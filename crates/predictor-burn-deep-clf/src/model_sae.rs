//! Per-model sparse autoencoder — dictionary learning on a *trained deep
//! classifier's* own hidden activations, the nonlinear sibling of the layer
//! probe. Ported from `predictor-burn-mlp::model_sae`: the analysis half (SAE
//! training via `lensing_sae`, capacity / segment / concept / dropped-signal
//! reads, depth probes via `lensing_interp`) is identical — only the head
//! differs, because this classifier splits its input into a standardized
//! continuous block + raw one-hot groups and assembles a per-topology
//! trunk input, rather than taking one scaler-standardized feature matrix.
//!
//! What it buys over the plain layer probe (same four reads as the MLP version):
//!   1. capacity — how much of each layer's width the model uses (dead / rare).
//!   2. segment representation — for every one-hot value, whether a dedicated
//!      atom separates that slice.
//!   3. concept-vs-decodability depth — where the target becomes linearly
//!      decodable vs. where interpretable concepts first form.
//!   4. dropped signal — target-relevant embedding concepts no model atom tracks
//!      (optional, needs `--dataset-sae`).
//!
//! Binary classifiers only (the model-output reference stage reads P(class==1));
//! any trunk topology (mlp / embeddings / wide_deep) is analyzable.

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use anyhow::{ensure, Context, Result};
use serde::Deserialize;
use serde_json::{json, Value};

use lensing_core::manifest::ColumnKind;
use lensing_core::Dataset;
use lensing_interp::{fit_stages, gather_targets, reference_stage, Stage, StageKind};
use lensing_sae::{autointerp, llm, sae};

use crate::model::{self, Activation, ArchConfig, Topology};

/// Everything the `model-sae` subcommand takes (mirrors the MLP version so the
/// server invocation and CLI are identical across the two predictors).
pub struct Args {
    pub model_dir: PathBuf,
    pub dataset_dir: PathBuf,
    pub output: PathBuf,
    pub layers: Vec<usize>,
    pub n_atoms: usize,
    pub l1: f64,
    pub epochs: usize,
    pub lr: f64,
    pub batch: usize,
    pub seed: u64,
    pub lambda: f64,
    pub min_segment: usize,
    pub dataset_sae: Option<PathBuf>,
    pub label_atoms: bool,
    pub cache_dir: Option<PathBuf>,
}

/// A concept is "interpretable" if it correlates with the target at least this
/// much and isn't vanishingly rare — the bar for counting it toward depth.
const CONCEPT_CORR: f64 = 0.10;
const MIN_FREQ: f64 = 0.005;
/// A segment is "represented" if some atom's mean activation separates it from
/// the rest by at least this many (pooled) SDs.
const SEG_SEP: f64 = 0.5;

/// Bump when the recipe changes so stale SAE caches are not reused.
const ENGINE_VERSION: u32 = 1;

/// Minimal view of the model dir's `hyperparams.json` — enough to rebuild the
/// architecture (mirrors the classifier's `Hyperparams`).
#[derive(Deserialize)]
struct HpSnapshot {
    #[serde(default = "default_hidden")]
    hidden: Vec<usize>,
    #[serde(default)]
    dropout: f64,
    #[serde(default)]
    activation: Activation,
    #[serde(default)]
    topology: Topology,
    #[serde(default = "default_embed_dim")]
    embed_dim: usize,
    #[serde(default = "default_batch")]
    batch_size: usize,
}
fn default_hidden() -> Vec<usize> {
    vec![128, 64]
}
fn default_embed_dim() -> usize {
    8
}
fn default_batch() -> usize {
    256
}

/// Continuous-block standardizer as persisted by the trainer (`scaler.json`).
#[derive(Deserialize)]
struct ContScaler {
    mean: Vec<f32>,
    std: Vec<f32>,
}
impl ContScaler {
    fn transform(&self, x: &mut [f32]) {
        let w = self.mean.len();
        if w == 0 {
            return;
        }
        for row in x.chunks_exact_mut(w) {
            for (j, v) in row.iter_mut().enumerate() {
                *v = (*v - self.mean[j]) / self.std[j];
            }
        }
    }
}

/// Embedding width per group, capped at the group's cardinality (mirrors the
/// trainer's `embed_dims`).
fn embed_dims(group_lens: &[usize], embed_dim: usize) -> Vec<usize> {
    group_lens.iter().map(|&len| embed_dim.min(len).max(1)).collect()
}

/// Gather a contiguous `[n_rows, sel.len()]` block from a row-major matrix, over
/// rows 0..n_rows in order (the same row space the segment masks / probes use).
fn gather_block(features: &[f32], n_cols: usize, sel: &[usize], n_rows: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n_rows * sel.len());
    for r in 0..n_rows {
        let base = r * n_cols;
        for &c in sel {
            out.push(features[base + c]);
        }
    }
    out
}

pub fn run(args: Args, emit: &dyn Fn(Value)) -> Result<()> {
    let Args {
        model_dir,
        dataset_dir,
        output,
        layers,
        n_atoms,
        l1,
        epochs,
        lr,
        batch,
        seed,
        lambda,
        min_segment,
        dataset_sae,
        label_atoms,
        cache_dir,
    } = args;

    let hp: HpSnapshot = serde_json::from_str(
        &std::fs::read_to_string(model_dir.join("hyperparams.json"))
            .context("read hyperparams.json from model dir")?,
    )
    .context("parse hyperparams.json")?;
    let scaler: ContScaler = serde_json::from_str(
        &std::fs::read_to_string(model_dir.join("scaler.json"))
            .context("read scaler.json from model dir")?,
    )
    .context("parse scaler.json")?;

    // Binary only: the model-output reference stage reads P(class==1).
    let contract: Value = serde_json::from_str(
        &std::fs::read_to_string(model_dir.join("contract.json"))
            .context("read contract.json from model dir")?,
    )?;
    let task = contract["target"]["task"].as_str().unwrap_or("");
    ensure!(task == "binary", "per-model SAE supports binary classifiers only; task is {task:?}");
    let k = 2usize;

    let ds = Dataset::load(&dataset_dir)?;
    let n_rows = ds.manifest.n_rows;
    let n_cols = ds.manifest.n_cols;

    // Column layout from the dataset manifest (same rule as train/predict):
    // continuous = PCA + numerics, one-hots grouped in first-appearance order.
    let mut cont_idx: Vec<usize> = Vec::new();
    let mut groups: Vec<(String, Vec<usize>)> = Vec::new();
    for (i, c) in ds.manifest.columns.iter().enumerate() {
        match &c.kind {
            ColumnKind::Pca { .. } | ColumnKind::Numeric { .. } => cont_idx.push(i),
            ColumnKind::Onehot { group, .. } => match groups.iter_mut().find(|(g, _)| g == group) {
                Some((_, idx)) => idx.push(i),
                None => groups.push((group.clone(), vec![i])),
            },
        }
    }
    let group_lens: Vec<usize> = groups.iter().map(|(_, v)| v.len()).collect();
    let oh_idx: Vec<usize> = groups.iter().flat_map(|(_, v)| v.iter().copied()).collect();
    let n_cont = cont_idx.len();
    let n_oh = oh_idx.len();
    ensure!(
        scaler.mean.len() == n_cont,
        "model's continuous block is {} wide, dataset has {n_cont}",
        scaler.mean.len()
    );

    let bs = hp.batch_size.max(1);
    let model_id = model_dir.file_name().and_then(|s| s.to_str()).unwrap_or("model").to_string();
    emit(json!({"event":"log","msg":format!(
        "model-sae: model {model_id} topology {} {:?} ({} activation), dataset {} ({} train / {} test)",
        hp.topology.name(), hp.hidden, hp.activation.name(), ds.manifest.dataset_id,
        ds.train_idx.len(), ds.test_idx.len())}));

    let arch = ArchConfig {
        n_cont,
        group_lens: group_lens.clone(),
        embed_dims: embed_dims(&group_lens, hp.embed_dim),
        hidden: hp.hidden.clone(),
        k,
        dropout: hp.dropout,
        activation: hp.activation,
        topology: hp.topology,
    };
    let mlp = model::load(&model_dir.join("model"), &arch)?;

    // Assemble the model's split input over ALL rows in order: standardized
    // continuous block + raw one-hots.
    let mut cont = gather_block(&ds.features, n_cols, &cont_idx, n_rows);
    let oh = gather_block(&ds.features, n_cols, &oh_idx, n_rows);
    scaler.transform(&mut cont);
    let captured = model::capture_stage_activations(&mlp, &cont, &oh, n_cont, n_oh, bs);

    // ---- Depth axis: linear decodability at each stage + the model's own head.
    let transform = ds.manifest.target.transform;
    let ytr = gather_targets(&ds.target, &ds.train_idx);
    let yte = gather_targets(&ds.target, &ds.test_idx);
    let clamp_hi = ytr.iter().cloned().fold(f64::MIN, f64::max) + std::f64::consts::LN_2;
    let net_stages: Vec<Stage> = captured
        .iter()
        .enumerate()
        .map(|(k2, la)| Stage {
            name: la.name.clone(),
            kind: if k2 == 0 { StageKind::Input } else { StageKind::Hidden },
            dim: la.dim,
            values: la.values.clone(),
        })
        .collect();
    let mut depth = fit_stages(
        &net_stages, &ds.train_idx, &ds.test_idx, &ytr, &yte, transform, clamp_hi, lambda, emit,
    );
    // The model's own output: P(class==1).
    let proba = model::predict_proba_loaded(&mlp, &cont, &oh, n_cont, n_oh, bs, k);
    let model_pred: Vec<f64> = (0..n_rows).map(|r| proba[r * k + 1] as f64).collect();
    depth.push(reference_stage(
        depth.len(), "output (model)", &model_pred, &ds.train_idx, &ds.test_idx, &ytr, &yte,
        transform, clamp_hi,
    ));

    // ---- Segment masks: one per one-hot column value, in manifest order.
    let segments: Vec<(String, usize)> = ds
        .manifest
        .columns
        .iter()
        .enumerate()
        .filter_map(|(ci, c)| match &c.kind {
            ColumnKind::Onehot { group, value } => Some((format!("{group}={value}"), ci)),
            _ => None,
        })
        .collect();

    // ---- Per hidden layer: train the SAE on its activations and analyze.
    let n_hidden = captured.len().saturating_sub(1);
    let want: BTreeSet<usize> = if layers.is_empty() {
        (1..=n_hidden).collect()
    } else {
        layers.into_iter().filter(|&k| k >= 1 && k <= n_hidden).collect()
    };
    ensure!(!want.is_empty(), "no valid hidden layers to analyze (model has {n_hidden})");

    let llm_cfg = if label_atoms { llm::LlmConfig::from_env() } else { None };
    if label_atoms && llm_cfg.is_none() {
        emit(json!({"event":"log","msg":
            "--label-atoms set but OPENAI_API_KEY missing; atoms returned unlabeled"}));
    }
    let vocab = llm::Vocab::from_domain(
        &lensing_core::domain::Domain::load_or_default(std::path::Path::new(".")).unwrap_or_default(),
    );
    let content: Vec<String> = if llm_cfg.is_some() {
        autointerp::load_content(&dataset_dir, &ds.row_ids, n_rows)
    } else {
        Vec::new()
    };

    let (ds_atoms, ds_missing_codes) = match &dataset_sae {
        Some(p) => load_ds_atoms(p, n_rows)?,
        None => (Vec::new(), 0),
    };
    if dataset_sae.is_some() {
        emit(json!({"event":"log","msg":format!(
            "embedding diff: {} target-relevant atoms loaded ({} lacked codes)",
            ds_atoms.len(), ds_missing_codes)}));
    }

    let mut layer_reports: Vec<Value> = Vec::new();

    for k in &want {
        let la = &captured[*k];
        let d = la.dim;
        let m = if n_atoms == 0 { 2 * d } else { n_atoms };

        // Standardize this layer's activations on TRAIN stats, then train/encode.
        let (mean, std) = col_stats(&la.values, &ds.train_idx, d);
        let mut act = la.values.clone();
        for r in 0..n_rows {
            for j in 0..d {
                act[r * d + j] = ((act[r * d + j] as f64 - mean[j]) / std[j]) as f32;
            }
        }
        let train_x = gather(&act, &ds.train_idx, d);

        let cfg = sae::SaeConfig { n_atoms: m, l1, epochs, lr, batch_size: batch, seed };
        let cache = cache_dir.as_ref().map(|c| {
            c.join(cache_key(&model_id, &ds.manifest.dataset_id, *k, d, m, l1, epochs, lr, seed))
        });
        let hit = cache.as_ref().map(|c| c.join("model.mpk").exists()).unwrap_or(false);
        let sae_model = if hit {
            emit(json!({"event":"log","msg":format!("layer {k}: loaded cached SAE")}));
            sae::load(cache.as_ref().unwrap(), d, m)?
        } else {
            emit(json!({"event":"log","msg":format!(
                "layer {k}: {d}-d activations → {m} atoms, l1 {l1}, {epochs} epochs")}));
            let (sm, _loss) = sae::train(&train_x, d, &cfg, emit);
            if let Some(c) = &cache {
                let _ = sae::save(&sm, c);
            }
            sm
        };
        let out = sae::encode_all(&sae_model, &act, d, batch);

        // (1) Capacity read: how much of the layer's width the model uses.
        let (freq, _amean, sd, corr) = atom_global(&out.code_all, m, &ds.target, n_rows);
        let active = freq.iter().filter(|&&f| f > 0.0).count();
        let rare = freq.iter().filter(|&&f| f > 0.0 && f < MIN_FREQ).count();
        let capacity = json!({
            "n_atoms": m,
            "active_atoms": active,
            "dead_atoms": m - active,
            "rare_atoms": rare,
            "utilization": active as f64 / m as f64,
            "l0_mean": out.l0_mean,
            "l0_frac": out.l0_mean / m as f64,
            "var_explained": out.var_explained,
        });

        // Atom → target-correlation ranking (interpretable concepts).
        let mut by_target: Vec<usize> = (0..m).filter(|&i| freq[i] >= MIN_FREQ).collect();
        by_target.sort_by(|&a, &b| corr[b].abs().partial_cmp(&corr[a].abs()).unwrap());
        let n_concepts = by_target.iter().filter(|&&i| corr[i].abs() >= CONCEPT_CORR).count();
        let shown: Vec<usize> = by_target.iter().take(12).copied().collect();
        let labels: HashMap<usize, String> = match &llm_cfg {
            Some(cfg) => {
                emit(json!({"event":"log","msg":format!(
                    "layer {k}: labeling {} atoms via {}", shown.len(), cfg.model)}));
                autointerp::label_atoms_concurrent(cfg, &vocab, &out.code_all, m, &shown, &content)
            }
            None => HashMap::new(),
        };
        let atoms_by_target: Vec<Value> = shown
            .iter()
            .map(|&i| json!({
                "atom": i, "target_corr": corr[i], "freq": freq[i],
                "label": labels.get(&i),
                "top_rows": autointerp::top_k_rows(&out.code_all, m, i, 12),
            }))
            .collect();

        // Does sparsifying the layer decode the target BETTER than the raw layer?
        let probe_stages = vec![
            Stage { name: format!("hidden_{k}:raw"), kind: StageKind::Input, dim: d, values: act },
            Stage {
                name: format!("hidden_{k}:sae"),
                kind: StageKind::Hidden,
                dim: m,
                values: out.code_all.clone(),
            },
        ];
        let probe = fit_stages(
            &probe_stages, &ds.train_idx, &ds.test_idx, &ytr, &yte, transform, clamp_hi, lambda,
            emit,
        );

        // (2) Segment representation, generalized over every one-hot value.
        let seg_reports: Vec<Value> = segments
            .iter()
            .filter_map(|(label, ci)| {
                let mask: Vec<bool> =
                    (0..n_rows).map(|r| ds.features[r * n_cols + ci] > 0.5).collect();
                let nseg = mask.iter().filter(|&&b| b).count();
                if nseg < min_segment || nseg == n_rows {
                    return None;
                }
                let sep = segment_sep(&out.code_all, m, &mask, &sd, n_rows);
                let mut order: Vec<usize> = (0..m).filter(|&i| freq[i] >= MIN_FREQ).collect();
                order.sort_by(|&a, &b| sep[b].abs().partial_cmp(&sep[a].abs()).unwrap());
                let best = order.first().copied();
                let top: Vec<Value> = order
                    .iter()
                    .take(3)
                    .map(|&i| json!({"atom": i, "separation": sep[i], "freq": freq[i]}))
                    .collect();
                Some(json!({
                    "segment": label,
                    "n_rows": nseg,
                    "represented": best.map(|i| sep[i].abs() >= SEG_SEP).unwrap_or(false),
                    "top_atoms": top,
                }))
            })
            .collect();

        // (4) Dropped signal FOR THIS LAYER.
        let dropped_vs_embedding = if ds_atoms.is_empty() {
            Value::Null
        } else {
            dropped_for_layer(&ds_atoms, &out.code_all, m, n_rows)
        };

        layer_reports.push(json!({
            "layer": k,
            "dim": d,
            "capacity": capacity,
            "n_interpretable_concepts": n_concepts,
            "probe": probe,
            "atoms_by_target_corr": atoms_by_target,
            "segments": seg_reports,
            "dropped_vs_embedding": dropped_vs_embedding,
        }));
    }

    // (3) Concept-vs-decodability depth.
    let depth_summary: Vec<Value> = layer_reports
        .iter()
        .map(|lr_| {
            let k = lr_["layer"].as_u64().unwrap_or(0) as usize;
            let lin = depth.get(k);
            json!({
                "layer": k,
                "linear_r2_target": lin.map(|s| s.test_r2_target),
                "linear_r2_log": lin.map(|s| s.test_r2_log),
                "n_interpretable_concepts": lr_["n_interpretable_concepts"],
            })
        })
        .collect();

    if dataset_sae.is_none() {
        emit(json!({"event":"log","msg":
            "no --dataset-sae given; skipping dropped-signal diff"}));
    }

    let result = json!({
        "tool": "model-sae",
        "method": "sparse autoencoder (dictionary learning) on a trained deep classifier's hidden activations",
        "engine_version": ENGINE_VERSION,
        "model_dir": model_id,
        "dataset_id": ds.manifest.dataset_id,
        "n_train": ds.train_idx.len(),
        "n_test": ds.test_idx.len(),
        "topology": hp.topology.name(),
        "hidden": hp.hidden,
        "activation": hp.activation.name(),
        "config": {"n_atoms": n_atoms, "l1": l1, "epochs": epochs, "lr": lr, "seed": seed},
        "depth_linear_probe": depth,
        "layers": layer_reports,
        "concept_vs_decodability": depth_summary,
        "embedding_diff": if dataset_sae.is_some() {
            json!({
                "ran": true,
                "match_corr_threshold": DROPPED_MATCH_CORR,
                "n_target_atoms": ds_atoms.len(),
                "n_missing_codes": ds_missing_codes,
            })
        } else {
            json!({"ran": false})
        },
    });

    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(&output, serde_json::to_vec_pretty(&result)?)
        .with_context(|| format!("write {}", output.display()))?;
    emit(json!({"event":"log","msg":format!("wrote {}", output.display())}));
    emit(json!({"event":"done"}));
    Ok(())
}

/// Per-atom frequency, mean, SD, and correlation with the target, over all rows.
fn atom_global(code: &[f32], m: usize, y: &[f32], n_rows: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let nf = n_rows as f64;
    let ybar = y.iter().map(|&v| v as f64).sum::<f64>() / nf;
    let ysd = (y.iter().map(|&v| (v as f64 - ybar).powi(2)).sum::<f64>() / nf).sqrt().max(1e-9);
    let (mut cnt, mut s, mut ss, mut sy) =
        (vec![0f64; m], vec![0f64; m], vec![0f64; m], vec![0f64; m]);
    for r in 0..n_rows {
        let yr = y[r] as f64;
        let base = r * m;
        for i in 0..m {
            let v = code[base + i] as f64;
            if v > 1e-6 {
                cnt[i] += 1.0;
            }
            s[i] += v;
            ss[i] += v * v;
            sy[i] += v * yr;
        }
    }
    let (mut freq, mut mean, mut sd, mut corr) =
        (vec![0f64; m], vec![0f64; m], vec![0f64; m], vec![0f64; m]);
    for i in 0..m {
        mean[i] = s[i] / nf;
        sd[i] = (ss[i] / nf - mean[i] * mean[i]).max(0.0).sqrt();
        let cov = sy[i] / nf - mean[i] * ybar;
        corr[i] = if sd[i] > 1e-9 { cov / (sd[i] * ysd) } else { 0.0 };
        freq[i] = cnt[i] / nf;
    }
    (freq, mean, sd, corr)
}

/// Per-atom segment-vs-rest mean-activation separation, in pooled-SD units.
fn segment_sep(code: &[f32], m: usize, mask: &[bool], sd: &[f64], n_rows: usize) -> Vec<f64> {
    let (mut sseg, mut srest) = (vec![0f64; m], vec![0f64; m]);
    let (mut nseg, mut nrest) = (0f64, 0f64);
    for r in 0..n_rows {
        let base = r * m;
        if mask[r] {
            nseg += 1.0;
            for i in 0..m {
                sseg[i] += code[base + i] as f64;
            }
        } else {
            nrest += 1.0;
            for i in 0..m {
                srest[i] += code[base + i] as f64;
            }
        }
    }
    (0..m)
        .map(|i| {
            if nseg > 0.0 && nrest > 0.0 && sd[i] > 1e-9 {
                (sseg[i] / nseg - srest[i] / nrest) / sd[i]
            } else {
                0.0
            }
        })
        .collect()
}

/// A target-relevant embedding concept from a dataset SAE, standardized.
struct DsAtom {
    atom: Value,
    target_corr: f64,
    label: Option<Value>,
    code_std: Vec<f64>,
}

/// A concept whose best correlation with any model atom is below this is
/// "dropped".
const DROPPED_MATCH_CORR: f64 = 0.30;

fn load_ds_atoms(path: &Path, n_rows: usize) -> Result<(Vec<DsAtom>, usize)> {
    let ds_sae: Value = serde_json::from_str(
        &std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?,
    )?;
    let atoms = ds_sae["atoms_by_target_corr"].as_array().cloned().unwrap_or_default();
    let mut out = Vec::new();
    let mut missing = 0usize;
    for a in &atoms {
        let corr = a["target_corr"].as_f64().unwrap_or(0.0);
        if corr.abs() < 0.15 {
            continue;
        }
        let code: Vec<f32> = match a["code"].as_array() {
            Some(v) if v.len() == n_rows => {
                v.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect()
            }
            _ => {
                missing += 1;
                continue;
            }
        };
        out.push(DsAtom {
            atom: a["atom"].clone(),
            target_corr: corr,
            label: a.get("label").cloned(),
            code_std: standardize_col(&code, 1, 0, n_rows),
        });
    }
    Ok((out, missing))
}

fn dropped_for_layer(ds_atoms: &[DsAtom], model_code: &[f32], model_m: usize, n_rows: usize) -> Value {
    let model_cols: Vec<Vec<f64>> =
        (0..model_m).map(|j| standardize_col(model_code, model_m, j, n_rows)).collect();
    let mut dropped: Vec<Value> = Vec::new();
    for a in ds_atoms {
        let best = model_cols
            .iter()
            .map(|mc| pearson_pre(&a.code_std, mc, n_rows).abs())
            .fold(0.0f64, f64::max);
        if best < DROPPED_MATCH_CORR {
            dropped.push(json!({
                "dataset_atom": a.atom,
                "target_corr": a.target_corr,
                "label": a.label,
                "best_match_corr": best,
            }));
        }
    }
    json!({
        "n_checked": ds_atoms.len(),
        "n_dropped": dropped.len(),
        "dropped": dropped,
    })
}

fn standardize_col(x: &[f32], stride: usize, j: usize, n_rows: usize) -> Vec<f64> {
    let mut v: Vec<f64> = (0..n_rows).map(|r| x[r * stride + j] as f64).collect();
    let mean = v.iter().sum::<f64>() / n_rows.max(1) as f64;
    let sd = (v.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / n_rows.max(1) as f64).sqrt();
    if sd < 1e-9 {
        return vec![0.0; n_rows];
    }
    for e in &mut v {
        *e = (*e - mean) / sd;
    }
    v
}

fn pearson_pre(a: &[f64], b: &[f64], n_rows: usize) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>() / n_rows.max(1) as f64
}

fn col_stats(x: &[f32], train_idx: &[u32], d: usize) -> (Vec<f64>, Vec<f64>) {
    let n = train_idx.len().max(1) as f64;
    let mut mean = vec![0f64; d];
    for &r in train_idx {
        let base = r as usize * d;
        for k in 0..d {
            mean[k] += x[base + k] as f64;
        }
    }
    for mk in mean.iter_mut() {
        *mk /= n;
    }
    let mut var = vec![0f64; d];
    for &r in train_idx {
        let base = r as usize * d;
        for k in 0..d {
            let e = x[base + k] as f64 - mean[k];
            var[k] += e * e;
        }
    }
    let std = var.iter().map(|v| (v / n).sqrt().max(1e-8)).collect();
    (mean, std)
}

fn gather(x: &[f32], idx: &[u32], d: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(idx.len() * d);
    for &r in idx {
        out.extend_from_slice(&x[r as usize * d..(r as usize + 1) * d]);
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn cache_key(
    model_id: &str,
    dataset_id: &str,
    layer: usize,
    d: usize,
    m: usize,
    l1: f64,
    epochs: usize,
    lr: f64,
    seed: u64,
) -> String {
    let f = |x: f64| format!("{x}").replace('.', "p").replace('-', "m");
    format!(
        "{model_id}__{dataset_id}__L{layer}_d{d}_a{m}_l1{}_e{epochs}_lr{}_s{seed}_v{ENGINE_VERSION}",
        f(l1),
        f(lr)
    )
}
