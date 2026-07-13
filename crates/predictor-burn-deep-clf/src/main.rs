//! Representation-aware deep classifier on the language-neutral dataset
//! artifact. burn 0.21, ndarray CPU backend. Splits the feature matrix into a
//! standardized continuous block (PCA + numerics) and raw one-hot identity
//! groups, then trains one of three topologies (mlp / embeddings / wide_deep)
//! with softmax cross-entropy. Emits classification metrics (accuracy, logloss,
//! auc, brier) the server trusts verbatim.

mod model;
mod model_sae;
mod onnx_export;

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use serde::Deserialize;
use serde_json::json;

use lensing_core::manifest::ColumnKind;
use lensing_core::{Dataset, InferenceInput};

use model::{ArchConfig, Activation, Topology};

#[derive(Parser)]
#[command(name = "predictor-burn-deep-clf")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Train {
        #[arg(long)]
        dataset: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        hyperparams: PathBuf,
    },
    Predict {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Export the trained model as a portable `model.onnx` (binary only):
    /// input = assembled feature vector, output = P(class 1), shape [N, 1].
    Export {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Per-model sparse autoencoder: train an SAE on this promoted classifier's
    /// own hidden (trunk) activations per layer and report capacity, per-segment
    /// representation, concept-vs-decodability depth, and (with --dataset-sae)
    /// signal the model drops relative to the embedding. The nonlinear sibling
    /// of a layer probe; binary classifiers only. Writes JSON to `output`.
    ModelSae {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        dataset: PathBuf,
        #[arg(long)]
        output: PathBuf,
        /// Comma-separated 1-based hidden layers to analyze; empty ⇒ all.
        #[arg(long, default_value = "")]
        layers: String,
        /// SAE atom count; 0 ⇒ 2× the layer width.
        #[arg(long, default_value_t = 0)]
        n_atoms: usize,
        /// L1 sparsity penalty. Matches the dataset-SAE default so CLI and the
        /// `/api/interp/model-sae` endpoint behave identically by default.
        #[arg(long, default_value_t = 0.0015)]
        l1: f64,
        #[arg(long, default_value_t = 50)]
        epochs: usize,
        #[arg(long, default_value_t = 1e-3)]
        lr: f64,
        #[arg(long, default_value_t = 256)]
        batch: usize,
        #[arg(long, default_value_t = 1337)]
        seed: u64,
        /// Ridge penalty for the probes; 0 ⇒ internal CV.
        #[arg(long, default_value_t = 0.0)]
        lambda: f64,
        /// Skip a one-hot group whose slice has fewer than this many rows.
        #[arg(long, default_value_t = 30)]
        min_segment: usize,
        /// A prior `lensing-sae analyze --emit-codes` result JSON, to diff for
        /// dropped signal (its target atoms must carry `code` vectors).
        #[arg(long)]
        dataset_sae: Option<PathBuf>,
        /// Label the surfaced atoms with an LLM (auto-interp); no-op without
        /// OPENAI_API_KEY.
        #[arg(long)]
        label_atoms: bool,
        /// Cache dir for trained SAEs (content-addressed by model+layer+config).
        #[arg(long)]
        cache_dir: Option<PathBuf>,
    },
}

#[derive(Debug, Deserialize)]
struct Hyperparams {
    #[serde(default = "d_epochs")]
    epochs: usize,
    #[serde(default = "d_lr")]
    lr: f64,
    #[serde(default = "d_batch")]
    batch_size: usize,
    #[serde(default = "d_hidden")]
    hidden: Vec<usize>,
    #[serde(default = "d_dropout")]
    dropout: f64,
    #[serde(default)]
    activation: Activation,
    #[serde(default)]
    topology: Topology,
    /// Embedding width per one-hot group (Embeddings topology); capped at the
    /// group's cardinality so it never exceeds a lossless encoding.
    #[serde(default = "d_embed_dim")]
    embed_dim: usize,
    #[serde(default)]
    patience: usize,
    #[serde(default = "d_val_fraction")]
    val_fraction: f64,
    #[serde(default)]
    checkpoint_every: usize,
}

fn d_epochs() -> usize { 100 }
fn d_lr() -> f64 { 1e-3 }
fn d_batch() -> usize { 256 }
fn d_hidden() -> Vec<usize> { vec![128, 64] }
fn d_dropout() -> f64 { 0.1 }
fn d_embed_dim() -> usize { 8 }
fn d_val_fraction() -> f64 { 0.15 }

const SEED: u64 = 1337;

fn emit(v: serde_json::Value) {
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "{v}");
    let _ = out.flush();
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Train { dataset, output, hyperparams } => train(dataset, output, hyperparams),
        Command::Predict { model, input, output } => predict(model, input, output),
        Command::Export { model, output } => export(model, output),
        Command::ModelSae {
            model,
            dataset,
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
        } => {
            let layers = layers
                .split(',')
                .filter_map(|s| s.trim().parse::<usize>().ok())
                .collect();
            model_sae::run(
                model_sae::Args {
                    model_dir: model,
                    dataset_dir: dataset,
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
                },
                &emit,
            )
        }
    }
}

// ----------------------------------------------------------------------------
// Column layout: continuous indices + one-hot groups (first-appearance order).
// ----------------------------------------------------------------------------

struct Layout {
    cont_idx: Vec<usize>,
    /// (group_name, column indices) in canonical order.
    groups: Vec<(String, Vec<usize>)>,
}

impl Layout {
    fn from_columns(cols: &[lensing_core::manifest::ColumnDesc]) -> Self {
        let mut cont_idx = Vec::new();
        let mut groups: Vec<(String, Vec<usize>)> = Vec::new();
        for (i, c) in cols.iter().enumerate() {
            match &c.kind {
                ColumnKind::Pca { .. } | ColumnKind::Numeric { .. } => cont_idx.push(i),
                ColumnKind::Onehot { group, .. } => {
                    match groups.iter_mut().find(|(g, _)| g == group) {
                        Some((_, idx)) => idx.push(i),
                        None => groups.push((group.clone(), vec![i])),
                    }
                }
            }
        }
        Layout { cont_idx, groups }
    }
    fn group_lens(&self) -> Vec<usize> {
        self.groups.iter().map(|(_, v)| v.len()).collect()
    }
    fn oh_idx(&self) -> Vec<usize> {
        self.groups.iter().flat_map(|(_, v)| v.iter().copied()).collect()
    }
}

fn embed_dims(group_lens: &[usize], embed_dim: usize) -> Vec<usize> {
    group_lens.iter().map(|&len| embed_dim.min(len).max(1)).collect()
}

/// Gather a contiguous [rows.len(), sel.len()] block from a row-major matrix.
fn gather(features: &[f32], n_cols: usize, rows: &[u32], sel: &[usize]) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows.len() * sel.len());
    for &r in rows {
        let base = r as usize * n_cols;
        for &c in sel {
            out.push(features[base + c]);
        }
    }
    out
}

fn encode_labels(target: &[f32]) -> (Vec<i64>, Vec<i64>, usize) {
    let raw: Vec<i64> = target.iter().map(|v| v.round() as i64).collect();
    let mut classes: Vec<i64> = raw.clone();
    classes.sort_unstable();
    classes.dedup();
    let remap = |c: i64| classes.iter().position(|&x| x == c).unwrap() as i64;
    let y: Vec<i64> = raw.iter().map(|&c| remap(c)).collect();
    let k = classes.len();
    (y, classes, k)
}

// ----------------------------------------------------------------------------
// Continuous-block standardizer (fit on the train/fit split only).
// ----------------------------------------------------------------------------

#[derive(serde::Serialize, serde::Deserialize)]
struct ContScaler {
    mean: Vec<f32>,
    std: Vec<f32>,
}

impl ContScaler {
    fn fit(x: &[f32], w: usize) -> Self {
        if w == 0 {
            return ContScaler { mean: vec![], std: vec![] };
        }
        let n = x.len() / w;
        let mut mean = vec![0.0f64; w];
        for row in x.chunks_exact(w) {
            for (m, v) in mean.iter_mut().zip(row) {
                *m += *v as f64;
            }
        }
        for m in &mut mean {
            *m /= n as f64;
        }
        let mut var = vec![0.0f64; w];
        for row in x.chunks_exact(w) {
            for (j, v) in row.iter().enumerate() {
                let d = *v as f64 - mean[j];
                var[j] += d * d;
            }
        }
        let std: Vec<f32> = var
            .iter()
            .map(|v| {
                let s = (v / n as f64).sqrt();
                if s < 1e-12 { 1.0 } else { s as f32 }
            })
            .collect();
        ContScaler { mean: mean.into_iter().map(|m| m as f32).collect(), std }
    }
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
    fn save(&self, p: &Path) -> Result<()> {
        std::fs::write(p, serde_json::to_vec(self)?)?;
        Ok(())
    }
    fn load(p: &Path) -> Result<Self> {
        Ok(serde_json::from_str(&std::fs::read_to_string(p)?)?)
    }
}

// ----------------------------------------------------------------------------
// Classification metrics (binary primary; mirror _clf_common keys).
// ----------------------------------------------------------------------------

fn binary_auc(y: &[i64], p1: &[f32]) -> f64 {
    let n = y.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| p1[a].partial_cmp(&p1[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut rank = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && p1[idx[j + 1]] == p1[idx[i]] {
            j += 1;
        }
        let avg = (i + j) as f64 / 2.0 + 1.0; // 1-based average rank for ties
        for &k in &idx[i..=j] {
            rank[k] = avg;
        }
        i = j + 1;
    }
    let n_pos = y.iter().filter(|&&v| v == 1).count();
    let n_neg = n - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return 0.5;
    }
    let sum_pos: f64 = (0..n).filter(|&k| y[k] == 1).map(|k| rank[k]).sum();
    (sum_pos - n_pos as f64 * (n_pos as f64 + 1.0) / 2.0) / (n_pos as f64 * n_neg as f64)
}

/// Build metrics.json object from test labels + row-major [n,k] proba.
fn metrics_obj(y: &[i64], proba: &[f32], k: usize) -> serde_json::Value {
    let n = y.len();
    let mut correct = 0usize;
    let mut logloss = 0.0f64;
    for (i, &lab) in y.iter().enumerate() {
        let row = &proba[i * k..(i + 1) * k];
        let am = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(j, _)| j)
            .unwrap_or(0);
        if am as i64 == lab {
            correct += 1;
        }
        let p = (row[lab as usize] as f64).clamp(1e-15, 1.0 - 1e-15);
        logloss -= p.ln();
    }
    let mut m = json!({
        "n_test": n,
        "accuracy": correct as f64 / n as f64,
        "logloss": logloss / n as f64,
    });
    if k == 2 {
        let p1: Vec<f32> = (0..n).map(|i| proba[i * 2 + 1]).collect();
        let auc = binary_auc(y, &p1);
        let brier = (0..n)
            .map(|i| {
                let d = p1[i] as f64 - y[i] as f64;
                d * d
            })
            .sum::<f64>()
            / n as f64;
        m["auc"] = json!(auc);
        m["brier"] = json!(brier);
    }
    m
}

// ----------------------------------------------------------------------------
// Train
// ----------------------------------------------------------------------------

fn train(dataset_dir: PathBuf, run_dir: PathBuf, hp_path: PathBuf) -> Result<()> {
    let hp: Hyperparams =
        serde_json::from_str(&std::fs::read_to_string(&hp_path).context("read hyperparams")?)
            .context("parse hyperparams")?;
    anyhow::ensure!(hp.epochs >= 1, "epochs must be >= 1");
    anyhow::ensure!(hp.batch_size >= 1, "batch_size must be >= 1");
    anyhow::ensure!(!hp.hidden.is_empty(), "hidden must have at least one layer");
    std::fs::create_dir_all(&run_dir)?;

    let ds = Dataset::load(&dataset_dir)?;
    let n_cols = ds.manifest.n_cols;
    let layout = Layout::from_columns(&ds.manifest.columns);
    let group_lens = layout.group_lens();
    let oh_idx = layout.oh_idx();
    let n_cont = layout.cont_idx.len();
    let n_oh = oh_idx.len();
    let (y_all, classes, k) = encode_labels(&ds.target);
    anyhow::ensure!(k >= 2, "classification needs >= 2 classes, target has {k}");

    // Persist the column layout so `export` can rebuild the Gather indices
    // without the dataset (predict reconstructs it from the input manifest).
    let groups_json: Vec<serde_json::Value> = layout
        .groups
        .iter()
        .map(|(name, idx)| json!({"name": name, "idx": idx}))
        .collect();
    std::fs::write(
        run_dir.join("layout.json"),
        serde_json::to_vec(&json!({
            "n_cols": n_cols, "cont_idx": layout.cont_idx, "groups": groups_json}))?,
    )?;

    let group_names: Vec<&str> = layout.groups.iter().map(|(g, _)| g.as_str()).collect();
    emit(json!({"event":"log","msg":format!(
        "dataset {}: {} train / {} test rows, {} cols → {} continuous + {} one-hot in {} groups {:?}",
        ds.manifest.dataset_id, ds.train_idx.len(), ds.test_idx.len(), n_cols,
        n_cont, n_oh, group_lens.len(), group_names)}));
    let edims = embed_dims(&group_lens, hp.embed_dim);
    emit(json!({"event":"log","msg":format!(
        "topology {}, hidden {:?}, embed_dims {:?}, {} activation, dropout {}, lr {}, batch {}, {} epochs, {} classes {:?}",
        hp.topology.name(), hp.hidden, edims, hp.activation.name(), hp.dropout, hp.lr,
        hp.batch_size, hp.epochs, k, classes)}));

    // Early-stop carve from the TRAIN split (test stays untouched).
    let (fit_idx, val_idx) = if hp.patience > 0 {
        anyhow::ensure!(
            hp.val_fraction > 0.0 && hp.val_fraction < 1.0,
            "val_fraction must be in (0,1) when patience > 0"
        );
        let (fit_pos, val_pos) =
            lensing_core::shuffle::train_test_split(ds.train_idx.len(), hp.val_fraction, SEED);
        let map = |pos: Vec<u32>| -> Vec<u32> {
            pos.into_iter().map(|p| ds.train_idx[p as usize]).collect()
        };
        (map(fit_pos), map(val_pos))
    } else {
        (ds.train_idx.clone(), Vec::new())
    };

    // Assemble per-split contiguous blocks. Continuous standardized on fit only.
    let mut fit_cont = gather(&ds.features, n_cols, &fit_idx, &layout.cont_idx);
    let fit_oh = gather(&ds.features, n_cols, &fit_idx, &oh_idx);
    let fit_y: Vec<i64> = fit_idx.iter().map(|&r| y_all[r as usize]).collect();

    let mut val_cont = gather(&ds.features, n_cols, &val_idx, &layout.cont_idx);
    let val_oh = gather(&ds.features, n_cols, &val_idx, &oh_idx);
    let val_y: Vec<i64> = val_idx.iter().map(|&r| y_all[r as usize]).collect();

    let mut test_cont = gather(&ds.features, n_cols, &ds.test_idx, &layout.cont_idx);
    let test_oh = gather(&ds.features, n_cols, &ds.test_idx, &oh_idx);
    let test_y: Vec<i64> = ds.test_idx.iter().map(|&r| y_all[r as usize]).collect();

    let scaler = ContScaler::fit(&fit_cont, n_cont);
    scaler.transform(&mut fit_cont);
    scaler.transform(&mut val_cont);
    scaler.transform(&mut test_cont);
    scaler.save(&run_dir.join("scaler.json"))?;

    let arch = ArchConfig {
        n_cont,
        group_lens: group_lens.clone(),
        embed_dims: edims,
        hidden: hp.hidden.clone(),
        k,
        dropout: hp.dropout,
        activation: hp.activation,
        topology: hp.topology,
    };

    let trained = model::run_training(
        model::TrainInputs {
            train_cont: &fit_cont,
            train_oh: &fit_oh,
            train_y: &fit_y,
            val_cont: &val_cont,
            val_oh: &val_oh,
            val_y: &val_y,
        },
        model::TrainConfig {
            arch: arch.clone(),
            n_cont,
            n_oh,
            epochs: hp.epochs,
            lr: hp.lr,
            batch_size: hp.batch_size,
            seed: SEED,
            patience: hp.patience,
            checkpoint_every: hp.checkpoint_every,
        },
        &run_dir,
        &emit,
    )?;

    let proba = model::predict_proba(&trained, &test_cont, &test_oh, n_cont, n_oh, hp.batch_size, k);
    let metrics = metrics_obj(&test_y, &proba, k);

    // predictions.json: predicted = P(class 1) (binary) / argmax id (multiclass).
    let mut predictions = Vec::with_capacity(test_y.len());
    for (j, &row) in ds.test_idx.iter().enumerate() {
        let p: Vec<f64> = (0..k).map(|c| proba[j * k + c] as f64).collect();
        let predicted = if k == 2 {
            p[1]
        } else {
            p.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as f64
        };
        predictions.push(json!({
            "row_id": ds.row_ids[row as usize],
            "actual": test_y[j] as f64,
            "predicted": predicted,
            "proba": p,
        }));
    }

    std::fs::write(run_dir.join("metrics.json"), serde_json::to_vec(&metrics)?)?;
    std::fs::write(run_dir.join("predictions.json"), serde_json::to_vec(&predictions)?)?;
    std::fs::write(run_dir.join("meta.json"), serde_json::to_vec(&json!({"classes": classes, "k": k}))?)?;
    model::save_atomic(&trained, &run_dir)?;

    let auc = metrics.get("auc").and_then(|v| v.as_f64()).unwrap_or(f64::NAN);
    let ll = metrics["logloss"].as_f64().unwrap();
    let acc = metrics["accuracy"].as_f64().unwrap();
    emit(json!({"event":"log","msg":format!("AUC {auc:.4}  logloss {ll:.4}  acc {acc:.3}")}));
    emit(json!({"event":"done"}));
    Ok(())
}

// ----------------------------------------------------------------------------
// Predict
// ----------------------------------------------------------------------------

fn predict(model_dir: PathBuf, input_dir: PathBuf, output: PathBuf) -> Result<()> {
    let hp: Hyperparams = serde_json::from_str(
        &std::fs::read_to_string(model_dir.join("hyperparams.json"))
            .context("read hyperparams.json from model dir")?,
    )
    .context("parse hyperparams.json")?;
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(model_dir.join("meta.json"))?)?;
    let k = meta["k"].as_u64().context("meta.json missing k")? as usize;

    let input = InferenceInput::load(&input_dir)?;
    let n_cols = input.manifest.n_cols;
    let layout = Layout::from_columns(&input.manifest.columns);
    let group_lens = layout.group_lens();
    let oh_idx = layout.oh_idx();
    let n_cont = layout.cont_idx.len();
    let n_oh = oh_idx.len();

    let all_rows: Vec<u32> = (0..input.manifest.n_rows as u32).collect();
    let mut cont = gather(&input.features, n_cols, &all_rows, &layout.cont_idx);
    let oh = gather(&input.features, n_cols, &all_rows, &oh_idx);
    let scaler = ContScaler::load(&model_dir.join("scaler.json"))?;
    anyhow::ensure!(scaler.mean.len() == n_cont, "scaler/continuous-width mismatch");
    scaler.transform(&mut cont);

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
    let m = model::load(&model_dir.join("model"), &arch)?;
    let proba = model::predict_proba_loaded(&m, &cont, &oh, n_cont, n_oh, hp.batch_size, k);
    emit(json!({"event":"log","msg":format!(
        "predicting {} rows, topology {}, {k} classes", input.manifest.n_rows, hp.topology.name())}));

    let mut rows = Vec::with_capacity(input.row_ids.len());
    for (j, &row_id) in input.row_ids.iter().enumerate() {
        let p: Vec<f64> = (0..k).map(|c| proba[j * k + c] as f64).collect();
        let predicted = if k == 2 {
            p[1]
        } else {
            p.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as f64
        };
        rows.push(json!({"row_id": row_id, "predicted": predicted, "proba": p}));
    }
    std::fs::write(&output, serde_json::to_vec(&rows)?)
        .with_context(|| format!("write {}", output.display()))?;
    emit(json!({"event":"done"}));
    Ok(())
}

// ----------------------------------------------------------------------------
// Export (ONNX) — binary only
// ----------------------------------------------------------------------------

#[derive(Deserialize)]
struct LayoutFile {
    n_cols: usize,
    cont_idx: Vec<usize>,
    groups: Vec<LayoutGroup>,
}

#[derive(Deserialize)]
struct LayoutGroup {
    #[allow(dead_code)]
    name: String,
    idx: Vec<usize>,
}

fn export(model_dir: PathBuf, output: PathBuf) -> Result<()> {
    let hp: Hyperparams = serde_json::from_str(
        &std::fs::read_to_string(model_dir.join("hyperparams.json"))
            .context("read hyperparams.json from model dir")?,
    )
    .context("parse hyperparams.json")?;
    // `meta.json` is not copied into a promoted model dir (the server excludes
    // it on promotion); read the binary-only contract from the frozen contract.
    let contract: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(model_dir.join("contract.json"))
            .context("read contract.json from model dir")?,
    )?;
    let task = contract["target"]["task"].as_str().unwrap_or("");
    anyhow::ensure!(
        task == "binary",
        "ONNX export supports binary classifiers only; target task is {task:?}"
    );
    let k = 2usize;
    let lf: LayoutFile = serde_json::from_str(
        &std::fs::read_to_string(model_dir.join("layout.json"))
            .context("read layout.json from model dir")?,
    )
    .context("parse layout.json")?;
    let scaler = ContScaler::load(&model_dir.join("scaler.json"))?;

    let groups: Vec<Vec<usize>> = lf.groups.iter().map(|g| g.idx.clone()).collect();
    let group_lens: Vec<usize> = groups.iter().map(|v| v.len()).collect();
    let arch = ArchConfig {
        n_cont: lf.cont_idx.len(),
        group_lens: group_lens.clone(),
        embed_dims: embed_dims(&group_lens, hp.embed_dim),
        hidden: hp.hidden.clone(),
        k,
        dropout: hp.dropout,
        activation: hp.activation,
        topology: hp.topology,
    };
    let m = model::load(&model_dir.join("model"), &arch)?;
    let bundle = m.export_bundle();
    onnx_export::write_onnx(
        &lf.cont_idx, &groups, &scaler.mean, &scaler.std, &bundle, lf.n_cols, &output,
    )?;
    emit(json!({"event":"log","msg":format!(
        "exported model.onnx: burn-deep-classifier topology {}, {} cols, binary P(class=1)",
        hp.topology.name(), lf.n_cols)}));
    emit(json!({"event":"done"}));
    Ok(())
}
