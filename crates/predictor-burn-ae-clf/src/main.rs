//! Autoencoder-pretrained deep classifier on the language-neutral dataset
//! artifact. burn 0.21, ndarray CPU backend. Phase 1 trains an autoencoder to
//! reconstruct the full assembled feature vector (standardized continuous block
//! + raw one-hots); phase 2 transplants the encoder into a deep classifier head
//! and fine-tunes it with softmax cross-entropy. Emits the same classification
//! metrics (accuracy, logloss, auc, brier) the server trusts verbatim.

mod model;
mod onnx_export;

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use serde::Deserialize;
use serde_json::json;

use lensing_core::manifest::ColumnKind;
use lensing_core::{Dataset, InferenceInput};

use model::{AeArch, Activation};

#[derive(Parser)]
#[command(name = "predictor-burn-ae-clf")]
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
}

#[derive(Debug, Deserialize)]
struct Hyperparams {
    #[serde(default = "d_enc_hidden")]
    enc_hidden: Vec<usize>,
    #[serde(default = "d_ae_epochs")]
    ae_epochs: usize,
    #[serde(default = "d_lr")]
    ae_lr: f64,
    #[serde(default)]
    ae_dropout: f64,
    #[serde(default = "d_clf_hidden")]
    clf_hidden: Vec<usize>,
    #[serde(default)]
    freeze_encoder: bool,
    #[serde(default = "d_epochs")]
    epochs: usize,
    #[serde(default = "d_lr")]
    lr: f64,
    #[serde(default = "d_batch")]
    batch_size: usize,
    #[serde(default = "d_dropout")]
    dropout: f64,
    #[serde(default)]
    activation: Activation,
    #[serde(default)]
    patience: usize,
    #[serde(default = "d_val_fraction")]
    val_fraction: f64,
    #[serde(default)]
    checkpoint_every: usize,
}

fn d_enc_hidden() -> Vec<usize> { vec![128, 64] }
fn d_ae_epochs() -> usize { 60 }
fn d_clf_hidden() -> Vec<usize> { vec![64] }
fn d_epochs() -> usize { 100 }
fn d_lr() -> f64 { 1e-3 }
fn d_batch() -> usize { 256 }
fn d_dropout() -> f64 { 0.1 }
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
    }
}

// ----------------------------------------------------------------------------
// Column layout: continuous indices + one-hot indices (first-appearance order).
// ----------------------------------------------------------------------------

struct Layout {
    cont_idx: Vec<usize>,
    oh_idx: Vec<usize>,
}

impl Layout {
    fn from_columns(cols: &[lensing_core::manifest::ColumnDesc]) -> Self {
        let mut cont_idx = Vec::new();
        let mut oh_idx = Vec::new();
        for (i, c) in cols.iter().enumerate() {
            match &c.kind {
                ColumnKind::Pca { .. } | ColumnKind::Numeric { .. } => cont_idx.push(i),
                ColumnKind::Onehot { .. } => oh_idx.push(i),
            }
        }
        Layout { cont_idx, oh_idx }
    }
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

/// Interleave two row-major blocks into `[cont | oh]` rows of width `n_cont + n_oh`.
fn interleave(cont: &[f32], n_cont: usize, oh: &[f32], n_oh: usize) -> Vec<f32> {
    let n = if n_cont > 0 { cont.len() / n_cont } else { oh.len() / n_oh.max(1) };
    let mut out = Vec::with_capacity(n * (n_cont + n_oh));
    for r in 0..n {
        out.extend_from_slice(&cont[r * n_cont..(r + 1) * n_cont]);
        out.extend_from_slice(&oh[r * n_oh..(r + 1) * n_oh]);
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
// Continuous-block standardizer (fit on the fit split only).
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

fn arch_from(hp: &Hyperparams, n_cont: usize, n_oh: usize, k: usize) -> AeArch {
    AeArch {
        n_cont,
        n_oh,
        enc_hidden: hp.enc_hidden.clone(),
        clf_hidden: hp.clf_hidden.clone(),
        k,
        ae_dropout: hp.ae_dropout,
        dropout: hp.dropout,
        activation: hp.activation,
        freeze_encoder: hp.freeze_encoder,
    }
}

// ----------------------------------------------------------------------------
// Train
// ----------------------------------------------------------------------------

fn train(dataset_dir: PathBuf, run_dir: PathBuf, hp_path: PathBuf) -> Result<()> {
    let hp: Hyperparams =
        serde_json::from_str(&std::fs::read_to_string(&hp_path).context("read hyperparams")?)
            .context("parse hyperparams")?;
    anyhow::ensure!(hp.epochs >= 1, "epochs must be >= 1");
    anyhow::ensure!(hp.ae_epochs >= 1, "ae_epochs must be >= 1");
    anyhow::ensure!(hp.batch_size >= 1, "batch_size must be >= 1");
    anyhow::ensure!(!hp.enc_hidden.is_empty(), "enc_hidden must have at least one layer");
    std::fs::create_dir_all(&run_dir)?;

    let ds = Dataset::load(&dataset_dir)?;
    let n_cols = ds.manifest.n_cols;
    let layout = Layout::from_columns(&ds.manifest.columns);
    let n_cont = layout.cont_idx.len();
    let n_oh = layout.oh_idx.len();
    let (y_all, classes, k) = encode_labels(&ds.target);
    anyhow::ensure!(k >= 2, "classification needs >= 2 classes, target has {k}");

    // Persist the column layout so `export` can rebuild the Gather indices.
    std::fs::write(
        run_dir.join("layout.json"),
        serde_json::to_vec(&json!({
            "n_cols": n_cols, "cont_idx": layout.cont_idx, "oh_idx": layout.oh_idx}))?,
    )?;

    emit(json!({"event":"log","msg":format!(
        "dataset {}: {} train / {} test rows, {} cols → {} continuous + {} one-hot (encoder D={})",
        ds.manifest.dataset_id, ds.train_idx.len(), ds.test_idx.len(), n_cols,
        n_cont, n_oh, n_cont + n_oh)}));
    emit(json!({"event":"log","msg":format!(
        "AE enc_hidden {:?} (latent {}), ae_epochs {}, ae_lr {}; clf_hidden {:?}, freeze_encoder {}, {} activation, dropout {}, lr {}, batch {}, {} epochs, {} classes {:?}",
        hp.enc_hidden, hp.enc_hidden.last().unwrap(), hp.ae_epochs, hp.ae_lr, hp.clf_hidden,
        hp.freeze_encoder, hp.activation.name(), hp.dropout, hp.lr, hp.batch_size, hp.epochs, k, classes)}));

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
    let fit_oh = gather(&ds.features, n_cols, &fit_idx, &layout.oh_idx);
    let fit_y: Vec<i64> = fit_idx.iter().map(|&r| y_all[r as usize]).collect();

    let mut val_cont = gather(&ds.features, n_cols, &val_idx, &layout.cont_idx);
    let val_oh = gather(&ds.features, n_cols, &val_idx, &layout.oh_idx);
    let val_y: Vec<i64> = val_idx.iter().map(|&r| y_all[r as usize]).collect();

    let mut test_cont = gather(&ds.features, n_cols, &ds.test_idx, &layout.cont_idx);
    let test_oh = gather(&ds.features, n_cols, &ds.test_idx, &layout.oh_idx);
    let test_y: Vec<i64> = ds.test_idx.iter().map(|&r| y_all[r as usize]).collect();

    let scaler = ContScaler::fit(&fit_cont, n_cont);
    scaler.transform(&mut fit_cont);
    scaler.transform(&mut val_cont);
    scaler.transform(&mut test_cont);
    scaler.save(&run_dir.join("scaler.json"))?;

    let arch = arch_from(&hp, n_cont, n_oh, k);
    let d = n_cont + n_oh;
    let total_epochs = hp.ae_epochs + hp.epochs;

    // Phase 1: pretrain the autoencoder on the fit-split feature vector.
    let x_fit = interleave(&fit_cont, n_cont, &fit_oh, n_oh);
    let ae = model::run_pretrain(
        &x_fit,
        model::PretrainConfig {
            arch: arch.clone(),
            d,
            epochs: hp.ae_epochs,
            lr: hp.ae_lr,
            batch_size: hp.batch_size,
            seed: SEED,
            base_epoch: 0,
            total_epochs,
        },
        &run_dir,
        &emit,
    )?;
    emit(json!({"event":"log","msg":"autoencoder pretraining done; transplanting encoder"}));

    // Phase 2: fine-tune the transplanted encoder + classifier head.
    let trained = model::run_finetune(
        ae.into_encoder(),
        model::TrainInputs {
            train_cont: &fit_cont,
            train_oh: &fit_oh,
            train_y: &fit_y,
            val_cont: &val_cont,
            val_oh: &val_oh,
            val_y: &val_y,
        },
        model::FinetuneConfig {
            arch: arch.clone(),
            n_cont,
            n_oh,
            epochs: hp.epochs,
            lr: hp.lr,
            batch_size: hp.batch_size,
            seed: SEED,
            patience: hp.patience,
            checkpoint_every: hp.checkpoint_every,
            base_epoch: hp.ae_epochs,
            total_epochs,
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
    let n_cont = layout.cont_idx.len();
    let n_oh = layout.oh_idx.len();

    let all_rows: Vec<u32> = (0..input.manifest.n_rows as u32).collect();
    let mut cont = gather(&input.features, n_cols, &all_rows, &layout.cont_idx);
    let oh = gather(&input.features, n_cols, &all_rows, &layout.oh_idx);
    let scaler = ContScaler::load(&model_dir.join("scaler.json"))?;
    anyhow::ensure!(scaler.mean.len() == n_cont, "scaler/continuous-width mismatch");
    scaler.transform(&mut cont);

    let arch = arch_from(&hp, n_cont, n_oh, k);
    let m = model::load(&model_dir.join("model"), &arch)?;
    let proba = model::predict_proba_loaded(&m, &cont, &oh, n_cont, n_oh, hp.batch_size, k);
    emit(json!({"event":"log","msg":format!(
        "predicting {} rows, {k} classes", input.manifest.n_rows)}));

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
    oh_idx: Vec<usize>,
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

    let arch = arch_from(&hp, lf.cont_idx.len(), lf.oh_idx.len(), k);
    let m = model::load(&model_dir.join("model"), &arch)?;
    let bundle = m.export_bundle();
    onnx_export::write_onnx(
        &lf.cont_idx, &lf.oh_idx, &scaler.mean, &scaler.std, &bundle, lf.n_cols, &output,
    )?;
    emit(json!({"event":"log","msg":format!(
        "exported model.onnx: burn-ae-classifier, {} cols, binary P(class=1)", lf.n_cols)}));
    emit(json!({"event":"done"}));
    Ok(())
}
