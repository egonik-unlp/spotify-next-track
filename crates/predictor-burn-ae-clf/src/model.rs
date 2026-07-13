//! Autoencoder-pretrained deep classifier (burn 0.21, ndarray CPU, hand-written
//! training loops).
//!
//! The feature matrix is split into a standardized continuous block (PCA +
//! numerics) and raw 0/1 one-hots; the encoder input is the concatenation
//! `x = [cont | oh]` of width `D = n_cont + n_oh`.
//!
//! Phase 1 — an `Autoencoder` (encoder + mirror decoder) is trained to
//! reconstruct `x` (MSE, unsupervised). Phase 2 — the trained encoder is
//! transplanted into an `AeClf` (encoder → clf head → K logits), fine-tuned with
//! softmax cross-entropy. `freeze_encoder` detaches the latent so the
//! transplanted weights stay frozen; otherwise the whole network fine-tunes.

use std::path::Path;

use anyhow::Result;
use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::backend::Autodiff;
use burn::module::{AutodiffModule, Module};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::record::CompactRecorder;
use burn::tensor::backend::Backend;
use burn::tensor::{activation, Int, Tensor, TensorData};

use lensing_core::shuffle::SplitMix64;

pub type Inner = NdArray<f32>;
pub type Auto = Autodiff<Inner>;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Activation {
    #[default]
    Relu,
    Gelu,
    Silu,
    Tanh,
}

impl Activation {
    pub fn name(self) -> &'static str {
        match self {
            Activation::Relu => "ReLU",
            Activation::Gelu => "GELU",
            Activation::Silu => "SiLU",
            Activation::Tanh => "Tanh",
        }
    }
    fn apply<B: Backend>(self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        match self {
            Activation::Relu => activation::relu(x),
            Activation::Gelu => activation::gelu(x),
            Activation::Silu => activation::silu(x),
            Activation::Tanh => activation::tanh(x),
        }
    }
}

/// Static description of the column layout + architecture, recomputed
/// identically at train / predict / export from the manifest + hyperparams.
#[derive(Clone, Debug)]
pub struct AeArch {
    pub n_cont: usize,
    pub n_oh: usize,
    /// Encoder layer widths; the last entry is the latent (bottleneck) dim.
    pub enc_hidden: Vec<usize>,
    /// Classifier-head layers applied to the latent before the K-logit output.
    pub clf_hidden: Vec<usize>,
    pub k: usize,
    pub ae_dropout: f64,
    /// Classifier-head dropout.
    pub dropout: f64,
    pub activation: Activation,
    pub freeze_encoder: bool,
}

impl AeArch {
    fn d(&self) -> usize {
        self.n_cont + self.n_oh
    }
}

fn build_encoder<B: Backend>(cfg: &AeArch, device: &B::Device) -> Vec<Linear<B>> {
    let mut encoder = Vec::with_capacity(cfg.enc_hidden.len());
    let mut prev = cfg.d();
    for &h in &cfg.enc_hidden {
        encoder.push(LinearConfig::new(prev, h).init(device));
        prev = h;
    }
    encoder
}

// ----------------------------------------------------------------------------
// Phase 1: Autoencoder (encoder + mirror decoder), unsupervised reconstruction.
// ----------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct Autoencoder<B: Backend> {
    encoder: Vec<Linear<B>>,
    decoder: Vec<Linear<B>>,
    dropout: Dropout,
    #[module(skip)]
    activation: Activation,
}

impl<B: Backend> Autoencoder<B> {
    pub fn new(cfg: &AeArch, device: &B::Device) -> Self {
        let encoder = build_encoder::<B>(cfg, device);
        // Decoder mirrors the encoder: latent → reversed hidden (minus the
        // bottleneck) → D.
        let mut decoder = Vec::new();
        let mut prev = *cfg.enc_hidden.last().expect("enc_hidden non-empty");
        let mut rev: Vec<usize> = cfg.enc_hidden.iter().rev().skip(1).copied().collect();
        rev.push(cfg.d());
        for &h in &rev {
            decoder.push(LinearConfig::new(prev, h).init(device));
            prev = h;
        }
        Autoencoder {
            encoder,
            decoder,
            dropout: DropoutConfig::new(cfg.ae_dropout).init(),
            activation: cfg.activation,
        }
    }

    /// Encode with activation + dropout after every encoder layer (incl. latent).
    fn encode(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut h = x;
        for layer in &self.encoder {
            h = self.dropout.forward(self.activation.apply(layer.forward(h)));
        }
        h
    }

    /// Reconstruct `x`: encode then decode (linear output on the final layer).
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut h = self.encode(x);
        let n = self.decoder.len();
        for (i, layer) in self.decoder.iter().enumerate() {
            h = layer.forward(h);
            if i + 1 < n {
                h = self.activation.apply(h);
            }
        }
        h
    }

    /// Consume the trained AE, yielding the encoder for transplant.
    pub fn into_encoder(self) -> Vec<Linear<B>> {
        self.encoder
    }
}

// ----------------------------------------------------------------------------
// Phase 2: AeClf (transplanted encoder → clf head → K logits), supervised CE.
// ----------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct AeClf<B: Backend> {
    /// Transplanted encoder (`D → latent`); same topology as the AE encoder.
    encoder: Vec<Linear<B>>,
    /// Classifier-head hidden layers over the latent.
    head: Vec<Linear<B>>,
    /// Final layer: last head width (or latent) → K logits.
    out: Linear<B>,
    dropout: Dropout,
    #[module(skip)]
    activation: Activation,
    #[module(skip)]
    freeze_encoder: bool,
}

impl<B: Backend> AeClf<B> {
    pub fn new(cfg: &AeArch, device: &B::Device) -> Self {
        let encoder = build_encoder::<B>(cfg, device);
        let mut prev = *cfg.enc_hidden.last().expect("enc_hidden non-empty");
        let mut head = Vec::with_capacity(cfg.clf_hidden.len());
        for &h in &cfg.clf_hidden {
            head.push(LinearConfig::new(prev, h).init(device));
            prev = h;
        }
        let out = LinearConfig::new(prev, cfg.k).init(device);
        AeClf {
            encoder,
            head,
            out,
            dropout: DropoutConfig::new(cfg.dropout).init(),
            activation: cfg.activation,
            freeze_encoder: cfg.freeze_encoder,
        }
    }

    /// Replace the freshly-initialized encoder with transplanted weights.
    pub fn with_encoder(mut self, encoder: Vec<Linear<B>>) -> Self {
        self.encoder = encoder;
        self
    }

    /// Encode with activation after every encoder layer (no dropout — the
    /// encoder is a feature extractor; matches the AE's eval-mode encode).
    fn encode(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut h = x;
        for layer in &self.encoder {
            h = self.activation.apply(layer.forward(h));
        }
        h
    }

    /// `cont`: [n, n_cont] standardized continuous block. `oh`: [n, n_oh] raw
    /// one-hots. Returns [n, K] logits.
    pub fn forward(&self, cont: Tensor<B, 2>, oh: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = Tensor::cat(vec![cont, oh], 1);
        let mut h = self.encode(x);
        if self.freeze_encoder {
            h = h.detach();
        }
        for layer in &self.head {
            h = self.dropout.forward(self.activation.apply(layer.forward(h)));
        }
        self.out.forward(h)
    }
}

// ----------------------------------------------------------------------------
// Batched feature/label staging (shared by both phases).
// ----------------------------------------------------------------------------

fn slab<B: Backend>(x: &[f32], rows: &[usize], w: usize, device: &B::Device) -> Tensor<B, 2> {
    if w == 0 {
        return Tensor::zeros([rows.len(), 0], device);
    }
    let mut flat = Vec::with_capacity(rows.len() * w);
    for &r in rows {
        flat.extend_from_slice(&x[r * w..(r + 1) * w]);
    }
    Tensor::from_data(TensorData::new(flat, [rows.len(), w]), device)
}

fn labels<B: Backend>(y: &[i64], rows: &[usize], device: &B::Device) -> Tensor<B, 1, Int> {
    let v: Vec<i64> = rows.iter().map(|&r| y[r]).collect();
    Tensor::from_data(TensorData::new(v, [rows.len()]), device)
}

// ----------------------------------------------------------------------------
// Phase 1 trainer.
// ----------------------------------------------------------------------------

pub struct PretrainConfig {
    pub arch: AeArch,
    pub d: usize,
    pub epochs: usize,
    pub lr: f64,
    pub batch_size: usize,
    pub seed: u64,
    /// Cumulative-counter base (epochs already emitted before this phase).
    pub base_epoch: usize,
    pub total_epochs: usize,
}

/// Train the autoencoder on the fit-split feature block `x_fit` (row-major,
/// width `d`). Honors the STOP file; emits `phase=pretrain` epoch events.
pub fn run_pretrain(
    x_fit: &[f32],
    cfg: PretrainConfig,
    run_dir: &Path,
    emit: &dyn Fn(serde_json::Value),
) -> Result<Autoencoder<Auto>> {
    let device = NdArrayDevice::default();
    Auto::seed(&device, cfg.seed);

    let n = x_fit.len() / cfg.d;
    let mut ae: Autoencoder<Auto> = Autoencoder::new(&cfg.arch, &device);
    let mut optimizer = AdamConfig::new().init();

    let mut order: Vec<usize> = (0..n).collect();
    let mut rng = SplitMix64(cfg.seed ^ 0x5151_5151_5151_5151);

    for epoch in 1..=cfg.epochs {
        if run_dir.join("STOP").exists() {
            emit(serde_json::json!({"event":"stopping"}));
            break;
        }
        for i in (1..n).rev() {
            let j = (rng.next_u64() % (i as u64 + 1)) as usize;
            order.swap(i, j);
        }

        let mut loss_sum = 0.0f64;
        for batch in order.chunks(cfg.batch_size) {
            let xb = slab::<Auto>(x_fit, batch, cfg.d, &device);
            let recon = ae.forward(xb.clone());
            let diff = recon - xb;
            let loss = (diff.clone() * diff).mean();
            let lv = loss.clone().into_data().to_vec::<f32>().unwrap()[0] as f64;
            loss_sum += lv * batch.len() as f64;
            let grads = GradientsParams::from_grads(loss.backward(), &ae);
            ae = optimizer.step(cfg.lr, ae, grads);
        }
        let train_loss = loss_sum / n as f64;
        emit(serde_json::json!({
            "event":"epoch","phase":"pretrain","epoch":cfg.base_epoch + epoch,
            "total_epochs":cfg.total_epochs,"train_loss":train_loss,"val_loss":train_loss}));
    }
    Ok(ae)
}

// ----------------------------------------------------------------------------
// Phase 2 trainer.
// ----------------------------------------------------------------------------

pub struct TrainInputs<'a> {
    pub train_cont: &'a [f32],
    pub train_oh: &'a [f32],
    pub train_y: &'a [i64],
    pub val_cont: &'a [f32],
    pub val_oh: &'a [f32],
    pub val_y: &'a [i64],
}

pub struct FinetuneConfig {
    pub arch: AeArch,
    pub n_cont: usize,
    pub n_oh: usize,
    pub epochs: usize,
    pub lr: f64,
    pub batch_size: usize,
    pub seed: u64,
    pub patience: usize,
    pub checkpoint_every: usize,
    pub base_epoch: usize,
    pub total_epochs: usize,
}

/// Build the classifier from the transplanted `encoder`, then fine-tune with
/// softmax cross-entropy. Mirrors the deep-clf training loop: STOP file,
/// checkpointing, and best-epoch early-stop on val logloss.
pub fn run_finetune(
    encoder: Vec<Linear<Auto>>,
    inputs: TrainInputs<'_>,
    cfg: FinetuneConfig,
    run_dir: &Path,
    emit: &dyn Fn(serde_json::Value),
) -> Result<AeClf<Auto>> {
    let device = NdArrayDevice::default();
    Auto::seed(&device, cfg.seed ^ 0xBEEF_BEEF_BEEF_BEEF);

    let n_train = inputs.train_y.len();
    let mut model: AeClf<Auto> = AeClf::new(&cfg.arch, &device).with_encoder(encoder);
    let mut optimizer = AdamConfig::new().init();
    let loss_fn = CrossEntropyLossConfig::new().init(&device);

    let mut order: Vec<usize> = (0..n_train).collect();
    let mut rng = SplitMix64(cfg.seed ^ 0xA5A5_A5A5);
    let monitor = !inputs.val_y.is_empty();
    let mut best: Option<(f64, usize, AeClf<Auto>)> = None;

    for epoch in 1..=cfg.epochs {
        if run_dir.join("STOP").exists() {
            emit(serde_json::json!({"event":"stopping"}));
            break;
        }
        for i in (1..n_train).rev() {
            let j = (rng.next_u64() % (i as u64 + 1)) as usize;
            order.swap(i, j);
        }

        let mut loss_sum = 0.0f64;
        for batch in order.chunks(cfg.batch_size) {
            let cont = slab::<Auto>(inputs.train_cont, batch, cfg.n_cont, &device);
            let oh = slab::<Auto>(inputs.train_oh, batch, cfg.n_oh, &device);
            let y = labels::<Auto>(inputs.train_y, batch, &device);
            let logits = model.forward(cont, oh);
            let loss = loss_fn.forward(logits, y);
            let lv = loss.clone().into_data().to_vec::<f32>().unwrap()[0] as f64;
            loss_sum += lv * batch.len() as f64;
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optimizer.step(cfg.lr, model, grads);
        }
        let train_loss = loss_sum / n_train as f64;

        let val_loss = if monitor {
            let vm = model.valid();
            mean_logloss(&vm, inputs.val_cont, inputs.val_oh, inputs.val_y, &cfg)
        } else {
            train_loss
        };
        emit(serde_json::json!({
            "event":"epoch","phase":"finetune","epoch":cfg.base_epoch + epoch,
            "total_epochs":cfg.total_epochs,"train_loss":train_loss,"val_loss":val_loss}));

        if cfg.patience > 0 && best.as_ref().is_none_or(|(b, _, _)| val_loss < *b) {
            best = Some((val_loss, epoch, model.clone()));
        }
        if cfg.checkpoint_every > 0 && epoch % cfg.checkpoint_every == 0 && epoch < cfg.epochs {
            save_atomic(best.as_ref().map_or(&model, |(_, _, m)| m), run_dir)?;
            emit(serde_json::json!({"event":"checkpoint","epoch":cfg.base_epoch + epoch}));
        }
        if cfg.patience > 0 {
            if let Some((_, be, _)) = &best {
                if epoch - be >= cfg.patience {
                    emit(serde_json::json!({"event":"log","msg":format!(
                        "early stop at epoch {epoch}: no val improvement in {} epochs", cfg.patience)}));
                    break;
                }
            }
        }
    }

    if let Some((bv, be, bm)) = best {
        emit(serde_json::json!({"event":"log","msg":format!(
            "restoring best epoch {be} (val logloss {bv:.6})")}));
        return Ok(bm);
    }
    Ok(model)
}

/// Mean cross-entropy (logloss) on an eval-mode model over a split.
fn mean_logloss(
    model: &AeClf<Inner>,
    cont: &[f32],
    oh: &[f32],
    y: &[i64],
    cfg: &FinetuneConfig,
) -> f64 {
    let proba = predict_proba_inner(model, cont, oh, cfg.n_cont, cfg.n_oh, cfg.batch_size, cfg.arch.k);
    let n = y.len();
    let mut s = 0.0f64;
    for (i, &lab) in y.iter().enumerate() {
        let p = (proba[i * cfg.arch.k + lab as usize] as f64).clamp(1e-15, 1.0 - 1e-15);
        s -= p.ln();
    }
    s / n as f64
}

fn predict_proba_inner(
    model: &AeClf<Inner>,
    cont: &[f32],
    oh: &[f32],
    n_cont: usize,
    n_oh: usize,
    batch_size: usize,
    k: usize,
) -> Vec<f32> {
    let device = NdArrayDevice::default();
    let n = if n_cont > 0 { cont.len() / n_cont } else { oh.len() / n_oh.max(1) };
    let rows: Vec<usize> = (0..n).collect();
    let mut out = Vec::with_capacity(n * k);
    for batch in rows.chunks(batch_size.max(1)) {
        let c = slab::<Inner>(cont, batch, n_cont, &device);
        let o = slab::<Inner>(oh, batch, n_oh, &device);
        let logits = model.forward(c, o);
        let proba = activation::softmax(logits, 1);
        out.extend(proba.into_data().to_vec::<f32>().unwrap());
    }
    out
}

/// Public proba predictor (eval mode), row-major [n, k].
pub fn predict_proba(
    model: &AeClf<Auto>,
    cont: &[f32],
    oh: &[f32],
    n_cont: usize,
    n_oh: usize,
    batch_size: usize,
    k: usize,
) -> Vec<f32> {
    predict_proba_inner(&model.valid(), cont, oh, n_cont, n_oh, batch_size, k)
}

pub fn predict_proba_loaded(
    model: &AeClf<Inner>,
    cont: &[f32],
    oh: &[f32],
    n_cont: usize,
    n_oh: usize,
    batch_size: usize,
    k: usize,
) -> Vec<f32> {
    predict_proba_inner(model, cont, oh, n_cont, n_oh, batch_size, k)
}

pub fn save_atomic(model: &AeClf<Auto>, dir: &Path) -> Result<()> {
    let tmp = dir.join("model-tmp");
    model
        .valid()
        .save_file(tmp.to_string_lossy().as_ref(), &CompactRecorder::new())
        .map_err(|e| anyhow::anyhow!("save checkpoint: {e}"))?;
    std::fs::rename(dir.join("model-tmp.mpk"), dir.join("model.mpk"))?;
    Ok(())
}

pub fn load(path: &Path, cfg: &AeArch) -> Result<AeClf<Inner>> {
    let device = NdArrayDevice::default();
    AeClf::<Inner>::new(cfg, &device)
        .load_file(path.to_string_lossy().as_ref(), &CompactRecorder::new(), &device)
        .map_err(|e| anyhow::anyhow!("load checkpoint {}: {e}", path.display()))
}

// ----------------------------------------------------------------------------
// ONNX export bundle.
// ----------------------------------------------------------------------------

/// One dense layer's parameters. `weight` is row-major `[w_in, w_out]` (burn's
/// `Linear` layout → ONNX `Gemm` with no transpose).
pub struct ExportLayer {
    pub weight: Vec<f32>,
    pub w_in: usize,
    pub w_out: usize,
    pub bias: Option<Vec<f32>>,
}

/// Everything the ONNX exporter needs from the trained classifier weights.
pub struct ExportBundle {
    pub activation: Activation,
    /// Encoder layers (`D → latent`), each followed by an activation.
    pub encoder: Vec<ExportLayer>,
    /// Classifier-head hidden layers, each followed by an activation.
    pub head: Vec<ExportLayer>,
    /// Final layer → K logits (no trailing activation).
    pub out: ExportLayer,
}

impl AeClf<Inner> {
    pub fn export_bundle(&self) -> ExportBundle {
        let ex = |l: &Linear<Inner>| {
            let [w_in, w_out] = l.weight.dims();
            ExportLayer {
                weight: l.weight.val().into_data().to_vec::<f32>().unwrap(),
                w_in,
                w_out,
                bias: l.bias.as_ref().map(|b| b.val().into_data().to_vec::<f32>().unwrap()),
            }
        };
        ExportBundle {
            activation: self.activation,
            encoder: self.encoder.iter().map(ex).collect(),
            head: self.head.iter().map(ex).collect(),
            out: ex(&self.out),
        }
    }
}
