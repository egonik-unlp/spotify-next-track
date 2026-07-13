//! Representation-aware deep classifier (burn 0.21, ndarray CPU, hand-written
//! training loop). Three topologies over a feature matrix split into a
//! continuous block (PCA latents + numerics, standardized) and one-hot identity
//! groups (raw 0/1):
//!
//! - `mlp`        — dense MLP over concat(continuous, all one-hots). The plain
//!                  baseline: one-hots fed raw into the first dense layer.
//! - `embeddings` — each one-hot group g is projected by a bias-free
//!                  Linear(|g| → emb_dim) (an entity embedding: one-hot @ W
//!                  selects a learned row), concatenated with the continuous
//!                  block, then an MLP. The fix for high-cardinality identities.
//! - `wide_deep`  — a wide linear path Linear(all one-hots → K) summed at the
//!                  logits with a deep MLP over the continuous block only.
//!
//! Loss is softmax cross-entropy in class space; the head emits K logits.

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
pub enum Topology {
    /// Dense MLP over concat(continuous, all raw one-hots).
    #[default]
    Mlp,
    /// Per-group entity embeddings + continuous → MLP.
    Embeddings,
    /// Wide linear over one-hots + deep MLP over continuous, summed at logits.
    WideDeep,
}

impl Topology {
    pub fn name(self) -> &'static str {
        match self {
            Topology::Mlp => "mlp",
            Topology::Embeddings => "embeddings",
            Topology::WideDeep => "wide_deep",
        }
    }
}

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
/// identically at train and predict from the manifest columns + hyperparams.
#[derive(Clone, Debug)]
pub struct ArchConfig {
    pub n_cont: usize,
    /// One-hot group lengths, in canonical (first-appearance) order.
    pub group_lens: Vec<usize>,
    /// Embedding width per group (Embeddings topology), aligned with group_lens.
    pub embed_dims: Vec<usize>,
    pub hidden: Vec<usize>,
    pub k: usize,
    pub dropout: f64,
    pub activation: Activation,
    pub topology: Topology,
}

impl ArchConfig {
    fn onehot_total(&self) -> usize {
        self.group_lens.iter().sum()
    }
    /// Input width of the deep trunk for this topology.
    fn trunk_in(&self) -> usize {
        match self.topology {
            Topology::Mlp => self.n_cont + self.onehot_total(),
            Topology::Embeddings => self.n_cont + self.embed_dims.iter().sum::<usize>(),
            Topology::WideDeep => self.n_cont,
        }
    }
}

#[derive(Module, Debug)]
pub struct DeepClf<B: Backend> {
    /// One bias-free Linear per one-hot group (Embeddings topology); empty otherwise.
    embeds: Vec<Linear<B>>,
    /// Deep-trunk hidden layers.
    trunk: Vec<Linear<B>>,
    /// Deep head: last hidden (or trunk_in if no hidden) → K logits.
    head: Linear<B>,
    /// Wide path Linear(onehot_total → K); len 1 for WideDeep, else 0.
    wide: Vec<Linear<B>>,
    dropout: Dropout,
    #[module(skip)]
    activation: Activation,
    #[module(skip)]
    topology: Topology,
    #[module(skip)]
    group_lens: Vec<usize>,
}

impl<B: Backend> DeepClf<B> {
    pub fn new(cfg: &ArchConfig, device: &B::Device) -> Self {
        let mut embeds = Vec::new();
        if cfg.topology == Topology::Embeddings {
            for (g, &len) in cfg.group_lens.iter().enumerate() {
                embeds.push(
                    LinearConfig::new(len, cfg.embed_dims[g]).with_bias(false).init(device),
                );
            }
        }
        let mut trunk = Vec::with_capacity(cfg.hidden.len());
        let mut prev = cfg.trunk_in();
        for &h in &cfg.hidden {
            trunk.push(LinearConfig::new(prev, h).init(device));
            prev = h;
        }
        let head = LinearConfig::new(prev, cfg.k).init(device);
        let wide = if cfg.topology == Topology::WideDeep {
            vec![LinearConfig::new(cfg.onehot_total(), cfg.k).init(device)]
        } else {
            Vec::new()
        };
        DeepClf {
            embeds,
            trunk,
            head,
            wide,
            dropout: DropoutConfig::new(cfg.dropout).init(),
            activation: cfg.activation,
            topology: cfg.topology,
            group_lens: cfg.group_lens.clone(),
        }
    }

    /// `cont`: [n, n_cont] standardized continuous block. `oh`: [n, onehot_total]
    /// raw one-hots in group order. Returns [n, K] logits.
    pub fn forward(&self, cont: Tensor<B, 2>, oh: Tensor<B, 2>) -> Tensor<B, 2> {
        let n = oh.dims()[0];
        let trunk_in = match self.topology {
            Topology::Mlp => Tensor::cat(vec![cont, oh.clone()], 1),
            Topology::Embeddings => {
                let mut parts = vec![cont];
                let mut off = 0usize;
                for (g, &len) in self.group_lens.iter().enumerate() {
                    let slice = oh.clone().slice([0..n, off..off + len]);
                    parts.push(self.embeds[g].forward(slice));
                    off += len;
                }
                Tensor::cat(parts, 1)
            }
            Topology::WideDeep => cont,
        };

        let mut x = trunk_in;
        for layer in &self.trunk {
            x = self.dropout.forward(self.activation.apply(layer.forward(x)));
        }
        let mut logits = self.head.forward(x);
        if self.topology == Topology::WideDeep {
            logits = logits + self.wide[0].forward(oh);
        }
        logits
    }

    /// Assemble the deep-trunk input for this topology (the same `trunk_in` the
    /// forward pass builds) and, for each trunk hidden layer, its post-activation
    /// output in forward order. Excludes the wide path, the head logits, and
    /// dropout (inference-inactive on the eval backend) — the representation
    /// stages a probe/SAE reads (Alain & Bengio 2016). Returns `(trunk_in, acts)`.
    pub fn forward_capture(&self, cont: Tensor<B, 2>, oh: Tensor<B, 2>) -> (Tensor<B, 2>, Vec<Tensor<B, 2>>) {
        let n = oh.dims()[0];
        let trunk_in = match self.topology {
            Topology::Mlp => Tensor::cat(vec![cont, oh.clone()], 1),
            Topology::Embeddings => {
                let mut parts = vec![cont];
                let mut off = 0usize;
                for (g, &len) in self.group_lens.iter().enumerate() {
                    let slice = oh.clone().slice([0..n, off..off + len]);
                    parts.push(self.embeds[g].forward(slice));
                    off += len;
                }
                Tensor::cat(parts, 1)
            }
            Topology::WideDeep => cont,
        };
        let mut acts = Vec::with_capacity(self.trunk.len());
        let mut x = trunk_in.clone();
        for layer in &self.trunk {
            x = self.activation.apply(layer.forward(x));
            acts.push(x.clone());
        }
        (trunk_in, acts)
    }
}

/// One captured stage: row-major `[n_rows, dim]` activations a probe/SAE fits on.
pub struct LayerActs {
    pub name: String,
    pub dim: usize,
    pub values: Vec<f32>,
}

/// Run a loaded model over the split feature blocks (`cont` already
/// scaler-standardized, `oh` raw one-hots, both row-major over the SAME rows)
/// and materialize the representation at each probe stage: stage 0 is the
/// assembled deep-trunk input, stages 1..=H are the post-activation outputs of
/// each trunk hidden layer. The head logits are the model's own output and are
/// scored separately. Powers the per-model SAE / layer-probe analysis.
pub fn capture_stage_activations(
    model: &DeepClf<Inner>,
    cont: &[f32],
    oh: &[f32],
    n_cont: usize,
    n_oh: usize,
    batch_size: usize,
) -> Vec<LayerActs> {
    let device = NdArrayDevice::default();
    let n = if n_cont > 0 { cont.len() / n_cont } else { oh.len() / n_oh.max(1) };
    let rows: Vec<usize> = (0..n).collect();
    let mut trunk_in: Vec<f32> = Vec::new();
    let mut trunk_in_dim = 0usize;
    let mut hidden: Vec<Vec<f32>> = Vec::new();
    let mut hidden_dims: Vec<usize> = Vec::new();
    for batch in rows.chunks(batch_size.max(1)) {
        let c = slab::<Inner>(cont, batch, n_cont, &device);
        let o = slab::<Inner>(oh, batch, n_oh, &device);
        let (ti, acts) = model.forward_capture(c, o);
        trunk_in_dim = ti.dims()[1];
        trunk_in.extend(ti.into_data().to_vec::<f32>().unwrap());
        if hidden.is_empty() {
            hidden = acts.iter().map(|_| Vec::new()).collect();
            hidden_dims = acts.iter().map(|a| a.dims()[1]).collect();
        }
        for (i, a) in acts.into_iter().enumerate() {
            hidden[i].extend(a.into_data().to_vec::<f32>().unwrap());
        }
    }
    let mut stages = Vec::with_capacity(hidden.len() + 1);
    stages.push(LayerActs { name: "input".into(), dim: trunk_in_dim, values: trunk_in });
    for (i, d) in hidden_dims.into_iter().enumerate() {
        stages.push(LayerActs {
            name: format!("hidden_{}", i + 1),
            dim: d,
            values: std::mem::take(&mut hidden[i]),
        });
    }
    stages
}

pub struct TrainInputs<'a> {
    pub train_cont: &'a [f32],
    pub train_oh: &'a [f32],
    pub train_y: &'a [i64],
    pub val_cont: &'a [f32],
    pub val_oh: &'a [f32],
    pub val_y: &'a [i64],
}

pub struct TrainConfig {
    pub arch: ArchConfig,
    pub n_cont: usize,
    pub n_oh: usize,
    pub epochs: usize,
    pub lr: f64,
    pub batch_size: usize,
    pub seed: u64,
    pub patience: usize,
    pub checkpoint_every: usize,
}

fn slab<B: Backend>(x: &[f32], rows: &[usize], w: usize, device: &B::Device) -> Tensor<B, 2> {
    if w == 0 {
        // A zero-width block still needs the right row count for cat/slice.
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

pub fn run_training(
    inputs: TrainInputs<'_>,
    cfg: TrainConfig,
    run_dir: &Path,
    emit: &dyn Fn(serde_json::Value),
) -> Result<DeepClf<Auto>> {
    let device = NdArrayDevice::default();
    Auto::seed(&device, cfg.seed);

    let n_train = inputs.train_y.len();
    let mut model: DeepClf<Auto> = DeepClf::new(&cfg.arch, &device);
    let mut optimizer = AdamConfig::new().init();
    let loss_fn = CrossEntropyLossConfig::new().init(&device);

    let mut order: Vec<usize> = (0..n_train).collect();
    let mut rng = SplitMix64(cfg.seed ^ 0xA5A5_A5A5);
    let monitor = !inputs.val_y.is_empty();
    let mut best: Option<(f64, usize, DeepClf<Auto>)> = None;

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
            "event":"epoch","epoch":epoch,"total_epochs":cfg.epochs,
            "train_loss":train_loss,"val_loss":val_loss}));

        if cfg.patience > 0 && best.as_ref().is_none_or(|(b, _, _)| val_loss < *b) {
            best = Some((val_loss, epoch, model.clone()));
        }
        if cfg.checkpoint_every > 0 && epoch % cfg.checkpoint_every == 0 && epoch < cfg.epochs {
            save_atomic(best.as_ref().map_or(&model, |(_, _, m)| m), run_dir)?;
            emit(serde_json::json!({"event":"checkpoint","epoch":epoch}));
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
    model: &DeepClf<Inner>,
    cont: &[f32],
    oh: &[f32],
    y: &[i64],
    cfg: &TrainConfig,
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
    model: &DeepClf<Inner>,
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
    model: &DeepClf<Auto>,
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
    model: &DeepClf<Inner>,
    cont: &[f32],
    oh: &[f32],
    n_cont: usize,
    n_oh: usize,
    batch_size: usize,
    k: usize,
) -> Vec<f32> {
    predict_proba_inner(model, cont, oh, n_cont, n_oh, batch_size, k)
}

pub fn save_atomic(model: &DeepClf<Auto>, dir: &Path) -> Result<()> {
    let tmp = dir.join("model-tmp");
    model
        .valid()
        .save_file(tmp.to_string_lossy().as_ref(), &CompactRecorder::new())
        .map_err(|e| anyhow::anyhow!("save checkpoint: {e}"))?;
    std::fs::rename(dir.join("model-tmp.mpk"), dir.join("model.mpk"))?;
    Ok(())
}

pub fn load(path: &Path, cfg: &ArchConfig) -> Result<DeepClf<Inner>> {
    let device = NdArrayDevice::default();
    DeepClf::<Inner>::new(cfg, &device)
        .load_file(path.to_string_lossy().as_ref(), &CompactRecorder::new(), &device)
        .map_err(|e| anyhow::anyhow!("load checkpoint {}: {e}", path.display()))
}

/// One dense layer's parameters, extracted for ONNX export. `weight` is
/// row-major `[w_in, w_out]` (burn's `Linear` layout → ONNX `Gemm`/`MatMul`
/// with no transpose). `bias` is `None` for the bias-free embedding projections.
pub struct ExportLayer {
    pub weight: Vec<f32>,
    pub w_in: usize,
    pub w_out: usize,
    pub bias: Option<Vec<f32>>,
}

/// Everything the ONNX exporter needs from the trained weights.
pub struct ExportBundle {
    pub topology: Topology,
    pub activation: Activation,
    /// One bias-free projection per one-hot group (Embeddings topology only).
    pub embeds: Vec<ExportLayer>,
    /// Deep-trunk hidden layers (Mlp/Embeddings: over the assembled trunk input;
    /// WideDeep: over the continuous block).
    pub trunk: Vec<ExportLayer>,
    /// Deep head → K logits.
    pub head: ExportLayer,
    /// Wide path Linear(onehot_total → K), present only for WideDeep.
    pub wide: Option<ExportLayer>,
}

impl DeepClf<Inner> {
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
            topology: self.topology,
            activation: self.activation,
            embeds: self.embeds.iter().map(ex).collect(),
            trunk: self.trunk.iter().map(ex).collect(),
            head: ex(&self.head),
            wide: self.wide.first().map(ex),
        }
    }
}
