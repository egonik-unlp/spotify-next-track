//! Export a trained `AeClf` to a portable `model.onnx`.
//!
//! The graph's input is the *assembled feature vector* (the same matrix the
//! server's featurizer produces), so the input contract matches every other
//! predictor family. Everything the native model does is baked in:
//!
//!   input[N, n_cols]
//!     → Gather(cont_idx) → Sub(mean) → Div(std)   (continuous block, standardized)
//!     → Gather(oh_idx)                             (raw one-hots)
//!     → Concat([scont, oh])                        (encoder input, width D)
//!     → Gemm·act · …  (encoder → latent)
//!     → Gemm·act · …  (classifier head)
//!     → Gemm (out → [N, K] logits)
//!     → Softmax → Gather(class 1) → output[N, 1] = P(class 1)
//!
//! Binary only: the `[N, 1]` positive-class probability matches the regression
//! bundle contract (`featurize.json` transform `none`).

use std::path::Path;

use anyhow::{Context, Result};
use lensing_onnx::{Activation as OnnxAct, Attr, Dim, GraphBuilder};

use crate::model::{Activation, ExportBundle};

fn map_act(a: Activation) -> OnnxAct {
    match a {
        Activation::Relu => OnnxAct::Relu,
        Activation::Gelu => OnnxAct::Gelu,
        Activation::Silu => OnnxAct::Silu,
        Activation::Tanh => OnnxAct::Tanh,
    }
}

/// `Gather(input, idx, axis=1)` → the selected columns as a fresh tensor.
fn gather_cols(g: &mut GraphBuilder, src: &str, idx: &[usize]) -> String {
    let i64s: Vec<i64> = idx.iter().map(|&i| i as i64).collect();
    let name = g.add_int_tensor(vec![i64s.len() as i64], i64s);
    g.op("Gather", &[src, &name], vec![("axis".into(), Attr::Int(1))])
}

/// Build and write `<output_dir>/model.onnx`. `cont_idx` / `oh_idx` are the
/// column indices of the continuous block and the one-hots (canonical order),
/// matching the `[cont | oh]` concatenation the native encoder consumes.
pub fn write_onnx(
    cont_idx: &[usize],
    oh_idx: &[usize],
    cont_mean: &[f32],
    cont_std: &[f32],
    bundle: &ExportBundle,
    n_cols: usize,
    output_dir: &Path,
) -> Result<()> {
    let mut g = GraphBuilder::new();
    let n_cont = cont_idx.len();

    // Continuous block: Gather then standardize (x - mean) / std.
    let cont = gather_cols(&mut g, "input", cont_idx);
    g.add_weight("cont_mean", vec![n_cont as i64], cont_mean.to_vec());
    g.add_weight("cont_std", vec![n_cont as i64], cont_std.to_vec());
    let centered = g.op("Sub", &[&cont, "cont_mean"], vec![]);
    let scont = g.op("Div", &[&centered, "cont_std"], vec![]);

    // Encoder input = concat(standardized continuous, raw one-hots).
    let trunk_in = if oh_idx.is_empty() {
        scont
    } else {
        let oh = gather_cols(&mut g, "input", oh_idx);
        g.op("Concat", &[&scont, &oh], vec![("axis".into(), Attr::Int(1))])
    };

    let act = map_act(bundle.activation);
    let mut cur = trunk_in;

    // Encoder: Gemm·act per layer (every layer activated, incl. the latent).
    for (i, l) in bundle.encoder.iter().enumerate() {
        cur = g.gemm(
            &cur, &format!("enc_w{i}"), l.weight.clone(), l.w_in, l.w_out,
            &format!("enc_b{i}"), l.bias.clone(),
        );
        cur = g.activation(&cur, act);
    }
    // Classifier head: Gemm·act per layer.
    for (i, l) in bundle.head.iter().enumerate() {
        cur = g.gemm(
            &cur, &format!("head_w{i}"), l.weight.clone(), l.w_in, l.w_out,
            &format!("head_b{i}"), l.bias.clone(),
        );
        cur = g.activation(&cur, act);
    }
    // Output layer → K logits (no trailing activation).
    let o = &bundle.out;
    let logits = g.gemm(&cur, "out_w", o.weight.clone(), o.w_in, o.w_out, "out_b", o.bias.clone());

    // Softmax over K logits, then gather the positive class → [N, 1].
    let proba = g.op("Softmax", &[&logits], vec![("axis".into(), Attr::Int(1))]);
    let pos = g.add_int_tensor(vec![1], vec![1]);
    let out = g.op("Gather", &[&proba, &pos], vec![("axis".into(), Attr::Int(1))]);

    let bytes = g.build(
        "input",
        &[Dim::Param("N".into()), Dim::Value(n_cols as i64)],
        &out,
        &[Dim::Param("N".into()), Dim::Value(1)],
        concat!("lensing/predictor-burn-ae-clf ", env!("CARGO_PKG_VERSION")),
    );
    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("create export dir {}", output_dir.display()))?;
    std::fs::write(output_dir.join("model.onnx"), &bytes)
        .with_context(|| format!("write {}/model.onnx", output_dir.display()))?;
    Ok(())
}
