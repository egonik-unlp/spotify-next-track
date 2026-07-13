//! Export a trained DeepClf to a portable `model.onnx`.
//!
//! The graph's input is the *assembled feature vector* (the same matrix the
//! server's featurizer produces), so the input contract is identical across
//! every predictor family. Everything the native model does is baked in:
//!
//!   input[N, n_cols]
//!     → Gather(cont_idx) → Sub(mean) → Div(std)         (continuous block, standardized)
//!     ┊  (mlp)       Gather(oh_idx) ───────────────┐
//!     ┊  (embeddings) per group: Gather → MatMul(W) ┤  concat with the continuous block
//!     ┊  (wide_deep) Gather(oh_idx) → Gemm(wide) ───┘  (summed at the logits instead)
//!     → Gemm·activation · … → Gemm (head → [N, K] logits)
//!     → Softmax → Gather(class 1) → output[N, 1] = P(class 1)
//!
//! Binary only: the `[N, 1]` positive-class probability matches the regression
//! bundle contract (`featurize.json` transform `none`, non-negative clamp a
//! no-op on a probability).

use std::path::Path;

use anyhow::{Context, Result};
use lensing_onnx::{Activation as OnnxAct, Attr, Dim, GraphBuilder};

use crate::model::{Activation, ExportBundle, Topology};

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

/// Build and write `<output_dir>/model.onnx`. `groups` holds each one-hot
/// group's column indices in canonical order (aligned with `bundle.embeds` /
/// `bundle.group_lens`).
pub fn write_onnx(
    cont_idx: &[usize],
    groups: &[Vec<usize>],
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

    let all_oh: Vec<usize> = groups.iter().flatten().copied().collect();
    let mut wide_logits: Option<String> = None;

    let trunk_in = match bundle.topology {
        Topology::Mlp => {
            let oh = gather_cols(&mut g, "input", &all_oh);
            g.op("Concat", &[&scont, &oh], vec![("axis".into(), Attr::Int(1))])
        }
        Topology::Embeddings => {
            let mut parts: Vec<String> = vec![scont.clone()];
            for (gi, gcols) in groups.iter().enumerate() {
                let og = gather_cols(&mut g, "input", gcols);
                let l = &bundle.embeds[gi];
                let wname = format!("embed_w{gi}");
                g.add_weight(&wname, vec![l.w_in as i64, l.w_out as i64], l.weight.clone());
                parts.push(g.op("MatMul", &[&og, &wname], vec![]));
            }
            let refs: Vec<&str> = parts.iter().map(|s| s.as_str()).collect();
            g.op("Concat", &refs, vec![("axis".into(), Attr::Int(1))])
        }
        Topology::WideDeep => {
            let oh = gather_cols(&mut g, "input", &all_oh);
            let w = bundle
                .wide
                .as_ref()
                .context("wide_deep export needs a wide layer")?;
            wide_logits = Some(g.gemm(
                &oh, "wide_w", w.weight.clone(), w.w_in, w.w_out, "wide_b", w.bias.clone(),
            ));
            scont.clone()
        }
    };

    // Deep trunk (Gemm·activation per layer) then the head → K logits.
    let act = map_act(bundle.activation);
    let mut cur = trunk_in;
    for (i, l) in bundle.trunk.iter().enumerate() {
        cur = g.gemm(
            &cur, &format!("trunk_w{i}"), l.weight.clone(), l.w_in, l.w_out,
            &format!("trunk_b{i}"), l.bias.clone(),
        );
        cur = g.activation(&cur, act);
    }
    let h = &bundle.head;
    let mut logits = g.gemm(&cur, "head_w", h.weight.clone(), h.w_in, h.w_out, "head_b", h.bias.clone());
    if let Some(wl) = wide_logits {
        logits = g.op("Add", &[&logits, &wl], vec![]);
    }

    // Softmax over K logits, then gather the positive class → [N, 1].
    let proba = g.op("Softmax", &[&logits], vec![("axis".into(), Attr::Int(1))]);
    let pos = g.add_int_tensor(vec![1], vec![1]);
    let out = g.op("Gather", &[&proba, &pos], vec![("axis".into(), Attr::Int(1))]);

    let bytes = g.build(
        "input",
        &[Dim::Param("N".into()), Dim::Value(n_cols as i64)],
        &out,
        &[Dim::Param("N".into()), Dim::Value(1)],
        concat!("lensing/predictor-burn-deep-clf ", env!("CARGO_PKG_VERSION")),
    );
    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("create export dir {}", output_dir.display()))?;
    std::fs::write(output_dir.join("model.onnx"), &bytes)
        .with_context(|| format!("write {}/model.onnx", output_dir.display()))?;
    Ok(())
}
