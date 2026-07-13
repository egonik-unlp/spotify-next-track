#!/usr/bin/env python3
"""Shared ONNX-export helpers for the Python predictors.

Every predictor's `export` subcommand writes a single `model.onnx` whose
**input is the assembled feature vector** — the exact matrix the lensing
server's featurizer produces (`features.f32`) — and whose **output is the
transformed-space target** (`[N, 1]`). That uniform I/O contract is what lets
an MLP, a gradient-boosted forest and an SVR all be consumed by one downstream
runtime; the inverse target transform lives in the export's `featurize.json`,
applied by the consumer, not here.

Predictors that standardize features before fitting (ridge, svm, kernel-ridge,
the neural nets) bake the standardizer `(x - mean) / std` into the head of the
graph, so the ONNX input stays the raw assembled vector regardless of family.
Tree ensembles are scale-invariant and need no prefix.

Two construction styles live here:
  * library conversion (`finalize`) for the tree models, via skl2onnx /
    onnxmltools, with the output renamed to the canonical `output`;
  * hand-built graphs (`linear_graph`, `kernel_graph`) for the JSON-serialized
    linear / kernel models, whose math the predictors already recompute in
    numpy — the ONNX mirrors that numpy exactly.

Import lazily from a predictor's `export()` (these pull in onnx/onnxmltools):
    import sys; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import onnx_common
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

INPUT = "input"
OUTPUT = "output"
OPSET = 13


def _f32(name: str, arr: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(np.ascontiguousarray(arr, dtype=np.float32), name)


def save(model: onnx.ModelProto, output_dir: Path) -> None:
    """Validate and write `<output_dir>/model.onnx`."""
    onnx.checker.check_model(model)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_dir / "model.onnx"))


def _model(nodes, inits, n_cols: int, out_dims, producer: str) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes,
        "lensing-export",
        [helper.make_tensor_value_info(INPUT, TensorProto.FLOAT, ["N", n_cols])],
        [helper.make_tensor_value_info(OUTPUT, TensorProto.FLOAT, out_dims)],
        inits,
    )
    return helper.make_model(
        graph,
        producer_name=producer,
        opset_imports=[helper.make_opsetid("", OPSET)],
        ir_version=7,
    )


def _standardizer(mean: np.ndarray, std: np.ndarray):
    """Nodes + initializers mapping `input` → `scaled` = (input - mean) / std."""
    inits = [_f32("scaler_mean", mean), _f32("scaler_std", std)]
    nodes = [
        helper.make_node("Sub", [INPUT, "scaler_mean"], ["centered"]),
        helper.make_node("Div", ["centered", "scaler_std"], ["scaled"]),
    ]
    return nodes, inits, "scaled"


# ---------- hand-built graphs (ridge / svm / kernel-ridge) ----------

def linear_graph(coef: np.ndarray, intercept: float, mean: np.ndarray,
                 std: np.ndarray) -> onnx.ModelProto:
    """Ridge: output = ((input - mean) / std) @ coef + intercept."""
    n_cols = len(mean)
    nodes, inits, x = _standardizer(mean, std)
    inits.append(_f32("coef", coef.reshape(n_cols, 1)))
    inits.append(_f32("intercept", np.asarray([intercept])))
    nodes.append(helper.make_node("MatMul", [x, "coef"], ["proj"]))
    nodes.append(helper.make_node("Add", ["proj", "intercept"], [OUTPUT]))
    return _model(nodes, inits, n_cols, ["N", 1], "lensing/ridge")


def kernel_graph(mean: np.ndarray, std: np.ndarray, basis: np.ndarray,
                 dual_coef: np.ndarray, bias: float, kernel: str, gamma: float,
                 degree: int, coef0: float) -> onnx.ModelProto:
    """SVR / kernel-ridge decision function over support vectors / basis rows:
        f(x) = K(scaled(x), basis) @ dual_coef + bias
    mirroring the predictors' numpy `decision_function` for every kernel."""
    n_cols = len(mean)
    m = basis.shape[0]
    nodes, inits, x = _standardizer(mean, std)
    inits.append(_f32("basis_t", basis.T))  # [F, M]
    nodes.append(helper.make_node("MatMul", [x, "basis_t"], ["xsv"]))  # [N, M]

    if kernel == "linear":
        kname = "xsv"
    elif kernel == "rbf":
        # ||x - sv||^2 = sum(x^2) - 2 x·sv + sum(sv^2), then exp(-gamma·max(·,0)).
        inits.append(_f32("sv2", np.sum(basis**2, axis=1).reshape(1, m)))
        inits.append(_f32("two", np.asarray([2.0])))
        inits.append(_f32("neg_gamma", np.asarray([-float(gamma)])))
        nodes += [
            helper.make_node("ReduceSumSquare", [x], ["x2"], axes=[1], keepdims=1),
            helper.make_node("Mul", ["xsv", "two"], ["two_xsv"]),
            helper.make_node("Sub", ["x2", "two_xsv"], ["sq0"]),
            helper.make_node("Add", ["sq0", "sv2"], ["sq1"]),
            helper.make_node("Relu", ["sq1"], ["sq"]),  # max(·, 0)
            helper.make_node("Mul", ["sq", "neg_gamma"], ["scaled_sq"]),
            helper.make_node("Exp", ["scaled_sq"], ["kmat"]),
        ]
        kname = "kmat"
    elif kernel in ("poly", "sigmoid"):
        inits.append(_f32("gamma", np.asarray([float(gamma)])))
        inits.append(_f32("coef0", np.asarray([float(coef0)])))
        nodes += [
            helper.make_node("Mul", ["xsv", "gamma"], ["g_xsv"]),
            helper.make_node("Add", ["g_xsv", "coef0"], ["affine"]),
        ]
        if kernel == "sigmoid":
            nodes.append(helper.make_node("Tanh", ["affine"], ["kmat"]))
        else:
            # Integer power via repeated multiply: exact for negative bases,
            # unlike Pow with a float exponent.
            prev = "affine"
            for i in range(int(degree) - 1):
                out = f"pow{i}"
                nodes.append(helper.make_node("Mul", [prev, "affine"], [out]))
                prev = out
            nodes.append(helper.make_node("Identity", [prev], ["kmat"]))
        kname = "kmat"
    else:
        raise ValueError(f"unknown kernel {kernel!r}")

    inits.append(_f32("dual_coef", dual_coef.reshape(m, 1)))
    inits.append(_f32("bias", np.asarray([float(bias)])))
    nodes.append(helper.make_node("MatMul", [kname, "dual_coef"], ["f"]))
    nodes.append(helper.make_node("Add", ["f", "bias"], [OUTPUT]))
    return _model(nodes, inits, n_cols, ["N", 1], f"lensing/{kernel}-kernel")


# ---------- library conversion (tree models) ----------

def finalize(model: onnx.ModelProto) -> onnx.ModelProto:
    """Normalize a skl2onnx / onnxmltools conversion to a single graph output
    named `output`. The converters are called with the input already named
    `input`. Some converters (e.g. lightgbm) already use `output` for an
    intermediate tensor, so we pick a collision-free name when needed — the
    exact name doesn't matter to consumers (they read `outputNames[0]`)."""
    g = model.graph
    assert len(g.output) == 1, f"expected one output, got {len(g.output)}"
    old = g.output[0].name
    # Every tensor name in use, excluding the output we're about to rename.
    used = set()
    for node in g.node:
        used.update(node.input)
        used.update(node.output)
    used.update(i.name for i in g.initializer)
    used.update(i.name for i in g.input)
    used.discard(old)
    target = OUTPUT
    if target in used:
        i = 0
        while f"{OUTPUT}_{i}" in used:
            i += 1
        target = f"{OUTPUT}_{i}"
    if old != target:
        for node in g.node:
            node.output[:] = [target if o == old else o for o in node.output]
            node.input[:] = [target if x == old else x for x in node.input]
        g.output[0].name = target
    onnx.checker.check_model(model)
    return model


def positive_class_proba(model: onnx.ModelProto) -> onnx.ModelProto:
    """Reduce a converted *binary* classifier to the single-output `[N, 1]`
    regression-style contract: `output` = P(class index 1), the positive-class
    probability (for the rotation target, the engagement probability).

    The skl2onnx / onnxmltools classifier converters emit two graph outputs — a
    hard `label` and a `[N, k]` `probabilities` tensor. We gather the positive
    column off the probabilities tensor into the canonical `output` and drop the
    label, so the bundle stays a single `[N, 1]` output like every regressor
    (consumers read `outputNames[0]`; `featurize.json` transform is `none`, and
    the non-negative clamp is a no-op on a probability).

    Multiclass is unsupported: an `[N, k>2]` output doesn't fit the `[N, 1]`
    contract."""
    g = model.graph
    probs = [
        o for o in g.output
        if o.type.tensor_type.elem_type == TensorProto.FLOAT
        and len(o.type.tensor_type.shape.dim) == 2
    ]
    assert len(probs) == 1, f"expected one [N, k] probabilities output, got {len(probs)}"
    prob_name = probs[0].name
    k = probs[0].type.tensor_type.shape.dim[1].dim_value
    assert k == 2, f"ONNX export supports binary classifiers only (got {k} classes)"

    g.initializer.append(numpy_helper.from_array(np.asarray([1], dtype=np.int64), "pos_class_idx"))
    g.node.append(helper.make_node("Gather", [prob_name, "pos_class_idx"], [OUTPUT], axis=1))
    del g.output[:]
    g.output.append(helper.make_tensor_value_info(OUTPUT, TensorProto.FLOAT, ["N", 1]))
    # Gather lives in the default ONNX domain; the ml converters only import
    # ai.onnx.ml, so add a default-domain opset if one isn't already present.
    if not any((op.domain or "") == "" for op in model.opset_import):
        model.opset_import.append(helper.make_opsetid("", OPSET))
    onnx.checker.check_model(model)
    return model


def sklearn_binary_classifier(model, n_cols: int) -> onnx.ModelProto:
    """Convert an sklearn binary classifier (or a Pipeline ending in one) to the
    single-output `[N, 1]` P(class 1) contract via skl2onnx. A StandardScaler in
    the pipeline is converted into the graph too, so the ONNX input stays the raw
    assembled feature vector. `zipmap=False` keeps the probabilities as a plain
    `[N, 2]` tensor (not a sequence-of-maps) so `positive_class_proba` can gather
    the positive column."""
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    onx = convert_sklearn(
        model,
        initial_types=[(INPUT, FloatTensorType([None, n_cols]))],
        options={"zipmap": False},
        target_opset=OPSET,
    )
    return positive_class_proba(onx)


def catboost_binary_classifier(cbm_path, scratch_dir) -> onnx.ModelProto:
    """Convert a CatBoost binary classifier (`model.cbm`) to the `[N, 1]`
    P(class 1) contract. CatBoost's native ONNX export emits
    `TreeEnsembleClassifier → ZipMap`, where the ZipMap turns the raw `[N, 2]`
    probability tensor into a sequence-of-maps that doesn't fit the contract.
    We bypass the ZipMap: rename the input to the canonical `input`, drop the
    ZipMap node, and gather the positive column straight off the pre-ZipMap
    probability tensor. (onnxmltools' `convert_catboost` is unusable here — it
    trips a CatBoost metadata-serialization bug.)"""
    from pathlib import Path

    from catboost import CatBoostClassifier

    model = CatBoostClassifier()
    model.load_model(str(cbm_path))
    native = Path(scratch_dir) / "_catboost_native.onnx"
    model.save_model(
        str(native),
        format="onnx",
        export_parameters={
            "onnx_domain": "ai.catboost",
            "onnx_model_version": 1,
            "onnx_doc_string": "lensing catboost-classifier",
            "onnx_graph_name": "catboost",
        },
    )
    onx = onnx.load(str(native))
    native.unlink(missing_ok=True)
    # CatBoost emits an old IR version under which every initializer must also be
    # a graph input; bump it so the appended `pos_class_idx` initializer is legal.
    onx.ir_version = 7
    g = onx.graph

    old_input = g.input[0].name
    for node in g.node:
        node.input[:] = [INPUT if x == old_input else x for x in node.input]
    g.input[0].name = INPUT

    zipmaps = [n for n in g.node if n.op_type == "ZipMap"]
    assert len(zipmaps) == 1, f"expected one ZipMap, got {len(zipmaps)}"
    prob_tensor = zipmaps[0].input[0]
    g.node.remove(zipmaps[0])

    g.initializer.append(numpy_helper.from_array(np.asarray([1], dtype=np.int64), "pos_class_idx"))
    g.node.append(helper.make_node("Gather", [prob_tensor, "pos_class_idx"], [OUTPUT], axis=1))
    del g.output[:]
    g.output.append(helper.make_tensor_value_info(OUTPUT, TensorProto.FLOAT, ["N", 1]))
    if not any((op.domain or "") == "" for op in onx.opset_import):
        onx.opset_import.append(helper.make_opsetid("", OPSET))
    onnx.checker.check_model(onx)
    return onx
