#!/usr/bin/env python3
"""Pyramid-MLP classifier (binary + multiclass). A feed-forward net whose hidden
layers taper geometrically from a wide first layer down to a narrow last one (a
"pyramid": top_width, top_width*decay, top_width*decay^2, ...). Standardizes
features, then fits an sklearn MLPClassifier with adam + ReLU; emits calibrated
class probabilities (softmax output) and classification metrics. The dense /
neural family to set against the tree boosters on the PCA + identity features.

Model + scaler are persisted via joblib; meta.json records the class vocabulary
and the realized hidden-layer schedule.

Run from the shared predictor venv (predictors/.venv)."""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _clf_common as clf  # noqa: E402

DEFAULTS = {
    "top_width": 256,     # width of the first (widest) hidden layer
    "n_layers": 3,        # number of hidden layers in the pyramid
    "decay": 0.5,         # each layer = round(previous * decay), floored at 8
    "alpha": 1e-4,        # L2 regularization
    "learning_rate_init": 1e-3,
    "max_iter": 300,
    "seed": 42,
}

MIN_WIDTH = 8


def pyramid_layers(top_width: int, n_layers: int, decay: float) -> tuple:
    """Geometric taper: [top, top*decay, top*decay^2, ...], each >= MIN_WIDTH."""
    widths = []
    w = float(top_width)
    for _ in range(max(1, int(n_layers))):
        widths.append(max(MIN_WIDTH, int(round(w))))
        w *= float(decay)
    return tuple(widths)


def build_model(hp: dict, hidden: tuple) -> MLPClassifier:
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=hidden,
            activation="relu",
            solver="adam",
            alpha=float(hp["alpha"]),
            learning_rate_init=float(hp["learning_rate_init"]),
            max_iter=int(hp["max_iter"]),
            random_state=int(hp["seed"]),
        ),
    )


def train(dataset: Path, output: Path, hp_path: Path) -> None:
    hp = {**DEFAULTS, **json.loads(hp_path.read_text())}
    manifest, x, target, row_ids, train_idx, test_idx = clf.load_dataset(dataset)
    y, classes, k = clf.encode_labels(target)
    assert k >= 2, f"classification needs >=2 classes, target has {k}"
    hidden = pyramid_layers(hp["top_width"], hp["n_layers"], hp["decay"])
    clf.emit({"event": "log", "msg": f"dataset {manifest['dataset_id']}: "
              f"{len(train_idx)} train / {len(test_idx)} test rows, {manifest['n_cols']} cols; "
              f"{k} classes {classes}"})
    clf.emit({"event": "log", "msg": f"pyramid-mlp-classifier hidden {list(hidden)}, "
              f"alpha {hp['alpha']}, lr {hp['learning_rate_init']}, max_iter {hp['max_iter']}"})

    model = build_model(hp, hidden)
    model.fit(x[train_idx], y[train_idx])
    mlp = model.steps[-1][1]
    if mlp.n_iter_ >= int(hp["max_iter"]):
        clf.emit({"event": "log", "msg": f"WARNING: hit max_iter ({hp['max_iter']}) "
                  "without converging — consider raising it"})

    proba = clf.align_proba(model.predict_proba(x[test_idx]), mlp.classes_, k)
    y_test = y[test_idx]
    metrics = clf.classification_metrics(y_test, proba, k)
    predictions = clf.predictions_payload(row_ids, test_idx, y_test, proba, k)

    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics))
    (output / "predictions.json").write_text(json.dumps(predictions))
    joblib.dump(model, output / "model.joblib")
    (output / "meta.json").write_text(json.dumps(
        {"classes": classes, "k": k, "hidden_layers": list(hidden)}))

    clf.emit({"event": "log", "msg": f"AUC {metrics.get('auc'):.4f}  "
              f"logloss {metrics['logloss']:.4f}  acc {metrics['accuracy']:.3f}  "
              f"({mlp.n_iter_} iters)"})
    clf.emit({"event": "done"})


def predict(model_dir: Path, input_dir: Path, output: Path) -> None:
    meta = json.loads((model_dir / "meta.json").read_text())
    k = int(meta["k"])
    model = joblib.load(model_dir / "model.joblib")
    mlp = model.steps[-1][1]

    manifest = json.loads((input_dir / "manifest.json").read_text())
    n_rows, n_cols = manifest["n_rows"], manifest["n_cols"]
    x = clf.read_features(input_dir, n_rows, n_cols)
    row_ids = np.fromfile(input_dir / "row_ids.u64", dtype="<u8")
    proba = clf.align_proba(model.predict_proba(x), mlp.classes_, k)
    clf.emit({"event": "log", "msg": f"predicting {n_rows} rows, {k} classes"})
    output.write_text(json.dumps(clf.inference_payload(row_ids, proba, k)))
    clf.emit({"event": "done"})


def export(model_dir: Path, output: Path) -> None:
    """Contract export: a single `model.onnx` whose input is the assembled
    feature vector and whose output is P(class=1), shape [N, 1] (binary only).
    The StandardScaler is baked into the graph by skl2onnx, so the input stays
    the raw assembled vector."""
    import onnx_common
    # Derive the class count from the model itself — `meta.json` is not copied
    # into a promoted model dir (the server excludes it on promotion).
    model = joblib.load(model_dir / "model.joblib")
    k = len(model.steps[-1][1].classes_)
    if k != 2:
        raise SystemExit(
            f"ONNX export supports binary classifiers only; this model has {k} classes")
    n_cols = int(model.steps[0][1].n_features_in_)
    onx = onnx_common.sklearn_binary_classifier(model, n_cols)
    onnx_common.save(onx, output)
    clf.emit({"event": "log", "msg": "exported model.onnx: pyramid-mlp-classifier "
              f"({n_cols} features, scaler + dense stack baked in) binary P(class=1)"})
    clf.emit({"event": "done"})


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--dataset", required=True, type=Path)
    tr.add_argument("--output", required=True, type=Path)
    tr.add_argument("--hyperparams", required=True, type=Path)
    pr = sub.add_parser("predict")
    pr.add_argument("--model", required=True, type=Path)
    pr.add_argument("--input", required=True, type=Path)
    pr.add_argument("--output", required=True, type=Path)
    ex = sub.add_parser("export")
    ex.add_argument("--model", required=True, type=Path)
    ex.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    if args.cmd == "train":
        train(args.dataset, args.output, args.hyperparams)
    elif args.cmd == "export":
        export(args.model, args.output)
    else:
        predict(args.model, args.input, args.output)


if __name__ == "__main__":
    main()
