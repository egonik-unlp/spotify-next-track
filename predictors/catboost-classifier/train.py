#!/usr/bin/env python3
"""CatBoost classifier (binary + multiclass). Ordered gradient boosting with
symmetric (oblivious) trees optimizing Logloss / MultiClass; emits calibrated
class probabilities and classification metrics. A strong tree family to set
against xgboost-classifier on the categorical-identity features.

Model file is CatBoost's native binary (model.cbm); meta.json records the class
vocabulary so predict can reconstruct canonical column order.

Run from the shared predictor venv (predictors/.venv)."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _clf_common as clf  # noqa: E402

DEFAULTS = {
    "iterations": 500,
    "depth": 6,
    "learning_rate": 0.05,
    "l2_leaf_reg": 3.0,
    "subsample": 0.8,
    "border_count": 254,
    "seed": 42,
}


def build_model(hp: dict, k: int) -> CatBoostClassifier:
    loss = "Logloss" if k == 2 else "MultiClass"
    return CatBoostClassifier(
        loss_function=loss,
        iterations=int(hp["iterations"]),
        depth=int(hp["depth"]),
        learning_rate=float(hp["learning_rate"]),
        l2_leaf_reg=float(hp["l2_leaf_reg"]),
        subsample=float(hp["subsample"]),
        border_count=int(hp["border_count"]),
        random_seed=int(hp["seed"]),
        bootstrap_type="Bernoulli",  # subsample is only honored for Bernoulli/Poisson
        thread_count=-1,
        allow_writing_files=False,
        logging_level="Silent",
    )


def train(dataset: Path, output: Path, hp_path: Path) -> None:
    hp = {**DEFAULTS, **json.loads(hp_path.read_text())}
    manifest, x, target, row_ids, train_idx, test_idx = clf.load_dataset(dataset)
    y, classes, k = clf.encode_labels(target)
    assert k >= 2, f"classification needs >=2 classes, target has {k}"
    clf.emit({"event": "log", "msg": f"dataset {manifest['dataset_id']}: "
              f"{len(train_idx)} train / {len(test_idx)} test rows, {manifest['n_cols']} cols; "
              f"{k} classes {classes}"})
    clf.emit({"event": "log", "msg": f"catboost-classifier {hp['iterations']} iters, "
              f"depth {hp['depth']}, lr {hp['learning_rate']}"})

    model = build_model(hp, k)
    model.fit(x[train_idx], y[train_idx])

    proba = clf.align_proba(model.predict_proba(x[test_idx]), model.classes_, k)
    y_test = y[test_idx]
    metrics = clf.classification_metrics(y_test, proba, k)
    predictions = clf.predictions_payload(row_ids, test_idx, y_test, proba, k)

    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics))
    (output / "predictions.json").write_text(json.dumps(predictions))
    model.save_model(str(output / "model.cbm"))
    (output / "meta.json").write_text(json.dumps({"classes": classes, "k": k}))

    clf.emit({"event": "log", "msg": f"AUC {metrics.get('auc'):.4f}  "
              f"logloss {metrics['logloss']:.4f}  acc {metrics['accuracy']:.3f}"})
    clf.emit({"event": "done"})


def predict(model_dir: Path, input_dir: Path, output: Path) -> None:
    meta = json.loads((model_dir / "meta.json").read_text())
    k = int(meta["k"])
    model = CatBoostClassifier()
    model.load_model(str(model_dir / "model.cbm"))

    manifest = json.loads((input_dir / "manifest.json").read_text())
    n_rows, n_cols = manifest["n_rows"], manifest["n_cols"]
    x = clf.read_features(input_dir, n_rows, n_cols)
    row_ids = np.fromfile(input_dir / "row_ids.u64", dtype="<u8")
    proba = clf.align_proba(model.predict_proba(x), model.classes_, k)
    clf.emit({"event": "log", "msg": f"predicting {n_rows} rows, {k} classes"})
    output.write_text(json.dumps(clf.inference_payload(row_ids, proba, k)))
    clf.emit({"event": "done"})


def export(model_dir: Path, output: Path) -> None:
    """Contract export: a single `model.onnx` whose input is the assembled
    feature vector and whose output is P(class=1), shape [N, 1] (binary only —
    trees are scale-invariant, so no standardizer prefix is needed)."""
    import onnx_common
    # Derive the class count from the model itself — `meta.json` is not copied
    # into a promoted model dir (the server excludes it on promotion).
    probe = CatBoostClassifier()
    probe.load_model(str(model_dir / "model.cbm"))
    k = len(probe.classes_)
    if k != 2:
        raise SystemExit(
            f"ONNX export supports binary classifiers only; this model has {k} classes")
    output.mkdir(parents=True, exist_ok=True)
    onx = onnx_common.catboost_binary_classifier(model_dir / "model.cbm", output)
    onnx_common.save(onx, output)
    clf.emit({"event": "log", "msg": "exported model.onnx: catboost-classifier binary P(class=1)"})
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
