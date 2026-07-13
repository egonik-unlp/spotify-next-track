#!/usr/bin/env python3
"""NGBoost classifier (binary + multiclass). Natural-gradient boosting over a
k-categorical distribution: each round fits decision-tree base learners to the
natural gradient of the LogScore, yielding probabilistic predictions. Emits
calibrated class probabilities and classification metrics — a different boosting
philosophy (probabilistic / natural gradient) to set against xgboost/catboost.

The k_categorical target distribution expects integer labels 0..K-1, which is
exactly what `_clf_common.encode_labels` produces, so predict_proba columns are
already in canonical class order. Model is persisted via joblib; meta.json
records the class vocabulary.

Run from the shared predictor venv (predictors/.venv)."""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from ngboost import NGBClassifier
from ngboost.distns import k_categorical
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _clf_common as clf  # noqa: E402

DEFAULTS = {
    "n_estimators": 400,
    "learning_rate": 0.03,
    "minibatch_frac": 0.5,
    "col_sample": 1.0,
    "base_max_depth": 3,
    "seed": 42,
}


def build_model(hp: dict, k: int) -> NGBClassifier:
    base = DecisionTreeRegressor(
        max_depth=int(hp["base_max_depth"]),
        random_state=int(hp["seed"]),
    )
    return NGBClassifier(
        Dist=k_categorical(k),
        Base=base,
        n_estimators=int(hp["n_estimators"]),
        learning_rate=float(hp["learning_rate"]),
        minibatch_frac=float(hp["minibatch_frac"]),
        col_sample=float(hp["col_sample"]),
        random_state=int(hp["seed"]),
        verbose=False,
    )


def train(dataset: Path, output: Path, hp_path: Path) -> None:
    hp = {**DEFAULTS, **json.loads(hp_path.read_text())}
    manifest, x, target, row_ids, train_idx, test_idx = clf.load_dataset(dataset)
    y, classes, k = clf.encode_labels(target)
    assert k >= 2, f"classification needs >=2 classes, target has {k}"
    clf.emit({"event": "log", "msg": f"dataset {manifest['dataset_id']}: "
              f"{len(train_idx)} train / {len(test_idx)} test rows, {manifest['n_cols']} cols; "
              f"{k} classes {classes}"})
    clf.emit({"event": "log", "msg": f"ngboost-classifier {hp['n_estimators']} rounds, "
              f"lr {hp['learning_rate']}, base depth {hp['base_max_depth']}, "
              f"minibatch {hp['minibatch_frac']}"})

    model = build_model(hp, k)
    model.fit(x[train_idx], y[train_idx])

    # k_categorical maps onto labels 0..K-1, so predict_proba columns are
    # already in canonical class order — identity reorder.
    proba = clf.align_proba(model.predict_proba(x[test_idx]), list(range(k)), k)
    y_test = y[test_idx]
    metrics = clf.classification_metrics(y_test, proba, k)
    predictions = clf.predictions_payload(row_ids, test_idx, y_test, proba, k)

    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics))
    (output / "predictions.json").write_text(json.dumps(predictions))
    joblib.dump(model, output / "model.joblib")
    (output / "meta.json").write_text(json.dumps({"classes": classes, "k": k}))

    clf.emit({"event": "log", "msg": f"AUC {metrics.get('auc'):.4f}  "
              f"logloss {metrics['logloss']:.4f}  acc {metrics['accuracy']:.3f}"})
    clf.emit({"event": "done"})


def predict(model_dir: Path, input_dir: Path, output: Path) -> None:
    meta = json.loads((model_dir / "meta.json").read_text())
    k = int(meta["k"])
    model = joblib.load(model_dir / "model.joblib")

    manifest = json.loads((input_dir / "manifest.json").read_text())
    n_rows, n_cols = manifest["n_rows"], manifest["n_cols"]
    x = clf.read_features(input_dir, n_rows, n_cols)
    row_ids = np.fromfile(input_dir / "row_ids.u64", dtype="<u8")
    proba = clf.align_proba(model.predict_proba(x), list(range(k)), k)
    clf.emit({"event": "log", "msg": f"predicting {n_rows} rows, {k} classes"})
    output.write_text(json.dumps(clf.inference_payload(row_ids, proba, k)))
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
    args = ap.parse_args()
    if args.cmd == "train":
        train(args.dataset, args.output, args.hyperparams)
    else:
        predict(args.model, args.input, args.output)


if __name__ == "__main__":
    main()
