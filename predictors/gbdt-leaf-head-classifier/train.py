#!/usr/bin/env python3
"""GBDT-leaf-embedding classifier (binary). The classic Facebook GBDT→LR trick,
generalized to GBDT→MLP: a gradient-boosted tree (frozen at the on-record
xgboost-classifier champion config) emits per-row leaf indices, those are
one-hot-encoded into a learned, axis-aligned representation (the tree's own
partition), and a linear (L2 logistic) or shallow-MLP head is trained on top of
that representation.

This is a CEILING-PROBE predictor: the tree producing the leaves is held at the
champion hyperparams in every run so the only axis is "bare tree vs head over
the tree's leaves". See experiments/PROJECT-FACTS.md.

Artifacts: booster.ubj (the leaf-emitting xgboost), head.joblib (OneHotEncoder
+ fitted head pipeline), meta.json (class vocabulary + head kind). No ONNX
export (the two-stage stack is not in the [N,1] bundle contract).

Run from the shared predictor venv (predictors/.venv)."""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _clf_common as clf  # noqa: E402

# The leaf-emitting GBDT is FROZEN at the xgboost-classifier-rotation-p64
# champion config (run-20260609-204726-439d8). These are NOT exposed as tunable
# params: the experiment's only axis is the head.
TREE = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1.0,
    "reg_lambda": 1.0,
    "seed": 42,
}

DEFAULTS = {
    "head": "linear",       # "linear" (L2 logistic) | "mlp" (shallow MLP)
    "hidden": [128],        # mlp hidden layer sizes
    "l2": 1.0,              # linear: inverse-C maps below; mlp: alpha (L2)
    "head_lr": 1e-3,        # mlp: learning_rate_init (ignored by linear head)
    "max_iter": 300,        # linear: lbfgs iters; mlp: adam epoch cap (early-stops)
    "seed": 42,
}


def build_tree() -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=TREE["n_estimators"],
        max_depth=TREE["max_depth"],
        learning_rate=TREE["learning_rate"],
        subsample=TREE["subsample"],
        colsample_bytree=TREE["colsample_bytree"],
        min_child_weight=TREE["min_child_weight"],
        reg_lambda=TREE["reg_lambda"],
        random_state=TREE["seed"],
        eval_metric="logloss",
        n_jobs=-1,
    )


def build_head(hp: dict):
    """OneHotEncoder over leaf ids (unseen test leaves → all-zero) → head."""
    ohe = OneHotEncoder(handle_unknown="ignore", dtype=np.float64)
    kind = str(hp["head"])
    if kind == "linear":
        # L2 logistic regression — the literal Facebook GBDT→LR. l2 → inverse-C.
        head = LogisticRegression(
            C=1.0 / max(float(hp["l2"]), 1e-9),
            max_iter=int(hp["max_iter"]),
            random_state=int(hp["seed"]),
        )
    elif kind == "mlp":
        hidden = tuple(int(h) for h in hp["hidden"])
        # early_stopping: holds out 10% of train, stops when val score stalls for
        # n_iter_no_change epochs — keeps adam tractable over the wide sparse leaf
        # input AND guards the leaf-overfitting/overconfidence risk.
        head = MLPClassifier(
            hidden_layer_sizes=hidden,
            activation="relu",
            alpha=float(hp["l2"]),
            learning_rate_init=float(hp["head_lr"]),
            max_iter=int(hp["max_iter"]),
            early_stopping=True,
            n_iter_no_change=10,
            random_state=int(hp["seed"]),
        )
    else:
        raise SystemExit(f"unknown head '{kind}' (expected 'linear' or 'mlp')")
    return Pipeline([("ohe", ohe), ("head", head)])


def leaves(tree: xgb.XGBClassifier, x: np.ndarray) -> np.ndarray:
    """Per-row, per-tree leaf indices, shape (n_rows, n_trees), as strings so the
    OneHotEncoder treats each tree's leaf-id set as its own categorical vocab."""
    return tree.apply(x).astype(np.int32).astype(str)


def train(dataset: Path, output: Path, hp_path: Path) -> None:
    hp = {**DEFAULTS, **json.loads(hp_path.read_text())}
    manifest, x, target, row_ids, train_idx, test_idx = clf.load_dataset(dataset)
    y, classes, k = clf.encode_labels(target)
    assert k == 2, f"gbdt-leaf-head supports binary only; target has {k} classes"
    clf.emit({"event": "log", "msg": f"dataset {manifest['dataset_id']}: "
              f"{len(train_idx)} train / {len(test_idx)} test rows, {manifest['n_cols']} cols; "
              f"binary {classes}"})
    clf.emit({"event": "log", "msg": f"leaf-emitting tree FROZEN at champion config "
              f"({TREE['n_estimators']} rounds, depth {TREE['max_depth']}, "
              f"lr {TREE['learning_rate']}); head={hp['head']}"})

    tree = build_tree()
    tree.fit(x[train_idx], y[train_idx])
    leaf_train = leaves(tree, x[train_idx])
    leaf_test = leaves(tree, x[test_idx])
    clf.emit({"event": "log", "msg": f"leaf representation: {leaf_train.shape[1]} trees → "
              f"one-hot leaves; fitting {hp['head']} head"})

    head = build_head(hp)
    head.fit(leaf_train, y[train_idx])
    n_feat = head.named_steps["ohe"].transform(leaf_train[:1]).shape[1]

    proba = clf.align_proba(head.predict_proba(leaf_test), head.classes_, k)
    y_test = y[test_idx]
    metrics = clf.classification_metrics(y_test, proba, k)
    predictions = clf.predictions_payload(row_ids, test_idx, y_test, proba, k)

    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics))
    (output / "predictions.json").write_text(json.dumps(predictions))
    tree.get_booster().save_model(output / "booster.ubj")
    joblib.dump(head, output / "head.joblib")
    (output / "meta.json").write_text(json.dumps(
        {"classes": classes, "k": k, "head": hp["head"], "leaf_dim": int(n_feat)}))

    auc = metrics.get("auc")
    clf.emit({"event": "log", "msg": f"AUC {auc:.4f}  logloss {metrics['logloss']:.4f}  "
              f"acc {metrics['accuracy']:.3f}  (leaf one-hot dim {n_feat})"})
    clf.emit({"event": "done"})


def predict(model_dir: Path, input_dir: Path, output: Path) -> None:
    meta = json.loads((model_dir / "meta.json").read_text())
    k = int(meta["k"])
    tree = xgb.XGBClassifier()
    tree.load_model(model_dir / "booster.ubj")
    head = joblib.load(model_dir / "head.joblib")

    manifest = json.loads((input_dir / "manifest.json").read_text())
    n_rows, n_cols = manifest["n_rows"], manifest["n_cols"]
    x = clf.read_features(input_dir, n_rows, n_cols)
    row_ids = np.fromfile(input_dir / "row_ids.u64", dtype="<u8")
    proba = clf.align_proba(head.predict_proba(leaves(tree, x)), head.classes_, k)
    clf.emit({"event": "log", "msg": f"predicting {n_rows} rows via leaf-head ({meta['head']})"})
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
