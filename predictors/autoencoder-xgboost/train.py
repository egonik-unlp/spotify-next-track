#!/usr/bin/env python3
"""Autoencoder-latent + XGBoost blend ensemble.

A weighted-blend ensemble of two predictors that split the feature space along
the manifest's column kinds:

  - AE branch  (the track's *identity*): a regressor over the embedding/latent
    columns (manifest kind == "pca"). On a dataset built off the
    `spotify_tracks_song_ae` collection with pca_dims = latent_dim, those
    columns are the 64-dim song-autoencoder latent (an orthonormal rotation +
    centering of it). Two interchangeable AE-branch regressors via `ae_branch`:
      * "knn"   — KNeighborsRegressor on the RAW latent columns, Euclidean
                  metric. PCA(k=d) is an isometry for Euclidean distance, so
                  these are exactly the latent's native neighbor relations; we
                  deliberately do NOT standardize (that would warp the geometry).
      * "ridge" — StandardScaler + RidgeCV on the latent columns.
  - XGB branch (the *rest*): XGBRegressor over the metadata columns (manifest
    kind in {numeric, onehot}) — popularity, genre, artist, release year, ...

Final prediction: yhat = w * yhat_ae + (1 - w) * yhat_xgb, blended in the
TRANSFORMED (log) target space both branches train in, then inverted to target
space. `w` (the AE-branch weight) is tuned by grid search on an internal
holdout carved from the train split (test split untouched); set blend_weight
>= 0 to fix it instead.

metrics.json carries the canonical blended-model metrics (mae/rmse/r2/...) plus
diagnostics mae_ae / mae_xgb (each branch alone, refit on full train) and
blend_weight, so a run reports whether the latent carries signal and whether
blending beats either branch alone.

Run from the repo's shared predictor venv (predictors/.venv, see
`zig build py-setup`)."""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DEFAULTS = {
    "ae_branch": "knn",          # "knn" | "ridge"
    "knn_k": 25,
    "knn_weights": "distance",   # "distance" | "uniform"
    "ridge_auto": True,          # RidgeCV over a log-spaced grid
    "ridge_alpha": 1.0,          # used when ridge_auto is False
    # XGB branch (mirrors the current-best xgboost-metadata config)
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1.0,
    "reg_lambda": 1.0,
    # Blend
    "blend_weight": -1.0,        # < 0 = auto-tune on holdout; else fixed AE weight
    "blend_grid": 21,            # grid resolution for the auto blend search
    "val_fraction": 0.2,         # internal holdout carved from train for tuning
    "seed": 42,
}
RIDGE_ALPHA_GRID = np.logspace(-2, 5, 29)


def emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def read_features(dir: Path, n_rows: int, n_cols: int) -> np.ndarray:
    x = np.fromfile(dir / "features.f32", dtype="<f4")
    assert x.size == n_rows * n_cols, (
        f"features.f32 has {x.size} values, manifest says {n_rows}x{n_cols}")
    return x.reshape(n_rows, n_cols).astype(np.float64)


def invert_target(y: np.ndarray, transform: str) -> np.ndarray:
    """Map transformed-space targets back to target space."""
    if transform == "log1p":
        return np.expm1(y)
    return y


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Target-space metrics, formula identical to lensing-core::compute_metrics
    (medape for even n = mean of the two middle APEs). APE skips actual==0."""
    n = len(actual)
    err = predicted - actual
    nz = actual != 0.0
    apes = np.sort(np.abs(err[nz] / actual[nz]))
    m = len(apes)
    if m == 0:
        medape = mape = 0.0
    else:
        medape = apes[m // 2] if m % 2 == 1 else (apes[m // 2 - 1] + apes[m // 2]) / 2.0
        mape = float(np.mean(apes))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    sq_err = float(np.sum(err**2))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(sq_err / n)),
        "r2": (1.0 - sq_err / ss_tot) if ss_tot > 0 else 0.0,
        "mape": mape,
        "medape": float(medape),
        "n_test": n,
    }


def column_groups(manifest: dict) -> tuple[list[int], list[int]]:
    """Split columns by manifest kind: latent (pca) vs metadata (the rest)."""
    latent, meta = [], []
    for i, col in enumerate(manifest["columns"]):
        (latent if col["kind"]["type"] == "pca" else meta).append(i)
    return latent, meta


def build_ae(hp: dict, n_fit: int):
    """The AE-branch regressor over the latent columns."""
    branch = str(hp["ae_branch"])
    if branch == "knn":
        k = min(int(hp["knn_k"]), max(1, n_fit - 1))
        return KNeighborsRegressor(
            n_neighbors=k, weights=str(hp["knn_weights"]), metric="euclidean")
    if branch == "ridge":
        reg = (RidgeCV(alphas=RIDGE_ALPHA_GRID, fit_intercept=True)
               if bool(hp["ridge_auto"])
               else Ridge(alpha=float(hp["ridge_alpha"]), fit_intercept=True))
        # standardize the latent for the linear branch (penalty is scale-sensitive)
        return make_pipeline(StandardScaler(), reg)


def build_xgb(hp: dict) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        n_estimators=int(hp["n_estimators"]),
        max_depth=int(hp["max_depth"]),
        learning_rate=float(hp["learning_rate"]),
        subsample=float(hp["subsample"]),
        colsample_bytree=float(hp["colsample_bytree"]),
        min_child_weight=float(hp["min_child_weight"]),
        reg_lambda=float(hp["reg_lambda"]),
        random_state=int(hp["seed"]),
        eval_metric="rmse",
        n_jobs=-1,
    )


def tune_blend(ae_val_t, xgb_val_t, val_y_t, transform, grid) -> tuple[float, float]:
    """Grid-search the AE weight w on holdout target-space MAE.
    Predictions are blended in transformed space, then inverted."""
    actual = invert_target(val_y_t, transform)
    best_w, best_mae = 0.0, np.inf
    for w in np.linspace(0.0, 1.0, int(grid)):
        blended = invert_target(w * ae_val_t + (1.0 - w) * xgb_val_t, transform)
        mae = float(np.mean(np.abs(np.maximum(blended, 0.0) - actual)))
        if mae < best_mae:
            best_w, best_mae = float(w), mae
    return best_w, best_mae


def train(dataset: Path, output: Path, hp_path: Path) -> None:
    hp = {**DEFAULTS, **json.loads(hp_path.read_text())}

    manifest = json.loads((dataset / "manifest.json").read_text())
    n_rows, n_cols = manifest["n_rows"], manifest["n_cols"]
    features = read_features(dataset, n_rows, n_cols)
    target = np.fromfile(dataset / "target.f32", dtype="<f4").astype(np.float64)
    row_ids = np.fromfile(dataset / "row_ids.u64", dtype="<u8")
    train_idx = np.fromfile(dataset / "train_idx.u32", dtype="<u4")
    test_idx = np.fromfile(dataset / "test_idx.u32", dtype="<u4")
    transform = manifest["target"]["transform"]

    latent_cols, meta_cols = column_groups(manifest)
    assert latent_cols, "no PCA/latent columns in manifest — build with pca_dims>=1"
    assert meta_cols, "no metadata columns in manifest"
    emit({"event": "log", "msg": f"dataset {manifest['dataset_id']}: "
          f"{len(train_idx)} train / {len(test_idx)} test rows; "
          f"{len(latent_cols)} latent + {len(meta_cols)} metadata cols"})
    emit({"event": "log", "msg": f"AE branch: {hp['ae_branch']}"
          + (f" (k={hp['knn_k']}, {hp['knn_weights']})" if hp["ae_branch"] == "knn"
             else f" (auto_alpha={hp['ridge_auto']})")
          + f"; XGB branch: {hp['n_estimators']} trees, depth {hp['max_depth']}, "
          f"lr {hp['learning_rate']}"})

    X_lat, X_meta = features[:, latent_cols], features[:, meta_cols]
    train_y = target[train_idx]

    # ----- tune the blend weight on an internal holdout carved from train -----
    fixed_w = float(hp["blend_weight"])
    if fixed_w >= 0.0:
        best_w = min(max(fixed_w, 0.0), 1.0)
        emit({"event": "log", "msg": f"blend weight fixed at w={best_w:.3f}"})
    else:
        n_val = max(1, round(len(train_idx) * float(hp["val_fraction"])))
        assert n_val < len(train_idx), "holdout carve left no training rows"
        perm = np.random.default_rng(int(hp["seed"])).permutation(len(train_idx))
        val_i, fit_i = train_idx[perm[:n_val]], train_idx[perm[n_val:]]

        ae_v = build_ae(hp, len(fit_i))
        ae_v.fit(X_lat[fit_i], target[fit_i])
        xgb_v = build_xgb(hp)
        xgb_v.fit(X_meta[fit_i], target[fit_i])
        best_w, val_mae = tune_blend(
            ae_v.predict(X_lat[val_i]), xgb_v.predict(X_meta[val_i]),
            target[val_i], transform, hp["blend_grid"])
        emit({"event": "log", "msg": f"tuned blend on {n_val} holdout rows: "
              f"w={best_w:.3f} (AE) / {1 - best_w:.3f} (XGB), holdout MAE {val_mae:.3f}"})

    # ----- refit both branches on the FULL train split, predict test -----
    ae = build_ae(hp, len(train_idx))
    ae.fit(X_lat[train_idx], train_y)
    xgb_model = build_xgb(hp)
    xgb_model.fit(X_meta[train_idx], train_y)

    ae_test_t = ae.predict(X_lat[test_idx])
    xgb_test_t = xgb_model.predict(X_meta[test_idx])
    blended_t = best_w * ae_test_t + (1.0 - best_w) * xgb_test_t

    actual = invert_target(target[test_idx], transform)
    predicted = np.maximum(invert_target(blended_t, transform), 0.0)
    pred_ae = np.maximum(invert_target(ae_test_t, transform), 0.0)
    pred_xgb = np.maximum(invert_target(xgb_test_t, transform), 0.0)

    metrics = compute_metrics(actual, predicted)
    metrics["mae_ae"] = float(np.mean(np.abs(pred_ae - actual)))
    metrics["mae_xgb"] = float(np.mean(np.abs(pred_xgb - actual)))
    metrics["blend_weight"] = best_w

    predictions = [
        {"row_id": int(row_ids[i]), "actual": float(a), "predicted": float(p)}
        for i, a, p in zip(test_idx, actual, predicted)
    ]

    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics))
    (output / "predictions.json").write_text(json.dumps(predictions))
    # snapshot for predict: branch artifacts + blend/column spec
    joblib.dump(ae, output / "ae_model.joblib")
    xgb_model.save_model(output / "xgb_model.ubj")
    (output / "ensemble.json").write_text(json.dumps({
        "ae_branch": str(hp["ae_branch"]),
        "blend_weight": best_w,
        "latent_cols": latent_cols,
        "meta_cols": meta_cols,
        "n_features": n_cols,
        "transform": transform,
    }))

    emit({"event": "log", "msg": f"blended MAE {metrics['mae']:.3f}  "
          f"(AE-only {metrics['mae_ae']:.3f}, XGB-only {metrics['mae_xgb']:.3f})  "
          f"R² {metrics['r2']:.3f}  w={best_w:.3f}"})
    emit({"event": "done"})


def predict(model_dir: Path, input_dir: Path, output: Path) -> None:
    """Load both branch artifacts, blend on the server-featurized mini-artifact,
    write target-space predictions."""
    spec = json.loads((model_dir / "ensemble.json").read_text())
    ae = joblib.load(model_dir / "ae_model.joblib")
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(model_dir / "xgb_model.ubj")

    manifest = json.loads((input_dir / "manifest.json").read_text())
    n_rows, n_cols = manifest["n_rows"], manifest["n_cols"]
    assert n_cols == spec["n_features"], (
        f"model was fit on {spec['n_features']} columns, input has {n_cols}")
    features = read_features(input_dir, n_rows, n_cols)
    row_ids = np.fromfile(input_dir / "row_ids.u64", dtype="<u8")
    transform = manifest["target"]["transform"]
    w = float(spec["blend_weight"])
    emit({"event": "log", "msg": f"predicting {n_rows} rows; AE branch "
          f"{spec['ae_branch']}, blend w={w:.3f}"})

    ae_t = ae.predict(features[:, spec["latent_cols"]])
    xgb_t = xgb_model.predict(features[:, spec["meta_cols"]])
    predicted = np.maximum(invert_target(w * ae_t + (1.0 - w) * xgb_t, transform), 0.0)

    output.write_text(json.dumps([
        {"row_id": int(rid), "predicted": float(p)}
        for rid, p in zip(row_ids, predicted)
    ]))
    emit({"event": "done"})


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
