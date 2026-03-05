#!/usr/bin/env python
"""Phase 14: Hybrid approaches for extrapolation.

Key idea: LGBM is excellent within training range but cannot extrapolate.
Use a hybrid approach:
1. LGBM for the main prediction
2. Linear correction for extrapolation beyond training range
3. Clip-aware prediction adjustments

Also: post-processing tricks and calibration.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spectral_challenge.config import Config
from spectral_challenge.data.load import load_train, load_test
from spectral_challenge.metrics import rmse
from spectral_challenge.models.factory import create_model
from spectral_challenge.preprocess.pipeline import build_preprocess_pipeline

DATA_DIR = Path("data/raw")
RUNS_DIR = Path("runs")


def run_cv_with_postprocess(name, X_train, y_train, groups, preprocess_cfg,
                            model_type, model_params, postprocess_fn=None, n_folds=5):
    """Run CV with optional post-processing of predictions."""
    gkf = GroupKFold(n_splits=n_folds)
    oof_preds = np.full(len(y_train), np.nan)
    fold_rmses = []

    run_dir = RUNS_DIR / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}"
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        pipe = build_preprocess_pipeline(preprocess_cfg)
        X_tr_t = pipe.fit_transform(X_tr)
        X_val_t = pipe.transform(X_val)

        model = create_model(model_type, model_params)
        model.fit(X_tr_t, y_tr)

        val_pred = model.predict(X_val_t).ravel()

        if postprocess_fn:
            val_pred = postprocess_fn(val_pred, y_tr, X_tr_t, X_val_t)

        oof_preds[val_idx] = val_pred
        fold_rmses.append(rmse(y_val, val_pred))

        joblib.dump(pipe, models_dir / f"pipe_fold{fold_idx}.joblib")
        joblib.dump(model, models_dir / f"model_fold{fold_idx}.joblib")

    mean_score = rmse(y_train, oof_preds)
    print(f"  {name}: RMSE={mean_score:.4f}  folds=[{', '.join(f'{f:.1f}' for f in fold_rmses)}]")

    np.save(run_dir / "oof_preds.npy", oof_preds)
    with open(run_dir / "metrics.json", "w") as f:
        json.dump({"mean_rmse": float(mean_score), "fold_rmses": [float(x) for x in fold_rmses]}, f, indent=2)

    return mean_score


def hybrid_lgbm_linear_cv(name, X_train, y_train, groups, preprocess_cfg,
                          lgbm_params, linear_alpha=1.0, blend_weight=0.7, n_folds=5):
    """Hybrid: blend LGBM predictions with Ridge predictions.

    Ridge can extrapolate, LGBM cannot. Blend to get best of both.
    """
    gkf = GroupKFold(n_splits=n_folds)
    oof_preds = np.full(len(y_train), np.nan)
    oof_lgbm = np.full(len(y_train), np.nan)
    oof_ridge = np.full(len(y_train), np.nan)
    fold_rmses = []

    run_dir = RUNS_DIR / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}"
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        pipe = build_preprocess_pipeline(preprocess_cfg)
        X_tr_t = pipe.fit_transform(X_tr)
        X_val_t = pipe.transform(X_val)

        # LGBM
        model_lgbm = create_model("lgbm", lgbm_params)
        model_lgbm.fit(X_tr_t, y_tr)
        pred_lgbm = model_lgbm.predict(X_val_t).ravel()
        oof_lgbm[val_idx] = pred_lgbm

        # Ridge
        model_ridge = Ridge(alpha=linear_alpha)
        model_ridge.fit(X_tr_t, y_tr)
        pred_ridge = model_ridge.predict(X_val_t).ravel()
        oof_ridge[val_idx] = pred_ridge

        # Blend
        pred_blend = blend_weight * pred_lgbm + (1 - blend_weight) * pred_ridge
        oof_preds[val_idx] = pred_blend

        fold_rmses.append(rmse(y_val, pred_blend))

        joblib.dump(pipe, models_dir / f"pipe_fold{fold_idx}.joblib")
        joblib.dump(model_lgbm, models_dir / f"model_fold{fold_idx}.joblib")

    mean_score = rmse(y_train, oof_preds)
    lgbm_score = rmse(y_train, oof_lgbm)
    ridge_score = rmse(y_train, oof_ridge)
    print(f"  {name}: RMSE={mean_score:.4f} (lgbm={lgbm_score:.4f}, ridge={ridge_score:.4f})  folds=[{', '.join(f'{f:.1f}' for f in fold_rmses)}]")

    np.save(run_dir / "oof_preds.npy", oof_preds)
    with open(run_dir / "metrics.json", "w") as f:
        json.dump({
            "mean_rmse": float(mean_score),
            "fold_rmses": [float(x) for x in fold_rmses],
            "lgbm_rmse": float(lgbm_score),
            "ridge_rmse": float(ridge_score),
        }, f, indent=2)

    return mean_score


def adaptive_blend_cv(name, X_train, y_train, groups, preprocess_cfg, lgbm_params, n_folds=5):
    """Adaptive blending: use Ridge more where LGBM likely saturates (near max/min of training range)."""
    gkf = GroupKFold(n_splits=n_folds)
    oof_preds = np.full(len(y_train), np.nan)
    fold_rmses = []

    run_dir = RUNS_DIR / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}"
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        pipe = build_preprocess_pipeline(preprocess_cfg)
        X_tr_t = pipe.fit_transform(X_tr)
        X_val_t = pipe.transform(X_val)

        # LGBM
        model_lgbm = create_model("lgbm", lgbm_params)
        model_lgbm.fit(X_tr_t, y_tr)
        pred_lgbm = model_lgbm.predict(X_val_t).ravel()

        # Ridge
        model_ridge = Ridge(alpha=100.0)
        model_ridge.fit(X_tr_t, y_tr)
        pred_ridge = model_ridge.predict(X_val_t).ravel()

        # Adaptive blending: use Ridge more when LGBM pred is near training boundary
        y_train_max = y_tr.max()
        y_train_min = y_tr.min()
        y_range = y_train_max - y_train_min

        # LGBM weight decreases near boundaries
        dist_to_boundary = np.minimum(
            pred_lgbm - y_train_min,
            y_train_max - pred_lgbm,
        )
        # Sigmoid-like weighting: LGBM gets less weight near boundaries
        lgbm_weight = 1.0 / (1.0 + np.exp(-5 * dist_to_boundary / y_range))
        lgbm_weight = np.clip(lgbm_weight, 0.3, 0.95)

        pred_blend = lgbm_weight * pred_lgbm + (1 - lgbm_weight) * pred_ridge
        oof_preds[val_idx] = pred_blend
        fold_rmses.append(rmse(y_val, pred_blend))

        joblib.dump(pipe, models_dir / f"pipe_fold{fold_idx}.joblib")
        joblib.dump(model_lgbm, models_dir / f"model_fold{fold_idx}.joblib")

    mean_score = rmse(y_train, oof_preds)
    print(f"  {name}: RMSE={mean_score:.4f}  folds=[{', '.join(f'{f:.1f}' for f in fold_rmses)}]")

    np.save(run_dir / "oof_preds.npy", oof_preds)
    with open(run_dir / "metrics.json", "w") as f:
        json.dump({"mean_rmse": float(mean_score), "fold_rmses": [float(x) for x in fold_rmses]}, f, indent=2)

    return mean_score


def main():
    print("=== Phase 14: Hybrid Approaches ===\n")

    cfg = Config(
        train_file="train.csv", test_file="test.csv",
        id_col="sample number", target_col="含水率",
        group_col="species number",
    )
    X_train, y_train, ids = load_train(cfg, DATA_DIR)
    df = pd.read_csv(DATA_DIR / "train.csv", encoding="cp932")
    groups = df["species number"].values

    print(f"Train: {X_train.shape}")

    results = []

    best_preprocess = [
        {"name": "emsc", "poly_order": 2},
        {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
        {"name": "binning", "bin_size": 8},
        {"name": "standard_scaler"},
    ]
    lgbm_params = {
        "n_estimators": 400, "max_depth": 5, "num_leaves": 20,
        "learning_rate": 0.05, "min_child_samples": 20,
        "subsample": 0.7, "colsample_bytree": 0.7,
        "reg_alpha": 0.1, "reg_lambda": 1.0,
        "verbose": -1, "n_jobs": -1,
    }

    # === Section A: Hybrid LGBM + Ridge blending ===
    print("\n--- Section A: LGBM + Ridge blending ---")
    for bw in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        for alpha in [10.0, 100.0, 1000.0]:
            try:
                score = hybrid_lgbm_linear_cv(
                    f"hybrid_bw{bw}_a{alpha}",
                    X_train, y_train, groups,
                    best_preprocess, lgbm_params,
                    linear_alpha=alpha, blend_weight=bw,
                )
                results.append(("hybrid", f"bw{bw}_a{alpha}", score))
            except Exception as e:
                print(f"  ERROR hybrid: {e}")

    # === Section B: Adaptive blending ===
    print("\n--- Section B: Adaptive blending ---")
    try:
        score = adaptive_blend_cv(
            "adaptive_blend",
            X_train, y_train, groups,
            best_preprocess, lgbm_params,
        )
        results.append(("adaptive", "", score))
    except Exception as e:
        print(f"  ERROR adaptive: {e}")
        traceback.print_exc()

    # === Section C: Post-processing (clip extension) ===
    print("\n--- Section C: Post-processing ---")

    # C1: Scale up predictions that are near the max of LGBM output
    def scale_up_max(pred, y_tr, X_tr_t, X_val_t):
        """Scale predictions that are near the training max upward."""
        train_max = y_tr.max()
        train_95 = np.percentile(y_tr, 95)
        high_mask = pred > train_95
        if high_mask.any():
            # Scale high predictions by a factor
            scale = 1.1
            pred[high_mask] = train_95 + (pred[high_mask] - train_95) * scale
        return pred

    try:
        score = run_cv_with_postprocess(
            "pp_scaleup_1.1",
            X_train, y_train, groups,
            best_preprocess, "lgbm", lgbm_params,
            postprocess_fn=scale_up_max,
        )
        results.append(("pp_scaleup", "1.1", score))
    except Exception as e:
        print(f"  ERROR pp: {e}")

    # C2: Quantile-based calibration
    def quantile_calibration(pred, y_tr, X_tr_t, X_val_t):
        """Match prediction quantile distribution to training distribution."""
        from scipy.stats import rankdata
        ranks = rankdata(pred) / (len(pred) + 1)
        calibrated = np.quantile(y_tr, ranks)
        return calibrated

    try:
        score = run_cv_with_postprocess(
            "pp_quantile_cal",
            X_train, y_train, groups,
            best_preprocess, "lgbm", lgbm_params,
            postprocess_fn=quantile_calibration,
        )
        results.append(("pp_quantile", "", score))
    except Exception as e:
        print(f"  ERROR pp_q: {e}")

    # === Section D: LGBM with sqrt target ===
    print("\n--- Section D: Target transforms + LGBM ---")
    # sqrt transform: can help with right-skewed targets
    gkf = GroupKFold(n_splits=5)
    for transform_name, fwd, inv in [
        ("sqrt", np.sqrt, np.square),
        ("log1p", np.log1p, np.expm1),
        ("cbrt", lambda x: np.cbrt(x), lambda x: np.power(x, 3)),
    ]:
        oof = np.full(len(y_train), np.nan)
        fold_rmses_t = []
        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            y_tr_t = fwd(y_tr)

            pipe = build_preprocess_pipeline(best_preprocess)
            X_tr_t2 = pipe.fit_transform(X_tr)
            X_val_t2 = pipe.transform(X_val)

            model = create_model("lgbm", lgbm_params)
            model.fit(X_tr_t2, y_tr_t)
            pred_t = model.predict(X_val_t2).ravel()
            pred = inv(pred_t)

            oof[val_idx] = pred
            fold_rmses_t.append(rmse(y_val, pred))

        score = rmse(y_train, oof)
        print(f"  target_{transform_name}: RMSE={score:.4f}  folds=[{', '.join(f'{f:.1f}' for f in fold_rmses_t)}]")
        results.append(("target", transform_name, score))

        # Save
        run_dir = RUNS_DIR / f"target_{transform_name}_lgbm_{time.strftime('%Y%m%d_%H%M%S')}"
        run_dir.mkdir(parents=True, exist_ok=True)
        np.save(run_dir / "oof_preds.npy", oof)
        with open(run_dir / "metrics.json", "w") as f:
            json.dump({"mean_rmse": float(score), "fold_rmses": [float(x) for x in fold_rmses_t]}, f, indent=2)

    # === Section E: Multiple seeds with best config ===
    print("\n--- Section E: Multi-seed ensemble ---")
    seed_oofs = []
    for seed in range(10):
        p = dict(lgbm_params)
        p["random_state"] = seed * 7 + 42
        p["bagging_seed"] = seed * 11 + 17

        oof = np.full(len(y_train), np.nan)
        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            pipe = build_preprocess_pipeline(best_preprocess)
            X_tr_t2 = pipe.fit_transform(X_tr)
            X_val_t2 = pipe.transform(X_val)

            model = create_model("lgbm", p)
            model.fit(X_tr_t2, y_tr)
            oof[val_idx] = model.predict(X_val_t2).ravel()

        score = rmse(y_train, oof)
        seed_oofs.append(oof)
        print(f"  seed_{seed}: RMSE={score:.4f}")

    # Average of all seeds
    avg_oof = np.mean(seed_oofs, axis=0)
    avg_score = rmse(y_train, avg_oof)
    print(f"  10-seed average: RMSE={avg_score:.4f}")
    results.append(("multiseed_10", "", avg_score))

    run_dir = RUNS_DIR / f"multiseed_10_avg_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "oof_preds.npy", avg_oof)
    with open(run_dir / "metrics.json", "w") as f:
        json.dump({"mean_rmse": float(avg_score), "fold_rmses": []}, f, indent=2)

    # Summary
    print("\n\n=== Phase 14 Summary ===")
    results.sort(key=lambda x: x[2])
    for method, params, score in results:
        print(f"  {score:.4f}  {method} {params}")

    if results:
        print(f"\nBest: {results[0][2]:.4f} ({results[0][0]} {results[0][1]})")
        print(f"Baseline: 17.745")


if __name__ == "__main__":
    main()
