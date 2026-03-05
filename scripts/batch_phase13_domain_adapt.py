#!/usr/bin/env python
"""Phase 13: Domain adaptation and novel approaches.

Since train/test species are completely disjoint, domain adaptation can help.
Also tries CORAL alignment, feature-space tricks, and novel LGBM configurations.
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
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spectral_challenge.config import Config
from spectral_challenge.data.load import load_train, load_test
from spectral_challenge.metrics import rmse
from spectral_challenge.models.factory import create_model
from spectral_challenge.preprocess.pipeline import build_preprocess_pipeline

DATA_DIR = Path("data/raw")
RUNS_DIR = Path("runs")


def coral_transform(X_source, X_target):
    """CORAL: Correlate alignment - align source feature distribution to target.

    Returns transformed X_source that has similar covariance to X_target.
    """
    n_s = X_source.shape[0]
    n_t = X_target.shape[0]

    # Center
    mu_s = X_source.mean(axis=0)
    mu_t = X_target.mean(axis=0)

    X_s_c = X_source - mu_s
    X_t_c = X_target - mu_t

    # Covariance
    C_s = (X_s_c.T @ X_s_c) / (n_s - 1) + np.eye(X_source.shape[1]) * 1e-6
    C_t = (X_t_c.T @ X_t_c) / (n_t - 1) + np.eye(X_target.shape[1]) * 1e-6

    # Whitening + coloring
    # X_aligned = X_s_c @ C_s^{-1/2} @ C_t^{1/2} + mu_t
    from scipy.linalg import sqrtm

    C_s_inv_sqrt = np.real(sqrtm(np.linalg.inv(C_s)))
    C_t_sqrt = np.real(sqrtm(C_t))

    transform = C_s_inv_sqrt @ C_t_sqrt
    X_aligned = X_s_c @ transform + mu_t

    return X_aligned.astype(np.float64), transform, mu_s, mu_t


def run_cv_experiment(name, X_train, y_train, groups, preprocess_cfg, model_type, model_params, n_folds=5):
    """Run a standard CV experiment."""
    gkf = GroupKFold(n_splits=n_folds)
    oof_preds = np.full(len(y_train), np.nan)
    fold_rmses = []

    models_dir_base = RUNS_DIR / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}"
    models_dir = models_dir_base / "models"
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
        oof_preds[val_idx] = val_pred
        fold_rmses.append(rmse(y_val, val_pred))

        # Save fold models for test prediction
        joblib.dump(pipe, models_dir / f"pipe_fold{fold_idx}.joblib")
        joblib.dump(model, models_dir / f"model_fold{fold_idx}.joblib")

    mean_score = rmse(y_train, oof_preds)
    print(f"  {name}: RMSE={mean_score:.4f}  folds=[{', '.join(f'{f:.1f}' for f in fold_rmses)}]")

    np.save(models_dir_base / "oof_preds.npy", oof_preds)
    with open(models_dir_base / "metrics.json", "w") as f:
        json.dump({"mean_rmse": float(mean_score), "fold_rmses": [float(x) for x in fold_rmses]}, f, indent=2)

    return mean_score


def main():
    print("=== Phase 13: Domain Adaptation & Novel Approaches ===\n")

    cfg = Config(
        train_file="train.csv", test_file="test.csv",
        id_col="sample number", target_col="含水率",
        group_col="species number",
    )
    X_train, y_train, ids = load_train(cfg, DATA_DIR)
    X_test, test_ids = load_test(cfg, DATA_DIR)
    df = pd.read_csv(DATA_DIR / "train.csv", encoding="cp932")
    groups = df["species number"].values

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    results = []

    # Best preprocessing / model config
    best_preprocess = [
        {"name": "emsc", "poly_order": 2},
        {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
        {"name": "binning", "bin_size": 8},
        {"name": "standard_scaler"},
    ]
    best_params = {
        "n_estimators": 400, "max_depth": 5, "num_leaves": 20,
        "learning_rate": 0.05, "min_child_samples": 20,
        "subsample": 0.7, "colsample_bytree": 0.7,
        "reg_alpha": 0.1, "reg_lambda": 1.0,
        "verbose": -1, "n_jobs": -1,
    }

    # === Section A: CORAL Domain Adaptation ===
    print("\n--- Section A: CORAL Domain Adaptation ---")
    # Apply CORAL to align train features toward test distribution
    # Note: CORAL works on processed features, not raw spectra
    try:
        pipe_coral = build_preprocess_pipeline(best_preprocess)
        X_train_proc = pipe_coral.fit_transform(X_train)
        X_test_proc = pipe_coral.transform(X_test)

        X_aligned, transform, mu_s, mu_t = coral_transform(X_train_proc, X_test_proc)
        print(f"  CORAL aligned: shape={X_aligned.shape}")

        # Test with aligned features directly (no preprocessing, already processed)
        gkf = GroupKFold(n_splits=5)
        oof = np.full(len(y_train), np.nan)
        fold_rmses = []

        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_aligned, y_train, groups)):
            X_tr, X_val = X_aligned[train_idx], X_aligned[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            model = create_model("lgbm", best_params)
            model.fit(X_tr, y_tr)
            pred = model.predict(X_val).ravel()
            oof[val_idx] = pred
            fold_rmses.append(rmse(y_val, pred))

        score = rmse(y_train, oof)
        print(f"  coral_lgbm: RMSE={score:.4f}  folds=[{', '.join(f'{f:.1f}' for f in fold_rmses)}]")
        results.append(("coral_lgbm", "", score))

        # Save
        run_dir = RUNS_DIR / f"coral_lgbm_{time.strftime('%Y%m%d_%H%M%S')}"
        run_dir.mkdir(parents=True, exist_ok=True)
        np.save(run_dir / "oof_preds.npy", oof)
        with open(run_dir / "metrics.json", "w") as f:
            json.dump({"mean_rmse": float(score), "fold_rmses": [float(x) for x in fold_rmses]}, f, indent=2)
    except Exception as e:
        print(f"  ERROR CORAL: {e}")
        traceback.print_exc()

    # === Section B: LGBM with different configurations ===
    print("\n--- Section B: LGBM Advanced Configs ---")

    # B1: DART boosting
    print("\n  B1: DART boosting")
    for drop_rate in [0.05, 0.1, 0.2]:
        dart_params = {
            "boosting_type": "dart",
            "n_estimators": 500,
            "max_depth": 5,
            "num_leaves": 20,
            "learning_rate": 0.05,
            "drop_rate": drop_rate,
            "skip_drop": 0.5,
            "min_child_samples": 20,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "verbose": -1,
            "n_jobs": -1,
        }
        try:
            score = run_cv_experiment(
                f"dart_dr{drop_rate}", X_train, y_train, groups,
                best_preprocess, "lgbm", dart_params
            )
            results.append(("dart", drop_rate, score))
        except Exception as e:
            print(f"  ERROR dart: {e}")

    # B2: GOSS boosting
    print("\n  B2: GOSS boosting")
    goss_params = {
        "boosting_type": "goss",
        "n_estimators": 400,
        "max_depth": 5,
        "num_leaves": 20,
        "learning_rate": 0.05,
        "top_rate": 0.2,
        "other_rate": 0.1,
        "min_child_samples": 20,
        "colsample_bytree": 0.7,
        "verbose": -1,
        "n_jobs": -1,
    }
    try:
        score = run_cv_experiment(
            "goss_lgbm", X_train, y_train, groups,
            best_preprocess, "lgbm", goss_params
        )
        results.append(("goss", "", score))
    except Exception as e:
        print(f"  ERROR goss: {e}")

    # B3: Very deep LGBM with strong regularization
    print("\n  B3: Deep LGBM with strong reg")
    for depth in [8, 10, 15]:
        deep_params = {
            "n_estimators": 800,
            "max_depth": depth,
            "num_leaves": min(2**depth - 1, 63),
            "learning_rate": 0.02,
            "min_child_samples": 30,
            "subsample": 0.6,
            "colsample_bytree": 0.5,
            "reg_alpha": 1.0,
            "reg_lambda": 10.0,
            "min_split_gain": 0.01,
            "verbose": -1,
            "n_jobs": -1,
        }
        try:
            score = run_cv_experiment(
                f"deep_d{depth}_reg", X_train, y_train, groups,
                best_preprocess, "lgbm", deep_params
            )
            results.append(("deep_lgbm", depth, score))
        except Exception as e:
            print(f"  ERROR deep: {e}")

    # B4: Very shallow LGBM (linear-like)
    print("\n  B4: Shallow LGBM")
    shallow_params = {
        "n_estimators": 1000,
        "max_depth": 2,
        "num_leaves": 4,
        "learning_rate": 0.02,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbose": -1,
        "n_jobs": -1,
    }
    try:
        score = run_cv_experiment(
            "shallow_d2_l4", X_train, y_train, groups,
            best_preprocess, "lgbm", shallow_params
        )
        results.append(("shallow_lgbm", "d2", score))
    except Exception as e:
        print(f"  ERROR shallow: {e}")

    # B5: MAE objective (L1 loss — more robust to outliers)
    print("\n  B5: MAE objective")
    mae_params = {
        "objective": "mae",
        "n_estimators": 500,
        "max_depth": 5,
        "num_leaves": 20,
        "learning_rate": 0.05,
        "min_child_samples": 20,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "verbose": -1,
        "n_jobs": -1,
    }
    try:
        score = run_cv_experiment(
            "mae_lgbm", X_train, y_train, groups,
            best_preprocess, "lgbm", mae_params
        )
        results.append(("mae_lgbm", "", score))
    except Exception as e:
        print(f"  ERROR mae: {e}")

    # B6: Huber loss with correct parameter (alpha)
    print("\n  B6: Huber loss (correct alpha param)")
    for alpha in [0.7, 0.8, 0.9, 0.95]:
        huber_params = {
            "objective": "huber",
            "alpha": alpha,  # This is the correct LGBM parameter
            "n_estimators": 500,
            "max_depth": 5,
            "num_leaves": 20,
            "learning_rate": 0.05,
            "min_child_samples": 20,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "verbose": -1,
            "n_jobs": -1,
        }
        try:
            score = run_cv_experiment(
                f"huber_a{alpha}_lgbm", X_train, y_train, groups,
                best_preprocess, "lgbm", huber_params
            )
            results.append(("huber", alpha, score))
        except Exception as e:
            print(f"  ERROR huber: {e}")

    # === Section C: Different preprocessing combos ===
    print("\n--- Section C: Preprocessing Variations ---")

    # C1: No binning (use all features)
    print("\n  C1: No binning")
    nobin_preprocess = [
        {"name": "emsc", "poly_order": 2},
        {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
        {"name": "standard_scaler"},
    ]
    try:
        score = run_cv_experiment(
            "emsc_nobin", X_train, y_train, groups,
            nobin_preprocess, "lgbm", best_params
        )
        results.append(("nobin", "", score))
    except Exception as e:
        print(f"  ERROR nobin: {e}")

    # C2: Larger bin
    print("\n  C2: Bin=16")
    bin16_preprocess = [
        {"name": "emsc", "poly_order": 2},
        {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
        {"name": "binning", "bin_size": 16},
        {"name": "standard_scaler"},
    ]
    try:
        score = run_cv_experiment(
            "emsc_bin16", X_train, y_train, groups,
            bin16_preprocess, "lgbm", best_params
        )
        results.append(("bin16", "", score))
    except Exception as e:
        print(f"  ERROR bin16: {e}")

    # C3: Bin=4 (smaller)
    print("\n  C3: Bin=4")
    bin4_preprocess = [
        {"name": "emsc", "poly_order": 2},
        {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
        {"name": "binning", "bin_size": 4},
        {"name": "standard_scaler"},
    ]
    try:
        score = run_cv_experiment(
            "emsc_bin4", X_train, y_train, groups,
            bin4_preprocess, "lgbm", best_params
        )
        results.append(("bin4", "", score))
    except Exception as e:
        print(f"  ERROR bin4: {e}")

    # C4: SG deriv=2
    print("\n  C4: SG 2nd deriv")
    deriv2_preprocess = [
        {"name": "emsc", "poly_order": 2},
        {"name": "sg", "window_length": 11, "polyorder": 3, "deriv": 2},
        {"name": "binning", "bin_size": 8},
        {"name": "standard_scaler"},
    ]
    try:
        score = run_cv_experiment(
            "emsc_deriv2", X_train, y_train, groups,
            deriv2_preprocess, "lgbm", best_params
        )
        results.append(("deriv2", "", score))
    except Exception as e:
        print(f"  ERROR deriv2: {e}")

    # C5: EMSC + area normalize + SG
    print("\n  C5: EMSC + area normalize")
    area_preprocess = [
        {"name": "emsc", "poly_order": 2},
        {"name": "area_normalize"},
        {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
        {"name": "binning", "bin_size": 8},
        {"name": "standard_scaler"},
    ]
    try:
        score = run_cv_experiment(
            "emsc_area_sg", X_train, y_train, groups,
            area_preprocess, "lgbm", best_params
        )
        results.append(("emsc_area", "", score))
    except Exception as e:
        print(f"  ERROR emsc_area: {e}")

    # === Section D: Multi-seed ensembles ===
    print("\n--- Section D: Multi-seed ---")
    for seed in [0, 1, 2, 3, 4]:
        seed_params = dict(best_params)
        seed_params["random_state"] = seed * 11 + 7
        seed_params["bagging_seed"] = seed * 13 + 5
        seed_params["feature_fraction_seed"] = seed * 17 + 3
        try:
            score = run_cv_experiment(
                f"multiseed_s{seed}", X_train, y_train, groups,
                best_preprocess, "lgbm", seed_params
            )
            results.append(("multiseed", seed, score))
        except Exception as e:
            print(f"  ERROR seed: {e}")

    # === Section E: XGBoost with tuned params ===
    print("\n--- Section E: XGBoost tuned ---")
    for lr, n_est in [(0.05, 400), (0.02, 800), (0.1, 200)]:
        xgb_params = {
            "n_estimators": n_est,
            "max_depth": 5,
            "learning_rate": lr,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "min_child_weight": 5,
            "verbosity": 0,
            "n_jobs": -1,
        }
        try:
            score = run_cv_experiment(
                f"xgb_lr{lr}_n{n_est}", X_train, y_train, groups,
                best_preprocess, "xgb", xgb_params
            )
            results.append(("xgb", f"lr{lr}_n{n_est}", score))
        except Exception as e:
            print(f"  ERROR xgb: {e}")

    # Summary
    print("\n\n=== Phase 13 Summary ===")
    results.sort(key=lambda x: x[2])
    for method, params, score in results:
        print(f"  {score:.4f}  {method} {params}")

    if results:
        print(f"\nBest: {results[0][2]:.4f} ({results[0][0]} {results[0][1]})")
        print(f"Baseline: 17.745")


if __name__ == "__main__":
    main()
