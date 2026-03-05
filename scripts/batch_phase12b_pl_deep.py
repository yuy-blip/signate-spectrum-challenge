#!/usr/bin/env python
"""Phase 12b: Deep pseudo-labeling exploration.

Since Phase 12 showed significant improvement with pseudo-labeling
(16.60 vs 17.75 baseline), explore more thoroughly:
- Fine-grained weight tuning
- Different preprocessing with pseudo-labels
- Multi-round iterative pseudo-labeling
- Pseudo-label + ensemble
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


def get_test_predictions(X_test, top_n=10):
    """Get test predictions from top N models."""
    runs = []
    for f in RUNS_DIR.glob("*/metrics.json"):
        oof_file = f.parent / "oof_preds.npy"
        if not oof_file.exists():
            continue
        with open(f) as fh:
            m = json.load(fh)
        mean_rmse = m.get("mean_rmse", 999)
        if mean_rmse > 20:
            continue
        models_dir = f.parent / "models"
        has_models = all(
            (models_dir / f"model_fold{i}.joblib").exists() and
            (models_dir / f"pipe_fold{i}.joblib").exists()
            for i in range(5)
        )
        if not has_models:
            continue
        runs.append({"name": f.parent.name, "rmse": mean_rmse, "run_dir": f.parent})
    runs.sort(key=lambda x: x["rmse"])
    runs = runs[:top_n]

    all_preds = []
    for run in runs:
        fold_preds = []
        for fold_idx in range(5):
            pipe = joblib.load(run["run_dir"] / "models" / f"pipe_fold{fold_idx}.joblib")
            model = joblib.load(run["run_dir"] / "models" / f"model_fold{fold_idx}.joblib")
            X_t = pipe.transform(X_test)
            fold_preds.append(model.predict(X_t).ravel())
        all_preds.append(np.mean(fold_preds, axis=0))

    test_pred = np.mean(all_preds, axis=0)
    test_std = np.std(all_preds, axis=0)
    return test_pred, test_std


def run_pl_cv(name, X_train, y_train, groups, X_pseudo, y_pseudo,
              preprocess_cfg, model_type, model_params,
              pseudo_weight=1.0, save_models=True):
    """Run CV with pseudo-labeled data."""
    gkf = GroupKFold(n_splits=5)
    oof_preds = np.full(len(y_train), np.nan)
    fold_rmses = []

    ts = time.strftime('%Y%m%d_%H%M%S')
    run_dir = RUNS_DIR / f"{name}_{ts}"
    if save_models:
        models_dir = run_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir.mkdir(parents=True, exist_ok=True)

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        X_tr_aug = np.vstack([X_tr, X_pseudo])
        y_tr_aug = np.concatenate([y_tr, y_pseudo])
        sample_weights = np.concatenate([
            np.ones(len(y_tr)),
            np.full(len(y_pseudo), pseudo_weight),
        ])

        pipe = build_preprocess_pipeline(preprocess_cfg)
        X_tr_t = pipe.fit_transform(X_tr_aug)
        X_val_t = pipe.transform(X_val)

        model = create_model(model_type, model_params)
        try:
            model.fit(X_tr_t, y_tr_aug, sample_weight=sample_weights)
        except TypeError:
            model.fit(X_tr_t, y_tr_aug)

        val_pred = model.predict(X_val_t).ravel()
        oof_preds[val_idx] = val_pred
        fold_rmses.append(rmse(y_val, val_pred))

        if save_models:
            joblib.dump(pipe, models_dir / f"pipe_fold{fold_idx}.joblib")
            joblib.dump(model, models_dir / f"model_fold{fold_idx}.joblib")

    mean_score = rmse(y_train, oof_preds)
    print(f"  {name}: RMSE={mean_score:.4f}  folds=[{', '.join(f'{f:.1f}' for f in fold_rmses)}]")

    np.save(run_dir / "oof_preds.npy", oof_preds)
    with open(run_dir / "metrics.json", "w") as f:
        json.dump({
            "mean_rmse": float(mean_score),
            "fold_rmses": [float(x) for x in fold_rmses],
        }, f, indent=2)

    return mean_score, oof_preds


def main():
    print("=== Phase 12b: Deep Pseudo-labeling ===\n")

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

    # Get test predictions
    print("\n--- Getting test predictions from top 10 models ---")
    test_pred, test_std = get_test_predictions(X_test, top_n=10)
    print(f"Test pred: [{test_pred.min():.1f}, {test_pred.max():.1f}], mean={test_pred.mean():.1f}")

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

    results = []

    # === Section A: Fine-grained weight tuning ===
    print("\n--- Section A: Fine-grained weight tuning ---")
    for pw in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8, 1.0]:
        try:
            score, _ = run_pl_cv(
                f"pl_w{pw}",
                X_train, y_train, groups,
                X_test, test_pred,
                best_preprocess, "lgbm", best_params,
                pseudo_weight=pw, save_models=True,
            )
            results.append(("pl_fine", pw, score))
        except Exception as e:
            print(f"  ERROR: {e}")

    # === Section B: PL with different LGBM params ===
    print("\n--- Section B: PL with tuned LGBM ---")
    # More trees since we have more data
    for n_est, lr in [(600, 0.03), (800, 0.02), (400, 0.05), (1000, 0.01)]:
        params_v = dict(best_params)
        params_v["n_estimators"] = n_est
        params_v["learning_rate"] = lr
        try:
            score, _ = run_pl_cv(
                f"pl_n{n_est}_lr{lr}",
                X_train, y_train, groups,
                X_test, test_pred,
                best_preprocess, "lgbm", params_v,
                pseudo_weight=0.3, save_models=True,
            )
            results.append(("pl_lgbm_tuned", f"n{n_est}_lr{lr}", score))
        except Exception as e:
            print(f"  ERROR: {e}")

    # === Section C: PL with different depths ===
    print("\n--- Section C: PL with different tree depth ---")
    for depth, leaves in [(3, 8), (4, 15), (6, 31), (7, 40)]:
        params_d = dict(best_params)
        params_d["max_depth"] = depth
        params_d["num_leaves"] = leaves
        try:
            score, _ = run_pl_cv(
                f"pl_d{depth}_l{leaves}",
                X_train, y_train, groups,
                X_test, test_pred,
                best_preprocess, "lgbm", params_d,
                pseudo_weight=0.3, save_models=True,
            )
            results.append(("pl_depth", f"d{depth}_l{leaves}", score))
        except Exception as e:
            print(f"  ERROR: {e}")

    # === Section D: PL with different preprocessings ===
    print("\n--- Section D: PL with different preprocessing ---")

    # SNV + EMSC
    snv_emsc_pp = [
        {"name": "snv"},
        {"name": "emsc", "poly_order": 2},
        {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
        {"name": "binning", "bin_size": 8},
        {"name": "standard_scaler"},
    ]
    try:
        score, _ = run_pl_cv(
            "pl_snvemsc_w03",
            X_train, y_train, groups,
            X_test, test_pred,
            snv_emsc_pp, "lgbm", best_params,
            pseudo_weight=0.3, save_models=True,
        )
        results.append(("pl_snvemsc", 0.3, score))
    except Exception as e:
        print(f"  ERROR: {e}")

    # No bin
    nobin_pp = [
        {"name": "emsc", "poly_order": 2},
        {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
        {"name": "standard_scaler"},
    ]
    try:
        score, _ = run_pl_cv(
            "pl_nobin_w03",
            X_train, y_train, groups,
            X_test, test_pred,
            nobin_pp, "lgbm", best_params,
            pseudo_weight=0.3, save_models=True,
        )
        results.append(("pl_nobin", 0.3, score))
    except Exception as e:
        print(f"  ERROR: {e}")

    # Bin 16
    bin16_pp = [
        {"name": "emsc", "poly_order": 2},
        {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
        {"name": "binning", "bin_size": 16},
        {"name": "standard_scaler"},
    ]
    try:
        score, _ = run_pl_cv(
            "pl_bin16_w03",
            X_train, y_train, groups,
            X_test, test_pred,
            bin16_pp, "lgbm", best_params,
            pseudo_weight=0.3, save_models=True,
        )
        results.append(("pl_bin16", 0.3, score))
    except Exception as e:
        print(f"  ERROR: {e}")

    # === Section E: PL with XGBoost ===
    print("\n--- Section E: PL with XGBoost ---")
    xgb_params = {
        "n_estimators": 400, "max_depth": 5, "learning_rate": 0.05,
        "subsample": 0.7, "colsample_bytree": 0.7,
        "reg_alpha": 0.1, "reg_lambda": 1.0,
        "min_child_weight": 5,
        "verbosity": 0, "n_jobs": -1,
    }
    for pw in [0.2, 0.3, 0.5]:
        try:
            score, _ = run_pl_cv(
                f"pl_xgb_w{pw}",
                X_train, y_train, groups,
                X_test, test_pred,
                best_preprocess, "xgb", xgb_params,
                pseudo_weight=pw, save_models=True,
            )
            results.append(("pl_xgb", pw, score))
        except Exception as e:
            print(f"  ERROR: {e}")

    # === Section F: Multi-seed PL ===
    print("\n--- Section F: Multi-seed pseudo-labeling ---")
    seed_oofs = []
    for seed in range(5):
        p = dict(best_params)
        p["random_state"] = seed * 7 + 42
        p["bagging_seed"] = seed * 11 + 17
        try:
            score, oof = run_pl_cv(
                f"pl_seed{seed}_w03",
                X_train, y_train, groups,
                X_test, test_pred,
                best_preprocess, "lgbm", p,
                pseudo_weight=0.3, save_models=True,
            )
            seed_oofs.append(oof)
            results.append(("pl_seed", seed, score))
        except Exception as e:
            print(f"  ERROR: {e}")

    if len(seed_oofs) >= 3:
        avg_oof = np.mean(seed_oofs, axis=0)
        avg_score = rmse(y_train, avg_oof)
        print(f"\n  PL 5-seed average: RMSE={avg_score:.4f}")
        results.append(("pl_multiseed_avg", "5seed", avg_score))

        run_dir = RUNS_DIR / f"pl_multiseed5_avg_{time.strftime('%Y%m%d_%H%M%S')}"
        run_dir.mkdir(parents=True, exist_ok=True)
        np.save(run_dir / "oof_preds.npy", avg_oof)
        with open(run_dir / "metrics.json", "w") as f:
            json.dump({"mean_rmse": float(avg_score), "fold_rmses": []}, f, indent=2)

    # === Section G: Iterative PL (3 rounds) ===
    print("\n--- Section G: Iterative pseudo-labeling ---")
    current_test_pred = test_pred.copy()
    for round_num in range(3):
        print(f"\n  Round {round_num + 1}:")
        # Train with current pseudo-labels to get new predictions
        gkf = GroupKFold(n_splits=5)

        # Retrain on all data + pseudo to get new test predictions
        X_aug = np.vstack([X_train, X_test])
        y_aug = np.concatenate([y_train, current_test_pred])
        sw = np.concatenate([np.ones(len(y_train)), np.full(len(current_test_pred), 0.3)])

        pipe = build_preprocess_pipeline(best_preprocess)
        X_all_t = pipe.fit_transform(X_aug)

        model = create_model("lgbm", best_params)
        model.fit(X_all_t, y_aug, sample_weight=sw)

        # Predict on test again
        X_test_t = pipe.transform(X_test)
        new_test_pred = model.predict(X_test_t).ravel()
        diff = np.abs(new_test_pred - current_test_pred).mean()
        print(f"    New test pred range: [{new_test_pred.min():.1f}, {new_test_pred.max():.1f}], diff={diff:.4f}")
        current_test_pred = new_test_pred

        # Evaluate with CV
        score, _ = run_pl_cv(
            f"pl_iter{round_num+1}_w03",
            X_train, y_train, groups,
            X_test, current_test_pred,
            best_preprocess, "lgbm", best_params,
            pseudo_weight=0.3, save_models=(round_num == 2),
        )
        results.append(("pl_iterative", f"round{round_num+1}", score))

    # Summary
    print("\n\n=== Phase 12b Summary ===")
    results.sort(key=lambda x: x[2])
    for method, params, score in results:
        print(f"  {score:.4f}  {method} {params}")

    if results:
        print(f"\nBest: {results[0][2]:.4f} ({results[0][0]} {results[0][1]})")
        print(f"Baseline (no PL): 17.745")


if __name__ == "__main__":
    main()
