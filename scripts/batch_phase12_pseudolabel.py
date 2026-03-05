#!/usr/bin/env python
"""Phase 12: Pseudo-labeling for domain adaptation.

Since test species are completely different from train species,
pseudo-labeling can help by:
1. Predicting test labels with current best model
2. Adding confident test predictions to training data
3. Retraining with expanded coverage

This is especially useful when train and test distributions differ.
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


def get_test_predictions(X_test, top_n=5):
    """Get test predictions from top N models (fold-averaged)."""
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
        model_pred = np.mean(fold_preds, axis=0)
        all_preds.append(model_pred)
        print(f"  Test pred from {run['name'][:40]}: [{model_pred.min():.1f}, {model_pred.max():.1f}]")

    # Average across models
    test_pred = np.mean(all_preds, axis=0)
    # Also get prediction variance for confidence
    test_std = np.std(all_preds, axis=0)

    return test_pred, test_std


def run_pseudolabel_cv(
    name, X_train, y_train, groups, X_pseudo, y_pseudo,
    preprocess_cfg, model_type, model_params,
    pseudo_weight=1.0, n_folds=5,
):
    """Run CV with pseudo-labeled data added to training."""
    gkf = GroupKFold(n_splits=n_folds)
    oof_preds = np.full(len(y_train), np.nan)
    fold_rmses = []

    # Create pseudo groups (assign group=-1 for pseudo data)
    n_pseudo = len(y_pseudo)

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        # Add pseudo-labeled data to training
        X_tr_aug = np.vstack([X_tr, X_pseudo])
        y_tr_aug = np.concatenate([y_tr, y_pseudo])

        # Build pipeline and transform
        pipe = build_preprocess_pipeline(preprocess_cfg)
        X_tr_t = pipe.fit_transform(X_tr_aug)
        X_val_t = pipe.transform(X_val)

        # Sample weights: real data = 1.0, pseudo data = pseudo_weight
        sample_weights = np.concatenate([
            np.ones(len(y_tr)),
            np.full(n_pseudo, pseudo_weight),
        ])

        # Create model
        model = create_model(model_type, model_params)

        # Fit with sample weights if supported
        try:
            model.fit(X_tr_t, y_tr_aug, sample_weight=sample_weights)
        except TypeError:
            # Model doesn't support sample_weight
            model.fit(X_tr_t, y_tr_aug)

        val_pred = model.predict(X_val_t).ravel()
        oof_preds[val_idx] = val_pred
        fold_rmses.append(rmse(y_val, val_pred))

    mean_score = rmse(y_train, oof_preds)
    print(f"  {name}: RMSE={mean_score:.4f}  folds=[{', '.join(f'{f:.1f}' for f in fold_rmses)}]")

    # Save
    run_dir = RUNS_DIR / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "oof_preds.npy", oof_preds)
    with open(run_dir / "metrics.json", "w") as f:
        json.dump({"mean_rmse": float(mean_score), "fold_rmses": [float(x) for x in fold_rmses]}, f, indent=2)

    return mean_score


def main():
    print("=== Phase 12: Pseudo-labeling ===\n")

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
    print(f"y range: [{y_train.min():.1f}, {y_train.max():.1f}]")

    # Step 1: Get test predictions
    print("\n--- Getting test predictions ---")
    test_pred, test_std = get_test_predictions(X_test, top_n=5)
    print(f"Test pred: [{test_pred.min():.1f}, {test_pred.max():.1f}], mean={test_pred.mean():.1f}")
    print(f"Test std: [{test_std.min():.2f}, {test_std.max():.2f}], mean={test_std.mean():.2f}")

    # Best preprocessing config
    best_preprocess = [
        {"name": "emsc", "poly_order": 2},
        {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
        {"name": "binning", "bin_size": 8},
        {"name": "standard_scaler"},
    ]
    best_model = "lgbm"
    best_params = {
        "n_estimators": 400, "max_depth": 5, "num_leaves": 20,
        "learning_rate": 0.05, "min_child_samples": 20,
        "subsample": 0.7, "colsample_bytree": 0.7,
        "reg_alpha": 0.1, "reg_lambda": 1.0,
        "verbose": -1, "n_jobs": -1,
    }

    results = []

    # Experiment 1: Add ALL pseudo-labeled test data
    print("\n--- All pseudo-labels ---")
    for pw in [0.1, 0.3, 0.5, 0.7, 1.0]:
        try:
            score = run_pseudolabel_cv(
                f"pl_all_w{pw}",
                X_train, y_train, groups,
                X_test, test_pred,
                best_preprocess, best_model, best_params,
                pseudo_weight=pw,
            )
            results.append(("pl_all", pw, score))
        except Exception as e:
            print(f"  ERROR pl_all_w{pw}: {e}")
            traceback.print_exc()

    # Experiment 2: Only confident pseudo-labels (low std)
    print("\n--- Confident pseudo-labels (low variance) ---")
    for conf_pct in [25, 50, 75]:
        threshold = np.percentile(test_std, conf_pct)
        confident_mask = test_std <= threshold
        X_conf = X_test[confident_mask]
        y_conf = test_pred[confident_mask]
        print(f"  Confidence {conf_pct}%: {confident_mask.sum()} samples (std<={threshold:.2f})")

        for pw in [0.3, 0.5, 1.0]:
            try:
                score = run_pseudolabel_cv(
                    f"pl_conf{conf_pct}_w{pw}",
                    X_train, y_train, groups,
                    X_conf, y_conf,
                    best_preprocess, best_model, best_params,
                    pseudo_weight=pw,
                )
                results.append(("pl_confident", f"p{conf_pct}_w{pw}", score))
            except Exception as e:
                print(f"  ERROR: {e}")

    # Experiment 3: Iterative pseudo-labeling (2 rounds)
    print("\n--- Iterative pseudo-labeling ---")
    for pw in [0.3, 0.5]:
        try:
            # Round 1: Train with pseudo-labels
            X_aug = np.vstack([X_train, X_test])
            y_aug = np.concatenate([y_train, test_pred])

            # Retrain to get round-2 predictions
            pipe = build_preprocess_pipeline(best_preprocess)
            X_all_t = pipe.fit_transform(X_aug)
            sample_w = np.concatenate([np.ones(len(y_train)), np.full(len(test_pred), pw)])

            model = create_model(best_model, best_params)
            model.fit(X_all_t[:len(y_train)], y_train)  # Fit on train portion only after transform

            # Actually need to do this properly with the augmented set
            # Get round 2 predictions on test
            pipe2 = build_preprocess_pipeline(best_preprocess)
            X_train_t2 = pipe2.fit_transform(X_aug)
            model2 = create_model(best_model, best_params)
            try:
                model2.fit(X_train_t2, y_aug, sample_weight=sample_w)
            except TypeError:
                model2.fit(X_train_t2, y_aug)

            X_test_t2 = pipe2.transform(X_test)
            test_pred_r2 = model2.predict(X_test_t2).ravel()
            print(f"  Round 2 test pred: [{test_pred_r2.min():.1f}, {test_pred_r2.max():.1f}]")

            # Now use round 2 predictions for final CV
            score = run_pseudolabel_cv(
                f"pl_iter2_w{pw}",
                X_train, y_train, groups,
                X_test, test_pred_r2,
                best_preprocess, best_model, best_params,
                pseudo_weight=pw,
            )
            results.append(("pl_iterative", f"w{pw}", score))
        except Exception as e:
            print(f"  ERROR pl_iter2_w{pw}: {e}")
            traceback.print_exc()

    # Experiment 4: Pseudo-labels with different model types
    print("\n--- Pseudo-labels with XGBoost ---")
    xgb_params = {
        "n_estimators": 400, "max_depth": 5, "learning_rate": 0.05,
        "subsample": 0.7, "colsample_bytree": 0.7,
        "reg_alpha": 0.1, "reg_lambda": 1.0,
        "verbosity": 0, "n_jobs": -1,
    }
    for pw in [0.3, 0.5]:
        try:
            score = run_pseudolabel_cv(
                f"pl_xgb_w{pw}",
                X_train, y_train, groups,
                X_test, test_pred,
                best_preprocess, "xgb", xgb_params,
                pseudo_weight=pw,
            )
            results.append(("pl_xgb", pw, score))
        except Exception as e:
            print(f"  ERROR: {e}")

    # Experiment 5: Pseudo-labels with different preprocessing
    print("\n--- Pseudo-labels with SNV+EMSC ---")
    snv_emsc_preprocess = [
        {"name": "snv"},
        {"name": "emsc", "poly_order": 2},
        {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
        {"name": "binning", "bin_size": 8},
        {"name": "standard_scaler"},
    ]
    for pw in [0.3, 0.5]:
        try:
            score = run_pseudolabel_cv(
                f"pl_snvemsc_w{pw}",
                X_train, y_train, groups,
                X_test, test_pred,
                snv_emsc_preprocess, best_model, best_params,
                pseudo_weight=pw,
            )
            results.append(("pl_snvemsc", pw, score))
        except Exception as e:
            print(f"  ERROR: {e}")

    # Summary
    print("\n\n=== Phase 12 Summary ===")
    results.sort(key=lambda x: x[2])
    for method, params, score in results:
        print(f"  {score:.4f}  {method} {params}")

    if results:
        print(f"\nBest: {results[0][2]:.4f} ({results[0][0]} {results[0][1]})")
        print(f"Baseline (no pseudo): 17.745")


if __name__ == "__main__":
    main()
