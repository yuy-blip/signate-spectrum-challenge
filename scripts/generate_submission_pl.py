#!/usr/bin/env python
"""Generate submission using pseudo-labeling (best approach so far).

Train LGBM on train + pseudo-labeled test data, then predict test.
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spectral_challenge.config import Config
from spectral_challenge.data.load import load_train, load_test
from spectral_challenge.metrics import rmse
from spectral_challenge.models.factory import create_model
from spectral_challenge.preprocess.pipeline import build_preprocess_pipeline

DATA_DIR = Path("data/raw")
RUNS_DIR = Path("runs")
SUBMISSIONS_DIR = Path("submissions")


def get_best_test_predictions(X_test, top_n=10):
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
        print(f"  {run['rmse']:.4f}  {run['name'][:50]}")

    return np.mean(all_preds, axis=0)


def train_pl_model(X_train, y_train, X_pseudo, y_pseudo, preprocess_cfg, model_type, model_params, pseudo_weight=0.5):
    """Train a single model on train + pseudo-labeled data."""
    X_aug = np.vstack([X_train, X_pseudo])
    y_aug = np.concatenate([y_train, y_pseudo])
    sample_weights = np.concatenate([
        np.ones(len(y_train)),
        np.full(len(y_pseudo), pseudo_weight),
    ])

    pipe = build_preprocess_pipeline(preprocess_cfg)
    X_aug_t = pipe.fit_transform(X_aug)

    model = create_model(model_type, model_params)
    try:
        model.fit(X_aug_t, y_aug, sample_weight=sample_weights)
    except TypeError:
        model.fit(X_aug_t, y_aug)

    return pipe, model


def main():
    print("=== Final Submission with Pseudo-labeling ===\n")

    cfg = Config(
        train_file="train.csv", test_file="test.csv",
        id_col="sample number", target_col="含水率",
        group_col="species number",
    )
    X_train, y_train, ids = load_train(cfg, DATA_DIR)
    X_test, test_ids = load_test(cfg, DATA_DIR)

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Step 1: Get initial test predictions
    print("\n--- Getting test predictions from top models ---")
    test_pred_initial = get_best_test_predictions(X_test, top_n=10)
    print(f"Initial test pred: [{test_pred_initial.min():.1f}, {test_pred_initial.max():.1f}], mean={test_pred_initial.mean():.1f}")

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

    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Step 2: Generate multi-seed PL ensemble predictions
    print("\n--- Multi-seed PL ensemble ---")
    all_test_preds = []

    for seed in range(10):
        params = dict(best_params)
        params["random_state"] = seed * 7 + 42
        params["bagging_seed"] = seed * 11 + 17

        for pw in [0.25, 0.5, 1.0]:
            pipe, model = train_pl_model(
                X_train, y_train, X_test, test_pred_initial,
                best_preprocess, "lgbm", params,
                pseudo_weight=pw,
            )
            X_test_t = pipe.transform(X_test)
            pred = model.predict(X_test_t).ravel()
            all_test_preds.append(pred)
            print(f"  seed={seed} pw={pw}: pred range [{pred.min():.1f}, {pred.max():.1f}]")

    # Step 3: Also get predictions from existing top PL runs
    print("\n--- Loading existing PL run predictions ---")
    pl_runs = []
    for f in RUNS_DIR.glob("pl_*/metrics.json"):
        with open(f) as fh:
            m = json.load(fh)
        mean_rmse = m.get("mean_rmse", 999)
        if mean_rmse > 17:
            continue
        models_dir = f.parent / "models"
        has_models = all(
            (models_dir / f"model_fold{i}.joblib").exists() and
            (models_dir / f"pipe_fold{i}.joblib").exists()
            for i in range(5)
        )
        if has_models:
            pl_runs.append({"name": f.parent.name, "rmse": mean_rmse, "run_dir": f.parent})

    pl_runs.sort(key=lambda x: x["rmse"])
    print(f"Found {len(pl_runs)} PL runs with RMSE<17")

    for run in pl_runs[:5]:
        fold_preds = []
        for fold_idx in range(5):
            pipe = joblib.load(run["run_dir"] / "models" / f"pipe_fold{fold_idx}.joblib")
            model = joblib.load(run["run_dir"] / "models" / f"model_fold{fold_idx}.joblib")
            X_t = pipe.transform(X_test)
            fold_preds.append(model.predict(X_t).ravel())
        model_pred = np.mean(fold_preds, axis=0)
        all_test_preds.append(model_pred)
        print(f"  {run['rmse']:.4f}  {run['name'][:50]}  [{model_pred.min():.1f}, {model_pred.max():.1f}]")

    # Step 4: Also include non-PL top models
    print("\n--- Including top non-PL models ---")
    non_pl_runs = []
    for f in RUNS_DIR.glob("*/metrics.json"):
        name = f.parent.name
        if name.startswith("pl_") or name.startswith("stack_"):
            continue
        with open(f) as fh:
            m = json.load(fh)
        mean_rmse = m.get("mean_rmse", 999)
        if mean_rmse > 19:
            continue
        models_dir = f.parent / "models"
        has_models = all(
            (models_dir / f"model_fold{i}.joblib").exists() and
            (models_dir / f"pipe_fold{i}.joblib").exists()
            for i in range(5)
        )
        if has_models:
            non_pl_runs.append({"name": name, "rmse": mean_rmse, "run_dir": f.parent})

    non_pl_runs.sort(key=lambda x: x["rmse"])
    for run in non_pl_runs[:10]:
        fold_preds = []
        for fold_idx in range(5):
            pipe = joblib.load(run["run_dir"] / "models" / f"pipe_fold{fold_idx}.joblib")
            model = joblib.load(run["run_dir"] / "models" / f"model_fold{fold_idx}.joblib")
            X_t = pipe.transform(X_test)
            fold_preds.append(model.predict(X_t).ravel())
        model_pred = np.mean(fold_preds, axis=0)
        all_test_preds.append(model_pred)
        print(f"  {run['rmse']:.4f}  {run['name'][:50]}  [{model_pred.min():.1f}, {model_pred.max():.1f}]")

    # Step 5: Final ensemble (simple average)
    print(f"\n--- Final ensemble of {len(all_test_preds)} predictions ---")
    final_pred = np.mean(all_test_preds, axis=0)
    print(f"Final pred: [{final_pred.min():.1f}, {final_pred.max():.1f}], mean={final_pred.mean():.1f}")

    sub = pd.DataFrame({"sample number": test_ids.values, "含水率": final_pred})
    path = SUBMISSIONS_DIR / f"submission_pl_ensemble_{ts}.csv"
    sub.to_csv(path, index=False)
    print(f"\nSaved: {path}")

    # Step 6: Also save PL-only ensemble (no non-PL models)
    pl_only_preds = all_test_preds[:30 + len(pl_runs[:5])]
    pl_final = np.mean(pl_only_preds, axis=0)
    sub_pl = pd.DataFrame({"sample number": test_ids.values, "含水率": pl_final})
    path_pl = SUBMISSIONS_DIR / f"submission_pl_only_{ts}.csv"
    sub_pl.to_csv(path_pl, index=False)
    print(f"Saved PL-only: {path_pl}")

    # Step 7: Iterative refinement - use ensemble prediction as new pseudo-labels
    print("\n--- Iterative refinement (2 rounds) ---")
    current_pl = final_pred.copy()
    for round_num in range(2):
        iter_preds = []
        for seed in range(5):
            params = dict(best_params)
            params["random_state"] = seed * 13 + 100 + round_num * 50
            pipe, model = train_pl_model(
                X_train, y_train, X_test, current_pl,
                best_preprocess, "lgbm", params,
                pseudo_weight=0.3,
            )
            X_test_t = pipe.transform(X_test)
            iter_preds.append(model.predict(X_test_t).ravel())

        current_pl = np.mean(iter_preds, axis=0)
        print(f"  Round {round_num+1}: [{current_pl.min():.1f}, {current_pl.max():.1f}], mean={current_pl.mean():.1f}")

    sub_iter = pd.DataFrame({"sample number": test_ids.values, "含水率": current_pl})
    path_iter = SUBMISSIONS_DIR / f"submission_pl_iterative_{ts}.csv"
    sub_iter.to_csv(path_iter, index=False)
    print(f"Saved iterative: {path_iter}")

    print("\n=== Done ===")
    print(f"Submissions saved to {SUBMISSIONS_DIR}/")


if __name__ == "__main__":
    main()
