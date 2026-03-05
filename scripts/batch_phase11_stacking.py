#!/usr/bin/env python
"""Phase 11: Stacking / Blending with diverse base models.

Use OOF predictions from multiple diverse models as features for a meta-learner.
This can capture complementary strengths of different model types.
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
from scipy.optimize import minimize
from sklearn.linear_model import Ridge, ElasticNet, BayesianRidge, HuberRegressor
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spectral_challenge.config import Config
from spectral_challenge.data.load import load_train, load_test
from spectral_challenge.metrics import rmse
from spectral_challenge.models.factory import create_model
from spectral_challenge.preprocess.pipeline import build_preprocess_pipeline

DATA_DIR = Path("data/raw")
RUNS_DIR = Path("runs")


def load_all_oofs(min_rmse=15, max_rmse=25):
    """Load OOF predictions from all valid runs."""
    runs = []
    for f in RUNS_DIR.glob("*/metrics.json"):
        oof_file = f.parent / "oof_preds.npy"
        if not oof_file.exists():
            continue
        with open(f) as fh:
            m = json.load(fh)
        mean_rmse = m.get("mean_rmse", 999)
        if mean_rmse < min_rmse or mean_rmse > max_rmse:
            continue
        oof = np.load(oof_file)
        if np.any(np.isnan(oof)):
            continue
        models_dir = f.parent / "models"
        has_models = all(
            (models_dir / f"model_fold{i}.joblib").exists() and
            (models_dir / f"pipe_fold{i}.joblib").exists()
            for i in range(5)
        )
        if not has_models:
            continue
        runs.append({
            "name": f.parent.name,
            "rmse": mean_rmse,
            "oof": oof,
            "run_dir": f.parent,
            "fold_rmses": m.get("fold_rmses", []),
        })
    runs.sort(key=lambda x: x["rmse"])
    return runs


def greedy_diverse_select(runs, y, max_models=15, corr_threshold=0.99):
    """Select diverse models - avoid highly correlated OOFs."""
    if not runs:
        return []
    selected = [runs[0]]

    for r in runs[1:]:
        if len(selected) >= max_models:
            break
        # Check correlation with all selected
        max_corr = max(np.corrcoef(r["oof"], s["oof"])[0, 1] for s in selected)
        if max_corr < corr_threshold:
            # Also check if it improves ensemble
            current_avg = np.mean([s["oof"] for s in selected], axis=0)
            cur_rmse = rmse(y, current_avg)
            new_avg = np.mean([s["oof"] for s in selected] + [r["oof"]], axis=0)
            new_rmse = rmse(y, new_avg)
            if new_rmse < cur_rmse:
                selected.append(r)

    return selected


def stacking_cv(base_oofs, y, groups, meta_model, n_folds=5):
    """Train a meta-learner on OOF predictions using nested CV."""
    gkf = GroupKFold(n_splits=n_folds)
    oof_meta = np.full(len(y), np.nan)

    X_stack = np.column_stack(base_oofs)

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_stack, y, groups)):
        X_tr, X_val = X_stack[train_idx], X_stack[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        meta_model_copy = type(meta_model)(**meta_model.get_params())
        meta_model_copy.fit(X_tr, y_tr)
        oof_meta[val_idx] = meta_model_copy.predict(X_val)

    return oof_meta, rmse(y, oof_meta)


def run_stacking_experiment(name, base_runs, y, groups, meta_model, augment_features=None):
    """Run a single stacking experiment."""
    base_oofs = [r["oof"] for r in base_runs]

    if augment_features is not None:
        base_oofs = base_oofs + augment_features

    oof_meta, score = stacking_cv(base_oofs, y, groups, meta_model)

    print(f"  {name}: RMSE={score:.4f} (n_base={len(base_oofs)})")

    # Save
    run_dir = RUNS_DIR / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "oof_preds.npy", oof_meta)
    with open(run_dir / "metrics.json", "w") as f:
        json.dump({
            "mean_rmse": float(score),
            "fold_rmses": [],
            "method": "stacking",
            "n_base_models": len(base_oofs),
            "base_models": [r["name"] for r in base_runs],
        }, f, indent=2)

    return score


def main():
    print("=== Phase 11: Stacking / Blending ===\n")

    cfg = Config(
        train_file="train.csv", test_file="test.csv",
        id_col="sample number", target_col="含水率",
        group_col="species number",
    )
    X_raw, y, ids = load_train(cfg, DATA_DIR)
    df = pd.read_csv(DATA_DIR / "train.csv", encoding="cp932")
    groups = df["species number"].values

    print(f"Train: {X_raw.shape}, y range: [{y.min():.1f}, {y.max():.1f}]")

    # Load all runs
    all_runs = load_all_oofs(max_rmse=22)
    print(f"Loaded {len(all_runs)} runs with RMSE<22")

    # Select diverse base models
    diverse_runs = greedy_diverse_select(all_runs, y, max_models=20, corr_threshold=0.98)
    print(f"Selected {len(diverse_runs)} diverse models:")
    for r in diverse_runs:
        print(f"  {r['rmse']:.4f}  {r['name'][:50]}")

    results = []

    # Experiment 1: Ridge meta-learner
    print("\n--- Stacking with Ridge meta-learners ---")
    for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
        try:
            score = run_stacking_experiment(
                f"stack_ridge_a{alpha}",
                diverse_runs,
                y, groups,
                Ridge(alpha=alpha)
            )
            results.append(("stack_ridge", alpha, score))
        except Exception as e:
            print(f"  ERROR stack_ridge_a{alpha}: {e}")

    # Experiment 2: ElasticNet meta-learner
    print("\n--- Stacking with ElasticNet ---")
    for alpha in [0.1, 1.0, 10.0]:
        for l1 in [0.1, 0.5, 0.9]:
            try:
                score = run_stacking_experiment(
                    f"stack_enet_a{alpha}_l1{l1}",
                    diverse_runs,
                    y, groups,
                    ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=10000)
                )
                results.append(("stack_enet", f"a{alpha}_l1{l1}", score))
            except Exception as e:
                print(f"  ERROR: {e}")

    # Experiment 3: BayesianRidge
    print("\n--- Stacking with BayesianRidge ---")
    try:
        score = run_stacking_experiment(
            "stack_bayesian",
            diverse_runs,
            y, groups,
            BayesianRidge()
        )
        results.append(("stack_bayesian", "", score))
    except Exception as e:
        print(f"  ERROR: {e}")

    # Experiment 4: Huber meta-learner (robust to outliers)
    print("\n--- Stacking with Huber ---")
    for eps in [1.1, 1.35, 2.0]:
        try:
            score = run_stacking_experiment(
                f"stack_huber_e{eps}",
                diverse_runs,
                y, groups,
                HuberRegressor(epsilon=eps, max_iter=1000)
            )
            results.append(("stack_huber", eps, score))
        except Exception as e:
            print(f"  ERROR: {e}")

    # Experiment 5: Stacking with fewer but more diverse models
    print("\n--- Stacking with fewer models (high diversity) ---")
    diverse_few = greedy_diverse_select(all_runs, y, max_models=8, corr_threshold=0.95)
    for alpha in [1.0, 10.0, 100.0]:
        try:
            score = run_stacking_experiment(
                f"stack_diverse8_ridge_a{alpha}",
                diverse_few,
                y, groups,
                Ridge(alpha=alpha)
            )
            results.append(("stack_diverse8", alpha, score))
        except Exception as e:
            print(f"  ERROR: {e}")

    # Experiment 6: Stacking with augmented features (add raw spectral PCA)
    print("\n--- Stacking with PCA augmentation ---")
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # Build EMSC pipeline for feature extraction
    emsc_pipe = build_preprocess_pipeline([
        {"name": "emsc", "poly_order": 2},
        {"name": "sg", "window_length": 7, "polyorder": 2, "deriv": 1},
        {"name": "binning", "bin_size": 8},
        {"name": "standard_scaler"},
    ])
    X_emsc = emsc_pipe.fit_transform(X_raw)

    # PCA on processed features
    for n_comp in [5, 10, 20]:
        pca = PCA(n_components=n_comp)
        X_pca = pca.fit_transform(X_emsc)
        pca_features = [X_pca[:, i] for i in range(n_comp)]

        for alpha in [10.0, 100.0]:
            try:
                score = run_stacking_experiment(
                    f"stack_pca{n_comp}_ridge_a{alpha}",
                    diverse_runs,
                    y, groups,
                    Ridge(alpha=alpha),
                    augment_features=pca_features,
                )
                results.append(("stack_pca", f"n{n_comp}_a{alpha}", score))
            except Exception as e:
                print(f"  ERROR: {e}")

    # Summary
    print("\n\n=== Phase 11 Summary ===")
    results.sort(key=lambda x: x[2])
    for method, params, score in results:
        print(f"  {score:.4f}  {method} {params}")

    if results:
        print(f"\nBest: {results[0][2]:.4f} ({results[0][0]} {results[0][1]})")


if __name__ == "__main__":
    main()
