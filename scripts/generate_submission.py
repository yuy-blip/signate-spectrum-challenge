#!/usr/bin/env python
"""Generate final submission using best ensemble of all available runs.

Loads all runs, performs greedy model selection + weight optimization,
then averages fold-level predictions on test data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spectral_challenge.metrics import rmse

DATA_DIR = Path("data/raw")
RUNS_DIR = Path("runs")
SUBMISSIONS_DIR = Path("submissions")


def load_all_runs():
    """Load all valid runs with OOF predictions."""
    all_runs = []
    for f in RUNS_DIR.glob("*/metrics.json"):
        oof_file = f.parent / "oof_preds.npy"
        if not oof_file.exists():
            continue
        with open(f) as fh:
            m = json.load(fh)
        mean_rmse = m.get("mean_rmse", 999)
        if mean_rmse > 30:
            continue
        oof = np.load(oof_file)
        if np.any(np.isnan(oof)):
            continue
        # Check if fold models exist (for test prediction)
        models_dir = f.parent / "models"
        has_models = all(
            (models_dir / f"model_fold{i}.joblib").exists() and
            (models_dir / f"pipe_fold{i}.joblib").exists()
            for i in range(5)
        )
        all_runs.append({
            "name": f.parent.name,
            "rmse": mean_rmse,
            "oof": oof,
            "run_dir": f.parent,
            "fold_rmses": m.get("fold_rmses", []),
            "has_models": has_models,
        })
    all_runs.sort(key=lambda x: x["rmse"])
    return all_runs


def greedy_select(all_runs, y, max_models=25):
    """Greedy diverse selection."""
    selected = [all_runs[0]]
    remaining = list(all_runs[1:])

    for step in range(max_models - 1):
        if not remaining:
            break
        current_oofs = np.column_stack([r["oof"] for r in selected])
        current_avg = current_oofs.mean(axis=1)
        cur_rmse = rmse(y, current_avg)
        best_score = float("inf")
        best_idx = -1

        for i, r in enumerate(remaining):
            new_avg = (current_avg * len(selected) + r["oof"]) / (len(selected) + 1)
            score = rmse(y, new_avg)
            if score < best_score:
                best_score = score
                best_idx = i

        if best_idx >= 0 and best_score < cur_rmse - 0.001:
            added = remaining.pop(best_idx)
            selected.append(added)
            print(f"  +{step+1}: {added['name'][:50]:50s} ind={added['rmse']:.4f} ens={best_score:.4f}")
        else:
            break

    return selected


def optimize_weights(selected, y, n_trials=200):
    """Optimize ensemble weights."""
    n = len(selected)
    oof_matrix = np.column_stack([r["oof"] for r in selected])
    avg_pred = oof_matrix.mean(axis=1)
    avg_rmse_val = rmse(y, avg_pred)

    def obj(w):
        w_pos = np.abs(w)
        w_norm = w_pos / w_pos.sum()
        pred = (oof_matrix * w_norm).sum(axis=1)
        return rmse(y, pred)

    best_w_rmse = avg_rmse_val
    best_weights = np.ones(n) / n

    for trial in range(n_trials):
        w0 = np.random.dirichlet(np.ones(n) * 3)
        res = minimize(obj, w0, method="Nelder-Mead",
                       options={"maxiter": 10000, "xatol": 1e-8, "fatol": 1e-8})
        if res.fun < best_w_rmse:
            best_w_rmse = res.fun
            w = np.abs(res.x)
            best_weights = w / w.sum()

    return best_weights, best_w_rmse, avg_rmse_val


def predict_test_ensemble(selected, weights, n_folds=5):
    """Generate test predictions using ensemble of fold models."""
    from spectral_challenge.config import Config
    from spectral_challenge.data.load import load_test

    cfg = Config(
        train_file="train.csv", test_file="test.csv",
        id_col="sample number", target_col="含水率",
        group_col="species number",
    )
    X_test, ids = load_test(cfg, DATA_DIR)
    print(f"Test data: {X_test.shape[0]} samples, {X_test.shape[1]} features")

    # For each selected model, predict using all fold models and average
    all_preds = []
    for i, (run, weight) in enumerate(zip(selected, weights)):
        if weight < 0.001:
            continue  # Skip negligible weights

        run_dir = run["run_dir"]
        models_dir = run_dir / "models"

        if not run["has_models"]:
            print(f"  SKIP {run['name'][:40]} (no fold models)")
            continue

        fold_preds = []
        for fold_idx in range(n_folds):
            pipe = joblib.load(models_dir / f"pipe_fold{fold_idx}.joblib")
            model = joblib.load(models_dir / f"model_fold{fold_idx}.joblib")

            X_test_t = pipe.transform(X_test)
            pred = model.predict(X_test_t).ravel()
            fold_preds.append(pred)

        # Average across folds for this model
        model_pred = np.mean(fold_preds, axis=0)
        all_preds.append((weight, model_pred))
        print(f"  Model {i+1}: w={weight:.4f} {run['name'][:40]} pred_range=[{model_pred.min():.1f}, {model_pred.max():.1f}]")

    # Weighted average
    total_weight = sum(w for w, _ in all_preds)
    final_pred = sum(w * p for w, p in all_preds) / total_weight

    return ids, final_pred


def main():
    print("=== Final Submission Generation ===\n")

    # Load y and groups
    df = pd.read_csv(DATA_DIR / "train.csv", encoding="cp932")
    y = df["含水率"].values
    groups = df["species number"].values

    # Load all runs
    all_runs = load_all_runs()
    print(f"Loaded {len(all_runs)} runs (RMSE < 30)")
    top5_info = [(r["name"][:40], round(r["rmse"], 4)) for r in all_runs[:5]]
    print(f"Top 5: {top5_info}")

    # Greedy selection
    print("\n--- Greedy Selection ---")
    selected = greedy_select(all_runs, y, max_models=25)
    n = len(selected)

    # Optimize weights
    print(f"\n--- Weight Optimization ({n} models) ---")
    weights, weighted_rmse, avg_rmse = optimize_weights(selected, y)

    # Show results
    print(f"\nSimple average RMSE: {avg_rmse:.4f}")
    print(f"Weighted RMSE: {weighted_rmse:.4f}")
    print(f"\nModel weights:")
    for r, w in sorted(zip(selected, weights), key=lambda x: -x[1]):
        if w > 0.01:
            print(f"  {w:.4f}  RMSE={r['rmse']:.4f}  {r['name'][:60]}")

    # Generate test predictions
    print("\n--- Generating Test Predictions ---")

    # Use both weighted and simple average
    ids, pred_weighted = predict_test_ensemble(selected, weights)

    # Simple average version
    _, pred_avg = predict_test_ensemble(selected, np.ones(n) / n)

    # Save submissions
    SUBMISSIONS_DIR.mkdir(exist_ok=True)

    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Weighted ensemble
    sub_weighted = pd.DataFrame({"sample number": ids.values, "含水率": pred_weighted})
    path_w = SUBMISSIONS_DIR / f"submission_weighted_{ts}.csv"
    sub_weighted.to_csv(path_w, index=False)
    print(f"\nWeighted submission: {path_w}")

    # Simple average ensemble
    sub_avg = pd.DataFrame({"sample number": ids.values, "含水率": pred_avg})
    path_a = SUBMISSIONS_DIR / f"submission_avg_{ts}.csv"
    sub_avg.to_csv(path_a, index=False)
    print(f"Average submission: {path_a}")

    # Prediction stats
    print(f"\nWeighted pred range: [{pred_weighted.min():.1f}, {pred_weighted.max():.1f}], mean={pred_weighted.mean():.1f}")
    print(f"Average pred range: [{pred_avg.min():.1f}, {pred_avg.max():.1f}], mean={pred_avg.mean():.1f}")

    # Save ensemble info
    ensemble_dir = RUNS_DIR / "final_submission_ensemble"
    ensemble_dir.mkdir(exist_ok=True)
    with open(ensemble_dir / "ensemble_info.json", "w") as f:
        json.dump({
            "n_models": n,
            "simple_avg_rmse": float(avg_rmse),
            "weighted_rmse": float(weighted_rmse),
            "models": [{"name": r["name"], "rmse": r["rmse"], "weight": float(w)}
                       for r, w in zip(selected, weights)],
        }, f, indent=2)


if __name__ == "__main__":
    main()
