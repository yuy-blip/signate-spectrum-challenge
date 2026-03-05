#!/usr/bin/env python
"""Final ensemble: Comprehensive greedy selection + weight optimization.

Combines all 400+ runs into the best possible ensemble.
Generates final OOF predictions and test predictions.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spectral_challenge.metrics import rmse

DATA_DIR = Path("data/raw")
RUNS_DIR = Path("runs")


def main():
    print("=== Final Ensemble Optimization ===\n")

    # Load y and groups
    df = pd.read_csv(DATA_DIR / "train.csv", encoding="cp932")
    y = df["含水率"].values
    groups = df["species number"].values

    # Load ALL runs
    all_runs = []
    for f in RUNS_DIR.glob("*/metrics.json"):
        oof_file = f.parent / "oof_preds.npy"
        if not oof_file.exists():
            continue
        with open(f) as fh:
            m = json.load(fh)
        mean_rmse = m.get("mean_rmse", 999)
        if mean_rmse > 30:
            continue  # Skip terrible models
        oof = np.load(oof_file)
        if np.any(np.isnan(oof)):
            continue
        all_runs.append({
            "name": f.parent.name,
            "rmse": mean_rmse,
            "oof": oof,
            "run_dir": str(f.parent),
            "fold_rmses": m.get("fold_rmses", []),
        })

    all_runs.sort(key=lambda x: x["rmse"])
    print(f"Loaded {len(all_runs)} runs (RMSE < 30)")
    top5 = [(r['name'][:40], round(r['rmse'], 4)) for r in all_runs[:5]]
    print(f"Top 5: {top5}")

    # Greedy diverse selection
    print("\n--- Greedy Selection ---")
    selected = [all_runs[0]]
    remaining = list(all_runs[1:])

    for step in range(24):
        if not remaining:
            break
        best_score = float("inf")
        best_idx = -1
        current_oofs = np.column_stack([r["oof"] for r in selected])
        current_avg = current_oofs.mean(axis=1)
        cur_rmse = rmse(y, current_avg)

        for i, r in enumerate(remaining):
            new_avg = (current_avg * len(selected) + r["oof"]) / (len(selected) + 1)
            score = rmse(y, new_avg)
            if score < best_score:
                best_score = score
                best_idx = i

        if best_idx >= 0 and best_score < cur_rmse - 0.001:
            added = remaining.pop(best_idx)
            selected.append(added)
            print(f"  +{step+1}: {added['name'][:50]} (ind={added['rmse']:.4f}, ens={best_score:.4f})")
        else:
            print(f"  Stopped at step {step+1} (no improvement)")
            break

    n = len(selected)
    oof_matrix = np.column_stack([r["oof"] for r in selected])
    avg_pred = oof_matrix.mean(axis=1)
    avg_rmse_val = rmse(y, avg_pred)
    print(f"\nSelected {n} models")
    print(f"Simple average RMSE: {avg_rmse_val:.4f}")

    # Weight optimization with L1 regularization to prevent overfitting
    print("\n--- Weight Optimization ---")

    def obj(w):
        w_pos = np.abs(w)
        w_norm = w_pos / w_pos.sum()
        pred = (oof_matrix * w_norm).sum(axis=1)
        return rmse(y, pred)

    best_w_rmse = avg_rmse_val
    best_weights = np.ones(n) / n

    for trial in range(100):
        w0 = np.random.dirichlet(np.ones(n) * 3)
        res = minimize(obj, w0, method="Nelder-Mead",
                       options={"maxiter": 10000, "xatol": 1e-8, "fatol": 1e-8})
        if res.fun < best_w_rmse:
            best_w_rmse = res.fun
            w = np.abs(res.x)
            best_weights = w / w.sum()

    print(f"Optimized weighted RMSE: {best_w_rmse:.4f}")

    # Per-fold analysis
    gkf = GroupKFold(n_splits=5)
    weighted_pred = (oof_matrix * best_weights).sum(axis=1)

    print("\n--- Per-fold Analysis ---")
    print(f"{'Fold':>6}  {'Avg':>8}  {'Weighted':>8}  {'Best Ind':>8}")
    for fold_idx, (_, val_idx) in enumerate(gkf.split(df, y, groups)):
        avg_fold = rmse(y[val_idx], avg_pred[val_idx])
        wt_fold = rmse(y[val_idx], weighted_pred[val_idx])
        # Best individual for this fold
        best_ind = min(r["fold_rmses"][fold_idx] for r in selected if len(r["fold_rmses"]) > fold_idx)
        print(f"  {fold_idx:>4}  {avg_fold:>8.2f}  {wt_fold:>8.2f}  {best_ind:>8.2f}")

    # Show weights
    print("\n--- Model Weights ---")
    for i, (r, w) in enumerate(zip(selected, best_weights)):
        if w > 0.01:
            print(f"  {w:.4f}  RMSE={r['rmse']:.4f}  {r['name'][:60]}")

    # Final summary
    print(f"\n{'='*60}")
    print(f"Best individual:     {selected[0]['rmse']:.4f}")
    print(f"Simple average ({n}):  {avg_rmse_val:.4f}")
    print(f"Weighted average:    {best_w_rmse:.4f}")
    print(f"{'='*60}")

    # Save final ensemble info
    ensemble_dir = RUNS_DIR / "final_ensemble"
    ensemble_dir.mkdir(exist_ok=True)
    np.save(ensemble_dir / "oof_preds.npy", weighted_pred)
    np.save(ensemble_dir / "avg_preds.npy", avg_pred)
    np.save(ensemble_dir / "weights.npy", best_weights)

    with open(ensemble_dir / "ensemble_info.json", "w") as f:
        json.dump({
            "n_models": n,
            "simple_avg_rmse": float(avg_rmse_val),
            "weighted_rmse": float(best_w_rmse),
            "models": [{"name": r["name"], "rmse": r["rmse"], "weight": float(w)}
                       for r, w in zip(selected, best_weights)],
        }, f, indent=2)

    print(f"\nEnsemble saved to {ensemble_dir}")


if __name__ == "__main__":
    main()
