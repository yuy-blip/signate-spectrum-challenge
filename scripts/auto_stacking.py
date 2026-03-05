#!/usr/bin/env python
"""Auto-stacking: Find the best N runs and build a stacking ensemble.

Scans all run directories for metrics.json, ranks by RMSE,
picks top-K diverse models, and stacks them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spectral_challenge.metrics import rmse


def find_all_runs(runs_dir: Path) -> list[dict]:
    """Find all completed runs with metrics and OOF predictions."""
    results = []
    for metrics_file in runs_dir.glob("*/metrics.json"):
        run_dir = metrics_file.parent
        oof_file = run_dir / "oof_preds.npy"
        if not oof_file.exists():
            continue
        with open(metrics_file) as f:
            metrics = json.load(f)
        results.append({
            "name": run_dir.name,
            "run_dir": str(run_dir),
            "mean_rmse": metrics.get("mean_rmse", 999),
            "fold_rmses": metrics.get("fold_rmses", []),
        })
    results.sort(key=lambda x: x["mean_rmse"])
    return results


def load_y_and_groups(data_dir: Path):
    """Load true y and species groups."""
    for enc in ["utf-8", "cp932", "shift_jis"]:
        try:
            df = pd.read_csv(data_dir / "train.csv", encoding=enc)
            break
        except (UnicodeDecodeError, LookupError):
            continue

    target_col = None
    for candidate in ["含水率", "target", "y"]:
        if candidate in df.columns:
            target_col = candidate
            break
    y = df[target_col].values
    groups = df["species number"].values
    return y, groups


def greedy_select_diverse(runs: list[dict], y: np.ndarray, max_models: int = 10,
                          min_rmse: float = 50.0) -> list[dict]:
    """Greedily select diverse models for stacking.

    Start with the best model, then iteratively add the model that
    most reduces the simple-average RMSE.
    """
    # Filter to models below threshold
    candidates = [r for r in runs if r["mean_rmse"] < min_rmse]
    if not candidates:
        candidates = runs[:20]

    # Load OOF predictions
    for r in candidates:
        r["oof"] = np.load(Path(r["run_dir"]) / "oof_preds.npy")

    selected = [candidates[0]]
    remaining = candidates[1:]

    while len(selected) < max_models and remaining:
        best_score = float("inf")
        best_idx = -1

        current_oof = np.column_stack([r["oof"] for r in selected])
        current_avg = current_oof.mean(axis=1)

        for i, r in enumerate(remaining):
            # Try adding this model
            new_avg = (current_avg * len(selected) + r["oof"]) / (len(selected) + 1)
            score = rmse(y, new_avg)
            if score < best_score:
                best_score = score
                best_idx = i

        if best_idx >= 0:
            selected.append(remaining.pop(best_idx))
            print(f"  Added model {len(selected)}: {selected[-1]['name']} "
                  f"(individual RMSE={selected[-1]['mean_rmse']:.4f}, "
                  f"avg RMSE={best_score:.4f})")
        else:
            break

    return selected


def stack_with_cv(oof_matrix: np.ndarray, y: np.ndarray, groups: np.ndarray,
                  meta_type: str = "ridge", n_folds: int = 5) -> dict:
    """Stack with GroupKFold CV."""
    gkf = GroupKFold(n_splits=n_folds)
    meta_oof = np.zeros(len(y))
    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(oof_matrix, y, groups)):
        if meta_type == "ridge":
            meta = Ridge(alpha=1.0)
        elif meta_type == "enet":
            meta = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)
        else:
            meta = Ridge(alpha=1.0)

        meta.fit(oof_matrix[train_idx], y[train_idx])
        pred = meta.predict(oof_matrix[val_idx]).ravel()
        meta_oof[val_idx] = pred
        fold_scores.append(rmse(y[val_idx], pred))

    overall = rmse(y, meta_oof)
    return {"rmse": overall, "fold_scores": fold_scores, "oof": meta_oof}


def main():
    runs_dir = Path("runs")
    data_dir = Path("data/raw")

    print("=== Auto Stacking ===\n")

    # Find all runs
    all_runs = find_all_runs(runs_dir)
    print(f"Found {len(all_runs)} completed runs")
    print(f"Top 10:")
    for i, r in enumerate(all_runs[:10]):
        print(f"  {i+1}. RMSE={r['mean_rmse']:.4f}  {r['name']}")

    y, groups = load_y_and_groups(data_dir)

    # Greedy diverse selection
    print(f"\n--- Greedy Selection ---")
    selected = greedy_select_diverse(all_runs, y, max_models=15, min_rmse=30.0)

    print(f"\nSelected {len(selected)} models")

    # Stack
    oof_matrix = np.column_stack([r["oof"] for r in selected])
    print(f"\nOOF matrix: {oof_matrix.shape}")

    # Simple average
    avg = oof_matrix.mean(axis=1)
    avg_rmse = rmse(y, avg)
    print(f"Simple average RMSE: {avg_rmse:.4f}")

    # Weighted average (inverse RMSE)
    weights = np.array([1.0 / r["mean_rmse"] for r in selected])
    weights /= weights.sum()
    weighted = (oof_matrix * weights).sum(axis=1)
    weighted_rmse = rmse(y, weighted)
    print(f"Inverse-RMSE weighted average: {weighted_rmse:.4f}")

    # Ridge stacking with GKF
    print(f"\n--- Ridge Stacking (GKF) ---")
    result = stack_with_cv(oof_matrix, y, groups, "ridge")
    print(f"Ridge stacking RMSE: {result['rmse']:.4f}")
    print(f"Fold RMSEs: {[f'{s:.2f}' for s in result['fold_scores']]}")

    # Try different subsets
    print(f"\n--- Subset Analysis ---")
    for k in range(2, min(len(selected)+1, 12)):
        sub_oof = oof_matrix[:, :k]
        sub_avg = sub_oof.mean(axis=1)
        sub_rmse_val = rmse(y, sub_avg)
        sub_stack = stack_with_cv(sub_oof, y, groups, "ridge")
        print(f"  Top-{k:2d}: avg={sub_rmse_val:.4f}, stack={sub_stack['rmse']:.4f}")

    # Best individual
    best_individual = min(r["mean_rmse"] for r in selected)
    print(f"\n{'='*50}")
    print(f"Best individual:  {best_individual:.4f}")
    print(f"Simple average:   {avg_rmse:.4f}")
    print(f"Weighted average: {weighted_rmse:.4f}")
    print(f"Ridge stacking:   {result['rmse']:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
