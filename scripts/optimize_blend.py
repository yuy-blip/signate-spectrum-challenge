#!/usr/bin/env python
"""Optimize blending weights using scipy on OOF predictions.

Finds optimal weights for combining multiple model predictions
by minimizing RMSE on OOF data.

Usage:
    python scripts/optimize_blend.py \
        runs/lgbm_run1 runs/pls_run1 runs/ridge_run1

    python scripts/optimize_blend.py \
        runs/lgbm_run1 runs/pls_run1 runs/ridge_run1 \
        --output submissions/optimized_blend.csv
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spectral_challenge.metrics import rmse


def optimize_weights(oof_matrix: np.ndarray, y_true: np.ndarray, method: str = "nelder-mead"):
    """Find optimal blending weights via RMSE minimization."""
    n_models = oof_matrix.shape[1]

    def objective(weights):
        # Softmax to ensure positive weights summing to 1
        w = np.exp(weights) / np.exp(weights).sum()
        blended = oof_matrix @ w
        return rmse(y_true, blended)

    # Try multiple initializations
    best_result = None
    best_score = float("inf")

    # Init 1: equal weights
    inits = [np.zeros(n_models)]
    # Init 2: weight proportional to 1/RMSE
    individual_rmses = [rmse(y_true, oof_matrix[:, i]) for i in range(n_models)]
    inv_rmse = 1.0 / np.array(individual_rmses)
    inits.append(np.log(inv_rmse / inv_rmse.sum()))
    # Init 3: best model gets most weight
    best_idx = np.argmin(individual_rmses)
    w_init = np.full(n_models, -2.0)
    w_init[best_idx] = 2.0
    inits.append(w_init)

    for init in inits:
        result = minimize(objective, init, method=method, options={"maxiter": 10000, "xatol": 1e-8})
        if result.fun < best_score:
            best_score = result.fun
            best_result = result

    # Convert to proper weights
    final_weights = np.exp(best_result.x) / np.exp(best_result.x).sum()
    return final_weights, best_score


def main():
    parser = argparse.ArgumentParser(description="Optimize blending weights on OOF")
    parser.add_argument("run_dirs", nargs="+", help="Run directories containing oof_preds.npy")
    parser.add_argument("--output", type=str, default=None, help="Output CSV")
    parser.add_argument("--data-dir", type=str, default="data/raw")
    args = parser.parse_args()

    run_dirs = [Path(d) for d in args.run_dirs]
    n_models = len(run_dirs)

    print("=== Optimized Blending ===")
    print(f"Models: {n_models}")

    # Load OOF predictions
    oof_list = []
    test_list = []
    for rd in run_dirs:
        oof = np.load(rd / "oof_preds.npy")
        oof_list.append(oof)
        test_file = rd / "test_preds.npy"
        if test_file.exists():
            test_list.append(np.load(test_file))
        print(f"  {rd.name}: loaded")

    oof_matrix = np.column_stack(oof_list)
    test_matrix = np.column_stack(test_list) if test_list else None

    # Load y_true
    data_dir = Path(args.data_dir)
    train_file = data_dir / "train.csv"
    for enc in ["utf-8", "cp932", "shift_jis"]:
        try:
            df = pd.read_csv(train_file, encoding=enc)
            break
        except (UnicodeDecodeError, LookupError):
            continue

    target_col = None
    for candidate in ["含水率", "target", "y"]:
        if candidate in df.columns:
            target_col = candidate
            break
    y_true = df[target_col].values

    # Handle NaN
    valid = ~np.isnan(oof_matrix).any(axis=1)
    if not valid.all():
        print(f"WARNING: {(~valid).sum()} NaN samples removed")
        oof_matrix = oof_matrix[valid]
        y_true = y_true[valid]

    # Individual model scores
    print("\nIndividual model scores:")
    for i, rd in enumerate(run_dirs):
        score = rmse(y_true, oof_matrix[:, i])
        print(f"  {rd.name}: RMSE = {score:.4f}")

    # Equal average
    avg_rmse = rmse(y_true, oof_matrix.mean(axis=1))
    print(f"\nEqual average: RMSE = {avg_rmse:.4f}")

    # Optimize
    print("\nOptimizing weights...")
    opt_weights, opt_rmse = optimize_weights(oof_matrix, y_true)

    print(f"\nOptimized weights:")
    for i, rd in enumerate(run_dirs):
        print(f"  {rd.name}: {opt_weights[i]:.4f}")
    print(f"\nOptimized blend RMSE: {opt_rmse:.4f}")
    print(f"Improvement over equal avg: {avg_rmse - opt_rmse:.4f}")

    # Generate submission
    if test_matrix is not None and test_matrix.shape[1] == n_models:
        final_preds = test_matrix @ opt_weights

        test_file = data_dir / "test.csv"
        for enc in ["utf-8", "cp932", "shift_jis"]:
            try:
                df_test = pd.read_csv(test_file, encoding=enc)
                break
            except (UnicodeDecodeError, LookupError):
                continue

        id_col = None
        for candidate in ["sample number", "id", "ID"]:
            if candidate in df_test.columns:
                id_col = candidate
                break
        if id_col is None:
            id_col = df_test.columns[0]

        if args.output:
            out_path = Path(args.output)
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = Path("submissions") / f"optblend_{n_models}m_{timestamp}.csv"

        out_path.parent.mkdir(parents=True, exist_ok=True)
        submission = pd.DataFrame({id_col: df_test[id_col].values, "含水率": final_preds})
        submission.to_csv(out_path, index=False)
        print(f"\nSubmission saved to: {out_path}")
    else:
        print("\nWARNING: test predictions missing for some runs. Run predict first.")

    # Save metadata
    blend_dir = Path("runs") / f"optblend_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    blend_dir.mkdir(parents=True, exist_ok=True)
    with open(blend_dir / "metrics.json", "w") as f:
        json.dump({
            "mean_rmse": float(opt_rmse),
            "fold_rmses": [],
            "equal_avg_rmse": float(avg_rmse),
            "weights": opt_weights.tolist(),
            "runs": [str(rd) for rd in run_dirs],
        }, f, indent=2)
    print(f"Blend metadata saved to: {blend_dir}")


if __name__ == "__main__":
    main()
