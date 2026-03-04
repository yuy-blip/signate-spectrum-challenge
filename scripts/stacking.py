#!/usr/bin/env python
"""2-Level Stacking Ensemble.

Level-1: Multiple diverse base models produce OOF predictions
Level-2: Meta-learner (Ridge/LightGBM) learns to combine them

Usage:
    python scripts/stacking.py \
        --runs runs/lgbm_run1 runs/pls_run1 runs/ridge_run1 \
        --meta ridge \
        --output submissions/stacking_submission.csv

    python scripts/stacking.py \
        --runs runs/lgbm_run1 runs/pls_run1 runs/ridge_run1 \
        --meta lgbm \
        --n-folds 5
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spectral_challenge.metrics import rmse


def load_oof_and_test_preds(run_dirs: list[Path]):
    """Load OOF predictions and test predictions from run directories."""
    oof_list = []
    test_list = []

    for rd in run_dirs:
        oof_file = rd / "oof_preds.npy"
        test_file = rd / "test_preds.npy"

        if not oof_file.exists():
            raise FileNotFoundError(f"No oof_preds.npy in {rd}")

        oof = np.load(oof_file)
        oof_list.append(oof)

        if test_file.exists():
            test_preds = np.load(test_file)
            test_list.append(test_preds)

    oof_matrix = np.column_stack(oof_list)  # (n_train, n_models)
    test_matrix = np.column_stack(test_list) if test_list else None

    return oof_matrix, test_matrix


def train_meta_learner(
    oof_matrix: np.ndarray,
    y_true: np.ndarray,
    test_matrix: np.ndarray | None,
    meta_type: str = "ridge",
    n_folds: int = 5,
    seed: int = 42,
    groups: np.ndarray | None = None,
):
    """Train a level-2 meta-learner on OOF predictions.

    Uses GroupKFold (if groups provided) to avoid species leakage.
    """
    n_samples = oof_matrix.shape[0]

    # Meta-model CV — use GroupKFold if groups available
    if groups is not None:
        kf = GroupKFold(n_splits=n_folds)
        split_iter = kf.split(oof_matrix, y_true, groups)
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        split_iter = kf.split(oof_matrix)
    meta_oof = np.zeros(n_samples)
    meta_test_preds = []
    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(split_iter):
        X_train = oof_matrix[train_idx]
        y_train = y_true[train_idx]
        X_val = oof_matrix[val_idx]
        y_val = y_true[val_idx]

        if meta_type == "ridge":
            meta = Ridge(alpha=1.0)
        elif meta_type == "lgbm":
            from lightgbm import LGBMRegressor
            meta = LGBMRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=3,
                num_leaves=7, verbose=-1,
            )
        elif meta_type == "linear":
            from sklearn.linear_model import LinearRegression
            meta = LinearRegression(positive=True)  # positive weights = blend
        else:
            meta = Ridge(alpha=1.0)

        meta.fit(X_train, y_train)
        val_pred = meta.predict(X_val).ravel()
        meta_oof[val_idx] = val_pred

        fold_score = rmse(y_val, val_pred)
        fold_scores.append(fold_score)
        print(f"  Meta fold {fold_idx}: RMSE = {fold_score:.4f}")

        if test_matrix is not None:
            meta_test_preds.append(meta.predict(test_matrix).ravel())

    overall_rmse = rmse(y_true, meta_oof)
    print(f"  Meta overall OOF RMSE: {overall_rmse:.4f}")

    # Average test predictions across meta folds
    if meta_test_preds:
        final_test = np.mean(meta_test_preds, axis=0)
    else:
        final_test = None

    # Also fit final meta-model on all OOF data
    if meta_type == "ridge":
        final_meta = Ridge(alpha=1.0)
    elif meta_type == "lgbm":
        from lightgbm import LGBMRegressor
        final_meta = LGBMRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=3,
            num_leaves=7, verbose=-1,
        )
    elif meta_type == "linear":
        from sklearn.linear_model import LinearRegression
        final_meta = LinearRegression(positive=True)
    else:
        final_meta = Ridge(alpha=1.0)

    final_meta.fit(oof_matrix, y_true)

    if hasattr(final_meta, "coef_"):
        print(f"  Meta weights: {final_meta.coef_}")
        if hasattr(final_meta, "intercept_"):
            print(f"  Meta intercept: {final_meta.intercept_:.4f}")

    return {
        "meta_oof_rmse": overall_rmse,
        "fold_scores": fold_scores,
        "meta_oof": meta_oof,
        "test_preds": final_test,
        "meta_model": final_meta,
    }


def main():
    parser = argparse.ArgumentParser(description="2-Level Stacking Ensemble")
    parser.add_argument("--runs", nargs="+", required=True, help="Run directories for level-1 models")
    parser.add_argument("--meta", default="ridge", choices=["ridge", "lgbm", "linear"],
                        help="Meta-learner type")
    parser.add_argument("--n-folds", type=int, default=5, help="Meta CV folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output submission CSV")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Data directory")
    parser.add_argument("--y-file", type=str, default=None,
                        help="Path to y_true (auto-loaded from train.csv if not specified)")
    args = parser.parse_args()

    run_dirs = [Path(d) for d in args.runs]

    print(f"=== Stacking Ensemble ===")
    print(f"Level-1 models: {len(run_dirs)}")
    for rd in run_dirs:
        print(f"  - {rd.name}")
    print(f"Meta-learner: {args.meta}")
    print()

    # Load OOF and test predictions
    print("Loading predictions...")
    oof_matrix, test_matrix = load_oof_and_test_preds(run_dirs)
    print(f"OOF matrix: {oof_matrix.shape}")
    if test_matrix is not None:
        print(f"Test matrix: {test_matrix.shape}")

    # Load true y
    data_dir = Path(args.data_dir)
    if args.y_file:
        y_true = np.load(args.y_file)
    else:
        # Load from train.csv
        train_file = data_dir / "train.csv"
        for enc in ["utf-8", "cp932", "shift_jis"]:
            try:
                df = pd.read_csv(train_file, encoding=enc)
                break
            except (UnicodeDecodeError, LookupError):
                continue

        # Find target column
        target_col = None
        for candidate in ["含水率", "target", "y", "moisture"]:
            if candidate in df.columns:
                target_col = candidate
                break
        if target_col is None:
            raise ValueError(f"Cannot find target column in {list(df.columns)}")

        y_true = df[target_col].values

    print(f"y_true: shape={y_true.shape}, mean={y_true.mean():.2f}, std={y_true.std():.2f}")

    # Check for NaN in OOF
    nan_mask = np.isnan(oof_matrix).any(axis=1)
    if nan_mask.any():
        print(f"WARNING: {nan_mask.sum()} samples with NaN in OOF, excluding from meta-training")
        valid = ~nan_mask
        oof_matrix = oof_matrix[valid]
        y_true = y_true[valid]

    # Print individual model scores
    print("\nLevel-1 model scores:")
    for i, rd in enumerate(run_dirs):
        score = rmse(y_true, oof_matrix[:, i])
        print(f"  {rd.name}: OOF RMSE = {score:.4f}")

    # Simple average baseline
    avg_pred = oof_matrix.mean(axis=1)
    avg_rmse = rmse(y_true, avg_pred)
    print(f"\nSimple average: OOF RMSE = {avg_rmse:.4f}")

    # Train meta-learner
    print(f"\nTraining {args.meta} meta-learner ({args.n_folds}-fold)...")
    result = train_meta_learner(
        oof_matrix, y_true, test_matrix,
        meta_type=args.meta, n_folds=args.n_folds, seed=args.seed,
    )

    print(f"\n{'='*50}")
    print(f"Individual best: {min(rmse(y_true, oof_matrix[:, i]) for i in range(oof_matrix.shape[1])):.4f}")
    print(f"Simple average:  {avg_rmse:.4f}")
    print(f"Stacking ({args.meta}):  {result['meta_oof_rmse']:.4f}")
    print(f"{'='*50}")

    # Generate submission
    if result["test_preds"] is not None:
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

        ids = df_test[id_col].values
        submission = pd.DataFrame({id_col: ids, "含水率": result["test_preds"]})

        if args.output:
            out_path = Path(args.output)
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = Path("submissions") / f"stacking_{args.meta}_{timestamp}.csv"

        out_path.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(out_path, index=False)
        print(f"\nSubmission saved to: {out_path}")

    # Save stacking result
    stacking_dir = Path("runs") / f"stacking_{args.meta}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    stacking_dir.mkdir(parents=True, exist_ok=True)
    np.save(stacking_dir / "meta_oof.npy", result["meta_oof"])
    if result["test_preds"] is not None:
        np.save(stacking_dir / "test_preds.npy", result["test_preds"])
    joblib.dump(result["meta_model"], stacking_dir / "meta_model.joblib")
    with open(stacking_dir / "metrics.json", "w") as f:
        json.dump({
            "mean_rmse": result["meta_oof_rmse"],
            "fold_rmses": result["fold_scores"],
            "level1_runs": [str(rd) for rd in run_dirs],
            "meta_type": args.meta,
        }, f, indent=2)
    print(f"Stacking artifacts saved to: {stacking_dir}")


if __name__ == "__main__":
    main()
