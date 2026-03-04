#!/usr/bin/env python
"""Pseudo-labeling (semi-supervised learning).

1. Train model on labeled train data
2. Predict on test data (pseudo-labels)
3. Filter confident predictions (low uncertainty)
4. Retrain on train + pseudo-labeled test data
5. Predict again for final submission

Usage:
    python scripts/pseudo_labeling.py \
        --config configs/lgbm_pls_features.yaml \
        --confidence-threshold 0.8 \
        --iterations 2

    python scripts/pseudo_labeling.py \
        --base-run runs/lgbm_best_run \
        --config configs/lgbm_pls_features.yaml
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
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spectral_challenge.config import Config
from spectral_challenge.data.load import load_test, load_train
from spectral_challenge.metrics import rmse
from spectral_challenge.models.factory import create_model
from spectral_challenge.preprocess.pipeline import build_preprocess_pipeline


def pseudo_label_iteration(
    cfg: Config,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
    confidence_top_pct: float = 0.8,
):
    """One iteration of pseudo-labeling.

    Returns
    -------
    dict with:
        - pseudo_labels: predictions for test data
        - confident_mask: boolean mask of confident samples
        - oof_rmse: OOF score on original training data
        - fold_stds: std of predictions across folds (uncertainty proxy)
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    oof_preds = np.zeros(n_train)
    test_preds_all = np.zeros((n_folds, n_test))

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        Xtr, Xvl = X_train[train_idx], X_train[val_idx]
        ytr, yvl = y_train[train_idx], y_train[val_idx]

        pipe = build_preprocess_pipeline(cfg.preprocess)
        Xtr_t = pipe.fit_transform(Xtr)
        Xvl_t = pipe.transform(Xvl)
        Xte_t = pipe.transform(X_test)

        model = create_model(cfg.model_type, cfg.model_params)
        model.fit(Xtr_t, ytr)

        oof_preds[val_idx] = model.predict(Xvl_t).ravel()
        test_preds_all[fold_idx] = model.predict(Xte_t).ravel()

    # Test predictions: mean across folds
    pseudo_labels = test_preds_all.mean(axis=0)
    # Uncertainty: std across folds (lower = more confident)
    fold_stds = test_preds_all.std(axis=0)

    # Select confident samples: those with lowest std
    threshold = np.percentile(fold_stds, confidence_top_pct * 100)
    confident_mask = fold_stds <= threshold

    oof_score = rmse(y_train[:len(oof_preds)], oof_preds[:len(y_train)])

    return {
        "pseudo_labels": pseudo_labels,
        "confident_mask": confident_mask,
        "oof_rmse": oof_score,
        "fold_stds": fold_stds,
        "oof_preds": oof_preds,
        "test_preds_all": test_preds_all,
    }


def main():
    parser = argparse.ArgumentParser(description="Pseudo-labeling semi-supervised pipeline")
    parser.add_argument("--config", required=True, help="Config YAML")
    parser.add_argument("--iterations", type=int, default=2, help="Number of PL iterations")
    parser.add_argument("--confidence-threshold", type=float, default=0.8,
                        help="Top %% of confident samples to use (0.5=50%%, 0.8=80%%)")
    parser.add_argument("--pl-weight", type=float, default=1.0,
                        help="Weight for pseudo-labeled samples (0-1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data-dir", type=str, default="data/raw")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    data_dir = Path(args.data_dir)

    print("=== Pseudo-Labeling Pipeline ===")
    print(f"Config: {args.config}")
    print(f"Iterations: {args.iterations}")
    print(f"Confidence threshold: {args.confidence_threshold}")
    print()

    # Load data
    X_train_orig, y_train_orig, train_ids = load_train(cfg, data_dir)
    X_test, test_ids = load_test(cfg, data_dir)
    print(f"Train: {X_train_orig.shape}, Test: {X_test.shape}")

    X_train = X_train_orig.copy()
    y_train = y_train_orig.copy()

    for iteration in range(args.iterations):
        print(f"\n--- Iteration {iteration + 1}/{args.iterations} ---")
        print(f"Training data: {X_train.shape[0]} samples")

        result = pseudo_label_iteration(
            cfg, X_train, y_train, X_test,
            n_folds=cfg.n_folds, seed=args.seed + iteration,
            confidence_top_pct=args.confidence_threshold,
        )

        print(f"OOF RMSE: {result['oof_rmse']:.4f}")
        print(f"Confident test samples: {result['confident_mask'].sum()}/{len(result['confident_mask'])}")
        print(f"Pseudo-label std: mean={result['fold_stds'].mean():.4f}, "
              f"max={result['fold_stds'].max():.4f}")

        if iteration < args.iterations - 1:
            # Add confident pseudo-labeled samples to training data
            confident_X = X_test[result["confident_mask"]]
            confident_y = result["pseudo_labels"][result["confident_mask"]]

            # Combine: original train + confident pseudo-labels
            X_train = np.vstack([X_train_orig, confident_X])
            y_train = np.concatenate([y_train_orig, confident_y])

    # Final predictions
    final_preds = result["pseudo_labels"]

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / f"pseudo_label_{cfg.model_type}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    np.save(run_dir / "test_preds.npy", final_preds)
    np.save(run_dir / "oof_preds.npy", result["oof_preds"][:len(y_train_orig)])
    with open(run_dir / "metrics.json", "w") as f:
        json.dump({
            "mean_rmse": result["oof_rmse"],
            "fold_rmses": [],
            "iterations": args.iterations,
            "confidence_threshold": args.confidence_threshold,
        }, f, indent=2)

    # Submission
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
        out_path = Path("submissions") / f"pseudo_label_{timestamp}.csv"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame({id_col: df_test[id_col].values, "含水率": final_preds})
    submission.to_csv(out_path, index=False)
    print(f"\nSubmission saved to: {out_path}")
    print(f"Run artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
