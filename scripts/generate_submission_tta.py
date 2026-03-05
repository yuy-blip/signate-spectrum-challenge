#!/usr/bin/env python
"""Generate submission with Test-Time Augmentation (TTA).

For each test sample, create multiple augmented versions by:
1. Adding small Gaussian noise
2. Slight spectral shifting
Then average all predictions.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spectral_challenge.config import Config
from spectral_challenge.data.load import load_test
from spectral_challenge.metrics import rmse

DATA_DIR = Path("data/raw")
RUNS_DIR = Path("runs")
SUBMISSIONS_DIR = Path("submissions")


def load_best_runs(n_top=20):
    """Load top runs sorted by RMSE."""
    all_runs = []
    for f in RUNS_DIR.glob("*/metrics.json"):
        oof_file = f.parent / "oof_preds.npy"
        if not oof_file.exists():
            continue
        with open(f) as fh:
            m = json.load(fh)
        mean_rmse = m.get("mean_rmse", 999)
        if mean_rmse > 25:
            continue
        models_dir = f.parent / "models"
        has_models = all(
            (models_dir / f"model_fold{i}.joblib").exists() and
            (models_dir / f"pipe_fold{i}.joblib").exists()
            for i in range(5)
        )
        if not has_models:
            continue
        all_runs.append({
            "name": f.parent.name,
            "rmse": mean_rmse,
            "oof": np.load(oof_file),
            "run_dir": f.parent,
        })
    all_runs.sort(key=lambda x: x["rmse"])
    return all_runs[:n_top]


def predict_with_tta(run_dir, X_test, n_aug=10, noise_std=0.001, seed=42):
    """Predict with TTA: original + augmented versions."""
    rng = np.random.RandomState(seed)
    models_dir = run_dir / "models"

    all_preds = []

    for fold_idx in range(5):
        pipe = joblib.load(models_dir / f"pipe_fold{fold_idx}.joblib")
        model = joblib.load(models_dir / f"model_fold{fold_idx}.joblib")

        # Original prediction
        X_t = pipe.transform(X_test)
        pred_orig = model.predict(X_t).ravel()
        all_preds.append(pred_orig)

        # Augmented predictions
        for aug in range(n_aug):
            # Add Gaussian noise to raw spectra
            noise = rng.normal(0, noise_std * np.std(X_test, axis=0, keepdims=True),
                               size=X_test.shape)
            X_aug = X_test + noise

            X_aug_t = pipe.transform(X_aug)
            pred_aug = model.predict(X_aug_t).ravel()
            all_preds.append(pred_aug)

    return np.mean(all_preds, axis=0)


def main():
    print("=== Submission with TTA ===\n")

    cfg = Config(
        train_file="train.csv", test_file="test.csv",
        id_col="sample number", target_col="含水率",
        group_col="species number",
    )
    X_test, ids = load_test(cfg, DATA_DIR)
    print(f"Test data: {X_test.shape}")

    # Load best runs
    runs = load_best_runs(n_top=10)
    print(f"Loaded {len(runs)} best runs")
    for r in runs[:5]:
        print(f"  {r['rmse']:.4f}  {r['name'][:50]}")

    # Predict with TTA for top 5 models
    ensemble_preds = []
    for i, run in enumerate(runs[:5]):
        print(f"\n[{i+1}/5] Predicting: {run['name'][:50]}")
        pred = predict_with_tta(run["run_dir"], X_test, n_aug=5, noise_std=0.001)
        ensemble_preds.append(pred)
        print(f"  pred range: [{pred.min():.1f}, {pred.max():.1f}]")

    # Simple average
    final_pred = np.mean(ensemble_preds, axis=0)
    print(f"\nFinal pred range: [{final_pred.min():.1f}, {final_pred.max():.1f}], mean={final_pred.mean():.1f}")

    # Save
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sub = pd.DataFrame({"sample number": ids.values, "含水率": final_pred})
    path = SUBMISSIONS_DIR / f"submission_tta_{ts}.csv"
    sub.to_csv(path, index=False)
    print(f"\nTTA submission: {path}")

    # Also no-TTA version with same models
    no_tta_preds = []
    for run in runs[:5]:
        fold_preds = []
        for fold_idx in range(5):
            pipe = joblib.load(run["run_dir"] / "models" / f"pipe_fold{fold_idx}.joblib")
            model = joblib.load(run["run_dir"] / "models" / f"model_fold{fold_idx}.joblib")
            X_t = pipe.transform(X_test)
            fold_preds.append(model.predict(X_t).ravel())
        no_tta_preds.append(np.mean(fold_preds, axis=0))
    no_tta_final = np.mean(no_tta_preds, axis=0)
    sub2 = pd.DataFrame({"sample number": ids.values, "含水率": no_tta_final})
    path2 = SUBMISSIONS_DIR / f"submission_top5avg_{ts}.csv"
    sub2.to_csv(path2, index=False)
    print(f"Top-5 average submission: {path2}")

    # Difference between TTA and no-TTA
    diff = np.abs(final_pred - no_tta_final)
    print(f"\nTTA vs no-TTA: mean diff={diff.mean():.4f}, max diff={diff.max():.4f}")


if __name__ == "__main__":
    main()
