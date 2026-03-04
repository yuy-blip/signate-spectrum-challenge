"""CV training loop.

Responsibilities
- Run K-fold cross-validation
- Save OOF predictions, fold-level RMSE, models, and the pipeline
- Copy config + git hash into the run directory
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

import joblib
import numpy as np
import yaml

from spectral_challenge.config import Config
from spectral_challenge.data.split import cv_splits
from spectral_challenge.metrics import rmse
from spectral_challenge.models.factory import create_model
from spectral_challenge.preprocess.pipeline import build_preprocess_pipeline

logger = logging.getLogger("spectral_challenge")


def run_cv(
    cfg: Config,
    X: np.ndarray,
    y: np.ndarray,
    run_dir: Path,
    groups: np.ndarray | None = None,
) -> dict:
    """Execute cross-validation and persist artefacts.

    Returns
    -------
    dict with keys: oof_preds, fold_rmses, mean_rmse
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config snapshot
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(cfg.to_dict(), f, default_flow_style=False)

    # Save git hash if available
    _save_git_info(run_dir)

    n_samples = X.shape[0]
    oof_preds = np.full(n_samples, np.nan)
    fold_rmses: list[float] = []
    models_dir = run_dir / "models"
    models_dir.mkdir(exist_ok=True)

    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits(cfg, X, y, groups)):
        logger.info("=== Fold %d ===", fold_idx)

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Build & fit preprocessing pipeline per fold
        pipe = build_preprocess_pipeline(cfg.preprocess)
        X_train_t = pipe.fit_transform(X_train)
        X_val_t = pipe.transform(X_val)

        # Build & fit model
        model = create_model(cfg.model_type, cfg.model_params)
        model.fit(X_train_t, y_train)

        # Predict
        val_pred = model.predict(X_val_t).ravel()
        oof_preds[val_idx] = val_pred

        fold_score = rmse(y_val, val_pred)
        fold_rmses.append(fold_score)
        logger.info("Fold %d RMSE: %.6f", fold_idx, fold_score)

        # Save fold artefacts
        joblib.dump(model, models_dir / f"model_fold{fold_idx}.joblib")
        joblib.dump(pipe, models_dir / f"pipe_fold{fold_idx}.joblib")

    mean_score = rmse(y, oof_preds)
    logger.info("Overall OOF RMSE: %.6f", mean_score)

    # Persist OOF & metrics
    np.save(run_dir / "oof_preds.npy", oof_preds)
    metrics = {"fold_rmses": fold_rmses, "mean_rmse": mean_score}
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return {"oof_preds": oof_preds, "fold_rmses": fold_rmses, "mean_rmse": mean_score}


def _save_git_info(run_dir: Path) -> None:
    """Best-effort save of the current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        (run_dir / "git_hash.txt").write_text(result.stdout.strip())
    except Exception:
        pass
