"""Prediction using trained fold models.

Loads all fold models + pipelines from a run directory, applies each to
the input data, and returns the averaged prediction.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np

logger = logging.getLogger("spectral_challenge")


def predict_test(X_test: np.ndarray, run_dir: Path) -> np.ndarray:
    """Generate averaged predictions from all fold models in *run_dir*.

    Parameters
    ----------
    X_test : ndarray of shape (n_samples, n_features)
        Raw (un-preprocessed) test features.
    run_dir : Path
        Directory containing ``models/model_fold*.joblib`` and
        ``models/pipe_fold*.joblib``.

    Returns
    -------
    ndarray of shape (n_samples,)
        Averaged predictions across folds.
    """
    models_dir = run_dir / "models"
    model_files = sorted(models_dir.glob("model_fold*.joblib"))
    pipe_files = sorted(models_dir.glob("pipe_fold*.joblib"))

    if not model_files:
        raise FileNotFoundError(f"No fold models found in {models_dir}")
    if len(model_files) != len(pipe_files):
        raise RuntimeError("Mismatch between number of model and pipeline files")

    preds = np.zeros(X_test.shape[0], dtype=np.float64)
    n_folds = len(model_files)

    for mf, pf in zip(model_files, pipe_files):
        pipe = joblib.load(pf)
        model = joblib.load(mf)
        X_t = pipe.transform(X_test)
        preds += model.predict(X_t).ravel()
        logger.info("Predicted with %s", mf.name)

    preds /= n_folds
    return preds
