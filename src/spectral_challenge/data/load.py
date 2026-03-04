"""Data loading utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from spectral_challenge.config import Config
from spectral_challenge.paths import DATA_RAW


def load_train(cfg: Config, data_dir: Path | None = None) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    """Load training data and return (X, y, ids).

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)
    ids : pandas Index
    """
    data_dir = data_dir or DATA_RAW
    df = pd.read_csv(data_dir / cfg.train_file)
    ids = df[cfg.id_col]
    y = df[cfg.target_col].values.astype(np.float64)
    X = _extract_features(df, cfg)
    return X, y, ids


def load_test(cfg: Config, data_dir: Path | None = None) -> tuple[np.ndarray, pd.Index]:
    """Load test data and return (X, ids).

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
    ids : pandas Index
    """
    data_dir = data_dir or DATA_RAW
    df = pd.read_csv(data_dir / cfg.test_file)
    ids = df[cfg.id_col]
    X = _extract_features(df, cfg)
    return X, ids


def _extract_features(df: pd.DataFrame, cfg: Config) -> np.ndarray:
    """Extract feature columns from a DataFrame.

    If ``cfg.feature_prefix`` is set, only columns starting with that prefix
    are used.  Otherwise all numeric columns except id and target are used.
    """
    if cfg.feature_prefix:
        feat_cols = [c for c in df.columns if c.startswith(cfg.feature_prefix)]
    else:
        exclude = {cfg.id_col, cfg.target_col}
        feat_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    return df[feat_cols].values.astype(np.float64)
