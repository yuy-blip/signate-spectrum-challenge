"""Data loading utilities with robust column auto-detection."""

from __future__ import annotations

import logging
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from spectral_challenge.config import Config
from spectral_challenge.paths import DATA_RAW

logger = logging.getLogger("spectral_challenge")

_ENCODING_CANDIDATES = ["utf-8", "cp932", "shift_jis", "euc-jp", "latin-1"]

# Candidate names tried (in order) when the configured column is missing.
_ID_CANDIDATES = ["sample number", "id", "sample_number", "sample_id", "index"]
_TARGET_CANDIDATES = ["含水率", "target", "y", "moisture", "moisture_content"]


# ---------------------------------------------------------------------------
# Column name normalisation
# ---------------------------------------------------------------------------

def _normalize(name: str) -> str:
    """Normalise a column name for fuzzy matching.

    * NFKC unicode normalisation (full-width → half-width)
    * strip whitespace
    * lower-case
    """
    return unicodedata.normalize("NFKC", name).strip().lower()


def _build_lookup(columns: pd.Index) -> dict[str, str]:
    """Map normalised column names → original column names."""
    lookup: dict[str, str] = {}
    for col in columns:
        key = _normalize(str(col))
        # first occurrence wins (shouldn't have duplicates in practice)
        if key not in lookup:
            lookup[key] = col
    return lookup


# ---------------------------------------------------------------------------
# Auto-detection helpers
# ---------------------------------------------------------------------------

def _find_column(
    columns: pd.Index,
    configured: str,
    candidates: list[str],
    role: str,
) -> str:
    """Return the actual column name to use for *role* (e.g. 'id' / 'target').

    Strategy:
    1. If *configured* exists verbatim in *columns*, use it.
    2. Try normalised matching against *configured*.
    3. Walk through *candidates* with normalised matching.
    4. For the 'id' role only: fall back to ``columns[0]``.
    5. Raise ``KeyError`` if nothing found.
    """
    lookup = _build_lookup(columns)

    # 1. Exact match
    if configured in columns:
        return configured

    # 2. Normalised match for the configured name
    norm_cfg = _normalize(configured)
    if norm_cfg in lookup:
        detected = lookup[norm_cfg]
        logger.info("Matched configured %s column '%s' → '%s' (normalised)", role, configured, detected)
        return detected

    # 3. Candidate list
    for cand in candidates:
        norm_cand = _normalize(cand)
        if norm_cand in lookup:
            detected = lookup[norm_cand]
            logger.info("Auto-detected %s column: '%s'", role, detected)
            return detected

    # 4. Fallback for id: first column
    if role == "id":
        detected = columns[0]
        logger.info("Auto-detected %s column (fallback to first column): '%s'", role, detected)
        return detected

    raise KeyError(
        f"Could not detect {role} column. Configured='{configured}', "
        f"candidates={candidates}, available={list(columns)}"
    )


def _is_float_column_name(name: str) -> bool:
    """Return True if *name* looks like a numeric wavelength (e.g. '9993.76781')."""
    try:
        float(name)
        return True
    except (ValueError, TypeError):
        return False


def _detect_feature_columns(
    df: pd.DataFrame,
    cfg: Config,
    id_col: str,
    target_col: str,
) -> list[str]:
    """Detect feature columns.

    Priority:
    1. ``cfg.feature_prefix`` — columns starting with that prefix.
    2. Columns whose *name* parses as a float (spectral wavelength convention).
    3. All numeric-dtype columns excluding id/target and known categoricals.
    """
    if cfg.feature_prefix:
        feat_cols = [c for c in df.columns if c.startswith(cfg.feature_prefix)]
        if feat_cols:
            return feat_cols

    exclude = {id_col, target_col}
    if cfg.group_col:
        exclude.add(cfg.group_col)

    # Try float-name columns first (spectral wavelengths)
    float_cols = [
        c for c in df.columns
        if c not in exclude and _is_float_column_name(c)
    ]
    if float_cols:
        return float_cols

    # Generic fallback: all numeric-dtype columns except id/target
    return [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]


# ---------------------------------------------------------------------------
# CSV reading
# ---------------------------------------------------------------------------

def _read_csv(path: Path, encoding: str = "") -> pd.DataFrame:
    """Read a CSV, auto-detecting encoding if not specified."""
    if encoding:
        return pd.read_csv(path, encoding=encoding)
    for enc in _ENCODING_CANDIDATES:
        try:
            df = pd.read_csv(path, encoding=enc)
            logger.info("Read %s with encoding=%s", path.name, enc)
            return df
        except (UnicodeDecodeError, UnicodeError):
            continue
    return pd.read_csv(path, encoding="latin-1")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_train(cfg: Config, data_dir: Path | None = None) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    """Load training data and return (X, y, ids).

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)
    ids : pandas Index
    """
    data_dir = data_dir or DATA_RAW
    df = _read_csv(data_dir / cfg.train_file, cfg.encoding)
    logger.info("Train columns (%d): %s", len(df.columns), list(df.columns))

    id_col = _find_column(df.columns, cfg.id_col, _ID_CANDIDATES, "id")
    target_col = _find_column(df.columns, cfg.target_col, _TARGET_CANDIDATES, "target")

    ids = df[id_col]
    y = df[target_col].values.astype(np.float64)

    feat_cols = _detect_feature_columns(df, cfg, id_col, target_col)
    logger.info("Detected id column: '%s'", id_col)
    logger.info("Detected target column: '%s'", target_col)
    logger.info("Feature columns: %d", len(feat_cols))

    X = df[feat_cols].values.astype(np.float64)
    logger.info("Train shape: X=%s, y=%s", X.shape, y.shape)
    return X, y, ids


def load_test(cfg: Config, data_dir: Path | None = None) -> tuple[np.ndarray, pd.Index]:
    """Load test data and return (X, ids).

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
    ids : pandas Index
    """
    data_dir = data_dir or DATA_RAW
    df = _read_csv(data_dir / cfg.test_file, cfg.encoding)
    logger.info("Test columns (%d): %s", len(df.columns), list(df.columns))

    id_col = _find_column(df.columns, cfg.id_col, _ID_CANDIDATES, "id")
    ids = df[id_col]

    # For test data target_col may not exist; use a safe dummy for exclusion
    target_col = cfg.target_col if cfg.target_col in df.columns else "__absent__"

    feat_cols = _detect_feature_columns(df, cfg, id_col, target_col)
    logger.info("Detected id column: '%s'", id_col)
    logger.info("Feature columns: %d", len(feat_cols))

    X = df[feat_cols].values.astype(np.float64)
    logger.info("Test shape: X=%s", X.shape)
    return X, ids
