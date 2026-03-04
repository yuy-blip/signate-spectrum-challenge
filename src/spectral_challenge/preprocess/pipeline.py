"""Build a preprocessing pipeline from config entries.

Each entry in ``cfg.preprocess`` is a dict like::

    {"name": "snv"}
    {"name": "sg", "window_length": 11, "polyorder": 2, "deriv": 1}
    {"name": "absorbance", "base": "10"}
    {"name": "binning", "bin_size": 4}
    {"name": "standard_scaler"}

The pipeline returned is an sklearn ``Pipeline`` that supports
``fit`` / ``transform`` / ``fit_transform``.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from spectral_challenge.preprocess.absorbance import AbsorbanceTransformer
from spectral_challenge.preprocess.binning import BinningTransformer
from spectral_challenge.preprocess.msc import EMSCTransformer, MSCTransformer
from spectral_challenge.preprocess.sg import DerivativeTransformer, SavitzkyGolayTransformer
from spectral_challenge.preprocess.snv import SNVTransformer
from spectral_challenge.preprocess.wavelength_selector import WavelengthSelector
from spectral_challenge.preprocess.feature_engineering import (
    BandRatioTransformer,
    PCAFeatureTransformer,
    SpectralStatsTransformer,
)

logger = logging.getLogger("spectral_challenge")

_REGISTRY: dict[str, type] = {
    "snv": SNVTransformer,
    "msc": MSCTransformer,
    "emsc": EMSCTransformer,
    "sg": SavitzkyGolayTransformer,
    "savitzky_golay": SavitzkyGolayTransformer,
    "derivative": DerivativeTransformer,
    "absorbance": AbsorbanceTransformer,
    "binning": BinningTransformer,
    "standard_scaler": StandardScaler,
    "wavelength_selector": WavelengthSelector,
    "select_wn": WavelengthSelector,
    "band_ratio": BandRatioTransformer,
    "spectral_stats": SpectralStatsTransformer,
    "pca_features": PCAFeatureTransformer,
}


class _NaNInfGuard(BaseEstimator, TransformerMixin):
    """Check for NaN/inf after a specific preprocessing step."""

    def __init__(self, after_step: str) -> None:
        self.after_step = after_step

    def fit(self, X: np.ndarray, y: None = None) -> _NaNInfGuard:
        self.is_fitted_ = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not np.all(np.isfinite(X)):
            n_nan = int(np.isnan(X).sum())
            n_inf = int(np.isinf(X).sum())
            raise ValueError(
                f"NaN/inf detected after '{self.after_step}': "
                f"{n_nan} NaN, {n_inf} inf in array of shape {X.shape}"
            )
        return X


def build_preprocess_pipeline(steps_cfg: list[dict[str, Any]]) -> Pipeline:
    """Construct an sklearn Pipeline from a list of step dicts.

    Parameters
    ----------
    steps_cfg : list[dict]
        Each dict must have a ``"name"`` key matching a registered
        transformer.  Any other keys are passed as ``__init__`` kwargs.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    steps: list[tuple[str, Any]] = []
    seen: dict[str, int] = {}
    for entry in steps_cfg:
        entry = dict(entry)  # shallow copy
        name = entry.pop("name")
        cls = _REGISTRY.get(name)
        if cls is None:
            raise ValueError(
                f"Unknown preprocessor '{name}'. Available: {sorted(_REGISTRY.keys())}"
            )
        # deduplicate step names
        count = seen.get(name, 0)
        seen[name] = count + 1
        step_name = name if count == 0 else f"{name}_{count}"
        steps.append((step_name, cls(**entry)))
        # Insert a NaN/inf guard after each real step
        steps.append((f"_guard_{step_name}", _NaNInfGuard(after_step=step_name)))
    if not steps:
        steps.append(("passthrough", "passthrough"))
    return Pipeline(steps)
