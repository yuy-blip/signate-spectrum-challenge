"""Absorbance transformation for reflectance/transmittance spectra.

Converts positive spectral values to absorbance: ``A = -log(X)``.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class AbsorbanceTransformer(BaseEstimator, TransformerMixin):
    """Convert reflectance / transmittance to absorbance.

    Parameters
    ----------
    base : str
        ``"e"`` for natural log (default), ``"10"`` for log10.
    eps : float
        Small value to clamp X away from zero before taking log.
    """

    def __init__(self, base: str = "e", eps: float = 1e-8) -> None:
        self.base = base
        self.eps = eps

    def fit(self, X: np.ndarray, y: None = None) -> AbsorbanceTransformer:
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_clamped = np.clip(X, self.eps, None)
        if self.base == "10":
            return -np.log10(X_clamped)
        return -np.log(X_clamped)
