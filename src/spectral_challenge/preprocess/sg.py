"""Savitzky-Golay filter and derivative transforms (sklearn-compatible)."""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin


class SavitzkyGolayTransformer(BaseEstimator, TransformerMixin):
    """Apply Savitzky-Golay smoothing (optionally with derivative).

    Parameters
    ----------
    window_length : int
        Must be odd and >= polyorder+1.
    polyorder : int
        Polynomial order for the filter.
    deriv : int
        Derivative order (0 = smoothing only, 1 = 1st derivative, ...).
    """

    def __init__(self, window_length: int = 11, polyorder: int = 2, deriv: int = 0) -> None:
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv

    def fit(self, X: np.ndarray, y: None = None) -> SavitzkyGolayTransformer:
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return savgol_filter(
            X,
            window_length=self.window_length,
            polyorder=self.polyorder,
            deriv=self.deriv,
            axis=1,
        )


class DerivativeTransformer(BaseEstimator, TransformerMixin):
    """Simple finite-difference derivative (numpy diff).

    Parameters
    ----------
    order : int
        1 = first derivative, 2 = second derivative.
    """

    def __init__(self, order: int = 1) -> None:
        self.order = order

    def fit(self, X: np.ndarray, y: None = None) -> DerivativeTransformer:
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        out = X.copy()
        for _ in range(self.order):
            out = np.diff(out, axis=1)
        return out
