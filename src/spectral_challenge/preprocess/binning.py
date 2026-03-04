"""Wavelength binning (averaging adjacent channels).

Reduces spectral dimensionality by averaging groups of adjacent features.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class BinningTransformer(BaseEstimator, TransformerMixin):
    """Average adjacent spectral channels into bins.

    Parameters
    ----------
    bin_size : int
        Number of adjacent channels to average per bin (default 4).
        The last bin may contain fewer channels if ``n_features`` is not
        evenly divisible.
    """

    def __init__(self, bin_size: int = 4) -> None:
        self.bin_size = bin_size

    def fit(self, X: np.ndarray, y: None = None) -> BinningTransformer:
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        n_samples, n_features = X.shape
        bs = self.bin_size
        # Number of complete bins (last partial bin handled separately)
        n_full = n_features // bs
        remainder = n_features % bs

        parts = []
        if n_full > 0:
            # Reshape complete bins and average along last axis
            X_full = X[:, : n_full * bs].reshape(n_samples, n_full, bs)
            parts.append(X_full.mean(axis=2))
        if remainder > 0:
            parts.append(X[:, n_full * bs :].mean(axis=1, keepdims=True))

        return np.hstack(parts)
