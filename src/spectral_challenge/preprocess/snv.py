"""Standard Normal Variate (SNV) transformation.

SNV normalises each *sample* (row) to zero mean and unit variance.
It is stateless – fit is a no-op.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class SNVTransformer(BaseEstimator, TransformerMixin):
    """Row-wise Standard Normal Variate."""

    def fit(self, X: np.ndarray, y: None = None) -> SNVTransformer:
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        mean = X.mean(axis=1, keepdims=True)
        std = X.std(axis=1, keepdims=True)
        std[std == 0] = 1.0
        return (X - mean) / std
