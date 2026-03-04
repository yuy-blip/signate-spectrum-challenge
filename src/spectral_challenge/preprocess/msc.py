"""Multiplicative Scatter Correction (MSC).

Fit computes the mean spectrum from the training set.  Transform corrects
each sample by linear regression against that reference.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class MSCTransformer(BaseEstimator, TransformerMixin):
    """Multiplicative Scatter Correction."""

    def __init__(self) -> None:
        self.reference_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: None = None) -> MSCTransformer:
        self.reference_ = X.mean(axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.reference_ is None:
            raise RuntimeError("MSCTransformer has not been fitted.")
        X_corrected = np.empty_like(X)
        for i in range(X.shape[0]):
            coef = np.polyfit(self.reference_, X[i], deg=1)
            X_corrected[i] = (X[i] - coef[1]) / coef[0]
        return X_corrected
