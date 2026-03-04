"""Multiplicative Scatter Correction (MSC) and Extended MSC (EMSC).

Fit computes the mean spectrum from the training set.  Transform corrects
each sample by linear regression against that reference.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class MSCTransformer(BaseEstimator, TransformerMixin):
    """Multiplicative Scatter Correction.

    Parameters
    ----------
    eps : float
        Guard value — if the estimated slope *b* is smaller than *eps*
        in absolute value, it is clamped to ``sign(b) * eps``.
    """

    def __init__(self, eps: float = 1e-10) -> None:
        self.eps = eps
        self.reference_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: None = None) -> MSCTransformer:
        self.reference_ = X.mean(axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.reference_ is None:
            raise RuntimeError("MSCTransformer has not been fitted.")
        ref = self.reference_
        X_corrected = np.empty_like(X)
        for i in range(X.shape[0]):
            coef = np.polyfit(ref, X[i], deg=1)  # [slope, intercept]
            b = coef[0]
            if abs(b) < self.eps:
                b = np.sign(b) * self.eps if b != 0 else self.eps
            X_corrected[i] = (X[i] - coef[1]) / b
        return X_corrected


class EMSCTransformer(BaseEstimator, TransformerMixin):
    """Extended Multiplicative Scatter Correction (EMSC).

    Fits each sample as::

        x = a_0 + a_1 * ref + a_2 * w + a_3 * w^2 + ... + a_{p+1} * w^p + residual

    and returns ``x_corr = (x - baseline) / a_1`` where *baseline* is the
    polynomial part.

    Parameters
    ----------
    poly_order : int
        Maximum polynomial degree for baseline modelling (default 2).
    eps : float
        Guard for the reference coefficient *a_1*.
    """

    def __init__(self, poly_order: int = 2, eps: float = 1e-10) -> None:
        self.poly_order = poly_order
        self.eps = eps
        self.reference_: np.ndarray | None = None
        self._design: np.ndarray | None = None  # (n_wl, n_basis)

    def fit(self, X: np.ndarray, y: None = None) -> EMSCTransformer:
        self.reference_ = X.mean(axis=0)
        n_wl = X.shape[1]
        # Normalised wavelength indices [-1, 1]
        w = np.linspace(-1, 1, n_wl)
        # Design matrix: [ref, 1, w, w^2, ..., w^p]
        cols = [self.reference_, np.ones(n_wl)]
        for p in range(1, self.poly_order + 1):
            cols.append(w ** p)
        self._design = np.column_stack(cols)  # (n_wl, 2+poly_order)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._design is None:
            raise RuntimeError("EMSCTransformer has not been fitted.")
        D = self._design
        # lstsq for all samples at once: coefs shape (n_basis, n_samples)
        coefs, _, _, _ = np.linalg.lstsq(D, X.T, rcond=None)
        a1 = coefs[0]  # reference coefficient per sample
        a1_safe = np.where(np.abs(a1) < self.eps, self.eps, a1)
        # baseline = D[:, 1:] @ coefs[1:]  (everything except ref column)
        baseline = D[:, 1:] @ coefs[1:]  # (n_wl, n_samples)
        X_corrected = ((X.T - baseline) / a1_safe).T
        return X_corrected
