"""Water-aware EMSC: EMSC with water spectrum as an additional basis function.

The standard EMSC decomposes each spectrum as:
    x = a_0 + a_1 * ref + poly_terms + residual

This version adds a "water component" as an additional basis function:
    x = a_0 + a_1 * ref + a_water * water_spec + poly_terms + residual

The water_spec is estimated from training data as the correlation between
each wavelength and the target variable. The a_water coefficient directly
indicates water content and is species-invariant.

Also includes WaterCoefficientExtractor that outputs the EMSC coefficients
(especially a_water) as features for the model.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class WaterAwareEMSCTransformer(BaseEstimator, TransformerMixin):
    """EMSC with water spectrum estimated from training data.

    During fit, estimates the "water direction" in spectral space from
    the correlation between spectra and target values. During transform,
    decomposes each spectrum using this water direction as an additional
    basis function, then returns the corrected spectrum.

    Parameters
    ----------
    poly_order : int
        Maximum polynomial degree for baseline modelling.
    eps : float
        Guard for coefficients.
    return_coefficients : bool
        If True, append the EMSC coefficients (including water coefficient)
        as additional features.
    """

    def __init__(
        self,
        poly_order: int = 2,
        eps: float = 1e-10,
        return_coefficients: bool = False,
    ) -> None:
        self.poly_order = poly_order
        self.eps = eps
        self.return_coefficients = return_coefficients

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> WaterAwareEMSCTransformer:
        self.reference_ = X.mean(axis=0)
        n_wl = X.shape[1]

        # Estimate water spectrum from correlation with target
        if y is not None:
            # Water direction: how each wavelength correlates with moisture
            self.water_spec_ = np.zeros(n_wl)
            for j in range(n_wl):
                corr = np.corrcoef(X[:, j], y)[0, 1]
                self.water_spec_[j] = corr if np.isfinite(corr) else 0.0
            # Normalize
            norm = np.linalg.norm(self.water_spec_)
            if norm > self.eps:
                self.water_spec_ /= norm
            self.has_water_ = True
        else:
            self.has_water_ = False

        # Normalised wavelength indices [-1, 1]
        w = np.linspace(-1, 1, n_wl)

        # Design matrix: [ref, water_spec, 1, w, w^2, ..., w^p]
        cols = [self.reference_]
        if self.has_water_:
            cols.append(self.water_spec_)
        cols.append(np.ones(n_wl))
        for p in range(1, self.poly_order + 1):
            cols.append(w ** p)
        self._design = np.column_stack(cols)
        self._n_basis = len(cols)
        # Index of water coefficient (1 if present, None otherwise)
        self._water_idx = 1 if self.has_water_ else None
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._design is None:
            raise RuntimeError("WaterAwareEMSCTransformer has not been fitted.")

        D = self._design
        coefs, _, _, _ = np.linalg.lstsq(D, X.T, rcond=None)

        a1 = coefs[0]  # reference coefficient
        a1_safe = np.where(np.abs(a1) < self.eps, self.eps, a1)

        # baseline = everything except reference (and optionally water)
        # We want to remove: intercept + polynomial terms
        # Keep: reference scaling and water component
        if self.has_water_:
            # baseline = cols[2:] (intercept + poly)
            baseline_start = 2
        else:
            baseline_start = 1

        baseline = D[:, baseline_start:] @ coefs[baseline_start:]
        X_corrected = ((X.T - baseline) / a1_safe).T

        if self.return_coefficients and self.has_water_:
            # Append water coefficient as an extra feature
            a_water = coefs[self._water_idx]  # (n_samples,)
            X_corrected = np.column_stack([X_corrected, a_water])

        return X_corrected


class EMSCCoefficientExtractor(BaseEstimator, TransformerMixin):
    """Extract EMSC coefficients as features.

    Instead of returning the corrected spectrum, returns the EMSC
    decomposition coefficients (reference scaling, polynomial coefficients).
    These coefficients encode scattering properties which may carry
    species-independent moisture information.

    Parameters
    ----------
    poly_order : int
        Maximum polynomial degree.
    include_corrected : bool
        If True, append coefficients to the corrected spectrum.
        If False, return only the coefficients.
    """

    def __init__(self, poly_order: int = 2, include_corrected: bool = True) -> None:
        self.poly_order = poly_order
        self.include_corrected = include_corrected

    def fit(self, X: np.ndarray, y: None = None) -> EMSCCoefficientExtractor:
        self.reference_ = X.mean(axis=0)
        n_wl = X.shape[1]
        w = np.linspace(-1, 1, n_wl)
        cols = [self.reference_, np.ones(n_wl)]
        for p in range(1, self.poly_order + 1):
            cols.append(w ** p)
        self._design = np.column_stack(cols)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        D = self._design
        coefs, _, _, _ = np.linalg.lstsq(D, X.T, rcond=None)

        # coefs shape: (n_basis, n_samples)
        coef_features = coefs.T  # (n_samples, n_basis)

        if self.include_corrected:
            a1 = coefs[0]
            a1_safe = np.where(np.abs(a1) < 1e-10, 1e-10, a1)
            baseline = D[:, 1:] @ coefs[1:]
            X_corrected = ((X.T - baseline) / a1_safe).T
            return np.column_stack([X_corrected, coef_features])
        else:
            return coef_features
