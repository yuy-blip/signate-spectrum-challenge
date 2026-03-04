"""Wavelength (wavenumber) region selector for spectral data.

Selects columns corresponding to specific wavenumber ranges,
useful for focusing on known absorption bands (e.g., water O-H bands).
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class WavelengthSelector(BaseEstimator, TransformerMixin):
    """Select spectral columns within specified wavenumber ranges.

    Parameters
    ----------
    ranges : list of (float, float)
        List of (low, high) wavenumber ranges to keep.
        Columns whose wavenumber falls within any range are retained.
    wavenumbers : list of float or None
        The wavenumber for each column. If None, will be set during fit
        from the column count (assuming linearly spaced from high to low).
    exclude : bool
        If True, *exclude* the specified ranges instead of including them.
    """

    def __init__(
        self,
        ranges: list[tuple[float, float]] | None = None,
        wavenumbers: list[float] | None = None,
        exclude: bool = False,
    ) -> None:
        self.ranges = ranges
        self.wavenumbers = wavenumbers
        self.exclude = exclude

    def fit(self, X: np.ndarray, y: None = None) -> WavelengthSelector:
        n_features = X.shape[1]
        if self.wavenumbers is not None:
            wn = np.array(self.wavenumbers, dtype=float)
        else:
            # Default: linearly spaced from ~10000 to ~4000 cm-1 (typical NIR)
            wn = np.linspace(9993.77, 3999.82, n_features)

        if self.ranges is None:
            # No selection: keep all
            self.mask_ = np.ones(n_features, dtype=bool)
        else:
            mask = np.zeros(n_features, dtype=bool)
            for lo, hi in self.ranges:
                mask |= (wn >= lo) & (wn <= hi)
            if self.exclude:
                mask = ~mask
            self.mask_ = mask
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.mask_]
