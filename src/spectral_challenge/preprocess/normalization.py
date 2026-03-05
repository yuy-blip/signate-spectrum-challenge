"""Additional normalization methods for spectral cross-species generalization.

- AreaNormalize: divide each spectrum by its total area (L1 norm)
- MaxNormalize: divide each spectrum by its maximum value
- RangeNormalize: min-max normalize each row
- ContinuumRemoval: normalize by convex hull upper envelope
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class AreaNormalizeTransformer(BaseEstimator, TransformerMixin):
    """Normalize each spectrum by its total area (L1 norm).

    This removes overall intensity differences between samples,
    making the model focus on spectral shape rather than magnitude.
    Species-invariant since it removes scattering effects.
    """

    def __init__(self, eps: float = 1e-10) -> None:
        self.eps = eps

    def fit(self, X: np.ndarray, y: None = None) -> AreaNormalizeTransformer:
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        area = np.abs(X).sum(axis=1, keepdims=True)
        area = np.maximum(area, self.eps)
        return X / area


class MaxNormalizeTransformer(BaseEstimator, TransformerMixin):
    """Normalize each spectrum by its maximum absolute value."""

    def __init__(self, eps: float = 1e-10) -> None:
        self.eps = eps

    def fit(self, X: np.ndarray, y: None = None) -> MaxNormalizeTransformer:
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        max_val = np.abs(X).max(axis=1, keepdims=True)
        max_val = np.maximum(max_val, self.eps)
        return X / max_val


class RangeNormalizeTransformer(BaseEstimator, TransformerMixin):
    """Min-max normalize each spectrum row-wise."""

    def __init__(self, eps: float = 1e-10) -> None:
        self.eps = eps

    def fit(self, X: np.ndarray, y: None = None) -> RangeNormalizeTransformer:
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        xmin = X.min(axis=1, keepdims=True)
        xmax = X.max(axis=1, keepdims=True)
        denom = np.maximum(xmax - xmin, self.eps)
        return (X - xmin) / denom


class ContinuumRemovalTransformer(BaseEstimator, TransformerMixin):
    """Continuum removal: divide spectrum by its convex hull upper envelope.

    This normalizes absorption features relative to a smooth continuum,
    making band depths comparable across different species with different
    baseline characteristics. Widely used in remote sensing.
    """

    def __init__(self) -> None:
        pass

    def fit(self, X: np.ndarray, y: None = None) -> ContinuumRemovalTransformer:
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        result = np.empty_like(X)
        n_wl = X.shape[1]
        x_idx = np.arange(n_wl)

        for i in range(X.shape[0]):
            spectrum = X[i]
            # Build upper convex hull
            hull = self._upper_hull(x_idx, spectrum)
            # Divide by hull (continuum removal)
            hull_safe = np.maximum(np.abs(hull), 1e-10) * np.sign(hull + 1e-10)
            result[i] = spectrum / hull_safe

        return result

    @staticmethod
    def _upper_hull(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute upper convex hull envelope by linear interpolation."""
        n = len(x)
        # Find convex hull vertices (upper hull)
        hull_x = [0]  # start with first point

        for j in range(1, n):
            while len(hull_x) > 1:
                # Check if last hull point is below the line from prev to current
                x0, x1, x2 = x[hull_x[-2]], x[hull_x[-1]], x[j]
                y0, y1, y2 = y[hull_x[-2]], y[hull_x[-1]], y[j]
                # Cross product: if (x1-x0, y1-y0) x (x2-x0, y2-y0) >= 0, remove middle
                cross = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
                if cross >= 0:
                    hull_x.pop()
                else:
                    break
            hull_x.append(j)

        # Linear interpolation between hull vertices
        hull_y = np.interp(x, x[hull_x], y[hull_x])
        return hull_y
