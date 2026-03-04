"""Spectral feature engineering transformers.

Generate additional features from raw spectra:
- Band ratios (known water indices)
- Statistical features (slope, curvature at key regions)
- PCA components as features
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA


class BandRatioTransformer(BaseEstimator, TransformerMixin):
    """Compute ratios between spectral bands as additional features.

    Useful for creating moisture-sensitive indices from NIR spectra.
    The ratios are appended to the original features.

    Parameters
    ----------
    ratios : list of (int, int) or list of ((int,int), (int,int))
        Each entry defines a ratio. If (a, b), computes X[:,a]/X[:,b].
        If ((a1,a2),(b1,b2)), computes mean(X[:,a1:a2]) / mean(X[:,b1:b2]).
    append : bool
        If True, append ratios to original X. If False, return only ratios.
    """

    def __init__(
        self,
        ratios: list | None = None,
        append: bool = True,
    ) -> None:
        self.ratios = ratios
        self.append = append

    def fit(self, X: np.ndarray, y: None = None) -> BandRatioTransformer:
        n_feat = X.shape[1]
        if self.ratios is None:
            # Auto-generate: water-sensitive ratios for typical NIR
            # Divide spectrum into ~10 equal regions, compute cross-ratios
            step = max(1, n_feat // 10)
            self.ratios_ = []
            regions = list(range(0, n_feat, step))
            for i in range(len(regions) - 1):
                for j in range(i + 1, min(i + 3, len(regions))):
                    self.ratios_.append(
                        ((regions[i], regions[i] + step), (regions[j], regions[j] + step))
                    )
        else:
            self.ratios_ = self.ratios
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        ratio_features = []
        for r in self.ratios_:
            if isinstance(r[0], (list, tuple)):
                (a1, a2), (b1, b2) = r
                num = X[:, a1:a2].mean(axis=1)
                den = X[:, b1:b2].mean(axis=1)
            else:
                a, b = r
                num = X[:, a]
                den = X[:, b]
            den = np.where(np.abs(den) < 1e-10, 1e-10, den)
            ratio_features.append((num / den).reshape(-1, 1))

        ratios = np.hstack(ratio_features) if ratio_features else np.empty((X.shape[0], 0))

        if self.append:
            return np.hstack([X, ratios])
        return ratios


class SpectralStatsTransformer(BaseEstimator, TransformerMixin):
    """Extract statistical features from spectral regions.

    Computes per-sample statistics (mean, std, slope, max, min) over
    configurable spectral windows. Appends to original features.

    Parameters
    ----------
    n_regions : int
        Split spectrum into this many equal-width regions.
    stats : list of str
        Statistics to compute: 'mean', 'std', 'slope', 'max', 'min', 'skew'.
    append : bool
        If True, append stats to original X.
    """

    def __init__(
        self,
        n_regions: int = 8,
        stats: list[str] | None = None,
        append: bool = True,
    ) -> None:
        self.n_regions = n_regions
        self.stats = stats or ["mean", "std", "slope", "max", "min"]
        self.append = append

    def fit(self, X: np.ndarray, y: None = None) -> SpectralStatsTransformer:
        self.n_features_ = X.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        n_samples, n_feat = X.shape
        step = max(1, n_feat // self.n_regions)
        features = []

        for i in range(0, n_feat, step):
            region = X[:, i : i + step]
            if region.shape[1] == 0:
                continue
            for stat in self.stats:
                if stat == "mean":
                    features.append(region.mean(axis=1, keepdims=True))
                elif stat == "std":
                    features.append(region.std(axis=1, keepdims=True))
                elif stat == "slope":
                    # Linear slope across the region
                    x_axis = np.arange(region.shape[1], dtype=float)
                    x_centered = x_axis - x_axis.mean()
                    denom = (x_centered**2).sum()
                    if denom > 0:
                        slopes = (region @ x_centered) / denom
                    else:
                        slopes = np.zeros(n_samples)
                    features.append(slopes.reshape(-1, 1))
                elif stat == "max":
                    features.append(region.max(axis=1, keepdims=True))
                elif stat == "min":
                    features.append(region.min(axis=1, keepdims=True))
                elif stat == "skew":
                    m = region.mean(axis=1, keepdims=True)
                    s = region.std(axis=1, keepdims=True) + 1e-10
                    sk = ((region - m) ** 3).mean(axis=1, keepdims=True) / (s**3)
                    features.append(sk)

        stats_arr = np.hstack(features) if features else np.empty((n_samples, 0))
        if self.append:
            return np.hstack([X, stats_arr])
        return stats_arr


class PCAFeatureTransformer(BaseEstimator, TransformerMixin):
    """Extract PCA components and optionally append to original features.

    Parameters
    ----------
    n_components : int
        Number of PCA components.
    append : bool
        If True, append PCA components to original features.
    """

    def __init__(self, n_components: int = 20, append: bool = True) -> None:
        self.n_components = n_components
        self.append = append

    def fit(self, X: np.ndarray, y: None = None) -> PCAFeatureTransformer:
        self.pca_ = PCA(n_components=min(self.n_components, X.shape[1]))
        self.pca_.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        components = self.pca_.transform(X)
        if self.append:
            return np.hstack([X, components])
        return components
