"""LightGBM with Mixup data augmentation.

Generates synthetic training samples by linearly interpolating between
existing samples. This helps the model learn to predict in ranges beyond
what individual training samples show.

Physics justification: NIR follows Beer-Lambert law (A = εlc), so
linearly mixing spectra of different moisture contents produces a
spectrum with linearly interpolated moisture content.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class MixupLGBMRegressor(BaseEstimator, RegressorMixin):
    """LightGBM with Mixup augmentation during training.

    Parameters
    ----------
    n_augmented : int
        Number of synthetic samples to generate.
    alpha : float
        Beta distribution parameter for Mixup interpolation.
        alpha=1.0 gives uniform mixing, alpha=0.2 gives near-boundary mixing.
    focus_high : bool
        If True, preferentially generate high-value synthetic samples.
    lgbm_params : dict
        Parameters passed to LGBMRegressor.
    """

    def __init__(
        self,
        n_augmented: int = 500,
        alpha: float = 0.4,
        focus_high: bool = True,
        seed: int = 42,
        **lgbm_params,
    ):
        self.n_augmented = n_augmented
        self.alpha = alpha
        self.focus_high = focus_high
        self.seed = seed
        self.lgbm_params = lgbm_params

    def fit(self, X, y):
        from lightgbm import LGBMRegressor

        rng = np.random.RandomState(self.seed)
        n = len(y)

        # Generate Mixup samples
        X_aug_list = [X]
        y_aug_list = [y]

        for _ in range(self.n_augmented):
            # Sample two indices
            if self.focus_high:
                # Preferentially pick at least one high-value sample
                p = np.maximum(y - np.median(y), 0.0)
                p = p / p.sum() if p.sum() > 0 else np.ones(n) / n
                i = rng.choice(n, p=p)
            else:
                i = rng.randint(n)
            j = rng.randint(n)

            # Mixup interpolation
            lam = rng.beta(self.alpha, self.alpha)
            x_new = lam * X[i] + (1 - lam) * X[j]
            y_new = lam * y[i] + (1 - lam) * y[j]

            X_aug_list.append(x_new.reshape(1, -1))
            y_aug_list.append(np.array([y_new]))

        X_train = np.vstack(X_aug_list)
        y_train = np.concatenate(y_aug_list)

        self.model_ = LGBMRegressor(**self.lgbm_params)
        self.model_.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model_.predict(X)
