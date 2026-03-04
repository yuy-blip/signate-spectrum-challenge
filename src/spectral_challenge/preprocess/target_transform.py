"""Target transformation wrappers for sklearn regressors.

Wraps a regressor to apply a transformation to y before fitting
and inverse-transforms predictions. Useful for skewed targets.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone


class TransformedTargetRegressor(BaseEstimator, RegressorMixin):
    """Regressor that transforms the target before fitting.

    Parameters
    ----------
    regressor : estimator
        The base regressor.
    transform : str
        'log1p' : y' = log1p(y), pred = expm1(y')
        'log'   : y' = log(y + offset), pred = exp(y') - offset
        'sqrt'  : y' = sqrt(y), pred = y'^2
        'boxcox': y' = boxcox(y, lmbda), pred = inv_boxcox(y', lmbda)
        'none'  : identity
    lmbda : float
        Lambda for Box-Cox (only used when transform='boxcox').
    """

    def __init__(
        self,
        regressor=None,
        transform: str = "none",
        lmbda: float = 0.5,
    ) -> None:
        self.regressor = regressor
        self.transform = transform
        self.lmbda = lmbda

    def _forward(self, y: np.ndarray) -> np.ndarray:
        if self.transform == "log1p":
            return np.log1p(np.clip(y, 0, None))
        elif self.transform == "log":
            self.offset_ = max(0, -y.min() + 1.0)
            return np.log(y + self.offset_)
        elif self.transform == "sqrt":
            return np.sqrt(np.clip(y, 0, None))
        elif self.transform == "boxcox":
            from scipy.special import boxcox1p
            return boxcox1p(np.clip(y, 0, None), self.lmbda)
        return y

    def _inverse(self, y: np.ndarray) -> np.ndarray:
        if self.transform == "log1p":
            return np.expm1(y)
        elif self.transform == "log":
            return np.exp(y) - self.offset_
        elif self.transform == "sqrt":
            return y**2
        elif self.transform == "boxcox":
            from scipy.special import inv_boxcox1p
            return inv_boxcox1p(y, self.lmbda)
        return y

    def fit(self, X, y):
        self.regressor_ = clone(self.regressor) if self.regressor else None
        y_t = self._forward(y)
        if self.regressor_ is not None:
            self.regressor_.fit(X, y_t)
        return self

    def predict(self, X):
        y_t = self.regressor_.predict(X)
        return self._inverse(y_t.ravel())
