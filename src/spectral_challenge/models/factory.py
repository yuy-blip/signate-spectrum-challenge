"""Model factory – create estimators from config.

Supported model types
---------------------
- ``ridge``      → sklearn Ridge
- ``pls``        → sklearn PLSRegression
- ``svr``        → sklearn SVR
- ``lgbm``       → LightGBM LGBMRegressor

To add a new model (e.g. a 1-D CNN), register it in ``_REGISTRY``.
"""

from __future__ import annotations

from typing import Any

from sklearn.base import RegressorMixin

_REGISTRY: dict[str, tuple[str, str]] = {
    "ridge": ("sklearn.linear_model", "Ridge"),
    "pls": ("sklearn.cross_decomposition", "PLSRegression"),
    "svr": ("sklearn.svm", "SVR"),
    "lgbm": ("lightgbm", "LGBMRegressor"),
    "elastic_net": ("sklearn.linear_model", "ElasticNet"),
    "lasso": ("sklearn.linear_model", "Lasso"),
}


def create_model(model_type: str, params: dict[str, Any] | None = None) -> RegressorMixin:
    """Instantiate a regression model.

    Parameters
    ----------
    model_type : str
        Key in ``_REGISTRY``.
    params : dict, optional
        ``__init__`` keyword arguments forwarded to the model constructor.

    Returns
    -------
    sklearn-compatible estimator
    """
    params = params or {}
    entry = _REGISTRY.get(model_type)
    if entry is None:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Available: {sorted(_REGISTRY.keys())}"
        )

    module_path, class_name = entry
    import importlib

    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**params)
