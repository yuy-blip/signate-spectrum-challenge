"""Cross-validation split strategies."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.model_selection import GroupKFold, KFold

if TYPE_CHECKING:
    from collections.abc import Iterator

    from spectral_challenge.config import Config


def get_cv_splitter(cfg: Config) -> KFold | GroupKFold:
    """Return a CV splitter based on config."""
    if cfg.split_method == "group_kfold":
        return GroupKFold(n_splits=cfg.n_folds)
    return KFold(n_splits=cfg.n_folds, shuffle=cfg.shuffle, random_state=cfg.seed)


def cv_splits(
    cfg: Config,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, val_idx) for each fold."""
    splitter = get_cv_splitter(cfg)
    if cfg.split_method == "group_kfold":
        if groups is None:
            raise ValueError("group_col must be provided for group_kfold")
        yield from splitter.split(X, y, groups)
    else:
        yield from splitter.split(X, y)
