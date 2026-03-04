"""Generate a submission CSV."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("spectral_challenge")


def make_submission(
    ids: pd.Index,
    preds: np.ndarray,
    id_col: str,
    target_col: str,
    out_path: Path,
) -> Path:
    """Write a submission CSV and return its path.

    Parameters
    ----------
    ids : array-like
        Sample identifiers.
    preds : ndarray
        Predicted target values.
    id_col : str
        Name of the ID column in the output CSV.
    target_col : str
        Name of the target column in the output CSV.
    out_path : Path
        Destination file path.

    Returns
    -------
    Path
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({id_col: ids, target_col: preds})
    df.to_csv(out_path, index=False)
    logger.info("Submission saved to %s (%d rows)", out_path, len(df))
    return out_path
