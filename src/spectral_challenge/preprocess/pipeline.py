"""Build a preprocessing pipeline from config entries.

Each entry in ``cfg.preprocess`` is a dict like::

    {"name": "snv"}
    {"name": "sg", "window_length": 11, "polyorder": 2, "deriv": 1}
    {"name": "standard_scaler"}

The pipeline returned is an sklearn ``Pipeline`` that supports
``fit`` / ``transform`` / ``fit_transform``.
"""

from __future__ import annotations

from typing import Any

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from spectral_challenge.preprocess.msc import MSCTransformer
from spectral_challenge.preprocess.sg import DerivativeTransformer, SavitzkyGolayTransformer
from spectral_challenge.preprocess.snv import SNVTransformer

_REGISTRY: dict[str, type] = {
    "snv": SNVTransformer,
    "msc": MSCTransformer,
    "sg": SavitzkyGolayTransformer,
    "savitzky_golay": SavitzkyGolayTransformer,
    "derivative": DerivativeTransformer,
    "standard_scaler": StandardScaler,
}


def build_preprocess_pipeline(steps_cfg: list[dict[str, Any]]) -> Pipeline:
    """Construct an sklearn Pipeline from a list of step dicts.

    Parameters
    ----------
    steps_cfg : list[dict]
        Each dict must have a ``"name"`` key matching a registered
        transformer.  Any other keys are passed as ``__init__`` kwargs.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    steps: list[tuple[str, Any]] = []
    seen: dict[str, int] = {}
    for entry in steps_cfg:
        entry = dict(entry)  # shallow copy
        name = entry.pop("name")
        cls = _REGISTRY.get(name)
        if cls is None:
            raise ValueError(
                f"Unknown preprocessor '{name}'. Available: {sorted(_REGISTRY.keys())}"
            )
        # deduplicate step names
        count = seen.get(name, 0)
        seen[name] = count + 1
        step_name = name if count == 0 else f"{name}_{count}"
        steps.append((step_name, cls(**entry)))
    if not steps:
        steps.append(("passthrough", "passthrough"))
    return Pipeline(steps)
