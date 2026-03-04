"""Hydra-style config override parsing and application.

Supports CLI strings like::

    --override model_params.n_components=60
    --override n_folds=3
    --override preprocess[0].deriv=1
    --override shuffle=false
"""

from __future__ import annotations

import re
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Value parsing (safe – no eval)
# ---------------------------------------------------------------------------

def _parse_value(raw: str) -> Any:
    """Convert a CLI value string to an appropriate Python type.

    Uses ``yaml.safe_load`` for safe type inference:
    true/false → bool, 123 → int, 0.1 → float, null → None, [1,2] → list, etc.
    Falls back to the raw string if yaml parsing fails.
    """
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


# ---------------------------------------------------------------------------
# Key path parsing  (dot notation + optional array index)
# ---------------------------------------------------------------------------

_SEGMENT_RE = re.compile(r"([^\.\[\]]+)|\[(\d+)\]")


def _parse_key_path(key: str) -> list[str | int]:
    """Parse a dotted key path into segments.

    Examples::

        "model_params.n_components"  → ["model_params", "n_components"]
        "preprocess[0].deriv"        → ["preprocess", 0, "deriv"]
        "n_folds"                    → ["n_folds"]
    """
    segments: list[str | int] = []
    for m in _SEGMENT_RE.finditer(key):
        if m.group(1) is not None:
            segments.append(m.group(1))
        else:
            segments.append(int(m.group(2)))
    if not segments:
        raise ValueError(f"Invalid override key: {key!r}")
    return segments


# ---------------------------------------------------------------------------
# Deep-set into a nested dict / list
# ---------------------------------------------------------------------------

def _deep_set(d: dict | list, segments: list[str | int], value: Any) -> None:
    """Set a value in a nested dict/list structure at the given path."""
    for i, seg in enumerate(segments[:-1]):
        if isinstance(seg, int):
            d = d[seg]
        else:
            if seg not in d:
                # auto-create intermediate dicts
                next_seg = segments[i + 1]
                d[seg] = [] if isinstance(next_seg, int) else {}
            d = d[seg]

    last = segments[-1]
    if isinstance(last, int):
        d[last] = value
    else:
        d[last] = value


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_overrides(raw_overrides: list[str]) -> list[tuple[str, list[str | int], Any]]:
    """Parse a list of ``key=value`` strings.

    Returns
    -------
    list of (original_key, segments, parsed_value)
    """
    results = []
    for item in raw_overrides:
        if "=" not in item:
            raise ValueError(
                f"Invalid override format: {item!r}. Expected 'key=value'."
            )
        key, _, raw_val = item.partition("=")
        key = key.strip()
        raw_val = raw_val.strip()
        segments = _parse_key_path(key)
        value = _parse_value(raw_val)
        results.append((key, segments, value))
    return results


def apply_overrides(
    raw_cfg: dict[str, Any],
    overrides: list[tuple[str, list[str | int], Any]],
    *,
    valid_top_keys: set[str] | None = None,
) -> dict[str, Any]:
    """Apply parsed overrides to a raw config dict (in-place) and return it.

    Parameters
    ----------
    raw_cfg : dict
        The raw YAML-loaded config dict.
    overrides : list
        Output of :func:`parse_overrides`.
    valid_top_keys : set, optional
        If provided, the top-level key of each override is validated against
        this set.  A ``ValueError`` is raised on unknown keys.
    """
    for original_key, segments, value in overrides:
        top_key = segments[0]
        if valid_top_keys is not None and isinstance(top_key, str) and top_key not in valid_top_keys:
            raise ValueError(
                f"Unknown config key: {original_key!r}. "
                f"Valid top-level keys: {sorted(valid_top_keys)}"
            )
        _deep_set(raw_cfg, segments, value)
    return raw_cfg
