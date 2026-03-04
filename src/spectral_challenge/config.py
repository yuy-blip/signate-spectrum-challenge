"""Config loading from YAML with sensible defaults."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    """Experiment configuration."""

    # --- data ---
    train_file: str = "train.csv"
    test_file: str = "test.csv"
    id_col: str = "id"
    target_col: str = "y"
    feature_prefix: str = ""  # empty = auto-detect (all numeric cols except id/target)
    encoding: str = ""  # empty = auto-detect (try utf-8 → cp932)

    # --- preprocessing ---
    preprocess: list[dict[str, Any]] = field(default_factory=lambda: [{"name": "standard_scaler"}])

    # --- model ---
    model_type: str = "ridge"
    model_params: dict[str, Any] = field(default_factory=dict)

    # --- CV ---
    n_folds: int = 5
    split_method: str = "kfold"  # kfold | group_kfold
    group_col: str | None = None
    seed: int = 42
    shuffle: bool = True

    # --- output ---
    outdir: str = ""  # filled at runtime if empty
    experiment_name: str = "default"

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        overrides: dict[str, Any] | None = None,
        cli_overrides: list[str] | None = None,
    ) -> Config:
        """Load config from a YAML file, with optional CLI overrides.

        Parameters
        ----------
        path : str or Path
            Path to the YAML config file.
        overrides : dict, optional
            Simple key-value overrides (legacy, flat dict only).
        cli_overrides : list of str, optional
            Hydra-style ``key=value`` strings (supports dot-notation).
        """
        import logging

        from spectral_challenge.config_override import apply_overrides, parse_overrides

        log = logging.getLogger("spectral_challenge")

        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        # Legacy flat overrides
        if overrides:
            raw.update(overrides)

        # Hydra-style dot-notation overrides
        if cli_overrides:
            valid_keys = set(cls.__dataclass_fields__)
            parsed = parse_overrides(cli_overrides)
            apply_overrides(raw, parsed, valid_top_keys=valid_keys)
            for orig_key, _segments, value in parsed:
                log.info("Config override: %s = %r", orig_key, value)

        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (for saving alongside runs)."""
        from dataclasses import asdict

        return asdict(self)
