"""Logging helpers."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "spectral_challenge",
    log_file: Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Return a logger that writes to stderr and optionally to a file."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    if not logger.handlers:
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
