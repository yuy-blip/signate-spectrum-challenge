"""Canonical path resolution for the project."""

from pathlib import Path

# Repo root = two levels up from this file (src/spectral_challenge/paths.py)
ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
RUNS_DIR = ROOT / "runs"
SUBMISSIONS_DIR = ROOT / "submissions"
CONFIGS_DIR = ROOT / "configs"
