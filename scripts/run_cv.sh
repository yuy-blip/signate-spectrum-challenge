#!/usr/bin/env bash
# Convenience wrapper for CV training.
# Usage: bash scripts/run_cv.sh [configs/baseline_ridge.yaml]

set -euo pipefail

CONFIG="${1:-configs/baseline_ridge.yaml}"
echo "Running CV with config: ${CONFIG}"
python -m spectral_challenge.cli cv --config "${CONFIG}"
