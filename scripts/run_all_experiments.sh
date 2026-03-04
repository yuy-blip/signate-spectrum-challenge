#!/usr/bin/env bash
# Master script: Run ALL experiment phases sequentially
# Each phase can also be run independently
#
# Usage:
#   bash scripts/run_all_experiments.sh          # run all phases
#   bash scripts/run_all_experiments.sh 2        # start from phase 2
#   bash scripts/run_all_experiments.sh 1 3      # run phases 1 through 3
set -euo pipefail

START_PHASE="${1:-1}"
END_PHASE="${2:-5}"

cd "$(dirname "$0")/.."

echo "============================================================"
echo "  SIGNATE Spectrum Challenge - Full Experiment Suite"
echo "  Phases: $START_PHASE → $END_PHASE"
echo "  Started: $(date)"
echo "============================================================"
echo ""

if [ "$START_PHASE" -le 1 ] && [ "$END_PHASE" -ge 1 ]; then
  echo "[$(date +%H:%M:%S)] Starting Phase 1..."
  bash scripts/phase1_groupkfold.sh 2>&1 | tee logs/phase1.log
  echo ""
fi

if [ "$START_PHASE" -le 2 ] && [ "$END_PHASE" -ge 2 ]; then
  echo "[$(date +%H:%M:%S)] Starting Phase 2..."
  bash scripts/phase2_preprocess_sweep.sh 2>&1 | tee logs/phase2.log
  echo ""
fi

if [ "$START_PHASE" -le 3 ] && [ "$END_PHASE" -ge 3 ]; then
  echo "[$(date +%H:%M:%S)] Starting Phase 3..."
  bash scripts/phase3_wavelength_selection.sh 2>&1 | tee logs/phase3.log
  echo ""
fi

if [ "$START_PHASE" -le 4 ] && [ "$END_PHASE" -ge 4 ]; then
  echo "[$(date +%H:%M:%S)] Starting Phase 4..."
  bash scripts/phase4_model_variations.sh 2>&1 | tee logs/phase4.log
  echo ""
fi

if [ "$START_PHASE" -le 5 ] && [ "$END_PHASE" -ge 5 ]; then
  echo "[$(date +%H:%M:%S)] Starting Phase 5..."
  bash scripts/phase5_multiseed_best.sh 2>&1 | tee logs/phase5.log
  echo ""
fi

echo "============================================================"
echo "  All experiments complete! $(date)"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Review results: python scripts/analyze_results.py --top 30"
echo "  2. Edit scripts/phase6_ensemble_submit.sh with best run dirs"
echo "  3. Run: bash scripts/phase6_ensemble_submit.sh"
