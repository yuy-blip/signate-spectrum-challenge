#!/usr/bin/env bash
# ULTIMATE RUN: Execute Phases 7-9 (advanced experiments)
#
# Phase 7: LightGBM deep tuning + feature engineering (~40 experiments)
# Phase 8: Multi-seed LightGBM for ensemble diversity (~25 experiments)
# Phase 9: Pseudo-labeling experiments (~15 experiments)
#
# After this completes:
#   1. python scripts/analyze_results.py --top 30
#   2. Edit scripts/phase10_final_ensemble.sh with best runs
#   3. bash scripts/phase10_final_ensemble.sh
#
# Usage:
#   bash scripts/run_ultimate.sh          # run phases 7-9
#   bash scripts/run_ultimate.sh 8        # start from phase 8
set -euo pipefail

START="${1:-7}"
cd "$(dirname "$0")/.."
mkdir -p logs

echo "============================================================"
echo "  ULTIMATE EXPERIMENT SUITE"
echo "  Starting from Phase $START"
echo "  $(date)"
echo "============================================================"

if [ "$START" -le 7 ]; then
  echo "[$(date +%H:%M:%S)] Phase 7: Advanced LightGBM & Features..."
  bash scripts/phase7_advanced.sh 2>&1 | tee logs/phase7.log
fi

if [ "$START" -le 8 ]; then
  echo "[$(date +%H:%M:%S)] Phase 8: Multi-seed LightGBM..."
  bash scripts/phase8_multiseed_lgbm.sh 2>&1 | tee logs/phase8.log
fi

if [ "$START" -le 9 ]; then
  echo "[$(date +%H:%M:%S)] Phase 9: Pseudo-labeling..."
  bash scripts/phase9_pseudo_label.sh 2>&1 | tee logs/phase9.log
fi

echo ""
echo "============================================================"
echo "  DONE! $(date)"
echo "============================================================"
echo ""
echo "Next:"
echo "  1. python scripts/analyze_results.py --top 30"
echo "  2. Edit scripts/phase10_final_ensemble.sh"
echo "  3. bash scripts/phase10_final_ensemble.sh"
