#!/usr/bin/env bash
# ULTIMATE RUN: Cross-species generalization strategy
#
# Phase 7:  PLS/Ridge/LightGBM × preprocessing × hyperparams (ALL GKF)
# Phase 7b: Extreme push — new models, EMSC, feature eng, target transforms
# Phase 9:  Final ensemble & submission
#
# Usage:
#   bash scripts/run_ultimate.sh          # phases 7 + 7b
#   bash scripts/run_ultimate.sh 7b       # start from phase 7b only
set -euo pipefail

START="${1:-7}"
cd "$(dirname "$0")/.."
mkdir -p logs

echo "============================================================"
echo "  CROSS-SPECIES GENERALIZATION SUITE (ALL GKF)"
echo "  Train: 13 species → Test: 6 UNSEEN species"
echo "  Starting from Phase $START"
echo "  $(date)"
echo "============================================================"

if [ "$START" = "7" ]; then
  echo "[$(date +%H:%M:%S)] Phase 7: Preprocessing & model sweep (GKF)..."
  bash scripts/phase7_advanced.sh 2>&1 | tee logs/phase7.log
fi

if [ "$START" = "7" ] || [ "$START" = "7b" ]; then
  echo "[$(date +%H:%M:%S)] Phase 7b: Extreme optimization..."
  bash scripts/phase7b_extreme.sh 2>&1 | tee logs/phase7b.log
fi

echo ""
echo "============================================================"
echo "  DONE! $(date)"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. python scripts/analyze_results.py --top 50"
echo "  2. Edit scripts/phase9_final_ensemble.sh with best runs"
echo "  3. bash scripts/phase9_final_ensemble.sh"
