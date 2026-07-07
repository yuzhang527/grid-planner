#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-.}:."

RUN_DIR="outputs/logs/strategy_A_qwen_100"
CONSOLE_LOG="outputs/logs/strategy_A_qwen_100_console.log"

mkdir -p outputs/logs
rm -rf "$RUN_DIR"

echo "========== Strategy A Qwen 100 episodes =========="
echo "RUN_DIR=$RUN_DIR"

PYTHONPATH=. python scripts/smoke_test.py \
  --config configs/exp_A_adjacent_qwen_100.yaml \
  2>&1 | tee "$CONSOLE_LOG"

echo
echo "========== Behavior analysis =========="
if [ -f scripts/analyze_strategy_A_run.py ]; then
  PYTHONPATH=. python scripts/analyze_strategy_A_run.py \
    --run "$RUN_DIR" \
    2>&1 | tee "$RUN_DIR/behavior_analysis.txt"
else
  echo "scripts/analyze_strategy_A_run.py not found, skipped."
fi

echo
echo "========== World-model analysis =========="
PYTHONPATH=. python scripts/score_world_model_A.py \
  --run "$RUN_DIR" \
  2>&1 | tee "$RUN_DIR/world_model_analysis.txt"

echo
echo "Saved:"
echo "  $RUN_DIR/steps.jsonl"
echo "  $RUN_DIR/summary.json"
echo "  $RUN_DIR/world_model_summary.json"
echo "  $RUN_DIR/world_model_step_metrics.jsonl"
