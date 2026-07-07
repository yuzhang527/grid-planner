#!/usr/bin/env bash
set -euo pipefail

REPO="/workspace/luoyuzhang/grid-planner"
cd "$REPO"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"

RUN_DIR="outputs/logs/strategy_A_qwen_5_vllm"

mkdir -p outputs/logs
rm -rf "$RUN_DIR"

python -m py_compile scripts/smoke_test_vllm_A.py
python -m py_compile scripts/check_strategy_A_vllm_outputs.py

time python scripts/smoke_test_vllm_A.py \
  --config configs/exp_A_adjacent_qwen_100_vllm.yaml \
  --num-episodes 5 \
  --output-dir "$RUN_DIR" \
  --batch-size 8 \
  --max-num-seqs 8 \
  2>&1 | tee outputs/logs/strategy_A_qwen_5_vllm_console.log

python scripts/check_strategy_A_vllm_outputs.py \
  --run "$RUN_DIR"

if [ -f scripts/score_world_model_A.py ]; then
  python scripts/score_world_model_A.py \
    --run "$RUN_DIR"
fi
