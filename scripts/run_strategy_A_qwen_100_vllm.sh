#!/usr/bin/env bash
set -euo pipefail

REPO="/workspace/luoyuzhang/grid-planner"
cd "$REPO"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"

CONFIG="${CONFIG:-configs/exp_A_adjacent_qwen_100_vllm.yaml}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
RUN_DIR="${RUN_DIR:-outputs/logs/strategy_A_qwen_100_vllm}"
GPU_UTIL="${GPU_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
TP_SIZE="${TP_SIZE:-1}"

mkdir -p outputs/logs
rm -rf "$RUN_DIR"

echo "======================================================================"
echo "Strategy A Qwen vLLM rollout"
echo "======================================================================"
echo "REPO=$REPO"
echo "CONFIG=$CONFIG"
echo "RUN_DIR=$RUN_DIR"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "MAX_NUM_SEQS=$MAX_NUM_SEQS"
echo "GPU_UTIL=$GPU_UTIL"
echo "MAX_MODEL_LEN=$MAX_MODEL_LEN"
echo "TP_SIZE=$TP_SIZE"
echo "======================================================================"

python -m py_compile scripts/smoke_test_vllm_A.py
python -m py_compile scripts/check_strategy_A_vllm_outputs.py

time python scripts/smoke_test_vllm_A.py \
  --config "$CONFIG" \
  --output-dir "$RUN_DIR" \
  --batch-size "$BATCH_SIZE" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --tensor-parallel-size "$TP_SIZE" \
  2>&1 | tee "${RUN_DIR}_console.log"

echo
echo "======================================================================"
echo "Check output files"
echo "======================================================================"

python scripts/check_strategy_A_vllm_outputs.py \
  --run "$RUN_DIR" \
  2>&1 | tee "$RUN_DIR/output_check.txt"

echo
echo "======================================================================"
echo "Behavior analysis"
echo "======================================================================"

if [ -f scripts/analyze_strategy_A_run.py ]; then
  python scripts/analyze_strategy_A_run.py \
    --run "$RUN_DIR" \
    2>&1 | tee "$RUN_DIR/behavior_analysis.txt"
else
  echo "scripts/analyze_strategy_A_run.py not found, skipped."
fi

echo
echo "======================================================================"
echo "World-model analysis"
echo "======================================================================"

if [ -f scripts/score_world_model_A.py ]; then
  python scripts/score_world_model_A.py \
    --run "$RUN_DIR" \
    2>&1 | tee "$RUN_DIR/world_model_analysis.txt"
else
  echo "scripts/score_world_model_A.py not found, skipped."
fi

echo
echo "======================================================================"
echo "Done"
echo "======================================================================"
echo "Main files:"
echo "  $RUN_DIR/steps.jsonl"
echo "  $RUN_DIR/summary.json"
echo "  $RUN_DIR/output_check.txt"
echo "  $RUN_DIR/behavior_analysis.txt"
echo "  $RUN_DIR/world_model_analysis.txt"

if [ -f "$RUN_DIR/world_model_summary.json" ]; then
  echo
  echo "world_model_summary.json:"
  cat "$RUN_DIR/world_model_summary.json"
fi
