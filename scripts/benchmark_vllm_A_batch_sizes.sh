#!/usr/bin/env bash
set -euo pipefail

REPO="/workspace/luoyuzhang/grid-planner"
cd "$REPO"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"

CONFIG="${CONFIG:-configs/exp_A_adjacent_qwen_100_vllm.yaml}"
NUM_EPISODES="${NUM_EPISODES:-20}"
GPU_UTIL="${GPU_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
TP_SIZE="${TP_SIZE:-1}"

mkdir -p outputs/logs

echo "======================================================================"
echo "Strategy A vLLM batch-size benchmark"
echo "======================================================================"
echo "REPO=$REPO"
echo "CONFIG=$CONFIG"
echo "NUM_EPISODES=$NUM_EPISODES"
echo "GPU_UTIL=$GPU_UTIL"
echo "MAX_MODEL_LEN=$MAX_MODEL_LEN"
echo "TP_SIZE=$TP_SIZE"
echo "======================================================================"

python -m py_compile scripts/smoke_test_vllm_A.py
python -m py_compile scripts/check_strategy_A_vllm_outputs.py

for BS in 8 16 32 64; do
  RUN_DIR="outputs/logs/strategy_A_qwen_benchmark_bs${BS}"

  echo
  echo "######################################################################"
  echo "Running batch size: $BS"
  echo "RUN_DIR=$RUN_DIR"
  echo "######################################################################"

  rm -rf "$RUN_DIR"

  set +e
  /usr/bin/time -f "ELAPSED_SECONDS=%e" \
    env PYTHONPATH="$PYTHONPATH" \
    python scripts/smoke_test_vllm_A.py \
      --config "$CONFIG" \
      --num-episodes "$NUM_EPISODES" \
      --output-dir "$RUN_DIR" \
      --batch-size "$BS" \
      --max-num-seqs "$BS" \
      --gpu-memory-utilization "$GPU_UTIL" \
      --max-model-len "$MAX_MODEL_LEN" \
      --tensor-parallel-size "$TP_SIZE" \
      2>&1 | tee "${RUN_DIR}_console.log"

  STATUS=${PIPESTATUS[0]}
  set -e

  if [ "$STATUS" -ne 0 ]; then
    echo "Batch size $BS failed. Possibly OOM. Stop benchmark here."
    exit 0
  fi

  python scripts/check_strategy_A_vllm_outputs.py \
    --run "$RUN_DIR" \
    2>&1 | tee "$RUN_DIR/output_check.txt"

  if [ -f scripts/score_world_model_A.py ]; then
    python scripts/score_world_model_A.py \
      --run "$RUN_DIR" \
      2>&1 | tee "$RUN_DIR/world_model_analysis.txt"
  fi
done

echo
echo "Benchmark finished."
echo "Check logs under outputs/logs/strategy_A_qwen_benchmark_bs*"
