#!/usr/bin/env bash
set -euo pipefail

REPO="/workspace/luoyuzhang/grid-planner"
cd "$REPO"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"

BASE_CONFIG="${BASE_CONFIG:-configs/exp_A_adjacent_qwen_100_vllm.yaml}"
ROOT_DIR="${ROOT_DIR:-outputs/logs/strategy_A_qwen_100_vllm_4gpu}"
MERGED_DIR="${MERGED_DIR:-outputs/logs/strategy_A_qwen_100_vllm_4gpu_merged}"

GPUS="${GPUS:-0,1,2,3}"
TOTAL_EPISODES="${TOTAL_EPISODES:-100}"
BASE_SEED="${BASE_SEED:-123}"

BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-32}"
MAX_NUM_SEQS_PER_GPU="${MAX_NUM_SEQS_PER_GPU:-32}"
GPU_UTIL="${GPU_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS="${#GPU_ARRAY[@]}"

if [ "$NUM_GPUS" -lt 1 ]; then
  echo "No GPUs specified. Set GPUS=0,1,2,3"
  exit 1
fi

echo "======================================================================"
echo "Strategy A Qwen vLLM 4-GPU data parallel rollout"
echo "======================================================================"
echo "REPO=$REPO"
echo "BASE_CONFIG=$BASE_CONFIG"
echo "ROOT_DIR=$ROOT_DIR"
echo "MERGED_DIR=$MERGED_DIR"
echo "GPUS=$GPUS"
echo "NUM_GPUS=$NUM_GPUS"
echo "TOTAL_EPISODES=$TOTAL_EPISODES"
echo "BASE_SEED=$BASE_SEED"
echo "BATCH_SIZE_PER_GPU=$BATCH_SIZE_PER_GPU"
echo "MAX_NUM_SEQS_PER_GPU=$MAX_NUM_SEQS_PER_GPU"
echo "GPU_UTIL=$GPU_UTIL"
echo "MAX_MODEL_LEN=$MAX_MODEL_LEN"
echo "======================================================================"

python -m py_compile scripts/smoke_test_vllm_A.py
python -m py_compile scripts/check_strategy_A_vllm_outputs.py
python -m py_compile scripts/merge_strategy_A_vllm_shards.py

rm -rf "$ROOT_DIR" "$MERGED_DIR"
mkdir -p "$ROOT_DIR" configs/vllm_shards outputs/logs

python - <<PY
from pathlib import Path
import yaml

base_config_path = Path("$BASE_CONFIG")
cfg = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))

total = int("$TOTAL_EPISODES")
num_gpus = int("$NUM_GPUS")
base_seed = int("$BASE_SEED")

base = total // num_gpus
rem = total % num_gpus

start_seed = base_seed

for shard_id in range(num_gpus):
    n = base + (1 if shard_id < rem else 0)

    shard_cfg = dict(cfg)
    shard_cfg["seed"] = start_seed
    shard_cfg["num_episodes"] = n
    shard_cfg["tensor_parallel_size"] = 1
    shard_cfg["gpu_memory_utilization"] = float("$GPU_UTIL")
    shard_cfg["max_model_len"] = int("$MAX_MODEL_LEN")
    shard_cfg["max_num_seqs"] = int("$MAX_NUM_SEQS_PER_GPU")
    shard_cfg["batch_size"] = int("$BATCH_SIZE_PER_GPU")
    shard_cfg["output_dir"] = f"$ROOT_DIR/shard_{shard_id}"

    out = Path(f"configs/vllm_shards/exp_A_qwen_vllm_shard_{shard_id}.yaml")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(shard_cfg, sort_keys=False), encoding="utf-8")

    print(
        f"shard_{shard_id}: "
        f"seed={start_seed}, episodes={n}, config={out}"
    )

    start_seed += n
PY

PIDS=()
SHARD_IDS=()

for IDX in "${!GPU_ARRAY[@]}"; do
  GPU_ID="${GPU_ARRAY[$IDX]}"
  SHARD_CONFIG="configs/vllm_shards/exp_A_qwen_vllm_shard_${IDX}.yaml"
  SHARD_DIR="$ROOT_DIR/shard_${IDX}"
  SHARD_LOG="$ROOT_DIR/shard_${IDX}_console.log"

  echo
  echo "######################################################################"
  echo "Launching shard_$IDX on GPU $GPU_ID"
  echo "SHARD_CONFIG=$SHARD_CONFIG"
  echo "SHARD_DIR=$SHARD_DIR"
  echo "######################################################################"

  (
    set -euo pipefail
    cd "$REPO"
    export PYTHONPATH="$REPO:${PYTHONPATH:-}"
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    export VLLM_WORKER_MULTIPROC_METHOD=spawn

    python scripts/smoke_test_vllm_A.py \
      --config "$SHARD_CONFIG" \
      --output-dir "$SHARD_DIR" \
      --batch-size "$BATCH_SIZE_PER_GPU" \
      --max-num-seqs "$MAX_NUM_SEQS_PER_GPU" \
      --gpu-memory-utilization "$GPU_UTIL" \
      --max-model-len "$MAX_MODEL_LEN" \
      --tensor-parallel-size 1 \
      2>&1 | tee "$SHARD_LOG"

    python scripts/check_strategy_A_vllm_outputs.py \
      --run "$SHARD_DIR" \
      2>&1 | tee "$SHARD_DIR/output_check.txt"
  ) &

  PIDS+=("$!")
  SHARD_IDS+=("$IDX")
done

FAILED=0

for I in "${!PIDS[@]}"; do
  PID="${PIDS[$I]}"
  SHARD_ID="${SHARD_IDS[$I]}"

  if wait "$PID"; then
    echo "shard_$SHARD_ID finished successfully."
  else
    echo "ERROR: shard_$SHARD_ID failed."
    FAILED=1
  fi
done

if [ "$FAILED" -ne 0 ]; then
  echo "At least one shard failed. Check logs under $ROOT_DIR"
  exit 1
fi

echo
echo "======================================================================"
echo "Merging shards"
echo "======================================================================"

python scripts/merge_strategy_A_vllm_shards.py \
  --root "$ROOT_DIR" \
  --out "$MERGED_DIR" \
  2>&1 | tee "$ROOT_DIR/merge.log"

echo
echo "======================================================================"
echo "Checking merged output"
echo "======================================================================"

python scripts/check_strategy_A_vllm_outputs.py \
  --run "$MERGED_DIR" \
  2>&1 | tee "$MERGED_DIR/output_check.txt"

echo
echo "======================================================================"
echo "Behavior analysis on merged output"
echo "======================================================================"

if [ -f scripts/analyze_strategy_A_run.py ]; then
  python scripts/analyze_strategy_A_run.py \
    --run "$MERGED_DIR" \
    2>&1 | tee "$MERGED_DIR/behavior_analysis.txt"
else
  echo "scripts/analyze_strategy_A_run.py not found, skipped."
fi

echo
echo "======================================================================"
echo "World-model analysis on merged output"
echo "======================================================================"

if [ -f scripts/score_world_model_A.py ]; then
  python scripts/score_world_model_A.py \
    --run "$MERGED_DIR" \
    2>&1 | tee "$MERGED_DIR/world_model_analysis.txt"
else
  echo "scripts/score_world_model_A.py not found, skipped."
fi

echo
echo "======================================================================"
echo "Done"
echo "======================================================================"
echo "Shard root:"
echo "  $ROOT_DIR"
echo "Merged output:"
echo "  $MERGED_DIR"
echo
echo "Main merged files:"
echo "  $MERGED_DIR/steps.jsonl"
echo "  $MERGED_DIR/summary.json"
echo "  $MERGED_DIR/output_check.txt"
echo "  $MERGED_DIR/world_model_summary.json"
echo "  $MERGED_DIR/world_model_step_metrics.jsonl"

if [ -f "$MERGED_DIR/world_model_summary.json" ]; then
  echo
  echo "world_model_summary.json:"
  cat "$MERGED_DIR/world_model_summary.json"
fi
