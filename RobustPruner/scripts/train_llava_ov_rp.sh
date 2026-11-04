#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export WANDB_DISABLED=true
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export DATASETS_CACHE="${DATASETS_CACHE:-${HF_DATASETS_CACHE}}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PROJECT_ROOT}/.cache}"
export TMPDIR="${TMPDIR:-${PROJECT_ROOT}/.cache/tmp}"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$TMPDIR"

ngpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ ! -z "$CUDA_VISIBLE_DEVICES" ]; then
    ngpus=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
fi

port=${PORT:-12345}
config=${1:-train_configs/llava_ov_1_5_8b_rp/llava_ov_1_5_8b_rp.yaml}

echo "Number of GPUs: $ngpus"
echo "Running LLaVA-OneVision RP training with config: $config"

accelerate launch \
    --num_processes "$ngpus" \
    --mixed_precision bf16 \
    --main_process_port "$port" \
    train_llava_ov_rp.py \
    --config "$config"
