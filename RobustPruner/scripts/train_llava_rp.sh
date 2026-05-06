#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export WANDB_DISABLED=true
export HF_HOME=/mnt/nvme/fjh/hf_cache
export HF_DATASETS_CACHE=/mnt/nvme/fjh/hf_cache/datasets
export DATASETS_CACHE=/mnt/nvme/fjh/hf_cache/datasets
export TRANSFORMERS_CACHE=/mnt/nvme/fjh/hf_cache/transformers
export XDG_CACHE_HOME=/mnt/nvme/fjh/hf_cache
export TMPDIR=/mnt/nvme/fjh/hf_cache/tmp
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$TMPDIR"

CUDA_VISIBLE_DEVICES=3 accelerate launch \
  --num_processes 1 \
  --mixed_precision bf16 \
  --main_process_port 12345 \
  train_llava_rp.py \
  --config train_configs/llava1_5_7b_rp/llava1_5_7b_rp.yaml

