#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export HF_HOME=/RobustPruner_2026/hf_cache
export HF_DATASETS_CACHE=/RobustPruner_2026/hf_cache/datasets
export TRANSFORMERS_CACHE=/RobustPruner_2026/hf_cache/transformers
export XDG_CACHE_HOME=/RobustPruner_2026/hf_cache
export TMPDIR=/RobustPruner_2026/hf_cache/tmp
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$TMPDIR"

ngpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    ngpus=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
fi

echo "Number of GPUs: $ngpus"
port=${PORT:-12345}

# Stage 1: train on fixed 20k set (gqa_rand_seed42_20000.jsonl)
# with uncertainty-aware penalty-head supervision.
accelerate launch \
    --num_processes "$ngpus" \
    --main_process_port "$port" \
    train_qwen_rp.py \
    --config train_configs/qwen2_5_7b_rp/qwen2_5_7b_rp_uncertainty_stage1.yaml
