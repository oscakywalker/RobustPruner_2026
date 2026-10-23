#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${EVAL_ROOT}/.." && pwd)"
ROBUSTPRUNER_ROOT="${ROBUSTPRUNER_ROOT:-${WORKSPACE_ROOT}/RobustPruner}"

export PYTHONPATH="${EVAL_ROOT}/lmms-eval:${ROBUSTPRUNER_ROOT}:${PYTHONPATH:-}"
export LMMS_EVAL_PLUGINS="my_lmms_eval"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export GP_DPP_PROJECTED_MARGINAL="${GP_DPP_PROJECTED_MARGINAL:-True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
RP_MODULES_DIR="${RP_MODULES_DIR:-${ROBUSTPRUNER_ROOT}/output/qwen2_5_7b_rp_uncertainty_stage1_unbind_layer12_newest_testy}"
ROOT_DIR="${ROOT_DIR:-${SCRIPT_DIR}/result_robustpruner_qwen_dpp_marginal}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
MASTER_PORT="${MASTER_PORT:-29531}"
BASE_SCORE_POOL_MULTIPLIER="${BASE_SCORE_POOL_MULTIPLIER:-1.1}"
read -r -a TASK_LIST <<< "${TASKS:-gqa}"
read -r -a BUDGET_LIST <<< "${BUDGETS:-0.1}"

for TASK in "${TASK_LIST[@]}"; do
  for BUDGET in "${BUDGET_LIST[@]}"; do
    OUTPUT_PATH="${ROOT_DIR}/${TASK}/${BUDGET}"
    mkdir -p "${OUTPUT_PATH}"
    echo "Task=${TASK}; retention=${BUDGET}; projected marginal=${GP_DPP_PROJECTED_MARGINAL}"

    accelerate launch \
      --num_processes="${NUM_PROCESSES}" \
      --main_process_port="${MASTER_PORT}" \
      --mixed_precision=bf16 \
      -m lmms_eval \
      --model qwen2_5_vl_rp_dpp_marginal \
      --model_args "pretrained=${MODEL_PATH},new_modules_dir=${RP_MODULES_DIR},max_remain_ratio=${BUDGET},base_score_pool_multiplier=${BASE_SCORE_POOL_MULTIPLIER},attn_implementation=flash_attention_2,text_perturb=False" \
      --tasks "${TASK}" \
      --batch_size 1 \
      --output_path "${OUTPUT_PATH}" \
      --log_samples \
      --verbosity INFO \
      2>&1 | tee "${OUTPUT_PATH}/eval.log"
  done
done
