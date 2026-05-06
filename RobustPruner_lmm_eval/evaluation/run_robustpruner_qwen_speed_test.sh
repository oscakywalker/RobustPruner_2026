#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${EVAL_ROOT}/.." && pwd)"
LOCAL_LMMS_EVAL_ROOT="${EVAL_ROOT}/lmms-eval"
ROBUSTPRUNER_ROOT="${ROBUSTPRUNER_ROOT:-${WORKSPACE_ROOT}/RobustPruner}"

###########################################
# Load my_lmms_eval plugin (CRITICAL)
###########################################
export PYTHONPATH="${LOCAL_LMMS_EVAL_ROOT}:${ROBUSTPRUNER_ROOT}:${PYTHONPATH:-}"
export LMMS_EVAL_PLUGINS="my_lmms_eval"
export EVAL_TIME="${EVAL_TIME:-True}"

###########################################
# GPU config (MANUAL, SIMPLE)
###########################################
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"

echo "Using GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "NUM_PROCESSES: ${NUM_PROCESSES}"

###########################################
# Basic Config
###########################################
set -e

MODEL_PATH="${MODEL_PATH:-/path/to/Qwen2.5-VL-7B-Instruct}"
MODEL_NAME="qwen25vl_7b"
RP_MODULES_DIR="/RobustPruner_2026/RobustPruner/output/qwen2_5_7b_rp_uncertainty_stage1_unbind_layer12_newest_testy"
ROOT_DIR="${ROOT_DIR:-${SCRIPT_DIR}/result_robustpruner_qwen}"
TASKS=("docvqa_val" "textvqa_val" "scienceqa_img" "ai2d_no_mask" "mmmu_val" "mme" "pope")
BUDGETS=("0.2")
MASTER_PORT=28176

# Dual-ranking hyperparameters:
# 1) first expand the base-score candidate pool to budget * a
# 2) then keep the lowest 1/a uncertainty tokens inside that pool
BASE_SCORE_POOL_MULTIPLIER="1.1"

###########################################
# Main Loop
###########################################
for TASK in "${TASKS[@]}"; do
  for BUDGET in "${BUDGETS[@]}"; do
    BASE_SCORE_POOL_RATIO=$(python3 -c "b=float('${BUDGET}'); a=float('${BASE_SCORE_POOL_MULTIPLIER}'); print(f'{b*a:.8f}')")
    OUTPUT_PATH="${ROOT_DIR}/${TASK}/${BUDGET}/${MODEL_NAME}_robustpruner_${BUDGET}"
    LOG_SUFFIX="${TASK}"
    LOG_FILE="${OUTPUT_PATH}/log_time.log"

    rm -rf "${OUTPUT_PATH}"
    mkdir -p "${OUTPUT_PATH}"
    mkdir -p "${OUTPUT_PATH}/token_scores"

    echo "============================================"
    echo "Task      : ${TASK}"
    echo "Budget    : ${BUDGET}"
    echo "Base pool : ${BASE_SCORE_POOL_RATIO} (= budget * ${BASE_SCORE_POOL_MULTIPLIER})"
    echo "Unc keep  : $(python3 -c "a=float('${BASE_SCORE_POOL_MULTIPLIER}'); print(f'{1.0/a:.8f}')") (= 1 / ${BASE_SCORE_POOL_MULTIPLIER})"
    echo "GPUs      : ${CUDA_VISIBLE_DEVICES}"
    echo "Processes : ${NUM_PROCESSES}"
    echo "Output    : ${OUTPUT_PATH}"
    echo "============================================"

    accelerate launch \
        --num_processes=${NUM_PROCESSES} \
        --main_process_port=${MASTER_PORT} \
        --mixed_precision=bf16 \
        -m lmms_eval \
        --model qwen2_5_vl_rp \
        --model_args "pretrained=${MODEL_PATH},new_modules_dir=${RP_MODULES_DIR},max_remain_ratio=${BUDGET},base_score_pool_multiplier=${BASE_SCORE_POOL_MULTIPLIER},attn_implementation=flash_attention_2,timing_output_dir=${OUTPUT_PATH},token_scores_output_dir=${OUTPUT_PATH}/token_scores,save_token_scores=True" \
        --tasks ${TASK} \
        --batch_size 1 \
        --output_path "${OUTPUT_PATH}" \
        --log_samples \
        --log_samples_suffix "${LOG_SUFFIX}" \
        --verbosity INFO \
        2>&1 | tee "${LOG_FILE}"
  done
done

echo "All done."
