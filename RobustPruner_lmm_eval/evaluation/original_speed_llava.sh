#!/bin/bash

###########################################
# Load my_lmms_eval plugin (CRITICAL)
###########################################
export PYTHONPATH="/home/fjh/VisionSelector/lmms-eval:/home/fjh/GlimpsePrune/llava_vscan:/home/fjh/GlimpsePrune:$PYTHONPATH"
export LMMS_EVAL_PLUGINS="my_lmms_eval"
export PYTHONNOUSERSITE=1

###########################################
# (Recommended) run from lmms-eval root
###########################################
cd /home/fjh/VisionSelector/lmms-eval || exit 1

###########################################
# Detect GPU Count
###########################################
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    ngpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    ngpus=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
fi
echo "GPUs detected: $ngpus"

###########################################
# Auto-detect perturbations version from GQA utils.py
###########################################
UTILS_PATH="/home/fjh/VisionSelector/lmms-eval/lmms_eval/tasks/gqa/utils.py"
if [[ ! -f "$UTILS_PATH" ]]; then
    echo "WARNING: utils.py not found at $UTILS_PATH"
    echo "Using default: text_perturbations_0"
    PERTURB_VERSION="0"
else
    PERTURB_VERSION=$(grep -oP 'text_perturbations_\K\d+' "$UTILS_PATH" | head -1)
    if [[ -z "$PERTURB_VERSION" ]]; then
        echo "WARNING: Cannot detect perturbations version from utils.py"
        echo "Using default: text_perturbations_0"
        PERTURB_VERSION="0"
    else
        echo "Auto-detected perturbations version: text_perturbations_${PERTURB_VERSION}.json"
    fi
fi

###########################################
# Basic Config
###########################################
set -e

# VSCAN model path.
MODEL_PATH="/mnt/nvme/fjh/llava-v1.5-7b"
MODEL_NAME="llava_v1_5_7b"

# Output root directory.
ROOT_DIR="/mnt/nvme/fjh/llava-evaluation/vscan_ablation_results"

# Task configuration.
#TASKS=("gqa" "mme" "ai2d" "vizwiz_vqa")
#PERTURB_NAME="text_perturb"
# Fixed retained visual token counts and corresponding ratios, matched by index.
#KEEP_TOKENS=("87" "173" "260")
#KEEP_RATIOS=("0.1" "0.2" "0.3")
#LAYER_LIST=${LAYER_LIST:-"[21]"}
#IMAGE_TOKEN_LISTS=("58" "115" "173")
# Adaptive switches (grid)
#STAGE1_ADAPTIVE_LIST=(${STAGE1_ADAPTIVE_LIST:-false true})
#STAGE2_ADAPTIVE_LIST=(${STAGE2_ADAPTIVE_LIST:-false true})

# Ablation
TASKS=("mme")
PERTURB_NAME="text_perturb"
# Fixed retained visual token counts and corresponding ratios, matched by index.
KEEP_TOKENS=("576")
KEEP_RATIOS=("1")
LAYER_LIST=${LAYER_LIST:-"[21]"}
IMAGE_TOKEN_LISTS=("576")
# Adaptive switches (grid)
STAGE1_ADAPTIVE_LIST=(${STAGE1_ADAPTIVE_LIST:-false})
STAGE2_ADAPTIVE_LIST=(${STAGE2_ADAPTIVE_LIST:-false})
# Alpha grids
VISION_ALPHAS=(${VISION_ALPHAS:-0})
LLM_ALPHAS=(${LLM_ALPHAS:-0})
# Configuration for multiple GPUs.
NUM_PROCESSES=${ngpus}
MASTER_PORT=29523
EVAL_TIME=false

###########################################
# Main Loop
###########################################
for TASK in "${TASKS[@]}"; do
  for i in "${!KEEP_TOKENS[@]}"; do

    KEEP=${KEEP_TOKENS[$i]}
    RATIO=${KEEP_RATIOS[$i]}

    for STAGE1_ADAPTIVE in "${STAGE1_ADAPTIVE_LIST[@]}"; do
      for STAGE2_ADAPTIVE in "${STAGE2_ADAPTIVE_LIST[@]}"; do
        for VISION_ALPHA in "${VISION_ALPHAS[@]}"; do
          for LLM_ALPHA in "${LLM_ALPHAS[@]}"; do

        RATIO_LABEL="${RATIO}"
        KEEP_LABEL="${KEEP}"
        IMAGE_TOKEN_LIST=${IMAGE_TOKEN_LISTS[$i]}
        OUTPUT_PATH="${ROOT_DIR}/${TASK}/${RATIO_LABEL}/vscan_${KEEP_LABEL}/stage1_adaptive_${STAGE1_ADAPTIVE}_stage2_adaptive_${STAGE2_ADAPTIVE}/vision_alpha_${VISION_ALPHA}_llm_alpha_${LLM_ALPHA}/${MODEL_NAME}_vscan_text_perturbations_${PERTURB_VERSION}"
        LOG_FILE="${OUTPUT_PATH}/log_time.log"

        if [ -d "${OUTPUT_PATH}" ]; then
            echo "Output folder exists, removing: ${OUTPUT_PATH}"
            rm -rf "${OUTPUT_PATH}"
        fi
        mkdir -p "${OUTPUT_PATH}"

        echo "================================================"
        echo "Running LLaVA-VSCAN + TEXT PERTURB (Multi-GPU)"
        echo "Task           : ${TASK}"
        echo "Keep tokens    : ${KEEP}"
        echo "Keep ratio     : ${RATIO}"
        echo "Stage1 adaptive: ${STAGE1_ADAPTIVE}"
        echo "Stage2 adaptive: ${STAGE2_ADAPTIVE}"
        echo "Vision alpha   : ${VISION_ALPHA}"
        echo "LLM alpha      : ${LLM_ALPHA}"
        echo "Perturbations  : text_perturbations_${PERTURB_VERSION}.json"
        echo "GPUs           : ${NUM_PROCESSES}"
        echo "Output path    : ${OUTPUT_PATH}"
        echo "================================================"

        MODEL_ARGS="pretrained=${MODEL_PATH},visual_token_num=${KEEP},attn_implementation=flash_attention_2,text_perturb=True,save_token_scores=False,stage1_adaptive=${STAGE1_ADAPTIVE},stage2_adaptive=${STAGE2_ADAPTIVE},vision_uncertainty_alpha=${VISION_ALPHA},uncertainty_alpha=${LLM_ALPHA}"
        if [[ -n "${LAYER_LIST}" && -n "${IMAGE_TOKEN_LIST}" ]]; then
            MODEL_ARGS="${MODEL_ARGS},llm_pruning=True,layer_list=${LAYER_LIST},image_token_list=${IMAGE_TOKEN_LIST}"
            echo "Stage2 enabled: layer_list=${LAYER_LIST}, image_token_list=${IMAGE_TOKEN_LIST}"
        else
            echo "Stage2 disabled (missing layer_list/image_token_list)."
        fi
        accelerate launch \
            --num_processes=${NUM_PROCESSES} \
            --main_process_port=${MASTER_PORT} \
            --mixed_precision=bf16 \
            -m lmms_eval \
            --model llava_vscan_execution \
            --model_args "${MODEL_ARGS}" \
            --tasks ${TASK} \
            --batch_size 1 \
            --output_path "${OUTPUT_PATH}" \
            --log_samples \
            --log_samples_suffix "${TASK}" \
            --verbosity DEBUG \
            2>&1 | tee "${LOG_FILE}"

        if [[ "${EVAL_TIME,,}" == "true" ]]; then
            echo "===== Time Summary ====="
            python3 extract_time.py --path "${LOG_FILE}"
        fi
          done
        done
      done
    done

  done

done

echo "All done! Results saved to:"
echo "${ROOT_DIR}/${TASKS[0]}/${RATIO_LABEL}/vscan_${KEEP_LABEL}/vision_alpha_*_llm_alpha_*/${MODEL_NAME}_vscan_*_text_perturbations_${PERTURB_VERSION}/"