SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VISIONSELECTOR_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOCAL_LMMS_EVAL_ROOT="${VISIONSELECTOR_ROOT}/lmms-eval"

# save result path
ROOT_DIR="${ROOT_DIR:-${SCRIPT_DIR}/result/eval_time}"
mkdir -p "$ROOT_DIR"
NUM_PROCESSES=2

export PYTHONPATH="${LOCAL_LMMS_EVAL_ROOT}:${PYTHONPATH:-}"
export LMMS_DISABLE_TQDM="${LMMS_DISABLE_TQDM:-False}"
# export OPENAI_API_BASE="${OPENAI_API_BASE:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
# export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-c653b84b9d74460dbb8099eacca2a5ed}"
# export HALLUSION_BENCH_JUDGE_MODEL="${HALLUSION_BENCH_JUDGE_MODEL:-qwen3-vl-plus-2025-09-23}"
set -x

EVAL_TIME=False
BASE_COMMAND="CUDA_VISIBLE_DEVICES=2,3 python3 -m accelerate.commands.launch \
    --main_process_port=28175 \
    --mixed_precision=bf16 \
    --num_processes=$NUM_PROCESSES \
    -m lmms_eval \
    --model qwen2_5_vl_with_token_compression \
    --batch_size 1 \
    --log_samples"

# first for method name ; second for FILENAME ; third for additional args
METHODS=(
    # Qwen2-5-VL token compression method for example
    "orig new_preprocess attn_implementation="flash_attention_2""  
     "fastv fastv attn_implementation="flash_attention_2""
     "prumerge+ prumerge+ attn_implementation="flash_attention_2""
     "visionzip_official visionzip_official attn_implementation="flash_attention_2""
     "dart dart attn_implementation="flash_attention_2""
     "divprune divprune attn_implementation="flash_attention_2""
)


# budgets
BUDGETS=(0.1 0.2 0.3)

# model path
MODEL_PATH="${MODEL_PATH:-/mnt/nvme/fjh/Qwen2.5-VL-7B-Instruct}"
MODEL_NAME="qwen25vl_7b"

TASKS=("realworldqa" "vstar_bench")

declare -A TASK_YAMLS=(
    ["ocrbench"]="${LOCAL_LMMS_EVAL_ROOT}/lmms_eval/tasks/ocrbench/ocrbench.yaml"
    ["ai2d_no_mask"]="${LOCAL_LMMS_EVAL_ROOT}/lmms_eval/tasks/ai2d/ai2d_no_mask.yaml"
    ["mme"]="${LOCAL_LMMS_EVAL_ROOT}/lmms_eval/tasks/mme/mme.yaml"
    ["gqa"]="${LOCAL_LMMS_EVAL_ROOT}/lmms_eval/tasks/gqa/gqa.yaml"
    ["chartqa"]="${LOCAL_LMMS_EVAL_ROOT}/lmms_eval/tasks/chartqa/chartqa.yaml"
)

BACKUP_DIR="$(mktemp -d)"

cleanup() {
    for task in "${!TASK_YAMLS[@]}"; do
        yaml_path="${TASK_YAMLS[$task]}"
        backup_path="${BACKUP_DIR}/${task}.yaml"
        if [[ -f "${backup_path}" ]]; then
            cp "${backup_path}" "${yaml_path}"
        fi
    done
    rm -rf "${BACKUP_DIR}"
}
trap cleanup EXIT

for task in "${!TASK_YAMLS[@]}"; do
    cp "${TASK_YAMLS[$task]}" "${BACKUP_DIR}/${task}.yaml"
done

set_task_yaml_original() {
    local yaml_path="$1"

    python3 - "${yaml_path}" <<'PY'
from pathlib import Path
import re
import sys

yaml_path = Path(sys.argv[1])
text = yaml_path.read_text(encoding="utf-8")

if re.search(r'(?m)^    text_perturb:', text):
    text = re.sub(r'(?m)^    text_perturb:.*$', '    text_perturb: false', text, count=1)

if re.search(r'(?m)^    perturbation_name:', text):
    text = re.sub(r'(?m)^    perturbation_name:.*$', '    perturbation_name: "perturbations_0"', text, count=1)

yaml_path.write_text(text, encoding="utf-8")
PY
}


for TASK in "${TASKS[@]}"; do
    for METHOD_CONFIG in "${METHODS[@]}"; do
        METHOD=$(echo "$METHOD_CONFIG" | awk '{print $1}')
        FILENAME=$(echo "$METHOD_CONFIG" | awk '{print $2}')
        ADDITIONAL_ARGS=$(echo "$METHOD_CONFIG" | awk '{$1="";$2=""; print $0}' | xargs)

        for BUDGET in "${BUDGETS[@]}"; do
            MODEL_TEXT_PERTURB="True"
            OUTPUT_SUFFIX=""
            if [[ -n "${TASK_YAMLS[$TASK]:-}" ]]; then
                set_task_yaml_original "${TASK_YAMLS[$TASK]}"
                MODEL_TEXT_PERTURB="False"
                OUTPUT_SUFFIX="_perturbations_0"
            fi

            OUTPUT_PATH="$ROOT_DIR/${TASK}/${BUDGET}/${MODEL_NAME}_${METHOD}_${BUDGET}_${FILENAME}${OUTPUT_SUFFIX}"
            LOG_FILE="${OUTPUT_PATH}/log_time.log"

            if [ -d "$OUTPUT_PATH" ]; then
                echo "folder exists, skip."
                continue
            fi

            mkdir -p "$OUTPUT_PATH"

            MODEL_ARGS="pretrained=${MODEL_PATH},method=${METHOD},budgets=${BUDGET},eval_time=${EVAL_TIME},timing_output_dir=${OUTPUT_PATH},text_perturb=${MODEL_TEXT_PERTURB},${ADDITIONAL_ARGS}"

            COMMAND="${BASE_COMMAND} --tasks ${TASK} --output_path ${OUTPUT_PATH} --log_samples_suffix ${TASK} --model_args \"${MODEL_ARGS}\""
            
            {
                eval "${COMMAND}"
                if [[ "${EVAL_TIME,,}" == "true" ]]; then
                    echo "===== Summary ====="
                    RESULT_JSON=$(find "${OUTPUT_PATH}" -name '*_results.json' | head -n 1)
                    shopt -s nullglob
                    TIMING_FILES=("${OUTPUT_PATH}"/timing_records_rank*.log)
                    shopt -u nullglob
                    if (( ${#TIMING_FILES[@]} == 0 )); then
                        echo "No timing records found."
                    else
                        TIMING_LOG_FILE="${OUTPUT_PATH}/timing_records_all.log"
                        cat "${TIMING_FILES[@]}" > "${TIMING_LOG_FILE}"
                        python3 "${SCRIPT_DIR}/extract_time.py" --path "${TIMING_LOG_FILE}" --results_json "${RESULT_JSON}"
                    fi
                fi

            } 2>&1 | tee "${LOG_FILE}"

        done
    done
done
