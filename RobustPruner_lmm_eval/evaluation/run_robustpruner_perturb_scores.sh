#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VISIONSELECTOR_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${VISIONSELECTOR_ROOT}/.." && pwd)"
LOCAL_LMMS_EVAL_ROOT="${VISIONSELECTOR_ROOT}/lmms-eval"
ROBUSTPRUNER_ROOT="${ROBUSTPRUNER_ROOT:-/RobustPruner_2026/RobustPruner}"

export PYTHONPATH="${LOCAL_LMMS_EVAL_ROOT}:${ROBUSTPRUNER_ROOT}:${PYTHONPATH:-}"
export LMMS_EVAL_PLUGINS="my_lmms_eval"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
MODEL_NAME="qwen25vl_7b"
RP_MODULES_DIR="${RP_MODULES_DIR:-/RobustPruner_2026/RobustPruner/output/qwen2_5_7b_rp_uncertainty_stage1_unbind_layer12_newest_testy}"
ROOT_DIR="${ROOT_DIR:-/RobustPruner_2026/RobustPruner_lmm_eval/evaluation/result_perturbations/result_robustpruner_qwen}"
TASKS=("ocrbench" "ai2d_no_mask" "mme" "gqa" "chartqa")
PERTURBATIONS=("perturbations_0" "perturbations_1" "perturbations_2" "perturbations_3" "perturbations_4")
BUDGETS=("0.2")
MASTER_PORT="${MASTER_PORT:-28176}"
BASE_SCORE_POOL_MULTIPLIER="1.1"

declare -A TASK_YAMLS=(
  ["ocrbench"]="${LOCAL_LMMS_EVAL_ROOT}/lmms_eval/tasks/ocrbench/ocrbench.yaml"
  ["ai2d_no_mask"]="${LOCAL_LMMS_EVAL_ROOT}/lmms_eval/tasks/ai2d/ai2d_no_mask.yaml"
  ["mme"]="${LOCAL_LMMS_EVAL_ROOT}/lmms_eval/tasks/mme/mme.yaml"
  ["gqa"]="${LOCAL_LMMS_EVAL_ROOT}/lmms_eval/tasks/gqa/gqa.yaml"
  ["chartqa"]="${LOCAL_LMMS_EVAL_ROOT}/lmms_eval/tasks/chartqa/chartqa.yaml"
)

BACKUP_DIR="$(mktemp -d)"
GLOBAL_SCORE_FILE="${ROOT_DIR}/all_scores.tsv"
mkdir -p "${ROOT_DIR}"

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

cd "${LOCAL_LMMS_EVAL_ROOT}" || exit 1

echo -e "task\tbudget\tperturbation\tmetric\tvalue" > "${GLOBAL_SCORE_FILE}"

set_task_yaml_perturbation() {
  local yaml_path="$1"
  local perturbation_name="$2"
  local enable_text_perturb="true"
  if [[ "${perturbation_name}" == "perturbations_0" ]]; then
    enable_text_perturb="false"
  fi

  python3 - "${yaml_path}" "${enable_text_perturb}" "${perturbation_name}" <<'PY'
from pathlib import Path
import re
import sys

yaml_path = Path(sys.argv[1])
enable_text_perturb = sys.argv[2]
perturbation_name = sys.argv[3]
text = yaml_path.read_text(encoding="utf-8")

if re.search(r'(?m)^    text_perturb:', text):
    text = re.sub(r'(?m)^    text_perturb:.*$', f'    text_perturb: {enable_text_perturb}', text, count=1)
else:
    raise SystemExit(f"Missing `text_perturb` field in {yaml_path}")

if re.search(r'(?m)^    perturbation_name:', text):
    text = re.sub(r'(?m)^    perturbation_name:.*$', f'    perturbation_name: "{perturbation_name}"', text, count=1)
else:
    raise SystemExit(f"Missing `perturbation_name` field in {yaml_path}")

yaml_path.write_text(text, encoding="utf-8")
PY
}

extract_score() {
  local results_json_path="$1"
  python3 - "${results_json_path}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
results = data.get("results", {})
for task_metrics in results.values():
    for key, value in task_metrics.items():
        if key == "alias" or "stderr" in key:
            continue
        if isinstance(value, (int, float)):
            print(f"{key}\t{value}")
            raise SystemExit(0)
raise SystemExit("No numeric score found.")
PY
}

for TASK in "${TASKS[@]}"; do
  YAML_PATH="${TASK_YAMLS[$TASK]}"
  TASK_SCORE_FILE="${ROOT_DIR}/${TASK}/scores.tsv"
  mkdir -p "${ROOT_DIR}/${TASK}"
  echo -e "task\tbudget\tperturbation\tmetric\tvalue" > "${TASK_SCORE_FILE}"

  for BUDGET in "${BUDGETS[@]}"; do
    BASE_SCORE_POOL_RATIO=$(python3 -c "b=float('${BUDGET}'); a=float('${BASE_SCORE_POOL_MULTIPLIER}'); print(f'{b*a:.8f}')")

    for PERTURBATION in "${PERTURBATIONS[@]}"; do
      set_task_yaml_perturbation "${YAML_PATH}" "${PERTURBATION}"

      OUTPUT_PATH="${ROOT_DIR}/${TASK}/${BUDGET}/${MODEL_NAME}_robustpruner_${BUDGET}_${PERTURBATION}"
      LOG_FILE="${OUTPUT_PATH}/eval.log"

      rm -rf "${OUTPUT_PATH}"
      mkdir -p "${OUTPUT_PATH}"

      echo "============================================"
      echo "Task         : ${TASK}"
      echo "Budget       : ${BUDGET}"
      echo "Perturbation : ${PERTURBATION}"
      echo "Base pool    : ${BASE_SCORE_POOL_RATIO}"
      echo "Output       : ${OUTPUT_PATH}"
      echo "============================================"

      accelerate launch \
        --num_processes="${NUM_PROCESSES}" \
        --main_process_port="${MASTER_PORT}" \
        --mixed_precision=bf16 \
        -m lmms_eval \
        --model qwen2_5_vl_rp \
        --model_args "pretrained=${MODEL_PATH},new_modules_dir=${RP_MODULES_DIR},max_remain_ratio=${BUDGET},base_score_pool_multiplier=${BASE_SCORE_POOL_MULTIPLIER},attn_implementation=flash_attention_2,text_perturb=False" \
        --tasks "${TASK}" \
        --batch_size 1 \
        --output_path "${OUTPUT_PATH}" \
        --verbosity ERROR \
        > "${LOG_FILE}" 2>&1

      RESULTS_JSON=$(find "${OUTPUT_PATH}" -name '*_results.json' | head -n 1)
      if [[ -z "${RESULTS_JSON}" ]]; then
        echo "No results json found for task=${TASK}, perturbation=${PERTURBATION}" >&2
        exit 1
      fi

      SCORE_RECORD="$(extract_score "${RESULTS_JSON}")"
      METRIC_NAME="${SCORE_RECORD%%$'\t'*}"
      SCORE_VALUE="${SCORE_RECORD#*$'\t'}"

      echo -e "${TASK}\t${BUDGET}\t${PERTURBATION}\t${METRIC_NAME}\t${SCORE_VALUE}" | tee -a "${TASK_SCORE_FILE}" >> "${GLOBAL_SCORE_FILE}"
    done
  done
done

echo "All done. Score summaries are in ${ROOT_DIR}."
