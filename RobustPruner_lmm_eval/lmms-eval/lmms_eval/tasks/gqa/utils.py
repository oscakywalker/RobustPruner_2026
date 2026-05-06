import json
from pathlib import Path

from datasets import load_dataset

GQA_RAW_IMAGE_DATASET = None
GQA_ID2IMAGE = None
DEFAULT_PERTURBATION_NAME = "perturbations_1"
PERTURBATIONS_DIR = Path(__file__).parent
_PERTURBATIONS_CACHE = {}


def _get_perturbations_file(perturbation_name: str) -> Path:
    return PERTURBATIONS_DIR / f"text_{perturbation_name}.json"


def load_perturbations(perturbation_name: str):
    if perturbation_name in _PERTURBATIONS_CACHE:
        return _PERTURBATIONS_CACHE[perturbation_name]

    perturbations_file = _get_perturbations_file(perturbation_name)
    if perturbations_file.exists():
        with open(perturbations_file, "r", encoding="utf-8") as f:
            perturbations = json.load(f)
    else:
        perturbations = {}

    _PERTURBATIONS_CACHE[perturbation_name] = perturbations
    return perturbations


def get_perturbed_question(question: str, perturbation_name: str) -> str:
    perturbations = load_perturbations(perturbation_name)
    return perturbations.get(question, question)


def gqa_doc_to_visual(doc):
    global GQA_RAW_IMAGE_DATASET
    global GQA_ID2IMAGE
    if GQA_RAW_IMAGE_DATASET is None:
        GQA_RAW_IMAGE_DATASET = load_dataset("lmms-lab/GQA", "testdev_balanced_images", split="testdev", token=True)
        GQA_ID2IMAGE = {}
        for row in GQA_RAW_IMAGE_DATASET:
            GQA_ID2IMAGE[row["id"]] = row["image"].convert("RGB")
    image = GQA_ID2IMAGE[doc["imageId"]]
    return [image]

def gqa_doc_to_text(doc, lmms_eval_specific_kwargs):
    question = doc["question"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    if lmms_eval_specific_kwargs.get("text_perturb", False):
        perturbation_name = lmms_eval_specific_kwargs.get("perturbation_name", DEFAULT_PERTURBATION_NAME)
        question = get_perturbed_question(question, perturbation_name)
    return f"{pre_prompt}{question}{post_prompt}"