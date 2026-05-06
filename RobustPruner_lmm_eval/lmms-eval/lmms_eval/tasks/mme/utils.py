import datetime
import json
import os
from collections import defaultdict
from pathlib import Path

from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

dir_name = os.path.dirname(os.path.abspath(__file__))

eval_type_dict = {
    "Perception": [
        "existence",
        "count",
        "position",
        "color",
        "posters",
        "celebrity",
        "scene",
        "landmark",
        "artwork",
        "OCR",
    ],
    "Cognition": [
        "commonsense_reasoning",
        "numerical_calculation",
        "text_translation",
        "code_reasoning",
    ],
}


replace_prompt = " Please answer yes or no."
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
        eval_logger.info(f"[MME] Loaded {len(perturbations)} text perturbations from {perturbations_file.name}")
    else:
        eval_logger.warning(f"[MME] Perturbations file not found: {perturbations_file}")
        eval_logger.warning("[MME] Text perturbation disabled for missing file; using original questions.")
        perturbations = {}

    _PERTURBATIONS_CACHE[perturbation_name] = perturbations
    return perturbations


def get_perturbed_question(question: str, perturbation_name: str) -> str:
    perturbations = load_perturbations(perturbation_name)
    return perturbations.get(question, question)


def mme_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def mme_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"].strip()
    if lmms_eval_specific_kwargs.get("text_perturb", False):
        perturbation_name = lmms_eval_specific_kwargs.get("perturbation_name", DEFAULT_PERTURBATION_NAME)
        question = get_perturbed_question(question, perturbation_name)
    if "pre_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["pre_prompt"] != "":
        question = question.replace(replace_prompt, "")
        question = f"{lmms_eval_specific_kwargs['pre_prompt']}{question}"
    if "post_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["post_prompt"] != "":
        question = question.replace(replace_prompt, "")
        question = f"{question}{lmms_eval_specific_kwargs['post_prompt']}"
    return question


def parse_pred_ans(pred_ans):
    """Brought from Otter Eval"""
    pred_ans = pred_ans.lower().strip().replace(".", "")
    pred_label = None
    if pred_ans in ["yes", "no"]:
        pred_label = pred_ans
    elif len(pred_ans) == 1:
        if pred_ans == "y":
            pred_label = "yes"
        elif pred_ans == "n":
            pred_label = "no"
        else:
            pred_label = "other"
    else:
        prefix_pred_ans = pred_ans[:4]
        if "yes" in prefix_pred_ans:
            pred_label = "yes"
        elif "no" in prefix_pred_ans:
            pred_label = "no"
        else:
            pred_label = "other"
    return pred_label


def mme_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case mme score), value: metric value
    """
    pred = results[0]
    pred_ans = parse_pred_ans(pred)
    gt_ans = doc["answer"].lower().strip().replace(".", "")
    assert gt_ans in ["yes", "no"]
    assert pred_ans in ["yes", "no", "other"]
    score = 1.0 if pred_ans == gt_ans else 0.0
    category = doc["category"]
    key_name = "mme_perception_score" if category in eval_type_dict["Perception"] else "mme_cognition_score"
    # Note: the key name here is very important. It decides which aggregation function will receive the results
    # We note down the question id/category to help us aggregate the results later
    return {key_name: {"question_id": doc["question_id"], "category": category, "score": score}}


def mme_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    category2score = defaultdict(dict)
    for result in results:
        question_id = result["question_id"]
        score = result["score"]
        category = result["category"]
        if question_id not in category2score[category]:
            category2score[category][question_id] = []
        category2score[category][question_id].append(score)
    category2avg_score = {}
    for category, question2scores in category2score.items():
        total_score = 0
        for question_id, scores in question2scores.items():
            assert len(scores) == 2, "MME only supports pairwise evaluation"
            acc = sum(scores) / len(scores) * 100.0
            acc_plus = (sum(scores) == 2) * 100.0
            score = acc_plus + acc
            total_score += score
        avg_score = total_score / len(question2scores)
        category2avg_score[category] = avg_score
    for category, avg_score in category2avg_score.items():
        eval_logger.info(f"{category}: {avg_score:.2f}")
    total_score = sum(category2avg_score.values())
    return total_score