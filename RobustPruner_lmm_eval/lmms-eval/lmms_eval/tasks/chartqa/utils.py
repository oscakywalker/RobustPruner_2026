import json
from pathlib import Path

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


def chartqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def chartqa_doc_to_text(doc, lmms_eval_specific_kwargs):
    question = doc["question"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    if lmms_eval_specific_kwargs.get("text_perturb", False):
        perturbation_name = lmms_eval_specific_kwargs.get("perturbation_name", DEFAULT_PERTURBATION_NAME)
        question = get_perturbed_question(question, perturbation_name)
    return f"{pre_prompt}{question}{post_prompt}"


def chartqa_process_results(doc, results):
    pred = results[0]
    type = doc["type"]
    score = relaxed_correctness(pred, doc["answer"])
    score = 1.0 if score else 0.0
    return_dict = {"relaxed_overall": score}
    if type == "human_test":
        return_dict["relaxed_human_split"] = score
    else:
        return_dict["relaxed_augmented_split"] = score
    return return_dict


def relaxed_correctness(prediction, target, max_relative_change: float = 0.05) -> bool:
    """Calculates relaxed correctness.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    “Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct.”

    This funcion is taken from https://github.com/QwenLM/Qwen-VL/blob/34b4c0ee7b07726371b960911f249fe61b362ca3/eval_mm/evaluate_vqa.py#L113
    Args:
      target: List of target string.
      prediction: List of predicted string.
      max_relative_change: Maximum relative change.

    Returns:
      Whether the prediction was correct given the specified tolerance.
    """

    def _to_float(text: str):
        try:
            if text.endswith("%"):
                # Convert percentages to floats.
                return float(text.rstrip("%")) / 100.0
            else:
                return float(text)
        except ValueError:
            return None

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float - target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()
