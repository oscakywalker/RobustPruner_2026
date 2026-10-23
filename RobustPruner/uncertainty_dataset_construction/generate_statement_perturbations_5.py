import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


TASK_SPECS = {
    "chartqa": {
        "output_subdir": "chartqa",
        "datasets": [
            {"path": "lmms-lab/ChartQA", "name": None, "split": "test"},
        ],
    },
    "gqa": {
        "output_subdir": "gqa",
        "datasets": [
            {"path": "lmms-lab/GQA", "name": "testdev_balanced_instructions", "split": "testdev"},
        ],
    },
    "mme": {
        "output_subdir": "mme",
        "datasets": [
            {"path": "lmms-lab/MME", "name": None, "split": "test"},
        ],
    },
    "ocrbench": {
        "output_subdir": "ocrbench",
        "datasets": [
            {"path": "echo840/OCRBench", "name": None, "split": "test"},
        ],
    },
    "ai2d": {
        "output_subdir": "ai2d",
        "datasets": [
            {"path": "lmms-lab/ai2d", "name": None, "split": "test"},
            {"path": "Efficient-Large-Model/ai2d-no-mask", "name": None, "split": "test"},
        ],
    },
}

STATEMENT_STYLE_INSTRUCTION = (
    "Rewrite the question into a natural declarative/request-style prompt "
    "(e.g., 'Please determine ...', 'Identify ...', 'Report ...', or 'Indicate whether ...'). "
    "Keep the exact meaning, answer, and visual grounding unchanged."
)


def load_existing(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items() if isinstance(v, str)}
    except Exception as exc:
        print(f"[WARN] failed to load existing file {path}: {exc}")
    return {}


def save_json(path: Path, data: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def progress(iterable, **kwargs):
    try:
        from tqdm import tqdm

        return tqdm(iterable, **kwargs)
    except ImportError:
        return iterable


def iter_questions(dataset_specs: List[Dict[str, Optional[str]]], hf_token: bool) -> Iterable[str]:
    from datasets import load_dataset

    seen = set()
    for spec in dataset_specs:
        dataset_path = spec["path"]
        dataset_name = spec.get("name")
        split = spec["split"]
        print(f"Loading dataset: path={dataset_path}, name={dataset_name}, split={split}")
        if dataset_name:
            dataset = load_dataset(dataset_path, dataset_name, split=split, token=hf_token)
        else:
            dataset = load_dataset(dataset_path, split=split, token=hf_token)
        print(f"Loaded {len(dataset)} docs")

        for doc in dataset:
            question = doc.get("question")
            if not isinstance(question, str):
                continue
            question = question.strip()
            if not question or question in seen:
                continue
            seen.add(question)
            yield question


def build_prompt(question: str, task_name: str) -> str:
    return (
        "You are rewriting a visual question.\n"
        "Requirements:\n"
        "- Keep semantics exactly the same\n"
        "- Keep answerability and scope unchanged\n"
        "- Output only one rewritten prompt\n"
        "- Do not add explanation, prefix, or quotes\n\n"
        f"Rewrite strategy: {STATEMENT_STYLE_INSTRUCTION}\n\n"
        "Examples:\n"
        "Original: What cooking utensil is round?\n"
        "Rewritten: Please answer the question what cooking utensil is round.\n"
        "Original: Is the sky overcast?\n"
        "Rewritten: Indicate whether the sky is overcast.\n\n"
        f"Original question: {question}"
    )


def clean_response(text: str, fallback: str) -> str:
    if not text:
        return fallback
    out = text.strip()
    if out.startswith(("\"", "'")) and out.endswith(("\"", "'")) and len(out) >= 2:
        out = out[1:-1].strip()
    prefixes = [
        "rewritten:",
        "rewritten prompt:",
        "rephrased:",
        "rephrased question:",
        "statement:",
        "answer:",
    ]
    lower = out.lower()
    for prefix in prefixes:
        if lower.startswith(prefix):
            out = out[len(prefix):].strip()
            break
    return out or fallback


def call_rewrite(
    client: Any,
    model: str,
    question: str,
    task_name: str,
    max_retries: int,
    sleep_seconds: float,
) -> str:
    prompt = build_prompt(question, task_name)
    for attempt in range(1, max_retries + 1):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                temperature=0,
                stream=False,
            )
            content = completion.choices[0].message.content if completion.choices else ""
            return clean_response(content or "", question)
        except Exception as exc:
            if attempt == max_retries:
                print(f"[ERROR] retries exhausted for question={question!r}: {exc}")
                return question
            wait = sleep_seconds * attempt
            print(f"[WARN] retry {attempt}/{max_retries} after error: {exc}; sleep {wait:.2f}s")
            time.sleep(wait)
    return question


def generate_for_task(
    task_name: str,
    task_spec: Dict[str, Any],
    tasks_root: Path,
    client: Any,
    model: str,
    hf_token: bool,
    save_every: int,
    sleep_seconds: float,
    max_retries: int,
    overwrite: bool,
) -> None:
    output_path = tasks_root / task_spec["output_subdir"] / "text_perturbations_5.json"
    outputs = {} if overwrite else load_existing(output_path)
    processed = set(outputs.keys())

    questions = list(iter_questions(task_spec["datasets"], hf_token=hf_token))
    print(f"\n[{task_name}] total unique questions: {len(questions)}")
    print(f"[{task_name}] resume existing: {len(processed)}")
    print(f"[{task_name}] output: {output_path}")

    new_count = 0
    for question in progress(questions, desc=f"{task_name}: perturbations_5", ncols=110):
        if question in processed:
            continue
        outputs[question] = call_rewrite(
            client=client,
            model=model,
            question=question,
            task_name=task_name,
            max_retries=max_retries,
            sleep_seconds=sleep_seconds,
        )
        processed.add(question)
        new_count += 1

        if new_count % save_every == 0:
            save_json(output_path, outputs)

        time.sleep(sleep_seconds)

    save_json(output_path, outputs)
    print(f"[{task_name}] saved {len(outputs)} entries -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate statement-style text_perturbations_5.json for ChartQA, GQA, MME, "
            "OCRBench, and AI2D task directories."
        )
    )
    parser.add_argument(
        "--tasks-root",
        type=str,
        default=str(
            Path(__file__).resolve().parents[2]
            / "RobustPruner_lmm_eval" / "lmms-eval" / "lmms_eval" / "tasks"
        ),
        help="Root directory containing chartqa/gqa/mme/ocrbench/ai2d task folders.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["chartqa", "gqa", "mme", "ocrbench", "ai2d"],
        choices=sorted(TASK_SPECS.keys()),
    )
    parser.add_argument("--model", type=str, default="qwen3-vl-plus-2025-09-23")
    parser.add_argument("--base-url", type=str, default=os.environ.get("DASHSCOPE_BASE_URL", ""))
    parser.add_argument("--api-key", type=str, default=os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--hf-token", action="store_true", default=True)
    parser.add_argument("--no-hf-token", dest="hf_token", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("Missing API key. Set DASHSCOPE_API_KEY or OPENAI_API_KEY, or pass --api-key.")

    from openai import OpenAI

    tasks_root = Path(args.tasks_root)
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    for task_name in args.tasks:
        generate_for_task(
            task_name=task_name,
            task_spec=TASK_SPECS[task_name],
            tasks_root=tasks_root,
            client=client,
            model=args.model,
            hf_token=args.hf_token,
            save_every=args.save_every,
            sleep_seconds=args.sleep_seconds,
            max_retries=args.max_retries,
            overwrite=args.overwrite,
        )

    print("\nDone. Use perturbation_name: \"perturbations_5\" in the corresponding YAML.")


if __name__ == "__main__":
    main()
