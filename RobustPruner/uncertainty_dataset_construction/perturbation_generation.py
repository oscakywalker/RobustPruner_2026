import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openai import OpenAI
from tqdm import tqdm


METHOD_SPECS = [
    {
        "name": "synonym_replacement",
        "instruction": (
            "Rewrite the question by replacing key words with close synonyms. "
            "Keep the exact meaning and answer unchanged."
        ),
    },
    {
        "name": "sentence_structure_change",
        "instruction": (
            "Rewrite the question with a different sentence structure (e.g., change clause order or phrasing style) "
            "while preserving the exact meaning and answer."
        ),
    },
    {
        "name": "advanced_vocabulary",
        "instruction": (
            "Rewrite the question with more formal and advanced vocabulary while preserving the exact meaning and answer."
        ),
    },
    {
        "name": "concise_rephrase",
        "instruction": (
            "Rewrite the question into a concise and natural alternative wording. "
            "Keep the exact meaning and answer unchanged."
        ),
    },
]


def load_questions_by_row(jsonl_path: Path) -> List[Tuple[int, str]]:
    rows: List[Tuple[int, str]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] Skip invalid JSON at line {line_no}")
                continue
            question = obj.get("question")
            if not isinstance(question, str):
                continue
            rows.append((line_no, question))
    return rows


def build_user_prompt(question: str, method_instruction: str) -> str:
    return (
        "You are rewriting a visual question.\n"
        "Requirements:\n"
        "- Keep semantics exactly the same\n"
        "- Keep answerability and scope unchanged\n"
        "- Output only one rewritten question\n"
        "- Do not add explanation, prefix, or quotes\n\n"
        f"Rewrite strategy: {method_instruction}\n\n"
        f"Original question: {question}"
    )


def clean_response(text: str, fallback: str) -> str:
    if not text:
        return fallback
    out = text.strip()
    prefixes = [
        "rephrased question:",
        "rephrased:",
        "rewrite:",
        "paraphrase:",
        "answer:",
    ]
    lower = out.lower()
    for p in prefixes:
        if lower.startswith(p):
            out = out[len(p):].strip()
            break
    return out if out else fallback


def call_rewrite(
    client: OpenAI,
    model: str,
    question: str,
    method_instruction: str,
    max_retries: int,
    sleep_seconds: float,
) -> str:
    prompt = build_user_prompt(question, method_instruction)
    for attempt in range(1, max_retries + 1):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
            )
            content = completion.choices[0].message.content if completion.choices else ""
            return clean_response(content or "", question)
        except Exception as e:
            if attempt == max_retries:
                print(f"[ERROR] retries exhausted: {e}")
                return question
            backoff = sleep_seconds * attempt
            print(f"[WARN] retry {attempt}/{max_retries} after error: {e}")
            time.sleep(backoff)
    return question


def load_existing_map(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
            if isinstance(obj, dict):
                return obj
    except Exception as e:
        print(f"[WARN] failed to load {path}: {e}")
    return {}


def save_map(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create 4 perturbation JSON files for questions in gqa_rand_seed42_20000.jsonl"
    )
    parser.add_argument(
        "--input-jsonl",
        type=str,
        default="/home/fjh/GlimpsePrune_2/uncertainty_dataset_construction/gqa_rand_seed42_20000.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/fjh/GlimpsePrune_2/uncertainty_dataset_construction",
    )
    parser.add_argument("--model", type=str, default="qwen3-vl-plus-2025-09-23")
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    args = parser.parse_args()

    input_path = Path(args.input_jsonl)
    output_dir = Path(args.output_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")

    question_rows = load_questions_by_row(input_path)
    print(f"Loaded {len(question_rows)} question rows from {input_path}")

    client = OpenAI(
        api_key="sk-c653b84b9d74460dbb8099eacca2a5ed",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    for method in METHOD_SPECS:
        method_name = method["name"]
        instruction = method["instruction"]
        out_path = output_dir / f"text_perturbations_{method_name}.json"

        if out_path.exists():
            print(f"\n[{method_name}] output already exists, skip -> {out_path}")
            continue

        perturbations = load_existing_map(out_path)
        print(f"\n[{method_name}] resume with {len(perturbations)} existing row entries")

        new_count = 0
        for line_no, question in tqdm(question_rows, desc=method_name, ncols=110):
            row_key = str(line_no)
            if row_key in perturbations:
                continue
            perturbed = call_rewrite(
                client=client,
                model=args.model,
                question=question,
                method_instruction=instruction,
                max_retries=args.max_retries,
                sleep_seconds=args.sleep_seconds,
            )
            perturbations[row_key] = {
                "question": question,
                "perturbed_question": perturbed,
            }
            new_count += 1

            if new_count % args.save_every == 0:
                save_map(out_path, perturbations)

            time.sleep(args.sleep_seconds)

        save_map(out_path, perturbations)
        print(f"[{method_name}] saved {len(perturbations)} entries -> {out_path}")

    print("\nDone: generated 4 perturbation json files.")


if __name__ == "__main__":
    main()