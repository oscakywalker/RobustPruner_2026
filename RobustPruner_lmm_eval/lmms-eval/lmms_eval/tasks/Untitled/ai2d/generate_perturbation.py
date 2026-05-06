import json
import os
import time
from pathlib import Path

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

MODEL_NAME = "qwen3-vl-plus-2025-09-23"
SAVE_EVERY = 10
SLEEP_SECONDS = 0.2
OUTPUT_DIR = Path(__file__).parent

METHOD_SPECS = [
    {
        "name": "perturbations_1",
        "instruction": (
            "Rewrite the question by replacing key words with close synonyms. "
            "Keep the exact meaning and answer unchanged."
        ),
    },
    {
        "name": "perturbations_2",
        "instruction": (
            "Rewrite the question with a different sentence structure (e.g., change clause order or phrasing style) "
            "while preserving the exact meaning and answer."
        ),
    },
    {
        "name": "perturbations_3",
        "instruction": (
            "Rewrite the question with more formal and advanced vocabulary while preserving the exact meaning and answer."
        ),
    },
    {
        "name": "perturbations_4",
        "instruction": (
            "Rewrite the question into a concise and natural alternative wording. "
            "Keep the exact meaning and answer unchanged."
        ),
    },
]

client = OpenAI(
    api_key=os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY") or "sk-c653b84b9d74460dbb8099eacca2a5ed",
    base_url=os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
)


def build_prompt(question: str) -> str:
    method_lines = "\n".join(f"- {spec['name']}: {spec['instruction']}" for spec in METHOD_SPECS)
    return f"""Generate four rewritten versions of the AI2D multiple-choice question stem.

Requirements:
- Rewrite only the question stem, not the answer options
- Preserve the exact meaning and answer in every rewrite
- Return valid JSON only
- The JSON keys must be exactly the method names below
- Each value must contain only the rewritten question string

Methods:
{method_lines}

Question: {question}
"""


def normalize_outputs(question: str, content: str):
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return {spec["name"]: question for spec in METHOD_SPECS}

    normalized = {}
    for spec in METHOD_SPECS:
        value = parsed.get(spec["name"], question)
        if not isinstance(value, str):
            value = question
        value = value.strip()
        normalized[spec["name"]] = value or question
    return normalized


def generate_perturbations(question: str):
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": build_prompt(question)}],
                }
            ],
            temperature=0,
        )
        content = completion.choices[0].message.content.strip()
        return normalize_outputs(question, content)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return {spec["name"]: question for spec in METHOD_SPECS}


def load_existing_outputs():
    outputs = {}
    for spec in METHOD_SPECS:
        output_path = OUTPUT_DIR / f"text_{spec['name']}.json"
        if output_path.exists():
            with open(output_path, "r", encoding="utf-8") as f:
                outputs[spec["name"]] = json.load(f)
        else:
            outputs[spec["name"]] = {}
    return outputs


def save_outputs(outputs):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for spec in METHOD_SPECS:
        output_path = OUTPUT_DIR / f"text_{spec['name']}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(outputs[spec["name"]], f, ensure_ascii=False, indent=2)


def main():
    print("Loading AI2D dataset...")
    dataset = load_dataset("lmms-lab/ai2d", split="test", token=True)
    print(f"Total docs: {len(dataset)}")

    outputs = load_existing_outputs()
    processed_questions = set(outputs[METHOD_SPECS[0]["name"]].keys())

    for index, doc in enumerate(tqdm(dataset, desc="Generating perturbations", ncols=100)):
        question = doc["question"].strip()
        if question in processed_questions:
            continue

        perturbations = generate_perturbations(question)
        for spec in METHOD_SPECS:
            outputs[spec["name"]][question] = perturbations[spec["name"]]
        processed_questions.add(question)

        if (index + 1) % SAVE_EVERY == 0:
            save_outputs(outputs)

        time.sleep(SLEEP_SECONDS)

    save_outputs(outputs)
    for spec in METHOD_SPECS:
        print(f"Saved {len(outputs[spec['name']])} perturbations to text_{spec['name']}.json")


if __name__ == "__main__":
    main()