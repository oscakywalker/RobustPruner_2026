import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

try:
    from qwen_vl_utils import process_vision_info
    _HAS_QWEN_VL_UTILS = True
except ImportError:
    _HAS_QWEN_VL_UTILS = False

GLIMPSE_LLAVA_BASE = "/mnt/nvme/fjh/glimpse_llava_ov_eval"
JSONL_FILENAME = "gqa_rand_seed42_20000_predictions.jsonl"
SCORES_SUBDIR = "token_scores/gqa_rand_seed42_20000"

DEFAULT_MODEL_PATH = (
    "/mnt/nvme/fjh/models--lmms-lab--LLaVA-OneVision-1.5-8B-Instruct"
    "/snapshots/bdf95183e0f91549cc71820b12d920b1a63689a2"
)
DEFAULT_INPUT_JSONL = str(Path(__file__).parent / "gqa_rand_seed42_20000.jsonl")

PERTURBATION_JSONS = [
    str(Path(__file__).parent / "text_perturbations_advanced_vocabulary.json"),
    str(Path(__file__).parent / "text_perturbations_concise_rephrase.json"),
    str(Path(__file__).parent / "text_perturbations_sentence_structure_change.json"),
    str(Path(__file__).parent / "text_perturbations_synonym_replacement.json"),
]


def load_rows(jsonl_path: Path) -> List[Tuple[int, Dict]]:
    rows: List[Tuple[int, Dict]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            if "question" not in obj or "image" not in obj:
                continue
            rows.append((line_no, obj))
    return rows


def load_perturbation_map(path: Path) -> Dict[str, Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Perturbation JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Perturbation JSON must be an object: {path}")
    return obj


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def safe_text(x) -> str:
    return str(x).strip() if x is not None else ""


def get_image_token_id(processor) -> Optional[int]:
    """Detect the image pad/placeholder token ID from the processor."""
    for attr in ("image_token_id", "img_token_id"):
        val = getattr(processor, attr, None)
        if val is not None:
            return int(val)
    tok = processor.tokenizer
    for token_str in ("<|image_pad|>", "<image>", "<img>", "<|img|>"):
        try:
            tid = tok.convert_tokens_to_ids(token_str)
            if tid is not None and tid != tok.unk_token_id:
                return int(tid)
        except Exception:
            pass
    return None


def compute_question_to_visual_scores(
    model,
    processor,
    inputs: Dict[str, torch.Tensor],
    question_text: str,
    image_token_id: Optional[int],
    device: str,
) -> Dict:
    """
    Per-layer cosine similarity between mean(question token embeddings) and each visual token hidden state.
    In LLaVA-OV-1.5 (Qwen2-based), all image pad tokens are already expanded in input_ids,
    so visual positions are exactly where image_token_id appears.
    """
    input_ids = inputs["input_ids"]

    empty = {
        "num_visual_tokens": 0,
        "visual_token_positions": [],
        "question_token_ids": [],
        "question_token_texts": [],
        "text_to_visual_attention": [],
        "text_to_visual_similarity": [],
        "has_last_layer_attention": True,
        "layer_sampling": "from_last_layer_step_2",
        "sampled_layer_indices_1based": [],
        "per_layer_text_to_visual": {},
        "question_token_alignment_found": False,
    }

    if image_token_id is None:
        return empty

    visual_pos_tensor = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]
    num_visual_tokens = len(visual_pos_tensor)
    if num_visual_tokens == 0:
        result = empty.copy()
        result["question_token_alignment_found"] = True
        return result

    visual_pos_list = visual_pos_tensor.tolist()

    with torch.inference_mode():
        fwd = model(
            **inputs,
            return_dict=True,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
        )

    tok = processor.tokenizer
    q_ids = tok(question_text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    with torch.inference_mode():
        mean_q_embed = model.model.language_model.embed_tokens(q_ids[0]).float().mean(0)  # [D]
    question_token_ids = q_ids[0].tolist()
    question_token_texts = tok.convert_ids_to_tokens(question_token_ids)

    num_layers = len(model.model.language_model.layers)
    sampled_layer_indices = list(range(num_layers - 1, -1, -2))
    sampled_layer_indices_1based = [int(i + 1) for i in sampled_layer_indices]

    per_layer_text_to_visual = {}
    for layer_idx in sampled_layer_indices:
        vis_hidden = fwd.hidden_states[layer_idx + 1][0, visual_pos_tensor, :].float()  # [V, D]
        sims = F.cosine_similarity(mean_q_embed.unsqueeze(0), vis_hidden, dim=-1)       # [V]
        layer_key = f"layer_{layer_idx + 1}"
        per_layer_text_to_visual[layer_key] = {
            "layer_index_0based": int(layer_idx),
            "layer_index_1based": int(layer_idx + 1),
            "text_to_visual_attention": [],
            "text_to_visual_similarity": [float(v) for v in sims.tolist()],
        }

    last_layer_key = f"layer_{num_layers}"
    last_sims = per_layer_text_to_visual.get(last_layer_key, {}).get("text_to_visual_similarity", [])

    return {
        "num_visual_tokens": num_visual_tokens,
        "visual_token_positions": visual_pos_list,
        "question_token_ids": question_token_ids,
        "question_token_texts": question_token_texts,
        "text_to_visual_attention": [],
        "text_to_visual_similarity": last_sims,
        "has_last_layer_attention": True,
        "layer_sampling": "from_last_layer_step_2",
        "sampled_layer_indices_1based": sampled_layer_indices_1based,
        "per_layer_text_to_visual": per_layer_text_to_visual,
        "question_token_alignment_found": True,
    }


def run_one_perturbation(
    args,
    model,
    processor,
    image_token_id: Optional[int],
    all_rows: List[Tuple[int, Dict]],
    perturbation_json: str,
    use_perturbed: bool,
    model_name: str,
    device: str,
) -> None:
    if use_perturbed and perturbation_json:
        run_subdir = Path(perturbation_json).stem
        perturbation_map = load_perturbation_map(Path(perturbation_json))
    else:
        run_subdir = "original"
        perturbation_map = {}

    run_dir = Path(GLIMPSE_LLAVA_BASE) / model_name / run_subdir
    output_jsonl = run_dir / JSONL_FILENAME
    token_scores_dir = run_dir / SCORES_SUBDIR

    ensure_parent(output_jsonl)
    if args.save_token_scores:
        token_scores_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        (line_no, obj)
        for line_no, obj in all_rows
        if (line_no - 1) % args.num_shards == args.shard_index
    ]
    print(f"\n{'='*60}")
    print(f"Run subdir : {run_subdir}")
    print(f"Rows       : {len(rows)}")
    print(f"Output     : {output_jsonl}")
    print(f"{'='*60}")

    image_root = Path(args.image_root)
    new_count = 0

    with open(output_jsonl, "w", encoding="utf-8") as fout:
        for line_no, obj in tqdm(rows, desc=f"infer [{run_subdir[:30]}]", ncols=120):
            question_original = safe_text(obj.get("question", ""))
            question = question_original
            if use_perturbed and perturbation_map:
                one = perturbation_map.get(str(line_no))
                if isinstance(one, dict):
                    question = safe_text(one.get("perturbed_question", question_original)) or question_original

            answer = safe_text(obj.get("answer", ""))
            image_name = safe_text(obj.get("image", ""))
            image_path = image_root / image_name

            if not image_path.exists():
                out_obj = {
                    "line_no": line_no,
                    "question": question,
                    "question_original": question_original,
                    "answer": answer,
                    "image": image_name,
                    "prediction": "",
                    "correct": 0,
                    "error": f"missing_image:{image_path}",
                }
                fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                new_count += 1
                continue

            content_text = question + args.post_prompt
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(image_path)},
                        {"type": "text", "text": content_text},
                    ],
                }
            ]

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            if _HAS_QWEN_VL_UTILS:
                image_inputs, video_inputs = process_vision_info(messages)
            else:
                image_inputs = [Image.open(image_path).convert("RGB")]
                video_inputs = None

            proc_kwargs: Dict = dict(
                text=[text], images=image_inputs, padding=True, return_tensors="pt"
            )
            if video_inputs:
                proc_kwargs["videos"] = video_inputs
            inputs = processor(**proc_kwargs)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            attn_sim = None
            if args.save_token_scores:
                attn_sim = compute_question_to_visual_scores(
                    model=model,
                    processor=processor,
                    inputs=inputs,
                    question_text=question,
                    image_token_id=image_token_id,
                    device=device,
                )

            with torch.inference_mode():
                generate_out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            num_new = len(generate_out.scores) if generate_out.scores else 0
            gen_ids = (
                generate_out.sequences[0][-num_new:]
                if num_new > 0
                else generate_out.sequences[0][0:0]
            )
            tok = processor.tokenizer
            pred = tok.batch_decode(
                [gen_ids],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            if args.save_token_scores:
                token_ids = gen_ids.tolist()
                token_texts = tok.convert_ids_to_tokens(token_ids)
                token_logprobs: List[float] = []
                for t, step_scores in enumerate(generate_out.scores or []):
                    logits = step_scores[0]
                    log_probs = torch.log_softmax(logits.float(), dim=-1)
                    token_logprobs.append(float(log_probs[token_ids[t]].item()))

                one_score = {
                    "line_no": line_no,
                    "image": image_name,
                    "question": question,
                    "question_original": question_original,
                    "prediction": pred,
                    "generated_token_ids": token_ids,
                    "generated_tokens": token_texts,
                    "generated_token_logprobs": token_logprobs,
                }
                one_score.update(attn_sim or {})
                one_score["score_type"] = "question_tokens_to_visual_tokens_mean"
                score_path = token_scores_dir / f"gqa_rand_seed42_20000_{line_no}.json"
                with open(score_path, "w", encoding="utf-8") as sf:
                    json.dump(one_score, sf, ensure_ascii=False, indent=2)

            out_obj = {
                "line_no": line_no,
                "question": question,
                "question_original": question_original,
                "answer": answer,
                "image": image_name,
                "prediction": pred,
                "correct": int(pred.strip().lower() == answer.strip().lower()) if pred else 0,
                "error": "",
            }
            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            new_count += 1

            if new_count % args.save_every == 0:
                fout.flush()
                os.fsync(fout.fileno())

    print(f"Done [{run_subdir}]. Rows: {new_count}  →  {output_jsonl}")
    if args.save_token_scores:
        print(f"Token scores → {token_scores_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LLaVA-OV-1.5 on GQA-rand JSONL for all text perturbations."
    )
    parser.add_argument("--input-jsonl", type=str, default=DEFAULT_INPUT_JSONL)
    parser.add_argument(
        "--image-root",
        type=str,
        default="/mnt/nvme/fjh/datasets/Visual-CoT/cot/cot_image_data/gqa",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model-name", type=str, default=None,
                        help="Override model directory name used in output paths.")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--save-token-scores", action="store_true")
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--post-prompt",
        type=str,
        default="\nAnswer the question using a single word or phrase.",
    )
    parser.add_argument(
        "--perturbation-json",
        type=str,
        default="",
        help="Run a single perturbation JSON instead of auto-looping all four.",
    )
    parser.add_argument(
        "--use-perturbed-question",
        action="store_true",
        help="Activate perturbed questions when using --perturbation-json.",
    )
    parser.add_argument(
        "--run-original",
        action="store_true",
        help="Also run original (unperturbed) questions before the perturbations.",
    )
    parser.add_argument("--attn-implementation", type=str, default="flash_attention_2")
    parser.add_argument("--torch-dtype", type=str, default="bfloat16",
                        help="torch dtype string, e.g. bfloat16, float16, float32.")
    args = parser.parse_args()

    model_name = args.model_name or Path(args.model).name

    input_jsonl = Path(args.input_jsonl)
    if not input_jsonl.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_jsonl}")
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be > 0")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard-index out of range")

    all_rows = load_rows(input_jsonl)
    print(f"Loaded {len(all_rows)} valid rows from {input_jsonl}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = getattr(torch, args.torch_dtype, torch.bfloat16)

    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        attn_implementation=args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    ).eval()

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    image_token_id = get_image_token_id(processor)
    print(f"Image token ID: {image_token_id}")

    if args.perturbation_json:
        tasks = [(args.perturbation_json, args.use_perturbed_question)]
    else:
        tasks = [(pj, True) for pj in PERTURBATION_JSONS]
        if args.run_original:
            tasks = [("", False)] + tasks

    for perturbation_json, use_perturbed in tasks:
        run_one_perturbation(
            args=args,
            model=model,
            processor=processor,
            image_token_id=image_token_id,
            all_rows=all_rows,
            perturbation_json=perturbation_json,
            use_perturbed=use_perturbed,
            model_name=model_name,
            device=device,
        )

    print("\nAll perturbations complete.")


if __name__ == "__main__":
    main()