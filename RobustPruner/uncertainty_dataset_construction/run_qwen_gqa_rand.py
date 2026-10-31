import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

try:
    from qwen_vl_utils import process_vision_info
except ImportError as e:
    raise ImportError("Please install qwen-vl-utils: pip install qwen-vl-utils") from e

PROJECT_ROOT = Path("/RobustPruner_2026/RobustPruner")
DEFAULT_INPUT_JSONL = str(PROJECT_ROOT / "uncertainty_dataset_construction" / "gqa_rand_seed42_20000.jsonl")
DEFAULT_OUTPUT_JSONL = str(PROJECT_ROOT / "robustpruner_qwen_eval" / "gqa_rand_seed42_20000_predictions.jsonl")
DEFAULT_TOKEN_SCORES_DIR = str(PROJECT_ROOT / "robustpruner_qwen_eval" / "token_scores" / "gqa_rand_seed42_20000")
DEFAULT_IMAGE_ROOT = str(PROJECT_ROOT / "datas" / "cot" / "gqa")


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


def safe_text(x: str) -> str:
    return str(x).strip() if x is not None else ""


def resolve_image_path(image_root: Path, image_value: str) -> Path:
    image_value = safe_text(image_value)
    raw_path = Path(image_value)

    candidates: List[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.append(image_root / raw_path.name)
        candidates.append(image_root / raw_path)
        parts = raw_path.parts
        if len(parts) >= 2 and parts[-2] == "gqa":
            candidates.append(image_root / parts[-1])
        if len(parts) >= 3 and parts[-3:] == ("cot", "gqa", raw_path.name):
            candidates.append(image_root / raw_path.name)

    seen = set()
    for candidate in candidates:
        candidate = candidate.expanduser()
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate

    return (image_root / raw_path.name).expanduser()


def _find_question_token_span(
    prompt_input_ids: List[int],
    valid_positions: List[int],
    visual_pos_list: List[int],
    tokenizer,
    content_text: str,
    suffix_text: str,
) -> Tuple[List[int], List[int], List[str], bool]:
    if not prompt_input_ids:
        return [], [], [], False

    visual_end_seq_idx = visual_pos_list[-1] if visual_pos_list else -1
    question_start_compact = 0
    if visual_end_seq_idx >= 0:
        for compact_idx, seq_idx in enumerate(valid_positions):
            if seq_idx > visual_end_seq_idx:
                question_start_compact = compact_idx
                break

    content_ids = tokenizer(content_text, add_special_tokens=False)["input_ids"]
    suffix_ids = tokenizer(suffix_text, add_special_tokens=False)["input_ids"]
    if not content_ids or len(suffix_ids) > len(content_ids):
        return [], [], [], False

    question_token_count = len(content_ids) - len(suffix_ids)
    if question_token_count <= 0:
        return [], [], [], False

    content_start = -1
    limit = len(prompt_input_ids) - len(content_ids) + 1
    for start in range(question_start_compact, max(question_start_compact, limit)):
        if prompt_input_ids[start : start + len(content_ids)] == content_ids:
            content_start = start
            break
    if content_start < 0:
        return [], [], [], False

    question_end_compact = content_start + question_token_count
    question_seq_indices = [int(valid_positions[i]) for i in range(content_start, question_end_compact)]
    question_token_ids = [int(prompt_input_ids[i]) for i in range(content_start, question_end_compact)]
    question_token_texts = tokenizer.convert_ids_to_tokens(question_token_ids)
    return question_seq_indices, question_token_ids, question_token_texts, True


def compute_question_to_visual_scores(model, processor, inputs, content_text: str, suffix_text: str) -> Dict:
    image_token_id = getattr(model.config, "image_token_id", None)
    video_token_id = getattr(model.config, "video_token_id", None)
    seq_input_ids = inputs["input_ids"][0]
    valid_mask = inputs["attention_mask"][0].bool()
    visual_mask = torch.zeros_like(seq_input_ids, dtype=torch.bool)
    if image_token_id is not None:
        visual_mask |= seq_input_ids.eq(image_token_id)
    if video_token_id is not None:
        visual_mask |= seq_input_ids.eq(video_token_id)
    visual_mask &= valid_mask
    visual_pos = torch.where(visual_mask)[0]
    visual_pos_list = [int(v) for v in visual_pos.tolist()]
    valid_positions = torch.where(valid_mask)[0].tolist()
    prompt_input_ids = seq_input_ids[valid_mask].tolist()
    question_seq_indices, question_token_ids, question_token_texts, alignment_found = _find_question_token_span(
        prompt_input_ids=prompt_input_ids,
        valid_positions=valid_positions,
        visual_pos_list=visual_pos_list,
        tokenizer=processor.tokenizer,
        content_text=content_text,
        suffix_text=suffix_text,
    )
    if not alignment_found:
        return {
            "num_visual_tokens": int(len(visual_pos)),
            "visual_token_positions": visual_pos_list,
            "question_token_seq_indices": [],
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

    with torch.inference_mode():
        fwd = model(
            **inputs,
            return_dict=True,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
        )

    llm_layers = model.model.layers
    num_layers = len(llm_layers)
    sampled_layer_indices = list(range(num_layers - 1, -1, -2))
    sampled_layer_indices_1based = [int(i + 1) for i in sampled_layer_indices]

    if len(visual_pos) == 0:
        return {
            "num_visual_tokens": 0,
            "visual_token_positions": [],
            "question_token_seq_indices": question_seq_indices,
            "question_token_ids": question_token_ids,
            "question_token_texts": question_token_texts,
            "text_to_visual_attention": [],
            "text_to_visual_similarity": [],
            "has_last_layer_attention": True,
            "layer_sampling": "from_last_layer_step_2",
            "sampled_layer_indices_1based": sampled_layer_indices_1based,
            "per_layer_text_to_visual": {},
            "question_token_alignment_found": True,
        }

    per_layer_text_to_visual = {}
    for layer_idx in sampled_layer_indices:
        layer = llm_layers[layer_idx]
        hidden_before_layer = fwd.hidden_states[layer_idx]  # [1,S,D]
        hidden_after_layer = fwd.hidden_states[layer_idx + 1]  # [1,S,D]

        self_attn = layer.self_attn
        proj_dtype = self_attn.q_proj.weight.dtype
        x = layer.input_layernorm(hidden_before_layer).to(proj_dtype)
        q_states = self_attn.q_proj(x)
        k_states = self_attn.k_proj(x)

        bsz, seqlen, _ = q_states.shape
        num_heads = self_attn.num_heads
        num_kv_heads = self_attn.num_key_value_heads
        head_dim = self_attn.head_dim
        num_kv_groups = self_attn.num_key_value_groups
        q_states = q_states.view(bsz, seqlen, num_heads, head_dim).transpose(1, 2)  # [B,H,S,D]
        k_states = k_states.view(bsz, seqlen, num_kv_heads, head_dim).transpose(1, 2)  # [B,Hkv,S,D]
        if num_kv_groups > 1:
            k_states = (
                k_states[:, :, None, :, :]
                .expand(bsz, num_kv_heads, num_kv_groups, seqlen, head_dim)
                .reshape(bsz, num_heads, seqlen, head_dim)
            )

        if len(question_seq_indices) == 0:
            text_to_visual_attention = []
            text_to_visual_similarity = []
        else:
            attn_vectors = []
            sim_vectors = []
            k_all = k_states[0:1]  # [1,H,S,D]
            vis_hidden = hidden_after_layer[0, visual_pos, :].float()  # [V,D]
            for seq_idx, tok_id, tok_text in zip(question_seq_indices, question_token_ids, question_token_texts):
                q_t = q_states[0:1, :, seq_idx : seq_idx + 1, :]  # [1,H,1,D]
                attn_logits = torch.matmul(q_t, k_all.transpose(-1, -2)) / (head_dim**0.5)  # [1,H,1,S]
                attn_logits = attn_logits.masked_fill(
                    (~valid_mask).view(1, 1, 1, seqlen),
                    torch.finfo(attn_logits.dtype).min,
                )
                attn_prob = torch.softmax(attn_logits, dim=-1)  # [1,H,1,S]
                attn_query = attn_prob[0, :, 0, :].mean(dim=0)  # [S]
                attn_vec = attn_query[visual_pos].float()
                attn_vectors.append(attn_vec)

                tok_vec = hidden_after_layer[0, seq_idx, :].float().unsqueeze(0)  # [1,D]
                sims = F.cosine_similarity(tok_vec, vis_hidden, dim=-1)  # [V]
                sim_vectors.append(sims)

            text_to_visual_attention = [float(v) for v in torch.stack(attn_vectors, dim=0).mean(dim=0).tolist()]
            text_to_visual_similarity = [float(v) for v in torch.stack(sim_vectors, dim=0).mean(dim=0).tolist()]

        layer_key = f"layer_{layer_idx + 1}"
        per_layer_text_to_visual[layer_key] = {
            "layer_index_0based": int(layer_idx),
            "layer_index_1based": int(layer_idx + 1),
            "text_to_visual_attention": text_to_visual_attention,
            "text_to_visual_similarity": text_to_visual_similarity,
        }

    # Keep last-layer fields for backward compatibility.
    last_layer_key = f"layer_{num_layers}"
    last_layer_payload = per_layer_text_to_visual.get(
        last_layer_key,
        {"text_to_visual_attention": [], "text_to_visual_similarity": []},
    )

    return {
        "num_visual_tokens": int(len(visual_pos)),
        "visual_token_positions": visual_pos_list,
        "question_token_seq_indices": question_seq_indices,
        "question_token_ids": question_token_ids,
        "question_token_texts": question_token_texts,
        "text_to_visual_attention": last_layer_payload["text_to_visual_attention"],
        "text_to_visual_similarity": last_layer_payload["text_to_visual_similarity"],
        "has_last_layer_attention": True,
        "layer_sampling": "from_last_layer_step_2",
        "sampled_layer_indices_1based": sampled_layer_indices_1based,
        "per_layer_text_to_visual": per_layer_text_to_visual,
        "question_token_alignment_found": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Qwen2.5-VL on local GQA random JSONL directly.")
    parser.add_argument(
        "--input-jsonl",
        type=str,
        default=DEFAULT_INPUT_JSONL,
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default=DEFAULT_IMAGE_ROOT,
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default=DEFAULT_OUTPUT_JSONL,
    )
    parser.add_argument(
        "--token-scores-dir",
        type=str,
        default=DEFAULT_TOKEN_SCORES_DIR,
    )
    parser.add_argument("--model", type=str, default="/mnt/nvme/fjh/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--attn-implementation", type=str, default="flash_attention_2")
    parser.add_argument("--save-token-scores", action="store_true")
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--one-keyword", action="store_true")
    parser.add_argument(
        "--post-prompt",
        type=str,
        default="\nAnswer the question using a single word or phrase.",
    )
    parser.add_argument("--perturbation-json", type=str, default="")
    parser.add_argument("--use-perturbed-question", action="store_true")
    args = parser.parse_args()

    input_jsonl = Path(args.input_jsonl)
    image_root = Path(args.image_root)
    output_jsonl = Path(args.output_jsonl)
    token_scores_dir = Path(args.token_scores_dir)
    perturbation_map: Dict[str, Dict] = {}
    if args.perturbation_json:
        perturbation_map = load_perturbation_map(Path(args.perturbation_json))
    if args.use_perturbed_question and args.perturbation_json:
        perturb_name = Path(args.perturbation_json).stem
        if args.output_jsonl == DEFAULT_OUTPUT_JSONL:
            # Keep perturbed runs isolated from original-run outputs.
            output_jsonl = output_jsonl.parent / perturb_name / output_jsonl.name
        if args.token_scores_dir == DEFAULT_TOKEN_SCORES_DIR:
            token_scores_dir = token_scores_dir.parent / perturb_name

    if not input_jsonl.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_jsonl}")
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be > 0")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards")

    all_rows = load_rows(input_jsonl)
    rows = [(line_no, obj) for line_no, obj in all_rows if (line_no - 1) % args.num_shards == args.shard_index]
    rows_todo = rows
    ensure_parent(output_jsonl)
    if args.save_token_scores:
        token_scores_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(all_rows)} valid rows from {input_jsonl}")
    print(f"Shard {args.shard_index}/{args.num_shards}: {len(rows)} rows")
    print(f"Resume disabled: always run from scratch")
    print(f"Rows to run: {len(rows_todo)}")
    if args.use_perturbed_question:
        print(
            f"Perturbed question mode: enabled "
            f"(source={args.perturbation_json if args.perturbation_json else 'N/A'})"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto" if device == "cuda" else None,
        attn_implementation=args.attn_implementation,
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model)

    new_count = 0
    with open(output_jsonl, "w", encoding="utf-8") as fout:
        for line_no, obj in tqdm(rows_todo, desc="infer", ncols=120):

            question_original = safe_text(obj.get("question", ""))
            question = question_original
            if args.use_perturbed_question and perturbation_map:
                one = perturbation_map.get(str(line_no))
                if isinstance(one, dict):
                    question = safe_text(one.get("perturbed_question", question_original)) or question_original
            answer = safe_text(obj.get("answer", ""))
            image_name = safe_text(obj.get("image", ""))
            image_path = resolve_image_path(image_root, image_name)

            if not image_path.exists():
                pred = ""
                error = f"missing_image:{image_path}"
            else:
                error = ""
                answer_instruction = (
                    "\nAnswer with exactly one keyword (one word) only. "
                    "No article, no punctuation."
                    if args.one_keyword
                    else "\nAnswer with a short phrase only, no full sentence."
                )
                suffix_text = args.post_prompt + answer_instruction
                content_text = question + suffix_text
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": str(image_path)},
                            {
                                "type": "text",
                                "text": content_text,
                            },
                        ],
                    }
                ]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                if device == "cuda":
                    inputs = inputs.to("cuda")

                attn_sim = None
                if args.save_token_scores:
                    attn_sim = compute_question_to_visual_scores(
                        model=model,
                        processor=processor,
                        inputs=inputs,
                        content_text=content_text,
                        suffix_text=suffix_text,
                    )

                with torch.inference_mode():
                    generate_out = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        eos_token_id=processor.tokenizer.eos_token_id,
                        pad_token_id=processor.tokenizer.pad_token_id,
                        return_dict_in_generate=True,
                        output_scores=bool(args.save_token_scores),
                    )

                in_len = int(inputs.input_ids.shape[1])
                gen_ids = generate_out.sequences[0][in_len:]
                pred = processor.batch_decode(
                    [gen_ids],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0].strip()

                if args.save_token_scores:
                    token_ids = gen_ids.tolist()
                    token_texts = processor.tokenizer.convert_ids_to_tokens(token_ids)
                    token_logprobs: List[float] = []
                    for t, step_scores in enumerate(generate_out.scores or []):
                        # scores shape: [batch, vocab_size], batch=1
                        logits = step_scores[0]
                        log_probs = torch.log_softmax(logits, dim=-1)
                        tok_id = token_ids[t]
                        token_logprobs.append(float(log_probs[tok_id].item()))

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
                "error": error,
            }
            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            new_count += 1

            if new_count % args.save_every == 0:
                fout.flush()
                os.fsync(fout.fileno())

    print(f"Done. Newly processed rows: {new_count}")
    print(f"Predictions written to: {output_jsonl}")
    if args.save_token_scores:
        print(f"Token scores written to: {token_scores_dir}")


if __name__ == "__main__":
    main()
