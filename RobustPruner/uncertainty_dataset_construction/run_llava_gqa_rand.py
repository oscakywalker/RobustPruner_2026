import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "LLaVA"))

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.utils import disable_torch_init

GLIMPSE_LLAVA_BASE = "/mnt/nvme/fjh/glimpse_llava_eval"
JSONL_FILENAME = "gqa_rand_seed42_20000_predictions.jsonl"
SCORES_SUBDIR = "token_scores/gqa_rand_seed42_20000"


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


def compute_question_to_visual_scores(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    image_tensor: torch.Tensor,
    image_sizes: List,
    question_text: str,
    device: str,
) -> Dict:
    """
    Per-layer cosine similarity: mean(embed_tokens(question)) vs each visual token hidden state.
    No positional alignment needed — question embeddings come directly from the embedding table.
    """
    img_pos_candidates = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
    if len(img_pos_candidates) == 0:
        return {
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
            "question_token_alignment_found": True,
        }
    img_pos = int(img_pos_candidates[0])

    with torch.inference_mode():
        fwd = model(
            input_ids=input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            return_dict=True,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
        )

    expanded_seq_len = fwd.hidden_states[0].shape[1]
    num_visual_tokens = expanded_seq_len - (input_ids.shape[1] - 1)
    visual_pos_list = list(range(img_pos, img_pos + num_visual_tokens))
    visual_pos_tensor = torch.tensor(visual_pos_list, device=device)

    # Question token embeddings directly from embedding table — no position alignment needed
    q_ids = tokenizer(question_text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    with torch.inference_mode():
        mean_q_embed = model.model.embed_tokens(q_ids[0]).float().mean(0)  # [D]
    question_token_ids = q_ids[0].tolist()
    question_token_texts = tokenizer.convert_ids_to_tokens(question_token_ids)

    llm_layers = model.model.layers
    num_layers = len(llm_layers)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLaVA-1.5 on local GQA random JSONL directly.")
    parser.add_argument(
        "--input-jsonl",
        type=str,
        default="/home/fjh/GlimpsePrune_2/uncertainty_dataset_construction/gqa_rand_seed42_20000.jsonl",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default="/mnt/nvme/fjh/datasets/Visual-CoT/cot/cot_image_data/gqa",
    )
    parser.add_argument("--output-jsonl", type=str, default=None,
                        help="Override output JSONL path (auto-derived from model name if omitted).")
    parser.add_argument("--token-scores-dir", type=str, default=None,
                        help="Override token scores dir (auto-derived from model name if omitted).")
    parser.add_argument("--model", type=str, default="/mnt/nvme/fjh/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None,
                        help="Model name hint for load_pretrained_model (e.g. 'llava-v1.5-7b').")
    parser.add_argument("--conv-mode", type=str, default="llava_v1",
                        help="Conversation template key (e.g. llava_v1, llava_llama_2).")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--use-flash-attn", action="store_true", default=True)
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

    model_name = args.model_name or Path(args.model).name

    # Determine run subdirectory: "original" or the perturbation name
    if args.use_perturbed_question and args.perturbation_json:
        run_subdir = Path(args.perturbation_json).stem
    else:
        run_subdir = "original"

    run_dir = Path(GLIMPSE_LLAVA_BASE) / model_name / run_subdir

    output_jsonl = Path(args.output_jsonl) if args.output_jsonl else run_dir / JSONL_FILENAME
    token_scores_dir = Path(args.token_scores_dir) if args.token_scores_dir else run_dir / SCORES_SUBDIR

    input_jsonl = Path(args.input_jsonl)
    image_root = Path(args.image_root)
    perturbation_map: Dict[str, Dict] = {}
    if args.perturbation_json:
        perturbation_map = load_perturbation_map(Path(args.perturbation_json))

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
    disable_torch_init()

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=args.model,
        model_base=args.model_base,
        model_name=model_name,
        use_flash_attn=args.use_flash_attn,
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()

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
            image_path = image_root / image_name

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

                # Build prompt via conversation template
                conv = conv_templates[args.conv_mode].copy()
                user_msg = DEFAULT_IMAGE_TOKEN + "\n" + content_text
                conv.append_message(conv.roles[0], user_msg)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

                input_ids = (
                    tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                    .unsqueeze(0)
                    .to(device)
                )

                image = Image.open(image_path).convert("RGB")
                image_tensor = (
                    process_images([image], image_processor, model.config)[0]
                    .unsqueeze(0)
                    .half()
                    .to(device)
                )
                image_sizes = [image.size]  # (W, H)

                attn_sim = None
                if args.save_token_scores:
                    attn_sim = compute_question_to_visual_scores(
                        model=model,
                        tokenizer=tokenizer,
                        input_ids=input_ids,
                        image_tensor=image_tensor,
                        image_sizes=image_sizes,
                        question_text=question,
                        device=device,
                    )

                with torch.inference_mode():
                    generate_out = model.generate(
                        input_ids,
                        images=image_tensor,
                        image_sizes=image_sizes,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        use_cache=True,
                        return_dict_in_generate=True,
                        output_scores=True,  # always on: needed to compute in_len
                    )

                # LLaVA's generate uses inputs_embeds internally, so sequences[0]
                # has length = (HF prefix len) + num_new_tokens, NOT original prompt len.
                # Use len(scores) to reliably identify the generated portion.
                num_new = len(generate_out.scores) if generate_out.scores else 0
                gen_ids = generate_out.sequences[0][-num_new:] if num_new > 0 else generate_out.sequences[0][0:0]
                pred = tokenizer.batch_decode(
                    [gen_ids],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0].strip()

                if args.save_token_scores:
                    token_ids = gen_ids.tolist()
                    token_texts = tokenizer.convert_ids_to_tokens(token_ids)
                    token_logprobs: List[float] = []
                    for t, step_scores in enumerate(generate_out.scores or []):
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