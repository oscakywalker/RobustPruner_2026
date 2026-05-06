import os
import json
from pathlib import Path
from statistics import mean

ROOT = Path("/mnt/nvme/fjh/qwen-evaluation")

INPUT_DIRS = [
    ROOT / "visualization_text_perturb/gqa/text_perturb/0.3/qwen25vl_7b_glimpse_0.3_text_perturbations_0/token_scores",
    ROOT / "visualization_text_perturb/gqa/text_perturb/0.3/qwen25vl_7b_glimpse_0.3_text_perturbations_1/token_scores",
    ROOT / "visualization_text_perturb/gqa/text_perturb/0.3/qwen25vl_7b_glimpse_0.3_text_perturbations_2/token_scores",
    ROOT / "visualization_text_perturb/gqa/text_perturb/0.3/qwen25vl_7b_glimpse_0.3_text_perturbations_3/token_scores",
    ROOT / "visualization_text_perturb/gqa/text_perturb/0.3/qwen25vl_7b_glimpse_0.3_text_perturbations_4/token_scores",
]

# Output dictionary path.
OUTPUT_DIR = ROOT / "qwen_combined_gqa_token_score_vscan"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def population_variance(values):
    if not values:
        return 0.0
    m = mean(values)
    return sum((x - m) ** 2 for x in values) / len(values)

def extract_scores_from_mask_flat(data, fpath):
    if "mask_flat" not in data:
        raise ValueError(f"'mask_flat' missing in {fpath}")

    raw = data["mask_flat"]

    # æƒ…å†µ Aï¼šå·²ç»æ˜¯ dict æˆ– list
    if isinstance(raw, dict):
        scores_dict = raw
    elif isinstance(raw, list):
        # å¦‚æžœå·²ç»æ˜¯åˆ—è¡¨ï¼Œå‡è®¾å®ƒå°±æ˜¯ token é¡ºåºçš„ scoresï¼Œç›´æŽ¥è¿”å›ž
        return list(raw)
    elif isinstance(raw, str):
        # æƒ…å†µ Bï¼šæ˜¯å­—ç¬¦ä¸²ï¼Œéœ€è¦å†è§£æžä¸€æ¬¡
        try:
            scores_dict = json.loads(raw)
        except Exception as e:
            raise ValueError(f"Failed to parse 'mask_flat' JSON in {fpath}: {e}")
    else:
        raise TypeError(
            f"Unexpected type for 'mask_flat' in {fpath}: {type(raw)}. "
            f"Expected str/dict/list."
        )

    try:
        items = sorted(scores_dict.items(), key=lambda kv: int(kv[0]))
    except Exception as e:
        raise ValueError(
            f"mask_flat keys are not numeric strings in {fpath}: {e}"
        )
    scores = [v for _, v in items]

    # ç”¨ num_tokens åš sanity checkï¼ˆå¦‚æžœæœ‰çš„è¯ï¼‰
    num_tokens = data.get("num_tokens")
    if num_tokens is not None and len(scores) != num_tokens:
        raise ValueError(
            f"Length of mask_flat ({len(scores)}) != num_tokens ({num_tokens}) in {fpath}"
        )

    return scores

def main():
    base_dir = INPUT_DIRS[0]
    json_files = sorted(f for f in base_dir.iterdir() if f.suffix == ".json")

    print(f"Found {len(json_files)} JSON files in base dir: {base_dir}")

    for idx, base_file in enumerate(json_files, start=1):
        filename = base_file.name
        token_score_lists = []
        doc_id = None

        for d in INPUT_DIRS:
            fpath = d / filename
            if not fpath.exists():
                raise FileNotFoundError(f"Missing file: {fpath}")

            with fpath.open("r") as f:
                data = json.load(f)

            if doc_id is None:
                doc_id = data.get("doc_id")

            # å…³é”®é€»è¾‘ï¼šåªç”¨ mask_flat ä½œä¸º token scores
            scores = extract_scores_from_mask_flat(data, fpath)
            token_score_lists.append(scores)

        lengths = {len(lst) for lst in token_score_lists}
        if len(lengths) != 1:
            raise ValueError(f"Token lengths mismatch for {filename}: {lengths}")

        num_tokens = lengths.pop()

        token_stats = {}
        has_nonzero_variance = False

        for i in range(num_tokens):
            values = [lst[i] for lst in token_score_lists]
            m = mean(values)
            v = population_variance(values)

            if v > 0:
                has_nonzero_variance = True

            token_stats[str(i)] = {"avg": m, "var": v}

        if has_nonzero_variance:
            print(f"[VAR>0] {filename}")

        # Save the json file.
        output = {
            "doc_id": doc_id,
            "token_stats": token_stats
        }

        out_path = OUTPUT_DIR / filename
        with out_path.open("w") as f:
            json.dump(output, f)

        if idx % 100 == 0 or idx == len(json_files):
            print(f"Processed {idx}/{len(json_files)} files")

    print("All done! Results saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()