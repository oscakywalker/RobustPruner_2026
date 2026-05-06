import argparse
import json
from pathlib import Path


def format_flops(flops):
    if flops is None:
        return None
    if flops >= 1e12:
        return f"{flops / 1e12:.2f} T"
    if flops >= 1e9:
        return f"{flops / 1e9:.2f} B"
    if flops >= 1e6:
        return f"{flops / 1e6:.2f} M"
    return f"{flops:.2f} F"

def process_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    metrics = {
        "after_generation_memory": [],
        "visual_token_num": [],
        "input_token_num": [],
        "generated_token_num": [],
        "kv_cache_size": [],
        "reduced_visual_token_num": [],
        "reduced_input_token_num": [],
        "prefill_flops": [],
        "decode_flops": [],
        "kv_cache_length": [],
    }
    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("{"):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                record = None
            if isinstance(record, dict):
                mapping = {
                    "after_generation_memory": "after_generation_memory",
                    "visual_token_num": "visual_token_num",
                    "input_token_num": "input_token_num",
                    "generated_token_num": "generated_token_num",
                    "kv_cache_size": "kv_cache_size",
                    "reduced_visual_token_num": "reduced_visual_token_num",
                    "reduced_input_token_num": "reduced_input_token_num",
                    "prefill_flops": "prefill_flops",
                    "decode_flops": "decode_flops",
                    "kv_cache_length": "kv_cache_length",
                }
                for record_key, metric_key in mapping.items():
                    value = record.get(record_key)
                    if value is not None:
                        metrics[metric_key].append(value)
                continue
    
    return metrics

def average(values):
    return None if len(values) == 0 else sum(values) / len(values)

def max_or_none(values):
    return None if len(values) == 0 else max(values)

def extract_score(results_json_path):
    if not results_json_path:
        return None
    path = Path(results_json_path)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    results = data.get("results", {})
    for task_metrics in results.values():
        for key, value in task_metrics.items():
            if key == "alias" or "stderr" in key:
                continue
            if isinstance(value, (int, float)):
                return key, float(value)
    return None

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="input path arg")
    parser.add_argument('--path', type=str, help='path to the log', default = './log_eval.log')
    parser.add_argument('--results_json', type=str, help='path to results json', default = '')
    args = parser.parse_args()

    metrics = process_file(args.path)
    if not any(len(values) > 0 for values in metrics.values()):
        print(f"No timing records found in {args.path}")
    else:
        avg_visual_token_num = average(metrics["visual_token_num"])
        avg_input_token_num = average(metrics["input_token_num"])
        avg_generated_token_num = average(metrics["generated_token_num"])
        avg_kv_cache_size = average(metrics["kv_cache_size"])
        avg_reduced_visual_token_num = average(metrics["reduced_visual_token_num"])
        avg_reduced_input_token_num = average(metrics["reduced_input_token_num"])
        avg_prefill_flops = average(metrics["prefill_flops"])
        avg_decode_flops = average(metrics["decode_flops"])
        avg_kv_cache_length = average(metrics["kv_cache_length"])
        max_gpu_memory_gb = max_or_none(metrics["after_generation_memory"])

        if max_gpu_memory_gb is not None:
            print(f"Max GPU memory: {max_gpu_memory_gb/(1024*1024*1024)} GB")
        if avg_visual_token_num is not None:
            print(f"Average visual token num: {avg_visual_token_num}")
        if avg_input_token_num is not None:
            print(f"Average input token num: {avg_input_token_num}")
        if avg_reduced_visual_token_num is not None:
            print(f"Average reduced visual token num: {avg_reduced_visual_token_num}")
        if avg_reduced_input_token_num is not None:
            print(f"Average reduced input token num: {avg_reduced_input_token_num}")
        if avg_generated_token_num is not None:
            print(f"Average generated token num: {avg_generated_token_num}")
        if avg_prefill_flops is not None:
            print(f"Average prefill FLOPs: {format_flops(avg_prefill_flops)}")
        if avg_decode_flops is not None:
            print(f"Average decode FLOPs: {format_flops(avg_decode_flops)}")
        if avg_kv_cache_length is not None:
            print(f"Average KV cache length: {avg_kv_cache_length}")
        if avg_kv_cache_size is not None:
            print(f"Average KV cache size: {avg_kv_cache_size} MB")

    score = extract_score(args.results_json)
    if score is not None:
        metric_name, value = score
        print(f"Score ({metric_name}): {value}")
