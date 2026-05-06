import csv
import json
import os
import threading
import time

import numpy as np
import requests
from openai import OpenAI
from tqdm import tqdm

API_TYPE = os.getenv("API_TYPE", "openai")

from loguru import logger as eval_logger


def _normalize_openai_base_url(api_url):
    api_url = api_url.rstrip("/")
    if api_url.endswith("/chat/completions"):
        return api_url[: -len("/chat/completions")]
    return api_url


def _resolve_api_config():
    if API_TYPE == "azure":
        api_url = os.getenv("AZURE_ENDPOINT") or os.getenv("AZURE_OPENAI_API_BASE") or "https://api.cognitive.microsoft.com/sts/v1.0/issueToken"
        api_key = os.getenv("AZURE_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY", "")
        headers = {
            "api-key": api_key,
            "Content-Type": "application/json",
        }
    else:
        api_url = os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_API_URL") or "https://api.openai.com/v1"
        api_key = os.getenv("OPENAI_API_KEY", "")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
    return api_url, api_key, headers


def _build_openai_client():
    api_url, api_key, _ = _resolve_api_config()
    if not api_key:
        eval_logger.error("Judge API key is empty. Set OPENAI_API_KEY before running HallusionBench evaluation.")
        return None
    return OpenAI(
        api_key=api_key,
        base_url=_normalize_openai_base_url(api_url),
    )


def _request_judge_completion(payload, retries):
    if API_TYPE == "azure":
        api_url, api_key, headers = _resolve_api_config()
        if not api_key:
            eval_logger.error("Judge API key is empty. Set AZURE_API_KEY or AZURE_OPENAI_API_KEY before running HallusionBench evaluation.")
            return None

        last_error = None
        for attempt in range(retries):
            try:
                response = requests.post(api_url, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                last_error = e
                eval_logger.info(f"Attempt {attempt + 1} failed with error: {str(e)}")
                if attempt < retries - 1:
                    time.sleep(5)

        eval_logger.error(f"All {retries} attempts failed. Last error message: {str(last_error)}")
        return None

    client = _build_openai_client()
    if client is None:
        return None

    last_error = None
    for attempt in range(retries):
        try:
            return client.chat.completions.create(**payload)
        except Exception as e:
            last_error = e
            eval_logger.info(f"Attempt {attempt + 1} failed with error: {str(e)}")
            if attempt < retries - 1:
                time.sleep(5)

    eval_logger.error(f"All {retries} attempts failed. Last error message: {str(last_error)}")
    return None


def evaluate_by_chatgpt(data, output_entry, correctness_entry, gpt_model="gpt-4", load_json=False, save_json_path="./hallusion_output.json", retries=3):
    if load_json and os.path.exists(save_json_path):
        with open(save_json_path, "r") as f:
            output = json.load(f)
    else:
        output = []
    for sample in tqdm(data[len(output) :], desc="Eval by GPT"):
        prompt = "Imagine you are an intelligent teacher. Thoroughly read the question, reference answer and the prediction answer to ensure a clear understanding of the information provided. Assess the correctness of the predictions. "
        prompt += 'If the prediction answer does not conflict with the reference answer, please generate “correct”. If the prediction answer conflict with the reference answer, please generate “incorrect”. If the prediction answer is unclear about the answer, please generate "unclear". \n\n Question:'
        prompt += sample["question"]
        prompt += "\nReference answer: "
        prompt += sample["gt_answer_details"]
        prompt += "\nPrediction answer:"
        prompt += sample[output_entry]
        prompt += "\nOutput:"

        # https://github.com/openai/openai-python/issues/322#issuecomment-1767841683
        judge_model = os.getenv("HALLUSION_BENCH_JUDGE_MODEL", os.getenv("OPENAI_MODEL", gpt_model))
        messages = [{"role": "user", "content": prompt}]
        payload = {
            "model": judge_model,
            "messages": messages,
            "max_tokens": 16,
        }
        response = _request_judge_completion(payload, retries)
        try:
            if API_TYPE == "azure":
                output_text = response["choices"][0]["message"]["content"]
            else:
                output_text = response.choices[0].message.content
        except Exception as e:
            eval_logger.info(f"Get error {str(e)} when extracting response")
            output_text = "unclear"

        if "incorrect" in output_text.lower():
            gpt_correctness = "0"

        elif "correct" in output_text.lower():
            gpt_correctness = "1"
        else:
            gpt_correctness = "2"

        sample[correctness_entry] = gpt_correctness
        sample["gpt_answer"] = prompt + output_text

        output.append(sample)

        with open(save_json_path, "w") as f:
            json.dump(output, f, indent=4)

    return output


def check_same_by_chatgpt(data, output_entry, gpt_model="gpt-4", load_json=False, save_json_path="./hallusion_output.json", retries=3):
    orig_response = {}

    for r in data:
        if str(r["figure_id"]) == "0":
            key = "_".join([r["category"], r["subcategory"], str(r["set_id"]), str(r["question_id"])])
            orig_response[key] = r[output_entry]

    for sample in tqdm(data, desc="Check same by GPT"):
        if "same" not in sample.keys():
            key = "_".join([sample["category"], sample["subcategory"], str(sample["set_id"]), str(sample["question_id"])])
            response2 = orig_response[key]

            prompt = "Imagine you are an intelligent teacher. Thoroughly read the two responses to two different questions. Assess the consistency of the information provided within those two responses. "
            prompt += "You do not know the specific questions, but you can asssess the consistency among the two responses by checking for logical conflicts if both responses are correct. "
            prompt += 'If response1 does not conflict with response2, please generate “same”. Otherwise, generate "different". \n\n response1:'
            prompt += sample[output_entry]
            prompt += "\nresponse2: "
            prompt += response2
            prompt += "\nOutput:"

            # https://github.com/openai/openai-python/issues/322#issuecomment-1767841683
            judge_model = os.getenv("HALLUSION_BENCH_JUDGE_MODEL", os.getenv("OPENAI_MODEL", gpt_model))
            messages = [{"role": "user", "content": prompt}]
            payload = {
                "model": judge_model,
                "messages": messages,
                "max_tokens": 16,
            }
            response = _request_judge_completion(payload, retries)

            try:
                if API_TYPE == "azure":
                    output_text = response["choices"][0]["message"]["content"]
                else:
                    output_text = response.choices[0].message.content
            except Exception as e:
                eval_logger.info(f"Get error {str(e)} when extracting response")
                output_text = "different"

            gpt_same = "0"

            if "same" in output_text.lower():
                gpt_same = "1"

            elif "different" in output_text.lower():
                gpt_same = "0"

            sample["same"] = gpt_same

            with open(save_json_path, "w") as f:
                json.dump(data, f, indent=4)

    return data


def assign_correctness(data_arr, correctness_entry):
    for r in data_arr:
        assert int(r[correctness_entry]) == 0 or int(r[correctness_entry]) == 1 or int(r[correctness_entry]) == 2
        if r["category"] == "VS" and int(r["figure_id"]) == 0:  # if there is no visual supplement and the model does not know, count it as correct
            r["correct"] = 1 if int(r[correctness_entry]) == 1 or int(r[correctness_entry]) == 2 else 0
        else:
            r["correct"] = 1 if int(r[correctness_entry]) == 1 else 0
    return data_arr


def get_eval_fig(data):  # per figure
    eval_fig_dict = dict()

    for r in data:
        if r["category"] == "VS" and str(r["figure_id"]) == "0":  # no figure
            continue
        name = "_".join([r["category"], r["subcategory"], str(r["set_id"]), str(r["figure_id"])])
        if name in eval_fig_dict:
            c, t = eval_fig_dict[name]
            eval_fig_dict[name] = (c + r["correct"], t + 1)
        else:
            eval_fig_dict[name] = (r["correct"], 1)

    eval_fig_stat = {}
    eval_fig_stat["note"] = "all accuracy per image (consistency test)"
    eval_fig_stat["total"] = len(eval_fig_dict.keys())
    eval_fig_stat["correct"] = 0
    eval_fig_stat["wrong"] = 0
    eval_fig_stat["inconsistent"] = 0
    eval_fig_stat["score"] = 0

    for v in eval_fig_dict.values():
        if v[0] == v[1]:
            eval_fig_stat["correct"] += 1
        elif v[0] == 0:
            eval_fig_stat["wrong"] += 1
        else:
            eval_fig_stat["inconsistent"] += 1
        eval_fig_stat["score"] += v[0] / v[1]

    eval_fig_stat["score"] = eval_fig_stat["score"] / eval_fig_stat["total"]
    return eval_fig_stat


def get_eval_all(data, model_correctness_entry):  # per question
    eval_all_dict = dict()
    eval_all_stat = {}
    eval_all_stat["LH"] = 0
    eval_all_stat["VI"] = 0
    eval_all_stat["Mix"] = 0

    for r in data:
        name = "_".join([r["category"], r["subcategory"], str(r["set_id"]), str(r["figure_id"]), str(r["question_id"])])
        assert name not in eval_all_dict

        eval_all_dict[name] = r["correct"]

        if str(r["category"]) == "VD":  # VD
            if str(r["figure_id"]) == "0":
                if str(r[model_correctness_entry]) == "0" or str(r[model_correctness_entry]) == "2":
                    eval_all_stat["VI"] += 1
            else:
                if str(r[model_correctness_entry]) == "0":
                    eval_all_stat["Mix"] += 1
                elif str(r[model_correctness_entry]) == "2":
                    eval_all_stat["VI"] += 1
        else:  # VS
            if str(r["visual_input"]) == "0":  # no visual
                if str(r[model_correctness_entry]) == "0":
                    eval_all_stat["LH"] += 1
            else:  # original visual or modified visual (isual_input == 1 or 2)
                if str(r[model_correctness_entry]) == "0":
                    eval_all_stat["Mix"] += 1
                elif str(r[model_correctness_entry]) == "2":
                    eval_all_stat["VI"] += 1

    eval_all_stat["note"] = "all accuracy per question"
    eval_all_stat["total"] = len(eval_all_dict.keys())
    eval_all_stat["correct"] = np.count_nonzero(list(eval_all_dict.values()))
    eval_all_stat["wrong"] = eval_all_stat["total"] - eval_all_stat["correct"]

    return eval_all_stat


def get_eval_pair_all(data, model_correctness_entry):  # per question pair
    orig_correctness = dict()
    counter = 0
    lh_counter = 0
    vi_counter = 0
    both_counter = 0

    for r in data:
        if str(r["figure_id"]) == "0":
            key = "_".join([r["category"], r["subcategory"], str(r["set_id"]), str(r["question_id"])])
            orig_correctness[key] = r[model_correctness_entry]

    get_eval_pair_dict = dict()

    for r in data:
        name = "_".join([r["category"], r["subcategory"], str(r["set_id"]), str(r["question_id"])])
        if name in get_eval_pair_dict:
            c, t = get_eval_pair_dict[name]
            get_eval_pair_dict[name] = (c + r["correct"], t + 1)
        else:
            get_eval_pair_dict[name] = (r["correct"], 1)
        counter += 1

    eval_all_pair_stat = {}
    eval_all_pair_stat["note"] = "all accuracy per question pair"
    eval_all_pair_stat["total"] = len(get_eval_pair_dict.keys())
    eval_all_pair_stat["total_q"] = counter
    eval_all_pair_stat["correct"] = 0
    eval_all_pair_stat["wrong"] = 0
    eval_all_pair_stat["LH"] = 0
    eval_all_pair_stat["VI"] = 0
    eval_all_pair_stat["Mix"] = 0

    eval_all_pair_stat["LH_cg"] = lh_counter
    eval_all_pair_stat["VI_cg"] = vi_counter
    eval_all_pair_stat["Mix_cg"] = both_counter

    # for v in get_eval_pair_dict.values():
    #     if v[0] == v[1]:
    #         eval_all_pair_stat["correct"] += 1
    #     else:
    #         eval_all_pair_stat["wrong"] += 1

    # for v in get_analysis_pair_dict.values():
    #     if v[0] > 0 and v[1] > 0:
    #         eval_all_pair_stat["Mix"] += 1
    #     elif v[0] > 0:
    #         eval_all_pair_stat["LH"] += 1
    #     elif v[1] > 0:
    #         eval_all_pair_stat["VI"] += 1

    for k in get_eval_pair_dict.keys():
        v = get_eval_pair_dict[k]
        if v[0] == v[1]:
            eval_all_pair_stat["correct"] += 1
        else:
            eval_all_pair_stat["wrong"] += 1

    return eval_all_pair_stat
