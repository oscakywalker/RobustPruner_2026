import json
from pathlib import Path
from openai import OpenAI
from datasets import load_dataset
from tqdm import tqdm
import time

# 初始化客户端
client = OpenAI(
    api_key="sk",
    base_url="",
)

# 输出文件路径
OUTPUT_FILE = ""

def perturb_question(question):
    """使用API对问题进行同义替换"""
    try:
        completion = client.chat.completions.create(
            model="qwen3-vl-plus-2025-09-23",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""You may use similar but more formal and advanced words to rephrase the question.
Requirements:
- Keep the semantics of the question 
- Only return the new question you want to ask, no explanation or prefix

Question: {question}"""
                        }
                    ]
                }
            ],
            stream=True,
        )
        
        full_content = ""
        for chunk in completion:
            if chunk.choices[0].delta.content is None:
                continue
            full_content += chunk.choices[0].delta.content
        
        perturbed = full_content.strip()
        
        # 清理可能的前缀
        prefixes = ["rephrased question:", "rephrased:", "answer:"]
        for prefix in prefixes:
            if perturbed.lower().startswith(prefix):
                perturbed = perturbed[len(prefix):].strip()
        
        if not perturbed:
            return question
        
        return perturbed
        
    except Exception as e:
        print(f"[ERROR] {e}")
        return question

def main():
    print("Loading GQA dataset...")
    dataset = load_dataset("lmms-lab/GQA", "testdev_balanced_instructions", split="testdev", token=True)
    total = len(dataset)
    print(f"Total questions: {total}")
    
    perturbations = {}

    # tqdm 包裹整个 dataset
    for i, doc in enumerate(tqdm(dataset, desc="Paraphrasing", ncols=100)):
        question = doc["question"]
        
        # 如果已存在，跳过
        if question in perturbations:
            continue
        
        perturbed = perturb_question(question)
        perturbations[question] = perturbed
        
        # 每10个保存一次
        if (i + 1) % 10 == 0:
            output_path = Path(OUTPUT_FILE)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(perturbations, f, ensure_ascii=False, indent=2)

        # 避免API限流
        time.sleep(0.2)
    
    # 最终保存
    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(perturbations, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Done! Saved {len(perturbations)} perturbations to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
