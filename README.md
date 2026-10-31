# RobustPruner: Decoupled Relevance and Uncertainty for Efficient Visual Token Pruning in MLLMs

[📄 [Paper](https://github.com/oscakywalker/RobustPruner_2026/tree/main)] [🌐 [Project Page](https://oscakywalker.github.io/RobustPruner_2026/)]

## 📰 News

🔥 **[2026/10/28]** Our [RobustPruner Project Page](https://oscakywalker.github.io/RobustPruner_2026/) is updated!

🔥 **[2026/09/25]** [RobustPruner](https://github.com/oscakywalker/RobustPruner_2026/tree/main) is accepted by NeurIPS 2026! 🎉

## ⚙️ Setup

### 1. Qwen2.5-VL Environment

Use the `robustpruner_qwen` environment.

```bash
cd /RobustPruner_2026/RobustPruner_requirements
conda create -n robustpruner_qwen python=3.10
conda activate robustpruner_qwen

pip install torch==2.7.0
pip install flash-attn==2.7.4.post1
pip install -r qwen_requirements.txt
```

You may also need:

- `qwen-vl-utils[decord]`
- `lmms-eval==0.3.5`
- `vllm==0.9.0.1`
- `gradio==5.39.0`

These packages are useful for evaluation and quick testing. If needed, download the [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) model in the same environment.

### 2. Data Download

Training data uses the [Visual-CoT](https://huggingface.co/datasets/deepcs233/Visual-CoT/tree/main) dataset.

```bash
# Download the dataset
huggingface-cli download --repo-type dataset --local-dir datas deepcs233/Visual-CoT cot_images_tar_split

# Extract
cd /RobustPruner_2026/RobustPruner/datas/cot_images_tar_split
cat cot_images_* | tar -xvf - -C ../cot
cd ../..
```

After downloading, prepare the `gqa` images and place them under `datas/cot/gqa/`.

Check the directory with:

```bash
cd /RobustPruner_2026/RobustPruner
ls datas/cot/gqa | head
```

### 3. Prepared GQA Files

We already prepared a 20k random GQA subset and four text perturbation files. They are stored at:

```bash
/RobustPruner_2026/RobustPruner/uncertainty_dataset_construction/gqa_rand_seed42_20000.jsonl
/RobustPruner_2026/RobustPruner/uncertainty_dataset_construction/text_perturbations_concise_rephrase.json
/RobustPruner_2026/RobustPruner/uncertainty_dataset_construction/text_perturbations_synonym_replacement.json
/RobustPruner_2026/RobustPruner/uncertainty_dataset_construction/text_perturbations_sentence_structure_change.json
/RobustPruner_2026/RobustPruner/uncertainty_dataset_construction/text_perturbations_advanced_vocabulary.json
```

### 4. Qwen Inference and Training

#### 4.1 Build uncertainty training data

Run the script for the original data:

```bash
python /RobustPruner_2026/RobustPruner/uncertainty_dataset_construction/run_qwen_gqa_rand.py \
  --model /path/to/Qwen2.5-VL-7B-Instruct \
  --save-token-scores
```

Run the script for a perturbed version:

```bash
python /RobustPruner_2026/RobustPruner/uncertainty_dataset_construction/run_qwen_gqa_rand.py \
  --model /path/to/Qwen2.5-VL-7B-Instruct \
  --perturbation-json /RobustPruner_2026/RobustPruner/uncertainty_dataset_construction/text_perturbations_concise_rephrase.json \
  --use-perturbed-question \
  --save-token-scores
```

By default, this script uses:

- `input-jsonl`: `/RobustPruner_2026/RobustPruner/uncertainty_dataset_construction/gqa_rand_seed42_20000.jsonl`
- `image-root`: `/RobustPruner_2026/RobustPruner/datas/cot/gqa`
- `output-jsonl`: `/RobustPruner_2026/RobustPruner/robustpruner_qwen_eval/gqa_rand_seed42_20000_predictions.jsonl`

The output structure looks like this:

```bash
RobustPruner_2026/
└── RobustPruner/
    └── robustpruner_qwen_eval/
        ├── gqa_rand_seed42_20000_predictions.jsonl
        └── token_scores/
            └── gqa_rand_seed42_20000/
                ├── gqa_rand_seed42_20000_1.json
                ├── gqa_rand_seed42_20000_2.json
                ├── gqa_rand_seed42_20000_3.json
                └── ...
```

After the token scores are ready, open `gqa_layer_qwen_similarity.ipynb`. This notebook is used to build attention-based uncertainty and similarity-based uncertainty, and to visualize layer-wise attention and similarity behavior. Run the notebook step by step. The final output will be written to:

```bash
/RobustPruner_2026/RobustPruner/robustpruner_qwen_eval/token_scores/layer12_token_stats_per_doc
```

#### 4.2 Training

The current training script is:

```bash
/RobustPruner_2026/RobustPruner/scripts/train_qwen_rp.sh
```

It uses the following config:

```bash
/RobustPruner_2026/RobustPruner/train_configs/qwen2_5_7b_rp/qwen2_5_7b_rp_uncertainty_stage1.yaml
```

Start training with:

```bash
cd /RobustPruner_2026/RobustPruner
conda activate robustpruner_qwen

bash scripts/train_qwen_rp.sh
```

The default training output directory is:

```bash
/RobustPruner_2026/RobustPruner/output/qwen2_5_7b_rp_uncertainty_stage1_unbind_layer12_newest_testy
```

#### 4.3 Evaluation

Basic evaluation:

```bash
cd /RobustPruner_2026/RobustPruner_lmm_eval/evaluation
bash run_robustpruner_qwen_speed_test.sh
```

Other token compression baselines:

```bash
bash run_token_compression.sh
```

Perturbation evaluation:

Before running perturbation evaluation, enable the perturbation fields in these task YAML files:

- `/RobustPruner_2026/RobustPruner_lmm_eval/lmms-eval/lmms_eval/tasks/ai2d/ai2d_no_mask.yaml`
- `/RobustPruner_2026/RobustPruner_lmm_eval/lmms-eval/lmms_eval/tasks/ocrbench/ocrbench.yaml`
- `/RobustPruner_2026/RobustPruner_lmm_eval/lmms-eval/lmms_eval/tasks/mme/mme.yaml`
- `/RobustPruner_2026/RobustPruner_lmm_eval/lmms-eval/lmms_eval/tasks/gqa/gqa.yaml`
- `/RobustPruner_2026/RobustPruner_lmm_eval/lmms-eval/lmms_eval/tasks/chartqa/chartqa.yaml`

Set:

```yaml
text_perturb: true
perturbation_name: "perturbations_1"
```

Then run:

```bash
bash run_robustpruner_perturb_scores.sh
```
If you would like to test other three question-style text-perturbed test sets (perturbations_{2,3,4}), and a brand-new imperative-sentence-style test-perturbed test set (perturbations_5), please replace the `perturbation_name` in the yaml file.

#### 4.4 Projected DPP marginal selection

An independent inference entry point uses projected determinantal leverage
scores to construct the candidate pool, followed by the same low-uncertainty
refinement. The projected DPP-marginal alternative is a method to reduce the relatively high time and memory costs of constructing the full similarity matrix $S$ and performing sequential greedy MAP inference while maintaining close performances with original DPP selection.

```bash
cd /RobustPruner_2026/RobustPruner_lmm_eval/evaluation
conda activate robustpruner_qwen

MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct \
RP_MODULES_DIR=/path/to/robustpruner_checkpoint \
bash run_robustpruner_qwen_dpp_marginal.sh
```

The script defaults to GQA at 10% retention. To change the original setting and evaluate all five benchmarks at 20% retention, set:

```bash
TASKS="ai2d_no_mask chartqa gqa mme ocrbench" BUDGETS="0.2" \
MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct \
RP_MODULES_DIR=/path/to/robustpruner_checkpoint \
bash run_robustpruner_qwen_dpp_marginal.sh
```

The model registration is `qwen2_5_vl_rp_dpp_marginal`, backed by
`RobustPruner/transformers_rp_dpp_marginal/`. Model code defaults to the
original greedy DPP selection when `GP_DPP_PROJECTED_MARGINAL` is unset.
`run_robustpruner_qwen_dpp_marginal.sh` enables marginal selection by
default through `GP_DPP_PROJECTED_MARGINAL=True`; setting it to `False`
keeps greedy DPP selection.

### 5 Citation

If you find CDPruner useful for your research and applications, please cite using this BibTeX:
```bibtex
@inproceedings{zhu2026robustpruner,
  title = {RobustPruner: Decoupled Relevance and Uncertainty for Efficient Visual Token Pruning in MLLMs},
  author = {Zhu, Hanwei and Fu, Junhan and Zhang, Xi and Wang, Jiamang and Lin, Weisi},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2026},
}
```
