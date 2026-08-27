# SAMark

Official implementation for **SAMark: A Self-Anchored Text Watermarking with Paragraph-Level Paraphrase Robustness**.

## 1. Included Components

Core scripts:
- `samark_gen.py`: SAMark generation
- `samark_gen_unwatermarked.py`: unwatermarked reference generation for blind pairwise judge
- `samark_detect.py`: sentence-level detection
- `run_pipeline.sh`: full end-to-end pipeline

Attack scripts:
- `attack_gpt.py`
- `attack_trans.py`
- `attack_utils.py`

Evaluation scripts:
- `hierarchical_tpr.py`
- `diversity_eval.py`
- `jsd_eval.py`
- `llm_judge_eval.py`

Bundled dependencies from MarkLLM:
- `evaluation/tools/`
- `utils/openai_utils.py`
- `exceptions/exceptions.py`

## 2. Setup

Install dependencies:

```bash
pip install -r SAMark/requirements.txt
```

Install NLTK tokenizer data:

```bash
python -c "import nltk; nltk.download('punkt')"
```

If you run GPT-based attacks or LLM judge:

```bash
export OPENAI_API_KEY="your_api_key"
# optional override
export OPENAI_BASE_URL="https://xxx/v1"
```

## 3. Data and Models

Expected datasets:
- `./data/booksum-train-500`
- `./data/c4-val-500`

Example model paths:
- base model: `../models/Mistral-Small-3.1-24B-Base-2503`
- embedder: `../models/all-mpnet-base-v2`

## 4. End-to-End Usage

Run full pipeline:

```bash
bash SAMark/run_pipeline.sh MODEL_PATH EMBEDDER_PATH DATA_NAME START END [EPS] [REF_DIR] [RUN_LLM_JUDGE]
```

Arguments:
- `MODEL_PATH`: generation model path
- `EMBEDDER_PATH`: sentence embedder path
- `DATA_NAME`: `booksum` or `c4`
- `START`, `END`: sample range `[START, END)`
- `EPS` (optional): sentence selection scale, default `80`
- `REF_DIR` (optional): reference logs for JSD and pairwise judge
- `RUN_LLM_JUDGE` (optional): `1` enables pairwise judge, default `0`

Example:

```bash
bash SAMark/run_pipeline.sh \
  ../models/Mistral-Small-3.1-24B-Base-2503 \
  ../models/all-mpnet-base-v2 \
  booksum \
  0 500 \
  80 \
  ./logs/booksum_logs/Mistral-Small-3.1-24B-Base-2503_log/unwatermarked_ref \
  1
```

If `RUN_LLM_JUDGE=1` and `REF_DIR` is empty, the pipeline auto-generates an unwatermarked reference set at:

```text
./logs/{data_name}_logs/{model_name}_log/unwatermarked_ref/
```

## 5. Pipeline Stages

`run_pipeline.sh` executes:
1. SAMark generation
2. Sentence-level detection (baseline and tanh)
3. Attacks: Doc-P I / Doc-P II and Doc-T(GPT)
4. Sentence-level detection on attacked texts
5. Sentence-level TPR/FPR/AUC summary calibrated on 1,024 independent human texts,
   each truncated to its first 12 sentences
6. Diversity metrics
7. Optional JSD against reference
8. Optional blind pairwise LLM judge

## 6. Output Layout

Main run directory:

```text
./logs/{data_name}_logs/{model_name}_log/samark_{suffix}/
```

Contains:
- `{i}.json` generation logs
- `detect/` baseline sentence-level detection logs
- `detect_tanh30/` tanh sentence-level detection logs
- `attack/{attack_name}/` attacked logs and corresponding detection logs
- `attack_results/` and `attack_results_tanh30/` summary files
- diversity/JSD/judge outputs

## 7. Detection Log Schema

Detection output (`detect/{i}.json`):

```json
{
  "watermark_log": {
    "sentence_level": {
      "z_score": 3.8,
      "mean_score": 0.62
    }
  }
}
```

Detection logs contain raw scores only. `detect/results_full.csv` records the
human-null threshold, empirical FPR, TPR, and AUROC at each operating point.
`detect/results.csv` and `detect/results_wm.csv` use the shared compact schema:
`auroc`, `fpr1`, and `fpr5`, where the historical `fpr1`/`fpr5` names mean TPR
at 1%/5% FPR. No fixed normal-theory threshold is used.

## 8. Notes
| Dataset + Model | Recommended Parameters |  
|------|------|
| BookSum + Mistral-24B | `div0.35_nov0.2_sim0.8_op0.4` |  
| BookSum + Nemotron-9B | `div0.35_nov0.1_sim0.85_op0.5` |  
| C4 + Mistral-24B | `div0.15_nov0.15_sim0.85_op0.7` |  
| C4 + Nemotron-9B | `div0.3_nov0.15_sim0.85_op0.5` |  
