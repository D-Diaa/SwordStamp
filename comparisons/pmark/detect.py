"""
PMark detection script.

Two modes
---------
--mode legacy  (default)
    Original per-file JSON mode.  Reads {log_dir}/{sub_dir}/{i}.json,
    writes detect results to {log_dir}/{sub_dir}/detect/{i}.json.

--mode hf
    HuggingFace dataset mode (bridges PMark detection onto our attacked datasets).
    --hf_dataset PATH   : dataset with a "text" column (watermarked).
                          Must also contain "pmark_idx" to align outputs.
                          If a "para_text" column exists it is detected too.
    --output_dir PATH   : where to write z_scores_*.npy and results.csv.
                          Defaults to {hf_dataset}/detect/.

Online detection uses PMark's analytical N(0,1) null. Offline/prior detection
uses an independent human-document null because the zero-median approximation
does not provide the online method's calibrated analytical null.
"""

import json
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import argparse

import numpy as np
import torch
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from scipy import stats as scipy_stats
from sentence_transformers import SentenceTransformer
from vllm import LLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk

from calibration import calibrated_threshold, empirical_auroc
from utils.detect import detect_paragraph
from utils.rand import secret_mbit


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_llm(args):
    if args.backend == "hf":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype="auto"
        ).eval().to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return None, model, tokenizer
    elif args.backend == "vllm":
        llm = LLM(model=args.model_name, gpu_memory_utilization=0.8, tensor_parallel_size=1)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return llm, None, tokenizer
    else:  # openai
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return None, None, tokenizer


def _load_embedder(embedder_name):
    if "sonar" in embedder_name:
        from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
        return TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder",
            tokenizer="text_sonar_basic_encoder",
            device=torch.device("cuda"),
            dtype=torch.float16,
        )
    return SentenceTransformer(embedder_name, device="cuda")


def _detect_text(text, args, llm, model, tokenizer, embedder, max_sentences=None):
    """Segment text and run detect_paragraph; return z_score (float)."""
    sentences = sent_tokenize(text)
    if max_sentences is not None:
        sentences = sentences[:max_sentences]
    if len(sentences) < 2:
        return 0.0
    result = detect_paragraph(
        sentences,
        num_samples=args.num_samples,
        api_base=getattr(args, "api_base", None),
        model_name=args.model_name,
        llm=llm,
        embedder=embedder,
        pivot=args.pivot,
        median_method=args.median_method,
        debug=False,
        dedup=args.dedup,
        msig=args.msig,
        model=model,
        tokenizer=tokenizer,
        soft_k=args.soft_k,
        soft_delta=args.soft_delta,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    return result["z_score"]


# ── legacy mode ──────────────────────────────────────────────────────────────

def run_legacy(args, llm, model, tokenizer, embedder):
    log_dir = args.log_dir
    sub_dir = args.sub_dir
    os.makedirs(f"{log_dir}/{sub_dir}/detect", exist_ok=True)
    if not os.path.exists(f"{log_dir}/{sub_dir}/detect/config.json"):
        with open(f"{log_dir}/{sub_dir}/detect/config.json", "w", encoding="utf-8") as f:
            json.dump(args.__dict__, f, ensure_ascii=False, indent=4)

    origin_pos = []
    watermark_pos = []

    for i in tqdm(range(args.start, args.end)):
        sample_path = f"{log_dir}/{sub_dir}/{i}.json"
        if not os.path.exists(sample_path):
            print(f"Sample {i} not found, skipping.")
            continue

        with open(sample_path, "r", encoding="utf-8") as f:
            generated_log = json.load(f)

        gen_sentences = [d["text"] for d in generated_log["log"]]
        gen_sentences.insert(0, generated_log["prompt"])

        watermark_log = detect_paragraph(
            gen_sentences,
            num_samples=args.num_samples,
            llm=llm,
            embedder=embedder,
            pivot=args.pivot,
            median_method=args.median_method,
            debug=True,
            dedup=args.dedup,
            msig=args.msig,
            model=model,
            tokenizer=tokenizer,
            soft_k=args.soft_k,
            soft_delta=args.soft_delta,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        watermark_pos.append(watermark_log["is_watermarked"])
        print(f"[{i}] watermarked={watermark_log['is_watermarked']}  z={watermark_log['z_score']:.3f}")

        origin_log = None
        if args.detect_origin:
            original_text = generated_log["original_text"]
            origin_ids = tokenizer(original_text, return_tensors="pt")["input_ids"]
            if origin_ids.shape[1] > args.max_context:
                origin_ids = origin_ids[:, : args.max_context - 500]
            original_text = tokenizer.decode(origin_ids[0])
            origin_sens = sent_tokenize(original_text)[:12]
            origin_log = detect_paragraph(
                origin_sens,
                num_samples=args.num_samples,
                api_base=getattr(args, "api_base", None),
                model_name=args.model_name,
                llm=llm,
                embedder=embedder,
                pivot=args.pivot,
                median_method=args.median_method,
                debug=True,
                dedup=args.dedup,
                msig=args.msig,
                model=model,
                tokenizer=tokenizer,
                soft_k=args.soft_k,
                soft_delta=args.soft_delta,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            )
            origin_pos.append(origin_log["is_watermarked"])

        with open(f"{log_dir}/{sub_dir}/detect/{i}.json", "w", encoding="utf-8") as f:
            json.dump({"origin_log": origin_log, "watermark_log": watermark_log}, f,
                      ensure_ascii=False, indent=4)

    print("Finished!")


# ── HF dataset mode (Q4 + Q5) ────────────────────────────────────────────────

def run_hf(args, llm, model, tokenizer, embedder):
    wm_dataset = load_from_disk(args.hf_dataset)
    has_para = "para_text" in wm_dataset.column_names
    has_idx = "pmark_idx" in wm_dataset.column_names

    output_dir = args.output_dir or os.path.join(args.hf_dataset, "detect")
    os.makedirs(output_dir, exist_ok=True)

    # Keep generation/attack metadata intact when detection writes into the same
    # HuggingFace dataset directory.
    with open(os.path.join(output_dir, "detection_config.json"), "w", encoding="utf-8") as f:
        json.dump(args.__dict__, f, ensure_ascii=False, indent=4)

    n = len(wm_dataset)
    if args.end is not None:
        n = min(n, args.end)
    start = args.start or 0

    z_text  = []
    z_para  = [] if has_para else None
    indices = []

    print(f"Detecting watermarked text ({n - start} samples)…")
    for i in tqdm(range(start, n)):
        row = wm_dataset[i]
        idx = row["pmark_idx"] if has_idx else i
        indices.append(idx)

        z_text.append(_detect_text(row["text"], args, llm, model, tokenizer, embedder))

        if has_para and row.get("para_text"):
            z_para.append(_detect_text(row["para_text"], args, llm, model, tokenizer, embedder))
        elif has_para:
            z_para.append(float("nan"))

    z_text = np.array(z_text, dtype=np.float32)
    np.save(os.path.join(output_dir, "z_scores_text.npy"), z_text)
    if has_para:
        z_para = np.array(z_para, dtype=np.float32)
        np.save(os.path.join(output_dir, "z_scores_para.npy"), z_para)

    # Online PMark retains its analytical N(0,1) null. Offline/prior PMark is
    # calibrated against 1,024 independent human documents. A precomputed null
    # may be supplied so every attacked directory reuses the unattacked cell's
    # identical scores.
    human_scores = None
    if args.median_method == "prior":
        if args.human_scores:
            human_scores = np.load(args.human_scores)
        else:
            if not args.human_text:
                raise ValueError("--human_text is required for offline/prior detection")
            human_dataset = load_from_disk(args.human_text)
            human_texts = human_dataset["text"][:min(args.human_n, len(human_dataset))]
            print(
                f"Scoring {len(human_texts)} independent human documents "
                f"(first {args.human_max_sentences} sentences)…"
            )
            human_scores = np.asarray([
                _detect_text(
                    text, args, llm, model, tokenizer, embedder,
                    max_sentences=args.human_max_sentences,
                )
                for text in tqdm(human_texts)
            ], dtype=np.float32)
        np.save(os.path.join(output_dir, "human_z_scores.npy"), human_scores)

    def _summarise(label, scores):
        valid = scores[~np.isnan(scores)]
        rows = []
        if args.median_method == "prior":
            null_label = f"human:{args.human_text}"
            auc_value = empirical_auroc(valid, human_scores)
        else:
            null_label = "analytical_N(0,1)"
            auc_value = float(np.mean(scipy_stats.norm.cdf(valid)))

        for target_fpr in (0.01, 0.05, 0.10):
            if args.median_method == "prior":
                threshold = calibrated_threshold(human_scores, target_fpr)
                empirical_fpr = float(np.mean(human_scores > threshold))
            else:
                threshold = float(scipy_stats.norm.ppf(1 - target_fpr))
                empirical_fpr = target_fpr
            rows.append({
                "column": label,
                "fpr": target_fpr,
                "tpr": round(float(np.mean(valid > threshold)), 4),
                "threshold": round(threshold, 4),
                "empirical_fpr": round(empirical_fpr, 4),
                "auroc": round(auc_value, 4),
                "n_wm": len(valid),
                "n_null": 0 if human_scores is None else len(human_scores),
                "null": null_label,
                "z_mean": round(float(np.mean(valid)), 4),
                "z_std": round(float(np.std(valid)), 4),
            })
        return rows

    all_rows = _summarise("text", z_text)
    if has_para:
        all_rows += _summarise("para_text", z_para)

    import csv
    full_path = os.path.join(output_dir, "results_full.csv")
    fieldnames = ["column", "fpr", "tpr", "threshold", "empirical_fpr", "auroc",
                  "n_wm", "n_null", "null", "z_mean", "z_std"]
    with open(full_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    # Pipeline-compatible single-row outputs. Historical column names fpr1/fpr5
    # mean TPR at the corresponding FPR operating point.
    def _compact(label):
        rows = {row["fpr"]: row for row in all_rows if row["column"] == label}
        if not rows:
            return None
        return {
            "auroc": rows[0.01]["auroc"],
            "fpr1": rows[0.01]["tpr"],
            "fpr5": rows[0.05]["tpr"],
            "threshold_source": rows[0.01]["null"],
        }

    def _write_row(path, row):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
            w.writerow(row)

    wm = _compact("text")
    if wm is not None:
        _write_row(os.path.join(output_dir, "results_wm.csv"),
                   {**wm, "mean_z": round(float(np.nanmean(z_text)), 4),
                    "human_mean_z": (0.0 if human_scores is None
                                     else round(float(np.nanmean(human_scores)), 4))})
    para = _compact("para_text") if has_para else None
    _write_row(os.path.join(output_dir, "results.csv"), para if para is not None else (wm or {}))

    null_name = "analytical N(0,1)" if human_scores is None else "independent human"
    print(f"\nResults → {full_path}  (+ results.csv / results_wm.csv, {null_name} null)")
    for r in all_rows:
        print(f"  [{r['column']}] FPR={r['fpr']}  TPR={r['tpr']}  "
              f"AUROC={r['auroc']}  null={r['null']}")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="legacy",
                        choices=["legacy", "hf"],
                        help="'legacy': per-file JSON mode; 'hf': HuggingFace dataset mode")

    # shared
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--embedder_name", type=str, default=None)
    parser.add_argument("--backend", type=str, default="vllm",
                        choices=["vllm", "hf", "openai"])
    parser.add_argument("--api_base", type=str, default=None)
    parser.add_argument("--pivot", type=str, default="rand")
    parser.add_argument("--median_method", type=str, default="torch")
    parser.add_argument("--dedup", type=bool, default=True)
    parser.add_argument("--msig", type=int, default=0)
    parser.add_argument("--soft_k", type=float, default=150.0)
    parser.add_argument("--soft_delta", type=float, default=0.001)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)

    # legacy mode
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--sub_dir", type=str, default=None)
    parser.add_argument("--detect_origin", type=bool, default=False)
    parser.add_argument("--max_context", type=int, default=1024)

    # hf mode (bridges PMark detection onto our attacked HF datasets)
    parser.add_argument("--hf_dataset", type=str, default=None,
                        help="path to HF dataset (text column = watermarked)")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--human_text", type=str, default=None)
    parser.add_argument("--human_scores", type=str, default=None,
                        help="optional cached offline human-null z-score array")
    parser.add_argument("--human_n", type=int, default=1024)
    parser.add_argument("--human_max_sentences", type=int, default=12)

    args = parser.parse_args()

    if args.median_method == "hd":
        args.dedup = False
    if args.msig > 1:
        secret_mbit.set_signum(args.msig)

    # prior mode needs only the embedder — skip expensive LLM load
    if args.median_method == "prior":
        llm, model, tokenizer = None, None, None
    else:
        llm, model, tokenizer = _load_llm(args)
    embedder = _load_embedder(args.embedder_name)

    if args.mode == "hf":
        if not args.hf_dataset:
            parser.error("--hf_dataset is required for --mode hf")
        run_hf(args, llm, model, tokenizer, embedder)
    else:
        if not args.log_dir or not args.sub_dir:
            parser.error("--log_dir and --sub_dir are required for --mode legacy")
        # legacy --end defaulted to 500 in original; keep that default
        if args.end is None:
            args.end = 500
        run_legacy(args, llm, model, tokenizer, embedder)
