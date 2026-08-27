import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import json
import argparse

import torch
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from datasets import load_from_disk, Dataset
from vllm import LLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

from utils.sample import speculative_sample_next_sentence, sample_next_sentence_msignal, secret_mbit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--log_dir", type=str, default="logs/watermark_log")
    # Q1: explicit dataset path; --data_name kept for legacy convenience
    parser.add_argument("--data_path", type=str, default=None,
                        help="path to a HuggingFace dataset saved to disk (takes precedence over --data_name)")
    parser.add_argument("--data_name", type=str, default="c4",
                        help="legacy shorthand: 'c4' or 'booksum' (ignored when --data_path is set)")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--backend", type=str, default="vllm",
                        help="backend for generation: vllm | hf | openai")
    parser.add_argument("--api_base", type=str, default=None,
                        help="base URL for OpenAI-compatible API")
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--pivot", type=str, default="rand")
    parser.add_argument("--median_method", type=str, default="torch",
                        help="median estimator: torch | hd | kde | prior")
    parser.add_argument("--dedup", type=bool, default=False)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None,
                        help="exclusive end row; defaults to the full dataset length")
    parser.add_argument("--embedder_path", type=str, default=None)
    parser.add_argument("--fast_llm", type=str, default=None)
    parser.add_argument("--msig", type=int, default=0)
    parser.add_argument("--parallel", action="store_true",
                        help="use PMark's offline max-evidence channel selection")
    parser.add_argument("--device0", type=str, default="0")
    parser.add_argument("--device1", type=str, default="1")
    parser.add_argument("--max_new_sentences", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    args = parser.parse_args()

    # ── inference backend ────────────────────────────────────────────────────
    model = llm = fast_llm = None
    if args.backend == "hf":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype="auto"
        ).eval().to("cuda")
    elif args.backend == "vllm":
        llm = LLM(model=args.model_path, gpu_memory_utilization=0.8, tensor_parallel_size=1)
    # openai: nothing to init here

    # ── tokenizer + embedder ─────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if "sonar" in args.embedder_path:
        from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
        embedder = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder",
            tokenizer="text_sonar_basic_encoder",
            device=torch.device("cuda"),
            dtype=torch.float16,
        )
    else:
        embedder = SentenceTransformer(args.embedder_path, device="cuda")

    if args.fast_llm:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device1
        fast_llm = LLM(model=args.fast_llm, gpu_memory_utilization=0.8, tensor_parallel_size=1)

    if args.msig > 1:
        secret_mbit.set_signum(args.msig)
    if args.median_method == "hd":
        args.dedup = False

    # ── Q1: dataset loading ──────────────────────────────────────────────────
    if args.data_path:
        dataset = load_from_disk(args.data_path)
    elif args.data_name == "c4":
        dataset = load_from_disk("./data/c4-val-500")
    elif args.data_name == "booksum":
        dataset = load_from_disk("./data/booksum-train-500")
    else:
        raise ValueError(f"Unknown --data_name '{args.data_name}'; pass --data_path instead.")

    # default to the whole dataset; clamp an explicit slice to its bounds
    if args.end is None:
        args.end = len(dataset)
    args.end = min(args.end, len(dataset))
    args.start = max(0, args.start)

    os.makedirs(args.log_dir, exist_ok=True)
    with open(f"{args.log_dir}/config.json", "w", encoding="utf-8") as f:
        json.dump(args.__dict__, f, ensure_ascii=False, indent=4)

    # ── generation loop ──────────────────────────────────────────────────────
    for i in tqdm(range(args.start, args.end)):
        if os.path.exists(f"{args.log_dir}/{i}.json"):
            continue
        text = dataset[i]["text"]
        prompt = sent_tokenize(text)[0]

        generated_text = ""
        generated_log = {"prompt": prompt, "generated_text": "", "original_text": text, "log": []}
        it = 0

        while True:
            it += 1
            for _ in range(5):
                if args.fast_llm:
                    next_sample = speculative_sample_next_sentence(
                        embedder=embedder, prompt=prompt, num_samples=args.num_samples,
                        debug=True, savefig=None, pivot=args.pivot, sen_id=it,
                        median_method=args.median_method, llm=llm, fast_llm=fast_llm,
                    )
                else:
                    next_sample = sample_next_sentence_msignal(
                        model=model, tokenizer=tokenizer, embedder=embedder,
                        prompt=prompt, model_name=args.model_path,
                        num_samples=args.num_samples, debug=True,
                        openai_api_base=args.api_base, sen_id=it,
                        median_method=args.median_method, llm=llm, dedup=args.dedup,
                        parallel=args.parallel,
                        temperature=args.temperature, top_p=args.top_p,
                        repetition_penalty=args.repetition_penalty,
                    )
                if next_sample:
                    break
            if not next_sample:
                break

            next_sentence = next_sample["text"]
            generated_log["log"].append(next_sample)
            prompt = prompt + " " + next_sentence
            generated_text = generated_text + " " + next_sentence
            generated_log["generated_text"] = generated_text

            if it >= args.max_new_sentences:
                break

        print(f"Complete paragraph: {prompt}")
        print("******************")
        with open(f"{args.log_dir}/{i}.json", "w", encoding="utf-8") as f:
            json.dump(generated_log, f, ensure_ascii=False, indent=4)

    # ── Q3: export HF dataset ────────────────────────────────────────────────
    # Collect all JSON files in log_dir (covers partial/sharded runs).
    rows = []
    for fname in sorted(os.listdir(args.log_dir)):
        stem, extension = os.path.splitext(fname)
        if extension != ".json" or not stem.isdigit():
            continue
        idx = int(stem)
        with open(f"{args.log_dir}/{fname}", "r", encoding="utf-8") as f:
            log = json.load(f)
        rows.append({
            "pmark_idx":    idx,
            "text":         (log["prompt"] + log["generated_text"]).strip(),
            "original_text": log["original_text"],
            "prompt":       log["prompt"],
        })

    if rows:
        rows.sort(key=lambda r: r["pmark_idx"])
        hf_path = args.log_dir
        Dataset.from_list(rows).save_to_disk(hf_path)
        print(f"Saved HF dataset ({len(rows)} samples) → {hf_path}")
