"""
Pairwise Blind LLM-as-Judge Evaluation Script

For each sample, the judge compares watermarked text vs unwatermarked text
in a blind pairwise review (randomized A/B order) and decides which is better.

Supports both pairwise mode (--ref_dir) and legacy single-grading mode.

Usage (pairwise):
    python llm_judge_eval.py \
        --log_dir ./logs/booksum_logs/Mistral-Small-3.1-24B-Base-2503_log/KGW \
        --ref_dir ./logs/booksum_logs/Mistral-Small-3.1-24B-Base-2503_log/None \
        --start 0 --end 500 \
        --model gpt-3.5-turbo \
        --num_workers 4

Usage (legacy single-grade):
    python llm_judge_eval.py \
        --log_dir ./logs/booksum_logs/Mistral-Small-3.1-24B-Base-2503_log/KGW \
        --start 0 --end 500 \
        --text_key generated_text \
        --model gpt-3.5-turbo
"""

import os
import re
import json
import time
import random
import argparse
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from utils.openai_utils import OpenAIAPI

# ── Prompt templates per dataset ─────────────────────────────
SYSTEM_PROMPTS = {
    "booksum": (
        "You are a strict but fair teacher grading a student's book summary homework. "
        "You will be given the opening context of a book chapter and the student's summary. "
        "Grade the summary on a 0-100 scale based on: accuracy, completeness, coherence, "
        "writing quality, and absence of obvious errors (repetition, fabricated facts, "
        "irrelevant content, formatting artifacts). "
        "Return ONLY a JSON object: {\"grade\": <integer 0-100>, \"reason\": \"<one sentence>\"}"
    ),
    "c4": (
        "You are a strict but fair evaluator grading a text continuation. "
        "You will be given the opening sentence of a web document and a continuation. "
        "Grade the continuation on a 0-100 scale based on: topical relevance, coherence, "
        "writing quality, informativeness, and absence of obvious errors (repetition, "
        "fabricated facts, irrelevant content, formatting artifacts). "
        "Return ONLY a JSON object: {\"grade\": <integer 0-100>, \"reason\": \"<one sentence>\"}"
    ),
}

PAIRWISE_SYSTEM_PROMPTS = {
    "booksum": (
        "You are a strict but fair teacher comparing two students' book summary homework. "
        "You will be given the opening context of a book chapter and two summaries (A and B). "
        "Compare them on: accuracy, completeness, coherence, writing quality, and absence of "
        "obvious errors (repetition, fabricated facts, irrelevant content, formatting artifacts). "
        "Decide which summary is better overall. "
        "Return ONLY a JSON object: {\"winner\": \"A\" or \"B\" or \"tie\", \"reason\": \"<one sentence>\"}"
    ),
    "c4": (
        "You are a strict but fair evaluator comparing two text continuations. "
        "You will be given the opening sentence of a web document and two continuations (A and B). "
        "Compare them on: topical relevance, coherence, writing quality, informativeness, "
        "and absence of obvious errors (repetition, fabricated facts, irrelevant content, "
        "formatting artifacts). Decide which continuation is better overall. "
        "Return ONLY a JSON object: {\"winner\": \"A\" or \"B\" or \"tie\", \"reason\": \"<one sentence>\"}"
    ),
}

DEFAULT_DATASET = "booksum"


def get_prompts(dataset: str) -> tuple[str, str]:
    """Return (single_prompt, pairwise_prompt) for the given dataset."""
    ds = dataset if dataset in SYSTEM_PROMPTS else DEFAULT_DATASET
    return SYSTEM_PROMPTS[ds], PAIRWISE_SYSTEM_PROMPTS[ds]


def build_query(prompt: str, text: str) -> str:
    """Build the grading query for the LLM judge (legacy mode)."""
    return (
        f"## Book Chapter Opening\n{prompt}\n\n"
        f"## Student's Summary\n{text}\n\n"
        f"Grade this summary (0-100). Return ONLY JSON: {{\"grade\": <int>, \"reason\": \"...\"}}"
    )


def build_pairwise_query(prompt: str, text_a: str, text_b: str) -> str:
    """Build the pairwise comparison query for the LLM judge."""
    return (
        f"## Book Chapter Opening\n{prompt}\n\n"
        f"## Summary A\n{text_a}\n\n"
        f"## Summary B\n{text_b}\n\n"
        f"Which summary is better? Return ONLY JSON: {{\"winner\": \"A\" or \"B\" or \"tie\", \"reason\": \"...\"}}"
    )


def parse_grade(response: str) -> tuple[int | None, str]:
    """Extract grade and reason from LLM response. Returns (grade, reason)."""
    try:
        obj = json.loads(response)
        return int(obj["grade"]), obj.get("reason", "")
    except (json.JSONDecodeError, KeyError, ValueError):
        pass
    m = re.search(r'"grade"\s*:\s*(\d+)', response)
    if m:
        return int(m.group(1)), response
    m = re.search(r'\b(\d{1,3})\b', response)
    if m:
        val = int(m.group(1))
        if 0 <= val <= 100:
            return val, response
    return None, response


def parse_pairwise(response: str) -> tuple[str | None, str]:
    """Extract winner from pairwise LLM response. Returns (winner, reason)."""
    try:
        obj = json.loads(response)
        winner = obj.get("winner", "").strip().upper()
        if winner in ("A", "B", "TIE"):
            return winner, obj.get("reason", "")
    except (json.JSONDecodeError, KeyError, ValueError):
        pass
    # Fallback regex
    m = re.search(r'"winner"\s*:\s*"([ABab]|[Tt]ie)"', response)
    if m:
        return m.group(1).strip().upper(), response
    return None, response


def grade_single(api: OpenAIAPI, prompt: str, text: str, max_retries: int = 3) -> dict:
    """Grade a single sample with retries (legacy mode)."""
    query = build_query(prompt, text)
    for attempt in range(max_retries):
        try:
            response = api.get_result(query)
            grade, reason = parse_grade(response)
            if grade is not None:
                return {"grade": grade, "reason": reason, "raw_response": response}
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
            else:
                return {"grade": None, "reason": f"Error: {str(e)}", "raw_response": ""}
    return {"grade": None, "reason": "Failed to parse grade", "raw_response": ""}


def pairwise_single(api: OpenAIAPI, prompt: str, wm_text: str, ref_text: str,
                     index: int, max_retries: int = 3) -> dict:
    """Pairwise blind comparison of watermarked vs unwatermarked text."""
    # Randomize order to eliminate position bias
    rng = random.Random(index)  # deterministic per sample
    if rng.random() < 0.5:
        text_a, text_b = wm_text, ref_text
        wm_position = "A"
    else:
        text_a, text_b = ref_text, wm_text
        wm_position = "B"

    query = build_pairwise_query(prompt, text_a, text_b)
    for attempt in range(max_retries):
        try:
            response = api.get_result(query)
            winner, reason = parse_pairwise(response)
            if winner is not None:
                # Map back: did the watermarked text win?
                if winner == "TIE":
                    wm_result = "tie"
                elif winner == wm_position:
                    wm_result = "win"
                else:
                    wm_result = "lose"
                return {
                    "winner_raw": winner,
                    "wm_position": wm_position,
                    "wm_result": wm_result,
                    "reason": reason,
                    "raw_response": response,
                }
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
            else:
                return {"wm_result": None, "reason": f"Error: {str(e)}", "raw_response": ""}
    return {"wm_result": None, "reason": "Failed to parse", "raw_response": ""}


def load_samples(log_dir: str, start: int, end: int, text_key: str) -> list[dict]:
    """Load samples from per-sample JSON logs."""
    samples = []
    for i in range(start, end):
        path = os.path.join(log_dir, f"{i}.json")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        text = data.get(text_key, "")
        prompt = data.get("prompt", "")
        if text and prompt:
            samples.append({"index": i, "prompt": prompt, "text": text})
    return samples


def load_paired_samples(wm_dir: str, ref_dir: str, start: int, end: int,
                        text_key: str = "generated_text") -> list[dict]:
    """Load paired (watermarked, unwatermarked) samples."""
    pairs = []
    for i in range(start, end):
        wm_path = os.path.join(wm_dir, f"{i}.json")
        ref_path = os.path.join(ref_dir, f"{i}.json")
        if not os.path.exists(wm_path) or not os.path.exists(ref_path):
            continue
        with open(wm_path, "r", encoding="utf-8") as f:
            wm_data = json.load(f)
        with open(ref_path, "r", encoding="utf-8") as f:
            ref_data = json.load(f)
        wm_text = wm_data.get(text_key, "")
        ref_text = ref_data.get(text_key, "")
        prompt = wm_data.get("prompt", "")
        if wm_text and ref_text and prompt:
            pairs.append({
                "index": i, "prompt": prompt,
                "wm_text": wm_text, "ref_text": ref_text,
            })
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Blind LLM-as-judge evaluation")
    parser.add_argument("--log_dir", type=str, required=True,
                        help="Directory containing watermarked per-sample JSON logs")
    parser.add_argument("--ref_dir", type=str, default=None,
                        help="Directory containing unwatermarked (None) per-sample JSON logs. "
                             "If provided, runs pairwise blind review mode.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=500)
    parser.add_argument("--text_key", type=str, default="generated_text",
                        choices=["generated_text", "original_text"])
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo",
                        choices=["gpt-3.5-turbo", "gpt-4"],
                        help="OpenAI model for judging")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of parallel API workers")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to evaluate (for cost control). None = all.")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional path to save result JSON")
    parser.add_argument("--dataset", type=str, default="booksum",
                        choices=["booksum", "c4"],
                        help="Dataset name, used to select appropriate grading prompt")
    args = parser.parse_args()

    system_prompt, pairwise_system_prompt = get_prompts(args.dataset)

    # ── Pairwise mode ───────────────────────────────────────
    if args.ref_dir is not None:
        out_path = args.output or os.path.join(args.log_dir, "llm_judge_pairwise.json")
        if os.path.exists(out_path):
            print(f"Result already exists: {out_path}, loading existing results...")
            with open(out_path, "r", encoding="utf-8") as f:
                output = json.load(f)
            
            # Display results
            total_valid = output.get("num_valid", 0)
            failed = output.get("num_failed", 0)
            wins = output.get("wm_wins", 0)
            losses = output.get("wm_losses", 0)
            ties = output.get("ties", 0)
            win_rate = output.get("win_rate", 0)
            lose_rate = output.get("lose_rate", 0)
            tie_rate = output.get("tie_rate", 0)
            
            print(f"\n{'='*60}")
            print(f"Pairwise Blind Review Results ({total_valid} valid, {failed} failed)")
            print(f"{'='*60}")
            print(f"  WM Win   : {wins} ({win_rate:.1f}%)")
            print(f"  WM Lose  : {losses} ({lose_rate:.1f}%)")
            print(f"  Tie      : {ties} ({tie_rate:.1f}%)")
            print(f"{'='*60}")
            return

        pairs = load_paired_samples(args.log_dir, args.ref_dir, args.start, args.end, args.text_key)
        if not pairs:
            print("No paired samples found.")
            return

        if args.max_samples and len(pairs) > args.max_samples:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(pairs), size=args.max_samples, replace=False)
            pairs = [pairs[i] for i in sorted(indices)]

        print(f"Pairwise evaluating {len(pairs)} samples with {args.model} (workers={args.num_workers})")

        api = OpenAIAPI(model=args.model, temperature=0.2, system_content=pairwise_system_prompt)
        results = {}

        if args.num_workers <= 1:
            for p in tqdm(pairs, desc="Pairwise"):
                r = pairwise_single(api, p["prompt"], p["wm_text"], p["ref_text"], p["index"])
                results[p["index"]] = r
        else:
            with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
                futures = {
                    pool.submit(pairwise_single, api, p["prompt"], p["wm_text"], p["ref_text"], p["index"]): p["index"]
                    for p in pairs
                }
                for future in tqdm(as_completed(futures), total=len(futures), desc="Pairwise"):
                    idx = futures[future]
                    results[idx] = future.result()

        # Aggregate
        wins = sum(1 for r in results.values() if r.get("wm_result") == "win")
        losses = sum(1 for r in results.values() if r.get("wm_result") == "lose")
        ties = sum(1 for r in results.values() if r.get("wm_result") == "tie")
        failed = sum(1 for r in results.values() if r.get("wm_result") is None)
        total_valid = wins + losses + ties

        win_rate = wins / total_valid * 100 if total_valid else 0
        lose_rate = losses / total_valid * 100 if total_valid else 0
        tie_rate = ties / total_valid * 100 if total_valid else 0

        print(f"\n{'='*60}")
        print(f"Pairwise Blind Review Results ({total_valid} valid, {failed} failed)")
        print(f"{'='*60}")
        print(f"  WM Win   : {wins} ({win_rate:.1f}%)")
        print(f"  WM Lose  : {losses} ({lose_rate:.1f}%)")
        print(f"  Tie      : {ties} ({tie_rate:.1f}%)")
        print(f"{'='*60}")

        output = {
            "log_dir": args.log_dir,
            "ref_dir": args.ref_dir,
            "model": args.model,
            "num_samples": len(results),
            "num_valid": total_valid,
            "num_failed": failed,
            "wm_wins": wins,
            "wm_losses": losses,
            "ties": ties,
            "win_rate": win_rate,
            "lose_rate": lose_rate,
            "tie_rate": tie_rate,
            "per_sample": {str(k): v for k, v in results.items()},
        }
        with open(out_path, "w") as f:
            json.dump(output, f, indent=4)
        print(f"Result saved to: {out_path}")
        return

    # ── Legacy single-grade mode ────────────────────────────
    out_path = args.output or os.path.join(args.log_dir, f"llm_judge_{args.text_key}.json")
    if os.path.exists(out_path):
        print(f"Result already exists: {out_path}, loading existing results...")
        with open(out_path, "r", encoding="utf-8") as f:
            output = json.load(f)
        
        # Display results
        n_valid = output.get("num_valid", 0)
        n_failed = output.get("num_failed", 0)
        avg = output.get("avg_grade", 0)
        med = output.get("median_grade", 0)
        std = output.get("std_grade", 0)
        
        print(f"\n{'='*60}")
        print(f"Blind LLM-as-Judge Results ({n_valid} graded, {n_failed} failed)")
        print(f"{'='*60}")
        print(f"  Model       : {output.get('model', 'N/A')}")
        print(f"  text_key    : {output.get('text_key', 'N/A')}")
        print(f"  Avg Grade   : {avg:.1f}")
        print(f"  Median      : {med:.1f}")
        print(f"  Std         : {std:.1f}")
        if "per_sample" in output and output["per_sample"]:
            grades = [r["grade"] for r in output["per_sample"].values() if r.get("grade") is not None]
            if grades:
                print(f"  Min / Max   : {min(grades)} / {max(grades)}")
        print(f"{'='*60}")
        # return

    samples = load_samples(args.log_dir, args.start, args.end, args.text_key)
    if not samples:
        print("No samples found. Check --log_dir, --start, --end, and --text_key.")
        return

    if args.max_samples and len(samples) > args.max_samples:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(samples), size=args.max_samples, replace=False)
        samples = [samples[i] for i in sorted(indices)]

    print(f"Evaluating {len(samples)} samples with {args.model} (workers={args.num_workers})")

    api = OpenAIAPI(model=args.model, temperature=0.2, system_content=system_prompt)
    results = {}

    if args.num_workers <= 1:
        for s in tqdm(samples, desc="Grading"):
            r = grade_single(api, s["prompt"], s["text"])
            results[s["index"]] = r
    else:
        with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
            futures = {
                pool.submit(grade_single, api, s["prompt"], s["text"]): s["index"]
                for s in samples
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Grading"):
                idx = futures[future]
                results[idx] = future.result()

    grades = [r["grade"] for r in results.values() if r["grade"] is not None]
    n_valid = len(grades)
    n_failed = len(results) - n_valid

    if grades:
        avg = float(np.mean(grades))
        med = float(np.median(grades))
        std = float(np.std(grades))
    else:
        avg = med = std = 0.0

    print(f"\n{'='*60}")
    print(f"Blind LLM-as-Judge Results ({n_valid} graded, {n_failed} failed)")
    print(f"{'='*60}")
    print(f"  Model       : {args.model}")
    print(f"  text_key    : {args.text_key}")
    print(f"  Avg Grade   : {avg:.1f}")
    print(f"  Median      : {med:.1f}")
    print(f"  Std         : {std:.1f}")
    if grades:
        print(f"  Min / Max   : {min(grades)} / {max(grades)}")
    print(f"{'='*60}")

    output = {
        "log_dir": args.log_dir,
        "text_key": args.text_key,
        "model": args.model,
        "num_samples": len(results),
        "num_valid": n_valid,
        "num_failed": n_failed,
        "avg_grade": avg,
        "median_grade": med,
        "std_grade": std,
        "per_sample": {str(k): v for k, v in results.items()},
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=4)
    print(f"Result saved to: {out_path}")


if __name__ == "__main__":
    main()
