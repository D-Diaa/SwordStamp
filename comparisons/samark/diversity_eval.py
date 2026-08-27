"""
Diversity Metrics Evaluation Script (§3.2 of SAMARK_REVIEW.md)

Computes per-sample and aggregate diversity metrics on generated text:
  - 4-gram Repeat%: fraction of 4-grams that are repeated within a sample
  - Sentence Dup%: fraction of sentences that are exact duplicates within a sample
  - Distinct-1: ratio of unique unigrams to total unigrams
  - Distinct-2: ratio of unique bigrams to total bigrams

Reads per-sample JSON logs (same format as ppl_eval.py / gen.py output).

Usage:
    python diversity_eval.py \
        --log_dir ./logs/booksum_logs/Mistral-7B-v0.1_log/KGW \
        --start 0 --end 500 \
        --text_key generated_text

    # Evaluate original (human) text:
    python diversity_eval.py \
        --log_dir ./logs/booksum_logs/Mistral-7B-v0.1_log/KGW \
        --start 0 --end 500 \
        --text_key original_text
"""

import os
import json
import argparse
import numpy as np
from collections import Counter
from nltk.tokenize import sent_tokenize


# ─── Per-sample metric functions ─────────────────────────────────────────────

def ngram_repeat_rate(text: str, n: int = 4) -> float:
    """Fraction of n-grams that appear more than once in the text."""
    tokens = text.lower().split()
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    total = len(ngrams)
    counts = Counter(ngrams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / total if total > 0 else 0.0


def sentence_dup_rate(text: str) -> float:
    """Fraction of sentences that are duplicates (not counting first occurrence)."""
    sentences = sent_tokenize(text)
    if len(sentences) <= 1:
        return 0.0
    seen = set()
    dups = 0
    for s in sentences:
        norm = s.strip().lower()
        if norm in seen:
            dups += 1
        else:
            seen.add(norm)
    return dups / len(sentences)


def distinct_n(text: str, n: int) -> float:
    """Ratio of unique n-grams to total n-grams."""
    tokens = text.lower().split()
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    total = len(ngrams)
    unique = len(set(ngrams))
    return unique / total if total > 0 else 0.0


def compute_sample_metrics(text: str) -> dict:
    """Compute all diversity metrics for a single text sample."""
    return {
        "4g_repeat_pct": ngram_repeat_rate(text, 4) * 100,
        "sent_dup_pct": sentence_dup_rate(text) * 100,
        "distinct_1": distinct_n(text, 1),
        "distinct_2": distinct_n(text, 2),
    }


# ─── I/O ─────────────────────────────────────────────────────────────────────

def load_texts(log_dir: str, start: int, end: int, text_key: str) -> list[tuple[int, str]]:
    """Load (index, text) pairs from per-sample JSON logs."""
    results = []
    for i in range(start, end):
        path = os.path.join(log_dir, f"{i}.json")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        text = data.get(text_key, "")
        if text:
            results.append((i, text))
    return results


def main():
    parser = argparse.ArgumentParser(description="Compute diversity metrics on generated/original texts")
    parser.add_argument("--log_dir", type=str, required=True,
                        help="Directory containing per-sample JSON logs")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=500)
    parser.add_argument("--text_key", type=str, default="generated_text",
                        choices=["generated_text", "original_text"],
                        help="Which text field to evaluate")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional path to save result JSON")
    args = parser.parse_args()

    samples = load_texts(args.log_dir, args.start, args.end, args.text_key)
    if not samples:
        print("No texts found. Check --log_dir, --start, --end, and --text_key.")
        return

    # Compute per-sample metrics
    per_sample = {}
    all_metrics = {"4g_repeat_pct": [], "sent_dup_pct": [], "distinct_1": [], "distinct_2": []}

    for idx, text in samples:
        m = compute_sample_metrics(text)
        per_sample[idx] = m
        for k in all_metrics:
            all_metrics[k].append(m[k])

    # Aggregate
    agg = {}
    for k, vals in all_metrics.items():
        agg[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "median": float(np.median(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    # Print results
    n = len(samples)
    print(f"\n{'='*60}")
    print(f"Diversity Metrics ({n} samples, text_key={args.text_key})")
    print(f"{'='*60}")
    print(f"  4-gram Repeat%  : {agg['4g_repeat_pct']['mean']:.1f}% (±{agg['4g_repeat_pct']['std']:.1f}%)")
    print(f"  Sentence Dup%   : {agg['sent_dup_pct']['mean']:.1f}% (±{agg['sent_dup_pct']['std']:.1f}%)")
    print(f"  Distinct-1      : {agg['distinct_1']['mean']:.3f} (±{agg['distinct_1']['std']:.3f})")
    print(f"  Distinct-2      : {agg['distinct_2']['mean']:.3f} (±{agg['distinct_2']['std']:.3f})")
    print(f"{'='*60}")

    result = {
        "log_dir": args.log_dir,
        "text_key": args.text_key,
        "num_samples": n,
        "aggregate": agg,
    }

    out_path = args.output or os.path.join(args.log_dir, f"diversity_{args.text_key}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"Result saved to: {out_path}")


if __name__ == "__main__":
    main()
