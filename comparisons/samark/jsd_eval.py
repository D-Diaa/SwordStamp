"""
Distributional Quality Metrics (§6.6 of SAMARK_REVIEW.md)

Computes Jensen-Shannon Divergence (JSD) between a target method's text
distribution and a reference method's text distribution, plus vocabulary
density. All metrics are pure text statistics — no model needed.

Metrics:
  - JSD(unigram): JSD between unigram frequency distributions
  - JSD(bigram):  JSD between bigram frequency distributions
  - JSD(sent-len): JSD between sentence-length distributions
  - Vocab/1k tokens: unique word types per 1000 tokens

The reference method defaults to SynthID (known low-distortion watermark).
When the target IS the reference, all JSDs will be 0.

Usage:
    # Compare KGW against SynthID reference:
    python jsd_eval.py \
        --target_dir ./logs/booksum_logs/Mistral-7B-v0.1_log/KGW \
        --ref_dir ./logs/booksum_logs/Mistral-7B-v0.1_log/SynthID \
        --start 0 --end 500

    # For hierarchical methods:
    python jsd_eval.py \
        --target_dir ./logs/booksum_logs/Mistral-7B-v0.1_log/hierarchical_KGW_top5 \
        --ref_dir ./logs/booksum_logs/Mistral-7B-v0.1_log/SynthID \
        --start 0 --end 500
"""

import os
import json
import argparse
import numpy as np
from collections import Counter
from nltk.tokenize import sent_tokenize


# ─── Distribution helpers ────────────────────────────────────────────────────

def ngram_freq(texts: list[str], n: int) -> Counter:
    """Aggregate n-gram frequency counts across all texts."""
    counts = Counter()
    for text in texts:
        tokens = text.lower().split()
        for i in range(len(tokens) - n + 1):
            counts[tuple(tokens[i:i + n])] += 1
    return counts


def sent_len_freq(texts: list[str]) -> Counter:
    """Aggregate sentence-length (word count) frequency distribution."""
    counts = Counter()
    for text in texts:
        for sent in sent_tokenize(text):
            wc = len(sent.split())
            counts[wc] += 1
    return counts


def counter_to_prob(counter: Counter) -> dict:
    """Normalize a Counter into a probability distribution."""
    total = sum(counter.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in counter.items()}


def jsd(p_counter: Counter, q_counter: Counter) -> float:
    """Jensen-Shannon Divergence between two frequency distributions.

    JSD(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M), where M = 0.5*(P+Q).
    Uses natural log, result in [0, ln2] ≈ [0, 0.693].
    """
    p = counter_to_prob(p_counter)
    q = counter_to_prob(q_counter)

    if not p or not q:
        return 0.0

    # Union of all keys
    all_keys = set(p.keys()) | set(q.keys())

    jsd_val = 0.0
    for k in all_keys:
        pk = p.get(k, 0.0)
        qk = q.get(k, 0.0)
        mk = 0.5 * (pk + qk)
        if mk == 0:
            continue
        if pk > 0:
            jsd_val += 0.5 * pk * np.log(pk / mk)
        if qk > 0:
            jsd_val += 0.5 * qk * np.log(qk / mk)

    return float(jsd_val)


def vocab_per_1k(texts: list[str]) -> float:
    """Unique word types per 1000 tokens across all texts."""
    all_tokens = []
    for text in texts:
        all_tokens.extend(text.lower().split())
    total = len(all_tokens)
    if total == 0:
        return 0.0
    unique = len(set(all_tokens))
    return unique / total * 1000


# ─── I/O ─────────────────────────────────────────────────────────────────────

def load_texts(log_dir: str, start: int, end: int,
               text_key: str = "generated_text") -> list[str]:
    """Load texts from per-sample JSON logs."""
    texts = []
    for i in range(start, end):
        path = os.path.join(log_dir, f"{i}.json")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        text = data.get(text_key, "")
        if text:
            texts.append(text)
    return texts


def main():
    parser = argparse.ArgumentParser(
        description="Compute JSD distributional quality metrics against a reference method")
    parser.add_argument("--target_dir", type=str, required=True,
                        help="Log dir of the method to evaluate")
    parser.add_argument("--ref_dir", type=str, required=True,
                        help="Log dir of the reference method (e.g. SynthID)")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=500)
    parser.add_argument("--text_key", type=str, default="generated_text",
                        choices=["generated_text", "original_text"])
    parser.add_argument("--output", type=str, default=None,
                        help="Optional path to save result JSON")
    args = parser.parse_args()

    print(f"Loading target texts from: {args.target_dir}")
    target_texts = load_texts(args.target_dir, args.start, args.end, args.text_key)
    print(f"Loading reference texts from: {args.ref_dir}")
    ref_texts = load_texts(args.ref_dir, args.start, args.end, args.text_key)

    if not target_texts:
        print("Error: No target texts found.")
        return
    if not ref_texts:
        print("Error: No reference texts found.")
        return

    # Build distributions
    tgt_uni = ngram_freq(target_texts, 1)
    ref_uni = ngram_freq(ref_texts, 1)
    tgt_bi = ngram_freq(target_texts, 2)
    ref_bi = ngram_freq(ref_texts, 2)
    tgt_sl = sent_len_freq(target_texts)
    ref_sl = sent_len_freq(ref_texts)

    # Compute metrics
    jsd_unigram = jsd(tgt_uni, ref_uni)
    jsd_bigram = jsd(tgt_bi, ref_bi)
    jsd_sentlen = jsd(tgt_sl, ref_sl)
    vocab_1k = vocab_per_1k(target_texts)
    ref_vocab_1k = vocab_per_1k(ref_texts)

    print(f"\n{'='*60}")
    print(f"Distributional Quality Metrics (§6.6)")
    print(f"{'='*60}")
    print(f"  Target     : {args.target_dir} ({len(target_texts)} samples)")
    print(f"  Reference  : {args.ref_dir} ({len(ref_texts)} samples)")
    print(f"  text_key   : {args.text_key}")
    print(f"{'='*60}")
    print(f"  JSD(unigram)   : {jsd_unigram:.3f}")
    print(f"  JSD(bigram)    : {jsd_bigram:.3f}")
    print(f"  JSD(sent-len)  : {jsd_sentlen:.3f}")
    print(f"  Vocab/1k tok   : {vocab_1k:.0f}  (ref: {ref_vocab_1k:.0f})")
    print(f"{'='*60}")

    result = {
        "target_dir": args.target_dir,
        "ref_dir": args.ref_dir,
        "text_key": args.text_key,
        "num_target_samples": len(target_texts),
        "num_ref_samples": len(ref_texts),
        "jsd_unigram": jsd_unigram,
        "jsd_bigram": jsd_bigram,
        "jsd_sentlen": jsd_sentlen,
        "vocab_per_1k": vocab_1k,
        "ref_vocab_per_1k": ref_vocab_1k,
    }

    out_path = args.output or os.path.join(args.target_dir, f"jsd_{args.text_key}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"Result saved to: {out_path}")


if __name__ == "__main__":
    main()
