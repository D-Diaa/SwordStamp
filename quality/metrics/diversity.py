"""Deterministic repetition and diversity metrics used by SAMark."""

from __future__ import annotations

import numpy as np

from segmentation import segment

from .common import _confidence_interval


_METRIC_KEYS = ("sent_dup_pct", "distinct_2", "4g_repeat_pct")


def _word_ngrams(text: str, n: int) -> list[tuple[str, ...]]:
    """Return lowercased whitespace-tokenized word n-grams."""
    tokens = text.lower().split()
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def sentence_duplicate_pct(text: str) -> float:
    """Percentage of sentences duplicating an earlier normalized sentence."""
    # Route NLTK tokenization through the repository's central boundary API so
    # every package observes the same sentence policy.
    sentences = [
        unit.display.strip().lower()
        for unit in segment(text, type="sentence", backend="nltk")
    ]
    if not sentences:
        return 0.0
    return 100.0 * (len(sentences) - len(set(sentences))) / len(sentences)


def distinct_n(text: str, n: int) -> float:
    """Ratio of unique word n-grams to all word n-grams, or zero if empty."""
    ngrams = _word_ngrams(text, n)
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


def ngram_repeat_pct(text: str, n: int = 4) -> float:
    """Percentage of word n-gram occurrences beyond their first occurrence."""
    ngrams = _word_ngrams(text, n)
    if not ngrams:
        return 0.0
    return 100.0 * (len(ngrams) - len(set(ngrams))) / len(ngrams)


def compute_diversity_metrics(text: str) -> dict[str, float]:
    """Compute the three deterministic per-sample diversity metrics."""
    return {
        "sent_dup_pct": sentence_duplicate_pct(text),
        "distinct_2": distinct_n(text, 2),
        "4g_repeat_pct": ngram_repeat_pct(text, 4),
    }


def evaluate_diversity(texts) -> dict[str, object]:
    """Aggregate deterministic diversity metrics and retain per-sample values."""
    per_metric = {key: [] for key in _METRIC_KEYS}
    for text in texts:
        sample = compute_diversity_metrics(str(text))
        for key in _METRIC_KEYS:
            per_metric[key].append(sample[key])

    results: dict[str, object] = {}
    for key, values in per_metric.items():
        per_sample = np.asarray(values, dtype=float)
        if per_sample.size:
            mean = float(np.mean(per_sample))
            median = float(np.median(per_sample))
        else:
            mean = float("nan")
            median = float("nan")
        results[key] = mean
        results[f"{key}_ci"] = _confidence_interval(per_sample)
        results[f"{key}_median"] = median
        results[f"{key}_per_sample"] = per_sample
    return results
