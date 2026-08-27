"""ROUGE pairwise metrics."""

import numpy as np

from .common import _confidence_interval


def evaluate_rouge(gens, refs):
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rL = [], [], []
    for gen, ref in zip(gens, refs):
        scores = scorer.score(ref, gen)
        r1.append(scores["rouge1"].fmeasure)
        r2.append(scores["rouge2"].fmeasure)
        rL.append(scores["rougeL"].fmeasure)
    return {
        "rouge1": np.mean(r1),
        "rouge1_ci": _confidence_interval(r1),
        "rouge1_median": np.median(r1),
        "rouge2": np.mean(r2),
        "rouge2_ci": _confidence_interval(r2),
        "rouge2_median": np.median(r2),
        "rougeL": np.mean(rL),
        "rougeL_ci": _confidence_interval(rL),
        "rougeL_median": np.median(rL),
        "rouge1_per_sample": np.asarray(r1, dtype=float),
        "rouge2_per_sample": np.asarray(r2, dtype=float),
        "rougeL_per_sample": np.asarray(rL, dtype=float),
    }
