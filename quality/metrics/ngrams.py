"""N-gram metrics adapted from original SemStamp's ``eval_quality.py``."""

import numpy as np

from .common import _confidence_interval, rep_ngram, tokenize_untokenize


def evaluate_ngrams(gens, tokenizer):
    gen_sen_lis = [tokenize_untokenize(g, tokenizer) for g in gens]
    results = {}
    for k in [2, 3, 4]:
        per_sample = np.asarray(rep_ngram(gen_sen_lis, k, return_per_sample=True), dtype=float)
        results[f"rep_{k}"] = float(np.nanmean(per_sample))
        results[f"rep_{k}_ci"] = _confidence_interval(per_sample)
        results[f"rep_{k}_median"] = float(np.nanmedian(per_sample))
        results[f"rep_{k}_per_sample"] = per_sample
    return results
