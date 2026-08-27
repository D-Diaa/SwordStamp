"""Perplexity adapted from original SemStamp's ``eval_quality.py``."""

import numpy as np

from .common import _confidence_interval, eval_perplexity


def evaluate_perplexity(model, tokenizer, texts):
    corpus_ppl, per_sample_ppls, total_nlls, scored_tokens = eval_perplexity(
        model, tokenizer, texts, return_evidence=True,
    )
    per_sample_ppls = np.asarray(per_sample_ppls, dtype=float)
    total_nlls = np.asarray(total_nlls, dtype=float)
    scored_tokens = np.asarray(scored_tokens, dtype=np.int64)
    return {
        "gen_ppl": corpus_ppl,
        "gen_ppl_ci": _confidence_interval(per_sample_ppls),
        "gen_ppl_median": float(np.nanmedian(per_sample_ppls)),
        "gen_ppl_per_sample": per_sample_ppls,
        "gen_ppl_total_nll_per_sample": total_nlls,
        "gen_ppl_scored_tokens_per_sample": scored_tokens,
        "gen_ppl_total_nll": float(np.nansum(total_nlls)),
        "gen_ppl_scored_tokens": int(scored_tokens.sum()),
    }
