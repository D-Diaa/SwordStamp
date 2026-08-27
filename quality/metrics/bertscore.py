"""BERTScore adapted from original SemStamp's ``detection_utils.py``."""

import numpy as np

from .common import _confidence_interval


def _sanitize(texts):
    """Replace blank strings to avoid a Roberta tokenizer failure."""
    return [t if t and t.strip() else "." for t in texts]


def evaluate_bertscore(gens, refs, max_retries=3):
    from bert_score import score as bert_score

    gens = _sanitize(gens)
    refs = _sanitize(refs)

    for attempt in range(max_retries):
        try:
            P, R, F1 = bert_score(gens, refs, lang="en", verbose=False)
            p_vals, r_vals, f1_vals = P.numpy(), R.numpy(), F1.numpy()
            return {
                "bert_P": np.mean(p_vals),
                "bert_P_ci": _confidence_interval(p_vals),
                "bert_P_median": np.median(p_vals),
                "bert_R": np.mean(r_vals),
                "bert_R_ci": _confidence_interval(r_vals),
                "bert_R_median": np.median(r_vals),
                "bert_F1": np.mean(f1_vals),
                "bert_F1_ci": _confidence_interval(f1_vals),
                "bert_F1_median": np.median(f1_vals),
                "bert_P_per_sample": np.asarray(p_vals, dtype=float),
                "bert_R_per_sample": np.asarray(r_vals, dtype=float),
                "bert_F1_per_sample": np.asarray(f1_vals, dtype=float),
            }
        except Exception as e:
            if attempt < max_retries - 1:
                import time

                wait = 30 * (attempt + 1)
                print(f"BERTScore attempt {attempt + 1} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
