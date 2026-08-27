"""Null-calibration helpers for the SemStamp comparison bridge."""

import numpy as np


def calibrated_threshold(null_scores, target_fpr):
    """Return an empirical strict-tail cutoff with FPR no greater than target."""
    if not 0 < target_fpr < 1:
        raise ValueError(f"target_fpr must be between 0 and 1, got {target_fpr!r}")
    valid = np.asarray(null_scores, dtype=float)
    valid = valid[~np.isnan(valid)]
    if not len(valid):
        raise ValueError("Cannot calibrate from an empty human-null distribution")
    return float(np.quantile(valid, 1 - target_fpr, method="higher"))


def empirical_auroc(pos_scores, null_scores):
    """Mann-Whitney AUROC, using every positive and human-null document."""
    pos = np.asarray(pos_scores, dtype=float)
    null = np.asarray(null_scores, dtype=float)
    pos = pos[~np.isnan(pos)]
    null = null[~np.isnan(null)]
    if not len(pos) or not len(null):
        return float("nan")
    comparisons = pos[:, None] - null[None, :]
    return float(np.mean(comparisons > 0) + 0.5 * np.mean(comparisons == 0))
