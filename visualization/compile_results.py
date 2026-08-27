#!/usr/bin/env python
"""Compile results using each detector's persisted null distribution."""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as scipy_stats
import yaml
from datasets import load_from_disk
from sklearn.metrics import auc, roc_curve

from config.paper import (
    DATASETS as PAPER_DATASETS,
    HUMAN_NULL_DATASET,
    HUMAN_NULL_DOCUMENTS,
    LADDERS as PAPER_LADDERS,
    ORACLE_KS,
    PMARK,
    PROVIDER_CANDIDATES,
    SAMARK,
)
from quality.metrics.common import _confidence_interval, is_per_sample_scores
from segmentation import segment
from watermarking.primitives import extract_prompt_from_text

from .encoder_transfer import OUTPUT_STEM, build_transfer_table
from .metric_specs import (
    CORPUS_QUALITY_KEYS,
    LLM_JUDGE_METRICS,
    LLM_QUALITY_METRICS,
    PER_SAMPLE_QUALITY_KEYS,
    quality_metric_manifest,
)

ROOT = str(Path(__file__).resolve().parents[1])
OUT = os.path.join(ROOT, "results/paper")

DATASETS = list(PAPER_DATASETS)
HUMAN_CORPUS = HUMAN_NULL_DATASET  # empirical null (stage="human" rows)
FAMILIES = {
    "lsh": "SemStamp",
    "kmeans": "k-SemStamp",
    "pmark": "PMark",
    "samark": "SAMark",
    "none": "No watermark",
}

EDA_KS = (1, 2, 4, 8, 16, 32, 64)
EDA_POSITIONAL_COMPARISON_KS = (4, 16, 64)
DIPPER_SETTINGS = ((20, 20), (20, 40), (20, 80), (40, 20), (80, 20))
CONTROLLED_SETTINGS = (
    ("controlled_reorder", ("0.1", "0.25", "0.5", "0.75", "1.0")),
    ("synonym_substitution", ("0.1", "0.25", "0.5", "0.75", "1.0")),
    ("split_midpoint", ("0.2", "0.5", "1.0")),
    ("merge_adjacent", ("0.2", "0.5", "1.0")),
)

PHI10, PHI5, PHI1 = 1.2815515594457412, 1.6448536269514722, 2.3263478740408408


def empirical_null_thresholds(null_scores):
    """Return conservative empirical cutoffs for 10%, 5%, and 1% FPR."""
    return tuple(
        float(np.quantile(null_scores, q, method="higher"))
        for q in (0.90, 0.95, 0.99)
    )


# Pool corpus-only metrics by sample-weighted mean.
CORPUS_SCALARS = list(CORPUS_QUALITY_KEYS)
# Perplexity retains per-document diagnostics, but its headline aggregate is
# persisted in the CSV because it cannot be reconstructed by averaging those
# per-document perplexities.
PERPLEXITY_EVIDENCE_KEYS = {
    "gen_ppl_total_nll",
    "gen_ppl_scored_tokens",
}
PERSISTED_QUALITY_SCALARS = CORPUS_SCALARS + [
    "gen_ppl",
    *sorted(PERPLEXITY_EVIDENCE_KEYS),
]

# Success requires both evasion and preserved content.
QUALITY_METRIC = "llm_judge"          # per-sample pairwise judge, [0, 1]
QUALITY_BAR = 0.90                    # headline bar -> ASR_at5 / ASR_at1
ASR_BARS = [percent / 100 for percent in range(65, 96, 5)]
ASR_FPRS = [("at5", "thr5"), ("at1", "thr1")]

# Quality evaluation persists the three independent rubric decisions as
# criterion-by-repetition arrays.  Keep the compiled export wide so its row
# identity remains exactly one physical generated sample.
JUDGE_RUN_COUNT = 3
JUDGE_METRIC_GROUPS = (LLM_QUALITY_METRICS, LLM_JUDGE_METRICS)
JUDGE_RUN_KEYS = tuple(
    f"{spec.key}_run_{run}"
    for metrics in JUDGE_METRIC_GROUPS
    for spec in metrics
    for run in range(1, JUDGE_RUN_COUNT + 1)
)

# Provider-candidate accounting. Comparison systems do not expose a candidate
# budget in their path identity, but the selected paper configuration uses the
# same fixed pool as the N=64 SemStamp arms.
COMPARISON_PROVIDER_POOL = float(PROVIDER_CANDIDATES)
PROVIDER_METRIC_FIELDS = (
    "provider_draws_per_unit",
    "provider_draws_per_unit_ci_lo_95",
    "provider_draws_per_unit_ci_hi_95",
    "provider_units_total",
    "provider_sentences_total",
    "provider_units_per_sentence",
    "provider_draws_per_sentence",
    "provider_draws_per_sentence_ci_lo_95",
    "provider_draws_per_sentence_ci_hi_95",
)
REJECTION_STATS_FIELDS = (
    "mean_tries",
    "tries_ci_lo_95",
    "tries_ci_hi_95",
    "n_units_tracked",
    "mean_p_green_per_unit",
    "std_p_green_per_unit",
)


class ProviderEvidenceError(ValueError):
    """Selected clean-generation evidence is missing or malformed."""


def empty_provider_metrics():
    """Stable provider-cost schema for non-clean rows."""
    return {field: np.nan for field in PROVIDER_METRIC_FIELDS}


def _require_finite_number(values, field, source, *, positive=False,
                           nonnegative=False):
    """Read one finite JSON number or raise an evidence-focused error."""
    if field not in values:
        raise ProviderEvidenceError(f"{source}: missing field {field!r}")
    value = values[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProviderEvidenceError(f"{source}: field {field!r} is not numeric")
    value = float(value)
    if not np.isfinite(value):
        raise ProviderEvidenceError(f"{source}: field {field!r} is not finite")
    if positive and value <= 0:
        raise ProviderEvidenceError(f"{source}: field {field!r} must be positive")
    if nonnegative and value < 0:
        raise ProviderEvidenceError(f"{source}: field {field!r} must be nonnegative")
    return value


@lru_cache(maxsize=None)
def _load_source_dataset(directory):
    """Load a selected source shard once while auditing all of its clean cells."""
    try:
        dataset = load_from_disk(directory)
    except Exception as exc:
        raise ProviderEvidenceError(
            f"{directory}: cannot load source dataset: {exc}"
        ) from exc
    if "text" not in dataset.column_names:
        raise ProviderEvidenceError(
            f"{directory}: source dataset is missing the 'text' column"
        )
    return dataset


def _semstamp_prompt_policy(directory, source_directory):
    """Recover SemStamp's exact prompt function from resolved generation config."""
    config_path = os.path.join(directory, "resolved_config.yaml")
    if not os.path.exists(config_path):
        raise ProviderEvidenceError(f"{config_path}: missing source file")
    try:
        with open(config_path, encoding="utf-8") as source:
            resolved = yaml.safe_load(source)
    except (OSError, yaml.YAMLError) as exc:
        raise ProviderEvidenceError(
            f"{config_path}: cannot read YAML: {exc}"
        ) from exc
    if not isinstance(resolved, dict):
        raise ProviderEvidenceError(f"{config_path}: expected a YAML mapping")

    configured_source = resolved.get("io", {}).get("data_path")
    if not isinstance(configured_source, str) or not configured_source:
        raise ProviderEvidenceError(
            f"{config_path}: missing field 'io.data_path'"
        )
    configured_source = (
        configured_source if os.path.isabs(configured_source)
        else os.path.join(ROOT, configured_source)
    )
    if os.path.realpath(configured_source) != os.path.realpath(source_directory):
        raise ProviderEvidenceError(
            f"{config_path}: io.data_path={configured_source!r} disagrees with "
            f"selected source shard {source_directory!r}"
        )

    len_prompt = resolved.get("generation", {}).get("len_prompt")
    if isinstance(len_prompt, bool) or not isinstance(len_prompt, int) or len_prompt <= 0:
        raise ProviderEvidenceError(
            f"{config_path}: field 'generation.len_prompt' must be a positive integer"
        )

    return (
        lambda text: extract_prompt_from_text(text, len_prompt),
        config_path,
        f"extract_prompt_from_text(source_text, len_prompt={len_prompt})",
    )


def count_clean_output_sentences(directory, family, source_directory):
    """Strip each verified generation prompt, then count continuation sentences."""
    try:
        marked = load_from_disk(directory)
    except Exception as exc:
        raise ProviderEvidenceError(
            f"{directory}: cannot load clean watermarked dataset: {exc}"
        ) from exc
    if "text" not in marked.column_names:
        raise ProviderEvidenceError(
            f"{directory}: clean watermarked dataset is missing the 'text' column"
        )
    source = _load_source_dataset(source_directory)

    if family in {"lsh", "kmeans", "none"}:
        if len(marked) != len(source):
            raise ProviderEvidenceError(
                f"{directory}: clean/source row-count mismatch "
                f"({len(marked)} != {len(source)})"
            )
        prompt_for_source, prompt_config, prompt_rule = _semstamp_prompt_policy(
            directory, source_directory,
        )
        source_indices = range(len(marked))
        prompts = [prompt_for_source(text) for text in source["text"]]
    else:
        required = {"prompt", "original_text"}
        missing = sorted(required - set(marked.column_names))
        if missing:
            raise ProviderEvidenceError(
                f"{directory}: clean comparison dataset is missing column(s) {missing}"
            )
        index_column = "pmark_idx" if family == "pmark" else "samark_idx"
        source_indices = (
            marked[index_column] if index_column in marked.column_names
            else list(range(len(marked)))
        )
        prompts = marked["prompt"]
        prompt_config = os.path.join(directory, "config.json")
        if not os.path.exists(prompt_config):
            prompt_config = None
        prompt_rule = "persisted first-source-sentence prompt"

    sentences_total = 0
    seen_source_indices = set()
    for output_index, (source_index, prompt, marked_text) in enumerate(zip(
        source_indices, prompts, marked["text"],
    )):
        if isinstance(source_index, bool) or not isinstance(source_index, (int, np.integer)):
            raise ProviderEvidenceError(
                f"{directory}: sample {output_index} has invalid source index {source_index!r}"
            )
        source_index = int(source_index)
        if source_index < 0 or source_index >= len(source):
            raise ProviderEvidenceError(
                f"{directory}: sample {output_index} source index {source_index} is out of range"
            )
        if source_index in seen_source_indices:
            raise ProviderEvidenceError(
                f"{directory}: duplicate source index {source_index}"
            )
        seen_source_indices.add(source_index)
        source_text = source[source_index]["text"]
        if not isinstance(prompt, str) or not prompt:
            raise ProviderEvidenceError(
                f"{directory}: sample {output_index} recovered prompt is empty or non-text"
            )
        if family in {"pmark", "samark"}:
            if marked[output_index]["original_text"] != source_text:
                raise ProviderEvidenceError(
                    f"{directory}: sample {output_index} original_text does not match "
                    f"source row {source_index}"
                )
            if not source_text.startswith(prompt):
                raise ProviderEvidenceError(
                    f"{directory}: sample {output_index} persisted prompt is not a "
                    f"prefix of source row {source_index}"
                )
        if not marked_text.startswith(prompt):
            raise ProviderEvidenceError(
                f"{directory}: sample {output_index} stored text does not begin with "
                f"the exact recovered prompt {prompt[:80]!r}"
            )
        continuation = marked_text[len(prompt):]
        sentences_total += len(segment(
            continuation, type="sentence", backend="nltk",
        ))

    return sentences_total, dict(
        source_dataset=source_directory,
        prompt_config=prompt_config,
        prompt_rule=prompt_rule,
        prompts_verified=len(marked),
    )


def load_provider_evidence(directory, family, sampling, segmentation,
                           generation_num_candidates, source_directory):
    """Load auditable provider-cost evidence for one clean physical shard."""
    sentences_total, prompt_evidence = count_clean_output_sentences(
        directory, family, source_directory,
    )
    if sentences_total <= 0:
        raise ProviderEvidenceError(
            f"{directory}: clean watermarked dataset has no final output sentences"
        )

    evidence = dict(
        directory=directory,
        sentences_total=sentences_total,
        stats_path=None,
        stats=None,
        prompt_evidence=prompt_evidence,
    )
    if family in {"pmark", "samark"}:
        evidence.update(
            pool=COMPARISON_PROVIDER_POOL,
            units_total=sentences_total,
        )
        return evidence

    stats_path = os.path.join(directory, "generation_stats.json")
    if not os.path.exists(stats_path):
        raise ProviderEvidenceError(f"{stats_path}: missing source file")
    try:
        with open(stats_path, encoding="utf-8") as source:
            stats = json.load(source)
    except (OSError, json.JSONDecodeError) as exc:
        raise ProviderEvidenceError(f"{stats_path}: cannot read JSON: {exc}") from exc
    if not isinstance(stats, dict):
        raise ProviderEvidenceError(f"{stats_path}: expected a JSON object")

    pool = _require_finite_number(stats, "num_candidates", stats_path, positive=True)
    if not np.isfinite(generation_num_candidates):
        raise ProviderEvidenceError(
            f"{directory}: missing generation_num_candidates cell identity"
        )
    if pool != float(generation_num_candidates):
        raise ProviderEvidenceError(
            f"{stats_path}: num_candidates={pool:g} disagrees with selected "
            f"cell budget {float(generation_num_candidates):g}"
        )
    units_total = _require_finite_number(
        stats, "total_units", stats_path, positive=True,
    )
    if not units_total.is_integer():
        raise ProviderEvidenceError(f"{stats_path}: total_units must be an integer")

    if sampling == "rejection":
        for field in REJECTION_STATS_FIELDS:
            _require_finite_number(
                stats, field, stats_path,
                positive=field != "std_p_green_per_unit",
                nonnegative=field == "std_p_green_per_unit",
            )
        mean_p = float(stats["mean_p_green_per_unit"])
        if mean_p > 1:
            raise ProviderEvidenceError(
                f"{stats_path}: mean_p_green_per_unit must be at most 1"
            )
        tracked = float(stats["n_units_tracked"])
        if not tracked.is_integer():
            raise ProviderEvidenceError(
                f"{stats_path}: n_units_tracked must be an integer"
            )
    elif sampling != "best-of-n":
        raise ProviderEvidenceError(
            f"{directory}: unsupported clean sampling mode {sampling!r}"
        )

    evidence.update(
        pool=pool,
        units_total=int(units_total),
        stats_path=stats_path,
        stats=stats,
    )
    return evidence


def pool_valid_candidate_rate(evidence):
    """Pool shard moments, then form and invert a 95% t interval."""
    parts = []
    for shard in evidence:
        stats = shard["stats"]
        parts.append((
            int(stats["n_units_tracked"]),
            float(stats["mean_p_green_per_unit"]),
            float(stats["std_p_green_per_unit"]),
        ))
    n_total = sum(n for n, _mean, _std in parts)
    mean = sum(n * shard_mean for n, shard_mean, _std in parts) / n_total
    if n_total > 1:
        ss = sum(
            (n - 1) * shard_std ** 2 + n * (shard_mean - mean) ** 2
            for n, shard_mean, shard_std in parts
        )
        variance = ss / (n_total - 1)
        half_width = float(scipy_stats.t.ppf(0.975, df=n_total - 1)) * (
            variance / n_total
        ) ** 0.5
    else:
        half_width = 0.0
    # A valid-candidate rate is supported on [0, 1]. The positive floor makes
    # inversion finite for an interval whose sampling uncertainty reaches 0.
    p_lo = max(1e-9, mean - half_width)
    p_hi = min(1.0, mean + half_width)
    return 1.0 / mean, 1.0 / p_hi, 1.0 / p_lo


def aggregate_provider_metrics(family, sampling, segmentation, evidence):
    """Compute one dataset/cell summary from complete clean shard evidence."""
    if not evidence:
        return empty_provider_metrics()
    units_total = int(sum(shard["units_total"] for shard in evidence))
    sentences_total = int(sum(shard["sentences_total"] for shard in evidence))
    is_semspan = parse_segmentation(segmentation)["segmentation_type"] == "semspan"
    units_per_sentence = (
        units_total / sentences_total if is_semspan else 1.0
    )

    if family in {"pmark", "samark"} or sampling == "best-of-n":
        draws_per_unit = float(evidence[0]["pool"])
        if any(float(shard["pool"]) != draws_per_unit for shard in evidence):
            raise ProviderEvidenceError(
                "clean shards in one cell have inconsistent candidate-pool sizes"
            )
        unit_lo = unit_hi = draws_per_unit
    elif sampling == "rejection":
        draws_per_unit, unit_lo, unit_hi = pool_valid_candidate_rate(evidence)
    else:
        raise ProviderEvidenceError(f"unsupported clean sampling mode {sampling!r}")

    return dict(
        provider_draws_per_unit=draws_per_unit,
        provider_draws_per_unit_ci_lo_95=unit_lo,
        provider_draws_per_unit_ci_hi_95=unit_hi,
        provider_units_total=units_total,
        provider_sentences_total=sentences_total,
        provider_units_per_sentence=units_per_sentence,
        provider_draws_per_sentence=draws_per_unit * units_per_sentence,
        provider_draws_per_sentence_ci_lo_95=unit_lo * units_per_sentence,
        provider_draws_per_sentence_ci_hi_95=unit_hi * units_per_sentence,
    )


# Detection metrics.
def auroc_empirical(para, human):
    """Compute AUROC against every available human-null document."""
    para = np.nan_to_num(np.asarray(para, dtype=float))
    human = np.nan_to_num(np.asarray(human, dtype=float))
    n = len(para)
    if n == 0 or len(human) == 0:
        return float("nan")
    fpr, tpr, _ = roc_curve(
        [1] * n + [0] * len(human), np.concatenate([para, human]),
    )
    return float(auc(fpr, tpr))


def auroc_analytic(para):
    """Compute PMark AUROC against its analytic normal null."""
    v = np.asarray(para, dtype=float)
    v = v[~np.isnan(v)]
    return float(np.mean(scipy_stats.norm.cdf(v))) if len(v) else float("nan")


def load_z(d, family, stage):
    """Load attacked, watermarked, and null z-scores for one directory."""
    source = "direct"
    if family == "none":
        # The no-watermark baseline is quality-only. Preserve one row per
        # generated sample without inventing detector scores or thresholds.
        n = len(load_from_disk(d))
        unavailable = np.full(n, np.nan)
        return unavailable, unavailable.copy(), None, "unavailable"
    if family == "pmark":
        para = _npy(d, "z_scores_para.npy")
        wm = _npy(d, "z_scores_text.npy")
        # Paper PMark is online-only and intentionally uses an N(0,1) null.
        human = None
        if stage == "watermark" and wm is None:
            wm = _pmark_unattacked_z(d)
            source = "median-of-siblings"
    elif family == "samark":
        detect_dir = os.path.join(d, "detect")
        para = _npy(detect_dir, "z_scores_para.npy")
        wm = _npy(detect_dir, "z_scores_text.npy")
        human = _npy(detect_dir, "human_z_scores.npy")
    else:
        para = _npy(d, "para_z_scores.npy")
        wm = _npy(d, "z_scores.npy")
        human = _npy(d, "human_z_scores.npy")
    if stage == "watermark":
        # No attack: the watermarked z IS the score under test.
        para = wm if wm is not None else para
    return para, wm, human, source


def _pmark_unattacked_z(watermarked_dir):
    """Per-sample median of the siblings' z_scores_text.npy (see load_z)."""
    sib = sorted(glob.glob(os.path.join(os.path.dirname(watermarked_dir), "*", "z_scores_text.npy")))
    arrs = [np.load(p) for p in sib]
    arrs = [a for a in arrs if a.ndim == 1]
    if not arrs:
        return None
    n = min(len(a) for a in arrs)
    stack = np.vstack([a[:n] for a in arrs])
    print(f"    pmark unattacked z derived from {len(arrs)} siblings "
          f"(per-sample sd {np.nanmean(np.nanstd(stack, axis=0)):.3f}): "
          f"{os.path.relpath(watermarked_dir, ROOT)}")
    return np.nanmedian(stack, axis=0)


def _npy(d, name):
    p = os.path.join(d, name)
    return np.load(p) if os.path.exists(p) else None


# Attack names.
ADAPT_RE = re.compile(
    r"^adaptive-(?P<model>Qwen2\.5-3B-Instruct)-(?P<style>standard)"
    r"-K(?P<K>1|2|4|8|16|32|64)-(?P<agg>min)"
    r"(?:-bag=(?P<bag>min))?-surr=(?P<surr>bge)"
    r"(?:-aseg=(?P<aseg>semspan-nltk-max15-win5))?$"
)
DIPPER_RE = re.compile(r"^dipper-lex(?P<lex>\d+)-order(?P<order>\d+)$")
PM_ADP_RE = re.compile(r"^adp(?P<bag>bag)?-K(?P<K>1|2|4|8|16|32|64)$")
ORACLE_RE = re.compile(
    r"^oracle-(?P<model>Qwen2\.5-3B-Instruct)-(?P<style>standard)"
    r"-K(?P<K>4|8|16|32|64)$"
)
PM_DIPPER_RE = re.compile(r"^dipper-l(?P<lex>\d+)-o(?P<order>\d+)$")
BIGRAM_RE = re.compile(
    r"^(?P<base>pegasus|parrot)-bigram(?:-threshold=(?P<thr>0\.03))?$"
)


def parse_segmentation(value: str) -> dict:
    """Expose parameters for one of the paper's two segmentation identities."""
    if value == "sentence-nltk":
        return dict(
            segmentation_type="sentence",
            segmentation_backend="nltk",
            segmentation_max_words=np.nan,
            segmentation_window=np.nan,
        )
    if value == "semspan-spacy-max15-win5":
        return dict(
            segmentation_type="semspan",
            segmentation_backend="spacy",
            segmentation_max_words=15.0,
            segmentation_window=5.0,
        )
    raise ValueError(f"unsupported paper segmentation: {value!r}")


def parse_attack(leaf: str) -> dict:
    """Parse one exact paper attack leaf; reject stale experiment names."""
    out = dict(attack=leaf, attack_family="other", attack_setting=leaf, K=np.nan,
               surrogate="", anchor="", attacker_model="", attacker_seg="",
               candidate_agg="", bag_agg="",
               dipper_lex=np.nan, dipper_order=np.nan)
    if leaf == "none":
        return {**out, "attack_family": "none", "attack_setting": "none"}

    m = ADAPT_RE.match(leaf)
    if m and not (m["aseg"] and not m["bag"]):
        return {**out, "attack_family": "adaptive",
                "attack_setting": (
                    f"K{m['K']}-{m['agg']}-"
                    f"{'bag=' + m['bag'] if m['bag'] else 'positional'}-{m['surr']}"
                ),
                "K": float(m["K"]), "surrogate": m["surr"],
                "anchor": "bag" if m["bag"] else "positional",
                "candidate_agg": m["agg"], "bag_agg": m["bag"] or "",
                "attacker_model": m["model"], "attacker_seg": m["aseg"] or "sentence-nltk"}
    m = PM_ADP_RE.match(leaf)
    if m:
        surr = "bge"
        anchor = "bag" if m["bag"] else "positional"
        return {**out, "attack_family": "adaptive",
                "attack_setting": f"K{m['K']}-{anchor}-{surr}", "K": float(m["K"]),
                "surrogate": surr, "anchor": anchor,
                "bag_agg": "min" if anchor == "bag" else "",
                "attacker_model": "Qwen2.5-3B-Instruct", "attacker_seg": "sentence-nltk"}
    m = ORACLE_RE.match(leaf)
    if m:
        return {**out, "attack_family": "oracle",
                "attack_setting": f"K{m['K']}-detector-aware", "K": float(m["K"]),
                "surrogate": "detector", "anchor": "detector",
                "attacker_model": m["model"], "attacker_seg": "detector"}
    m = DIPPER_RE.match(leaf) or PM_DIPPER_RE.match(leaf)
    if m and (int(m["lex"]), int(m["order"])) in DIPPER_SETTINGS:
        return {**out, "attack_family": "dipper",
                "attack_setting": f"lex{m['lex']}-order{m['order']}",
                "dipper_lex": float(m["lex"]), "dipper_order": float(m["order"])}
    m = BIGRAM_RE.match(leaf)
    if m:
        return {**out, "attack_family": m["base"],
                "attack_setting": f"bigram{'-' + m['thr'] if m['thr'] else ''}"}
    if leaf in ("pegasus", "parrot"):
        return {**out, "attack_family": leaf, "attack_setting": "default"}
    controlled_leaves = {
        f"{attack}-ratio={ratio}"
        for attack, ratios in CONTROLLED_SETTINGS
        for ratio in ratios
    }
    if leaf in controlled_leaves:
        return out
    raise ValueError(f"unsupported paper attack leaf: {leaf!r}")


def decorate_transfer_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the same cell identity columns used by cell_summary.csv."""
    if frame.empty:
        return frame
    result = frame.copy()
    result["scheme"] = result["family"].map(FAMILIES)
    result["flag_scope"] = pd.Series("", index=result.index, dtype=object)
    result["msig"] = np.nan
    result["stage"] = "attack"
    result["z_source"] = "direct"
    result["generation_num_candidates"] = float(PROVIDER_CANDIDATES)
    segmentations = {
        value: parse_segmentation(value)
        for value in result["segmentation"].drop_duplicates()
    }
    for key in next(iter(segmentations.values())):
        result[key] = result["segmentation"].map(
            lambda value, column=key: segmentations[value][column]
        )
    attacks = {
        value: parse_attack(value) for value in result["attack"].drop_duplicates()
    }
    for key in next(iter(attacks.values())):
        if key != "K":
            result[key] = result["attack"].map(
                lambda value, column=key: attacks[value][column]
            )

    lead = ["scheme", "family", "mask", "sampling", "segmentation", "flag_scope", "msig",
            "segmentation_type", "segmentation_backend", "segmentation_max_words",
            "segmentation_window", "generation_num_candidates", "stage", "z_source", "attack",
            "attack_family", "attack_setting", "K", "surrogate", "anchor",
            "attacker_model", "attacker_seg", "candidate_agg", "bag_agg",
            "dipper_lex", "dipper_order"]
    unit = ["dataset", "document_id", "sample_id", "sentence_index",
            "marked_sentence_count", "attacked_sentence_count", "aligned_sentence_count",
            "marked_sentence", "attacked_sentence", "surrogate_encoder", "provider_encoder",
            "surrogate_cosine_displacement", "provider_cosine_displacement"]
    return result[lead + unit + [
        column for column in result.columns if column not in lead + unit
    ]]


# Quality inputs.
def per_sample_arrays(d, n):
    """Every per-sample quality score array for dir `d`, padded/truncated to n.

    Judge auditing arrays share the NPZ but are not score columns; skip them.
    """
    out = {}
    p = os.path.join(d, "eval_quality_per_sample.npz")
    if not os.path.exists(p):
        return out
    z = np.load(p, allow_pickle=True)
    for k in z.files:
        v = z[k]
        if not is_per_sample_scores(k, v):
            continue
        v = np.asarray(v, dtype=float)
        out[k] = v[:n] if len(v) >= n else np.concatenate([v, np.full(n - len(v), np.nan)])
    return out


def judge_run_arrays(d, n):
    """Return the three independent runs for both LLM judge rubrics.

    Criterion repetitions are stored directly in the quality NPZ.  The
    per-run rubric headline is reconstructed with the same normalized weights
    and valence as the evaluator.  Missing samples, criteria, or repetitions
    remain NaN; the returned schema is stable for every compiled row.
    """
    out = {key: np.full(n, np.nan) for key in JUDGE_RUN_KEYS}
    path = os.path.join(d, "eval_quality_per_sample.npz")
    if not os.path.exists(path):
        return out

    with np.load(path, allow_pickle=True) as quality:
        for metrics in JUDGE_METRIC_GROUPS:
            headline, *criteria = metrics
            repeated = []
            for spec in criteria:
                key = f"{spec.key}_repeats"
                if key not in quality:
                    repeated = []
                    break
                values = np.asarray(quality[key], dtype=float)
                if values.ndim != 2:
                    repeated = []
                    break
                padded = np.full((n, JUDGE_RUN_COUNT), np.nan)
                rows = min(n, values.shape[0])
                runs = min(JUDGE_RUN_COUNT, values.shape[1])
                padded[:rows, :runs] = values[:rows, :runs]
                repeated.append((spec, padded))
                for run in range(JUDGE_RUN_COUNT):
                    out[f"{spec.key}_run_{run + 1}"] = padded[:, run]

            if len(repeated) != len(criteria):
                continue

            adjusted = np.stack([
                1.0 - values if spec.lower_is_better else values
                for spec, values in repeated
            ])
            complete = np.all(np.isfinite(adjusted), axis=0)
            overall = np.where(complete, np.mean(adjusted, axis=0), np.nan)

            # Empty generations are a deliberate evaluator special case: all
            # rubric dimensions and the headline are zero.  Applying negative
            # valence to their zero-valued pairwise dimensions would otherwise
            # reconstruct a nonzero headline.
            if headline.key in quality:
                reduced = np.asarray(quality[headline.key], dtype=float)[:n]
                empty_rows = np.flatnonzero(reduced == 0.0)
                if len(empty_rows):
                    all_zero = np.all(
                        np.stack([values[empty_rows] for _spec, values in repeated]) == 0.0,
                        axis=(0, 2),
                    )
                    overall[empty_rows[all_zero], :] = 0.0

            for run in range(JUDGE_RUN_COUNT):
                out[f"{headline.key}_run_{run + 1}"] = overall[:, run]
    return out


def judge_mean_arrays(run_arrays, n):
    """Rebuild both judge headlines and all criteria from their three runs.

    Persisted aggregate arrays may predate mean aggregation.  The run columns
    are the auditable source of truth, so compilation requires three finite
    runs and never carries a stale NPZ aggregate into summaries or ASR gates.
    """
    out = {}
    for metrics in JUDGE_METRIC_GROUPS:
        for spec in metrics:
            runs = np.column_stack([
                np.asarray(run_arrays[f"{spec.key}_run_{run}"], dtype=float)
                for run in range(1, JUDGE_RUN_COUNT + 1)
            ])
            if runs.shape != (n, JUDGE_RUN_COUNT):
                raise ValueError(
                    f"invalid judge run shape for {spec.key}: {runs.shape}"
                )
            complete = np.all(np.isfinite(runs), axis=1)
            means = np.full(n, np.nan)
            means[complete] = np.mean(runs[complete], axis=1)
            out[spec.key] = means
    return out


def scalars(d):
    p = os.path.join(d, "eval_quality.csv")
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    out = {}
    for k, v in rows[0].items():
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            out[k] = np.nan
    return out


def aggregate_corpus_perplexity(segments):
    """Return corpus PPL and its exact summed evidence across shards."""
    evidence = [
        (
            segment["sc"].get("gen_ppl_total_nll", np.nan),
            segment["sc"].get("gen_ppl_scored_tokens", np.nan),
        )
        for segment in segments
    ]
    evidence = [
        (total_nll, scored_tokens)
        for total_nll, scored_tokens in evidence
        if total_nll is not None and scored_tokens is not None
        and np.isfinite(total_nll) and np.isfinite(scored_tokens)
        and scored_tokens > 0
    ]
    if not evidence:
        return float("nan"), float("nan"), 0
    total_nll = float(sum(value for value, _ in evidence))
    scored_tokens = int(sum(value for _, value in evidence))
    return float(np.exp(total_nll / scored_tokens)), total_nll, scored_tokens


# Result discovery.
def paper_attack_leaves(family, mask, sampling, segmentation):
    """Return every and only physical attack leaf reported for one paper cell."""
    identity = (family, mask, sampling, segmentation)
    watermark_cells = {
        (rung.family, rung.mask, rung.sampling, rung.segmentation)
        for rungs in PAPER_LADDERS.values()
        for rung in rungs
    }
    comparison_cells = {
        (PMARK.family, PMARK.mask, PMARK.sampling, PMARK.segmentation),
        (SAMARK.family, SAMARK.mask, SAMARK.sampling, SAMARK.segmentation),
    }
    if identity not in watermark_cells | comparison_cells:
        return frozenset()

    leaves = {"watermarked", "pegasus", "parrot"}
    if family in {"pmark", "samark"}:
        leaves.update({"pegasus-bigram", "parrot-bigram"})
        leaves.update(
            f"dipper-l{lex}-o{order}" for lex, order in DIPPER_SETTINGS
        )
    else:
        leaves.update({
            "pegasus-bigram-threshold=0.03",
            "parrot-bigram-threshold=0.03",
        })
        leaves.update(
            f"dipper-lex{lex}-order{order}" for lex, order in DIPPER_SETTINGS
        )
        leaves.update(
            f"{attack}-ratio={ratio}"
            for attack, ratios in CONTROLLED_SETTINGS
            for ratio in ratios
        )

    if family == "pmark":
        leaves.update(f"adp-K{k}" for k in EDA_KS)
        return frozenset(leaves)
    if family == "samark":
        leaves.update(f"adpbag-K{k}" for k in EDA_KS)
        leaves.update(f"adp-K{k}" for k in EDA_POSITIONAL_COMPARISON_KS)
        return frozenset(leaves)

    stem = "adaptive-Qwen2.5-3B-Instruct-standard"
    positional_ks = (
        EDA_KS if mask == "context" else EDA_POSITIONAL_COMPARISON_KS
    )
    leaves.update(f"{stem}-K{k}-min-surr=bge" for k in positional_ks)
    if mask != "context":
        attacker_seg = (
            "-aseg=semspan-nltk-max15-win5"
            if segmentation == "semspan-spacy-max15-win5"
            else ""
        )
        leaves.update(
            f"{stem}-K{k}-min-bag=min-surr=bge{attacker_seg}"
            for k in EDA_KS
        )

    detector_access_cells = {
        ("kmeans", "context", "rejection", "sentence-nltk"),
        (
            "kmeans", "fixed_diverse", "best-of-n",
            "semspan-spacy-max15-win5",
        ),
    }
    if identity in detector_access_cells:
        leaves.update(
            f"oracle-Qwen2.5-3B-Instruct-standard-K{k}" for k in ORACLE_KS
        )
    return frozenset(leaves)


def discover(datasets=None):
    """Yield only physical leaves in the final paper experiment registry."""
    candidate_filter = {float(PROVIDER_CANDIDATES)}
    paper_cells = {
        (rung.family, rung.mask, rung.sampling, rung.segmentation)
        for rungs in PAPER_LADDERS.values()
        for rung in rungs
    }
    for ds in datasets or DATASETS:
        # The shared no-watermark reference and the human null are both
        # required alongside every paper watermark family.
        for d in sorted(glob.glob(os.path.join(ROOT, "data", ds, "none", "*"))):
            if not (
                os.path.isdir(d)
                and os.path.exists(os.path.join(d, "dataset_info.json"))
            ):
                continue
            seg = os.path.basename(d)
            if seg != "sentence-nltk":
                continue
            # The canonical no-watermark generator draws exactly one
            # continuation candidate. Provider evidence validates this
            # identity against generation_stats.json.
            yield (ds, "none", "none", "rejection", seg, 1.0,
                   "", "", "watermark", "none", d)

        for family in ("lsh", "kmeans", "pmark"):
            for sd in sorted(glob.glob(os.path.join(ROOT, "data", ds, family, "*/*/*"))):
                if not os.path.isdir(sd):
                    continue
                mask, sampling, seg = sd.split(os.sep)[-3:]
                if family in {"lsh", "kmeans"}:
                    if (family, mask, sampling, seg) not in paper_cells:
                        continue
                elif (mask, sampling, seg) != ("online", "rejection", "sentence-nltk"):
                    continue
                selected_leaves = paper_attack_leaves(
                    family, mask, sampling, seg,
                )
                parents = [(sd, float(PMARK.num_candidates))] if family == "pmark" else []
                if family in {"lsh", "kmeans"}:
                    for candidate_dir in sorted(glob.glob(os.path.join(sd, "candidates-*"))):
                        try:
                            budget = float(os.path.basename(candidate_dir).split("-", 1)[1])
                        except (IndexError, ValueError):
                            continue
                        parents.append((candidate_dir, budget))
                for parent, generation_num_candidates in parents:
                    if (
                        family in {"lsh", "kmeans"}
                        and
                        candidate_filter is not None
                        and generation_num_candidates not in candidate_filter
                    ):
                        continue
                    for d in sorted(glob.glob(parent + "/*")):
                        if not (os.path.isdir(d) and os.path.exists(os.path.join(d, "dataset_info.json"))):
                            continue
                        leaf = os.path.basename(d)
                        if leaf not in selected_leaves:
                            continue
                        stage = "watermark" if leaf == "watermarked" else "attack"
                        yield (ds, family, mask, sampling, seg, generation_num_candidates,
                               "", "", stage,
                               ("none" if stage == "watermark" else leaf), d)

        pattern = os.path.join(
            ROOT, "data", ds, "samark", "flags-*", "msig*", "*", "*"
        )
        for sd in sorted(glob.glob(pattern)):
            if not os.path.isdir(sd):
                continue
            flag_dir, msig_dir, sampling, seg = sd.split(os.sep)[-4:]
            flag_scope = flag_dir.removeprefix("flags-")
            if (
                flag_scope != SAMARK.mask.removeprefix("flags-")
                or sampling != SAMARK.sampling
                or seg != SAMARK.segmentation
            ):
                continue
            try:
                msig = int(msig_dir.removeprefix("msig"))
            except ValueError:
                continue
            if msig != SAMARK.msig:
                continue
            selected_leaves = paper_attack_leaves(
                "samark", f"flags-{flag_scope}", sampling, seg,
            )
            for d in sorted(glob.glob(sd + "/*")):
                if not (os.path.isdir(d) and os.path.exists(os.path.join(d, "dataset_info.json"))):
                    continue
                leaf = os.path.basename(d)
                if leaf not in selected_leaves:
                    continue
                stage = "watermark" if leaf == "watermarked" else "attack"
                yield (ds, "samark", f"flags-{flag_scope}", sampling, seg,
                       float(SAMARK.num_candidates),
                       flag_scope, msig, stage,
                       ("none" if stage == "watermark" else leaf), d)


def required_generation_cells():
    """Return the thirteen clean-generation identities required per dataset."""
    cells = {
        (rung.family, rung.mask, rung.sampling, rung.segmentation)
        for rungs in PAPER_LADDERS.values()
        for rung in rungs
    }
    cells.update({
        ("none", "none", "rejection", "sentence-nltk"),
        (PMARK.family, PMARK.mask, PMARK.sampling, PMARK.segmentation),
        (SAMARK.family, SAMARK.mask, SAMARK.sampling, SAMARK.segmentation),
    })
    return frozenset(cells)


def require_complete_generation(discovered, datasets) -> None:
    """Fail before compilation when any clean paper cell is absent or duplicated."""
    expected = required_generation_cells()
    counts = defaultdict(int)
    for (dataset, family, mask, sampling, segmentation, budget,
         _flag_scope, _msig, stage, _leaf, _directory) in discovered:
        if stage == "watermark":
            budget_ok = (
                budget == float(PROVIDER_CANDIDATES)
                if family in {"lsh", "kmeans", "pmark", "samark"}
                else budget == 1.0 if family == "none"
                else False
            )
            if not budget_ok:
                continue
            counts[(dataset, family, mask, sampling, segmentation)] += 1
    missing = [
        (dataset, *cell)
        for dataset in datasets
        for cell in sorted(expected)
        if counts[(dataset, *cell)] == 0
    ]
    duplicates = [identity for identity, count in counts.items()
                  if identity[1:] in expected and count > 1]
    if missing or duplicates:
        details = []
        if missing:
            details.append("missing: " + ", ".join(": ".join(item) for item in missing))
        if duplicates:
            details.append("duplicated: " + ", ".join(": ".join(item) for item in duplicates))
        raise RuntimeError(
            "paper generation matrix is incomplete; expected 13 clean cells for "
            f"each of {len(datasets)} datasets (N={PROVIDER_CANDIDATES}): "
            + "; ".join(details)
        )


def compile_results(
    argv=None, *, _datasets=None, _require_complete=True, _skip_transfer=False
):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=OUT, help="Output directory (default results/paper).")
    ap.add_argument(
        "--unit-encoder-device", default="cpu",
        help="Device for a new encoder-transfer pass (default: cpu).",
    )
    ap.add_argument(
        "--unit-encoder-batch-size", type=int, default=128,
        help="Sentence embedding batch size for the encoder-transfer table.",
    )
    ap.add_argument(
        "--force-unit-encoder-transfer", action="store_true",
        help="Recompute encoder-transfer distances even when its source fingerprint matches.",
    )
    args = ap.parse_args(argv)
    args.datasets = list(DATASETS if _datasets is None else _datasets)
    args.families = ["lsh", "kmeans", "pmark", "samark"]
    args.oracle_ks = list(ORACLE_KS)
    args.generation_num_candidates = [float(PROVIDER_CANDIDATES)]
    args.no_unit_encoder_transfer = _skip_transfer

    # Store one human-null copy per configuration.
    cells = defaultdict(lambda: dict(segments=[], human=None))
    human_by_cfg = {}
    skipped = []
    provider_evidence_errors = []
    provider_sources = []
    discovered = list(discover(args.datasets))
    if _require_complete:
        require_complete_generation(discovered, args.datasets)
    for (ds, family, mask, sampling, seg, generation_num_candidates,
         flag_scope, msig, stage, leaf, d) in discovered:
        para, wm, human, z_source = load_z(d, family, stage)
        if para is None:
            if stage == "watermark":
                raise RuntimeError(
                    f"required clean paper cell has no detector scores: {d}"
                )
            skipped.append((d, "no z-scores"))
            continue
        if (
            stage == "watermark"
            and family != "none"
            and not np.isfinite(np.asarray(para, dtype=float)).any()
        ):
            raise RuntimeError(
                f"required clean paper cell has no finite detector scores: {d}"
            )
        n = len(para)
        key = (family, mask, sampling, seg, generation_num_candidates,
               flag_scope, msig, leaf)
        c = cells[key]
        c["stage"] = stage
        c["z_source"] = z_source
        ps = per_sample_arrays(d, n)
        jr = judge_run_arrays(d, n)
        ps.update(judge_mean_arrays(jr, n))
        if stage == "watermark":
            missing_quality = [
                spec.key for spec in LLM_QUALITY_METRICS
                if spec.key not in ps
                or not np.isfinite(np.asarray(ps[spec.key], dtype=float)).any()
            ]
            if missing_quality:
                raise RuntimeError(
                    f"required clean paper cell lacks quality evidence {missing_quality}: {d}"
                )
        provider = None
        if stage == "watermark":
            try:
                provider = load_provider_evidence(
                    d, family, sampling, seg, generation_num_candidates,
                    os.path.join(ROOT, "data", ds),
                )
                provider_sources.append(dict(
                    dataset=ds,
                    family=family,
                    mask=mask,
                    sampling=sampling,
                    segmentation=seg,
                    marked_dataset=os.path.relpath(d, ROOT),
                    generation_stats=(
                        None if provider["stats_path"] is None
                        else os.path.relpath(provider["stats_path"], ROOT)
                    ),
                    source_dataset=os.path.relpath(
                        provider["prompt_evidence"]["source_dataset"], ROOT,
                    ),
                    prompt_config=(
                        None if provider["prompt_evidence"]["prompt_config"] is None
                        else os.path.relpath(
                            provider["prompt_evidence"]["prompt_config"], ROOT,
                        )
                    ),
                    prompt_rule=provider["prompt_evidence"]["prompt_rule"],
                    prompts_verified=provider["prompt_evidence"]["prompts_verified"],
                ))
            except ProviderEvidenceError as exc:
                provider_evidence_errors.append(str(exc))
        c["segments"].append(dict(dataset=ds, dir=d, n=n, para=np.asarray(para, float),
                                  wm=np.asarray(wm, float) if wm is not None else np.full(n, np.nan),
                                  ps=ps, jr=jr, sc=scalars(d), provider=provider))
        if human is not None:
            cfg_key = (family, mask, sampling, seg, flag_scope, msig)
            human_by_cfg.setdefault(cfg_key, np.asarray(human, dtype=float))
            c["human"] = human_by_cfg[cfg_key]

    if provider_evidence_errors:
        details = "\n".join(
            f"  - {message}" for message in sorted(set(provider_evidence_errors))
        )
        raise ProviderEvidenceError(
            "provider-draw compilation requires complete clean-generation evidence; "
            f"found {len(set(provider_evidence_errors))} problem(s):\n{details}"
        )

    os.makedirs(args.out, exist_ok=True)

    print(f"cells: {len(cells)}   dirs: {sum(len(c['segments']) for c in cells.values())}"
          f"   skipped: {len(skipped)}")

    # The registry defines a stable compiled schema. Missing physical arrays
    # remain NaN per sample and required paper metrics are checked by extraction.
    ps_keys = sorted(
        set(PER_SAMPLE_QUALITY_KEYS)
        | {k for c in cells.values() for s in c["segments"] for k in s["ps"]}
    )
    print(f"per-sample metrics carried: {len(ps_keys)}")

    # Pass 2: pool thresholds and AUROC, then emit rows.
    ps_rows, cell_rows, dataset_scalar_rows = [], [], []
    for (family, mask, sampling, seg, generation_num_candidates,
         flag_scope, msig, leaf), c in sorted(cells.items()):
        para_all = np.concatenate([s["para"] for s in c["segments"]])
        has_detection = c["z_source"] != "unavailable"
        if not has_detection:
            thr10 = thr5 = thr1 = float("nan")
            auroc = float("nan")
        elif family == "pmark" and mask == "online":
            thr10, thr5, thr1 = PHI10, PHI5, PHI1
            auroc = auroc_analytic(para_all)
        else:
            human = np.nan_to_num(c["human"]) if c["human"] is not None else np.array([])
            if human.size:
                thr10, thr5, thr1 = empirical_null_thresholds(human)
            else:
                thr10, thr5, thr1 = PHI10, PHI5, PHI1
            auroc = auroc_empirical(para_all, human)

        meta = dict(scheme=FAMILIES[family], family=family, mask=mask, sampling=sampling,
                    segmentation=seg, flag_scope=flag_scope, msig=msig,
                    generation_num_candidates=generation_num_candidates,
                    **parse_segmentation(seg),
                    stage=c["stage"], z_source=c["z_source"],
                    **parse_attack(leaf))
        provider_metrics = (
            aggregate_provider_metrics(
                family, sampling, seg,
                [s["provider"] for s in c["segments"]],
            )
            if c["stage"] == "watermark"
            else empty_provider_metrics()
        )

        pooled = defaultdict(list)
        for s in c["segments"]:
            dataset_scalar_rows.append(
                dict(meta, dataset=s["dataset"], **{
                    key: s["sc"].get(key, np.nan)
                    for key in PERSISTED_QUALITY_SCALARS
                }, **(
                    aggregate_provider_metrics(
                        family, sampling, seg, [s["provider"]],
                    )
                    if c["stage"] == "watermark"
                    else empty_provider_metrics()
                ))
            )
            z, wm, n = s["para"], s["wm"], s["n"]
            qual = s["ps"].get(QUALITY_METRIC, np.full(n, np.nan))
            for i in range(n):
                if has_detection:
                    evades = not z[i] > thr5
                    # NaN >= bar is False: unjudged samples never count.
                    q_ok = bool(qual[i] >= QUALITY_BAR)
                    detection = dict(
                        z=float(z[i]), z_wm=float(wm[i]), z_drop=float(wm[i] - z[i]),
                        thr10=thr10, thr5=thr5, thr1=thr1,
                        detected_at10=int(z[i] > thr10), detected_at5=int(z[i] > thr5),
                        detected_at1=int(z[i] > thr1), evades_at5=int(evades),
                        quality_ok=int(q_ok), success_at5=int(evades and q_ok),
                        success_at1=int((not z[i] > thr1) and q_ok),
                    )
                else:
                    detection = dict(
                        z=np.nan, z_wm=np.nan, z_drop=np.nan,
                        thr10=np.nan, thr5=np.nan, thr1=np.nan,
                        detected_at10=np.nan, detected_at5=np.nan,
                        detected_at1=np.nan, evades_at5=np.nan,
                        quality_ok=np.nan, success_at5=np.nan, success_at1=np.nan,
                    )
                row = dict(meta, dataset=s["dataset"], sample_id=i, **detection)
                for k in ps_keys:
                    row[k] = float(s["ps"][k][i]) if k in s["ps"] else np.nan
                for k in JUDGE_RUN_KEYS:
                    row[k] = float(s["jr"][k][i])
                ps_rows.append(row)
            for k in ps_keys:
                pooled[k].append(s["ps"].get(k, np.full(n, np.nan)))

        n_tot = len(para_all)
        wm_all = np.concatenate([s["wm"] for s in c["segments"]])
        detection_summary = (
            dict(
                TPR_at10=float(np.mean(para_all > thr10)),
                TPR_at5=float(np.mean(para_all > thr5)),
                TPR_at1=float(np.mean(para_all > thr1)),
                evade_rate_at5=float(np.mean(para_all <= thr5)),
                mean_z=float(np.nanmean(para_all)),
                mean_z_wm=float(np.nanmean(wm_all)),
                mean_z_drop=float(np.nanmean(wm_all - para_all)),
            )
            if has_detection
            else dict(
                TPR_at10=np.nan, TPR_at5=np.nan, TPR_at1=np.nan,
                evade_rate_at5=np.nan, mean_z=np.nan, mean_z_wm=np.nan,
                mean_z_drop=np.nan,
            )
        )
        cell = dict(meta, n_samples=n_tot, n_sets=len(c["segments"]),
                    datasets="|".join(sorted(s["dataset"] for s in c["segments"])),
                    thr10=thr10, thr5=thr5, thr1=thr1, auroc=auroc,
                    **detection_summary)
        cell.update(provider_metrics)

        # Quality-bounded attack success: evades AND preserves content at/above the bar.
        qual_all = (np.concatenate(pooled[QUALITY_METRIC]) if pooled.get(QUALITY_METRIC)
                    else np.full(n_tot, np.nan))
        if c["stage"] != "attack" or not has_detection or np.isnan(qual_all).all():
            # ASR is undefined before an attack or without detector scores.
            cell.update({f"ASR_{tag}_q{int(round(b * 100))}": np.nan
                         for tag, _ in ASR_FPRS for b in ASR_BARS})
            cell.update(ASR_at5=np.nan, ASR_at1=np.nan, evade_rate_at1=np.nan,
                        quality_pass_rate=np.nan)
        else:
            with np.errstate(invalid="ignore"):
                for tag, thr_key in ASR_FPRS:
                    evade = para_all <= {"thr5": thr5, "thr1": thr1}[thr_key]
                    for b in ASR_BARS:
                        cell[f"ASR_{tag}_q{int(round(b * 100))}"] = float(np.mean(evade & (qual_all >= b)))
                    cell[f"ASR_{tag}"] = float(np.mean(evade & (qual_all >= QUALITY_BAR)))
                cell["evade_rate_at1"] = float(np.mean(para_all <= thr1))
                cell["quality_pass_rate"] = float(np.mean(qual_all >= QUALITY_BAR))
        for k in ps_keys:
            if k in PERPLEXITY_EVIDENCE_KEYS:
                continue
            v = np.concatenate(pooled[k]) if pooled[k] else np.array([])
            valid = v[~np.isnan(v)] if v.size else v
            cell[k] = float(np.mean(valid)) if valid.size else np.nan
            cell[f"{k}_ci"] = _confidence_interval(v) if valid.size else np.nan
            cell[f"{k}_median"] = float(np.median(valid)) if valid.size else np.nan
        # Override the diagnostic per-document mean with the persisted corpus
        # aggregate. CI and median remain descriptions of document-level PPL.
        corpus_ppl, total_nll, scored_tokens = aggregate_corpus_perplexity(
            c["segments"]
        )
        cell["gen_ppl"] = corpus_ppl
        cell["gen_ppl_total_nll"] = total_nll
        cell["gen_ppl_scored_tokens"] = scored_tokens
        # Corpus-level scalars have no per-sample array: sample-weighted mean over sets.
        for k in CORPUS_SCALARS:
            vals = [(s["sc"].get(k, np.nan), s["n"]) for s in c["segments"]]
            vals = [(v, w) for v, w in vals if not (v is None or np.isnan(v))]
            cell[k] = float(np.average([v for v, _ in vals], weights=[w for _, w in vals])) \
                if vals else np.nan
        cell_rows.append(cell)

    # Emit empirical human nulls as first-class rows.
    for (family, mask, sampling, seg, flag_scope, msig), hz in sorted(human_by_cfg.items()):
        thr = next(((cl["thr10"], cl["thr5"], cl["thr1"]) for cl in cell_rows
                    if (cl["family"], cl["mask"], cl["sampling"], cl["segmentation"],
                        cl["flag_scope"], cl["msig"])
                    == (family, mask, sampling, seg, flag_scope, msig)), None)
        if thr is None:
            continue
        thr10, thr5, thr1 = thr
        meta = dict(scheme=FAMILIES[family], family=family, mask=mask, sampling=sampling,
                    segmentation=seg, flag_scope=flag_scope, msig=msig,
                    generation_num_candidates=np.nan,
                    **parse_segmentation(seg),
                    stage="human", z_source="direct", **parse_attack("none"))
        clean = np.nan_to_num(hz)
        for i, z in enumerate(hz):
            # Detection flags on human text are false positives.
            ps_rows.append(dict(meta, dataset=HUMAN_CORPUS, sample_id=i,
                                z=float(z), z_wm=np.nan, z_drop=np.nan,
                                thr10=thr10, thr5=thr5, thr1=thr1,
                                detected_at10=int(z > thr10), detected_at5=int(z > thr5),
                                detected_at1=int(z > thr1), evades_at5=np.nan,
                                quality_ok=np.nan, success_at5=np.nan, success_at1=np.nan,
                                **{k: np.nan for k in ps_keys},
                                **{k: np.nan for k in JUDGE_RUN_KEYS}))
        cell = dict(meta, n_samples=len(hz), n_sets=1, datasets=HUMAN_CORPUS,
                    thr10=thr10, thr5=thr5, thr1=thr1, auroc=np.nan,
                    mean_z=float(np.nanmean(hz)), mean_z_wm=np.nan, mean_z_drop=np.nan,
                    FPR_at10=float(np.mean(clean > thr10)), FPR_at5=float(np.mean(clean > thr5)),
                    FPR_at1=float(np.mean(clean > thr1)),
                    null_sd=float(np.nanstd(hz)),
                    null_q90=float(np.quantile(clean, 0.90)),
                    null_q95=float(np.quantile(clean, 0.95)),
                    null_q99=float(np.quantile(clean, 0.99)))
        cell.update(empty_provider_metrics())
        cell_rows.append(cell)

    ps_df = pd.DataFrame(ps_rows)
    cell_df = pd.DataFrame(cell_rows)

    lead = ["scheme", "family", "mask", "sampling", "segmentation", "flag_scope", "msig",
            "segmentation_type", "segmentation_backend", "segmentation_max_words",
            "segmentation_window", "generation_num_candidates", "stage", "z_source", "attack",
            "attack_family", "attack_setting", "K", "surrogate", "anchor",
            "attacker_model", "attacker_seg", "candidate_agg", "bag_agg",
            "dipper_lex", "dipper_order"]
    ps_df = ps_df[lead + ["dataset", "sample_id"] +
                  [c for c in ps_df.columns if c not in lead + ["dataset", "sample_id"]]]
    cell_df = cell_df[lead + [c for c in cell_df.columns if c not in lead]]
    cell_df = cell_df.sort_values(lead).reset_index(drop=True)

    # Preserve one summary per physical dataset shard. The pooled cell table is
    # appropriate for headline estimates, but attack arms are often completed
    # incrementally. Pairing pooled cells with different shard sizes would be a
    # confound; dataset_summary lets plots compare only their shared shards.
    dataset_rows = []
    group_keys = lead + ["dataset"]
    for identity, group in ps_df.groupby(group_keys, dropna=False, sort=False):
        row = dict(zip(group_keys, identity))
        row.update(
            n_samples=len(group), n_sets=1, datasets=row["dataset"],
            thr10=float(group.thr10.iloc[0]),
            thr5=float(group.thr5.iloc[0]),
            thr1=float(group.thr1.iloc[0]),
            mean_z=float(group.z.mean()),
            mean_z_wm=float(group.z_wm.mean()),
            mean_z_drop=float(group.z_drop.mean()),
        )
        row.update(empty_provider_metrics())
        if row["stage"] == "human":
            row.update(
                FPR_at10=float(group.detected_at10.mean()),
                FPR_at5=float(group.detected_at5.mean()),
                FPR_at1=float(group.detected_at1.mean()),
                null_sd=float(group.z.std(ddof=0)),
                null_q90=float(group.z.quantile(0.90)),
                null_q95=float(group.z.quantile(0.95)),
                null_q99=float(group.z.quantile(0.99)),
            )
        else:
            row.update(
                TPR_at10=float(group.detected_at10.mean()),
                TPR_at5=float(group.detected_at5.mean()),
                TPR_at1=float(group.detected_at1.mean()),
                evade_rate_at5=float(group.evades_at5.mean()),
            )
            quality = group.get(QUALITY_METRIC)
            if quality is not None and quality.notna().any():
                for tag, threshold_column in ASR_FPRS:
                    evades = group.z <= group[threshold_column]
                    for bar in ASR_BARS:
                        row[f"ASR_{tag}_q{int(round(bar * 100))}"] = float(
                            (evades & (quality >= bar)).mean()
                        )
                    row[f"ASR_{tag}"] = float(
                        (evades & (quality >= QUALITY_BAR)).mean()
                    )
                row["evade_rate_at1"] = float((group.z <= group.thr1).mean())
                row["quality_pass_rate"] = float((quality >= QUALITY_BAR).mean())
        for key in ps_keys:
            if key in PERPLEXITY_EVIDENCE_KEYS:
                continue
            values = group[key].dropna() if key in group else pd.Series(dtype=float)
            row[key] = float(values.mean()) if len(values) else np.nan
            row[f"{key}_ci"] = _confidence_interval(values) if len(values) else np.nan
            row[f"{key}_median"] = float(values.median()) if len(values) else np.nan
        dataset_rows.append(row)
    dataset_df = pd.DataFrame(dataset_rows)
    if dataset_scalar_rows:
        scalar_df = pd.DataFrame(dataset_scalar_rows).drop_duplicates(group_keys)
        # gen_ppl was provisionally summarized from its diagnostic per-sample
        # array above. Replace only that aggregate with the persisted corpus PPL.
        dataset_df = dataset_df.drop(
            columns=[
                key for key in (*PERSISTED_QUALITY_SCALARS, *PROVIDER_METRIC_FIELDS)
                if key in dataset_df
            ],
        )
        dataset_df = dataset_df.merge(scalar_df, on=group_keys, how="left", validate="one_to_one")
    dataset_df = dataset_df.sort_values(group_keys).reset_index(drop=True)

    # Comparison families have no signature width while SAMark stores an
    # integer width. Keep one nullable numeric dtype so Parquet does not see a
    # mixed object column of empty strings and integers.
    for frame in (ps_df, cell_df, dataset_df):
        frame["msig"] = pd.to_numeric(frame["msig"], errors="coerce")

    ps_path = os.path.join(args.out, "per_sample.parquet")
    cell_path = os.path.join(args.out, "cell_summary.csv")
    dataset_path = os.path.join(args.out, "dataset_summary.csv")
    ps_df.to_parquet(ps_path, index=False)
    cell_df.to_csv(cell_path, index=False)
    dataset_df.to_csv(dataset_path, index=False)

    transfer_df = pd.DataFrame()
    transfer_manifest = None
    if not args.no_unit_encoder_transfer:
        transfer_df, transfer_manifest = build_transfer_table(
            ROOT, args.datasets, args.out,
            device=args.unit_encoder_device,
            batch_size=args.unit_encoder_batch_size,
            force=args.force_unit_encoder_transfer,
        )
        transfer_df = decorate_transfer_rows(transfer_df)
        transfer_df.to_parquet(
            os.path.join(args.out, f"{OUTPUT_STEM}.parquet"), index=False,
        )

    manifest = dict(
        generated_by="visualization.compile_results",
        datasets=args.datasets,
        selected_families=args.families,
        selected_oracle_ks=args.oracle_ks,
        generation_num_candidates=args.generation_num_candidates,
        families={k: v for k, v in FAMILIES.items()},
        pooling=("cell = (family, mask, sampling, segmentation, attack); pooled over "
                 "the three disjoint paper datasets; dataset_summary retains shards"),
        segmentation_identity=dict(
            exact_column="segmentation",
            parsed_columns=["segmentation_type", "segmentation_backend",
                            "segmentation_max_words", "segmentation_window"],
            note="semantic-span max/window/backend variants remain distinct cells",
        ),
        detection=dict(
            none=("quality-only no-watermark baseline; detector scores, thresholds, "
                  "detection rates, and attack-success fields are undefined"),
            lsh=("empirical human null pooled over sets; conservative higher-order "
                 "thr@10/5/1 = q90/q95/q99; AUROC uses every null document"),
            kmeans="same as lsh",
            pmark=("online only: analytic N(0,1), thr = "
                   "1.2816/1.6449/2.3263 and AUROC = mean(Phi(z))"),
            samark=("flags-run only: empirical human null with conservative "
                    "higher-order quantiles; msig is retained as cell metadata")),
        cell_aggregation=dict(
            per_sample_metrics="pooled mean + 95% t CI half-width + median over all sets",
            corpus_scalars={k: "sample-weighted mean of per-set values" for k in CORPUS_SCALARS},
            gen_ppl=("exp(sum(gen_ppl_total_nll) / "
                     "sum(gen_ppl_scored_tokens)) across available shards"),
        ),
        provider_draws=dict(
            headline=("expected language-model candidate draws needed to produce one "
                      "final clean output sentence"),
            domain=("stage=watermark rows only, including the one-draw no-watermark "
                    "baseline; attacked and human rows are undefined"),
            interpretation=("algorithmic sequential-draw estimate, not physical runtime "
                            "work; the current batched implementation generates the full "
                            "candidate pool before selection"),
            source_files=dict(
                generation=("each clean SemStamp/k-SemStamp/no-watermark generation "
                            "directory's generation_stats.json"),
                output_sentences=("the text column of each clean watermarked HuggingFace "
                                  "dataset after exact prompt-prefix removal, segmented with "
                                  "sentence-NLTK; prompts are stored but are verified and not "
                                  "counted"),
                prompts=("SemStamp/k-SemStamp/no-watermark: recover "
                         "extract_prompt_from_text from each "
                         "resolved_config.yaml's io.data_path and generation.len_prompt; "
                         "PMark/SAMark: use the persisted first-source-sentence prompt; every "
                         "source row and stored output prefix must match exactly"),
                comparison_systems=("clean PMark/SAMark watermarked datasets; their fixed "
                                    "64-candidate-per-sentence rule supplies draw counts"),
                consumed=sorted(
                    provider_sources,
                    key=lambda item: (
                        item["dataset"], item["family"], item["mask"],
                        item["sampling"], item["segmentation"],
                    ),
                ),
            ),
            aggregation=dict(
                rejection=("pool n_units_tracked, mean_p_green_per_unit, and "
                           "std_p_green_per_unit across complete shards using within- plus "
                           "between-shard sums of squares; form a two-sided 95% Student-t "
                           "interval for the pooled valid-candidate rate; clip rate endpoints "
                           "to [1e-9, 1], then invert in reverse order"),
                fixed_pool=("PMark, SAMark, and best-of-N use their full fixed pool per "
                            "provider unit; both candidate-draw confidence bounds equal the "
                            "point value and describe draw count, not runtime variance"),
                sentence=("provider_units_per_sentence is defined as 1; raw total_units and "
                          "the independent NLTK final-sentence recount are retained for audit"),
                semspan_best_of_n=("sum generation_stats.total_units across shards divided by "
                                  "the summed final NLTK output-sentence count, then multiplied "
                                  "by the fixed candidate pool"),
                dataset_summary="one physical shard, reproducing that shard's raw evidence",
                cell_summary="pool complete shard evidence before any rate or ratio",
            ),
            rejection_required_fields=list(REJECTION_STATS_FIELDS),
            count_fields=["total_units", "final output sentence count"],
            interval_limitation=("the observed provider-unit-to-output-sentence ratio is "
                                 "treated as fixed when scaling per-unit confidence bounds to "
                                 "per-sentence bounds"),
        ),
        quality_metrics=quality_metric_manifest(),
        judge_runs=dict(
            table="per_sample.parquet",
            runs=JUDGE_RUN_COUNT,
            aggregation="mean across three judge runs",
            columns=list(JUDGE_RUN_KEYS),
            note=("criterion runs are exported directly from the quality NPZ; "
                  "per-run llm_quality and llm_judge headlines are reconstructed "
                  "with the evaluator's weights and criterion valence; all ten "
                  "document aggregates are rebuilt from these run columns before "
                  "summary statistics and quality-gated ASR are computed"),
        ),
        attack_success=dict(
            definition=f"a sample succeeds iff it evades detection at the operating point AND "
                       f"{QUALITY_METRIC} >= bar; unjudged samples never count as success",
            quality_metric=QUALITY_METRIC, headline_bar=QUALITY_BAR,
            columns="ASR_at5 / ASR_at1 at the headline bar; ASR_{at5,at1}_q65..q95 sweep the bar in 0.05 increments",
            bars=ASR_BARS, operating_points=["5% FPR", "1% FPR"]),
        human_null=dict(
            rows=f'stage="human" rows carry the empirical null per config, from {HUMAN_CORPUS}',
            note="identical across attacks and eval sets, so stored once per config "
                 "(family, mask, sampling, segmentation); online PMark has no human rows",
            cell_columns="FPR_at10/5/1 (empirical false-positive rates) + null_sd/q90/q95/q99"),
        unit_encoder_transfer=(
            None if transfer_manifest is None else dict(
                table=f"{OUTPUT_STEM}.parquet",
                rows=len(transfer_df),
                families=sorted(transfer_df.family.unique()) if len(transfer_df) else [],
                budgets=sorted(transfer_df.K.unique()) if len(transfer_df) else [],
                alignment="positional sentence index, fixed before encoder evaluation",
                distance="max(0, 1 - cosine) on normalized sentence text",
                surrogate="BAAI/bge-base-en-v1.5",
                providers=(sorted(transfer_df.provider_encoder.unique())
                           if len(transfer_df) else []),
                selected_draws_only=True,
                provenance_file=f"{OUTPUT_STEM}.manifest.json",
            )
        ),
        counts=dict(cells=len(cell_df), dataset_cells=len(dataset_df),
                    per_sample_rows=len(ps_df),
                    unit_encoder_transfer_rows=len(transfer_df),
                    dirs=int(cell_df["n_sets"].sum()), metrics_per_sample=len(ps_keys),
                    judge_run_columns=len(JUDGE_RUN_KEYS)),
        skipped=[dict(dir=os.path.relpath(d, ROOT), why=w) for d, w in skipped],
    )
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nwrote {cell_path}  {cell_df.shape}")
    print(f"wrote {dataset_path}  {dataset_df.shape}")
    print(f"wrote {ps_path}  {ps_df.shape}")
    if not args.no_unit_encoder_transfer:
        print(
            f"wrote {os.path.join(args.out, f'{OUTPUT_STEM}.parquet')}  "
            f"{transfer_df.shape}"
        )
    for p in (cell_path, dataset_path, ps_path):
        print(f"   {os.path.basename(p):18s} {os.path.getsize(p) / 1e6:8.1f} MB")
    return Path(args.out)


def main(argv=None):
    """Command-line entry point."""
    return compile_results(argv)


if __name__ == "__main__":
    main()
