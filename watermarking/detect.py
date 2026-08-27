"""Detection adapted from original SemStamp's ``detection_utils.py``."""

import argparse
import csv
import json
import os

import numpy as np
import torch
from tqdm import tqdm, trange
from datasets import load_from_disk
from sklearn.metrics import roc_curve, auc
from sentence_transformers import SentenceTransformer
from segmentation import (
    DEFAULT_BACKEND,
    DEFAULT_TYPE,
    Segmenter,
        display_with_boundary_space,
)
from config.cli import add_config_args, resolve
from config.loader import dump_config
from config.paths import target_dir, watermark_dir
from watermarking.primitives import (
    SBERTLSHModel,
    get_mask_from_seed,
    get_cluster_mask,
    get_cluster_id,
    hash_key,
)


def compute_zscore(n_watermark, n_test_sent, lmbd):
    """Compute z-score from watermark count."""
    num = n_watermark - lmbd * n_test_sent
    denom = np.sqrt(n_test_sent * lmbd * (1 - lmbd))
    return num / denom


def null_distribution_threshold(null_scores, fpr=0.01):
    """Return the z cutoff calibrated to the requested null false-positive rate."""
    if not 0 < fpr < 1:
        raise ValueError(f"fpr must be between 0 and 1, got {fpr!r}")
    scores = np.nan_to_num(np.asarray(null_scores, dtype=float))
    if scores.size == 0:
        raise ValueError("Cannot calibrate a threshold from an empty null distribution.")
    return float(np.percentile(scores, 100 * (1 - fpr)))


def count_lsh_watermark_hits(units, lsh_model, lmbd, lsh_dim, secret_message=None, key=None):
    """Count LSH hits across normalized units."""
    if key is None:
        key = hash_key
    if not units:
        return 0, 0
    n_watermark = 0
    n_test_units = 0
    unit_hashes = lsh_model.get_hash([unit.normalized for unit in units])
    if secret_message is None:
        lsh_seed = unit_hashes[0]
        start_idx = 1
    else:
        lsh_seed = lsh_model.get_hash([secret_message])[0]
        start_idx = 0
    accept_mask = get_mask_from_seed(lsh_dim, lmbd, lsh_seed, key=key)
    for i in range(start_idx, len(units)):
        if unit_hashes[i] in accept_mask:
            n_watermark += 1
        n_test_units += 1
        if secret_message is None:
            lsh_seed = unit_hashes[i]
            accept_mask = get_mask_from_seed(lsh_dim, lmbd, lsh_seed, key=key)
    return n_watermark, n_test_units


def detect_lsh(units, lsh_model, lmbd, lsh_dim, secret_message=None, key=None):
    """Return the LSH watermark z-score for units."""
    n_watermark, n_test_units = count_lsh_watermark_hits(
        units, lsh_model, lmbd, lsh_dim, secret_message=secret_message, key=key,
    )
    return compute_zscore(n_watermark, n_test_units, lmbd)


def count_kmeans_watermark_hits(units, embedder, lmbd, k_dim, cluster_centers, secret_message=None, key=None):
    """Count KMeans hits across normalized units."""
    if key is None:
        key = hash_key
    if not units:
        return 0, 0
    n_watermark = 0
    n_test_units = 0
    unit_cluster_ids = [
        get_cluster_id(unit.normalized, embedder=embedder, cluster_centers=cluster_centers)
        for unit in units
    ]
    if secret_message is None:
        curr_cluster_id = unit_cluster_ids[0]
        start_idx = 1
    else:
        curr_cluster_id = get_cluster_id(secret_message, embedder=embedder, cluster_centers=cluster_centers)
        start_idx = 0
    cluster_mask = get_cluster_mask(curr_cluster_id, k_dim, lmbd, key=key)
    for i in range(start_idx, len(units)):
        if unit_cluster_ids[i] in cluster_mask:
            n_watermark += 1
        n_test_units += 1
        if secret_message is None:
            curr_cluster_id = unit_cluster_ids[i]
            cluster_mask = get_cluster_mask(curr_cluster_id, k_dim, lmbd, key=key)
    return n_watermark, n_test_units


def detect_kmeans(units, embedder, lmbd, k_dim, cluster_centers, secret_message=None, key=None):
    """Return the KMeans watermark z-score for units."""
    n_watermark, n_test_units = count_kmeans_watermark_hits(
        units, embedder, lmbd, k_dim, cluster_centers, secret_message=secret_message, key=key,
    )
    return compute_zscore(n_watermark, n_test_units, lmbd)


def get_roc_metrics(labels, preds):
    fpr, tpr, _ = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), float(roc_auc)


def get_roc_metrics_from_zscores(mp, h, dataset_path, suffix=""):
    mp = np.nan_to_num(mp)
    h = np.nan_to_num(h)
    len_z = len(mp)
    mp_fpr, mp_tpr, mp_area = get_roc_metrics(
        [1] * len_z + [0] * len_z, np.concatenate((mp, h[:len_z])))
    np.save(os.path.join(dataset_path, f"fpr{suffix}.npy"), mp_fpr)
    np.save(os.path.join(dataset_path, f"tpr{suffix}.npy"), mp_tpr)
    return mp_area, mp_fpr


def evaluate_z_scores(mpz, hz, dataset_path, suffix=""):
    mpz = np.array(mpz)
    hz = np.nan_to_num(np.array(hz))
    fpr_1_threshold = null_distribution_threshold(hz, fpr=0.01)
    fpr_5_threshold = null_distribution_threshold(hz, fpr=0.05)
    mp_area, _ = get_roc_metrics_from_zscores(mpz, hz, dataset_path, suffix=suffix)
    return mp_area, len(mpz[mpz > fpr_1_threshold]) / len(mpz), len(mpz[mpz > fpr_5_threshold]) / len(mpz)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    return resolve(parser.parse_args())


def setup_detector(cfg):
    """Build the configured detector and description prefix."""
    wm = cfg.watermark
    mode = cfg.detection.mode or wm.mode
    if mode is None:
        raise ValueError("detection.mode (or watermark.mode) is required.")
    # "_fixed" also matches "_fixed_diverse": the diverse constraint is
    # generation-only, so its detector is exactly the fixed-mask detector.
    is_fixed = "_fixed" in mode
    secret_msg = wm.secret_message if is_fixed else None

    if "lsh" in mode:
        lsh_model = SBERTLSHModel(
            lsh_model_path=wm.embedder, device='cuda', batch_size=1, lsh_dim=wm.sp_dim, sbert_type='base'
        )
        detect_fn = lambda units: detect_lsh(
            units=units, lsh_model=lsh_model, lmbd=wm.lmbd, lsh_dim=wm.sp_dim,
            secret_message=secret_msg, key=wm.hash_key,
        )
        encoder = lsh_model.embedder
    elif "kmeans" in mode:
        cluster_centers = torch.load(wm.cc_path)
        embedder = SentenceTransformer(wm.embedder)
        detect_fn = lambda units: detect_kmeans(
            units=units, embedder=embedder, lmbd=wm.lmbd, k_dim=wm.sp_dim,
            cluster_centers=cluster_centers, secret_message=secret_msg, key=wm.hash_key,
        )
        encoder = embedder
    else:
        raise ValueError(f"Unknown detection mode: {mode}")

    return detect_fn, mode, encoder


def run_detection(detect_fn, texts, desc, segmenter: Segmenter):
    """Run detection on a list of texts."""
    scores = []
    for i in trange(len(texts), desc=desc):
        units = segmenter.segment(texts[i])
        score = detect_fn(units)
        scores.append(score)
    return scores


def truncate_to_generation_budget(
    text, sentence_segmenter, segmenter, tokenizer, max_new_tokens,
):
    """Cut one human document to the shape a watermarked generation has.

    Generation prompts with the document's first sentence and then appends
    units until the token count past that prompt crosses ``max_new_tokens``,
    keeping the unit that crosses it. Replaying the rule over human text gives
    the null the same scored length as the positives it calibrates. Documents
    that never reach the budget are returned whole.
    """
    sentences = sentence_segmenter.segment(text)
    if not sentences:
        return text
    prompt = sentences[0].display.strip()
    tail = "".join(unit.display for unit in sentences[1:])
    prompt_len = len(tokenizer.encode(prompt))
    out = prompt
    for unit in segmenter.segment(tail):
        out += display_with_boundary_space(out, unit.display)
        if len(tokenizer.encode(out)) - prompt_len >= max_new_tokens:
            break
    return out


def truncate_human_null(texts, cfg, segmenter):
    """Length-match the human null to this run's watermarked positives."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.generation.model)
    # Generation extracts its prompt with the default sentence splitter
    # whatever the arm's segmentation is, so the seed sentence matches it.
    sentence_segmenter = Segmenter(DEFAULT_TYPE, DEFAULT_BACKEND)
    return [
        truncate_to_generation_budget(
            text, sentence_segmenter, segmenter, tokenizer,
            cfg.generation.max_new_tokens,
        )
        for text in tqdm(texts, desc="truncating human null")
    ]


def _human_cache_metadata_path(dataset_path):
    return os.path.join(dataset_path, "human_z_scores.segmentation.json")


def _human_cache_matches(dataset_path, segmenter: Segmenter):
    metadata_path = _human_cache_metadata_path(dataset_path)
    if not os.path.exists(metadata_path):
        return False
    try:
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    return metadata == segmenter.metadata()


def _write_human_cache_metadata(dataset_path, segmenter: Segmenter):
    with open(_human_cache_metadata_path(dataset_path), "w", encoding="utf-8") as f:
        json.dump(segmenter.metadata(), f, sort_keys=True)


def save_results(
    dataset_path,
    segmenter: Segmenter,
    z_scores,
    para_scores,
    human_scores,
):
    """Save detection results and compute metrics."""
    print("Saving scores...")
    print(f"Average z-score of generations: {np.nanmean(z_scores):.3f}")
    print(f"Average z-score of human texts: {np.nanmean(human_scores):.3f}")

    np.save(os.path.join(dataset_path, "z_scores.npy"), z_scores)
    np.save(os.path.join(dataset_path, "human_z_scores.npy"), human_scores)
    _write_human_cache_metadata(dataset_path, segmenter)

    if len(para_scores) > 0:
        print(f"Average z-score of paraphrased texts: {np.nanmean(para_scores):.3f}")
        np.save(os.path.join(dataset_path, "para_z_scores.npy"), para_scores)

    print("Evaluating watermarked z-scores...")
    wm_auroc, wm_fpr1, wm_fpr5 = evaluate_z_scores(z_scores, human_scores, dataset_path, suffix="_wm")
    wm_results = {
        "auroc": f"{wm_auroc:.3f}",
        "fpr1": f"{wm_fpr1:.3f}",
        "fpr5": f"{wm_fpr5:.3f}",
        "mean_z": f"{np.nanmean(z_scores):.3f}",
        "human_mean_z": f"{np.nanmean(human_scores):.3f}",
    }
    wm_csv_path = os.path.join(dataset_path, "results_wm.csv")
    with open(wm_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=wm_results.keys())
        writer.writeheader()
        writer.writerow(wm_results)

    if len(para_scores) > 0:
        print("Evaluating paraphrased z-scores...")
        auroc, fpr1, fpr5 = evaluate_z_scores(para_scores, human_scores, dataset_path)

        para_results = {
            "auroc": f"{auroc:.3f}",
            "fpr1": f"{fpr1:.3f}",
            "fpr5": f"{fpr5:.3f}",
        }
        para_csv_path = os.path.join(dataset_path, "results.csv")
        with open(para_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=para_results.keys())
            writer.writeheader()
            writer.writerow(para_results)


def main():
    cfg = parse_args()
    if not cfg.io.data_path:
        raise ValueError("No data_path set. Pass it positionally or via io.data_path.")
    # Select the configured or explicit target directory.
    dataset_path = cfg.io.output_dir or target_dir(cfg)
    seg_type = cfg.segmentation.type
    seg_backend = cfg.segmentation.backend

    dataset = load_from_disk(dataset_path)
    gens = dataset['text']
    paras = dataset['para_text'] if 'para_text' in dataset.column_names else None
    human_texts = load_from_disk(cfg.detection.human_text)['text'][:max(len(gens), 1024)]

    detect_fn, mode_desc, encoder = setup_detector(cfg)
    segmenter = (
        Segmenter.from_config(
            cfg.segmentation,
            encoder=encoder,
            encoder_id=cfg.watermark.embedder,
            batch_size=cfg.runtime.semcut_batch_size,
        )
        if seg_type == "semspan"
        else Segmenter(seg_type, seg_backend)
    )

    z_scores = run_detection(
        detect_fn, gens, f'{mode_desc}_detection', segmenter=segmenter,
    )
    para_scores = run_detection(
        detect_fn, paras, f'{mode_desc}_para', segmenter=segmenter,
    ) if paras else []

    # Reuse the watermarked directory's human-score cache for attacks.
    wm_data_path = cfg.detection.wm_data_path
    if wm_data_path is None and cfg.io.target == "attack":
        wm_data_path = watermark_dir(cfg)
    cached_human_dir = wm_data_path if wm_data_path else dataset_path
    cached_human_path = os.path.join(cached_human_dir, "human_z_scores.npy")
    if os.path.exists(cached_human_path) and _human_cache_matches(
        cached_human_dir, segmenter,
    ):
        print(f"Loading cached human z-scores from {cached_human_path}")
        human_scores = np.load(cached_human_path).tolist()
    else:
        if os.path.exists(cached_human_path):
            print(
                f"Ignoring cached human z-scores from {cached_human_path}: "
                "segmentation settings do not match."
            )
        human_scores = run_detection(
            detect_fn, truncate_human_null(human_texts, cfg, segmenter),
            f'{mode_desc}_human', segmenter=segmenter,
        )

    print("First 5 z-scores of generated texts:", z_scores[:5])
    print("First 5 z-scores of paraphrased texts:", para_scores[:5] if para_scores else "N/A")
    print("First 5 z-scores of human texts:", human_scores[:5])

    save_results(
        dataset_path, segmenter, z_scores, para_scores, human_scores,
    )
    dump_config(cfg, os.path.join(dataset_path, "resolved_config.detect.yaml"))


if __name__ == '__main__':
    main()
