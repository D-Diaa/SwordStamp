"""Run an end-to-end watermark smoke test."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, load_from_disk
from sentence_transformers import SentenceTransformer

from config.cli import add_config_args, resolve
from config.loader import to_dict
from segmentation import Segmenter
from watermarking.detect import (
    compute_zscore,
    count_kmeans_watermark_hits,
    count_lsh_watermark_hits,
    null_distribution_threshold,
)
from watermarking.generate import effective_num_candidates, generate_dataset
from watermarking.primitives import SBERTLSHModel


_PROMPTS = [
    "The history of the printing press is a story of gradual refinement.",
    "Marine biologists have long studied the migration patterns of whales.",
    "In the early days of aviation, every flight was a genuine experiment.",
    "The recipe calls for fresh basil, ripe tomatoes, and good olive oil.",
    "Astronomers detected an unusual signal from a distant spiral galaxy.",
    "The committee gathered to discuss proposed changes to the city budget.",
    "Hikers who reached the summit were rewarded with a panoramic view.",
    "The novel opens with a long description of a quiet seaside village.",
    "Engineers tested the new bridge design under a simulated heavy load.",
    "Volunteers spent the entire weekend cleaning up the muddy riverbank.",
]

_WRONG_KEY = 87178291199
_DEFAULT_SMOKE_MODEL = "meta-llama/Llama-3.1-8B"


def parse_args():
    parser = argparse.ArgumentParser(description="Watermark generate/detect smoke test.")
    add_config_args(parser)
    parser.add_argument("--model", default=None, help="Convenience override for generation.model.")
    parser.add_argument("--embedder", default=None, help="Convenience override for watermark.embedder.")
    parser.add_argument("--backend", choices=["vllm", "hf"], default=None,
                        help="Convenience override for generation.backend.")
    parser.add_argument("--num-candidates", "--num_candidates", dest="num_candidates",
                        type=int, default=None,
                        help="Convenience override for generation.num_candidates.")
    parser.add_argument("--max-new-tokens", "--max_new_tokens", dest="max_new_tokens",
                        type=int, default=None,
                        help="Convenience override for generation.max_new_tokens.")
    parser.add_argument("--sp-dim", "--sp_dim", dest="sp_dim", type=int, default=None,
                        help="Convenience override for watermark.sp_dim.")
    parser.add_argument("--lmbd", type=float, default=None,
                        help="Convenience override for watermark.lmbd.")
    parser.add_argument("--n", type=int, default=10,
                        help="Number of prompt examples to generate.")
    parser.add_argument("--examples", type=int, default=3,
                        help="Number of generated examples to print.")
    parser.add_argument("--null-fpr", type=float, default=0.01,
                        help="Target false-positive rate for null-calibrated detection.")
    parser.add_argument("--null-n", type=int, default=1024,
                        help="Number of human-null texts to score when a null corpus is available.")
    parser.add_argument("--null-data-path", default=None,
                        help="Optional null corpus. Defaults to detection.human_text.")
    parser.add_argument("--null-cache", default=None,
                        help="Optional JSON cache for null detection scores.")
    parser.add_argument("--metrics-json", default=None,
                        help="Optional path to write machine-readable smoke metrics.")
    parser.add_argument("--assert-detected", action="store_true",
                        help="Exit nonzero unless the correct-key scores separate from the null.")
    parser.add_argument("--verbose-generation", action="store_true",
                        help="Print per-prompt generation logs from watermarking.generate.")
    return parser.parse_args()


def _override_was_set(overrides, key):
    return any(item.partition("=")[0].strip() == key for item in overrides or [])


def _replace_section(cfg, section_name, **updates):
    updates = {k: v for k, v in updates.items() if v is not None}
    if not updates:
        return cfg
    section = getattr(cfg, section_name)
    return dataclasses.replace(cfg, **{section_name: dataclasses.replace(section, **updates)})


def _resolve_config(args):
    cfg = resolve(args)

    # Use smoke defaults only without explicit config.
    no_config = not args.config
    gen_defaults = {}
    if no_config and not args.model and not _override_was_set(args.overrides, "generation.model"):
        gen_defaults["model"] = _DEFAULT_SMOKE_MODEL
    if no_config and args.max_new_tokens is None and not _override_was_set(args.overrides, "generation.max_new_tokens"):
        gen_defaults["max_new_tokens"] = 160
    cfg = _replace_section(cfg, "generation", **gen_defaults)

    cfg = _replace_section(
        cfg,
        "generation",
        model=args.model,
        backend=args.backend,
        num_candidates=args.num_candidates,
        max_new_tokens=args.max_new_tokens,
    )
    cfg = _replace_section(
        cfg,
        "watermark",
        embedder=args.embedder,
        sp_dim=args.sp_dim,
        lmbd=args.lmbd,
    )
    return cfg


def _load_generation_dataset(cfg, n):
    if cfg.io.data_path:
        dataset = load_from_disk(cfg.io.data_path)
        count = min(n, len(dataset))
        return Dataset.from_dict({"text": dataset["text"][:count]}), cfg.io.data_path

    prompts = _PROMPTS[: min(n, len(_PROMPTS))]
    return Dataset.from_dict({"text": prompts}), "built-in prompts"


def _mode(cfg):
    mode = cfg.detection.mode or cfg.watermark.mode
    if mode in {None, "none"}:
        raise ValueError(
            "watermark/detection mode must be one of lsh, lsh_fixed, "
            "lsh_fixed_diverse, kmeans, kmeans_fixed, kmeans_fixed_diverse."
        )
    return mode


def _secret_message_for(mode, cfg):
    # "_fixed" also matches "_fixed_diverse" (generation-only variant).
    return cfg.watermark.secret_message if "_fixed" in mode else None


def _build_detector(cfg, device):
    mode = _mode(cfg)
    wm = cfg.watermark
    seg_type = cfg.segmentation.type
    seg_backend = cfg.segmentation.backend
    secret_message = _secret_message_for(mode, cfg)

    if mode in {"lsh", "lsh_fixed", "lsh_fixed_diverse"}:
        lsh_model = SBERTLSHModel(
            lsh_model_path=wm.embedder,
            device=device,
            batch_size=16,
            lsh_dim=wm.sp_dim,
            sbert_type="base",
        )
        segmenter = (
            Segmenter.from_config(
                cfg.segmentation,
                encoder=lsh_model.embedder,
                encoder_id=wm.embedder,
                batch_size=cfg.runtime.semcut_batch_size,
            )
            if seg_type == "semspan" else Segmenter(seg_type, seg_backend)
        )

        def detect_stats(text, key):
            units = segmenter.segment(text)
            hits, total = count_lsh_watermark_hits(
                units,
                lsh_model,
                wm.lmbd,
                wm.sp_dim,
                secret_message=secret_message,
                key=key,
            )
            return {
                "z": float(compute_zscore(hits, total, wm.lmbd)),
                "hits": int(hits),
                "total": int(total),
                "units": len(units),
            }

        return detect_stats, mode, segmenter

    if mode in {"kmeans", "kmeans_fixed", "kmeans_fixed_diverse"}:
        if not wm.cc_path:
            raise ValueError("watermark.cc_path is required for kmeans detection.")
        cluster_centers = torch.load(wm.cc_path)
        embedder = SentenceTransformer(wm.embedder, device=device)
        segmenter = (
            Segmenter.from_config(
                cfg.segmentation,
                encoder=embedder,
                encoder_id=wm.embedder,
                batch_size=cfg.runtime.semcut_batch_size,
            )
            if seg_type == "semspan" else Segmenter(seg_type, seg_backend)
        )

        def detect_stats(text, key):
            units = segmenter.segment(text)
            hits, total = count_kmeans_watermark_hits(
                units,
                embedder,
                wm.lmbd,
                wm.sp_dim,
                cluster_centers,
                secret_message=secret_message,
                key=key,
            )
            return {
                "z": float(compute_zscore(hits, total, wm.lmbd)),
                "hits": int(hits),
                "total": int(total),
                "units": len(units),
            }

        return detect_stats, mode, segmenter

    raise ValueError(f"Unknown detection mode: {mode!r}")


def _detect_many(texts, detect_stats, key):
    return [detect_stats(text, key) for text in texts]


def _null_cache_metadata(cfg, args, source_path, mode_desc, segmenter):
    wm = cfg.watermark
    metadata = {
        "source_path": source_path,
        "null_n": args.null_n,
        "mode": mode_desc,
        "segmentation_type": cfg.segmentation.type,
        "segmentation_backend": cfg.segmentation.backend,
        "embedder": wm.embedder,
        "cc_path": wm.cc_path,
        "sp_dim": wm.sp_dim,
        "lmbd": wm.lmbd,
        "hash_key": wm.hash_key,
        "secret_message": wm.secret_message if "_fixed" in mode_desc else None,
    }
    if segmenter.type == "semspan":
        segmentation = segmenter.metadata()
        metadata.update({
            key: segmentation[key]
            for key in ("semcut_max_words", "semcut_window")
        })
    return metadata


def _read_null_cache(path, metadata):
    if not path or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("metadata") != metadata:
        return None
    return payload.get("stats")


def _write_null_cache(path, metadata, stats):
    if not path:
        return
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump({"metadata": metadata, "stats": stats}, f, indent=2)
    os.replace(tmp_path, cache_path)


def _load_or_compute_null_stats(cfg, args, detect_stats, mode_desc, segmenter):
    source_path = args.null_data_path or cfg.detection.human_text
    if not source_path or not os.path.isdir(source_path):
        return None, None

    metadata = _null_cache_metadata(
        cfg, args, source_path, mode_desc, segmenter,
    )
    cached = _read_null_cache(args.null_cache, metadata)
    if cached is not None:
        return cached, f"human null cache: {args.null_cache}"

    dataset = load_from_disk(source_path)
    texts = dataset["text"][: min(args.null_n, len(dataset))]
    stats = _detect_many(texts, detect_stats, cfg.watermark.hash_key)
    _write_null_cache(args.null_cache, metadata, stats)
    return stats, f"human null: {source_path}"


def _clean_scores(stats):
    return np.nan_to_num(np.array([row["z"] for row in stats], dtype=float))


def _rate(num, denom):
    return (num / denom) if denom else 0.0


def _snippet(text, max_chars=360):
    one_line = " ".join(text.split())
    if len(one_line) <= max_chars:
        return one_line
    return one_line[: max_chars - 3].rstrip() + "..."


def _summed_hits(stats):
    hits = sum(row["hits"] for row in stats)
    total = sum(row["total"] for row in stats)
    return hits, total


def main():
    args = parse_args()
    cfg = _resolve_config(args)
    if torch.cuda.device_count() == 0:
        raise RuntimeError("This smoke test requires a GPU.")
    device = "cuda:0"

    dataset, prompt_source = _load_generation_dataset(cfg, args.n)
    print(f"== Generating {len(dataset)} watermarked continuations ==")
    generated, gen_stats = generate_dataset(
        dataset,
        cfg,
        device,
        verbose=args.verbose_generation,
    )
    accepted = gen_stats["accepted_units"]
    total = gen_stats["total_units"]
    gens = list(generated["text"])
    gen_accepts = (
        list(generated["_semstamp_units_accepted"])
        if "_semstamp_units_accepted" in generated.column_names
        else [0] * len(gens)
    )
    gen_totals = (
        list(generated["_semstamp_units_total"])
        if "_semstamp_units_total" in generated.column_names
        else [0] * len(gens)
    )
    gen_unit_displays = (
        list(generated["_semstamp_unit_displays"])
        if "_semstamp_unit_displays" in generated.column_names
        else [[] for _ in gens]
    )
    gen_unit_normalized = (
        list(generated["_semstamp_unit_normalized"])
        if "_semstamp_unit_normalized" in generated.column_names
        else [[] for _ in gens]
    )
    print("\n== Detecting watermark and calibrating from null ==")
    detect_stats, mode_desc, segmenter = _build_detector(cfg, device)
    wm_stats = _detect_many(gens, detect_stats, cfg.watermark.hash_key)
    wrong_key_stats = _detect_many(gens, detect_stats, _WRONG_KEY)
    null_stats, null_source = _load_or_compute_null_stats(
        cfg, args, detect_stats, mode_desc, segmenter,
    )
    if null_stats is None:
        null_stats = wrong_key_stats
        null_source = "wrong-key scores on generated texts"

    wm_z = _clean_scores(wm_stats)
    wrong_key_z = _clean_scores(wrong_key_stats)
    null_z = _clean_scores(null_stats)
    cutoff = null_distribution_threshold(null_z, fpr=args.null_fpr)
    detected = wm_z > cutoff
    null_over_cutoff = null_z > cutoff
    wrong_key_over_cutoff = wrong_key_z > cutoff

    wm_hits, wm_total = _summed_hits(wm_stats)
    null_hits, null_total = _summed_hits(null_stats)
    wrong_hits, wrong_total = _summed_hits(wrong_key_stats)
    detected_count = int(np.count_nonzero(detected))
    tpr_at_null_fpr = float(detected_count / len(wm_z)) if len(wm_z) else 0.0
    empirical_null_fpr = float(np.count_nonzero(null_over_cutoff) / len(null_z)) if len(null_z) else 0.0
    wrong_key_fpr = float(np.count_nonzero(wrong_key_over_cutoff) / len(wrong_key_z)) if len(wrong_key_z) else 0.0
    acceptance_rate = _rate(accepted, total)

    print("\n" + "=" * 72)
    print("WATERMARK SMOKE TEST")
    print("=" * 72)
    print(f"Prompt source:                  {prompt_source}")
    print(f"Config files:                   {', '.join(args.config) if args.config else 'schema defaults'}")
    print(f"Detection mode:                 {mode_desc}")
    print(f"Generation backend/model:       {cfg.generation.backend} / {cfg.generation.model}")
    print(f"num_candidates:                 {effective_num_candidates(cfg.generation)}")
    print(f"Examples generated:             {len(gens)}")
    print(f"Generated units accepted:       {accepted}/{total} ({acceptance_rate * 100:.1f}%)")
    print(f"Detector hits, correct key:     {wm_hits}/{wm_total} ({_rate(wm_hits, wm_total) * 100:.1f}%)")
    print(f"Detector hits, wrong key:       {wrong_hits}/{wrong_total} ({_rate(wrong_hits, wrong_total) * 100:.1f}%)")
    print(f"Null source:                    {null_source} (n={len(null_z)})")
    print(f"Null detector hits:             {null_hits}/{null_total} ({_rate(null_hits, null_total) * 100:.1f}%)")
    print(f"Null-calibrated z cutoff:       {cutoff:.3f} at {args.null_fpr * 100:.2f}% FPR")
    print(f"Mean z, correct key:            {float(np.mean(wm_z)):.3f}")
    print(f"Mean z, wrong key:              {float(np.mean(wrong_key_z)):.3f}")
    print(f"Mean z, null:                   {float(np.mean(null_z)):.3f}")
    print(f"Detected examples:              {detected_count}/{len(wm_z)} ({tpr_at_null_fpr * 100:.1f}%)")
    print(f"Empirical null FPR at cutoff:   {empirical_null_fpr * 100:.2f}%")
    print(f"Wrong-key rate at cutoff:       {wrong_key_fpr * 100:.2f}%")

    if args.examples > 0:
        print("\nExamples:")
        for idx in range(min(args.examples, len(gens))):
            gen_acc = gen_accepts[idx]
            gen_total = gen_totals[idx]
            row = wm_stats[idx]
            wrong = wrong_key_stats[idx]
            status = "detected" if detected[idx] else "not detected"
            print(
                f"{idx + 1:02d}. gen_accept={gen_acc}/{gen_total} "
                f"detect_hits={row['hits']}/{row['total']} "
                f"posthoc_units={row['units']} "
                f"z={row['z']:.3f} wrong_key_z={wrong['z']:.3f} {status}"
            )
            print(f"    {_snippet(gens[idx])}")
    print("=" * 72)

    metrics = {
        "prompt_source": prompt_source,
        "config_files": args.config,
        "mode": mode_desc,
        "num_candidates": effective_num_candidates(cfg.generation),
        "num_examples": len(gens),
        "generated_units_accepted": int(accepted),
        "generated_units_total": int(total),
        "generation_acceptance_rate": float(acceptance_rate),
        "detector_hits": int(wm_hits),
        "detector_total": int(wm_total),
        "detector_hit_rate": float(_rate(wm_hits, wm_total)),
        "wrong_key_hits": int(wrong_hits),
        "wrong_key_total": int(wrong_total),
        "wrong_key_hit_rate": float(_rate(wrong_hits, wrong_total)),
        "null_source": null_source,
        "null_n": len(null_z),
        "null_hits": int(null_hits),
        "null_total": int(null_total),
        "null_hit_rate": float(_rate(null_hits, null_total)),
        "null_fpr": float(args.null_fpr),
        "null_calibrated_cutoff": float(cutoff),
        "wm_mean_z": float(np.mean(wm_z)),
        "wrong_key_mean_z": float(np.mean(wrong_key_z)),
        "null_mean_z": float(np.mean(null_z)),
        "detected_count": detected_count,
        "tpr_at_null_fpr": tpr_at_null_fpr,
        "tpr_at_1pct_fpr": tpr_at_null_fpr if abs(args.null_fpr - 0.01) < 1e-12 else None,
        "empirical_null_fpr": empirical_null_fpr,
        "wrong_key_rate_at_cutoff": wrong_key_fpr,
        "per_example": [
            {
                "index": idx,
                "generated_units_accepted": int(gen_accepts[idx]),
                "generated_units_total": int(gen_totals[idx]),
                "detector_hits": int(wm_stats[idx]["hits"]),
                "detector_total": int(wm_stats[idx]["total"]),
                "detector_units": int(wm_stats[idx]["units"]),
                "z": float(wm_stats[idx]["z"]),
                "wrong_key_z": float(wrong_key_stats[idx]["z"]),
                "detected": bool(detected[idx]),
                "generated_unit_displays": list(gen_unit_displays[idx]),
                "generated_unit_normalized": list(gen_unit_normalized[idx]),
                "text": gens[idx],
            }
            for idx in range(len(gens))
        ],
        "config": to_dict(cfg),
    }

    if args.metrics_json:
        metrics_path = Path(args.metrics_json)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Wrote metrics: {metrics_path}")

    if args.assert_detected:
        if float(np.mean(wm_z)) <= float(np.mean(null_z)):
            raise AssertionError(
                f"watermark not separable from null mean: {float(np.mean(wm_z)):.3f} <= {float(np.mean(null_z)):.3f}"
            )
        if detected_count == 0:
            raise AssertionError(
                f"no examples cleared the null-calibrated cutoff {cutoff:.3f}"
            )
        print("PASS: correct-key scores separate from the null distribution.")


if __name__ == "__main__":
    main()
