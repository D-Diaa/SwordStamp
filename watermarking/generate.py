"""Generation adapted from original SemStamp's ``sampling.py``."""

import argparse
import json
import os
import pprint

import torch
from sentence_transformers import SentenceTransformer
from transformers import GenerationConfig
import datasets
from datasets import load_from_disk, Dataset

from config.cli import add_config_args, resolve
from config.loader import dump_config, to_dict
from config.paths import watermark_dir
from config.runtime import vllm_gpu_memory_utilization
from sampling.base_sampler import create_sampler
from segmentation import Segmenter
from watermarking.primitives import extract_prompt_from_text
from watermarking.scoring import (
    create_none_score_fn,
    setup_score_factory,
)

datasets.disable_caching()


def effective_num_candidates(gen):
    """Return the explicit candidate budget recorded in the run config."""
    return gen.num_candidates


def generate_dataset(dataset, cfg, device, verbose=True):
    """Generate watermarked continuations and aggregate statistics."""
    wm = cfg.watermark
    gen = cfg.generation
    sampling_method = gen.sampling_method
    num_candidates = effective_num_candidates(gen)
    seg_type = cfg.segmentation.type
    seg_backend = cfg.segmentation.backend
    unit_label = "sentences" if seg_type == "sentence" else "EDUs"

    backend = gen.backend
    # Bind semantic cuts to the same encoder used by the defender's watermark.
    score_factory = None
    if wm.mode == "none":
        encoder = (
            SentenceTransformer(wm.embedder, device=device)
            if seg_type == "semspan" else None
        )
    else:
        score_factory = setup_score_factory(wm, device, num_candidates)
        encoder = score_factory.encoder
    if seg_type == "semspan":
        segmenter = Segmenter.from_config(
            cfg.segmentation,
            encoder=encoder,
            encoder_id=wm.embedder,
            batch_size=cfg.runtime.semcut_batch_size,
        )
    else:
        segmenter = Segmenter(seg_type, seg_backend)

    # Reserve GPU memory for the embedder.
    backend_kwargs = (
        {"device": device}
        if backend == "hf"
        else {"gpu_memory_utilization": vllm_gpu_memory_utilization(cfg.runtime.vllm_utilization)}
    )
    sampler = create_sampler(
        backend,
        gen.model,
        num_candidates=num_candidates,
        chunk_tokens=gen.chunk_tokens,
        adapter_path=gen.adapter_path,
        adapter_strength=gen.adapter_strength,
        segmentation_type=seg_type,
        segmentation_backend=seg_backend,
        segmenter=segmenter,
        **backend_kwargs,
    )
    gen_config = GenerationConfig(
        max_new_tokens=gen.max_new_tokens,
        do_sample=gen.do_sample,
        temperature=gen.temperature,
        top_k=gen.top_k,
        top_p=gen.top_p,
        repetition_penalty=gen.rep_p,
        pad_token_id=sampler.tokenizer.pad_token_id,
    )

    # One score function drives both selection modes.
    if score_factory is None:
        step_fn = create_none_score_fn()
    else:
        step_fn = score_factory()

    # Keep prompts identical across segmentation arms.
    prompts = [
        extract_prompt_from_text(text, gen.len_prompt)
        for text in dataset["text"]
    ]

    if verbose:
        max_active = sampler._resolve_max_active(gen.max_active_docs, len(prompts))
        print(
            f"[generate] {len(prompts)} docs, backend={backend}, "
            f"selection={sampling_method}, unit={unit_label}, "
            f"num_candidates={num_candidates}, max_active={max_active}, "
            f"max_new_tokens={gen.max_new_tokens}",
            flush=True,
        )
    # Sampling method maps directly to selection mode.
    results = sampler.generate_batched_continuation(
        prompts,
        gen_config,
        step_fn,
        selection_mode=sampling_method,
        margin=wm.margin if wm.mode != "none" else 0.0,
        max_tokens=gen.max_new_tokens,
        max_active=gen.max_active_docs,
        progress=verbose,
    )

    # Retain per-unit pool counts for E[tries].
    per_unit_n_green = []
    per_unit_n_candidates = []
    if sampling_method == "rejection":
        for _, info in results:
            per_unit_n_green.extend(info["n_accepted_candidates_per_unit"])
            per_unit_n_candidates.extend(info["n_candidates_per_unit"])

    if verbose:
        label = "scored green" if sampling_method == "best-of-n" else "accepted"
        for i, (prompt, (text, info)) in enumerate(zip(prompts, results)):
            print(
                f"\n[doc {i}] {info['accepted_count']}/{info['unit_count']} {unit_label} {label}\n"
                f"prompt: {prompt}\n"
                f"response: {text}",
                flush=True,
            )

    processed_dataset = Dataset.from_dict({
        "text": [text.strip() for text, _ in results],
        "_semstamp_units_accepted": [int(info["accepted_count"]) for _, info in results],
        "_semstamp_units_total": [int(info["unit_count"]) for _, info in results],
        "_semstamp_unit_displays": [
            [unit.display for unit in info["units"]] for _, info in results
        ],
        "_semstamp_unit_normalized": [
            [unit.normalized for unit in info["units"]] for _, info in results
        ],
    })
    stats = {
        "accepted_units": int(sum(info["accepted_count"] for _, info in results)),
        "total_units": int(sum(info["unit_count"] for _, info in results)),
        "per_unit_n_green": per_unit_n_green,
        "per_unit_n_candidates": per_unit_n_candidates,
    }
    return processed_dataset, stats


def generate(cfg):
    data_path = cfg.io.data_path
    if not data_path:
        raise ValueError("No data_path set. Pass it positionally or via io.data_path.")
    if not cfg.watermark.mode:
        raise ValueError("watermark.mode is required (lsh, kmeans, lsh_fixed, kmeans_fixed, none).")

    dataset = load_from_disk(data_path)
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs detected. This script requires at least one GPU.")

    device = "cuda:0"
    print(f"Detected {num_gpus} GPU(s). Running generation on {device}.")

    generated_dataset, gen_stats = generate_dataset(dataset, cfg, device)
    total_accepted = gen_stats["accepted_units"]
    total_units = gen_stats["total_units"]
    per_unit_n_green = gen_stats["per_unit_n_green"]
    per_unit_n_candidates = gen_stats["per_unit_n_candidates"]

    merged_dataset = Dataset.from_dict({'text': generated_dataset['text']})
    # Prefer an explicit output override.
    output_path = cfg.io.output_dir or watermark_dir(cfg)
    os.makedirs(output_path, exist_ok=True)
    merged_dataset.save_to_disk(output_path)
    # Save run provenance.
    dump_config(cfg, os.path.join(output_path, "resolved_config.yaml"))

    acceptance_rate = (total_accepted / total_units * 100) if total_units > 0 else 0
    unit_label = "sentences" if cfg.segmentation.type == "sentence" else "EDUs"

    # Estimate trials from per-unit green rates.
    tries_stats: dict = {}
    if per_unit_n_green:
        import numpy as np
        n_green_arr = np.array(per_unit_n_green, dtype=float)
        n_cand_arr = np.array(per_unit_n_candidates, dtype=float)
        p_per_unit = np.where(n_cand_arr > 0, n_green_arr / n_cand_arr, 0.0)
        n = len(p_per_unit)
        mean_p = float(np.mean(p_per_unit))
        std_p = float(np.std(p_per_unit, ddof=1)) if n > 1 else 0.0
        sem_p = std_p / (n ** 0.5) if n > 0 else 0.0
        # Invert the green-rate confidence interval.
        import scipy.stats as _stats
        t_crit = _stats.t.ppf(0.975, df=max(n - 1, 1))
        p_lo = max(1e-9, mean_p - t_crit * sem_p)
        p_hi = min(1.0, mean_p + t_crit * sem_p)
        mean_tries = 1.0 / mean_p if mean_p > 0 else None
        tries_stats = {
            "n_units_tracked": n,
            "mean_p_green_per_unit": mean_p,
            "std_p_green_per_unit": std_p,
            "mean_tries": mean_tries,
            "tries_ci_lo_95": 1.0 / p_hi if p_hi > 0 else None,
            "tries_ci_hi_95": 1.0 / p_lo if p_lo > 0 else None,
        }

    stats = {
        "num_examples": len(generated_dataset),
        "num_candidates": effective_num_candidates(cfg.generation),
        "unit_label": unit_label,
        "total_units": int(total_units),
        "accepted_units": int(total_accepted),
        "acceptance_rate": total_accepted / total_units if total_units else 0.0,
        **tries_stats,
    }
    with open(os.path.join(output_path, "generation_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 60)
    print("GENERATION STATISTICS")
    print("=" * 60)
    print(f"Total {unit_label} generated: {total_units}")
    print(f"{unit_label} accepted by watermark criterion: {total_accepted}")
    print(f"Acceptance rate: {acceptance_rate:.2f}%")
    if tries_stats:
        mt = tries_stats.get("mean_tries")
        lo = tries_stats.get("tries_ci_lo_95")
        hi = tries_stats.get("tries_ci_hi_95")
        mt_str = f"{mt:.2f}" if mt else "n/a"
        ci_str = f"[{lo:.2f}, {hi:.2f}]" if lo and hi else "n/a"
        print(f"E[tries until green]: {mt_str}  95% CI: {ci_str}")
    print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    cfg = resolve(parser.parse_args())
    pprint.PrettyPrinter(indent=4).pprint(to_dict(cfg))
    return cfg


if __name__ == '__main__':
    generate(parse_args())
