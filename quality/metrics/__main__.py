"""Smoke-test the LLM judge rubrics on authored samples."""

import argparse
from pathlib import Path

import numpy as np
import yaml

from config.schema import QualityConfig

from .llm_judge import (
    PAIRWISE_RUBRIC,
    QUALITY_RUBRIC,
    _is_openai_model,
    build_llm_judge_pipe,
    evaluate_llm_judge,
)

_SAMPLES_PATH = Path(__file__).with_name("judge_samples.yaml")
DEFAULT_MODEL = "Qwen/Qwen3-14B"

# Keep run order aligned with the sample-file keys.
RUBRICS = {r.name: r for r in (QUALITY_RUBRIC, PAIRWISE_RUBRIC)}


def _load_samples(path):
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def _short_headers(dim_keys):
    """Compact, unique column headers for the per-criterion score table."""
    headers, seen = [], set()
    for k in dim_keys:
        h = k[:4]
        i = 0
        while h in seen:  # disambiguate any collision deterministically
            i += 1
            h = f"{k[:3]}{i}"
        seen.add(h)
        headers.append(h)
    return headers


def _print_rubric_report(rubric, samples, stats):
    dim_keys = rubric.dim_keys
    headers = _short_headers(dim_keys)
    # per-criterion arrays are stored normalized 0-1 (raw/5); recover the 0-5 score
    per_dim = {k: stats[f"{rubric.name}_{k}_per_sample"] * 5.0 for k in dim_keys}
    agg = stats[f"{rubric.name}_per_sample"]

    id_w = max(4, *(len(s.get("id", "")) for s in samples))
    head = f"  {'#':>2}  {'id':<{id_w}}  " + "  ".join(f"{h:>4}" for h in headers) + "   agg   note"
    print(head)
    print("  " + "-" * (len(head) - 2))
    for i, s in enumerate(samples):
        cells = []
        for k in dim_keys:
            v = per_dim[k][i]
            cells.append(" nan" if np.isnan(v) else f"{int(round(v)):>4}")
        a = agg[i]
        a_str = " nan" if np.isnan(a) else f"{a:.2f}"
        print(
            f"  {i + 1:>2}  {s.get('id', ''):<{id_w}}  "
            + "  ".join(cells)
            + f"  {a_str:>5}   {s.get('note', '')}"
        )
    print(
        f"\n  mean {stats[rubric.name]:.4f}  |  median {stats[f'{rubric.name}_median']:.4f}"
        f"  |  CI(95%) +/-{stats[f'{rubric.name}_ci']:.4f}"
    )
    print(f"  legend: {', '.join(f'{h}={k}' for h, k in zip(headers, dim_keys))}")


def _print_raw(rubric, samples, stats):
    for i, (s, raw_group) in enumerate(zip(samples, stats["raw_outputs"])):
        print(f"\n  --- {rubric.name} #{i + 1} [{s.get('id', '')}] ---")
        for repeat_idx, raw in enumerate(raw_group, start=1):
            print(f"\n  [repetition {repeat_idx}]\n  {raw}")


def run(rubric, samples, qcfg, pipe, show_raw):
    print(f"\n{'=' * 70}")
    print(f"=== {rubric.name}  |  {len(samples)} samples ===")
    print("=" * 70)
    stats = evaluate_llm_judge(samples, qcfg, rubric=rubric, debug=True, pipe=pipe)
    _print_rubric_report(rubric, samples, stats)
    if show_raw:
        _print_raw(rubric, samples, stats)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="judge model (HF id or OpenAI model)")
    parser.add_argument("--samples", type=Path, default=_SAMPLES_PATH, help="path to samples YAML")
    parser.add_argument(
        "--rubric",
        choices=sorted(RUBRICS),
        action="append",
        help="run only this rubric (repeatable); default runs all three",
    )
    parser.add_argument("--show-raw", action="store_true", help="also dump each judge's raw output")
    args = parser.parse_args()

    all_samples = _load_samples(args.samples)
    names = args.rubric or list(RUBRICS)
    qcfg = QualityConfig(judge_model=args.model)

    print(f"Judge model : {args.model}")
    print(f"Samples file: {args.samples}")
    print(f"Rubrics     : {', '.join(names)}")

    # One vLLM engine reused across every rubric (OpenAI models need no engine).
    pipe = None if _is_openai_model(args.model) else build_llm_judge_pipe(args.model)

    summary = {}
    stats_all = {}
    for name in names:
        samples = all_samples.get(name)
        if not samples:
            print(f"\n[skip] no samples for rubric '{name}' in {args.samples}")
            continue
        stats = run(RUBRICS[name], samples, qcfg, pipe, args.show_raw)
        summary[name] = stats[name]
        stats_all[name] = stats

    print(f"\n{'=' * 70}")
    print("SUMMARY (mean aggregate score per rubric)")
    for name, mean in summary.items():
        print(f"  {name:<14} {mean:.4f}")
    print("=" * 70)

    import json
    import os

    # Format the model name to be a safe filename
    safe_model_name = args.model.replace("/", "_").replace("-", "_")
    output_path = Path(f"{safe_model_name}_judge_results.json")

    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(convert_to_serializable(stats_all), f, indent=2)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
