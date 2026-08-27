#!/usr/bin/env python3
"""Evaluate anchor metrics on the authored structural-edit fixtures."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import yaml

from .anchors import CHANNELS, evaluate_anchor_structure

FIELDS = ("original", "reorder", "reword", "merge", "split")
DEFAULT_SAMPLES_PATH = Path(__file__).with_name("anchor_samples.yaml")


def load_samples(path: Path = DEFAULT_SAMPLES_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        samples = yaml.safe_load(f) or []
    if not isinstance(samples, list):
        raise ValueError(f"{path} must contain a YAML list")

    required = {"id", *FIELDS}
    bad = []
    for i, sample in enumerate(samples):
        if not isinstance(sample, dict):
            bad.append((i, "<not a mapping>", sorted(required)))
            continue
        missing = sorted(required - set(sample))
        extra = sorted(set(sample) - required)
        if missing or extra:
            bad.append((i, sample.get("id", "<missing id>"), missing, extra))
    if bad:
        preview = "; ".join(str(x) for x in bad[:5])
        raise ValueError(f"{path} does not match the fixture schema: {preview}")
    return samples


def _summary(values) -> dict[str, float | int]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"n": 0, "mean": np.nan, "sd": np.nan, "p10": np.nan, "p50": np.nan, "p90": np.nan}
    p10, p50, p90 = np.percentile(vals, [10, 50, 90])
    return {
        "n": int(vals.size),
        "mean": float(vals.mean()),
        "sd": float(vals.std()),
        "p10": float(p10),
        "p50": float(p50),
        "p90": float(p90),
    }


def _fmt(summary: dict[str, float | int]) -> str:
    return (
        f"n={summary['n']:>3} "
        f"mean={summary['mean']:.3f} sd={summary['sd']:.3f} "
        f"p10={summary['p10']:.3f} p50={summary['p50']:.3f} p90={summary['p90']:.3f}"
    )


def evaluate_samples(samples: list[dict], fields: tuple[str, ...]) -> list[dict]:
    refs = [sample["original"] for sample in samples]
    rows = []
    for field in fields:
        gens = refs if field == "original" else [sample[field] for sample in samples]
        result = evaluate_anchor_structure(gens, refs)
        for channel in CHANNELS:
            summary = _summary(result[f"anchor_{channel}_per_sample"])
            rows.append({"field": field, "channel": channel, **summary})
    return rows


def print_table(rows: list[dict], n_samples: int, samples_path: Path) -> None:
    print(f"=== ANCHOR SAMPLE DISTRIBUTIONS (n={n_samples}, file={samples_path}) ===")
    current = None
    for row in rows:
        if row["field"] != current:
            current = row["field"]
            print(f"\n[{current}]")
        print(f"  {row['channel']:10s} {_fmt(row)}")


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=("field", "channel", "n", "mean", "sd", "p10", "p50", "p90"))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples",
        type=Path,
        default=DEFAULT_SAMPLES_PATH,
        help="YAML fixture file to evaluate",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        choices=FIELDS,
        default=list(FIELDS),
        help="fixture fields to compare against original",
    )
    parser.add_argument("--csv", type=Path, default=None, help="optional CSV output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples_path = args.samples
    samples = load_samples(samples_path)
    rows = evaluate_samples(samples, tuple(args.fields))
    print_table(rows, len(samples), samples_path)
    if args.csv:
        write_csv(rows, args.csv)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
