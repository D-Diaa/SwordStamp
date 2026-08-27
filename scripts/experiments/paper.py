#!/usr/bin/env python3
"""Print the canonical paper experiment registry for shell orchestration."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config.paper import (  # noqa: E402,F401 - deliberately re-exported
    COMPARISONS,
    CENTER_DATASET,
    CENTER_DOCUMENTS,
    DATASETS,
    HUMAN_NULL_DATASET,
    HUMAN_NULL_DOCUMENTS,
    LADDERS,
    MAIN_SEMSPAN,
    NONE_PRESET,
    ORACLE_KS,
    PAPER_RUNGS,
    PMARK,
    PAPER_CORPORA,
    PROMPT_DATASETS,
    PROVIDER_CANDIDATES,
    SAMARK,
    SENTENCE,
    Comparison,
    PaperRung,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "query",
        choices=(
            "rungs", "datasets", "oracle-ks", "comparisons", "candidates", "baseline",
        ),
    )
    args = parser.parse_args(argv)

    if args.query == "rungs":
        for rung in PAPER_RUNGS:
            print("\t".join((
                rung.key,
                rung.family,
                rung.mask,
                rung.sampling,
                rung.segmentation,
                rung.preset,
                str(rung.num_candidates),
            )))
    elif args.query == "datasets":
        print("\n".join(DATASETS))
    elif args.query == "oracle-ks":
        print(" ".join(map(str, ORACLE_KS)))
    elif args.query == "comparisons":
        for comparison in COMPARISONS:
            print("\t".join((
                comparison.key,
                comparison.family,
                comparison.mask,
                comparison.sampling,
                comparison.segmentation,
                str(comparison.msig),
                str(comparison.num_candidates),
            )))
    elif args.query == "candidates":
        print(PROVIDER_CANDIDATES)
    else:
        print(NONE_PRESET)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
