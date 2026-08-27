#!/usr/bin/env python3
"""Report config-derived paths and completed stages for one experiment cell."""
import glob
import os
import sys

# Import from the repo root regardless of CWD.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import paths  # noqa: E402
from config.loader import load_config  # noqa: E402


def _glob(pattern: str) -> str:
    return "1" if glob.glob(pattern) else "0"


def _file(path: str) -> str:
    return "1" if os.path.isfile(path) else "0"


def _set_overrides(tokens: list[str]) -> list[str]:
    """Pull the values following each `--set` out of a token list."""
    out, i = [], 0
    while i < len(tokens):
        if tokens[i] == "--set" and i + 1 < len(tokens):
            out.append(tokens[i + 1])
            i += 2
        else:
            i += 1
    return out


def main() -> int:
    base, cfg_file = sys.argv[1], sys.argv[2]
    base_over = [f"io.data_path={base}"]
    common_over = _set_overrides(sys.argv[3:])

    cell = load_config([cfg_file], base_over + common_over)
    wm = paths.watermark_dir(cell)
    print("\t".join(["CELL", wm, _glob(f"{wm}/data*.arrow"),
                      _file(f"{wm}/results_wm.csv"), _file(f"{wm}/eval_quality.csv")]))

    for line in sys.stdin:
        if not line.strip():
            continue
        label, _, tokstr = line.rstrip("\n").partition("\t")
        attack_over = _set_overrides(tokstr.split())
        ad = paths.attack_dir(load_config(
            [cfg_file], base_over + attack_over + common_over,
        ))
        print("\t".join(["ATTACK", label, ad, _glob(f"{ad}/data*.arrow"),
                          _file(f"{ad}/results.csv"), _file(f"{ad}/eval_quality.csv")]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
