"""Deterministic PGFPlots and LaTeX extractors for the paper bundle."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from . import common


EXTRACTORS = (
    "teaser",
    "fidelity_robustness",
    "cross_scheme",
    "whitebox",
    "channel_response",
    "attack_channels",
    "isolation",
    "costs",
    "quality_criteria",
    "transfer",
)


def extract_all(bundle: str | Path, output: str | Path) -> list[Path]:
    """Run every authoritative paper extractor and return generated files."""
    common.configure(bundle=bundle, output=output)
    for name in EXTRACTORS:
        qualified = f"{__name__}.{name}"
        module = importlib.import_module(qualified)
        if qualified in sys.modules:
            module = importlib.reload(module)
        module.main()
    return sorted(
        path for path in (common.output_root() / "tables").rglob("*")
        if path.is_file()
    )


__all__ = ["EXTRACTORS", "extract_all"]
