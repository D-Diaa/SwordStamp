"""Shared arguments for config-driven commands."""

from __future__ import annotations

import argparse
import dataclasses

from config.loader import load_config
from config.schema import AppConfig


def add_config_args(parser: argparse.ArgumentParser, *, positional_data: bool = True) -> None:
    if positional_data:
        parser.add_argument(
            "data_path",
            nargs="?",
            default=None,
            help="Dataset path; convenience override for io.data_path.",
        )
    parser.add_argument(
        "-c",
        "--config",
        action="append",
        default=[],
        metavar="FILE",
        help="YAML config file(s); later files override earlier ones (repeatable).",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Dotted config override, e.g. watermark.margin=0.1 (repeatable).",
    )


def resolve(args: argparse.Namespace) -> AppConfig:
    """Resolve parsed arguments into an application config."""
    cfg = load_config(getattr(args, "config", None), getattr(args, "overrides", None))
    data_path = getattr(args, "data_path", None)
    if data_path:
        cfg = dataclasses.replace(cfg, io=dataclasses.replace(cfg.io, data_path=data_path))
    return cfg


def _main():
    """Run a CLI smoke test."""
    parser = argparse.ArgumentParser(description="SemStamp CLI smoke test")
    add_config_args(parser)

    args = parser.parse_args([
        "data/c4-val-def",
        "--set", "watermark.sp_dim=16",
        "--set", "generation.backend=hf",
    ])
    cfg = resolve(args)
    assert cfg.watermark.sp_dim == 16
    assert cfg.generation.backend == "hf"
    assert cfg.io.data_path == "data/c4-val-def"
    print(f"with data_path: sp_dim={cfg.watermark.sp_dim}, backend={cfg.generation.backend!r}, "
          f"data_path={cfg.io.data_path!r}")

    args2 = parser.parse_args(["--set", "watermark.lmbd=0.3"])
    cfg2 = resolve(args2)
    assert cfg2.io.data_path is None
    assert cfg2.watermark.lmbd == 0.3
    print(f"no data_path: lmbd={cfg2.watermark.lmbd}, data_path={cfg2.io.data_path!r}")

    print("cli smoke ok")


if __name__ == "__main__":
    _main()
