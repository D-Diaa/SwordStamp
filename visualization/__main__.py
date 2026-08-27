"""Command-line interface for compiling, extracting, and rendering results."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT = Path(__file__).resolve().parents[1] / "results" / "paper"


def _paths(parser: argparse.ArgumentParser, *, output: bool = True) -> None:
    parser.add_argument(
        "--bundle", type=Path, default=DEFAULT,
        help="Compiled bundle or extracted-results root (default: results/paper).",
    )
    if output:
        parser.add_argument(
            "--output", type=Path, default=DEFAULT,
            help="Destination root (default: results/paper).",
        )


def _compile_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--unit-encoder-device", default="cpu")
    parser.add_argument("--unit-encoder-batch-size", type=int, default=128)
    parser.add_argument("--force-unit-encoder-transfer", action="store_true")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="python -m visualization")
    commands = root.add_subparsers(dest="command", required=True)

    compile_parser = commands.add_parser("compile", help="Compile experiment outputs.")
    _paths(compile_parser, output=False)
    _compile_arguments(compile_parser)

    extract_parser = commands.add_parser(
        "extract", help="Emit PGFPlots data and LaTeX fragments."
    )
    _paths(extract_parser)

    render_parser = commands.add_parser("render", help="Render PNG/PDF figures.")
    _paths(render_parser)

    all_parser = commands.add_parser("all", help="Compile, extract, and render.")
    _paths(all_parser)
    _compile_arguments(all_parser)
    return root


def _compile(args) -> Path:
    from .compile_results import compile_results

    argv = ["--out", str(args.bundle)]
    argv.extend([
        "--unit-encoder-device", args.unit_encoder_device,
        "--unit-encoder-batch-size", str(args.unit_encoder_batch_size),
    ])
    if args.force_unit_encoder_transfer:
        argv.append("--force-unit-encoder-transfer")
    return compile_results(argv)


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    written: list[Path] = []
    if args.command == "compile":
        _compile(args)
        return 0
    if args.command == "extract":
        from .extractors import extract_all

        written = extract_all(args.bundle, args.output)
    elif args.command == "render":
        from .renderers import render_all

        written = render_all(args.bundle, args.output)
    elif args.command == "all":
        _compile(args)
        from .extractors import extract_all
        from .renderers import render_all

        written.extend(extract_all(args.bundle, args.output))
        written.extend(render_all(args.output, args.output))
    print(f"visualization: wrote {len(written)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
