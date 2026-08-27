"""Validate the immutable pieces of a SwordStamp artifact checkout."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "artifacts" / "revisions.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _hub_cache() -> Path:
    explicit = os.getenv("HF_HUB_CACHE")
    if explicit:
        return Path(explicit).expanduser()
    hf_home = Path(os.getenv("HF_HOME", "~/.cache/huggingface")).expanduser()
    return hf_home / "hub"


def _snapshot_path(identifier: str, revision: str) -> Path:
    cache_name = "models--" + identifier.replace("/", "--")
    return _hub_cache() / cache_name / "snapshots" / revision


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", action="store_true",
        help="report whether every recorded model revision is locally cached",
    )
    parser.add_argument(
        "--require-models", action="store_true",
        help="fail if any recorded model revision is not locally cached",
    )
    parser.add_argument(
        "--scheduler", action="store_true",
        help="also require the gpu-scheduler and gpu-enqueue commands",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    errors: list[str] = []

    if sys.version_info[:2] != (3, 11):
        errors.append(f"Python 3.11 is required; found {sys.version.split()[0]}")

    with MANIFEST.open(encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream)
    print(f"manifest: {MANIFEST.relative_to(ROOT)} (schema {manifest['schema_version']})")

    for name in ("pmark", "samark"):
        record = manifest["repositories"][name]
        path = ROOT / "external" / ("SAMark" if name == "samark" else name)
        actual = _git_head(path)
        expected = record["revision"]
        if actual is None:
            errors.append(f"submodule is not initialized: {path.relative_to(ROOT)}")
        elif actual != expected:
            errors.append(f"{path.relative_to(ROOT)} is at {actual}, expected {expected}")
        else:
            print(f"submodule: {name} {actual}")

    for record in manifest["artifacts"]:
        path = ROOT / record["path"]
        if not path.is_file():
            errors.append(f"artifact is missing: {record['path']}")
            continue
        actual = _sha256(path)
        if actual != record["sha256"]:
            errors.append(f"checksum mismatch for {record['path']}: {actual}")
        else:
            print(f"artifact: {record['path']} {actual}")

    if args.models or args.require_models:
        for record in manifest["models"]:
            snapshot = _snapshot_path(record["identifier"], record["revision"])
            status = "cached" if snapshot.is_dir() else "missing"
            print(f"model: {record['identifier']}@{record['revision']} [{status}]")
            if args.require_models and not snapshot.is_dir():
                errors.append(
                    f"model revision is not cached: {record['identifier']}@{record['revision']}"
                )

    if args.scheduler:
        for command in ("gpu-scheduler", "gpu-enqueue"):
            if shutil.which(command) is None:
                errors.append(f"scheduler command is not on PATH: {command}")
            else:
                print(f"scheduler: {command} found")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("artifact checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
