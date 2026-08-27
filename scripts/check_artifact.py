#!/usr/bin/env python3
"""Validate anonymous-review artifacts, checksums, and optional runtime inputs."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shlex
import shutil
import stat
import sys

import yaml


ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "artifacts" / "revisions.yaml"
MIRROR_FILE_LIMIT = 8_000_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_inventory(directory: Path) -> tuple[list[Path], str]:
    """Return sorted source files and a path-sensitive aggregate checksum."""
    files = sorted(
        (
            path for path in directory.rglob("*")
            if path.is_file()
            and not {".venv", "__pycache__"}.intersection(
                path.relative_to(directory).parts
            )
            and path.suffix != ".pyc"
        ),
        key=lambda path: path.relative_to(directory).as_posix(),
    )
    rows = "".join(
        f"{_sha256(path)}  {path.relative_to(directory).as_posix()}\n"
        for path in files
    )
    return files, hashlib.sha256(rows.encode("utf-8")).hexdigest()


def _hub_cache() -> Path:
    explicit = os.getenv("HF_HUB_CACHE")
    if explicit:
        return Path(explicit).expanduser()
    return Path(os.getenv("HF_HOME", "~/.cache/huggingface")).expanduser() / "hub"


def _snapshot_path(identifier: str, revision: str) -> Path:
    return _hub_cache() / ("models--" + identifier.replace("/", "--")) / "snapshots" / revision


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", action="store_true", help="report public model cache state")
    parser.add_argument("--require-models", action="store_true", help="require public models")
    parser.add_argument("--scheduler", action="store_true", help="require scheduler commands")
    parser.add_argument(
        "--comparisons", action="store_true",
        help="also require the isolated PMark environment",
    )
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    errors: list[str] = []
    if sys.version_info[:2] != (3, 11):
        errors.append(f"Python 3.11 is required; found {sys.version.split()[0]}")

    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    print(f"manifest: {MANIFEST.relative_to(ROOT)} (schema {manifest['schema_version']})")

    if (ROOT / ".gitmodules").exists() or (ROOT / "external").exists():
        errors.append("anonymous review tree must not contain submodules or external gitlinks")

    for record in manifest["comparison_sources"]:
        directory = ROOT / record["path"]
        if not directory.is_dir():
            errors.append(f"missing comparison source: {record['path']}")
            continue
        files, aggregate = _source_inventory(directory)
        if len(files) != record["source_files"]:
            errors.append(
                f"comparison source file count mismatch for {record['name']}: "
                f"{len(files)} != {record['source_files']}"
            )
        if aggregate != record["aggregate_sha256"]:
            errors.append(
                f"comparison source checksum mismatch for {record['name']}: {aggregate}"
            )
        forbidden = []
        for path in directory.rglob("*"):
            relative_parts = path.relative_to(directory).parts
            if ".venv" in relative_parts:
                continue
            if path.name == ".git" or "data" in relative_parts:
                forbidden.append(path)
        if forbidden:
            errors.append(
                f"comparison source contains data/cache/git metadata: "
                f"{forbidden[0].relative_to(ROOT)}"
            )
        oversized = [path for path in files if path.stat().st_size >= MIRROR_FILE_LIMIT]
        if oversized:
            errors.append(
                f"anonymous mirror file limit exceeded: {oversized[0].relative_to(ROOT)}"
            )
        if record.get("license_file") != "absent":
            errors.append(f"unexpected comparison license declaration: {record['name']}")
        print(
            f"comparison: {record['name']} {len(files)} files {aggregate} "
            "[upstream publishes no license file]"
        )

    for section in (
        "artifacts", "vendored_tools", "third_party_licenses", "compiled_bundle"
    ):
        for record in manifest[section]:
            path = ROOT / record["path"]
            if not path.is_file():
                errors.append(f"missing {section} file: {record['path']}")
                continue
            actual = _sha256(path)
            if actual != record["sha256"]:
                errors.append(f"checksum mismatch for {record['path']}: {actual}")
            elif path.stat().st_size >= MIRROR_FILE_LIMIT:
                errors.append(f"anonymous mirror file limit exceeded: {record['path']}")
            else:
                print(f"{section}: {record['path']} {actual}")

    scheduler_license = ROOT / "tools" / "gpu_scheduler" / "LICENSE"
    if scheduler_license.is_file():
        license_text = scheduler_license.read_text(encoding="utf-8")
        if "Apache License" not in license_text or "Version 2.0" not in license_text:
            errors.append("bundled scheduler must retain its Apache-2.0 license")

    scheduler_installer = ROOT / "scripts" / "install_bundled_scheduler.sh"
    if not scheduler_installer.is_file():
        errors.append("missing safe bundled scheduler installer")
    elif not scheduler_installer.stat().st_mode & stat.S_IXUSR:
        errors.append("bundled scheduler installer is not executable")

    if args.models or args.require_models:
        for record in manifest["models"]:
            revision = record["revision"]
            snapshot = _snapshot_path(record["identifier"], revision)
            status = "cached" if snapshot.is_dir() else "missing"
            print(f"model: {record['identifier']}@{revision} [{status}]")
            if args.require_models and not snapshot.is_dir():
                errors.append(f"model revision is not cached: {record['identifier']}@{revision}")

    if args.scheduler:
        expected_python = str(ROOT / ".venv" / "bin" / "python")
        for command in ("gpu-scheduler", "gpu-enqueue", "gpu-delete", "gpu-status"):
            resolved = shutil.which(command)
            if resolved is None:
                errors.append(f"scheduler command is not on PATH: {command}")
                continue
            launcher = Path(resolved)
            try:
                lines = launcher.read_text(encoding="utf-8").splitlines()
                tokens = shlex.split(lines[1])
            except (IndexError, OSError, UnicodeDecodeError, ValueError) as exc:
                errors.append(f"invalid bundled scheduler launcher {launcher}: {exc}")
                continue
            if len(tokens) != 4 or tokens[:2] != ["exec", expected_python]:
                errors.append(
                    f"scheduler launcher is not bound to {expected_python}: {launcher}"
                )
                continue
            expected_script = str(
                Path.home() / ".gpu-scheduler" / "bin" / command
            )
            if tokens[2:] != [expected_script, "$@"]:
                errors.append(f"scheduler launcher has unexpected target: {launcher}")
                continue
            print(f"scheduler: {command} uses frozen Python 3.11")

    if args.comparisons:
        pmark_python = ROOT / "comparisons" / "pmark" / ".venv" / "bin" / "python"
        if not pmark_python.is_file() or not os.access(pmark_python, os.X_OK):
            errors.append(
                "PMark environment is missing; run bash scripts/setup.sh --comparisons"
            )
        else:
            print("comparison environment: isolated PMark Python found")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("artifact checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
