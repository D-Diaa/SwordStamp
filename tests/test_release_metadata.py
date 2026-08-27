"""Release contracts for the self-contained double-anonymous artifact."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import stat
import subprocess
import sys
import tomllib
import unittest

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as parquet
import yaml


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "results" / "paper"
MIRROR_FILE_LIMIT = 8_000_000

EXPECTED_BUNDLE_HASHES = {
    "cell_summary.csv": "cab0a5f5674e0fb0cfa7374743f9a71800f110ba3eee03c51f1d9e9fe0a825cc",
    "dataset_summary.csv": "8d0c7e8ba1a206f5294e3b072833fd2aca0acb6de956b1eafb9d052a278bca2e",
    "per_sample.parquet": "9f3149256993090c50d0e30caf2cab9286f21d7ce1113f83f9d19b0c35f08b30",
    "unit_encoder_transfer.parquet": "eb674dc4c53c2f7caf10ad552e2dbb61aa5124e0983d1b262c15b603d6a1e61e",
}

EXPECTED_SCHEDULER_HASHES = {
    "tools/gpu_scheduler/LICENSE":
        "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
    "tools/gpu_scheduler/bin/gpu-delete":
        "85d367e4a7ca67c363e8eb8ec8f7e71dcf5d3822ac1375e9a60b435007340d1e",
    "tools/gpu_scheduler/bin/gpu-enqueue":
        "07dc6ce2f56d2f2af071abfe251bf46062b04998979461857d5deecd545c4a46",
    "tools/gpu_scheduler/bin/gpu-scheduler":
        "1135c3f2849b9faa41bc3b84298de4e03f2eafb03172fb8a3e8cb6ecd07f6ccc",
    "tools/gpu_scheduler/bin/gpu-status":
        "409b9e8235914791c78b998fdaf806a7eed0377979e4f6666beeb90df3170e68",
    "tools/gpu_scheduler/src/common.py":
        "1815b5582ea109e20e7f736c8730291b2370aae8decb24eac305fe41c2acf413",
    "tools/gpu_scheduler/src/dispatcher.py":
        "4a5a90aceadbb061b5c7272887b90be4aaa8786723fed8b942554fc6bd8c2401",
    "tools/gpu_scheduler/src/warmup.py":
        "d7669bdccd10fad342808f9631e6d8c5ee247a5e3ba65cf36b65fae5e8af2e2f",
}

EXPECTED_COMPARISON_SOURCES = {
    "PMark": {
        "path": "comparisons/pmark",
        "repository": "https://github.com/PMark-repo/PMark",
        "upstream_revision": "75140fe7142f88c51ba50984d9eb07f44f5961b3",
        "source_files": 15,
        "aggregate_sha256":
            "ee623782f772e72aa9b566e58a73b316e75a54e004eed796292f28f51f5f1f03",
        "changed_files": {
            "calibration.py", "detect.py", "pmark.py", "utils/detect.py",
            "utils/sample.py",
        },
        "insertions": 635,
        "deletions": 204,
    },
    "SAMark": {
        "path": "comparisons/samark",
        "repository": "https://github.com/Z1zs/SAMark",
        "upstream_revision": "e098bd347ba936c3e9335db6f24632fd0c319141",
        "source_files": 21,
        "aggregate_sha256":
            "e0b4cc27a82269371b8475bd6061b4964f8447656323b134cbdbd13a3866ae26",
        "changed_files": {
            "README.md", "attack_utils.py", "hierarchical_tpr.py",
            "run_pipeline.sh", "samark_detect.py", "samark_gen.py",
            "samark_gen_unwatermarked.py",
        },
        "insertions": 462,
        "deletions": 191,
    },
}

EXPECTED_PER_SAMPLE_COLUMNS = (
    "scheme", "family", "mask", "sampling", "segmentation", "flag_scope",
    "msig", "generation_num_candidates", "stage", "z_source", "attack",
    "attack_family", "attack_setting", "K", "surrogate", "anchor",
    "attacker_seg", "candidate_agg", "bag_agg", "dipper_lex", "dipper_order",
    "dataset", "sample_id", "z", "z_wm", "detected_at1", "detected_at5",
    "evades_at5", "llm_judge", "anchor_reword", "anchor_reorder", "anchor_reseg",
)

EXPECTED_TRANSFER_COLUMNS = (
    "scheme", "mask", "attack_family", "surrogate", "dataset", "document_id",
    "surrogate_encoder", "provider_encoder", "surrogate_cosine_displacement",
    "provider_cosine_displacement",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_inventory(directory: Path) -> tuple[list[Path], str]:
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


def _joined(*parts: str) -> str:
    """Build deny-list entries without placing those entries in this source."""
    return "".join(parts).lower()


IDENTITY_MARKERS = (
    _joined("a2", "di", "aa"),
    _joined("d-", "di", "aa"),
    _joined("d", "di", "aa"),
    _joined("abdul", "rahman"),
    _joined("di", "aa"),
    _joined("auce", "gypt"),
    _joined("semstamp", "2.0"),
    _joined("git", "@", "github.com"),
    _joined("/", "home", "/"),
    _joined("/", "scratch", "/"),
    _joined("a86c356d13c8b5fad009", "7e0847e11dcda4579f3d"),
    _joined("8c1ac5cd6d9cfcd76c40", "51a2457854e03576bbfd"),
)

TEXT_SUFFIXES = {
    ".cff", ".csv", ".json", ".lock", ".md", ".py", ".sh", ".tex",
    ".txt", ".toml", ".yaml", ".yml",
}
TEXT_NAMES = {".gitignore", "LICENSE"}
IGNORED_PARTS = {".git", ".venv", ".tools", "__pycache__"}


def _source_text_files():
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or any(part in IGNORED_PARTS for part in path.parts):
            continue
        if path.suffix.lower() in TEXT_SUFFIXES or path.name in TEXT_NAMES:
            yield path


class ReleaseMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (ROOT / "pyproject.toml").open("rb") as stream:
            cls.pyproject = tomllib.load(stream)
        with (ROOT / "artifacts/revisions.yaml").open(encoding="utf-8") as stream:
            cls.revisions = yaml.safe_load(stream)
        cls.bundle_manifest = json.loads(
            (BUNDLE / "manifest.json").read_text(encoding="utf-8")
        )

    def test_project_metadata_is_anonymous_mit_python311(self):
        project = self.pyproject["project"]
        self.assertEqual(project["name"], "swordstamp")
        self.assertEqual(project["version"], "1.0.0")
        self.assertEqual(project["requires-python"], ">=3.11,<3.12")
        self.assertEqual(project["license"], "MIT")
        self.assertNotIn("authors", project)
        self.assertNotIn("maintainers", project)
        self.assertNotIn("urls", project)
        self.assertIn(
            "License :: OSI Approved :: MIT License", project["classifiers"]
        )

        license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
        self.assertTrue(license_text.startswith("MIT License\n"))
        self.assertIn(
            "Copyright (c) 2026 Anonymous software author", license_text
        )
        self.assertIn("Permission is hereby granted", license_text)

    def test_dependency_surface_and_plot_group_are_explicit(self):
        dependencies = self.pyproject["project"]["dependencies"]
        self.assertLessEqual(len(dependencies), 32)
        self.assertIn("vllm", dependencies)
        self.assertIn("mauve-text", dependencies)
        self.assertEqual(
            set(self.pyproject["dependency-groups"]["plot"]),
            {"matplotlib", "numpy", "pyarrow"},
        )
        packages = self.pyproject["tool"]["hatch"]["build"]["targets"]["wheel"][
            "packages"
        ]
        self.assertEqual(
            set(packages),
            {
                "attacks", "config", "quality", "sampling", "segmentation",
                "visualization", "watermarking",
            },
        )

    def test_anonymous_tree_has_no_gitlinks_or_fork_layout(self):
        absent = (
            ".gitmodules",
            "external",
            "CITATION.cff",
            "scripts/install_gpu_scheduler.sh",
        )
        for relative in absent:
            with self.subTest(path=relative):
                self.assertFalse((ROOT / relative).exists(), relative)

        staged = subprocess.run(
            ["git", "ls-files", "--stage"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        self.assertFalse(
            any(line.startswith("160000 ") for line in staged.splitlines()),
            "anonymous branch must not contain gitlinks",
        )

        required = (
            "requirements/pmark.txt",
            "scripts/experiments/pmark.sh",
            "scripts/experiments/samark.sh",
            "scripts/experiments/_pmark_gen.sh",
            "scripts/experiments/_pmark_detect.sh",
            "scripts/experiments/_samark_gen.sh",
            "scripts/experiments/_samark_detect.sh",
            "tests/test_pmark_bridge.py",
            "tests/test_samark_bridge.py",
        )
        for relative in required:
            with self.subTest(path=relative):
                self.assertTrue((ROOT / relative).is_file(), relative)

    def test_revision_manifest_is_review_safe_and_reproducible(self):
        self.assertEqual(self.revisions["schema_version"], 1)
        self.assertEqual(
            self.revisions["artifact"]["review_state"], "double-anonymous"
        )
        self.assertNotIn("repository", self.revisions["artifact"])
        self.assertNotIn("paper_source_revision", self.revisions["artifact"])
        self.assertNotIn("repositories", self.revisions)

        models = {record["role"]: record for record in self.revisions["models"]}
        embedder = models["watermark-embedder"]
        self.assertEqual(embedder["identifier"], "AbeHou/SemStamp-c4-sbert")
        self.assertEqual(
            embedder["revision"], "a77dce8530e100ac1101981e4be390546c1cb282"
        )
        for record in models.values():
            self.assertRegex(record["revision"], r"^[0-9a-f]{40}$")

        dataset = self.revisions["datasets"][0]
        self.assertEqual(dataset["identifier"], "allenai/c4")
        self.assertEqual(
            dataset["revision"], "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
        )
        withheld_items = " ".join(self.revisions["withheld_during_review"]).lower()
        self.assertIn("scheduler", withheld_items)
        self.assertNotIn("pmark", withheld_items)
        self.assertNotIn("samark", withheld_items)

    def test_comparison_sources_are_complete_pinned_public_snapshots(self):
        records = {
            record["name"]: record
            for record in self.revisions["comparison_sources"]
        }
        self.assertEqual(set(records), set(EXPECTED_COMPARISON_SOURCES))

        for name, expected in EXPECTED_COMPARISON_SOURCES.items():
            with self.subTest(comparison=name):
                record = records[name]
                self.assertEqual(record["path"], expected["path"])
                self.assertEqual(
                    record["original_repository"], expected["repository"]
                )
                self.assertEqual(
                    record["upstream_revision"], expected["upstream_revision"]
                )
                self.assertEqual(record["source_files"], expected["source_files"])
                self.assertEqual(
                    record["aggregate_sha256"], expected["aggregate_sha256"]
                )
                self.assertEqual(record["license_file"], "absent")
                self.assertNotIn("evaluation_revision", record)
                upstream_diff = record["diff_from_upstream"]
                self.assertEqual(
                    set(upstream_diff["files"]), expected["changed_files"]
                )
                self.assertEqual(
                    upstream_diff["insertions"], expected["insertions"]
                )
                self.assertEqual(
                    upstream_diff["deletions"], expected["deletions"]
                )

                directory = ROOT / record["path"]
                files, aggregate = _source_inventory(directory)
                self.assertEqual(len(files), expected["source_files"])
                self.assertEqual(aggregate, expected["aggregate_sha256"])
                self.assertFalse(
                    any(
                        "data" in path.relative_to(directory).parts
                        for path in files
                    )
                )
                self.assertFalse(any(path.suffix == ".pyc" for path in files))
                self.assertFalse(
                    any(path.name == ".git" for path in directory.rglob("*"))
                )
                self.assertTrue(
                    all(path.stat().st_size < MIRROR_FILE_LIMIT for path in files)
                )

        documentation = "\n".join(
            (ROOT / relative).read_text(encoding="utf-8")
            for relative in ("README.md", "comparisons/README.md", "THIRD_PARTY.md")
        )
        for marker in (
            "https://github.com/PMark-repo/PMark",
            "https://github.com/Z1zs/SAMark",
            "https://github.com/THU-BPM/MarkLLM",
            "comparisons/LICENSES/MarkLLM-APACHE-2.0.txt",
            "does not cover",
        ):
            self.assertIn(marker, documentation)

        markllm_license = ROOT / "comparisons/LICENSES/MarkLLM-APACHE-2.0.txt"
        self.assertEqual(
            _sha256(markllm_license),
            "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
        )
        self.assertIn(
            "Apache License", markllm_license.read_text(encoding="utf-8")
        )

    def test_bundled_scheduler_is_pinned_anonymous_and_apache_licensed(self):
        manifest_hashes = {
            record["path"]: record["sha256"]
            for record in self.revisions["vendored_tools"]
        }
        self.assertEqual(manifest_hashes, EXPECTED_SCHEDULER_HASHES)

        vendor = ROOT / "tools" / "gpu_scheduler"
        actual_files = {
            str(path.relative_to(ROOT))
            for path in vendor.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        }
        self.assertEqual(actual_files, set(EXPECTED_SCHEDULER_HASHES))
        for relative, expected in EXPECTED_SCHEDULER_HASHES.items():
            with self.subTest(path=relative):
                self.assertEqual(_sha256(ROOT / relative), expected)

        license_text = (vendor / "LICENSE").read_text(encoding="utf-8")
        self.assertIn("Apache License", license_text)
        self.assertIn("Version 2.0", license_text)

    def test_compiled_bundle_hashes_and_mirror_limit(self):
        self.assertEqual(
            self.bundle_manifest["counts"],
            {
                "cells": 407,
                "dataset_cells": 1_199,
                "per_sample_rows": 416_768,
                "unit_encoder_transfer_rows": 154_326,
            },
        )
        selected_cells = self.bundle_manifest["selected_family_cells"]
        self.assertEqual(len(selected_cells), 13)
        self.assertIn("none:none:rejection:sentence-nltk", selected_cells)
        self.assertNotIn("none:none:best-of-n:sentence-nltk", selected_cells)

        checksum_rows = {}
        for line in (BUNDLE / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
            digest, name = line.split(maxsplit=1)
            checksum_rows[name] = digest
        self.assertEqual(checksum_rows, EXPECTED_BUNDLE_HASHES)
        self.assertEqual(
            self.bundle_manifest["sha256"], EXPECTED_BUNDLE_HASHES
        )
        revision_rows = {
            Path(record["path"]).name: record["sha256"]
            for record in self.revisions["compiled_bundle"]
        }
        self.assertEqual(revision_rows, EXPECTED_BUNDLE_HASHES)

        for name, expected in EXPECTED_BUNDLE_HASHES.items():
            path = BUNDLE / name
            with self.subTest(path=name):
                self.assertTrue(path.is_file())
                self.assertEqual(_sha256(path), expected)
                self.assertLess(path.stat().st_size, MIRROR_FILE_LIMIT)

        for path in BUNDLE.rglob("*"):
            if path.is_file():
                self.assertLess(
                    path.stat().st_size,
                    MIRROR_FILE_LIMIT,
                    f"anonymous mirror limit exceeded: {path.relative_to(ROOT)}",
                )

    def test_every_tracked_file_fits_the_anonymous_mirror(self):
        tracked = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout.split(b"\0")
        paths = [ROOT / item.decode("utf-8") for item in tracked if item]
        self.assertTrue(paths, "release tree must be staged or committed")
        for path in paths:
            with self.subTest(path=path.relative_to(ROOT)):
                self.assertLess(path.stat().st_size, MIRROR_FILE_LIMIT)

    def test_compiled_parquet_projection_has_exact_safe_schemas(self):
        projection = self.bundle_manifest["review_projection"]
        self.assertFalse(projection["raw_text_included"])
        self.assertFalse(projection["numerical_values_changed"])
        self.assertEqual(
            tuple(projection["per_sample_columns"]), EXPECTED_PER_SAMPLE_COLUMNS
        )
        self.assertEqual(
            tuple(projection["unit_encoder_transfer_columns"]),
            EXPECTED_TRANSFER_COLUMNS,
        )

        per_sample = parquet.ParquetFile(BUNDLE / "per_sample.parquet")
        transfer = parquet.ParquetFile(BUNDLE / "unit_encoder_transfer.parquet")
        self.assertEqual(tuple(per_sample.schema_arrow.names), EXPECTED_PER_SAMPLE_COLUMNS)
        self.assertEqual(tuple(transfer.schema_arrow.names), EXPECTED_TRANSFER_COLUMNS)
        self.assertEqual(per_sample.metadata.num_rows, 416_768)
        self.assertEqual(transfer.metadata.num_rows, 154_326)
        self.assertNotIn("marked_sentence", transfer.schema_arrow.names)
        self.assertNotIn("attacked_sentence", transfer.schema_arrow.names)

        provider_column = parquet.read_table(
            BUNDLE / "unit_encoder_transfer.parquet", columns=["provider_encoder"]
        ).column(0)
        self.assertEqual(
            set(pc.unique(provider_column).to_pylist()),
            {
                "AbeHou/SemStamp-c4-sbert",
                "sentence-transformers/all-mpnet-base-v2",
            },
        )

        transfer_manifest = json.loads(
            (BUNDLE / "unit_encoder_transfer.manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(transfer_manifest["rows"], 154_326)
        self.assertEqual(
            set(transfer_manifest["review_projection"]["dropped_columns"]),
            {"marked_sentence", "attacked_sentence"},
        )
        self.assertFalse(
            transfer_manifest["review_projection"]["numerical_values_changed"]
        )

    def test_source_and_parquet_metadata_have_no_identity_markers(self):
        for path in _source_text_files():
            text = path.read_text(encoding="utf-8").lower()
            for marker in IDENTITY_MARKERS:
                with self.subTest(path=path.relative_to(ROOT), marker=marker):
                    self.assertNotIn(marker, text)

        for name in ("per_sample.parquet", "unit_encoder_transfer.parquet"):
            table = parquet.read_table(BUNDLE / name)
            values: list[str] = []
            for field, column in zip(table.schema, table.columns):
                if pa.types.is_string(field.type) or pa.types.is_large_string(field.type):
                    values.extend(
                        str(value).lower()
                        for value in pc.unique(column).to_pylist()
                        if value is not None
                    )
            metadata = table.schema.metadata or {}
            values.extend(
                value.decode("utf-8", errors="replace").lower()
                for value in metadata.values()
            )
            joined = "\n".join(values)
            for marker in IDENTITY_MARKERS:
                with self.subTest(path=name, marker=marker):
                    self.assertNotIn(marker, joined)

    def test_readme_is_concise_and_covers_review_workflows(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertLessEqual(len(readme.splitlines()), 220)
        self.assertLessEqual(len(readme.split()), 1_100)
        self.assertLessEqual(len(re.findall(r"^## ", readme, flags=re.MULTILINE)), 6)

        for marker in (
            "docs/figures/teaser.png",
            "docs/figures/fidelity-robustness.png",
            "docs/figures/encoder-transfer.png",
            "18.7%",
            "10.8%",
            "embedding displacement attack (EDA)",
            "EDA-P",
            "EDA-S",
            "EDA-D",
            "attack.bag_agg=min",
            "semcut_max_words=15",
            "semcut_window=5",
            "adaptive_attack_paraphrase",
            "uv sync --frozen --only-group plot --no-install-project",
            "python -m visualization extract",
            "python -m visualization render",
            "scripts/experiments/swordstamp.sh",
            "scripts/install_bundled_scheduler.sh",
            "data preparation → generation → attacks → detection",
            "committed bundle",
            "scripts/install_bundled_scheduler.sh --print-plan",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, readme)

        for stale_runner in ("semstamp.sh",):
            self.assertNotIn(stale_runner, readme)
        self.assertIn("scripts/experiments/pmark.sh", readme)
        self.assertIn("scripts/experiments/samark.sh", readme)
        for figure in ("teaser.png", "fidelity-robustness.png", "encoder-transfer.png"):
            self.assertGreater((ROOT / "docs" / "figures" / figure).stat().st_size, 1024)

        bundle_readme = (BUNDLE / "README.md").read_text(encoding="utf-8")
        self.assertLessEqual(len(bundle_readme.splitlines()), 30)
        for marker in ("--only-group plot", "visualization extract", "visualization render"):
            self.assertIn(marker, bundle_readme)

    def test_release_entry_points_are_executable_and_checker_passes(self):
        for relative in (
            "scripts/setup.sh",
            "scripts/check_artifact.py",
            "scripts/prepare_c4.py",
            "scripts/experiments/swordstamp.sh",
            "scripts/experiments/pmark.sh",
            "scripts/experiments/samark.sh",
            "scripts/experiments/_pmark_gen.sh",
            "scripts/experiments/_pmark_detect.sh",
            "scripts/experiments/_samark_gen.sh",
            "scripts/experiments/_samark_detect.sh",
            "scripts/install_bundled_scheduler.sh",
        ):
            with self.subTest(path=relative):
                mode = (ROOT / relative).stat().st_mode
                self.assertTrue(mode & stat.S_IXUSR, relative)

        setup = (ROOT / "scripts/setup.sh").read_text(encoding="utf-8").lower()
        for forbidden in (
            "git submodule", "git clone", "external/",
            "curl ", "wget ", "--scheduler",
        ):
            self.assertNotIn(forbidden, setup)
        self.assertIn("scripts/check_artifact.py", setup)
        self.assertIn("--comparisons", setup)
        self.assertIn("comparisons/pmark/.venv", setup)
        self.assertNotRegex(
            setup,
            r"(?m)^\s*bash\s+.*install_bundled_scheduler",
        )

        workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn('--bundle "$output" --output "$output"', workflow)
        self.assertIn("tables/pgfplots/teaser/swordstamp.dat", workflow)
        self.assertIn("install_bundled_scheduler.sh --print-plan", workflow)
        self.assertIn("install_bundled_scheduler.sh --yes", workflow)

        result = subprocess.run(
            [sys.executable, "scripts/check_artifact.py"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("artifact checks passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
