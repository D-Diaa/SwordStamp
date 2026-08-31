import configparser
import os
from pathlib import Path
import stat
import subprocess
import sys
import tomllib
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]


class ReleaseMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (ROOT / "pyproject.toml").open("rb") as stream:
            cls.pyproject = tomllib.load(stream)
        with (ROOT / "artifacts/revisions.yaml").open(encoding="utf-8") as stream:
            cls.revisions = yaml.safe_load(stream)

    def test_project_metadata_targets_only_python_311(self):
        project = self.pyproject["project"]
        self.assertEqual(project["name"], "swordstamp")
        self.assertEqual(project["version"], "1.0.0")
        self.assertEqual(project["requires-python"], ">=3.11,<3.12")
        self.assertEqual(project["license"], "MIT")
        self.assertEqual(project["authors"], [{"name": "Abdulrahman Diaa"}])

    def test_software_and_paper_citation_metadata_are_distinct(self):
        citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
        self.assertEqual(
            citation["authors"],
            [{"family-names": "Diaa", "given-names": "Abdulrahman"}],
        )
        self.assertEqual(
            citation["preferred-citation"]["authors"],
            [
                {"family-names": "Diaa", "given-names": "Abdulrahman"},
                {"family-names": "Petit", "given-names": "Jonathan"},
                {"family-names": "Kerschbaum", "given-names": "Florian"},
            ],
        )
        preferred = citation["preferred-citation"]
        self.assertEqual(preferred["journal"], "arXiv")
        self.assertEqual(preferred["year"], 2026)
        self.assertEqual(preferred["url"], "https://arxiv.org/abs/2608.27666")
        self.assertIn(
            {
                "type": "other",
                "value": "arXiv:2608.27666",
                "description": "arXiv identifier",
            },
            preferred["identifiers"],
        )
        self.assertEqual(
            self.pyproject["project"]["urls"]["Paper"],
            "https://arxiv.org/abs/2608.27666",
        )
        self.assertEqual(
            self.revisions["artifact"]["paper_identifier"],
            "arXiv:2608.27666",
        )
        self.assertEqual(
            self.revisions["artifact"]["paper_url"],
            "https://arxiv.org/abs/2608.27666",
        )
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("https://arxiv.org/abs/2608.27666", readme)
        self.assertIn("@misc{diaa2026semantic", readme)

    def test_dependency_surface_names_only_direct_requirements(self):
        project = self.pyproject["project"]
        dependencies = project["dependencies"]
        self.assertLessEqual(len(dependencies), 32)
        self.assertIn("vllm", dependencies)
        self.assertIn("mauve-text", dependencies)
        parrot = [item for item in dependencies if item.startswith("parrot @ ")]
        self.assertEqual(len(parrot), 1)
        self.assertIn("03084c54b64019ba5fa0b620b9c70ad81123e458", parrot[0])
        for transitive_only in ("aiohttp", "certifi", "filelock", "wandb"):
            self.assertNotIn(transitive_only, dependencies)

    def test_plot_group_is_small_and_complete(self):
        self.assertEqual(
            set(self.pyproject["dependency-groups"]["plot"]),
            {"matplotlib", "numpy", "pyarrow"},
        )

    def test_wheel_includes_every_first_party_package(self):
        packages = self.pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
        self.assertEqual(
            set(packages),
            {"attacks", "config", "quality", "sampling", "segmentation", "visualization", "watermarking"},
        )

    def test_source_distribution_does_not_vendor_submodules(self):
        excluded = self.pyproject["tool"]["hatch"]["build"]["targets"]["sdist"]["exclude"]
        self.assertEqual(excluded, ["/external", "/external/**"])

    def test_exact_repository_pins_match_gitmodules(self):
        parser = configparser.ConfigParser()
        parser.read(ROOT / ".gitmodules")
        expected = {
            "pmark": (
                "git@github.com:D-Diaa/PMark.git",
                "5a8ad3008fdc2c58e607de2f60941c2d1911deb9",
            ),
            "samark": (
                "git@github.com:D-Diaa/SAMark.git",
                "9160b8025fda05be9a36e02abf0450c1249693cc",
            ),
        }
        for name, (url, revision) in expected.items():
            manifest = self.revisions["repositories"][name]
            self.assertEqual(manifest["url"], url)
            self.assertEqual(manifest["revision"], revision)
        self.assertEqual(parser['submodule "external/pmark"']["url"], expected["pmark"][0])
        self.assertEqual(parser['submodule "external/SAMark"']["url"], expected["samark"][0])

        staged = subprocess.run(
            ["git", "ls-files", "--stage", "external/pmark", "external/SAMark"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        gitlinks = {
            line.split("\t", 1)[1]: line.split()[1]
            for line in staged
            if line.startswith("160000 ")
        }
        self.assertEqual(gitlinks["external/pmark"], expected["pmark"][1])
        self.assertEqual(gitlinks["external/SAMark"], expected["samark"][1])

    def test_model_and_dataset_records_are_content_pinned(self):
        paper_models = [
            record for record in self.revisions["models"]
            if record["scope"] in {"paper", "paper-quality"}
        ]
        self.assertGreaterEqual(len(paper_models), 10)
        for record in paper_models:
            self.assertRegex(record["revision"], r"^[0-9a-f]{40}$")
        dataset = self.revisions["datasets"][0]
        self.assertEqual(dataset["identifier"], "allenai/c4")
        self.assertEqual(dataset["revision"], "1588ec454efa1a09f29cd18ddd04fe05fc8653a2")
        model_ids = {record["identifier"] for record in self.revisions["models"]}
        self.assertIn("DDiaa/WM-Removal-Unigram-Qwen2.5-3B", model_ids)
        self.assertIn("Qwen/Qwen3-32B", model_ids)
        external_ids = {
            record.get("identifier")
            for record in self.revisions["unverified_external_models"]
        }
        self.assertIn("gpt-3.5-turbo", external_ids)

    def test_pmark_environment_uses_known_fair_evaluation_stack(self):
        requirements = (ROOT / "requirements/pmark.txt").read_text(encoding="utf-8")
        for pin in (
            "torch==2.6.0+cu124",
            "transformers==4.55.2",
            "vllm==0.8.5.post1",
            "datasets==4.0.0",
            "sentence-transformers==5.1.0",
        ):
            self.assertIn(pin, requirements)

    def test_release_entry_points_are_executable(self):
        for relative in (
            "scripts/setup.sh",
            "scripts/install_gpu_scheduler.sh",
            "scripts/check_artifact.py",
            "scripts/prepare_c4.py",
        ):
            mode = (ROOT / relative).stat().st_mode
            self.assertTrue(mode & stat.S_IXUSR, relative)

    def test_scheduler_plan_is_pinned_and_has_no_implicit_initialization(self):
        result = subprocess.run(
            ["bash", "scripts/install_gpu_scheduler.sh", "--print-plan"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("a86c356d13c8b5fad0097e0847e11dcda4579f3d", result.stdout)
        self.assertIn("does NOT initialize a GPU pool", result.stdout)
        self.assertIn("~/.gpu-scheduler", result.stdout)

    def test_artifact_checker_succeeds_without_models_or_scheduler(self):
        result = subprocess.run(
            [sys.executable, "scripts/check_artifact.py"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("artifact checks passed", result.stdout)

    def test_root_uses_the_mit_license(self):
        text = (ROOT / "LICENSE").read_text(encoding="utf-8")
        self.assertTrue(text.startswith("MIT License\n"))
        self.assertIn("Copyright (c) 2026 Abdulrahman Diaa", text)
        self.assertIn("Permission is hereby granted", text)
        for relative in ("README.md", "ARTIFACT.md", "THIRD_PARTY.md"):
            documentation = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn("MIT License", documentation)

    def test_docs_cover_functional_full_and_visualization_paths(self):
        docs = "\n".join(
            (ROOT / relative).read_text(encoding="utf-8")
            for relative in ("README.md", "ARTIFACT.md")
        )
        for command in (
            "python -m visualization compile",
            "python -m visualization extract",
            "python -m visualization render",
            "python -m visualization all",
            "scripts/prepare_c4.py --output-dir data",
            "scripts/install_gpu_scheduler.sh --print-plan",
        ):
            self.assertIn(command, docs)

    def test_readme_covers_headlines_and_the_public_eda_surface(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        for marker in (
            "docs/figures/teaser.png",
            "docs/figures/fidelity-robustness.png",
            "docs/figures/encoder-transfer.png",
            "embedding displacement attack (EDA)",
            "EDA-P",
            "EDA-S",
            "EDA-D",
            "adaptive_attack_paraphrase",
            "scripts/experiments/swordstamp.sh",
        ):
            self.assertIn(marker, readme)
        for figure in ("teaser.png", "fidelity-robustness.png", "encoder-transfer.png"):
            path = ROOT / "docs" / "figures" / figure
            self.assertGreater(path.stat().st_size, 1024)


if __name__ == "__main__":
    unittest.main()
