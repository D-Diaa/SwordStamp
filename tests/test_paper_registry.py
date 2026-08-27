"""Contracts for the exact paper-only experiment registry and dry runs."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from config.paper import (
    COMPARISONS,
    DATASETS,
    LADDERS,
    NONE_PRESET,
    ORACLE_KS,
    PAPER_RUNGS,
    PROVIDER_CANDIDATES,
)


ROOT = Path(__file__).resolve().parents[1]


class PaperRegistryTests(unittest.TestCase):
    def test_registry_is_exactly_two_five_rung_additive_ladders(self):
        expected = (
            ("base", "lsh", "context", "rejection", "sentence-nltk"),
            ("bestn", "lsh", "context", "best-of-n", "sentence-nltk"),
            ("fixed", "lsh", "fixed", "best-of-n", "sentence-nltk"),
            ("diverse", "lsh", "fixed_diverse", "best-of-n", "sentence-nltk"),
            ("span", "lsh", "fixed_diverse", "best-of-n", "semspan-spacy-max15-win5"),
            ("kbase", "kmeans", "context", "rejection", "sentence-nltk"),
            ("kbestn", "kmeans", "context", "best-of-n", "sentence-nltk"),
            ("kfixed", "kmeans", "fixed", "best-of-n", "sentence-nltk"),
            ("kdiverse", "kmeans", "fixed_diverse", "best-of-n", "sentence-nltk"),
            ("kspan", "kmeans", "fixed_diverse", "best-of-n", "semspan-spacy-max15-win5"),
        )
        actual = tuple(
            (rung.key, rung.family, rung.mask, rung.sampling, rung.segmentation)
            for rung in PAPER_RUNGS
        )
        self.assertEqual(actual, expected)
        self.assertEqual(tuple(LADDERS), ("lsh", "kmeans"))
        self.assertEqual({rung.num_candidates for rung in PAPER_RUNGS}, {64})
        self.assertEqual(PROVIDER_CANDIDATES, 64)

    def test_comparisons_datasets_and_oracle_effort_are_exact(self):
        self.assertEqual(
            tuple(
                (cell.key, cell.family, cell.mask, cell.sampling,
                 cell.segmentation, cell.msig, cell.num_candidates)
                for cell in COMPARISONS
            ),
            (
                ("pmark-online", "pmark", "online", "rejection", "sentence-nltk", 4, 64),
                ("samark", "samark", "flags-run", "rejection", "sentence-nltk", 2, 64),
            ),
        )
        self.assertEqual(
            DATASETS,
            ("c4-val-def-256", "c4-val-def-256b", "c4-val-def-512"),
        )
        self.assertEqual(ORACLE_KS, (4, 8, 16, 32, 64))

    def test_preset_directory_contains_only_the_registry_and_baseline(self):
        expected = {Path(rung.preset).name for rung in PAPER_RUNGS}
        expected.add(Path(NONE_PRESET).name)
        actual = {path.name for path in (ROOT / "config" / "presets").glob("*.yaml")}
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), 11)

    def test_registry_cli_prints_the_exact_ten_rungs(self):
        result = subprocess.run(
            [sys.executable, "scripts/experiments/paper.py", "rungs"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        rows = [line.split("\t") for line in result.stdout.splitlines()]
        self.assertEqual(len(rows), 10)
        self.assertEqual([row[0] for row in rows], [rung.key for rung in PAPER_RUNGS])
        self.assertTrue(all(row[-1] == "64" for row in rows))

    def test_swordstamp_orchestrator_dry_run_is_paper_only(self):
        with tempfile.TemporaryDirectory() as cache:
            env = os.environ.copy()
            env.update({
                "ATTACKS_FILTER": "^$",
                "DRY_RUN": "1",
                "QUALITY_BATCH": "off",
                "UV_CACHE_DIR": cache,
                "UV_NO_SYNC": "1",
                "UV_PROJECT_ENVIRONMENT": sys.prefix,
            })
            result = subprocess.run(
                ["bash", "scripts/experiments/swordstamp.sh"],
                cwd=ROOT,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(
                "paper cells: 11 (10 rungs + baseline)", result.stdout
            )
            self.assertIn("DRY RUN — nothing was enqueued", result.stdout)
            self.assertNotIn("ablation", result.stdout.lower())
            self.assertEqual(result.stdout.count("--- rung "), 10)
            self.assertEqual(result.stdout.count("--- none baseline"), 1)


if __name__ == "__main__":
    unittest.main()
