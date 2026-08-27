"""Exact row-level scope contract for the committed paper-results bundle."""

from __future__ import annotations

from collections import Counter
import csv
import json
import math
from pathlib import Path
import unittest

import pyarrow.compute as pc
import pyarrow.parquet as parquet

from config.paper import DATASETS, LADDERS, PMARK, SAMARK
from visualization.compile_results import paper_attack_leaves


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "results" / "paper"
HUMAN_DATASET = "c4-human-def"
FULL_K = (1, 2, 4, 8, 16, 32, 64)
POSITIONAL_COMPARISON_K = (4, 16, 64)
DETECTOR_ACCESS_K = (4, 8, 16, 32, 64)

SCHEMES = {
    "lsh": "SemStamp",
    "kmeans": "k-SemStamp",
    "pmark": "PMark",
    "samark": "SAMark",
    "none": "No watermark",
}

Cell = tuple[str, str, str, str]
Run = tuple[Cell, str, str]


def _cell(family: str, mask: str, sampling: str, segmentation: str) -> Cell:
    return family, mask, sampling, segmentation


PAPER_CELLS = tuple(
    _cell(rung.family, rung.mask, rung.sampling, rung.segmentation)
    for family in ("lsh", "kmeans")
    for rung in LADDERS[family]
) + (
    _cell(PMARK.family, PMARK.mask, PMARK.sampling, PMARK.segmentation),
    _cell(SAMARK.family, SAMARK.mask, SAMARK.sampling, SAMARK.segmentation),
    _cell("none", "none", "rejection", "sentence-nltk"),
)

CONTROLLED_ATTACKS = tuple(
    f"{name}-ratio={ratio}"
    for name, ratios in (
        ("controlled_reorder", ("0.1", "0.25", "0.5", "0.75", "1.0")),
        ("synonym_substitution", ("0.1", "0.25", "0.5", "0.75", "1.0")),
        ("split_midpoint", ("0.2", "0.5", "1.0")),
        ("merge_adjacent", ("0.2", "0.5", "1.0")),
    )
    for ratio in ratios
)

DIPPER_SETTINGS = ((20, 20), (20, 40), (20, 80), (40, 20), (80, 20))


def _passive_attacks(family: str) -> tuple[str, ...]:
    suffix = "-bigram" if family in {"pmark", "samark"} else "-bigram-threshold=0.03"
    return "pegasus", f"pegasus{suffix}", "parrot", f"parrot{suffix}"


def _dipper_attacks(family: str) -> tuple[str, ...]:
    if family in {"pmark", "samark"}:
        return tuple(f"dipper-l{lex}-o{order}" for lex, order in DIPPER_SETTINGS)
    return tuple(f"dipper-lex{lex}-order{order}" for lex, order in DIPPER_SETTINGS)


def _adaptive_attacks(cell: Cell) -> tuple[str, ...]:
    family, mask, _sampling, segmentation = cell
    if family == "pmark":
        return tuple(f"adp-K{k}" for k in FULL_K)
    if family == "samark":
        return (
            *(f"adpbag-K{k}" for k in FULL_K),
            *(f"adp-K{k}" for k in POSITIONAL_COMPARISON_K),
        )
    if family not in {"lsh", "kmeans"}:
        return ()

    stem = "adaptive-Qwen2.5-3B-Instruct-standard"
    positional = tuple(
        f"{stem}-K{k}-min-surr=bge"
        for k in (FULL_K if mask == "context" else POSITIONAL_COMPARISON_K)
    )
    if mask == "context":
        return positional
    attacker_seg = (
        "-aseg=semspan-nltk-max15-win5"
        if segmentation == "semspan-spacy-max15-win5"
        else ""
    )
    scheme_specific = tuple(
        f"{stem}-K{k}-min-bag=min-surr=bge{attacker_seg}" for k in FULL_K
    )
    return (*scheme_specific, *positional)


def _oracle_attacks(cell: Cell) -> tuple[str, ...]:
    if cell not in {
        _cell("kmeans", "context", "rejection", "sentence-nltk"),
        _cell(
            "kmeans", "fixed_diverse", "best-of-n",
            "semspan-spacy-max15-win5",
        ),
    }:
        return ()
    return tuple(
        f"oracle-Qwen2.5-3B-Instruct-standard-K{k}"
        for k in DETECTOR_ACCESS_K
    )


def _attack_family(attack: str) -> str:
    if attack == "none":
        return "none"
    if attack.startswith(("adaptive-", "adp-", "adpbag-")):
        return "adaptive"
    if attack.startswith("oracle-"):
        return "oracle"
    if attack.startswith("dipper-"):
        return "dipper"
    if attack.startswith("parrot"):
        return "parrot"
    if attack.startswith("pegasus"):
        return "pegasus"
    if attack in CONTROLLED_ATTACKS:
        return "other"
    raise AssertionError(f"unregistered paper attack {attack!r}")


def _expected_runs() -> tuple[Run, ...]:
    runs: list[Run] = []
    for cell in PAPER_CELLS:
        family = cell[0]
        runs.append((cell, "watermark", "none"))
        if family in {"lsh", "kmeans", "samark"}:
            runs.append((cell, "human", "none"))
        if family == "none":
            continue
        attacks = [
            *_passive_attacks(family),
            *_dipper_attacks(family),
            *_adaptive_attacks(cell),
            *_oracle_attacks(cell),
        ]
        if family in {"lsh", "kmeans"}:
            attacks.extend(CONTROLLED_ATTACKS)
        runs.extend((cell, "attack", attack) for attack in attacks)
    return tuple(runs)


EXPECTED_RUNS = _expected_runs()
EXPECTED_DATASET_RUNS = tuple(
    (cell, stage, attack, dataset)
    for cell, stage, attack in EXPECTED_RUNS
    for dataset in ((HUMAN_DATASET,) if stage == "human" else DATASETS)
)


def _read_csv(name: str) -> list[dict[str, str]]:
    with (BUNDLE / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _row_cell(row: dict) -> Cell:
    return _cell(
        str(row["family"]), str(row["mask"]), str(row["sampling"]),
        str(row["segmentation"]),
    )


def _is_blank_number(value) -> bool:
    if value is None or value == "":
        return True
    return isinstance(value, float) and math.isnan(value)


class CompiledBundleScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cells = _read_csv("cell_summary.csv")
        cls.datasets = _read_csv("dataset_summary.csv")
        cls.manifest = json.loads((BUNDLE / "manifest.json").read_text())

    def test_expected_scope_definition_is_exact_and_nonduplicated(self):
        self.assertEqual(len(PAPER_CELLS), 13)
        self.assertEqual(len(set(PAPER_CELLS)), 13)
        self.assertEqual(len(EXPECTED_RUNS), 407)
        self.assertEqual(len(set(EXPECTED_RUNS)), 407)
        self.assertEqual(len(EXPECTED_DATASET_RUNS), 1199)
        self.assertEqual(len(set(EXPECTED_DATASET_RUNS)), 1199)

    def test_cell_summary_is_exactly_the_paper_runs(self):
        actual = [
            (_row_cell(row), row["stage"], row["attack"])
            for row in self.cells
        ]
        self.assertEqual(Counter(actual), Counter(EXPECTED_RUNS))
        self._assert_row_metadata(self.cells)

    def test_compiler_accepts_exactly_the_paper_attack_leaves(self):
        for cell in PAPER_CELLS:
            if cell[0] == "none":
                continue
            expected = {
                "watermarked",
                *(
                    attack
                    for run_cell, stage, attack in EXPECTED_RUNS
                    if run_cell == cell and stage == "attack"
                ),
            }
            with self.subTest(cell=cell):
                self.assertEqual(paper_attack_leaves(*cell), expected)

    def test_dataset_summary_is_exactly_the_paper_runs_and_shards(self):
        actual = [
            (_row_cell(row), row["stage"], row["attack"], row["dataset"])
            for row in self.datasets
        ]
        self.assertEqual(Counter(actual), Counter(EXPECTED_DATASET_RUNS))
        self._assert_row_metadata(self.datasets)

    def _assert_row_metadata(self, rows: list[dict[str, str]]) -> None:
        for row in rows:
            with self.subTest(
                family=row["family"], stage=row["stage"], attack=row["attack"]
            ):
                family = row["family"]
                self.assertEqual(row["scheme"], SCHEMES[family])
                self.assertEqual(row["attack_family"], _attack_family(row["attack"]))

                if family == "samark":
                    self.assertEqual(row["flag_scope"], "run")
                    self.assertEqual(float(row["msig"]), 2.0)
                else:
                    self.assertEqual(row["flag_scope"], "")
                    self.assertTrue(_is_blank_number(row["msig"]))

                if row["stage"] == "human" and family in {"lsh", "kmeans"}:
                    self.assertTrue(_is_blank_number(row["generation_num_candidates"]))
                else:
                    expected = 1.0 if family == "none" else 64.0
                    self.assertEqual(float(row["generation_num_candidates"]), expected)

                if row["segmentation"] == "sentence-nltk":
                    self.assertEqual(row["segmentation_type"], "sentence")
                    self.assertEqual(row["segmentation_backend"], "nltk")
                    self.assertTrue(_is_blank_number(row["segmentation_max_words"]))
                    self.assertTrue(_is_blank_number(row["segmentation_window"]))
                else:
                    self.assertEqual(
                        row["segmentation"], "semspan-spacy-max15-win5"
                    )
                    self.assertEqual(row["segmentation_type"], "semspan")
                    self.assertEqual(row["segmentation_backend"], "spacy")
                    self.assertEqual(float(row["segmentation_max_words"]), 15.0)
                    self.assertEqual(float(row["segmentation_window"]), 5.0)

    def test_per_sample_rows_match_every_dataset_cell_exactly(self):
        keys = [
            "family", "mask", "sampling", "segmentation", "stage", "attack",
            "dataset",
        ]
        table = parquet.read_table(
            BUNDLE / "per_sample.parquet", columns=[*keys, "sample_id"]
        )
        grouped = table.group_by(keys).aggregate([
            ("sample_id", "count"),
            ("sample_id", "count_distinct"),
            ("sample_id", "min"),
            ("sample_id", "max"),
        ])
        actual = []
        for row in grouped.to_pylist():
            cell = _cell(
                row["family"], row["mask"], row["sampling"], row["segmentation"]
            )
            actual.append((cell, row["stage"], row["attack"], row["dataset"]))
            expected_count = {
                "c4-val-def-256": 256,
                "c4-val-def-256b": 256,
                "c4-val-def-512": 512,
                HUMAN_DATASET: 1024,
            }[row["dataset"]]
            with self.subTest(cell=cell, stage=row["stage"], attack=row["attack"]):
                self.assertEqual(row["sample_id_count"], expected_count)
                self.assertEqual(row["sample_id_count_distinct"], expected_count)
                self.assertEqual(row["sample_id_min"], 0)
                self.assertEqual(row["sample_id_max"], expected_count - 1)
        self.assertEqual(Counter(actual), Counter(EXPECTED_DATASET_RUNS))
        self.assertEqual(table.num_rows, 416_768)

    def test_transfer_is_only_the_two_paper_curves(self):
        table = parquet.read_table(BUNDLE / "unit_encoder_transfer.parquet")
        expected_pairs = {
            ("SemStamp", "context", "AbeHou/SemStamp-c4-sbert"),
            (
                "PMark", "online",
                "sentence-transformers/all-mpnet-base-v2",
            ),
        }
        actual_pairs = set(zip(
            pc.unique(table["scheme"]).to_pylist(),
            pc.unique(table["mask"]).to_pylist(),
            pc.unique(table["provider_encoder"]).to_pylist(),
        ))
        # unique() is column-wise, so also verify row-wise combinations below.
        actual_pairs = {
            (row["scheme"], row["mask"], row["provider_encoder"])
            for row in table.select(
                ["scheme", "mask", "provider_encoder"]
            ).group_by(
                ["scheme", "mask", "provider_encoder"]
            ).aggregate([]).to_pylist()
        }
        self.assertEqual(actual_pairs, expected_pairs)
        self.assertEqual(set(pc.unique(table["attack_family"]).to_pylist()), {"adaptive"})
        self.assertEqual(set(pc.unique(table["surrogate"]).to_pylist()), {"bge"})
        self.assertEqual(
            set(pc.unique(table["surrogate_encoder"]).to_pylist()),
            {"BAAI/bge-base-en-v1.5"},
        )
        counts = {
            (row["scheme"], row["dataset"]): row["document_id_count"]
            for row in table.group_by(["scheme", "dataset"]).aggregate([
                ("document_id", "count"),
            ]).to_pylist()
        }
        self.assertEqual(counts, {
            ("SemStamp", "c4-val-def-256"): 21_225,
            ("SemStamp", "c4-val-def-256b"): 19_178,
            ("SemStamp", "c4-val-def-512"): 39_998,
            ("PMark", "c4-val-def-256"): 18_403,
            ("PMark", "c4-val-def-256b"): 18_447,
            ("PMark", "c4-val-def-512"): 37_075,
        })
        self.assertEqual(table.num_rows, 154_326)

    def test_manifest_declares_the_same_exact_scope(self):
        self.assertEqual(set(self.manifest["selected_family_cells"]), {
            ":".join(cell) for cell in PAPER_CELLS
        })
        self.assertEqual(self.manifest["counts"], {
            "cells": 407,
            "dataset_cells": 1199,
            "per_sample_rows": 416_768,
            "unit_encoder_transfer_rows": 154_326,
        })
        self.assertEqual(
            set(self.manifest["unit_encoder_transfer_cells"]),
            {
                "lsh:context:rejection:sentence-nltk",
                "pmark:online:rejection:sentence-nltk",
            },
        )
        serialized = json.dumps(self.manifest, sort_keys=True).lower()
        for stale in (
            "max12", "win4", "boundary_exchange", "pmark:offline",
            "samark:flags-document",
        ):
            self.assertNotIn(stale, serialized)


if __name__ == "__main__":
    unittest.main()
