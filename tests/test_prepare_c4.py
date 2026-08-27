"""Deterministic, network-free tests for the paper C4 preparation script."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prepare_c4", ROOT / "scripts" / "prepare_c4.py")
prepare_c4 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(prepare_c4)


class _SavedDataset:
    def __init__(self, data):
        self.data = data

    def save_to_disk(self, path):
        path = Path(path)
        path.mkdir()
        _FakeDataset.saved[path.name] = list(self.data["text"])
        (path / "fixture.json").write_text("{}\n", encoding="utf-8")


class _FakeDataset:
    saved = {}

    @classmethod
    def from_dict(cls, data):
        return _SavedDataset(data)


class PrepareC4Tests(unittest.TestCase):
    def test_source_and_partition_constants_are_exact(self):
        self.assertEqual(prepare_c4.DATASET_ID, "allenai/c4")
        self.assertEqual(prepare_c4.DATASET_CONFIG, "realnewslike")
        self.assertEqual(prepare_c4.SOURCE_SPLIT, "train")
        self.assertEqual(
            prepare_c4.DATASET_REVISION,
            "1588ec454efa1a09f29cd18ddd04fe05fc8653a2",
        )
        self.assertEqual(prepare_c4.SEED, 42)
        self.assertEqual(
            prepare_c4.OUTPUT_SIZES,
            (
                ("c4-center-8192", 8192),
                ("c4-val-def-256", 256),
                ("c4-val-def-256b", 256),
                ("c4-val-def-512", 512),
                ("c4-human-def", 1024),
            ),
        )

    def test_source_scan_filters_empty_and_duplicate_texts(self):
        rows = [
            {"text": None},
            {"text": "   "},
            {"text": "document-0"},
            {"text": "document-0"},
            *({"text": f"document-{index}"}
              for index in range(1, prepare_c4.TOTAL_DOCUMENTS)),
            {"text": "unreached"},
        ]
        with patch.object(prepare_c4, "load_dataset", return_value=iter(rows)) as load:
            texts, scanned = prepare_c4._source_texts()

        load.assert_called_once_with(
            "allenai/c4",
            "realnewslike",
            split="train",
            revision=prepare_c4.DATASET_REVISION,
            streaming=True,
        )
        self.assertEqual(len(texts), prepare_c4.TOTAL_DOCUMENTS)
        self.assertEqual(len(set(texts)), prepare_c4.TOTAL_DOCUMENTS)
        self.assertEqual(scanned, prepare_c4.TOTAL_DOCUMENTS + 3)

    def test_prepare_writes_disjoint_seeded_partitions_and_manifest(self):
        texts = [f"text-{index}" for index in range(prepare_c4.TOTAL_DOCUMENTS)]
        expected_order = [
            texts[index]
            for index in sorted(
                range(prepare_c4.TOTAL_DOCUMENTS),
                key=prepare_c4._selection_key,
            )
        ]
        _FakeDataset.saved = {}

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "data"
            with (
                patch.object(prepare_c4, "Dataset", _FakeDataset),
                patch.object(prepare_c4, "_source_texts", return_value=(texts, 12345)),
            ):
                manifest = prepare_c4.prepare(output)

            on_disk = json.loads((output / "c4_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(on_disk, manifest)
            self.assertEqual(manifest["source_rows_scanned"], 12345)
            self.assertEqual(manifest["total_documents"], 10240)
            self.assertTrue(manifest["pairwise_content_disjoint"])

            flattened = []
            observed_sets = []
            for name, size in prepare_c4.OUTPUT_SIZES:
                selected = _FakeDataset.saved[name]
                self.assertEqual(len(selected), size)
                self.assertEqual(manifest["outputs"][name]["rows"], size)
                self.assertEqual(
                    manifest["outputs"][name]["ordered_text_sha256"],
                    prepare_c4._digest(selected),
                )
                self.assertTrue((output / name / "fixture.json").is_file())
                flattened.extend(selected)
                observed_sets.append(set(selected))

            self.assertEqual(flattened, expected_order)
            for index, left in enumerate(observed_sets):
                for right in observed_sets[index + 1:]:
                    self.assertTrue(left.isdisjoint(right))

    def test_prepare_refuses_existing_outputs_before_reading_source(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "data"
            occupied = output / prepare_c4.OUTPUT_SIZES[0][0]
            occupied.mkdir(parents=True)
            with patch.object(prepare_c4, "_source_texts") as source:
                with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                    prepare_c4.prepare(output)
            source.assert_not_called()

    def test_prepare_refuses_a_symlinked_output_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            target.mkdir()
            output = root / "data"
            output.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(FileExistsError, "symlinked output"):
                prepare_c4.prepare(output)


if __name__ == "__main__":
    unittest.main()
