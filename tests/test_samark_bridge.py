import importlib.util
import os
import unittest

import numpy as np
import torch


_ROOT = os.path.dirname(os.path.dirname(__file__))
_DETECT_PATH = os.path.join(_ROOT, "external", "SAMark", "samark_detect.py")
_SPEC = importlib.util.spec_from_file_location("samark_detect_bridge", _DETECT_PATH)
samark_detect = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(samark_detect)


class _FakeEmbedder:
    def encode(self, texts, convert_to_tensor=True):
        rows = [[float(i + 1), float(len(text)), 1.0] for i, text in enumerate(texts)]
        return torch.tensor(rows)


class SAMarkBridgeTests(unittest.TestCase):
    def test_raw_detector_does_not_make_a_binary_decision(self):
        original = samark_detect.get_random_vectors
        samark_detect.get_random_vectors = lambda dim, num, seed=42: torch.eye(dim)[:num]
        try:
            result = samark_detect.detect_sentence_level(
                ["one.", "a longer two."], _FakeEmbedder(), 2, transform="tanh", transform_k=30.0,
            )
        finally:
            samark_detect.get_random_vectors = original

        self.assertIn("z_score", result)
        self.assertNotIn("is_watermarked", result)
        self.assertNotIn("p_value", result)
        self.assertTrue(np.isfinite(result["z_score"]))

    def test_calibrated_cutoff_respects_empirical_fpr(self):
        null = np.arange(1024, dtype=float)
        threshold = samark_detect.calibrated_threshold(null, 0.01)
        self.assertLessEqual(float(np.mean(null > threshold)), 0.01)

    def test_standard_operating_points_include_one_and_five_percent(self):
        null = np.arange(1024, dtype=float)
        positive = np.arange(900, 1100, dtype=float)
        rows = samark_detect.summarize_operating_points(
            positive, "text", null, "c4-human-def",
        )
        self.assertEqual([row["fpr_target"] for row in rows], [0.01, 0.05])
        compact = samark_detect.compact_result(rows, "text")
        self.assertEqual(compact["fpr1"], rows[0]["tpr"])
        self.assertEqual(compact["fpr5"], rows[1]["tpr"])

    def test_custom_operating_point_is_additive(self):
        self.assertEqual(
            samark_detect.operating_points(0.10),
            (0.01, 0.05, 0.10),
        )

    def test_empty_document_is_a_raw_zero_score(self):
        result = samark_detect.detect_sentence_level([], _FakeEmbedder(), 2)
        self.assertEqual(result["z_score"], 0.0)
        self.assertNotIn("is_watermarked", result)


if __name__ == "__main__":
    unittest.main()
