"""Tests for segmentation-sensitive, current-schema embedding caches."""

import pickle
import tempfile
import unittest
from unittest.mock import patch

import torch

from watermarking.primitives import embedding_cache_path, load_embeds


class EmbeddingCacheIdentityTest(unittest.TestCase):
    def test_non_semcut_cache_name_is_unchanged(self):
        self.assertEqual(
            embedding_cache_path(
                "data/corpus", "org/encoder", "sentence", "nltk",
            ),
            "data/corpus/embeds_sentence_nltk_encoder.pkl",
        )

    def test_every_semcut_policy_value_separates_the_cache(self):
        base = embedding_cache_path(
            "data/corpus", "org/encoder", "semspan", "spacy",
        )
        self.assertIn("semspan_spacy_max15_win5", base)
        variants = {
            embedding_cache_path(
                "data/corpus", "org/encoder", "semspan", "spacy", 14, 5,
            ),
            embedding_cache_path(
                "data/corpus", "org/encoder", "semspan", "spacy", 15, 4,
            ),
        }
        self.assertNotIn(base, variants)
        self.assertEqual(len(variants), 2)
        self.assertEqual(
            base,
            embedding_cache_path(
                "data/corpus", "org/encoder", "semspan", "spacy",
            ),
        )


class EmbeddingCacheSchemaTest(unittest.TestCase):
    def _write(self, directory, payload):
        path = f"{directory}/embeds.pkl"
        with open(path, "wb") as handle:
            pickle.dump(payload, handle)
        return path

    def test_current_tensor_schema_loads(self):
        expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        with tempfile.TemporaryDirectory() as tmp, patch(
            "watermarking.primitives.device", "cpu",
        ):
            actual = load_embeds(self._write(tmp, {"text": expected}))
        self.assertTrue(torch.equal(actual, expected))

    def test_tensor_list_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, patch(
            "watermarking.primitives.device", "cpu",
        ):
            path = self._write(tmp, {"text": [torch.tensor([1.0, 2.0])]})
            with self.assertRaisesRegex(TypeError, "must be a torch.Tensor"):
                load_embeds(path)

    def test_noncurrent_mapping_schema_is_rejected(self):
        payloads = (
            torch.tensor([[1.0]]),
            {"embeddings": torch.tensor([[1.0]])},
            {"text": torch.tensor([[1.0]]), "metadata": {}},
        )
        with tempfile.TemporaryDirectory() as tmp, patch(
            "watermarking.primitives.device", "cpu",
        ):
            for payload in payloads:
                with self.subTest(payload=payload):
                    path = self._write(tmp, payload)
                    with self.assertRaisesRegex(ValueError, "current"):
                        load_embeds(path)


if __name__ == "__main__":
    unittest.main()
