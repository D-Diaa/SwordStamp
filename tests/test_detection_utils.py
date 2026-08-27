"""Test detection math with mocked GPU primitives."""
import unittest
from unittest.mock import MagicMock, patch
import json
import tempfile
import torch
from types import SimpleNamespace

from watermarking.detect import (
    _human_cache_matches,
    _write_human_cache_metadata,
    compute_zscore,
    count_kmeans_watermark_hits,
    count_lsh_watermark_hits,
    detect_lsh,
    detect_kmeans,
    truncate_to_generation_budget,
)
from segmentation import Segmenter, Unit



def _u(s: str) -> Unit:
    """Wrap a plain string as a sentence Unit for testing."""
    return Unit("sentence", s, s)


class _FakeLSHModel:
    """Returns predetermined hash ints; no embedder or GPU needed."""

    def __init__(self, hash_map: dict):
        self._map = hash_map

    def get_hash(self, sents, embeds=None):
        return [self._map[s] for s in sents]


def _cpu_mask(*values):
    """CPU tensor mask for use in patches (avoids CUDA dependency)."""
    return torch.tensor(list(values))



class TestComputeZscore(unittest.TestCase):

    def test_at_null_rate_is_zero(self):
        lmbd = 0.25
        n = 100
        z = compute_zscore(lmbd * n, n, lmbd)
        self.assertAlmostEqual(z, 0.0, places=9)

    def test_above_null_is_positive(self):
        z = compute_zscore(50, 100, 0.25)  # 50% > 25%
        self.assertGreater(z, 0)

    def test_below_null_is_negative(self):
        z = compute_zscore(10, 100, 0.25)  # 10% < 25%
        self.assertLess(z, 0)

    def test_all_accepted_gives_high_positive(self):
        n = 100
        z = compute_zscore(n, n, 0.25)
        self.assertGreater(z, 5.0)

    def test_none_accepted_gives_high_negative(self):
        z = compute_zscore(0, 100, 0.25)
        self.assertLess(z, -2.0)

    def test_symmetric_around_null(self):
        lmbd = 0.25
        n = 100
        k = 10  # deviation
        z_high = compute_zscore(lmbd * n + k, n, lmbd)
        z_low = compute_zscore(lmbd * n - k, n, lmbd)
        self.assertAlmostEqual(z_high, -z_low, places=9)

    def test_larger_n_more_significant(self):
        # Same fraction above lmbd, but larger n → higher z
        lmbd = 0.25
        z_small = compute_zscore(0.5 * 50, 50, lmbd)
        z_large = compute_zscore(0.5 * 500, 500, lmbd)
        self.assertGreater(z_large, z_small)


class TestSemcutDetectionContract(unittest.TestCase):

    def test_run_detection_uses_the_bound_segmenter(self):
        from watermarking.detect import run_detection

        segmenter = MagicMock(spec=Segmenter)
        segmenter.segment.side_effect = [[_u("a")], [_u("b")]]
        detect_fn = MagicMock(side_effect=[1.0, 2.0])
        scores = run_detection(
            detect_fn, ["first", "second"], "test", segmenter=segmenter,
        )
        self.assertEqual(scores, [1.0, 2.0])
        self.assertEqual(
            [call.args[0] for call in segmenter.segment.call_args_list],
            ["first", "second"],
        )

    def test_human_cache_identity_includes_semcut_encoder(self):
        encoder_a = Segmenter(
            "semspan", "nltk", encode=MagicMock(),
            encoder_id="defender/encoder-a",
        )
        encoder_b = Segmenter(
            "semspan", "nltk", encode=MagicMock(),
            encoder_id="defender/encoder-b",
        )
        with tempfile.TemporaryDirectory() as tmp:
            _write_human_cache_metadata(tmp, encoder_a)
            self.assertTrue(_human_cache_matches(tmp, encoder_a))
            self.assertFalse(_human_cache_matches(tmp, encoder_b))
            with open(
                f"{tmp}/human_z_scores.segmentation.json", encoding="utf-8",
            ) as f:
                self.assertEqual(json.load(f)["semantic_encoder"], "defender/encoder-a")

    def test_human_cache_without_segmentation_metadata_is_a_miss(self):
        segmenter = Segmenter("sentence", "nltk")
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(_human_cache_matches(tmp, segmenter))

    def test_invalid_human_cache_metadata_is_a_miss(self):
        segmenter = Segmenter("sentence", "nltk")
        with tempfile.TemporaryDirectory() as tmp:
            with open(
                f"{tmp}/human_z_scores.segmentation.json", "w", encoding="utf-8",
            ) as handle:
                handle.write("not json")
            self.assertFalse(_human_cache_matches(tmp, segmenter))

    def test_human_cache_identity_includes_every_semcut_policy_value(self):
        base = Segmenter(
            "semspan", "nltk", encode=MagicMock(),
            encoder_id="defender/encoder",
        )
        variants = (
            Segmenter(
                "semspan", "nltk", encode=MagicMock(),
                encoder_id="defender/encoder", semcut_max_words=14,
            ),
            Segmenter(
                "semspan", "nltk", encode=MagicMock(),
                encoder_id="defender/encoder", semcut_window=6,
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            _write_human_cache_metadata(tmp, base)
            self.assertTrue(_human_cache_matches(tmp, base))
            for variant in variants:
                with self.subTest(metadata=variant.metadata()):
                    self.assertFalse(_human_cache_matches(tmp, variant))

    def test_smoke_null_cache_identity_includes_semcut_policy(self):
        from config.loader import load_config
        from watermarking.__main__ import _null_cache_metadata

        cfg = load_config(overrides=["segmentation.type=semspan"])
        args = SimpleNamespace(null_n=1024)
        base = Segmenter(
            "semspan", "nltk", encode=MagicMock(),
            encoder_id=cfg.watermark.embedder,
        )
        metadata = _null_cache_metadata(
            cfg, args, "data/human", "lsh", base,
        )
        self.assertEqual(metadata["semcut_max_words"], 15)
        self.assertEqual(metadata["semcut_window"], 5)

        changed = Segmenter(
            "semspan", "nltk", encode=MagicMock(),
            encoder_id=cfg.watermark.embedder, semcut_window=6,
        )
        self.assertNotEqual(
            metadata,
            _null_cache_metadata(cfg, args, "data/human", "lsh", changed),
        )


class TestSmokeDetectorModes(unittest.TestCase):

    def test_lsh_fixed_diverse_uses_fixed_lsh_detector(self):
        from config.loader import load_config
        from watermarking.__main__ import _build_detector

        cfg = load_config(overrides=["watermark.mode=lsh_fixed_diverse"])
        segmenter = MagicMock(spec=Segmenter)
        segmenter.segment.return_value = [_u("unit")]
        lsh_model = SimpleNamespace(embedder=object())

        with patch(
            "watermarking.__main__.SBERTLSHModel", return_value=lsh_model,
        ), patch(
            "watermarking.__main__.Segmenter", return_value=segmenter,
        ), patch(
            "watermarking.__main__.count_lsh_watermark_hits",
            return_value=(1, 1),
        ) as count_hits:
            detect_stats, mode, returned_segmenter = _build_detector(cfg, "cpu")
            stats = detect_stats("text", cfg.watermark.hash_key)

        self.assertEqual(mode, "lsh_fixed_diverse")
        self.assertIs(returned_segmenter, segmenter)
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(
            count_hits.call_args.kwargs["secret_message"],
            cfg.watermark.secret_message,
        )

    def test_kmeans_fixed_diverse_uses_fixed_kmeans_detector(self):
        from config.loader import load_config
        from watermarking.__main__ import _build_detector

        cfg = load_config(overrides=[
            "watermark.mode=kmeans_fixed_diverse",
            "watermark.cc_path=/tmp/centers.pt",
        ])
        segmenter = MagicMock(spec=Segmenter)
        segmenter.segment.return_value = [_u("unit")]
        embedder = object()

        with patch(
            "watermarking.__main__.torch.load", return_value=object(),
        ), patch(
            "watermarking.__main__.SentenceTransformer", return_value=embedder,
        ), patch(
            "watermarking.__main__.Segmenter", return_value=segmenter,
        ), patch(
            "watermarking.__main__.count_kmeans_watermark_hits",
            return_value=(1, 1),
        ) as count_hits:
            detect_stats, mode, returned_segmenter = _build_detector(cfg, "cpu")
            stats = detect_stats("text", cfg.watermark.hash_key)

        self.assertEqual(mode, "kmeans_fixed_diverse")
        self.assertIs(returned_segmenter, segmenter)
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(
            count_hits.call_args.kwargs["secret_message"],
            cfg.watermark.secret_message,
        )



class TestDetectLSH(unittest.TestCase):

    def _detect(self, sents, hash_map, mask_values, secret_message=None, key=None):
        """Run detect_lsh with mocked get_mask_from_seed."""
        lsh_model = _FakeLSHModel(hash_map)
        mask = _cpu_mask(*mask_values)
        kwargs = {}
        if key is not None:
            kwargs["key"] = key
        units = [_u(s) for s in sents]
        with patch("watermarking.detect.get_mask_from_seed", return_value=mask):
            return detect_lsh(
                units, lsh_model, lmbd=0.25, lsh_dim=2,
                secret_message=secret_message, **kwargs
            )

    def test_all_accepted_context_dependent(self):
        # Every sentence hashes to a value that is in the mask.
        sents = ["s0", "s1", "s2", "s3", "s4"]
        hash_map = {s: 7 for s in sents}  # all hash to 7
        z = self._detect(sents, hash_map, mask_values=[7])
        expected_z = compute_zscore(4, 4, 0.25)  # s1-s4 checked
        self.assertAlmostEqual(z, expected_z, places=9)

    def test_none_accepted_context_dependent(self):
        sents = ["s0", "s1", "s2", "s3"]
        hash_map = {"s0": 9, "s1": 1, "s2": 2, "s3": 3}
        # mask only accepts 9; sentences s1-s3 hash to 1,2,3 → rejected
        z = self._detect(sents, hash_map, mask_values=[9])
        expected_z = compute_zscore(0, 3, 0.25)
        self.assertAlmostEqual(z, expected_z, places=9)

    def test_mixed_acceptance(self):
        # Alternating: s1 accepted, s2 rejected, s3 accepted
        sents = ["s0", "s1", "s2", "s3"]
        hash_map = {"s0": 0, "s1": 5, "s2": 1, "s3": 5}
        z = self._detect(sents, hash_map, mask_values=[5])
        expected_z = compute_zscore(2, 3, 0.25)
        self.assertAlmostEqual(z, expected_z, places=9)

    def test_context_mode_uses_unit_zero_only_as_preceding_seed(self):
        units = [_u("s0"), _u("s1"), _u("s2")]
        lsh_model = _FakeLSHModel({"s0": 0, "s1": 1, "s2": 2})

        def mask_for_seed(_dim, _lmbd, seed, **_kwargs):
            return _cpu_mask(seed + 1)

        with patch(
            "watermarking.detect.get_mask_from_seed", side_effect=mask_for_seed,
        ) as mock_mask:
            hits, total = count_lsh_watermark_hits(
                units, lsh_model, lmbd=0.25, lsh_dim=2,
            )

        self.assertEqual((hits, total), (2, 2))
        self.assertEqual(
            [call.args[2] for call in mock_mask.call_args_list],
            [0, 1, 2],
        )

    def test_single_sentence_gives_nan_or_zero(self):
        # No tested units remain after the context seed, so z is undefined.
        import math
        sents = ["s0"]
        hash_map = {"s0": 5}
        z = self._detect(sents, hash_map, mask_values=[5])
        self.assertTrue(math.isnan(z) or isinstance(z, float))

    # Fixed mode.

    def test_fixed_mode_all_accepted(self):
        secret = "the magic words"
        sents = ["s0", "s1", "s2"]
        hash_map = {secret: 42, "s0": 42, "s1": 42, "s2": 42}
        z = self._detect(sents, hash_map, mask_values=[42], secret_message=secret)
        expected_z = compute_zscore(3, 3, 0.25)
        self.assertAlmostEqual(z, expected_z, places=9)

    def test_fixed_mode_permutation_preserves_hits_and_score(self):
        secret = "secret"
        units = [_u("green-a"), _u("red"), _u("green-b")]
        lsh_model = _FakeLSHModel({
            secret: 99,
            "green-a": 7,
            "red": 3,
            "green-b": 7,
        })
        with patch(
            "watermarking.detect.get_mask_from_seed", return_value=_cpu_mask(7),
        ):
            original = count_lsh_watermark_hits(
                units, lsh_model, 0.25, 2, secret_message=secret,
            )
            permuted = count_lsh_watermark_hits(
                [units[2], units[0], units[1]],
                lsh_model,
                0.25,
                2,
                secret_message=secret,
            )
            original_z = detect_lsh(
                units, lsh_model, 0.25, 2, secret_message=secret,
            )
            permuted_z = detect_lsh(
                [units[2], units[0], units[1]],
                lsh_model,
                0.25,
                2,
                secret_message=secret,
            )

        self.assertEqual(original, (2, 3))
        self.assertEqual(permuted, original)
        self.assertAlmostEqual(permuted_z, original_z, places=9)

    def test_fixed_mode_uses_secret_seed_not_s0_seed(self):
        """Require fixed mode to seed masks from the secret."""
        secret = "secret"
        units = [_u("different_s0"), _u("s1")]
        # secret hashes to 99; "different_s0" hashes to 0
        lsh_model = _FakeLSHModel({secret: 99, "different_s0": 0, "s1": 0})
        mask = _cpu_mask()
        with patch("watermarking.detect.get_mask_from_seed", return_value=mask) as mock_fn:
            detect_lsh(units, lsh_model, lmbd=0.25, lsh_dim=2, secret_message=secret)
        # First positional arg to call is lsh_dim, second is lmbd, third is the seed
        actual_seed = mock_fn.call_args[0][2]
        self.assertEqual(actual_seed, 99)



class TestDetectKMeans(unittest.TestCase):

    def _cluster_id(self, val):
        return torch.tensor(val)

    def _detect(self, sents, cluster_id_map, mask_values, secret_message=None):
        embedder = object()  # never called; patched away
        cluster_centers = object()
        mask = _cpu_mask(*mask_values)

        def fake_get_cluster_id(text, **kwargs):
            return self._cluster_id(cluster_id_map[text])

        units = [_u(s) for s in sents]
        with patch("watermarking.detect.get_cluster_id", side_effect=fake_get_cluster_id), \
             patch("watermarking.detect.get_cluster_mask", return_value=mask):
            return detect_kmeans(
                units, embedder, lmbd=0.25, k_dim=4,
                cluster_centers=cluster_centers, secret_message=secret_message,
            )

    def test_all_accepted(self):
        sents = ["s0", "s1", "s2", "s3"]
        cluster_id_map = {s: 2 for s in sents}
        z = self._detect(sents, cluster_id_map, mask_values=[2])
        expected_z = compute_zscore(3, 3, 0.25)
        self.assertAlmostEqual(z, expected_z, places=9)

    def test_none_accepted(self):
        sents = ["s0", "s1", "s2"]
        cluster_id_map = {"s0": 0, "s1": 1, "s2": 1}
        z = self._detect(sents, cluster_id_map, mask_values=[0])
        # mask accepts cluster 0; s1 and s2 are cluster 1 → not counted
        expected_z = compute_zscore(0, 2, 0.25)
        self.assertAlmostEqual(z, expected_z, places=9)

    def test_context_mode_uses_unit_zero_only_as_preceding_seed(self):
        units = [_u("s0"), _u("s1"), _u("s2")]
        cluster_id_map = {"s0": 0, "s1": 1, "s2": 2}

        def fake_get_cluster_id(text, **_kwargs):
            return self._cluster_id(cluster_id_map[text])

        def mask_for_cluster(cluster_id, _k_dim, _lmbd, **_kwargs):
            return _cpu_mask(int(cluster_id) + 1)

        with patch(
            "watermarking.detect.get_cluster_id",
            side_effect=fake_get_cluster_id,
        ), patch(
            "watermarking.detect.get_cluster_mask",
            side_effect=mask_for_cluster,
        ) as mock_mask:
            hits, total = count_kmeans_watermark_hits(
                units,
                object(),
                lmbd=0.25,
                k_dim=4,
                cluster_centers=object(),
            )

        self.assertEqual((hits, total), (2, 2))
        self.assertEqual(
            [int(call.args[0]) for call in mock_mask.call_args_list],
            [0, 1, 2],
        )

    def test_fixed_mode_permutation_preserves_hits_and_score(self):
        secret = "secret"
        units = [_u("green-a"), _u("red"), _u("green-b")]
        cluster_id_map = {
            secret: 9,
            "green-a": 2,
            "red": 1,
            "green-b": 2,
        }

        def fake_get_cluster_id(text, **_kwargs):
            return self._cluster_id(cluster_id_map[text])

        with patch(
            "watermarking.detect.get_cluster_id",
            side_effect=fake_get_cluster_id,
        ), patch(
            "watermarking.detect.get_cluster_mask", return_value=_cpu_mask(2),
        ):
            original = count_kmeans_watermark_hits(
                units,
                object(),
                0.25,
                4,
                object(),
                secret_message=secret,
            )
            permuted = count_kmeans_watermark_hits(
                [units[2], units[0], units[1]],
                object(),
                0.25,
                4,
                object(),
                secret_message=secret,
            )
            original_z = detect_kmeans(
                units,
                object(),
                0.25,
                4,
                object(),
                secret_message=secret,
            )
            permuted_z = detect_kmeans(
                [units[2], units[0], units[1]],
                object(),
                0.25,
                4,
                object(),
                secret_message=secret,
            )

        self.assertEqual(original, (2, 3))
        self.assertEqual(permuted, original)
        self.assertAlmostEqual(permuted_z, original_z, places=9)


class _WordTokenizer:
    """Count whitespace words; enough to exercise the budget arithmetic."""

    def encode(self, text):
        return text.split()


class TestTruncateToGenerationBudget(unittest.TestCase):
    """Human-null truncation must mirror generation's stopping rule."""

    def setUp(self):
        self.seg = Segmenter("sentence", "nltk")
        self.tok = _WordTokenizer()

    def _truncate(self, text, budget):
        return truncate_to_generation_budget(
            text, self.seg, self.seg, self.tok, budget,
        )

    def test_keeps_the_unit_that_crosses_the_budget(self):
        # Two words per sentence past the seed: the budget is crossed inside
        # the third one, which generation would still have emitted in full.
        text = "Seed sentence. Aa bb. Cc dd. Ee ff. Gg hh."
        self.assertEqual(self._truncate(text, 5), "Seed sentence. Aa bb. Cc dd. Ee ff.")

    def test_seed_sentence_is_outside_the_budget(self):
        # A long seed must not consume any of the budget, exactly as the
        # generation prompt does not count against max_new_tokens.
        short = "Seed. Aa bb. Cc dd."
        long_seed = "One two three four five six seven. Aa bb. Cc dd."
        for text in (short, long_seed):
            seed_words = len(text.split(".")[0].split())
            kept = len(self._truncate(text, 2).split())
            self.assertEqual(kept - seed_words, 2)

    def test_documents_under_budget_are_kept_whole(self):
        text = "Seed sentence. Only one more."
        self.assertEqual(self._truncate(text, 256), text)

    def test_detector_units_are_a_prefix_of_the_originals(self):
        text = "Seed sentence. Aa bb. Cc dd. Ee ff. Gg hh."
        truncated = self._truncate(text, 5)
        full_units = [u.normalized for u in self.seg.segment(text)]
        kept_units = [u.normalized for u in self.seg.segment(truncated)]
        self.assertLess(len(kept_units), len(full_units))
        self.assertEqual(full_units[:len(kept_units)], kept_units)

    def test_unsegmentable_text_is_returned_unchanged(self):
        self.assertEqual(self._truncate("", 256), "")
        self.assertEqual(self._truncate("   ", 256), "   ")

    def test_single_sentence_has_no_scored_units(self):
        self.assertEqual(self._truncate("Only one sentence here.", 256),
                         "Only one sentence here.")


if __name__ == "__main__":
    unittest.main()
