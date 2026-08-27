"""CPU tests for generation-side watermark score functions."""

import unittest
from dataclasses import replace
from unittest.mock import MagicMock, patch, sentinel

import numpy as np
import torch

from sampling import CandidateScore, Region, score_accepts
from segmentation import Unit
from watermarking.scoring import (
    create_kmeans_score_fn,
    create_lsh_score_fn,
    create_none_score_fn,
)
from config.schema import AppConfig


def _unit(text: str, display: str | None = None) -> Unit:
    return Unit("sentence", text, display or text)


def _predecessor(text: str) -> Unit:
    return Unit("sentence", text.lower(), text)


class _FakeLSH:
    def __init__(self, hash_map):
        self._map = hash_map

    def get_hash(self, texts, embeds=None):
        return [self._map.get(text, -1) for text in texts]

    def get_embeddings(self, texts):
        return np.zeros((len(texts), 4), dtype=np.float32)


def _lsh_scores(hash_map, mask_values, depths, candidates, *, lmbd=0.25):
    model = _FakeLSH(hash_map)
    with patch("watermarking.scoring.get_mask_from_seed", return_value=torch.tensor(mask_values)):
        fn = create_lsh_score_fn(
            model, lsh_dim=2, lmbd=lmbd, fixed_seed=999,
        )
    with patch(
        "watermarking.scoring.compute_lsh_margins",
        return_value=torch.tensor(depths, dtype=torch.float32),
    ):
        return fn(Unit(), candidates, 0)


def _kmeans_scores(cluster_map, mask_values, candidates, *, lmbd=0.25):
    embedder = MagicMock()
    cluster_centers = object()

    def fake_margins(texts, _embedder, _centers):
        margins = torch.tensor([cluster_map[text][1] for text in texts], dtype=torch.float32)
        ids = torch.tensor([cluster_map[text][0] for text in texts])
        return margins, ids

    with patch("watermarking.scoring.get_cluster_mask", return_value=torch.tensor(mask_values)), \
         patch("watermarking.scoring.get_cluster_id", return_value=torch.tensor(0)):
        fn = create_kmeans_score_fn(
            embedder, cluster_centers, k_dim=4, lmbd=lmbd,
            fixed_cluster_id=torch.tensor(0),
        )
    with patch("watermarking.scoring.compute_kmeans_margins", side_effect=fake_margins):
        return fn(Unit(), candidates, 0)


class TestCandidateScore(unittest.TestCase):
    def test_region_and_depth_are_explicit(self):
        scores = _lsh_scores(
            {"green": 5, "red": 9}, [5], [0.4, 0.3],
            [_unit("green"), _unit("red")],
        )
        self.assertTrue(all(isinstance(score, CandidateScore) for score in scores))
        self.assertEqual(scores[0].region, Region.GREEN)
        self.assertEqual(scores[1].region, Region.RED)
        self.assertAlmostEqual(scores[0].depth, 0.4, places=6)
        self.assertAlmostEqual(scores[1].depth, 0.3, places=6)

    def test_score_uses_normalized_not_display(self):
        scores = _lsh_scores(
            {"normalized": 5}, [5], [0.2],
            [_unit("normalized", display="Different display")],
        )
        self.assertEqual(scores[0].region, Region.GREEN)

    def test_depth_is_not_margin_adjusted(self):
        scores = _lsh_scores(
            {"deep": 5, "shallow": 5, "red": 9}, [5], [0.5, 0.2, 0.1],
            [_unit("deep"), _unit("shallow"), _unit("red")],
        )
        self.assertEqual([score.region for score in scores], [
            Region.GREEN, Region.GREEN, Region.RED,
        ])
        self.assertEqual([round(score.depth, 6) for score in scores], [0.5, 0.2, 0.1])

    def test_boundary_depth_orders_candidates_with_same_color(self):
        green_scores = _lsh_scores(
            {"deep": 5, "shallow": 5}, [5], [0.6, 0.4],
            [_unit("deep"), _unit("shallow")],
        )
        red_scores = _lsh_scores(
            {"far": 9, "near": 9}, [5], [0.6, 0.2],
            [_unit("far"), _unit("near")],
        )
        self.assertGreater(green_scores[0].rank_key(), green_scores[1].rank_key())
        self.assertGreater(red_scores[1].rank_key(), red_scores[0].rank_key())

    def test_lmbd_changes_mask_not_score_scale(self):
        low = _lsh_scores({"green": 5}, [5], [0.4], [_unit("green")], lmbd=0.1)
        high = _lsh_scores({"green": 5}, [5], [0.4], [_unit("green")], lmbd=0.9)
        self.assertEqual(low, high)

    def test_kmeans_uses_same_structured_contract(self):
        scores = _kmeans_scores(
            {"green": (2, 0.5), "red": (3, 0.2)}, [2],
            [_unit("green"), _unit("red")],
        )
        self.assertEqual(scores[0].region, Region.GREEN)
        self.assertEqual(scores[1].region, Region.RED)
        self.assertAlmostEqual(scores[0].depth, 0.5)
        self.assertAlmostEqual(scores[1].depth, 0.2)


class TestPredecessorDrivenContext(unittest.TestCase):

    def test_none_scorer_conforms_and_ignores_predecessor(self):
        candidates = [_unit("one"), _unit("two")]
        self.assertEqual(
            create_none_score_fn()(_predecessor("anything"), candidates, 9),
            [CandidateScore(Region.GREEN, 1.0)] * 2,
        )

    def test_lsh_context_mask_uses_supplied_predecessor_unit(self):
        model = _FakeLSH({"prompt predecessor.": 3, "green": 5, "red": 7})
        with patch(
            "watermarking.scoring.get_mask_from_seed",
            return_value=torch.tensor([5]),
        ) as get_mask:
            fn = create_lsh_score_fn(model, lsh_dim=3, lmbd=0.25)
            with patch(
                "watermarking.scoring.compute_lsh_margins",
                return_value=torch.tensor([0.4, 0.3]),
            ):
                scores = fn(
                    _predecessor("Prompt predecessor."),
                    [_unit("green"), _unit("red")],
                    0,
                )

        self.assertEqual(scores[0].region, Region.GREEN)
        self.assertEqual(scores[1].region, Region.RED)
        self.assertEqual(get_mask.call_args.args[2], 3)

    def test_kmeans_context_mask_uses_supplied_predecessor_unit(self):
        embedder = MagicMock()
        centers = object()

        def fake_margins(texts, _embedder, _centers):
            return torch.tensor([0.4, 0.3]), torch.tensor([2, 3])

        with patch(
            "watermarking.scoring.get_cluster_id", return_value=torch.tensor(1),
        ) as get_id, patch(
            "watermarking.scoring.get_cluster_mask", return_value=torch.tensor([2]),
        ) as get_mask, patch(
            "watermarking.scoring.compute_kmeans_margins",
            side_effect=fake_margins,
        ):
            fn = create_kmeans_score_fn(embedder, centers, k_dim=4, lmbd=0.25)
            scores = fn(
                _predecessor("Prompt predecessor."),
                [_unit("green"), _unit("red")],
                0,
            )

        self.assertEqual(scores[0].region, Region.GREEN)
        self.assertEqual(scores[1].region, Region.RED)
        get_id.assert_called_once_with("prompt predecessor.", centers, embedder)
        self.assertEqual(int(get_mask.call_args.args[0]), 1)


class TestGenerationConfigWiring(unittest.TestCase):
    def test_do_sample_reaches_transformers_generation_config(self):
        from watermarking.generate import generate_dataset

        sampler = MagicMock()
        sampler.tokenizer.pad_token_id = 0
        sampler.generate_batched_continuation.return_value = [(
            "generated",
            {
                "accepted_count": 0,
                "unit_count": 0,
                "units": [],
                "n_accepted_candidates_per_unit": [],
                "n_candidates_per_unit": [],
            },
        )]
        cfg = AppConfig()
        cfg = replace(
            cfg,
            watermark=replace(cfg.watermark, mode="none"),
            generation=replace(cfg.generation, backend="hf", do_sample=False),
        )
        with patch("watermarking.generate.create_sampler", return_value=sampler):
            generate_dataset({"text": ["A sufficiently long source sentence for a prompt."]}, cfg, "cpu", verbose=False)
        gen_config = sampler.generate_batched_continuation.call_args.args[1]
        self.assertFalse(gen_config.do_sample)
        self.assertEqual(
            sampler.generate_batched_continuation.call_args.kwargs["margin"], 0.0,
        )

    def test_semcut_reuses_the_defender_watermark_encoder(self):
        from watermarking.generate import generate_dataset

        sampler = MagicMock()
        sampler.tokenizer.pad_token_id = 0
        sampler.generate_batched_continuation.return_value = [(
            "generated",
            {
                "accepted_count": 0,
                "unit_count": 0,
                "units": [],
                "n_accepted_candidates_per_unit": [],
                "n_candidates_per_unit": [],
            },
        )]
        factory = MagicMock()
        factory.encoder = sentinel.watermark_encoder
        factory.return_value = sentinel.score_fn
        cfg = AppConfig()
        cfg = replace(
            cfg,
            watermark=replace(cfg.watermark, mode="lsh_fixed"),
            generation=replace(cfg.generation, backend="hf"),
            segmentation=replace(cfg.segmentation, type="semspan"),
        )
        with patch("watermarking.generate.setup_score_factory", return_value=factory), \
             patch("watermarking.generate.Segmenter.from_config",
                   return_value=sentinel.segmenter) as bind, \
             patch("watermarking.generate.create_sampler", return_value=sampler) as create:
            generate_dataset(
                {"text": ["A sufficiently long source sentence for a prompt."]},
                cfg, "cpu", verbose=False,
            )

        bind.assert_called_once_with(
            cfg.segmentation,
            encoder=sentinel.watermark_encoder,
            encoder_id=cfg.watermark.embedder,
            batch_size=cfg.runtime.semcut_batch_size,
        )
        self.assertIs(create.call_args.kwargs["segmenter"], sentinel.segmenter)
        self.assertEqual(
            sampler.generate_batched_continuation.call_args.kwargs["margin"],
            cfg.watermark.margin,
        )
        factory.assert_called_once_with()


class TestFixedDiverse(unittest.TestCase):
    """The *_fixed_diverse modes exclude the previous unit's region from green."""

    MASK = [3, 5]

    def _scores(self, context, candidates, *, diverse):
        model = _FakeLSH({
            "the previous unit.": 3,
            "an outside unit.": 7,
            "copy of prev": 3,
            "fresh region": 5,
        })
        with patch("watermarking.scoring.get_mask_from_seed",
                   return_value=torch.tensor(self.MASK)):
            fn = create_lsh_score_fn(
                model, lsh_dim=2, lmbd=0.25, fixed_seed=999, diverse=diverse,
            )
        with patch("watermarking.scoring.compute_lsh_margins",
                   return_value=torch.tensor([0.9] * len(candidates))):
            return fn(_predecessor(context), candidates, 0)

    def test_copy_of_previous_unit_is_yellow(self):
        scores = self._scores(
            "The previous unit.",
            [_unit("copy of prev"), _unit("fresh region")],
            diverse=True,
        )
        self.assertEqual(scores[0].region, Region.YELLOW)
        self.assertEqual(scores[1].region, Region.GREEN)
        self.assertAlmostEqual(scores[0].depth, 0.9)

    def test_yellow_outranks_red_but_loses_to_green(self):
        """A yellow unit is a detection hit, so it beats an out-of-mask miss."""
        model = _FakeLSH({
            "the previous unit.": 3,
            "copy of prev": 3,      # in mask, predecessor's region -> yellow
            "fresh region": 5,      # in mask, different region     -> green
            "outside": 7,           # not in mask                   -> red
        })
        with patch("watermarking.scoring.get_mask_from_seed",
                   return_value=torch.tensor(self.MASK)):
            fn = create_lsh_score_fn(
                model, lsh_dim=2, lmbd=0.25, fixed_seed=999, diverse=True,
            )
        # A deep red must still lose to the yellow candidate.
        with patch("watermarking.scoring.compute_lsh_margins",
                   return_value=torch.tensor([0.2, 0.9, 0.9])):
            green, yellow, red = fn(
                _predecessor("The previous unit."),
                [_unit("fresh region"), _unit("copy of prev"), _unit("outside")],
                0,
            )
        self.assertGreater(green.rank_key(), yellow.rank_key())
        self.assertGreater(yellow.rank_key(), red.rank_key())

    def test_yellow_is_not_accepted_by_the_sampler(self):
        """Rejection mode never accepts the predecessor's yellow region."""
        scores = self._scores(
            "The previous unit.",
            [_unit("copy of prev"), _unit("fresh region")],
            diverse=True,
        )
        self.assertFalse(score_accepts(scores[0]))
        self.assertTrue(score_accepts(scores[1]))

    def test_plain_fixed_accepts_the_copy(self):
        scores = self._scores(
            "The previous unit.",
            [_unit("copy of prev"), _unit("fresh region")],
            diverse=False,
        )
        self.assertEqual([score.region for score in scores], [Region.GREEN] * 2)

    def test_mask_intact_when_previous_region_not_green(self):
        scores = self._scores(
            "An outside unit.",
            [_unit("copy of prev"), _unit("fresh region")],
            diverse=True,
        )
        self.assertEqual([score.region for score in scores], [Region.GREEN] * 2)

    def test_red_predecessor_does_not_promote_same_region_to_yellow(self):
        model = _FakeLSH({
            "red predecessor.": 7,
            "same red region": 7,
            "fresh green": 5,
        })
        with patch(
            "watermarking.scoring.get_mask_from_seed",
            return_value=torch.tensor(self.MASK),
        ):
            fn = create_lsh_score_fn(
                model, lsh_dim=3, lmbd=0.25, fixed_seed=999, diverse=True,
            )
        with patch(
            "watermarking.scoring.compute_lsh_margins",
            return_value=torch.tensor([0.9, 0.9]),
        ):
            same_red, fresh_green = fn(
                _predecessor("Red predecessor."),
                [_unit("same red region"), _unit("fresh green")],
                1,
            )

        self.assertEqual(same_red.region, Region.RED)
        self.assertEqual(fresh_green.region, Region.GREEN)

    def test_margin_failed_in_mask_fallback_keeps_next_yellow_nonaccepting(self):
        """A shallow green fallback may define yellow, but yellow stays rejected."""
        model = _FakeLSH({
            "outside prompt.": 7,
            "shallow in-mask": 3,
            "same in-mask region": 3,
        })
        with patch(
            "watermarking.scoring.get_mask_from_seed",
            return_value=torch.tensor(self.MASK),
        ):
            fn = create_lsh_score_fn(
                model,
                lsh_dim=3,
                lmbd=0.25,
                fixed_seed=999,
                diverse=True,
            )

        with patch(
            "watermarking.scoring.compute_lsh_margins",
            return_value=torch.tensor([0.1]),
        ):
            fallback_score = fn(
                _predecessor("Outside prompt."), [_unit("shallow in-mask")], 0,
            )[0]
        with patch(
            "watermarking.scoring.compute_lsh_margins",
            return_value=torch.tensor([0.9]),
        ):
            next_score = fn(
                _unit("shallow in-mask"), [_unit("same in-mask region")], 1,
            )[0]

        self.assertEqual(fallback_score.region, Region.GREEN)
        self.assertAlmostEqual(fallback_score.depth, 0.1)
        self.assertFalse(score_accepts(fallback_score, margin=0.2))
        self.assertEqual(next_score.region, Region.YELLOW)
        self.assertAlmostEqual(next_score.depth, 0.9)
        self.assertFalse(score_accepts(next_score, margin=0.2))

    def test_kmeans_diverse_excludes_previous_cluster(self):
        embedder = MagicMock()

        def fake_margins(texts, _embedder, _centers):
            ids = {"copy of prev": 3, "fresh region": 5}
            return (torch.tensor([0.9] * len(texts)),
                    torch.tensor([ids[t] for t in texts]))

        # get_cluster_id stays patched through the call: the diverse branch
        # resolves the previous unit's cluster per step, not at construction.
        with patch("watermarking.scoring.get_cluster_mask",
                   return_value=torch.tensor(self.MASK)), \
             patch("watermarking.scoring.get_cluster_id", return_value=3):
            fn = create_kmeans_score_fn(
                embedder, object(), k_dim=4, lmbd=0.25,
                fixed_cluster_id=0, diverse=True,
            )
            with patch("watermarking.scoring.compute_kmeans_margins",
                       side_effect=fake_margins):
                scores = fn(_predecessor("The previous unit."),
                            [_unit("copy of prev"), _unit("fresh region")], 0)
        self.assertEqual(scores[0].region, Region.YELLOW)
        self.assertEqual(scores[1].region, Region.GREEN)


if __name__ == "__main__":
    unittest.main()
