"""CPU tests for the adaptive attack's single-surrogate displacement scorer."""

import unittest
from unittest.mock import patch, sentinel

import torch

from attacks.paraphrasing.adaptive import SurrogateScorer


class _FakeEncoder:
    """Return preloaded embedding tensors in call order."""

    def __init__(self, embeddings_per_call):
        self._embeddings = iter(embeddings_per_call)

    def encode(self, sentences, convert_to_tensor=True, normalize_embeddings=True):
        return next(self._embeddings)


def _make_scorer(encoder_outputs):
    scorer = SurrogateScorer.__new__(SurrogateScorer)
    scorer.encoder = _FakeEncoder(encoder_outputs)
    return scorer


class TestConstruction(unittest.TestCase):

    def test_loads_one_named_encoder(self):
        with patch(
            "attacks.paraphrasing.adaptive.SentenceTransformer",
            return_value=sentinel.encoder,
        ) as load:
            scorer = SurrogateScorer("example/surrogate", device="cpu")
        load.assert_called_once_with("example/surrogate", device="cpu")
        self.assertIs(scorer.encoder, sentinel.encoder)

    def test_semcut_segmenter_reuses_the_surrogate_encoder(self):
        scorer = SurrogateScorer.__new__(SurrogateScorer)
        scorer.encoder = sentinel.encoder
        with patch(
            "attacks.paraphrasing.adaptive.Segmenter.from_sentence_transformer",
            return_value=sentinel.segmenter,
        ) as bind:
            got = scorer.segmenter("semspan", "nltk", "example/surrogate")
        self.assertIs(got, sentinel.segmenter)
        bind.assert_called_once_with(
            "semspan", "nltk", sentinel.encoder, "example/surrogate",
            batch_size=512,
            semcut_max_words=15,
            semcut_window=5,
        )


class TestPositionalDisplacement(unittest.TestCase):

    def _scores(self, anchors, candidates, anchor_index):
        scorer = _make_scorer([anchors, candidates])
        anchor_embs = scorer.encode_anchors(["a"] * len(anchors))
        texts = [f"c{i}" for i in range(len(candidates))]
        return scorer.score_displacement(
            anchor_embs, texts, anchor_index=anchor_index,
        )

    def test_identical_candidate_scores_near_zero(self):
        scores = self._scores(
            torch.tensor([[1.0, 0.0]]), torch.tensor([[1.0, 0.0]]), 0,
        )
        self.assertAlmostEqual(scores[0], 0.0, places=4)

    def test_orthogonal_candidate_scores_near_one(self):
        scores = self._scores(
            torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]]), 0,
        )
        self.assertAlmostEqual(scores[0], 1.0, places=4)

    def test_opposite_candidate_scores_near_two(self):
        scores = self._scores(
            torch.tensor([[1.0, 0.0]]), torch.tensor([[-1.0, 0.0]]), 0,
        )
        self.assertAlmostEqual(scores[0], 2.0, places=4)

    def test_anchor_index_selects_correct_anchor(self):
        anchors = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ])
        scores = self._scores(anchors, torch.tensor([[0.0, 1.0]]), 1)
        self.assertAlmostEqual(scores[0], 0.0, places=4)

    def test_empty_candidates_returns_empty(self):
        scorer = _make_scorer([torch.tensor([[1.0, 0.0]])])
        anchor_embs = scorer.encode_anchors(["a"])
        self.assertEqual(
            scorer.score_displacement(anchor_embs, [], anchor_index=0), [],
        )

    def test_multiple_candidates_scored_independently(self):
        anchors = torch.tensor([[1.0, 0.0]])
        candidates = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ])
        scores = self._scores(anchors, candidates, 0)
        self.assertEqual(len(scores), 3)
        self.assertAlmostEqual(scores[0], 0.0, places=4)
        self.assertAlmostEqual(scores[1], 1.0, places=4)
        self.assertAlmostEqual(scores[2], 2.0, places=4)


class TestBagDisplacement(unittest.TestCase):

    def test_min_bag_agg_picks_nearest_neighbor_distance(self):
        anchors = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        candidates = torch.tensor([[1.0, 0.0], [-0.7071, -0.7071]])
        scorer = _make_scorer([anchors, candidates])
        anchor_embs = scorer.encode_anchors(["a", "b"])
        scores = scorer.score_displacement(anchor_embs, ["c1", "c2"], bag_agg="min")
        self.assertAlmostEqual(scores[0], 0.0, places=4)
        self.assertGreater(scores[1], 1.0)

    def test_mean_bag_agg_averages_anchor_distances(self):
        anchors = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
        candidates = torch.tensor([[0.0, 1.0]])
        scorer = _make_scorer([anchors, candidates])
        anchor_embs = scorer.encode_anchors(["a", "b"])
        scores = scorer.score_displacement(anchor_embs, ["c"], bag_agg="mean")
        self.assertAlmostEqual(scores[0], 1.0, places=4)

    def test_min_and_mean_are_distinct_anchor_reductions(self):
        anchors = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
        candidates = torch.tensor([[1.0, 0.0]])
        min_scorer = _make_scorer([anchors, candidates])
        mean_scorer = _make_scorer([anchors, candidates])
        min_score = min_scorer.score_displacement(
            min_scorer.encode_anchors(["a", "b"]), ["c"], bag_agg="min",
        )[0]
        mean_score = mean_scorer.score_displacement(
            mean_scorer.encode_anchors(["a", "b"]), ["c"], bag_agg="mean",
        )[0]
        self.assertAlmostEqual(min_score, 0.0, places=4)
        self.assertAlmostEqual(mean_score, 1.0, places=4)

    def test_invalid_bag_agg_raises(self):
        scorer = _make_scorer([torch.zeros(1, 2)])
        anchor_embs = scorer.encode_anchors(["a"])
        with self.assertRaises(ValueError):
            scorer.score_displacement(anchor_embs, ["c"], bag_agg="median")


if __name__ == "__main__":
    unittest.main()
