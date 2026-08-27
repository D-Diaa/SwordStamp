"""CPU tests for the oracle attack's region-inverting fidelity scorer."""

import unittest

import torch

from attacks.paraphrasing.oracle import OracleScorer, _fidelity
from sampling.base_sampler import (
    BaseSampler,
    CandidateScore,
    Region,
    default_accept_fn,
)
from segmentation import Unit


class _FakeEncoder:
    """Return preloaded embedding tensors in call order."""

    def __init__(self, embeddings_per_call):
        self._embeddings = iter(embeddings_per_call)

    def encode(self, sentences, convert_to_tensor=True, normalize_embeddings=True):
        return next(self._embeddings)


def _units(n):
    return [Unit("sentence", f"source {i}.", f"Source {i}.") for i in range(n)]


def _candidates(n):
    return [Unit("sentence", f"candidate {i}.", f"Candidate {i}.") for i in range(n)]


def _make_scorer(encoder_outputs, regions_per_call, scores_first_unit=True):
    """Build a scorer over a fake encoder and a fake defender partition."""
    scorer = OracleScorer.__new__(OracleScorer)
    scorer.encoder = _FakeEncoder(encoder_outputs)
    scorer.scores_first_unit = scores_first_unit
    calls = iter(regions_per_call)
    seen = []

    def region_score_fn(predecessor, candidates, unit_idx):
        seen.append((predecessor, list(candidates), unit_idx))
        return [CandidateScore(r, 0.5) for r in next(calls)]

    scorer.region_score_fn = region_score_fn
    scorer.seen = seen
    return scorer


def _select(scores, candidates):
    """Return the index best-of-n would commit."""
    unit, _info = BaseSampler._select(
        list(zip(candidates, scores)), "best-of-n", default_accept_fn,
    )
    return candidates.index(unit)


class TestFidelity(unittest.TestCase):

    def test_maps_cosine_range_onto_unit_interval(self):
        self.assertAlmostEqual(_fidelity(1.0), 1.0)
        self.assertAlmostEqual(_fidelity(0.0), 0.5)
        self.assertAlmostEqual(_fidelity(-1.0), 0.0)

    def test_clamps_floating_point_overshoot(self):
        # Depth must stay nonnegative or CandidateScore rejects it.
        self.assertEqual(_fidelity(1.0000002), 1.0)
        self.assertEqual(_fidelity(-1.0000002), 0.0)


class TestRegionInversion(unittest.TestCase):

    def test_closest_red_candidate_wins(self):
        anchors = torch.tensor([[1.0, 0.0]])
        cands = torch.tensor([
            [1.0, 0.0],                    # identical, but a detector hit
            [0.9, 0.43589],                # near miss — the intended pick
            [-1.0, 0.0],                   # far miss
        ])
        scorer = _make_scorer(
            [anchors, cands], [[Region.GREEN, Region.RED, Region.RED]],
        )
        score_fn = scorer.make_doc_score_fn(_units(1))
        candidates = _candidates(3)
        scores = score_fn(Unit(), candidates, 0)
        self.assertEqual(
            [s.region for s in scores],
            [Region.RED, Region.GREEN, Region.GREEN],
        )
        self.assertEqual(_select(scores, candidates), 1)

    def test_yellow_counts_as_a_detector_hit(self):
        # Detection counts yellow under *_fixed_diverse, so the attacker must
        # treat it as a hit even though it is closer to the source.
        anchors = torch.tensor([[1.0, 0.0]])
        cands = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        scorer = _make_scorer([anchors, cands], [[Region.YELLOW, Region.RED]])
        score_fn = scorer.make_doc_score_fn(_units(1))
        candidates = _candidates(2)
        scores = score_fn(Unit(), candidates, 0)
        self.assertEqual([s.region for s in scores], [Region.RED, Region.GREEN])
        self.assertEqual(_select(scores, candidates), 1)

    def test_all_green_fallback_keeps_the_most_displaced_candidate(self):
        anchors = torch.tensor([[1.0, 0.0]])
        cands = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        scorer = _make_scorer(
            [anchors, cands], [[Region.GREEN, Region.GREEN, Region.GREEN]],
        )
        score_fn = scorer.make_doc_score_fn(_units(1))
        candidates = _candidates(3)
        scores = score_fn(Unit(), candidates, 0)
        self.assertTrue(all(s.region is Region.RED for s in scores))
        # A unit with no escape falls back to the adaptive objective.
        self.assertEqual(_select(scores, candidates), 2)

    def test_both_tiers_rank_on_the_same_fidelity_scale(self):
        # Green tier takes the deepest, red tier the shallowest, so one depth
        # definition serves both: win on closest, fail on most displaced.
        anchors = torch.tensor([[1.0, 0.0]])
        cands = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
        scorer = _make_scorer([anchors, cands], [[Region.GREEN, Region.RED]])
        score_fn = scorer.make_doc_score_fn(_units(1))
        scores = score_fn(Unit(), _candidates(2), 0)
        self.assertEqual(scores[0].depth, 1.0)
        self.assertEqual(scores[1].depth, 0.0)


class TestUnitIndexing(unittest.TestCase):

    def test_context_mode_ignores_regions_at_the_seed_unit(self):
        # Context-dependent detection spends unit 0 as the mask seed and never
        # counts it, so there is nothing to evade there.
        anchors = torch.tensor([[1.0, 0.0]])
        cands = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        scorer = _make_scorer([anchors, cands], [], scores_first_unit=False)
        score_fn = scorer.make_doc_score_fn(_units(1))
        candidates = _candidates(2)
        scores = score_fn(Unit(), candidates, 0)
        self.assertEqual(scorer.seen, [])
        self.assertTrue(all(s.region is Region.GREEN for s in scores))
        self.assertEqual(_select(scores, candidates), 0)

    def test_fixed_mode_scores_the_first_unit(self):
        anchors = torch.tensor([[1.0, 0.0]])
        cands = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        scorer = _make_scorer(
            [anchors, cands], [[Region.GREEN, Region.RED]], scores_first_unit=True,
        )
        score_fn = scorer.make_doc_score_fn(_units(1))
        candidates = _candidates(2)
        scores = score_fn(Unit(), candidates, 0)
        self.assertEqual(len(scorer.seen), 1)
        self.assertEqual(_select(scores, candidates), 1)

    def test_anchor_is_reused_past_the_source_unit_count(self):
        anchors = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        cands = torch.tensor([[0.0, 1.0]])
        scorer = _make_scorer([anchors, cands], [[Region.RED]])
        score_fn = scorer.make_doc_score_fn(_units(2))
        # Unit 7 exceeds the source length; the last anchor applies.
        scores = score_fn(Unit(), _candidates(1), 7)
        self.assertAlmostEqual(scores[0].depth, 1.0, places=5)

    def test_predecessor_and_unit_index_reach_the_partition(self):
        # Context-dependent masks are seeded from the committed predecessor.
        anchors = torch.tensor([[1.0, 0.0]])
        scorer = _make_scorer([anchors, torch.tensor([[1.0, 0.0]])], [[Region.RED]])
        score_fn = scorer.make_doc_score_fn(_units(1))
        predecessor = Unit("sentence", "committed.", "Committed.")
        score_fn(predecessor, _candidates(1), 3)
        seen_predecessor, _seen_candidates, seen_idx = scorer.seen[0]
        self.assertIs(seen_predecessor, predecessor)
        self.assertEqual(seen_idx, 3)


class TestMarkerStripping(unittest.TestCase):

    def test_partition_scores_the_text_detection_will_see(self):
        anchors = torch.tensor([[1.0, 0.0]])
        scorer = _make_scorer([anchors, torch.tensor([[1.0, 0.0]])], [[Region.RED]])
        score_fn = scorer.make_doc_score_fn(_units(1))
        marked = Unit(
            "sentence",
            "a rewritten sentence. [[END OF PARAPHRASE]]",
            "A rewritten sentence. [[END OF PARAPHRASE]]",
        )
        score_fn(Unit(), [marked], 0)
        _pred, seen_candidates, _idx = scorer.seen[0]
        self.assertEqual(seen_candidates[0].normalized, "a rewritten sentence.")


if __name__ == "__main__":
    unittest.main()
