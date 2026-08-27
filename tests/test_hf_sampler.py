"""Test shared sampler logic with fake generation backends."""
import unittest
from unittest.mock import MagicMock

from sampling.base_sampler import (
    BaseSampler,
    CandidateScore,
    GeneratedCandidate,
    Region,
    default_accept_fn,
)
from segmentation import (
    Segmenter,
    Unit,
    display_with_boundary_space,
    normalize_generated_whitespace,
)


def _s(text: str) -> Unit:
    """Convenience: build a Unit from a plain string."""
    return Unit(type="sentence", normalized=text.lower().strip(), display=text)


def _norms(units) -> list:
    """Extract normalized strings from a list of Units."""
    return [u.normalized for u in units]


def _cand(unit: Unit, index: int = 0) -> GeneratedCandidate:
    """Wrap a Unit in a complete GeneratedCandidate."""
    return GeneratedCandidate(
        source_index=index,
        raw_text=unit.display,
        finished=True,
        boundary_complete=True,
        complete=True,
        unit=unit,
    )


def _green(depth: float = 1.0) -> CandidateScore:
    return CandidateScore(Region.GREEN, depth)


def _yellow(depth: float = 1.0) -> CandidateScore:
    return CandidateScore(Region.YELLOW, depth)


def _red(depth: float = 1.0) -> CandidateScore:
    return CandidateScore(Region.RED, depth)


def _length_scores(candidates) -> list[CandidateScore]:
    return [_green(float(len(candidate.normalized))) for candidate in candidates]


def _mock_tokenizer():
    tok = MagicMock()
    tok.encode.return_value = [0, 0, 0]
    tok.pad_token_id = 0
    return tok


class _FakeSampler(BaseSampler):
    """Return predetermined waves in call order."""

    def __init__(
        self,
        batches,
        num_candidates=4,
        segmentation_type="sentence",
        segmentation_backend="nltk",
    ):
        self.num_candidates = num_candidates
        self.chunk_tokens = 64
        self.segmentation_type = segmentation_type
        self.segmentation_backend = segmentation_backend
        self.segmenter = Segmenter(segmentation_type, segmentation_backend)
        self.stop_segmenter = Segmenter("sentence", segmentation_backend)
        self.tokenizer = _mock_tokenizer()
        # Auto-convert strings to Unit objects
        self._pending = [
            [_s(u) if isinstance(u, str) else u for u in batch]
            for batch in batches
        ]

    def _generate_batches_with_metadata(self, prompts, gen_config):
        return [
            [_cand(u, i) for i, u in enumerate(self._pending.pop(0) if self._pending else [])]
            for _ in prompts
        ]

    def generate_raw(self, prompts, n, max_tokens, gen_config):
        # Unused: waves are injected above the token engine.
        raise NotImplementedError


class _FakeBatchedSampler(BaseSampler):
    """Serve per-document wave queues keyed by prompt prefix."""

    def __init__(self, queues, num_candidates=4):
        self.num_candidates = num_candidates
        self.chunk_tokens = 64
        self.segmenter = Segmenter("sentence", "nltk")
        self.stop_segmenter = Segmenter("sentence", "nltk")
        self.tokenizer = _mock_tokenizer()
        self._queues = {
            prefix: [[_s(u) if isinstance(u, str) else u for u in batch] for batch in batches]
            for prefix, batches in queues.items()
        }
        self.wave_prompt_counts: list = []  # len(prompts) per driver wave

    def _generate_batches_with_metadata(self, prompts, gen_config):
        self.wave_prompt_counts.append(len(prompts))
        waves = []
        for p in prompts:
            key = next(k for k in self._queues if p.startswith(k))
            queue = self._queues[key]
            batch = queue.pop(0) if queue else []
            waves.append([_cand(u, i) for i, u in enumerate(batch)])
        return waves

    def generate_raw(self, prompts, n, max_tokens, gen_config):
        raise NotImplementedError


_DUMMY_CFG = None


class TestUniqueCandidates(unittest.TestCase):

    def test_deduplication_within_wave(self):
        s = _FakeSampler([["a.", "b.", "a."]])
        cands = s._unique_candidates("p", _DUMMY_CFG)
        self.assertEqual(_norms(cands).count("a."), 1)
        self.assertEqual(_norms(cands), ["a.", "b."])

    def test_empty_candidates_excluded(self):
        s = _FakeSampler([["", "  ", "real."]])
        self.assertEqual(_norms(s._unique_candidates("p", _DUMMY_CFG)), ["real."])

    def test_empty_wave_returns_empty(self):
        s = _FakeSampler([])
        self.assertEqual(s._unique_candidates("p", _DUMMY_CFG), [])


class TestPromptPredecessor(unittest.TestCase):

    def test_uses_final_configured_prompt_unit(self):
        sampler = _FakeSampler([])
        predecessor = sampler.prompt_predecessor("First sentence. Final sentence.")
        self.assertEqual(predecessor.normalized, "final sentence.")

    def test_empty_prompt_has_explicit_synthetic_unit(self):
        sampler = _FakeSampler([])
        self.assertEqual(
            sampler.prompt_predecessor(""), Unit("sentence", "", ""),
        )

    def test_unsegmentable_prompt_preserves_normalized_fallback(self):
        sampler = _FakeSampler([])
        sampler.segmenter = MagicMock()
        sampler.segmenter.type = "semspan"
        sampler.segmenter.segment.return_value = []
        self.assertEqual(
            sampler.prompt_predecessor("  Unsegmentable PROMPT  "),
            Unit("semspan", "unsegmentable prompt", "  Unsegmentable PROMPT  "),
        )


class TestGeneratedWhitespaceNormalization(unittest.TestCase):

    def test_nbsp_and_horizontal_runs_canonicalized(self):
        text = "A\xa0\xa0B\t\tC\n  D"
        self.assertEqual(normalize_generated_whitespace(text), "A B C\n D")

    def test_boundary_space_inserted_between_generated_units(self):
        self.assertEqual(display_with_boundary_space("First.", "Second."), " Second.")

    def test_boundary_space_not_inserted_before_punctuation(self):
        self.assertEqual(display_with_boundary_space("First", ","), ",")


class TestSelectionModes(unittest.TestCase):
    """generate() with selection_mode rejection vs best-of-n."""

    def test_rejection_returns_first_accepted(self):
        s = _FakeSampler([["red one.", "green one.", "green two."]])

        def score_fn(_predecessor, cands, _unit_idx):
            return [_green() if "green" in c.normalized else _red() for c in cands]

        unit, info = s.generate("p", _DUMMY_CFG, score_fn)
        self.assertEqual(unit.normalized, "green one.")
        self.assertTrue(info["accepted"])
        self.assertEqual(info["n_accepted_candidates"], 2)
        self.assertEqual(info["n_candidates"], 3)

    def test_rejection_falls_back_to_last_candidate(self):
        s = _FakeSampler([["red one.", "red two."]])

        def reject_all(_predecessor, cands, _unit_idx):
            return [_red() for _ in cands]

        unit, info = s.generate("p", _DUMMY_CFG, reject_all)
        self.assertEqual(unit.normalized, "red two.")
        self.assertFalse(info["accepted"])
        self.assertEqual(info["n_accepted_candidates"], 0)

    def test_rejection_falls_back_to_last_yellow(self):
        s = _FakeSampler([[
            "red one.", "yellow one.", "yellow two.",
            "shallow green.", "red two.",
        ]])

        def score_fn(_predecessor, _cands, _unit_idx):
            return [
                _red(0.4), _yellow(0.2), _yellow(0.7),
                _green(0.1), _red(0.2),
            ]

        unit, info = s.generate("p", _DUMMY_CFG, score_fn, margin=0.2)
        self.assertEqual(unit.normalized, "yellow two.")
        self.assertEqual(info["score"], _yellow(0.7))
        self.assertFalse(info["accepted"])
        self.assertEqual(info["n_accepted_candidates"], 0)

    def test_best_of_n_returns_highest_scoring(self):
        s = _FakeSampler([["a.", "much longer sentence here.", "bb."]])

        def length_score(_predecessor, cands, _unit_idx):
            return _length_scores(cands)

        unit, info = s.generate("p", _DUMMY_CFG, length_score, selection_mode="best-of-n")
        self.assertEqual(unit.normalized, "much longer sentence here.")
        self.assertEqual(
            info["score"], _green(float(len("much longer sentence here."))),
        )
        self.assertTrue(info["accepted"])

    def test_rejection_accepts_first_green_that_clears_external_margin(self):
        s = _FakeSampler([["shallow green.", "yellow.", "deep green."]])

        def score_fn(_predecessor, _cands, _unit_idx):
            return [_green(0.1), _yellow(0.9), _green(0.3)]

        unit, info = s.generate("p", _DUMMY_CFG, score_fn, margin=0.2)
        self.assertEqual(unit.normalized, "deep green.")
        self.assertTrue(info["accepted"])
        self.assertEqual(info["n_accepted_candidates"], 1)

    def test_best_of_n_ranks_regions_then_depth(self):
        cases = (
            (["deep yellow.", "shallow green.", "deep red."],
             [_yellow(0.9), _green(0.1), _red(0.9)], "shallow green."),
            (["shallow yellow.", "deep yellow."],
             [_yellow(0.1), _yellow(0.9)], "deep yellow."),
            (["deep red.", "shallow red."],
             [_red(0.9), _red(0.1)], "shallow red."),
        )
        for texts, scores, expected in cases:
            with self.subTest(expected=expected):
                sampler = _FakeSampler([texts])
                unit, _ = sampler.generate(
                    "p", _DUMMY_CFG,
                    lambda _predecessor, _cands, _unit_idx, scores=scores: scores,
                    selection_mode="best-of-n",
                )
                self.assertEqual(unit.normalized, expected)

    def test_numeric_scores_are_rejected(self):
        s = _FakeSampler([["candidate."]])
        with self.assertRaisesRegex(TypeError, "CandidateScore"):
            s.generate(
                "p", _DUMMY_CFG,
                lambda _predecessor, _cands, _unit_idx: [1.0],
            )

    def test_empty_pool_returns_empty_unit(self):
        s = _FakeSampler([])
        for mode in ("rejection", "best-of-n"):
            unit, info = s.generate(
                "p", _DUMMY_CFG,
                lambda _predecessor, cands, _unit_idx: [],
                selection_mode=mode,
            )
            self.assertFalse(unit.normalized)
            self.assertFalse(info["accepted"])
            self.assertEqual(info["n_candidates"], 0)

    def test_unknown_selection_mode_raises(self):
        s = _FakeSampler([["a."]])
        with self.assertRaises(ValueError):
            s.generate(
                "p", _DUMMY_CFG,
                lambda _predecessor, cands, _unit_idx: [_green()],
                selection_mode="bogus",
            )


class TestAcceptFn(unittest.TestCase):
    """Pluggable, pool-aware acceptance (PMark-style criteria)."""

    def test_default_accept_fn_uses_region_and_external_margin(self):
        self.assertEqual(
            default_accept_fn(
                [_green(0.5), _yellow(1.0), _red(0.1), _green(0.2)],
                margin=0.2,
            ),
            [True, False, False, False],
        )

    def test_pool_dependent_acceptance_changes_rejection_selection(self):
        # A median criterion skips the first positive score.
        s = _FakeSampler([["one.", "two.", "three.", "four four four."]])

        def length_score(_predecessor, cands, _unit_idx):
            return _length_scores(cands)

        def above_median(scores):
            ordered = sorted(scores, key=lambda score: score.depth)
            median = ordered[len(ordered) // 2]
            return [score.depth > median.depth for score in scores]

        unit, info = s.generate("p", _DUMMY_CFG, length_score, accept_fn=above_median)
        self.assertEqual(unit.normalized, "four four four.")
        self.assertTrue(info["accepted"])
        self.assertEqual(info["n_accepted_candidates"], 1)

    def test_best_of_n_winner_acceptance_uses_accept_fn(self):
        s = _FakeSampler([["a.", "long sentence wins."]])

        def length_score(_predecessor, cands, _unit_idx):
            return _length_scores(cands)

        def accept_nothing(scores):
            return [False] * len(scores)

        unit, info = s.generate(
            "p", _DUMMY_CFG, length_score, selection_mode="best-of-n",
            accept_fn=accept_nothing,
        )
        self.assertEqual(unit.normalized, "long sentence wins.")  # best-of-n unaffected
        self.assertFalse(info["accepted"])                        # flag reflects accept_fn
        self.assertEqual(info["n_accepted_candidates"], 0)

    def test_with_rejects_acceptance_partition_uses_accept_fn(self):
        s = _FakeSampler([["a.", "bb.", "ccc.", "dddd."]])

        def length_score(_predecessor, cands, _unit_idx):
            return _length_scores(cands)

        def longer_than_two(scores):
            return [score.depth > 3 for score in scores]

        accepted, rejected = s.generate_with_rejects(
            "p", _DUMMY_CFG, length_score, num_accepted=4, num_rejected=4,
            accept_fn=longer_than_two,
        )
        self.assertEqual(_norms(accepted), ["ccc.", "dddd."])
        self.assertEqual(_norms(rejected), ["a.", "bb."])


class TestGenericUnitGeneration(unittest.TestCase):

    def test_generate_scores_same_semspan_unit_that_is_returned(self):
        span_a = Unit("semspan", "first a", "Sentence A ")
        span_b = Unit("semspan", "first b", "Sentence B ")

        s = _FakeSampler(
            [[span_a, span_b]],
            segmentation_type="sentence",
            segmentation_backend="spacy",
        )

        seen = []
        predecessors = []

        def score_fn(predecessor, cands, unit_idx):
            predecessors.append((predecessor, unit_idx))
            seen.extend(cands)
            return [_green() if u.normalized == "first b" else _red() for u in cands]

        unit, info = s.generate(
            "p", _DUMMY_CFG, score_fn, predecessor=_s("previous span"),
        )

        self.assertTrue(info["accepted"])
        self.assertEqual(unit, span_b)
        self.assertEqual(_norms(seen), ["first a", "first b"])
        self.assertEqual(predecessors, [(_s("previous span"), 0)])

    def test_generic_continuation_appends_semspan_units(self):
        span_a = Unit("semspan", "first a", "First A")
        span_b = Unit("semspan", "first b", " because B")
        s = _FakeSampler(
            [[span_a], [span_b]],
            segmentation_type="sentence",
            segmentation_backend="spacy",
        )

        seen_indices = []

        def score_fn(_predecessor, cands, unit_idx):
            seen_indices.append(unit_idx)
            return [_green() for _ in cands]

        text, info = s.generate_continuation(
            "Prompt.", _DUMMY_CFG, score_fn, max_tokens=999,
        )

        self.assertEqual(text, "Prompt. First A because B")
        self.assertEqual(info["accepted_count"], 2)
        self.assertEqual(info["unit_count"], 2)
        self.assertEqual(seen_indices, [0, 1])

    def test_generic_continuation_info_returns_units_and_scores(self):
        span_a = Unit("semspan", "first a", "First A")
        span_b = Unit("semspan", "first b", " because B")
        s = _FakeSampler(
            [[span_a], [span_b]],
            segmentation_type="sentence",
            segmentation_backend="spacy",
        )

        def score_fn(_predecessor, cands, unit_idx):
            return [_green() for _ in cands]

        text, info = s.generate_continuation(
            "Prompt.", _DUMMY_CFG, score_fn, max_tokens=999,
        )

        self.assertEqual(text, "Prompt. First A because B")
        self.assertEqual(info["units"], [span_a, span_b])
        self.assertEqual(info["scores"], [_green(), _green()])
        self.assertEqual(info["accepted"], [True, True])


class TestGenerateWithRejectsRankPartition(unittest.TestCase):
    """generate_with_rejects(selection_mode='best-of-n.) — rank partition."""

    def _run(self, batches, num_accepted, num_rejected):
        s = _FakeSampler(batches)
        def score_fn(_predecessor, cands, _unit_idx):
            return _length_scores(cands)
        return s.generate_with_rejects(
            "p", _DUMMY_CFG, score_fn, selection_mode="best-of-n",
            num_accepted=num_accepted, num_rejected=num_rejected,
        )

    def test_accepted_are_top_scoring(self):
        accepted, _ = self._run([["a.", "longer.", "longest sentence here.", "short."]], 2, 1)
        self.assertIn("longest sentence here.", _norms(accepted))
        self.assertIn("longer.", _norms(accepted))

    def test_rejected_are_bottom_scoring(self):
        _, rejected = self._run([["a.", "bb.", "longest sentence here.", "ccc."]], 1, 2)
        self.assertIn("a.", _norms(rejected))
        self.assertIn("bb.", _norms(rejected))

    def test_disjoint(self):
        accepted, rejected = self._run([["a.", "bb.", "ccc.", "dddd.", "eeeee."]], 2, 2)
        self.assertEqual(set(_norms(accepted)) & set(_norms(rejected)), set())

    def test_accepted_takes_priority_when_pool_small(self):
        accepted, rejected = self._run([["a.", "bb.", "ccc."]], 2, 2)
        self.assertEqual(len(accepted), 2)
        self.assertLessEqual(len(rejected), 1)
        self.assertEqual(set(_norms(accepted)) & set(_norms(rejected)), set())

    def test_accepted_sorted_best_first(self):
        accepted, _ = self._run([["a.", "bb.", "ccc.", "dddd."]], 4, 0)
        lengths = [len(u.normalized) for u in accepted]
        self.assertEqual(lengths, sorted(lengths, reverse=True))


class TestGenerateWithRejectsAcceptancePartition(unittest.TestCase):
    """generate_with_rejects(selection_mode='rejection') acceptance partition."""

    def _run(self, batches, score_fn, num_accepted=1, num_rejected=1):
        s = _FakeSampler(batches)
        return s.generate_with_rejects(
            "p", _DUMMY_CFG, score_fn,
            num_accepted=num_accepted, num_rejected=num_rejected,
        )

    def test_partitioned_correctly(self):
        def fn(_predecessor, cands, _unit_idx):
            return [_green() if "good" in c.normalized else _red() for c in cands]
        accepted, rejected = self._run(
            [["good sent.", "bad sent.", "good again.", "bad again."]], fn, 2, 2)
        self.assertTrue(all("good" in u.normalized for u in accepted))
        self.assertTrue(all("bad" in u.normalized for u in rejected))

    def test_rejected_trimmed_to_len_accepted(self):
        def fn(_predecessor, cands, _unit_idx):
            return [_green() if "good" in c.normalized else _red() for c in cands]
        accepted, rejected = self._run([["bad1.", "bad2.", "bad3.", "bad4.", "good."]], fn, 1, 4)
        self.assertEqual(len(rejected), len(accepted))

    def test_empty_when_nothing_accepted(self):
        def fn(_predecessor, cands, _unit_idx):
            return [_red() for _ in cands]
        accepted, rejected = self._run([["bad.", "worse."]], fn, 1, 1)
        self.assertEqual(accepted, [])
        self.assertEqual(rejected, [])


class _FakeMetadataSampler(BaseSampler):
    """Returns predetermined metadata candidates for scored wave tests."""

    def __init__(self, items):
        self._items = items
        self.segmenter = Segmenter("sentence", "nltk")
        self.stop_segmenter = Segmenter("sentence", "nltk")
        self.num_candidates = len(items)
        self.chunk_tokens = 64

    def _generate_batch_with_metadata(self, prompt, gen_config):
        return self._items

    def generate_raw(self, prompts, n, max_tokens, gen_config):
        raise NotImplementedError


class TestScoredCandidateWave(unittest.TestCase):

    def test_preserves_metadata_and_scores_non_empty_units(self):
        items = [
            GeneratedCandidate(0, "raw a", False, True, True, _s("Alpha sentence.")),
            GeneratedCandidate(1, "", False, False, False, Unit()),
            GeneratedCandidate(2, "raw b", True, False, True, _s("Beta sentence.")),
        ]
        sampler = _FakeMetadataSampler(items)
        seen_predecessors = []

        def score_fn(predecessor, candidates, unit_idx):
            seen_predecessors.append((predecessor, unit_idx))
            return [_green(float(i)) for i, _ in enumerate(candidates)]

        scored = sampler.scored_candidate_wave(
            "prompt", _DUMMY_CFG, score_fn,
            predecessor=_s("explicit predecessor"), unit_idx=7,
        )

        self.assertEqual(
            seen_predecessors, [(_s("explicit predecessor"), 7)],
        )
        self.assertEqual([item.source_index for item in scored], [0, 1, 2])
        self.assertEqual(scored[0].raw_text, "raw a")
        self.assertEqual(scored[0].score, _green(0.0))
        self.assertIsNone(scored[1].score)
        self.assertEqual(scored[2].score, _green(1.0))
        self.assertTrue(scored[2].finished)


class TestPositionAwareIndexContract(unittest.TestCase):
    """Verify sequential unit indices in both selection modes."""

    def test_rejection_continuation_passes_sequential_indices(self):
        sentences = ["Sentence zero.", "Sentence one.", "Sentence two."]
        s = _FakeSampler([[sent] for sent in sentences])

        seen = []
        def score_fn(_predecessor, cands, unit_idx):
            seen.append(unit_idx)
            return [_green() for _ in cands]

        _, info = s.generate_continuation("p", _DUMMY_CFG, score_fn,
                                          max_tokens=999)
        self.assertEqual(sorted(set(seen)), list(range(info["unit_count"])))

    def test_best_of_n_continuation_passes_sequential_indices(self):
        sentences = ["First.", "Second.", "Third."]
        s = _FakeSampler([[sent] for sent in sentences])

        seen = []
        def score_fn(_predecessor, cands, unit_idx):
            seen.append(unit_idx)
            return _length_scores(cands)

        _, info = s.generate_continuation("p", _DUMMY_CFG, score_fn,
                                          selection_mode="best-of-n",
                                          max_tokens=999)
        self.assertEqual(sorted(set(seen)), list(range(info["unit_count"])))
        self.assertEqual(len(info["scores"]), info["unit_count"])


class TestBatchedContinuation(unittest.TestCase):
    """Pooled multi-document continuation driver."""

    @staticmethod
    def _accept_all(_predecessor, cands, _unit_idx):
        return [_green() for _ in cands]

    def test_documents_complete_independently_with_refill(self):
        queues = {
            "P0": [["Doc zero one."], ["Doc zero two."]],
            "P1": [["Doc one one."]],   # second wave is empty -> stops at 1 unit
            "P2": [["Doc two one."], ["Doc two two."]],
        }
        s = _FakeBatchedSampler(queues)
        results = s.generate_batched_continuation(
            ["P0", "P1", "P2"], _DUMMY_CFG, self._accept_all,
            max_tokens=999, max_active=2,
        )
        self.assertEqual(len(results), 3)
        texts = [r[0] for r in results]
        totals = [r[1]["unit_count"] for r in results]
        self.assertEqual(texts[0], "P0 Doc zero one. Doc zero two.")
        self.assertEqual(texts[1], "P1 Doc one one.")
        self.assertEqual(texts[2], "P2 Doc two one. Doc two two.")
        self.assertEqual(totals, [2, 1, 2])
        # The pool never exceeds max_active documents per wave.
        self.assertTrue(all(c <= 2 for c in s.wave_prompt_counts))
        # Refill happened: doc 2 entered after a slot freed (>= 3 waves total).
        self.assertGreaterEqual(len(s.wave_prompt_counts), 3)

    def test_passes_per_document_predecessors_and_indices(self):
        queues = {
            "A": [["A one."], ["A two."]],
            "B": [["B one."], ["B two."]],
        }
        s = _FakeBatchedSampler(queues)
        seen = []

        def score_fn(predecessor, cands, unit_idx):
            seen.append((predecessor.display[0], unit_idx))
            return [_green() for _ in cands]

        s.generate_batched_continuation(
            ["A", "B"], _DUMMY_CFG, score_fn, max_tokens=999,
        )
        self.assertEqual(sorted(seen), [("A", 0), ("A", 1), ("B", 0), ("B", 1)])

    def test_passes_exact_selected_unit_as_next_predecessor(self):
        first = _s("First selected unit.")
        second = _s("Second selected unit.")
        sampler = _FakeSampler([[first], [second]])
        seen = []

        def score(predecessor, candidates, unit_idx):
            seen.append((predecessor, unit_idx))
            return [_green() for _ in candidates]

        sampler.generate_continuation(
            "Prompt opening. Prompt predecessor.", _DUMMY_CFG, score,
            max_tokens=999,
        )
        self.assertEqual(seen[0][0].normalized, "prompt predecessor.")
        self.assertEqual(seen[0][1], 0)
        self.assertIs(seen[1][0], first)
        self.assertEqual(seen[1][1], 1)

    def test_red_fallback_is_the_next_predecessor_without_becoming_accepted(self):
        red_first = _s("First red fallback.")
        red_last = _s("Committed red fallback.")
        next_unit = _s("Next unit.")
        sampler = _FakeSampler([[red_first, red_last], [next_unit]])
        seen = []

        def reject_all(predecessor, candidates, unit_idx):
            seen.append(predecessor)
            return [_red() for _ in candidates]

        _text, info = sampler.generate_continuation(
            "Prompt.", _DUMMY_CFG, reject_all,
            max_tokens=999,
        )
        self.assertIs(seen[1], red_last)
        self.assertEqual(info["accepted"], [False, False])

    def test_semspan_prompt_recovery_runs_once_per_document(self):
        queues = {
            "A": [["A one."], ["A two."]],
            "B": [["B one."], ["B two."]],
        }
        sampler = _FakeBatchedSampler(queues)
        semspan = MagicMock()
        semspan.type = "semspan"
        semspan.segment.side_effect = lambda prompt: [
            Unit("semspan", prompt.lower(), prompt),
        ]
        sampler.segmenter = semspan

        sampler.generate_batched_continuation(
            ["A", "B"], _DUMMY_CFG, self._accept_all,
            max_tokens=999,
        )
        self.assertEqual(semspan.segment.call_count, 2)
        self.assertEqual(
            [call.args[0] for call in semspan.segment.call_args_list], ["A", "B"],
        )

    def test_pooled_documents_keep_predecessors_isolated(self):
        a1, a2 = _s("A one."), _s("A two.")
        b1, b2 = _s("B one."), _s("B two.")
        sampler = _FakeBatchedSampler({
            "A": [[a1], [a2]],
            "B": [[b1], [b2]],
        })
        later = {}

        def score(predecessor, candidates, unit_idx):
            if unit_idx == 1:
                later[candidates[0].normalized[0]] = predecessor
            return [_green() for _ in candidates]

        sampler.generate_batched_continuation(
            ["A", "B"], _DUMMY_CFG, score,
            max_tokens=999,
        )
        self.assertIs(later["a"], a1)
        self.assertIs(later["b"], b1)

    def test_best_of_n_selects_best_per_document(self):
        queues = {
            "A": [["short.", "a much longer candidate sentence."]],
            "B": [["tiny.", "an even much longer candidate sentence indeed."]],
        }
        s = _FakeBatchedSampler(queues)

        def score(_predecessor, cands, _unit_idx):
            return _length_scores(cands)

        results = s.generate_batched_continuation(
            ["A", "B"], _DUMMY_CFG, score, selection_mode="best-of-n",
            max_tokens=999,
        )
        self.assertIn("a much longer candidate sentence.", results[0][0])
        self.assertIn("an even much longer candidate sentence indeed.", results[1][0])
        for text, info in results:
            self.assertEqual(len(info["scores"]), info["unit_count"])
            self.assertEqual(info["unit_count"], 1)

    def test_rejection_fallback_counts_as_not_accepted(self):
        queues = {"A": [["red one.", "red two."]]}
        s = _FakeBatchedSampler(queues)

        def reject_all(_predecessor, cands, _unit_idx):
            return [_red() for _ in cands]

        results = s.generate_batched_continuation(
            ["A"], _DUMMY_CFG, reject_all, max_tokens=999,
        )
        text, info = results[0]
        self.assertEqual(info["accepted_count"], 0)
        self.assertEqual(info["unit_count"], 1)  # fallback unit still appended
        self.assertEqual(text, "A red two.")

    def test_info_shapes_match_single_document_wrapper(self):
        queues = {"A": [["A one."], ["A two."]]}
        s = _FakeBatchedSampler(queues)
        results = s.generate_batched_continuation(
            ["A"], _DUMMY_CFG, self._accept_all,
            max_tokens=999,
        )
        text, info = results[0]
        self.assertEqual(info["accepted_count"], 2)
        self.assertEqual(info["unit_count"], 2)
        for key in ("units", "steps", "accepted", "scores",
                    "n_accepted_candidates_per_unit", "n_candidates_per_unit"):
            self.assertIn(key, info)
            self.assertEqual(len(info[key]), 2)

    def test_max_active_one_is_fully_sequential(self):
        queues = {
            "A": [["A one."], ["A two."]],
            "B": [["B one."], ["B two."]],
        }
        s = _FakeBatchedSampler(queues)
        results = s.generate_batched_continuation(
            ["A", "B"], _DUMMY_CFG, self._accept_all,
            max_tokens=999, max_active=1,
        )
        self.assertEqual(results[0][0], "A A one. A two.")
        self.assertEqual(results[1][0], "B B one. B two.")
        self.assertTrue(all(c == 1 for c in s.wave_prompt_counts))

    def test_per_document_score_fns(self):
        queues = {
            "A": [["A one."], ["A two."]],
            "B": [["B one."], ["B two."]],
        }
        s = _FakeBatchedSampler(queues)
        seen_a, seen_b = [], []

        def score_a(predecessor, cands, unit_idx):
            seen_a.append(predecessor.display[0])
            return [_green() for _ in cands]

        def score_b(predecessor, cands, unit_idx):
            seen_b.append(predecessor.display[0])
            return [_green() for _ in cands]

        results = s.generate_batched_continuation(
            ["A", "B"], _DUMMY_CFG, [score_a, score_b],
            max_tokens=999,
        )
        # Each document was scored only by its own score_fn.
        self.assertEqual(set(seen_a), {"A"})
        self.assertEqual(set(seen_b), {"B"})
        self.assertEqual(results[0][1]["unit_count"], 2)
        self.assertEqual(results[1][1]["unit_count"], 2)

    def test_per_document_sequences_validate_length(self):
        s = _FakeBatchedSampler({"A": [["A one."]]})
        with self.assertRaises(ValueError):
            s.generate_batched_continuation(
                ["A"], _DUMMY_CFG, [self._accept_all, self._accept_all],
                max_tokens=999,
            )

    def test_pool_dependent_accept_fn_in_continuation(self):
        queues = {"A": [["one.", "two.", "three.", "four four four."]]}
        s = _FakeBatchedSampler(queues)

        def length_score(_predecessor, cands, _unit_idx):
            return _length_scores(cands)

        def above_median(scores):
            ordered = sorted(scores, key=lambda score: score.depth)
            median = ordered[len(ordered) // 2]
            return [score.depth > median.depth for score in scores]

        results = s.generate_batched_continuation(
            ["A"], _DUMMY_CFG, length_score,
            max_tokens=999, accept_fn=above_median,
        )
        text, info = results[0]
        self.assertIn("four four four.", text)
        self.assertEqual(info["accepted"], [True])
        self.assertEqual(info["n_accepted_candidates_per_unit"], [1])

    def test_empty_prompt_list_returns_empty(self):
        s = _FakeBatchedSampler({})
        self.assertEqual(
            s.generate_batched_continuation([], _DUMMY_CFG, self._accept_all), [],
        )

    def test_resolve_max_active_scales_with_num_candidates(self):
        s = _FakeBatchedSampler({}, num_candidates=128)
        self.assertEqual(s._resolve_max_active(None, n_docs=1000), 4)
        s1 = _FakeBatchedSampler({}, num_candidates=1)
        self.assertEqual(s1._resolve_max_active(None, n_docs=1000), 512)
        # Explicit value wins; both are clamped to the document count.
        self.assertEqual(s._resolve_max_active(16, n_docs=1000), 16)
        self.assertEqual(s._resolve_max_active(None, n_docs=2), 2)


if __name__ == "__main__":
    unittest.main()
