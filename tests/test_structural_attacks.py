"""Probe atoms must move one channel and pin the rest.

The attribution regression is only identified if each atom is separable, so
these tests assert the pinning properties directly against the anchor channels
rather than merely checking that the text changed.
"""

import random
import unittest

import numpy as np

from attacks.base import AttackConfig
from attacks.simple import structural as st
from quality.metrics.anchors import anchor_channels
from segmentation import segment

S1 = "Gray wolves live in tightly organized packs across northern forests."
S2 = "A wolf pack hunts large prey such as elk and deer."
S3 = "The pack leader coordinates each hunt through scent marks and howls."
S4 = "Wolf pups are born blind in spring dens each year."
S5 = "Researchers track collared animals through several winter seasons."
S6 = "Territory sizes shift when prey density changes across the range."
DOC = " ".join([S1, S2, S3, S4, S5, S6])

OTHER = (
    "Coastal tide pools shelter anemones and small crabs. "
    "Volunteers survey the shoreline every autumn morning. "
    "Salinity readings guide the sampling schedule."
)

# Stands in for the unwatermarked donor corpus. It is deliberately disjoint from
# the attacked batch: batch-internal donors would splice the watermark back in.
DONORS = [
    "Freight schedules shifted after the new terminal opened downtown. "
    "Dispatchers rebuilt the timetable over a single weekend. "
    "Commuters noticed the change within a few days.",
    "Bakers proof the dough overnight in a cool room. "
    "Steam injected early gives the crust its shine.",
]


def units(text):
    """Sentence texts, stripped of the leading padding ``Unit.display`` keeps."""
    return [u.display.strip() for u in segment(text, type="sentence", backend="nltk")]


def n_units(text):
    return len(segment(text, type="sentence", backend="nltk"))


def rng():
    return random.Random(0)


class TestUnitCountAtoms(unittest.TestCase):
    """Atoms whose defining effect is on the segment count."""

    def test_sentence_deletion_leaves_survivors_verbatim(self):
        out = st.sentence_deletion(DOC, 0.5, rng())
        originals = set(units(DOC))
        for unit in units(out):
            self.assertIn(unit, originals)

    def test_sentence_deletion_does_not_resegment(self):
        ch = anchor_channels(DOC, st.sentence_deletion(DOC, 0.5, rng()))
        self.assertAlmostEqual(ch["reseg"], 0.0, places=12)
        self.assertAlmostEqual(ch["reorder"], 0.0, places=12)

    def test_merge_lowers_unit_count_without_losing_words(self):
        out = st.merge_adjacent(DOC, 1.0, rng())
        self.assertLess(n_units(out), n_units(DOC))
        self.assertEqual(len(out.split()), len(DOC.split()))

    def test_split_raises_unit_count_without_losing_words(self):
        out = st.split_midpoint(DOC, 1.0, rng())
        self.assertGreater(n_units(out), n_units(DOC))
        self.assertEqual(len(out.split()), len(DOC.split()))

    def test_merge_and_split_resegment(self):
        for fn in (st.merge_adjacent, st.split_midpoint):
            with self.subTest(atom=fn.__name__):
                self.assertGreater(anchor_channels(DOC, fn(DOC, 1.0, rng()))["reseg"], 0.0)

    def test_zero_ratio_is_a_no_op(self):
        for name, fn in st.RATIO_ATOMS.items():
            with self.subTest(atom=name):
                ch = anchor_channels(DOC, fn(DOC, 0.0, rng()))
                self.assertAlmostEqual(ch["reseg"], 0.0, places=12)
                self.assertAlmostEqual(ch["reword"], 0.0, places=12)


class TestFixedCountAtoms(unittest.TestCase):
    """Atoms that must hold the segment count fixed."""

    def test_clause_migrate_preserves_unit_count(self):
        out = st.clause_migrate(DOC, 1.0, rng())
        self.assertEqual(n_units(out), n_units(DOC))

    def test_clause_migrate_resegments_without_rewording(self):
        ch = anchor_channels(DOC, st.clause_migrate(DOC, 1.0, rng()))
        self.assertGreater(ch["reseg"], 0.0)
        self.assertAlmostEqual(ch["reword"], 0.0, places=12)

    def test_permute_preserves_units_exactly(self):
        out = st.permute_sentences(DOC, 1.0, rng())
        self.assertEqual(sorted(units(out)), sorted(units(DOC)))

    def test_permute_moves_only_the_reorder_channel(self):
        # Seeded to a permutation that actually reorders.
        out = st.permute_sentences(DOC, 1.0, random.Random(3))
        self.assertNotEqual(out, DOC)
        ch = anchor_channels(DOC, out)
        self.assertAlmostEqual(ch["reseg"], 0.0, places=12)
        self.assertAlmostEqual(ch["reword"], 0.0, places=12)
        self.assertGreater(ch["reorder"], 0.0)

    def test_controlled_reorder_hits_exact_inversion_targets(self):
        # Two unique anchors per sentence make anchor_reorder exactly equal to
        # the sentence-pair inversion fraction.
        doc = " ".join([
            "Albatross glides.", "Badger burrows.", "Cheetah sprints.",
            "Dolphin swims.", "Eagle soars.", "Falcon dives.",
        ])
        original = units(doc)
        maximum = len(original) * (len(original) - 1) // 2
        observed = []
        for ratio in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
            with self.subTest(ratio=ratio):
                out = st.controlled_reorder(doc, ratio, rng())
                order = [original.index(sentence) for sentence in units(out)]
                inversions = sum(
                    order[i] > order[j]
                    for i in range(len(order))
                    for j in range(i + 1, len(order))
                )
                expected = int(ratio * maximum + 0.5) / maximum
                channels = anchor_channels(doc, out)
                self.assertAlmostEqual(inversions / maximum, expected, places=12)
                self.assertAlmostEqual(channels["reorder"], expected, places=12)
                self.assertAlmostEqual(channels["reword"], 0.0, places=12)
                self.assertAlmostEqual(channels["reseg"], 0.0, places=12)
                observed.append(channels["reorder"])
        self.assertEqual(observed, sorted(observed))

    def test_controlled_reorder_preserves_units_and_spans_endpoints(self):
        for ratio, expected in ((0.0, 0.0), (1.0, 1.0)):
            with self.subTest(ratio=ratio):
                out = st.controlled_reorder(DOC, ratio, rng())
                self.assertEqual(sorted(units(out)), sorted(units(DOC)))
                self.assertEqual(n_units(out), n_units(DOC))
                self.assertAlmostEqual(
                    anchor_channels(DOC, out)["reorder"], expected, places=12,
                )

    def test_controlled_reorder_rejects_invalid_strength(self):
        for ratio in (-0.1, 1.1, float("nan")):
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                st.controlled_reorder(DOC, ratio, rng())


class TestUniformRechunk(unittest.TestCase):

    def test_chunks_respect_requested_length(self):
        out = st.uniform_rechunk(DOC, 5, rng())
        lengths = [len(u.display.split())
                   for u in segment(out, type="sentence", backend="nltk")]
        self.assertTrue(all(n <= 5 for n in lengths), lengths)

    def test_word_sequence_is_preserved_modulo_terminal_punctuation(self):
        out = st.uniform_rechunk(DOC, 7, rng())
        strip = lambda s: [w.rstrip(".!?").lower() for w in s.split()]  # noqa: E731
        self.assertEqual(strip(out), strip(DOC))

    def test_resegments_without_rewording(self):
        ch = anchor_channels(DOC, st.uniform_rechunk(DOC, 5, rng()))
        self.assertGreater(ch["reseg"], 0.0)
        self.assertAlmostEqual(ch["reword"], 0.0, places=12)

    def test_rejects_non_positive_chunk_length(self):
        with self.assertRaises(ValueError):
            st.uniform_rechunk(DOC, 0, rng())


class TestBoundaryExchange(unittest.TestCase):

    def test_preserves_words_order_count_and_anchor_coverage(self):
        lexical = lambda text: st._WORD_RE.findall(text.lower())  # noqa: E731
        for ratio in (0.1, 0.5, 1.0):
            with self.subTest(ratio=ratio):
                out = st.boundary_exchange(DOC, ratio, rng())
                channels = anchor_channels(DOC, out)
                self.assertEqual(lexical(out), lexical(DOC))
                self.assertEqual(n_units(out), n_units(DOC))
                self.assertAlmostEqual(channels["reword"], 0.0, places=12)
                self.assertAlmostEqual(channels["reorder"], 0.0, places=12)
                self.assertAlmostEqual(channels["coverage"], 1.0, places=12)

    def test_strength_monotonically_tracks_anchor_reseg(self):
        scores = []
        for ratio in (0.0, 0.25, 0.5, 0.75, 1.0):
            out = st.boundary_exchange(DOC, ratio, rng())
            channels = anchor_channels(DOC, out)
            scores.append(channels["reseg"])
            self.assertAlmostEqual(
                channels["merge"], channels["split"], delta=0.2,
            )
        self.assertEqual(scores, sorted(scores))
        self.assertGreater(scores[-1], 0.75)

    def test_zero_strength_is_exact_no_op(self):
        self.assertEqual(st.boundary_exchange(DOC, 0.0, rng()), DOC)

    def test_rejects_invalid_strength(self):
        for ratio in (-0.1, 1.1, float("nan")):
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                st.boundary_exchange(DOC, ratio, rng())


class TestDonorAtoms(unittest.TestCase):

    def test_random_sub_draws_only_from_donors(self):
        donors = st._content_words(OTHER)
        out = st.random_content_sub(DOC, 1.0, rng(), donors)
        original = {w.lower() for w in st._content_words(DOC)}
        allowed = {w.lower() for w in donors}
        for word in st._content_words(out):
            self.assertIn(word.lower(), allowed | original)

    def test_random_sub_rewords_without_resegmenting(self):
        out = st.random_content_sub(DOC, 0.5, rng(), st._content_words(OTHER))
        ch = anchor_channels(DOC, out)
        self.assertGreater(ch["reword"], 0.0)
        self.assertEqual(n_units(out), n_units(DOC))

    def test_random_sub_never_reuses_own_vocabulary(self):
        """Self-substitution manufactures false anchors, so it must not happen."""
        donors = st._content_words(OTHER)
        out = st.random_content_sub(DOC, 1.0, rng(), donors)
        own = {w.lower() for w in st._content_words(DOC)}
        substituted = [w for w in st._content_words(out) if w.lower() not in own]
        self.assertTrue(substituted)
        self.assertTrue(all(w.lower() in {d.lower() for d in donors} for w in substituted))

    def test_insertion_raises_unit_count_and_keeps_originals(self):
        out = st.sentence_insertion(DOC, 0.5, rng(), units(OTHER))
        self.assertGreater(n_units(out), n_units(DOC))
        produced = set(units(out))
        for unit in units(DOC):
            self.assertIn(unit, produced)

    def test_donor_pool_draws_only_from_the_donor_corpus(self):
        pool = st._donor_pool(DONORS, "random_content_sub")
        self.assertEqual(pool, [w for t in DONORS for w in st._content_words(t)])
        # Function words are shared by any two English texts; what must not leak
        # through is the attacked document's own distinctive vocabulary.
        lowered = {w.lower() for w in pool}
        for distinctive in ("wolves", "pack", "pups", "elk"):
            self.assertNotIn(distinctive, lowered)


class TestBatchApplication(unittest.TestCase):

    def test_every_atom_runs_and_returns_one_output_per_input(self):
        texts = [DOC, OTHER]
        for atom in st.PROBE_ATOMS:
            with self.subTest(atom=atom):
                out = st.apply_atom(texts, atom, ratio=0.5, words_per_chunk=8,
                                    donor_texts=DONORS)
                self.assertEqual(len(out), len(texts))
                self.assertTrue(all(isinstance(t, str) and t.strip() for t in out))

    def test_splicing_atoms_refuse_to_run_without_a_donor_corpus(self):
        """Batch-internal donors carry the watermark, so silence is not an option."""
        for atom in st.DONOR_ATOMS:
            with self.subTest(atom=atom):
                with self.assertRaises(ValueError):
                    st.apply_atom([DOC, OTHER], atom, ratio=0.5)

    def test_atoms_are_deterministic_under_a_fixed_seed(self):
        texts = [DOC, OTHER]
        for atom in st.PROBE_ATOMS:
            with self.subTest(atom=atom):
                a = st.apply_atom(texts, atom, ratio=0.5, words_per_chunk=8, seed=7,
                                  donor_texts=DONORS)
                b = st.apply_atom(texts, atom, ratio=0.5, words_per_chunk=8, seed=7,
                                  donor_texts=DONORS)
                self.assertEqual(a, b)

    def test_seed_changes_output(self):
        texts = [DOC, OTHER]
        a = st.apply_atom(texts, "permute_sentences", ratio=1.0, seed=1)
        b = st.apply_atom(texts, "permute_sentences", ratio=1.0, seed=2)
        self.assertNotEqual(a, b)

    def test_unknown_atom_rejected(self):
        with self.assertRaises(ValueError):
            st.apply_atom([DOC], "not_an_atom")

    def test_empty_and_degenerate_inputs_do_not_raise(self):
        for atom in st.PROBE_ATOMS:
            with self.subTest(atom=atom):
                out = st.apply_atom(["", "   ", "One short sentence."], atom,
                                    ratio=0.5, donor_texts=DONORS)
                self.assertEqual(len(out), 3)

    def test_runners_exist_and_return_matching_lengths(self):
        texts = [DOC, OTHER]
        cfg = AttackConfig(word_edit_ratio=0.5, rechunk_words=8, output_path=None)
        for atom in set(st.PROBE_ATOMS) - set(st.DONOR_ATOMS):
            with self.subTest(atom=atom):
                runner = getattr(st, f"run_{atom}_attack")
                result = runner(texts, cfg)
                self.assertEqual(len(result.para_text), len(texts))
                self.assertIsNone(result.save_path)

    def test_splicing_runners_require_a_configured_donor_corpus(self):
        cfg = AttackConfig(word_edit_ratio=0.5, output_path=None)
        for atom in st.DONOR_ATOMS:
            with self.subTest(atom=atom):
                with self.assertRaises(ValueError):
                    getattr(st, f"run_{atom}_attack")([DOC, OTHER], cfg)


class TestSeparation(unittest.TestCase):
    """The property the attribution regression depends on."""

    def test_each_atom_moves_its_own_channel_most(self):
        cases = [
            ("clause_migrate", {"ratio": 0.6}, "reseg"),
            ("merge_adjacent", {"ratio": 0.6}, "reseg"),
            ("split_midpoint", {"ratio": 0.6}, "reseg"),
            ("permute_sentences", {"ratio": 1.0}, "reorder"),
            ("controlled_reorder", {"ratio": 1.0}, "reorder"),
            ("boundary_exchange", {"ratio": 1.0}, "reseg"),
            ("random_content_sub", {"ratio": 0.4}, "reword"),
        ]
        for atom, kwargs, expected in cases:
            with self.subTest(atom=atom):
                out = st.apply_atom([DOC, OTHER], atom, seed=3,
                                    donor_texts=DONORS, **kwargs)[0]
                ch = anchor_channels(DOC, out)
                scores = {c: ch[c] for c in ("reseg", "reorder", "reword")
                          if not np.isnan(ch[c])}
                self.assertEqual(max(scores, key=scores.get), expected, scores)

    def test_lexically_null_atoms_leave_wording_untouched(self):
        for atom, kwargs in (("merge_adjacent", {"ratio": 1.0}),
                             ("split_midpoint", {"ratio": 1.0}),
                             ("clause_migrate", {"ratio": 1.0}),
                             ("permute_sentences", {"ratio": 1.0}),
                             ("controlled_reorder", {"ratio": 1.0}),
                             ("boundary_exchange", {"ratio": 1.0}),
                             ("uniform_rechunk", {"words_per_chunk": 6})):
            with self.subTest(atom=atom):
                out = st.apply_atom([DOC, OTHER], atom, seed=3, **kwargs)[0]
                self.assertAlmostEqual(
                    anchor_channels(DOC, out)["reword"], 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
