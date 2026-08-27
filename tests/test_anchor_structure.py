import unittest

import numpy as np

from quality.metrics.anchors import CHANNELS, anchor_channels, evaluate_anchor_structure

S1 = "Gray wolves live in tightly organized packs across northern forests."
S2 = "A wolf pack hunts large prey such as elk and deer."
S3 = "The pack leader coordinates each hunt through scent marks and howls."
S4 = "Wolf pups are born blind in spring dens."
DOC = " ".join([S1, S2, S3, S4])


class TestIdentity(unittest.TestCase):

    def test_identity_all_channels_zero(self):
        ch = anchor_channels(DOC, DOC)
        for c in ("reword", "reword_novel", "reorder", "merge", "split", "reseg"):
            self.assertAlmostEqual(ch[c], 0.0, places=12, msg=c)
        self.assertAlmostEqual(ch["coverage"], 1.0)
        self.assertGreater(ch["n_anchors"], 10)


class TestReorder(unittest.TestCase):

    def test_full_reversal_is_max_reorder_and_nothing_else(self):
        rev = " ".join([S4, S3, S2, S1])
        ch = anchor_channels(DOC, rev)
        self.assertAlmostEqual(ch["reorder"], 1.0)
        self.assertAlmostEqual(ch["reword"], 0.0, places=12)
        self.assertAlmostEqual(ch["merge"], 0.0)
        self.assertAlmostEqual(ch["split"], 0.0)
        self.assertAlmostEqual(ch["reseg"], 0.0)

    def test_single_block_move_is_partial_reorder(self):
        moved = " ".join([S2, S3, S4, S1])
        ch = anchor_channels(DOC, moved)
        self.assertGreater(ch["reorder"], 0.0)
        self.assertLess(ch["reorder"], 1.0)
        self.assertAlmostEqual(ch["reseg"], 0.0)

    def test_within_sentence_reshuffle_is_not_reorder(self):
        # Reorder within S1 without crossing a unit boundary.
        s1_flipped = "Across northern forests, gray wolves live in tightly organized packs."
        ch = anchor_channels(DOC, " ".join([s1_flipped, S2, S3, S4]))
        self.assertAlmostEqual(ch["reorder"], 0.0)
        self.assertAlmostEqual(ch["reseg"], 0.0)


class TestReseg(unittest.TestCase):

    def test_merge_two_sentences(self):
        merged = " ".join([S1[:-1] + ", a wolf pack hunts large prey such as elk and deer.",
                           S3, S4])
        ch = anchor_channels(DOC, merged)
        self.assertAlmostEqual(ch["merge"], 1 / 3)     # 1 of 3 reference boundaries erased
        self.assertAlmostEqual(ch["split"], 0.0)       # no new candidate boundaries
        self.assertGreater(ch["reseg"], 0.0)
        self.assertAlmostEqual(ch["reword"], 0.0, places=12)  # only stopwords changed
        self.assertAlmostEqual(ch["reorder"], 0.0)

    def test_split_one_sentence(self):
        split = " ".join(["Gray wolves live in tightly organized packs. They range across "
                          "northern forests.", S2, S3, S4])
        ch = anchor_channels(DOC, split)
        self.assertGreater(ch["split"], 0.0)
        self.assertAlmostEqual(ch["merge"], 0.0)
        self.assertGreater(ch["reseg"], 0.0)
        self.assertAlmostEqual(ch["reorder"], 0.0)

    def test_full_merge_saturates_reseg(self):
        joined = DOC.replace(". ", ", ").replace(" A ", " a ").replace(" The ", " the ")
        ch = anchor_channels(DOC, joined)
        self.assertAlmostEqual(ch["merge"], 1.0)
        self.assertAlmostEqual(ch["reseg"], 1.0)


class TestReword(unittest.TestCase):

    REWORDED = " ".join([
        "Grey wolves dwell in tightly structured groups across northern woodlands.",
        "A wolf unit chases big quarry such as elk and deer.",
        S3, S4])

    def test_inplace_rewording_hits_only_reword(self):
        ch = anchor_channels(DOC, self.REWORDED)
        self.assertGreater(ch["reword"], 0.1)
        self.assertGreater(ch["reword_novel"], 0.1)
        self.assertAlmostEqual(ch["reorder"], 0.0)
        self.assertAlmostEqual(ch["merge"], 0.0)
        self.assertAlmostEqual(ch["split"], 0.0)
        self.assertAlmostEqual(ch["reseg"], 0.0)

    def test_reword_is_order_invariant(self):
        # the bag measure must score a pure shuffle at exactly 0
        shuffled = " ".join([S3, S1, S4, S2])
        self.assertAlmostEqual(anchor_channels(DOC, shuffled)["reword"], 0.0, places=12)

    def test_coverage_drops_with_rewording(self):
        full = anchor_channels(DOC, DOC)["coverage"]
        rew = anchor_channels(DOC, self.REWORDED)["coverage"]
        self.assertLess(rew, full)


class TestDegenerate(unittest.TestCase):

    def test_empty_candidate(self):
        ch = anchor_channels(DOC, "")
        self.assertAlmostEqual(ch["reword"], 1.0)
        self.assertTrue(np.isnan(ch["reorder"]))
        self.assertEqual(ch["n_anchors"], 0.0)

    def test_both_empty(self):
        ch = anchor_channels("", "")
        self.assertTrue(np.isnan(ch["reword"]))
        self.assertEqual(ch["n_anchors"], 0.0)

    def test_disjoint_texts_have_no_anchors(self):
        ch = anchor_channels("One two three alpha.", "Completely different words here.")
        self.assertEqual(ch["n_anchors"], 0.0)
        self.assertTrue(np.isnan(ch["reorder"]))
        self.assertGreater(ch["reword"], 0.9)

    def test_single_unit_reference_has_no_reorder_pairs(self):
        ch = anchor_channels(S1, S1)
        self.assertTrue(np.isnan(ch["reorder"]))   # no cross-unit pairs exist
        self.assertTrue(np.isnan(ch["merge"]))     # no reference boundaries to erase


class TestBatch(unittest.TestCase):

    def test_evaluate_shapes_and_keys(self):
        res = evaluate_anchor_structure([DOC, ""], [DOC, DOC])
        for c in CHANNELS:
            self.assertIn(f"anchor_{c}", res)
            self.assertEqual(len(res[f"anchor_{c}_per_sample"]), 2)
        self.assertAlmostEqual(res["anchor_reword_per_sample"][0], 0.0, places=12)
        self.assertAlmostEqual(res["anchor_reword_per_sample"][1], 1.0)


if __name__ == "__main__":
    unittest.main()
