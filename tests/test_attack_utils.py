import unittest

from attacks.utils import (
    strip_paraphrase_markers,
    split_texts_into_sentences,
    join_sentences_by_document,
    well_formed_sentence,
    _build_bigrams,
)


class TestStripParaphraseMarkers(unittest.TestCase):

    def test_no_marker_unchanged(self):
        text = "The cat sat on the mat."
        self.assertEqual(strip_paraphrase_markers(text), text)

    def test_marker_truncates(self):
        text = "Good sentence. [[END OF PARAPHRASE]] garbage"
        self.assertEqual(strip_paraphrase_markers(text), "Good sentence.")

    def test_marker_at_start_gives_empty(self):
        text = "[[END OF PARAPHRASE]] nothing useful"
        self.assertEqual(strip_paraphrase_markers(text), "")

    def test_result_stripped(self):
        text = "  Sentence.  [[END OF PARAPHRASE]] rest"
        result = strip_paraphrase_markers(text)
        self.assertFalse(result.startswith(" "))
        self.assertFalse(result.endswith(" "))

    def test_multiple_markers_truncates_at_first(self):
        text = "A. [[END OF PARAPHRASE]] B. [[END OF PARAPHRASE]] C."
        result = strip_paraphrase_markers(text)
        self.assertNotIn("B.", result)
        self.assertNotIn("C.", result)

    def test_end_of_without_paraphrase_not_stripped(self):
        # Ordinary prose is not a sentinel.
        text = "We reached the end of the road by noon."
        self.assertEqual(strip_paraphrase_markers(text), text)

    def test_empty_input(self):
        self.assertEqual(strip_paraphrase_markers(""), "")

    def test_quoted_bracket_variant(self):
        text = "Good sentence. [['END OF PARAPHRASE']]"
        self.assertEqual(strip_paraphrase_markers(text), "Good sentence.")

    def test_unbalanced_bracket_variant(self):
        text = "Good sentence. [] END OF PARAPHRASE]"
        self.assertEqual(strip_paraphrase_markers(text), "Good sentence.")

    def test_bare_phrase_variant(self):
        text = "Good sentence. END OF PARAPHRASE"
        self.assertEqual(strip_paraphrase_markers(text), "Good sentence.")

    def test_underscore_variant(self):
        text = "Good sentence. [[END_OF_PARAPHRASE]]"
        self.assertEqual(strip_paraphrase_markers(text), "Good sentence.")

    def test_misspelled_trailing_sentinel(self):
        text = "Good sentence. [[END OF PARAPRASE]]"  # Misspelled sentinel.
        self.assertEqual(strip_paraphrase_markers(text), "Good sentence.")

    def test_trailing_bracket_only_at_end(self):
        # Preserve mid-text bracketed blocks.
        text = "See [[note]] and keep reading here."
        self.assertEqual(strip_paraphrase_markers(text), text)

    def test_bracketed_prose_not_stripped(self):
        # Preserve bracketed prose.
        text = "We met near the [end of the] road."
        self.assertEqual(strip_paraphrase_markers(text), text)


class TestSplitTextsIntoSentencesAndJoin(unittest.TestCase):

    def test_single_document_roundtrip(self):
        texts = ["The cat sat on the mat. The dog ran away."]
        sents, doc_lengths = split_texts_into_sentences(texts)
        rejoined = join_sentences_by_document(sents, doc_lengths)
        self.assertEqual(len(rejoined), 1)
        # Both source sentences survive.
        self.assertIn("cat", rejoined[0])
        self.assertIn("dog", rejoined[0])

    def test_multi_document_lengths_correct(self):
        texts = [
            "First sentence. Second sentence.",
            "Only one sentence.",
            "A. B. C.",
        ]
        sents, doc_lengths = split_texts_into_sentences(texts)
        self.assertEqual(len(doc_lengths), 3)
        self.assertEqual(doc_lengths[1], 1)
        self.assertEqual(sum(doc_lengths), len(sents))

    def test_join_output_count_matches_input(self):
        texts = ["Doc one. Sentence two.", "Doc two."]
        sents, doc_lengths = split_texts_into_sentences(texts)
        rejoined = join_sentences_by_document(sents, doc_lengths)
        self.assertEqual(len(rejoined), len(texts))

    def test_empty_document_handled(self):
        texts = [""]
        sents, doc_lengths = split_texts_into_sentences(texts)
        rejoined = join_sentences_by_document(sents, doc_lengths)
        self.assertEqual(len(rejoined), 1)
        self.assertEqual(rejoined[0], "")

    def test_sentence_order_preserved(self):
        texts = ["Alpha. Beta. Gamma.", "Delta. Epsilon."]
        sents, doc_lengths = split_texts_into_sentences(texts)
        # Document order survives.
        doc0_sents = sents[:doc_lengths[0]]
        self.assertTrue(any("Alpha" in s for s in doc0_sents))
        doc1_sents = sents[doc_lengths[0]:]
        self.assertTrue(any("Delta" in s for s in doc1_sents))


class TestWellFormedSentence(unittest.TestCase):

    def test_first_letter_uppercased(self):
        self.assertEqual(well_formed_sentence("hello world.")[0], "H")

    def test_double_space_collapsed(self):
        result = well_formed_sentence("hello  world.")
        self.assertNotIn("  ", result)

    def test_end_sent_adds_period_if_missing(self):
        result = well_formed_sentence("hello world", end_sent=True)
        self.assertTrue(result.endswith("."))

    def test_end_sent_false_no_period_added(self):
        result = well_formed_sentence("hello world", end_sent=False)
        self.assertFalse(result.endswith("."))

    def test_existing_terminal_punct_not_doubled(self):
        result = well_formed_sentence("Hello world.", end_sent=True)
        self.assertFalse(result.endswith(".."))

    def test_i_uppercased(self):
        result = well_formed_sentence("they said i would come.")
        self.assertIn(" I ", result)

    def test_empty_string_ok(self):
        # Empty text is valid.
        result = well_formed_sentence("")
        self.assertEqual(result, "")


class TestBuildBigrams(unittest.TestCase):

    def test_empty_ids(self):
        import torch
        ids = torch.tensor([], dtype=torch.long)
        self.assertEqual(_build_bigrams(ids), [])

    def test_single_token(self):
        import torch
        ids = torch.tensor([5], dtype=torch.long)
        self.assertEqual(_build_bigrams(ids), [])

    def test_two_tokens_one_bigram(self):
        import torch
        ids = torch.tensor([1, 2], dtype=torch.long)
        self.assertEqual(_build_bigrams(ids), [(1, 2)])

    def test_three_tokens_two_bigrams(self):
        import torch
        ids = torch.tensor([1, 2, 3], dtype=torch.long)
        self.assertEqual(_build_bigrams(ids), [(1, 2), (2, 3)])

    def test_no_overlap_skipped(self):
        import torch
        ids = torch.tensor([10, 20, 30, 40], dtype=torch.long)
        bigrams = _build_bigrams(ids)
        self.assertEqual(len(bigrams), 3)
        self.assertEqual(bigrams[0], (10, 20))
        self.assertEqual(bigrams[-1], (30, 40))


if __name__ == "__main__":
    unittest.main()
