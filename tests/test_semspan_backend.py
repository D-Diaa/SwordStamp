"""Tests for bounded semantic-span segmentation."""

import unittest

import numpy as np

from segmentation import (
    DEFAULT_SEMCUT_MAX_WORDS,
    DEFAULT_SEMCUT_WINDOW,
    Segmenter,
    segment,
)


TEXTS = [
    "Thailand arrested the suspect, who was detained in Bangkok and is suspected of "
    "making a million from the website, which police shut down.",
    "He said that his teachers no longer had free speech rights because of a law he "
    "had signed, and that they needed to file records requests.",
    "Short one.",
    "",
]


class SemspanTest(unittest.TestCase):
    """Semantic spans are bounded, lossless, and explicitly encoder-bound."""

    DEGENERATE = "40 minutes before kickoff " * 40

    @classmethod
    def setUpClass(cls):
        from segmentation._backends import _semspan

        def encode(texts):
            rows = []
            for text in texts:
                row = np.zeros(16, dtype=np.float32)
                for i, byte in enumerate(text.encode("utf-8")):
                    row[(byte + i) % len(row)] += 1.0
                rows.append(row)
            return np.stack(rows)

        cls.backend = _semspan
        cls.encode = staticmethod(encode)
        cls.segmenter = Segmenter(
            "semspan", "nltk", encode=encode, encoder_id="test/encoder",
        )

    def test_display_invariant(self):
        for text in TEXTS + [self.DEGENERATE]:
            if not text.strip():
                continue
            got = "".join(unit.display for unit in self.segmenter.segment(text))
            want = "".join(
                unit.display for unit in segment(text, type="sentence", backend="nltk")
            )
            self.assertEqual(got, want)

    def test_units_are_bounded_semantic_spans(self):
        for text in TEXTS + [self.DEGENERATE]:
            if not text.strip():
                continue
            for unit in self.segmenter.segment(text):
                self.assertEqual(unit.type, "semspan")
                self.assertLessEqual(
                    len(unit.normalized.split()), 4 * DEFAULT_SEMCUT_MAX_WORDS,
                )

    def test_degenerate_text_is_split(self):
        self.assertGreater(len(self.segmenter.segment(self.DEGENERATE)), 5)

    def test_batch_matches_scalar(self):
        batched = self.backend.segment_semspans_batch(TEXTS, encode=self.encode)
        for got, text in zip(batched, TEXTS):
            want = self.backend.segment_semspans(text, encode=self.encode)
            self.assertEqual([unit.normalized for unit in got],
                             [unit.normalized for unit in want])
            self.assertEqual([unit.display for unit in got],
                             [unit.display for unit in want])

    def test_first_units_match_scalar_calls(self):
        batched = self.segmenter.first_units(TEXTS)
        scalar = [self.segmenter.first_unit(text) for text in TEXTS]
        self.assertEqual([unit.normalized for unit in batched],
                         [unit.normalized for unit in scalar])
        self.assertEqual([unit.display for unit in batched],
                         [unit.display for unit in scalar])

    def test_interspan_whitespace_belongs_to_right_unit(self):
        units = self.segmenter.segment(self.DEGENERATE)
        self.assertGreater(len(units), 1)
        for left, right in zip(units, units[1:]):
            self.assertFalse(left.display.endswith((" ", "\t")))
            self.assertTrue(right.display.startswith((" ", "\t")))

    def test_policy_is_explicit_and_changes_boundaries(self):
        custom = Segmenter(
            "semspan", "nltk", encode=self.encode,
            encoder_id="test/encoder", semcut_max_words=6, semcut_window=2,
        )
        self.assertGreater(
            len(custom.segment(self.DEGENERATE)),
            len(self.segmenter.segment(self.DEGENERATE)),
        )
        self.assertEqual(custom.metadata(), {
            "type": "semspan",
            "backend": "nltk",
            "semantic_encoder": "test/encoder",
            "semcut_max_words": 6,
            "semcut_window": 2,
        })

    def test_default_policy_is_fifteen_five(self):
        self.assertEqual(
            (self.segmenter.semcut_max_words, self.segmenter.semcut_window),
            (DEFAULT_SEMCUT_MAX_WORDS, DEFAULT_SEMCUT_WINDOW),
        )

    def test_empty(self):
        self.assertEqual(self.backend.segment_semspans(""), [])
        self.assertEqual(self.backend.segment_semspans_batch([]), [])

    def test_nonempty_semspan_requires_explicit_encoder(self):
        with self.assertRaisesRegex(ValueError, "explicit semantic encoder"):
            segment(TEXTS[0], type="semspan", backend="nltk")

    def test_semspan_segmenter_rejects_missing_encoder(self):
        with self.assertRaisesRegex(ValueError, "explicit semantic encoder"):
            Segmenter("semspan", "spacy")


if __name__ == "__main__":
    unittest.main()
