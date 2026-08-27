"""Test segmentation units, backends, and display idempotency."""

import unittest

from segmentation import (
    DEFAULT_BACKEND,
    DEFAULT_TYPE,
    Segmenter,
    Unit,
    check_split_idempotent,
    first_unit,
    normalize_text,
    resolve_segmentation_backend,
    resolve_segmentation_type,
    segment,
    segmentation_kwargs,
)
from segmentation import insert_missing_punctuation


def _spacy_available() -> bool:
    try:
        import spacy
        spacy.load("en_core_web_md")
        return True
    except Exception:
        return False


_SPACY_AVAILABLE = _spacy_available()
_spacy_skip = unittest.skipUnless(_SPACY_AVAILABLE, "en_core_web_md not installed")



def _joined(text: str) -> str:
    """Return ''.join(u.display for u in segment(text))."""
    return "".join(u.display for u in segment(text))



class TestDispatch(unittest.TestCase):

    def test_empty_string_returns_empty_list(self):
        self.assertEqual(segment(""), [])

    def test_whitespace_only_returns_empty_list(self):
        self.assertEqual(segment("   \t\n  "), [])

    def test_returns_list_of_units(self):
        units = segment("Hello world.")
        self.assertIsInstance(units, list)
        self.assertTrue(all(isinstance(u, Unit) for u in units))

    def test_first_unit_matches_first_segment(self):
        text = "Hello world. Goodbye moon."
        self.assertEqual(first_unit(text), segment(text)[0])

    def test_first_unit_empty_text_returns_empty_unit(self):
        self.assertEqual(first_unit("   "), Unit())

    def test_unit_type_field_is_sentence(self):
        units = segment("Hello world.")
        self.assertTrue(all(u.type == "sentence" for u in units))

    def test_invalid_type_raises_value_error(self):
        with self.assertRaises(ValueError):
            segment("Hello.", type="paragraph")

    def test_invalid_backend_raises_value_error(self):
        with self.assertRaises(ValueError):
            segment("Hello.", backend="nonexistent")

    def test_invalid_combo_raises_value_error(self):
        with self.assertRaises(ValueError):
            segment("Hello.", type="paragraph", backend="nonexistent")

    def test_resolve_segmentation_defaults_without_config(self):
        self.assertEqual(resolve_segmentation_type(), DEFAULT_TYPE)
        self.assertEqual(resolve_segmentation_backend(), DEFAULT_BACKEND)

    def test_resolve_segmentation_from_config(self):
        class Config:
            type = "sentence"
            backend = "spacy"

        self.assertEqual(resolve_segmentation_type(Config), "sentence")
        self.assertEqual(resolve_segmentation_backend(Config), "spacy")
        self.assertEqual(segmentation_kwargs(Config), {"type": "sentence", "backend": "spacy"})
        self.assertEqual(
            segmentation_kwargs(segmentation_type="sentence", segmentation_backend="spacy"),
            {"type": "sentence", "backend": "spacy"},
        )

    def test_semcut_provenance_resolves_from_typed_config_shape(self):
        class Config:
            type = "semspan"
            backend = "spacy"
            semcut_max_words = 15
            semcut_window = 5

        self.assertEqual(segmentation_kwargs(Config), {
            "type": "semspan",
            "backend": "spacy",
            "semcut_max_words": 15,
            "semcut_window": 5,
        })



class TestSentenceCount(unittest.TestCase):

    def test_single_sentence_with_period(self):
        self.assertEqual(len(segment("Hello world.")), 1)

    def test_single_sentence_without_punct(self):
        self.assertEqual(len(segment("Hello world")), 1)

    def test_question_exclamation(self):
        self.assertEqual(len(segment("Really? Yes! Definitely.")), 3)

    def test_two_sentences(self):
        self.assertEqual(len(segment("The cat sat. The dog ran.")), 2)

    def test_three_sentences(self):
        text = "The cat sat on the mat. The dog chased the cat. The cat ran away."
        self.assertEqual(len(segment(text)), 3)

    def test_newline_separated(self):
        self.assertEqual(len(segment("First line\nSecond line")), 2)

    def test_three_newline_lines(self):
        self.assertEqual(len(segment("Line one\nLine two\nLine three")), 3)

    def test_double_newline_paragraph(self):
        self.assertEqual(len(segment("First sentence.\n\nSecond sentence.")), 2)



class TestNormalizedField(unittest.TestCase):

    def test_normalized_is_lowercase(self):
        units = segment("Hello World.")
        self.assertEqual(units[0].normalized, "hello world.")

    def test_normalized_is_stripped(self):
        units = segment("  Hello world.  ")
        self.assertFalse(units[0].normalized.startswith(" "))
        self.assertFalse(units[0].normalized.endswith(" "))

    def test_normalized_collapses_tabs(self):
        units = segment("Hello\tworld.")
        self.assertNotIn("\t", units[0].normalized)

    def test_normalized_collapses_double_spaces(self):
        units = segment("Hello  world.  How  are  you?")
        for u in units:
            self.assertNotIn("  ", u.normalized)

    def test_normalized_has_no_newlines(self):
        units = segment("First line\nSecond line")
        for u in units:
            self.assertNotIn("\n", u.normalized)

    def test_normalized_missing_punct_gets_period(self):
        units = segment("Hello world")
        self.assertTrue(units[0].normalized.endswith("."))

    def test_normalized_retains_terminal_punct(self):
        units = segment("A. B? C!")
        self.assertTrue(all(u.normalized[-1] in ".?!" for u in units))

    def test_normalized_equals_display_collapsed_lowercase(self):
        import re
        units = segment("The Cat Sat. The Dog Ran.")
        for u in units:
            expected = re.sub(r'\s+', ' ', u.display).strip().lower()
            self.assertEqual(u.normalized, expected)



class TestDisplayField(unittest.TestCase):

    def test_first_unit_no_added_separator(self):
        # First unit should not have a leading space added artificially
        units = segment("The cat sat. The dog ran.")
        self.assertFalse(units[0].display.startswith(" "))

    def test_second_unit_carries_space_separator(self):
        units = segment("The cat sat. The dog ran.")
        self.assertTrue(units[1].display.startswith(" "))

    def test_newline_separator_preserved(self):
        units = segment("The cat sat.\nThe dog ran.")
        self.assertTrue(units[1].display.startswith("\n"))

    def test_display_case_preserved(self):
        units = segment("Hello World.")
        self.assertIn("H", units[0].display)
        self.assertIn("W", units[0].display)

    def test_display_case_preserved_when_abbreviations_are_case_sensitive(self):
        text = "I met A. B. Smith. He Left."
        units = segment(text)
        self.assertEqual("".join(u.display for u in units), insert_missing_punctuation(text))

    def test_display_case_preserved_for_month_abbreviation(self):
        text = "GREAT NEWS! We extended the contest deadline to Dec. 15. Everybody's a critic."
        units = segment(text)
        self.assertEqual("".join(u.display for u in units), insert_missing_punctuation(text))

    def test_display_case_preserved_for_title_abbreviation(self):
        text = "State Services Commissioner Peter Hughes thanked Chai Chuah. Mr Chuah replied."
        units = segment(text)
        self.assertEqual("".join(u.display for u in units), insert_missing_punctuation(text))

    def test_display_retains_internal_tabs(self):
        units = segment("Hello\tworld. How are you?")
        self.assertIn("\t", units[0].display)

    def test_display_retains_double_spaces(self):
        units = segment("Hello  world. How?")
        self.assertIn("  ", units[0].display)

    def test_model_output_leading_space_captured_in_first_unit(self):
        # Preserve a model-generated leading space.
        units = segment(" The cat sat. More text.")
        self.assertTrue(units[0].display.startswith(" "))

    def test_display_stripped_gives_normalized_case_preserved(self):
        import re
        units = segment("The cat sat. The dog ran.")
        for u in units:
            stripped = re.sub(r'\s+', ' ', u.display).strip()
            self.assertEqual(stripped.lower(), u.normalized)


class TestPromptExtractionCasing(unittest.TestCase):

    def test_extract_prompt_preserves_month_abbreviation_casing(self):
        from watermarking.primitives import extract_prompt_from_text

        text = (
            "GREAT NEWS! We extended the contest deadline to Dec. 15. "
            "Everybody's a critic. Few dare to sign their name. Enter the "
            '"1999 Acura Book Review" contest and find out what your opinion is really worth.'
        )
        self.assertEqual(
            extract_prompt_from_text(text, 32),
            "We extended the contest deadline to Dec. 15.",
        )

    def test_extract_prompt_preserves_title_abbreviation_casing(self):
        from watermarking.primitives import extract_prompt_from_text

        text = (
            "State Services Commissioner Peter Hughes today thanked the "
            "Director-General of Health Chai Chuah for his public service. "
            "Mr Chuah has been the Director-General of Health since November 2013."
        )
        self.assertEqual(
            extract_prompt_from_text(text, 32),
            "State Services Commissioner Peter Hughes today thanked the "
            "Director-General of Health Chai Chuah for his public service.",
        )



class TestIdempotency(unittest.TestCase):

    def _check(self, text: str):
        expected = insert_missing_punctuation(text)
        got = _joined(text)
        self.assertEqual(
            got, expected,
            f"\nInput:    {text!r}\nExpected: {expected!r}\nGot:      {got!r}"
        )
        # Joined display spans recover the punctuation-fixed paragraph.
        sents = [u.display for u in segment(text)]
        self.assertTrue(
            check_split_idempotent(insert_missing_punctuation(text), sents),
            f"sentence split dropped text from {text!r}",
        )

    def test_simple_two_sentence(self):
        self._check("The cat sat. The dog ran.")

    def test_newline_separated_with_punct(self):
        self._check("The cat sat.\nThe dog ran.")

    def test_newline_separated_without_punct(self):
        self._check("First line\nSecond line")

    def test_leading_space_model_output(self):
        self._check(" The cat sat. More text.")

    def test_double_spaces_internal(self):
        self._check("Hello  world.  How  are  you?")

    def test_double_newline_paragraph(self):
        self._check("First sentence.\n\nSecond sentence.")

    def test_trailing_newline(self):
        self._check("Sentence one.\nSentence two.\n")

    def test_three_sentences(self):
        self._check("The cat sat on the mat. The dog chased the cat. The cat ran away.")

    def test_internal_tab(self):
        self._check("Hello\tworld. How are you?")

    def test_single_sentence_no_punct(self):
        self._check("Hello world")

    def test_rejoined_display_re_segments_identically(self):
        text = "The cat sat. The dog ran. The bird flew."
        units_first = segment(text)
        rejoined = "".join(u.display for u in units_first)
        units_second = segment(rejoined)
        self.assertEqual(
            [u.normalized for u in units_first],
            [u.normalized for u in units_second],
        )


class TestInsertMissingPunctuation(unittest.TestCase):

    def test_line_without_punct_gets_period(self):
        self.assertEqual(insert_missing_punctuation("Hello world"), "Hello world.")

    def test_terminal_punct_inside_closing_quote_is_respected(self):
        # Closing quotes do not hide existing terminal punctuation.
        self.assertEqual(
            insert_missing_punctuation('She asked "why?"'),
            'She asked "why?"',
        )
        self.assertEqual(
            insert_missing_punctuation("He shouted “stop!”"),
            "He shouted “stop!”",
        )
        self.assertEqual(
            insert_missing_punctuation("a fact (see above.)"),
            "a fact (see above.)",
        )

    def test_quote_without_terminal_punct_still_gets_period(self):
        self.assertEqual(
            insert_missing_punctuation('He said "hello"'),
            'He said "hello".',
        )

    def test_multiline_mixed(self):
        text = 'First line\nShe asked "why?"\nlast line'
        self.assertEqual(
            insert_missing_punctuation(text),
            'First line.\nShe asked "why?"\nlast line.',
        )


class TestNormalizeText(unittest.TestCase):

    def test_lowercases(self):
        self.assertEqual(normalize_text("Hello World"), "hello world")

    def test_collapses_whitespace(self):
        self.assertEqual(normalize_text("hello  world"), "hello world")

    def test_collapses_tabs_and_newlines(self):
        self.assertEqual(normalize_text("hello\t world\n"), "hello world")

    def test_strips_edges(self):
        self.assertEqual(normalize_text("  hello world  "), "hello world")

    def test_nbsp_converted(self):
        self.assertEqual(normalize_text("hello world"), "hello world")

    def test_empty_string(self):
        self.assertEqual(normalize_text(""), "")

    def test_matches_unit_normalized(self):
        # Account for NLTK's missing-punctuation insertion.
        units = segment("Hello World. Goodbye Moon.")
        for u in units:
            self.assertEqual(normalize_text(u.display), u.normalized)



class TestCheckSplitIdempotent(unittest.TestCase):

    def test_clean_split_is_true(self):
        para = "The cat sat. The dog ran."
        sents = ["The cat sat.", "The dog ran."]
        self.assertTrue(check_split_idempotent(para, sents))

    def test_dropped_sentence_is_false(self):
        para = "The cat sat. The dog ran."
        self.assertFalse(check_split_idempotent(para, ["The cat sat."]))

    def test_extra_sentence_is_false(self):
        para = "The cat sat."
        self.assertFalse(check_split_idempotent(para, ["The cat sat.", "Extra."]))

    def test_whitespace_difference_is_tolerated(self):
        para = "The cat sat.  The dog ran."
        sents = ["The cat sat.", "The dog ran."]
        self.assertTrue(check_split_idempotent(para, sents))

    def test_segment_output_is_idempotent(self):
        para = "The cat sat. The dog ran. The bird flew."
        sents = [u.display.strip() for u in segment(para)]
        self.assertTrue(check_split_idempotent(para, sents))

    def test_multiline_paragraph_is_idempotent(self):
        para = "First sentence.\nSecond sentence."
        sents = [u.display.strip() for u in segment(para)]
        self.assertTrue(check_split_idempotent(para, sents))



@_spacy_skip
class TestSpacySentenceBackend(unittest.TestCase):

    def _seg(self, text: str):
        return segment(text, backend="spacy")

    def test_returns_list_of_units(self):
        units = self._seg("Hello world.")
        self.assertIsInstance(units, list)
        self.assertTrue(all(isinstance(u, Unit) for u in units))

    def test_unit_type_is_sentence(self):
        for u in self._seg("The cat sat. The dog ran."):
            self.assertEqual(u.type, "sentence")

    def test_empty_input_returns_empty(self):
        self.assertEqual(self._seg(""), [])

    def test_whitespace_only_returns_empty(self):
        self.assertEqual(self._seg("   "), [])

    def test_single_sentence(self):
        self.assertEqual(len(self._seg("The cat sat.")), 1)

    def test_two_sentences(self):
        self.assertEqual(len(self._seg("The cat sat. The dog ran.")), 2)

    def test_normalized_is_lowercase(self):
        units = self._seg("Hello World.")
        self.assertEqual(units[0].normalized, "hello world.")

    def test_normalized_collapses_whitespace(self):
        units = self._seg("Hello  world.")
        self.assertNotIn("  ", units[0].normalized)

    def test_display_idempotency_two_sentences(self):
        text = "The cat sat. The dog ran."
        expected = insert_missing_punctuation(text)
        got = "".join(u.display for u in self._seg(text))
        self.assertEqual(got, expected)

    def test_display_idempotency_newline(self):
        text = "The cat sat.\nThe dog ran."
        expected = insert_missing_punctuation(text)
        got = "".join(u.display for u in self._seg(text))
        self.assertEqual(got, expected)

    def test_display_idempotency_no_punct(self):
        text = "First line\nSecond line"
        expected = insert_missing_punctuation(text)
        got = "".join(u.display for u in self._seg(text))
        self.assertEqual(got, expected)

    def test_first_unit_no_added_leading_space(self):
        units = self._seg("The cat sat. The dog ran.")
        self.assertFalse(units[0].display.startswith(" "))

    def test_second_unit_carries_separator(self):
        units = self._seg("The cat sat. The dog ran.")
        self.assertTrue(units[1].display.startswith(" "))

if __name__ == "__main__":
    unittest.main()
