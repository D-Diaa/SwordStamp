"""Lossless NLTK Punkt sentence backend."""

from typing import List

import nltk

from segmentation import normalize_text, insert_missing_punctuation
from segmentation._unit import Unit


_punkt = None


def _get_punkt():
    """Load and cache the English Punkt tokenizer."""
    global _punkt
    if _punkt is None:
        try:
            _punkt = nltk.data.load('tokenizers/punkt_tab/english.pickle')
        except LookupError:
            _punkt = nltk.data.load('tokenizers/punkt/PY3/english.pickle')
    return _punkt


def segment_sentences_nltk(text: str) -> List[Unit]:
    """Split punctuation-fixed text with Punkt spans."""
    punct_fixed = insert_missing_punctuation(text)

    tokenizer = _get_punkt()
    spans = list(tokenizer.span_tokenize(punct_fixed))

    if not spans:
        return []

    # Lowercasing here would change Punkt abbreviation decisions.
    sent_tokens = nltk.sent_tokenize(punct_fixed)
    if len(spans) != len(sent_tokens):
        # Preserve sentence count when span tokenization diverges.
        units = []
        for i, sent in enumerate(sent_tokens):
            display = sent if i == 0 else ' ' + sent
            normalized_sent = normalize_text(sent)
            units.append(Unit("sentence", normalized_sent, display))
        return units

    units: List[Unit] = []
    prev_end = 0
    for i, (start, end) in enumerate(spans):
        if i == 0:
            display = punct_fixed[:end]
        else:
            display = punct_fixed[prev_end:end]
        normalized_sent = normalize_text(punct_fixed[start:end])
        units.append(Unit("sentence", normalized_sent, display))
        prev_end = end

    # Keep trailing text in the display stream.
    tail = punct_fixed[prev_end:]
    if tail:
        last = units[-1]
        units[-1] = Unit(last.type, last.normalized, last.display + tail)

    return units
