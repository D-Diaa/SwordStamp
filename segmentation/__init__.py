"""Public sentence and semantic-span segmentation API."""

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Sequence
from segmentation._unit import Unit

_HORIZONTAL_WS_RE = re.compile(r"[^\S\n]+")
_NO_PREFIX_SPACE_BEFORE = frozenset('.,;:!?)]}’”\'"')


_PUNCTS = frozenset(".!?")
# Closers allowed after terminal punctuation.
_CLOSING_TRAIL = "\"'”’»)]}"


def insert_missing_punctuation(text: str) -> str:
    """Add periods to non-empty lines without terminal punctuation."""
    lines = text.split("\n")
    fixed: list[str] = []
    for line in lines:
        stripped = line.rstrip()
        core = stripped.rstrip(_CLOSING_TRAIL)
        if stripped and (not core or core[-1] not in _PUNCTS):
            stripped += "."
        fixed.append(stripped)
    return "\n".join(fixed)


def normalize_text(text: str) -> str:
    """Normalize whitespace and case for scoring."""
    text = text.replace(" ", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def check_split_idempotent(paragraph: str, sentences: List[str]) -> bool:
    """Check reconstruction after normalizing whitespace."""
    joined = " ".join(re.sub(r"\s+", " ", s).strip() for s in sentences)
    original = re.sub(r"\s+", " ", paragraph).strip()
    return joined == original


DEFAULT_TYPE = "sentence"
DEFAULT_BACKEND = "nltk"
DEFAULT_SEMCUT_MAX_WORDS = 15
# The window doubles as the minimum unit size: a gap is legal only when it
# leaves at least this many content words on each side, so the text compared
# across a candidate boundary is exactly the text guaranteed to stay there.
DEFAULT_SEMCUT_WINDOW = 5
DEFAULT_SEMCUT_BATCH_SIZE = 512
SUPPORTED_TYPES = ("sentence", "semspan")
SUPPORTED_BACKENDS = ("nltk", "spacy")


def _validate_segmentation(type: str, backend: str) -> None:
    """Validate a segmentation type and backend."""
    if type not in SUPPORTED_TYPES or backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported combination: type={type!r}, backend={backend!r}. "
            f"Supported types: {SUPPORTED_TYPES}. "
            f"Supported backends: {SUPPORTED_BACKENDS}."
        )


def validate_semcut_policy(max_words: int, window: int) -> None:
    """Validate boundary-affecting semantic-cut parameters."""
    for name, value in (
        ("semcut_max_words", max_words),
        ("semcut_window", window),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value!r}")
    if max_words < 2 * window:
        raise ValueError(
            "semcut_max_words must be at least twice semcut_window, "
            f"got {max_words} < 2 * {window}"
        )


def resolve_segmentation_type(config: Any = None, segmentation_type: str | None = None) -> str:
    """Resolve an explicit or configured segmentation type."""
    if segmentation_type is not None:
        return segmentation_type
    if config is None:
        return DEFAULT_TYPE
    return getattr(config, "type", getattr(config, "segmentation_type", DEFAULT_TYPE))


def resolve_segmentation_backend(config: Any = None, segmentation_backend: str | None = None) -> str:
    """Resolve an explicit or configured segmentation backend."""
    if segmentation_backend is not None:
        return segmentation_backend
    if config is None:
        return DEFAULT_BACKEND
    return getattr(
        config, "segmentation_backend", getattr(config, "backend", DEFAULT_BACKEND),
    )


def segmentation_kwargs(
    config: Any = None,
    *,
    segmentation_type: str | None = None,
    segmentation_backend: str | None = None,
    semantic_encoder: str | None = None,
    semcut_max_words: int | None = None,
    semcut_window: int | None = None,
) -> Dict[str, Any]:
    """Build serializable segmentation provenance."""
    values = {
        "type": resolve_segmentation_type(config, segmentation_type),
        "backend": resolve_segmentation_backend(config, segmentation_backend),
    }
    semcut_max_words = (
        getattr(config, "semcut_max_words", DEFAULT_SEMCUT_MAX_WORDS)
        if semcut_max_words is None else semcut_max_words
    )
    semcut_window = (
        getattr(config, "semcut_window", DEFAULT_SEMCUT_WINDOW)
        if semcut_window is None else semcut_window
    )
    if values["type"] == "semspan" and semantic_encoder is not None:
        values["semantic_encoder"] = semantic_encoder
    if values["type"] == "semspan":
        validate_semcut_policy(semcut_max_words, semcut_window)
        values.update({
            "semcut_max_words": semcut_max_words,
            "semcut_window": semcut_window,
        })
    return values


def segment(text: str, type: str = DEFAULT_TYPE, backend: str = DEFAULT_BACKEND,
            *, encode=None,
            semcut_max_words: int = DEFAULT_SEMCUT_MAX_WORDS,
            semcut_window: int = DEFAULT_SEMCUT_WINDOW) -> List[Unit]:
    """Split text into sentence or semantic-span units."""
    _validate_segmentation(type, backend)
    if not text or not text.strip():
        return []
    if type == "sentence":
        if backend == "nltk":
            from segmentation._backends._nltk import segment_sentences_nltk
            return segment_sentences_nltk(text)
        from segmentation._backends._spacy import segment_sentences_spacy
        return segment_sentences_spacy(text)
    if type == "semspan":
        from segmentation._backends._semspan import segment_semspans
        return segment_semspans(
            text,
            sentence_backend=backend,
            encode=encode,
            max_words=semcut_max_words,
            window=semcut_window,
        )
    raise AssertionError(f"Validated segmentation type was not dispatched: {type!r}")


def first_unit(text: str, type: str = DEFAULT_TYPE, backend: str = DEFAULT_BACKEND,
               *, encode=None,
               semcut_max_words: int = DEFAULT_SEMCUT_MAX_WORDS,
               semcut_window: int = DEFAULT_SEMCUT_WINDOW) -> Unit:
    """Return the first unit in text."""
    units = segment(
        text, type=type, backend=backend, encode=encode,
        semcut_max_words=semcut_max_words,
        semcut_window=semcut_window,
    )
    return units[0] if units else Unit()


def first_units(texts: List[str], type: str = DEFAULT_TYPE,
                backend: str = DEFAULT_BACKEND, *, encode=None,
                semcut_max_words: int = DEFAULT_SEMCUT_MAX_WORDS,
                semcut_window: int = DEFAULT_SEMCUT_WINDOW) -> List[Unit]:
    """Return the first unit from each text."""
    _validate_segmentation(type, backend)
    if not texts:
        return []
    if type == "semspan":
        from segmentation._backends._semspan import first_semspan_batch
        return first_semspan_batch(
            texts,
            sentence_backend=backend,
            encode=encode,
            max_words=semcut_max_words,
            window=semcut_window,
        )
    return [
        first_unit(
            t, type=type, backend=backend, encode=encode,
            semcut_max_words=semcut_max_words,
            semcut_window=semcut_window,
        )
        for t in texts
    ]


def sentence_transformer_encode(encoder, batch_size: int | None = None):
    """Adapt one SentenceTransformer instance to semcut's NumPy batch API."""
    if batch_size is None:
        batch_size = DEFAULT_SEMCUT_BATCH_SIZE
    if batch_size <= 0:
        raise ValueError(f"semcut_batch_size must be positive, got {batch_size!r}")

    def encode(texts: Sequence[str]):
        return encoder.encode(
            list(texts), show_progress_bar=False, batch_size=batch_size,
            convert_to_numpy=True,
        )
    return encode


@dataclass(frozen=True)
class Segmenter:
    """A segmentation policy bound to its semantic encoder dependency."""

    type: str = DEFAULT_TYPE
    backend: str = DEFAULT_BACKEND
    encode: Callable[[Sequence[str]], Any] | None = field(
        default=None, repr=False, compare=False,
    )
    encoder_id: str | None = None
    semcut_max_words: int = DEFAULT_SEMCUT_MAX_WORDS
    semcut_window: int = DEFAULT_SEMCUT_WINDOW

    def __post_init__(self) -> None:
        _validate_segmentation(self.type, self.backend)
        validate_semcut_policy(
            self.semcut_max_words, self.semcut_window,
        )
        if self.type == "semspan" and self.encode is None:
            raise ValueError(
                "semspan requires an explicit semantic encoder; use "
                "Segmenter.from_sentence_transformer(...)"
            )

    @classmethod
    def from_sentence_transformer(
        cls,
        type: str,
        backend: str,
        encoder,
        encoder_id: str,
        *,
        batch_size: int | None = None,
        semcut_max_words: int = DEFAULT_SEMCUT_MAX_WORDS,
        semcut_window: int = DEFAULT_SEMCUT_WINDOW,
    ) -> "Segmenter":
        if type == "semspan" and encoder is None:
            raise ValueError("semspan requires an explicit semantic encoder")
        encode = (
            sentence_transformer_encode(encoder, batch_size)
            if type == "semspan" else None
        )
        return cls(
            type,
            backend,
            encode=encode,
            encoder_id=encoder_id if type == "semspan" else None,
            semcut_max_words=semcut_max_words,
            semcut_window=semcut_window,
        )

    @classmethod
    def from_config(
        cls,
        config,
        *,
        encoder=None,
        encoder_id: str | None = None,
        batch_size: int | None = None,
    ) -> "Segmenter":
        """Construct a segmenter from a typed or runtime config object."""
        type = resolve_segmentation_type(config)
        backend = resolve_segmentation_backend(config)
        policy = {
            "semcut_max_words": getattr(
                config, "semcut_max_words", DEFAULT_SEMCUT_MAX_WORDS,
            ),
            "semcut_window": getattr(
                config, "semcut_window", DEFAULT_SEMCUT_WINDOW,
            ),
        }
        if type == "semspan":
            return cls.from_sentence_transformer(
                type, backend, encoder, encoder_id, batch_size=batch_size, **policy,
            )
        return cls(type, backend, **policy)

    def segment(self, text: str) -> List[Unit]:
        return segment(
            text, type=self.type, backend=self.backend, encode=self.encode,
            semcut_max_words=self.semcut_max_words,
            semcut_window=self.semcut_window,
        )

    def first_unit(self, text: str) -> Unit:
        return first_unit(
            text, type=self.type, backend=self.backend, encode=self.encode,
            semcut_max_words=self.semcut_max_words,
            semcut_window=self.semcut_window,
        )

    def first_units(self, texts: List[str]) -> List[Unit]:
        return first_units(
            texts, type=self.type, backend=self.backend, encode=self.encode,
            semcut_max_words=self.semcut_max_words,
            semcut_window=self.semcut_window,
        )

    def metadata(self) -> Dict[str, Any]:
        return segmentation_kwargs(
            segmentation_type=self.type,
            segmentation_backend=self.backend,
            semantic_encoder=self.encoder_id,
            semcut_max_words=self.semcut_max_words,
            semcut_window=self.semcut_window,
        )


def normalize_generated_whitespace(text: str) -> str:
    """Collapse horizontal whitespace in decoded text."""
    return _HORIZONTAL_WS_RE.sub(" ", text.replace("\xa0", " "))


def display_with_boundary_space(context: str, display: str) -> str:
    """Separate a unit from adjacent context when needed."""
    if (
        context
        and display
        and not context[-1].isspace()
        and not display[0].isspace()
        and display[0] not in _NO_PREFIX_SPACE_BEFORE
    ):
        return " " + display
    return display


def _main():
    """Run segmentation smoke tests."""
    para = (
        "The history of the printing press is fascinating"
        "\nIt transformed how knowledge spread across Europe."
        "\nEvery scholar who could afford it wanted a printed copy."
    )

    fixed = insert_missing_punctuation(para)
    assert fixed.split("\n")[0].endswith(".")
    print(f"insert_missing_punctuation (first line): {fixed.split(chr(10))[0]!r}")

    units = segment(fixed)
    print(f"segment: {len(units)} unit(s)")
    for u in units:
        print(f"  normalized={u.normalized!r}")

    reconstructed = "".join(u.display for u in units)
    assert reconstructed == fixed, f"reconstruction mismatch:\n  got:      {reconstructed!r}\n  expected: {fixed!r}"
    print("display reconstruction: ok")

    first = first_unit(fixed)
    assert first.normalized == units[0].normalized
    print(f"first_unit: {first.normalized!r}")

    n = normalize_text("  Hello   World  ")
    assert n == "hello world", repr(n)
    print(f"normalize_text: {n!r}")

    assert check_split_idempotent(fixed, [u.display.strip() for u in units])
    print("check_split_idempotent: ok")

    print("segmentation smoke ok")




__all__ = [
    "DEFAULT_BACKEND",
    "DEFAULT_SEMCUT_BATCH_SIZE",
    "DEFAULT_SEMCUT_MAX_WORDS",
    "DEFAULT_SEMCUT_WINDOW",
    "DEFAULT_TYPE",
    "SUPPORTED_BACKENDS",
    "SUPPORTED_TYPES",
    "Segmenter",
    "Unit",
    "check_split_idempotent",
    "display_with_boundary_space",
    "first_unit",
    "first_units",
    "insert_missing_punctuation",
    "normalize_generated_whitespace",
    "normalize_text",
    "resolve_segmentation_backend",
    "resolve_segmentation_type",
    "segment",
    "segmentation_kwargs",
    "sentence_transformer_encode",
    "validate_semcut_policy",
]
