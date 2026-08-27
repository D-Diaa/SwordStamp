"""Single-channel probe attacks for channel-to-z attribution.

Each atom moves one structural channel and pins the others, so the resulting
design matrix has full rank. The model-based attack grid does not: its
conditions are almost collinear, which leaves per-channel effects on the
detection z-score unidentified.

Atoms pair up to separate segment *contamination* from segment *count*, since a
segment-aggregated z grows roughly with the square root of the segment count:

    sentence_deletion   drops whole units, leaves survivors intact -> pure sqrt(n)
    merge_adjacent      resegments, unit count falls
    split_midpoint      resegments, unit count rises
    clause_migrate      resegments at a fixed unit count
    permute_sentences   reorders without touching units or wording
    controlled_reorder  targets a chosen fraction of sentence inversions
    boundary_exchange   replaces boundaries while holding unit count fixed
    uniform_rechunk     maximal resegmentation at zero lexical cost

``random_content_sub`` and ``sentence_insertion`` splice in outside text, so they
draw their donor material from an *unwatermarked* corpus. The batch itself is not
a valid donor source: its units carry the watermark, and under a fixed mask the
green set is a constant key shared across documents, so a donor unit that was
green in its source document stays green in the recipient. Splicing those in adds
signal instead of diluting it, which inverts the channel being measured. An
outside corpus also rules out resampling a document's own vocabulary, which would
manufacture false anchors -- a token substituted elsewhere stays a hapax but
points at the wrong unit.
"""

import argparse
import math
import random
import re

from tqdm import tqdm

from attacks.base import AttackConfig, AttackResult, save_dataset
from segmentation import segment

_PUNCT_RE = re.compile(r"^[^\w]+$")
_TERMINAL = ".!?"
_WORD_RE = re.compile(r"[A-Za-z]{3,}")
_ALPHA_RE = re.compile(r"[A-Za-z]")
_CLOSING_RE = re.compile(r"(['\"\)\]]*)$")
_END_PUNCT_RE = re.compile(r"[.!?]+(?=(?:['\"\)\]]*)$)")
# A migrated clause is a quarter of its sentence, floored so short sentences
# still move something rather than silently opting out.
_MIGRATE_FRACTION = 4
_MIGRATE_MIN_WORDS = 3
_MIGRATE_MIN_SENTENCE = 8
_SPLIT_MIN_WORDS = 10


def _sentences(text: str) -> list[str]:
    """Sentence display strings via the shared segmentation package.

    ``Unit.display`` keeps the original leading whitespace so a document can be
    reconstructed losslessly. Atoms rebuild documents by joining on a single
    space, so that padding is stripped here to avoid emitting double spaces.
    """
    return [u.display.strip() for u in segment(text, type="sentence", backend="nltk")]


def _lower_first(sentence: str) -> str:
    return sentence[0].lower() + sentence[1:] if sentence else sentence


def _upper_first(sentence: str) -> str:
    return sentence[0].upper() + sentence[1:] if sentence else sentence


def _terminate(sentence: str) -> str:
    """Ensure a sentence ends in terminal punctuation."""
    stripped = sentence.rstrip()
    if not stripped:
        return stripped
    return stripped if stripped[-1] in _TERMINAL else stripped + "."


# --------------------------------------------------------------------------
# Unit-count atoms
# --------------------------------------------------------------------------

def sentence_deletion(text: str, ratio: float, rng: random.Random) -> str:
    """Delete whole sentences. Survivors are untouched, so this is pure sqrt(n)."""
    sents = _sentences(text)
    if len(sents) < 2:
        return text
    keep = [s for s in sents if rng.random() >= ratio]
    return " ".join(keep) if keep else sents[0]


def merge_adjacent(text: str, ratio: float, rng: random.Random) -> str:
    """Join adjacent sentence pairs. Wording is preserved; unit count falls."""
    sents = _sentences(text)
    out: list[str] = []
    i = 0
    while i < len(sents):
        if i + 1 < len(sents) and rng.random() < ratio:
            head = sents[i].rstrip().rstrip(_TERMINAL).rstrip()
            out.append(f"{head}, {_lower_first(sents[i + 1])}")
            i += 2
        else:
            out.append(sents[i])
            i += 1
    return " ".join(out)


def split_midpoint(text: str, ratio: float, rng: random.Random) -> str:
    """Cut sentences at their midpoint. Wording is preserved; unit count rises."""
    out: list[str] = []
    for sent in _sentences(text):
        words = sent.split()
        if rng.random() < ratio and len(words) >= _SPLIT_MIN_WORDS:
            cut = len(words) // 2
            head = " ".join(words[:cut]).rstrip(",;:")
            out.append(_terminate(head))
            out.append(_upper_first(" ".join(words[cut:])))
        else:
            out.append(sent)
    return " ".join(out)


# --------------------------------------------------------------------------
# Fixed-unit-count atoms
# --------------------------------------------------------------------------

def clause_migrate(text: str, ratio: float, rng: random.Random) -> str:
    """Move a sentence's tail clause onto the next sentence.

    Every boundary survives and the unit count is unchanged, but two units
    change content -- the case that separates resegmentation from rewording.
    """
    sents = _sentences(text)
    out = list(sents)
    for i in range(len(out) - 1):
        words = out[i].split()
        if rng.random() >= ratio or len(words) < _MIGRATE_MIN_SENTENCE:
            continue
        tail_len = max(_MIGRATE_MIN_WORDS, len(words) // _MIGRATE_FRACTION)
        cut = len(words) - tail_len
        out[i] = _terminate(" ".join(words[:cut]).rstrip(",;:"))
        tail = " ".join(words[cut:]).rstrip(_TERMINAL).rstrip(",;:")
        # The tail now opens its sentence, so it must be capitalized -- a
        # lowercase opener reads as a fragment and can make the segmenter
        # re-merge the pair, silently undoing the unit-count guarantee.
        out[i + 1] = f"{_upper_first(tail)}, {_lower_first(out[i + 1])}"
    return " ".join(out)


def permute_sentences(text: str, ratio: float, rng: random.Random) -> str:
    """Permute a fraction of whole sentences. Units and wording are untouched."""
    sents = _sentences(text)
    count = int(round(ratio * len(sents)))
    if count < 2:
        return " ".join(sents) if sents else text
    positions = sorted(rng.sample(range(len(sents)), count))
    moved = [sents[p] for p in positions]
    rng.shuffle(moved)
    for position, value in zip(positions, moved):
        sents[position] = value
    return " ".join(sents)


def _permutation_with_inversions(size: int, target: int) -> list[int]:
    """Construct a permutation with exactly ``target`` pair inversions.

    Choosing the ``digit``-th smallest remaining item contributes exactly that
    many inversions with items placed later. Greedily taking the largest legal
    digit therefore realizes every target from zero through ``size choose 2``.
    """
    maximum = size * (size - 1) // 2
    if target < 0 or target > maximum:
        raise ValueError(
            f"target inversions must be in [0, {maximum}], got {target}"
        )
    remaining = list(range(size))
    permutation: list[int] = []
    inversions_left = target
    while remaining:
        digit = min(inversions_left, len(remaining) - 1)
        permutation.append(remaining.pop(digit))
        inversions_left -= digit
    return permutation


def controlled_reorder(text: str, ratio: float, rng: random.Random) -> str:
    """Reorder whole sentences to target a fraction of possible inversions.

    ``ratio`` is not the fraction of sentences selected. It is the requested
    fraction of the ``n * (n - 1) / 2`` sentence-pair inversions: zero preserves
    the document, and one reverses it. Units, wording, boundaries, and unit
    count are unchanged. ``rng`` is unused because the target has a canonical,
    reproducible permutation.
    """
    del rng
    if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
        raise ValueError(f"controlled reorder ratio must be in [0, 1], got {ratio}")
    sents = _sentences(text)
    maximum = len(sents) * (len(sents) - 1) // 2
    target = min(maximum, int(ratio * maximum + 0.5))
    order = _permutation_with_inversions(len(sents), target)
    return " ".join(sents[i] for i in order) if sents else text


def _evenly_spaced_indices(size: int, count: int) -> list[int]:
    """Return ``count`` deterministic indices spread across ``range(size)``."""
    if count < 0 or count > size:
        raise ValueError(f"count must be in [0, {size}], got {count}")
    if count == 0:
        return []
    return [min(size - 1, int((i + 0.5) * size / count)) for i in range(count)]


def _strip_terminal_punctuation(text: str) -> str:
    return _END_PUNCT_RE.sub("", text)


def _add_period(text: str) -> str:
    stripped = _strip_terminal_punctuation(text)
    match = _CLOSING_RE.search(stripped)
    position = match.start() if match else len(stripped)
    return f"{stripped[:position]}.{stripped[position:]}"


def _upper_first_alpha(text: str) -> str:
    return _ALPHA_RE.sub(lambda match: match.group(0).upper(), text, count=1)


def _render_boundary_exchange(
    tokens: list[str],
    old_boundaries: list[int],
    new_boundaries: list[int],
) -> str:
    out = list(tokens)
    old, new = set(old_boundaries), set(new_boundaries)
    for boundary in old - new:
        out[boundary - 1] = _strip_terminal_punctuation(out[boundary - 1])
    for boundary in new - old:
        out[boundary - 1] = _add_period(out[boundary - 1])
    for start in [0, *sorted(new)]:
        if start < len(out):
            out[start] = _upper_first_alpha(out[start])
    return " ".join(out)


def boundary_exchange(text: str, ratio: float, rng: random.Random) -> str:
    """Move selected boundaries to preceding-sentence midpoints.

    Every movable selected boundary has exactly one replacement. Lexical tokens
    and their global order are unchanged. One-word sentences have no internal
    gap, so their selected boundary remains in place.
    """
    del rng
    if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
        raise ValueError(f"boundary exchange ratio must be in [0, 1], got {ratio}")
    source = _sentences(text)
    if len(source) < 2:
        return text
    tokenized = [sentence.split() for sentence in source]
    tokens = [token for sentence in tokenized for token in sentence]
    old_boundaries: list[int] = []
    cursor = 0
    for sentence in tokenized[:-1]:
        cursor += len(sentence)
        old_boundaries.append(cursor)
    exchange_count = min(
        len(old_boundaries), int(ratio * len(old_boundaries) + 0.5),
    )
    if exchange_count == 0:
        return text
    new_boundaries = list(old_boundaries)
    starts = [0, *old_boundaries]
    selected = _evenly_spaced_indices(len(old_boundaries), exchange_count)
    for index in selected:
        sentence_length = len(tokenized[index])
        if sentence_length >= 2:
            new_boundaries[index] = starts[index] + sentence_length // 2
    return _render_boundary_exchange(tokens, old_boundaries, new_boundaries)


def uniform_rechunk(text: str, words_per_chunk: int, rng: random.Random) -> str:
    """Discard sentence boundaries and re-cut into fixed-length chunks.

    Maximal resegmentation at near-zero lexical cost: the word sequence is
    preserved and only terminal punctuation and sentence-initial capitalization
    move. Trailing ``.!?`` is stripped from every word, so abbreviations lose
    their period. ``rng`` is unused, kept for a uniform atom signature.
    """
    del rng
    if words_per_chunk < 1:
        raise ValueError(f"words_per_chunk must be >= 1, got {words_per_chunk}")
    words = [w.rstrip(_TERMINAL) for w in " ".join(_sentences(text)).split()]
    words = [w for w in words if w]
    if not words:
        return text
    chunks = [
        " ".join(words[i:i + words_per_chunk])
        for i in range(0, len(words), words_per_chunk)
    ]
    return " ".join(_upper_first(_terminate(c)) for c in chunks)


# --------------------------------------------------------------------------
# Donor atoms (batch-level: material comes from other documents)
# --------------------------------------------------------------------------

def _content_words(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def random_content_sub(text: str, ratio: float, rng: random.Random,
                       donor_words: list[str]) -> str:
    """Replace content words with unrelated words drawn from other documents.

    Matched against ``synonym_substitution`` at equal measured rewording, this
    isolates semantic displacement from lexical displacement.
    """
    if not donor_words:
        return text
    own = {w.lower() for w in _content_words(text)}
    pool = [w for w in donor_words if w.lower() not in own] or donor_words

    def replace(match: re.Match) -> str:
        if rng.random() >= ratio:
            return match.group(0)
        word = rng.choice(pool)
        return _upper_first(word) if match.group(0)[0].isupper() else word

    return _WORD_RE.sub(replace, text)


def sentence_insertion(text: str, ratio: float, rng: random.Random,
                       donor_sentences: list[str]) -> str:
    """Splice unwatermarked sentences in from an outside corpus.

    Dilutes the signal with clean units rather than corrupting existing ones.
    ``donor_sentences`` must come from unwatermarked text; see the module
    docstring for why batch-internal donors invert this channel.
    """
    sents = _sentences(text)
    count = int(round(ratio * len(sents)))
    if not sents or not donor_sentences or count < 1:
        return " ".join(sents) if sents else text
    for _ in range(count):
        sents.insert(rng.randrange(len(sents) + 1), rng.choice(donor_sentences))
    return " ".join(sents)


# --------------------------------------------------------------------------
# Batch application
# --------------------------------------------------------------------------

#: Atoms parameterized by an edit ratio in [0, 1].
RATIO_ATOMS = {
    "sentence_deletion": sentence_deletion,
    "merge_adjacent": merge_adjacent,
    "split_midpoint": split_midpoint,
    "clause_migrate": clause_migrate,
    "permute_sentences": permute_sentences,
    "controlled_reorder": controlled_reorder,
    "boundary_exchange": boundary_exchange,
}

#: Atoms needing donor material pooled across the batch.
DONOR_ATOMS = {
    "random_content_sub": random_content_sub,
    "sentence_insertion": sentence_insertion,
}

#: Atoms parameterized by a chunk length in words.
CHUNK_ATOMS = {"uniform_rechunk": uniform_rechunk}

PROBE_ATOMS = tuple(RATIO_ATOMS) + tuple(DONOR_ATOMS) + tuple(CHUNK_ATOMS)


def _donor_pool(donor_texts: list[str], atom: str) -> list[str]:
    """Flatten an unwatermarked corpus into donor units for ``atom``."""
    if atom == "random_content_sub":
        return [w for t in donor_texts for w in _content_words(t)]
    return [s for t in donor_texts for s in _sentences(t)]


def apply_atom(texts: list[str], atom: str, *, ratio: float = 0.3,
               words_per_chunk: int = 20, seed: int = 42,
               donor_texts: list[str] | None = None) -> list[str]:
    """Apply one probe atom across a batch of documents.

    ``donor_texts`` is required by the splicing atoms and must be unwatermarked.
    """
    if atom not in PROBE_ATOMS:
        raise ValueError(f"Unknown probe atom {atom!r}; choose from {sorted(PROBE_ATOMS)}")
    rng = random.Random(seed)
    if atom in CHUNK_ATOMS:
        fn = CHUNK_ATOMS[atom]
        return [fn(t, words_per_chunk, rng) for t in tqdm(texts, desc=atom)]
    if atom in DONOR_ATOMS:
        if not donor_texts:
            raise ValueError(
                f"{atom!r} needs donor_texts from an unwatermarked corpus; "
                "donors taken from the attacked batch splice the watermark back in."
            )
        pool = _donor_pool(donor_texts, atom)
        if not pool:
            raise ValueError(f"{atom!r} donor corpus yielded no usable material")
        fn = DONOR_ATOMS[atom]
        return [fn(t, ratio, rng, pool) for t in tqdm(texts, desc=atom)]
    fn = RATIO_ATOMS[atom]
    return [fn(t, ratio, rng) for t in tqdm(texts, desc=atom)]


def _load_donor_texts(config: AttackConfig, atom: str) -> list[str]:
    """Read the unwatermarked donor corpus named by ``attack.donor_corpus``."""
    if not config.donor_corpus:
        raise ValueError(
            f"{atom!r} requires attack.donor_corpus pointing at an unwatermarked "
            "dataset; donors from the attacked batch splice the watermark back in."
        )
    from datasets import load_from_disk

    return list(load_from_disk(config.donor_corpus)["text"])


def _run(texts, config: AttackConfig, atom: str) -> AttackResult:
    donor_texts = _load_donor_texts(config, atom) if atom in DONOR_ATOMS else None
    output = apply_atom(
        texts,
        atom,
        ratio=config.word_edit_ratio,
        words_per_chunk=config.rechunk_words,
        donor_texts=donor_texts,
    )
    save_path = config.output_path
    if save_path is not None:
        save_dataset(texts, output, save_path)
    return AttackResult(texts, output, save_path=save_path)


def _make_runner(atom: str):
    def run(texts, config: AttackConfig) -> AttackResult:
        return _run(texts, config, atom)
    run.__name__ = f"run_{atom}_attack"
    run.__doc__ = f"Run the {atom!r} probe atom over a batch."
    return run


run_sentence_deletion_attack = _make_runner("sentence_deletion")
run_merge_adjacent_attack = _make_runner("merge_adjacent")
run_split_midpoint_attack = _make_runner("split_midpoint")
run_clause_migrate_attack = _make_runner("clause_migrate")
run_permute_sentences_attack = _make_runner("permute_sentences")
run_controlled_reorder_attack = _make_runner("controlled_reorder")
run_boundary_exchange_attack = _make_runner("boundary_exchange")
run_uniform_rechunk_attack = _make_runner("uniform_rechunk")
run_random_content_sub_attack = _make_runner("random_content_sub")
run_sentence_insertion_attack = _make_runner("sentence_insertion")


def _parse_args():
    parser = argparse.ArgumentParser(description="Smoke-test structural probe atoms.")
    parser.add_argument("--atom", choices=sorted(PROBE_ATOMS), default="clause_migrate")
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--words-per-chunk", type=int, default=20)
    parser.add_argument("--text", action="append")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    sample = args.text or [
        "The satellite crossed the night sky before sunrise. "
        "Observers logged its track from three separate stations. "
        "Cloud cover interrupted the final measurement window.",
        "Researchers calibrated the telescope through the afternoon. "
        "The mirror required a second alignment pass. "
        "They published the corrected figures the following week.",
    ]
    # The smoke test has no corpus on hand, so the second sample stands in as
    # the donor: enough to exercise the code path, never a valid experiment.
    results = apply_atom(
        sample, args.atom, ratio=args.ratio, words_per_chunk=args.words_per_chunk,
        donor_texts=sample[1:] or sample,
    )
    for original, attacked in zip(sample, results):
        print(f"original : {original}")
        print(f"attacked : {attacked}\n")
