"""Attack helpers adapted from original SemStamp's ``paraphrase_gen_utils.py``."""

import os
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import re
from collections import Counter
from itertools import groupby
from string import punctuation
from segmentation import DEFAULT_BACKEND, DEFAULT_TYPE, segment

PUNCTS = "!.?"
END_OF_PARAPHRASE_MARKER = "[[END OF PARAPHRASE]]"

# Match decorated variants of the full sentinel phrase.
_MARKER_RE = re.compile(r"end[\s_]+of[\s_]+paraphrase\b", re.IGNORECASE)
_MARKER_DECORATION = "[]'\" \t_"
# Also drop malformed trailing sentinel blocks.
_TRAILING_SENTINEL_RE = re.compile(r"\s*\[\[[^\[\]]*\]\]\s*$")


def strip_paraphrase_markers(text: str) -> str:
    """Strip valid and malformed end-of-paraphrase sentinels."""
    m = _MARKER_RE.search(text)
    if m is not None:
        start = m.start()
        # Remove decoration before the marker.
        while start > 0 and text[start - 1] in _MARKER_DECORATION:
            start -= 1
        text = text[:start]
    text = _TRAILING_SENTINEL_RE.sub("", text)
    return text.strip()


def batched(items, batch_size):
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def _first_upper(s):
    if len(s) == 0:
        return s
    return s[0].upper() + s[1:]


def _clean_text(s):
    punc = set(punctuation) - set(".")
    punc.add("\n")
    newtext = []
    for k, g in groupby(s):
        if k in punc:
            newtext.append(k)
        else:
            newtext.extend(g)
    return "".join(newtext)


def well_formed_sentence(sent, end_sent=False):
    sent = _first_upper(sent)
    sent = sent.replace("  ", " ")
    sent = sent.replace(" i ", " I ")
    if end_sent and len(sent) > 0 and sent[-1] not in PUNCTS:
        sent += "."
    return _clean_text(sent)


def split_texts_into_sentences(
    texts,
    segmentation_type=DEFAULT_TYPE,
    segmentation_backend=DEFAULT_BACKEND,
):
    sents, doc_lengths = [], []
    for text in texts:
        sent_list = [u.display.strip() for u in segment(text, type=segmentation_type, backend=segmentation_backend)]
        sents.extend(sent_list)
        doc_lengths.append(len(sent_list))
    return sents, doc_lengths


def join_sentences_by_document(sentences, doc_lengths):
    output = []
    start = 0
    for length in doc_lengths:
        output.append(" ".join(sentences[start:start + length]))
        start += length
    return output


def get_bert_scorer(device=None):
    """Create a BERTScorer instance."""
    from bert_score import BERTScorer

    device = device or "cuda"
    return BERTScorer(model_type="roberta-large", device=device, lang="en")


def _build_bigrams(input_ids):
    bigrams = []
    for i in range(len(input_ids) - 1):
        bigram = tuple(input_ids[i:i+2].tolist())
        bigrams.append(bigram)
    return bigrams


def _compare_ngram_overlap(input_ngram, para_ngram):
    input_c = Counter(input_ngram)
    para_c = Counter(para_ngram)
    intersection = list(input_c.keys() & para_c.keys())
    overlap = 0
    for i in intersection:
        overlap += para_c[i]
    return overlap


def _main():
    """Run text-cleaning smoke tests."""
    docs = [
        "the quick brown fox jumps over the lazy dog. an interesting fact about foxes is that they are cunning.",
        "marine biologists study whale migration. the patterns repeat every year.",
    ]

    sents, lengths = split_texts_into_sentences(docs)
    assert lengths == [2, 2], lengths
    print(f"split_texts_into_sentences: doc lengths={lengths}, total={len(sents)} sentences")
    for s in sents:
        print(f"  {s!r}")

    cleaned = [well_formed_sentence(s, end_sent=True) for s in sents]
    assert all(s[0].isupper() and s.endswith(".") for s in cleaned), cleaned
    print(f"well_formed_sentence: {cleaned}")

    raw = "This is a paraphrase output. [[END OF PARAPHRASE]] trailing garbage"
    stripped = strip_paraphrase_markers(raw)
    assert "[[END" not in stripped
    print(f"strip_paraphrase_markers: {stripped!r}")

    joined = join_sentences_by_document(cleaned, lengths)
    assert len(joined) == len(docs)
    print(f"join_sentences_by_document: {joined}")

    print("attacks/utils smoke ok")


def accept_by_bigram_overlap(sent, para_sents, tokenizer, bert_threshold=0.03, scorer=None):
    # BERTScore rejects empty strings.
    para_sents = [p for p in para_sents if p.strip()]
    if not para_sents:
        return sent
    device = "cuda"
    input_ids = tokenizer(sent, return_tensors='pt').input_ids[0].to(device)
    input_bigram = _build_bigrams(input_ids)
    para_ids = [tokenizer(para, return_tensors='pt').input_ids[0].to(device) for para in para_sents]
    para_bigrams = [_build_bigrams(para_id) for para_id in para_ids]
    min_overlap = len(input_ids)

    # Score all candidates in one batch.
    gen_repeated = [sent] * len(para_sents)
    if scorer is None:
        scorer = get_bert_scorer(device=device)
    _, _, F1 = scorer.score(gen_repeated, list(para_sents), batch_size=32)
    bert_scores = F1.tolist()

    # Rank candidates by BERTScore.
    sorted_indices = sorted(range(len(para_sents)), key=lambda i: bert_scores[i], reverse=True)
    para_sents = [para_sents[i] for i in sorted_indices]
    para_ids = [para_ids[i] for i in sorted_indices]
    para_bigrams = [para_bigrams[i] for i in sorted_indices]
    bert_scores = [bert_scores[i] for i in sorted_indices]

    max_score = bert_scores[0]
    best_paraphrased = para_sents[0]
    score_threshold = bert_threshold * max_score
    for i in range(len(para_bigrams)):
        para_bigram = para_bigrams[i]
        overlap = _compare_ngram_overlap(input_bigram, para_bigram)
        bert_score = bert_scores[i]
        diff = max_score - bert_score
        if overlap < min_overlap and len(para_ids[i]) <= 1.5 * len(input_ids) and (diff <= score_threshold):
            min_overlap = overlap
            best_paraphrased = para_sents[i]
    return best_paraphrased


if __name__ == "__main__":
    _main()
