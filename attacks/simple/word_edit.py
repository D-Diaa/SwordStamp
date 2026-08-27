"""Word-edit attacks adapted from MarkLLM's ``evaluation/tools/text_editor.py``."""

import argparse
import random
import re

import torch
from tqdm import tqdm

from attacks.base import AttackConfig, AttackResult, save_dataset

_PUNCT_RE = re.compile(r"^[^\w]+$")
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "to", "of", "in",
    "on", "at", "by", "for", "with", "about", "as", "into", "through",
    "and", "but", "or", "nor", "so", "yet", "both", "either", "neither",
    "not", "no", "i", "you", "he", "she", "it", "we", "they", "this",
    "that", "these", "those",
}

_BERT_MODEL = "bert-base-uncased"
_bert_cache: dict = {}


def _load_bert(device: str):
    if device not in _bert_cache:
        from transformers import BertForMaskedLM, BertTokenizerFast
        tok = BertTokenizerFast.from_pretrained(_BERT_MODEL)
        mdl = BertForMaskedLM.from_pretrained(_BERT_MODEL).to(device).eval()
        _bert_cache[device] = (tok, mdl)
    return _bert_cache[device]


def _word_deletion(text: str, ratio: float, rng: random.Random) -> str:
    tokens = text.split()
    if not tokens:
        return text
    keep = [tok for tok in tokens if _PUNCT_RE.match(tok) or rng.random() >= ratio]
    return " ".join(keep) if keep else text


def _get_wordnet_synonym(word: str) -> str | None:
    try:
        from nltk.corpus import wordnet
    except ImportError:
        return None
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            name = lemma.name().replace("_", " ")
            if name.lower() != word.lower():
                return name
    return None


def _synonym_substitution(text: str, ratio: float, rng: random.Random) -> str:
    tokens = text.split()
    result = []
    for tok in tokens:
        word = re.sub(r"\W+$", "", tok)
        suffix = tok[len(word):]
        if (
            word
            and word.lower() not in _STOPWORDS
            and not _PUNCT_RE.match(tok)
            and rng.random() < ratio
        ):
            syn = _get_wordnet_synonym(word)
            result.append((syn if syn else word) + suffix)
        else:
            result.append(tok)
    return " ".join(result)


def _candidate_positions(tokens: list[str], ratio: float, rng: random.Random) -> list[int]:
    """Indices of content words selected for substitution."""
    return [
        i for i, tok in enumerate(tokens)
        if (
            re.sub(r"\W+$", "", tok)
            and tok.lower() not in _STOPWORDS
            and not _PUNCT_RE.match(tok)
            and rng.random() < ratio
        )
    ]


def _bert_substitute(text: str, ratio: float, rng: random.Random, tokenizer, model, device: str, top_k: int = 10) -> str:
    tokens = text.split()
    positions = _candidate_positions(tokens, ratio, rng)
    if not positions:
        return text

    result = tokens[:]
    for pos in positions:
        word = re.sub(r"\W+$", "", tokens[pos])
        suffix = tokens[pos][len(word):]
        masked = result[:pos] + [tokenizer.mask_token] + result[pos + 1:]
        input_str = " ".join(masked)
        enc = tokenizer(input_str, return_tensors="pt", truncation=True, max_length=512).to(device)
        mask_idx = (enc.input_ids[0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
        if len(mask_idx) == 0:
            continue
        with torch.no_grad():
            logits = model(**enc).logits[0, mask_idx[0]]
        top_ids = logits.topk(top_k).indices.tolist()
        for tid in top_ids:
            candidate = tokenizer.decode([tid]).strip()
            if candidate.isalpha() and candidate.lower() != word.lower() and candidate.lower() not in _STOPWORDS:
                result[pos] = candidate + suffix
                break

    return " ".join(result)


def _context_synonym_batch(
    texts: list[str],
    ratio: float,
    device: str,
    seed: int = 42,
    top_k: int = 10,
) -> list[str]:
    tokenizer, model = _load_bert(device)
    rng = random.Random(seed)
    return [
        _bert_substitute(t, ratio, rng, tokenizer, model, device, top_k)
        for t in tqdm(texts, desc="context_synonym")
    ]


def _apply(texts: list[str], attack_fn, ratio: float, seed: int = 42) -> list[str]:
    rng = random.Random(seed)
    return [attack_fn(t, ratio, rng) for t in tqdm(texts, desc=attack_fn.__name__)]


def run_word_deletion_attack(texts: list[str], config: AttackConfig) -> AttackResult:
    output = _apply(texts, _word_deletion, config.word_edit_ratio)
    save_path = config.output_path
    if save_path is not None:
        save_dataset(texts, output, save_path)
    return AttackResult(texts, output, save_path=save_path)


def run_synonym_substitution_attack(texts: list[str], config: AttackConfig) -> AttackResult:
    output = _apply(texts, _synonym_substitution, config.word_edit_ratio)
    save_path = config.output_path
    if save_path is not None:
        save_dataset(texts, output, save_path)
    return AttackResult(texts, output, save_path=save_path)


def run_context_synonym_attack(texts: list[str], config: AttackConfig) -> AttackResult:
    output = _context_synonym_batch(
        texts,
        ratio=config.word_edit_ratio,
        device=config.device,
    )
    save_path = config.output_path
    if save_path is not None:
        save_dataset(texts, output, save_path)
    return AttackResult(texts, output, save_path=save_path)


def _parse_args():
    parser = argparse.ArgumentParser(description="Smoke-test word-level attacks.")
    parser.add_argument(
        "--attack",
        choices=["word_deletion", "synonym_substitution", "context_synonym"],
        default="context_synonym",
    )
    parser.add_argument("--ratio", type=float, default=0.3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--text", action="append")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    texts = args.text or [
        "The satellite crossed the night sky before sunrise.",
        "Researchers used the telescope to calibrate their instruments carefully.",
    ]
    rng = random.Random(42)
    if args.attack == "word_deletion":
        results = [_word_deletion(t, args.ratio, rng) for t in texts]
    elif args.attack == "synonym_substitution":
        results = [_synonym_substitution(t, args.ratio, rng) for t in texts]
    else:
        results = _context_synonym_batch(texts, args.ratio, device=args.device)
    for text, result in zip(texts, results):
        print(f"original : {text}")
        print(f"attacked : {result}\n")
