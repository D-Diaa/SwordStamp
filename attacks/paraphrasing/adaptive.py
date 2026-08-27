"""Adaptive paraphrasing by surrogate embedding displacement."""

from typing import List, Optional, Sequence

import torch
from transformers import GenerationConfig

from config.runtime import DEFAULT_VLLM_UTILIZATION

from sampling.base_sampler import (
    CandidateScore,
    Region,
    create_sampler,
    resolve_adapter,
)
from segmentation import (
    DEFAULT_BACKEND,
    DEFAULT_SEMCUT_BATCH_SIZE,
    DEFAULT_SEMCUT_MAX_WORDS,
    DEFAULT_SEMCUT_WINDOW,
    DEFAULT_TYPE,
    Segmenter,
    SUPPORTED_TYPES,
    resolve_segmentation_backend,
    resolve_segmentation_type,
)
from attacks.utils import strip_paraphrase_markers
from attacks.paraphrasing.custom import build_custom_prompt
from attacks.base import AttackResult, save_dataset
from sentence_transformers import SentenceTransformer


ANCHORS = ("positional", "bag")

DEFAULT_SURROGATE = "BAAI/bge-base-en-v1.5"


class SurrogateScorer:
    """Cosine-displacement scorer backed by one surrogate encoder."""

    def __init__(self, model_name: str = DEFAULT_SURROGATE, device: str = "cuda"):
        self.encoder = SentenceTransformer(model_name, device=device)

    @torch.no_grad()
    def encode_anchors(self, units: List[str]) -> torch.Tensor:
        """Return an ``(n_units, embedding_dim)`` normalized anchor tensor."""
        return self.encoder.encode(
            units, convert_to_tensor=True, normalize_embeddings=True,
        )

    def segmenter(self, segmentation_type: str, segmentation_backend: str,
                  model_name: str, *,
                  semcut_max_words: int = DEFAULT_SEMCUT_MAX_WORDS,
                  semcut_window: int = DEFAULT_SEMCUT_WINDOW,
                  semcut_batch_size: int = DEFAULT_SEMCUT_BATCH_SIZE) -> Segmenter:
        """Bind semantic cuts to this scorer's one surrogate encoder."""
        return Segmenter.from_sentence_transformer(
            segmentation_type, segmentation_backend, self.encoder, model_name,
            batch_size=semcut_batch_size,
            semcut_max_words=semcut_max_words,
            semcut_window=semcut_window,
        )

    @torch.no_grad()
    def score_displacement(
        self,
        anchor_embs: torch.Tensor,
        candidates: Sequence[str],
        anchor_index: Optional[int] = None,
        bag_agg: str = "min",
    ) -> List[float]:
        """Score candidate displacement from positional or bag anchors."""
        if bag_agg not in {"min", "mean"}:
            raise ValueError(f"bag_agg must be 'min' or 'mean', got {bag_agg!r}")
        if not candidates:
            return []
        anchors = anchor_embs
        if anchor_index is not None:
            anchors = anchors[anchor_index:anchor_index + 1]
        candidate_embs = self.encoder.encode(
            candidates, convert_to_tensor=True, normalize_embeddings=True,
        )
        distances = 1.0 - (candidate_embs @ anchors.T)
        scores = (
            distances.min(dim=1).values
            if bag_agg == "min"
            else distances.mean(dim=1)
        )
        # Identical candidate/anchor pairs round to a cosine above 1 in float32,
        # leaving a tiny negative distance that the depth contract rejects.
        return scores.clamp_min(0.0).cpu().tolist()


def unitwise_rewrite(
    texts: List[str],
    base_model: str,
    *,
    encoder,
    encoder_id: str,
    make_doc_score_fn,
    prompt_style: str = "standard",
    num_candidates: int = 32,
    backend: str = "vllm",
    device: str = "cuda",
    sentence_max_tokens: int = 96,
    temperature: float = 1.0,
    top_p: float = 0.95,
    max_doc_tokens: int = 512,
    segmentation_type: str = DEFAULT_TYPE,
    segmentation_backend: str = DEFAULT_BACKEND,
    semcut_max_words: int = DEFAULT_SEMCUT_MAX_WORDS,
    semcut_window: int = DEFAULT_SEMCUT_WINDOW,
    semcut_batch_size: int = DEFAULT_SEMCUT_BATCH_SIZE,
    vllm_utilization: float = DEFAULT_VLLM_UTILIZATION,
) -> List[str]:
    """Rewrite documents unitwise, picking each unit best-of-K.

    ``encoder`` owns both the attacker's segmentation and its scoring space.
    ``make_doc_score_fn`` receives one document's source units and returns the
    ``score_fn`` used for every candidate wave in that document; the ranking
    objective is the only thing that separates attackers built on this driver.
    """
    base, adapter = resolve_adapter(base_model)
    segmenter = Segmenter.from_sentence_transformer(
        segmentation_type, segmentation_backend, encoder, encoder_id,
        batch_size=semcut_batch_size,
        semcut_max_words=semcut_max_words,
        semcut_window=semcut_window,
    )
    # Reserve GPU memory for the attacker's encoder.
    backend_kwargs = (
        {"device": device}
        if backend == "hf"
        else {"gpu_memory_utilization": vllm_utilization}
    )
    sampler = create_sampler(
        backend,
        base,
        num_candidates=num_candidates,
        adapter_path=adapter,
        segmentation_type=segmentation_type,
        segmentation_backend=segmentation_backend,
        segmenter=segmenter,
        **backend_kwargs,
    )

    gen_config = GenerationConfig(
        max_new_tokens=sentence_max_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=sampler.tokenizer.pad_token_id,
    )

    # Precompute each document's prompt and scorer. The rewrite is free to
    # merge, split, or drop units, so only the token budget bounds it.
    doc_indices: List[int] = []
    prompts: List[str] = []
    score_fns: List = []
    for i, text in enumerate(texts):
        units = segmenter.segment(text)
        if not units:
            continue
        doc_indices.append(i)
        prompts.append(build_custom_prompt(sampler.tokenizer, prompt_style, text))
        score_fns.append(make_doc_score_fn(units))

    results = sampler.generate_batched_continuation(
        prompts, gen_config, score_fns, selection_mode="best-of-n",
        max_tokens=max_doc_tokens,
        progress=True,
    )

    output: List[str] = [""] * len(texts)
    for i, prompt, (full_text, _info) in zip(doc_indices, prompts, results):
        # Remove the chat prefix and normalize unit spacing.
        para = strip_paraphrase_markers(full_text[len(prompt):])
        output[i] = "".join(
            u.display
            for u in segmenter.segment(para)
            if u.normalized
        )
    return output


def adaptive_attack_paraphrase(
    texts: List[str],
    base_model: str,
    surrogate_model: str = DEFAULT_SURROGATE,
    prompt_style: str = "standard",
    num_candidates: int = 32,
    backend: str = "vllm",
    device: str = "cuda",
    sentence_max_tokens: int = 96,
    temperature: float = 1.0,
    top_p: float = 0.95,
    anchor: str = "positional",
    bag_agg: str = "min",
    max_doc_tokens: int = 512,
    segmentation_type: str = DEFAULT_TYPE,
    segmentation_backend: str = DEFAULT_BACKEND,
    semcut_max_words: int = DEFAULT_SEMCUT_MAX_WORDS,
    semcut_window: int = DEFAULT_SEMCUT_WINDOW,
    semcut_batch_size: int = DEFAULT_SEMCUT_BATCH_SIZE,
    vllm_utilization: float = DEFAULT_VLLM_UTILIZATION,
) -> List[str]:
    """Paraphrase documents unitwise with best-of-K displacement."""
    if anchor not in ANCHORS:
        raise ValueError(f"anchor must be one of {ANCHORS}, got {anchor!r}")
    scorer = SurrogateScorer(surrogate_model, device=device)

    def make_doc_score_fn(units):
        anchor_embs = scorer.encode_anchors([unit.normalized for unit in units])
        n_units = len(units)

        def score_fn(_predecessor, candidates, unit_idx):
            candidate_texts = [strip_paraphrase_markers(u.normalized) for u in candidates]
            if anchor == "bag":
                anchor_index = None
            else:
                # Reuse the last anchor for extra output units.
                idx = min(unit_idx, n_units - 1)
                anchor_index = idx
            depths = scorer.score_displacement(
                anchor_embs, candidate_texts,
                anchor_index=anchor_index, bag_agg=bag_agg,
            )
            return [CandidateScore(Region.GREEN, float(depth)) for depth in depths]

        return score_fn

    return unitwise_rewrite(
        texts,
        base_model,
        encoder=scorer.encoder,
        encoder_id=surrogate_model,
        make_doc_score_fn=make_doc_score_fn,
        prompt_style=prompt_style,
        num_candidates=num_candidates,
        backend=backend,
        device=device,
        sentence_max_tokens=sentence_max_tokens,
        temperature=temperature,
        top_p=top_p,
        max_doc_tokens=max_doc_tokens,
        segmentation_type=segmentation_type,
        segmentation_backend=segmentation_backend,
        semcut_max_words=semcut_max_words,
        semcut_window=semcut_window,
        semcut_batch_size=semcut_batch_size,
        vllm_utilization=vllm_utilization,
    )


def run_attack(texts, config):
    """Common attack interface."""
    if config.custom_model is None:
        raise ValueError("custom_model is required for adaptive")
    output = adaptive_attack_paraphrase(
        texts, config.custom_model,
        surrogate_model=config.surrogate_model,
        prompt_style=config.prompt_style,
        num_candidates=config.num_candidates,
        backend=config.backend,
        device=config.device,
        temperature=config.temperature if config.temperature is not None else 1.0,
        anchor=config.anchor,
        bag_agg=config.bag_agg,
        segmentation_type=resolve_segmentation_type(config),
        segmentation_backend=resolve_segmentation_backend(config),
        semcut_max_words=config.semcut_max_words,
        semcut_window=config.semcut_window,
        semcut_batch_size=config.semcut_batch_size,
        vllm_utilization=getattr(config, "vllm_utilization", DEFAULT_VLLM_UTILIZATION),
    )
    save_path = config.output_path
    if save_path is not None:
        save_dataset(texts, output, save_path)
    return AttackResult(texts, output, save_path=save_path)


def _parse_smoke_args():
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Smoke-test the adaptive paraphrasing attack.")
    parser.add_argument(
        "--model",
        default=os.getenv("CUSTOM_PARAPHRASER_MODEL", "Qwen/Qwen2.5-3B-Instruct"),
        help="Base model or adapter path to test.",
    )
    parser.add_argument("--text", action="append", help="Text to paraphrase. May be passed multiple times.")
    parser.add_argument("--prompt", default="standard", choices=["standard", "shuffle", "combine", "sentence"])
    parser.add_argument("--num-candidates", "--num_candidates", dest="num_candidates", type=int, default=2)
    parser.add_argument("--backend", default="vllm", choices=["hf", "vllm"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--surrogate-model", default=DEFAULT_SURROGATE)
    parser.add_argument("--anchor", default="positional", choices=list(ANCHORS))
    parser.add_argument("--bag-agg", default="min", choices=["min", "mean"])
    parser.add_argument(
        "--seg-type",
        default="sentence",
        choices=list(SUPPORTED_TYPES),
    )
    parser.add_argument("--seg-backend", default="nltk", choices=["nltk", "spacy"])
    return parser.parse_args()


def _main():
    args = _parse_smoke_args()
    texts = args.text or [
        "The satellite crossed the night sky before sunrise. Researchers used it to calibrate their instruments."
    ]
    outputs = adaptive_attack_paraphrase(
        texts,
        args.model,
        prompt_style=args.prompt,
        num_candidates=args.num_candidates,
        backend=args.backend,
        device=args.device,
        surrogate_model=args.surrogate_model,
        anchor=args.anchor,
        bag_agg=args.bag_agg,
        segmentation_type=args.seg_type,
        segmentation_backend=args.seg_backend,
    )
    for i, (text, para) in enumerate(zip(texts, outputs), start=1):
        print(f"\n[{i}] input:\n{text}")
        print(f"\n[{i}] output:\n{para}")


if __name__ == "__main__":
    _main()
