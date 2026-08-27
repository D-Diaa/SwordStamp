"""Oracle paraphrasing: the closest rewrite that the detector would score as a miss.

Threat model. The attacker is given the defender's encoder *and* its green/red
partition, so it can evaluate detection exactly rather than approximating it
with a surrogate. Per unit it draws N paraphrase candidates and keeps the one
closest to the source unit, measured in the defender's own embedding space,
among those the detector would count as a miss. This is the strongest member of
the sample-and-select family: it buys the cheapest available region flip rather
than paying the mean displacement of a blind paraphraser.

The point of running it is to price that knowledge. The attacker still has to
find a faithful rewrite that lands red, so it pays a content-preservation cost
and an effort cost (candidates drawn per unit), both of which are recorded.

Selection is best-of-N only: one wave per unit, no rejection loop and no second
wave. When no candidate escapes, the attacker falls back to the adaptive
attacker's objective and commits the most displaced candidate, which is the one
sitting shallowest inside the green region.

Yellow is not a third region here. Detection counts yellow exactly as it counts
green, so the attacker does too; the yellow tier is a generation-side
constraint the detector never sees.

Deliberate simplifications:

* Greedy, no lookahead. Under context-dependent modes the committed unit
  reseeds the next unit's mask; the attacker does not plan over that chain.
* No attacker-side margin. Landing exactly on a hyperplane is a measure-zero
  event under sampling, so a red candidate is simply taken as red.
* No calibration toward the null. The attack minimizes hits rather than
  targeting a hit rate of ``lmbd``, so its z-scores run below the human null.

This module owns only the ranking objective; the sampling, segmentation, and
reassembly pipeline is :func:`attacks.paraphrasing.adaptive.unitwise_rewrite`.
"""

from typing import List

import torch

from config.runtime import DEFAULT_VLLM_UTILIZATION

from sampling.base_sampler import CandidateScore, Region
from segmentation import (
    DEFAULT_BACKEND,
    DEFAULT_SEMCUT_BATCH_SIZE,
    DEFAULT_SEMCUT_MAX_WORDS,
    DEFAULT_SEMCUT_WINDOW,
    DEFAULT_TYPE,
    Unit,
)
from attacks.base import AttackResult, save_dataset
from attacks.paraphrasing.adaptive import unitwise_rewrite
from attacks.utils import strip_paraphrase_markers
from watermarking.scoring import setup_score_factory


def _fidelity(similarity: float) -> float:
    """Map a cosine similarity to a nonnegative depth in ``[0, 1]``."""
    return min(max((float(similarity) + 1.0) / 2.0, 0.0), 1.0)


class OracleScorer:
    """Rank rewrites by fidelity, preferring the defender's red region.

    Wraps the defender's own score function, so the partition the attacker
    optimizes against is by construction the one detection uses.
    """

    def __init__(self, watermark, device: str = "cuda", batch_size: int = 32):
        factory = setup_score_factory(watermark, device, batch_size)
        self.region_score_fn = factory()
        self.encoder = factory.encoder
        # Fixed-key detection counts every unit. Context-dependent detection
        # spends unit 0 as the mask seed and never counts it, so at unit 0 the
        # attacker has nothing to evade and should optimize fidelity alone.
        self.scores_first_unit = "_fixed" in (watermark.mode or "")

    @torch.no_grad()
    def encode(self, texts: List[str]) -> torch.Tensor:
        """Return an ``(n, dim)`` normalized tensor in the defender's space."""
        return self.encoder.encode(
            texts, convert_to_tensor=True, normalize_embeddings=True,
        )

    def make_doc_score_fn(self, units: List[Unit]):
        """Bind one document's source units as fidelity anchors."""
        anchors = self.encode([unit.normalized for unit in units])
        n_units = len(units)

        def score_fn(predecessor, candidates, unit_idx):
            # Score the text the detector will actually see.
            stripped = [
                Unit(u.type, strip_paraphrase_markers(u.normalized), u.display)
                for u in candidates
            ]
            # Reuse the last anchor for units beyond the source length.
            anchor = anchors[min(unit_idx, n_units - 1)]
            sims = (self.encode([u.normalized for u in stripped]) @ anchor).tolist()
            fidelities = [_fidelity(s) for s in sims]

            if unit_idx == 0 and not self.scores_first_unit:
                return [CandidateScore(Region.GREEN, f) for f in fidelities]

            regions = self.region_score_fn(predecessor, stripped, unit_idx)
            scores = []
            for region_score, fidelity in zip(regions, fidelities):
                if region_score.region is Region.RED:
                    # A detector miss. Green tier outranks every hit, and
                    # best-of-n takes the deepest, so depth is fidelity.
                    scores.append(CandidateScore(Region.GREEN, fidelity))
                else:
                    # A detector hit (green, or yellow under *_fixed_diverse).
                    # The red tier is ranked by *shallowest* depth, so storing
                    # fidelity makes the fallback keep the most displaced
                    # candidate, which sits shallowest in the green region.
                    scores.append(CandidateScore(Region.RED, fidelity))
            return scores

        return score_fn


def oracle_attack_paraphrase(
    texts: List[str],
    base_model: str,
    watermark,
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
    """Paraphrase documents unitwise, keeping the closest red candidate."""
    scorer = OracleScorer(watermark, device=device, batch_size=num_candidates)
    return unitwise_rewrite(
        texts,
        base_model,
        encoder=scorer.encoder,
        encoder_id=watermark.embedder,
        make_doc_score_fn=scorer.make_doc_score_fn,
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
        raise ValueError("custom_model is required for oracle")
    if config.watermark is None:
        raise ValueError(
            "the oracle attack needs the defender's watermark config; it is "
            "supplied by attacks.__main__.build_config"
        )
    output = oracle_attack_paraphrase(
        texts, config.custom_model,
        config.watermark,
        prompt_style=config.prompt_style,
        num_candidates=config.num_candidates,
        backend=config.backend,
        device=config.device,
        temperature=config.temperature if config.temperature is not None else 1.0,
        segmentation_type=config.segmentation_type,
        segmentation_backend=config.segmentation_backend,
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

    parser = argparse.ArgumentParser(description="Smoke-test the oracle paraphrasing attack.")
    parser.add_argument(
        "--model",
        default=os.getenv("CUSTOM_PARAPHRASER_MODEL", "Qwen/Qwen2.5-3B-Instruct"),
        help="Base model or adapter path to test.",
    )
    parser.add_argument("--text", action="append", help="Text to paraphrase. May be passed multiple times.")
    parser.add_argument("--prompt", default="standard", choices=["standard", "shuffle", "combine", "sentence"])
    parser.add_argument("--num-candidates", "--num_candidates", dest="num_candidates", type=int, default=8)
    parser.add_argument("--backend", default="vllm", choices=["hf", "vllm"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", default="lsh_fixed")
    parser.add_argument("--seg-type", default="sentence", choices=["sentence", "semspan"])
    parser.add_argument("--seg-backend", default="nltk", choices=["nltk", "spacy"])
    return parser.parse_args()


def _main():
    import dataclasses

    from config.schema import WatermarkConfig

    args = _parse_smoke_args()
    texts = args.text or [
        "The satellite crossed the night sky before sunrise. Researchers used it to calibrate their instruments."
    ]
    watermark = dataclasses.replace(WatermarkConfig(), mode=args.mode)
    outputs = oracle_attack_paraphrase(
        texts,
        args.model,
        watermark,
        prompt_style=args.prompt,
        num_candidates=args.num_candidates,
        backend=args.backend,
        device=args.device,
        segmentation_type=args.seg_type,
        segmentation_backend=args.seg_backend,
    )
    for i, (text, para) in enumerate(zip(texts, outputs), start=1):
        print(f"\n[{i}] input:\n{text}")
        print(f"\n[{i}] output:\n{para}")


if __name__ == "__main__":
    _main()
