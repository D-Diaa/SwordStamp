"""Generation scoring adapted from original SemStamp's ``sampling.py``."""

from dataclasses import dataclass
from typing import Any, Callable

import torch
from sentence_transformers import SentenceTransformer
from sampling import CandidateScore, Region
from segmentation import Unit
from watermarking.primitives import (
    SBERTLSHModel,
    hash_key,
    get_mask_from_seed,
    compute_lsh_margins,
    get_cluster_mask,
    get_cluster_id,
    compute_kmeans_margins,
)

_LSH_MODES = frozenset({"lsh", "lsh_fixed", "lsh_fixed_diverse"})
_KMEANS_MODES = frozenset({"kmeans", "kmeans_fixed", "kmeans_fixed_diverse"})


def _candidate_scores(green, boundary_depths, yellow=None):
    """Return explicit region labels and nonnegative boundary depths."""
    if yellow is None:
        yellow = [False] * len(green)
    return [
        CandidateScore(
            Region.YELLOW if is_yellow else Region.GREEN if is_green else Region.RED,
            float(depth),
        )
        for is_green, is_yellow, depth in zip(green, yellow, boundary_depths)
    ]


@dataclass(frozen=True)
class ScoreFnFactory:
    """Build score functions while exposing their one shared text encoder."""

    build: Callable[[], Callable]
    encoder: Any

    def __call__(self):
        return self.build()


def _make_score_fn(get_accept_set, score_candidates):
    """Compose mask selection and candidate scoring.

    ``get_accept_set`` returns ``(accept_set, yellow_region)``. ``yellow_region``
    is the predecessor unit's region under the ``*_fixed_diverse`` modes and
    ``None`` everywhere else; the mask itself is never narrowed, so it keeps
    matching what detection counts against.
    """
    def score_fn(predecessor: Unit, candidates: list, unit_idx: int) -> list:
        accept_set, yellow_region = get_accept_set(predecessor)
        texts = [unit.normalized for unit in candidates]
        green, yellow, boundary_depths = score_candidates(
            texts, accept_set, yellow_region,
        )
        return _candidate_scores(green, boundary_depths, yellow)

    return score_fn


def _split_tiers(regions, accept_set, yellow_region):
    """Split in-mask regions into green and yellow candidate tiers."""
    in_mask = [r in accept_set for r in regions]
    yellow = [ok and r == yellow_region for ok, r in zip(in_mask, regions)]
    green = [ok and not is_yellow for ok, is_yellow in zip(in_mask, yellow)]
    return green, yellow


def create_lsh_score_fn(
    lsh_model,
    lsh_dim,
    lmbd,
    fixed_seed=None,
    key=None,
    diverse=False,
):
    """Watermark score_fn under an LSH partition. See :func:`_make_score_fn`."""
    if key is None:
        key = hash_key

    if fixed_seed is not None:
        _fixed = set(get_mask_from_seed(lsh_dim, lmbd, fixed_seed, key=key).tolist())
        if diverse:
            # Non-lazy walk: a candidate in the predecessor's region is demoted
            # to the yellow tier, so a copy of the last committed unit can never
            # win on greenness. Generation-only — the mask is unchanged, and
            # detection still counts yellow units as hits.
            def get_accept_set(predecessor):
                prev = lsh_model.get_hash([predecessor.normalized])[0]
                return _fixed, prev
        else:
            get_accept_set = lambda _predecessor: (_fixed, None)
    else:
        # Recompute the mask from the previously committed unit.
        def get_accept_set(predecessor):
            seed = lsh_model.get_hash([predecessor.normalized])[0]
            return set(get_mask_from_seed(lsh_dim, lmbd, seed, key=key).tolist()), None

    def score_candidates(texts, accept_set, yellow_region):
        embeds = lsh_model.get_embeddings(texts)
        margins = compute_lsh_margins(lsh_model, texts, embeds=embeds).tolist()
        hashes = lsh_model.get_hash(texts, embeds=embeds)
        green, yellow = _split_tiers(hashes, accept_set, yellow_region)
        return green, yellow, margins

    return _make_score_fn(get_accept_set, score_candidates)


def create_kmeans_score_fn(
    embedder,
    cluster_centers,
    k_dim,
    lmbd,
    fixed_cluster_id=None,
    key=None,
    diverse=False,
):
    """Watermark score_fn under a KMeans partition. See :func:`_make_score_fn`."""
    if key is None:
        key = hash_key

    if fixed_cluster_id is not None:
        _fixed = set(get_cluster_mask(fixed_cluster_id, k_dim, lmbd, key=key).tolist())
        if diverse:
            # Non-lazy walk; see the LSH branch for the rationale.
            def get_accept_set(predecessor):
                prev = int(get_cluster_id(predecessor.normalized,
                                          cluster_centers, embedder))
                return _fixed, prev
        else:
            get_accept_set = lambda _predecessor: (_fixed, None)
    else:
        def get_accept_set(predecessor):
            cid = get_cluster_id(predecessor.normalized,
                                 cluster_centers, embedder)
            return set(get_cluster_mask(cid, k_dim, lmbd, key=key).tolist()), None

    def score_candidates(texts, accept_set, yellow_region):
        margins, cluster_ids = compute_kmeans_margins(texts, embedder, cluster_centers)
        green, yellow = _split_tiers(
            cluster_ids.tolist(), accept_set, yellow_region,
        )
        return green, yellow, margins.tolist()

    return _make_score_fn(get_accept_set, score_candidates)


def setup_lsh_mode(wm, device, batch_size):
    """Build the LSH score_fn factory from a :class:`config.schema.WatermarkConfig`."""
    lsh_model = SBERTLSHModel(
        lsh_model_path=wm.embedder, device=device, batch_size=batch_size,
        lsh_dim=wm.sp_dim, sbert_type='base',
    )
    fixed_seed = (
        lsh_model.get_hash([wm.secret_message])[0]
        if wm.mode in ("lsh_fixed", "lsh_fixed_diverse") else None
    )
    return ScoreFnFactory(
        lambda: create_lsh_score_fn(
            lsh_model, wm.sp_dim, wm.lmbd, fixed_seed,
            key=wm.hash_key,
            diverse=wm.mode.endswith("_diverse"),
        ),
        lsh_model.embedder,
    )


def setup_kmeans_mode(wm, device):
    """Build the KMeans score_fn factory from a :class:`config.schema.WatermarkConfig`."""
    cluster_centers = torch.load(wm.cc_path)
    embedder = SentenceTransformer(wm.embedder, device=device)
    fixed_cluster_id = (
        get_cluster_id(wm.secret_message, cluster_centers, embedder)
        if wm.mode in ("kmeans_fixed", "kmeans_fixed_diverse") else None
    )
    return ScoreFnFactory(
        lambda: create_kmeans_score_fn(
            embedder, cluster_centers, wm.sp_dim, wm.lmbd, fixed_cluster_id,
            key=wm.hash_key,
            diverse=wm.mode.endswith("_diverse"),
        ),
        embedder,
    )


def setup_score_factory(wm, device, batch_size=1):
    """Build the score_fn factory for one watermark mode.

    Shared by defender generation and by attackers that are given the
    partition, so both sides score against exactly the same code path.
    """
    if wm.mode in _LSH_MODES:
        return setup_lsh_mode(wm, device, batch_size)
    if wm.mode in _KMEANS_MODES:
        return setup_kmeans_mode(wm, device)
    raise NotImplementedError(f"Unknown watermark mode: {wm.mode!r}")


def create_none_score_fn():
    """Return a scorer that accepts every candidate."""
    def score_fn(predecessor: Unit, candidates: list, unit_idx: int) -> list:
        return [CandidateScore(Region.GREEN, 1.0) for _ in candidates]
    return score_fn


def _main():
    """Run CPU-only scoring smoke tests."""
    from segmentation import segment

    none_fn = create_none_score_fn()
    text = "Scientists discovered a new species of deep-sea fish. Its patterns have never been seen before."
    units = segment(text)
    scores = none_fn(Unit("sentence", "some prior context sentence.", "Prior."), units, 0)
    assert all(s == CandidateScore(Region.GREEN, 1.0) for s in scores), scores
    print(f"create_none_score_fn: {len(units)} candidate(s) → {scores}")

    green = [True, False]
    margins = [0.8, 0.4]
    scores2 = _candidate_scores(green, margins)
    assert scores2[0].region is Region.GREEN, scores2
    assert scores2[1].region is Region.RED, scores2
    print(f"_candidate_scores: {scores2[0]} (green), {scores2[1]} (red)")

    print("scoring smoke ok")


if __name__ == "__main__":
    _main()
