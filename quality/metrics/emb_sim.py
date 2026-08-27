"""Embedding-based pairwise similarity metric."""

import numpy as np

from .common import _confidence_interval


_EMB_SIM_MODEL_CACHE = {}


def _get_emb_sim_model(model_name, device):
    key = (model_name, device)
    if key not in _EMB_SIM_MODEL_CACHE:
        from sentence_transformers import SentenceTransformer

        _EMB_SIM_MODEL_CACHE[key] = SentenceTransformer(model_name, device=device)
    return _EMB_SIM_MODEL_CACHE[key]


def clear_emb_sim_model_cache():
    """Drop cached embedding-similarity models before starting memory-heavy work."""
    _EMB_SIM_MODEL_CACHE.clear()


def emb_sim_pairs(gens, refs, model=None, model_name="google/embeddinggemma-300m", batch_size=32, device=None):
    """Return cosine similarity for each generated/reference pair."""
    if model is None:
        if device is None:
            device = "cuda"
        model = _get_emb_sim_model(model_name, device)

    encode_kwargs = {
        "batch_size": batch_size,
        "convert_to_numpy": True,
        "normalize_embeddings": True,
        "show_progress_bar": False,
    }
    try:
        gen_emb = model.encode(list(gens), prompt_name="STS", **encode_kwargs)
        ref_emb = model.encode(list(refs), prompt_name="STS", **encode_kwargs)
    except (KeyError, ValueError):
        # Model doesn't declare an STS prompt - fall back to plain encoding.
        gen_emb = model.encode(list(gens), **encode_kwargs)
        ref_emb = model.encode(list(refs), **encode_kwargs)

    return np.sum(gen_emb * ref_emb, axis=1)


def evaluate_embedding_similarity(gens, refs, model_name="google/embeddinggemma-300m", batch_size=32, device=None):
    """Mean cosine similarity + 95% CI across gen/ref pairs."""
    sims = emb_sim_pairs(gens, refs, model_name=model_name, batch_size=batch_size, device=device)
    return {
        "emb_sim": float(np.mean(sims)),
        "emb_sim_ci": _confidence_interval(sims),
        "emb_sim_median": float(np.median(sims)),
        "emb_sim_per_sample": np.asarray(sims, dtype=float),
    }
