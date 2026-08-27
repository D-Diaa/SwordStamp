"""Semantic watermark generation and detection."""

from importlib import import_module


_EXPORTS = {
    "hash_key": ("watermarking.primitives", "hash_key"),
    "extract_prompt_from_text": ("watermarking.primitives", "extract_prompt_from_text"),
    "cosine_distance_matrix": ("watermarking.primitives", "cosine_distance_matrix"),
    "get_mask_from_seed": ("watermarking.primitives", "get_mask_from_seed"),
    "compute_lsh_margins": ("watermarking.primitives", "compute_lsh_margins"),
    "get_cluster_mask": ("watermarking.primitives", "get_cluster_mask"),
    "compute_kmeans_margins": ("watermarking.primitives", "compute_kmeans_margins"),
    "get_cluster_id": ("watermarking.primitives", "get_cluster_id"),
    "kmeans_predict": ("watermarking.primitives", "kmeans_predict"),
    "get_cluster_centers": ("watermarking.primitives", "get_cluster_centers"),
    "load_embeds": ("watermarking.primitives", "load_embeds"),
    "embed_gen_list": ("watermarking.primitives", "embed_gen_list"),
    "embedding_cache_path": ("watermarking.primitives", "embedding_cache_path"),
    "pairwise_cosine": ("watermarking.primitives", "pairwise_cosine"),
    "SBERTLSHModel": ("watermarking.primitives", "SBERTLSHModel"),
    "LSHModel": ("watermarking.primitives", "LSHModel"),
    "create_lsh_score_fn": ("watermarking.scoring", "create_lsh_score_fn"),
    "create_kmeans_score_fn": ("watermarking.scoring", "create_kmeans_score_fn"),
    "create_none_score_fn": ("watermarking.scoring", "create_none_score_fn"),
    "setup_lsh_mode": ("watermarking.scoring", "setup_lsh_mode"),
    "setup_kmeans_mode": ("watermarking.scoring", "setup_kmeans_mode"),
    "generate": ("watermarking.generate", "generate"),
    "generate_dataset": ("watermarking.generate", "generate_dataset"),
    "compute_zscore": ("watermarking.detect", "compute_zscore"),
    "detect_lsh": ("watermarking.detect", "detect_lsh"),
    "detect_kmeans": ("watermarking.detect", "detect_kmeans"),
    "get_roc_metrics": ("watermarking.detect", "get_roc_metrics"),
    "get_roc_metrics_from_zscores": ("watermarking.detect", "get_roc_metrics_from_zscores"),
    "evaluate_z_scores": ("watermarking.detect", "evaluate_z_scores"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
