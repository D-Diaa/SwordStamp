"""Expose quality metrics with lazy optional dependencies."""

from importlib import import_module

_EXPORTS = {
    "_confidence_interval": ("quality.metrics.common", "_confidence_interval"),
    "eval_perplexity": ("quality.metrics.common", "eval_perplexity"),
    "path_wo_ext": ("quality.metrics.common", "path_wo_ext"),
    "rep_ngram": ("quality.metrics.common", "rep_ngram"),
    "text_entropy": ("quality.metrics.common", "text_entropy"),
    "tokenize_untokenize": ("quality.metrics.common", "tokenize_untokenize"),
    "evaluate_perplexity": ("quality.metrics.perplexity", "evaluate_perplexity"),
    "evaluate_entropy": ("quality.metrics.entropy", "evaluate_entropy"),
    "evaluate_ngrams": ("quality.metrics.ngrams", "evaluate_ngrams"),
    "evaluate_diversity": ("quality.metrics.diversity", "evaluate_diversity"),
    "evaluate_mauve": ("quality.metrics.mauve", "evaluate_mauve"),
    "evaluate_bertscore": ("quality.metrics.bertscore", "evaluate_bertscore"),
    "_get_emb_sim_model": ("quality.metrics.emb_sim", "_get_emb_sim_model"),
    "clear_emb_sim_model_cache": ("quality.metrics.emb_sim", "clear_emb_sim_model_cache"),
    "emb_sim_pairs": ("quality.metrics.emb_sim", "emb_sim_pairs"),
    "evaluate_embedding_similarity": ("quality.metrics.emb_sim", "evaluate_embedding_similarity"),
    "evaluate_bleu": ("quality.metrics.bleu", "evaluate_bleu"),
    "evaluate_rouge": ("quality.metrics.rouge", "evaluate_rouge"),
    "evaluate_anchor_structure": ("quality.metrics.anchors", "evaluate_anchor_structure"),
    "evaluate_semantic_entropy": ("quality.metrics.semantic_entropy", "evaluate_semantic_entropy"),
    "JudgeRubric": ("quality.metrics.llm_judge", "JudgeRubric"),
    "PAIRWISE_RUBRIC": ("quality.metrics.llm_judge", "PAIRWISE_RUBRIC"),
    "QUALITY_RUBRIC": ("quality.metrics.llm_judge", "QUALITY_RUBRIC"),
    "_is_openai_model": ("quality.metrics.llm_judge", "_is_openai_model"),
    "_clip05": ("quality.metrics.llm_judge", "_clip05"),
    "_parse_dimension_scores": ("quality.metrics.llm_judge", "_parse_dimension_scores"),
    "_overall_from_dims": ("quality.metrics.llm_judge", "_overall_from_dims"),
    "empty_rubric_metrics": ("quality.metrics.llm_judge", "empty_rubric_metrics"),
    "_run_llm_judge_openai": ("quality.metrics.llm_judge", "_run_llm_judge_openai"),
    "_format_chat_for_generation": ("quality.metrics.llm_judge", "_format_chat_for_generation"),
    "VLLMJudge": ("quality.metrics.llm_judge", "VLLMJudge"),
    "build_llm_judge_pipe": ("quality.metrics.llm_judge", "build_llm_judge_pipe"),
    "_run_llm_judge_local": ("quality.metrics.llm_judge", "_run_llm_judge_local"),
    "evaluate_llm_judge": ("quality.metrics.llm_judge", "evaluate_llm_judge"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
