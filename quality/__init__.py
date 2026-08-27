"""Quality evaluation package exports."""

import os
import re
from importlib import import_module
from pathlib import Path


def _load_dotenv():
    """Load the first project ``.env`` file, overriding existing variables."""
    candidates = [Path.cwd() / ".env", Path(__file__).resolve().parent.parent / ".env"]
    seen = set()
    for p in candidates:
        if p in seen or not p.is_file():
            continue
        seen.add(p)
        for line in p.read_text().splitlines():
            m = re.match(r'\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)', line)
            if not m:
                continue
            key, val = m.group(1), m.group(2).strip()
            # strip surrounding single or double quotes
            if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
                val = val[1:-1]
            os.environ[key] = val
        return  # first .env wins


_load_dotenv()


_EXPORTS = {
    "eval_perplexity": ("quality.metrics.common", "eval_perplexity"),
    "text_entropy": ("quality.metrics.common", "text_entropy"),
    "rep_ngram": ("quality.metrics.common", "rep_ngram"),
    "tokenize_untokenize": ("quality.metrics", "tokenize_untokenize"),
    "evaluate_perplexity": ("quality.metrics", "evaluate_perplexity"),
    "evaluate_entropy": ("quality.metrics", "evaluate_entropy"),
    "evaluate_ngrams": ("quality.metrics", "evaluate_ngrams"),
    "evaluate_diversity": ("quality.metrics", "evaluate_diversity"),
    "evaluate_mauve": ("quality.metrics", "evaluate_mauve"),
    "evaluate_bertscore": ("quality.metrics", "evaluate_bertscore"),
    "evaluate_embedding_similarity": ("quality.metrics", "evaluate_embedding_similarity"),
    "evaluate_bleu": ("quality.metrics", "evaluate_bleu"),
    "evaluate_rouge": ("quality.metrics", "evaluate_rouge"),
    "evaluate_semantic_entropy": ("quality.metrics", "evaluate_semantic_entropy"),
    "evaluate_llm_judge": ("quality.metrics", "evaluate_llm_judge"),
    "evaluate_model_quality_metrics": ("quality.evaluator", "evaluate_model_quality_metrics"),
    "evaluate_pair_quality_metrics": ("quality.evaluator", "evaluate_pair_quality_metrics"),
    "evaluate_llm_judge_metrics": ("quality.evaluator", "evaluate_llm_judge_metrics"),
    "evaluate_quality": ("quality.evaluator", "evaluate_quality"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
