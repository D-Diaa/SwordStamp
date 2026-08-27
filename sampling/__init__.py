"""Watermark-agnostic candidate sampling."""

from importlib import import_module


_EXPORTS = {
    "BaseSampler": ("sampling.base_sampler", "BaseSampler"),
    "CandidateScore": ("sampling.base_sampler", "CandidateScore"),
    "GeneratedCandidate": ("sampling.base_sampler", "GeneratedCandidate"),
    "Region": ("sampling.base_sampler", "Region"),
    "ScoreFn": ("sampling.base_sampler", "ScoreFn"),
    "create_sampler": ("sampling.base_sampler", "create_sampler"),
    "resolve_adapter": ("sampling.base_sampler", "resolve_adapter"),
    "score_accepts": ("sampling.base_sampler", "score_accepts"),
    "default_accept_fn": ("sampling.base_sampler", "default_accept_fn"),
    "DEFAULT_CHUNK_TOKENS": ("sampling.base_sampler", "DEFAULT_CHUNK_TOKENS"),
    "HFSampler": ("sampling.hf_sampler", "HFSampler"),
    "VLLMSampler": ("sampling.vllm_sampler", "VLLMSampler"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
