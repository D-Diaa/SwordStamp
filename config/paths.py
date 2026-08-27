"""Canonical experiment path and naming helpers for the current layout."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import Optional

from segmentation import (
    DEFAULT_SEMCUT_MAX_WORDS,
    DEFAULT_SEMCUT_WINDOW,
    validate_semcut_policy,
)


WATERMARK_LEAF = "watermarked"
DEFAULT_WATERMARK_CANDIDATES = 64
CANDIDATE_BUDGET_PREFIX = "candidates-"

DEFAULT_SEGMENTATION_TYPE = "sentence"
DEFAULT_SEGMENTATION_BACKEND = "nltk"
DEFAULT_SAMPLING_METHOD = "rejection"

METHODS = ("lsh", "lsh_fixed", "lsh_fixed_diverse", "kmeans", "kmeans_fixed",
           "kmeans_fixed_diverse")

# Attacks swept over an edit ratio. The ratio must reach the path or the
# strength levels of one sweep overwrite each other. Names are duplicated from
# ``attacks.simple`` rather than imported to keep config free of attack deps.
RATIO_SWEEP_ATTACKS = frozenset({
    "word_deletion", "synonym_substitution", "context_synonym",
    "sentence_deletion", "merge_adjacent", "split_midpoint",
    "clause_migrate", "permute_sentences", "controlled_reorder",
    "boundary_exchange",
    "random_content_sub",
    "sentence_insertion",
})
CHUNK_SWEEP_ATTACKS = frozenset({"uniform_rechunk"})


@dataclass(frozen=True)
class AttackSpec:
    kind: str
    model: Optional[str] = None
    prompt: Optional[str] = None
    k: Optional[str | int] = None
    suffixes: tuple[str, ...] = field(default_factory=tuple)


def model_tag(model_path: str) -> str:
    return str(model_path).rstrip("/").split("/")[-1]


def semcut_policy_slug(max_words: int, window: int) -> str:
    """Return the readable identity of one semantic-span boundary policy."""
    validate_semcut_policy(max_words, window)
    return f"max{max_words}-win{window}"


def segmentation_slug(
    segmentation_type: str,
    segmentation_backend: str,
    semcut_max_words: int = DEFAULT_SEMCUT_MAX_WORDS,
    semcut_window: int = DEFAULT_SEMCUT_WINDOW,
) -> str:
    base = f"{segmentation_type}-{segmentation_backend}"
    if segmentation_type != "semspan":
        return base
    return f"{base}-{semcut_policy_slug(semcut_max_words, semcut_window)}"


def segmentation_cache_tag(
    segmentation_type: str,
    segmentation_backend: str,
    semcut_max_words: int = DEFAULT_SEMCUT_MAX_WORDS,
    semcut_window: int = DEFAULT_SEMCUT_WINDOW,
) -> str:
    """Return a filename-safe segmentation identity with stable sentence tags."""
    base = f"{segmentation_type}_{segmentation_backend}"
    if segmentation_type != "semspan":
        return base
    policy = semcut_policy_slug(
        semcut_max_words, semcut_window,
    ).replace("-", "_")
    return f"{base}_{policy}"


def method_to_algo_mode(method: str) -> tuple[str, str]:
    if method.startswith("lsh"):
        algo = "lsh"
    elif method.startswith("kmeans"):
        algo = "kmeans"
    else:
        raise ValueError(f"Unknown watermark method: {method!r}")

    # Order matters: "_fixed" is a substring of "_fixed_diverse".
    if "_fixed_diverse" in method:
        mode = "fixed_diverse"
    elif "_fixed" in method:
        mode = "fixed"
    else:
        mode = "context"
    return algo, mode


def generation_subpath(
    sp_mode: Optional[str],
    sampling_method: str = DEFAULT_SAMPLING_METHOD,
    segmentation_type: str = DEFAULT_SEGMENTATION_TYPE,
    segmentation_backend: str = DEFAULT_SEGMENTATION_BACKEND,
    semcut_max_words: int = DEFAULT_SEMCUT_MAX_WORDS,
    semcut_window: int = DEFAULT_SEMCUT_WINDOW,
    num_candidates: int = DEFAULT_WATERMARK_CANDIDATES,
) -> str:
    seg = segmentation_slug(
        segmentation_type, segmentation_backend,
        semcut_max_words, semcut_window,
    )
    if not sp_mode or sp_mode == "none":
        return os.path.join("none", seg)

    algo, mode = method_to_algo_mode(sp_mode)
    parts = [algo, mode, sampling_method, seg]
    # Candidate count is part of the experiment identity even at the paper
    # default, so every watermarked path states it explicitly.
    parts.append(f"{CANDIDATE_BUDGET_PREFIX}{num_candidates}")
    return os.path.join(*parts, WATERMARK_LEAF)


def generation_path(
    data_folder: str,
    sp_mode: Optional[str],
    sampling_method: str = DEFAULT_SAMPLING_METHOD,
    segmentation_type: str = DEFAULT_SEGMENTATION_TYPE,
    segmentation_backend: str = DEFAULT_SEGMENTATION_BACKEND,
    semcut_max_words: int = DEFAULT_SEMCUT_MAX_WORDS,
    semcut_window: int = DEFAULT_SEMCUT_WINDOW,
    num_candidates: int = DEFAULT_WATERMARK_CANDIDATES,
) -> str:
    return os.path.join(
        data_folder,
        generation_subpath(
            sp_mode, sampling_method, segmentation_type, segmentation_backend,
            semcut_max_words, semcut_window, num_candidates,
        ),
    )


def attack_base(data_path: str) -> str:
    normalized = os.path.normpath(data_path)
    if os.path.basename(normalized) == WATERMARK_LEAF:
        return os.path.dirname(normalized) or "."
    return normalized


def _with_suffixes(base: str, suffixes: tuple[str, ...]) -> str:
    clean = tuple(str(s).strip("-") for s in suffixes if s)
    return "-".join((base, *clean)) if clean else base


def attack_leaf(spec: AttackSpec) -> str:
    if spec.kind in {"custom", "custom_sent"}:
        return f"{spec.kind}-{model_tag(spec.model or '')}-{spec.prompt}"

    if spec.kind in {"adaptive", "oracle"}:
        # The candidate budget is the search axis and must stay in the path.
        base = (
            f"{spec.kind}-{model_tag(spec.model or '')}-"
            f"{spec.prompt}-K{spec.k}"
        )
        if spec.kind == "adaptive":
            # Keep ``-min`` for existing result paths.
            base += "-min"
        return _with_suffixes(base, spec.suffixes)

    return _with_suffixes(spec.kind, spec.suffixes)


def attack_path(data_path: str, spec: AttackSpec) -> str:
    return os.path.join(attack_base(data_path), attack_leaf(spec))


def all_beams_path(save_path: str) -> str:
    return os.path.join(save_path, "all_beams.pkl")


def watermarked_sibling_for(dir_path: str) -> Optional[str]:
    normalized = os.path.normpath(dir_path)
    if os.path.basename(normalized) == WATERMARK_LEAF:
        return None
    base = os.path.join(os.path.dirname(normalized), WATERMARK_LEAF)
    return base if os.path.isdir(base) else None


def _attacker_seg_suffix(seg_cfg) -> str:
    """Return a suffix for non-default attacker segmentation."""
    attacker_type = seg_cfg.attacker_type or DEFAULT_SEGMENTATION_TYPE
    attacker_backend = seg_cfg.attacker_backend or DEFAULT_SEGMENTATION_BACKEND
    if (attacker_type, attacker_backend) == (DEFAULT_SEGMENTATION_TYPE, DEFAULT_SEGMENTATION_BACKEND):
        return ""
    slug = segmentation_slug(
        attacker_type,
        attacker_backend,
        seg_cfg.semcut_max_words,
        seg_cfg.semcut_window,
    )
    return f"aseg={slug}"


def attack_spec(attack_cfg, seg_cfg) -> AttackSpec:
    """Build the canonical attack output specification."""
    kind = attack_cfg.paraphraser
    suffixes: list[str] = []

    if kind == "parrot-bigram":
        suffixes.append(f"threshold={attack_cfg.bert_threshold}")
    elif kind in {"openai", "openai-bigram"}:
        suffixes.append(f"num_beams={attack_cfg.num_beams}")
    elif kind in {"pegasus", "pegasus-bigram"}:
        if attack_cfg.temperature:
            suffixes.append(f"temp={attack_cfg.temperature}")
        if kind == "pegasus-bigram":
            suffixes.append(f"bigram-threshold={attack_cfg.bert_threshold}")
        kind = "pegasus"
    elif kind == "adaptive":
        if attack_cfg.anchor == "bag":
            suffixes.append(f"bag={attack_cfg.bag_agg}")
        if attack_cfg.surrogate_tag:
            suffixes.append(f"surr={attack_cfg.surrogate_tag}")
        aseg = _attacker_seg_suffix(seg_cfg)
        if aseg:
            suffixes.append(aseg)
    elif kind == "dipper":
        # Both controls affect output and must appear in the path.
        suffixes.append(f"lex{attack_cfg.dipper_lex}")
        suffixes.append(f"order{attack_cfg.dipper_order}")
    elif kind == "back_translation":
        suffixes.append(f"lang={attack_cfg.back_translation_lang}")
    elif kind in RATIO_SWEEP_ATTACKS:
        suffixes.append(f"ratio={attack_cfg.word_edit_ratio}")
    elif kind in CHUNK_SWEEP_ATTACKS:
        suffixes.append(f"words={attack_cfg.rechunk_words}")

    return AttackSpec(
        kind,
        model=attack_cfg.custom_model,
        prompt=attack_cfg.prompt_style,
        k=attack_cfg.num_candidates,
        suffixes=tuple(suffixes),
    )


def watermark_dir(cfg) -> str:
    """Return the configured watermarked-data directory."""
    return generation_path(
        cfg.io.data_path,
        cfg.watermark.mode,
        cfg.generation.sampling_method,
        cfg.segmentation.type,
        cfg.segmentation.backend,
        cfg.segmentation.semcut_max_words,
        cfg.segmentation.semcut_window,
        cfg.generation.num_candidates,
    )


def attack_dir(cfg) -> str:
    """Return the configured attack directory."""
    return attack_path(watermark_dir(cfg), attack_spec(cfg.attack, cfg.segmentation))


def target_dir(cfg) -> str:
    """Return the configured detection or quality target."""
    return attack_dir(cfg) if cfg.io.target == "attack" else watermark_dir(cfg)


_QUERIES = {
    "base-dir": lambda cfg: cfg.io.data_path,
    "watermark-dir": watermark_dir,
    "attack-dir": attack_dir,
    "target-dir": target_dir,
}


def main(argv: Optional[list[str]] = None) -> int:
    """Print a configured experiment path."""
    from config.cli import add_config_args, resolve  # Avoid an import cycle.

    parser = argparse.ArgumentParser(description="Print a config-derived experiment path.")
    parser.add_argument("query", choices=sorted(_QUERIES))
    add_config_args(parser)
    args = parser.parse_args(argv)
    print(_QUERIES[args.query](resolve(args)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
