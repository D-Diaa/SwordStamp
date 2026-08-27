"""Typed application configuration and defaults."""

from __future__ import annotations

from dataclasses import dataclass, field

from segmentation import (
    DEFAULT_BACKEND,
    DEFAULT_SEMCUT_BATCH_SIZE,
    DEFAULT_SEMCUT_MAX_WORDS,
    DEFAULT_SEMCUT_WINDOW,
    DEFAULT_TYPE,
    SUPPORTED_BACKENDS,
    SUPPORTED_TYPES,
    validate_semcut_policy,
)

from config.paths import METHODS
from config.runtime import DEFAULT_VLLM_UTILIZATION

_GEN_MODES = (*METHODS, "none")
_DETECT_MODES = METHODS
_BACKENDS = ("vllm", "hf")
_SAMPLING = ("rejection", "best-of-n")
_AGG = ("min", "mean")
_ANCHORS = ("positional", "bag")
_SEM_ENT_MODES = ("last_token", "last_mean_pooling", "all_mean_pooling")


def _check(name: str, value, allowed) -> None:
    if value is not None and value not in allowed:
        raise ValueError(
            f"{name} must be one of {tuple(allowed)}, got {value!r}"
        )


@dataclass(frozen=True)
class IOConfig:
    """Input and output locations."""

    data_path: str | None = None
    target: str = "watermark"  # watermark | attack — which derived dir to operate on
    output_dir: str | None = None  # explicit override of the derived working/output dir

    def __post_init__(self) -> None:
        _check("io.target", self.target, ("watermark", "attack"))


@dataclass(frozen=True)
class WatermarkConfig:
    """Watermark algorithm parameters shared by generation and detection."""

    # lsh | kmeans | {lsh,kmeans}_fixed | {lsh,kmeans}_fixed_diverse | none.
    # The *_fixed_diverse modes are generation-only variants of *_fixed: a
    # candidate in the previously committed unit's region is yellow rather than
    # green (a non-lazy walk), so it outranks any red candidate but loses to any
    # diverse green one. Detection is identical to *_fixed.
    mode: str | None = "lsh"
    sp_dim: int = 3
    lmbd: float = 0.25
    margin: float = 0.02
    secret_message: str = "The magic words are squeamish ossifrage."
    embedder: str = "AbeHou/SemStamp-c4-sbert"  # official SemStamp c4-tuned SBERT
    cc_path: str | None = None
    hash_key: int = 15485863  # PRF salt; must match between generation and detection

    def __post_init__(self) -> None:
        _check("watermark.mode", self.mode, _GEN_MODES)


@dataclass(frozen=True)
class GenConfig:
    """Generation and sampling settings."""

    model: str = "meta-llama/Llama-3.1-8B"
    backend: str = "vllm"
    sampling_method: str = "rejection"
    num_candidates: int = 64
    # None sizes the pool from the candidate count.
    max_active_docs: int | None = None
    max_new_tokens: int = 256
    len_prompt: int = 32
    do_sample: bool = True
    rep_p: float = 1.05
    temperature: float = 0.9
    top_k: int = 0
    top_p: float = 0.9
    chunk_tokens: int = 64
    adapter_path: str | None = None
    adapter_strength: float = 1.0

    def __post_init__(self) -> None:
        _check("generation.backend", self.backend, _BACKENDS)
        _check("generation.sampling_method", self.sampling_method, _SAMPLING)
        if self.num_candidates <= 0:
            raise ValueError(
                "generation.num_candidates must be positive, "
                f"got {self.num_candidates!r}"
            )


@dataclass(frozen=True)
class SegmentationConfig:
    """Watermark and optional attacker segmentation settings."""

    type: str = DEFAULT_TYPE
    backend: str = DEFAULT_BACKEND
    attacker_type: str | None = None
    attacker_backend: str | None = None
    semcut_max_words: int = DEFAULT_SEMCUT_MAX_WORDS
    semcut_window: int = DEFAULT_SEMCUT_WINDOW

    def __post_init__(self) -> None:
        _check("segmentation.type", self.type, SUPPORTED_TYPES)
        _check("segmentation.backend", self.backend, SUPPORTED_BACKENDS)
        _check("segmentation.attacker_type", self.attacker_type, SUPPORTED_TYPES)
        _check("segmentation.attacker_backend", self.attacker_backend, SUPPORTED_BACKENDS)
        validate_semcut_policy(
            self.semcut_max_words, self.semcut_window,
        )


@dataclass(frozen=True)
class DetectionConfig:
    """Detection settings."""

    mode: str | None = None
    human_text: str = "data/c4-human-def"
    wm_data_path: str | None = None

    def __post_init__(self) -> None:
        _check("detection.mode", self.mode, _DETECT_MODES)


@dataclass(frozen=True)
class AttackConfig:
    """Paraphrase and attack settings."""

    paraphraser: str = "parrot"
    model_path: str = "meta-llama/Llama-3.1-8B"
    custom_model: str | None = None
    prompt_style: str = "standard"
    batch_size: int = 1
    num_beams: int = 10
    bert_threshold: float = 0.03
    temperature: float | None = None
    do_sample: bool = True
    surrogate_model: str = "BAAI/bge-base-en-v1.5"
    num_candidates: int = 32
    backend: str = "vllm"
    # Positional uses unit i; bag uses every source unit.
    anchor: str = "positional"
    bag_agg: str = "min"
    surrogate_tag: str = ""  # path-only disambiguator appended to adaptive output dirs
    dipper_lex: int = 60       # lexical diversity control (0-100, step 20)
    dipper_order: int = 0      # order diversity control (0-100, step 20)
    dipper_sent_interval: int = 3  # preceding sentences used as context
    word_edit_ratio: float = 0.3  # fraction of words to edit
    back_translation_lang: str = "zh"  # pivot language (zh, de, fr, ru, ar)
    rechunk_words: int = 20  # chunk length for the uniform_rechunk probe atom
    # Unwatermarked corpus supplying donor material to the probe atoms that
    # splice in outside text. Required by random_content_sub and
    # sentence_insertion; unused by every other attack.
    donor_corpus: str | None = None

    def __post_init__(self) -> None:
        _check("attack.backend", self.backend, _BACKENDS)
        _check("attack.anchor", self.anchor, _ANCHORS)
        _check("attack.bag_agg", self.bag_agg, _AGG)


@dataclass(frozen=True)
class QualityConfig:
    """Quality evaluation settings."""

    model_path: str = "meta-llama/Llama-3.1-8B"
    cluster_size: int = 50
    corpus: str | None = None
    sem_ent_mode: str = "last_token"
    # These defaults derive from io.target.
    column: str | None = None
    reference: str | None = None
    skip_per_pair: bool | None = None
    load_kmeans_path: str | None = None
    load_testgen_path: str | None = None
    judge_model: str | None = None
    judge_batch_size: int = 16
    judge_repeats: int = 3
    # None inherits generation.len_prompt.
    judge_len_prompt: int | None = None
    emb_sim_model: str = "google/embeddinggemma-300m"

    def __post_init__(self) -> None:
        _check("quality.sem_ent_mode", self.sem_ent_mode, _SEM_ENT_MODES)
        if self.judge_repeats <= 0:
            raise ValueError(
                "quality.judge_repeats must be positive, "
                f"got {self.judge_repeats!r}"
            )


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime settings."""

    vllm_utilization: float = DEFAULT_VLLM_UTILIZATION
    semcut_batch_size: int = DEFAULT_SEMCUT_BATCH_SIZE

    def __post_init__(self) -> None:
        if self.semcut_batch_size <= 0:
            raise ValueError(
                "runtime.semcut_batch_size must be positive, "
                f"got {self.semcut_batch_size!r}"
            )


@dataclass(frozen=True)
class AppConfig:
    """Complete application configuration."""

    io: IOConfig = field(default_factory=IOConfig)
    watermark: WatermarkConfig = field(default_factory=WatermarkConfig)
    generation: GenConfig = field(default_factory=GenConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def _main():
    """Run schema smoke tests."""
    import dataclasses

    cfg = AppConfig()
    print(f"defaults: sp_dim={cfg.watermark.sp_dim}, lmbd={cfg.watermark.lmbd}, "
          f"backend={cfg.generation.backend!r}, seg_type={cfg.segmentation.type!r}")

    cfg2 = dataclasses.replace(
        cfg,
        watermark=dataclasses.replace(cfg.watermark, sp_dim=16, mode="lsh"),
        generation=dataclasses.replace(cfg.generation, backend="hf"),
    )
    assert cfg2.watermark.sp_dim == 16
    assert cfg2.watermark.mode == "lsh"
    assert cfg2.generation.backend == "hf"
    assert cfg.watermark.sp_dim == 3
    print(f"replace: sp_dim={cfg2.watermark.sp_dim}, mode={cfg2.watermark.mode!r}, backend={cfg2.generation.backend!r}")

    for bad in [("watermark.mode", lambda: WatermarkConfig(mode="bad")),
                ("generation.backend", lambda: GenConfig(backend="bad")),
                ("io.target", lambda: IOConfig(target="bad"))]:
        name, factory = bad
        try:
            factory()
            raise AssertionError(f"expected ValueError for {name}")
        except ValueError as e:
            print(f"enum guard {name}: {e}")

    d = dataclasses.asdict(cfg)
    assert set(d) == {"io", "watermark", "generation", "segmentation", "detection", "attack", "quality", "runtime"}
    print("schema smoke ok")


if __name__ == "__main__":
    _main()
