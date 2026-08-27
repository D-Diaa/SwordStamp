"""Shared runtime and result contracts for attack implementations."""

from dataclasses import dataclass
from typing import Any, Optional

from datasets import Dataset
from config.runtime import DEFAULT_VLLM_UTILIZATION
from segmentation import (
    DEFAULT_BACKEND,
    DEFAULT_SEMCUT_BATCH_SIZE,
    DEFAULT_SEMCUT_MAX_WORDS,
    DEFAULT_SEMCUT_WINDOW,
    DEFAULT_TYPE,
)


@dataclass
class AttackConfig:
    """Runtime attack parameters."""

    model_path: str = "meta-llama/Llama-3.1-8B"
    custom_model: Optional[str] = None
    prompt_style: str = "combine"
    batch_size: int = 1
    num_beams: int = 10
    bigram: bool = False
    bert_threshold: float = 0.03
    device: str = "cuda"
    # None uses the attack-specific default.
    temperature: Optional[float] = None
    do_sample: bool = True
    surrogate_model: str = "BAAI/bge-base-en-v1.5"
    # Adaptive candidate budget and sampler backend.
    num_candidates: int = 32
    backend: str = "vllm"
    # vLLM memory fraction with room for the surrogate.
    vllm_utilization: float = DEFAULT_VLLM_UTILIZATION
    # Adaptive anchor mode and bag reduction.
    anchor: str = "positional"
    bag_agg: str = "min"
    # Defender WatermarkConfig, for attackers that are given the partition.
    watermark: Optional[Any] = None
    segmentation_type: str = DEFAULT_TYPE
    segmentation_backend: str = DEFAULT_BACKEND
    semcut_max_words: int = DEFAULT_SEMCUT_MAX_WORDS
    semcut_window: int = DEFAULT_SEMCUT_WINDOW
    semcut_batch_size: int = DEFAULT_SEMCUT_BATCH_SIZE
    output_path: Optional[str] = None
    dipper_lex: int = 60
    dipper_order: int = 0
    dipper_sent_interval: int = 3
    word_edit_ratio: float = 0.3
    back_translation_lang: str = "zh"
    rechunk_words: int = 20
    # Unwatermarked corpus supplying donor material to the splicing probe atoms.
    donor_corpus: Optional[str] = None


@dataclass
class AttackResult:
    text: Any
    para_text: Any
    save_path: Optional[str] = None
    beams_path: Optional[str] = None


def save_dataset(texts, para_texts, save_path):
    Dataset.from_dict({"text": texts, "para_text": para_texts}).save_to_disk(save_path)
    return save_path
