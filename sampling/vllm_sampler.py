"""vLLM token-generation backend."""

import os
import json
from typing import Optional

from segmentation import DEFAULT_BACKEND, DEFAULT_TYPE, Segmenter
from sampling.base_sampler import (
    BaseSampler,
    DEFAULT_CHUNK_TOKENS,
    resolve_gen_params,
)
from config.runtime import vllm_gpu_memory_utilization

DEFAULT_MAX_MODEL_LEN = 16_384


def _as_vllm_dtype(dtype) -> str:
    name = str(dtype).replace("torch.", "")
    if name in ("float16", "bfloat16", "float32", "auto"):
        return name
    return "bfloat16"


def _load_adapter_config(adapter_path: str) -> Optional[dict]:
    """Load a local or remote adapter config."""
    config_path = os.path.join(adapter_path, "adapter_config.json")
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            return json.load(f)
    # Fall back to a Hub repository.
    try:
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(adapter_path, "adapter_config.json")
        with open(config_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _adapter_rank(adapter_config: dict) -> Optional[int]:
    rank = adapter_config.get("r")
    if rank is not None:
        return int(rank)
    pattern = adapter_config.get("rank_pattern") or {}
    if pattern:
        return max(int(v) for v in pattern.values())
    return None


def _vllm_lora_rank_bucket(rank: int) -> int:
    for valid in (1, 8, 16, 32, 64, 128, 256, 320, 512):
        if rank <= valid:
            return valid
    return rank


def _merge_and_save_adapter(
    base_model_path: str,
    adapter_path: str,
    adapter_strength: float,
) -> str:
    """Merge a scaled LoRA adapter into a temporary CPU checkpoint."""
    import tempfile
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tmpdir = tempfile.mkdtemp(prefix="semstamp_merged_adapter_")
    print(
        f"Merging LoRA adapter (strength={adapter_strength}) from {adapter_path} "
        f"into {base_model_path} -> {tmpdir}"
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    if adapter_strength != 1.0:
        for module in model.modules():
            if hasattr(module, "set_scale"):
                module.set_scale("default", adapter_strength)
    model = model.merge_and_unload(safe_merge=True, progressbar=True)
    model.save_pretrained(tmpdir)
    AutoTokenizer.from_pretrained(base_model_path).save_pretrained(tmpdir)
    del model
    return tmpdir


class VLLMSampler(BaseSampler):
    """Generate candidate waves with vLLM."""

    def __init__(
        self,
        model_path: str,
        num_candidates: int = 32,
        chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
        dtype="bfloat16",
        enable_prefix_caching: bool = True,
        adapter_path: Optional[str] = None,
        adapter_strength: float = 1.0,
        segmentation_type: str = DEFAULT_TYPE,
        segmentation_backend: str = DEFAULT_BACKEND,
        segmenter: Optional[Segmenter] = None,
        gpu_memory_utilization: Optional[float] = None,
        tensor_parallel_size: int = 1,
        max_model_len: Optional[int] = None,
    ):
        try:
            from vllm import LLM, SamplingParams
        except ImportError as e:
            raise ImportError(
                "VLLMSampler requires vLLM. Install it with: pip install vllm"
            ) from e

        self._SamplingParams = SamplingParams
        self.num_candidates = num_candidates
        self.chunk_tokens = chunk_tokens
        self.segmentation_type = segmentation_type
        self.segmentation_backend = segmentation_backend
        self.segmenter = segmenter or Segmenter(segmentation_type, segmentation_backend)
        # Chunk completion always uses sentence boundaries.
        self.stop_segmenter = Segmenter("sentence", segmentation_backend)
        self._lora_request = None
        # Keep merged checkpoints alive with the sampler.
        self._merged_tmpdir: Optional[str] = None

        if gpu_memory_utilization is None:
            gpu_memory_utilization = vllm_gpu_memory_utilization()
        if max_model_len is None:
            max_model_len = DEFAULT_MAX_MODEL_LEN

        llm_model_path = model_path
        llm_kwargs: dict = {}

        if adapter_path is not None and adapter_strength != 0.0:
            adapter_config = _load_adapter_config(adapter_path)
            if adapter_config is None:
                raise ValueError(f"No adapter_config.json found at {adapter_path!r}")

            if adapter_strength != 1.0:
                self._merged_tmpdir = _merge_and_save_adapter(
                    model_path, adapter_path, adapter_strength
                )
                llm_model_path = self._merged_tmpdir
            else:
                from vllm.lora.request import LoRARequest

                llm_kwargs["enable_lora"] = True
                rank = _adapter_rank(adapter_config)
                if rank is not None:
                    llm_kwargs["max_lora_rank"] = _vllm_lora_rank_bucket(rank)
                self._lora_request = LoRARequest("watermark_adapter", 1, adapter_path)
                print(f"Using vLLM native LoRA from {adapter_path!r}")

        init_kwargs: dict = dict(
            model=llm_model_path,
            dtype=_as_vllm_dtype(dtype),
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=enable_prefix_caching,
            max_model_len=max_model_len,
            trust_remote_code=os.getenv("VLLM_TRUST_REMOTE_CODE", "0") == "1",
            **llm_kwargs,
        )

        self.llm = LLM(**init_kwargs)
        self._lora_kwargs: dict = (
            {"lora_request": self._lora_request} if self._lora_request else {}
        )

        if hasattr(self.llm, "get_tokenizer"):
            self.tokenizer = self.llm.get_tokenizer()
        else:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(llm_model_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        print(
            f"Initialized vLLM sampler: {model_path} "
            f"({num_candidates} candidates/sentence in one wave, "
            f"chunk_tokens={chunk_tokens}, max_model_len={max_model_len})"
        )

    def _to_sampling_params(self, gen_config, *, n: int, max_tokens: int):
        """Translate shared generation settings to vLLM."""
        params = resolve_gen_params(gen_config)
        temp = params["temperature"] if params["do_sample"] else 0.0
        # vLLM uses -1 rather than 0 to disable top-k.
        top_k = params["top_k"]
        top_k = -1 if top_k is None or top_k <= 0 else top_k
        return self._SamplingParams(
            n=n,
            temperature=temp,
            top_k=top_k,
            top_p=params["top_p"],
            repetition_penalty=params["repetition_penalty"],
            max_tokens=max_tokens,
        )

    def generate_raw(self, prompts, n, max_tokens, gen_config):
        """Generate continuations in one vLLM request batch."""
        params = self._to_sampling_params(gen_config, n=n, max_tokens=max_tokens)
        outs = self.llm.generate(prompts, params, use_tqdm=False, **self._lora_kwargs)
        return [
            [(o.text, o.finish_reason == "stop") for o in req.outputs]
            for req in outs
        ]
