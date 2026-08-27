"""HuggingFace token-generation backend."""

import torch
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from segmentation import DEFAULT_BACKEND, DEFAULT_TYPE, Segmenter
from sampling.base_sampler import (
    BaseSampler,
    DEFAULT_CHUNK_TOKENS,
    resolve_gen_params,
)


def _scale_adapter(model, adapter_name, scale):
    """Scale an adapter's LoRA layers."""
    for module in model.modules():
        if hasattr(module, "set_scale"):
            module.set_scale(adapter_name, scale)


class HFSampler(BaseSampler):
    """Generate candidate waves with HuggingFace models."""

    def __init__(
        self,
        model_path: str,
        num_candidates: int = 32,
        device: str = "cuda",
        batch_size: int = 32,
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: Optional[str] = "sdpa",
        adapter_path: Optional[str] = None,
        adapter_strength: float = 1.0,
        segmentation_type: str = DEFAULT_TYPE,
        segmentation_backend: str = DEFAULT_BACKEND,
        segmenter: Optional[Segmenter] = None,
        chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    ):
        self.model_path = model_path
        self.device = device
        self.num_candidates = num_candidates
        self.batch_size = batch_size
        self.dtype = dtype
        self.chunk_tokens = chunk_tokens
        self.segmentation_type = segmentation_type
        self.segmentation_backend = segmentation_backend
        self.segmenter = segmenter or Segmenter(segmentation_type, segmentation_backend)
        self.stop_segmenter = Segmenter("sentence", segmentation_backend)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = dict(dtype=dtype)
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        # OPT repositories may contain unusable safetensors indexes.
        if "opt" in model_path.lower():
            model_kwargs["use_safetensors"] = False
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs,
        )
        if adapter_path is not None and adapter_strength != 0.0:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            if adapter_strength != 1.0:
                _scale_adapter(self.model, "default", adapter_strength)
            self.model = self.model.merge_and_unload(safe_merge=True, progressbar=True)
            print(f"Loaded LoRA adapter from {adapter_path} (strength={adapter_strength})")
        self.model.to(device)
        self.model.eval()
        print(
            f"Initialized HF sampler on {device}: {num_candidates} candidates/sentence "
            f"(physical batch_size {batch_size}, chunk_tokens {chunk_tokens})"
        )

    def generate_raw(self, prompts, n, max_tokens, gen_config):
        """Generate continuations in memory-bounded micro-batches."""
        params = resolve_gen_params(gen_config)
        batch_gen_config = GenerationConfig(
            max_new_tokens=max_tokens,
            do_sample=params["do_sample"],
            temperature=params["temperature"],
            top_k=params["top_k"],
            top_p=params["top_p"],
            repetition_penalty=params["repetition_penalty"],
            pad_token_id=self.tokenizer.pad_token_id,
            num_return_sequences=1,
        )
        eos_id = self.tokenizer.eos_token_id

        # Track prompt indices across micro-batches.
        work = [p_idx for p_idx in range(len(prompts)) for _ in range(n)]
        results = [[] for _ in prompts]

        prev_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            for start in range(0, len(work), self.batch_size):
                row_prompt_idxs = work[start:start + self.batch_size]
                pass_prompts = [prompts[i] for i in row_prompt_idxs]
                enc = self.tokenizer(
                    pass_prompts, return_tensors="pt", padding=True
                ).to(self.device)
                with torch.no_grad():
                    outputs = self.model.generate(
                        **enc, generation_config=batch_gen_config,
                    )
                # Left padding gives every prompt the same suffix offset.
                gen_only = outputs[:, enc["input_ids"].shape[1]:]
                for row, p_idx in enumerate(row_prompt_idxs):
                    seq = gen_only[row]
                    finished = bool((seq == eos_id).any().item()) if eos_id is not None else False
                    text = self.tokenizer.decode(seq, skip_special_tokens=True)
                    results[p_idx].append((text, finished))
        finally:
            self.tokenizer.padding_side = prev_side

        return results
