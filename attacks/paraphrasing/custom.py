import os
from pathlib import Path

import yaml
from tqdm import tqdm
from transformers import GenerationConfig

from config.runtime import DEFAULT_VLLM_UTILIZATION
from segmentation import (
    DEFAULT_BACKEND,
    DEFAULT_TYPE,
    resolve_segmentation_backend,
    resolve_segmentation_type,
    segment,
)
from sampling.base_sampler import create_sampler, resolve_adapter
from attacks.utils import join_sentences_by_document, strip_paraphrase_markers
from attacks.base import AttackResult, save_dataset


_PROMPTS_PATH = Path(__file__).with_name("custom_prompts.yaml")


def _load_custom_prompts(path=_PROMPTS_PATH):
    """Load the custom paraphraser system prompts, keyed by attack.prompt_style."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


CUSTOM_PROMPTS = _load_custom_prompts()


def build_custom_prompt(tokenizer, prompt_style, text):
    """Build a chat-template prompt for the custom paraphraser."""
    system_prompt = CUSTOM_PROMPTS[prompt_style]
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"\n[[START OF TEXT]]\n{text}\n[[END OF TEXT]]"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    ) + "[[START OF PARAPHRASE]]\n"


def _build_sampler(
    custom_model, backend="vllm", device="cuda", bsz=16,
    vllm_utilization=DEFAULT_VLLM_UTILIZATION,
):
    """Build a sampler for unscored chat generation."""
    base, adapter = resolve_adapter(custom_model)
    backend_kwargs = (
        {"device": device, "batch_size": bsz}
        if backend == "hf"
        else {"gpu_memory_utilization": vllm_utilization}
    )
    return create_sampler(backend, base, adapter_path=adapter, **backend_kwargs)


def _generate_paraphrases(sampler, texts, prompt_style, max_new_tokens, temperature=1.0):
    """Generate one unscored paraphrase per text."""
    if not texts:
        return []
    if temperature is None:
        temperature = 1.0
    prompts = [build_custom_prompt(sampler.tokenizer, prompt_style, t) for t in texts]
    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        top_p=1.0,
        pad_token_id=sampler.tokenizer.pad_token_id,
    )
    raw = sampler.generate_raw(prompts, n=1, max_tokens=max_new_tokens, gen_config=gen_config)
    return [
        strip_paraphrase_markers(group[0][0]) if group and group[0] else ""
        for group in raw
    ]


def run_attack(texts, config):
    """Common attack interface."""
    if config.custom_model is None:
        raise ValueError("custom_model is required for custom attacks")
    output = _paraphrase_documents(
        texts, config.custom_model, backend=config.backend, device=config.device,
        bsz=config.batch_size, custom_prompt=config.prompt_style,
        vllm_utilization=getattr(config, "vllm_utilization", DEFAULT_VLLM_UTILIZATION),
        temperature=config.temperature if config.temperature is not None else 1.0,
    )

    save_path = config.output_path
    if save_path is not None:
        save_dataset(texts, output, save_path)
    return AttackResult(texts, output, save_path=save_path)


def run_sentence_attack(texts, config):
    """Common attack interface for the `custom_sent` CLI option."""
    if config.custom_model is None:
        raise ValueError("custom_model is required for custom_sent")
    output = _paraphrase_sentences(
        texts, config.custom_model, backend=config.backend, device=config.device,
        bsz=config.batch_size, custom_prompt=config.prompt_style,
        segmentation_type=resolve_segmentation_type(config),
        segmentation_backend=resolve_segmentation_backend(config),
        vllm_utilization=getattr(config, "vllm_utilization", DEFAULT_VLLM_UTILIZATION),
        temperature=config.temperature if config.temperature is not None else 1.0,
    )
    save_path = config.output_path
    if save_path is not None:
        save_dataset(texts, output, save_path)
    return AttackResult(texts, output, save_path=save_path)




def _paraphrase_documents(
    texts, custom_model, backend="vllm", device="cuda", bsz=16, custom_prompt="combine",
    max_new_tokens=512, vllm_utilization=DEFAULT_VLLM_UTILIZATION, temperature=1.0,
):
    sampler = _build_sampler(custom_model, backend=backend, device=device, bsz=bsz,
                             vllm_utilization=vllm_utilization)
    return _generate_paraphrases(sampler, texts, custom_prompt, max_new_tokens, temperature=temperature)


def _paraphrase_sentences(
    texts,
    custom_model,
    backend="vllm",
    device="cuda",
    bsz=16,
    custom_prompt="sentence",
    max_new_tokens=128,
    segmentation_type=DEFAULT_TYPE,
    segmentation_backend=DEFAULT_BACKEND,
    vllm_utilization=DEFAULT_VLLM_UTILIZATION,
    temperature=1.0,
):
    sampler = _build_sampler(custom_model, backend=backend, device=device, bsz=bsz,
                             vllm_utilization=vllm_utilization)

    sents, doc_lengths = [], []
    for text in tqdm(texts, desc="Splitting"):
        sent_list = [u.display.strip() for u in segment(text, type=segmentation_type, backend=segmentation_backend)]
        sents.extend(sent_list)
        doc_lengths.append(len(sent_list))

    paras = _generate_paraphrases(sampler, sents, custom_prompt, max_new_tokens, temperature=temperature)
    return join_sentences_by_document(paras, doc_lengths)





_SAMPLES_PATH = Path(__file__).with_name("custom_samples.yaml")
DEFAULT_SMOKE_MODEL = "Qwen/Qwen2.5-3B-Instruct"


def _load_samples(path):
    """Load the smoke-test paragraphs (list of {id, note, text}) from YAML."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return (data or {}).get("samples", [])


def _n_sentences(text):
    """Sentence count via the canonical segmenter (for before/after diagnostics)."""
    return len(segment(text, type="sentence", backend="nltk"))


def _run_style(sampler, texts, style, max_new_tokens, temperature=1.0):
    """Generate outputs for one prompt style."""
    if style == "sentence":
        sents, doc_lengths = [], []
        for text in texts:
            units = [u.display.strip() for u in segment(text, type="sentence", backend="nltk")]
            sents.extend(units)
            doc_lengths.append(len(units))
        paras = _generate_paraphrases(sampler, sents, style, max_new_tokens, temperature=temperature)
        return join_sentences_by_document(paras, doc_lengths)
    return _generate_paraphrases(sampler, texts, style, max_new_tokens, temperature=temperature)


def _parse_smoke_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Smoke-test the custom paraphraser prompts over loadable sample paragraphs."
    )
    parser.add_argument(
        "--model",
        default=os.getenv("CUSTOM_PARAPHRASER_MODEL", DEFAULT_SMOKE_MODEL),
        help="HF model or PEFT adapter path to test.",
    )
    parser.add_argument("--samples", type=Path, default=_SAMPLES_PATH, help="path to samples YAML")
    parser.add_argument(
        "--prompt",
        choices=sorted(CUSTOM_PROMPTS),
        action="append",
        help="prompt style to run (repeatable); default runs all styles",
    )
    parser.add_argument("--backend", default="vllm", choices=["hf", "vllm"])
    parser.add_argument("--device", default="cuda", help="Device (hf backend only).")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--vllm-utilization", type=float, default=DEFAULT_VLLM_UTILIZATION)
    parser.add_argument("--limit", type=int, default=None, help="run only the first N samples")
    return parser.parse_args()


def _main():
    args = _parse_smoke_args()
    samples = _load_samples(args.samples)
    if args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        raise SystemExit(f"no samples found in {args.samples}")
    texts = [s["text"] for s in samples]
    styles = args.prompt or sorted(CUSTOM_PROMPTS)

    print(f"Model       : {args.model}")
    print(f"Samples file: {args.samples}  ({len(samples)} paragraphs)")
    print(f"Prompt styles: {', '.join(styles)}")

    sampler = _build_sampler(
        args.model, backend=args.backend, device=args.device,
        vllm_utilization=args.vllm_utilization,
    )

    for style in styles:
        print(f"\n{'=' * 78}\n=== prompt style: {style}  |  {len(samples)} samples ===\n{'=' * 78}")
        outputs = _run_style(sampler, texts, style, args.max_new_tokens)
        for s, text, para in zip(samples, texts, outputs):
            n_in, n_out = _n_sentences(text), _n_sentences(para)
            print(f"\n--- [{s.get('id', '')}] ({s.get('note', '')})  sentences {n_in} -> {n_out} ---")
            print(f"INPUT:\n{text}")
            print(f"OUTPUT:\n{para}")


if __name__ == "__main__":
    _main()
