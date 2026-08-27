"""Back translation adapted from MarkLLM's ``evaluation/tools/text_editor.py``."""

import argparse

import torch
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer

from attacks.base import AttackConfig, AttackResult, save_dataset

# Pivot-language model IDs.
_FWD = "Helsinki-NLP/opus-mt-en-{lang}"
_BWD = "Helsinki-NLP/opus-mt-{lang}-en"

# Pivots with multilingual reverse models.
_MULTI_BWD = {
    "zh": "Helsinki-NLP/opus-mt-zh-en",
    "de": "Helsinki-NLP/opus-mt-de-en",
    "fr": "Helsinki-NLP/opus-mt-fr-en",
    "ru": "Helsinki-NLP/opus-mt-ru-en",
    "ar": "Helsinki-NLP/opus-mt-ar-en",
}


def _load_marian(model_id: str, device: str):
    tok = MarianTokenizer.from_pretrained(model_id)
    mdl = MarianMTModel.from_pretrained(model_id).to(device).eval()
    return tok, mdl


def _translate_batch(
    texts: list[str],
    tokenizer: MarianTokenizer,
    model: MarianMTModel,
    batch_size: int,
    device: str,
    desc: str,
) -> list[str]:
    outputs = []
    for i in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            gen_ids = model.generate(**enc, num_beams=4)
        decoded = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        outputs.extend(decoded)
    return outputs


def back_translate(
    texts: list[str],
    pivot_lang: str = "zh",
    batch_size: int = 16,
    device: str = "cuda",
) -> list[str]:
    fwd_model_id = _FWD.format(lang=pivot_lang)
    bwd_model_id = _MULTI_BWD.get(pivot_lang, _BWD.format(lang=pivot_lang))

    fwd_tok, fwd_mdl = _load_marian(fwd_model_id, device)
    pivot_texts = _translate_batch(texts, fwd_tok, fwd_mdl, batch_size, device, f"→{pivot_lang}")
    del fwd_tok, fwd_mdl
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    bwd_tok, bwd_mdl = _load_marian(bwd_model_id, device)
    back_texts = _translate_batch(pivot_texts, bwd_tok, bwd_mdl, batch_size, device, f"→en")
    del bwd_tok, bwd_mdl
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return back_texts


def run_attack(texts: list[str], config: AttackConfig) -> AttackResult:
    pivot = getattr(config, "back_translation_lang", "zh")
    output = back_translate(
        texts,
        pivot_lang=pivot,
        batch_size=config.batch_size,
        device=config.device,
    )
    save_path = config.output_path
    if save_path is not None:
        save_dataset(texts, output, save_path)
    return AttackResult(texts, output, save_path=save_path)




def _parse_args():
    parser = argparse.ArgumentParser(description="Smoke-test the back-translation attack.")
    parser.add_argument("--pivot", default="zh", choices=["zh", "de", "fr", "ru", "ar"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--text", action="append")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    texts = args.text or [
        "The James Webb Space Telescope launched on December 25, 2021. "
        "It observes infrared light from distant galaxies and stars.",
        "Python is widely used for data analysis and machine learning "
        "because its ecosystem has mature libraries.",
    ]
    results = back_translate(texts, pivot_lang=args.pivot, batch_size=args.batch_size, device=args.device)
    for orig, para in zip(texts, results):
        print(f"original  : {orig}")
        print(f"translated: {para}\n")
