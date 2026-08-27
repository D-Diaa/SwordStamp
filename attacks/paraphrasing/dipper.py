"""Dipper attack adapted from MarkLLM's ``evaluation/tools/text_editor.py``."""

import argparse

import torch
from nltk import sent_tokenize
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer

from attacks.base import AttackConfig, AttackResult, save_dataset

DIPPER_MODEL = "kalpeshk2011/dipper-paraphraser-xxl"
# Dipper uses the T5 v1.1 XXL tokenizer.
DIPPER_TOKENIZER = "google/t5-v1_1-xxl"
MAX_INPUT_TOKENS = 512   # T5 positional limit; the encoder is O(L^2) in memory.
PREFIX_BUDGET = 256      # cap on the rolling paraphrased-context window (tokens).


def _load_model():
    tokenizer = T5Tokenizer.from_pretrained(DIPPER_TOKENIZER)
    # bf16 leaves memory for long-input attention.
    model = T5ForConditionalGeneration.from_pretrained(DIPPER_MODEL, torch_dtype=torch.bfloat16)
    model.cuda()
    model.eval()
    return tokenizer, model


def _paraphrase_text(
    text: str,
    tokenizer,
    model,
    lex: int,
    order: int,
    sent_interval: int,
) -> str:
    # Dipper encodes diversity inversely.
    lex_code = 100 - lex
    order_code = 100 - order

    text = " ".join(text.split())
    sentences = sent_tokenize(text)
    prefix = ""
    output_text = ""

    for sent_idx in range(0, len(sentences), sent_interval):
        curr_window = " ".join(sentences[sent_idx : sent_idx + sent_interval])

        input_text = f"lexical = {lex_code}, order = {order_code}"
        if prefix:
            input_text += f" {prefix}"
        input_text += f" <sent> {curr_window} </sent>"

        final_input = tokenizer([input_text], return_tensors="pt",
                                truncation=True, max_length=MAX_INPUT_TOKENS)
        final_input = {k: v.cuda() for k, v in final_input.items()}

        with torch.inference_mode():
            outputs = model.generate(
                **final_input,
                do_sample=True,
                top_p=0.75,
                top_k=None,
                max_length=512,
            )
        out = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        output_text += " " + out
        # Release window tensors promptly.
        del final_input, outputs

        # Bound rolling context to T5's input budget.
        prefix = (prefix + " " + out).strip()
        prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
        if len(prefix_ids) > PREFIX_BUDGET:
            prefix = tokenizer.decode(prefix_ids[-PREFIX_BUDGET:], skip_special_tokens=True)

    # Defragment the allocator between documents.
    torch.cuda.empty_cache()
    return output_text.strip()


def run_attack(texts: list[str], config: AttackConfig) -> AttackResult:
    tokenizer, model = _load_model()
    output = [
        _paraphrase_text(
            text,
            tokenizer,
            model,
            lex=config.dipper_lex,
            order=config.dipper_order,
            sent_interval=config.dipper_sent_interval,
        )
        for text in tqdm(texts, desc="Dipper")
    ]
    save_path = config.output_path
    if save_path is not None:
        save_dataset(texts, output, save_path)
    return AttackResult(texts, output, save_path=save_path)


def _parse_args():
    parser = argparse.ArgumentParser(description="Smoke-test the Dipper paraphrasing attack.")
    parser.add_argument("--lex", type=int, default=60, choices=[0, 20, 40, 60, 80, 100])
    parser.add_argument("--order", type=int, default=0, choices=[0, 20, 40, 60, 80, 100])
    parser.add_argument("--sent-interval", type=int, default=3)
    parser.add_argument("--text", action="append")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    texts = args.text or [
        "The James Webb Space Telescope launched on December 25, 2021. "
        "It observes infrared light from distant galaxies and stars. "
        "Scientists use it to study the formation of the first galaxies."
    ]
    tokenizer, model = _load_model()
    for i, text in enumerate(texts, 1):
        result = _paraphrase_text(
            text,
            tokenizer,
            model,
            lex=args.lex,
            order=args.order,
            sent_interval=args.sent_interval,
        )
        print(f"\n[{i}] original:\n{text}")
        print(f"\n[{i}] paraphrase:\n{result}")
