"""Pegasus attack adapted from original SemStamp's ``paraphrase_gen.py``."""

import pickle

import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from attacks.utils import (
    accept_by_bigram_overlap,
    batched,
    get_bert_scorer,
    join_sentences_by_document,
    split_texts_into_sentences,
    well_formed_sentence,
)
from config.paths import all_beams_path
from attacks.base import AttackResult, save_dataset
from segmentation import (
    DEFAULT_BACKEND,
    DEFAULT_TYPE,
    resolve_segmentation_backend,
    resolve_segmentation_type,
)


def run_attack(texts, config, tokenizer=None):
    """Common attack interface."""
    if config.do_sample:
        temperature = config.temperature if config.temperature is not None else 2.0
    else:
        temperature = None  # pure beam search; do_sample stays False in gen_kwargs
    output, total_paraphrased = _paraphrase(
        texts, tokenizer,
        bigram=config.bigram,
        num_beams=config.num_beams,
        bert_threshold=config.bert_threshold,
        bsz=config.batch_size,
        device=config.device,
        temperature=temperature,
        segmentation_type=resolve_segmentation_type(config),
        segmentation_backend=resolve_segmentation_backend(config),
    )
    beams_path = None
    save_path = config.output_path
    if save_path is not None:
        save_dataset(texts, output, save_path)
        beams_path = all_beams_path(save_path)
        with open(beams_path, "wb") as f:
            pickle.dump(total_paraphrased, f)
    return AttackResult(texts, output, save_path=save_path, beams_path=beams_path)




def _paraphrase(
    texts,
    tokenizer,
    bigram=False,
    num_beams=10,
    bert_threshold=0.03,
    bsz=1,
    device="cuda",
    temperature=2.0,
    segmentation_type=DEFAULT_TYPE,
    segmentation_backend=DEFAULT_BACKEND,
):
    if bigram and tokenizer is None:
        raise ValueError("tokenizer is required for pegasus-bigram")

    model_tag = "tuner007/pegasus_paraphrase"
    peg_tokenizer = AutoTokenizer.from_pretrained(model_tag)
    peg_model = AutoModelForSeq2SeqLM.from_pretrained(model_tag).to(device)
    peg_model.eval()

    scorer = get_bert_scorer() if bigram else None
    sents, doc_lengths = split_texts_into_sentences(
        tqdm(texts, desc="Tokenizing"),
        segmentation_type=segmentation_type,
        segmentation_backend=segmentation_backend,
    )
    n_return = num_beams if bigram else 1

    gen_kwargs = dict(
        max_length=60,
        num_beams=num_beams,
        num_return_sequences=n_return,
        repetition_penalty=1.03,
    )
    if temperature is not None:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature

    paras = []
    total_paraphrased = []
    for batch in tqdm(batched(sents, bsz), desc="Paraphrasing with Pegasus"):
        inputs = peg_tokenizer(
            list(batch), return_tensors="pt", padding="longest",
            max_length=60, truncation=True,
        ).to(device)
        with torch.no_grad():
            outputs = peg_model.generate(**inputs, **gen_kwargs)

        all_decoded = peg_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for i, sent in enumerate(batch):
            start = i * n_return
            end = start + n_return
            paraphrased = [well_formed_sentence(p, end_sent=True) for p in all_decoded[start:end]]
            total_paraphrased.append(paraphrased)
            if bigram:
                para = accept_by_bigram_overlap(
                    sent, paraphrased, tokenizer,
                    bert_threshold=bert_threshold, scorer=scorer,
                )
            else:
                para = paraphrased[0]
            paras.append(para)

    return join_sentences_by_document(paras, doc_lengths), total_paraphrased


def _parse_smoke_args():
    import argparse

    parser = argparse.ArgumentParser(description="Smoke-test the Pegasus paraphrasing attack.")
    parser.add_argument("--text", action="append", help="Text to paraphrase. May be passed multiple times.")
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--bigram", action="store_true")
    parser.add_argument("--bert-threshold", type=float, default=0.03)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--model-path",
        default="meta-llama/Llama-3.1-8B",
        help="Tokenizer path used only with --bigram.",
    )
    return parser.parse_args()


def _main():
    args = _parse_smoke_args()
    texts = args.text or [
        "The satellite crossed the night sky before sunrise. Researchers used it to calibrate their instruments."
    ]
    tokenizer = None
    if args.bigram:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    device = args.device or "cuda"
    output, beams = _paraphrase(
        texts, tokenizer,
        bigram=args.bigram,
        num_beams=args.num_beams,
        bert_threshold=args.bert_threshold,
        bsz=args.batch_size,
        device=device,
        temperature=args.temperature,
    )
    for i, (text, para) in enumerate(zip(texts, output), start=1):
        print(f"\n[{i}] input:\n{text}")
        print(f"\n[{i}] output:\n{para}")
    print(f"\nGenerated beams for {len(beams)} sentences.")


if __name__ == "__main__":
    _main()
