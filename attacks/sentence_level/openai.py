"""OpenAI attack adapted from original SemStamp's ``paraphrase_gen_utils.py``."""

import os
import re

import backoff
import openai
from tqdm import tqdm

from segmentation import (
    DEFAULT_BACKEND,
    DEFAULT_TYPE,
    resolve_segmentation_backend,
    resolve_segmentation_type,
    segment,
)
from attacks.base import AttackResult, save_dataset
from attacks.utils import accept_by_bigram_overlap, get_bert_scorer


def extract_list(text):
    pattern = re.compile(r"^[0-9]+[.)\]\*·:] (.*(?:\n(?![0-9]+[.)\]\*·:]).*)*)", re.MULTILINE)
    return pattern.findall(text)


def gen_prompt(sent, context):
    return f"Previous context: {context} \n Current sentence to paraphrase: {sent}"


def gen_bigram_prompt(sent, context, num_beams):
    return (
        f"Previous context: {context} \n Paraphrase in {num_beams} "
        f"different ways and return a numbered list : {sent}"
    )


@backoff.on_exception(backoff.expo, openai.RateLimitError)
def query_openai(client, prompt):
    while True:
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=1,
                max_tokens=256,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
        except openai.APIError:
            continue
        break
    return response.choices[0].message.content


# Use the long-context deployment.
@backoff.on_exception(backoff.expo, openai.RateLimitError)
def query_openai_bigram(client, prompt):
    while True:
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=1,
                max_tokens=4096,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
        except openai.APIError:
            continue
        break
    return response.choices[0].message.content


def run_attack(texts, config, tokenizer=None, client=None):
    """Run OpenAI paraphrasing with optional bigram selection."""
    if client is None:
        import os
        import openai

        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    new_texts, all_paras = _paraphrase(
        client,
        texts,
        num_beams=config.num_beams,
        bigram=config.bigram,
        bert_threshold=config.bert_threshold,
        tokenizer=tokenizer,
        segmentation_type=resolve_segmentation_type(config),
        segmentation_backend=resolve_segmentation_backend(config),
    )
    save_path = config.output_path
    if save_path is not None:
        save_dataset(new_texts, all_paras, save_path)
    return AttackResult(new_texts, all_paras, save_path=save_path)





def _paraphrase(
    client,
    texts,
    num_beams=10,
    bigram=False,
    bert_threshold=0.03,
    tokenizer=None,
    segmentation_type=DEFAULT_TYPE,
    segmentation_backend=DEFAULT_BACKEND,
):
    new_texts = []
    all_paras = []
    max_iter = 10

    # Reuse one BERTScorer for bigram selection.
    scorer = get_bert_scorer() if bigram else None

    for text in tqdm(texts, desc="Paraphrasing with OpenAI"):
        sents = [u.display.strip() for u in segment(text, type=segmentation_type, backend=segmentation_backend)]
        para_sents = []
        fail = False

        for i, sent in enumerate(sents):
            context = sents[:i]
            num_iter = 0

            if bigram:
                para_ls = []
                prompt = gen_bigram_prompt(sent, context, num_beams)
                while len(para_ls) < 5 and num_iter < max_iter:
                    para_str = query_openai_bigram(client, prompt)
                    para_ls = extract_list(para_str)
                    num_iter += 1
                if num_iter > max_iter:
                    fail = True
                    break
                # Select by bigram overlap.
                best = accept_by_bigram_overlap(
                    sent, para_ls, tokenizer, bert_threshold, scorer
                )
                para_sents.append(best)
            else:
                prompt = gen_prompt(sent, context)
                para_sents.append(query_openai(client, prompt))

        if fail:
            continue
        new_texts.append(text)
        all_paras.append(" ".join(para_sents))

    return new_texts, all_paras


def _parse_smoke_args():
    import argparse

    parser = argparse.ArgumentParser(description="Smoke-test the OpenAI paraphrasing attack.")
    parser.add_argument("--text", action="append", help="Text to paraphrase. May be passed multiple times.")
    parser.add_argument("--num-beams", type=int, default=10)
    parser.add_argument("--bigram", action="store_true")
    return parser.parse_args()


def _main():
    args = _parse_smoke_args()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required to run this smoke test.")

    texts = args.text or [
        "The satellite crossed the night sky before sunrise. Researchers used it to calibrate their instruments."
    ]
    client = openai.OpenAI(api_key=api_key)
    source, paraphrases = _paraphrase(client, texts, num_beams=args.num_beams, bigram=args.bigram)

    for i, (text, para) in enumerate(zip(source, paraphrases), start=1):
        print(f"\n[{i}] input:\n{text}")
        print(f"\n[{i}] output:\n{para}")


if __name__ == "__main__":
    _main()
