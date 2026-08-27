"""Parrot attack adapted from original SemStamp's ``paraphrase_gen_utils.py``."""

import pickle
import re
import os

if os.getenv("PARROT_HF_OFFLINE", "1") != "0":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from tqdm import tqdm
from parrot import Parrot

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


# Adapt Parrot to the current transformers API.
class SParrot(Parrot):
    def __init__(self, model_tag="prithivida/parrot_paraphraser_on_T5", use_gpu=True):
        # Avoid the deprecated use_auth_token parameter.
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
        from parrot.filters import Adequacy, Fluency, Diversity

        self.device = "cuda" if use_gpu else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_tag)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_tag, use_safetensors=False).to(self.device)
        self.adequacy_score = Adequacy()
        self.fluency_score = Fluency()
        self.diversity_score = Diversity()

        # Reuse generation configs.
        self._GenerationConfig = GenerationConfig

    def _clean_and_prefix(self, input_phrase):
        cleaned = re.sub('[^a-zA-Z0-9 \?\'\-\/\:\.]', '', input_phrase)
        return "paraphrase: " + cleaned

    def _filter_and_rank(self, input_phrase, prefixed_phrase, paraphrases,
                         adequacy_threshold, fluency_threshold, diversity_ranker):
        adequacy_filtered = self.adequacy_score.filter(
            prefixed_phrase, paraphrases, adequacy_threshold, self.device)
        if len(adequacy_filtered) == 0:
            adequacy_filtered = paraphrases
        fluency_filtered = self.fluency_score.filter(
            adequacy_filtered, fluency_threshold, self.device)
        if len(fluency_filtered) == 0:
            fluency_filtered = adequacy_filtered
        diversity_scored = self.diversity_score.rank(
            input_phrase, fluency_filtered, diversity_ranker)
        para_phrases = sorted(diversity_scored.items(), key=lambda x: x[1], reverse=True)
        return [phrase for phrase, score in para_phrases]

    def augment(self, input_phrase, use_gpu=True, diversity_ranker="levenshtein", do_diverse=False,
                max_return_phrases=10, max_length=32, adequacy_threshold=0.90, fluency_threshold=0.90):
        if len(input_phrase) >= max_length:
            max_length += 32

        prefixed_phrase = self._clean_and_prefix(input_phrase)
        input_ids = self.tokenizer.encode(prefixed_phrase, return_tensors='pt').to(self.device)

        if do_diverse:
            # Choose a valid beam-group count.
            for n in range(2, 9):
                if max_return_phrases % n == 0:
                    break
            gen_config = self._GenerationConfig(
                do_sample=False,
                max_length=max_length,
                num_beams=max_return_phrases,
                num_beam_groups=n,
                diversity_penalty=2.0,
                early_stopping=True,
                num_return_sequences=max_return_phrases,
            )
            preds = self.model.generate(input_ids, generation_config=gen_config, trust_remote_code=True, custom_generate='transformers-community/group-beam-search')
        else:
            gen_config = self._GenerationConfig(
                do_sample=True,
                max_length=max_length,
                top_k=50,
                top_p=0.95,
                early_stopping=True,
                num_return_sequences=max_return_phrases,
            )
            preds = self.model.generate(input_ids, generation_config=gen_config)

        paraphrases = set()
        for pred in preds:
            gen_pp = self.tokenizer.decode(pred, skip_special_tokens=True).lower()
            gen_pp = re.sub('[^a-zA-Z0-9 \?\'\-]', '', gen_pp)
            paraphrases.add(gen_pp)

        return self._filter_and_rank(
            input_phrase, prefixed_phrase, paraphrases,
            adequacy_threshold, fluency_threshold, diversity_ranker)

    def augment_batch(self, input_phrases, use_gpu=True, diversity_ranker="levenshtein", do_diverse=False,
                      max_return_phrases=10, max_length=32, adequacy_threshold=0.90, fluency_threshold=0.90):
        """Generate paraphrases for several sentences in one pass."""
        # Use one batch-wide maximum length.
        effective_max_length = max_length
        for phrase in input_phrases:
            if len(phrase) >= max_length:
                effective_max_length = max(effective_max_length, max_length + 32)

        prefixed_phrases = [self._clean_and_prefix(p) for p in input_phrases]
        batch_encoding = self.tokenizer(
            prefixed_phrases, return_tensors='pt', padding=True, truncation=True
        ).to(self.device)

        if do_diverse:
            for n in range(2, 9):
                if max_return_phrases % n == 0:
                    break
            gen_config = self._GenerationConfig(
                do_sample=False,
                max_length=effective_max_length,
                num_beams=max_return_phrases,
                num_beam_groups=n,
                diversity_penalty=2.0,
                early_stopping=True,
                num_return_sequences=max_return_phrases,
            )
            preds = self.model.generate(
                **batch_encoding, generation_config=gen_config,
                trust_remote_code=True,
                custom_generate='transformers-community/group-beam-search')
        else:
            gen_config = self._GenerationConfig(
                do_sample=True,
                max_length=effective_max_length,
                top_k=50,
                top_p=0.95,
                early_stopping=True,
                num_return_sequences=max_return_phrases,
            )
            preds = self.model.generate(**batch_encoding, generation_config=gen_config)

        # Regroup decoded outputs by input.
        all_decoded = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        batch_size = len(input_phrases)
        results = []
        for i in range(batch_size):
            start = i * max_return_phrases
            end = start + max_return_phrases
            paraphrases = set()
            for gen_pp in all_decoded[start:end]:
                gen_pp = gen_pp.lower()
                gen_pp = re.sub('[^a-zA-Z0-9 \?\'\-]', '', gen_pp)
                paraphrases.add(gen_pp)

            results.append(self._filter_and_rank(
                input_phrases[i], prefixed_phrases[i], paraphrases,
                adequacy_threshold, fluency_threshold, diversity_ranker))

        return results



def run_attack(texts, config, tokenizer=None, parrot=None):
    """Common attack interface."""
    parrot = parrot or SParrot()
    output, total_paraphrased = _paraphrase(
        parrot, texts, tokenizer,
        num_beams=config.num_beams,
        bigram=config.bigram,
        bert_threshold=config.bert_threshold,
        bsz=config.batch_size,
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
    parrot,
    texts,
    tokenizer,
    num_beams=10,
    bigram=False,
    bert_threshold=0.03,
    bsz=1,
    segmentation_type=DEFAULT_TYPE,
    segmentation_backend=DEFAULT_BACKEND,
):
    if bigram and tokenizer is None:
        raise ValueError("tokenizer is required for parrot-bigram")

    augment_kwargs = dict(
        use_gpu=True,
        diversity_ranker="levenshtein",
        do_diverse=True,
        max_return_phrases=num_beams,
        max_length=60,
        adequacy_threshold=0.8,
        fluency_threshold=0.8,
    )

    scorer = get_bert_scorer() if bigram else None
    sents, doc_lengths = split_texts_into_sentences(
        tqdm(texts, desc="Tokenizing"),
        segmentation_type=segmentation_type,
        segmentation_backend=segmentation_backend,
    )

    paras = []
    total_paraphrased = []
    for batch in tqdm(batched(sents, bsz), desc="Paraphrasing"):
        if bsz == 1:
            batch_results = [parrot.augment(input_phrase=batch[0], **augment_kwargs)]
        else:
            batch_results = parrot.augment_batch(input_phrases=batch, **augment_kwargs)

        for sent, paraphrased in zip(batch, batch_results):
            paraphrased = [well_formed_sentence(para, end_sent=True) for para in paraphrased]
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

    parser = argparse.ArgumentParser(description="Smoke-test the Parrot paraphrasing attack.")
    parser.add_argument("--text", action="append", help="Text to paraphrase. May be passed multiple times.")
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--bigram", action="store_true")
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
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    parrot = SParrot(use_gpu=True)
    output, beams = _paraphrase(
        parrot, texts, tokenizer,
        num_beams=args.num_beams,
        bigram=args.bigram,
        bsz=args.batch_size,
    )
    for i, (text, para) in enumerate(zip(texts, output), start=1):
        print(f"\n[{i}] input:\n{text}")
        print(f"\n[{i}] output:\n{para}")
    print(f"\nGenerated beams for {len(beams)} sentences.")


if __name__ == "__main__":
    _main()
