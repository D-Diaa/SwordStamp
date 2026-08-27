"""Quality orchestration adapted from original SemStamp's ``eval_quality.py``."""

import gc

import numpy as np
import torch

from watermarking.primitives import extract_prompt_from_text

from .metrics import (
    evaluate_perplexity,
    evaluate_entropy,
    evaluate_ngrams,
    evaluate_diversity,
    evaluate_mauve,
    evaluate_bertscore,
    evaluate_embedding_similarity,
    evaluate_bleu,
    evaluate_rouge,
    evaluate_anchor_structure,
    evaluate_semantic_entropy,
    evaluate_llm_judge,
    build_llm_judge_pipe,
    empty_rubric_metrics,
    PAIRWISE_RUBRIC,
    QUALITY_RUBRIC,
    _is_openai_model,
)


def _gens_to_text(gens):
    if type(gens[0]) == list:
        return [" ".join(g) for g in gens]
    return gens


def _empty_pair_results(n=0):
    nan_ci = float('nan')
    ps = lambda: np.full(n, np.nan)
    return {
        "bert_P": nan_ci, "bert_P_ci": nan_ci, "bert_P_median": nan_ci,
        "bert_R": nan_ci, "bert_R_ci": nan_ci, "bert_R_median": nan_ci,
        "bert_F1": nan_ci, "bert_F1_ci": nan_ci, "bert_F1_median": nan_ci,
        "emb_sim": nan_ci, "emb_sim_ci": nan_ci, "emb_sim_median": nan_ci,
        "bleu": nan_ci,
        "rouge1": nan_ci, "rouge1_ci": nan_ci, "rouge1_median": nan_ci,
        "rouge2": nan_ci, "rouge2_ci": nan_ci, "rouge2_median": nan_ci,
        "rougeL": nan_ci, "rougeL_ci": nan_ci, "rougeL_median": nan_ci,
        "bert_P_per_sample": ps(), "bert_R_per_sample": ps(), "bert_F1_per_sample": ps(),
        "emb_sim_per_sample": ps(),
        "rouge1_per_sample": ps(), "rouge2_per_sample": ps(), "rougeL_per_sample": ps(),
    }


def evaluate_model_quality_metrics(model, gens, corpus_texts, ref_texts, tokenizer, qcfg, dataset_dir):
    gens_text = _gens_to_text(gens)

    print("Evaluating perplexity...")
    ppl_results = evaluate_perplexity(model, tokenizer, gens_text)

    print("Evaluating semantic entropy")
    sem_ent = evaluate_semantic_entropy(model, gens, tokenizer, corpus_texts, qcfg, dataset_dir)

    print("Evaluating n-gram repetition...")
    ngram_results = evaluate_ngrams(gens_text, tokenizer)

    print("Evaluating deterministic diversity metrics...")
    diversity_results = evaluate_diversity(gens_text)

    print("Evaluating entropy...")
    gen_entros = evaluate_entropy(gens_text, tokenizer)

    print("Evaluating MAUVE...")
    mauve_score = evaluate_mauve(list(gens_text), list(ref_texts))

    return {
        **ppl_results,
        "bi_entro": gen_entros[0],
        "tri_entro": gen_entros[1],
        **ngram_results,
        **diversity_results,
        "sem_ent": sem_ent,
        "mauve": mauve_score,
    }


def evaluate_pair_quality_metrics(gens, ref_texts, qcfg, skip_per_pair):
    gens_text = _gens_to_text(gens)
    if skip_per_pair:
        print("Skipping per-pair metrics (BERTScore, EmbSim, BLEU, ROUGE)")
        return _empty_pair_results(len(gens_text))

    print("Evaluating BERTScore...")
    bert_results = evaluate_bertscore(list(gens_text), list(ref_texts))

    print("Evaluating embedding similarity (EmbeddingGemma)...")
    emb_sim_results = evaluate_embedding_similarity(list(gens_text), list(ref_texts), model_name=qcfg.emb_sim_model)

    print("Evaluating BLEU...")
    bleu = evaluate_bleu(list(gens_text), list(ref_texts))

    print("Evaluating ROUGE...")
    rouge_results = evaluate_rouge(list(gens_text), list(ref_texts))

    print("Evaluating structural edits (patience anchors)...")
    anchor_results = evaluate_anchor_structure(list(gens_text), list(ref_texts))

    return {
        **bert_results,
        **emb_sim_results,
        "bleu": bleu,
        **rouge_results,
        **anchor_results,
    }


def evaluate_llm_judge_metrics(
    gens, orig_wm_texts, qcfg, judge_pipe=None, eval_intrinsic=True,
    eval_pairwise=False,
    vllm_utilization=None,
):
    """Run the requested intrinsic and pairwise rubrics with one engine."""
    n = len(gens)
    judge_model = qcfg.judge_model
    if judge_model is None:
        return {
            **empty_rubric_metrics(QUALITY_RUBRIC, n, qcfg.judge_repeats),
            **empty_rubric_metrics(PAIRWISE_RUBRIC, n, qcfg.judge_repeats),
        }

    gens_text = _gens_to_text(gens)
    orig_text = _gens_to_text(orig_wm_texts)
    len_prompt = qcfg.judge_len_prompt

    quality_samples = []
    if eval_intrinsic:
        for gen, orig in zip(gens_text, orig_text):
            # Strip the original prompt only when the generation includes it.
            prompt = extract_prompt_from_text(orig, len_prompt)
            judged = gen[len(prompt):].lstrip() if gen.startswith(prompt) else gen
            quality_samples.append({"prompt": prompt, "gen": judged})
    pair_samples = (
        [{"gen": g, "ref": o} for g, o in zip(gens_text, orig_text)]
        if eval_pairwise
        else []
    )

    # Build one vLLM engine and reuse it across every rubric (local judges only).
    has_nonempty = any(
        sample["gen"].strip() for sample in quality_samples + pair_samples
    )
    owns_pipe = (
        judge_pipe is None
        and not _is_openai_model(judge_model)
        and has_nonempty
    )
    if owns_pipe:
        judge_pipe = build_llm_judge_pipe(
            judge_model, gpu_memory_utilization=vllm_utilization,
        )

    try:
        results = {}

        if eval_intrinsic:
            print("Evaluating LLM Judge [generation quality]...")
            results.update(
                evaluate_llm_judge(
                    quality_samples, qcfg, rubric=QUALITY_RUBRIC, pipe=judge_pipe
                )
            )
        else:
            print("Skipping LLM Judge [generation quality] (attack evaluation)")
            results.update(
                empty_rubric_metrics(QUALITY_RUBRIC, n, qcfg.judge_repeats)
            )

        if eval_pairwise:
            print("Evaluating LLM Judge [pairwise semantic preservation]...")
            results.update(evaluate_llm_judge(pair_samples, qcfg, rubric=PAIRWISE_RUBRIC, pipe=judge_pipe))
        else:
            print("Skipping LLM Judge [pairwise] (watermarked-only evaluation)")
            results.update(
                empty_rubric_metrics(PAIRWISE_RUBRIC, n, qcfg.judge_repeats)
            )

        return results
    finally:
        if owns_pipe:
            del judge_pipe
            gc.collect()
            torch.cuda.empty_cache()


def evaluate_quality(model, gens, corpus_texts, ref_texts, tokenizer, qcfg, dataset_dir,
                     orig_wm_texts=None, eval_pairwise=False, judge_pipe=None, skip_per_pair=False):
    if orig_wm_texts is None:
        orig_wm_texts = gens
    return {
        **evaluate_model_quality_metrics(model, gens, corpus_texts, ref_texts, tokenizer, qcfg, dataset_dir),
        **evaluate_pair_quality_metrics(gens, ref_texts, qcfg, skip_per_pair),
        **evaluate_llm_judge_metrics(
            gens,
            orig_wm_texts,
            qcfg,
            judge_pipe=judge_pipe,
            eval_intrinsic=not eval_pairwise,
            eval_pairwise=eval_pairwise,
        ),
    }
