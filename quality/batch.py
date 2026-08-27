"""Evaluate many directories while loading each quality model once."""
from __future__ import annotations

import argparse
import csv
import dataclasses
import gc
import os

import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM

from config.cli import add_config_args, resolve
from config.paths import watermarked_sibling_for
from quality.evaluator import (
    _empty_pair_results,
    _gens_to_text,
    evaluate_llm_judge_metrics,
)
from quality.metrics import (
    _is_openai_model,
    build_llm_judge_pipe,
    clear_emb_sim_model_cache,
    evaluate_anchor_structure,
    evaluate_bertscore,
    evaluate_bleu,
    evaluate_embedding_similarity,
    evaluate_diversity,
    evaluate_entropy,
    evaluate_mauve,
    evaluate_ngrams,
    evaluate_perplexity,
    evaluate_rouge,
    evaluate_semantic_entropy,
)


class Target:
    """Hold one directory's inputs and accumulated results."""

    def __init__(self, d, qcfg):
        self.dir = d
        ds = load_from_disk(d)
        cols = ds.column_names
        self.stage = "attack" if "para_text" in cols else "watermark"
        self.column = qcfg.column or ("para_text" if self.stage == "attack" else "text")
        self.gens = ds[self.column]
        self.gens_text = _gens_to_text(self.gens)
        self.n = len(self.gens_text)
        self.orig_wm = ds["text"] if "text" in cols else self.gens
        self.eval_pairwise = self.stage == "attack" and "text" in cols

        skip = qcfg.skip_per_pair
        self.skip_per_pair = (self.stage == "watermark") if skip is None else skip

        # Reference for pair / MAUVE metrics.
        if qcfg.reference is not None:
            self.ref_texts = load_from_disk(qcfg.reference)["text"]
        elif self.stage == "attack":
            sib = watermarked_sibling_for(d)
            if sib is None:
                raise ValueError(f"no watermarked sibling for attack dir {d}")
            self.ref_texts = load_from_disk(sib)["text"]
        else:
            if qcfg.corpus is None:
                raise ValueError(f"watermark-stage dir {d} needs quality.corpus (or quality.reference)")
            self.ref_texts = load_from_disk(qcfg.corpus)["text"]

        # Clustering corpus for semantic entropy (None when a prebuilt k-means is used).
        if qcfg.corpus is not None and qcfg.load_kmeans_path is None:
            self.corpus_texts = load_from_disk(qcfg.corpus)["text"]
        else:
            self.corpus_texts = None

        self.results = {}
        self.ok = True
        # Seed the skipped pair-metric schema for watermark targets.
        if self.skip_per_pair:
            self.results.update(_empty_pair_results(self.n))


def _write_results(dataset_dir, results):
    """Write scalar, per-sample, and ragged results separately."""
    per_sample = {k[: -len("_per_sample")]: np.asarray(v)
                  for k, v in results.items() if k.endswith("_per_sample")}
    scalars = {k: v for k, v in results.items() if not k.endswith("_per_sample")}
    aux = {k: v for k, v in per_sample.items() if v.dtype == object}
    per_sample = {k: v for k, v in per_sample.items() if v.dtype != object}

    csv_path = os.path.join(dataset_dir, "eval_quality.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=scalars.keys())
        w.writeheader()
        w.writerow(scalars)
    if per_sample:
        np.savez(os.path.join(dataset_dir, "eval_quality_per_sample.npz"), **per_sample)
    if aux:
        np.savez(os.path.join(dataset_dir, "eval_quality_per_sample_aux.npz"), **aux)


def _phase(name, targets, fn):
    """Run fn(t) for each still-ok target, isolating per-target failures."""
    print(f"\n===== phase: {name}  ({sum(t.ok for t in targets)} dirs) =====")
    for t in targets:
        if not t.ok:
            continue
        try:
            fn(t)
        except Exception as e:
            print(f"  ! {t.dir}: {name} failed: {e} — dropping dir from this run")
            t.ok = False


def build_qcfg(cfg):
    q = cfg.quality
    load_kmeans_path = q.load_kmeans_path
    if load_kmeans_path is None and q.corpus is not None and \
            any(f.endswith(".pkl") for f in os.listdir(q.corpus)):
        print("Found pre-computed k-means centroid in corpus path")
        load_kmeans_path = q.corpus
    judge_len_prompt = q.judge_len_prompt if q.judge_len_prompt is not None else cfg.generation.len_prompt
    return dataclasses.replace(q, load_kmeans_path=load_kmeans_path, judge_len_prompt=judge_len_prompt)


def _is_stale(directory, done_path):
    """True if any dataset shard in ``directory`` is newer than ``done_path``."""
    try:
        done_at = os.path.getmtime(done_path)
    except OSError:
        return True
    for name in os.listdir(directory):
        if name.endswith(".arrow") or name == "dataset_info.json":
            try:
                if os.path.getmtime(os.path.join(directory, name)) > done_at:
                    return True
            except OSError:
                continue
    return False


def run(dirs, cfg, *, force=False, cpu_only=False):
    """Evaluate explicit directories with phase-ordered model lifetimes."""
    qcfg = build_qcfg(cfg)

    # Select + load targets (skip already-done unless --force). A result that
    # predates the text it describes is not "done": regenerating a cell leaves
    # its old eval_quality.csv in place, and a presence-only check would keep
    # reporting quality measured on text that no longer exists.
    selected_dirs = []
    for d in dirs:
        if not os.path.isdir(d):
            print(f"skip (not a dir): {d}"); continue
        done = os.path.join(d, "eval_quality.csv")
        if not force and os.path.exists(done):
            if _is_stale(d, done):
                print(f"redo (eval_quality.csv predates its dataset): {d}")
            else:
                print(f"skip (has eval_quality.csv): {d}"); continue
        selected_dirs.append(d)
    if not selected_dirs:
        print("Nothing to do (all targets present; use --force to redo)."); return

    print(f"Loading {len(selected_dirs)} targets...")
    targets = []
    for d in selected_dirs:
        try:
            targets.append(Target(d, qcfg))
        except Exception as e:
            print(f"  ! {d}: load failed: {e} — skipped")
    if not targets:
        print("No loadable targets."); return

    # Pair metrics apply only to attack targets.
    pair_targets = [t for t in targets if not t.skip_per_pair]

    _phase(
        "CPU diversity (SD / D-2 / 4g)",
        targets,
        lambda t: t.results.update(evaluate_diversity(t.gens_text)),
    )

    def cpu_phase(t):
        t.results["bleu"] = evaluate_bleu(list(t.gens_text), list(t.ref_texts))
        t.results.update(evaluate_rouge(list(t.gens_text), list(t.ref_texts)))
        t.results.update(evaluate_anchor_structure(list(t.gens_text), list(t.ref_texts)))

    if not cpu_only:
    # Phase 1: causal-LM metrics.
        print(f"\nLoading causal LM {qcfg.model_path} (bf16)...")
        tokenizer = AutoTokenizer.from_pretrained(qcfg.model_path)
        tokenizer.pad_token = tokenizer.eos_token
        pad_id = tokenizer.encode(tokenizer.eos_token)[0]
        model = AutoModelForCausalLM.from_pretrained(
            qcfg.model_path, return_dict=True, pad_token_id=pad_id, dtype=torch.bfloat16,
        ).to("cuda")
        model.eval()

        def lm_phase(t):
            t.results.update(evaluate_perplexity(model, tokenizer, t.gens_text))
            t.results["sem_ent"] = evaluate_semantic_entropy(
                model, t.gens, tokenizer, t.corpus_texts, qcfg, t.dir)
            t.results.update(evaluate_ngrams(t.gens_text, tokenizer))
            bi, tri = evaluate_entropy(t.gens_text, tokenizer)
            t.results["bi_entro"], t.results["tri_entro"] = bi, tri

        _phase("causal-LM (ppl / sem_ent / ngrams / entropy)", targets, lm_phase)

        del model
        clear_emb_sim_model_cache()
        gc.collect()
        torch.cuda.empty_cache()

        # Phase 2: MAUVE.
        _phase("MAUVE", targets,
               lambda t: t.results.__setitem__("mauve", evaluate_mauve(list(t.gens_text), list(t.ref_texts))))

        # Phases 3–4: GPU pairwise metrics.
        _phase("BERTScore", pair_targets,
               lambda t: t.results.update(evaluate_bertscore(list(t.gens_text), list(t.ref_texts))))
        _phase("embedding similarity", pair_targets,
               lambda t: t.results.update(evaluate_embedding_similarity(
                   list(t.gens_text), list(t.ref_texts), model_name=qcfg.emb_sim_model)))
        clear_emb_sim_model_cache()
        gc.collect()
        torch.cuda.empty_cache()

    # Phase 5: CPU pairwise metrics.
    _phase("CPU (BLEU / ROUGE / anchors)", pair_targets, cpu_phase)

    # Phase 6: one shared LLM judge.
    written_dirs = set()
    if not cpu_only:
        judge_pipe = None
        if qcfg.judge_model and not _is_openai_model(qcfg.judge_model):
            print(f"\nBuilding vLLM judge {qcfg.judge_model} (reused across all dirs)...")
            judge_pipe = build_llm_judge_pipe(
                qcfg.judge_model,
                gpu_memory_utilization=cfg.runtime.vllm_utilization,
            )
        try:
            def judge_phase(t):
                t.results.update(
                    evaluate_llm_judge_metrics(
                        t.gens,
                        t.orig_wm,
                        qcfg,
                        judge_pipe=judge_pipe,
                        eval_intrinsic=t.stage == "watermark",
                        eval_pairwise=t.eval_pairwise,
                    )
                )
                _write_results(t.dir, t.results)
                written_dirs.add(t.dir)

            _phase("LLM judge", targets, judge_phase)
        finally:
            if judge_pipe is not None:
                del judge_pipe
                gc.collect()
                torch.cuda.empty_cache()

    # Write results.
    written = 0
    for t in targets:
        if not t.ok:
            print(f"NOT written (had a phase failure): {t.dir}"); continue
        if t.dir not in written_dirs:
            _write_results(t.dir, t.results)
        written += 1
    print(f"\nDone. Wrote eval_quality.csv (+ .npz) to {written}/{len(targets)} dirs.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser, positional_data=False)
    parser.add_argument("dirs", nargs="+", help="Explicit list of directories to evaluate.")
    parser.add_argument("--force", action="store_true",
                        help="Re-evaluate even dirs that already have eval_quality.csv.")
    parser.add_argument("--cpu-only", action="store_true",
                        help="Run only the CPU metrics (diversity/BLEU/ROUGE/anchors); skip every "
                             "GPU phase and the judge. For validating the orchestration/piping "
                             "without a GPU.")
    args = parser.parse_args()
    run(args.dirs, resolve(args), force=args.force, cpu_only=args.cpu_only)


if __name__ == "__main__":
    main()
