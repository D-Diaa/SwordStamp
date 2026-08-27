import os
import sys
import json
import random
import argparse

import torch
import numpy as np
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from datasets import load_from_disk, Dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import torch.nn.functional as F


def set_random_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_random_vectors(dim, num, seed=42):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((dim, num * 6))
    Q, _ = np.linalg.qr(A)
    return torch.tensor(Q.T[:num, :])


def get_random_flags(num=4, rng=random):
    return rng.choices([True, False], k=num)


def get_doc_random_flags(num, pivot_seed, doc_idx):
    """Per-document flag pattern R.

    The paper specifies R is "chosen uniformly at random for each query on the generation
    side and is not shared with the detection side" -- i.e. a fresh draw per document, which
    is what makes the detector's blind sign-inference necessary in the first place. Drawing
    it from a (pivot_seed, doc_idx)-derived stream keeps runs reproducible and resume-safe:
    skipping rows that already exist on disk cannot shift the sequence for later rows.
    """
    return get_random_flags(num, random.Random(f"{pivot_seed}:{doc_idx}"))


def get_text_embeddings(texts, embedder):
    return embedder.encode(texts, convert_to_tensor=True).cpu()


def get_cosine_similarities(A, B):
    B = B.to(dtype=A.dtype, device=A.device)
    return F.cosine_similarity(A[:, None, :], B[None, :, :], dim=-1)


def clean_sentence(sentence):
    sentence = sentence.replace("\n", "").replace("\u201c", '"').replace("\u201d", '"').rstrip()
    endings = {".", "?", "!", '"'}
    if not sentence or sentence[-1] not in endings:
        sentence += "."
    return sentence


def extract_first_new_sentence(full_text, prompt):
    sentences = sent_tokenize(full_text)
    prompt_sentences = sent_tokenize(prompt)
    if len(sentences) > len(prompt_sentences):
        return clean_sentence(sentences[len(prompt_sentences)])
    return None


def get_ngrams(text, n=4):
    tokens = text.lower().split()
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def ngram_overlap_ratio(candidate, context_ngrams, n=4):
    cand_ngrams = get_ngrams(candidate, n)
    if not cand_ngrams:
        return 0.0
    overlap = cand_ngrams & context_ngrams
    return len(overlap) / len(cand_ngrams)


def filter_candidates_by_diversity(candidates, context_text, prev_embeddings,
                                   embedder, ngram_overlap_thresh=1,
                                   semantic_sim_thresh=1, n=4):
    context_ngrams = get_ngrams(context_text, n)
    if context_ngrams:
        stage1 = [
            c for c in candidates
            if ngram_overlap_ratio(c, context_ngrams, n) < ngram_overlap_thresh
        ]
        if stage1:
            candidates = stage1

    if prev_embeddings is not None and len(prev_embeddings) > 0 and len(candidates) > 1:
        cand_embs = get_text_embeddings(candidates, embedder)
        sims = get_cosine_similarities(cand_embs, prev_embeddings)
        max_sims = sims.max(dim=1).values
        keep_mask = max_sims < semantic_sim_thresh
        if keep_mask.any():
            candidates = [c for c, keep in zip(candidates, keep_mask.tolist()) if keep]

    return candidates


def vocab_novelty_scores(candidates, context_vocab):
    scores = []
    for c in candidates:
        words = set(c.lower().split())
        if not words:
            scores.append(0.0)
            continue
        new_words = words - context_vocab
        scores.append(len(new_words) / len(words))
    return torch.tensor(scores)


def select_best_sentence(candidates, embedder, msig, rand_flags,
                         epsilon=20, prev_embeddings=None, diversity_weight=0.0,
                         novelty_weight=0.0, context_vocab=None,
                         cossim_transform="none", cossim_transform_param=30.0,
                         pivot_seed=42):
    if not candidates:
        return None

    gen_embeddings = get_text_embeddings(candidates, embedder)
    pivot_vec = get_random_vectors(gen_embeddings.shape[1], msig, seed=pivot_seed)
    pivot_vec = pivot_vec.to(gen_embeddings.device)
    gen_cossim = get_cosine_similarities(gen_embeddings, pivot_vec)

    if not isinstance(rand_flags, torch.Tensor):
        rand_flags = torch.tensor(rand_flags)
    rand_flags = rand_flags.to(gen_cossim.device)

    weights = torch.where(rand_flags, 1.0, -1.0)
    signed_cossim = gen_cossim * weights
    if cossim_transform == "tanh":
        signed_cossim = torch.tanh(cossim_transform_param * signed_cossim)
    scores = torch.sum(signed_cossim, dim=1)

    if diversity_weight > 0 and prev_embeddings is not None and len(prev_embeddings) > 0:
        sims = get_cosine_similarities(gen_embeddings, prev_embeddings)
        max_sims = sims.max(dim=1).values
        scores = scores + diversity_weight * (1.0 - max_sims)

    if novelty_weight > 0 and context_vocab is not None:
        novelty = vocab_novelty_scores(candidates, context_vocab).to(scores.device)
        scores = scores + novelty_weight * novelty

    sign_match = (gen_cossim > 0) == rand_flags
    channels_matched = sign_match.sum(dim=1)
    max_match = channels_matched.max().item()
    mask = channels_matched == max_match

    masked_scores = scores.clone()
    masked_scores[~mask] = -1e9
    valid_scores = masked_scores[mask]
    probs = F.softmax(epsilon * valid_scores, dim=0)
    chosen_idx = torch.multinomial(probs, 1).item()
    chosen_pos = mask.nonzero(as_tuple=True)[0][chosen_idx].item()

    return {
        "text": candidates[chosen_pos],
        "cossim": gen_cossim[chosen_pos].tolist(),
        "score": scores[chosen_pos].item(),
        "channels_satisfied": int(channels_matched[chosen_pos].item()),
        "total_num": len(candidates),
        "rand_flag": rand_flags.tolist(),
        "embedding": gen_embeddings[chosen_pos],
    }


def generate_candidates_vllm(llm, prompt, num_samples, max_tokens=128,
                             temperature=0.7, top_p=0.95, repetition_penalty=1.0):
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        n=num_samples,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_tokens=max_tokens,
    )
    outputs = llm.generate([prompt], sampling_params, use_tqdm=False)

    candidates = []
    for output in outputs:
        for gen in output.outputs:
            full_text = prompt + gen.text
            new_sen = extract_first_new_sentence(full_text, prompt)
            if new_sen:
                candidates.append(new_sen)
    return list(set(candidates))


def parse_args():
    p = argparse.ArgumentParser(description="SAMark generation (Token-level=None only)")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--embedder_path", type=str, required=True)
    p.add_argument("--data_name", type=str, default="booksum", choices=["c4", "booksum"],
                    help="legacy shorthand: 'c4' or 'booksum' (ignored when --data_path is set)")
    p.add_argument("--data_path", type=str, default=None,
                    help="path to a HuggingFace dataset saved to disk (takes precedence over --data_name)")
    p.add_argument("--log_dir", type=str, required=True)
    p.add_argument("--msig", type=int, default=2)
    p.add_argument("--num_samples", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--repetition_penalty", type=float, default=1.0)
    p.add_argument("--flag_scope", choices=["run", "document"], default="run",
                   help="'run' reproduces the released code; 'document' follows the paper's "
                        "per-query target-pattern description")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None,
                    help="exclusive end row; defaults to the full dataset length")
    p.add_argument("--min_new_tokens", type=int, default=205)
    p.add_argument("--min_new_sentences", type=int, default=12)
    p.add_argument("--epsilon", type=float, default=80.0)
    p.add_argument("--ngram_overlap_thresh", type=float, default=0.5)
    p.add_argument("--semantic_sim_thresh", type=float, default=0.85)
    p.add_argument("--diversity_weight", type=float, default=0.3)
    p.add_argument("--novelty_weight", type=float, default=0.15)
    p.add_argument("--cossim_transform", type=str, default="tanh", choices=["none", "tanh"])
    p.add_argument("--cossim_transform_param", type=float, default=30.0)
    p.add_argument("--pivot_seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    if args.data_path:
        dataset = load_from_disk(args.data_path)
    elif args.data_name == "c4":
        dataset = load_from_disk("./data/c4-val-500")
    else:
        dataset = load_from_disk("./data/booksum-train-500")

    if args.end is None:
        args.end = len(dataset)
    args.end = min(args.end, len(dataset))
    args.start = max(0, args.start)

    all_done = all(os.path.exists(f"{args.log_dir}/{i}.json") for i in range(args.start, args.end))
    if all_done:
        print(f"All samples [{args.start}, {args.end}) already exist in {args.log_dir}, skipping generation.")
        _export_hf_dataset(args.log_dir)
        return

    set_random_seed(args.pivot_seed)

    run_rand_flags = None
    if args.flag_scope == "run":
        run_rand_flags = torch.tensor(get_random_flags(args.msig), dtype=torch.bool)

    from vllm import LLM

    llm = LLM(
        model=args.model_path,
        gpu_memory_utilization=0.8,
        trust_remote_code=True,
        tensor_parallel_size=1,
        max_model_len=4096,
        seed=42,
        dtype="bfloat16",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    embedder = SentenceTransformer(args.embedder_path, device="cuda")

    with open(f"{args.log_dir}/config.json", "w") as f:
        flag_metadata = (run_rand_flags.tolist() if run_rand_flags is not None
                         else "per-document (see rand_flags in each {i}.json)")
        json.dump({**args.__dict__, "rand_flags": flag_metadata},
                  f, ensure_ascii=False, indent=4)

    for i in tqdm(range(args.start, args.end)):
        out_path = f"{args.log_dir}/{i}.json"
        if os.path.exists(out_path):
            continue

        rand_flags = (
            run_rand_flags
            if run_rand_flags is not None
            else torch.tensor(get_doc_random_flags(args.msig, args.pivot_seed, i), dtype=torch.bool)
        )

        text = dataset[i]["text"]
        prompt = sent_tokenize(text)[0]

        generated_text = ""
        log = {"prompt": prompt, "generated_text": "", "original_text": text,
               "flag_scope": args.flag_scope, "rand_flags": rand_flags.tolist(), "log": []}

        it = 0
        prev_embeddings = []
        context_vocab = set(prompt.lower().split())

        while True:
            it += 1
            candidates = generate_candidates_vllm(
                llm,
                prompt,
                args.num_samples,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            )
            if not candidates:
                break

            prev_emb_tensor = torch.stack(prev_embeddings) if prev_embeddings else None
            filtered = filter_candidates_by_diversity(
                candidates,
                generated_text,
                prev_emb_tensor,
                embedder,
                ngram_overlap_thresh=args.ngram_overlap_thresh,
                semantic_sim_thresh=args.semantic_sim_thresh,
            )

            sample = select_best_sentence(
                filtered,
                embedder,
                args.msig,
                rand_flags,
                epsilon=args.epsilon,
                prev_embeddings=prev_emb_tensor,
                diversity_weight=args.diversity_weight,
                novelty_weight=args.novelty_weight,
                context_vocab=context_vocab,
                cossim_transform=args.cossim_transform,
                cossim_transform_param=args.cossim_transform_param,
                pivot_seed=args.pivot_seed,
            )
            if sample is None:
                break

            next_sentence = sample["text"]
            prev_embeddings.append(sample.pop("embedding"))
            log["log"].append(sample)

            context_vocab.update(next_sentence.lower().split())
            prompt = prompt + " " + next_sentence
            generated_text = generated_text + " " + next_sentence
            log["generated_text"] = generated_text

            gen_ids = tokenizer(generated_text, return_tensors="pt")["input_ids"]
            if gen_ids.shape[1] > args.min_new_tokens and it >= args.min_new_sentences:
                break

        with open(out_path, "w") as f:
            json.dump(log, f, ensure_ascii=False, indent=4)

    _export_hf_dataset(args.log_dir)
    print("Done!")


def _export_hf_dataset(log_dir):
    """Bridge SAMark's per-sample JSON logs into an HF dataset with a 'text' column,
    mirroring PMark's Q3 export. Written into log_dir itself (alongside
    the {i}.json logs and config.json) so io.output_dir can point our paraphrasing/quality
    modules directly at it, the same way the PMark comparison is bridged."""
    rows = []
    for fname in sorted(os.listdir(log_dir)):
        if not (fname.endswith(".json") and fname[:-5].isdigit()):
            continue
        idx = int(fname[:-5])
        with open(f"{log_dir}/{fname}", "r", encoding="utf-8") as f:
            log = json.load(f)
        rows.append({
            "samark_idx":    idx,
            "text":          (log["prompt"] + log["generated_text"]).strip(),
            "original_text": log["original_text"],
            "prompt":        log["prompt"],
            "flag_scope":    log.get("flag_scope"),
            # Per-document ground-truth flag pattern; absent in pre-per-doc-flag runs.
            "rand_flags":    log.get("rand_flags"),
        })
    if rows:
        rows.sort(key=lambda r: r["samark_idx"])
        Dataset.from_list(rows).save_to_disk(log_dir)
        print(f"Saved HF dataset ({len(rows)} samples) -> {log_dir}")


if __name__ == "__main__":
    main()
