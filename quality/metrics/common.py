"""Quality helpers adapted from original SemStamp's ``eval_clm.py``."""

import math
import os

import numpy as np


device = "cuda"

# Bound outlier documents during causal-LM scoring.
_PPL_MAX_LENGTH = 2048


def _confidence_interval(values, confidence=0.95):
    """Compute confidence interval half-width for the mean using t-distribution."""
    from scipy import stats as sp_stats

    a = np.asarray(values, dtype=float)
    a = a[~np.isnan(a)]
    n = len(a)
    if n < 2:
        return float("nan")
    se = np.std(a, ddof=1) / np.sqrt(n)
    t_crit = sp_stats.t.ppf((1 + confidence) / 2, df=n - 1)
    return t_crit * se


_JUDGE_AUDIT_SUFFIXES = (
    "_input_tokens",
    "_truncated_tokens",
    "_valid_repetitions",
    "_accepted_attempt",
    "_attempt_tokens",
    "_attempt_max_tokens",
    "_attempt_seeds",
    "_attempt_finish_reason",
    "_attempt_error",
)


def is_per_sample_scores(name, array):
    """True if ``name`` and ``array`` describe a scalar score per sample.

    ``eval_quality_per_sample.npz`` carries judge auditing arrays alongside the
    scores: per-repetition criterion scores and accepted-attempt indices
    (``N x repeats``), per-attempt token counts, caps and seeds (``N x repeats
    x attempts``), and string finish reasons/errors. Shape alone is not enough:
    input-token, truncation and valid-repetition telemetry are also 1-D numeric
    arrays, so audit names must be excluded explicitly.
    """
    a = np.asarray(array)
    is_audit = name.endswith(_JUDGE_AUDIT_SUFFIXES) or name.endswith("_repeats")
    return not is_audit and a.ndim == 1 and a.dtype.kind in "biuf"


def path_wo_ext(path):
    """Return path without file extension."""
    return os.path.splitext(path)[0]


def tokenize_untokenize(text, tokenizer):
    input_ids = tokenizer(text, return_tensors="pt").input_ids[0]
    tokens = []
    for token_id in input_ids:
        tokens.append(tokenizer.decode(token_id, skip_special_tokens=True))
    return tokens


def text_entropy(sen_lis, k):
    # sen_lis is like [["i", "am", "you", "</s>"], ...].
    counts, total = {}, 0
    for sen in sen_lis:
        for i in range(0, len(sen) - k + 1):
            total += 1
            ngram = " ".join(sen[i:i + k])
            if ngram not in counts:
                counts[ngram] = 0
            counts[ngram] += 1

    entropy = 0.0
    for ngram in counts:
        prob = float(counts[ngram]) / total
        entropy -= math.log(prob) * prob
    return entropy


def eval_perplexity(
    model, tokenizer, texts, generation_file=None, K=500, name_suffix="",
    return_evidence=False,
):
    """Return corpus perplexity and the diagnostic per-document perplexities."""
    import torch
    from tqdm import tqdm

    nlls = []
    ppls = []
    lengths = []
    for i, text in enumerate(tqdm(texts, desc="perplexity")):
        text = text.replace("Paraphrase:", "")
        text = text.replace("paraphrase:", "")

        prefix = tokenizer.bos_token if "GPT2Tokenizer" in type(tokenizer).__name__ else ""
        # Cap single-document attention memory.
        input_ids = tokenizer.encode(
            prefix + text, return_tensors="pt", truncation=True, max_length=_PPL_MAX_LENGTH,
        ).to(device)
        target_ids = input_ids.clone()
        try:
            # Metrics need no autograd graph.
            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
        except Exception:
            print("ppl model error")
            print(f"text=<{text}>")
            print(f"input_ids=<{input_ids}>")
            print(f"index: {i}")
            nlls.append(torch.tensor(float("nan")))
            ppls.append(torch.tensor(float("nan")))
            lengths.append(max(input_ids.size(1) - 1, 0))
            continue

        nll = outputs[0].cpu().detach()
        ppl = torch.exp(nll)
        # Causal-LM loss predicts every token after the first one.  Keep that
        # prediction count so the aggregate can be formed in loss space.
        length = max(input_ids.size(1) - 1, 0)
        nlls.append(nll)
        ppls.append(ppl)
        lengths.append(length)
    nlls = torch.tensor(nlls).float()
    ppls = torch.tensor(ppls).float()
    lengths = torch.tensor(lengths, dtype=torch.int64)
    valid = (~torch.isnan(nlls)) & (lengths > 0)
    scored_tokens = torch.where(valid, lengths, 0)
    total_nlls = torch.where(
        valid,
        nlls * lengths,
        torch.tensor(float("nan")),
    )
    total_length = scored_tokens.sum()
    total_nll = total_nlls.nansum()

    sent_avg_ppl = ppls.nanmean().item()
    if valid.any():
        corpus_nll = total_nll / total_length
        corpus_ppl = torch.exp(corpus_nll).item()
    else:
        corpus_ppl = float("nan")

    print(f"perplexity from model {name_suffix} out of {len(ppls)}:")
    print(f"corpus_ppl={corpus_ppl:.4f}")
    print(f"document_avg_ppl={sent_avg_ppl:.4f}")

    if generation_file is not None:
        k = min(K, len(texts))
        topk_ppls, topk_idx = ppls.topk(k)
        ppl_suffix = f"_{name_suffix}.ppl"
        with open(f"{path_wo_ext(generation_file)}{ppl_suffix}", "w") as f:
            print(f"corpus_ppl={corpus_ppl:.4f}", file=f)
            print(f"document_avg_ppl={sent_avg_ppl:.4f}", file=f)
            for i in range(len(topk_idx)):
                idx = topk_idx[i]
                print("-" * 50, file=f)
                print(f"rank={i}", file=f)
                print(f"index={idx}", file=f)
                print(f"ppl={topk_ppls[i]}", file=f)
                print(f"length={lengths[idx]}", file=f)
                weight = lengths[idx] / total_length if total_length > 0 else float("nan")
                print(f"weight={weight}", file=f)
                print(f"text={texts[idx]}", file=f)

    if return_evidence:
        return (
            corpus_ppl,
            ppls.numpy(),
            total_nlls.numpy(),
            scored_tokens.numpy(),
        )
    return corpus_ppl, ppls.numpy()


def rep_ngram(sen_lis, num_gram=4, return_per_sample=False):
    rep_lis = []
    for sen in sen_lis:
        uniq_ngram, all_ngram = {}, []
        for i in range(0, len(sen) - num_gram + 1):
            ngram = " ".join(sen[i:i + num_gram])
            if ngram not in uniq_ngram:
                uniq_ngram[ngram] = True
            all_ngram.append(ngram)
        if len(all_ngram) == 0:
            print(f"warning: len(all_ngram) is 0!!! sample: {str(sen)}")
            rep_lis.append(float("nan") if return_per_sample else None)
            continue
        rep = 1.0 - len(uniq_ngram) * 1.0 / len(all_ngram)
        rep_lis.append(rep)
    if not return_per_sample:
        rep_lis = [r for r in rep_lis if r is not None]
    if return_per_sample:
        return rep_lis
    return np.mean(rep_lis)


def _main():
    """Smoke-test entropy and repetition on toy texts."""
    texts = [
        "The history of the printing press is a fascinating subject worth studying carefully.",
        "Every scholar who could afford it wanted a printed copy of the newly published book.",
        "The press spread ideas faster than any medium that had existed before its invention.",
    ]
    tokens = [text.lower().split() for text in texts]

    # text_entropy: unigram entropy should be positive; bigram <= unigram
    h1 = text_entropy(tokens, k=1)
    h2 = text_entropy(tokens, k=2)
    assert h1 > 0 and h2 >= 0, (h1, h2)
    print(f"text_entropy: unigram={h1:.4f}, bigram={h2:.4f}")

    # rep_ngram: 4-gram repetition on diverse text should be near 0
    rep = rep_ngram(tokens, num_gram=4)
    assert 0 <= rep <= 1, rep
    print(f"rep_ngram(4): {rep:.4f}")

    print("quality/metrics/common smoke ok")


if __name__ == "__main__":
    _main()
