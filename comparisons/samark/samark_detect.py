import argparse
import csv
import json
import os
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


@lru_cache(maxsize=None)
def _random_vectors_cached(dim, num, seed):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((dim, num * 6))
    Q, _ = np.linalg.qr(A)
    return torch.tensor(Q.T[:num, :])


def get_random_vectors(dim, num, seed=42):
    """Return deterministic pivot directions, memoized for detection speed."""
    return _random_vectors_cached(dim, num, seed)


def get_text_embeddings(texts, embedder):
    return embedder.encode(texts, convert_to_tensor=True).cpu()


def get_cosine_similarities(A, B):
    B = B.to(dtype=A.dtype, device=A.device)
    return F.cosine_similarity(A[:, None, :], B[None, :, :], dim=-1)


def apply_transform(x, mode="none", k=30.0):
    if mode == "tanh":
        return torch.tanh(k * x)
    return x


def detect_sentence_level(sentences, embedder, msig, transform="none", transform_k=30.0):
    """Return SAMark's raw document score without making a detection decision."""
    if not sentences or msig == 0:
        return {
            "z_score": 0.0,
            "mean_score": 0.0,
            "aligned_channels": [],
            "guessed_flags": [],
            "transform": transform,
            "transform_k": transform_k,
        }

    embs = get_text_embeddings(list(sentences), embedder)
    pivot = get_random_vectors(embs.shape[1], msig, seed=42).to(embs.device)
    all_cossims = get_cosine_similarities(embs, pivot)

    channel_sums = all_cossims.sum(dim=0)
    guessed_flags = torch.where(channel_sums >= 0, 1.0, -1.0)

    aligned = apply_transform(all_cossims * guessed_flags, mode=transform, k=transform_k)
    per_sent_aligned = (aligned > 0).sum(dim=1).tolist()

    flat = aligned.view(-1)
    mean_s = flat.mean().item()
    std_s = flat.std().item()
    n = flat.numel()
    z_score = 0.0 if not np.isfinite(std_s) or std_s < 1e-12 else mean_s * (n ** 0.5) / std_s

    return {
        "z_score": float(z_score),
        "mean_score": mean_s,
        "guessed_flags": guessed_flags.tolist(),
        "aligned_channels": per_sent_aligned,
        "transform": transform,
        "transform_k": transform_k,
    }


def calibrated_threshold(null_scores, fpr_target):
    """Choose a cutoff whose empirical strict-tail FPR does not exceed the target."""
    if not 0 < fpr_target < 1:
        raise ValueError(f"fpr_target must be between 0 and 1, got {fpr_target!r}")
    valid = np.asarray(null_scores, dtype=float)
    valid = valid[~np.isnan(valid)]
    if not len(valid):
        raise ValueError("Cannot calibrate from an empty human-null distribution")
    return float(np.quantile(valid, 1 - fpr_target, method="higher"))


def auroc(pos_scores, null_scores):
    pos = np.asarray(pos_scores, dtype=float)
    neg = np.asarray(null_scores, dtype=float)
    pos = pos[~np.isnan(pos)]
    neg = neg[~np.isnan(neg)]
    if not len(pos) or not len(neg):
        return float("nan")
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    return float(roc_auc_score(labels, np.concatenate([pos, neg])))


def summarize(pos_scores, label, human_scores, fpr_target, human_text, human_max_sentences=12):
    pos = np.asarray(pos_scores, dtype=float)
    human = np.asarray(human_scores, dtype=float)
    pos = pos[~np.isnan(pos)]
    human = human[~np.isnan(human)]
    threshold = calibrated_threshold(human, fpr_target)
    return {
        "column": label,
        "threshold_source": (
            f"human:{os.path.basename(human_text)}:first-{human_max_sentences}-sentences"
        ),
        "fpr_target": float(fpr_target),
        "threshold": round(threshold, 6),
        "empirical_fpr": round(float(np.mean(human > threshold)), 6),
        "tpr": round(float(np.mean(pos > threshold)), 6) if len(pos) else "",
        "auroc": round(auroc(pos, human), 6),
        "n": int(len(pos)),
        "n_null": int(len(human)),
        "mean_z": round(float(np.mean(pos)), 6) if len(pos) else "",
        "human_mean_z": round(float(np.mean(human)), 6),
    }


def operating_points(extra_target=None):
    """Return the standard pipeline operating points plus an optional extra one."""
    targets = [0.01, 0.05]
    if extra_target is not None and extra_target not in targets:
        targets.append(float(extra_target))
    return tuple(sorted(targets))


def summarize_operating_points(
    pos_scores, label, human_scores, human_text, human_max_sentences=12,
    extra_target=None,
):
    return [
        summarize(
            pos_scores, label, human_scores, target, human_text,
            human_max_sentences,
        )
        for target in operating_points(extra_target)
    ]


def write_full_results(output_dir, rows):
    path = os.path.join(output_dir, "results_full.csv")
    fieldnames = [
        "column", "threshold_source", "fpr_target", "threshold", "empirical_fpr",
        "tpr", "auroc", "n", "n_null", "mean_z", "human_mean_z",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def compact_result(rows, label):
    """Return the shared single-row result schema for one score column."""
    by_fpr = {
        float(row["fpr_target"]): row
        for row in rows
        if row["column"] == label
    }
    if 0.01 not in by_fpr or 0.05 not in by_fpr:
        raise ValueError(f"Missing standard operating points for {label!r}")
    fpr1 = by_fpr[0.01]
    fpr5 = by_fpr[0.05]
    return {
        "auroc": fpr1["auroc"],
        # Historical pipeline names: these are TPR at the named FPR.
        "fpr1": fpr1["tpr"],
        "fpr5": fpr5["tpr"],
        "threshold_source": fpr1["threshold_source"],
        "mean_z": fpr1["mean_z"],
        "human_mean_z": fpr1["human_mean_z"],
    }


def write_compact_result(path, row):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def score_text(text, embedder, args, max_sentences=None):
    sentences = sent_tokenize(text)
    if max_sentences is not None:
        sentences = sentences[:max_sentences]
    return detect_sentence_level(
        sentences,
        embedder,
        args.msig,
        transform=args.transform,
        transform_k=args.transform_param,
    )["z_score"]


def score_human_null(args, embedder):
    dataset = load_from_disk(args.human_text)
    texts = dataset["text"][: min(args.human_n, len(dataset))]
    print(
        f"Scoring {len(texts)} independent human texts from {args.human_text} "
        f"(first {args.human_max_sentences} sentences)..."
    )
    return np.asarray([
        score_text(text, embedder, args, max_sentences=args.human_max_sentences)
        for text in tqdm(texts)
    ], dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(
        description="SAMark raw scoring with empirical human-null calibration"
    )
    parser.add_argument("--mode", default="legacy", choices=["legacy", "hf"])
    parser.add_argument("--embedder_path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--msig", type=int, default=2)
    parser.add_argument("--transform", default="tanh", choices=["none", "tanh"])
    parser.add_argument("--transform_param", type=float, default=30.0)
    parser.add_argument("--human_text", default="data/c4-human-def")
    parser.add_argument("--human_n", type=int, default=1024)
    parser.add_argument("--human_max_sentences", type=int, default=12)
    parser.add_argument(
        "--fpr_target", type=float, default=0.01,
        help="Optional extra operating point; 1%% and 5%% are always reported",
    )

    # Legacy per-file JSON mode.
    parser.add_argument("--log_dir", default=None)
    parser.add_argument("--sub_dir", default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=500)
    parser.add_argument("--detect_subdir", default="detect")

    # HuggingFace dataset bridge mode.
    parser.add_argument("--hf_dataset", default=None)
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


def run_hf(args, embedder):
    dataset = load_from_disk(args.hf_dataset)
    output_dir = args.output_dir or os.path.join(args.hf_dataset, args.detect_subdir)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(args.__dict__, f, ensure_ascii=False, indent=4)

    print(f"Scoring {len(dataset)} target texts...")
    z_text = np.asarray([score_text(row["text"], embedder, args) for row in tqdm(dataset)],
                        dtype=np.float32)
    np.save(os.path.join(output_dir, "z_scores_text.npy"), z_text)

    score_sets = [("text", z_text)]
    if "para_text" in dataset.column_names:
        print("Scoring para_text...")
        z_para = np.asarray([
            score_text(row["para_text"], embedder, args)
            if row.get("para_text") else float("nan")
            for row in tqdm(dataset)
        ], dtype=np.float32)
        np.save(os.path.join(output_dir, "z_scores_para.npy"), z_para)
        score_sets.append(("para_text", z_para))

    human_scores = score_human_null(args, embedder)
    np.save(os.path.join(output_dir, "human_z_scores.npy"), human_scores)
    rows = [
        row
        for label, scores in score_sets
        for row in summarize_operating_points(
            scores, label, human_scores, args.human_text,
            args.human_max_sentences, extra_target=args.fpr_target,
        )
    ]
    full_path = write_full_results(output_dir, rows)
    wm = compact_result(rows, "text")
    write_compact_result(os.path.join(output_dir, "results_wm.csv"), wm)
    result = compact_result(rows, "para_text") if "para_text" in dict(score_sets) else wm
    write_compact_result(os.path.join(output_dir, "results.csv"), result)
    print(f"Results -> {full_path} (+ results.csv / results_wm.csv)")
    for row in rows:
        print(
            f"  [{row['column']}] FPR={row['fpr_target']:.2f} "
            f"threshold={row['threshold']:.4f} "
            f"TPR={row['tpr']:.4f} AUROC={row['auroc']:.4f} "
            f"empirical FPR={row['empirical_fpr']:.4f}"
        )


def run_legacy(args, embedder):
    if not args.log_dir:
        raise SystemExit("--log_dir is required for --mode legacy")
    data_dir = os.path.join(args.log_dir, args.sub_dir) if args.sub_dir else args.log_dir
    detect_dir = os.path.join(data_dir, args.detect_subdir)
    os.makedirs(detect_dir, exist_ok=True)
    with open(os.path.join(detect_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(args.__dict__, f, ensure_ascii=False, indent=4)

    for index in tqdm(range(args.start, args.end)):
        log_path = os.path.join(data_dir, f"{index}.json")
        out_path = os.path.join(detect_dir, f"{index}.json")
        if not os.path.exists(log_path) or os.path.exists(out_path):
            continue
        with open(log_path, encoding="utf-8") as f:
            generated_text = json.load(f).get("generated_text", "")
        result = detect_sentence_level(
            sent_tokenize(generated_text),
            embedder,
            args.msig,
            transform=args.transform,
            transform_k=args.transform_param,
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"watermark_log": {"sentence_level": result}}, f,
                      ensure_ascii=False, indent=4)

    scores = []
    for index in range(args.start, args.end):
        path = os.path.join(detect_dir, f"{index}.json")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            scores.append(json.load(f)["watermark_log"]["sentence_level"]["z_score"])
    if not scores:
        print("No valid samples found.")
        return

    human_scores = score_human_null(args, embedder)
    np.save(os.path.join(detect_dir, "human_z_scores.npy"), human_scores)
    rows = summarize_operating_points(
        scores, "generated_text", human_scores, args.human_text,
        args.human_max_sentences, extra_target=args.fpr_target,
    )
    full_path = write_full_results(detect_dir, rows)
    result = compact_result(rows, "generated_text")
    write_compact_result(os.path.join(detect_dir, "results_wm.csv"), result)
    write_compact_result(os.path.join(detect_dir, "results.csv"), result)
    print(f"Results -> {full_path} (+ results.csv / results_wm.csv)")


def main():
    args = parse_args()
    embedder = SentenceTransformer(args.embedder_path, device=args.device)
    if args.mode == "hf":
        if not args.hf_dataset:
            raise SystemExit("--hf_dataset is required for --mode hf")
        run_hf(args, embedder)
    else:
        run_legacy(args, embedder)


if __name__ == "__main__":
    main()
