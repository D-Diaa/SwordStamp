#!/usr/bin/env python3
"""Render one directory's detection and quality summary."""

import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# Repo root on path so config.paths imports regardless of CWD.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.paths import watermarked_sibling_for  # noqa: E402
from visualization.metric_specs import (  # noqa: E402
    DETERMINISTIC_DIVERSITY_METRICS,
    LLM_JUDGE_METRICS,
    LLM_QUALITY_METRICS,
    metric_spec,
)



def setup_plot_style():
    plt.rcParams.update({
        "font.size": 11, "font.family": "serif",
        "axes.labelsize": 12, "axes.titlesize": 13,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 10, "figure.titlesize": 14,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.grid": True, "grid.alpha": 0.3,
    })


def save_figure(output_dir: str, filename: str):
    plt.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, f"{filename}.pdf"), bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}.png/pdf")



def _load_npy(dir_path: str, *names: str):
    """Load the first matching NumPy file alias."""
    for name in names:
        p = os.path.join(dir_path, name)
        if os.path.exists(p):
            return np.load(p)
    return None


def _normal_sf(x):
    """Survival function of the standard normal (analytical N(0,1) null)."""
    from scipy.special import erfc
    return 0.5 * erfc(np.asarray(x, dtype=float) / np.sqrt(2.0))


def _roc_from_z(model_z, null_z):
    """Build an ROC curve from z-scores and an analytic or empirical null."""
    m = np.asarray(model_z, dtype=float)
    m = m[~np.isnan(m)]
    if m.size == 0:
        return None
    thr = np.unique(m)
    thr = np.concatenate([[thr[-1] + 1.0], thr[::-1], [thr[0] - 1.0]])
    tpr = np.array([(m >= t).mean() for t in thr])
    if null_z is None:
        fpr = _normal_sf(thr)
    else:
        n = np.asarray(null_z, dtype=float)
        n = n[~np.isnan(n)]
        if n.size == 0:
            return None
        fpr = np.array([(n >= t).mean() for t in thr])
    order = np.argsort(fpr)
    fpr, tpr = fpr[order], tpr[order]
    trapz = getattr(np, "trapezoid", np.trapz)
    return fpr, tpr, float(trapz(tpr, fpr))


def _load_per_sample_npz(dir_path: str) -> dict:
    """Return {metric: array} from eval_quality_per_sample.npz, or {} if missing."""
    p = os.path.join(dir_path, "eval_quality_per_sample.npz")
    if not os.path.exists(p):
        return {}
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def _try_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _load_csv_row(dir_path: str, name: str) -> dict:
    p = os.path.join(dir_path, name)
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        row = next(csv.DictReader(f), None)
    return {k: _try_float(v) for k, v in row.items()} if row else {}


def _dir_label(dir_path: str) -> str:
    return os.path.basename(dir_path)



def _draw_roc(ax, dir_path: str, z_wm, z_para, null_z):
    """Draw pre- and post-attack ROC curves."""
    fpr_wm = _load_npy(dir_path, "fpr_wm.npy")
    tpr_wm = _load_npy(dir_path, "tpr_wm.npy")
    fpr_para = _load_npy(dir_path, "fpr.npy")
    tpr_para = _load_npy(dir_path, "tpr.npy")
    res_wm = _load_csv_row(dir_path, "results_wm.csv")
    res_para = _load_csv_row(dir_path, "results.csv")
    auroc_wm = res_wm.get("auroc", float("nan"))
    auroc_para = res_para.get("auroc", float("nan"))

    if (fpr_wm is None or tpr_wm is None) and z_wm is not None:
        syn = _roc_from_z(z_wm, null_z)
        if syn is not None:
            fpr_wm, tpr_wm, auc = syn
            if np.isnan(auroc_wm):
                auroc_wm = auc
    if (fpr_para is None or tpr_para is None) and z_para is not None:
        syn = _roc_from_z(z_para, null_z)
        if syn is not None:
            fpr_para, tpr_para, auc = syn
            if np.isnan(auroc_para):
                auroc_para = auc

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1, label="Random")
    if fpr_wm is not None and tpr_wm is not None:
        ax.plot(fpr_wm, tpr_wm, color="steelblue", linewidth=2,
                label=f"Before paraphrase (AUROC={auroc_wm:.3f})")
    if fpr_para is not None and tpr_para is not None:
        ax.plot(fpr_para, tpr_para, color="darkorange", linewidth=2, linestyle="--",
                label=f"After paraphrase (AUROC={auroc_para:.3f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=9)


def _draw_zhist(ax, z_scores, null_z, title: str, label: str):
    """Draw z-scores against an analytic or empirical null."""
    if z_scores is None or len(z_scores) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, color="gray")
        ax.set_title(title)
        return
    mpz = np.nan_to_num(z_scores)
    analytical = null_z is None

    if analytical:
        lo, hi = min(mpz.min(), -4.0), max(mpz.max(), 4.0)
        bins = np.linspace(lo - 0.5, hi + 0.5, 40)
        binw = bins[1] - bins[0]
        ax.hist(mpz, bins=bins, alpha=0.55, label=f"{label} (n={len(mpz)})",
                color="darkorange", edgecolor="black", linewidth=0.4)
        xs = np.linspace(lo, hi, 200)
        pdf = np.exp(-xs ** 2 / 2.0) / np.sqrt(2 * np.pi)
        ax.plot(xs, pdf * len(mpz) * binw, color="steelblue", linewidth=2, label="Null N(0,1)")
        thresholds = [(0.01, 2.3263, "-"), (0.05, 1.6449, "--"), (0.10, 1.2816, ":")]
        for fpr, thr, ls in thresholds:
            tpr = float((mpz >= thr).mean())
            ax.axvline(thr, color="red", linestyle=ls, linewidth=1.2,
                       label=f"FPR {fpr*100:.0f}%: z={thr:.2f}, TPR={tpr*100:.1f}%")
    else:
        hz = np.nan_to_num(null_z)
        bins = np.linspace(min(mpz.min(), hz.min()) - 0.5, max(mpz.max(), hz.max()) + 0.5, 40)
        ax.hist(hz, bins=bins, alpha=0.55, label=f"Human (n={len(hz)})",
                color="steelblue", edgecolor="black", linewidth=0.4)
        ax.hist(mpz, bins=bins, alpha=0.55, label=f"{label} (n={len(mpz)})",
                color="darkorange", edgecolor="black", linewidth=0.4)
        for fpr, ls in [(0.01, "-"), (0.05, "--"), (0.10, ":")]:
            thr = float(np.quantile(hz, 1 - fpr))
            tpr = float((mpz >= thr).mean())
            ax.axvline(thr, color="red", linestyle=ls, linewidth=1.2,
                       label=f"FPR {fpr*100:.0f}%: z={thr:.2f}, TPR={tpr*100:.1f}%")
    ax.set_xlabel("z-score"); ax.set_ylabel("Count"); ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)


def _delta_metric(base_key, source="csv", *, aggregate=None, nested=False):
    """Build a summary entry while keeping labels/directions/scales centralized."""
    spec = metric_spec(base_key)
    key = f"{base_key}_{aggregate}" if aggregate else base_key
    label = f"{'  · ' if nested else ''}{spec.label}"
    if aggregate:
        label = f"{label} ({aggregate})"
    return key, label, spec.lower_is_better, source, spec.scale


# Metric, label, direction, source, and display scale.
_DELTA_METRICS = [
    _delta_metric("gen_ppl"),
    *[_delta_metric(spec.key) for spec in DETERMINISTIC_DIVERSITY_METRICS],
    _delta_metric("emb_sim"),
    _delta_metric("bert_F1", aggregate="median"),
    _delta_metric("llm_quality"),
    *[_delta_metric(spec.key, "npz", nested=True)
      for spec in LLM_QUALITY_METRICS[1:]],
    _delta_metric("llm_judge"),
    *[_delta_metric(spec.key, "npz", nested=True)
      for spec in LLM_JUDGE_METRICS[1:]],
]


def _metric_value(key, source, csv_row, npz, scale=1.0):
    if source == "csv":
        v = csv_row.get(key, float("nan"))
        v = float(v) if v is not None else float("nan")
    else:
        a = npz.get(key)
        if a is None:
            return float("nan")
        a = np.asarray(a, dtype=float)
        a = a[~np.isnan(a)]
        v = float(a.mean()) if a.size else float("nan")
    return v * scale


def _draw_quality(ax, dir_path: str, base_dir):
    """Draw quality deltas, or absolute means when no base exists."""
    # Use absolute means when the base has no quality results.
    if base_dir and not os.path.exists(os.path.join(base_dir, "eval_quality.csv")):
        base_dir = None
    cur_csv = _load_csv_row(dir_path, "eval_quality.csv")
    cur_npz = _load_per_sample_npz(dir_path)
    base_csv = _load_csv_row(base_dir, "eval_quality.csv") if base_dir else {}
    base_npz = _load_per_sample_npz(base_dir) if base_dir else {}

    if not base_dir:
        rows = [(label, v) for key, label, _l, src, scale in _DELTA_METRICS
                if not np.isnan(v := _metric_value(key, src, cur_csv, cur_npz, scale))]
        if not rows:
            ax.text(0.5, 0.5, "No quality data", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
            ax.set_title("Quality Metrics")
            return
        labels, vals = [r[0] for r in rows], [r[1] for r in rows]
        y = np.arange(len(labels))
        ax.barh(y, vals, color="steelblue", edgecolor="black", linewidth=0.4, alpha=0.85)
        for i, v in enumerate(vals):
            ax.text(v, i, " " + (f"{v:.2f}" if abs(v) >= 10 else f"{v:.3f}"),
                    va="center", ha="left", fontsize=8)
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8); ax.invert_yaxis()
        ax.set_title("Quality Metrics (no base for Δ)")
        ax.grid(axis="x", linestyle=":", alpha=0.4)
        return

    rows = []
    for key, label, lower_is_better, src, scale in _DELTA_METRICS:
        cur_v = _metric_value(key, src, cur_csv, cur_npz, scale)
        base_v = _metric_value(key, src, base_csv, base_npz, scale)
        if np.isnan(cur_v) or np.isnan(base_v):
            continue
        raw = cur_v - base_v
        rows.append((label, raw, -raw if lower_is_better else raw))
    if not rows:
        ax.text(0.5, 0.5, "No overlapping quality metrics with base",
                ha="center", va="center", transform=ax.transAxes, color="gray")
        ax.set_title("Quality Δ vs base")
        return
    labels = [r[0] for r in rows]
    oriented = np.array([r[2] for r in rows], dtype=float)
    raw = np.array([r[1] for r in rows], dtype=float)
    colors = ["#43A047" if o >= 0 else "#E53935" for o in oriented]
    y = np.arange(len(labels))
    ax.barh(y, oriented, color=colors, edgecolor="black", linewidth=0.4, alpha=0.9)
    ax.axvline(0, color="black", linewidth=0.8)
    span = float(np.max(np.abs(oriented))) or 1.0
    for i, (r, o) in enumerate(zip(raw, oriented)):
        ax.text(o + (span * 0.02 if o >= 0 else -span * 0.02), i,
                f"{r:+.2f}" if abs(r) >= 10 else f"{r:+.3f}",
                va="center", ha="left" if o >= 0 else "right", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("← degradation        improvement →   (oriented Δ)")
    ax.set_title("Quality Δ vs base")
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.set_xlim(-span * 1.25, span * 1.25)


_CRITERIA = {
    prefix: [key[len(prefix) + 1:]
             for key, _label, _flag, source, _scale in _DELTA_METRICS
             if source == "npz" and key.startswith(prefix + "_")]
    for prefix in ("llm_quality", "llm_judge")
}
_LEVELS = 6
_LEVEL_COLORS = [plt.get_cmap("RdYlGn")(i / (_LEVELS - 1)) for i in range(_LEVELS)]


def _bin_levels(arr):
    """Bin 0–1 scores into 6 rubric levels; return (proportions[6], mean) or None."""
    if arr is None:
        return None
    a = np.asarray(arr, dtype=float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return None
    levels = np.clip(np.round(a * (_LEVELS - 1)), 0, _LEVELS - 1).astype(int)
    counts = np.bincount(levels, minlength=_LEVELS).astype(float)
    return counts / counts.sum(), float(a.mean())


def _draw_per_criterion(ax, dir_path: str, base_dir, prefix: str, title: str):
    """Likert-style stacked-proportion bars for per-criterion score distributions."""
    criteria = _CRITERIA[prefix]
    cur_npz = _load_per_sample_npz(dir_path)
    base_npz = _load_per_sample_npz(base_dir) if base_dir else {}

    rows, y_positions, y_cursor, GAP = [], [], 0.0, 0.6
    for c in criteria:
        cur_data = _bin_levels(cur_npz.get(f"{prefix}_{c}"))
        base_data = _bin_levels(base_npz.get(f"{prefix}_{c}")) if base_dir else None
        pair = []
        if cur_data is not None:
            pair.append((f"{c.replace('_', ' ')} · {'paraphrased' if base_dir else 'watermarked'}", *cur_data))
        if base_data is not None:
            pair.append((f"{c.replace('_', ' ')} · base", *base_data))
        for row in pair:
            rows.append(row); y_positions.append(y_cursor); y_cursor += 1.0
        if pair:
            y_cursor += GAP

    if not rows:
        ax.text(0.5, 0.5, "No per-criterion data\n(judge not run for this dir)",
                ha="center", va="center", transform=ax.transAxes, color="gray", fontsize=10)
        ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
        return

    for y, (_label, props, mean) in zip(y_positions, rows):
        left = 0.0
        for L in range(_LEVELS):
            w = float(props[L])
            if w > 0:
                ax.barh(y, w, left=left, height=0.78, color=_LEVEL_COLORS[L],
                        edgecolor="white", linewidth=0.6)
            left += w
        ax.plot(mean, y, marker="D", color="black", markersize=6,
                markeredgecolor="white", markeredgewidth=0.7, zorder=5)
    ax.set_yticks(y_positions); ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.invert_yaxis(); ax.set_xlim(0, 1)
    ax.set_xlabel("Proportion of samples at each 0–5 score level")
    ax.set_title(title); ax.grid(axis="x", linestyle=":", alpha=0.4)
    handles = [mpatches.Patch(color=_LEVEL_COLORS[L], label=f"{L}") for L in range(_LEVELS)]
    handles.append(Line2D([0], [0], marker="D", color="black", markeredgecolor="white",
                          linestyle="", label="mean"))
    ax.legend(handles=handles, loc="lower right", fontsize=7, ncol=4,
              title="score", title_fontsize=7, framealpha=0.92)


# Anchor channels are 0-1 edit rates; coverage measures preservation.
_ANCHOR_CHANNELS = [
    ("anchor_reorder", "reorder"),
    ("anchor_merge", "merge"),
    ("anchor_split", "split"),
    ("anchor_reword", "reword"),
    ("anchor_reword_novel", "reword (novel bigrams)"),
    ("anchor_coverage", "coverage (preserved)"),
]


def _draw_structure(ax, dir_path: str):
    """Draw deterministic structural-edit channels."""
    csv_row = _load_csv_row(dir_path, "eval_quality.csv")
    npz = _load_per_sample_npz(dir_path)
    rows = []
    for key, label in _ANCHOR_CHANNELS:
        v = csv_row.get(key, float("nan"))
        if v is None or np.isnan(v):
            a = npz.get(f"{key}_per_sample")
            if a is not None:
                a = np.asarray(a, dtype=float); a = a[~np.isnan(a)]
                v = float(a.mean()) if a.size else float("nan")
        if v is not None and not np.isnan(v):
            rows.append((label, float(v)))
    if not rows:
        ax.text(0.5, 0.5, "No structure data", ha="center", va="center",
                transform=ax.transAxes, color="gray", fontsize=10)
        ax.set_title("Structural edits — deterministic (anchor_*)")
        return
    labels = [r[0] for r in rows]; vals = [r[1] for r in rows]
    y = np.arange(len(labels))
    ax.barh(y, vals, color="mediumpurple", edgecolor="black", linewidth=0.4, alpha=0.9)
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.3f}", va="center", ha="left", fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9); ax.invert_yaxis()
    ax.set_xlim(0, max(1.0, max(vals) * 1.15))
    ax.set_xlabel("fraction of source units (0–1, higher = more of that edit)")
    ax.set_title("Structural edits — deterministic (anchor_*)")
    ax.grid(axis="x", linestyle=":", alpha=0.4)


def run_detection_summary(dir_path: str, output_dir=None):
    """Generate the detection + quality summary figure for one directory."""
    if not os.path.isdir(dir_path):
        print(f"Error: directory not found: {dir_path}")
        return
    out_dir = output_dir or dir_path
    os.makedirs(out_dir, exist_ok=True)
    base_dir = watermarked_sibling_for(dir_path)

    # PMark aliases use its analytical null when human scores are absent.
    z_wm = _load_npy(dir_path, "z_scores.npy", "z_scores_text.npy")
    z_para = _load_npy(dir_path, "para_z_scores.npy", "z_scores_para.npy")
    z_hum = _load_npy(dir_path, "human_z_scores.npy")

    fig, axes = plt.subplots(4, 2, figsize=(14, 20))
    _draw_roc(axes[0, 0], dir_path, z_wm, z_para, z_hum)
    _draw_quality(axes[0, 1], dir_path, base_dir)
    _draw_zhist(axes[1, 0], z_wm, z_hum, "Z-score Distribution (before paraphrase)", "Watermarked")
    if z_para is not None:
        _draw_zhist(axes[1, 1], z_para, z_hum, "Z-score Distribution (after paraphrase)", "Paraphrased")
    else:
        axes[1, 1].text(0.5, 0.5, "No paraphrase data", ha="center", va="center",
                        transform=axes[1, 1].transAxes, color="gray", fontsize=11)
        axes[1, 1].set_title("Z-score Distribution (after paraphrase)")
    _draw_per_criterion(axes[2, 0], dir_path, base_dir, "llm_quality",
                        "LLM Quality — per-criterion distribution")
    _draw_per_criterion(axes[2, 1], dir_path, base_dir, "llm_judge",
                        "LLM Judge — per-criterion distribution")
    _draw_structure(axes[3, 0], dir_path)
    axes[3, 1].axis("off")

    fig.suptitle(_dir_label(dir_path), fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_figure(out_dir, "detection_summary")
    print(f"Saved: detection_summary.png/pdf → {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Per-directory detection + quality summary figure.")
    parser.add_argument("dir_path", help="Generated or paraphrased dataset directory")
    parser.add_argument("--output", default=None, help="Output dir (default: dir_path)")
    args = parser.parse_args()
    setup_plot_style()
    run_detection_summary(args.dir_path, args.output)


if __name__ == "__main__":
    main()
