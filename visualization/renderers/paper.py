"""Matplotlib counterparts of the paper's PGFPlots figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCHEMES = (
    ("semstamp", "SemStamp"),
    ("ksemstamp", "k-SemStamp"),
    ("pmark-online", "PMark (online)"),
    ("samark", "SAMark"),
    ("swordstamp", "SwordStamp"),
    ("kswordstamp", "k-SwordStamp"),
)
COLORS = plt.get_cmap("tab10").colors


def _numeric(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"required extracted table is missing: {path}")
    values = np.genfromtxt(path, names=True, dtype=float, encoding="utf-8")
    return np.atleast_1d(values)


def _symbolic(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"required extracted table is missing: {path}")
    lines = [line.split() for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        return []
    return [dict(zip(lines[0], row)) for row in lines[1:]]


def _save(fig, output: Path, stem: str) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    paths = [output / f"{stem}.png", output / f"{stem}.pdf"]
    fig.savefig(paths[0], dpi=220, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def _direction_cue(
    ax,
    label: str,
    *,
    text_position: tuple[float, float],
    arrow_start: tuple[float, float],
    arrow_end: tuple[float, float],
    horizontal_alignment: str,
):
    """Draw one paper direction label and arrow in axes coordinates."""
    text_artist = ax.text(
        *text_position,
        label,
        transform=ax.transAxes,
        ha=horizontal_alignment,
        va="top",
        color="0.35",
        fontsize=8,
    )
    arrow_artist = ax.annotate(
        "",
        xy=arrow_end,
        xytext=arrow_start,
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops={
            "arrowstyle": "->",
            "color": "0.38",
            "linewidth": 0.8,
            "shrinkA": 0,
            "shrinkB": 0,
        },
    )
    return text_artist, arrow_artist


def _errorbar(ax, table, x: str, y: str, label: str, color, **kwargs) -> None:
    yerr = None
    if {"em", "ep"}.issubset(table.dtype.names or ()):
        yerr = np.vstack((table["em"], table["ep"]))
    ax.errorbar(
        table[x], table[y], yerr=yerr, label=label, color=color,
        marker="o", markersize=3, linewidth=1.4, capsize=2, **kwargs,
    )


def render_teaser(data: Path, output: Path) -> list[Path]:
    written: list[Path] = []
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    found = 0
    for index, (key, label) in enumerate(SCHEMES):
        path = data / "teaser" / f"{key}.dat"
        if not path.is_file():
            raise FileNotFoundError(f"required teaser table is missing: {path}")
        _errorbar(ax, _numeric(path), "bar", "y", label, COLORS[index])
        found += 1
    if not found:
        raise FileNotFoundError(f"no teaser tables under {data / 'teaser'}")
    ax.set(xlabel="Content-preservation requirement", ylabel="Strongest no-box ASR",
           ylim=(0, 1))
    ax.grid(alpha=.25)
    _direction_cue(
        ax,
        "more robust",
        text_position=(.98, .97),
        arrow_start=(.88, .86),
        arrow_end=(.88, .70),
        horizontal_alignment="right",
    )
    ax.legend(ncol=2, fontsize=8)
    written.extend(_save(fig, output, "teaser"))

    table = _numeric(data / "fidelity_robustness" / "points.dat")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for index, (_, label) in enumerate(SCHEMES[: len(table)]):
        row = table[index]
        ax.errorbar(
            row["fidelity"], row["asr"],
            xerr=[[row["fem"]], [row["fep"]]],
            yerr=[[row["aem"]], [row["aep"]]],
            fmt="o", color=COLORS[index], capsize=3, label=label,
        )
    ax.set(xlabel="Clean fidelity (%)", ylabel="Strongest no-box ASR (%)")
    ax.grid(alpha=.25)
    _direction_cue(
        ax,
        "better",
        text_position=(.03, .97),
        arrow_start=(.12, .84),
        arrow_end=(.22, .72),
        horizontal_alignment="left",
    )
    ax.legend(ncol=2, fontsize=8)
    written.extend(_save(fig, output, "fidelity-robustness"))
    return written


def render_channel_response(data: Path, output: Path) -> list[Path]:
    panels = (("reword", "Rewording"), ("reorder", "Reordering"),
              ("reseg", "Resegmentation"))
    ladders = (
        ("k-SemStamp ladder", ("kbase", "kbestn", "kfixed", "kdiverse", "kspan")),
        ("SemStamp ladder", ("base", "bestn", "fixed", "diverse", "span")),
    )
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.3), sharey=True)
    for row, (family_label, rungs) in enumerate(ladders):
        for col, (panel, title) in enumerate(panels):
            ax = axes[row, col]
            for index, rung in enumerate(rungs):
                table = _numeric(data / "channel_response" / f"{panel}-{rung}.dat")
                _errorbar(ax, table, "x", "y", rung, COLORS[index])
            ax.set_title(f"{family_label}: {title}", fontsize=9)
            ax.set_xlabel("Measured channel strength")
            ax.grid(alpha=.25)
            if col == 0:
                ax.set_ylabel("Normalized z retention (%)")
            if row == 0 and col == 2:
                ax.legend(fontsize=7)
    return _save(fig, output, "channel-response")


def render_cross_scheme(data: Path, output: Path) -> list[Path]:
    series = (("sentence", "Sentence controls"), ("dipper", "Dipper"),
              ("adaptive", "EDA-S"), ("adaptive-base", "EDA-P"))
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.3), sharex=True, sharey=True)
    for ax, (scheme, label) in zip(axes.flat, SCHEMES):
        found = False
        for index, (key, series_label) in enumerate(series):
            path = data / "cross_scheme" / f"{scheme}-{key}.dat"
            if not path.is_file():
                continue
            _errorbar(ax, _numeric(path), "bar", "y", series_label, COLORS[index])
            found = True
        if not found:
            raise FileNotFoundError(f"no cross-scheme tables for {scheme}")
        ax.set_title(label, fontsize=9)
        ax.grid(alpha=.25)
        ax.set_ylim(0, 1)
    for ax in axes[-1]:
        ax.set_xlabel("Content-preservation requirement")
    for ax in axes[:, 0]:
        ax.set_ylabel("ASR")
    axes[0, 0].legend(fontsize=7)
    return _save(fig, output, "cross-scheme")


def render_whitebox(data: Path, output: Path) -> list[Path]:
    schemes = (("ksemstamp", "k-SemStamp"), ("kswordstamp", "k-SwordStamp"))
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.7))
    for index, (key, label) in enumerate(schemes):
        for metric, style in (("evasion", "-"), ("asr", "--")):
            table = _numeric(data / "whitebox" / f"{metric}-{key}.dat")
            _errorbar(axes[0], table, "K", "y", f"{label}: {metric}",
                      COLORS[index], linestyle=style)
        table = _numeric(
            data / "whitebox" / f"quality-pass-given-evasion-{key}.dat"
        )
        _errorbar(axes[1], table, "K", "y", label, COLORS[index])
    axes[0].set_title("Evasion and quality-gated ASR")
    axes[1].set_title("Quality pass among evasions")
    for ax in axes:
        ax.set(xlabel="EDA-D search budget K", ylim=(0, 1))
        ax.grid(alpha=.25)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Rate")
    return _save(fig, output, "kmeans-whitebox")


def render_transfer(data: Path, output: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    for index, (key, label) in enumerate((("semstamp", "SemStamp"),
                                         ("pmark", "PMark"))):
        table = _numeric(data / "transfer" / f"{key}-band.dat")
        ax.plot(table["x"], table["y"], marker="o", markersize=3,
                color=COLORS[index], label=label)
        ax.fill_between(table["x"], table["lo"], table["hi"],
                        color=COLORS[index], alpha=.18)
    ax.set(xlabel="Surrogate cosine displacement",
           ylabel="Provider cosine displacement")
    ax.grid(alpha=.25)
    ax.legend()
    return _save(fig, output, "encoder-transfer")


def render_attack_channels(data: Path, output: Path) -> list[Path]:
    rows = _symbolic(data / "attack_channels" / "ksemstamp.dat")
    if not rows:
        raise ValueError("ksemstamp attack-channel table has no rows")
    panels = (("evading_reorder", "Evading reorder"),
              ("evading_reseg", "Evading resegmentation"),
              ("evading_quality_pass", "Quality pass among evasions"))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.7), sharex=True)
    series_order = list(dict.fromkeys(row["series"] for row in rows))
    for index, series in enumerate(series_order):
        selected = [row for row in rows if row["series"] == series]
        x = np.asarray([float(row["evading_reword"]) for row in selected])
        for ax, (column, _) in zip(axes, panels):
            y = np.asarray([float(row[column]) for row in selected])
            ax.plot(x, y, marker="o", markersize=3, linewidth=1.2,
                    color=COLORS[index % len(COLORS)], label=series)
    for ax, (_, title) in zip(axes, panels):
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Evading rewording")
        ax.grid(alpha=.25)
    axes[0].set_ylabel("Channel response")
    axes[-1].legend(fontsize=6, bbox_to_anchor=(1.02, 1), loc="upper left")
    return _save(fig, output, "ksemstamp-attack-channels")


RENDERERS = (
    render_teaser,
    render_channel_response,
    render_cross_scheme,
    render_whitebox,
    render_transfer,
    render_attack_channels,
)


def render_all(source: str | Path, output: str | Path) -> list[Path]:
    """Render every paper figure from ``source/tables/pgfplots``."""
    source = Path(source).resolve() / "tables" / "pgfplots"
    output = Path(output).resolve() / "figures"
    written: list[Path] = []
    with plt.rc_context({
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "figure.dpi": 120,
        "savefig.facecolor": "white",
    }):
        for renderer in RENDERERS:
            written.extend(renderer(source, output))
    return written
