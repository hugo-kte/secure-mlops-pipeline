"""Report figures (matplotlib, Agg), same visual spec as the rest of the portfolio."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]

plt.rcParams.update(
    {
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelcolor": INK2,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "legend.frameon": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    }
)


def _style(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _title(ax: plt.Axes, title: str, subtitle: str, y_sub: float = 1.035) -> None:
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=INK, pad=28)
    ax.text(0, y_sub, subtitle, transform=ax.transAxes, fontsize=9.5, color=INK2, va="bottom")


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return path


def duplicates_effect(metrics: dict[str, Any], path: Path) -> Path:
    """Error rate of the same model under three evaluation protocols."""
    effect = metrics["duplicates_effect"]
    keys = ["naive", "grouped", "deduplicated"]
    names = [
        "Row-level split\n(duplicates straddle\ntrain and test)",
        "Grouped split\n(identical vectors\nkept together)",
        "De-duplicated\n(one row per\ndistinct vector)",
    ]
    err = np.array([1 - effect[k]["accuracy"]["mean"] for k in keys])
    lo = np.array([1 - effect[k]["accuracy"]["ci95"][1] for k in keys])
    hi = np.array([1 - effect[k]["accuracy"]["ci95"][0] for k in keys])

    fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=100)
    _style(ax)
    ax.grid(axis="x", visible=False)
    colors = [SERIES[1], SERIES[0], SERIES[0]]
    xs = np.arange(len(keys))
    ax.bar(xs, err, 0.56, color=colors, linewidth=0)
    ax.errorbar(xs, err, yerr=[err - lo, hi - err], fmt="none", ecolor=INK2, elinewidth=1.2,
                capsize=4)  # fmt: skip
    for x, e, h in zip(xs, err, hi, strict=True):
        ax.text(x, h + 0.0015, f"{100 * e:.1f}%", ha="center", va="bottom", fontsize=10,
                color=INK, fontweight="bold")  # fmt: skip
    ax.set_xticks(xs, names)
    ax.set_ylim(0, max(hi) * 1.3)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("Error rate (1 − accuracy)")
    seen = effect["naive"]["test_rows_seen_in_train"]
    hidden = 1 - err[0] / err[1]  # share of the grouped-split errors a row-level split hides
    _title(
        ax,
        f"A row-level split hides {100 * hidden:.0f}% of the errors",
        f"Same model, {metrics['protocol']['cv_folds']}-fold CV, 95% CI over folds; "
        f"{100 * seen:.0f}% of row-level test rows have an identical twin in training",
    )
    return _save(fig, path)


def roc_pr(metrics: dict[str, Any], path: Path) -> Path:
    """ROC (zoomed on low false-positive rates) and precision-recall on the hold-out test set."""
    curves = metrics["curves"]
    sel_name = metrics["model"]["description"].split(" (")[0].capitalize()
    series = [("selected", sel_name, SERIES[0])]
    if "reference" in curves:
        ref_name = metrics["reference_model"]["description"].split(" (")[0].capitalize()
        series.append(("reference", ref_name, SERIES[1]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 4.1), dpi=100)
    for ax in (ax1, ax2):
        _style(ax)
    for key, name, color in series:
        roc = curves[key]["roc"]
        ax1.plot(roc["fpr"], roc["tpr"], color=color, lw=1.8, label=name)
        pr = curves[key]["pr"]
        ax2.plot(pr["recall"], pr["precision"], color=color, lw=1.8, label=name)

    # Operating point of the shipped model at its decision threshold.
    c, thr = metrics["test"]["confusion"], metrics["protocol"]["threshold"]
    tpr, fpr = c["tp"] / (c["tp"] + c["fn"]), c["fp"] / (c["fp"] + c["tn"])
    precision = c["tp"] / (c["tp"] + c["fp"])
    for ax, (x, y) in ((ax1, (fpr, tpr)), (ax2, (tpr, precision))):
        ax.scatter([x], [y], s=40, color=INK, zorder=5, edgecolors=SURFACE, linewidths=1.5)
    ax1.text(
        0.045, 0.70,
        f"\u25cf operating point (threshold {thr:g}):\n{100 * tpr:.1f}% caught, "
        f"{100 * fpr:.1f}% of legitimate flagged",
        fontsize=8.5, color=INK2,
    )  # fmt: skip

    ax1.set_xlim(-0.002, 0.25)
    ax1.set_ylim(0.5, 1.01)
    ax1.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax1.set_xlabel("Legitimate pages flagged (FPR, zoom 0-25%)")
    ax1.set_ylabel("Phishing caught (TPR)")
    ax2.set_xlim(-0.01, 1.01)
    ax2.set_ylim(0.4, 1.01)
    ax2.set_xlabel("Recall (phishing caught)")
    ax2.set_ylabel("Precision")
    ax1.legend(loc="lower right", fontsize=9, labelcolor=INK2)
    t = metrics["test"]
    _title(
        ax1,
        "Held-out performance on unseen feature vectors",
        f"{metrics['data']['test_rows']:,} de-duplicated test vectors; selected model ROC-AUC "
        f"{t['metrics']['roc_auc']:.4f} [{t['ci95']['roc_auc'][0]:.4f}, "
        f"{t['ci95']['roc_auc'][1]:.4f}], F1 {t['metrics']['f1']:.3f}",
    )
    fig.subplots_adjust(wspace=0.3)
    return _save(fig, path)


def render_all(metrics: dict[str, Any], out_dir: Path) -> list[Path]:
    return [
        duplicates_effect(metrics, out_dir / "duplicates_effect.png"),
        roc_pr(metrics, out_dir / "roc_pr.png"),
    ]
