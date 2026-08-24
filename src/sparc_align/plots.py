"""Figures for the SPARC-align convergence and batch-size studies.

    python -m src.sparc_align.plots --study convergence
    python -m src.sparc_align.plots --study batch

Writes PNGs under ``outputs/sparc_align/<run>/figures/``.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Colour-blind-safe qualitative palette (Okabe-Ito), so lines stay distinguishable
# in greyscale print and for the ~8% of readers with red-green deficiency.
PALETTE = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7",
    "#E69F00", "#56B4E9", "#F0E442", "#000000",
]
GRID = dict(alpha=0.25, linewidth=0.6)


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _style(ax, xlabel, ylabel, title=None):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=10)
    ax.grid(True, **GRID)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_per_section(rows, out_dir: Path, metric="ari", label="refined ARI"):
    """One panel per section: the metric against stage-1 epochs."""
    sections = sorted({r["section_id"] for r in rows})
    fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharex=True, sharey=True)
    for ax, section in zip(axes.ravel(), sections):
        sr = sorted((r for r in rows if r["section_id"] == section), key=lambda r: int(r["epoch"]))
        epochs = [int(r["epoch"]) for r in sr]
        for i, (key, name) in enumerate(
            [(metric, "two-stage (SPARC + attention)"),
             (f"stage1_support_{metric}", "stage 1 only, support"),
             (f"stage1_sum_{metric}", "stage 1 only, sum")]
        ):
            if key in sr[0]:
                ax.plot(epochs, [float(r[key]) for r in sr], marker="o", ms=3.5,
                        color=PALETTE[i], label=name, linewidth=1.6)
        best = max(sr, key=lambda r: float(r[metric]))
        ax.axvline(int(best["epoch"]), color=PALETTE[0], linestyle=":", linewidth=1, alpha=0.7)
        _style(ax, "stage-1 epochs", label, f"{section}  (best @ {best['epoch']} ep)")
        ax.set_xscale("log")
        ax.set_xticks(epochs)
        ax.set_xticklabels(epochs)
    for ax in axes.ravel()[len(sections):]:
        ax.set_visible(False)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"SPARC-align convergence per DLPFC section ({label})", fontsize=13)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    path = out_dir / f"convergence_per_section_{metric}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_mean(rows, out_dir: Path, metric="ari", label="refined ARI"):
    """Mean across sections, with an IQR band showing between-section spread."""
    by_epoch = defaultdict(list)
    for r in rows:
        by_epoch[int(r["epoch"])].append(float(r[metric]))
    epochs = sorted(by_epoch)
    means = np.array([np.mean(by_epoch[e]) for e in epochs])
    medians = np.array([np.median(by_epoch[e]) for e in epochs])
    q1 = np.array([np.percentile(by_epoch[e], 25) for e in epochs])
    q3 = np.array([np.percentile(by_epoch[e], 75) for e in epochs])

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.fill_between(epochs, q1, q3, color=PALETTE[0], alpha=0.15, label="IQR across 12 sections")
    ax.plot(epochs, means, marker="o", color=PALETTE[0], linewidth=2, label="mean")
    ax.plot(epochs, medians, marker="s", ms=4, color=PALETTE[1], linewidth=1.4,
            linestyle="--", label="median")

    peak = int(np.argmax(means))
    ax.annotate(
        f"best mean {means[peak]:.4f}\n@ {epochs[peak]} epochs",
        xy=(epochs[peak], means[peak]), xytext=(12, -28), textcoords="offset points",
        fontsize=9, arrowprops=dict(arrowstyle="->", color="0.4", linewidth=0.8),
    )
    _style(ax, "stage-1 epochs", f"mean {label}",
           f"SPARC-align mean {label} across 12 DLPFC sections\n(stage 2 fixed at 20 epochs)")
    ax.set_xscale("log")
    ax.set_xticks(epochs)
    ax.set_xticklabels(epochs)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    path = out_dir / f"convergence_mean_{metric}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_batch(rows, out_dir: Path, metric="ari", label="refined ARI"):
    """Batch size effect: per-section lines plus the mean."""
    sizes = sorted({int(r["batch_size"]) for r in rows})
    sections = sorted({r["section_id"] for r in rows})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    for i, section in enumerate(sections):
        sr = sorted((r for r in rows if r["section_id"] == section), key=lambda r: int(r["batch_size"]))
        ax1.plot([int(r["batch_size"]) for r in sr], [float(r[metric]) for r in sr],
                 marker="o", ms=3, linewidth=1.1, alpha=0.75,
                 color=PALETTE[i % len(PALETTE)], label=section)
    _style(ax1, "batch size", label, "per section")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(sizes); ax1.set_xticklabels(sizes)
    ax1.legend(frameon=False, fontsize=7, ncol=2)

    by_size = defaultdict(list)
    for r in rows:
        by_size[int(r["batch_size"])].append(float(r[metric]))
    means = [np.mean(by_size[s]) for s in sizes]
    q1 = [np.percentile(by_size[s], 25) for s in sizes]
    q3 = [np.percentile(by_size[s], 75) for s in sizes]
    ax2.fill_between(sizes, q1, q3, color=PALETTE[0], alpha=0.15, label="IQR")
    ax2.plot(sizes, means, marker="o", color=PALETTE[0], linewidth=2, label="mean")
    best = int(np.argmax(means))
    ax2.annotate(f"best {means[best]:.4f} @ bs={sizes[best]}",
                 xy=(sizes[best], means[best]), xytext=(10, -26), textcoords="offset points",
                 fontsize=9, arrowprops=dict(arrowstyle="->", color="0.4", linewidth=0.8))
    _style(ax2, "batch size", f"mean {label}", "mean across 12 sections")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(sizes); ax2.set_xticklabels(sizes)
    ax2.legend(frameon=False, fontsize=9)

    fig.suptitle(f"SPARC-align batch-size effect ({label})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = out_dir / f"batch_size_{metric}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_spatial_init(rows, out_dir: Path, metric="ari", label="refined ARI"):
    """Paired per-section comparison of spatial-smoothing placement."""
    from collections import defaultdict as _dd

    by = _dd(dict)
    for r in rows:
        by[r["section_id"]][r["arm"]] = float(r[metric])
    sections = sorted(by)
    arms = ["baseline", "before", "after"]
    x = np.arange(len(sections))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5),
                                   gridspec_kw={"width_ratios": [2.4, 1]})
    width = 0.27
    for i, arm in enumerate(arms):
        ax1.bar(x + (i - 1) * width, [by[s][arm] for s in sections], width,
                color=PALETTE[i], label=arm, alpha=0.9)
    _style(ax1, "section", label, "per section")
    ax1.set_xticks(x)
    ax1.set_xticklabels(sections, rotation=45, ha="right", fontsize=8)
    ax1.legend(frameon=False, fontsize=9)

    # Paired deltas make the heterogeneity legible in a way means hide.
    deltas = [by[s]["before"] - by[s]["baseline"] for s in sections]
    colors = [PALETTE[2] if d > 0 else PALETTE[1] for d in deltas]
    ax2.barh(x, deltas, color=colors, alpha=0.9)
    ax2.axvline(0, color="0.3", linewidth=0.9)
    ax2.axvline(float(np.mean(deltas)), color=PALETTE[0], linestyle="--", linewidth=1.2,
                label=f"mean {np.mean(deltas):+.4f}")
    _style(ax2, f"{label} change", "", "before − baseline")
    ax2.set_yticks(x)
    ax2.set_yticklabels(sections, fontsize=8)
    ax2.legend(frameon=False, fontsize=9)

    fig.suptitle("SPARC-align: spatial k-NN smoothing before vs after alignment", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = out_dir / f"spatial_init_{metric}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=["convergence", "batch", "spatial_init"], default="convergence")
    parser.add_argument("--run", default=None)
    parser.add_argument("--metric", default="ari", choices=["ari", "nmi"])
    args = parser.parse_args()

    default_run = {"convergence": "convergence_all12", "batch": "batch_size_all12",
                   "spatial_init": "spatial_init_all12"}[args.study]
    run = args.run or default_run
    base = Path("outputs/sparc_align") / run
    out_dir = base / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    label = "refined ARI" if args.metric == "ari" else "refined NMI"

    if args.study == "convergence":
        rows = _read(base / "convergence_metrics.csv")
        print(plot_per_section(rows, out_dir, args.metric, label))
        print(plot_mean(rows, out_dir, args.metric, label))
    elif args.study == "batch":
        rows = _read(base / "batch_metrics.csv")
        print(plot_batch(rows, out_dir, args.metric, label))
    else:
        rows = _read(base / "spatial_init_metrics.csv")
        print(plot_spatial_init(rows, out_dir, args.metric, label))


if __name__ == "__main__":
    main()
