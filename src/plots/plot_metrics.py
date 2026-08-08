"""Plot per-epoch training metrics from a ``src/train/train.py`` run.

Reads a metrics CSV (e.g. ``outputs/metrics/layer_projector_metrics.csv``) and renders
one PNG per group of scale-comparable metrics, train and validation series overlaid on
the same axes rather than split across separate figures:

- ``<run>_loss.png`` — train/val loss.
- ``<run>_similarity.png`` — same-layer and different-layer cosine similarity, train/val.
- ``<run>_similarity_gap.png`` — same-layer minus different-layer similarity, train/val.
- ``<run>_negative_margin_violation_rate.png`` — validation-only, no train counterpart.

``<run>`` is the CSV's filename with a trailing ``_metrics`` stripped, e.g.
``layer_projector_metrics.csv`` -> ``layer_projector``.

Usage
-----
    python -m src.plots.plot_metrics
    python -m src.plots.plot_metrics --metrics-csv outputs/metrics/other_run_metrics.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

# Fixed categorical slots (blue, orange, aqua), per the project's validated palette
# (skill: dataviz / references/palette.md) -- assigned by role, not cycled.
_BLUE = "#2a78d6"
_ORANGE = "#eb6834"
_AQUA = "#1baf7a"

_SURFACE = "#fcfcfb"
_GRIDLINE = "#e1e0d9"
_AXIS = "#c3c2b7"
_TICK_LABEL = "#898781"
_INK_PRIMARY = "#0b0b0b"
_INK_SECONDARY = "#52514e"

_TRAIN_STYLE = {"linestyle": "--"}
_VAL_STYLE = {"linestyle": "-"}


def _new_axes() -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    fig.patch.set_facecolor(_SURFACE)
    ax.set_facecolor(_SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(_AXIS)
    ax.spines["bottom"].set_color(_AXIS)
    ax.grid(axis="y", color=_GRIDLINE, linewidth=1.0, zorder=0)
    ax.tick_params(colors=_TICK_LABEL, labelsize=9)
    ax.set_xlabel("epoch", color=_INK_SECONDARY, fontsize=10)
    return fig, ax


def _save(fig: plt.Figure, ax: plt.Axes, title: str, ylabel: str, path: Path) -> None:
    ax.set_title(title, color=_INK_PRIMARY, fontsize=12, loc="left", pad=10)
    ax.set_ylabel(ylabel, color=_INK_SECONDARY, fontsize=10)
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        legend = ax.legend(frameon=False, fontsize=9, labelcolor=_INK_SECONDARY)
        legend.set_zorder(3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_train_val_pair(
    df: pd.DataFrame, epoch: Sequence[int], train_col: str, val_col: str,
    color: str, title: str, ylabel: str, path: Path,
) -> None:
    """One metric with a train/ and val/ column, both lines on the same axes."""
    fig, ax = _new_axes()
    ax.plot(epoch, df[train_col], color=color, linewidth=2, label="train", zorder=2, **_TRAIN_STYLE)
    ax.plot(epoch, df[val_col], color=color, linewidth=2, label="val", zorder=2, **_VAL_STYLE)
    _save(fig, ax, title, ylabel, path)


def plot_similarity(df: pd.DataFrame, epoch: Sequence[int], path: Path) -> None:
    """Same-layer and different-layer similarity, train/val, on one axes.

    Color encodes which metric (same-layer = blue, diff-layer = orange); linestyle
    encodes the split (train = dashed, val = solid) -- so identity never rests on
    color alone.
    """
    fig, ax = _new_axes()
    ax.axhline(0.0, color=_AXIS, linewidth=1.0, zorder=1)
    ax.plot(epoch, df["train_same_layer_similarity"], color=_BLUE, linewidth=2,
             label="same-layer (train)", zorder=2, **_TRAIN_STYLE)
    ax.plot(epoch, df["val_same_layer_similarity"], color=_BLUE, linewidth=2,
             label="same-layer (val)", zorder=2, **_VAL_STYLE)
    ax.plot(epoch, df["train_diff_layer_similarity"], color=_ORANGE, linewidth=2,
             label="diff-layer (train)", zorder=2, **_TRAIN_STYLE)
    ax.plot(epoch, df["val_diff_layer_similarity"], color=_ORANGE, linewidth=2,
             label="diff-layer (val)", zorder=2, **_VAL_STYLE)
    _save(fig, ax, "Same-layer vs. different-layer cosine similarity", "cosine similarity", path)


def plot_violation_rate(df: pd.DataFrame, epoch: Sequence[int], path: Path) -> None:
    """Validation-only metric -- single series, no train counterpart."""
    fig, ax = _new_axes()
    ax.plot(epoch, df["negative_margin_violation_rate"], color=_AQUA, linewidth=2, zorder=2)
    _save(
        fig, ax,
        "Negative-margin violation rate (validation)",
        "fraction of diff-layer pairs above margin",
        path,
    )


def plot_all(metrics_csv: Path, output_dir: Path) -> list[Path]:
    """Render every metric-group figure for one run's metrics CSV.

    Returns the list of PNG paths written.
    """
    df = pd.read_csv(metrics_csv)
    epoch = df["epoch"]
    run_name = metrics_csv.stem.removesuffix("_metrics")

    written = []

    loss_path = output_dir / f"{run_name}_loss.png"
    plot_train_val_pair(
        df, epoch, "train_loss", "val_loss", _BLUE,
        "Training vs. validation loss", "loss", loss_path,
    )
    written.append(loss_path)

    similarity_path = output_dir / f"{run_name}_similarity.png"
    plot_similarity(df, epoch, similarity_path)
    written.append(similarity_path)

    gap_path = output_dir / f"{run_name}_similarity_gap.png"
    plot_train_val_pair(
        df, epoch, "train_similarity_gap", "val_similarity_gap", _ORANGE,
        "Same-layer minus different-layer similarity gap", "similarity gap", gap_path,
    )
    written.append(gap_path)

    violation_path = output_dir / f"{run_name}_negative_margin_violation_rate.png"
    plot_violation_rate(df, epoch, violation_path)
    written.append(violation_path)

    return written


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics-csv", type=Path,
        default=Path("outputs/metrics/layer_projector_metrics.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/plots"))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    written = plot_all(args.metrics_csv, args.output_dir)
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
