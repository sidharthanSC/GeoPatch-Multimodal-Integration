"""Select the k-NN hyperparameter ``k`` via a dataset-size-informed elbow sweep.

Sweeps leave-one-out k-NN classification accuracy against ground-truth cortical layer
over a range of ``k`` values, on the *raw* (pre-projection) BYOL embeddings -- raw
embeddings are identical across every ``src/train/train.py`` run (verified: same
frozen encoder output, re-exported unchanged each run), so one sweep on that shared
reference space gives a ``k`` that applies consistently to every run's
``src/analysis/knn_projection_analysis.py`` comparison, rather than a different ``k``
per projection.

The swept range is anchored to the dataset size: with ``n`` spots, ``sqrt(n)`` is the
classic k-NN rule-of-thumb upper reference point (here ``n=47329`` -> ``sqrt(n)~=218``),
so the sweep runs from ``k=1`` out past that point (default up to 251) to see the full
error curve -- accuracy improving as k grows past the noise of very small neighborhoods,
then degrading again once neighborhoods grow large enough to blur across cortical
layers. Spacing is finer at small k (where the curve moves fastest) and coarser at
large k.

Elbow detection: normalize the ``(k, error)`` curve to ``[0, 1] x [0, 1]``, draw the
chord connecting its first and last point, and pick the point of maximum perpendicular
distance from that chord -- the point where the curve bends most sharply away from a
straight-line trade-off between ``k`` and error (the standard "distance from chord" /
kneedle rule).

Each projection run's own accuracy-vs-k curve is also computed and plotted alongside,
as a sanity check that the raw-curve-selected ``k`` isn't a poor choice for any of them.

Usage
-----
    python -m src.analysis.knn_k_selection

Outputs: ``outputs/analysis/knn_k_elbow.json``, ``outputs/analysis/knn_k_elbow.csv``,
``outputs/plots/knn_k_elbow.png``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.neighbors import NearestNeighbors

# Fixed categorical slots (blue, orange, aqua, yellow, ...), per the project's
# validated palette (skill: dataviz / references/palette.md) -- assigned by role.
_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#e34948"]
_SURFACE = "#fcfcfb"
_GRIDLINE = "#e1e0d9"
_AXIS = "#c3c2b7"
_TICK_LABEL = "#898781"
_INK_PRIMARY = "#0b0b0b"
_INK_SECONDARY = "#52514e"


def default_k_values(n_spots: int) -> List[int]:
    """Odd k values from 1 up to just past sqrt(n_spots), finer-grained at small k."""
    reference = round(n_spots ** 0.5)
    dense = list(range(1, 52, 2))
    sparse = [k for k in (61, 71, 91, 121, 151, 181, reference, reference + 34) if k > dense[-1]]
    return sorted(set(dense + sparse))


def sweep_k_accuracy(
    embeddings: np.ndarray, labels: np.ndarray, k_values: Sequence[int]
) -> Dict[int, float]:
    """Leave-one-out k-NN accuracy for every ``k`` in ``k_values``.

    Fits ``NearestNeighbors`` once with ``n_neighbors = max(k_values) + 1`` and reuses
    that single neighbor query for every ``k`` (slicing to the ``k`` closest, excluding
    self) rather than re-fitting per ``k`` -- the dominant cost either way is one
    nearest-neighbor search over all spots.
    """
    _, label_ints = np.unique(labels, return_inverse=True)
    max_k = max(k_values)
    n = embeddings.shape[0]

    finder = NearestNeighbors(n_neighbors=max_k + 1).fit(embeddings)
    _, neighbor_idx = finder.kneighbors(embeddings)

    trimmed = np.empty((n, max_k), dtype=neighbor_idx.dtype)
    for i in range(n):
        row = neighbor_idx[i]
        trimmed[i] = row[row != i][:max_k]

    neighbor_label_ints = label_ints[trimmed]  # (n, max_k)

    accuracies: Dict[int, float] = {}
    for k in k_values:
        mode_result = scipy_stats.mode(neighbor_label_ints[:, :k], axis=1, keepdims=False)
        accuracies[k] = float(np.mean(mode_result.mode == label_ints))
    return accuracies


def find_elbow(k_values: Sequence[int], errors: Sequence[float]) -> int:
    """Return the ``k`` at the point of maximum distance from the curve's endpoint chord."""
    k_arr = np.asarray(k_values, dtype=float)
    err_arr = np.asarray(errors, dtype=float)

    k_norm = (k_arr - k_arr.min()) / (k_arr.max() - k_arr.min())
    err_range = err_arr.max() - err_arr.min()
    err_norm = (err_arr - err_arr.min()) / err_range if err_range > 0 else np.zeros_like(err_arr)

    p1 = np.array([k_norm[0], err_norm[0]])
    p2 = np.array([k_norm[-1], err_norm[-1]])
    line_vec = p2 - p1
    line_vec = line_vec / np.linalg.norm(line_vec)

    points = np.stack([k_norm, err_norm], axis=1) - p1
    proj = (points @ line_vec)[:, None] * line_vec
    perpendicular = points - proj
    distances = np.linalg.norm(perpendicular, axis=1)

    return int(k_arr[int(np.argmax(distances))])


def run_k_selection(
    predictions_dir: Path, analysis_output_dir: Path, plot_output_dir: Path
) -> int:
    npz_paths = sorted(predictions_dir.glob("*_embeddings.npz"))
    if not npz_paths:
        raise FileNotFoundError(f"No *_embeddings.npz files found in {predictions_dir}")

    datasets = {p.stem.removesuffix("_embeddings"): np.load(p, allow_pickle=True) for p in npz_paths}
    run_names = list(datasets)

    reference = datasets[run_names[0]]
    labels = reference["cortical_labels"].astype(str)
    n_spots = len(labels)
    raw_embeddings = reference["raw_embeddings"]

    for name, d in datasets.items():
        if not np.array_equal(d["raw_embeddings"], raw_embeddings):
            raise ValueError(
                f"Run '{name}' has different raw_embeddings than '{run_names[0]}' -- "
                f"they were expected to share one frozen encoder output."
            )

    k_values = default_k_values(n_spots)
    print(f"n_spots={n_spots}, sqrt(n_spots)={n_spots ** 0.5:.1f}, sweeping k in {k_values}")

    raw_accuracy = sweep_k_accuracy(raw_embeddings, labels, k_values)
    raw_error = [1.0 - raw_accuracy[k] for k in k_values]
    elbow_k = find_elbow(k_values, raw_error)

    columns = {"k": list(k_values), "raw_accuracy": [raw_accuracy[k] for k in k_values]}
    per_run_accuracy: Dict[str, Dict[int, float]] = {}
    for name, d in datasets.items():
        acc = sweep_k_accuracy(d["projected_embeddings"], labels, k_values)
        per_run_accuracy[name] = acc
        columns[f"{name}_projected_accuracy"] = [acc[k] for k in k_values]

    df = pd.DataFrame(columns)
    analysis_output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(analysis_output_dir / "knn_k_elbow.csv", index=False)

    summary = {
        "n_spots": n_spots,
        "sqrt_n_spots": n_spots ** 0.5,
        "k_values": list(k_values),
        "method": "max distance from the (k, error) curve's endpoint chord, on the raw-embedding curve",
        "selected_k": elbow_k,
        "raw_accuracy_at_selected_k": raw_accuracy[elbow_k],
        "runs_swept": {
            name: {"projected_accuracy_at_selected_k": acc[elbow_k],
                   "best_k_for_this_run": max(acc, key=acc.get),
                   "accuracy_at_best_k_for_this_run": max(acc.values())}
            for name, acc in per_run_accuracy.items()
        },
    }
    with (analysis_output_dir / "knn_k_elbow.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    plot_elbow(df, elbow_k, run_names, plot_output_dir / "knn_k_elbow.png")
    return elbow_k


def plot_elbow(df: pd.DataFrame, elbow_k: int, run_names: Sequence[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    fig.patch.set_facecolor(_SURFACE)
    ax.set_facecolor(_SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(_AXIS)
    ax.spines["bottom"].set_color(_AXIS)
    ax.grid(axis="y", color=_GRIDLINE, linewidth=1.0, zorder=0)
    ax.tick_params(colors=_TICK_LABEL, labelsize=9)

    ax.plot(df["k"], 1.0 - df["raw_accuracy"], color=_PALETTE[0], linewidth=2.5,
             marker="o", markersize=4, label="raw embeddings", zorder=3)
    for i, name in enumerate(run_names):
        color = _PALETTE[(i + 1) % len(_PALETTE)]
        ax.plot(df["k"], 1.0 - df[f"{name}_projected_accuracy"], color=color, linewidth=1.5,
                 marker="o", markersize=3, linestyle="--", label=f"{name} (projected)", zorder=2)

    ax.axvline(elbow_k, color=_AXIS, linewidth=1.5, linestyle=":", zorder=1)
    ax.annotate(
        f"elbow k={elbow_k}", xy=(elbow_k, ax.get_ylim()[1]), xytext=(4, -4),
        textcoords="offset points", color=_INK_SECONDARY, fontsize=9, va="top",
    )

    ax.set_title("Leave-one-out k-NN error vs. k (elbow selected on raw-embedding curve)",
                 color=_INK_PRIMARY, fontsize=12, loc="left", pad=10)
    ax.set_xlabel("k", color=_INK_SECONDARY, fontsize=10)
    ax.set_ylabel("1 - accuracy (vs. ground-truth layer)", color=_INK_SECONDARY, fontsize=10)
    legend = ax.legend(frameon=False, fontsize=8, labelcolor=_INK_SECONDARY, loc="upper right")
    legend.set_zorder(4)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", type=Path, default=Path("outputs/predictions"))
    parser.add_argument("--analysis-output-dir", type=Path, default=Path("outputs/analysis"))
    parser.add_argument("--plot-output-dir", type=Path, default=Path("outputs/plots"))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    elbow_k = run_k_selection(args.predictions_dir, args.analysis_output_dir, args.plot_output_dir)
    print(f"\nSelected k={elbow_k}")


if __name__ == "__main__":
    main()
