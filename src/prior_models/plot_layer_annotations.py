"""Plot per-model spatial layer annotations for a DLPFC section.

Reuses the per-section trained checkpoints' saved predictions (the models are
already trained; this module only loads ``refined_predictions``, ``labels`` and
``coordinates`` from each run's embedding NPZ and plots them). Individual plots
are written to ``<output_root>/<model_name>/<dataset>_<section_id>.png`` and a
single 1x6 subplot of all five models plus ground truth is written to
``<output_root>/subplots/<dataset>_<section_id>.png``.

Colors are derived from the ground-truth cortical layers (``Layer_1``-``Layer_6``,
``WM``) so that every model uses a common spatial color reference. Predicted
cluster ids are mapped to ground-truth labels by Hungarian matching of the
confusion matrix before coloring.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config_imports import LAYER_LABELS, LAYER_COLORS


@dataclass(frozen=True)
class SectionPrediction:
    model_name: str
    predictions: np.ndarray
    labels: np.ndarray
    coordinates: np.ndarray


MODEL_RUN_EMBEDDINGS: dict[str, Path] = {
    "STAIG": Path("outputs/prior_models/staig_paper_default_img_emb_seed0_all12_v3/embeddings"),
    "SpaGCN": Path("outputs/prior_models/spagcn_paper_default_seed100_all12_v1/embeddings"),
    "GraphST": Path("outputs/prior_models/graphst_paper_default_seed41_all12_v1/embeddings"),
    "MuCoST": Path("outputs/prior_models/mucost_paper_default_seed2023_all12_v1/embeddings"),
    "MP-MNCA": Path("outputs/mp_mnca/phase1_all12/embeddings"),
}


def load_section_predictions(
    section_id: str, model_embeddings: dict[str, Path] | None = None
) -> dict[str, SectionPrediction]:
    """Load refined predictions, ground-truth labels and coordinates for a section."""
    model_embeddings = model_embeddings or MODEL_RUN_EMBEDDINGS
    result: dict[str, SectionPrediction] = {}
    for model_name, embed_dir in model_embeddings.items():
        path = embed_dir / f"{section_id}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing predictions for {model_name}: {path}")
        with np.load(path, allow_pickle=True) as data:
            result[model_name] = SectionPrediction(
                model_name=model_name,
                predictions=np.asarray(data["refined_predictions"], dtype=int),
                labels=np.asarray(data["labels"]),
                coordinates=np.asarray(data["coordinates"], dtype=float),
            )
    return result


def build_color_mapping(
    true_labels: np.ndarray, pred_labels: np.ndarray
) -> dict[int, str]:
    """Hungarian-match predicted cluster ids onto ground-truth labels.

    Returns a dict mapping each predicted cluster id to a ground-truth label name.
    """
    true_names = [str(x) for x in np.unique(true_labels)]
    pred_ids = np.unique(pred_labels)
    known = [label for label in true_names if label in LAYER_LABELS]
    if set(known) != set(true_names):
        raise ValueError(f"Unexpected ground-truth labels: {set(true_names) - set(known)}")
    truth_index = {label: i for i, label in enumerate(LAYER_LABELS)}
    # Weight of (pred cluster, truth label) = number of matching spots.
    weight = np.zeros((len(pred_ids), len(LAYER_LABELS)), dtype=float)
    for i, p in enumerate(pred_ids):
        mask = pred_labels == p
        truth_vals, counts = np.unique(true_labels[mask], return_counts=True)
        for t, c in zip(truth_vals, counts):
            weight[i, truth_index[str(t)]] = c
    assignment = _hungarian_maximize(weight)
    return {int(pred_ids[i]): LAYER_LABELS[j] for i, j in assignment.items()}


def _hungarian_maximize(weight: np.ndarray) -> dict[int, int]:
    """Maximal assignment via a greedy swap-free approach (scipy linear_sum_assignment)."""
    from scipy.optimize import linear_sum_assignment

    rows, cols = linear_sum_assignment(-weight)
    return {int(r): int(c) for r, c in zip(rows, cols)}


def _hex_to_rgb(color: str) -> tuple[float, float, float]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def _tissue_region(coordinates: np.ndarray) -> np.ndarray | None:
    """Return (x, y) polygon vertices of the section's convex hull as a tissue impression."""
    from scipy.spatial import ConvexHull

    try:
        hull = ConvexHull(coordinates)
    except Exception:
        return None
    pts = coordinates[hull.vertices]
    return np.vstack([pts, pts[0]])


def plot_section(
    ax: object,
    coordinates: np.ndarray,
    labels: np.ndarray,
    title: str,
    legend: bool = False,
    marker_size: float = 6.0,
    tissue_background: bool = True,
    title_fontsize: float = 11,
) -> None:
    """Scatter spots colored by label using the shared ground-truth color map.

    ``marker_size`` is in points^2; larger values give denser, more visible spots.
    ``tissue_background`` draws a light convex-hull fill behind the spots so the
    section outline reads like a tissue impression.
    ``title_fontsize`` controls the panel heading (e.g. the model name).
    """
    from matplotlib.colors import to_rgba

    if tissue_background:
        poly = _tissue_region(coordinates)
        if poly is not None:
            ax.fill(
                poly[:, 0],
                poly[:, 1],
                color="#e8e4dd",
                edgecolor="#c9c2b5",
                linewidth=0.8,
                zorder=0,
            )
            ax.set_facecolor("#f7f5f0")
    ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=[to_rgba(LAYER_COLORS[str(l)]) for l in labels],
        s=marker_size,
        linewidths=0,
        zorder=2,
    )
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    if legend:
        import matplotlib.patches as mpatches

        patches = [
            mpatches.Patch(color=LAYER_COLORS[l], label=l) for l in LAYER_LABELS
        ]
        ax.legend(
            handles=patches,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=8,
            frameon=False,
        )


def save_individual_plot(
    pred: SectionPrediction,
    dataset_name: str,
    section_id: str,
    output_root: Path,
    color_mapping: dict[int, str],
) -> Path:
    """Write one model's colored layer-annotation plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = output_root / pred.model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{dataset_name}_{section_id}.png"

    labels = np.array([color_mapping[int(p)] for p in pred.predictions])
    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=200)
    plot_section(
        ax,
        pred.coordinates,
        labels,
        pred.model_name,
        legend=True,
        marker_size=10.0,
        tissue_background=True,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_ground_truth_plot(
    pred: SectionPrediction,
    dataset_name: str,
    section_id: str,
    output_root: Path,
) -> Path:
    """Write the ground-truth layer annotation plot for the section."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = output_root / "Ground_Truth"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{dataset_name}_{section_id}.png"

    labels = np.array([str(l) for l in pred.labels])
    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=200)
    plot_section(
        ax,
        pred.coordinates,
        labels,
        "Ground Truth",
        legend=True,
        marker_size=10.0,
        tissue_background=True,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_subplot(
    predictions: dict[str, SectionPrediction],
    color_mappings: dict[str, dict[int, str]],
    dataset_name: str,
    section_id: str,
    output_root: Path,
) -> Path:
    """Write one 1x6 row of ground truth plus all five models with headings."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = output_root / "subplots"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{dataset_name}_{section_id}.png"

    # Ground truth first (left-most), then the five models for comparison.
    order = ["Ground Truth", *predictions.keys()]
    fig, axes = plt.subplots(1, 6, figsize=(26, 4.4), dpi=200)
    gt = next(iter(predictions.values()))
    for ax, panel in zip(axes, order):
        if panel == "Ground Truth":
            labels = np.array([str(l) for l in gt.labels])
            title = "Ground Truth"
        elif panel == "MP-MNCA":
            pred = predictions[panel]
            labels = np.array([color_mappings[panel][int(p)] for p in pred.predictions])
            title = "MP-MNCA (Ours)"
        else:
            pred = predictions[panel]
            labels = np.array([color_mappings[panel][int(p)] for p in pred.predictions])
            title = panel
        plot_section(
            ax,
            gt.coordinates if panel == "Ground Truth" else predictions[panel].coordinates,
            labels,
            title,
            marker_size=8.0,
            tissue_background=True,
            title_fontsize=18,
        )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_layer_annotation_plots(
    section_id: str,
    dataset_name: str,
    output_root: Path,
    model_embeddings: dict[str, Path] | None = None,
) -> dict[str, Path]:
    """Generate all individual plots plus the combined subplot for one section."""
    output_root = Path(output_root)
    predictions = load_section_predictions(section_id, model_embeddings)

    # Single common color mapping per model derived from its own confusion matrix.
    color_mappings: dict[str, dict[int, str]] = {}
    for model_name, pred in predictions.items():
        color_mappings[model_name] = build_color_mapping(pred.labels, pred.predictions)

    individual: dict[str, Path] = {}
    for model_name, pred in predictions.items():
        individual[model_name] = save_individual_plot(
            pred, dataset_name, section_id, output_root, color_mappings[model_name]
        )
    individual["Ground_Truth"] = save_ground_truth_plot(
        next(iter(predictions.values())), dataset_name, section_id, output_root
    )
    subplot = save_subplot(
        predictions, color_mappings, dataset_name, section_id, output_root
    )
    return {**individual, "subplot": subplot}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section-id", default="151507", help="DLPFC section ID")
    parser.add_argument("--dataset-name", default="DLPFC", help="prefix for plot file names")
    parser.add_argument(
        "--output-root",
        default="analysis/layer_annotation_plots",
        help="folder where per-model folders and subplots/ live",
    )
    args = parser.parse_args()

    paths = generate_layer_annotation_plots(
        args.section_id, args.dataset_name, Path(args.output_root)
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
