"""Tissue-annotation figures for SPARC-align across training epochs.

Mirrors ``analysis/layer_annotation_plots/`` (ground truth beside each model's spatial
domain assignment) but along the epoch axis instead of across models: how SPARC-align's
predicted layers sharpen -- or fail to -- as stage-1 training proceeds.

    python -m src.sparc_align.annotation_plots --section-id 151507

Reads the per-checkpoint assignments written by ``convergence.py --save-predictions``
and reuses ``src.prior_models.plot_layer_annotations`` for the colour mapping and panel
rendering, so the palette and Hungarian label matching are identical to the existing
figures rather than a lookalike.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.prior_models.plot_layer_annotations import build_color_mapping, plot_section


def _mapped(predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Hungarian-match cluster ids to ground-truth layer names for colouring."""
    mapping = build_color_mapping(labels, predictions)
    return np.array([mapping[int(p)] for p in predictions])


def build_epoch_figure(section_id: str, run: str, out_dir: Path, dataset_name: str = "DLPFC"):
    """Ground truth plus one panel per epoch checkpoint, in a single row."""
    pred_dir = Path("outputs/sparc_align") / run / "predictions"
    files = sorted(pred_dir.glob(f"{section_id}_epoch*.npz"))
    if not files:
        raise FileNotFoundError(
            f"no per-epoch predictions for {section_id} under {pred_dir}. "
            "Re-run convergence.py with --save-predictions "
            f"{section_id}"
        )

    from sklearn.metrics import adjusted_rand_score

    with np.load(files[0], allow_pickle=True) as first:
        labels = np.asarray(first["labels"])
        coordinates = np.asarray(first["coordinates"], dtype=float)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(files) + 1, figsize=(3.1 * (len(files) + 1), 3.8))

    plot_section(axes[0], coordinates, labels, "Ground Truth", legend=False, title_fontsize=11)

    per_epoch = []
    for ax, path in zip(axes[1:], files):
        epoch = int(path.stem.split("epoch")[-1])
        with np.load(path, allow_pickle=True) as data:
            predictions = np.asarray(data["predictions"], dtype=int)
        ari = adjusted_rand_score(labels, predictions)
        per_epoch.append((epoch, ari))
        plot_section(ax, coordinates, _mapped(predictions, labels),
                     f"{epoch} epochs\nARI={ari:.3f}", legend=False, title_fontsize=11)

    # Legend once, on the far right, so it does not repeat across panels.
    import matplotlib.patches as mpatches
    from src.prior_models.plot_layer_annotations import LAYER_COLORS, LAYER_LABELS
    present = [l for l in LAYER_LABELS if l in set(map(str, labels))]
    fig.legend(
        handles=[mpatches.Patch(color=LAYER_COLORS[l], label=l) for l in present],
        loc="center left", bbox_to_anchor=(0.995, 0.5), fontsize=8, frameon=False,
    )
    fig.suptitle(
        f"SPARC-align spatial domains vs ground truth — {dataset_name} {section_id}",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 0.99, 0.94])
    path = out_dir / f"{dataset_name}_{section_id}_epochs.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path, per_epoch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section-id", default="151507")
    parser.add_argument("--run", default="convergence_all12")
    parser.add_argument("--dataset-name", default="DLPFC")
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_root or f"outputs/sparc_align/{args.run}/figures/annotations")
    path, per_epoch = build_epoch_figure(args.section_id, args.run, out_dir, args.dataset_name)
    print(path)
    for epoch, ari in per_epoch:
        print(f"  epoch {epoch:3d}  ARI={ari:.4f}")


if __name__ == "__main__":
    main()
