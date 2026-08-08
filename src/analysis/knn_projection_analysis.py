"""Compare leave-one-out KNN label assignment before vs. after the learned projection.

Uses the ``raw_embeddings`` (frozen BYOL image embeddings, pre-projection) and
``projected_embeddings`` (post-projection) arrays exported by ``src/train/train.py``
(see ``export_embeddings`` in that module) to answer: for each spot, does its
neighborhood-majority cortical-layer label (``Layer_1``..``Layer_6``, ``WM``) change
once the frozen embedding is replaced by the trained projection?

For every spot, a K-nearest-neighbor vote (self excluded) over ``adata.obs["ground_truth"]``
gives a "neighborhood label" in each embedding space; spots where the raw-space and
projected-space neighborhood labels disagree are the ones whose local neighborhood
composition changed as a result of training. Barcodes are recovered from the
``DlpfcDataset`` checkpoint (the exported ``.npz`` does not carry them) using the same
per-section iteration and ground-truth filtering as ``src/train/train.py``'s
``load_embeddings_and_slice_ids``, and are verified to align positionally with the
``.npz`` arrays before use.

Usage
-----
    python -m src.analysis.knn_projection_analysis
    python -m src.analysis.knn_projection_analysis --k 10 --embeddings-npz outputs/predictions/other_run.npz

Outputs (under ``--output-dir``, default ``outputs/analysis``; ``<run>`` is the input
``.npz``'s filename with a trailing ``_embeddings`` stripped, e.g.
``layer_projector_e200_embeddings.npz`` -> ``layer_projector_e200``, so multiple runs'
analyses coexist instead of overwriting each other):

- ``<run>_knn_projection_summary.json`` — accuracies, change counts, category breakdown.
- ``<run>_knn_label_transitions.csv`` — confusion matrix of raw-space vs. projected-space
  neighborhood label, counted over all spots.
- ``<run>_changed_spots.csv`` — one row per spot whose neighborhood label changed: barcode,
  section id, ground truth, raw/projected neighborhood label, and change category.
- ``<run>_knn_predictions_full.csv`` — the same columns for every spot, changed or not.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.datasets.dlpfc import DlpfcDataset


def load_barcodes_labels_and_slices(
    checkpoint_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rebuild (barcodes, ground_truth_labels, slice_ids), aligned to the exported `.npz`.

    Mirrors ``src/train/train.py``'s ``load_embeddings_and_slice_ids`` exactly (same
    per-section iteration order via ``dataset.section_ids()``, same
    ``ground_truth.notna()`` filter) but also keeps ``adata.obs_names`` (the spot
    barcodes), which that function doesn't return and the `.npz` export doesn't store.
    """
    dataset = DlpfcDataset.from_checkpoint(checkpoint_path)

    barcodes = []
    labels = []
    slice_ids = []

    for section_id in dataset.section_ids():
        adata = dataset.get_section(section_id)
        gt = adata.obs["ground_truth"]
        valid = gt.notna().to_numpy()

        barcodes.append(adata.obs_names.to_numpy()[valid])
        labels.append(gt.to_numpy(dtype=object)[valid].astype(str))
        slice_ids.append(np.full(int(valid.sum()), str(section_id), dtype=object))

    return (
        np.concatenate(barcodes, axis=0),
        np.concatenate(labels, axis=0),
        np.concatenate(slice_ids, axis=0),
    )


def leave_one_out_knn_labels(embeddings: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Majority-vote label from each point's k nearest neighbors, excluding itself.

    Ties are broken in favor of the closer neighbor: votes are tallied in nearest-to-
    farthest order and ``Counter.most_common`` keeps the first-inserted label among
    ties (insertion order == neighbor proximity order).
    """
    n = embeddings.shape[0]
    finder = NearestNeighbors(n_neighbors=k + 1).fit(embeddings)
    _, neighbor_idx = finder.kneighbors(embeddings)

    predicted = np.empty(n, dtype=object)
    for i in range(n):
        row = neighbor_idx[i]
        row = row[row != i][:k]
        counts: dict = {}
        for idx in row:
            counts[labels[idx]] = counts.get(labels[idx], 0) + 1
        predicted[i] = max(counts.items(), key=lambda item: item[1])[0]
    return predicted


def categorize_change(true_label: str, raw_pred: str, proj_pred: str) -> str:
    if raw_pred == proj_pred:
        return "unchanged"
    raw_correct = raw_pred == true_label
    proj_correct = proj_pred == true_label
    if not raw_correct and proj_correct:
        return "improved"
    if raw_correct and not proj_correct:
        return "regressed"
    return "still_wrong_different_label"


def run_analysis(
    embeddings_npz: Path, checkpoint_path: Path, output_dir: Path, k: int
) -> None:
    data = np.load(embeddings_npz, allow_pickle=True)
    raw_embeddings = data["raw_embeddings"]
    projected_embeddings = data["projected_embeddings"]
    npz_labels = data["cortical_labels"].astype(str)
    npz_slice_ids = data["slice_ids"].astype(str)

    barcodes, labels, slice_ids = load_barcodes_labels_and_slices(checkpoint_path)

    if not (np.array_equal(labels, npz_labels) and np.array_equal(slice_ids, npz_slice_ids)):
        raise ValueError(
            f"Checkpoint-derived (labels, slice_ids) do not align positionally with "
            f"{embeddings_npz}. The two were likely built from different dataset "
            f"states/filters -- re-export embeddings before running this analysis."
        )

    run_name = embeddings_npz.stem.removesuffix("_embeddings")

    print(f"Loaded {len(labels)} spots, {raw_embeddings.shape[1]}-d raw / "
          f"{projected_embeddings.shape[1]}-d projected embeddings, k={k}")

    raw_pred = leave_one_out_knn_labels(raw_embeddings, labels, k)
    proj_pred = leave_one_out_knn_labels(projected_embeddings, labels, k)

    accuracy_raw = float(np.mean(raw_pred == labels))
    accuracy_projected = float(np.mean(proj_pred == labels))
    changed_mask = raw_pred != proj_pred
    categories = np.array(
        [categorize_change(t, r, p) for t, r, p in zip(labels, raw_pred, proj_pred)]
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    full_df = pd.DataFrame({
        "barcode": barcodes,
        "slice_id": slice_ids,
        "ground_truth": labels,
        "raw_knn_label": raw_pred,
        "projected_knn_label": proj_pred,
        "changed": changed_mask,
        "category": categories,
    })
    full_df.to_csv(output_dir / f"{run_name}_knn_predictions_full.csv", index=False)
    full_df[changed_mask].to_csv(output_dir / f"{run_name}_changed_spots.csv", index=False)

    transitions = pd.crosstab(
        pd.Series(raw_pred, name="raw_knn_label"),
        pd.Series(proj_pred, name="projected_knn_label"),
    )
    transitions.to_csv(output_dir / f"{run_name}_knn_label_transitions.csv")

    category_counts = {cat: int((categories == cat).sum()) for cat in np.unique(categories)}
    summary = {
        "embeddings_npz": str(embeddings_npz),
        "checkpoint_path": str(checkpoint_path),
        "k": k,
        "n_spots": int(len(labels)),
        "accuracy_raw": accuracy_raw,
        "accuracy_projected": accuracy_projected,
        "n_changed": int(changed_mask.sum()),
        "pct_changed": float(changed_mask.mean()),
        "category_counts": category_counts,
    }
    with (output_dir / f"{run_name}_knn_projection_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nWrote outputs to {output_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embeddings-npz", type=Path,
        default=Path("outputs/predictions/layer_projector_embeddings.npz"),
    )
    parser.add_argument("--checkpoint-path", type=Path, default=Path("checkpoints/dlpfc.pkl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/analysis"))
    parser.add_argument("--k", type=int, default=5)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_analysis(args.embeddings_npz, args.checkpoint_path, args.output_dir, args.k)


if __name__ == "__main__":
    main()
