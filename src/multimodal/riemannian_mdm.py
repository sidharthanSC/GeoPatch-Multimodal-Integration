"""Method 3 (supervised): Riemannian Minimum-Distance-to-Mean (MDM) classification.

Per training-set layer, the log-Euclidean mean of that layer's per-spot cross-modal
covariance matrices (``src/multimodal/covariance.py``) is its "layer representation" --
7 mean covariances (Layer_1..Layer_6, WM). Each held-out test spot is classified by
nearest layer mean, log-Euclidean distance (``||logm(spot) - mean_log_layer||_F``).

Uses the same stratified train/test split as method 4
(``src/multimodal/evaluation.py``'s ``shared_train_test_split``) so their accuracies
are directly comparable.

Usage
-----
    python -m src.multimodal.riemannian_mdm
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from src.multimodal.covariance import CovarianceSummary, covariance_summary, matrix_log_batch
from src.multimodal.data import load_embeddings_and_labels
from src.multimodal.evaluation import shared_train_test_split


def layer_log_euclidean_means(
    summary: CovarianceSummary, labels: np.ndarray, train_idx: np.ndarray, chunk_size: int = 2000
) -> Dict[str, np.ndarray]:
    """Log-Euclidean mean matrix log per layer, from training spots only."""
    means = {}
    for layer in sorted(np.unique(labels[train_idx])):
        layer_idx = train_idx[labels[train_idx] == layer]
        log_sum = np.zeros((summary.dim, summary.dim))
        for start in range(0, len(layer_idx), chunk_size):
            chunk_idx = layer_idx[start : start + chunk_size]
            log_sum += matrix_log_batch(summary, chunk_idx).sum(axis=0)
        means[layer] = log_sum / len(layer_idx)
    return means


def classify_by_nearest_mean(
    summary: CovarianceSummary, test_idx: np.ndarray, layer_means: Dict[str, np.ndarray], chunk_size: int = 2000
) -> Tuple[np.ndarray, np.ndarray]:
    """Nearest-mean (log-Euclidean distance) classification for every test spot."""
    layers = sorted(layer_means.keys())
    n_test = len(test_idx)
    distances = np.zeros((n_test, len(layers)))

    for layer_index, layer in enumerate(layers):
        mean_log = layer_means[layer]
        for start in range(0, n_test, chunk_size):
            chunk_idx = test_idx[start : start + chunk_size]
            logs = matrix_log_batch(summary, chunk_idx)
            diff = logs - mean_log[None, :, :]
            distances[start : start + len(chunk_idx), layer_index] = np.sqrt(
                np.sum(diff**2, axis=(1, 2))
            )

    predicted_index = distances.argmin(axis=1)
    predicted_labels = np.array([layers[i] for i in predicted_index])
    return predicted_labels, distances


def run(
    checkpoint_path: Path, epsilon: float, test_fraction: float, seed: int
) -> dict:
    gene_emb, image_emb, labels, barcodes, section_ids = load_embeddings_and_labels(checkpoint_path)
    train_idx, test_idx = shared_train_test_split(labels, test_fraction, seed)

    summary = covariance_summary(image_emb, gene_emb, epsilon)
    layer_means = layer_log_euclidean_means(summary, labels, train_idx)
    predicted_labels, distances = classify_by_nearest_mean(summary, test_idx, layer_means)

    accuracy = float(np.mean(predicted_labels == labels[test_idx]))
    return {
        "metrics": {"accuracy": accuracy, "n_train": len(train_idx), "n_test": len(test_idx)},
        "layer_means": layer_means,
        "predicted_labels": predicted_labels,
        "true_labels": labels[test_idx],
        "test_idx": test_idx,
        "barcodes": barcodes,
        "section_ids": section_ids,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, default=Path("checkpoints/dlpfc.pkl"))
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--test-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/multimodal"))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = run(args.checkpoint_path, args.epsilon, args.test_fraction, args.seed)

    print("Method 3 (Riemannian MDM, supervised):", json.dumps(result["metrics"], indent=2))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "method3_riemannian_mdm_metrics.json").open("w") as f:
        json.dump(result["metrics"], f, indent=2)
    np.savez_compressed(
        args.output_dir / "method3_riemannian_mdm.npz",
        predicted_labels=result["predicted_labels"],
        true_labels=result["true_labels"],
        test_idx=result["test_idx"],
        barcodes=result["barcodes"][result["test_idx"]],
        section_ids=result["section_ids"][result["test_idx"]],
    )
    print(f"Wrote {args.output_dir}/method3_riemannian_mdm.npz and method3_riemannian_mdm_metrics.json")


if __name__ == "__main__":
    main()
