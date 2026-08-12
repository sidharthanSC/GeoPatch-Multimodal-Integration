"""Shared utilities: cluster-to-label accuracy, and the one train/test split every
supervised method (3, 4) reuses, for a fair side-by-side comparison.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import train_test_split


def cluster_accuracy(true_labels: np.ndarray, cluster_ids: np.ndarray) -> float:
    """Best-case clustering accuracy: optimally permute cluster IDs to match ground-truth
    labels (Hungarian algorithm on the confusion matrix), then compute accuracy.

    Standard way to report an "accuracy" for unsupervised clustering -- cluster IDs
    themselves carry no inherent label identity, only groupings, so raw
    ``(cluster_ids == true_labels)`` isn't meaningful without this matching step.
    """
    true_names = np.unique(true_labels)
    cluster_names = np.unique(cluster_ids)
    confusion = np.zeros((len(cluster_names), len(true_names)), dtype=np.int64)
    for i, c in enumerate(cluster_names):
        for j, t in enumerate(true_names):
            confusion[i, j] = np.sum((cluster_ids == c) & (true_labels == t))

    row_ind, col_ind = linear_sum_assignment(-confusion)
    matched = confusion[row_ind, col_ind].sum()
    return float(matched / len(true_labels))


def clustering_metrics(true_labels: np.ndarray, cluster_ids: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": cluster_accuracy(true_labels, cluster_ids),
        "ari": float(adjusted_rand_score(true_labels, cluster_ids)),
        "nmi": float(normalized_mutual_info_score(true_labels, cluster_ids)),
    }


def shared_train_test_split(
    labels: np.ndarray, test_fraction: float = 0.3, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """One stratified-by-layer split, reused by every supervised method (3, 4) so their
    reported accuracies are directly comparable -- same train spots, same test spots."""
    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_fraction, random_state=seed, stratify=labels
    )
    return train_idx, test_idx
