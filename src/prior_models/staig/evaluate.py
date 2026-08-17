"""Clustering and spatial refinement for STAIG embeddings."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors


def tied_gmm(embeddings: np.ndarray, n_clusters: int, seed: int = 2023) -> np.ndarray:
    """Approximate R mclust EEE with a full covariance shared by all components."""
    # sklearn otherwise preserves float32 and can reject a regularized covariance
    # as non-positive-definite for low-rank GCN embeddings.
    embeddings = np.asarray(embeddings, dtype=np.float64)
    return GaussianMixture(
        n_components=n_clusters,
        covariance_type="tied",
        random_state=seed,
        n_init=1,
        max_iter=1000,
        reg_covar=1e-6,
    ).fit_predict(embeddings)


def refine_labels(
    labels: np.ndarray, coordinates: np.ndarray, n_neighbors: int = 15
) -> np.ndarray:
    """Replace every label by the plurality among its nearest spatial neighbors."""
    indices = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(coordinates).kneighbors(
        coordinates, return_distance=False
    )[:, 1:]
    refined = np.empty_like(labels)
    for index, neighbors in enumerate(indices):
        values, counts = np.unique(labels[neighbors], return_counts=True)
        refined[index] = values[np.argmax(counts)]
    return refined


def clustering_metrics(labels: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "ari": float(adjusted_rand_score(labels, predicted)),
        "nmi": float(normalized_mutual_info_score(labels, predicted)),
    }
