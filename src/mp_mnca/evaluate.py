"""Clustering evaluation and spatial refinement for MP-MNCA embeddings."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors


def tied_gmm(
    embeddings: np.ndarray, n_clusters: int, seed: int = 2023
) -> np.ndarray:
    """Approximate R mclust EEE with a full covariance shared by all components.

    Uses float64 for numerical stability with low-rank GCN embeddings.
    """
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


def clustering_metrics(
    labels: np.ndarray, predicted: np.ndarray
) -> dict[str, float]:
    """Compute ARI and NMI for clustering evaluation."""
    return {
        "ari": float(adjusted_rand_score(labels, predicted)),
        "nmi": float(normalized_mutual_info_score(labels, predicted)),
    }


def cluster_accuracy(true_labels: np.ndarray, cluster_ids: np.ndarray) -> float:
    """Best-case clustering accuracy via Hungarian algorithm."""
    from scipy.optimize import linear_sum_assignment

    true_names = np.unique(true_labels)
    cluster_names = np.unique(cluster_ids)
    confusion = np.zeros((len(cluster_names), len(true_names)), dtype=np.int64)
    for i, c in enumerate(cluster_names):
        for j, t in enumerate(true_names):
            confusion[i, j] = np.sum((cluster_ids == c) & (true_labels == t))

    row_ind, col_ind = linear_sum_assignment(-confusion)
    matched = confusion[row_ind, col_ind].sum()
    return float(matched / len(true_labels))


def full_clustering_metrics(
    labels: np.ndarray, predicted: np.ndarray
) -> dict[str, float]:
    """Compute all clustering metrics."""
    return {
        "accuracy": cluster_accuracy(labels, predicted),
        "ari": float(adjusted_rand_score(labels, predicted)),
        "nmi": float(normalized_mutual_info_score(labels, predicted)),
    }


def attention_diagnostics(
    attention_weights: np.ndarray,
    morphology_prior: np.ndarray,
    labels: np.ndarray,
    neighbor_indices: np.ndarray,
) -> dict[str, float]:
    """Compute attention diagnostics.

    Args:
        attention_weights: (n_spots, n_neighbors) attention weights.
        morphology_prior: (n_spots, n_neighbors) image similarity prior.
        labels: (n_spots,) ground truth labels.
        neighbor_indices: (n_spots, n_neighbors) neighbor indices.

    Returns:
        Dictionary of diagnostic metrics.
    """
    n_spots, n_neighbors = attention_weights.shape

    # Attention entropy (per spot, then average)
    entropy = -np.sum(attention_weights * np.log(attention_weights + 1e-10), axis=1)
    mean_entropy = float(np.mean(entropy))

    # Correlation between attention and morphology prior
    corr_matrix = np.corrcoef(attention_weights.flatten(), morphology_prior.flatten())
    attention_morph_corr = float(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0

    # Attention to same-domain vs different-domain neighbors
    same_domain_attn = []
    diff_domain_attn = []
    for i in range(n_spots):
        center_label = labels[i]
        neighbor_labels = labels[neighbor_indices[i]]
        same_mask = neighbor_labels == center_label
        if same_mask.any():
            same_domain_attn.append(attention_weights[i, same_mask].mean())
        if (~same_mask).any():
            diff_domain_attn.append(attention_weights[i, ~same_mask].mean())

    same_domain_mean = float(np.mean(same_domain_attn)) if same_domain_attn else 0.0
    diff_domain_mean = float(np.mean(diff_domain_attn)) if diff_domain_attn else 0.0
    domain_attn_gap = same_domain_mean - diff_domain_mean

    # Effective rank of attention distribution
    # Compute singular values of attention matrix
    U, s, Vt = np.linalg.svd(attention_weights, full_matrices=False)
    s_normalized = s / (s.sum() + 1e-10)
    effective_rank = float(np.exp(-np.sum(s_normalized * np.log(s_normalized + 1e-10))))

    return {
        "attention_entropy": mean_entropy,
        "attention_morphology_correlation": attention_morph_corr,
        "same_domain_attention": same_domain_mean,
        "diff_domain_attention": diff_domain_mean,
        "domain_attention_gap": domain_attn_gap,
        "attention_effective_rank": effective_rank,
    }


def embedding_collapse_diagnostics(embeddings: np.ndarray) -> dict[str, float]:
    """Compute collapse diagnostics for embeddings.

    Args:
        embeddings: (n_spots, embed_dim) embedding matrix.

    Returns:
        Dictionary of collapse diagnostics.
    """
    n_spots, embed_dim = embeddings.shape

    # Per-dimension standard deviation
    std_per_dim = embeddings.std(axis=0)
    mean_std = float(std_per_dim.mean())
    min_std = float(std_per_dim.min())
    max_std = float(std_per_dim.max())
    zero_dim_count = int((std_per_dim < 1e-4).sum())

    # Effective rank
    centered = embeddings - embeddings.mean(axis=0)
    U, s, Vt = np.linalg.svd(centered, full_matrices=False)
    s_normalized = s / (s.sum() + 1e-10)
    effective_rank = float(np.exp(-np.sum(s_normalized * np.log(s_normalized + 1e-10))))

    # Covariance off-diagonal magnitude
    cov = centered.T @ centered / (n_spots - 1)
    off_diag = cov - np.diag(np.diag(cov))
    off_diag_fro = float(np.sqrt(np.sum(off_diag**2)))
    diag_fro = float(np.sqrt(np.sum(np.diag(cov)**2)))
    off_diag_ratio = off_diag_fro / (diag_fro + 1e-10)

    return {
        "embed_std_mean": mean_std,
        "embed_std_min": min_std,
        "embed_std_max": max_std,
        "embed_zero_dim_count": zero_dim_count,
        "embed_effective_rank": effective_rank,
        "embed_cov_off_diag_frobenius": off_diag_fro,
        "embed_cov_off_diag_ratio": off_diag_ratio,
    }