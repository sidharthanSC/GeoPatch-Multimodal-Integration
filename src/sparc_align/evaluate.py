"""Evaluation for SPARC latents.

The clustering path is deliberately imported from :mod:`src.mp_mnca.evaluate` rather
than reimplemented, so the SPARC arm and the MP-MNCA/STAIG arms provably share one
protocol: PCA -> tied GMM (sklearn's approximation of R mclust EEE) -> 15-NN spatial
plurality refinement -> ARI/NMI.

The SPARC-native diagnostics replace MP-MNCA's attention diagnostics, which have no
analogue here (SPARC has no attention): dead-latent rates per stream, the
activation-pattern table from the paper's Table 1, self- vs cross-reconstruction
NMSE, and support-overlap Jaccard.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from src.mp_mnca.evaluate import (
    clustering_metrics,
    embedding_collapse_diagnostics,
    refine_labels,
    tied_gmm,
)

__all__ = [
    "clustering_metrics",
    "cluster_sparc_latents",
    "embedding_collapse_diagnostics",
    "refine_labels",
    "tied_gmm",
    "activation_pattern_table",
    "support_jaccard",
    "reconstruction_report",
]


def build_cluster_input(latents: dict[str, np.ndarray], mode: str) -> np.ndarray:
    """Reduce the per-stream sparse codes to the single matrix that gets clustered.

    SPARC keeps ``z_gene`` and ``z_image`` separate by design (paper Section 3.2),
    so choosing among them is our decision, not the paper's -- and it materially
    changes the result. ``sum`` is the default: with Global TopK both codes share a
    support, so their sum stays k-sparse and keeps both streams' magnitudes.
    """
    if mode == "sum":
        return np.sum(list(latents.values()), axis=0)
    if mode == "concat":
        return np.concatenate(list(latents.values()), axis=1)
    if mode == "support":
        # Binary k-hot support pattern: discards magnitudes, keeps only "which
        # concepts fired". Only meaningful under Global TopK.
        stacked = np.sum(list(latents.values()), axis=0)
        return (stacked > 0).astype(np.float32)
    if mode in latents:
        return latents[mode]
    raise ValueError(f"unknown cluster_input mode: {mode!r}")


def cluster_sparc_latents(
    latents: dict[str, np.ndarray],
    labels: np.ndarray,
    coordinates: np.ndarray,
    config,
) -> dict:
    """Run the MP-MNCA clustering protocol over SPARC latents.

    PCA before the tied GMM is not optional here: a k-sparse L-dimensional code has
    many exactly-zero columns, and a tied full covariance over those is singular
    regardless of ``reg_covar``. MP-MNCA applies the same reduction to its 3000-d
    embeddings, so this keeps the arms matched.
    """
    matrix = build_cluster_input(latents, config.cluster_input)
    n_clusters = config.n_clusters or len(np.unique(labels))

    pca_dim = min(config.pca_dim, matrix.shape[0], matrix.shape[1])
    reduced = PCA(n_components=pca_dim, random_state=0).fit_transform(matrix)

    predictions = tied_gmm(reduced, n_clusters=n_clusters, seed=config.gmm_seed)
    refined = refine_labels(predictions, coordinates, config.refinement_neighbors)

    before = clustering_metrics(labels, predictions)
    after = clustering_metrics(labels, refined)
    metrics = {
        **before,
        "refined_ari": after["ari"],
        "refined_nmi": after["nmi"],
        "n_clusters": int(n_clusters),
        "pca_dim": int(pca_dim),
        **embedding_collapse_diagnostics(reduced),
    }
    return {"metrics": metrics, "predictions": predictions, "refined_predictions": refined}


def activation_pattern_table(latents: dict[str, np.ndarray]) -> dict[str, float]:
    """Reproduce the paper's Table 1: how many latents live in how many streams.

    Under Global TopK a latent should be either alive in every stream or dead in
    every stream; the mixed buckets are the failure mode Local TopK exhibits.
    """
    names = list(latents)
    alive = {name: (latents[name] > 0).any(axis=0) for name in names}
    alive_count = np.sum([alive[name] for name in names], axis=0)
    n_latents = alive_count.shape[0]
    n_streams = len(names)

    table = {
        "n_latents": int(n_latents),
        "all_dead_fraction": float((alive_count == 0).mean()),
        "all_alive_fraction": float((alive_count == n_streams).mean()),
        "mixed_fraction": float(((alive_count > 0) & (alive_count < n_streams)).mean()),
    }
    for name in names:
        table[f"dead_fraction_{name}"] = float((~alive[name]).mean())
    return table


def support_jaccard(latents: dict[str, np.ndarray]) -> dict[str, float]:
    """Mean per-spot Jaccard overlap between each stream pair's active supports.

    Under Global TopK this is 1.0 by construction; it is the Local TopK ablation
    that makes it informative, and it is the closest analogue available to the
    paper's concept-profile Jaccard given that DLPFC offers 7 coarse labels rather
    than Open Images' 432 classes.
    """
    names = list(latents)
    result: dict[str, float] = {}
    for i, source in enumerate(names):
        for target in names[i + 1 :]:
            a = latents[source] > 0
            b = latents[target] > 0
            intersection = np.logical_and(a, b).sum(axis=1)
            union = np.logical_or(a, b).sum(axis=1)
            overlap = np.divide(
                intersection, union, out=np.zeros(len(union), dtype=np.float64), where=union > 0
            )
            result[f"support_jaccard_{source}_{target}"] = float(overlap.mean())
    return result


def reconstruction_report(
    self_nmse: dict[str, float], cross_nmse: dict[str, float]
) -> dict[str, float]:
    """Flatten reconstruction NMSE into flat metric keys.

    A cross-NMSE at or above 1.0 means the latent code from one modality predicts
    the other no better than that modality's own mean -- i.e. the shared-concept
    premise SPARC rests on does not hold for this data.
    """
    report = {f"self_nmse_{name}": float(value) for name, value in self_nmse.items()}
    report.update({f"cross_nmse_{name}": float(value) for name, value in cross_nmse.items()})
    if self_nmse:
        report["self_nmse_mean"] = float(np.mean(list(self_nmse.values())))
    if cross_nmse:
        report["cross_nmse_mean"] = float(np.mean(list(cross_nmse.values())))
    return report
