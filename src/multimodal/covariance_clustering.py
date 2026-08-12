"""Method 2 (unsupervised): clustering on the per-spot cross-modal covariance matrix.

Two variants, both built from ``src/multimodal/covariance.py``'s per-spot ``(128, 128)``
covariance between image and gene embeddings:

- **2a -- log-Euclidean K-Means**: vectorize each spot's matrix logarithm via scaled
  vectorization (``svec`` -- upper triangle, off-diagonal entries scaled by
  ``sqrt(2)`` so Euclidean distance in vector space equals Frobenius distance in matrix
  space, the standard trick that makes "clustering in log-space" equivalent to
  log-Euclidean-metric clustering), then plain K-Means on those vectors.
- **2b -- spectral clustering on unit correlation vectors**: convert each spot's
  covariance to a correlation matrix (normalize by its own diagonal), vectorize the
  off-diagonal upper triangle (the diagonal is always exactly 1, no information),
  L2-normalize to a unit vector, then ``SpectralClustering`` (nearest-neighbors
  affinity) on those.

Both process spots in chunks (not all ``(128, 128)`` matrices materialized at once --
would be several GB) using the closed forms in ``src/multimodal/covariance.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans, SpectralClustering
from sklearn.decomposition import PCA

from src.multimodal.covariance import covariance_summary, matrix_log_batch
from src.multimodal.data import load_embeddings_and_labels
from src.multimodal.evaluation import clustering_metrics


def svec_batch(matrices: np.ndarray) -> np.ndarray:
    """Scaled vectorization of a batch of symmetric matrices: ``(b, dim, dim) -> (b, dim*(dim+1)/2)``."""
    b, dim, _ = matrices.shape
    scale = np.sqrt(2.0) * np.ones((dim, dim))
    np.fill_diagonal(scale, 1.0)
    scaled = matrices * scale[None, :, :]
    rows, cols = np.triu_indices(dim)
    return scaled[:, rows, cols]


def off_diagonal_unit_vectors(covariances: np.ndarray) -> np.ndarray:
    """Correlation matrix (from covariance), off-diagonal upper triangle, L2-normalized."""
    b, dim, _ = covariances.shape
    diag = np.diagonal(covariances, axis1=1, axis2=2)  # (b, dim)
    denom = np.sqrt(diag[:, :, None] * diag[:, None, :])
    correlation = covariances / denom
    rows, cols = np.triu_indices(dim, k=1)  # off-diagonal only
    vectors = correlation[:, rows, cols]
    norm = np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    return vectors / norm


def build_log_euclidean_vectors(image_emb, gene_emb, epsilon: float, chunk_size: int) -> np.ndarray:
    summary = covariance_summary(image_emb, gene_emb, epsilon)
    n = image_emb.shape[0]
    dim = summary.dim
    svec_dim = dim * (dim + 1) // 2
    out = np.empty((n, svec_dim), dtype=np.float32)
    for start in range(0, n, chunk_size):
        idx = np.arange(start, min(start + chunk_size, n))
        logs = matrix_log_batch(summary, idx)
        out[idx] = svec_batch(logs).astype(np.float32)
    return out


def build_correlation_unit_vectors(image_emb, gene_emb, epsilon: float, chunk_size: int) -> np.ndarray:
    summary = covariance_summary(image_emb, gene_emb, epsilon)
    n = image_emb.shape[0]
    dim = summary.dim
    vec_dim = dim * (dim - 1) // 2
    out = np.empty((n, vec_dim), dtype=np.float32)
    eye = np.eye(dim)
    for start in range(0, n, chunk_size):
        idx = np.arange(start, min(start + chunk_size, n))
        u = summary.unit_direction[idx]
        outer = np.einsum("bi,bj->bij", u, u)
        covariances = summary.floor_eigenvalue[idx, None, None] * eye[None, :, :] + (
            summary.top_eigenvalue[idx] - summary.floor_eigenvalue[idx]
        )[:, None, None] * outer
        out[idx] = off_diagonal_unit_vectors(covariances).astype(np.float32)
    return out


def run_log_euclidean_kmeans(
    image_emb, gene_emb, labels, epsilon: float, n_clusters: int, seed: int, chunk_size: int = 2000
) -> dict:
    vectors = build_log_euclidean_vectors(image_emb, gene_emb, epsilon, chunk_size)
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, n_init=10, random_state=seed, batch_size=4096)
    cluster_ids = kmeans.fit_predict(vectors)
    return {"metrics": clustering_metrics(labels, cluster_ids), "cluster_ids": cluster_ids}


def run_spectral_on_correlation(
    image_emb, gene_emb, labels, epsilon: float, n_clusters: int, seed: int,
    n_neighbors: int = 10, chunk_size: int = 2000, pca_dim: int = 50,
) -> dict:
    vectors = build_correlation_unit_vectors(image_emb, gene_emb, epsilon, chunk_size)

    # The 8128-d correlation vectors are highly redundant (each spot's underlying
    # covariance has only ~130 real degrees of freedom -- a 128-d direction plus 2
    # scalars, see src/multimodal/covariance.py). Nearest-neighbor search directly in
    # 8128-d degrades toward brute-force (curse of dimensionality) and is intractable
    # at 47k spots -- confirmed empirically (killed after 18+ minutes with no output).
    # PCA first, standard practice for spectral clustering on high-dim data.
    if pca_dim < vectors.shape[1]:
        vectors = PCA(n_components=pca_dim, random_state=seed).fit_transform(vectors)

    spectral = SpectralClustering(
        n_clusters=n_clusters, affinity="nearest_neighbors", n_neighbors=n_neighbors,
        random_state=seed, assign_labels="kmeans", n_jobs=-1,
    )
    cluster_ids = spectral.fit_predict(vectors)
    return {"metrics": clustering_metrics(labels, cluster_ids), "cluster_ids": cluster_ids}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, default=Path("checkpoints/dlpfc.pkl"))
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--n-clusters", type=int, default=7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-neighbors", type=int, default=10)
    parser.add_argument("--pca-dim", type=int, default=50, help="Dimensionality reduction before spectral clustering's nearest-neighbor graph.")
    parser.add_argument(
        "--spectral-subsample", type=int, default=15000,
        help="Spectral clustering's eigendecomposition scales poorly (confirmed: even "
             "with PCA pre-reduction, full 47k spots didn't finish in 15 minutes). "
             "Run 2b on a stratified-by-layer random subsample of this size instead; "
             "0 disables subsampling (uses all spots).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/multimodal"))
    parser.add_argument("--skip-spectral", action="store_true", help="Run only 2a (log-Euclidean K-Means).")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    gene_emb, image_emb, labels, barcodes, section_ids = load_embeddings_and_labels(args.checkpoint_path)

    print("Running 2a: log-Euclidean K-Means ...")
    result_2a = run_log_euclidean_kmeans(image_emb, gene_emb, labels, args.epsilon, args.n_clusters, args.seed)
    print("Method 2a (log-Euclidean K-Means):", json.dumps(result_2a["metrics"], indent=2))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "method2a_log_euclidean_kmeans_metrics.json").open("w") as f:
        json.dump(result_2a["metrics"], f, indent=2)
    np.savez_compressed(
        args.output_dir / "method2a_log_euclidean_kmeans.npz",
        cluster_ids=result_2a["cluster_ids"], labels=labels, barcodes=barcodes, section_ids=section_ids,
    )

    if not args.skip_spectral:
        print("Running 2b: spectral clustering on unit correlation vectors ...")
        result_2b = run_spectral_on_correlation(
            image_emb, gene_emb, labels, args.epsilon, args.n_clusters, args.seed, args.n_neighbors,
            pca_dim=args.pca_dim,
        )
        print("Method 2b (spectral clustering):", json.dumps(result_2b["metrics"], indent=2))
        with (args.output_dir / "method2b_spectral_metrics.json").open("w") as f:
            json.dump(result_2b["metrics"], f, indent=2)
        np.savez_compressed(
            args.output_dir / "method2b_spectral.npz",
            cluster_ids=result_2b["cluster_ids"], labels=labels, barcodes=barcodes, section_ids=section_ids,
        )


if __name__ == "__main__":
    main()
