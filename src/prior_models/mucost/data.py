"""Data preparation for the MuCoST reproduction on DLPFC sections.

The official pipeline builds a spatial ``radius_graph`` (r=150, up to 6
neighbors, then ``to_undirected``) and a gene co-expression ``knn_graph``
(k=6, cosine) on a PCA(latent_dim) projection of the expression.  Both are
implemented here with sklearn/scipy instead of ``torch_geometric`` while
preserving the published edge semantics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from anndata import AnnData
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from .config import MuCostConfig


@dataclass(frozen=True)
class MuCostSectionData:
    section_id: str
    features: np.ndarray          # (n_spots, 3000) HVG expression
    coordinates: np.ndarray       # (n_spots, 2) pixel coordinates
    labels: np.ndarray            # ground-truth cortical layer strings
    barcodes: np.ndarray          # spot barcodes
    spatial_edge_index: np.ndarray  # (2, E_s) undirected radius-graph edges
    feature_edge_index: np.ndarray  # (2, E_f) directed cosine k-NN edges
    combined_edge_index: np.ndarray # (2, E_s + E_f) concatenation
    feature_source: str = "adata.obsm['feat']"


def radius_graph_edges(
    coordinates: np.ndarray, radius: float, max_num_neighbors: int
) -> np.ndarray:
    """Nearest-within-radius edges, then undirected (radius_graph + to_undirected)."""
    coordinates = np.asarray(coordinates, dtype=np.float32)
    tree = NearestNeighbors(radius=radius).fit(coordinates)
    directed: list[tuple[int, int]] = []
    neighbors = tree.radius_neighbors(coordinates, radius=radius, sort_results=True)
    distances_list, indices_list = neighbors
    for node, (dists, inds) in enumerate(zip(distances_list, indices_list)):
        # Exclude self (radius query includes the point itself).
        mask = inds != node
        inds, dists = inds[mask], dists[mask]
        if len(inds) > max_num_neighbors:
            keep = np.argsort(dists)[:max_num_neighbors]
            inds = inds[keep]
        for neighbor in inds:
            directed.append((int(node), int(neighbor)))
    pairs = set(directed)
    undirected = pairs | {(dst, src) for (src, dst) in pairs}
    edges = np.asarray(sorted(undirected), dtype=np.int64).T
    return edges


def feature_knn_edges(features: np.ndarray, k: int, latent_dim: int, seed: int) -> np.ndarray:
    """Cosine k-NN edges on a PCA(latent_dim) projection (directed, knn_graph)."""
    pca = PCA(n_components=latent_dim, random_state=seed)
    embedding = pca.fit_transform(np.asarray(features, dtype=np.float32))
    neighbors = (
        NearestNeighbors(n_neighbors=k + 1, metric="cosine")
        .fit(embedding)
        .kneighbors(embedding, return_distance=False)[:, 1:]
    )
    src = np.repeat(np.arange(features.shape[0]), k)
    dst = neighbors.reshape(-1)
    return np.stack([src, dst], axis=0).astype(np.int64, copy=False)


def prepare_section(
    adata: AnnData, section_id: str, config: MuCostConfig
) -> MuCostSectionData:
    """Prepare one DLPFC section for MuCoST training."""
    if "feat" not in adata.obsm:
        raise KeyError("MuCoST reproduction requires adata.obsm['feat']")
    features = np.asarray(adata.obsm["feat"], dtype=np.float32)
    if features.ndim != 2 or features.shape[0] != adata.n_obs:
        raise ValueError(f"Invalid feature shape {features.shape} for {adata.n_obs} spots")
    if not np.isfinite(features).all():
        raise ValueError("Node features contain non-finite values")

    coordinates = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    spatial_edges = radius_graph_edges(coordinates, config.radius, config.rknn)
    feature_edges = feature_knn_edges(features, config.knn, config.latent_dim, config.seed)
    combined = np.concatenate([spatial_edges, feature_edges], axis=1)

    return MuCostSectionData(
        section_id=section_id,
        features=features,
        coordinates=coordinates,
        labels=adata.obs["ground_truth"].astype(str).to_numpy(),
        barcodes=adata.obs_names.astype(str).to_numpy(),
        spatial_edge_index=spatial_edges,
        feature_edge_index=feature_edges,
        combined_edge_index=combined,
    )