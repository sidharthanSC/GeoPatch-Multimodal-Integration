"""Data preparation for the GraphST reproduction on DLPFC sections.

The official ``construct_interaction`` builds a coordinate k-NN graph with
``ot.dist``.  ``ot`` is unavailable here, so the k-NN interaction is built with
``sklearn.neighbors.NearestNeighbors`` and then symmetrised exactly as the
official code does (``adj = adj + adj.T; adj[adj > 1] = 1``).

Feature inputs: the repository checkpoint already contains the GraphST-normalised
3,000-HVG expression in ``adata.obsm["feat"]`` (log1p, scaled to [0,10]).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from anndata import AnnData
from sklearn.neighbors import NearestNeighbors

from .config import GraphStConfig


@dataclass(frozen=True)
class GraphStSectionData:
    section_id: str
    features: np.ndarray          # (n_spots, 3000) HVG expression
    coordinates: np.ndarray       # (n_spots, 2) pixel coordinates
    labels: np.ndarray            # ground-truth cortical layer strings
    barcodes: np.ndarray          # spot barcodes
    adjacency: np.ndarray         # symmetric k-NN interaction matrix
    graph_neigh: np.ndarray       # interaction + identity (readout mask)
    feature_source: str = "adata.obsm['feat']"


def build_interaction(coordinates: np.ndarray, n_neighbors: int) -> np.ndarray:
    """Return the symmetric k-NN interaction matrix (official semantics)."""
    coordinates = np.asarray(coordinates, dtype=np.float32)
    n_spots = coordinates.shape[0]
    indices = (
        NearestNeighbors(n_neighbors=n_neighbors + 1)
        .fit(coordinates)
        .kneighbors(coordinates, return_distance=False)[:, 1:]
    )
    interaction = np.zeros((n_spots, n_spots), dtype=np.float32)
    src = np.repeat(np.arange(n_spots), n_neighbors)
    interaction[src, indices.reshape(-1)] = 1.0
    adjacency = interaction + interaction.T
    adjacency = np.where(adjacency > 1, 1, adjacency).astype(np.float32)
    return adjacency


def preprocess_adjacency(adjacency: np.ndarray) -> np.ndarray:
    """Symmetrically normalize the adjacency and add self loops."""
    n = adjacency.shape[0]
    adj = adjacency + np.eye(n)
    d = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat = np.diag(d_inv_sqrt)
    return (d_mat @ adj @ d_mat).astype(np.float32)


def prepare_section(
    adata: AnnData, section_id: str, config: GraphStConfig
) -> GraphStSectionData:
    """Prepare one DLPFC section for GraphST training."""
    if "feat" not in adata.obsm:
        raise KeyError("GraphST reproduction requires adata.obsm['feat']")
    features = np.asarray(adata.obsm["feat"], dtype=np.float32)
    if features.ndim != 2 or features.shape[0] != adata.n_obs:
        raise ValueError(f"Invalid feature shape {features.shape} for {adata.n_obs} spots")
    if not np.isfinite(features).all():
        raise ValueError("Node features contain non-finite values")

    coordinates = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    interaction = build_interaction(coordinates, config.n_neighbors)
    graph_neigh = (interaction + np.eye(interaction.shape[0])).astype(np.float32)

    return GraphStSectionData(
        section_id=section_id,
        features=features,
        coordinates=coordinates,
        labels=adata.obs["ground_truth"].astype(str).to_numpy(),
        barcodes=adata.obs_names.astype(str).to_numpy(),
        adjacency=interaction,
        graph_neigh=graph_neigh,
    )