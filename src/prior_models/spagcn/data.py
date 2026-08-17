"""Data preparation for the SpaGCN reproduction on DLPFC sections.

The official SpaGCN pipeline expects raw expression, coordinates, and an
optional histology image.  The repository checkpoint already contains the
3,000-HVG expression in ``adata.obsm["feat"]`` (log1p-scaled to [0,10]), so
that array is used directly as the node feature matrix.  No histology image
is available, so the adjacency is computed from coordinates only (the
histology-free SpaGCN branch); this deviation is recorded in the run summary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from anndata import AnnData
from scipy.spatial.distance import cdist

from .config import SpaGcnConfig


@dataclass(frozen=True)
class SpaGcnSectionData:
    section_id: str
    features: np.ndarray          # (n_spots, 3000) HVG expression
    coordinates: np.ndarray       # (n_spots, 2) pixel coordinates
    labels: np.ndarray            # ground-truth cortical layer strings
    barcodes: np.ndarray          # spot barcodes
    adjacency: np.ndarray         # (n_spots, n_spots) thresholded exp-Gaussian weights
    feature_source: str = "adata.obsm['feat']"


def build_spagcn_adjacency(coordinates: np.ndarray, l: float) -> np.ndarray:
    """Compute the SpaGCN weighted adjacency matrix.

    SpaGCN builds a dense pairwise distance matrix and converts it to edge
    weights ``exp(-d^2 / (2 l^2))``, zeroing weights below 0.01.  ``l`` is the
    Gaussian length scale found by ``search_l``.
    """
    coordinates = np.asarray(coordinates, dtype=np.float32)
    distances = cdist(coordinates, coordinates, metric="euclidean")
    adjacency = np.exp(-(distances**2) / (2.0 * l**2))
    adjacency[adjacency < 0.01] = 0.0
    return adjacency.astype(np.float32)


def calculate_p(adjacency: np.ndarray, l: float) -> float:
    """SpaGCN's neighbourhood-expression contribution function."""
    exp_adj = np.exp(-(adjacency**2) / (2.0 * l**2))
    return float(np.mean(np.sum(exp_adj, axis=1)) - 1)


def search_l(
    p: float,
    adjacency: np.ndarray,
    start: float = 0.01,
    end: float = 1000.0,
    tol: float = 0.01,
    max_run: int = 100,
) -> float:
    """Binary-search the Gaussian length scale ``l`` so that ``p`` holds."""
    run = 0
    p_low = calculate_p(adjacency, start)
    p_high = calculate_p(adjacency, end)
    if p_low > p + tol:
        raise ValueError("l not found: try smaller start point.")
    if p_high < p - tol:
        raise ValueError("l not found: try bigger end point.")
    if np.abs(p_low - p) <= tol:
        return start
    if np.abs(p_high - p) <= tol:
        return end
    while (p_low + tol) < p < (p_high - tol):
        run += 1
        if run > max_run:
            raise ValueError("Exact l not found within max_run.")
        mid = (start + end) / 2
        p_mid = calculate_p(adjacency, mid)
        if np.abs(p_mid - p) <= tol:
            return mid
        if p_mid <= p:
            start = mid
            p_low = p_mid
        else:
            end = mid
            p_high = p_mid
    raise ValueError("l search failed to converge.")


def prepare_section(
    adata: AnnData, section_id: str, config: SpaGcnConfig
) -> SpaGcnSectionData:
    """Prepare one DLPFC section for SpaGCN training."""
    if "feat" not in adata.obsm:
        raise KeyError("SpaGCN reproduction requires adata.obsm['feat']")
    features = np.asarray(adata.obsm["feat"], dtype=np.float32)
    if features.ndim != 2 or features.shape[0] != adata.n_obs:
        raise ValueError(f"Invalid feature shape {features.shape} for {adata.n_obs} spots")
    if not np.isfinite(features).all():
        raise ValueError("Node features contain non-finite values")

    coordinates = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    distances = cdist(coordinates, coordinates, metric="euclidean").astype(np.float32)
    # Official tutorial uses p=0.5 for the neighbourhood-expression contribution.
    l = search_l(p=0.5, adjacency=distances)
    adjacency = build_spagcn_adjacency(coordinates, l)

    return SpaGcnSectionData(
        section_id=section_id,
        features=features,
        coordinates=coordinates,
        labels=adata.obs["ground_truth"].astype(str).to_numpy(),
        barcodes=adata.obs_names.astype(str).to_numpy(),
        adjacency=adjacency,
    )