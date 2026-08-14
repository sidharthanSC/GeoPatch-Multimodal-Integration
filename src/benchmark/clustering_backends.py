"""Clustering backends for Table A: KMeans, Gaussian-mixture (mclust substitute), Leiden.

Per the user's decisions for this run:
- The Gaussian-mixture backend uses ``sklearn.mixture.GaussianMixture`` as the
  standard Python substitute for R's ``mclust`` (no new R/rpy2 dependency).
  ``covariance_type="diag"`` is used rather than ``"full"``: a full 128x128 covariance
  per component needs ~8.3k free parameters per cluster, more than the ~600 spots/
  cluster a 4200-spot section splits into at k=7, so `"full"` is frequently singular
  or badly overfit here.
- The Leiden backend uses ``leidenalg``/``python-igraph`` via scanpy, newly added as
  project dependencies for this run.

Both GMM and Leiden are label-count-informed (``k=7``, matching every other backend in
this report): GMM directly via ``n_components``, Leiden via a resolution search that
targets exactly 7 communities.
"""

from __future__ import annotations

from typing import Optional, Tuple

import anndata as ad
import numpy as np
import scanpy as sc
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


def kmeans_cluster(x: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    return KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(x)


def gmm_cluster(x: np.ndarray, n_clusters: int, seed: int, covariance_type: str = "diag") -> np.ndarray:
    model = GaussianMixture(n_components=n_clusters, covariance_type=covariance_type, random_state=seed, n_init=1)
    return model.fit_predict(x)


def _leiden_partition(adata: ad.AnnData, resolution: float, seed: int) -> np.ndarray:
    sc.tl.leiden(
        adata, resolution=resolution, random_state=seed, key_added="leiden",
        flavor="igraph", n_iterations=2, directed=False,
    )
    return adata.obs["leiden"].to_numpy().astype(int)


def build_leiden_neighbors(x: np.ndarray, n_neighbors: int = 15, seed: int = 0) -> ad.AnnData:
    """Build the shared neighbor graph once; reused by resolution search and every seed."""
    adata = ad.AnnData(X=x.astype(np.float32))
    sc.pp.neighbors(adata, use_rep="X", n_neighbors=n_neighbors, random_state=seed)
    return adata


def search_leiden_resolution(
    adata: ad.AnnData, target_k: int, seed: int = 0, max_iter: int = 15,
    res_lo: float = 0.01, res_hi: float = 3.0,
) -> Tuple[float, np.ndarray]:
    """Binary-search a Leiden resolution that yields exactly ``target_k`` clusters.

    Falls back to the closest achievable cluster count within ``[res_lo, res_hi]`` if
    the exact target is never hit in ``max_iter`` steps (rare, but resolution<->cluster-
    count isn't strictly monotonic in all graphs).
    """
    best_resolution, best_labels, best_gap = None, None, None
    lo, hi = res_lo, res_hi
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        labels = _leiden_partition(adata, resolution=mid, seed=seed)
        k = len(np.unique(labels))
        gap = abs(k - target_k)
        if best_gap is None or gap < best_gap:
            best_resolution, best_labels, best_gap = mid, labels, gap
        if k == target_k:
            return mid, labels
        if k < target_k:
            lo = mid
        else:
            hi = mid
    return best_resolution, best_labels


def leiden_cluster_multiseed(x: np.ndarray, n_clusters: int, seeds: Tuple[int, ...], n_neighbors: int = 15) -> dict:
    """Resolution-search once (seed ``seeds[0]``), then reuse that resolution across
    every seed, varying only Leiden's own ``random_state`` -- avoids rebuilding the
    neighbor graph and re-searching resolution for every one of the 10 clustering seeds.
    """
    adata = build_leiden_neighbors(x, n_neighbors=n_neighbors, seed=seeds[0])
    resolution, first_labels = search_leiden_resolution(adata, target_k=n_clusters, seed=seeds[0])

    results = {seeds[0]: first_labels}
    for seed in seeds[1:]:
        results[seed] = _leiden_partition(adata, resolution=resolution, seed=seed)
    return {"resolution": resolution, "labels_by_seed": results}
