"""Clustering evaluation helpers for SpaGCN outputs.

Re-exports the shared evaluation protocol (tied-covariance GMM as the
R ``mclust(modelNames='EEE')`` analogue, 15-neighbor spatial refinement, and
ARI/NMI) used across the repository.
"""

from __future__ import annotations

from ..staig.evaluate import clustering_metrics, refine_labels, tied_gmm

__all__ = ["clustering_metrics", "refine_labels", "tied_gmm"]