"""Per-spot cross-modal covariance matrix (image embedding vs. gene embedding).

For one spot, the only inputs are its 128-d image vector and 128-d gene vector -- so
"the covariance matrix between them" means treating these two vectors as 2 observations
of a 128-dim variable (numpy's ``cov(X, rowvar=False)`` convention), which is *always*
rank <= 1 with only 2 observations (confirmed, and expected, per the user): centered,
the two observations are exact negatives of each other, so

    Cov = 0.5 * outer(diff, diff),  where diff = image_vec - gene_vec

exactly -- no approximation, this is what ``np.cov`` computes (verified below). A small
epsilon floor is added to the diagonal, scaled to each spot's own trace (so it stays
meaningful regardless of embedding magnitude), to guarantee strict positive-definiteness
for the Riemannian/log-Euclidean computations methods 2 and 3 need.

Because every spot's (regularized) covariance has the closed form ``floor * I +
(top - floor) * outer(u, u)`` (identity plus one rank-1 term), its eigendecomposition,
matrix logarithm, and reconstruction are all available in closed form too -- no actual
``(128, 128)`` eigendecomposition needed per spot. This matters at 47,329 spots: naively
materializing every spot's ``(128, 128)`` matrix at once would be several GB.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class CovarianceSummary:
    """Every spot's regularized covariance, in closed form: ``floor*I + (top-floor)*uu^T``.

    Equivalent to the full ``(n, 128, 128)`` matrix stack, but stored as three
    ``(n,)``/``(n, 128)`` arrays -- exact, not an approximation (see module docstring).
    """

    unit_direction: np.ndarray  # (n, dim)
    top_eigenvalue: np.ndarray  # (n,) -- the one nontrivial eigenvalue, along unit_direction
    floor_eigenvalue: np.ndarray  # (n,) -- the regularization floor, (dim - 1)-fold degenerate
    dim: int


def build_covariance_matrix(image_vec: np.ndarray, gene_vec: np.ndarray, epsilon: float) -> np.ndarray:
    """One spot's ``(dim, dim)`` covariance, built the direct/brute-force way (for
    validating :func:`covariance_summary`'s closed form, not for bulk use)."""
    stacked = np.stack([image_vec, gene_vec], axis=0)
    cov = np.cov(stacked, rowvar=False)
    dim = cov.shape[0]
    reg = max(epsilon * (np.trace(cov) / dim), 1e-8)
    return cov + reg * np.eye(dim)


def covariance_summary(image_emb: np.ndarray, gene_emb: np.ndarray, epsilon: float) -> CovarianceSummary:
    """Closed-form regularized covariance for every spot at once. See module docstring."""
    diff = image_emb - gene_emb  # (n, dim)
    dim = image_emb.shape[1]

    squared_norm = np.sum(diff**2, axis=1)  # (n,) -- ||diff||^2
    norm = np.sqrt(np.maximum(squared_norm, 1e-24))
    unit_direction = diff / norm[:, None]

    unregularized_trace = 0.5 * squared_norm  # trace(0.5 * outer(diff, diff))
    floor_eigenvalue = np.maximum(epsilon * unregularized_trace / dim, 1e-8)
    top_eigenvalue = unregularized_trace + floor_eigenvalue

    return CovarianceSummary(unit_direction, top_eigenvalue, floor_eigenvalue, dim)


def reconstruct_matrix(summary: CovarianceSummary, index: int) -> np.ndarray:
    """One spot's full ``(dim, dim)`` covariance, reconstructed from the closed form."""
    u = summary.unit_direction[index]
    return summary.floor_eigenvalue[index] * np.eye(summary.dim) + (
        summary.top_eigenvalue[index] - summary.floor_eigenvalue[index]
    ) * np.outer(u, u)


def matrix_log(summary: CovarianceSummary, index: int) -> np.ndarray:
    """One spot's ``logm(covariance)``, closed form (see module docstring): a matrix
    with the same eigenvectors, eigenvalues replaced by their logs."""
    u = summary.unit_direction[index]
    log_floor = np.log(summary.floor_eigenvalue[index])
    log_top = np.log(summary.top_eigenvalue[index])
    return log_floor * np.eye(summary.dim) + (log_top - log_floor) * np.outer(u, u)


def matrix_log_batch(summary: CovarianceSummary, indices: np.ndarray) -> np.ndarray:
    """Vectorized :func:`matrix_log` over multiple spots at once: ``(len(indices), dim, dim)``."""
    u = summary.unit_direction[indices]  # (b, dim)
    log_floor = np.log(summary.floor_eigenvalue[indices])  # (b,)
    log_top = np.log(summary.top_eigenvalue[indices])  # (b,)
    eye = np.eye(summary.dim)
    outer = np.einsum("bi,bj->bij", u, u)  # (b, dim, dim)
    return log_floor[:, None, None] * eye[None, :, :] + (log_top - log_floor)[:, None, None] * outer
