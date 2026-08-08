"""Compute each cortical layer's Riemannian mean SPD matrix, pooling across sections.

Groups the canonicalized slice matrices from ``src/analysis/slice_eigen_sort.py`` by
**layer only** -- per the project hierarchy (spot -> slice -> layer -> section ->
donor), sections are not treated independently at this stage; every slice from every
section that has a given layer contributes to that one layer's mean.

Uses the Log-Euclidean mean: convert each SPD matrix to its matrix logarithm
(``V log(\\Lambda) V^T`` via eigendecomposition), average the logs arithmetically, then
matrix-exponentiate the average back to SPD. This is the standard closed-form
"Riemannian mean" for SPD matrices in covariance-pooling applications (Arsigny et al.),
and -- importantly for correctness here -- it is mathematically *exact* (not an
approximation) for commuting matrices such as our diagonal ones: for diagonal inputs it
reduces to the elementwise geometric mean of each diagonal entry across the group. The
function is written generically over any symmetric PD matrix, not diagonal-only, so it
stays correct if an earlier stage ever changes to produce non-diagonal matrices.

Usage
-----
    python -m src.analysis.layer_riemannian_mean

Reads ``outputs/covariance/slice_groups.pkl`` and
``outputs/covariance/slice_covariances_sorted.pkl``. Output:
``outputs/covariance/layer_riemannian_means.pkl`` -- ``{layer: {"mean": (n, n) ndarray,
"n_slices": int, "sections": List[str]}}``.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

_EIGENVALUE_FLOOR = 1e-10


def _symmetric_matrix_log(matrix: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.clip(eigenvalues, _EIGENVALUE_FLOOR, None)
    return eigenvectors @ np.diag(np.log(eigenvalues)) @ eigenvectors.T


def _symmetric_matrix_exp(matrix: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    return eigenvectors @ np.diag(np.exp(eigenvalues)) @ eigenvectors.T


def spd_log_euclidean_mean(matrices: Sequence[np.ndarray]) -> np.ndarray:
    """Log-Euclidean mean of a set of SPD matrices (see module docstring)."""
    if not matrices:
        raise ValueError("Need at least one matrix to average.")
    logs = np.stack([_symmetric_matrix_log(m) for m in matrices], axis=0)
    return _symmetric_matrix_exp(logs.mean(axis=0))


def compute_layer_means(
    slice_groups: Dict[str, dict], sorted_covariances: Dict[str, np.ndarray]
) -> Dict[str, dict]:
    matrices_by_layer: Dict[str, List[np.ndarray]] = {}
    sections_by_layer: Dict[str, set] = {}

    for slice_id, meta in slice_groups.items():
        layer = meta["layer"]
        matrices_by_layer.setdefault(layer, []).append(sorted_covariances[slice_id])
        sections_by_layer.setdefault(layer, set()).add(meta["section_id"])

    return {
        layer: {
            "mean": spd_log_euclidean_mean(matrices),
            "n_slices": len(matrices),
            "sections": sorted(sections_by_layer[layer]),
        }
        for layer, matrices in matrices_by_layer.items()
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice-groups", type=Path, default=Path("outputs/covariance/slice_groups.pkl"))
    parser.add_argument(
        "--slice-covariances-sorted", type=Path,
        default=Path("outputs/covariance/slice_covariances_sorted.pkl"),
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/covariance/layer_riemannian_means.pkl"))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    with args.slice_groups.open("rb") as f:
        slice_groups = pickle.load(f)
    with args.slice_covariances_sorted.open("rb") as f:
        sorted_covariances = pickle.load(f)

    layer_means = compute_layer_means(slice_groups, sorted_covariances)

    for layer in sorted(layer_means):
        info = layer_means[layer]
        diag = np.diag(info["mean"])
        print(
            f"{layer}: {info['n_slices']} slices from {len(info['sections'])} sections "
            f"({', '.join(info['sections'])}) | eigenvalue range "
            f"[{diag.min():.4g}, {diag.max():.4g}]"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as f:
        pickle.dump(layer_means, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
