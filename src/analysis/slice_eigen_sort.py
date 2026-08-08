"""Canonicalize each slice's SPD matrix to its sorted-eigenvalue diagonal form.

Each slice's ``(n, n)`` matrix from ``src/analysis/slice_covariance.py`` is built from
that slice's own randomly-chosen 20 spots, so row/column index ``i`` refers to a
*different, unrelated spot* in every other slice -- there is no shared basis across
slices to make raw matrix entries comparable, let alone poolable. Since we chose the
slot (spot) order randomly, it carries no information; only the matrix's *spectrum*
(its eigenvalues -- rotation/labeling-invariant) is meaningfully comparable across
slices.

So each slice's matrix ``C`` is replaced by ``D = diag(sorted eigenvalues of C,
descending)`` -- discarding the eigenvectors (which encode the arbitrary spot
ordering) and keeping only the canonical, cross-slice-comparable spectrum. This is
"arranging the slot order by sorting the eigenvalues": slot ``j`` now always holds the
``j``-th largest eigenvalue, for every slice, regardless of which spots happened to be
in it.

This also fixes up the floating-point noise from ``slice_covariance.py`` (eigenvalues
that are mathematically >= 0 by construction but can land a hair below 0 due to
eigensolver rounding): eigenvalues are floored to a small positive epsilon before
sorting, so every output matrix is strictly positive-definite -- required for the
matrix logarithm the Riemannian mean (``src/analysis/layer_riemannian_mean.py``) uses.

Usage
-----
    python -m src.analysis.slice_eigen_sort

Reads ``outputs/covariance/slice_covariances.pkl``. Output:
``outputs/covariance/slice_covariances_sorted.pkl`` -- ``{slice_id: (n, n) ndarray}``,
each a diagonal matrix.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict

import numpy as np

_EIGENVALUE_FLOOR = 1e-10


def canonicalize_by_eigenvalues(matrix: np.ndarray, descending: bool = True) -> np.ndarray:
    """Replace a symmetric matrix with the diagonal matrix of its sorted eigenvalues.

    Eigenvalues are floored to ``_EIGENVALUE_FLOOR`` first, so the result is always
    strictly positive-definite even if ``matrix`` had a near-zero or slightly negative
    eigenvalue from upstream floating-point noise.
    """
    eigenvalues = np.linalg.eigvalsh(matrix)
    eigenvalues = np.clip(eigenvalues, _EIGENVALUE_FLOOR, None)
    order = np.argsort(eigenvalues)
    if descending:
        order = order[::-1]
    return np.diag(eigenvalues[order])


def canonicalize_all(
    covariances: Dict[str, np.ndarray], descending: bool = True
) -> Dict[str, np.ndarray]:
    return {
        slice_id: canonicalize_by_eigenvalues(matrix, descending=descending)
        for slice_id, matrix in covariances.items()
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--slice-covariances", type=Path, default=Path("outputs/covariance/slice_covariances.pkl")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/covariance/slice_covariances_sorted.pkl")
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    with args.slice_covariances.open("rb") as f:
        covariances = pickle.load(f)

    sorted_covariances = canonicalize_all(covariances)

    min_eigenvalue = min(float(np.diag(m).min()) for m in sorted_covariances.values())
    max_eigenvalue = max(float(np.diag(m).max()) for m in sorted_covariances.values())
    print(f"Canonicalized {len(sorted_covariances)} slices to sorted-eigenvalue diagonal form")
    print(f"Eigenvalue range across all slices: [{min_eigenvalue:.6g}, {max_eigenvalue:.6g}]")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as f:
        pickle.dump(sorted_covariances, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
