"""Build each slice's SPD matrix from softmax-cosine-similarity probabilities.

For a slice's ``(slice_size, dim)`` block of projected embeddings:

1. ``S = cosine_similarity(embeddings)`` -- pairwise cosine similarity, ``(n, n)``.
2. ``P = softmax(S / tau, axis=1)`` -- row-wise softmax with temperature ``tau``,
   making every row a probability distribution over the slice's other members
   ("softmax-based cosine similarity probabilities").
3. ``C = P @ P.T`` -- the slice's ``(n, n)`` SPD matrix. This, not ``(P + P.T) / 2``,
   is what guarantees positive-semi-definiteness: for any real matrix ``P``,
   ``x^T (P P^T) x = ||P^T x||^2 >= 0`` always, whereas merely symmetrizing a
   row-stochastic matrix carries no such guarantee. Positive-definiteness (not just
   semi-definiteness) is required by the matrix logarithm the downstream Riemannian
   mean uses -- see ``src/analysis/layer_riemannian_mean.py``.

**Why temperature matters here**: a slice's 20 spots all share one (section, layer),
and this projector was trained specifically to pull same-layer spots together, so their
pairwise cosine similarities all land in a narrow band near 1.0 (empirically, spreads
of ~0.04-0.10 within a slice). Un-tempered softmax (``tau=1``) barely differentiates
values that close together -- every row comes out nearly uniform (``~1/n`` everywhere),
which makes ``P`` (and therefore ``C = P @ P.T``) collapse to near rank 1: one dominant
eigenvalue near 1.0, everything else at the numerical floor. Dividing by a small
``tau`` before the softmax rescales those small real differences back into a range
where they produce a meaningfully varied (higher-rank) distribution.

Usage
-----
    python -m src.analysis.slice_covariance --temperature 0.05

Reads ``outputs/covariance/slice_groups.pkl`` (see ``src/analysis/slice_grouping.py``)
and each section's ``obsm["proj_emb"]`` from the checkpoint. Output:
``outputs/covariance/slice_covariances.pkl`` -- ``{slice_id: (n, n) ndarray}``.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict

import numpy as np

from src.datasets.dlpfc import DlpfcDataset

SliceCovariances = Dict[str, np.ndarray]


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity, ``(n, n)``, for an ``(n, dim)`` embedding block."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / norms
    return normalized @ normalized.T


def softmax_rows(matrix: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Row-wise softmax with temperature (each row sums to 1).

    Smaller ``temperature`` sharpens the distribution (amplifies small differences in
    ``matrix``); ``temperature=1.0`` is plain softmax.
    """
    scaled = matrix / temperature
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def build_slice_covariance(embeddings: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """One slice's ``(n, dim)`` embeddings -> its ``(n, n)`` SPD matrix (see module docstring)."""
    similarity = cosine_similarity_matrix(embeddings)
    probabilities = softmax_rows(similarity, temperature=temperature)
    return probabilities @ probabilities.T


def build_all_slice_covariances(
    dataset: DlpfcDataset,
    slice_groups: Dict[str, dict],
    obsm_key: str = "proj_emb",
    temperature: float = 1.0,
) -> SliceCovariances:
    """Build every slice's SPD matrix, grouped by section for efficient barcode lookup."""
    slices_by_section: Dict[str, list] = {}
    for slice_id, meta in slice_groups.items():
        slices_by_section.setdefault(meta["section_id"], []).append(slice_id)

    covariances: SliceCovariances = {}
    for section_id, slice_ids in slices_by_section.items():
        adata = dataset.get_section(section_id)
        barcode_to_row = {barcode: i for i, barcode in enumerate(adata.obs_names)}
        proj_emb = adata.obsm[obsm_key]

        for slice_id in slice_ids:
            barcodes = slice_groups[slice_id]["barcodes"]
            rows = [barcode_to_row[bc] for bc in barcodes]
            covariances[slice_id] = build_slice_covariance(proj_emb[rows], temperature=temperature)

    return covariances


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, default=Path("checkpoints/dlpfc.pkl"))
    parser.add_argument("--slice-groups", type=Path, default=Path("outputs/covariance/slice_groups.pkl"))
    parser.add_argument("--obsm-key", type=str, default="proj_emb")
    parser.add_argument(
        "--temperature", type=float, default=0.05,
        help="Softmax temperature; smaller sharpens the distribution. Plain "
             "(tau=1.0) softmax collapses to near rank-1 for this data -- see "
             "module docstring.",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/covariance/slice_covariances.pkl"))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    with args.slice_groups.open("rb") as f:
        slice_groups = pickle.load(f)

    dataset = DlpfcDataset.from_checkpoint(args.checkpoint_path)
    covariances = build_all_slice_covariances(
        dataset, slice_groups, obsm_key=args.obsm_key, temperature=args.temperature
    )

    shapes = {c.shape for c in covariances.values()}
    all_eigvals = np.stack([np.linalg.eigvalsh(c) for c in covariances.values()])
    eigvals_min = float(all_eigvals.min())
    print(f"Built {len(covariances)} slice covariance matrices, shape(s)={shapes}, temperature={args.temperature}")
    print(f"Smallest eigenvalue across all slices: {eigvals_min:.6g} "
          f"({'all positive -- valid SPD' if eigvals_min > 0 else 'WARNING: non-positive found'})")
    mean_sorted = np.sort(all_eigvals, axis=1)[:, ::-1].mean(axis=0)
    print(f"Mean eigenvalue spectrum across all slices (largest -> smallest):\n  {np.round(mean_sorted, 4)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as f:
        pickle.dump(covariances, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
