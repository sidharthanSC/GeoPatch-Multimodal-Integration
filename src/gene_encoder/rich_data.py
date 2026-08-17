"""Sparse expression storage and views for information-preserving gene training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from src.datasets.dlpfc import DlpfcDataset


@dataclass(frozen=True)
class SparseExpressionStore:
    matrix: sparse.csr_matrix
    gene_names: np.ndarray
    barcodes: np.ndarray
    section_ids: np.ndarray
    gene_std: np.ndarray

    def rows(self, indices: np.ndarray) -> np.ndarray:
        return self.matrix[np.asarray(indices, dtype=np.int64)].toarray().astype(
            np.float32, copy=False
        )


def load_sparse_expression(
    dataset: DlpfcDataset, genes: Sequence[str] | None = None
) -> SparseExpressionStore:
    """Load all common genes or one ordered subset without global densification."""
    first = dataset.get_section(dataset.section_ids()[0])
    if genes is None:
        gene_names = np.asarray(first.var_names.astype(str))
    else:
        gene_names = np.asarray([str(gene) for gene in genes])
        if len(np.unique(gene_names)) != len(gene_names):
            raise ValueError("Gene feature list contains duplicates")

    blocks = []
    barcode_blocks = []
    section_blocks = []
    for section_id in dataset.section_ids():
        adata = dataset.get_section(section_id)
        if genes is None:
            observed = np.asarray(adata.var_names.astype(str))
            if not np.array_equal(observed, gene_names):
                raise ValueError(f"Common gene order differs in section {section_id}")
            block = adata.X
        else:
            positions = adata.var_names.get_indexer(gene_names)
            if np.any(positions < 0):
                missing = gene_names[positions < 0][:5].tolist()
                raise ValueError(f"Section {section_id} is missing genes: {missing}")
            block = adata.X[:, positions]
        blocks.append(sparse.csr_matrix(block, dtype=np.float32))
        barcode_blocks.append(np.asarray(adata.obs_names.astype(str)))
        section_blocks.append(np.full(adata.n_obs, str(section_id), dtype=object))

    matrix = sparse.vstack(blocks, format="csr", dtype=np.float32)
    mean = np.asarray(matrix.mean(axis=0)).ravel()
    second_moment = np.asarray(matrix.power(2).mean(axis=0)).ravel()
    gene_std = np.sqrt(np.maximum(second_moment - mean**2, 1e-8)).astype(np.float32)
    return SparseExpressionStore(
        matrix=matrix,
        gene_names=gene_names,
        barcodes=np.concatenate(barcode_blocks),
        section_ids=np.concatenate(section_blocks),
        gene_std=gene_std,
    )


def build_global_neighbor_rows(
    dataset: DlpfcDataset, spatial_graphs: dict[str, dict[str, list[str]]]
) -> np.ndarray:
    """Map every spot to fixed-width global coordinate-neighbor row indices."""
    rows = []
    offset = 0
    for section_id in dataset.section_ids():
        adata = dataset.get_section(section_id)
        barcodes = np.asarray(adata.obs_names.astype(str))
        lookup = {barcode: offset + index for index, barcode in enumerate(barcodes)}
        section_rows = [[lookup[neighbor] for neighbor in spatial_graphs[section_id][barcode]] for barcode in barcodes]
        widths = {len(neighbors) for neighbors in section_rows}
        if len(widths) != 1:
            raise ValueError(f"Section {section_id} has variable neighbor counts: {widths}")
        rows.extend(section_rows)
        offset += adata.n_obs
    return np.asarray(rows, dtype=np.int64)


def fit_expression_geometry(
    matrix: sparse.csr_matrix, n_components: int = 128, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit train-free transductive expression SVD and return scores/components/EVR."""
    n_components = min(n_components, matrix.shape[0] - 1, matrix.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=seed, n_iter=7)
    scores = svd.fit_transform(matrix).astype(np.float32)
    return (
        scores,
        svd.components_.astype(np.float32),
        svd.explained_variance_ratio_.astype(np.float32),
    )


class SparseMaskedView:
    """Stateful BYOL augmentation that caches its exact batch target and mask."""

    def __init__(
        self,
        store: SparseExpressionStore,
        device: torch.device,
        mask_rate: float,
        noise_std_fraction: float,
    ) -> None:
        if not 0 < mask_rate < 1:
            raise ValueError("mask_rate must be in (0, 1)")
        self.store = store
        self.device = device
        self.mask_rate = mask_rate
        self.noise_std_fraction = noise_std_fraction
        self.last_indices: np.ndarray | None = None
        self.last_target: torch.Tensor | None = None
        self.last_mask: torch.Tensor | None = None

    def __call__(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.dtype != torch.long or indices.ndim != 1:
            return torch.zeros(
                indices.shape[0], self.store.matrix.shape[1], device=self.device
            )
        rows = indices.detach().cpu().numpy()
        target = torch.from_numpy(self.store.rows(rows)).to(self.device)
        mask = torch.rand_like(target) < self.mask_rate
        view = target.masked_fill(mask, 0.0)
        if self.noise_std_fraction > 0:
            scale = torch.from_numpy(self.store.gene_std).to(self.device)
            noise = torch.randn_like(view) * (self.noise_std_fraction * scale)
            view = view + noise * (~mask & (target > 0))
        self.last_indices = rows.copy()
        self.last_target = target
        self.last_mask = mask
        return view
