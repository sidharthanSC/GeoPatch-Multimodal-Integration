"""Compute one shared highly-variable-gene (HVG) set across all twelve DLPFC sections.

Each section's ``adata.var["highly_variable"]`` was already computed (seurat_v3,
3000 genes), but **independently per section** -- checked directly: section
``151508``'s 3000 HVGs overlap only ~38% with ``151507``'s. Using either section's flag
as-is would give a feature space where the same vector position means a different gene
in different sections, which is unusable for one shared encoder across all sections.

This recomputes HVGs jointly, batch-aware (``batch_key="section_id"``): scanpy scores
each gene as highly-variable independently within each section, then ranks genes by how
many sections flag them as highly variable (ties broken by median rank across
sections), and keeps the top ``n_top_genes`` by that combined ranking -- the standard
approach for getting one gene set that's consistently informative across batches/samples
rather than dominated by one batch's technical variation.

``flavor="seurat"`` (dispersion-based, works on log-normalized data) is used rather than
the section-level default ``"seurat_v3"`` (which expects raw counts) because
``checkpoints/dlpfc.pkl`` only carries ``adata.X`` as already-log1p-normalized values
(``adata.raw`` is not present).

Usage
-----
    python -m src.gene_encoder.hvg_selection

Output: ``outputs/gene_encoder/shared_hvgs.json`` -- a JSON list of gene names.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import anndata as ad
import scanpy as sc

from src.datasets.dlpfc import DlpfcDataset


def compute_shared_hvgs(
    dataset: DlpfcDataset, n_top_genes: int = 3000, flavor: str = "seurat"
) -> List[str]:
    """Jointly select HVGs across every section in ``dataset``, batch-aware by section.

    Returns the shared gene names, in the dataset's native gene order (not HVG-rank
    order), so indexing is stable regardless of ``n_top_genes``.
    """
    section_frames = []
    for section_id in dataset.section_ids():
        section_adata = dataset.get_section(section_id)
        minimal = ad.AnnData(X=section_adata.X, var=section_adata.var[[]].copy())
        minimal.obs["section_id"] = section_id
        section_frames.append(minimal)

    combined = ad.concat(section_frames, join="inner", index_unique="-")
    sc.pp.highly_variable_genes(
        combined, flavor=flavor, n_top_genes=n_top_genes, batch_key="section_id"
    )

    hvg_mask = combined.var["highly_variable"].to_numpy()
    original_gene_order = dataset.get_section(dataset.section_ids()[0]).var_names
    return [gene for gene in original_gene_order if gene in set(combined.var_names[hvg_mask])]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, default=Path("checkpoints/dlpfc.pkl"))
    parser.add_argument("--n-top-genes", type=int, default=3000)
    parser.add_argument("--flavor", type=str, default="seurat")
    parser.add_argument("--output", type=Path, default=Path("outputs/gene_encoder/shared_hvgs.json"))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    dataset = DlpfcDataset.from_checkpoint(args.checkpoint_path)
    shared_hvgs = compute_shared_hvgs(dataset, n_top_genes=args.n_top_genes, flavor=args.flavor)

    print(f"Selected {len(shared_hvgs)} shared HVGs (target {args.n_top_genes}) "
          f"across {len(dataset.section_ids())} sections")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(shared_hvgs, f, indent=2)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
