"""Build each section's k-nearest-neighbor spatial graph (Visium hex-grid neighbors).

For every spot, finds its ``k`` nearest neighbors by physical position
(``adata.obsm["spatial"]``), within that spot's own section only (neighbors never cross
sections). ``k=6`` matches the 10x Visium hexagonal spot layout, where every interior
spot has exactly 6 physical neighbors.

Usage
-----
    python -m src.gene_encoder.spatial_graph

Output: ``outputs/gene_encoder/spatial_knn_graph.pkl`` -- ``{section_id: {barcode:
[neighbor_barcode, ...]}}``, ``k`` neighbor barcodes per spot (self excluded), ordered
nearest-to-farthest.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List

from anndata import AnnData
from sklearn.neighbors import NearestNeighbors

from src.datasets.dlpfc import DlpfcDataset

SpatialGraph = Dict[str, List[str]]


def build_section_knn_graph(adata: AnnData, k: int = 6) -> SpatialGraph:
    """One section's barcode -> k nearest neighbor barcodes (self excluded), by ``obsm["spatial"]``."""
    coords = adata.obsm["spatial"]
    barcodes = adata.obs_names.to_numpy()

    finder = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, neighbor_idx = finder.kneighbors(coords)

    graph: SpatialGraph = {}
    for i, row in enumerate(neighbor_idx):
        row = row[row != i][:k]
        graph[barcodes[i]] = barcodes[row].tolist()
    return graph


def build_all_spatial_graphs(dataset: DlpfcDataset, k: int = 6) -> Dict[str, SpatialGraph]:
    return {
        section_id: build_section_knn_graph(dataset.get_section(section_id), k=k)
        for section_id in dataset.section_ids()
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, default=Path("checkpoints/dlpfc.pkl"))
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--output", type=Path, default=Path("outputs/gene_encoder/spatial_knn_graph.pkl"))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    dataset = DlpfcDataset.from_checkpoint(args.checkpoint_path)
    graphs = build_all_spatial_graphs(dataset, k=args.k)

    total_spots = sum(len(g) for g in graphs.values())
    avg_neighbors = sum(len(v) for g in graphs.values() for v in g.values()) / total_spots
    print(f"Built k={args.k} spatial graphs for {len(graphs)} sections, {total_spots} spots total")
    print(f"Average neighbors per spot: {avg_neighbors:.2f} (== k unless a section has < k+1 spots)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as f:
        pickle.dump(graphs, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
