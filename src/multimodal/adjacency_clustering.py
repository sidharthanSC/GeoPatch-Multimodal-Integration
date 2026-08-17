"""Cluster DLPFC sections directly from sparse spatial adjacency matrices.

The candidate topology is a symmetric coordinate KNN graph. Three fixed conditions
isolate what supplies useful edge information: binary spatial adjacency, all-gene
expression-weighted adjacency, and fixed image-embedding-weighted adjacency.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.cluster import SpectralClustering

from src.datasets.dlpfc import DlpfcDataset
from src.multimodal.evaluation import clustering_metrics
from src.prior_models.staig.data import build_spatial_graph
from src.prior_models.staig.evaluate import refine_labels


@dataclass(frozen=True)
class AdjacencyClusteringConfig:
    checkpoint_path: Path = Path("checkpoints/dlpfc.pkl")
    output_dir: Path = Path("outputs/evaluation/20260815_adjacency_clustering_seed0_v1")
    coordinate_neighbors: int = 6
    refinement_neighbors: int = 15
    image_key: str = "img_emb"
    seed: int = 0


def edge_cosine_distance_squared(
    features: np.ndarray | sparse.spmatrix, edge_index: np.ndarray
) -> np.ndarray:
    """Compute squared distance between L2-normalized rows only on listed edges."""
    src, dst = np.asarray(edge_index, dtype=np.int64)
    if sparse.issparse(features):
        values = sparse.csr_matrix(features, dtype=np.float64)
        norms = np.sqrt(np.asarray(values.multiply(values).sum(axis=1)).ravel())
        dots = np.asarray(values[src].multiply(values[dst]).sum(axis=1)).ravel()
    else:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("features must be a 2-D matrix")
        norms = np.linalg.norm(values, axis=1)
        dots = np.einsum("ij,ij->i", values[src], values[dst])
    denominator = norms[src] * norms[dst]
    cosine = np.divide(
        dots,
        denominator,
        out=np.zeros_like(dots, dtype=np.float64),
        where=denominator > 0,
    )
    return np.maximum(2.0 - 2.0 * np.clip(cosine, -1.0, 1.0), 0.0)


def median_kernel_affinity(distance_squared: np.ndarray) -> tuple[np.ndarray, float]:
    """Convert edge distances to positive affinities with a label-free bandwidth."""
    distance_squared = np.asarray(distance_squared, dtype=np.float64)
    positive = distance_squared[distance_squared > 0]
    bandwidth_squared = float(np.median(positive)) if len(positive) else 1.0
    bandwidth_squared = max(bandwidth_squared, np.finfo(np.float64).eps)
    weights = np.exp(-distance_squared / (2.0 * bandwidth_squared))
    return weights.astype(np.float64), bandwidth_squared


def sparse_adjacency(
    n_nodes: int, edge_index: np.ndarray, weights: np.ndarray
) -> sparse.csr_matrix:
    """Build a finite symmetric CSR adjacency from a directed edge list."""
    edge_index = np.asarray(edge_index, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, n_edges)")
    if weights.shape != (edge_index.shape[1],):
        raise ValueError("weights must match the edge count")
    if np.any(weights <= 0) or not np.isfinite(weights).all():
        raise ValueError("adjacency weights must be finite and positive")
    adjacency = sparse.coo_matrix(
        (weights, (edge_index[0], edge_index[1])), shape=(n_nodes, n_nodes)
    ).tocsr()
    adjacency = adjacency.maximum(adjacency.T)
    adjacency.setdiag(0.0)
    adjacency.eliminate_zeros()
    return adjacency


def cluster_adjacency(
    adjacency: sparse.csr_matrix, n_clusters: int, seed: int
) -> np.ndarray:
    """Apply normalized spectral clustering to one precomputed sparse graph."""
    if adjacency.shape[0] <= n_clusters:
        raise ValueError("adjacency needs more nodes than clusters")
    return SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        assign_labels="kmeans",
        n_init=10,
        random_state=seed,
        eigen_solver="arpack",
    ).fit_predict(adjacency)


def _summary(records: list[dict[str, object]]) -> dict[str, dict[str, dict[str, float]]]:
    summary = {}
    for condition in sorted({str(record["condition"]) for record in records}):
        rows = [record for record in records if record["condition"] == condition]
        summary[condition] = {}
        for metric in ("ari", "nmi", "accuracy", "refined_ari", "refined_nmi", "refined_accuracy"):
            values = np.asarray([float(row[metric]) for row in rows])
            summary[condition][metric] = {
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
            }
    return summary


def run_adjacency_clustering(config: AdjacencyClusteringConfig) -> dict:
    """Run and persist the three adjacency-only clustering conditions."""
    if config.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {config.output_dir}")
    adjacency_dir = config.output_dir / "adjacency"
    assignment_dir = config.output_dir / "assignments"
    adjacency_dir.mkdir(parents=True)
    assignment_dir.mkdir()

    dataset = DlpfcDataset.from_checkpoint(config.checkpoint_path)
    records: list[dict[str, object]] = []
    for section_id in dataset.section_ids():
        adata = dataset.get_section(section_id)
        coordinates = np.asarray(adata.obsm["spatial"], dtype=np.float32)
        labels = adata.obs["ground_truth"].astype(str).to_numpy()
        barcodes = np.asarray(adata.obs_names.astype(str))
        edge_index = build_spatial_graph(coordinates, config.coordinate_neighbors)
        n_clusters = int(np.unique(labels).size)

        gene_distance = edge_cosine_distance_squared(adata.X, edge_index)
        gene_weights, gene_bandwidth = median_kernel_affinity(gene_distance)
        if config.image_key not in adata.obsm:
            raise KeyError(f"Section {section_id} lacks {config.image_key}")
        image_distance = edge_cosine_distance_squared(
            np.asarray(adata.obsm[config.image_key], dtype=np.float32), edge_index
        )
        image_weights, image_bandwidth = median_kernel_affinity(image_distance)
        conditions = {
            "spatial_binary": (np.ones(edge_index.shape[1]), None),
            "spatial_gene_all33538": (gene_weights, gene_bandwidth),
            "spatial_image128": (image_weights, image_bandwidth),
        }
        for condition, (weights, bandwidth_squared) in conditions.items():
            adjacency = sparse_adjacency(adata.n_obs, edge_index, weights)
            predicted = cluster_adjacency(adjacency, n_clusters, config.seed)
            refined = refine_labels(
                predicted, coordinates, config.refinement_neighbors
            )
            raw_metrics = clustering_metrics(labels, predicted)
            refined_metrics = clustering_metrics(labels, refined)
            record = {
                "section_id": str(section_id),
                "condition": condition,
                "n_spots": int(adata.n_obs),
                "n_clusters": n_clusters,
                "n_undirected_edges": int(adjacency.nnz // 2),
                "bandwidth_squared": bandwidth_squared,
                "seed": config.seed,
                **raw_metrics,
                **{f"refined_{key}": value for key, value in refined_metrics.items()},
            }
            records.append(record)
            sparse.save_npz(adjacency_dir / f"{section_id}_{condition}.npz", adjacency)
            np.savez_compressed(
                assignment_dir / f"{section_id}_{condition}.npz",
                cluster_ids=predicted.astype(np.int64),
                refined_cluster_ids=refined.astype(np.int64),
                labels=labels,
                barcodes=barcodes,
                section_id=np.asarray(str(section_id)),
                condition=np.asarray(condition),
            )
            print(json.dumps(record, sort_keys=True), flush=True)

    summary = {
        "run_id": config.output_dir.name,
        "config": {
            **{
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(config).items()
            },
            "node_input": "none; clustering operates on adjacency",
            "gene_source": "adata.X, all 33,538 log1p-normalized genes",
            "image_source": f"adata.obsm[{config.image_key!r}]",
            "affinity": "exp(-cosine_distance_squared / (2 * median_positive_edge_distance_squared))",
            "backend": "sklearn SpectralClustering(precomputed, normalized Laplacian, kmeans)",
            "cluster_count": "exact observed valid annotation count; labels used only after clustering",
        },
        "metrics": _summary(records),
    }
    with (config.output_dir / "section_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (config.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (config.output_dir / "config.json").write_text(
        json.dumps(summary["config"], indent=2), encoding="utf-8"
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    defaults = AdjacencyClusteringConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, default=defaults.checkpoint_path)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--coordinate-neighbors", type=int, default=defaults.coordinate_neighbors)
    parser.add_argument("--refinement-neighbors", type=int, default=defaults.refinement_neighbors)
    parser.add_argument("--image-key", default=defaults.image_key)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    return parser


def main() -> None:
    config = AdjacencyClusteringConfig(**vars(build_arg_parser().parse_args()))
    print(json.dumps(run_adjacency_clustering(config)["metrics"], indent=2))


if __name__ == "__main__":
    main()
