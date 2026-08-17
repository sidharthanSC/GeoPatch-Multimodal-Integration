"""Closed-form, collapse-free graph-positive cross-modal alignment."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from src.cross_modal.neighborhood import (
    NeighborhoodPruningConfig,
    build_split_neighborhood_graph,
)
from src.cross_modal.train import (
    load_embeddings,
    load_spatial_coordinates,
    three_way_split,
)
from src.datasets.dlpfc import DlpfcDataset
from src.multimodal.representations import l2_normalize


@dataclass
class GraphProcrustesConfig:
    checkpoint_path: Path = Path("checkpoints/dlpfc_gbssa_seed0.pkl")
    output_dir: Path = Path("outputs/cross_modal")
    run_name: str = "graph_positive_procrustes_img_k6r5_seed0"
    gene_key: str = "gene_emb"
    gene_npz: Path | None = None
    image_key: str = "img_emb"
    coordinate_neighbors: int = 6
    retained_neighbors: int = 5
    self_weight: float = 1.0
    neighbor_weight: float = 1.0
    pooling_weight: float = 0.0
    transductive_full_graph: bool = False
    val_fraction: float = 0.2
    seed: int = 0


def fit_graph_positive_procrustes(
    gene: np.ndarray,
    image: np.ndarray,
    train_indices: np.ndarray,
    neighbor_edges: np.ndarray,
    self_weight: float = 1.0,
    neighbor_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit two orthogonal maps from exact and undirected neighborhood positives.

    If ``gene @ U`` and ``image @ V`` are the aligned outputs, orthogonality preserves
    all within-modality dot products exactly while maximizing positive cross-modal
    agreement. No different-spot negative or dispersion term is used.
    """
    if self_weight < 0 or neighbor_weight < 0:
        raise ValueError("positive weights must be non-negative")
    gene = l2_normalize(gene)
    image = l2_normalize(image)
    if gene.shape != image.shape:
        raise ValueError("gene and image embeddings must have matching shapes")
    train_indices = np.asarray(train_indices, dtype=np.int64)
    edges = np.asarray(neighbor_edges, dtype=np.int64).reshape(-1, 2)
    cross_covariance = self_weight * gene[train_indices].T @ image[train_indices]
    if len(edges):
        source, target = edges.T
        cross_covariance += neighbor_weight * (
            gene[source].T @ image[target] + gene[target].T @ image[source]
        )
    left, singular_values, right_transpose = np.linalg.svd(
        cross_covariance, full_matrices=False
    )
    right = right_transpose.T
    return left.astype(np.float32), right.astype(np.float32), singular_values


def transform_procrustes(
    gene: np.ndarray, image: np.ndarray, gene_map: np.ndarray, image_map: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    return (
        l2_normalize(l2_normalize(gene) @ gene_map),
        l2_normalize(l2_normalize(image) @ image_map),
    )


def pool_spatial_neighbors(
    embeddings: np.ndarray,
    neighbors: tuple[np.ndarray, ...],
    node_indices: np.ndarray,
    pooling_weight: float,
) -> np.ndarray:
    """Average each normalized spot with its retained immediate neighbors."""
    if pooling_weight < 0:
        raise ValueError("pooling_weight must be non-negative")
    normalized = l2_normalize(embeddings)
    if pooling_weight == 0:
        return normalized
    pooled = normalized.copy()
    for row in np.asarray(node_indices, dtype=np.int64):
        adjacent = neighbors[int(row)]
        if len(adjacent):
            pooled[row] = normalized[row] + pooling_weight * normalized[adjacent].mean(axis=0)
    return l2_normalize(pooled)


def run_graph_positive_procrustes(config: GraphProcrustesConfig) -> Path:
    predictions_path = (
        config.output_dir / "predictions" / f"{config.run_name}_embeddings.npz"
    )
    model_dir = config.output_dir / "checkpoints" / config.run_name
    model_path = model_dir / "procrustes.npz"
    config_path = model_dir / "config.json"
    graph_dir = config.output_dir / "graphs" / config.run_name
    graph_path = graph_dir / "edges.npz"
    diagnostics_path = graph_dir / "diagnostics.json"
    existing = [
        path
        for path in (predictions_path, model_path, config_path, graph_path, diagnostics_path)
        if path.exists()
    ]
    if existing:
        raise FileExistsError("Refusing to overwrite: " + ", ".join(map(str, existing)))

    dataset = DlpfcDataset.from_checkpoint(config.checkpoint_path)
    gene, image, barcodes, section_ids = load_embeddings(
        dataset, config.gene_key, config.image_key
    )
    if config.gene_npz is not None:
        with np.load(config.gene_npz, allow_pickle=True) as gene_data:
            observed_ids = list(
                zip(
                    gene_data["section_ids"].astype(str),
                    gene_data["barcodes"].astype(str),
                )
            )
            expected_ids = list(zip(section_ids.astype(str), barcodes.astype(str)))
            if observed_ids != expected_ids:
                raise ValueError("gene NPZ row identities do not match the checkpoint")
            gene = np.asarray(gene_data["embeddings"], dtype=np.float32)
    coordinates = load_spatial_coordinates(dataset)
    pruning = NeighborhoodPruningConfig(
        coordinate_neighbors=config.coordinate_neighbors,
        retained_neighbors=config.retained_neighbors,
    )
    if config.transductive_full_graph:
        split = np.full(len(section_ids), "transductive", dtype=object)
        graphs = {
            "transductive": build_split_neighborhood_graph(
                coordinates, section_ids, split, "transductive", gene, image, pruning
            )
        }
        fit_indices = np.arange(len(section_ids), dtype=np.int64)
        fit_graph = graphs["transductive"]
    else:
        train_idx, val_idx, _ = three_way_split(
            dataset,
            section_ids,
            config.val_fraction,
            "none",
            [],
            [],
            config.seed,
        )
        split = np.full(len(section_ids), "train", dtype=object)
        split[val_idx] = "val"
        graphs = {
            name: build_split_neighborhood_graph(
                coordinates, section_ids, split, name, gene, image, pruning
            )
            for name in ("train", "val")
        }
        fit_indices = train_idx
        fit_graph = graphs["train"]

    pooled_gene = pool_spatial_neighbors(
        gene, fit_graph.neighbors, fit_graph.node_indices, config.pooling_weight
    )
    pooled_image = pool_spatial_neighbors(
        image, fit_graph.neighbors, fit_graph.node_indices, config.pooling_weight
    )
    gene_map, image_map, singular_values = fit_graph_positive_procrustes(
        pooled_gene,
        pooled_image,
        fit_indices,
        fit_graph.edges,
        config.self_weight,
        config.neighbor_weight,
    )
    gene_projected, image_projected = transform_procrustes(
        pooled_gene, pooled_image, gene_map, image_map
    )

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        predictions_path,
        gene_projected=gene_projected,
        image_projected=image_projected,
        barcodes=barcodes,
        section_ids=section_ids,
        split=split,
        image_key=config.image_key,
        objective="graph_positive_procrustes",
    )
    np.savez_compressed(
        model_path,
        gene_map=gene_map,
        image_map=image_map,
        singular_values=singular_values,
    )
    np.savez_compressed(
        graph_path,
        **{f"{name}_candidate_edges": graph.candidate_edges for name, graph in graphs.items()},
        **{f"{name}_retained_edges": graph.edges for name, graph in graphs.items()},
    )
    with config_path.open("w") as file:
        serialized_config = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        }
        json.dump(
            serialized_config,
            file,
            indent=2,
        )
    with diagnostics_path.open("w") as file:
        json.dump({name: graph.diagnostics for name, graph in graphs.items()}, file, indent=2)
    return predictions_path


def build_arg_parser() -> argparse.ArgumentParser:
    defaults = GraphProcrustesConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, default=defaults.checkpoint_path)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--run-name", default=defaults.run_name)
    parser.add_argument("--gene-key", default=defaults.gene_key)
    parser.add_argument("--gene-npz", type=Path, default=defaults.gene_npz)
    parser.add_argument("--image-key", choices=["img_emb", "proj_emb"], default=defaults.image_key)
    parser.add_argument("--coordinate-neighbors", type=int, default=defaults.coordinate_neighbors)
    parser.add_argument("--retained-neighbors", type=int, default=defaults.retained_neighbors)
    parser.add_argument("--self-weight", type=float, default=defaults.self_weight)
    parser.add_argument("--neighbor-weight", type=float, default=defaults.neighbor_weight)
    parser.add_argument("--pooling-weight", type=float, default=defaults.pooling_weight)
    parser.add_argument("--transductive-full-graph", action="store_true")
    parser.add_argument("--val-fraction", type=float, default=defaults.val_fraction)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    return parser


def main() -> None:
    run_graph_positive_procrustes(
        GraphProcrustesConfig(**vars(build_arg_parser().parse_args()))
    )


if __name__ == "__main__":
    main()
