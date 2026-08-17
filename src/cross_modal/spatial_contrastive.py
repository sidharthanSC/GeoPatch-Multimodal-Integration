"""Unified spatial multi-positive contrastive learning over frozen modality features."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.cross_modal.model import CrossModalConfig, CrossModalModel
from src.cross_modal.neighborhood import (
    NeighborhoodPruningConfig,
    SplitNeighborhoodGraph,
    build_split_neighborhood_graph,
)
from src.cross_modal.train import load_embeddings, load_spatial_coordinates
from src.datasets.dlpfc import DlpfcDataset


@dataclass
class SpatialContrastiveConfig:
    checkpoint_path: Path = Path("checkpoints/dlpfc_gbssa_seed0.pkl")
    output_dir: Path = Path("outputs/cross_modal")
    run_name: str = "spatial_multimodal_contrastive_img_k6r5_seed0"
    gene_key: str = "gene_emb"
    image_key: str = "img_emb"
    coordinate_neighbors: int = 6
    retained_neighbors: int = 5
    hidden_dim: int = 256
    output_dim: int = 128
    dropout: float = 0.1
    cross_modal_weight: float = 1.0
    gene_neighbor_weight: float = 1.0
    image_neighbor_weight: float = 1.0
    batch_size: int = 256
    epochs: int = 50
    lr: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 0
    device: str = "cuda"


@dataclass(frozen=True)
class SpatialContrastiveLoss:
    total: torch.Tensor
    gene_to_image: torch.Tensor
    image_to_gene: torch.Tensor
    gene_neighbor: torch.Tensor
    image_neighbor: torch.Tensor


def multi_positive_info_nce(
    logits: torch.Tensor, positives: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """InfoNCE with one or more positives and explicit false-negative masking."""
    if logits.shape != positives.shape or logits.shape != valid.shape:
        raise ValueError("logits, positives, and valid masks must have matching shapes")
    valid = valid | positives
    usable = positives.any(dim=1) & valid.any(dim=1)
    if not usable.any():
        return logits.new_zeros(())
    negative_infinity = torch.finfo(logits.dtype).min
    numerator = torch.logsumexp(logits.masked_fill(~positives, negative_infinity), dim=1)
    denominator = torch.logsumexp(logits.masked_fill(~valid, negative_infinity), dim=1)
    return (denominator[usable] - numerator[usable]).mean()


def spatial_multimodal_contrastive_loss(
    gene_queries: torch.Tensor,
    image_queries: torch.Tensor,
    gene_gallery: torch.Tensor,
    image_gallery: torch.Tensor,
    cross_positives: torch.Tensor,
    neighbor_positives: torch.Tensor,
    cross_valid: torch.Tensor,
    intra_valid: torch.Tensor,
    logit_scale: torch.Tensor,
    cross_modal_weight: float = 1.0,
    gene_neighbor_weight: float = 1.0,
    image_neighbor_weight: float = 1.0,
) -> SpatialContrastiveLoss:
    scale = logit_scale.exp()
    gene_image = scale * gene_queries @ image_gallery.T
    image_gene = scale * image_queries @ gene_gallery.T
    gene_gene = scale * gene_queries @ gene_gallery.T
    image_image = scale * image_queries @ image_gallery.T
    gene_to_image = multi_positive_info_nce(gene_image, cross_positives, cross_valid)
    image_to_gene = multi_positive_info_nce(image_gene, cross_positives, cross_valid)
    gene_neighbor = multi_positive_info_nce(gene_gene, neighbor_positives, intra_valid)
    image_neighbor = multi_positive_info_nce(image_image, neighbor_positives, intra_valid)
    total = (
        cross_modal_weight * (gene_to_image + image_to_gene) / 2.0
        + gene_neighbor_weight * gene_neighbor
        + image_neighbor_weight * image_neighbor
    )
    return SpatialContrastiveLoss(
        total=total,
        gene_to_image=gene_to_image.detach(),
        image_to_gene=image_to_gene.detach(),
        gene_neighbor=gene_neighbor.detach(),
        image_neighbor=image_neighbor.detach(),
    )


def build_graph_batch(
    anchors: np.ndarray,
    retained_neighbors: tuple[np.ndarray, ...],
    candidate_neighbors: tuple[np.ndarray, ...],
    section_ids: np.ndarray,
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather anchors/neighbors and construct positive and valid-gallery masks."""
    anchors = np.asarray(anchors, dtype=np.int64)
    gathered = anchors.tolist()
    positions = {row: position for position, row in enumerate(gathered)}
    for source in anchors:
        for target in retained_neighbors[int(source)]:
            target = int(target)
            if target not in positions:
                positions[target] = len(gathered)
                gathered.append(target)
    gathered_array = np.asarray(gathered, dtype=np.int64)
    batch_size = len(anchors)
    gallery_size = len(gathered_array)
    cross_positives = np.zeros((batch_size, gallery_size), dtype=bool)
    neighbor_positives = np.zeros_like(cross_positives)
    same_section = section_ids[anchors, None] == section_ids[gathered_array][None, :]
    cross_valid = same_section.copy()
    intra_valid = same_section.copy()
    for row, source in enumerate(anchors):
        cross_positives[row, positions[int(source)]] = True
        intra_valid[row, positions[int(source)]] = False
        retained = set(map(int, retained_neighbors[int(source)]))
        candidates = set(map(int, candidate_neighbors[int(source)]))
        for target in retained:
            if target in positions:
                position = positions[target]
                cross_positives[row, position] = True
                neighbor_positives[row, position] = True
        for target in candidates - retained:
            if target in positions:
                position = positions[target]
                cross_valid[row, position] = False
                intra_valid[row, position] = False
    return (
        gathered_array,
        torch.from_numpy(cross_positives),
        torch.from_numpy(neighbor_positives),
        torch.from_numpy(cross_valid),
        torch.from_numpy(intra_valid),
    )


def edge_neighbors(n_spots: int, edges: np.ndarray) -> tuple[np.ndarray, ...]:
    neighbors: list[list[int]] = [[] for _ in range(n_spots)]
    for source, target in np.asarray(edges, dtype=np.int64).reshape(-1, 2):
        neighbors[int(source)].append(int(target))
        neighbors[int(target)].append(int(source))
    return tuple(np.asarray(sorted(rows), dtype=np.int64) for rows in neighbors)


@torch.no_grad()
def export_embeddings(
    model: CrossModalModel,
    gene: torch.Tensor,
    image: torch.Tensor,
    batch_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    gene_output, image_output = [], []
    for start in range(0, len(gene), batch_size):
        gene_batch, image_batch = model(
            gene[start : start + batch_size], image[start : start + batch_size]
        )
        gene_output.append(gene_batch.cpu().numpy())
        image_output.append(image_batch.cpu().numpy())
    return np.concatenate(gene_output), np.concatenate(image_output)


def run_spatial_contrastive(config: SpatialContrastiveConfig) -> Path:
    checkpoint_dir = config.output_dir / "checkpoints" / config.run_name
    metrics_path = config.output_dir / "metrics" / f"{config.run_name}_metrics.csv"
    predictions_path = config.output_dir / "predictions" / f"{config.run_name}_embeddings.npz"
    graph_dir = config.output_dir / "graphs" / config.run_name
    expected = [checkpoint_dir / "last.pt", metrics_path, predictions_path, graph_dir / "edges.npz"]
    if any(path.exists() for path in expected):
        raise FileExistsError("Refusing to overwrite an existing run")

    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    device = torch.device(config.device)
    dataset = DlpfcDataset.from_checkpoint(config.checkpoint_path)
    gene_np, image_np, barcodes, section_ids = load_embeddings(
        dataset, config.gene_key, config.image_key
    )
    coordinates = load_spatial_coordinates(dataset)
    split = np.full(len(section_ids), "transductive", dtype=object)
    graph = build_split_neighborhood_graph(
        coordinates,
        section_ids,
        split,
        "transductive",
        gene_np,
        image_np,
        NeighborhoodPruningConfig(
            coordinate_neighbors=config.coordinate_neighbors,
            retained_neighbors=config.retained_neighbors,
        ),
    )
    candidate_neighbors = edge_neighbors(len(section_ids), graph.candidate_edges)
    gene = torch.as_tensor(gene_np, dtype=torch.float32, device=device)
    image = torch.as_tensor(image_np, dtype=torch.float32, device=device)
    model = CrossModalModel(
        CrossModalConfig(
            gene_input_dim=gene.shape[1],
            image_input_dim=image.shape[1],
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim,
            dropout=config.dropout,
        )
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)
    with (graph_dir / "diagnostics.json").open("w") as file:
        json.dump(graph.diagnostics, file, indent=2)
    np.savez_compressed(
        graph_dir / "edges.npz",
        candidate_edges=graph.candidate_edges,
        retained_edges=graph.edges,
    )
    with metrics_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["epoch", "loss", "gene_to_image", "image_to_gene", "gene_neighbor", "image_neighbor", "temperature"],
        )
        writer.writeheader()
        for epoch in range(1, config.epochs + 1):
            order = rng.permutation(len(gene_np))
            totals = {name: 0.0 for name in writer.fieldnames if name != "epoch"}
            batches = 0
            model.train()
            for start in range(0, len(order), config.batch_size):
                anchors = order[start : start + config.batch_size]
                if len(anchors) < 2:
                    continue
                gathered, cross_pos, neighbor_pos, cross_valid, intra_valid = build_graph_batch(
                    anchors, graph.neighbors, candidate_neighbors, section_ids
                )
                gene_gallery, image_gallery = model(gene[gathered], image[gathered])
                anchor_positions = torch.arange(len(anchors), device=device)
                output = spatial_multimodal_contrastive_loss(
                    gene_gallery[anchor_positions],
                    image_gallery[anchor_positions],
                    gene_gallery,
                    image_gallery,
                    cross_pos.to(device),
                    neighbor_pos.to(device),
                    cross_valid.to(device),
                    intra_valid.to(device),
                    model.logit_scale,
                    config.cross_modal_weight,
                    config.gene_neighbor_weight,
                    config.image_neighbor_weight,
                )
                optimizer.zero_grad(set_to_none=True)
                output.total.backward()
                optimizer.step()
                model.clamp_logit_scale()
                for name in ("gene_to_image", "image_to_gene", "gene_neighbor", "image_neighbor"):
                    totals[name] += float(getattr(output, name).item())
                totals["loss"] += float(output.total.item())
                totals["temperature"] += float(model.logit_scale.detach().exp().reciprocal().item())
                batches += 1
            row = {"epoch": epoch, **{name: value / batches for name, value in totals.items()}}
            writer.writerow(row)
            file.flush()
            print(json.dumps(row), flush=True)
            torch.save(
                {"model": model.state_dict(), "config": asdict(config), "epoch": epoch, "metrics": row},
                checkpoint_dir / "last.pt",
            )

    gene_projected, image_projected = export_embeddings(model, gene, image)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        predictions_path,
        gene_projected=gene_projected,
        image_projected=image_projected,
        barcodes=barcodes,
        section_ids=section_ids,
        split=split,
        image_key=config.image_key,
        objective="spatial_multimodal_contrastive",
    )
    return predictions_path


def build_arg_parser() -> argparse.ArgumentParser:
    defaults = SpatialContrastiveConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint_path", "output_dir"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, default=getattr(defaults, name))
    parser.add_argument("--run-name", default=defaults.run_name)
    parser.add_argument("--coordinate-neighbors", type=int, default=defaults.coordinate_neighbors)
    parser.add_argument("--retained-neighbors", type=int, default=defaults.retained_neighbors)
    parser.add_argument("--hidden-dim", type=int, default=defaults.hidden_dim)
    parser.add_argument("--output-dim", type=int, default=defaults.output_dim)
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument("--cross-modal-weight", type=float, default=defaults.cross_modal_weight)
    parser.add_argument("--gene-neighbor-weight", type=float, default=defaults.gene_neighbor_weight)
    parser.add_argument("--image-neighbor-weight", type=float, default=defaults.image_neighbor_weight)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--device", default=defaults.device)
    return parser


def main() -> None:
    run_spatial_contrastive(SpatialContrastiveConfig(**vars(build_arg_parser().parse_args())))


if __name__ == "__main__":
    main()
