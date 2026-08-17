"""Cross-modal alignment between ``gene_emb`` and an image-side embedding.

Run once per image-side source (``--image-key img_emb`` or ``--image-key proj_emb``) --
these are two independent trainings, not one multi-task run, since ``img_emb`` (raw,
frozen BYOL image embedding) and ``proj_emb`` (after the layer-aware projection network)
represent different information and each gets its own pair of cross-modal projection
heads. Both this script's projection heads are trainable (see
``src/cross_modal/model.py``), so each run produces **two** new embeddings -- a
cross-modal gene embedding and a cross-modal image embedding -- not just one.

Three-way spot split -- train / validation / a held-out test set:

- **train** / **val**: as in ``src/gene_encoder/train.py`` -- val drives early stopping,
  never seen by the optimizer.
- **test**: ``--test-unit {none,section,donor}`` (default ``none``, i.e. no test set is
  carved out -- every spot lands in train/val). Unlike the gene encoder's random
  ``test_fraction``, holdout here is by *whole group* -- entire sections
  (``--test-sections 151669 151670``) or entire donors (``--test-donors 2``, matching
  ``DlpfcDataset``'s donor grouping of 4 sections each) -- since the point of a
  cross-modal test set is checking generalization to tissue the model never saw at all,
  not just unseen spots within seen tissue. Test spots never contribute to training or
  the validation loss; their embeddings are still exported via a plain forward pass
  through the final trained heads, same as everyone else's.

Usage
-----
    python -m src.cross_modal.train --image-key img_emb
    python -m src.cross_modal.train --image-key proj_emb
    python -m src.cross_modal.train --image-key img_emb --test-unit donor --test-donors 2

Outputs (under ``--output-dir``, default ``outputs/cross_modal``; ``run_name`` defaults
to ``cross_modal_<image_key>`` if not given):

- ``checkpoints/<run_name>/{best.pt, last.pt}``
- ``metrics/<run_name>_metrics.csv`` -- one row per epoch.
- ``predictions/<run_name>_embeddings.npz`` -- ``gene_projected``, ``image_projected``
  (both ``(n_spots, output_dim)``), ``barcodes``, ``section_ids``, ``split``
  (``"train"``/``"val"``/``"test"``), plus ``image_key`` (which source this run used).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src.datasets.dlpfc import DlpfcDataset
from src.cross_modal.model import (
    CrossModalConfig,
    CrossModalModel,
    gradient_balanced_clip_loss,
    positive_neighborhood_loss,
    retrieval_accuracy,
)
from src.cross_modal.neighborhood import (
    NeighborhoodPruningConfig,
    SplitNeighborhoodGraph,
    build_split_neighborhood_graph,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    checkpoint_path: Path = Path("checkpoints/dlpfc.pkl")
    output_dir: Path = Path("outputs/cross_modal")
    run_name: str = ""

    gene_key: str = "gene_emb"
    gene_npz_path: Path | None = None
    gene_array_key: str = "embeddings"
    image_key: str = "img_emb"
    objective: str = "positive_neighborhood"

    hidden_dim: int = 256
    output_dim: int = 128
    dropout: float = 0.1
    same_spot_gradient_ratio: float = 1.0
    coordinate_neighbors: int = 6
    retained_neighbors: int = 3
    min_gene_cosine: float | None = None
    min_image_cosine: float | None = None
    alignment_weight: float = 25.0
    neighbor_weight: float = 1.0
    variance_weight: float = 25.0
    variance_gamma: float = 1.0
    covariance_weight: float = 1.0

    batch_size: int = 512
    num_epochs: int = 200
    lr: float = 3e-4
    weight_decay: float = 1e-4

    val_fraction: float = 0.2
    test_unit: str = "none"
    test_sections: List[str] = field(default_factory=list)
    test_donors: List[int] = field(default_factory=list)

    seed: int = 0
    patience: int = 20
    min_delta: float = 1e-4

    device: str = field(default_factory=lambda: default_device())
    log_every: int = 20


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_embeddings(
    dataset: DlpfcDataset, gene_key: str, image_key: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Every spot's gene and image embeddings, concatenated in ``dataset.section_ids()``
    order. Gene/image rows are aligned by construction (both obsm keys live on the same
    AnnData, same row order) -- no barcode reconciliation needed."""
    gene_blocks, image_blocks, barcode_blocks, section_id_blocks = [], [], [], []

    for section_id in dataset.section_ids():
        adata = dataset.get_section(section_id)
        gene_blocks.append(np.asarray(adata.obsm[gene_key], dtype=np.float32))
        image_blocks.append(np.asarray(adata.obsm[image_key], dtype=np.float32))
        barcode_blocks.append(adata.obs_names.to_numpy())
        section_id_blocks.append(np.full(adata.n_obs, str(section_id), dtype=object))

    return (
        np.concatenate(gene_blocks, axis=0),
        np.concatenate(image_blocks, axis=0),
        np.concatenate(barcode_blocks, axis=0),
        np.concatenate(section_id_blocks, axis=0),
    )


def load_external_gene_embeddings(
    dataset: DlpfcDataset,
    gene_npz_path: Path,
    gene_array_key: str,
    image_key: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a gene array from NPZ and reconcile it by ``(section_id, barcode)``."""
    with np.load(gene_npz_path, allow_pickle=True) as artifact:
        required = {gene_array_key, "barcodes", "section_ids"}
        missing = required.difference(artifact.files)
        if missing:
            raise KeyError(f"Missing external gene artifact arrays: {sorted(missing)}")
        source_gene = np.asarray(artifact[gene_array_key], dtype=np.float32)
        source_barcodes = np.asarray(artifact["barcodes"]).astype(str)
        source_sections = np.asarray(artifact["section_ids"]).astype(str)

    if source_gene.ndim != 2:
        raise ValueError(f"{gene_array_key} must be 2-D, got {source_gene.shape}")
    if not np.isfinite(source_gene).all():
        raise ValueError(f"{gene_array_key} contains non-finite values")
    if not (len(source_gene) == len(source_barcodes) == len(source_sections)):
        raise ValueError("External gene arrays have inconsistent row counts")

    source_identities = list(zip(source_sections.tolist(), source_barcodes.tolist()))
    if len(set(source_identities)) != len(source_identities):
        raise ValueError("External gene artifact contains duplicate (section_id, barcode) identities")
    source_lookup = {identity: index for index, identity in enumerate(source_identities)}

    image_blocks, barcode_blocks, section_blocks, source_indices = [], [], [], []
    expected_identities = []
    for section_id in dataset.section_ids():
        adata = dataset.get_section(section_id)
        barcodes = np.asarray(adata.obs_names).astype(str)
        identities = [(str(section_id), barcode) for barcode in barcodes]
        expected_identities.extend(identities)
        image_blocks.append(np.asarray(adata.obsm[image_key], dtype=np.float32))
        barcode_blocks.append(barcodes)
        section_blocks.append(np.full(adata.n_obs, str(section_id), dtype=object))
        for identity in identities:
            if identity not in source_lookup:
                raise ValueError(f"External gene artifact is missing spot {identity}")
            source_indices.append(source_lookup[identity])

    extras = set(source_lookup).difference(expected_identities)
    if extras:
        raise ValueError(f"External gene artifact contains {len(extras)} unexpected spots")
    return (
        source_gene[np.asarray(source_indices)],
        np.concatenate(image_blocks),
        np.concatenate(barcode_blocks),
        np.concatenate(section_blocks),
    )


def load_spatial_coordinates(dataset: DlpfcDataset) -> np.ndarray:
    """Spot coordinates in the exact row order returned by :func:`load_embeddings`."""
    return np.concatenate(
        [
            np.asarray(dataset.get_section(section_id).obsm["spatial"], dtype=np.float32)
            for section_id in dataset.section_ids()
        ],
        axis=0,
    )


def three_way_split(
    dataset: DlpfcDataset,
    section_ids: np.ndarray,
    val_fraction: float,
    test_unit: str,
    test_sections: List[str],
    test_donors: List[int],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split spot indices into (train, val, test).

    ``test_unit="none"`` (the default): no test set, every spot in train/val.
    ``test_unit="section"``: every spot in ``test_sections`` is held out entirely.
    ``test_unit="donor"``: every spot in a section belonging to any of ``test_donors``
    is held out entirely (via ``dataset.donor_id``).
    """
    indices = np.arange(len(section_ids))

    if test_unit == "none":
        test_idx = np.array([], dtype=np.int64)
        train_val_idx = indices
    elif test_unit == "section":
        if not test_sections:
            raise ValueError("--test-unit section requires --test-sections <id> [<id> ...]")
        test_mask = np.isin(section_ids, test_sections)
        test_idx, train_val_idx = indices[test_mask], indices[~test_mask]
    elif test_unit == "donor":
        if not test_donors:
            raise ValueError("--test-unit donor requires --test-donors <id> [<id> ...]")
        donor_of_section = {sid: dataset.donor_id(sid) for sid in dataset.section_ids()}
        spot_donor = np.array([donor_of_section[s] for s in section_ids])
        test_mask = np.isin(spot_donor, test_donors)
        test_idx, train_val_idx = indices[test_mask], indices[~test_mask]
    else:
        raise ValueError(f"Unknown --test-unit {test_unit!r} (expected none/section/donor)")

    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_fraction, random_state=seed,
        stratify=section_ids[train_val_idx],
    )
    return train_idx, val_idx, test_idx


def run_epoch(
    model: CrossModalModel,
    gene_emb: torch.Tensor,
    image_emb: torch.Tensor,
    indices: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    train: bool,
    optimizer: torch.optim.Optimizer = None,
    log_every: int = 0,
    epoch: int = 0,
    num_epochs: int = 0,
    same_spot_gradient_ratio: float = 1.0,
) -> Dict[str, float]:
    """One pass over ``indices`` (shuffled if ``train``). Batches smaller than 2 are
    skipped -- InfoNCE's in-batch negatives need more than one sample."""
    model.train() if train else model.eval()

    order = indices.copy()
    if train:
        rng.shuffle(order)

    totals = {
        "loss": 0.0,
        "clip_loss": 0.0,
        "attraction_loss": 0.0,
        "weighted_attraction_loss": 0.0,
        "paired_cosine": 0.0,
        "hardest_negative_cosine": 0.0,
        "positive_hardest_margin": 0.0,
        "temperature": 0.0,
        "retrieval_accuracy": 0.0,
    }
    n_batches = 0

    for start in range(0, len(order), batch_size):
        batch_idx = order[start : start + batch_size]
        if len(batch_idx) < 2:
            continue

        with torch.set_grad_enabled(train):
            gene_proj, image_proj = model(gene_emb[batch_idx], image_emb[batch_idx])
            loss_output = gradient_balanced_clip_loss(
                gene_proj,
                image_proj,
                model.logit_scale,
                same_spot_gradient_ratio,
            )
            loss = loss_output.total
            accuracy = retrieval_accuracy(gene_proj, image_proj)

            similarity = gene_proj @ image_proj.T
            positive = similarity.diagonal()
            negative_mask = ~torch.eye(
                similarity.shape[0], dtype=torch.bool, device=similarity.device
            )
            hardest_gene = similarity.masked_fill(~negative_mask, -torch.inf).max(dim=1).values
            hardest_image = similarity.masked_fill(~negative_mask, -torch.inf).max(dim=0).values
            hardest_negative = (hardest_gene.mean() + hardest_image.mean()) / 2.0

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            model.clamp_logit_scale()

        totals["loss"] += float(loss.item())
        totals["clip_loss"] += float(loss_output.clip.item())
        totals["attraction_loss"] += float(loss_output.attraction.item())
        totals["weighted_attraction_loss"] += float(loss_output.weighted_attraction.item())
        totals["paired_cosine"] += float(positive.mean().item())
        totals["hardest_negative_cosine"] += float(hardest_negative.item())
        totals["positive_hardest_margin"] += float((positive.mean() - hardest_negative).item())
        totals["temperature"] += float(model.logit_scale.detach().exp().reciprocal().item())
        totals["retrieval_accuracy"] += accuracy
        n_batches += 1
        if train and log_every and n_batches % log_every == 0:
            logger.info(
                "epoch %d/%d step %d | loss %.4f | retrieval_acc %.4f",
                epoch, num_epochs, n_batches, loss.item(), accuracy,
            )

    return {key: value / max(n_batches, 1) for key, value in totals.items()}


def make_anchor_batch(
    anchors: np.ndarray, neighbors: tuple[np.ndarray, ...]
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    """Gather unique anchors/neighbors and express positive edges in local positions."""
    gathered = [int(row) for row in anchors]
    local_position = {row: position for position, row in enumerate(gathered)}
    edge_sources: list[int] = []
    edge_targets: list[int] = []
    for source_position, source in enumerate(gathered.copy()):
        for target_value in neighbors[source]:
            target = int(target_value)
            if target not in local_position:
                local_position[target] = len(gathered)
                gathered.append(target)
            edge_sources.append(source_position)
            edge_targets.append(local_position[target])
    edges = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
    return (
        np.asarray(gathered, dtype=np.int64),
        torch.arange(len(anchors), dtype=torch.long),
        edges,
    )


def run_positive_neighborhood_epoch(
    model: CrossModalModel,
    gene_emb: torch.Tensor,
    image_emb: torch.Tensor,
    indices: np.ndarray,
    graph: SplitNeighborhoodGraph,
    batch_size: int,
    rng: np.random.Generator,
    train: bool,
    optimizer: torch.optim.Optimizer | None = None,
    alignment_weight: float = 25.0,
    neighbor_weight: float = 1.0,
    variance_weight: float = 25.0,
    variance_gamma: float = 1.0,
    covariance_weight: float = 1.0,
) -> Dict[str, float]:
    """Positive-only alignment over same spots and retained spatial neighbors."""
    model.train() if train else model.eval()
    order = indices.copy()
    if train:
        rng.shuffle(order)
    totals = {
        "loss": 0.0,
        "self_alignment_loss": 0.0,
        "neighbor_alignment_loss": 0.0,
        "gene_variance_loss": 0.0,
        "image_variance_loss": 0.0,
        "gene_covariance_loss": 0.0,
        "image_covariance_loss": 0.0,
        "self_positive_cosine": 0.0,
        "neighbor_positive_cosine": 0.0,
        "gene_std_mean": 0.0,
        "image_std_mean": 0.0,
    }
    n_batches = 0
    for start in range(0, len(order), batch_size):
        anchors = order[start : start + batch_size]
        if len(anchors) < 2:
            continue
        gathered, anchor_positions, neighbor_edges = make_anchor_batch(
            anchors, graph.neighbors
        )
        anchor_positions = anchor_positions.to(gene_emb.device)
        neighbor_edges = neighbor_edges.to(gene_emb.device)
        with torch.set_grad_enabled(train):
            gene_raw, image_raw = model.project_raw(
                gene_emb[gathered], image_emb[gathered]
            )
            output = positive_neighborhood_loss(
                gene_raw,
                image_raw,
                anchor_positions,
                neighbor_edges,
                alignment_weight=alignment_weight,
                neighbor_weight=neighbor_weight,
                variance_weight=variance_weight,
                variance_gamma=variance_gamma,
                covariance_weight=covariance_weight,
            )
            gene_normalized = torch.nn.functional.normalize(gene_raw, dim=1)
            image_normalized = torch.nn.functional.normalize(image_raw, dim=1)
            self_cosine = (
                gene_normalized[anchor_positions] * image_normalized[anchor_positions]
            ).sum(dim=1).mean()
            if neighbor_edges.shape[1]:
                source, target = neighbor_edges
                neighbor_cosine = (
                    (gene_normalized[source] * image_normalized[target]).sum(dim=1).mean()
                    + (image_normalized[source] * gene_normalized[target]).sum(dim=1).mean()
                ) / 2.0
            else:
                neighbor_cosine = gene_raw.new_zeros(())
        if train:
            if optimizer is None:
                raise ValueError("optimizer is required for a training epoch")
            optimizer.zero_grad(set_to_none=True)
            output.total.backward()
            optimizer.step()

        gene_anchor = gene_normalized[anchor_positions]
        image_anchor = image_normalized[anchor_positions]
        values = {
            "loss": output.total,
            "self_alignment_loss": output.self_alignment,
            "neighbor_alignment_loss": output.neighbor_alignment,
            "gene_variance_loss": output.gene_variance,
            "image_variance_loss": output.image_variance,
            "gene_covariance_loss": output.gene_covariance,
            "image_covariance_loss": output.image_covariance,
            "self_positive_cosine": self_cosine,
            "neighbor_positive_cosine": neighbor_cosine,
            "gene_std_mean": gene_anchor.std(dim=0).mean(),
            "image_std_mean": image_anchor.std(dim=0).mean(),
        }
        for name, value in values.items():
            totals[name] += float(value.detach().item())
        n_batches += 1
    return {key: value / max(n_batches, 1) for key, value in totals.items()}


@torch.no_grad()
def export_embeddings(
    model: CrossModalModel, gene_emb: torch.Tensor, image_emb: torch.Tensor, batch_size: int = 4096
) -> Tuple[np.ndarray, np.ndarray]:
    """Plain forward pass through the trained heads -- no gradient, every spot."""
    model.eval()
    gene_outputs, image_outputs = [], []
    for start in range(0, gene_emb.shape[0], batch_size):
        gene_proj, image_proj = model(gene_emb[start : start + batch_size], image_emb[start : start + batch_size])
        gene_outputs.append(gene_proj.cpu().numpy())
        image_outputs.append(image_proj.cpu().numpy())
    return np.concatenate(gene_outputs, axis=0), np.concatenate(image_outputs, axis=0)


def save_checkpoint(
    model: CrossModalModel, config: TrainConfig, epoch: int, metrics: Dict[str, dict], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "train_config": asdict(config), "epoch": epoch, "metrics": metrics}, path
    )


def build_arg_parser() -> argparse.ArgumentParser:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--checkpoint-path", type=Path, default=defaults.checkpoint_path)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--run-name", type=str, default=defaults.run_name)

    parser.add_argument("--gene-key", type=str, default=defaults.gene_key)
    parser.add_argument("--gene-npz-path", type=Path, default=defaults.gene_npz_path)
    parser.add_argument("--gene-array-key", type=str, default=defaults.gene_array_key)
    parser.add_argument(
        "--image-key", type=str, default=defaults.image_key, choices=["img_emb", "proj_emb"],
        help="Which image-side embedding to align gene_emb against.",
    )
    parser.add_argument(
        "--objective",
        choices=["positive_neighborhood", "clip"],
        default=defaults.objective,
        help="Positive-only graph alignment (default) or the retained CLIP/GBSSA baseline.",
    )

    parser.add_argument("--hidden-dim", type=int, default=defaults.hidden_dim)
    parser.add_argument("--output-dim", type=int, default=defaults.output_dim)
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument(
        "--same-spot-gradient-ratio",
        type=float,
        default=defaults.same_spot_gradient_ratio,
        help="GBSSA diagonal attraction/aggregate-repulsion score-gradient ratio; 1 is standard CLIP.",
    )
    parser.add_argument("--coordinate-neighbors", type=int, default=defaults.coordinate_neighbors)
    parser.add_argument("--retained-neighbors", type=int, default=defaults.retained_neighbors)
    parser.add_argument("--min-gene-cosine", type=float, default=defaults.min_gene_cosine)
    parser.add_argument("--min-image-cosine", type=float, default=defaults.min_image_cosine)
    parser.add_argument("--alignment-weight", type=float, default=defaults.alignment_weight)
    parser.add_argument("--neighbor-weight", type=float, default=defaults.neighbor_weight)
    parser.add_argument("--variance-weight", type=float, default=defaults.variance_weight)
    parser.add_argument("--variance-gamma", type=float, default=defaults.variance_gamma)
    parser.add_argument("--covariance-weight", type=float, default=defaults.covariance_weight)

    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--num-epochs", type=int, default=defaults.num_epochs)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)

    parser.add_argument("--val-fraction", type=float, default=defaults.val_fraction)
    parser.add_argument(
        "--test-unit", type=str, default=defaults.test_unit, choices=["none", "section", "donor"],
        help="Held out entirely from training/validation, by whole group (not a random "
             "per-spot fraction). Default 'none' -- use a later run to test generalization "
             "once the methodology is settled.",
    )
    parser.add_argument("--test-sections", type=str, nargs="+", default=list(defaults.test_sections))
    parser.add_argument("--test-donors", type=int, nargs="+", default=list(defaults.test_donors))

    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--patience", type=int, default=defaults.patience)
    parser.add_argument("--min-delta", type=float, default=defaults.min_delta)
    parser.add_argument("--device", type=str, default=defaults.device)
    parser.add_argument("--log-every", type=int, default=defaults.log_every)

    return parser


def train_cross_modal(config: TrainConfig) -> Path:
    """Train one configured alignment run and return its embedding artifact path."""
    if not config.run_name:
        config.run_name = f"{config.objective}_{config.image_key}"
    predictions_path = config.output_dir / "predictions" / f"{config.run_name}_embeddings.npz"
    checkpoint_dir = config.output_dir / "checkpoints" / config.run_name
    if predictions_path.exists() or checkpoint_dir.exists():
        raise FileExistsError(f"Refusing to overwrite cross-modal run {config.run_name}")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    rng = np.random.default_rng(config.seed)
    device = torch.device(config.device)

    logger.info("Loading dataset checkpoint: %s", config.checkpoint_path)
    dataset = DlpfcDataset.from_checkpoint(config.checkpoint_path)

    if config.gene_npz_path is None:
        gene_np, image_np, barcodes, section_ids = load_embeddings(
            dataset, config.gene_key, config.image_key
        )
        gene_source = config.gene_key
    else:
        gene_np, image_np, barcodes, section_ids = load_external_gene_embeddings(
            dataset, config.gene_npz_path, config.gene_array_key, config.image_key
        )
        gene_source = f"{config.gene_npz_path}:{config.gene_array_key}"
    coordinates = load_spatial_coordinates(dataset)
    logger.info(
        "Loaded %d spots | gene_key=%s (dim=%d) | image_key=%s (dim=%d)",
        gene_np.shape[0], gene_source, gene_np.shape[1], config.image_key, image_np.shape[1],
    )

    train_idx, val_idx, test_idx = three_way_split(
        dataset, section_ids, config.val_fraction, config.test_unit,
        config.test_sections, config.test_donors, config.seed,
    )
    logger.info(
        "Split: %d train, %d val, %d test (test_unit=%s)",
        len(train_idx), len(val_idx), len(test_idx), config.test_unit,
    )

    split = np.full(len(section_ids), "train", dtype=object)
    split[val_idx] = "val"
    split[test_idx] = "test"
    graphs: dict[str, SplitNeighborhoodGraph] = {}
    if config.objective == "positive_neighborhood":
        pruning_config = NeighborhoodPruningConfig(
            coordinate_neighbors=config.coordinate_neighbors,
            retained_neighbors=config.retained_neighbors,
            min_gene_cosine=config.min_gene_cosine,
            min_image_cosine=config.min_image_cosine,
        )
        for split_name in ("train", "val", "test"):
            if np.any(split == split_name):
                graphs[split_name] = build_split_neighborhood_graph(
                    coordinates,
                    section_ids,
                    split,
                    split_name,
                    gene_np,
                    image_np,
                    pruning_config,
                )
                logger.info("%s graph: %s", split_name, graphs[split_name].diagnostics)

        graph_dir = config.output_dir / "graphs" / config.run_name
        graph_dir.mkdir(parents=True, exist_ok=True)
        with (graph_dir / "diagnostics.json").open("w") as file:
            json.dump(
                {name: graph.diagnostics for name, graph in graphs.items()},
                file,
                indent=2,
            )
        np.savez_compressed(
            graph_dir / "edges.npz",
            **{f"{name}_candidate_edges": graph.candidate_edges for name, graph in graphs.items()},
            **{f"{name}_retained_edges": graph.edges for name, graph in graphs.items()},
        )

    gene_emb = torch.as_tensor(gene_np, dtype=torch.float32, device=device)
    image_emb = torch.as_tensor(image_np, dtype=torch.float32, device=device)

    model_config = CrossModalConfig(
        gene_input_dim=gene_np.shape[1], image_input_dim=image_np.shape[1],
        hidden_dim=config.hidden_dim, output_dim=config.output_dim, dropout=config.dropout,
    )
    model = CrossModalModel(model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    metrics_path = config.output_dir / "metrics" / f"{config.run_name}_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    with metrics_path.open("w", newline="") as file:
        writer = csv.writer(file)
        if config.objective == "positive_neighborhood":
            metric_names = [
                "loss", "self_alignment_loss", "neighbor_alignment_loss",
                "gene_variance_loss", "image_variance_loss",
                "gene_covariance_loss", "image_covariance_loss",
                "self_positive_cosine", "neighbor_positive_cosine",
                "gene_std_mean", "image_std_mean",
            ]
        else:
            metric_names = [
                "loss", "clip_loss", "attraction_loss", "weighted_attraction_loss",
                "paired_cosine", "hardest_negative_cosine", "positive_hardest_margin",
                "temperature", "retrieval_accuracy",
            ]
        writer.writerow(
            ["epoch"]
            + [f"train_{name}" for name in metric_names]
            + [f"val_{name}" for name in metric_names]
        )

        for epoch in range(1, config.num_epochs + 1):
            if config.objective == "positive_neighborhood":
                train_metrics = run_positive_neighborhood_epoch(
                    model, gene_emb, image_emb, train_idx, graphs["train"],
                    config.batch_size, rng, train=True, optimizer=optimizer,
                    alignment_weight=config.alignment_weight,
                    neighbor_weight=config.neighbor_weight,
                    variance_weight=config.variance_weight,
                    variance_gamma=config.variance_gamma,
                    covariance_weight=config.covariance_weight,
                )
                val_metrics = run_positive_neighborhood_epoch(
                    model, gene_emb, image_emb, val_idx, graphs["val"],
                    config.batch_size, rng, train=False,
                    alignment_weight=config.alignment_weight,
                    neighbor_weight=config.neighbor_weight,
                    variance_weight=config.variance_weight,
                    variance_gamma=config.variance_gamma,
                    covariance_weight=config.covariance_weight,
                )
            else:
                train_metrics = run_epoch(
                    model, gene_emb, image_emb, train_idx, config.batch_size, rng, train=True,
                    optimizer=optimizer, log_every=config.log_every, epoch=epoch, num_epochs=config.num_epochs,
                    same_spot_gradient_ratio=config.same_spot_gradient_ratio,
                )
                val_metrics = run_epoch(
                    model, gene_emb, image_emb, val_idx, config.batch_size, rng, train=False,
                    same_spot_gradient_ratio=config.same_spot_gradient_ratio,
                )

            logger.info(
                "epoch %d/%d | train loss %.4f | val loss %.4f",
                epoch, config.num_epochs,
                train_metrics["loss"], val_metrics["loss"],
            )
            writer.writerow(
                [epoch]
                + [train_metrics[name] for name in metric_names]
                + [val_metrics[name] for name in metric_names]
            )
            file.flush()

            metrics = {"train": train_metrics, "val": val_metrics}
            save_checkpoint(model, config, epoch, metrics, checkpoint_dir / "last.pt")

            if val_metrics["loss"] < best_val_loss - config.min_delta:
                best_val_loss = val_metrics["loss"]
                epochs_without_improvement = 0
                save_checkpoint(model, config, epoch, metrics, checkpoint_dir / "best.pt")
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= config.patience:
                logger.info("Early stopping after %d epochs without improvement.", config.patience)
                break

    best_checkpoint = torch.load(checkpoint_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model"])

    gene_projected, image_projected = export_embeddings(model, gene_emb, image_emb)

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        predictions_path,
        gene_projected=gene_projected,
        image_projected=image_projected,
        barcodes=barcodes,
        section_ids=section_ids,
        split=split,
        image_key=config.image_key,
        objective=config.objective,
        gene_source=gene_source,
        gene_array_key=config.gene_array_key,
    )

    logger.info("Training complete. Best val loss: %.4f", best_val_loss)
    logger.info(
        "Exported gene_projected %s, image_projected %s to %s",
        gene_projected.shape, image_projected.shape, predictions_path,
    )
    return predictions_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train_cross_modal(TrainConfig(**vars(build_arg_parser().parse_args())))


if __name__ == "__main__":
    main()
