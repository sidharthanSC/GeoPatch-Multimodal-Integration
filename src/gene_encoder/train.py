"""Train the BYOL gene-expression encoder (see ``src/gene_encoder/model.py``).

Three-way spot split -- train / validation / a held-out test set:

- **train**: used for gradient updates.
- **val**: used every epoch to compute a monitoring loss and drive early stopping.
  Never seen by the optimizer.
- **test**: ``--test-fraction`` (default ``0.0``, i.e. no test set is carved out --
  every spot lands in train/val for now). Test spots never contribute to training or
  to the validation loss; when a nonzero fraction is requested, their embeddings are
  still exported the same way as everyone else's, via a plain forward pass through the
  final trained encoder -- this is meant for a *separate* future run to check how well
  the learned representation generalizes to spots the model never trained on at all,
  not something this run's training loop uses.

Usage
-----
    python -m src.gene_encoder.train
    python -m src.gene_encoder.train --test-fraction 0.15 --run-name gene_byol_heldout

Outputs (under ``--output-dir``, default ``outputs/gene_encoder``):

- ``checkpoints/<run_name>/{best.pt, last.pt}``
- ``metrics/<run_name>_metrics.csv`` -- one row per epoch (train/val loss).
- ``predictions/<run_name>_embeddings.npz`` -- ``embeddings`` (``(n_spots,
  embedding_dim)``), ``barcodes``, ``section_ids``, ``split`` (``"train"``/``"val"``/
  ``"test"`` per spot), for every spot in the dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src.datasets.dlpfc import DlpfcDataset
from src.gene_encoder.augmentations import spatial_mean_aggregation
from src.gene_encoder.model import (
    GeneEncoder,
    GeneEncoderConfig,
    build_byol_learner,
    make_view_augmentations,
    variance_regularization,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    checkpoint_path: Path = Path("checkpoints/dlpfc.pkl")
    hvg_path: Path = Path("outputs/gene_encoder/shared_hvgs.json")
    spatial_graph_path: Path = Path("outputs/gene_encoder/spatial_knn_graph.pkl")
    output_dir: Path = Path("outputs/gene_encoder")
    run_name: str = "gene_byol"

    hidden_dim: int = 512
    embedding_dim: int = 128
    projection_dim: int = 128
    projection_hidden_dim: int = 256
    dropout: float = 0.1
    ema_decay: float = 0.996
    mask_rate: float = 0.5
    noise_std_fraction: float = 0.1
    smoothing_includes_self: bool = True
    variance_weight: float = 5.0
    variance_gamma: float = 1.0

    batch_size: int = 256
    num_epochs: int = 100
    lr: float = 1e-5  # validated to avoid collapse (with the variance regularizer); 3e-4 collapsed within 2 epochs.
    weight_decay: float = 1e-4

    val_fraction: float = 0.2
    test_fraction: float = 0.0
    seed: int = 0
    patience: int = 10
    min_delta: float = 1e-4

    device: str = field(default_factory=lambda: default_device())
    log_every: int = 20


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_shared_hvg_expression(
    dataset: DlpfcDataset, hvg_genes: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every spot's shared-HVG expression, concatenated in ``dataset.section_ids()`` order.

    Returns ``(expression (n_spots, n_hvg) float32, barcodes (n_spots,), section_ids
    (n_spots,))``.
    """
    expression_blocks, barcode_blocks, section_id_blocks = [], [], []

    for section_id in dataset.section_ids():
        adata = dataset.get_section(section_id)
        gene_idx = [adata.var_names.get_loc(gene) for gene in hvg_genes]
        block = adata.X[:, gene_idx]
        block = block.toarray() if hasattr(block, "toarray") else np.asarray(block)

        expression_blocks.append(block.astype(np.float32))
        barcode_blocks.append(adata.obs_names.to_numpy())
        section_id_blocks.append(np.full(adata.n_obs, str(section_id), dtype=object))

    return (
        np.concatenate(expression_blocks, axis=0),
        np.concatenate(barcode_blocks, axis=0),
        np.concatenate(section_id_blocks, axis=0),
    )


def build_smoothed_expression(
    dataset: DlpfcDataset,
    hvg_genes: List[str],
    spatial_graphs: Dict[str, Dict[str, List[str]]],
    include_self: bool = True,
) -> np.ndarray:
    """Spatially-smoothed expression for every spot, same global row order as
    :func:`load_shared_hvg_expression`."""
    blocks = []
    for section_id in dataset.section_ids():
        adata = dataset.get_section(section_id)
        gene_idx = [adata.var_names.get_loc(gene) for gene in hvg_genes]
        block = adata.X[:, gene_idx]
        block = block.toarray() if hasattr(block, "toarray") else np.asarray(block)
        block = block.astype(np.float32)

        barcodes = adata.obs_names.tolist()
        smoothed = spatial_mean_aggregation(
            block, barcodes, spatial_graphs[section_id], include_self=include_self
        )
        blocks.append(smoothed)

    return np.concatenate(blocks, axis=0)


def three_way_split(
    section_ids: np.ndarray, val_fraction: float, test_fraction: float, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split spot indices into (train, val, test), stratified by section at each stage.

    ``test_fraction=0`` (the default) skips carving out a test set entirely --
    ``test_idx`` comes back empty and every spot lands in train or val.
    """
    indices = np.arange(len(section_ids))

    if test_fraction > 0:
        train_val_idx, test_idx = train_test_split(
            indices, test_size=test_fraction, random_state=seed, stratify=section_ids
        )
    else:
        train_val_idx, test_idx = indices, np.array([], dtype=np.int64)

    relative_val_fraction = val_fraction / (1.0 - test_fraction)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=relative_val_fraction,
        random_state=seed,
        stratify=section_ids[train_val_idx],
    )
    return train_idx, val_idx, test_idx


def run_epoch(
    learner,
    encoder: GeneEncoder,
    augment_view_a,
    augment_view_b,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
    rng: np.random.Generator,
    train: bool,
    variance_weight: float,
    variance_gamma: float,
    optimizer: torch.optim.Optimizer = None,
    log_every: int = 0,
    epoch: int = 0,
    num_epochs: int = 0,
) -> Dict[str, float]:
    """One pass over ``indices`` (shuffled if ``train``).

    Returns mean ``byol_loss``, ``variance_loss``, ``total_loss`` (what's actually
    optimized: ``byol_loss + variance_weight * variance_loss``), and ``embed_std``
    (mean per-embedding-dimension std across batches -- the direct diagnostic for
    collapse; watch this stay near/above ``variance_gamma``, not drift toward 0).

    Batches smaller than 2 are skipped -- ``byol_pytorch``'s internal projector uses
    BatchNorm, which requires batch size > 1.
    """
    learner.train() if train else learner.eval()

    order = indices.copy()
    if train:
        rng.shuffle(order)

    totals = {"byol_loss": 0.0, "variance_loss": 0.0, "total_loss": 0.0, "embed_std": 0.0}
    n_batches = 0

    for start in range(0, len(order), batch_size):
        batch_idx = order[start : start + batch_size]
        if len(batch_idx) < 2:
            continue
        x = torch.as_tensor(batch_idx, dtype=torch.long, device=device)

        with torch.set_grad_enabled(train):
            byol_loss = learner(x)
            embed_a = encoder(augment_view_a(x))
            embed_b = encoder(augment_view_b(x))
            var_loss = (
                variance_regularization(embed_a, gamma=variance_gamma)
                + variance_regularization(embed_b, gamma=variance_gamma)
            ) / 2.0
            total_loss = byol_loss + variance_weight * var_loss
            embed_std = torch.cat([embed_a, embed_b], dim=0).std(dim=0).mean()

        if train:
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()
            learner.update_moving_average()

        totals["byol_loss"] += float(byol_loss.item())
        totals["variance_loss"] += float(var_loss.item())
        totals["total_loss"] += float(total_loss.item())
        totals["embed_std"] += float(embed_std.item())
        n_batches += 1

        if train and log_every and n_batches % log_every == 0:
            logger.info(
                "epoch %d/%d step %d | loss %.4f (byol %.4f, var %.4f) | embed_std %.4f",
                epoch, num_epochs, n_batches,
                total_loss.item(), byol_loss.item(), var_loss.item(), embed_std.item(),
            )

    return {key: value / max(n_batches, 1) for key, value in totals.items()}

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def export_embeddings(
    encoder: GeneEncoder, expression: torch.Tensor, batch_size: int = 4096
) -> np.ndarray:
    """Plain forward pass through the trained encoder -- no augmentation, no gradient."""
    encoder.eval()
    outputs = []
    for start in range(0, expression.shape[0], batch_size):
        outputs.append(encoder(expression[start : start + batch_size]).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def save_checkpoint(
    learner, config: TrainConfig, epoch: int, metrics: Dict[str, float], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder": learner.net.state_dict(),
            "byol_state_dict": learner.state_dict(),
            "train_config": asdict(config),
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--checkpoint-path", type=Path, default=defaults.checkpoint_path)
    parser.add_argument("--hvg-path", type=Path, default=defaults.hvg_path)
    parser.add_argument("--spatial-graph-path", type=Path, default=defaults.spatial_graph_path)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--run-name", type=str, default=defaults.run_name)

    parser.add_argument("--hidden-dim", type=int, default=defaults.hidden_dim)
    parser.add_argument("--embedding-dim", type=int, default=defaults.embedding_dim)
    parser.add_argument("--projection-dim", type=int, default=defaults.projection_dim)
    parser.add_argument("--projection-hidden-dim", type=int, default=defaults.projection_hidden_dim)
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument("--ema-decay", type=float, default=defaults.ema_decay)
    parser.add_argument("--mask-rate", type=float, default=defaults.mask_rate)
    parser.add_argument("--noise-std-fraction", type=float, default=defaults.noise_std_fraction)
    parser.add_argument(
        "--variance-weight", type=float, default=defaults.variance_weight,
        help="Coefficient on the VICReg-style variance-collapse penalty added to the "
             "BYOL loss (0 disables it). See src/gene_encoder/model.py's "
             "variance_regularization -- needed because the BYOL loss alone let this "
             "encoder collapse to a near-constant output for this data.",
    )
    parser.add_argument("--variance-gamma", type=float, default=defaults.variance_gamma)
    parser.add_argument(
        "--smoothing-includes-self", type=lambda s: s.lower() != "false",
        default=defaults.smoothing_includes_self,
    )

    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--num-epochs", type=int, default=defaults.num_epochs)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)

    parser.add_argument("--val-fraction", type=float, default=defaults.val_fraction)
    parser.add_argument(
        "--test-fraction", type=float, default=defaults.test_fraction,
        help="Held out entirely from training/validation; embeddings still exported "
             "via direct inference. Default 0.0 -- use a separate run to test "
             "generalization once the methodology is settled.",
    )
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--patience", type=int, default=defaults.patience)
    parser.add_argument("--min-delta", type=float, default=defaults.min_delta)
    parser.add_argument("--device", type=str, default=defaults.device)
    parser.add_argument("--log-every", type=int, default=defaults.log_every)

    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = TrainConfig(**vars(build_arg_parser().parse_args()))

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    rng = np.random.default_rng(config.seed)
    device = torch.device(config.device)

    logger.info("Loading dataset checkpoint: %s", config.checkpoint_path)
    dataset = DlpfcDataset.from_checkpoint(config.checkpoint_path)

    with config.hvg_path.open() as f:
        hvg_genes = json.load(f)
    with config.spatial_graph_path.open("rb") as f:
        spatial_graphs = pickle.load(f)

    expression_np, barcodes, section_ids = load_shared_hvg_expression(dataset, hvg_genes)
    smoothed_np = build_smoothed_expression(
        dataset, hvg_genes, spatial_graphs, include_self=config.smoothing_includes_self
    )
    logger.info(
        "Loaded %d spots, %d shared HVGs, %d sections",
        expression_np.shape[0], expression_np.shape[1], len(dataset.section_ids()),
    )

    train_idx, val_idx, test_idx = three_way_split(
        section_ids, config.val_fraction, config.test_fraction, config.seed
    )
    logger.info(
        "Split: %d train, %d val, %d test (test_fraction=%.3f)",
        len(train_idx), len(val_idx), len(test_idx), config.test_fraction,
    )

    expression = torch.as_tensor(expression_np, dtype=torch.float32, device=device)
    smoothed = torch.as_tensor(smoothed_np, dtype=torch.float32, device=device)
    gene_min = expression.min(dim=0).values
    gene_max = expression.max(dim=0).values

    augment_view_a, augment_view_b = make_view_augmentations(
        expression, smoothed, gene_min, gene_max,
        mask_rate=config.mask_rate, noise_std_fraction=config.noise_std_fraction,
    )

    encoder_config = GeneEncoderConfig(
        input_dim=expression_np.shape[1],
        hidden_dim=config.hidden_dim,
        embedding_dim=config.embedding_dim,
        projection_dim=config.projection_dim,
        projection_hidden_dim=config.projection_hidden_dim,
        dropout=config.dropout,
        ema_decay=config.ema_decay,
    )
    encoder = GeneEncoder(encoder_config).to(device)
    learner = build_byol_learner(encoder, encoder_config, augment_view_a, augment_view_b, device)

    optimizer = torch.optim.Adam(learner.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    checkpoint_dir = config.output_dir / "checkpoints" / config.run_name
    metrics_path = config.output_dir / "metrics" / f"{config.run_name}_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    with metrics_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "epoch",
            "train_total_loss", "train_byol_loss", "train_variance_loss", "train_embed_std",
            "val_total_loss", "val_byol_loss", "val_variance_loss", "val_embed_std",
        ])

        for epoch in range(1, config.num_epochs + 1):
            train_metrics = run_epoch(
                learner, encoder, augment_view_a, augment_view_b, train_idx,
                config.batch_size, device, rng, train=True,
                variance_weight=config.variance_weight, variance_gamma=config.variance_gamma,
                optimizer=optimizer, log_every=config.log_every, epoch=epoch, num_epochs=config.num_epochs,
            )
            val_metrics = run_epoch(
                learner, encoder, augment_view_a, augment_view_b, val_idx,
                config.batch_size, device, rng, train=False,
                variance_weight=config.variance_weight, variance_gamma=config.variance_gamma,
            )

            logger.info(
                "epoch %d/%d | train: total %.4f byol %.4f var %.4f std %.4f | "
                "val: total %.4f byol %.4f var %.4f std %.4f",
                epoch, config.num_epochs,
                train_metrics["total_loss"], train_metrics["byol_loss"],
                train_metrics["variance_loss"], train_metrics["embed_std"],
                val_metrics["total_loss"], val_metrics["byol_loss"],
                val_metrics["variance_loss"], val_metrics["embed_std"],
            )
            writer.writerow([
                epoch,
                train_metrics["total_loss"], train_metrics["byol_loss"],
                train_metrics["variance_loss"], train_metrics["embed_std"],
                val_metrics["total_loss"], val_metrics["byol_loss"],
                val_metrics["variance_loss"], val_metrics["embed_std"],
            ])
            file.flush()

            metrics = {"train": train_metrics, "val": val_metrics}
            save_checkpoint(learner, config, epoch, metrics, checkpoint_dir / "last.pt")

            val_total_loss = val_metrics["total_loss"]
            if val_total_loss < best_val_loss - config.min_delta:
                best_val_loss = val_total_loss
                epochs_without_improvement = 0
                save_checkpoint(learner, config, epoch, metrics, checkpoint_dir / "best.pt")
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= config.patience:
                logger.info("Early stopping after %d epochs without improvement.", config.patience)
                break

    best_checkpoint = torch.load(checkpoint_dir / "best.pt", map_location=device, weights_only=False)
    encoder.load_state_dict(best_checkpoint["encoder"])

    split = np.full(len(section_ids), "train", dtype=object)
    split[val_idx] = "val"
    split[test_idx] = "test"

    embeddings = export_embeddings(encoder, expression)

    predictions_path = config.output_dir / "predictions" / f"{config.run_name}_embeddings.npz"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        predictions_path,
        embeddings=embeddings,
        barcodes=barcodes,
        section_ids=section_ids,
        split=split,
    )

    logger.info("Training complete. Best val loss: %.4f", best_val_loss)
    logger.info("Exported %d embeddings (dim=%d) to %s", embeddings.shape[0], embeddings.shape[1], predictions_path)


if __name__ == "__main__":
    main()
