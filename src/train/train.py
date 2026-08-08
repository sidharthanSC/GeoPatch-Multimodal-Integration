"""Train a shared layer-aware projector on frozen BYOL image embeddings.

This script uses ground-truth cortical layer labels (``adata.obs["ground_truth"]``,
``Layer_1``..``Layer_6``/``WM``) as the pairing key, via ``BalancedPairSampler`` from
``src/train/pairs.py``:
- eta = 0 for two patches from the same cortical layer;
- eta = 1 for patches from different cortical layers.

Tissue section id is still tracked per spot (as ``slice_ids``) and exported alongside
the embeddings for downstream analysis, but it is not used for pairing.

The exact normalized projector outputs used in the loss are also used for validation
and export.
"""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src.datasets.dlpfc import DlpfcDataset
from src.train.model import (
    ProjectorConfig,
    SliceProjectionModel,
    WeightedSliceContrastiveLoss,
)
from src.train.pairs import BalancedPairSampler

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    checkpoint_path: Path = Path("checkpoints/dlpfc.pkl")
    output_dir: Path = Path("outputs")
    run_name: str = "layer_projector"

    hidden_dim: int = 256
    projection_dim: int = 128
    dropout: float = 0.1

    batch_size: int = 256
    steps_per_epoch: int = 200
    num_epochs: int = 40
    val_batches: int = 50

    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 5.0

    # Balanced batches already contain approximately 50% positive and 50% negative
    # pairs. A larger positive weight prevents the model from separating layers by
    # simply lowering all similarities, as happened in the earlier run.
    positive_weight: float = 2.0
    negative_weight: float = 1.0

    # Different-layer pairs are pushed to cosine similarity <= 0 by default.
    # Use 0.1 if zero is too aggressive for the dataset.
    negative_margin: float = 0.0

    val_fraction: float = 0.2
    seed: int = 0
    patience: int = 8
    min_delta: float = 1e-4

    # Which `validate()` metric selects the checkpoint saved as `best.pt`.
    best_metric: str = "loss"

    device: str = field(default_factory=lambda: default_device())
    log_every: int = 20


# Direction each supported `best_metric` choice is optimized in.
_BEST_METRIC_DIRECTIONS = {
    "loss": "min",
    "similarity_gap": "max",
}


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_embeddings_and_slice_ids(
    dataset: DlpfcDataset,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load frozen image embeddings, cortical labels, and tissue slice IDs."""
    embeddings: List[np.ndarray] = []
    cortical_labels: List[np.ndarray] = []
    slice_ids: List[np.ndarray] = []

    for section_id in dataset.section_ids():
        adata = dataset.get_section(section_id)

        section_embeddings = np.asarray(
            adata.obsm["img_emb"], dtype=np.float32
        )

        if "ground_truth" in adata.obs:
            gt = adata.obs["ground_truth"]
            valid = gt.notna().to_numpy()
            labels = gt.to_numpy(dtype=object)[valid].astype(str)
        else:
            valid = np.ones(section_embeddings.shape[0], dtype=bool)
            labels = np.full(valid.sum(), "unknown", dtype=object)

        embeddings.append(section_embeddings[valid])
        cortical_labels.append(labels)
        slice_ids.append(
            np.full(valid.sum(), str(section_id), dtype=object)
        )

    if not embeddings:
        raise ValueError("No sections were loaded from the dataset.")

    return (
        np.concatenate(embeddings, axis=0),
        np.concatenate(cortical_labels, axis=0),
        np.concatenate(slice_ids, axis=0),
    )


def stratified_spot_split(
    labels: np.ndarray,
    val_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split spots while retaining every label in both train and validation.

    This split is intended only to monitor whether the pairwise objective is learned.
    A separate leave-one-donor-out evaluation should be used for final reporting.
    """
    indices = np.arange(len(labels))
    return train_test_split(
        indices,
        test_size=val_fraction,
        random_state=seed,
        stratify=labels,
    )


def train_one_epoch(
    model: SliceProjectionModel,
    criterion: WeightedSliceContrastiveLoss,
    optimizer: torch.optim.Optimizer,
    embeddings: torch.Tensor,
    sampler: BalancedPairSampler,
    config: TrainConfig,
) -> Dict[str, float]:
    model.train()

    loss_total = 0.0
    same_similarity_total = 0.0
    diff_similarity_total = 0.0
    same_count = 0
    diff_count = 0

    for step in range(config.steps_per_epoch):
        idx_a, idx_b, eta_np = sampler.sample_batch(config.batch_size)

        a = embeddings[idx_a]
        b = embeddings[idx_b]
        eta = torch.as_tensor(
            eta_np, dtype=torch.float32, device=embeddings.device
        )

        z_a = model(a)
        z_b = model(b)
        loss, similarity = criterion(z_a, z_b, eta)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.grad_clip_norm
        )
        optimizer.step()

        loss_total += float(loss.item())

        same_mask = eta == 0
        diff_mask = eta == 1

        if same_mask.any():
            same_similarity_total += float(similarity[same_mask].sum().item())
            same_count += int(same_mask.sum().item())

        if diff_mask.any():
            diff_similarity_total += float(similarity[diff_mask].sum().item())
            diff_count += int(diff_mask.sum().item())

        if config.log_every and (step + 1) % config.log_every == 0:
            logger.info(
                "step %d/%d | loss %.4f",
                step + 1,
                config.steps_per_epoch,
                loss.item(),
            )

    same_similarity = same_similarity_total / max(same_count, 1)
    diff_similarity = diff_similarity_total / max(diff_count, 1)

    return {
        "loss": loss_total / config.steps_per_epoch,
        "same_layer_similarity": same_similarity,
        "diff_layer_similarity": diff_similarity,
        "similarity_gap": same_similarity - diff_similarity,
    }


@torch.no_grad()
def validate(
    model: SliceProjectionModel,
    criterion: WeightedSliceContrastiveLoss,
    embeddings: torch.Tensor,
    sampler: BalancedPairSampler,
    config: TrainConfig,
) -> Dict[str, float]:
    model.eval()

    losses: List[np.ndarray] = []
    similarities: List[np.ndarray] = []
    etas: List[np.ndarray] = []

    for _ in range(config.val_batches):
        idx_a, idx_b, eta_np = sampler.sample_batch(config.batch_size)

        a = embeddings[idx_a]
        b = embeddings[idx_b]
        eta = torch.as_tensor(
            eta_np, dtype=torch.float32, device=embeddings.device
        )

        z_a = model(a)
        z_b = model(b)

        # Reconstruct per-pair terms for detailed validation statistics.
        similarity = torch.sum(z_a * z_b, dim=-1).clamp(-1.0, 1.0)
        positive_loss = (1.0 - similarity).pow(2)
        negative_loss = torch.relu(
            similarity - criterion.negative_margin
        ).pow(2)
        pair_loss = (
            (1.0 - eta) * criterion.positive_weight * positive_loss
            + eta * criterion.negative_weight * negative_loss
        )

        losses.append(pair_loss.cpu().numpy())
        similarities.append(similarity.cpu().numpy())
        etas.append(eta_np)

    losses_np = np.concatenate(losses)
    similarities_np = np.concatenate(similarities)
    etas_np = np.concatenate(etas)

    same_mask = etas_np == 0.0
    diff_mask = etas_np == 1.0

    same_similarity = float(similarities_np[same_mask].mean())
    diff_similarity = float(similarities_np[diff_mask].mean())

    return {
        "loss": float(losses_np.mean()),
        "same_layer_similarity": same_similarity,
        "diff_layer_similarity": diff_similarity,
        "similarity_gap": same_similarity - diff_similarity,
        "negative_margin_violation_rate": float(
            (
                similarities_np[diff_mask]
                > criterion.negative_margin
            ).mean()
        ),
    }


@torch.no_grad()
def export_embeddings(
    model: SliceProjectionModel,
    embeddings: torch.Tensor,
    cortical_labels: np.ndarray,
    slice_ids: np.ndarray,
    path: Path,
) -> None:
    model.eval()
    projected = model(embeddings).cpu().numpy()

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        projected_embeddings=projected,
        raw_embeddings=embeddings.cpu().numpy(),
        cortical_labels=cortical_labels,
        slice_ids=slice_ids,
    )


def save_checkpoint(
    model: SliceProjectionModel,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    epoch: int,
    metrics: Dict[str, float],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "projector_config": asdict(model.config),
            "train_config": asdict(config),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--checkpoint-path", type=Path, default=defaults.checkpoint_path
    )
    parser.add_argument(
        "--output-dir", type=Path, default=defaults.output_dir
    )
    parser.add_argument(
        "--run-name", type=str, default=defaults.run_name,
        help="Names this run's checkpoint dir / metrics CSV / predictions .npz "
             "(e.g. 'layer_projector_e200'), so it doesn't overwrite other runs.",
    )
    parser.add_argument("--hidden-dim", type=int, default=defaults.hidden_dim)
    parser.add_argument(
        "--projection-dim", type=int, default=defaults.projection_dim
    )
    parser.add_argument("--dropout", type=float, default=defaults.dropout)

    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument(
        "--steps-per-epoch", type=int, default=defaults.steps_per_epoch
    )
    parser.add_argument("--num-epochs", type=int, default=defaults.num_epochs)
    parser.add_argument("--val-batches", type=int, default=defaults.val_batches)

    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument(
        "--weight-decay", type=float, default=defaults.weight_decay
    )
    parser.add_argument(
        "--grad-clip-norm", type=float, default=defaults.grad_clip_norm
    )

    parser.add_argument(
        "--positive-weight", type=float, default=defaults.positive_weight
    )
    parser.add_argument(
        "--negative-weight", type=float, default=defaults.negative_weight
    )
    parser.add_argument(
        "--negative-margin", type=float, default=defaults.negative_margin
    )

    parser.add_argument(
        "--val-fraction", type=float, default=defaults.val_fraction
    )
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--patience", type=int, default=defaults.patience)
    parser.add_argument("--min-delta", type=float, default=defaults.min_delta)
    parser.add_argument(
        "--best-metric", type=str, default=defaults.best_metric,
        choices=sorted(_BEST_METRIC_DIRECTIONS),
        help="validate() metric used to pick best.pt: 'loss' (lower is better) or "
             "'similarity_gap' (higher is better).",
    )
    parser.add_argument("--device", type=str, default=defaults.device)
    parser.add_argument("--log-every", type=int, default=defaults.log_every)

    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    config = TrainConfig(**vars(build_arg_parser().parse_args()))

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = torch.device(config.device)

    logger.info("Loading dataset checkpoint: %s", config.checkpoint_path)
    dataset = DlpfcDataset.from_checkpoint(config.checkpoint_path)

    embeddings_np, cortical_labels, slice_ids = (
        load_embeddings_and_slice_ids(dataset)
    )

    logger.info(
        "Loaded %d patches, embedding_dim=%d, layers=%d, slices=%d",
        embeddings_np.shape[0],
        embeddings_np.shape[1],
        len(np.unique(cortical_labels)),
        len(np.unique(slice_ids)),
    )

    train_idx, val_idx = stratified_spot_split(
        labels=cortical_labels,
        val_fraction=config.val_fraction,
        seed=config.seed,
    )

    embeddings = torch.as_tensor(
        embeddings_np, dtype=torch.float32, device=device
    )
    train_embeddings = embeddings[train_idx]
    val_embeddings = embeddings[val_idx]

    train_layer_labels = cortical_labels[train_idx]
    val_layer_labels = cortical_labels[val_idx]

    train_sampler = BalancedPairSampler(
        train_layer_labels, seed=config.seed
    )
    val_sampler = BalancedPairSampler(
        val_layer_labels, seed=config.seed + 1
    )

    projector_config = ProjectorConfig(
        input_dim=embeddings_np.shape[1],
        hidden_dim=config.hidden_dim,
        projection_dim=config.projection_dim,
        dropout=config.dropout,
    )

    model = SliceProjectionModel(projector_config).to(device)
    criterion = WeightedSliceContrastiveLoss(
        positive_weight=config.positive_weight,
        negative_weight=config.negative_weight,
        negative_margin=config.negative_margin,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    checkpoint_dir = (
        config.output_dir / "checkpoints" / config.run_name
    )
    metrics_path = (
        config.output_dir / "metrics" / f"{config.run_name}_metrics.csv"
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    best_direction = _BEST_METRIC_DIRECTIONS[config.best_metric]
    best_value = float("inf") if best_direction == "min" else -float("inf")
    epochs_without_improvement = 0

    with metrics_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_same_layer_similarity",
                "train_diff_layer_similarity",
                "train_similarity_gap",
                "val_loss",
                "val_same_layer_similarity",
                "val_diff_layer_similarity",
                "val_similarity_gap",
                "negative_margin_violation_rate",
            ]
        )

        for epoch in range(1, config.num_epochs + 1):
            train_metrics = train_one_epoch(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                embeddings=train_embeddings,
                sampler=train_sampler,
                config=config,
            )

            val_metrics = validate(
                model=model,
                criterion=criterion,
                embeddings=val_embeddings,
                sampler=val_sampler,
                config=config,
            )

            logger.info(
                "epoch %d/%d | train_loss %.4f | val_loss %.4f | "
                "same %.4f | diff %.4f | gap %.4f | violation %.4f",
                epoch,
                config.num_epochs,
                train_metrics["loss"],
                val_metrics["loss"],
                val_metrics["same_layer_similarity"],
                val_metrics["diff_layer_similarity"],
                val_metrics["similarity_gap"],
                val_metrics["negative_margin_violation_rate"],
            )

            writer.writerow(
                [
                    epoch,
                    train_metrics["loss"],
                    train_metrics["same_layer_similarity"],
                    train_metrics["diff_layer_similarity"],
                    train_metrics["similarity_gap"],
                    val_metrics["loss"],
                    val_metrics["same_layer_similarity"],
                    val_metrics["diff_layer_similarity"],
                    val_metrics["similarity_gap"],
                    val_metrics["negative_margin_violation_rate"],
                ]
            )
            file.flush()

            save_checkpoint(
                model,
                optimizer,
                config,
                epoch,
                val_metrics,
                checkpoint_dir / "last.pt",
            )

            current_value = val_metrics[config.best_metric]
            if best_direction == "min":
                improved = current_value < best_value - config.min_delta
            else:
                improved = current_value > best_value + config.min_delta

            if improved:
                best_value = current_value
                epochs_without_improvement = 0
                save_checkpoint(
                    model,
                    optimizer,
                    config,
                    epoch,
                    val_metrics,
                    checkpoint_dir / "best.pt",
                )
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= config.patience:
                logger.info(
                    "Early stopping after %d epochs without improvement.",
                    config.patience,
                )
                break

    best_checkpoint = torch.load(
        checkpoint_dir / "best.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(best_checkpoint["model"])

    export_embeddings(
        model=model,
        embeddings=embeddings,
        cortical_labels=cortical_labels,
        slice_ids=slice_ids,
        path=(
            config.output_dir
            / "predictions"
            / f"{config.run_name}_embeddings.npz"
        ),
    )

    torch.save(
        {
            "model": model.state_dict(),
            "projector_config": asdict(model.config),
        },
        checkpoint_dir / "projector_final.pt",
    )

    logger.info(
        "Training complete. Best validation %s: %.4f",
        config.best_metric,
        best_value,
    )


if __name__ == "__main__":
    main()
