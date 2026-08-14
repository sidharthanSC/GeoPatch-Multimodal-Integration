"""Cross-modal InfoNCE training between ``gene_emb`` and an image-side embedding.

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
    retrieval_accuracy,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    checkpoint_path: Path = Path("checkpoints/dlpfc.pkl")
    output_dir: Path = Path("outputs/cross_modal")
    run_name: str = ""

    gene_key: str = "gene_emb"
    image_key: str = "img_emb"

    hidden_dim: int = 256
    output_dim: int = 128
    dropout: float = 0.1
    same_spot_gradient_ratio: float = 1.0

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
    parser.add_argument(
        "--image-key", type=str, default=defaults.image_key, choices=["img_emb", "proj_emb"],
        help="Which image-side embedding to align gene_emb against.",
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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = TrainConfig(**vars(build_arg_parser().parse_args()))
    if not config.run_name:
        config.run_name = f"cross_modal_{config.image_key}"

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    rng = np.random.default_rng(config.seed)
    device = torch.device(config.device)

    logger.info("Loading dataset checkpoint: %s", config.checkpoint_path)
    dataset = DlpfcDataset.from_checkpoint(config.checkpoint_path)

    gene_np, image_np, barcodes, section_ids = load_embeddings(dataset, config.gene_key, config.image_key)
    logger.info(
        "Loaded %d spots | gene_key=%s (dim=%d) | image_key=%s (dim=%d)",
        gene_np.shape[0], config.gene_key, gene_np.shape[1], config.image_key, image_np.shape[1],
    )

    train_idx, val_idx, test_idx = three_way_split(
        dataset, section_ids, config.val_fraction, config.test_unit,
        config.test_sections, config.test_donors, config.seed,
    )
    logger.info(
        "Split: %d train, %d val, %d test (test_unit=%s)",
        len(train_idx), len(val_idx), len(test_idx), config.test_unit,
    )

    gene_emb = torch.as_tensor(gene_np, dtype=torch.float32, device=device)
    image_emb = torch.as_tensor(image_np, dtype=torch.float32, device=device)

    model_config = CrossModalConfig(
        gene_input_dim=gene_np.shape[1], image_input_dim=image_np.shape[1],
        hidden_dim=config.hidden_dim, output_dim=config.output_dim, dropout=config.dropout,
    )
    model = CrossModalModel(model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    checkpoint_dir = config.output_dir / "checkpoints" / config.run_name
    metrics_path = config.output_dir / "metrics" / f"{config.run_name}_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    with metrics_path.open("w", newline="") as file:
        writer = csv.writer(file)
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
                "epoch %d/%d | train loss %.4f acc %.4f | val loss %.4f acc %.4f",
                epoch, config.num_epochs,
                train_metrics["loss"], train_metrics["retrieval_accuracy"],
                val_metrics["loss"], val_metrics["retrieval_accuracy"],
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

    split = np.full(len(section_ids), "train", dtype=object)
    split[val_idx] = "val"
    split[test_idx] = "test"

    gene_projected, image_projected = export_embeddings(model, gene_emb, image_emb)

    predictions_path = config.output_dir / "predictions" / f"{config.run_name}_embeddings.npz"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        predictions_path,
        gene_projected=gene_projected,
        image_projected=image_projected,
        barcodes=barcodes,
        section_ids=section_ids,
        split=split,
        image_key=config.image_key,
    )

    logger.info("Training complete. Best val loss: %.4f", best_val_loss)
    logger.info(
        "Exported gene_projected %s, image_projected %s to %s",
        gene_projected.shape, image_projected.shape, predictions_path,
    )


if __name__ == "__main__":
    main()
