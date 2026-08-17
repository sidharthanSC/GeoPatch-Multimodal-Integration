"""Reconstruction-regularized GBSSA alignment for rich gene embeddings."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from src.cross_modal.model import (
    CrossModalConfig,
    CrossModalModel,
    covariance_loss,
    gradient_balanced_clip_loss,
    retrieval_accuracy,
    variance_loss,
)
from src.cross_modal.train import (
    default_device,
    export_embeddings,
    load_external_gene_embeddings,
    three_way_split,
)
from src.datasets.dlpfc import DlpfcDataset
from src.gene_encoder.model import (
    GeneReconstructionDecoder,
    cosine_geometry_regularization,
    masked_reconstruction_loss,
)
from src.gene_encoder.rich_data import load_sparse_expression

logger = logging.getLogger(__name__)


@dataclass
class RichCrossModalConfig:
    checkpoint_path: Path = Path("checkpoints/dlpfc.pkl")
    gene_npz_path: Path = Path()
    gene_array_key: str = "gene_embedding_128"
    image_key: str = "img_emb"
    output_dir: Path = Path("outputs/cross_modal")
    run_name: str = ""

    hidden_dim: int = 256
    output_dim: int = 128
    dropout: float = 0.1
    same_spot_gradient_ratio: float = 1.0
    reconstruction_weight: float = 2.0
    reconstruction_mask_rate: float = 0.3
    reconstruction_nonzero_weight: float = 5.0
    geometry_weight: float = 1.0
    variance_weight: float = 5.0
    variance_gamma: float = 1.0
    covariance_weight: float = 1.0

    batch_size: int = 128
    num_epochs: int = 200
    lr: float = 3e-4
    weight_decay: float = 1e-4
    val_fraction: float = 0.2
    test_unit: str = "none"
    test_sections: list[str] = field(default_factory=list)
    test_donors: list[int] = field(default_factory=list)
    seed: int = 0
    patience: int = 25
    min_delta: float = 1e-4
    device: str = field(default_factory=default_device)
    log_every: int = 50


def load_rich_provenance(gene_npz_path: Path) -> tuple[Path, Path]:
    with np.load(gene_npz_path, allow_pickle=True) as data:
        required = {"feature_genes_path", "geometry_path"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"Rich gene artifact lacks provenance arrays: {sorted(missing)}")
        return Path(str(data["feature_genes_path"].item())), Path(
            str(data["geometry_path"].item())
        )


def load_geometry_scores(
    path: Path, section_ids: np.ndarray, barcodes: np.ndarray
) -> np.ndarray:
    with np.load(path, allow_pickle=True) as data:
        scores = np.asarray(data["scores"], dtype=np.float32)
        observed = list(
            zip(data["section_ids"].astype(str), data["barcodes"].astype(str))
        )
    expected = list(zip(section_ids.astype(str), barcodes.astype(str)))
    if observed != expected:
        raise ValueError("Expression geometry identities do not match alignment rows")
    return scores


def run_epoch(
    model: CrossModalModel,
    decoder: GeneReconstructionDecoder,
    gene_embeddings: torch.Tensor,
    image_embeddings: torch.Tensor,
    expression_store,
    geometry_scores: torch.Tensor,
    indices: np.ndarray,
    config: RichCrossModalConfig,
    rng: np.random.Generator,
    train: bool,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int = 0,
) -> dict[str, float]:
    model.train(train)
    decoder.train(train)
    order = indices.copy()
    if train:
        rng.shuffle(order)
    names = (
        "loss",
        "clip_loss",
        "reconstruction_loss",
        "geometry_loss",
        "variance_loss",
        "covariance_loss",
        "paired_cosine",
        "retrieval_accuracy",
    )
    totals = {name: 0.0 for name in names}
    n_batches = 0
    device = gene_embeddings.device

    for start in range(0, len(order), config.batch_size):
        batch_idx = order[start : start + config.batch_size]
        if len(batch_idx) < 2:
            continue
        with torch.set_grad_enabled(train):
            gene_proj, image_proj = model(
                gene_embeddings[batch_idx], image_embeddings[batch_idx]
            )
            alignment = gradient_balanced_clip_loss(
                gene_proj,
                image_proj,
                model.logit_scale,
                config.same_spot_gradient_ratio,
            )
            target = torch.from_numpy(expression_store.rows(batch_idx)).to(device)
            mask = torch.rand_like(target) < config.reconstruction_mask_rate
            reconstruction = masked_reconstruction_loss(
                decoder(gene_proj),
                target,
                mask,
                config.reconstruction_nonzero_weight,
            )
            geometry = cosine_geometry_regularization(
                gene_proj, geometry_scores[batch_idx].to(device)
            )
            scale = config.output_dim**0.5
            variance = (
                variance_loss(gene_proj * scale, config.variance_gamma)
                + variance_loss(image_proj * scale, config.variance_gamma)
            ) / 2.0
            covariance = (
                covariance_loss(gene_proj * scale)
                + covariance_loss(image_proj * scale)
            ) / 2.0
            loss = (
                alignment.total
                + config.reconstruction_weight * reconstruction
                + config.geometry_weight * geometry
                + config.variance_weight * variance
                + config.covariance_weight * covariance
            )

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            model.clamp_logit_scale()

        values = {
            "loss": loss,
            "clip_loss": alignment.clip,
            "reconstruction_loss": reconstruction,
            "geometry_loss": geometry,
            "variance_loss": variance,
            "covariance_loss": covariance,
            "paired_cosine": (gene_proj * image_proj).sum(dim=1).mean(),
            "retrieval_accuracy": torch.as_tensor(
                retrieval_accuracy(gene_proj, image_proj), device=device
            ),
        }
        for name in names:
            totals[name] += float(values[name].detach().item())
        n_batches += 1
        if train and config.log_every and n_batches % config.log_every == 0:
            logger.info(
                "epoch %d step %d | total %.4f clip %.4f recon %.4f geometry %.4f variance %.4f covariance %.4f retrieval %.4f",
                epoch,
                n_batches,
                values["loss"],
                values["clip_loss"],
                values["reconstruction_loss"],
                values["geometry_loss"],
                values["variance_loss"],
                values["covariance_loss"],
                values["retrieval_accuracy"],
            )
    return {name: value / max(n_batches, 1) for name, value in totals.items()}


def save_checkpoint(
    path: Path,
    model: CrossModalModel,
    decoder: GeneReconstructionDecoder,
    optimizer: torch.optim.Optimizer,
    config: RichCrossModalConfig,
    epoch: int,
    metrics: dict,
    expression_dim: int,
    feature_genes_path: Path,
    geometry_path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "model_state_dict": model.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "model_config": asdict(model.config),
            "epoch": epoch,
            "metrics": metrics,
            "expression_dim": expression_dim,
            "feature_genes_path": str(feature_genes_path),
            "geometry_path": str(geometry_path),
        },
        path,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    defaults = RichCrossModalConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, default=defaults.checkpoint_path)
    parser.add_argument("--gene-npz-path", type=Path, required=True)
    parser.add_argument("--gene-array-key", default=defaults.gene_array_key)
    parser.add_argument("--image-key", default=defaults.image_key)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--hidden-dim", type=int, default=defaults.hidden_dim)
    parser.add_argument("--output-dim", type=int, default=defaults.output_dim)
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    for name in (
        "same_spot_gradient_ratio",
        "reconstruction_weight",
        "reconstruction_mask_rate",
        "reconstruction_nonzero_weight",
        "geometry_weight",
        "variance_weight",
        "variance_gamma",
        "covariance_weight",
        "lr",
        "weight_decay",
        "val_fraction",
        "min_delta",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=float, default=getattr(defaults, name))
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--num-epochs", type=int, default=defaults.num_epochs)
    parser.add_argument("--test-unit", choices=["none", "section", "donor"], default=defaults.test_unit)
    parser.add_argument("--test-sections", nargs="*", default=[])
    parser.add_argument("--test-donors", type=int, nargs="*", default=[])
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--patience", type=int, default=defaults.patience)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--log-every", type=int, default=defaults.log_every)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = RichCrossModalConfig(**vars(build_arg_parser().parse_args()))
    checkpoint_dir = config.output_dir / "checkpoints" / config.run_name
    metrics_path = config.output_dir / "metrics" / f"{config.run_name}_metrics.csv"
    predictions_path = config.output_dir / "predictions" / f"{config.run_name}_embeddings.npz"
    if any(path.exists() for path in (checkpoint_dir, metrics_path, predictions_path)):
        raise FileExistsError(f"Refusing to overwrite rich alignment run {config.run_name}")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    rng = np.random.default_rng(config.seed)
    device = torch.device(config.device)
    feature_genes_path, geometry_path = load_rich_provenance(config.gene_npz_path)
    genes = json.loads(feature_genes_path.read_text(encoding="utf-8"))

    dataset = DlpfcDataset.from_checkpoint(config.checkpoint_path)
    gene_np, image_np, barcodes, section_ids = load_external_gene_embeddings(
        dataset,
        config.gene_npz_path,
        config.gene_array_key,
        config.image_key,
    )
    expression_store = load_sparse_expression(dataset, genes)
    if list(zip(expression_store.section_ids.astype(str), expression_store.barcodes.astype(str))) != list(zip(section_ids.astype(str), barcodes.astype(str))):
        raise ValueError("Expression store identities do not match rich gene embeddings")
    geometry_np = load_geometry_scores(geometry_path, section_ids, barcodes)
    train_idx, val_idx, test_idx = three_way_split(
        dataset,
        section_ids,
        config.val_fraction,
        config.test_unit,
        config.test_sections,
        config.test_donors,
        config.seed,
    )
    del dataset
    gc.collect()

    gene = torch.from_numpy(gene_np).to(device)
    image = torch.from_numpy(image_np).to(device)
    geometry = torch.from_numpy(geometry_np)
    model_config = CrossModalConfig(
        gene_input_dim=gene.shape[1],
        image_input_dim=image.shape[1],
        hidden_dim=config.hidden_dim,
        output_dim=config.output_dim,
        dropout=config.dropout,
    )
    model = CrossModalModel(model_config).to(device)
    decoder = GeneReconstructionDecoder(config.output_dim, expression_store.matrix.shape[1]).to(device)
    optimizer = torch.optim.Adam(
        [*model.parameters(), *decoder.parameters()],
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    names = (
        "loss", "clip_loss", "reconstruction_loss", "geometry_loss",
        "variance_loss", "covariance_loss", "paired_cosine", "retrieval_accuracy",
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", *[f"train_{name}" for name in names], *[f"val_{name}" for name in names]])
        for epoch in range(1, config.num_epochs + 1):
            train_metrics = run_epoch(
                model, decoder, gene, image, expression_store, geometry,
                train_idx, config, rng, True, optimizer, epoch,
            )
            val_metrics = run_epoch(
                model, decoder, gene, image, expression_store, geometry,
                val_idx, config, rng, False, epoch=epoch,
            )
            logger.info(
                "epoch %d/%d | train %.4f val %.4f | val clip %.4f recon %.4f geometry %.4f retrieval %.4f",
                epoch, config.num_epochs, train_metrics["loss"], val_metrics["loss"],
                val_metrics["clip_loss"], val_metrics["reconstruction_loss"],
                val_metrics["geometry_loss"], val_metrics["retrieval_accuracy"],
            )
            writer.writerow([epoch, *[train_metrics[name] for name in names], *[val_metrics[name] for name in names]])
            file.flush()
            metrics = {"train": train_metrics, "val": val_metrics}
            save_checkpoint(
                checkpoint_dir / "last.pt", model, decoder, optimizer, config,
                epoch, metrics, expression_store.matrix.shape[1],
                feature_genes_path, geometry_path,
            )
            if val_metrics["loss"] < best_val_loss - config.min_delta:
                best_val_loss = val_metrics["loss"]
                epochs_without_improvement = 0
                save_checkpoint(
                    checkpoint_dir / "best.pt", model, decoder, optimizer, config,
                    epoch, metrics, expression_store.matrix.shape[1],
                    feature_genes_path, geometry_path,
                )
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    best = torch.load(checkpoint_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    gene_projected, image_projected = export_embeddings(model, gene, image)
    split = np.full(len(section_ids), "train", dtype=object)
    split[val_idx] = "val"
    split[test_idx] = "test"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        predictions_path,
        gene_projected=gene_projected,
        image_projected=image_projected,
        barcodes=barcodes,
        section_ids=section_ids,
        split=split,
        image_key=np.asarray(config.image_key),
        objective=np.asarray("rich_reconstruction_gbssa"),
        gene_source=np.asarray(str(config.gene_npz_path)),
        gene_array_key=np.asarray(config.gene_array_key),
        feature_genes_path=np.asarray(str(feature_genes_path)),
        geometry_path=np.asarray(str(geometry_path)),
        best_epoch=np.asarray(best["epoch"], dtype=np.int64),
        best_val_loss=np.asarray(best_val_loss, dtype=np.float64),
    )
    logger.info("Exported rich alignment to %s", predictions_path)


if __name__ == "__main__":
    main()
