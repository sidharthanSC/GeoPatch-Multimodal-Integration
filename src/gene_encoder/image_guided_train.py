"""Fine-tune the rich all-gene encoder with image-guided KNN local identities.

The gene encoder remains a reusable expression model. Fixed external image features
guide its exported embedding through symmetric multi-positive InfoNCE: an anchor and
its six direct spatial neighbors are positives, while other gathered spots from the
same section are negatives. The original same-spot gene BYOL and expression-preserving
losses remain active.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.datasets.dlpfc import DlpfcDataset
from src.gene_encoder.image_guided import (
    ImageProjectionHead,
    build_local_identity_batch,
    clamp_logit_scale,
    initial_logit_scale,
    neighborhood_cross_modal_loss,
)
from src.gene_encoder.model import (
    GeneEncoder,
    GeneEncoderConfig,
    GeneReconstructionDecoder,
    build_byol_learner,
)
from src.gene_encoder.rich_data import (
    SparseExpressionStore,
    SparseMaskedView,
    build_global_neighbor_rows,
    fit_expression_geometry,
    load_sparse_expression,
)
from src.gene_encoder.rich_train import (
    OnlineFeatureCapture,
    RichTrainConfig,
    _auxiliary_losses,
    _validate_view,
    export_sparse_multistage,
)
from src.gene_encoder.train import default_device, three_way_split

logger = logging.getLogger(__name__)


@dataclass
class ImageGuidedTrainConfig(RichTrainConfig):
    """Configuration for end-to-end image-guided rich gene fine-tuning."""

    run_name: str = "rich_gene_v3_all33538_imgk6_seed0"
    hidden_dims: tuple[int, ...] = (4096, 1024, 512)
    spatial_weight: float = 0.1
    batch_size: int = 4
    num_epochs: int = 50
    patience: int = 5
    checkpoint_every: int = 5

    image_key: str = "img_emb"
    image_hidden_dim: int = 256
    cross_modal_weight: float = 1.0
    image_lr: float = 3e-4
    initial_checkpoint: Path = Path(
        "outputs/gene_encoder/checkpoints/rich_gene_v2_all33538_seed0/best.pt"
    )
    max_batches_per_epoch: int | None = None


def load_image_features(
    dataset: DlpfcDataset,
    store: SparseExpressionStore,
    image_key: str,
) -> np.ndarray:
    """Load fixed image vectors in the sparse expression store's exact row order."""
    blocks = []
    section_blocks = []
    barcode_blocks = []
    for section_id in dataset.section_ids():
        adata = dataset.get_section(section_id)
        if image_key not in adata.obsm:
            raise KeyError(f"Section {section_id} lacks adata.obsm[{image_key!r}]")
        values = np.asarray(adata.obsm[image_key], dtype=np.float32)
        if values.ndim != 2 or len(values) != adata.n_obs:
            raise ValueError(f"Malformed {image_key} array in section {section_id}")
        blocks.append(values)
        section_blocks.append(np.full(adata.n_obs, str(section_id), dtype=object))
        barcode_blocks.append(np.asarray(adata.obs_names.astype(str)))
    section_ids = np.concatenate(section_blocks)
    barcodes = np.concatenate(barcode_blocks)
    if not np.array_equal(section_ids.astype(str), store.section_ids.astype(str)):
        raise ValueError("Image section identities do not match expression rows")
    if not np.array_equal(barcodes.astype(str), store.barcodes.astype(str)):
        raise ValueError("Image barcode identities do not match expression rows")
    image = np.concatenate(blocks)
    if not np.isfinite(image).all():
        raise ValueError(f"{image_key} contains non-finite values")
    return image


def section_anchor_batches(
    indices: np.ndarray,
    section_ids: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    shuffle: bool,
) -> list[np.ndarray]:
    """Build section-pure anchor batches so every gathered non-positive is valid."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    batches = []
    indexed_sections = section_ids[np.asarray(indices, dtype=np.int64)].astype(str)
    for section_id in np.unique(indexed_sections):
        rows = np.asarray(indices[indexed_sections == section_id], dtype=np.int64).copy()
        if shuffle:
            rng.shuffle(rows)
        batches.extend(rows[start : start + batch_size] for start in range(0, len(rows), batch_size))
    if shuffle:
        rng.shuffle(batches)
    return batches


def run_epoch(
    learner,
    encoder: GeneEncoder,
    decoder: GeneReconstructionDecoder,
    image_head: ImageProjectionHead,
    logit_scale: torch.Tensor,
    capture: OnlineFeatureCapture,
    view_a: SparseMaskedView,
    view_b: SparseMaskedView,
    store: SparseExpressionStore,
    image_embeddings: torch.Tensor,
    geometry_scores: torch.Tensor,
    neighbor_rows: np.ndarray,
    indices: np.ndarray,
    config: ImageGuidedTrainConfig,
    rng: np.random.Generator,
    train: bool,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    epoch: int = 0,
) -> dict[str, float]:
    learner.train(train)
    decoder.train(train)
    image_head.train(train)
    names = (
        "total_loss",
        "byol_loss",
        "cross_modal_loss",
        "gene_to_image_loss",
        "image_to_gene_loss",
        "reconstruction_loss",
        "variance_loss",
        "covariance_loss",
        "geometry_loss",
        "spatial_loss",
        "embed_std",
        "positive_cosine",
        "negative_cosine",
        "neighborhood_retrieval",
        "temperature",
        "mean_gallery_size",
    )
    totals = {name: 0.0 for name in names}
    n_anchors = 0
    batches = section_anchor_batches(
        indices, store.section_ids, config.batch_size, rng, train
    )
    if config.max_batches_per_epoch is not None:
        batches = batches[: config.max_batches_per_epoch]

    device = image_embeddings.device
    for step, anchors in enumerate(batches, start=1):
        gathered, positives = build_local_identity_batch(anchors, neighbor_rows)
        gathered_tensor = torch.as_tensor(gathered, dtype=torch.long)
        capture.clear()
        with torch.set_grad_enabled(train):
            with torch.autocast(
                device_type=device.type,
                enabled=config.amp and device.type == "cuda",
            ):
                byol_loss = learner(gathered_tensor)
                _validate_view(view_a, gathered)
                _validate_view(view_b, gathered)
                features = capture.consume(len(gathered))
                geometry_target = geometry_scores[gathered_tensor].to(device)
                auxiliary = _auxiliary_losses(
                    features,
                    decoder,
                    view_a,
                    view_b,
                    geometry_target,
                    encoder,
                    store,
                    gathered,
                    neighbor_rows,
                    rng,
                    config,
                )
                first, second = features["embedding"].chunk(2, dim=0)
                gene_gallery = (first + second) / 2.0
                image_gallery = image_head(image_embeddings[gathered])
                alignment = neighborhood_cross_modal_loss(
                    gene_gallery,
                    image_gallery,
                    len(anchors),
                    positives,
                    logit_scale,
                )
                total_loss = (
                    byol_loss
                    + config.cross_modal_weight * alignment.total
                    + config.reconstruction_weight * auxiliary["reconstruction_loss"]
                    + config.variance_weight * auxiliary["variance_loss"]
                    + config.covariance_weight * auxiliary["covariance_loss"]
                    + config.geometry_weight * auxiliary["geometry_loss"]
                    + config.spatial_weight * auxiliary["spatial_loss"]
                )

        if train:
            if optimizer is None:
                raise ValueError("optimizer is required for training")
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and scaler.is_enabled():
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()
            learner.update_moving_average()
            clamp_logit_scale(logit_scale)

        values = {
            "total_loss": total_loss,
            "byol_loss": byol_loss,
            "cross_modal_loss": alignment.total,
            "gene_to_image_loss": alignment.gene_to_image,
            "image_to_gene_loss": alignment.image_to_gene,
            "reconstruction_loss": auxiliary["reconstruction_loss"],
            "variance_loss": auxiliary["variance_loss"],
            "covariance_loss": auxiliary["covariance_loss"],
            "geometry_loss": auxiliary["geometry_loss"],
            "spatial_loss": auxiliary["spatial_loss"],
            "embed_std": auxiliary["embed_std"],
            "positive_cosine": alignment.positive_cosine,
            "negative_cosine": alignment.negative_cosine,
            "neighborhood_retrieval": alignment.neighborhood_retrieval,
            "temperature": logit_scale.detach().exp().reciprocal(),
            "mean_gallery_size": total_loss.new_tensor(float(len(gathered))),
        }
        weight = len(anchors)
        for name in names:
            totals[name] += float(values[name].detach().item()) * weight
        n_anchors += weight
        if train and config.log_every and step % config.log_every == 0:
            logger.info(
                "epoch %d step %d/%d | total %.4f cross %.4f pos %.4f neg %.4f retrieval %.4f std %.4f",
                epoch,
                step,
                len(batches),
                *[
                    float(values[name].detach())
                    for name in (
                        "total_loss", "cross_modal_loss", "positive_cosine",
                        "negative_cosine", "neighborhood_retrieval", "embed_std",
                    )
                ],
            )
    return {name: value / max(n_anchors, 1) for name, value in totals.items()}


def save_checkpoint(
    path: Path,
    learner,
    decoder: GeneReconstructionDecoder,
    image_head: ImageProjectionHead,
    logit_scale: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    config: ImageGuidedTrainConfig,
    encoder_config: GeneEncoderConfig,
    epoch: int,
    metrics: dict[str, Any],
    feature_count: int,
    explained_variance: float,
    include_training_state: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "objective": "rich_gene_byol_image_guided_k6_multi_positive_infonce",
        "encoder": learner.net.state_dict(),
        "decoder": decoder.state_dict(),
        "image_head": image_head.state_dict(),
        "logit_scale": logit_scale.detach().cpu(),
        "train_config": asdict(config),
        "encoder_config": asdict(encoder_config),
        "epoch": epoch,
        "metrics": metrics,
        "feature_count": feature_count,
        "geometry_explained_variance": explained_variance,
        "resumable": include_training_state,
    }
    if include_training_state:
        payload["byol_state_dict"] = learner.state_dict()
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path)


def run_image_guided_training(config: ImageGuidedTrainConfig) -> Path:
    """Train one immutable image-guided rich gene run and return its NPZ path."""
    if config.embedding_dim != config.geometry_dim and config.expression_residual:
        raise ValueError("Frozen expression residual requires embedding_dim == geometry_dim")
    if len(set(config.hidden_dims)) != len(config.hidden_dims):
        raise ValueError("hidden_dims must be unique for stage-key export")
    if config.cross_modal_weight <= 0:
        raise ValueError("cross_modal_weight must be positive")

    checkpoint_dir = config.output_dir / "checkpoints" / config.run_name
    metrics_path = config.output_dir / "metrics" / f"{config.run_name}_metrics.csv"
    predictions_path = config.output_dir / "predictions" / f"{config.run_name}_embeddings.npz"
    preprocessing_path = config.output_dir / "preprocessing" / f"{config.run_name}_geometry.npz"
    genes_path = config.output_dir / "preprocessing" / f"{config.run_name}_genes.json"
    expected = (checkpoint_dir, metrics_path, predictions_path, preprocessing_path, genes_path)
    if any(path.exists() for path in expected):
        raise FileExistsError(f"Refusing to overwrite image-guided run {config.run_name}")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    rng = np.random.default_rng(config.seed)
    device = torch.device(config.device)
    dataset = DlpfcDataset.from_checkpoint(config.checkpoint_path)
    genes = (
        json.loads(config.feature_genes_path.read_text(encoding="utf-8"))
        if config.feature_genes_path is not None
        else None
    )
    with config.spatial_graph_path.open("rb") as file:
        spatial_graphs = pickle.load(file)
    store = load_sparse_expression(dataset, genes)
    neighbor_rows = build_global_neighbor_rows(dataset, spatial_graphs)
    if neighbor_rows.shape != (store.matrix.shape[0], 6):
        raise ValueError(f"Expected a fixed K6 graph, observed {neighbor_rows.shape}")
    image_np = load_image_features(dataset, store, config.image_key)
    logger.info(
        "Loaded %d spots x %d genes and %d-d %s features",
        store.matrix.shape[0],
        store.matrix.shape[1],
        image_np.shape[1],
        config.image_key,
    )
    del dataset
    gc.collect()

    geometry_np, geometry_components, explained_variance = fit_expression_geometry(
        store.matrix, config.geometry_dim, config.seed
    )
    preprocessing_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        preprocessing_path,
        scores=geometry_np,
        components=geometry_components,
        explained_variance_ratio=explained_variance,
        section_ids=store.section_ids,
        barcodes=store.barcodes,
    )
    genes_path.write_text(
        json.dumps(store.gene_names.astype(str).tolist(), indent=2), encoding="utf-8"
    )
    geometry_scores = torch.from_numpy(geometry_np)
    train_idx, val_idx, test_idx = three_way_split(
        store.section_ids, config.val_fraction, config.test_fraction, config.seed
    )

    encoder_config = GeneEncoderConfig(
        input_dim=store.matrix.shape[1],
        hidden_dims=config.hidden_dims,
        embedding_dim=config.embedding_dim,
        projection_dim=config.projection_dim,
        projection_hidden_dim=config.projection_hidden_dim,
        dropout=config.dropout,
        ema_decay=config.ema_decay,
        activation="gelu",
        expression_residual=config.expression_residual,
        freeze_expression_residual=config.freeze_expression_residual,
    )
    encoder = GeneEncoder(encoder_config).to(device)
    if config.expression_residual:
        encoder.initialize_expression_residual(torch.from_numpy(geometry_components).to(device))
    decoder = GeneReconstructionDecoder(config.embedding_dim, store.matrix.shape[1]).to(device)
    initial = torch.load(config.initial_checkpoint, map_location=device, weights_only=False)
    initial_encoder_config = initial.get("encoder_config", {})
    expected_shape = (store.matrix.shape[1], config.hidden_dims, config.embedding_dim)
    observed_shape = (
        initial_encoder_config.get("input_dim"),
        tuple(initial_encoder_config.get("hidden_dims", ())),
        initial_encoder_config.get("embedding_dim"),
    )
    if observed_shape != expected_shape:
        raise ValueError(
            f"Initial encoder architecture {observed_shape} does not match {expected_shape}"
        )
    encoder.load_state_dict(initial["encoder"])
    decoder.load_state_dict(initial["decoder"])
    logger.info("Initialized trainable gene encoder and decoder from %s", config.initial_checkpoint)

    view_a = SparseMaskedView(store, device, config.mask_rate, config.noise_std_fraction)
    view_b = SparseMaskedView(store, device, config.mask_rate, config.noise_std_fraction)
    learner = build_byol_learner(encoder, encoder_config, view_a, view_b, device)
    capture = OnlineFeatureCapture(learner, encoder)
    image_head = ImageProjectionHead(
        image_np.shape[1], config.image_hidden_dim, config.embedding_dim, config.dropout
    ).to(device)
    logit_scale = torch.nn.Parameter(
        initial_logit_scale().detach().to(device)
    )
    image_embeddings = torch.from_numpy(image_np).to(device)
    optimizer = torch.optim.Adam(
        [
            {
                "params": [parameter for parameter in learner.parameters() if parameter.requires_grad],
                "lr": config.lr,
            },
            {"params": decoder.parameters(), "lr": config.lr},
            {"params": image_head.parameters(), "lr": config.image_lr},
            {"params": [logit_scale], "lr": config.image_lr},
        ],
        weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=config.amp and device.type == "cuda"
    )

    metric_names = (
        "total_loss", "byol_loss", "cross_modal_loss", "gene_to_image_loss",
        "image_to_gene_loss", "reconstruction_loss", "variance_loss",
        "covariance_loss", "geometry_loss", "spatial_loss", "embed_std",
        "positive_cosine", "negative_cosine", "neighborhood_retrieval",
        "temperature", "mean_gallery_size",
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    final_epoch = 0
    final_metrics: dict[str, Any] | None = None
    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["epoch", *[f"train_{name}" for name in metric_names], *[f"val_{name}" for name in metric_names]]
        )
        for epoch in range(1, config.num_epochs + 1):
            train_metrics = run_epoch(
                learner, encoder, decoder, image_head, logit_scale, capture,
                view_a, view_b, store, image_embeddings, geometry_scores,
                neighbor_rows, train_idx, config, rng, True, optimizer, scaler, epoch,
            )
            val_metrics = run_epoch(
                learner, encoder, decoder, image_head, logit_scale, capture,
                view_a, view_b, store, image_embeddings, geometry_scores,
                neighbor_rows, val_idx, config, rng, False, epoch=epoch,
            )
            logger.info(
                "epoch %d/%d | train %.4f val %.4f | val cross %.4f gap %.4f retrieval %.4f std %.4f",
                epoch,
                config.num_epochs,
                train_metrics["total_loss"],
                val_metrics["total_loss"],
                val_metrics["cross_modal_loss"],
                val_metrics["positive_cosine"] - val_metrics["negative_cosine"],
                val_metrics["neighborhood_retrieval"],
                val_metrics["embed_std"],
            )
            writer.writerow(
                [epoch, *[train_metrics[name] for name in metric_names], *[val_metrics[name] for name in metric_names]]
            )
            file.flush()
            metrics = {"train": train_metrics, "val": val_metrics}
            final_epoch = epoch
            final_metrics = metrics
            if epoch % config.checkpoint_every == 0:
                save_checkpoint(
                    checkpoint_dir / "last.pt", learner, decoder, image_head,
                    logit_scale, optimizer, config, encoder_config, epoch, metrics,
                    store.matrix.shape[1], float(explained_variance.sum()), True,
                )
            if val_metrics["total_loss"] < best_val_loss - config.min_delta:
                best_val_loss = val_metrics["total_loss"]
                epochs_without_improvement = 0
                save_checkpoint(
                    checkpoint_dir / "best.pt", learner, decoder, image_head,
                    logit_scale, optimizer, config, encoder_config, epoch, metrics,
                    store.matrix.shape[1], float(explained_variance.sum()), False,
                )
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                logger.info("Early stopping after %d epochs without improvement", config.patience)
                break

    save_checkpoint(
        checkpoint_dir / "last.pt", learner, decoder, image_head, logit_scale,
        optimizer, config, encoder_config, final_epoch, final_metrics or {},
        store.matrix.shape[1], float(explained_variance.sum()), True,
    )
    capture.close()
    best = torch.load(checkpoint_dir / "best.pt", map_location=device, weights_only=False)
    encoder.load_state_dict(best["encoder"])
    image_head.load_state_dict(best["image_head"])
    exported = export_sparse_multistage(encoder, store, device)
    image_head.eval()
    image_blocks = []
    with torch.no_grad():
        for start in range(0, len(image_embeddings), 4096):
            image_blocks.append(
                image_head(image_embeddings[start : start + 4096]).cpu().numpy().astype(np.float32)
            )
    projected_image = np.concatenate(image_blocks)
    split = np.full(store.matrix.shape[0], "train", dtype=object)
    split[val_idx] = "val"
    split[test_idx] = "test"
    stage_arrays = {
        f"gene_stage_{width}": exported[f"encoder_stage_{index}"]
        for index, width in enumerate(config.hidden_dims, start=1)
    }
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        predictions_path,
        embeddings=exported["embedding"],
        gene_embedding_128=exported["embedding"],
        image_projected=projected_image,
        **stage_arrays,
        barcodes=store.barcodes,
        section_ids=store.section_ids,
        split=split,
        schema_version=np.asarray(1, dtype=np.int64),
        objective=np.asarray("rich_gene_byol_image_guided_k6_multi_positive_infonce"),
        image_key=np.asarray(config.image_key),
        hidden_dims=np.asarray(config.hidden_dims, dtype=np.int64),
        embedding_dim=np.asarray(config.embedding_dim, dtype=np.int64),
        input_dim=np.asarray(store.matrix.shape[1], dtype=np.int64),
        feature_genes_path=np.asarray(str(config.feature_genes_path or genes_path)),
        geometry_path=np.asarray(str(preprocessing_path)),
        spatial_graph_path=np.asarray(str(config.spatial_graph_path)),
        initial_checkpoint=np.asarray(str(config.initial_checkpoint)),
        best_epoch=np.asarray(best["epoch"], dtype=np.int64),
        best_val_loss=np.asarray(best_val_loss, dtype=np.float64),
    )
    logger.info("Exported image-guided rich embeddings to %s", predictions_path)
    return predictions_path


def build_arg_parser() -> argparse.ArgumentParser:
    defaults = ImageGuidedTrainConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "checkpoint_path", "feature_genes_path", "spatial_graph_path", "output_dir",
        "initial_checkpoint",
    ):
        parser.add_argument(
            f"--{name.replace('_', '-')}", type=Path, default=getattr(defaults, name)
        )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--image-key", default=defaults.image_key)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=list(defaults.hidden_dims))
    for name in (
        "mask_rate", "noise_std_fraction", "reconstruction_weight",
        "reconstruction_nonzero_weight", "variance_weight", "final_variance_gamma",
        "stage_variance_gamma", "covariance_weight", "geometry_weight", "spatial_weight",
        "cross_modal_weight", "lr", "image_lr", "weight_decay", "val_fraction",
        "test_fraction", "min_delta",
    ):
        parser.add_argument(
            f"--{name.replace('_', '-')}", type=float, default=getattr(defaults, name)
        )
    for name in (
        "embedding_dim", "projection_dim", "projection_hidden_dim", "image_hidden_dim",
        "covariance_max_dimensions", "geometry_dim", "batch_size", "num_epochs", "seed",
        "patience", "log_every", "checkpoint_every", "max_batches_per_epoch",
    ):
        parser.add_argument(
            f"--{name.replace('_', '-')}", type=int, default=getattr(defaults, name)
        )
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument("--ema-decay", type=float, default=defaults.ema_decay)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--no-expression-residual", action="store_true")
    parser.add_argument("--train-expression-residual", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = vars(build_arg_parser().parse_args())
    args["hidden_dims"] = tuple(args["hidden_dims"])
    args["expression_residual"] = not args.pop("no_expression_residual")
    args["freeze_expression_residual"] = not args.pop("train_expression_residual")
    args["amp"] = not args.pop("no_amp")
    run_image_guided_training(ImageGuidedTrainConfig(**args))


if __name__ == "__main__":
    main()
