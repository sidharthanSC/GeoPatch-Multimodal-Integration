"""Train information-preserving gene BYOL encoders from sparse expression.

This superseding trainer keeps expression sparse on CPU, uses two independent masked
views, and augments BYOL with masked reconstruction, stage-wise variance/covariance,
expression-SVD geometry preservation, and weak spatial-neighbor consistency.
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
import torch.nn.functional as F

from src.datasets.dlpfc import DlpfcDataset
from src.gene_encoder.model import (
    GeneEncoder,
    GeneEncoderConfig,
    GeneReconstructionDecoder,
    build_byol_learner,
    cosine_geometry_regularization,
    covariance_regularization,
    masked_reconstruction_loss,
    variance_regularization,
)
from src.gene_encoder.rich_data import (
    SparseExpressionStore,
    SparseMaskedView,
    build_global_neighbor_rows,
    fit_expression_geometry,
    load_sparse_expression,
)
from src.gene_encoder.train import default_device, three_way_split

logger = logging.getLogger(__name__)


@dataclass
class RichTrainConfig:
    checkpoint_path: Path = Path("checkpoints/dlpfc.pkl")
    feature_genes_path: Path | None = None
    spatial_graph_path: Path = Path("outputs/gene_encoder/spatial_knn_graph_gbssa.pkl")
    output_dir: Path = Path("outputs/gene_encoder")
    run_name: str = "rich_gene_v1"

    hidden_dims: tuple[int, ...] = (4096, 2048, 1024, 512, 256)
    embedding_dim: int = 128
    projection_dim: int = 128
    projection_hidden_dim: int = 256
    dropout: float = 0.1
    ema_decay: float = 0.996
    expression_residual: bool = True
    freeze_expression_residual: bool = True

    mask_rate: float = 0.3
    noise_std_fraction: float = 0.05
    reconstruction_weight: float = 2.0
    reconstruction_nonzero_weight: float = 5.0
    variance_weight: float = 5.0
    final_variance_gamma: float = 1.0
    stage_variance_gamma: float = 0.5
    covariance_weight: float = 1.0
    covariance_max_dimensions: int = 256
    geometry_weight: float = 1.0
    geometry_dim: int = 128
    spatial_weight: float = 0.1

    batch_size: int = 32
    num_epochs: int = 200
    lr: float = 1e-5
    weight_decay: float = 1e-4
    val_fraction: float = 0.2
    test_fraction: float = 0.0
    seed: int = 0
    patience: int = 25
    min_delta: float = 1e-4
    device: str = field(default_factory=default_device)
    log_every: int = 100
    amp: bool = True
    checkpoint_every: int = 10


class OnlineFeatureCapture:
    """Capture exact online hidden/final representations from one BYOL forward."""

    def __init__(self, learner, encoder: GeneEncoder) -> None:
        self.stage_outputs: dict[str, torch.Tensor] = {}
        self.final_output: torch.Tensor | None = None
        self.handles = []
        for index, block in enumerate(encoder.hidden_blocks, start=1):
            self.handles.append(
                block.register_forward_hook(self._stage_hook(f"encoder_stage_{index}"))
            )
        self.handles.append(learner.online_encoder.register_forward_hook(self._final_hook))

    def _stage_hook(self, key: str):
        def hook(_module, _inputs, output):
            self.stage_outputs[key] = output

        return hook

    def _final_hook(self, _module, _inputs, output):
        self.final_output = output[1]

    def clear(self) -> None:
        self.stage_outputs = {}
        self.final_output = None

    def consume(self, expected_batch: int) -> dict[str, torch.Tensor]:
        if self.final_output is None:
            raise RuntimeError("BYOL online representation was not captured")
        outputs = {**self.stage_outputs, "embedding": self.final_output}
        if any(value.shape[0] != 2 * expected_batch for value in outputs.values()):
            raise RuntimeError("Captured BYOL feature batch has an unexpected shape")
        self.clear()
        return outputs

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def _validate_view(view: SparseMaskedView, expected_indices: np.ndarray) -> None:
    if view.last_indices is None or not np.array_equal(view.last_indices, expected_indices):
        raise RuntimeError("Sparse BYOL view cache does not match the current batch")


def _auxiliary_losses(
    features: dict[str, torch.Tensor],
    decoder: GeneReconstructionDecoder,
    view_a: SparseMaskedView,
    view_b: SparseMaskedView,
    geometry_target: torch.Tensor,
    encoder: GeneEncoder,
    store: SparseExpressionStore,
    batch_indices: np.ndarray,
    neighbor_rows: np.ndarray,
    rng: np.random.Generator,
    config: RichTrainConfig,
) -> dict[str, torch.Tensor]:
    batch_size = len(batch_indices)
    split_features = {
        key: value.chunk(2, dim=0) for key, value in features.items()
    }
    embedding_a, embedding_b = split_features["embedding"]
    reconstruction = (
        masked_reconstruction_loss(
            decoder(embedding_a),
            view_a.last_target,
            view_a.last_mask,
            config.reconstruction_nonzero_weight,
        )
        + masked_reconstruction_loss(
            decoder(embedding_b),
            view_b.last_target,
            view_b.last_mask,
            config.reconstruction_nonzero_weight,
        )
    ) / 2.0

    variance_terms = []
    covariance_terms = []
    geometry_terms = []
    for key, (first, second) in split_features.items():
        gamma = (
            config.final_variance_gamma
            if key == "embedding"
            else config.stage_variance_gamma
        )
        variance_terms.append(
            (
                variance_regularization(first, gamma=gamma)
                + variance_regularization(second, gamma=gamma)
            )
            / 2.0
        )
        combined = torch.cat([first, second], dim=0)
        covariance_terms.append(
            covariance_regularization(
                combined, max_dimensions=config.covariance_max_dimensions
            )
        )
        geometry_terms.append(
            (
                cosine_geometry_regularization(first, geometry_target)
                + cosine_geometry_regularization(second, geometry_target)
            )
            / 2.0
        )

    choices = rng.integers(0, neighbor_rows.shape[1], size=batch_size)
    selected_neighbors = neighbor_rows[batch_indices, choices]
    neighbor_expression = torch.from_numpy(store.rows(selected_neighbors)).to(
        embedding_a.device
    )
    neighbor_embedding = encoder(neighbor_expression)
    anchor_embedding = (embedding_a + embedding_b) / 2.0
    spatial = (1.0 - (F.normalize(anchor_embedding, dim=1) * F.normalize(neighbor_embedding, dim=1)).sum(dim=1)).mean()

    return {
        "reconstruction_loss": reconstruction,
        "variance_loss": torch.stack(variance_terms).mean(),
        "covariance_loss": torch.stack(covariance_terms).mean(),
        "geometry_loss": torch.stack(geometry_terms).mean(),
        "spatial_loss": spatial,
        "embed_std": torch.cat([embedding_a, embedding_b], dim=0).std(dim=0).mean(),
    }


def run_epoch(
    learner,
    encoder: GeneEncoder,
    decoder: GeneReconstructionDecoder,
    capture: OnlineFeatureCapture,
    view_a: SparseMaskedView,
    view_b: SparseMaskedView,
    store: SparseExpressionStore,
    geometry_scores: torch.Tensor,
    neighbor_rows: np.ndarray,
    indices: np.ndarray,
    config: RichTrainConfig,
    rng: np.random.Generator,
    train: bool,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    epoch: int = 0,
) -> dict[str, float]:
    learner.train(train)
    decoder.train(train)
    order = indices.copy()
    if train:
        rng.shuffle(order)
    names = (
        "total_loss",
        "byol_loss",
        "reconstruction_loss",
        "variance_loss",
        "covariance_loss",
        "geometry_loss",
        "spatial_loss",
        "embed_std",
    )
    totals = {name: 0.0 for name in names}
    n_batches = 0

    for start in range(0, len(order), config.batch_size):
        batch_indices = order[start : start + config.batch_size]
        if len(batch_indices) < 2:
            continue
        index_tensor = torch.as_tensor(batch_indices, dtype=torch.long)
        capture.clear()
        with torch.set_grad_enabled(train):
            with torch.autocast(
                device_type=torch.device(config.device).type,
                enabled=config.amp and torch.device(config.device).type == "cuda",
            ):
                byol_loss = learner(index_tensor)
                _validate_view(view_a, batch_indices)
                _validate_view(view_b, batch_indices)
                features = capture.consume(len(batch_indices))
                geometry_target = geometry_scores[index_tensor].to(torch.device(config.device))
                auxiliary = _auxiliary_losses(
                    features,
                    decoder,
                    view_a,
                    view_b,
                    geometry_target,
                    encoder,
                    store,
                    batch_indices,
                    neighbor_rows,
                    rng,
                    config,
                )
                total_loss = (
                    byol_loss
                    + config.reconstruction_weight * auxiliary["reconstruction_loss"]
                    + config.variance_weight * auxiliary["variance_loss"]
                    + config.covariance_weight * auxiliary["covariance_loss"]
                    + config.geometry_weight * auxiliary["geometry_loss"]
                    + config.spatial_weight * auxiliary["spatial_loss"]
                )

        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and scaler.is_enabled():
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()
            learner.update_moving_average()

        values = {"total_loss": total_loss, "byol_loss": byol_loss, **auxiliary}
        for name in names:
            totals[name] += float(values[name].detach().item())
        n_batches += 1
        if train and config.log_every and n_batches % config.log_every == 0:
            logger.info(
                "epoch %d step %d | total %.4f byol %.4f recon %.4f var %.4f cov %.4f geom %.4f spatial %.4f std %.4f",
                epoch,
                n_batches,
                *[float(values[name].detach()) for name in names],
            )
    return {name: value / max(n_batches, 1) for name, value in totals.items()}


@torch.no_grad()
def export_sparse_multistage(
    encoder: GeneEncoder,
    store: SparseExpressionStore,
    device: torch.device,
    batch_size: int = 256,
) -> dict[str, np.ndarray]:
    encoder.eval()
    blocks: dict[str, list[np.ndarray]] = {}
    for start in range(0, store.matrix.shape[0], batch_size):
        rows = np.arange(start, min(start + batch_size, store.matrix.shape[0]))
        expression = torch.from_numpy(store.rows(rows)).to(device)
        for key, value in encoder.forward_features(expression).items():
            blocks.setdefault(key, []).append(value.cpu().numpy().astype(np.float32))
    return {key: np.concatenate(values) for key, values in blocks.items()}


def save_checkpoint(
    path: Path,
    learner,
    decoder: GeneReconstructionDecoder,
    optimizer: torch.optim.Optimizer,
    config: RichTrainConfig,
    encoder_config: GeneEncoderConfig,
    epoch: int,
    metrics: dict[str, Any],
    feature_count: int,
    explained_variance: float,
    include_training_state: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "encoder": learner.net.state_dict(),
        "decoder": decoder.state_dict(),
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


def build_arg_parser() -> argparse.ArgumentParser:
    defaults = RichTrainConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, default=defaults.checkpoint_path)
    parser.add_argument("--feature-genes-path", type=Path)
    parser.add_argument("--spatial-graph-path", type=Path, default=defaults.spatial_graph_path)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=list(defaults.hidden_dims))
    parser.add_argument("--embedding-dim", type=int, default=defaults.embedding_dim)
    parser.add_argument("--projection-dim", type=int, default=defaults.projection_dim)
    parser.add_argument("--projection-hidden-dim", type=int, default=defaults.projection_hidden_dim)
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument("--ema-decay", type=float, default=defaults.ema_decay)
    parser.add_argument("--no-expression-residual", action="store_true")
    parser.add_argument("--train-expression-residual", action="store_true")
    for name in (
        "mask_rate",
        "noise_std_fraction",
        "reconstruction_weight",
        "reconstruction_nonzero_weight",
        "variance_weight",
        "final_variance_gamma",
        "stage_variance_gamma",
        "covariance_weight",
        "geometry_weight",
        "spatial_weight",
        "lr",
        "weight_decay",
        "val_fraction",
        "test_fraction",
        "min_delta",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=float, default=getattr(defaults, name))
    parser.add_argument("--covariance-max-dimensions", type=int, default=defaults.covariance_max_dimensions)
    parser.add_argument("--geometry-dim", type=int, default=defaults.geometry_dim)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--num-epochs", type=int, default=defaults.num_epochs)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--patience", type=int, default=defaults.patience)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--log-every", type=int, default=defaults.log_every)
    parser.add_argument("--checkpoint-every", type=int, default=defaults.checkpoint_every)
    parser.add_argument("--no-amp", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = vars(build_arg_parser().parse_args())
    args["hidden_dims"] = tuple(args["hidden_dims"])
    args["expression_residual"] = not args.pop("no_expression_residual")
    args["freeze_expression_residual"] = not args.pop("train_expression_residual")
    args["amp"] = not args.pop("no_amp")
    config = RichTrainConfig(**args)
    if config.embedding_dim != config.geometry_dim and config.expression_residual:
        raise ValueError("Frozen expression residual requires embedding_dim == geometry_dim")
    if len(set(config.hidden_dims)) != len(config.hidden_dims):
        raise ValueError("hidden_dims must be unique for stage-key export")

    checkpoint_dir = config.output_dir / "checkpoints" / config.run_name
    metrics_path = config.output_dir / "metrics" / f"{config.run_name}_metrics.csv"
    predictions_path = config.output_dir / "predictions" / f"{config.run_name}_embeddings.npz"
    preprocessing_path = config.output_dir / "preprocessing" / f"{config.run_name}_geometry.npz"
    genes_path = config.output_dir / "preprocessing" / f"{config.run_name}_genes.json"
    if any(path.exists() for path in (checkpoint_dir, metrics_path, predictions_path, preprocessing_path, genes_path)):
        raise FileExistsError(f"Refusing to overwrite rich gene run {config.run_name}")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    rng = np.random.default_rng(config.seed)
    device = torch.device(config.device)

    dataset = DlpfcDataset.from_checkpoint(config.checkpoint_path)
    genes = None
    if config.feature_genes_path is not None:
        genes = json.loads(config.feature_genes_path.read_text(encoding="utf-8"))
    with config.spatial_graph_path.open("rb") as file:
        spatial_graphs = pickle.load(file)
    store = load_sparse_expression(dataset, genes)
    neighbor_rows = build_global_neighbor_rows(dataset, spatial_graphs)
    logger.info(
        "Loaded sparse expression: %d spots x %d genes, %.2f%% nonzero",
        *store.matrix.shape,
        100.0 * store.matrix.nnz / np.prod(store.matrix.shape),
    )
    del dataset
    gc.collect()

    geometry_scores_np, geometry_components, explained_variance = fit_expression_geometry(
        store.matrix, config.geometry_dim, config.seed
    )
    preprocessing_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        preprocessing_path,
        scores=geometry_scores_np,
        components=geometry_components,
        explained_variance_ratio=explained_variance,
        section_ids=store.section_ids,
        barcodes=store.barcodes,
    )
    genes_path.write_text(
        json.dumps(store.gene_names.astype(str).tolist(), indent=2), encoding="utf-8"
    )
    geometry_scores = torch.from_numpy(geometry_scores_np)

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
        encoder.initialize_expression_residual(
            torch.from_numpy(geometry_components).to(device)
        )
    view_a = SparseMaskedView(store, device, config.mask_rate, config.noise_std_fraction)
    view_b = SparseMaskedView(store, device, config.mask_rate, config.noise_std_fraction)
    learner = build_byol_learner(encoder, encoder_config, view_a, view_b, device)
    decoder = GeneReconstructionDecoder(config.embedding_dim, store.matrix.shape[1]).to(device)
    capture = OnlineFeatureCapture(learner, encoder)
    parameters = [parameter for parameter in learner.parameters() if parameter.requires_grad]
    parameters.extend(decoder.parameters())
    optimizer = torch.optim.Adam(parameters, lr=config.lr, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=config.amp and device.type == "cuda"
    )

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metric_names = (
        "total_loss",
        "byol_loss",
        "reconstruction_loss",
        "variance_loss",
        "covariance_loss",
        "geometry_loss",
        "spatial_loss",
        "embed_std",
    )
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    final_epoch = 0
    final_metrics = None
    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", *[f"train_{name}" for name in metric_names], *[f"val_{name}" for name in metric_names]])
        for epoch in range(1, config.num_epochs + 1):
            train_metrics = run_epoch(
                learner, encoder, decoder, capture, view_a, view_b, store,
                geometry_scores, neighbor_rows, train_idx, config, rng, True,
                optimizer, scaler, epoch,
            )
            val_metrics = run_epoch(
                learner, encoder, decoder, capture, view_a, view_b, store,
                geometry_scores, neighbor_rows, val_idx, config, rng, False, epoch=epoch,
            )
            logger.info(
                "epoch %d/%d | train total %.4f recon %.4f geom %.4f std %.4f | val total %.4f recon %.4f geom %.4f std %.4f",
                epoch, config.num_epochs,
                train_metrics["total_loss"], train_metrics["reconstruction_loss"],
                train_metrics["geometry_loss"], train_metrics["embed_std"],
                val_metrics["total_loss"], val_metrics["reconstruction_loss"],
                val_metrics["geometry_loss"], val_metrics["embed_std"],
            )
            writer.writerow([epoch, *[train_metrics[name] for name in metric_names], *[val_metrics[name] for name in metric_names]])
            file.flush()
            metrics = {"train": train_metrics, "val": val_metrics}
            final_epoch = epoch
            final_metrics = metrics
            if epoch % config.checkpoint_every == 0:
                save_checkpoint(
                    checkpoint_dir / "last.pt", learner, decoder, optimizer, config,
                    encoder_config, epoch, metrics, store.matrix.shape[1],
                    float(explained_variance.sum()), include_training_state=True,
                )
            if val_metrics["total_loss"] < best_val_loss - config.min_delta:
                best_val_loss = val_metrics["total_loss"]
                epochs_without_improvement = 0
                save_checkpoint(
                    checkpoint_dir / "best.pt", learner, decoder, optimizer, config,
                    encoder_config, epoch, metrics, store.matrix.shape[1],
                    float(explained_variance.sum()), include_training_state=False,
                )
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                logger.info("Early stopping after %d epochs without improvement", config.patience)
                break

    save_checkpoint(
        checkpoint_dir / "last.pt", learner, decoder, optimizer, config,
        encoder_config, final_epoch, final_metrics, store.matrix.shape[1],
        float(explained_variance.sum()), include_training_state=True,
    )

    capture.close()
    best = torch.load(checkpoint_dir / "best.pt", map_location=device, weights_only=False)
    encoder.load_state_dict(best["encoder"])
    exported = export_sparse_multistage(encoder, store, device)
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
        **stage_arrays,
        barcodes=store.barcodes,
        section_ids=store.section_ids,
        split=split,
        schema_version=np.asarray(3, dtype=np.int64),
        hidden_dims=np.asarray(config.hidden_dims, dtype=np.int64),
        embedding_dim=np.asarray(config.embedding_dim, dtype=np.int64),
        input_dim=np.asarray(store.matrix.shape[1], dtype=np.int64),
        feature_genes_path=np.asarray(str(config.feature_genes_path or genes_path)),
        geometry_path=np.asarray(str(preprocessing_path)),
        best_epoch=np.asarray(best["epoch"], dtype=np.int64),
        best_val_loss=np.asarray(best_val_loss, dtype=np.float64),
    )
    logger.info("Exported rich multistage embeddings to %s", predictions_path)


if __name__ == "__main__":
    main()
