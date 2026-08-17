"""Three-phase label-assisted image/gene representation learning.

Phases 1 and 2 train independent online/predictor/EMA-target projectors with
same-layer attraction and lower-weight different-layer repulsion. Phase 3 aligns the
two target-projector outputs using exact same-spot orthogonal Procrustes fitted only on
the fold's training spots. Despite BYOL-style EMA mechanics, phases 1 and 2 are
supervised metric learning because cortical-layer labels define their pairs.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from src.cross_modal.procrustes import (
    fit_graph_positive_procrustes,
    transform_procrustes,
)
from src.multimodal.representations import normalized_mean
from src.train.model import ProjectorConfig, SliceProjectionModel, WeightedSliceContrastiveLoss
from src.train.pairs import BalancedPairSampler


@dataclass(frozen=True)
class SupervisedMultiphaseConfig:
    dataset_manifest: Path = Path(
        "outputs/evaluation/20260813_checkpoint_audit_v1/data/dataset_manifest.parquet"
    )
    split_manifest: Path = Path(
        "outputs/splits/20260815_dlpfc_lodo_v1/split_manifest.parquet"
    )
    embeddings_npz: Path = Path(
        "outputs/evaluation/20260813_checkpoint_audit_v1/data/compact_embeddings.npz"
    )
    output_root: Path = Path("outputs/supervised_multiphase")
    run_name: str = "lodo_phase123_seed0_v1"
    folds: tuple[str, ...] = ()
    image_key: str = "img_emb"
    gene_key: str = "gene_emb"
    hidden_dim: int = 256
    output_dim: int = 128
    dropout: float = 0.1
    positive_weight: float = 5.0
    negative_weight: float = 1.0
    negative_margin: float = 0.0
    ema_decay: float = 0.996
    epochs: int = 60
    steps_per_epoch: int = 100
    batch_size: int = 256
    validation_pairs: int = 4096
    lr: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 8
    min_delta: float = 1e-4
    seed: int = 0
    device: str = "cuda"


class EMAMetricProjector(nn.Module):
    """Online projector/predictor with a stop-gradient EMA target projector."""

    def __init__(self, config: ProjectorConfig, ema_decay: float = 0.996) -> None:
        super().__init__()
        if not 0.0 < ema_decay < 1.0:
            raise ValueError("ema_decay must lie in (0, 1)")
        self.config = config
        self.ema_decay = ema_decay
        self.online = SliceProjectionModel(config)
        self.predictor = nn.Sequential(
            nn.Linear(config.projection_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.projection_dim),
        )
        self.target = copy.deepcopy(self.online)
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)

    def online_prediction(self, values: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.predictor(self.online(values)), dim=1)

    @torch.no_grad()
    def target_projection(self, values: torch.Tensor) -> torch.Tensor:
        return self.target(values)

    @torch.no_grad()
    def update_target(self) -> None:
        for online, target in zip(self.online.parameters(), self.target.parameters()):
            target.data.mul_(self.ema_decay).add_(online.data, alpha=1.0 - self.ema_decay)


def symmetric_ema_metric_loss(
    model: EMAMetricProjector,
    first: torch.Tensor,
    second: torch.Tensor,
    pair_kind: torch.Tensor,
    criterion: WeightedSliceContrastiveLoss,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average both online-to-target pair directions."""
    online_first = model.online_prediction(first)
    online_second = model.online_prediction(second)
    with torch.no_grad():
        target_first = model.target_projection(first)
        target_second = model.target_projection(second)
    forward_loss, forward_similarity = criterion(online_first, target_second, pair_kind)
    reverse_loss, reverse_similarity = criterion(online_second, target_first, pair_kind)
    return (forward_loss + reverse_loss) / 2.0, (
        forward_similarity + reverse_similarity
    ) / 2.0


def _fixed_pairs(labels: np.ndarray, count: int, seed: int) -> tuple[np.ndarray, ...]:
    sampler = BalancedPairSampler(labels, seed=seed)
    return sampler.sample_batch(count)


def _pair_metrics(similarity: np.ndarray, pair_kind: np.ndarray) -> dict[str, float]:
    positive = similarity[pair_kind == 0]
    negative = similarity[pair_kind == 1]
    return {
        "same_similarity": float(positive.mean()),
        "different_similarity": float(negative.mean()),
        "similarity_gap": float(positive.mean() - negative.mean()),
    }


@torch.no_grad()
def export_target_embeddings(
    model: EMAMetricProjector,
    values: torch.Tensor,
    batch_size: int = 4096,
) -> np.ndarray:
    model.eval()
    outputs = []
    for start in range(0, len(values), batch_size):
        outputs.append(model.target_projection(values[start : start + batch_size]).cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


def train_metric_phase(
    values: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    validation_idx: np.ndarray,
    config: SupervisedMultiphaseConfig,
    seed_offset: int,
) -> tuple[EMAMetricProjector, list[dict[str, float]], int]:
    """Fit one image or gene EMA metric phase without reading test labels."""
    device = torch.device(config.device)
    torch.manual_seed(config.seed + seed_offset)
    values_tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
    model_config = ProjectorConfig(
        input_dim=values.shape[1],
        hidden_dim=config.hidden_dim,
        projection_dim=config.output_dim,
        dropout=config.dropout,
    )
    model = EMAMetricProjector(model_config, config.ema_decay).to(device)
    criterion = WeightedSliceContrastiveLoss(
        config.positive_weight, config.negative_weight, config.negative_margin
    )
    optimizer = torch.optim.AdamW(
        [*model.online.parameters(), *model.predictor.parameters()],
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    train_sampler = BalancedPairSampler(labels[train_idx], config.seed + seed_offset)
    validation_pairs = _fixed_pairs(
        labels[validation_idx], config.validation_pairs, config.seed + 10_000 + seed_offset
    )
    history: list[dict[str, float]] = []
    best_gap = -math.inf
    best_state = None
    best_epoch = 0
    stale = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_losses = []
        train_similarities = []
        train_kinds = []
        for _ in range(config.steps_per_epoch):
            first, second, pair_kind = train_sampler.sample_batch(config.batch_size)
            first_global, second_global = train_idx[first], train_idx[second]
            pair_tensor = torch.as_tensor(pair_kind, dtype=torch.float32, device=device)
            loss, similarity = symmetric_ema_metric_loss(
                model,
                values_tensor[first_global],
                values_tensor[second_global],
                pair_tensor,
                criterion,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            model.update_target()
            train_losses.append(float(loss.detach()))
            train_similarities.append(similarity.detach().cpu().numpy())
            train_kinds.append(pair_kind)

        model.eval()
        val_first, val_second, val_kind = validation_pairs
        val_kind_tensor = torch.as_tensor(val_kind, dtype=torch.float32, device=device)
        with torch.no_grad():
            val_loss, val_similarity = symmetric_ema_metric_loss(
                model,
                values_tensor[validation_idx[val_first]],
                values_tensor[validation_idx[val_second]],
                val_kind_tensor,
                criterion,
            )
        train_metric = _pair_metrics(
            np.concatenate(train_similarities), np.concatenate(train_kinds)
        )
        val_metric = _pair_metrics(val_similarity.cpu().numpy(), val_kind)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "validation_loss": float(val_loss),
            **{f"train_{key}": value for key, value in train_metric.items()},
            **{f"validation_{key}": value for key, value in val_metric.items()},
        }
        history.append(row)
        if val_metric["similarity_gap"] > best_gap + config.min_delta:
            best_gap = val_metric["similarity_gap"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= config.patience:
            break
    if best_state is None:
        raise RuntimeError("Metric phase failed to select a checkpoint")
    model.load_state_dict(best_state)
    return model, history, best_epoch


def _load_inputs(
    config: SupervisedMultiphaseConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    dataset = pd.read_parquet(config.dataset_manifest).sort_values("global_row")
    split = pd.read_parquet(config.split_manifest)
    if not np.array_equal(dataset["global_row"].to_numpy(), np.arange(len(dataset))):
        raise ValueError("Dataset manifest global_row must be contiguous and ordered")
    with np.load(config.embeddings_npz, allow_pickle=True) as data:
        for key in (config.image_key, config.gene_key):
            if key not in data:
                raise KeyError(f"Missing embedding key {key}")
        image = np.asarray(data[config.image_key], dtype=np.float32)
        gene = np.asarray(data[config.gene_key], dtype=np.float32)
    if image.shape != gene.shape or image.shape[0] != len(dataset):
        raise ValueError("Embedding arrays do not match the dataset manifest")
    if not np.isfinite(image).all() or not np.isfinite(gene).all():
        raise ValueError("Embedding arrays contain non-finite values")
    return dataset, split, image, gene


def run_supervised_multiphase(config: SupervisedMultiphaseConfig) -> dict[str, object]:
    """Train all requested folds and export phase-wise embeddings and provenance."""
    run_dir = config.output_root / config.run_name
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {run_dir}")
    run_dir.mkdir(parents=True)
    dataset, split, image, gene = _load_inputs(config)
    available_folds = split["fold_id"].drop_duplicates().astype(str).tolist()
    folds = list(config.folds) if config.folds else available_folds
    unknown = sorted(set(folds) - set(available_folds))
    if unknown:
        raise ValueError(f"Unknown folds: {unknown}")
    labels = dataset["ground_truth"].astype(str).to_numpy()
    summaries = []

    for fold_number, fold_id in enumerate(folds):
        fold_dir = run_dir / fold_id
        fold_dir.mkdir()
        fold = split.loc[split["fold_id"].astype(str) == fold_id]
        role_by_row = dict(zip(fold["global_row"].astype(int), fold["role"].astype(str)))
        train_idx = np.asarray(sorted(row for row, role in role_by_row.items() if role == "train"))
        validation_idx = np.asarray(
            sorted(row for row, role in role_by_row.items() if role == "validation")
        )
        test_idx = np.asarray(sorted(row for row, role in role_by_row.items() if role == "test"))
        if set(train_idx) & set(validation_idx) or set(train_idx) & set(test_idx):
            raise ValueError(f"Fold {fold_id} contains overlapping roles")

        image_model, image_history, image_epoch = train_metric_phase(
            image, labels, train_idx, validation_idx, config, 100 * fold_number + 1
        )
        gene_model, gene_history, gene_epoch = train_metric_phase(
            gene, labels, train_idx, validation_idx, config, 100 * fold_number + 2
        )
        image_tensor = torch.as_tensor(image, dtype=torch.float32, device=config.device)
        gene_tensor = torch.as_tensor(gene, dtype=torch.float32, device=config.device)
        image_phase1 = export_target_embeddings(image_model, image_tensor)
        gene_phase2 = export_target_embeddings(gene_model, gene_tensor)

        empty_edges = np.empty((0, 2), dtype=np.int64)
        gene_map, image_map, singular_values = fit_graph_positive_procrustes(
            gene_phase2,
            image_phase1,
            train_idx,
            empty_edges,
            self_weight=1.0,
            neighbor_weight=0.0,
        )
        gene_phase3, image_phase3 = transform_procrustes(
            gene_phase2, image_phase1, gene_map, image_map
        )
        role = np.full(len(dataset), "excluded", dtype=object)
        for row, value in role_by_row.items():
            role[int(row)] = value
        np.savez_compressed(
            fold_dir / "embeddings.npz",
            image_raw=image,
            gene_raw=gene,
            image_phase1=image_phase1,
            gene_phase2=gene_phase2,
            image_phase3=image_phase3,
            gene_phase3=gene_phase3,
            bisector_phase12=normalized_mean(image_phase1, gene_phase2),
            bisector_phase3=normalized_mean(image_phase3, gene_phase3),
            barcodes=dataset["barcode"].astype(str).to_numpy(),
            section_ids=dataset["section_id"].astype(str).to_numpy(),
            donor_ids=dataset["donor_id"].astype(int).to_numpy(),
            labels=labels,
            role=role,
            fold_id=np.asarray(fold_id),
            image_key=np.asarray(config.image_key),
            gene_key=np.asarray(config.gene_key),
            phase3_objective=np.asarray("exact_spot_orthogonal_procrustes"),
        )
        torch.save(
            {
                "image_model": image_model.state_dict(),
                "gene_model": gene_model.state_dict(),
                "projector_config": asdict(image_model.config),
                "ema_decay": config.ema_decay,
                "image_best_epoch": image_epoch,
                "gene_best_epoch": gene_epoch,
            },
            fold_dir / "phase12_best.pt",
        )
        np.savez_compressed(
            fold_dir / "phase3_procrustes.npz",
            gene_map=gene_map,
            image_map=image_map,
            singular_values=singular_values,
            fit_indices=train_idx,
        )
        with (fold_dir / "phase1_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(image_history[0]))
            writer.writeheader()
            writer.writerows(image_history)
        with (fold_dir / "phase2_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(gene_history[0]))
            writer.writeheader()
            writer.writerows(gene_history)
        summary = {
            "fold_id": fold_id,
            "n_train": int(len(train_idx)),
            "n_validation": int(len(validation_idx)),
            "n_test": int(len(test_idx)),
            "image_best_epoch": image_epoch,
            "gene_best_epoch": gene_epoch,
            "image_validation_gap": image_history[image_epoch - 1]["validation_similarity_gap"],
            "gene_validation_gap": gene_history[gene_epoch - 1]["validation_similarity_gap"],
            "phase3_train_paired_cosine": float(
                np.sum(gene_phase3[train_idx] * image_phase3[train_idx], axis=1).mean()
            ),
        }
        (fold_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summaries.append(summary)
        print(json.dumps(summary, sort_keys=True), flush=True)

    result = {
        "run_name": config.run_name,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "supervision": "ground_truth cortical layers define Phase 1/2 pairs",
        "strictness": "projectors and Phase 3 fit only train rows; source embeddings may be transductively pretrained",
        "folds": summaries,
    }
    (run_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    defaults = SupervisedMultiphaseConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset_manifest", "split_manifest", "embeddings_npz", "output_root"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, default=getattr(defaults, name))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--folds", nargs="*", default=[])
    parser.add_argument("--image-key", default=defaults.image_key)
    parser.add_argument("--gene-key", default=defaults.gene_key)
    for name in ("positive_weight", "negative_weight", "negative_margin", "ema_decay", "lr", "weight_decay", "min_delta", "dropout"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=float, default=getattr(defaults, name))
    for name in ("hidden_dim", "output_dim", "epochs", "steps_per_epoch", "batch_size", "validation_pairs", "patience", "seed"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=int, default=getattr(defaults, name))
    parser.add_argument("--device", default=defaults.device)
    return parser


def main() -> None:
    args = vars(build_arg_parser().parse_args())
    args["folds"] = tuple(args["folds"])
    result = run_supervised_multiphase(SupervisedMultiphaseConfig(**args))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
