"""Common supervised and mathematical evaluation of three-phase embeddings."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import SpectralClustering
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    normalized_mutual_info_score,
    recall_score,
)
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler
from torch import nn

from src.multimodal.attention_ablation import ConcatMLP, SelfAttentionFusion
from src.multimodal.attention_fusion import AttentionFusionModel, FusionConfig
from src.multimodal.covariance import covariance_summary
from src.multimodal.evaluation import cluster_accuracy
from src.multimodal.representations import concatenate, normalized_mean
from src.splits.dlpfc import GLOBAL_LABEL_ORDER


@dataclass(frozen=True)
class SupervisedEvaluationConfig:
    multiphase_run_dir: Path
    split_manifest: Path
    output_dir: Path
    dataset_manifest: Path = Path(
        "outputs/evaluation/20260813_checkpoint_audit_v1/data/dataset_manifest.parquet"
    )
    staig_run_dir: Path | None = Path(
        "outputs/prior_models/staig_paper_default_img_emb_seed0_all12_v3"
    )
    neural_epochs: int = 80
    neural_patience: int = 10
    neural_batch_size: int = 256
    neural_lr: float = 1e-4
    spectral_neighbors: int = 10
    covariance_epsilon: float = 1e-3
    seed: int = 0
    device: str = "cuda"


class SingleMLP(nn.Module):
    def __init__(self, dim: int, n_classes: int, hidden_dim: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


class BisectorMLP(nn.Module):
    def __init__(self, dim: int, n_classes: int, hidden_dim: int = 512) -> None:
        super().__init__()
        self.classifier = SingleMLP(dim, n_classes, hidden_dim)

    def forward(self, image: torch.Tensor, gene: torch.Tensor) -> torch.Tensor:
        bisector = F.normalize(F.normalize(image, dim=1) + F.normalize(gene, dim=1), dim=1)
        return self.classifier(bisector)


def classification_metrics(
    true: np.ndarray, predicted: np.ndarray, label_order: tuple[str, ...] = GLOBAL_LABEL_ORDER
) -> dict[str, object]:
    """Return support-aware metrics and fixed-axis per-layer diagnostics."""
    true = np.asarray(true).astype(str)
    predicted = np.asarray(predicted).astype(str)
    observed = [label for label in label_order if np.any(true == label)]
    recalls = recall_score(true, predicted, labels=list(label_order), average=None, zero_division=0)
    supports = np.asarray([np.sum(true == label) for label in label_order])
    return {
        "accuracy": float(accuracy_score(true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(true, predicted)),
        "macro_f1": float(f1_score(true, predicted, labels=observed, average="macro", zero_division=0)),
        "observed_test_labels": observed,
        "per_layer_recall": {
            label: (float(value) if support else None)
            for label, value, support in zip(label_order, recalls, supports)
        },
        "support": {label: int(value) for label, value in zip(label_order, supports)},
        "confusion_matrix": confusion_matrix(true, predicted, labels=list(label_order)).tolist(),
    }


def _representations(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    image_raw, gene_raw = data["image_raw"], data["gene_raw"]
    image1, gene2 = data["image_phase1"], data["gene_phase2"]
    image3, gene3 = data["image_phase3"], data["gene_phase3"]
    return {
        "raw_image": image_raw,
        "raw_gene": gene_raw,
        "raw_bisector": normalized_mean(image_raw, gene_raw),
        "raw_concat": concatenate(image_raw, gene_raw),
        "phase1_image": image1,
        "phase2_gene": gene2,
        "phase12_bisector": normalized_mean(image1, gene2),
        "phase12_concat": concatenate(image1, gene2),
        "phase3_image": image3,
        "phase3_gene": gene3,
        "phase3_bisector": normalized_mean(image3, gene3),
        "phase3_concat": concatenate(image3, gene3),
    }


def _fit_classical(
    values: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
) -> list[tuple[str, np.ndarray]]:
    scaler = StandardScaler().fit(values[train_idx])
    train = scaler.transform(values[train_idx])
    test = scaler.transform(values[test_idx])
    logistic = SGDClassifier(
        loss="log_loss",
        alpha=1e-4,
        max_iter=2000,
        tol=1e-4,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced",
    ).fit(train, labels[train_idx])
    centroid = NearestCentroid().fit(train, labels[train_idx])
    return [
        ("linear_logistic_sgd", logistic.predict(test)),
        ("nearest_centroid", centroid.predict(test)),
    ]


def _run_neural_epoch(
    model: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    targets: torch.Tensor,
    indices: np.ndarray,
    batch_size: int,
    train: bool,
    optimizer: torch.optim.Optimizer | None,
    rng: np.random.Generator,
    class_weight: torch.Tensor,
) -> tuple[float, np.ndarray]:
    model.train(train)
    order = indices.copy()
    if train:
        rng.shuffle(order)
    total = 0.0
    predictions = []
    for start in range(0, len(order), batch_size):
        batch = order[start : start + batch_size]
        with torch.set_grad_enabled(train):
            logits = model(*(values[batch] for values in inputs))
            loss = F.cross_entropy(logits, targets[batch], weight=class_weight)
        if train:
            if optimizer is None:
                raise ValueError("optimizer required for training")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        total += float(loss.detach()) * len(batch)
        predictions.append(logits.argmax(dim=1).detach().cpu().numpy())
    return total / len(order), np.concatenate(predictions)


def train_neural_classifier(
    model: nn.Module,
    inputs: tuple[np.ndarray, ...],
    encoded_labels: np.ndarray,
    train_idx: np.ndarray,
    validation_idx: np.ndarray,
    test_idx: np.ndarray,
    config: SupervisedEvaluationConfig,
    seed_offset: int,
) -> tuple[np.ndarray, int, int]:
    """Train one classifier with validation-only early stopping."""
    device = torch.device(config.device)
    torch.manual_seed(config.seed + seed_offset)
    model = model.to(device)
    tensors = tuple(torch.as_tensor(value, dtype=torch.float32, device=device) for value in inputs)
    targets = torch.as_tensor(encoded_labels, dtype=torch.long, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.neural_lr, weight_decay=1e-4)
    rng = np.random.default_rng(config.seed + seed_offset)
    train_counts = torch.bincount(targets[train_idx], minlength=len(GLOBAL_LABEL_ORDER)).float()
    class_weight = torch.where(
        train_counts > 0,
        train_counts.sum() / (len(GLOBAL_LABEL_ORDER) * train_counts.clamp_min(1.0)),
        torch.zeros_like(train_counts),
    )
    best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
    for epoch in range(1, config.neural_epochs + 1):
        _run_neural_epoch(
            model, tensors, targets, train_idx, config.neural_batch_size, True, optimizer, rng,
            class_weight,
        )
        validation_loss, _ = _run_neural_epoch(
            model, tensors, targets, validation_idx, config.neural_batch_size, False, None, rng,
            class_weight,
        )
        if validation_loss < best_loss - 1e-4:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= config.neural_patience:
            break
    if best_state is None:
        raise RuntimeError("Neural classifier did not select a checkpoint")
    model.load_state_dict(best_state)
    _, predicted = _run_neural_epoch(
        model, tensors, targets, test_idx, config.neural_batch_size, False, None, rng,
        class_weight,
    )
    return predicted, best_epoch, sum(parameter.numel() for parameter in model.parameters())


def log_euclidean_mdm(
    image: np.ndarray,
    gene: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    epsilon: float,
    chunk_size: int = 1024,
) -> np.ndarray:
    """Classify by train-only Log-Euclidean means using rank-one identities."""
    summary = covariance_summary(image, gene, epsilon)
    classes = [label for label in GLOBAL_LABEL_ORDER if np.any(labels[train_idx] == label)]
    means = []
    for label in classes:
        indices = train_idx[labels[train_idx] == label]
        unit = summary.unit_direction[indices].astype(np.float64)
        log_floor = np.log(summary.floor_eigenvalue[indices])
        log_gap = np.log(summary.top_eigenvalue[indices]) - log_floor
        mean = np.eye(summary.dim) * log_floor.mean()
        mean += unit.T @ (log_gap[:, None] * unit) / len(indices)
        means.append(mean)
    means_array = np.stack(means)
    mean_norm = np.square(means_array).sum(axis=(1, 2))
    mean_trace = np.trace(means_array, axis1=1, axis2=2)
    predicted = []
    for start in range(0, len(test_idx), chunk_size):
        indices = test_idx[start : start + chunk_size]
        unit = summary.unit_direction[indices].astype(np.float64)
        log_floor = np.log(summary.floor_eigenvalue[indices])
        log_gap = np.log(summary.top_eigenvalue[indices]) - log_floor
        spot_norm = (
            summary.dim * np.square(log_floor)
            + 2.0 * log_floor * log_gap
            + np.square(log_gap)
        )
        quadratic = np.stack(
            [np.einsum("bi,ij,bj->b", unit, mean, unit) for mean in means_array],
            axis=1,
        )
        inner = log_floor[:, None] * mean_trace[None] + log_gap[:, None] * quadratic
        distances = spot_norm[:, None] + mean_norm[None] - 2.0 * inner
        predicted.extend(classes[index] for index in distances.argmin(axis=1))
    return np.asarray(predicted)


def spectral_bisector_clustering(
    bisector: np.ndarray,
    labels: np.ndarray,
    section_ids: np.ndarray,
    test_idx: np.ndarray,
    neighbors: int,
    seed: int,
) -> list[dict[str, object]]:
    """Cluster test spots independently by section; labels set only cluster count."""
    records = []
    for section_id in np.unique(section_ids[test_idx]):
        indices = test_idx[section_ids[test_idx] == section_id]
        if len(indices) < 3:
            continue
        true = labels[indices]
        n_clusters = len(np.unique(true))
        predicted = SpectralClustering(
            n_clusters=n_clusters,
            affinity="nearest_neighbors",
            n_neighbors=min(neighbors, len(indices) - 1),
            assign_labels="kmeans",
            n_init=10,
            random_state=seed,
        ).fit_predict(bisector[indices])
        records.append(
            {
                "section_id": str(section_id),
                "n_test": int(len(indices)),
                "n_clusters": n_clusters,
                "accuracy": cluster_accuracy(true, predicted),
                "ari": float(adjusted_rand_score(true, predicted)),
                "nmi": float(normalized_mutual_info_score(true, predicted)),
            }
        )
    return records


def load_staig_embeddings(
    run_dir: Path, dataset: pd.DataFrame
) -> np.ndarray:
    """Load independently trained per-section STAIG vectors in global manifest order."""
    output = np.empty((len(dataset), 64), dtype=np.float32)
    for section_id in dataset["section_id"].drop_duplicates().astype(str):
        section = dataset.loc[dataset["section_id"].astype(str) == section_id]
        with np.load(run_dir / "embeddings" / f"{section_id}.npz", allow_pickle=True) as data:
            barcodes = data["barcodes"].astype(str)
            embeddings = np.asarray(data["embeddings"], dtype=np.float32)
        lookup = {barcode: index for index, barcode in enumerate(barcodes)}
        expected = section["barcode"].astype(str).tolist()
        if len(lookup) != len(barcodes) or any(barcode not in lookup for barcode in expected):
            raise ValueError(f"STAIG identities fail for section {section_id}")
        output[section["global_row"].astype(int)] = embeddings[
            [lookup[barcode] for barcode in expected]
        ]
    return output


def run_supervised_evaluation(config: SupervisedEvaluationConfig) -> dict[str, object]:
    """Evaluate every fold under one immutable split and write long-form results."""
    if config.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {config.output_dir}")
    config.output_dir.mkdir(parents=True)
    dataset = pd.read_parquet(config.dataset_manifest).sort_values("global_row")
    split = pd.read_parquet(config.split_manifest)
    labels = dataset["ground_truth"].astype(str).to_numpy()
    sections = dataset["section_id"].astype(str).to_numpy()
    label_to_int = {label: index for index, label in enumerate(GLOBAL_LABEL_ORDER)}
    encoded = np.asarray([label_to_int[label] for label in labels], dtype=np.int64)
    staig = (
        load_staig_embeddings(config.staig_run_dir, dataset)
        if config.staig_run_dir is not None
        else None
    )
    records: list[dict[str, object]] = []
    spectral_records: list[dict[str, object]] = []
    fold_ids = split["fold_id"].drop_duplicates().astype(str).tolist()

    for fold_number, fold_id in enumerate(fold_ids):
        fold_path = config.multiphase_run_dir / fold_id / "embeddings.npz"
        if not fold_path.exists():
            continue
        with np.load(fold_path, allow_pickle=True) as data:
            arrays = {key: np.asarray(data[key], dtype=np.float32) for key in (
                "image_raw", "gene_raw", "image_phase1", "gene_phase2",
                "image_phase3", "gene_phase3",
            )}
            role = data["role"].astype(str)
        train_idx = np.where(role == "train")[0]
        validation_idx = np.where(role == "validation")[0]
        test_idx = np.where(role == "test")[0]
        representations = _representations(arrays)
        if staig is not None:
            representations["staig64_transductive"] = staig

        for name, values in representations.items():
            for classifier, predicted in _fit_classical(
                values, labels, train_idx, test_idx, config.seed
            ):
                records.append(
                    {
                        "fold_id": fold_id,
                        "method": f"{name}:{classifier}",
                        "family": "frozen_probe",
                        "n_train": len(train_idx),
                        "n_validation": len(validation_idx),
                        "n_test": len(test_idx),
                        **classification_metrics(labels[test_idx], predicted),
                    }
                )

        image3, gene3 = arrays["image_phase3"], arrays["gene_phase3"]
        neural_specs: list[tuple[str, nn.Module, tuple[np.ndarray, ...]]] = [
            ("phase3_image:mlp", SingleMLP(128, 7), (image3,)),
            ("phase3_gene:mlp", SingleMLP(128, 7), (gene3,)),
            ("phase3_bisector:mlp", BisectorMLP(128, 7), (image3, gene3)),
            ("phase3_concat:mlp", ConcatMLP(128, 7), (image3, gene3)),
            ("phase3:self_attention", SelfAttentionFusion(128, 7), (image3, gene3)),
            (
                "phase3:directional_cross_plus_self",
                AttentionFusionModel(FusionConfig(dim=128, n_classes=7)),
                (image3, gene3),
            ),
        ]
        for model_number, (name, model, inputs) in enumerate(neural_specs):
            predicted_int, best_epoch, parameters = train_neural_classifier(
                model,
                inputs,
                encoded,
                train_idx,
                validation_idx,
                test_idx,
                config,
                100 * fold_number + model_number,
            )
            predicted = np.asarray([GLOBAL_LABEL_ORDER[index] for index in predicted_int])
            records.append(
                {
                    "fold_id": fold_id,
                    "method": name,
                    "family": "neural_classifier",
                    "n_train": len(train_idx),
                    "n_validation": len(validation_idx),
                    "n_test": len(test_idx),
                    "best_epoch": best_epoch,
                    "parameters": parameters,
                    **classification_metrics(labels[test_idx], predicted),
                }
            )

        mdm_predicted = log_euclidean_mdm(
            image3,
            gene3,
            labels,
            train_idx,
            test_idx,
            config.covariance_epsilon,
        )
        records.append(
            {
                "fold_id": fold_id,
                "method": "phase3_covariance:log_euclidean_mdm",
                "family": "mathematical_classifier",
                "n_train": len(train_idx),
                "n_validation": len(validation_idx),
                "n_test": len(test_idx),
                **classification_metrics(labels[test_idx], mdm_predicted),
            }
        )
        for item in spectral_bisector_clustering(
            representations["phase3_bisector"],
            labels,
            sections,
            test_idx,
            config.spectral_neighbors,
            config.seed,
        ):
            spectral_records.append({"fold_id": fold_id, **item})
        print(json.dumps({"fold_id": fold_id, "methods": len(records)}, sort_keys=True), flush=True)

    flat_records = []
    for record in records:
        flat = {key: value for key, value in record.items() if key not in {"per_layer_recall", "support", "confusion_matrix", "observed_test_labels"}}
        flat_records.append(flat)
    pd.DataFrame(flat_records).to_csv(config.output_dir / "metrics.csv", index=False)
    pd.DataFrame(spectral_records).to_csv(config.output_dir / "spectral_section_metrics.csv", index=False)
    (config.output_dir / "detailed_metrics.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    summary = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "n_folds": len({record["fold_id"] for record in records}),
        "n_method_fold_records": len(records),
        "native_spagcn": "NOT RUN: package absent and raw-count preprocessing unavailable",
        "native_staig_supervised": "NOT A NATIVE TASK: frozen transductive STAIG embeddings receive common probes",
    }
    (config.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multiphase-run-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, default=SupervisedEvaluationConfig.__dataclass_fields__["dataset_manifest"].default)
    parser.add_argument("--staig-run-dir", type=Path, default=SupervisedEvaluationConfig.__dataclass_fields__["staig_run_dir"].default)
    parser.add_argument("--neural-epochs", type=int, default=80)
    parser.add_argument("--neural-patience", type=int, default=10)
    parser.add_argument("--neural-batch-size", type=int, default=256)
    parser.add_argument("--neural-lr", type=float, default=1e-4)
    parser.add_argument("--spectral-neighbors", type=int, default=10)
    parser.add_argument("--covariance-epsilon", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    config = SupervisedEvaluationConfig(**vars(build_arg_parser().parse_args()))
    print(json.dumps(run_supervised_evaluation(config), indent=2))


if __name__ == "__main__":
    main()
