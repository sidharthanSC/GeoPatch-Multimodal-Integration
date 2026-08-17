"""Shared supervised spatial GCN baseline across DLPFC tissue sections.

This is a repository-local baseline built from STAIG's graph primitives. It is not
native STAIG, SpaGCN, or GraphST. Labels supervise only nodes assigned the training
role by an immutable split manifest.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch import nn

from src.prior_models.staig.data import build_spatial_graph
from src.prior_models.staig.model import GraphConvolution, normalized_adjacency
from src.splits.dlpfc import GLOBAL_LABEL_ORDER
from src.train.supervised_evaluation import classification_metrics


@dataclass(frozen=True)
class SupervisedGCNConfig:
    multiphase_run_dir: Path
    split_manifest: Path
    dataset_manifest: Path
    output_dir: Path
    feature_keys: tuple[str, ...] = (
        "gene_phase3",
        "image_phase3",
        "phase3_concat",
    )
    coordinate_neighbors: int = 6
    hidden_dim: int = 128
    dropout: float = 0.2
    epochs: int = 100
    patience: int = 12
    lr: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 0
    device: str = "cuda"


class SupervisedGCN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int, dropout: float) -> None:
        super().__init__()
        self.first = GraphConvolution(input_dim, hidden_dim)
        self.second = GraphConvolution(hidden_dim, n_classes)
        self.dropout = dropout

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.first(features, adjacency))
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        return self.second(hidden, adjacency)


def _feature_array(data: dict[str, np.ndarray], key: str) -> np.ndarray:
    if key == "phase3_concat":
        return np.concatenate([data["image_phase3"], data["gene_phase3"]], axis=1)
    if key not in data:
        raise KeyError(f"Unknown GCN feature key {key}")
    return data[key]


def _section_graphs(
    dataset: pd.DataFrame,
    coordinate_neighbors: int,
    device: torch.device,
) -> dict[str, tuple[np.ndarray, torch.Tensor]]:
    graphs = {}
    for section_id in dataset["section_id"].drop_duplicates().astype(str):
        section = dataset.loc[dataset["section_id"].astype(str) == section_id]
        rows = section["global_row"].astype(int).to_numpy()
        coordinates = section[["spatial_x", "spatial_y"]].to_numpy(dtype=np.float32)
        edges = build_spatial_graph(coordinates, coordinate_neighbors)
        adjacency = normalized_adjacency(
            torch.as_tensor(edges, dtype=torch.long, device=device),
            len(rows),
            dtype=torch.float32,
            device=device,
        )
        graphs[section_id] = (rows, adjacency)
    return graphs


def train_supervised_gcn_fold(
    values: np.ndarray,
    labels: np.ndarray,
    role: np.ndarray,
    dataset: pd.DataFrame,
    config: SupervisedGCNConfig,
    seed_offset: int,
) -> tuple[np.ndarray, int, int]:
    """Fit shared weights over section graphs and predict all test-role nodes."""
    device = torch.device(config.device)
    torch.manual_seed(config.seed + seed_offset)
    train_idx = np.where(role == "train")[0]
    scaler = StandardScaler().fit(values[train_idx])
    transformed = scaler.transform(values).astype(np.float32)
    features = torch.as_tensor(transformed, dtype=torch.float32, device=device)
    label_to_int = {label: index for index, label in enumerate(GLOBAL_LABEL_ORDER)}
    encoded = torch.as_tensor(
        [label_to_int[label] for label in labels], dtype=torch.long, device=device
    )
    graphs = _section_graphs(dataset, config.coordinate_neighbors, device)
    train_counts = torch.bincount(encoded[train_idx], minlength=len(GLOBAL_LABEL_ORDER)).float()
    class_weight = torch.where(
        train_counts > 0,
        train_counts.sum() / (len(GLOBAL_LABEL_ORDER) * train_counts.clamp_min(1.0)),
        torch.zeros_like(train_counts),
    )
    model = SupervisedGCN(values.shape[1], config.hidden_dim, 7, config.dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0

    def run_role(selected_role: str, train: bool) -> tuple[torch.Tensor, dict[int, int]]:
        model.train(train)
        total_loss = features.new_zeros(())
        total_nodes = 0
        predictions: dict[int, int] = {}
        for rows, adjacency in graphs.values():
            local_mask = role[rows] == selected_role
            if not np.any(local_mask):
                continue
            local_indices = np.where(local_mask)[0]
            logits = model(features[rows], adjacency)
            local_targets = encoded[rows[local_indices]]
            loss = F.cross_entropy(
                logits[local_indices], local_targets, weight=class_weight, reduction="sum"
            )
            total_loss = total_loss + loss
            total_nodes += len(local_indices)
            local_predicted = logits[local_indices].argmax(dim=1).detach().cpu().numpy()
            predictions.update(
                (int(global_row), int(prediction))
                for global_row, prediction in zip(rows[local_indices], local_predicted)
            )
        return total_loss / max(total_nodes, 1), predictions

    for epoch in range(1, config.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        train_loss, _ = run_role("train", True)
        train_loss.backward()
        optimizer.step()
        with torch.no_grad():
            validation_loss, _ = run_role("validation", False)
        value = float(validation_loss)
        if value < best_loss - 1e-4:
            best_loss = value
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= config.patience:
            break
    if best_state is None:
        raise RuntimeError("GCN did not select a validation checkpoint")
    model.load_state_dict(best_state)
    with torch.no_grad():
        _, prediction_map = run_role("test", False)
    test_idx = np.where(role == "test")[0]
    predicted = np.asarray([prediction_map[int(index)] for index in test_idx])
    return predicted, best_epoch, sum(parameter.numel() for parameter in model.parameters())


def run_supervised_gcn(config: SupervisedGCNConfig) -> dict[str, object]:
    if config.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {config.output_dir}")
    config.output_dir.mkdir(parents=True)
    dataset = pd.read_parquet(config.dataset_manifest).sort_values("global_row")
    split = pd.read_parquet(config.split_manifest)
    labels = dataset["ground_truth"].astype(str).to_numpy()
    records = []
    for fold_number, fold_id in enumerate(split["fold_id"].drop_duplicates().astype(str)):
        path = config.multiphase_run_dir / fold_id / "embeddings.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=True) as artifact:
            data = {
                key: np.asarray(artifact[key], dtype=np.float32)
                for key in ("image_raw", "gene_raw", "image_phase1", "gene_phase2", "image_phase3", "gene_phase3")
            }
            role = artifact["role"].astype(str)
        test_idx = np.where(role == "test")[0]
        for feature_number, key in enumerate(config.feature_keys):
            values = _feature_array(data, key)
            predicted_int, best_epoch, parameters = train_supervised_gcn_fold(
                values,
                labels,
                role,
                dataset,
                config,
                100 * fold_number + feature_number,
            )
            predicted = np.asarray([GLOBAL_LABEL_ORDER[index] for index in predicted_int])
            records.append(
                {
                    "fold_id": fold_id,
                    "method": f"supervised_gcn:{key}",
                    "family": "repository_local_supervised_gcn",
                    "best_epoch": best_epoch,
                    "parameters": parameters,
                    "n_test": len(test_idx),
                    **classification_metrics(labels[test_idx], predicted),
                }
            )
            print(json.dumps({"fold_id": fold_id, "feature": key, "accuracy": records[-1]["accuracy"]}), flush=True)
    flat = [
        {key: value for key, value in record.items() if key not in {"per_layer_recall", "support", "confusion_matrix", "observed_test_labels"}}
        for record in records
    ]
    pd.DataFrame(flat).to_csv(config.output_dir / "metrics.csv", index=False)
    (config.output_dir / "detailed_metrics.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    summary = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "n_records": len(records),
        "method_scope": "repository-local supervised GCN; not native STAIG, SpaGCN, or GraphST",
    }
    (config.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multiphase-run-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-keys", nargs="+", default=list(SupervisedGCNConfig.__dataclass_fields__["feature_keys"].default))
    parser.add_argument("--coordinate-neighbors", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    args = vars(build_arg_parser().parse_args())
    args["feature_keys"] = tuple(args["feature_keys"])
    print(json.dumps(run_supervised_gcn(SupervisedGCNConfig(**args)), indent=2))


if __name__ == "__main__":
    main()
