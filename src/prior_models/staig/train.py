"""Train and export the STAIG adaptation on repository DLPFC sections."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.datasets.dlpfc import DlpfcDataset

from .config import StaigConfig
from .data import (
    PrecomputedFeatureBundle,
    StaigSectionData,
    load_feature_bundle,
    prepare_section,
    sample_augmented_edges,
)
from .evaluate import clustering_metrics, refine_labels, tied_gmm
from .model import StaigModel, mask_features, neighbor_contrastive_loss, normalized_adjacency


@dataclass(frozen=True)
class StaigResult:
    embeddings: np.ndarray
    predictions: np.ndarray
    refined_predictions: np.ndarray
    metrics: dict[str, float]
    losses: list[float]
    state_dict: dict[str, torch.Tensor]


def _set_seed(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def fit_staig(
    data: StaigSectionData,
    config: StaigConfig,
    device: torch.device | str | None = None,
) -> StaigResult:
    """Fit one independent transductive STAIG model for one tissue section."""
    _set_seed(config.seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float64 if config.dtype == "float64" else torch.float32
    features = torch.as_tensor(data.features, dtype=dtype, device=device)
    pseudo = torch.as_tensor(data.pseudo_labels, dtype=torch.long, device=device)
    full_edges = torch.as_tensor(data.edge_index, dtype=torch.long, device=device)

    model = StaigModel(features.shape[1], config.hidden_dim, config.projection_dim)
    model = model.to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    numpy_rng = np.random.default_rng(config.seed)
    torch_generator = torch.Generator(device=device).manual_seed(config.seed)
    losses: list[float] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        edges_1 = torch.as_tensor(
            sample_augmented_edges(data.edge_index, data.edge_drop_probability, numpy_rng),
            dtype=torch.long,
            device=device,
        )
        edges_2 = torch.as_tensor(
            sample_augmented_edges(data.edge_index, data.edge_drop_probability, numpy_rng),
            dtype=torch.long,
            device=device,
        )
        adjacency_1 = normalized_adjacency(
            edges_1, features.shape[0], dtype=dtype, device=device
        )
        adjacency_2 = normalized_adjacency(
            edges_2, features.shape[0], dtype=dtype, device=device
        )
        z1 = model(mask_features(features, config.feature_mask_rate_1, torch_generator), adjacency_1)
        z2 = model(mask_features(features, config.feature_mask_rate_2, torch_generator), adjacency_2)
        loss = neighbor_contrastive_loss(
            z1, z2, full_edges, pseudo, config.temperature
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch % 50 == 0 or epoch == config.epochs:
            print(f"{data.section_id} epoch={epoch:03d} loss={losses[-1]:.6f}", flush=True)

    model.eval()
    full_adjacency = normalized_adjacency(
        full_edges, features.shape[0], dtype=dtype, device=device
    )
    with torch.no_grad():
        embeddings = model(features, full_adjacency).cpu().numpy().astype(np.float32)
    n_clusters = len(np.unique(data.labels))
    predictions = tied_gmm(embeddings, n_clusters=n_clusters)
    refined = refine_labels(predictions, data.coordinates, config.refinement_neighbors)
    before = clustering_metrics(data.labels, predictions)
    after = clustering_metrics(data.labels, refined)
    metrics = {
        **before,
        "refined_ari": after["ari"],
        "refined_nmi": after["nmi"],
        "n_spots": int(data.features.shape[0]),
        "n_clusters": int(n_clusters),
        "final_loss": losses[-1],
    }
    state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    return StaigResult(embeddings, predictions, refined, metrics, losses, state_dict)


def run_dlpfc(
    checkpoint_path: Path,
    output_dir: Path,
    config: StaigConfig,
    sections: list[str] | None = None,
    device: str | None = None,
    node_features: PrecomputedFeatureBundle | None = None,
    image_features: PrecomputedFeatureBundle | None = None,
) -> dict[str, object]:
    """Run independent STAIG models and persist complete provenance and outputs."""
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {output_dir}")
    output_dir.mkdir(parents=True)
    (output_dir / "checkpoints").mkdir()
    (output_dir / "embeddings").mkdir()
    dataset = DlpfcDataset.from_checkpoint(checkpoint_path)
    section_ids = sections or dataset.section_ids()
    rows: list[dict[str, object]] = []

    for section_id in section_ids:
        started = time.perf_counter()
        adata = dataset.get_section(section_id)
        expected_barcodes = adata.obs_names.astype(str).to_numpy()
        section_nodes = (
            None if node_features is None else node_features.for_section(section_id, expected_barcodes)
        )
        section_images = (
            None if image_features is None else image_features.for_section(section_id, expected_barcodes)
        )
        data = prepare_section(
            adata,
            section_id,
            config,
            node_features=section_nodes,
            image_features=section_images,
            node_feature_name=(
                "section_hvg_expression"
                if node_features is None
                else f"{node_features.source}:{node_features.array_key}"
            ),
            image_feature_name=(
                "img_emb"
                if image_features is None
                else f"{image_features.source}:{image_features.array_key}"
            ),
        )
        result = fit_staig(data, config, device=device)
        elapsed = time.perf_counter() - started
        row = {"section_id": section_id, **result.metrics, "elapsed_seconds": elapsed}
        rows.append(row)
        torch.save(
            {
                "section_id": section_id,
                "config": config.to_dict(),
                "input_dim": int(data.features.shape[1]),
                "state_dict": result.state_dict,
                "metrics": result.metrics,
                "node_feature_name": data.node_feature_name,
                "image_feature_name": data.image_feature_name,
            },
            output_dir / "checkpoints" / f"{section_id}.pt",
        )
        np.savez_compressed(
            output_dir / "embeddings" / f"{section_id}.npz",
            embeddings=result.embeddings,
            predictions=result.predictions,
            refined_predictions=result.refined_predictions,
            labels=data.labels,
            barcodes=data.barcodes,
            coordinates=data.coordinates,
            losses=np.asarray(result.losses, dtype=np.float32),
            node_feature_name=np.asarray(data.node_feature_name),
            image_feature_name=np.asarray(data.image_feature_name),
        )
        print(json.dumps(row, sort_keys=True), flush=True)

    with (output_dir / "section_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    numeric = ["ari", "nmi", "refined_ari", "refined_nmi"]
    summary: dict[str, object] = {
        "run_id": output_dir.name,
        "implementation": "repository-local STAIG adaptation",
        "input_image_key": (
            "img_emb" if image_features is None else image_features.array_key
        ),
        "node_feature_source": (
            "section_hvg_expression"
            if node_features is None
            else f"{node_features.source}:{node_features.array_key}"
        ),
        "image_feature_source": (
            "img_emb"
            if image_features is None
            else f"{image_features.source}:{image_features.array_key}"
        ),
        "config": config.to_dict(),
        "sections": section_ids,
        "n_sections": len(section_ids),
        "metrics": {
            key: {
                "mean": float(np.mean([float(row[key]) for row in rows])),
                "median": float(np.median([float(row[key]) for row in rows])),
                "iqr": float(
                    np.percentile([float(row[key]) for row in rows], 75)
                    - np.percentile([float(row[key]) for row in rows], 25)
                ),
            }
            for key in numeric
        },
        "deviations": [
            (
                "Uses fixed repository img_emb instead of STAIG filtered-patch BYOL embeddings."
                if image_features is None
                else "Uses the recorded precomputed guidance feature instead of STAIG filtered-patch BYOL embeddings."
            ),
            "Uses sklearn tied-covariance GMM as the R mclust EEE analogue.",
            f"Uses {config.dtype}; the official implementation casts the model and inputs to float64.",
            f"Uses seed {config.seed}; the paper supplement does not declare one DLPFC seed, while the released 151673 notebook uses 39788.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "config.json").write_text(
        json.dumps(config.to_dict(), indent=2), encoding="utf-8"
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, default=Path("checkpoints/dlpfc.pkl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sections", nargs="*")
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=10.0)
    parser.add_argument("--image-pseudo-clusters", type=int, default=40)
    parser.add_argument("--feature-mask-rate-1", type=float, default=0.1)
    parser.add_argument("--feature-mask-rate-2", type=float, default=0.1)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--node-features-npz", type=Path)
    parser.add_argument("--node-feature-key")
    parser.add_argument("--image-features-npz", type=Path)
    parser.add_argument("--image-feature-key")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = StaigConfig(
        epochs=args.epochs,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        projection_dim=args.projection_dim,
        temperature=args.temperature,
        image_pseudo_clusters=args.image_pseudo_clusters,
        feature_mask_rate_1=args.feature_mask_rate_1,
        feature_mask_rate_2=args.feature_mask_rate_2,
        dtype=args.dtype,
    )
    if bool(args.node_features_npz) != bool(args.node_feature_key):
        raise ValueError("--node-features-npz and --node-feature-key must be provided together")
    if bool(args.image_features_npz) != bool(args.image_feature_key):
        raise ValueError("--image-features-npz and --image-feature-key must be provided together")
    node_features = (
        load_feature_bundle(args.node_features_npz, args.node_feature_key)
        if args.node_features_npz
        else None
    )
    image_features = (
        load_feature_bundle(args.image_features_npz, args.image_feature_key)
        if args.image_features_npz
        else None
    )
    summary = run_dlpfc(
        args.checkpoint_path,
        args.output_dir,
        config,
        sections=args.sections,
        device=args.device,
        node_features=node_features,
        image_features=image_features,
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
