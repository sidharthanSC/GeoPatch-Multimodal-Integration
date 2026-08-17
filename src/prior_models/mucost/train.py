"""Train and export MuCoST on repository DLPFC sections."""

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

from .config import MuCostConfig
from .data import MuCostSectionData, prepare_section
from .model import Model
from ..staig.evaluate import clustering_metrics, refine_labels, tied_gmm


@dataclass(frozen=True)
class MuCostResult:
    embeddings: np.ndarray          # latent spatial-view embedding (latent_dim)
    reconstructions: np.ndarray     # decoder reconstruction in gene space
    predictions: np.ndarray         # common-protocol tied-GMM clustering
    refined_predictions: np.ndarray # 15-neighbor refinement
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


def fit_mucost(
    data: MuCostSectionData,
    config: MuCostConfig,
    device: torch.device | str | None = None,
) -> MuCostResult:
    """Fit one independent transductive MuCoST model for one tissue section."""
    _set_seed(config.seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float32

    x = torch.as_tensor(data.features, dtype=dtype, device=device)
    g_s = torch.as_tensor(data.spatial_edge_index, dtype=torch.long, device=device)
    g_f = torch.as_tensor(data.combined_edge_index, dtype=torch.long, device=device)

    model = Model(x.shape[1], config).to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    losses: list[float] = []

    model.train()
    for epoch in range(1, config.epochs + 1):
        optimizer.zero_grad()
        loss = model(x, g_s, g_f)[-1]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch % config.log_step == 0 or epoch == config.epochs:
            print(f"{data.section_id} epoch={epoch:04d} loss={losses[-1]:.4f}", flush=True)

    model.eval()
    with torch.no_grad():
        latent, reconstructed, _ = model(x, g_s, g_f)
        latent = latent.cpu().numpy().astype(np.float32)
        reconstructed = reconstructed.cpu().numpy().astype(np.float32)

    n_clusters = len(np.unique(data.labels))
    predictions = tied_gmm(latent, n_clusters=n_clusters)
    refined = refine_labels(predictions, data.coordinates, config.refinement_neighbors)
    refined_native = refine_labels(predictions, data.coordinates, config.n_refine)
    before = clustering_metrics(data.labels, predictions)
    after = clustering_metrics(data.labels, refined)
    native_after = clustering_metrics(data.labels, refined_native)
    metrics = {
        **before,
        "refined_ari": after["ari"],
        "refined_nmi": after["nmi"],
        "native_refined_ari": native_after["ari"],
        "native_refined_nmi": native_after["nmi"],
        "n_spots": int(data.features.shape[0]),
        "n_clusters": int(n_clusters),
        "final_loss": losses[-1],
    }
    state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    return MuCostResult(latent, reconstructed, predictions, refined, metrics, losses, state_dict)


def run_dlpfc(
    checkpoint_path: Path,
    output_dir: Path,
    config: MuCostConfig,
    sections: list[str] | None = None,
    device: str | None = None,
) -> dict[str, object]:
    """Run independent MuCoST models and persist complete provenance and outputs."""
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
        data = prepare_section(adata, section_id, config)
        result = fit_mucost(data, config, device=device)
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
            },
            output_dir / "checkpoints" / f"{section_id}.pt",
        )
        np.savez_compressed(
            output_dir / "embeddings" / f"{section_id}.npz",
            embeddings=result.embeddings,
            reconstructions=result.reconstructions,
            predictions=result.predictions,
            refined_predictions=result.refined_predictions,
            labels=data.labels,
            barcodes=data.barcodes,
            coordinates=data.coordinates,
            losses=np.asarray(result.losses, dtype=np.float32),
        )
        print(json.dumps(row, sort_keys=True), flush=True)

    with (output_dir / "section_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    numeric = [
        "ari",
        "nmi",
        "refined_ari",
        "refined_nmi",
        "native_refined_ari",
        "native_refined_nmi",
    ]
    summary: dict[str, object] = {
        "run_id": output_dir.name,
        "implementation": "repository-local MuCoST reproduction (official shared GCN autoencoder + InfoNCE)",
        "input_feature_source": "adata.obsm['feat'] (3,000 HVGs, scaled [0,10])",
        "graph_source": "coordinate radius graph (r=150, k<=6, undirected) + PCA-50 cosine k-NN (k=6)",
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
            "radius_graph/knn_graph/GCNConv/BatchNorm implemented dependency-light (no torch_geometric).",
            "Clustering uses the common tied-GMM protocol; the official pipeline uses R mclust EEE.",
            f"Common 15-neighbor refinement reported as refined_*; official MuCoST refinement uses n_refine={config.n_refine} (recorded as native_refined_*).",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, default=Path("checkpoints/dlpfc.pkl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sections", nargs="*")
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--latent-dim", type=int, default=50)
    parser.add_argument("--radius", type=float, default=150.0)
    parser.add_argument("--seed", type=int, default=2023)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = MuCostConfig(
        epochs=args.epochs,
        latent_dim=args.latent_dim,
        radius=args.radius,
        seed=args.seed,
    )
    summary = run_dlpfc(
        args.checkpoint_path,
        args.output_dir,
        config,
        sections=args.sections,
        device=args.device,
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()