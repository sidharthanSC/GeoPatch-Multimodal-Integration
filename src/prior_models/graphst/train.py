"""Train and export GraphST on repository DLPFC sections."""

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
import torch.nn.functional as F

from src.datasets.dlpfc import DlpfcDataset

from .config import GraphStConfig
from .data import GraphStSectionData, preprocess_adjacency, prepare_section
from .model import Encoder
from ..staig.evaluate import clustering_metrics, refine_labels, tied_gmm


@dataclass(frozen=True)
class GraphStResult:
    embeddings: np.ndarray          # reconstruction embedding (dim_output space)
    latent_embeddings: np.ndarray   # hidden (pre-readout) embedding
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


def permutation(features: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """Row-permute a feature tensor (official GraphST data augmentation)."""
    ids = rng.permutation(features.shape[0])
    return features[ids]


def fit_graphst(
    data: GraphStSectionData,
    config: GraphStConfig,
    device: torch.device | str | None = None,
) -> GraphStResult:
    """Fit one independent transductive GraphST model for one tissue section."""
    _set_seed(config.seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float32

    features = torch.as_tensor(data.features, dtype=dtype, device=device)
    adj = torch.as_tensor(preprocess_adjacency(data.adjacency), dtype=dtype, device=device)
    graph_neigh = torch.as_tensor(data.graph_neigh, dtype=dtype, device=device)
    label_csl = torch.tensor([[1.0, 0.0]] * data.features.shape[0], dtype=dtype, device=device)
    numpy_rng = np.random.default_rng(config.seed)

    model = Encoder(
        config.dim_input,
        config.dim_output,
        graph_neigh,
        dropout=config.dropout,
    ).to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loss_csl = torch.nn.BCEWithLogitsLoss()
    losses: list[float] = []

    model.train()
    for epoch in range(1, config.epochs + 1):
        features_a = permutation(features, numpy_rng)
        _, emb, ret, ret_a = model(features, features_a, adj)
        loss_sl_1 = loss_csl(ret, label_csl)
        loss_sl_2 = loss_csl(ret_a, label_csl)
        loss_feat = F.mse_loss(features, emb)
        loss = config.alpha * loss_feat + config.beta * (loss_sl_1 + loss_sl_2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch % 50 == 0 or epoch == config.epochs:
            print(f"{data.section_id} epoch={epoch:03d} loss={losses[-1]:.6f}", flush=True)

    model.eval()
    with torch.no_grad():
        hidden, reconstructed, _, _ = model(features, permutation(features, numpy_rng), adj)
        latent = hidden.cpu().numpy().astype(np.float32)
        reconstructed = reconstructed.cpu().numpy().astype(np.float32)

    n_clusters = len(np.unique(data.labels))
    # The paper clusters the reconstructed expression embedding after PCA
    # reduction (official mclust path uses PCA-20), then refines spatially.
    from sklearn.decomposition import PCA

    pca = PCA(n_components=min(20, reconstructed.shape[1]), random_state=config.seed)
    cluster_input = pca.fit_transform(reconstructed).astype(np.float32)
    predictions = tied_gmm(cluster_input, n_clusters=n_clusters)
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
    return GraphStResult(reconstructed, latent, predictions, refined, metrics, losses, state_dict)


def run_dlpfc(
    checkpoint_path: Path,
    output_dir: Path,
    config: GraphStConfig,
    sections: list[str] | None = None,
    device: str | None = None,
) -> dict[str, object]:
    """Run independent GraphST models and persist complete provenance and outputs."""
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
        result = fit_graphst(data, config, device=device)
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
            latent_embeddings=result.latent_embeddings,
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
    numeric = ["ari", "nmi", "refined_ari", "refined_nmi"]
    summary: dict[str, object] = {
        "run_id": output_dir.name,
        "implementation": "repository-local GraphST reproduction (official Encoder/Discriminator/AvgReadout)",
        "input_feature_source": "adata.obsm['feat'] (3,000 HVGs, scaled [0,10])",
        "graph_source": f"coordinate {config.n_neighbors}-NN interaction (sklearn NearestNeighbors analogue of ot.dist)",
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
            "Graph construction uses sklearn NearestNeighbors instead of ot.dist (same k-NN semantics).",
            "Clustering uses the common tied-GMM + 15-neighbor refinement protocol rather than R mclust.",
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
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--n-neighbors", type=int, default=3)
    parser.add_argument("--seed", type=int, default=41)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = GraphStConfig(
        epochs=args.epochs,
        n_neighbors=args.n_neighbors,
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