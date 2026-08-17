"""Train and export SpaGCN on repository DLPFC sections."""

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

from .config import SpaGcnConfig
from .data import SpaGcnSectionData, prepare_section
from .model import SimpleGcDec, pca_embedding
from ..staig.evaluate import clustering_metrics, refine_labels, tied_gmm


@dataclass(frozen=True)
class SpaGcnResult:
    embeddings: np.ndarray          # GCN embedding (before DEC head)
    predictions: np.ndarray         # native DEC argmax
    gmm_predictions: np.ndarray     # common-protocol tied-GMM clustering
    refined_predictions: np.ndarray # 15-neighbor refinement of GMM predictions
    metrics: dict[str, float]
    losses: list[float]
    state_dict: dict[str, torch.Tensor]
    l_scale: float


def _set_seed(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fit_spagcn(
    data: SpaGcnSectionData,
    config: SpaGcnConfig,
    device: torch.device | str | None = None,
) -> SpaGcnResult:
    """Fit one independent transductive SpaGCN model for one tissue section."""
    _set_seed(config.seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    X = pca_embedding(data.features, config.num_pcs, seed=config.seed)
    n_clusters = config.n_clusters or int(len(np.unique(data.labels)))
    model = SimpleGcDec(X.shape[1], X.shape[1])
    model = model.to(device=device)
    losses = model.fit(
        X,
        data.adjacency,
        lr=config.learning_rate,
        max_epochs=config.max_epochs,
        update_interval=config.update_interval,
        weight_decay=config.weight_decay,
        optimizer_name=config.optimizer,
        n_clusters=n_clusters,
        init_spa=config.init_spa,
        tol=config.tol,
        seed=config.seed,
        device=device,
    )
    z, dec_pred = model.predict(X, data.adjacency, device=device)
    embeddings = z.astype(np.float32)
    gmm_pred = tied_gmm(embeddings, n_clusters=n_clusters)
    refined = refine_labels(gmm_pred, data.coordinates, config.refinement_neighbors)

    # Native DEC clustering metrics plus the common-protocol GMM clustering.
    native = clustering_metrics(data.labels, dec_pred)
    native_refined = clustering_metrics(data.labels, refine_labels(dec_pred, data.coordinates, config.refinement_neighbors))
    common = clustering_metrics(data.labels, gmm_pred)
    common_refined = clustering_metrics(data.labels, refined)
    metrics = {
        **common,
        "refined_ari": common_refined["ari"],
        "refined_nmi": common_refined["nmi"],
        "dec_ari": native["ari"],
        "dec_nmi": native["nmi"],
        "dec_refined_ari": native_refined["ari"],
        "dec_refined_nmi": native_refined["nmi"],
        "n_spots": int(data.features.shape[0]),
        "n_clusters": int(n_clusters),
        "final_loss": losses[-1],
        "n_epochs_run": int(len(losses)),
    }
    state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    return SpaGcnResult(embeddings, dec_pred, gmm_pred, refined, metrics, losses, state_dict, 0.0)


def run_dlpfc(
    checkpoint_path: Path,
    output_dir: Path,
    config: SpaGcnConfig,
    sections: list[str] | None = None,
    device: str | None = None,
) -> dict[str, object]:
    """Run independent SpaGCN models and persist complete provenance and outputs."""
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
        result = fit_spagcn(data, config, device=device)
        elapsed = time.perf_counter() - started
        row = {"section_id": section_id, **result.metrics, "elapsed_seconds": elapsed}
        rows.append(row)
        torch.save(
            {
                "section_id": section_id,
                "config": config.to_dict(),
                "input_dim": int(data.features.shape[1]),
                "n_clusters": int(result.metrics["n_clusters"]),
                "state_dict": result.state_dict,
                "metrics": result.metrics,
                "l_scale": result.l_scale,
            },
            output_dir / "checkpoints" / f"{section_id}.pt",
        )
        np.savez_compressed(
            output_dir / "embeddings" / f"{section_id}.npz",
            embeddings=result.embeddings,
            predictions=result.predictions,
            gmm_predictions=result.gmm_predictions,
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
    numeric = ["ari", "nmi", "refined_ari", "refined_nmi", "dec_ari", "dec_nmi", "dec_refined_ari", "dec_refined_nmi"]
    summary: dict[str, object] = {
        "run_id": output_dir.name,
        "implementation": "repository-local SpaGCN reproduction (official GCN + DEC)",
        "input_feature_source": "adata.obsm['feat'] (3,000 HVGs, scaled [0,10])",
        "graph_source": "coordinate-only weighted adjacency (no histology image available)",
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
            "Uses repository 3,000-HVG features instead of full-filtered raw counts from the official tutorial.",
            "Coordinate-only adjacency (histology image unavailable); official tutorial uses histology-aware distances.",
            "k-means cluster-centre initialisation (igraph/louvain unavailable); official default is louvain with a resolution search.",
            "Primary metrics use the common tied-GMM + 15-neighbor refinement protocol; native DEC metrics are also recorded.",
            f"Uses {config.dtype} inputs; official tutorial also runs float32.",
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
    parser.add_argument("--num-pcs", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--n-clusters", type=int, default=None)
    parser.add_argument("--seed", type=int, default=100)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = SpaGcnConfig(
        num_pcs=args.num_pcs,
        max_epochs=args.epochs,
        learning_rate=args.lr,
        n_clusters=args.n_clusters,
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