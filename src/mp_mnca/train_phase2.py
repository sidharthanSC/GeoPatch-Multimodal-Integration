"""MP-MNCA Phase 2 Training: BYOL Encoder 3000->3000 with Reconstruction."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from src.datasets.dlpfc import DlpfcDataset
from .config import MpMncaConfig
from .data import MpMncaSectionData, prepare_section
from .phase2 import Phase2Model, Phase2Output


@dataclass(frozen=True)
class Phase2Result:
    embeddings: np.ndarray
    predictions: np.ndarray
    refined_predictions: np.ndarray
    metrics: dict[str, float]
    losses: list[dict[str, float]]
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


def prepare_section_phase2(
    adata,
    section_id: str,
    config: MpMncaConfig,
    n_neighbors: int = 6,
    image_key: str = "img_emb",
    hvg_key: str = "highly_variable",
    use_feat_obsm: bool = True,
) -> MpMncaSectionData:
    """Prepare section for Phase 2 (uses raw 3000-dim HVG expression)."""
    from .data import prepare_gene_expression
    features = prepare_gene_expression(adata, hvg_key, config.gene_dim, use_feat_obsm)

    image_features = np.asarray(adata.obsm["img_emb"], dtype=np.float32)
    image_features = image_features[:, :config.image_embedding_dim]  # use raw 128-dim
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    standardized = StandardScaler().fit_transform(image_features)
    image_features = PCA(n_components=config.image_pca_dim, random_state=42).fit_transform(standardized).astype(np.float32)

    coordinates = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    labels = adata.obs["ground_truth"].astype(str).to_numpy()
    barcodes = adata.obs_names.astype(str).to_numpy()

    from sklearn.neighbors import NearestNeighbors
    finder = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(coordinates)
    _, indices = finder.kneighbors(coordinates)
    neighbor_indices = indices[:, 1:]

    return MpMncaSectionData(
        section_id=section_id,
        gene_expression=features,
        image_features=image_features,
        coordinates=coordinates,
        labels=labels,
        barcodes=barcodes,
        neighbor_indices=neighbor_indices,
        gene_mask=None,
        n_neighbors=n_neighbors,
    )


def fit_phase2(
    section_data: MpMncaSectionData,
    config: MpMncaConfig,
    device: torch.device | str | None = None,
) -> Phase2Result:
    """Fit Phase 2 with reconstruction loss on 3000-dim embeddings."""
    _set_seed(config.seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float32 if config.dtype == "float32" else torch.float64

    n_spots = section_data.gene_expression.shape[0]

    model = Phase2Model(config)
    model = model.to(device=device, dtype=dtype)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    numpy_rng = np.random.default_rng(config.seed)
    torch_generator = torch.Generator(device=device).manual_seed(config.seed)

    losses: list[dict[str, float]] = []

    features = torch.as_tensor(section_data.gene_expression, dtype=dtype, device=device)

    batch_size = config.batch_size
    n_batches = (n_spots + batch_size - 1) // batch_size
    indices = np.arange(n_spots)

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_losses = {"total": 0.0, "reconstruction": 0.0, "byol": 0.0}

        numpy_rng.shuffle(indices)

        for b in range(n_batches):
            start = b * batch_size
            end = min(start + batch_size, n_spots)
            idx = torch.arange(start, end, device=device)

            optimizer.zero_grad(set_to_none=True)

            batch_features = features[idx]
            output = model.forward(batch_features, torch_generator)
            losses_dict = model.compute_losses(output, batch_features)

            loss = losses_dict["total"]
            loss.backward()
            optimizer.step()

            # EMA update
            model.byol_encoder.update_target_encoder()

            for k in epoch_losses:
                epoch_losses[k] += losses_dict[k].item() * len(idx)

        for k in epoch_losses:
            epoch_losses[k] /= n_spots

        losses.append(epoch_losses)

        if epoch == 1 or epoch % 20 == 0 or epoch == config.epochs:
            print(f"{section_data.section_id} epoch={epoch:03d} "
                  f"total={epoch_losses['total']:.6f} "
                  f"recon={epoch_losses['reconstruction']:.6f} "
                  f"byol={epoch_losses['byol']:.6f}", flush=True)

    # Evaluation: get encoder embeddings
    model.eval()
    with torch.no_grad():
        all_emb = []
        for b in range(n_batches):
            start = b * batch_size
            end = min(start + batch_size, n_spots)
            idx = torch.arange(start, end, device=device)

            output = model.forward(features[idx])
            all_emb.append(output.spot_embeddings.cpu().numpy())

        embeddings = np.concatenate(all_emb, axis=0).astype(np.float32)

    # Clustering (PCA for 3000-dim)
    from sklearn.decomposition import PCA
    pca_dim = min(128, embeddings.shape[0], embeddings.shape[1])
    embeddings_pca = PCA(n_components=pca_dim, random_state=0).fit_transform(embeddings)

    from .evaluate import clustering_metrics, refine_labels, tied_gmm
    n_clusters = config.n_clusters or len(np.unique(section_data.labels))
    predictions = tied_gmm(embeddings_pca, n_clusters=n_clusters)
    refined = refine_labels(predictions, section_data.coordinates, config.refinement_neighbors)
    before = clustering_metrics(section_data.labels, predictions)
    after = clustering_metrics(section_data.labels, refined)
    metrics = {
        **before,
        "refined_ari": after["ari"],
        "refined_nmi": after["nmi"],
        "n_spots": int(n_spots),
        "n_clusters": int(n_clusters),
        "final_loss": losses[-1]["total"],
    }

    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    return Phase2Result(
        embeddings=embeddings,
        predictions=predictions,
        refined_predictions=refined,
        metrics=metrics,
        losses=losses,
        state_dict=state_dict,
    )


def run_dlpfc_phase2(
    checkpoint_path: Path,
    output_dir: Path,
    config: MpMncaConfig,
    sections: list[str] | None = None,
    device: str | None = None,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite: {output_dir}")
    output_dir.mkdir(parents=True)
    (output_dir / "checkpoints").mkdir()
    (output_dir / "embeddings").mkdir()

    dataset = DlpfcDataset.from_checkpoint(checkpoint_path)
    section_ids = sections or dataset.section_ids()
    rows: list[dict] = []

    for section_id in section_ids:
        started = time.perf_counter()
        adata = dataset.get_section(section_id)
        data = prepare_section_phase2(adata, section_id, config)
        result = fit_phase2(data, config, device=device)
        elapsed = time.perf_counter() - started

        row = {"section_id": section_id, **result.metrics, "elapsed_seconds": elapsed}
        rows.append(row)

        torch.save(
            {"section_id": section_id, "config": config.to_dict(),
             "input_dim": int(data.gene_expression.shape[1]),
             "state_dict": result.state_dict, "metrics": result.metrics},
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
            losses=np.asarray([l["total"] for l in result.losses], dtype=np.float32),
        )

        print(json.dumps(row, sort_keys=True), flush=True)

    import csv
    with (output_dir / "section_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    numeric = ["ari", "nmi", "refined_ari", "refined_nmi"]
    summary = {
        "run_id": output_dir.name,
        "implementation": "MP-MNCA Phase 2 (BYOL 3000->3000 + Reconstruction)",
        "config": config.to_dict(),
        "sections": section_ids,
        "n_sections": len(section_ids),
        "metrics": {k: {"mean": float(np.mean([float(r[k]) for r in rows])),
                        "median": float(np.median([float(r[k]) for r in rows])),
                        "iqr": float(np.percentile([float(r[k]) for r in rows], 75) - np.percentile([float(r[k]) for r in rows], 25))}
                   for k in numeric},
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
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--mask-rate", type=float, default=0.3)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--n-neighbors", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=10.0)
    parser.add_argument("--image-pseudo-clusters", type=int, default=40)
    parser.add_argument("--image-pca-dim", type=int, default=16)
    parser.add_argument("--refinement-neighbors", type=int, default=15)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = MpMncaConfig(
        epochs=args.epochs, seed=args.seed, batch_size=args.batch_size,
        learning_rate=args.lr, mask_rate=args.mask_rate, dtype=args.dtype,
        num_heads=args.num_heads, temperature=args.temperature,
        image_pseudo_clusters=args.image_pseudo_clusters,
        image_pca_dim=args.image_pca_dim,
        refinement_neighbors=args.refinement_neighbors,
    )
    summary = run_dlpfc_phase2(args.checkpoint_path, args.output_dir, config, args.sections, args.device)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()