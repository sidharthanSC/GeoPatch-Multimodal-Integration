"""MP-MNCA Contrastive Training (STAIG-style: full-graph forward, augmented edges for loss)."""

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
from src.prior_models.staig.data import (
    build_spatial_graph,
    image_guided_edge_probabilities,
    sample_augmented_edges,
)
from src.prior_models.staig.evaluate import clustering_metrics, refine_labels, tied_gmm
from src.prior_models.staig.model import mask_features, normalized_adjacency

from .config import MpMncaConfig
from .data import MpMncaSectionData, prepare_image_features, prepare_section
from .model import contrastive_loss
from .encoder import MpMncaEncoder, MpMncaOutput


@dataclass(frozen=True)
class MpMncaContrastiveResult:
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


def prepare_section_contrastive(
    adata,
    section_id: str,
    config: MpMncaConfig,
    n_neighbors: int = 6,
    image_key: str = "img_emb",
    hvg_key: str = "highly_variable",
    use_feat_obsm: bool = True,
) -> MpMncaSectionData:
    """Prepare section for contrastive MP-MNCA (mirrors STAIG data prep)."""
    if config.use_byol_encoder:
        # BYOL encoder needs raw 3000-dim HVG expression
        from .data import prepare_gene_expression
        features = prepare_gene_expression(adata, hvg_key, config.gene_input_dim, use_feat_obsm)
    elif config.use_pretrained_gene_emb:
        if "gene_emb" not in adata.obsm:
            raise KeyError("adata.obsm['gene_emb'] not found")
        features = np.asarray(adata.obsm["gene_emb"], dtype=np.float32)
        if not np.isfinite(features).all():
            raise ValueError("Pre-trained gene_emb contains non-finite values")
    else:
        from .data import prepare_gene_expression
        features = prepare_gene_expression(adata, hvg_key, config.gene_input_dim, use_feat_obsm)

    image_features = prepare_image_features(adata, config.image_pca_dim, image_key)
    coordinates = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    labels = adata.obs["ground_truth"].astype(str).to_numpy()
    barcodes = adata.obs_names.astype(str).to_numpy()

    # Build k-NN neighbor indices for cross-attention: (n_spots, n_neighbors)
    from sklearn.neighbors import NearestNeighbors
    finder = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(coordinates)
    _, indices = finder.kneighbors(coordinates)
    neighbor_indices = indices[:, 1:]  # exclude self

    # STAIG-style edge index (2, n_edges) for contrastive loss
    edge_index = build_spatial_graph(coordinates, n_neighbors)
    edge_probability = image_guided_edge_probabilities(edge_index, image_features)

    return MpMncaSectionData(
        section_id=section_id,
        gene_expression=features,
        image_features=image_features,
        coordinates=coordinates,
        labels=labels,
        barcodes=barcodes,
        neighbor_indices=neighbor_indices,        # (n_spots, n_neighbors) for cross-attention
        gene_mask=None,
        n_neighbors=n_neighbors,
    )


def fit_mp_mnca_contrastive(
    section_data: MpMncaSectionData,
    config: MpMncaConfig,
    device: torch.device | str | None = None,
) -> MpMncaContrastiveResult:
    """Fit MP-MNCA with STAIG-style neighbor contrastive loss."""
    _set_seed(config.seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float32 if config.dtype == "float32" else torch.float64

    n_spots = section_data.gene_expression.shape[0]

    # Model
    model = MpMncaEncoder(config)
    model = model.to(device=device, dtype=dtype)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    numpy_rng = np.random.default_rng(config.seed)
    torch_generator = torch.Generator(device=device).manual_seed(config.seed)

    losses: list[float] = []

    # Tensors
    features = torch.as_tensor(section_data.gene_expression, dtype=dtype, device=device)  # (n_spots, gene_dim)
    center_image = torch.as_tensor(section_data.image_features, dtype=dtype, device=device)
    center_coords = torch.as_tensor(section_data.coordinates, dtype=dtype, device=device)
    neighbor_idx = torch.as_tensor(section_data.neighbor_indices, dtype=torch.long, device=device)  # (n_spots, n_neighbors)

    # Neighbor features for cross-attention
    neighbor_genes = features[neighbor_idx]          # (n_spots, n_neighbors, gene_dim)
    neighbor_images = center_image[neighbor_idx]     # (n_spots, n_neighbors, image_dim)
    neighbor_coords = center_coords[neighbor_idx]    # (n_spots, n_neighbors, 2)

    # STAIG edge index for contrastive loss: build from neighbor_indices
    # neighbor_indices: (n_spots, n_neighbors) -> edge_index: (2, n_spots * n_neighbors)
    src = np.repeat(np.arange(n_spots), section_data.neighbor_indices.shape[1])
    dst = section_data.neighbor_indices.ravel()
    edge_index_np = np.stack([src, dst], axis=0).astype(np.int64)
    edge_index = torch.as_tensor(edge_index_np, dtype=torch.long, device=device)

    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        # Sample two augmented edge sets (STAIG-style)
        # Need drop probabilities for each edge
        n_edges = edge_index_np.shape[1]
        edge_drop_prob = np.ones(n_edges) * 0.1
        edges_1 = torch.as_tensor(
            sample_augmented_edges(edge_index_np, edge_drop_prob, numpy_rng),
            dtype=torch.long, device=device
        )
        edges_2 = torch.as_tensor(
            sample_augmented_edges(edge_index_np, edge_drop_prob, numpy_rng),
            dtype=torch.long, device=device
        )

        adjacency_1 = normalized_adjacency(edges_1, n_spots, dtype=dtype, device=device)
        adjacency_2 = normalized_adjacency(edges_2, n_spots, dtype=dtype, device=device)

        # Mask features for two views (STAIG-style feature masking)
        masked_features_1 = mask_features(features, config.mask_rate, torch_generator)
        masked_features_2 = mask_features(features, config.mask_rate, torch_generator)

        # ---- MP-MNCA Forward Pass (Full Graph) ----
        # We need to compute embeddings for all spots
        # Process in batches to avoid OOM
        batch_size = config.batch_size
        n_batches = (n_spots + batch_size - 1) // batch_size

        # View 1 embeddings
        all_emb_1 = []
        for b in range(n_batches):
            start = b * batch_size
            end = min(start + batch_size, n_spots)
            idx = torch.arange(start, end, device=device)

            out = model.forward(
                masked_features_1[idx],                    # center genes (masked)
                neighbor_genes[idx],                       # neighbor genes (clean for view 1? STAIG masks both)
                center_image[idx],
                neighbor_images[idx],
                center_coords[idx],
                neighbor_coords[idx],
            )
            all_emb_1.append(out.embeddings)

        z1 = torch.cat(all_emb_1, dim=0)  # (n_spots, embed_dim)

        # View 2: different feature masking
        masked_features_2 = mask_features(features, config.mask_rate, torch_generator)

        all_emb_2 = []
        for b in range(n_batches):
            start = b * batch_size
            end = min(start + batch_size, n_spots)
            idx = torch.arange(start, end, device=device)

            out = model.forward(
                masked_features_2[idx],
                neighbor_genes[idx],
                center_image[idx],
                neighbor_images[idx],
                center_coords[idx],
                neighbor_coords[idx],
            )
            all_emb_2.append(out.embeddings)

        z2 = torch.cat(all_emb_2, dim=0)

        # Contrastive loss on full graph (STAIG)
        loss = contrastive_loss(z1, z2, edge_index, config.temperature)

        loss.backward()
        optimizer.step()

        losses.append(float(loss.detach().cpu()))

        if epoch == 1 or epoch % 20 == 0 or epoch == config.epochs:
            print(f"{section_data.section_id} epoch={epoch:03d} loss={losses[-1]:.6f}", flush=True)

    # Evaluation: full forward pass with clean features
    model.eval()
    with torch.no_grad():
        all_emb = []
        for b in range(n_batches):
            start = b * batch_size
            end = min(start + batch_size, n_spots)
            idx = torch.arange(start, end, device=device)

            out = model.forward(
                features[idx],
                neighbor_genes[idx],
                center_image[idx],
                neighbor_images[idx],
                center_coords[idx],
                neighbor_coords[idx],
            )
            all_emb.append(out.embeddings)

        embeddings = torch.cat(all_emb, dim=0).cpu().numpy().astype(np.float32)

    # Clustering
    n_clusters = len(np.unique(section_data.labels))
    predictions = tied_gmm(embeddings, n_clusters=n_clusters)
    refined = refine_labels(predictions, section_data.coordinates, config.refinement_neighbors)
    before = clustering_metrics(section_data.labels, predictions)
    after = clustering_metrics(section_data.labels, refined)
    metrics = {
        **before,
        "refined_ari": after["ari"],
        "refined_nmi": after["nmi"],
        "n_spots": int(n_spots),
        "n_clusters": int(n_clusters),
        "final_loss": losses[-1],
    }

    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    return MpMncaContrastiveResult(
        embeddings=embeddings,
        predictions=predictions,
        refined_predictions=refined,
        metrics=metrics,
        losses=losses,
        state_dict=state_dict,
    )


def run_dlpfc_contrastive(
    checkpoint_path: Path,
    output_dir: Path,
    config: MpMncaConfig,
    sections: list[str] | None = None,
    device: str | None = None,
) -> dict:
    """Run MP-MNCA contrastive on DLPFC sections."""
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

        data = prepare_section_contrastive(adata, section_id, config)
        result = fit_mp_mnca_contrastive(data, config, device=device)
        elapsed = time.perf_counter() - started

        row = {"section_id": section_id, **result.metrics, "elapsed_seconds": elapsed}
        rows.append(row)

        torch.save(
            {
                "section_id": section_id,
                "config": config.to_dict(),
                "input_dim": int(data.gene_expression.shape[1]),
                "state_dict": result.state_dict,
                "metrics": result.metrics,
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
        "implementation": "MP-MNCA Contrastive",
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
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--mask-rate", type=float, default=0.1)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--n-neighbors", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=10.0)
    parser.add_argument("--image-pseudo-clusters", type=int, default=40)
    parser.add_argument("--image-pca-dim", type=int, default=16)
    parser.add_argument("--refinement-neighbors", type=int, default=15)
    parser.add_argument("--morphology-prior-type", choices=["cosine", "rbf", "gene_cosine"], default="cosine")
    parser.add_argument("--use-bylol-encoder", action="store_true", default=False, help="Use BYOL encoder (3000->3000)")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = MpMncaConfig(
        epochs=args.epochs,
        seed=args.seed,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        mask_rate=args.mask_rate,
        dtype=args.dtype,
        num_heads=args.num_heads,
        temperature=args.temperature,
        image_pseudo_clusters=args.image_pseudo_clusters,
        image_pca_dim=args.image_pca_dim,
        refinement_neighbors=args.refinement_neighbors,
        morphology_prior_type=args.morphology_prior_type,
        use_pretrained_gene_emb=not args.use_bylol_encoder,
        freeze_gene_encoder=True,
        use_byol_encoder=args.use_bylol_encoder,
    )
    summary = run_dlpfc_contrastive(
        args.checkpoint_path,
        args.output_dir,
        config,
        sections=args.sections,
        device=args.device,
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()