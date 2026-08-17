"""MP-MNCA training loop."""

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
from .data import (
    MpMncaSectionData,
    collate_mp_mnca_batch,
    prepare_all_sections,
    prepare_section,
)
from .model import (
    MpMncaModel,
    MpMncaOutput,
    covariance_regularization,
    variance_regularization,
)


@dataclass(frozen=True)
class MpMncaResult:
    embeddings: np.ndarray
    predictions: np.ndarray
    refined_predictions: np.ndarray
    metrics: dict[str, float]
    losses: list[dict[str, float]]
    state_dict: dict[str, torch.Tensor]
    attention_weights: np.ndarray | None = None
    morphology_prior: np.ndarray | None = None


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


def fit_mp_mnca(
    section_data: MpMncaSectionData,
    config: MpMncaConfig,
    device: torch.device | str | None = None,
) -> MpMncaResult:
    """Fit MP-MNCA model for one tissue section."""
    _set_seed(config.seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float32 if config.dtype == "float32" else torch.float64

    n_spots = section_data.gene_expression.shape[0]
    n_neighbors = section_data.n_neighbors

    # Create model
    model = MpMncaModel(config)
    model = model.to(device=device, dtype=dtype)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    # Random generators
    numpy_rng = np.random.default_rng(config.seed)
    torch_generator = torch.Generator(device=device).manual_seed(config.seed)

    losses: list[dict[str, float]] = []

    # Create index array for batching
    indices = np.arange(n_spots)
    n_batches = (n_spots + config.batch_size - 1) // config.batch_size

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_losses = {
            "total": 0.0,
            "latent": 0.0,
            "variance": 0.0,
            "covariance": 0.0,
        }

        # Shuffle indices each epoch
        numpy_rng.shuffle(indices)

        for batch_idx in range(n_batches):
            start = batch_idx * config.batch_size
            end = min(start + config.batch_size, n_spots)
            batch_indices = indices[start:end]

            optimizer.zero_grad(set_to_none=True)

            # Collate batch
            batch = collate_mp_mnca_batch(
                section_data, batch_indices, config, torch_generator
            )

            # Online forward (masked input)
            output: MpMncaOutput = model.forward_online(
                batch["center_gene_masked"],
                batch["neighbor_genes_masked"],
                batch["center_image"],
                batch["neighbor_images"],
                batch["center_coords"],
                batch["neighbor_coords"],
            )

            # Target forward (clean input, stop-gradient)
            with torch.no_grad():
                target_emb = model.forward_target(batch["center_gene_clean"])

            # Latent prediction loss
            predicted = model.predict(output.contextual_embedding)
            latent_loss = latent_prediction_loss(predicted, target_emb)

            # Anti-collapse regularization on online embeddings
            var_loss = variance_regularization(
                output.online_embedding, gamma=config.variance_gamma
            )
            cov_loss = covariance_regularization(
                output.online_embedding, max_dimensions=config.covariance_max_dim
            )

            # Total loss
            loss = (
                latent_loss
                + config.variance_weight * var_loss
                + config.covariance_weight * cov_loss
            )

            loss.backward()
            optimizer.step()

            # Update EMA target encoder
            if config.use_ema_target:
                model.update_target_encoder()

            # Accumulate losses
            epoch_losses["total"] += loss.item() * len(batch_indices)
            epoch_losses["latent"] += latent_loss.item() * len(batch_indices)
            epoch_losses["variance"] += var_loss.item() * len(batch_indices)
            epoch_losses["covariance"] += cov_loss.item() * len(batch_indices)

        # Average losses over epoch
        for key in epoch_losses:
            epoch_losses[key] /= n_spots

        losses.append(epoch_losses)

        if epoch == 1 or epoch % 20 == 0 or epoch == config.epochs:
            print(
                f"{section_data.section_id} epoch={epoch:03d} "
                f"total={epoch_losses['total']:.6f} "
                f"latent={epoch_losses['latent']:.6f} "
                f"var={epoch_losses['variance']:.6f} "
                f"cov={epoch_losses['covariance']:.6f}",
                flush=True,
            )

    # Evaluation: generate embeddings for all spots
    model.eval()
    with torch.no_grad():
        # Process in batches
        all_embeddings = []
        all_attention = []
        all_morph_prior = []

        for batch_idx in range(n_batches):
            start = batch_idx * config.batch_size
            end = min(start + config.batch_size, n_spots)
            batch_indices = indices[start:end]

            batch = collate_mp_mnca_batch(
                section_data, batch_indices, config, torch_generator
            )

            output = model.forward_online(
                batch["center_gene_masked"],
                batch["neighbor_genes_masked"],
                batch["center_image"],
                batch["neighbor_images"],
                batch["center_coords"],
                batch["neighbor_coords"],
            )

            all_embeddings.append(output.contextual_embedding.cpu().numpy())
            all_attention.append(output.attention_weights.cpu().numpy())
            all_morph_prior.append(output.morphology_prior.cpu().numpy())

        embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
        attention_weights = np.concatenate(all_attention, axis=0).astype(np.float32)
        morphology_prior = np.concatenate(all_morph_prior, axis=0).astype(np.float32)

    # Clustering evaluation
    from .evaluate import clustering_metrics, refine_labels, tied_gmm

    n_clusters = config.n_clusters or len(np.unique(section_data.labels))
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
        "final_loss": losses[-1]["total"],
    }

    state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    return MpMncaResult(
        embeddings=embeddings,
        predictions=predictions,
        refined_predictions=refined,
        metrics=metrics,
        losses=losses,
        state_dict=state_dict,
        attention_weights=attention_weights,
        morphology_prior=morphology_prior,
    )


def run_dlpfc(
    checkpoint_path: Path,
    output_dir: Path,
    config: MpMncaConfig,
    sections: list[str] | None = None,
    device: str | None = None,
) -> dict:
    """Run MP-MNCA on DLPFC sections."""
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {output_dir}")
    output_dir.mkdir(parents=True)
    (output_dir / "checkpoints").mkdir()
    (output_dir / "embeddings").mkdir()
    (output_dir / "diagnostics").mkdir()

    dataset = DlpfcDataset.from_checkpoint(checkpoint_path)
    section_ids = sections or dataset.section_ids()
    rows: list[dict] = []

    for section_id in section_ids:
        started = time.perf_counter()
        adata = dataset.get_section(section_id)
        expected_barcodes = adata.obs_names.astype(str).to_numpy()

        data = prepare_section(adata, section_id, config, use_feat_obsm=True)
        result = fit_mp_mnca(data, config, device=device)
        elapsed = time.perf_counter() - started

        row = {"section_id": section_id, **result.metrics, "elapsed_seconds": elapsed}
        rows.append(row)

        # Save checkpoint
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

        # Save embeddings and diagnostics
        np.savez_compressed(
            output_dir / "embeddings" / f"{section_id}.npz",
            embeddings=result.embeddings,
            predictions=result.predictions,
            refined_predictions=result.refined_predictions,
            labels=data.labels,
            barcodes=data.barcodes,
            coordinates=data.coordinates,
            losses=np.asarray([l["total"] for l in result.losses], dtype=np.float32),
            attention_weights=result.attention_weights,
            morphology_prior=result.morphology_prior,
        )

        # Save diagnostics separately
        np.savez_compressed(
            output_dir / "diagnostics" / f"{section_id}.npz",
            attention_weights=result.attention_weights,
            morphology_prior=result.morphology_prior,
        )

        print(json.dumps(row, sort_keys=True), flush=True)

    # Save section metrics CSV
    import csv
    with (output_dir / "section_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Save summary
    numeric = ["ari", "nmi", "refined_ari", "refined_nmi"]
    summary = {
        "run_id": output_dir.name,
        "implementation": "MP-MNCA",
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
    parser.add_argument("--mask-rate", type=float, default=0.3)
    parser.add_argument("--masking-type", choices=["scalar", "contiguous_block", "gene_module", "neighbourhood"], default="contiguous_block")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--n-neighbors", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--variance-weight", type=float, default=1.0)
    parser.add_argument("--covariance-weight", type=float, default=0.5)
    parser.add_argument("--gene-input-dim", type=int, default=3000)
    parser.add_argument("--use-pretrained-gene-emb", action="store_true", default=False)
    parser.add_argument("--freeze-gene-encoder", action="store_true", default=False)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = MpMncaConfig(
        epochs=args.epochs,
        seed=args.seed,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        mask_rate=args.mask_rate,
        masking_type=args.masking_type,
        dtype=args.dtype,
        num_heads=args.num_heads,
        variance_weight=args.variance_weight,
        covariance_weight=args.covariance_weight,
        gene_input_dim=args.gene_input_dim,
        use_pretrained_gene_emb=args.use_pretrained_gene_emb,
        freeze_gene_encoder=args.freeze_gene_encoder,
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