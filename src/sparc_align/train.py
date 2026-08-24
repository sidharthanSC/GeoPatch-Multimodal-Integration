"""SPARC training and per-section DLPFC evaluation.

``python -m src.sparc_align.train --output-dir outputs/sparc_align/<run> [...]``

The CLI is a thin wrapper over :func:`fit_sparc` and :func:`run_dlpfc_sparc`, both
importable and fully parameterized by :class:`SparcConfig`.
"""

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

from .config import SparcConfig
from .data import SparcSectionData, prepare_section
from .evaluate import (
    activation_pattern_table,
    cluster_sparc_latents,
    reconstruction_report,
    support_jaccard,
)
from .model import SparcModel, nmse


@dataclass(frozen=True)
class SparcResult:
    """Everything one section's SPARC run produces."""

    latents: dict[str, np.ndarray]
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


def mask_columns(
    batch: dict[str, torch.Tensor], rate: float, generator: torch.Generator
) -> dict[str, torch.Tensor]:
    """Zero whole feature columns, matching MP-MNCA/STAIG's mask_features().

    Column-wise rather than element-wise: the same features are dropped for every
    spot in the batch, which removes a coherent slice of signal instead of adding
    iid noise, and is what the reference implementations do.
    """
    masked = {}
    for name, tensor in batch.items():
        drop = torch.rand(tensor.shape[1], generator=generator, device=tensor.device) < rate
        view = tensor.clone()
        view[:, drop] = 0
        masked[name] = view
    return masked


def _neighbor_batch(
    streams: dict[str, torch.Tensor], neighbor_idx: torch.Tensor, index: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Gather each selected spot's spatial neighbours, per stream."""
    neighbors = neighbor_idx[index]  # (batch, n_neighbors)
    return {name: tensor[neighbors] for name, tensor in streams.items()}


@torch.no_grad()
def encode_all(
    model: SparcModel,
    streams: dict[str, torch.Tensor],
    batch_size: int,
    neighbor_idx: torch.Tensor | None = None,
    neighbor_similarity: torch.Tensor | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, float]]:
    """Encode every spot, returning latents plus self/cross reconstruction NMSE."""
    model.eval()
    n_spots = next(iter(streams.values())).shape[0]
    collected: dict[str, list[np.ndarray]] = {name: [] for name in model.stream_names}
    self_totals: dict[str, list[float]] = {name: [] for name in model.stream_names}
    cross_totals: dict[str, list[float]] = {}
    needs_neighbors = model.config.spatial_topk_weight > 0.0

    for start in range(0, n_spots, batch_size):
        end = min(start + batch_size, n_spots)
        batch = {name: tensor[start:end] for name, tensor in streams.items()}
        neighbors = None
        similarity = None
        if needs_neighbors:
            index = torch.arange(start, end, device=neighbor_idx.device)
            neighbors = _neighbor_batch(streams, neighbor_idx, index)
            if neighbor_similarity is not None:
                similarity = neighbor_similarity[start:end]
        output = model(batch, neighbors, similarity)
        for name in model.stream_names:
            collected[name].append(output.latents[name].cpu().numpy())
            self_totals[name].append(float(nmse(batch[name], output.self_reconstructions[name])))
        for (source, target), recon in output.cross_reconstructions.items():
            cross_totals.setdefault(f"{source}_to_{target}", []).append(
                float(nmse(batch[target], recon))
            )

    latents = {name: np.concatenate(parts, axis=0) for name, parts in collected.items()}
    self_nmse = {name: float(np.mean(values)) for name, values in self_totals.items()}
    cross_nmse = {name: float(np.mean(values)) for name, values in cross_totals.items()}
    return latents, self_nmse, cross_nmse


def fit_sparc(
    section_data: SparcSectionData,
    config: SparcConfig,
    device: torch.device | str | None = None,
    verbose: bool = True,
    eval_epochs: tuple[int, ...] = (),
    on_checkpoint=None,
) -> SparcResult:
    """Train SPARC on one section and evaluate through the MP-MNCA protocol.

    ``eval_epochs`` requests mid-training evaluation at those epoch numbers, so one
    200-epoch run yields the whole convergence curve instead of five separate runs
    (which would each restart from scratch and not be points on a single trajectory).
    ``on_checkpoint(epoch, latents, self_nmse, cross_nmse)`` is called at each, and
    training resumes from the same optimizer state afterwards.
    """
    _set_seed(config.seed)
    device = torch.device(device or "cpu")
    dtype = torch.float32 if config.dtype == "float32" else torch.float64

    streams = {
        name: torch.as_tensor(array, dtype=dtype, device=device)
        for name, array in section_data.streams.items()
    }
    n_spots = next(iter(streams.values())).shape[0]
    neighbor_idx = torch.as_tensor(
        section_data.neighbor_indices, dtype=torch.long, device=device
    )
    neighbor_similarity = torch.as_tensor(
        section_data.neighbor_image_similarity, dtype=dtype, device=device
    )

    model = SparcModel(config, section_data.stream_dims).to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        betas=config.adam_betas,
        eps=config.adam_eps,
    )

    rng = np.random.default_rng(config.seed)
    mask_generator = torch.Generator(device=device).manual_seed(config.seed)
    indices = np.arange(n_spots)
    losses: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        rng.shuffle(indices)
        totals = {"total": 0.0, "self": 0.0, "cross": 0.0, "aux": 0.0}
        n_revived = 0

        for start in range(0, n_spots, config.batch_size):
            end = min(start + config.batch_size, n_spots)
            batch_idx = torch.as_tensor(indices[start:end], dtype=torch.long, device=device)
            batch = {name: tensor[batch_idx] for name, tensor in streams.items()}
            spatial = config.spatial_topk_weight > 0.0
            similarity = neighbor_similarity[batch_idx] if spatial else None

            optimizer.zero_grad(set_to_none=True)
            # Denoising: encode the masked view, but score reconstruction against the
            # clean batch, so the model must recover what was removed.
            #
            # The neighbour tokens are drawn from the SAME masked view as the centre,
            # not from the clean streams. Masking is column-wise, so a mixed pair
            # would have the centre missing genes its neighbours still carry -- the
            # spatial-TopK selection logits would then compare vectors living in
            # different feature spaces. MP-MNCA masks centre and neighbours together
            # for the same reason.
            if config.input_mask_rate > 0.0:
                masked_streams = mask_columns(streams, config.input_mask_rate, mask_generator)
                encoder_input = {name: t[batch_idx] for name, t in masked_streams.items()}
                neighbors = (
                    _neighbor_batch(masked_streams, neighbor_idx, batch_idx) if spatial else None
                )
            else:
                encoder_input = batch
                neighbors = _neighbor_batch(streams, neighbor_idx, batch_idx) if spatial else None

            output = model(encoder_input, neighbors, similarity)
            loss_dict = model.compute_losses(batch, output)
            loss_dict["total"].backward()
            model.pre_step()  # project decoder gradients off the unit sphere normal
            optimizer.step()
            model.post_step()  # renormalize decoder columns
            model.update_dead_latents(output)

            for key in totals:
                totals[key] += float(loss_dict[key]) * (end - start)

        n_revived = model.reinitialize_dead_latents()
        for key in totals:
            totals[key] /= n_spots
        totals["revived_latents"] = float(n_revived)
        losses.append(totals)

        if eval_epochs and epoch in eval_epochs and on_checkpoint is not None:
            # encode_all flips the model to eval(); restore train mode after.
            snapshot = encode_all(model, streams, config.batch_size, neighbor_idx, neighbor_similarity)
            on_checkpoint(epoch, *snapshot)
            model.train()

        if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == config.epochs):
            print(
                f"{section_data.section_id} epoch={epoch:03d} "
                f"total={totals['total']:.4f} self={totals['self']:.4f} "
                f"cross={totals['cross']:.4f} aux={totals['aux']:.4f} "
                f"revived={n_revived}",
                flush=True,
            )

    latents, self_nmse, cross_nmse = encode_all(
        model, streams, config.batch_size, neighbor_idx, neighbor_similarity
    )
    clustered = cluster_sparc_latents(
        latents, section_data.labels, section_data.coordinates, config
    )

    metrics = {
        **clustered["metrics"],
        **reconstruction_report(self_nmse, cross_nmse),
        **activation_pattern_table(latents),
        **support_jaccard(latents),
        "n_spots": int(n_spots),
        "final_loss": losses[-1]["total"],
    }
    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    return SparcResult(
        latents=latents,
        predictions=clustered["predictions"],
        refined_predictions=clustered["refined_predictions"],
        metrics=metrics,
        losses=losses,
        state_dict=state_dict,
    )


def run_dlpfc_sparc(
    checkpoint_path: Path,
    output_dir: Path,
    config: SparcConfig,
    sections: list[str] | None = None,
    device: str | None = None,
) -> dict:
    """Train and evaluate SPARC independently per section, persisting provenance."""
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {output_dir}")
    output_dir.mkdir(parents=True)
    (output_dir / "checkpoints").mkdir()
    (output_dir / "latents").mkdir()

    dataset = DlpfcDataset.from_checkpoint(checkpoint_path)
    section_ids = sections or dataset.section_ids()
    rows: list[dict] = []

    for section_id in section_ids:
        started = time.perf_counter()
        data = prepare_section(dataset.get_section(section_id), section_id, config)
        result = fit_sparc(data, config, device=device)
        elapsed = time.perf_counter() - started

        row = {"section_id": section_id, **result.metrics, "elapsed_seconds": elapsed}
        rows.append(row)

        torch.save(
            {
                "section_id": section_id,
                "config": config.to_dict(),
                "stream_dims": data.stream_dims,
                "state_dict": result.state_dict,
                "metrics": result.metrics,
            },
            output_dir / "checkpoints" / f"{section_id}.pt",
        )
        np.savez_compressed(
            output_dir / "latents" / f"{section_id}.npz",
            predictions=result.predictions,
            refined_predictions=result.refined_predictions,
            labels=data.labels,
            barcodes=data.barcodes,
            coordinates=data.coordinates,
            losses=np.asarray([entry["total"] for entry in result.losses], dtype=np.float32),
            **{f"latent_{name}": array for name, array in result.latents.items()},
        )
        print(json.dumps({k: v for k, v in row.items()}, sort_keys=True, default=float), flush=True)

    with (output_dir / "section_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    headline = ["ari", "nmi", "refined_ari", "refined_nmi", "self_nmse_mean", "cross_nmse_mean"]
    summary = {
        "run_id": output_dir.name,
        "implementation": "SPARC Global TopK cross-modal SAE (gene 3000-d + image 128-d)",
        "config": config.to_dict(),
        "sections": section_ids,
        "n_sections": len(section_ids),
        "metrics": {
            key: {
                "mean": float(np.mean([float(r[key]) for r in rows])),
                "median": float(np.median([float(r[key]) for r in rows])),
                "iqr": float(
                    np.percentile([float(r[key]) for r in rows], 75)
                    - np.percentile([float(r[key]) for r in rows], 25)
                ),
            }
            for key in headline
            if key in rows[0]
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
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-latents", type=int, default=1024)
    parser.add_argument("--k-active", type=int, default=32)
    parser.add_argument("--cross-loss-weight", type=float, default=1.0)
    parser.add_argument("--local-topk", action="store_true", help="Local TopK ablation")
    parser.add_argument(
        "--cluster-input", choices=["sum", "gene", "image", "concat", "support"], default="sum"
    )
    parser.add_argument("--refinement-neighbors", type=int, default=15)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = SparcConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        n_latents=args.n_latents,
        k_active=args.k_active,
        cross_loss_weight=args.cross_loss_weight,
        global_topk=not args.local_topk,
        cluster_input=args.cluster_input,
        refinement_neighbors=args.refinement_neighbors,
    )
    summary = run_dlpfc_sparc(
        args.checkpoint_path, args.output_dir, config, args.sections, args.device
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
