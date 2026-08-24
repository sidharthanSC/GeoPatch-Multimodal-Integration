"""Convergence and batch-size studies for SPARC-align on DLPFC.

Two studies, both writing under ``outputs/sparc_align/``:

**Convergence** -- one 200-epoch stage-1 run per section, evaluated at epochs
10/20/50/100/200. Evaluating inside a single run matters: five separate runs would
each reinitialize and are not points on one trajectory, so their differences would
mix convergence with seed noise.

Stage 2 (cross-attention + neighbour-contrastive) stays pinned at 20 epochs at every
checkpoint. Its own epoch sweep was flat-to-noisy (20 -> 0.337, 50 -> 0.235,
100 -> 0.316, 200 -> 0.252 refined ARI on 151507), so varying it would plot noise
rather than convergence; stage 1 is the representation learner and is what varies.

**Batch size** -- 32/64/128/256/512 across all 12 sections at a fixed epoch budget.

    python -m src.sparc_align.convergence --study convergence
    python -m src.sparc_align.convergence --study batch --epochs 200
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from src.datasets.dlpfc import DlpfcDataset

from .attention_stage import fit_attention_stage
from .config import SparcConfig
from .data import SparcSectionData, prepare_section
from .evaluate import build_cluster_input, cluster_sparc_latents
from .results_io import IncrementalCsv, results_dir, save_json
from .train import fit_sparc

CACHE_DIR = Path("outputs/sparc_align/_prepared_sections")

EVAL_EPOCHS = (10, 20, 50, 100, 200)
BATCH_SIZES = (32, 64, 128, 256, 512)

# The best configuration from the 12-section ablations: paper-faithful Global TopK
# plus spatially-aggregated TopK with morphology weighting, then stage 2.
# (outputs/sparc_align/ablation_all12_v3_attention, arm n: 0.3740 mean refined ARI.)
BEST = dict(
    n_latents=1024,
    k_active=128,
    spatial_topk_weight=0.75,
    spatial_attention=True,
    attention_epochs=20,
)


def cache_sections(checkpoint_path: Path, config: SparcConfig, sections=None) -> list[str]:
    """Materialize each section's prepared streams to a small npz.

    Loading dlpfc.pkl costs ~4 GB resident. Doing that once here, instead of in every
    study process, is the difference between running and being OOM-killed on a 16 GB
    machine whose swap is already near capacity -- the cached arrays are ~50 MB each.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dataset = DlpfcDataset.from_checkpoint(checkpoint_path)
    ids = sections or dataset.section_ids()
    for section_id in ids:
        target = CACHE_DIR / f"{section_id}.npz"
        if target.exists():
            continue
        d = prepare_section(dataset.get_section(section_id), section_id, config)
        np.savez_compressed(
            target, gene=d.gene, image=d.image, gene_raw=d.gene_raw, image_pca=d.image_pca,
            coordinates=d.coordinates, labels=d.labels, barcodes=d.barcodes,
            neighbor_indices=d.neighbor_indices, pseudo_labels=d.pseudo_labels,
            neighbor_image_similarity=d.neighbor_image_similarity,
        )
        print(f"cached {section_id}", flush=True)
    return list(ids)


def load_cached_section(section_id: str) -> SparcSectionData:
    """Rebuild a prepared section from its cache, never touching dlpfc.pkl."""
    z = np.load(CACHE_DIR / f"{section_id}.npz", allow_pickle=True)
    return SparcSectionData(
        section_id=section_id, gene=z["gene"], image=z["image"], gene_raw=z["gene_raw"],
        image_pca=z["image_pca"], coordinates=z["coordinates"], labels=z["labels"].astype(str),
        barcodes=z["barcodes"].astype(str), neighbor_indices=z["neighbor_indices"],
        pseudo_labels=z["pseudo_labels"], neighbor_image_similarity=z["neighbor_image_similarity"],
    )


def _evaluate_latents(latents, data, config, run_stage2: bool, device: str):
    """Score one stage-1 snapshot: stage-1 alone (both inputs) and, optionally, stage 2."""
    out = {}
    for mode in ("sum", "support"):
        m = cluster_sparc_latents(
            latents, data.labels, data.coordinates, replace(config, cluster_input=mode)
        )["metrics"]
        out[f"stage1_{mode}_ari"] = m["refined_ari"]
        out[f"stage1_{mode}_nmi"] = m["refined_nmi"]

    # Headline metric: the full two-stage pipeline. Feed stage 2 whichever stage-1
    # input scored higher, matching how the ablation selected its reported arm.
    best_mode = "support" if out["stage1_support_ari"] >= out["stage1_sum_ari"] else "sum"
    if run_stage2:
        matrix = build_cluster_input(latents, best_mode).astype(np.float32)
        s2 = fit_attention_stage(data, matrix, config, device=device, verbose=False)
        out["ari"] = s2.metrics["refined_ari"]
        out["nmi"] = s2.metrics["refined_nmi"]
        out["stage2_input"] = best_mode
        out["predictions"] = s2.refined_predictions
        # Stage 2 holds the largest allocations in the pipeline: the neighbour
        # contrastive loss materializes several (n, n) matrices (~67 MB each at
        # n=4093) inside a retained autograd graph. Without an explicit release the
        # peak from one checkpoint is still resident when the next one starts, which
        # was enough to get this run OOM-killed three times on a 16 GB machine.
        del s2, matrix
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    else:
        out["ari"] = out[f"stage1_{best_mode}_ari"]
        out["nmi"] = out[f"stage1_{best_mode}_nmi"]
        out["stage2_input"] = f"none({best_mode})"
    return out


def run_convergence(
    dataset, sections, name, device, attention_device, run_stage2=True, save_predictions=()
):
    """One 200-epoch run per section, scored at each checkpoint."""
    directory = results_dir(name)
    writer = IncrementalCsv(directory / "convergence_metrics.csv", resume=True)
    config = SparcConfig(**BEST, epochs=max(EVAL_EPOCHS))
    preds_dir = directory / "predictions"

    for section_id in sections:
        data = prepare_section(dataset.get_section(section_id), section_id, config)
        started = time.perf_counter()

        def on_checkpoint(epoch, latents, self_nmse, cross_nmse, _sid=section_id, _d=data):
            scored = _evaluate_latents(latents, _d, config, run_stage2, attention_device)
            predictions = scored.pop("predictions", None)
            if predictions is not None and _sid in save_predictions:
                preds_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    preds_dir / f"{_sid}_epoch{epoch:03d}.npz",
                    predictions=predictions,
                    labels=_d.labels,
                    coordinates=_d.coordinates,
                    barcodes=_d.barcodes,
                )
            row = {
                "section_id": _sid,
                "epoch": epoch,
                **scored,
                "self_nmse_gene": self_nmse["gene"],
                "self_nmse_image": self_nmse["image"],
                "cross_nmse_image_to_gene": cross_nmse["image_to_gene"],
            }
            writer.append(row)
            print(
                f"  {_sid} epoch={epoch:3d} ARI={row['ari']:.4f} NMI={row['nmi']:.4f} "
                f"(stage1 sum={row['stage1_sum_ari']:.4f} sup={row['stage1_support_ari']:.4f})",
                flush=True,
            )

        fit_sparc(
            data, config, device=device, verbose=False,
            eval_epochs=EVAL_EPOCHS, on_checkpoint=on_checkpoint,
        )
        print(f"{section_id} done in {time.perf_counter() - started:.0f}s", flush=True)

    return writer.rows


def run_batch_study(dataset, sections, name, device, attention_device, epochs, run_stage2=True):
    """Batch-size sweep at a fixed epoch budget."""
    directory = results_dir(name)
    writer = IncrementalCsv(directory / "batch_metrics.csv", resume=True)

    for batch_size in BATCH_SIZES:
        config = SparcConfig(**BEST, epochs=epochs, batch_size=batch_size)
        print(f"\n=== batch_size={batch_size} ===", flush=True)
        for section_id in sections:
            data = load_cached_section(section_id)
            started = time.perf_counter()
            result = fit_sparc(data, config, device=device, verbose=False)
            scored = _evaluate_latents(result.latents, data, config, run_stage2, attention_device)
            scored.pop("predictions", None)
            row = {
                "section_id": section_id,
                "batch_size": batch_size,
                "epochs": epochs,
                **scored,
                "self_nmse_gene": result.metrics["self_nmse_gene"],
                "elapsed_seconds": time.perf_counter() - started,
            }
            writer.append(row)
            print(f"  bs={batch_size:4d} {section_id} ARI={row['ari']:.4f}", flush=True)

    return writer.rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, default=Path("checkpoints/dlpfc.pkl"))
    parser.add_argument("--study", choices=["convergence", "batch"], default="convergence")
    parser.add_argument("--name", default=None)
    parser.add_argument("--sections", nargs="*")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--attention-device", default="cpu")
    parser.add_argument("--epochs", type=int, default=200, help="batch study only")
    parser.add_argument("--no-stage2", action="store_true")
    parser.add_argument(
        "--save-predictions", nargs="*", default=["151507"],
        help="sections whose per-checkpoint cluster assignments to store, for the annotation plots",
    )
    args = parser.parse_args()

    dataset = DlpfcDataset.from_checkpoint(args.checkpoint_path)
    sections = args.sections or dataset.section_ids()
    name = args.name or (
        "convergence_all12" if args.study == "convergence" else "batch_size_all12"
    )

    if args.study == "convergence":
        rows = run_convergence(
            dataset, sections, name, args.device, args.attention_device,
            not args.no_stage2, set(args.save_predictions),
        )
        meta = {"eval_epochs": list(EVAL_EPOCHS)}
    else:
        rows = run_batch_study(
            dataset, sections, name, args.device, args.attention_device,
            args.epochs, not args.no_stage2,
        )
        meta = {"batch_sizes": list(BATCH_SIZES), "epochs": args.epochs}

    save_json(name, {
        "study": args.study, "sections": sections, "n_rows": len(rows),
        "base_config": BEST, "stage2_epochs": BEST["attention_epochs"], **meta,
    })
    print(f"\nwrote {len(rows)} rows to outputs/sparc_align/{name}/", flush=True)


if __name__ == "__main__":
    main()
