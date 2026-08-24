"""Spatial k-NN smoothing for SPARC-align: before alignment vs after.

SPARC currently uses the spatial graph in exactly one place -- ``spatial_topk_weight``
smooths the logits that decide *which* latents activate. The inputs, the
reconstruction targets and the latent values all remain strictly per-spot. This module
tests pushing spatial information further, in the two places it could go:

- **before**: smooth the gene and image streams, then run SPARC on the smoothed data.
  This is what STAIG and GraphST do implicitly through graph convolution. It targets
  the measured bottleneck -- gene self-NMSE sits at ~0.88, so a linear encoder cannot
  fit raw expression, and raising the stream's signal-to-noise should make it more
  competitive in the Global TopK vote against the much cleaner image stream.
  Note this also makes the reconstruction *target* smoothed: an autoencoder
  reconstructs whatever it is fed, so "faithfulness" now means faithfulness to the
  denoised signal, not the raw spot.

- **after**: leave SPARC untouched and smooth the latent code before stage 2. Keeps
  per-spot reconstruction honest, but may be redundant since stage 2's cross-attention
  already aggregates over the same k=6 graph.

Both use the morphology kernel already validated in the ablations,
``w_ij = softmax(beta * log s_ij)`` over image-PCA cosine similarity, so
morphologically dissimilar neighbours contribute less and boundaries blur less.

    python -m src.sparc_align.spatial_init --arms baseline before after
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

from .config import SparcConfig
from .convergence import BEST, load_cached_section, cache_sections
from .data import SparcSectionData
from .evaluate import build_cluster_input, cluster_sparc_latents
from .results_io import IncrementalCsv, results_dir, save_json
from .train import fit_sparc

# Blend weight for the neighbourhood term. 0.5 gives self and neighbourhood equal
# say, which is the closest analogue to STAIG's normalized adjacency with self-loops.
# (The validated 0.75 applies to logit selection, a weaker intervention than
# rewriting the data, so it is not carried over unchanged.)
SMOOTH_ALPHA = 0.5
MORPHOLOGY_BETA = 1.0


def neighbour_weights(similarity: np.ndarray, beta: float = MORPHOLOGY_BETA) -> np.ndarray:
    """softmax(beta * log s_ij) over each spot's neighbours -- the ablation's kernel."""
    logits = beta * np.log(np.clip(similarity, 1e-6, None))
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    return weights / weights.sum(axis=1, keepdims=True)


def smooth_matrix(
    matrix: np.ndarray, neighbor_indices: np.ndarray, weights: np.ndarray, alpha: float
) -> np.ndarray:
    """(1 - alpha) * x_i + alpha * sum_j w_ij x_j."""
    neighbourhood = np.einsum("ij,ijd->id", weights, matrix[neighbor_indices])
    return ((1.0 - alpha) * matrix + alpha * neighbourhood).astype(np.float32)


def smooth_streams(data: SparcSectionData, alpha: float) -> SparcSectionData:
    """Return a copy with both input streams spatially smoothed."""
    weights = neighbour_weights(data.neighbor_image_similarity)
    return replace(
        data,
        gene=smooth_matrix(data.gene, data.neighbor_indices, weights, alpha),
        image=smooth_matrix(data.image, data.neighbor_indices, weights, alpha),
    )


def run_arm(arm: str, section_id: str, config: SparcConfig, device: str, attention_device: str):
    """One arm on one section; returns the metric row."""
    from .attention_stage import fit_attention_stage

    data = load_cached_section(section_id)
    if arm == "before":
        data = smooth_streams(data, SMOOTH_ALPHA)

    result = fit_sparc(data, config, device=device, verbose=False)
    latents = result.latents

    if arm == "after":
        weights = neighbour_weights(data.neighbor_image_similarity)
        latents = {
            name: smooth_matrix(value, data.neighbor_indices, weights, SMOOTH_ALPHA)
            for name, value in latents.items()
        }

    row = {"arm": arm, "section_id": section_id}
    for mode in ("sum", "support"):
        m = cluster_sparc_latents(
            latents, data.labels, data.coordinates, replace(config, cluster_input=mode)
        )["metrics"]
        row[f"stage1_{mode}_ari"] = m["refined_ari"]
    best_mode = "support" if row["stage1_support_ari"] >= row["stage1_sum_ari"] else "sum"

    matrix = build_cluster_input(latents, best_mode).astype(np.float32)
    stage2 = fit_attention_stage(data, matrix, config, device=attention_device, verbose=False)
    row.update(
        ari=stage2.metrics["refined_ari"],
        nmi=stage2.metrics["refined_nmi"],
        stage2_input=best_mode,
        self_nmse_gene=result.metrics["self_nmse_gene"],
        self_nmse_image=result.metrics["self_nmse_image"],
        cross_nmse_image_to_gene=result.metrics["cross_nmse_image_to_gene"],
    )
    del stage2, matrix, result, latents
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return row


def main() -> None:
    global SMOOTH_ALPHA
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, default=Path("checkpoints/dlpfc.pkl"))
    parser.add_argument("--name", default="spatial_init_all12")
    parser.add_argument("--arms", nargs="*", default=["baseline", "before", "after"])
    parser.add_argument("--sections", nargs="*")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--attention-device", default="cpu")
    parser.add_argument("--alpha", type=float, default=SMOOTH_ALPHA)
    args = parser.parse_args()

    SMOOTH_ALPHA = args.alpha
    config = SparcConfig(**BEST, epochs=args.epochs)
    sections = args.sections or cache_sections(args.checkpoint_path, config)
    directory = results_dir(args.name)
    writer = IncrementalCsv(directory / "spatial_init_metrics.csv", resume=True)

    for arm in args.arms:
        print(f"\n=== arm {arm} ===", flush=True)
        for section_id in sections:
            started = time.perf_counter()
            row = run_arm(arm, section_id, config, args.device, args.attention_device)
            row["elapsed_seconds"] = time.perf_counter() - started
            writer.append(row)
            print(f"  {arm:9s} {section_id} ARI={row['ari']:.4f} "
                  f"(stage1 sup={row['stage1_support_ari']:.4f})", flush=True)

    save_json(args.name, {
        "arms": args.arms, "sections": sections, "epochs": args.epochs,
        "alpha": SMOOTH_ALPHA, "beta": MORPHOLOGY_BETA, "base_config": BEST,
        "n_rows": len(writer.rows),
    })
    print(f"\nwrote {len(writer.rows)} rows to outputs/sparc_align/{args.name}/", flush=True)


if __name__ == "__main__":
    main()
