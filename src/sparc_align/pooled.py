"""Pooled SPARC: one dictionary across all 12 DLPFC sections.

Phase 1 currently fits a separate 1024-atom dictionary per section on ~4,200 spots
-- roughly 4 spots per atom, against the SPARC paper's ~200 (8,192 atoms on 1.7M
Open Images samples). The encoder alone is 1024x3000 = 3.07M parameters fit on 4,221
samples, i.e. badly underdetermined. Pooling gives 47,329 spots, ~46 per atom.

**Primary readout is gene self-NMSE, not ARI.** It sits at ~0.88 per-section, which is
the root of everything downstream: it is why the Global TopK support is
image-dominated, why image->gene cross-NMSE is 0.92, and why the annotation figures
cannot resolve layer boundaries. If 11x the data moves it, data starvation is
confirmed. If it stays at 0.88, a linear encoder simply cannot represent this
expression and no amount of pooling or capacity will fix it -- a decisive negative
that would redirect effort to the encoder architecture.

Phase 2 stays per-section: its spatial graph is within-section by construction.

Image batch correction: img_emb remains section-predictable at 28% (chance 8.3%) even
after per-section standardization, i.e. slide-level staining/illumination structure
survives centering. Pooling without correction would let the shared dictionary spend
atoms on slide identity, and it would land on the stream that already dominates the
support. ``--batch-correct`` projects out the leading section-discriminative
directions. Section identity is a technical covariate, not the biological label, so
using it is not label leakage.

    python -m src.sparc_align.pooled --smooth-alpha 0.5 --batch-correct
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
from .convergence import BEST, cache_sections, load_cached_section
from .evaluate import build_cluster_input, cluster_sparc_latents
from .results_io import IncrementalCsv, results_dir, save_json
from .train import fit_sparc

SECTIONS = ["151507", "151508", "151509", "151510", "151669", "151670",
            "151671", "151672", "151673", "151674", "151675", "151676"]


def project_out_section_directions(matrix: np.ndarray, section_ids: np.ndarray, n_dirs: int = 4):
    """Remove the leading directions that separate sections.

    Per-section standardization already zeroes each section's mean, so what remains is
    higher-order structure. This fits section-mean directions in the standardized space
    and projects them out, which is a linear, assumption-light batch correction --
    weaker than Harmony/ComBat but with no extra dependency and no risk of distorting
    biology through a nonlinear warp.
    """
    means = np.stack([matrix[section_ids == s].mean(0) for s in np.unique(section_ids)])
    means -= means.mean(0, keepdims=True)
    # Orthonormal basis of the space spanned by section-mean offsets.
    u, s, _ = np.linalg.svd(means, full_matrices=False)
    basis = (means.T @ u[:, :n_dirs]) / np.maximum(s[:n_dirs], 1e-8)
    basis, _ = np.linalg.qr(basis)
    return (matrix - (matrix @ basis) @ basis.T).astype(np.float32)


def build_pooled(smooth_alpha: float, batch_correct: bool):
    """Concatenate all sections into one training set, optionally corrected."""
    from .spatial_init import smooth_streams

    sections, gene, image, offsets = [], [], [], {}
    cursor = 0
    for section_id in SECTIONS:
        data = load_cached_section(section_id)
        if smooth_alpha > 0.0:
            data = smooth_streams(data, smooth_alpha)
        sections.append(data)
        gene.append(data.gene)
        image.append(data.image)
        offsets[section_id] = (cursor, cursor + data.gene.shape[0])
        cursor += data.gene.shape[0]

    gene = np.vstack(gene)
    image = np.vstack(image)
    ids = np.concatenate([[s.section_id] * s.gene.shape[0] for s in sections])
    if batch_correct:
        before = image.copy()
        image = project_out_section_directions(image, ids)
        print(f"batch correction: image drift {np.abs(before).mean():.4f} -> "
              f"{np.abs(image).mean():.4f}", flush=True)
    return sections, gene, image, offsets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, default=Path("checkpoints/dlpfc.pkl"))
    parser.add_argument("--name", default="pooled_all12")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--smooth-alpha", type=float, default=0.5)
    parser.add_argument("--batch-correct", action="store_true")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--attention-device", default="cpu")
    args = parser.parse_args()

    config = SparcConfig(**BEST, epochs=args.epochs)
    cache_sections(args.checkpoint_path, config)
    sections, gene, image, offsets = build_pooled(args.smooth_alpha, args.batch_correct)
    print(f"pooled: {gene.shape[0]} spots, {config.n_latents} atoms "
          f"({gene.shape[0] / config.n_latents:.1f} spots/atom vs "
          f"{sections[0].gene.shape[0] / config.n_latents:.1f} per-section)", flush=True)

    # One dictionary over every spot. Phase 2 stays per-section afterwards.
    from dataclasses import replace as dc_replace

    pooled_data = dc_replace(
        sections[0], gene=gene, image=image,
        neighbor_indices=np.vstack([
            s.neighbor_indices + offsets[s.section_id][0] for s in sections]),
        neighbor_image_similarity=np.vstack([s.neighbor_image_similarity for s in sections]),
        coordinates=np.vstack([s.coordinates for s in sections]),
        labels=np.concatenate([s.labels for s in sections]),
        barcodes=np.concatenate([s.barcodes for s in sections]),
        gene_raw=np.vstack([s.gene_raw for s in sections]),
        image_pca=np.vstack([s.image_pca for s in sections]),
        pseudo_labels=np.concatenate([s.pseudo_labels for s in sections]),
    )
    started = time.perf_counter()
    result = fit_sparc(pooled_data, config, device=args.device, verbose=True)
    print(f"pooled phase 1 trained in {time.perf_counter() - started:.0f}s", flush=True)
    print(f"POOLED gene self-NMSE {result.metrics['self_nmse_gene']:.4f} "
          f"(per-section reference ~0.88)", flush=True)

    writer = IncrementalCsv(results_dir(args.name) / "pooled_metrics.csv", resume=True)
    from .attention_stage import fit_attention_stage

    for data in sections:
        lo, hi = offsets[data.section_id]
        latents = {k: v[lo:hi] for k, v in result.latents.items()}
        row = {"section_id": data.section_id, "pooled_self_nmse_gene": result.metrics["self_nmse_gene"]}
        for mode in ("sum", "support"):
            m = cluster_sparc_latents(latents, data.labels, data.coordinates,
                                      replace(config, cluster_input=mode))["metrics"]
            row[f"stage1_{mode}_ari"] = m["refined_ari"]
        best = "support" if row["stage1_support_ari"] >= row["stage1_sum_ari"] else "sum"
        matrix = build_cluster_input(latents, best).astype(np.float32)
        s2 = fit_attention_stage(data, matrix, config, device=args.attention_device, verbose=False)
        row.update(ari=s2.metrics["refined_ari"], nmi=s2.metrics["refined_nmi"], stage2_input=best)
        writer.append(row)
        print(f"  {data.section_id} ARI={row['ari']:.4f}", flush=True)
        del s2, matrix
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    aris = [float(r["ari"]) for r in writer.rows]
    save_json(args.name, {
        "sections": SECTIONS, "epochs": args.epochs, "smooth_alpha": args.smooth_alpha,
        "batch_correct": args.batch_correct, "base_config": BEST,
        "pooled_self_nmse_gene": result.metrics["self_nmse_gene"],
        "mean_ari": float(np.mean(aris)), "median_ari": float(np.median(aris)),
    })
    print(f"\nmean ARI {np.mean(aris):.4f} | median {np.median(aris):.4f}", flush=True)


if __name__ == "__main__":
    main()
