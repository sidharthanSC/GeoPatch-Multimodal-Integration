"""Single-section parameter sweeps for SPARC, persisted to ``outputs/sparc_align/``.

Replaces the ad-hoc scratchpad scripts used during exploration: every sweep is an
importable function returning rows, and every CLI invocation writes a CSV plus a
JSON summary under ``outputs/sparc_align/<name>/`` instead of only printing.

    python -m src.sparc_align.sweeps --sweep capacity --section 151507
    python -m src.sparc_align.sweeps --sweep all --section 151507

Single-section results are exploratory. Section 151507 in particular runs about
+0.06 ARI above the twelve-section median, so tune on it only with that in mind.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np

from src.datasets.dlpfc import DlpfcDataset

from .config import SparcConfig
from .data import prepare_section
from .evaluate import build_cluster_input, cluster_sparc_latents
from .results_io import IncrementalCsv, results_dir, save_json
from .train import fit_sparc

BASE = dict(n_latents=1024, k_active=128, epochs=50)


def _evaluate(config: SparcConfig, adata, section: str, device: str) -> dict:
    """Train one SPARC configuration and score it under both cluster inputs."""
    data = prepare_section(adata, section, config)
    result = fit_sparc(data, config, device=device, verbose=False)
    metrics = result.metrics
    row = {
        "section_id": section,
        "sum_ari": metrics["refined_ari"],
        "sum_nmi": metrics["refined_nmi"],
        "self_nmse_gene": metrics["self_nmse_gene"],
        "self_nmse_image": metrics["self_nmse_image"],
        "cross_nmse_image_to_gene": metrics["cross_nmse_image_to_gene"],
        "cross_nmse_gene_to_image": metrics["cross_nmse_gene_to_image"],
        "support_jaccard": metrics.get("support_jaccard_gene_image"),
        "all_alive_fraction": metrics["all_alive_fraction"],
    }
    support = cluster_sparc_latents(
        result.latents, data.labels, data.coordinates, replace(config, cluster_input="support")
    )["metrics"]
    row["support_ari"] = support["refined_ari"]
    row["support_nmi"] = support["refined_nmi"]
    return row, result, data


def sweep_capacity(adata, section: str, device: str, writer: IncrementalCsv) -> list[dict]:
    """Dictionary size and sparsity level."""
    rows = []
    for n_latents, k in [(1024, 32), (1024, 128), (2048, 256), (4096, 512)]:
        config = SparcConfig(n_latents=n_latents, k_active=k, epochs=50)
        row, _, _ = _evaluate(config, adata, section, device)
        rows.append({"sweep": "capacity", "n_latents": n_latents, "k_active": k, **row})
        writer.append(rows[-1])
    return rows


def sweep_spatial_weight(adata, section: str, device: str, writer: IncrementalCsv) -> list[dict]:
    """Blend weight for the spatially-aggregated TopK."""
    rows = []
    for weight in [0.0, 0.25, 0.5, 0.75, 0.9]:
        config = SparcConfig(**BASE, spatial_topk_weight=weight)
        row, _, _ = _evaluate(config, adata, section, device)
        rows.append({"sweep": "spatial_weight", "spatial_topk_weight": weight, **row})
        writer.append(rows[-1])
    return rows


def sweep_topk_mode(adata, section: str, device: str, writer: IncrementalCsv) -> list[dict]:
    """Support-selection mechanism, at the best known spatial weight."""
    rows = []
    for mode in ["global_sum", "quota", "rank_fusion", "partitioned"]:
        config = SparcConfig(**BASE, topk_mode=mode, spatial_topk_weight=0.75)
        row, _, _ = _evaluate(config, adata, section, device)
        rows.append({"sweep": "topk_mode", "topk_mode": mode, **row})
        writer.append(rows[-1])
    return rows


def sweep_spatial_attention(adata, section: str, device: str, writer: IncrementalCsv) -> list[dict]:
    """Uniform neighbour mean vs morphology-weighted aggregation."""
    rows = []
    for attention in [False, True]:
        config = SparcConfig(**BASE, spatial_topk_weight=0.75, spatial_attention=attention)
        row, _, _ = _evaluate(config, adata, section, device)
        rows.append({"sweep": "spatial_attention", "spatial_attention": attention, **row})
        writer.append(rows[-1])
    return rows


def sweep_stage2_epochs(adata, section: str, device: str, writer: IncrementalCsv) -> list[dict]:
    """Cross-attention stage-2 training length, on fixed stage-1 latents."""
    from .attention_stage import fit_attention_stage

    config = SparcConfig(**BASE, spatial_topk_weight=0.75)
    _, result, data = _evaluate(config, adata, section, device)
    matrix = build_cluster_input(result.latents, "sum").astype(np.float32)
    rows = []
    for epochs in [20, 50, 100, 200]:
        stage2 = fit_attention_stage(
            data, matrix, replace(config, attention_epochs=epochs), device="cpu", verbose=False
        )
        rows.append(
            {
                "sweep": "stage2_epochs",
                "section_id": section,
                "attention_epochs": epochs,
                "attn_ari": stage2.metrics["refined_ari"],
                "attn_nmi": stage2.metrics["refined_nmi"],
            }
        )
        writer.append(rows[-1])
    return rows


SWEEPS: dict[str, Callable] = {
    "capacity": sweep_capacity,
    "spatial_weight": sweep_spatial_weight,
    "topk_mode": sweep_topk_mode,
    "spatial_attention": sweep_spatial_attention,
    "stage2_epochs": sweep_stage2_epochs,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, default=Path("checkpoints/dlpfc.pkl"))
    parser.add_argument("--sweep", default="all", choices=[*SWEEPS, "all"])
    parser.add_argument("--section", default="151507")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--name", help="output subdirectory; default sweeps_<section>")
    args = parser.parse_args()

    name = args.name or f"sweeps_{args.section}"
    directory = results_dir(name)
    dataset = DlpfcDataset.from_checkpoint(args.checkpoint_path)
    adata = dataset.get_section(args.section)

    selected = list(SWEEPS) if args.sweep == "all" else [args.sweep]
    all_rows: list[dict] = []
    for sweep_name in selected:
        print(f"\n=== sweep {sweep_name} ===", flush=True)
        writer = IncrementalCsv(directory / f"{sweep_name}.csv")
        rows = SWEEPS[sweep_name](adata, args.section, args.device, writer)
        all_rows.extend(rows)
        for row in rows:
            print(json.dumps(row, sort_keys=True, default=float), flush=True)

    save_json(
        name,
        {
            "section": args.section,
            "sweeps": selected,
            "base_config": BASE,
            "n_rows": len(all_rows),
            "caveat": "single-section; 151507 runs ~+0.06 ARI above the 12-section median",
        },
    )
    print(f"\nwrote {len(all_rows)} rows to {directory}", flush=True)


if __name__ == "__main__":
    main()
