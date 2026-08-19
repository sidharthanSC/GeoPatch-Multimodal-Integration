"""SPARC ablation ladder across all twelve DLPFC sections.

Each arm changes exactly one thing relative to the paper-faithful baseline, so the
contribution of every extension is attributable. The final arm is the best
combination validated on section 151507.

    python -m src.sparc_align.ablation --output-dir outputs/sparc_align/<run>

Every arm is evaluated through the identical protocol used by ``src.mp_mnca`` and
the regenerated STAIG baseline: PCA -> tied GMM -> 15-NN spatial refinement ->
ARI/NMI, reported per section plus mean/median/IQR.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.datasets.dlpfc import DlpfcDataset

from .config import SparcConfig
from .data import prepare_section
from .evaluate import build_cluster_input, cluster_sparc_latents
from .results_io import IncrementalCsv
from .train import fit_sparc

# Baseline shared by every arm. k=128 rather than the paper's 64: a DLPFC section
# has ~4k spots against Open Images' 1.7M, and k=128/L=1024 was the best-behaved
# capacity in the 151507 sweep (no dead latents, lowest gene self-NMSE).
BASE = dict(n_latents=1024, k_active=128, epochs=50)

ARMS: dict[str, dict] = {
    # Eq. 1-6 exactly as published: affine encoders, raw summed logits, no spatial term.
    "a_paper_faithful": {},
    # Extension 2: equalize per-stream logit scale before the Global TopK sum.
    "b_norm_logits": {"normalize_stream_logits": True},
    # Extension 1: the paper's own suggested nonlinear-encoder future work.
    "c_mlp_encoder": {"encoder_hidden_dim": 512},
    # Extension 3: spatially-aggregated TopK. k=6 neighbours are 92.3% same-layer.
    "d_spatial_topk": {"spatial_topk_weight": 0.75},
    # Best validated combination; stage 2 is applied on top of arm d.
    "e_spatial_plus_attention": {"spatial_topk_weight": 0.75, "use_cross_attention_stage": True},
    # --- Follow-up: gene-encoder capacity, not the selection rule ------------
    # The v1 ablation plus the topk_mode sweep showed that every intervention
    # making support selection more *balanced* between streams made clustering
    # worse (global_sum 0.354 > quota 0.243 > rank_fusion 0.217 on 151507, and
    # normalize_stream_logits cost 0.045 median ARI across 12 sections). Gene is
    # not being crowded out of the support; with self-NMSE 0.88 vs image's 0.23 it
    # cannot produce usable logits at all. The bottleneck is gene encoder capacity.
    # These arms test that directly, all on top of the validated spatial TopK.
    "f_partitioned": {"topk_mode": "partitioned", "spatial_topk_weight": 0.75},
    "g_mlp1024": {"encoder_hidden_dim": 1024, "spatial_topk_weight": 0.75},
    "h_mlp2048": {"encoder_hidden_dim": 2048, "spatial_topk_weight": 0.75},
    "i_partitioned_mlp1024": {
        "topk_mode": "partitioned",
        "encoder_hidden_dim": 1024,
        "spatial_topk_weight": 0.75,
    },
    "j_best_plus_attention": {
        "topk_mode": "partitioned",
        "encoder_hidden_dim": 1024,
        "spatial_topk_weight": 0.75,
        "use_cross_attention_stage": True,
    },
    # --- v3: attention-style weighting inside the stage-1 spatial aggregation --
    # The uniform neighbour mean was the single largest validated gain (+0.072
    # median ARI), but it treats all six neighbours as equally informative and so
    # blurs supports across layer boundaries. These arms weight the neighbourhood
    # by MP-MNCA's morphology kernel softmax(beta * log s_ij) instead.
    "k_spatial_attention": {"spatial_topk_weight": 0.75, "spatial_attention": True},
    "l_spatial_attention_mlp1024": {
        "spatial_topk_weight": 0.75,
        "spatial_attention": True,
        "encoder_hidden_dim": 1024,
    },
    "m_spatial_attention_partitioned": {
        "spatial_topk_weight": 0.75,
        "spatial_attention": True,
        "topk_mode": "partitioned",
    },
    "n_spatial_attention_plus_stage2": {
        "spatial_topk_weight": 0.75,
        "spatial_attention": True,
        "use_cross_attention_stage": True,
    },
}


def run_arm(
    arm_name: str,
    overrides: dict,
    dataset: DlpfcDataset,
    sections: list[str],
    device: str,
    attention_device: str,
    writer: IncrementalCsv | None = None,
) -> list[dict]:
    """Run one ablation arm across all sections, returning one row per section."""
    config = SparcConfig(**BASE, **{k: v for k, v in overrides.items() if k != "use_cross_attention_stage"})
    use_attention = overrides.get("use_cross_attention_stage", False)
    rows: list[dict] = []

    for section_id in sections:
        started = time.perf_counter()
        data = prepare_section(dataset.get_section(section_id), section_id, config)
        result = fit_sparc(data, config, device=device, verbose=False)

        row = {
            "arm": arm_name,
            "section_id": section_id,
            "sum_ari": result.metrics["refined_ari"],
            "sum_nmi": result.metrics["refined_nmi"],
            "self_nmse_gene": result.metrics["self_nmse_gene"],
            "self_nmse_image": result.metrics["self_nmse_image"],
            "cross_nmse_image_to_gene": result.metrics["cross_nmse_image_to_gene"],
            "cross_nmse_gene_to_image": result.metrics["cross_nmse_gene_to_image"],
            "all_alive_fraction": result.metrics["all_alive_fraction"],
        }
        support = cluster_sparc_latents(
            result.latents, data.labels, data.coordinates, replace(config, cluster_input="support")
        )["metrics"]
        row["support_ari"] = support["refined_ari"]
        row["support_nmi"] = support["refined_nmi"]

        if use_attention:
            from .attention_stage import fit_attention_stage

            best = max(("sum", row["sum_ari"]), ("support", row["support_ari"]), key=lambda p: p[1])[0]
            matrix = build_cluster_input(result.latents, best).astype(np.float32)
            stage2 = fit_attention_stage(data, matrix, config, device=attention_device, verbose=False)
            row["attn_input"] = best
            row["attn_ari"] = stage2.metrics["refined_ari"]
            row["attn_nmi"] = stage2.metrics["refined_nmi"]

        row["elapsed_seconds"] = time.perf_counter() - started
        rows.append(row)
        # Flush after every section so an interrupted run still leaves its
        # completed rows on disk instead of an empty directory.
        if writer is not None:
            writer.append(row)
        print(json.dumps(row, sort_keys=True, default=float), flush=True)
    return rows


def summarize(rows: list[dict], keys: list[str]) -> dict:
    """Mean/median/IQR over sections for each present metric."""
    summary = {}
    for key in keys:
        values = [float(r[key]) for r in rows if key in r and r[key] is not None]
        if not values:
            continue
        summary[key] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, default=Path("checkpoints/dlpfc.pkl"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sections", nargs="*")
    parser.add_argument("--arms", nargs="*", help="subset of arm names; default all")
    parser.add_argument("--device", default="mps", help="device for the SPARC stage")
    parser.add_argument("--attention-device", default="cpu", help="device for stage 2")
    args = parser.parse_args()

    # An empty directory from a previously crashed run is not a real conflict --
    # only refuse when it actually holds results.
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing run: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = DlpfcDataset.from_checkpoint(args.checkpoint_path)
    sections = args.sections or dataset.section_ids()
    arm_names = args.arms or list(ARMS)

    metrics_path = args.output_dir / "ablation_metrics.csv"
    writer = IncrementalCsv(metrics_path)
    summary_keys = [
        "sum_ari", "sum_nmi", "support_ari", "support_nmi", "attn_ari", "attn_nmi",
        "self_nmse_gene", "cross_nmse_image_to_gene",
    ]

    def write_summary(summaries: dict[str, dict], complete: bool) -> None:
        (args.output_dir / "summary.json").write_text(
            json.dumps(
                {
                    "run_id": args.output_dir.name,
                    "complete": complete,
                    "base_config": BASE,
                    "arms": {name: ARMS[name] for name in arm_names},
                    "sections": sections,
                    "baseline_reference": "outputs/prior_models/staig_baseline_seed0_all12_regen_v1",
                    "summaries": summaries,
                },
                indent=2,
                default=float,
            ),
            encoding="utf-8",
        )

    all_rows: list[dict] = []
    summaries: dict[str, dict] = {}
    write_summary(summaries, complete=False)
    for arm_name in arm_names:
        print(f"\n=== arm {arm_name} ===", flush=True)
        rows = run_arm(
            arm_name, ARMS[arm_name], dataset, sections, args.device,
            args.attention_device, writer,
        )
        all_rows.extend(rows)
        summaries[arm_name] = summarize(rows, summary_keys)
        # Refresh the summary after each arm so partial runs stay interpretable.
        write_summary(summaries, complete=False)

    write_summary(summaries, complete=True)
    print("\n" + json.dumps(summaries, indent=2, default=float), flush=True)


if __name__ == "__main__":
    main()
