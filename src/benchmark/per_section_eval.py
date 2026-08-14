"""Table A (per-section domain discovery) + its spatially-refined counterpart, from
``MODEL_BENCHMARK_REPORT.md``.

For each of the 12 DLPFC sections independently: build representations R1-R11
(``src/benchmark/representations.py``), cluster each with KMeans / GaussianMixture /
Leiden (``src/benchmark/clustering_backends.py``) across 10 seeds each, score against
``ground_truth`` (ARI/NMI primary, Hungarian accuracy secondary), then apply one common
spatial-neighbor refinement pass (``src/benchmark/refinement.py``) and score again. Raw
and refined metrics are computed in the same pass (refinement is cheap re-labeling, not
a separate clustering run) and written side by side.

Usage
-----
    python -u -m src.benchmark.per_section_eval
    python -u -m src.benchmark.per_section_eval --sections 151507 151508 --n-seeds 2  # smoke test
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from src.benchmark.clustering_backends import gmm_cluster, kmeans_cluster, leiden_cluster_multiseed
from src.benchmark.data import SectionData, load_section_data
from src.benchmark.refinement import refine_labels
from src.benchmark.representations import REPRESENTATION_IDS, build_section_representations
from src.multimodal.evaluation import clustering_metrics

BACKENDS = ("kmeans", "gmm", "leiden")


def evaluate_section(
    section: SectionData,
    n_clusters: int,
    seeds: Sequence[int],
    representation_ids: Sequence[str],
    backends: Sequence[str],
) -> List[dict]:
    representations = build_section_representations(section, pca_seed=seeds[0])
    rows: List[dict] = []

    for rep_id in representation_ids:
        x = representations[rep_id]
        for backend in backends:
            if backend == "leiden":
                result = leiden_cluster_multiseed(x, n_clusters, tuple(seeds))
                labels_by_seed = result["labels_by_seed"]
            elif backend == "kmeans":
                labels_by_seed = {seed: kmeans_cluster(x, n_clusters, seed) for seed in seeds}
            elif backend == "gmm":
                labels_by_seed = {seed: gmm_cluster(x, n_clusters, seed) for seed in seeds}
            else:
                raise ValueError(f"Unknown backend: {backend}")

            for seed, cluster_ids in labels_by_seed.items():
                raw_metrics = clustering_metrics(section.labels, cluster_ids)
                refined_ids = refine_labels(section.barcodes, cluster_ids, section.spatial_neighbors)
                refined_metrics = clustering_metrics(section.labels, refined_ids)

                row = {
                    "section_id": section.section_id,
                    "donor_id": section.donor_id,
                    "representation": rep_id,
                    "backend": backend,
                    "seed": seed,
                    "n_spots": len(section.labels),
                }
                row.update({f"raw_{k}": v for k, v in raw_metrics.items()})
                row.update({f"refined_{k}": v for k, v in refined_metrics.items()})
                rows.append(row)
    return rows


def summarize(df: pd.DataFrame, metric_prefix: str) -> pd.DataFrame:
    """Median/IQR across the 12 sections, first averaging over clustering seeds within
    each section (per the report: "Show all 12 section values and median/IQR")."""
    metrics = [f"{metric_prefix}_accuracy", f"{metric_prefix}_ari", f"{metric_prefix}_nmi"]
    per_section = df.groupby(["representation", "backend", "section_id"])[metrics].mean().reset_index()

    def agg(group: pd.DataFrame) -> pd.Series:
        out = {}
        for metric in metrics:
            values = group[metric]
            out[f"{metric}_median"] = values.median()
            out[f"{metric}_iqr_low"] = values.quantile(0.25)
            out[f"{metric}_iqr_high"] = values.quantile(0.75)
        return pd.Series(out)

    return per_section.groupby(["representation", "backend"]).apply(agg, include_groups=False).reset_index()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, default=Path("checkpoints/dlpfc.pkl"))
    parser.add_argument("--hvg-path", type=Path, default=Path("outputs/gene_encoder/shared_hvgs.json"))
    parser.add_argument("--spatial-graph-path", type=Path, default=Path("outputs/gene_encoder/spatial_knn_graph.pkl"))
    parser.add_argument("--n-clusters", type=int, default=7)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--sections", type=str, nargs="*", default=None, help="Restrict to a subset of section ids (smoke test).")
    parser.add_argument("--representations", type=str, nargs="*", default=None, choices=list(REPRESENTATION_IDS))
    parser.add_argument("--backends", type=str, nargs="*", default=list(BACKENDS), choices=list(BACKENDS))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/benchmark/table_a"))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    seeds = list(range(args.n_seeds))
    representation_ids = args.representations or list(REPRESENTATION_IDS)

    print(f"Loading section data from {args.checkpoint_path} ...", flush=True)
    sections = load_section_data(args.checkpoint_path, args.hvg_path, args.spatial_graph_path)
    section_ids = args.sections or list(sections.keys())
    print(f"Evaluating {len(section_ids)} sections x {len(representation_ids)} representations x "
          f"{len(args.backends)} backends x {len(seeds)} seeds", flush=True)

    all_rows: List[dict] = []
    for section_id in section_ids:
        start = time.time()
        rows = evaluate_section(sections[section_id], args.n_clusters, seeds, representation_ids, args.backends)
        all_rows.extend(rows)
        print(f"  section {section_id}: {len(rows)} rows in {time.time() - start:.1f}s", flush=True)

    df = pd.DataFrame(all_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_dir / "per_run_results.csv", index=False)

    summarize(df, "raw").to_csv(args.output_dir / "summary_raw.csv", index=False)
    summarize(df, "refined").to_csv(args.output_dir / "summary_refined.csv", index=False)

    print(f"Wrote {args.output_dir}/per_run_results.csv, summary_raw.csv, summary_refined.csv", flush=True)


if __name__ == "__main__":
    main()
