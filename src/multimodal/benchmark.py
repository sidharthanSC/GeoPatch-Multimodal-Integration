"""Common per-section clustering benchmark over an audited embedding bundle."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from src.multimodal.evaluation import clustering_metrics
from src.multimodal.representations import existing_embedding_representations


@dataclass(frozen=True)
class KMeansBenchmarkConfig:
    manifest_path: Path
    embeddings_npz: Path
    output_dir: Path
    gene_npz: Path | None = None
    cross_modal_npz: Path | None = None
    backend: str = "kmeans"
    seeds: tuple[int, ...] = tuple(range(10))
    pca_dim: int = 128
    gmm_pca_dim: int = 30


def _refuse_existing(paths: tuple[Path, ...]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite: " + ", ".join(existing))


def run_per_section_kmeans(config: KMeansBenchmarkConfig) -> dict:
    metrics_path = config.output_dir / "metrics.parquet"
    assignments_path = config.output_dir / "assignments.parquet"
    section_summary_path = config.output_dir / "section_summary.parquet"
    aggregate_path = config.output_dir / "aggregate_summary.json"
    config_path = config.output_dir / "config.json"
    _refuse_existing(
        (metrics_path, assignments_path, section_summary_path, aggregate_path, config_path)
    )

    manifest = pd.read_parquet(config.manifest_path)
    if config.backend not in {"kmeans", "gmm"}:
        raise ValueError("backend must be 'kmeans' or 'gmm'")
    with np.load(config.embeddings_npz) as data:
        source = {key: data[key] for key in data.files}
    if config.gene_npz is not None:
        with np.load(config.gene_npz, allow_pickle=True) as gene_data:
            source["gene_emb"] = gene_data["embeddings"]
    if config.cross_modal_npz is not None:
        with np.load(config.cross_modal_npz, allow_pickle=True) as aligned:
            source["gene_emb_cm_img"] = aligned["gene_projected"]
            source["img_emb_cm"] = aligned["image_projected"]
    representations = existing_embedding_representations(source)

    metric_records = []
    assignment_records = []
    for section_id in manifest["section_id"].drop_duplicates():
        section_mask = (
            (manifest["section_id"] == section_id)
            & manifest["ground_truth_valid"]
        ).to_numpy()
        labels = manifest.loc[section_mask, "ground_truth"].to_numpy()
        barcodes = manifest.loc[section_mask, "barcode"].to_numpy()
        n_clusters = int(np.unique(labels).size)

        section_representations = {
            name: values[section_mask] for name, values in representations.items()
        }
        for source_name, output_name in (
            ("R5_unaligned_concat", "R6_unaligned_concat_pca128"),
            ("R10_aligned_concat", "R11_aligned_concat_pca128"),
        ):
            values = section_representations[source_name]
            n_components = min(config.pca_dim, values.shape[0], values.shape[1])
            section_representations[output_name] = PCA(
                n_components=n_components,
                svd_solver="randomized",
                random_state=0,
            ).fit_transform(values).astype(np.float32)

        for representation_name, values in section_representations.items():
            clustering_values = values
            if config.backend == "gmm":
                n_components = min(
                    config.gmm_pca_dim, values.shape[0], values.shape[1]
                )
                clustering_values = PCA(
                    n_components=n_components,
                    svd_solver="randomized",
                    random_state=0,
                ).fit_transform(values)
            for seed in config.seeds:
                start = time.perf_counter()
                if config.backend == "kmeans":
                    cluster_ids = KMeans(
                        n_clusters=n_clusters,
                        n_init=1,
                        random_state=seed,
                    ).fit_predict(clustering_values)
                else:
                    cluster_ids = GaussianMixture(
                        n_components=n_clusters,
                        covariance_type="full",
                        reg_covar=1e-5,
                        n_init=1,
                        random_state=seed,
                    ).fit_predict(clustering_values)
                elapsed = time.perf_counter() - start
                scores = clustering_metrics(labels, cluster_ids)
                metric_records.append(
                    {
                        "section_id": str(section_id),
                        "representation": representation_name,
                        "backend": config.backend,
                        "refinement": "none",
                        "seed": seed,
                        "n_clusters": n_clusters,
                        "n_spots": len(labels),
                        "runtime_seconds": elapsed,
                        **scores,
                    }
                )
                assignment_records.extend(
                    {
                        "section_id": str(section_id),
                        "barcode": barcode,
                        "representation": representation_name,
                        "backend": config.backend,
                        "refinement": "none",
                        "seed": seed,
                        "true_label": true_label,
                        "cluster_id": int(cluster_id),
                    }
                    for barcode, true_label, cluster_id in zip(
                        barcodes, labels, cluster_ids
                    )
                )

    metrics = pd.DataFrame(metric_records)
    assignments = pd.DataFrame(assignment_records)
    section_summary = (
        metrics.groupby(["section_id", "representation"], as_index=False)
        .agg(
            ari=("ari", "mean"),
            ari_std=("ari", "std"),
            nmi=("nmi", "mean"),
            nmi_std=("nmi", "std"),
            accuracy=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
        )
    )

    aggregate = {}
    for representation, group in section_summary.groupby("representation"):
        aggregate[representation] = {
            "median_ari": float(group["ari"].median()),
            "iqr_ari": float(group["ari"].quantile(0.75) - group["ari"].quantile(0.25)),
            "median_nmi": float(group["nmi"].median()),
            "iqr_nmi": float(group["nmi"].quantile(0.75) - group["nmi"].quantile(0.25)),
            "median_accuracy": float(group["accuracy"].median()),
            "n_sections": int(len(group)),
        }

    config.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_parquet(metrics_path, index=False)
    assignments.to_parquet(assignments_path, index=False)
    section_summary.to_parquet(section_summary_path, index=False)
    with aggregate_path.open("w") as file:
        json.dump(aggregate, file, indent=2)
    with config_path.open("w") as file:
        json.dump(
            {
                **asdict(config),
                "manifest_path": str(config.manifest_path),
                "embeddings_npz": str(config.embeddings_npz),
                "output_dir": str(config.output_dir),
                "gene_npz": str(config.gene_npz) if config.gene_npz else None,
                "cross_modal_npz": (
                    str(config.cross_modal_npz) if config.cross_modal_npz else None
                ),
                "seeds": list(config.seeds),
            },
            file,
            indent=2,
        )
    return aggregate


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--embeddings-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--gene-npz",
        type=Path,
        default=None,
        help="Optional gene-BYOL prediction NPZ whose embeddings replace audited gene_emb.",
    )
    parser.add_argument(
        "--cross-modal-npz",
        type=Path,
        default=None,
        help="Optional cross-modal prediction NPZ whose projected arrays replace the audited aligned pair.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--pca-dim", type=int, default=128)
    parser.add_argument("--backend", choices=["kmeans", "gmm"], default="kmeans")
    parser.add_argument("--gmm-pca-dim", type=int, default=30)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = run_per_section_kmeans(
        KMeansBenchmarkConfig(
            manifest_path=args.manifest_path,
            embeddings_npz=args.embeddings_npz,
            output_dir=args.output_dir,
            gene_npz=args.gene_npz,
            cross_modal_npz=args.cross_modal_npz,
            backend=args.backend,
            seeds=tuple(args.seeds),
            pca_dim=args.pca_dim,
            gmm_pca_dim=args.gmm_pca_dim,
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
