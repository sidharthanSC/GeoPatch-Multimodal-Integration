"""Matched clustering and frozen-probe evaluation for multi-scale STAIG studies."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.multimodal.evaluation import clustering_metrics
from src.prior_models.staig.evaluate import refine_labels, tied_gmm


@dataclass(frozen=True)
class MultiscaleEvaluationConfig:
    manifest_path: Path
    gene_npz: Path
    staig_runs: dict[str, Path]
    output_dir: Path
    stage_keys: tuple[str, ...] = (
        "gene_stage_1024",
        "gene_stage_512",
        "gene_stage_256",
        "gene_embedding_128",
    )
    seed: int = 0
    test_fraction: float = 0.3
    probe_pca_dim: int = 128
    refinement_neighbors: int = 15


def _load_gene_stages(
    path: Path, manifest: pd.DataFrame, stage_keys: tuple[str, ...]
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        missing = [key for key in (*stage_keys, "section_ids", "barcodes") if key not in data]
        if missing:
            raise KeyError(f"Gene NPZ is missing arrays: {missing}")
        observed = list(zip(data["section_ids"].astype(str), data["barcodes"].astype(str)))
        expected = list(
            zip(manifest["section_id"].astype(str), manifest["barcode"].astype(str))
        )
        if observed != expected:
            raise ValueError("Gene NPZ row identities do not match the audit manifest")
        stages = {key: np.asarray(data[key], dtype=np.float32) for key in stage_keys}
    for key, values in stages.items():
        if values.ndim != 2 or values.shape[0] != len(manifest):
            raise ValueError(f"Invalid {key} shape: {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"{key} contains non-finite values")
    return stages


def _probe(
    values: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
    pca_dim: int | None,
) -> dict[str, float]:
    scaler = StandardScaler().fit(values[train_idx])
    train_values = scaler.transform(values[train_idx])
    test_values = scaler.transform(values[test_idx])
    if pca_dim is not None and train_values.shape[1] > pca_dim:
        pca = PCA(
            n_components=min(pca_dim, len(train_idx), train_values.shape[1]),
            svd_solver="randomized",
            random_state=seed,
        ).fit(train_values)
        train_values = pca.transform(train_values)
        test_values = pca.transform(test_values)
    predicted = LogisticRegression(
        C=1.0,
        max_iter=1000,
        tol=1e-4,
        solver="lbfgs",
        random_state=seed,
    ).fit(train_values, labels[train_idx]).predict(test_values)
    return {
        "accuracy": float(accuracy_score(labels[test_idx], predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels[test_idx], predicted)),
        "macro_f1": float(f1_score(labels[test_idx], predicted, average="macro")),
    }


def _summarize(
    frame: pd.DataFrame, group_keys: tuple[str, ...], metrics: tuple[str, ...]
) -> dict:
    grouped = frame.groupby(group_keys[0] if len(group_keys) == 1 else list(group_keys))
    return {
        "|".join(str(part) for part in (name if isinstance(name, tuple) else (name,))): {
            metric: {
                "mean": float(group[metric].mean()),
                "median": float(group[metric].median()),
                "iqr": float(group[metric].quantile(0.75) - group[metric].quantile(0.25)),
            }
            for metric in metrics
        }
        for name, group in grouped
    }


def evaluate_multiscale(config: MultiscaleEvaluationConfig) -> dict[str, object]:
    """Evaluate raw gene stages and independent STAIG runs on matched section splits."""
    paths = (
        config.output_dir / "direct_clustering.parquet",
        config.output_dir / "probe_metrics.parquet",
        config.output_dir / "summary.json",
        config.output_dir / "config.json",
    )
    if any(path.exists() for path in paths):
        raise FileExistsError(f"Refusing to overwrite artifacts under {config.output_dir}")

    manifest = pd.read_parquet(config.manifest_path)
    stages = _load_gene_stages(config.gene_npz, manifest, config.stage_keys)
    clustering_records: list[dict[str, object]] = []
    probe_records: list[dict[str, object]] = []

    for section_id in manifest["section_id"].drop_duplicates().astype(str):
        section_mask = (
            (manifest["section_id"].astype(str) == section_id)
            & manifest["ground_truth_valid"]
        ).to_numpy()
        section = manifest.loc[section_mask]
        labels = section["ground_truth"].astype(str).to_numpy()
        coordinates = section[["spatial_x", "spatial_y"]].to_numpy(dtype=np.float32)
        n_clusters = int(np.unique(labels).size)
        indices = np.arange(len(labels))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=config.test_fraction,
            random_state=config.seed,
            stratify=labels,
        )

        for stage_key, all_values in stages.items():
            values = all_values[section_mask]
            predicted = tied_gmm(values, n_clusters=n_clusters)
            refined = refine_labels(
                predicted, coordinates, n_neighbors=config.refinement_neighbors
            )
            for refinement, cluster_ids in (("none", predicted), ("knn15", refined)):
                clustering_records.append(
                    {
                        "section_id": section_id,
                        "representation": stage_key,
                        "refinement": refinement,
                        "seed": 2023,
                        "n_clusters": n_clusters,
                        "n_spots": len(labels),
                        **clustering_metrics(labels, cluster_ids),
                    }
                )
            probe_records.append(
                {
                    "section_id": section_id,
                    "representation": stage_key,
                    "family": "gene_stage_pca128",
                    "seed": config.seed,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    **_probe(
                        values,
                        labels,
                        train_idx,
                        test_idx,
                        config.seed,
                        config.probe_pca_dim,
                    ),
                }
            )

        expected_barcodes = section["barcode"].astype(str).to_numpy()
        for run_name, run_dir in config.staig_runs.items():
            embedding_path = run_dir / "embeddings" / f"{section_id}.npz"
            with np.load(embedding_path, allow_pickle=True) as data:
                observed_barcodes = data["barcodes"].astype(str)
                embeddings = np.asarray(data["embeddings"], dtype=np.float32)
            # STAIG exports all section spots; select valid labeled rows by barcode.
            positions = {barcode: index for index, barcode in enumerate(observed_barcodes)}
            if len(positions) != len(observed_barcodes):
                raise ValueError(f"Duplicate barcodes in {embedding_path}")
            missing = [barcode for barcode in expected_barcodes if barcode not in positions]
            if missing:
                raise ValueError(f"Missing barcodes in {embedding_path}: {missing[:5]}")
            values = embeddings[[positions[barcode] for barcode in expected_barcodes]]
            probe_records.append(
                {
                    "section_id": section_id,
                    "representation": run_name,
                    "family": "staig_embedding",
                    "seed": config.seed,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    **_probe(values, labels, train_idx, test_idx, config.seed, None),
                }
            )

    clustering = pd.DataFrame(clustering_records)
    probes = pd.DataFrame(probe_records)
    summary = {
        "direct_clustering": _summarize(
            clustering,
            ("representation", "refinement"),
            ("ari", "nmi", "accuracy"),
        ),
        "probes": _summarize(
            probes,
            ("representation",),
            ("accuracy", "balanced_accuracy", "macro_f1"),
        ),
    }
    config.output_dir.mkdir(parents=True, exist_ok=True)
    clustering.to_parquet(paths[0], index=False)
    probes.to_parquet(paths[1], index=False)
    paths[2].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    paths[3].write_text(
        json.dumps(
            {
                "manifest_path": str(config.manifest_path),
                "gene_npz": str(config.gene_npz),
                "staig_runs": {
                    name: str(path) for name, path in config.staig_runs.items()
                },
                "stage_keys": list(config.stage_keys),
                "seed": config.seed,
                "test_fraction": config.test_fraction,
                "probe_pca_dim": config.probe_pca_dim,
                "refinement_neighbors": config.refinement_neighbors,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--gene-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--staig-run", nargs=2, action="append", metavar=("NAME", "PATH"))
    parser.add_argument("--stage-keys", nargs="+", default=list(MultiscaleEvaluationConfig.stage_keys))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-fraction", type=float, default=0.3)
    parser.add_argument("--probe-pca-dim", type=int, default=128)
    parser.add_argument("--refinement-neighbors", type=int, default=15)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = evaluate_multiscale(
        MultiscaleEvaluationConfig(
            manifest_path=args.manifest_path,
            gene_npz=args.gene_npz,
            staig_runs={name: Path(path) for name, path in (args.staig_run or [])},
            output_dir=args.output_dir,
            stage_keys=tuple(args.stage_keys),
            seed=args.seed,
            test_fraction=args.test_fraction,
            probe_pca_dim=args.probe_pca_dim,
            refinement_neighbors=args.refinement_neighbors,
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
