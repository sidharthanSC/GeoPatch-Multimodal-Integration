"""Namespaced evaluation for rich-gene feature sources and STAIG runs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.multimodal.evaluation import clustering_metrics
from src.prior_models.staig.evaluate import refine_labels, tied_gmm
from src.prior_models.staig.multiscale_evaluation import _probe


STAGE_KEY_MAP = {
    "4096": "gene_stage_4096",
    "2048": "gene_stage_2048",
    "1024": "gene_stage_1024",
    "512": "gene_stage_512",
    "256": "gene_stage_256",
    "128": "gene_embedding_128",
}


@dataclass(frozen=True)
class FeatureSource:
    source_id: str
    npz_path: Path
    stage_keys: dict[str, str] | None = None


@dataclass(frozen=True)
class AlignmentSource:
    source_id: str
    npz_path: Path


@dataclass(frozen=True)
class StaigRunSource:
    run_id: str
    source_id: str
    stage: str
    run_dir: Path


def _load_arrays(path: Path, keys: tuple[str, ...], manifest: pd.DataFrame) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        missing = [key for key in (*keys, "section_ids", "barcodes") if key not in data]
        if missing:
            raise KeyError(f"{path} is missing arrays: {missing}")
        observed = list(zip(data["section_ids"].astype(str), data["barcodes"].astype(str)))
        expected = list(zip(manifest["section_id"].astype(str), manifest["barcode"].astype(str)))
        if observed != expected:
            raise ValueError(f"{path} identities do not match the audit manifest")
        arrays = {key: np.asarray(data[key], dtype=np.float32) for key in keys}
    if any(values.ndim != 2 or values.shape[0] != len(manifest) for values in arrays.values()):
        raise ValueError(f"{path} contains malformed representation arrays")
    if any(not np.isfinite(values).all() for values in arrays.values()):
        raise ValueError(f"{path} contains non-finite representation arrays")
    return arrays


def dimension_controlled_values(
    values: np.ndarray, target_dim: int = 128, seed: int = 0
) -> tuple[np.ndarray, float]:
    """Standardize and reduce wide representations before tied-GMM fitting."""
    standardized = StandardScaler().fit_transform(values)
    if standardized.shape[1] <= target_dim:
        return standardized.astype(np.float64), 1.0
    pca = PCA(
        n_components=min(target_dim, standardized.shape[0], standardized.shape[1]),
        svd_solver="randomized",
        random_state=seed,
    )
    transformed = pca.fit_transform(standardized)
    return transformed.astype(np.float64), float(pca.explained_variance_ratio_.sum())


def _effective_rank(values: np.ndarray) -> float:
    singular = np.linalg.svd(values - values.mean(axis=0), compute_uv=False)
    probabilities = np.square(singular)
    probabilities /= probabilities.sum() + 1e-12
    return float(np.exp(-(probabilities * np.log(probabilities + 1e-12)).sum()))


def _summary(frame: pd.DataFrame, groups: list[str], metrics: tuple[str, ...]) -> dict:
    output = {}
    for name, group in frame.groupby(groups):
        parts = name if isinstance(name, tuple) else (name,)
        output["|".join(map(str, parts))] = {
            metric: {
                "mean": float(group[metric].mean()),
                "median": float(group[metric].median()),
                "iqr": float(group[metric].quantile(0.75) - group[metric].quantile(0.25)),
            }
            for metric in metrics
        }
    return output


def evaluate_rich_study(
    manifest_path: Path,
    feature_sources: tuple[FeatureSource, ...],
    alignment_sources: tuple[AlignmentSource, ...],
    staig_runs: tuple[StaigRunSource, ...],
    output_dir: Path,
    seed: int = 0,
    test_fraction: float = 0.3,
    target_dim: int = 128,
    run_probes: bool = True,
) -> dict:
    paths = {
        "direct": output_dir / "direct_clustering.parquet",
        "probes": output_dir / "probe_metrics.parquet",
        "diagnostics": output_dir / "preprocessing_diagnostics.parquet",
        "staig": output_dir / "staig_section_metrics.parquet",
        "summary": output_dir / "summary.json",
        "config": output_dir / "config.json",
    }
    if any(path.exists() for path in paths.values()):
        raise FileExistsError(f"Refusing to overwrite rich evaluation under {output_dir}")
    manifest = pd.read_parquet(manifest_path)
    feature_arrays = {
        source.source_id: _load_arrays(
            source.npz_path,
            tuple((source.stage_keys or STAGE_KEY_MAP).values()),
            manifest,
        )
        for source in feature_sources
    }
    alignment_arrays = {
        source.source_id: _load_arrays(
            source.npz_path, ("gene_projected", "image_projected"), manifest
        )
        for source in alignment_sources
    }

    direct_records = []
    probe_records = []
    diagnostic_records = []
    staig_records = []
    for section_id in manifest["section_id"].drop_duplicates().astype(str):
        mask = (
            (manifest["section_id"].astype(str) == section_id)
            & manifest["ground_truth_valid"]
        ).to_numpy()
        section = manifest.loc[mask]
        labels = section["ground_truth"].astype(str).to_numpy()
        coordinates = section[["spatial_x", "spatial_y"]].to_numpy(dtype=np.float32)
        n_clusters = int(np.unique(labels).size)
        indices = np.arange(len(labels))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_fraction,
            random_state=seed,
            stratify=labels,
        )

        representations: dict[str, tuple[np.ndarray, str]] = {}
        for source in feature_sources:
            source_id = source.source_id
            arrays = feature_arrays[source_id]
            for stage, key in (source.stage_keys or STAGE_KEY_MAP).items():
                representations[f"{source_id}:gene_{stage}"] = (arrays[key][mask], "raw_stage")
        for source_id, arrays in alignment_arrays.items():
            gene = arrays["gene_projected"][mask]
            image = arrays["image_projected"][mask]
            gene_unit = gene / np.maximum(np.linalg.norm(gene, axis=1, keepdims=True), 1e-12)
            image_unit = image / np.maximum(np.linalg.norm(image, axis=1, keepdims=True), 1e-12)
            representations[f"{source_id}:aligned_gene"] = (gene, "aligned")
            representations[f"{source_id}:aligned_image"] = (image, "aligned")
            representations[f"{source_id}:aligned_bisector"] = (
                (gene_unit + image_unit) / np.maximum(
                    np.linalg.norm(gene_unit + image_unit, axis=1, keepdims=True), 1e-12
                ),
                "aligned_fusion",
            )
            representations[f"{source_id}:aligned_concat"] = (
                np.concatenate([gene, image], axis=1),
                "aligned_fusion",
            )

        for name, (values, family) in representations.items():
            transformed, explained = dimension_controlled_values(values, target_dim, seed)
            predicted = tied_gmm(transformed, n_clusters=n_clusters)
            refined = refine_labels(predicted, coordinates, n_neighbors=15)
            for refinement, cluster_ids in (("none", predicted), ("knn15", refined)):
                direct_records.append(
                    {
                        "section_id": section_id,
                        "representation": name,
                        "family": family,
                        "refinement": refinement,
                        "native_dim": values.shape[1],
                        "evaluated_dim": transformed.shape[1],
                        "seed": 2023,
                        **clustering_metrics(labels, cluster_ids),
                    }
                )
            diagnostic_records.append(
                {
                    "section_id": section_id,
                    "representation": name,
                    "native_dim": values.shape[1],
                    "evaluated_dim": transformed.shape[1],
                    "pca_explained_variance": explained,
                    "native_zero_std_dimensions": int(np.sum(values.std(axis=0) < 1e-8)),
                    "evaluated_effective_rank": _effective_rank(transformed),
                }
            )
            if run_probes:
                probe_records.append(
                    {
                        "section_id": section_id,
                        "representation": name,
                        "family": family,
                        "seed": seed,
                        **_probe(values, labels, train_idx, test_idx, seed, target_dim),
                    }
                )

        expected_barcodes = section["barcode"].astype(str).to_numpy()
        for run in staig_runs:
            metrics = pd.read_csv(run.run_dir / "section_metrics.csv")
            row = metrics.loc[metrics["section_id"].astype(str) == section_id]
            if len(row) != 1:
                raise ValueError(f"Missing unique section {section_id} in {run.run_dir}")
            staig_records.append(
                {
                    "run_id": run.run_id,
                    "source_id": run.source_id,
                    "stage": run.stage,
                    **row.iloc[0].to_dict(),
                }
            )
            with np.load(run.run_dir / "embeddings" / f"{section_id}.npz", allow_pickle=True) as data:
                barcodes = data["barcodes"].astype(str)
                embeddings = np.asarray(data["embeddings"], dtype=np.float32)
            lookup = {barcode: index for index, barcode in enumerate(barcodes)}
            if len(lookup) != len(barcodes) or any(barcode not in lookup for barcode in expected_barcodes):
                raise ValueError(f"STAIG embedding identities fail for {run.run_id}/{section_id}")
            values = embeddings[[lookup[barcode] for barcode in expected_barcodes]]
            if run_probes:
                probe_records.append(
                    {
                        "section_id": section_id,
                        "representation": f"STAIG:{run.run_id}",
                        "family": "staig_embedding",
                        "seed": seed,
                        **_probe(values, labels, train_idx, test_idx, seed, None),
                    }
                )

    direct = pd.DataFrame(direct_records)
    probes = pd.DataFrame(probe_records)
    diagnostics = pd.DataFrame(diagnostic_records)
    staig = pd.DataFrame(staig_records)
    summary = {
        "direct": _summary(direct, ["representation", "refinement"], ("ari", "nmi", "accuracy")),
        "probes": (
            _summary(probes, ["representation"], ("accuracy", "balanced_accuracy", "macro_f1"))
            if not probes.empty
            else {}
        ),
        "staig": (
            _summary(staig, ["run_id"], ("ari", "nmi", "refined_ari", "refined_nmi"))
            if not staig.empty
            else {}
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    direct.to_parquet(paths["direct"], index=False)
    probes.to_parquet(paths["probes"], index=False)
    diagnostics.to_parquet(paths["diagnostics"], index=False)
    staig.to_parquet(paths["staig"], index=False)
    paths["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    paths["config"].write_text(
        json.dumps(
            {
                "manifest_path": str(manifest_path),
                "feature_sources": [
                    {
                        "source_id": item.source_id,
                        "npz_path": str(item.npz_path),
                        "stage_keys": item.stage_keys,
                    }
                    for item in feature_sources
                ],
                "alignment_sources": [
                    {"source_id": item.source_id, "npz_path": str(item.npz_path)}
                    for item in alignment_sources
                ],
                "staig_runs": [
                    {
                        "run_id": item.run_id,
                        "source_id": item.source_id,
                        "stage": item.stage,
                        "run_dir": str(item.run_dir),
                    }
                    for item in staig_runs
                ],
                "seed": seed,
                "test_fraction": test_fraction,
                "target_dim": target_dim,
                "run_probes": run_probes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = json.loads(args.study_manifest.read_text(encoding="utf-8"))
    summary = evaluate_rich_study(
        Path(config["manifest_path"]),
        tuple(
            FeatureSource(
                item["source_id"],
                Path(item["npz_path"]),
                item.get("stage_keys"),
            )
            for item in config["feature_sources"]
        ),
        tuple(AlignmentSource(item["source_id"], Path(item["npz_path"])) for item in config["alignment_sources"]),
        tuple(
            StaigRunSource(item["run_id"], item["source_id"], item["stage"], Path(item["run_dir"]))
            for item in config["staig_runs"]
        ),
        args.output_dir,
        seed=int(config.get("seed", 0)),
        test_fraction=float(config.get("test_fraction", 0.3)),
        target_dim=int(config.get("target_dim", 128)),
        run_probes=bool(config.get("run_probes", True)),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
