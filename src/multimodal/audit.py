"""Validate the DLPFC checkpoint and export a compact evaluation bundle.

The source checkpoint is read-only. This module never attaches embeddings or writes
back to it; all outputs are ordinary evaluation artifacts under a new output directory.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.datasets.dlpfc import DlpfcDataset
from src.multimodal.bisector import bisector_embedding
from src.multimodal.evaluation import clustering_metrics


EMBEDDING_KEYS = (
    "img_emb",
    "proj_emb",
    "gene_emb",
    "gene_emb_cm_img",
    "img_emb_cm",
    "gene_emb_cm_proj",
    "proj_emb_cm",
)
VALID_LABELS = frozenset({f"Layer_{i}" for i in range(1, 7)} | {"WM"})


@dataclass(frozen=True)
class AuditConfig:
    checkpoint_path: Path
    output_dir: Path
    embedding_keys: tuple[str, ...] = EMBEDDING_KEYS
    seed: int = 0


def _ensure_new_outputs(paths: Iterable[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing audit artifacts: " + ", ".join(existing)
        )


def audit_and_export(config: AuditConfig) -> dict:
    """Validate all sections, export float32 arrays, and reproduce pooled bisector."""
    output_dir = config.output_dir
    manifest_path = output_dir / "data" / "dataset_manifest.parquet"
    bundle_path = output_dir / "data" / "compact_embeddings.npz"
    embedding_manifest_path = output_dir / "data" / "embedding_manifest.json"
    audit_path = output_dir / "audit.json"
    legacy_path = output_dir / "legacy_pooled_bisector.json"
    _ensure_new_outputs(
        (manifest_path, bundle_path, embedding_manifest_path, audit_path, legacy_path)
    )

    dataset = DlpfcDataset.from_checkpoint(config.checkpoint_path)
    records: list[pd.DataFrame] = []
    embedding_blocks = {key: [] for key in config.embedding_keys}
    section_summaries = []
    global_row = 0

    for section_id in dataset.section_ids():
        adata = dataset.get_section(section_id)
        labels_series = adata.obs["ground_truth"]
        valid_label = labels_series.notna().to_numpy()
        labels = np.full(adata.n_obs, "", dtype="U16")
        labels[valid_label] = labels_series.to_numpy(dtype=object)[valid_label].astype(str)
        unknown = sorted(set(labels[valid_label]) - VALID_LABELS)
        if unknown:
            raise ValueError(f"Section {section_id} has unexpected labels: {unknown}")

        spatial = np.asarray(adata.obsm["spatial"], dtype=np.float32)
        if spatial.ndim != 2 or spatial.shape != (adata.n_obs, 2):
            raise ValueError(
                f"Section {section_id} spatial shape is {spatial.shape}, expected {(adata.n_obs, 2)}"
            )
        if not np.isfinite(spatial).all():
            raise ValueError(f"Section {section_id} has non-finite spatial coordinates")

        barcodes = adata.obs_names.to_numpy(dtype=str)
        section_records = pd.DataFrame(
            {
                "global_row": np.arange(global_row, global_row + adata.n_obs),
                "section_row": np.arange(adata.n_obs),
                "section_id": str(section_id),
                "donor_id": dataset.donor_id(section_id),
                "barcode": barcodes,
                "ground_truth": labels,
                "ground_truth_valid": valid_label,
                "spatial_x": spatial[:, 0],
                "spatial_y": spatial[:, 1],
            }
        )
        records.append(section_records)
        global_row += adata.n_obs

        for key in config.embedding_keys:
            if key not in adata.obsm:
                raise KeyError(f"Section {section_id} is missing obsm[{key!r}]")
            values = np.asarray(adata.obsm[key], dtype=np.float32)
            if values.ndim != 2 or values.shape != (adata.n_obs, 128):
                raise ValueError(
                    f"Section {section_id} obsm[{key!r}] shape is {values.shape}, "
                    f"expected {(adata.n_obs, 128)}"
                )
            if not np.isfinite(values).all():
                raise ValueError(f"Section {section_id} obsm[{key!r}] is non-finite")
            embedding_blocks[key].append(values)

        counts = pd.Series(labels[valid_label]).value_counts().sort_index().to_dict()
        section_summaries.append(
            {
                "section_id": str(section_id),
                "donor_id": dataset.donor_id(section_id),
                "n_spots": int(adata.n_obs),
                "n_valid_labels": int(valid_label.sum()),
                "label_counts": {str(key): int(value) for key, value in counts.items()},
            }
        )

    manifest = pd.concat(records, ignore_index=True)
    if manifest.duplicated(["section_id", "barcode"]).any():
        raise ValueError("Duplicate (section_id, barcode) identities detected")
    if len(dataset.section_ids()) != 12 or manifest["donor_id"].nunique() != 3:
        raise ValueError("Expected exactly 12 sections and 3 donors")

    arrays = {
        key: np.concatenate(blocks, axis=0).astype(np.float32, copy=False)
        for key, blocks in embedding_blocks.items()
    }
    embedding_manifest = {}
    for key, values in arrays.items():
        norms = np.linalg.norm(values, axis=1)
        if np.any(norms == 0):
            raise ValueError(f"Embedding {key!r} contains zero vectors")
        embedding_manifest[key] = {
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "norm_min": float(norms.min()),
            "norm_mean": float(norms.mean()),
            "norm_max": float(norms.max()),
        }

    valid = manifest["ground_truth_valid"].to_numpy()
    aligned_bisector = bisector_embedding(
        arrays["img_emb_cm"][valid], arrays["gene_emb_cm_img"][valid]
    )
    cluster_ids = KMeans(
        n_clusters=7, n_init=10, random_state=config.seed
    ).fit_predict(aligned_bisector)
    legacy_metrics = clustering_metrics(
        manifest.loc[valid, "ground_truth"].to_numpy(), cluster_ids
    )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(manifest_path, index=False)
    np.savez_compressed(bundle_path, **arrays)
    with embedding_manifest_path.open("w") as file:
        json.dump(embedding_manifest, file, indent=2)

    audit = {
        "checkpoint_path": str(config.checkpoint_path.resolve()),
        "checkpoint_size_bytes": config.checkpoint_path.stat().st_size,
        "n_spots": int(len(manifest)),
        "n_valid_labels": int(valid.sum()),
        "n_sections": int(manifest["section_id"].nunique()),
        "n_donors": int(manifest["donor_id"].nunique()),
        "sections": section_summaries,
        "embedding_keys": list(config.embedding_keys),
    }
    with audit_path.open("w") as file:
        json.dump(audit, file, indent=2)
    with legacy_path.open("w") as file:
        json.dump(
            {
                "representation": "gene_emb_cm_img + img_emb_cm bisector",
                "protocol": "pooled annotated spots, KMeans(k=7,n_init=10)",
                "seed": config.seed,
                "metrics": legacy_metrics,
            },
            file,
            indent=2,
        )

    return {"audit": audit, "legacy_metrics": legacy_metrics}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-path", type=Path, default=Path("checkpoints/dlpfc.pkl")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = audit_and_export(
        AuditConfig(
            checkpoint_path=args.checkpoint_path,
            output_dir=args.output_dir,
            seed=args.seed,
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
