"""Aggregate fold-level supervised study artifacts into paper-ready tables."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.splits.dlpfc import GLOBAL_LABEL_ORDER


@dataclass(frozen=True)
class SupervisedStudySource:
    regime: str
    evaluation_dir: Path
    gcn_dir: Path
    multiphase_dir: Path


def _aggregate(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    metrics = ["accuracy", "balanced_accuracy", "macro_f1"]
    records = []
    for keys, group in frame.groupby(groups, sort=False):
        parts = keys if isinstance(keys, tuple) else (keys,)
        record = dict(zip(groups, parts))
        record["n_folds"] = int(group["fold_id"].nunique())
        for metric in metrics:
            values = group[metric].astype(float)
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            record[f"{metric}_median"] = float(values.median())
        records.append(record)
    return pd.DataFrame(records)


def _alignment_rows(source: SupervisedStudySource) -> list[dict[str, object]]:
    rows = []
    for fold_dir in sorted(path for path in source.multiphase_dir.iterdir() if path.is_dir()):
        artifact_path = fold_dir / "embeddings.npz"
        if not artifact_path.exists():
            continue
        with np.load(artifact_path, allow_pickle=True) as data:
            role = data["role"].astype(str)
            pairs = {
                "raw": (data["gene_raw"], data["image_raw"]),
                "phase12": (data["gene_phase2"], data["image_phase1"]),
                "phase3": (data["gene_phase3"], data["image_phase3"]),
            }
            for split_role in ("train", "test"):
                mask = role == split_role
                for phase, (gene, image) in pairs.items():
                    gene_unit = gene[mask] / np.maximum(
                        np.linalg.norm(gene[mask], axis=1, keepdims=True), 1e-12
                    )
                    image_unit = image[mask] / np.maximum(
                        np.linalg.norm(image[mask], axis=1, keepdims=True), 1e-12
                    )
                    rows.append(
                        {
                            "regime": source.regime,
                            "fold_id": fold_dir.name,
                            "role": split_role,
                            "phase": phase,
                            "paired_cosine": float(np.sum(gene_unit * image_unit, axis=1).mean()),
                        }
                    )
    return rows


def aggregate_supervised_study(
    sources: tuple[SupervisedStudySource, ...], output_dir: Path
) -> dict[str, object]:
    """Aggregate classifications, spectral clustering, alignment, and layer recall."""
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    metric_frames = []
    spectral_frames = []
    detail_records = []
    alignment_records = []
    for source in sources:
        metrics = pd.read_csv(source.evaluation_dir / "metrics.csv")
        metrics.insert(0, "regime", source.regime)
        metric_frames.append(metrics)
        gcn = pd.read_csv(source.gcn_dir / "metrics.csv")
        gcn.insert(0, "regime", source.regime)
        metric_frames.append(gcn)
        spectral = pd.read_csv(source.evaluation_dir / "spectral_section_metrics.csv")
        spectral.insert(0, "regime", source.regime)
        spectral_frames.append(spectral)
        details = json.loads(
            (source.evaluation_dir / "detailed_metrics.json").read_text(encoding="utf-8")
        )
        for record in details:
            record["regime"] = source.regime
            detail_records.append(record)
        gcn_details = json.loads(
            (source.gcn_dir / "detailed_metrics.json").read_text(encoding="utf-8")
        )
        for record in gcn_details:
            record["regime"] = source.regime
            detail_records.append(record)
        alignment_records.extend(_alignment_rows(source))

    all_metrics = pd.concat(metric_frames, ignore_index=True)
    aggregate = _aggregate(all_metrics, ["regime", "method", "family"])
    all_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    aggregate.to_csv(output_dir / "aggregate_metrics.csv", index=False)

    spectral_all = pd.concat(spectral_frames, ignore_index=True)
    spectral_summary = spectral_all.groupby("regime")[["accuracy", "ari", "nmi"]].agg(
        ["mean", "std", "median"]
    )
    spectral_summary.columns = ["_".join(column) for column in spectral_summary.columns]
    spectral_summary.reset_index().to_csv(output_dir / "spectral_summary.csv", index=False)

    recall_records = []
    for record in detail_records:
        for label in GLOBAL_LABEL_ORDER:
            value = record["per_layer_recall"].get(label)
            recall_records.append(
                {
                    "regime": record["regime"],
                    "fold_id": record["fold_id"],
                    "method": record["method"],
                    "family": record["family"],
                    "label": label,
                    "recall": value,
                    "support": record["support"].get(label, 0),
                }
            )
    recall = pd.DataFrame(recall_records)
    recall.to_csv(output_dir / "per_layer_recall.csv", index=False)
    recall_summary = (
        recall.dropna(subset=["recall"])
        .groupby(["regime", "method", "family", "label"], as_index=False)["recall"]
        .agg(["mean", "std", "median"])
        .reset_index()
    )
    recall_summary.to_csv(output_dir / "per_layer_recall_summary.csv", index=False)

    alignment = pd.DataFrame(alignment_records)
    alignment.to_csv(output_dir / "alignment_fold_metrics.csv", index=False)
    alignment_summary = (
        alignment.groupby(["regime", "role", "phase"], as_index=False)["paired_cosine"]
        .agg(["mean", "std", "median"])
        .reset_index()
    )
    alignment_summary.to_csv(output_dir / "alignment_summary.csv", index=False)
    summary = {
        "regimes": [source.regime for source in sources],
        "n_fold_metric_rows": int(len(all_metrics)),
        "n_aggregate_rows": int(len(aggregate)),
        "n_spectral_rows": int(len(spectral_all)),
        "primary_metric": "balanced_accuracy; accuracy and observed-class macro-F1 also reported",
        "aggregation_unit": "equal-weight folds (donors or sections), not pooled spots",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.study_manifest.read_text(encoding="utf-8"))
    sources = tuple(
        SupervisedStudySource(
            item["regime"],
            Path(item["evaluation_dir"]),
            Path(item["gcn_dir"]),
            Path(item["multiphase_dir"]),
        )
        for item in manifest["sources"]
    )
    print(json.dumps(aggregate_supervised_study(sources, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
