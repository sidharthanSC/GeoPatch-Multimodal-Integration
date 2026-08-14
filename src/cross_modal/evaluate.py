"""Full per-section cross-modal retrieval and geometry evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.multimodal.representations import l2_normalize


def _effective_rank(values: np.ndarray) -> float:
    centered = values - values.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    energy = singular_values**2
    probabilities = energy / np.maximum(energy.sum(), 1e-12)
    positive = probabilities[probabilities > 0]
    return float(np.exp(-(positive * np.log(positive)).sum()))


def _directional_ranks(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    similarity = query @ gallery.T
    paired = similarity.diagonal()[:, None]
    greater = (similarity > paired).sum(axis=1)
    equal = (similarity == paired).sum(axis=1) - 1
    return 1.0 + greater + 0.5 * equal


def evaluate_pair(gene: np.ndarray, image: np.ndarray) -> dict[str, float]:
    gene = l2_normalize(gene)
    image = l2_normalize(image)
    gene_ranks = _directional_ranks(gene, image)
    image_ranks = _directional_ranks(image, gene)
    ranks = np.concatenate([gene_ranks, image_ranks])
    paired_cosine = np.sum(gene * image, axis=1)

    rng = np.random.default_rng(0)
    permutation = rng.permutation(len(gene))
    while np.any(permutation == np.arange(len(gene))):
        permutation = rng.permutation(len(gene))
    unpaired_cosine = np.sum(gene * image[permutation], axis=1)

    metrics = {
        f"recall_at_{k}": float(np.mean(ranks <= k)) for k in (1, 5, 10, 50)
    }
    metrics.update(
        {
            "mrr": float(np.mean(1.0 / ranks)),
            "median_rank": float(np.median(ranks)),
            "normalized_median_rank": float(np.median(ranks) / len(gene)),
            "foscttm": float(np.mean((ranks - 1.0) / max(len(gene) - 1, 1))),
            "paired_cosine": float(paired_cosine.mean()),
            "unpaired_cosine": float(unpaired_cosine.mean()),
            "cosine_gap": float(paired_cosine.mean() - unpaired_cosine.mean()),
            "gene_effective_rank": _effective_rank(gene),
            "image_effective_rank": _effective_rank(image),
            "gene_mean_dimension_std": float(gene.std(axis=0).mean()),
            "image_mean_dimension_std": float(image.std(axis=0).mean()),
        }
    )
    return metrics


def evaluate_npz(
    manifest_path: Path,
    predictions_npz: Path,
    output_path: Path,
) -> pd.DataFrame:
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    manifest = pd.read_parquet(manifest_path)
    with np.load(predictions_npz, allow_pickle=True) as data:
        gene = data["gene_projected"]
        image = data["image_projected"]
        section_ids = data["section_ids"].astype(str)
        barcodes = data["barcodes"].astype(str)

    expected_ids = list(zip(manifest["section_id"].astype(str), manifest["barcode"].astype(str)))
    observed_ids = list(zip(section_ids, barcodes))
    if expected_ids != observed_ids:
        raise ValueError("Prediction NPZ row identities do not match the audit manifest")

    records = []
    for section_id in manifest["section_id"].drop_duplicates():
        mask = section_ids == str(section_id)
        records.append(
            {
                "section_id": str(section_id),
                "n_spots": int(mask.sum()),
                **evaluate_pair(gene[mask], image[mask]),
            }
        )
    frame = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    summary_path = output_path.with_suffix(".summary.json")
    summary = {
        column: float(frame[column].median())
        for column in frame.columns
        if column not in {"section_id", "n_spots"}
    }
    with summary_path.open("w") as file:
        json.dump(summary, file, indent=2)
    return frame


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--predictions-npz", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = evaluate_npz(args.manifest_path, args.predictions_npz, args.output_path)
    print(result.median(numeric_only=True).to_json(indent=2))


if __name__ == "__main__":
    main()
