"""Joint-section clustering with biological-conservation and batch-mixing metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from src.multimodal.evaluation import clustering_metrics
from src.multimodal.representations import existing_embedding_representations


def _ilisi(values: np.ndarray, batch: np.ndarray, k: int = 30) -> float:
    neighbors = NearestNeighbors(
        n_neighbors=min(k + 1, len(values)), metric="cosine"
    ).fit(values).kneighbors(values, return_distance=False)[:, 1:]
    scores = []
    for row in neighbors:
        _, counts = np.unique(batch[row], return_counts=True)
        proportions = counts / counts.sum()
        scores.append(1.0 / np.sum(proportions**2))
    return float(np.mean(scores))


def _batch_kl(values: np.ndarray, batch: np.ndarray, k: int = 30) -> float:
    names, global_counts = np.unique(batch, return_counts=True)
    expected = global_counts / global_counts.sum()
    index = {name: position for position, name in enumerate(names)}
    neighbors = NearestNeighbors(
        n_neighbors=min(k + 1, len(values)), metric="cosine"
    ).fit(values).kneighbors(values, return_distance=False)[:, 1:]
    divergences = []
    for row in neighbors:
        counts = np.zeros(len(names), dtype=np.float64)
        for value in batch[row]:
            counts[index[value]] += 1
        observed = (counts + 1e-6) / (counts.sum() + 1e-6 * len(names))
        divergences.append(np.sum(observed * np.log(observed / expected)))
    return float(np.mean(divergences))


def run(
    manifest_path: Path,
    embeddings_npz: Path,
    gene_npz: Path,
    cross_modal_npz: Path,
    sections: tuple[str, ...],
    output_dir: Path,
    seeds: tuple[int, ...] = tuple(range(10)),
) -> dict:
    metrics_path = output_dir / "metrics.parquet"
    summary_path = output_dir / "summary.json"
    if metrics_path.exists() or summary_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    manifest = pd.read_parquet(manifest_path)
    selected = manifest["section_id"].astype(str).isin(sections) & manifest["ground_truth_valid"]
    subset = manifest[selected]
    labels = subset["ground_truth"].to_numpy()
    section_ids = subset["section_id"].astype(str).to_numpy()
    donor_ids = subset["donor_id"].astype(str).to_numpy()
    with np.load(embeddings_npz) as data:
        source = {key: data[key] for key in data.files}
    with np.load(gene_npz, allow_pickle=True) as data:
        source["gene_emb"] = data["embeddings"]
    with np.load(cross_modal_npz, allow_pickle=True) as data:
        source["gene_emb_cm_img"] = data["gene_projected"]
        source["img_emb_cm"] = data["image_projected"]
    representations = {
        name: values[selected.to_numpy()]
        for name, values in existing_embedding_representations(source).items()
    }
    n_clusters = len(np.unique(labels))
    records = []
    for name, values in representations.items():
        section_ilisi = _ilisi(values, section_ids)
        section_batchkl = _batch_kl(values, section_ids)
        donor_ilisi = _ilisi(values, donor_ids) if len(np.unique(donor_ids)) > 1 else 1.0
        donor_batchkl = _batch_kl(values, donor_ids) if len(np.unique(donor_ids)) > 1 else 0.0
        layer_conditioned_ilisi = np.mean([
            _ilisi(values[labels == layer], section_ids[labels == layer])
            for layer in np.unique(labels)
            if len(np.unique(section_ids[labels == layer])) > 1
        ])
        for seed in seeds:
            clusters = KMeans(
                n_clusters=n_clusters, n_init=1, random_state=seed
            ).fit_predict(values)
            records.append({
                "representation": name,
                "seed": seed,
                "n_spots": len(labels),
                "n_clusters": n_clusters,
                **clustering_metrics(labels, clusters),
                "section_ilisi": section_ilisi,
                "section_batchkl": section_batchkl,
                "donor_ilisi": donor_ilisi,
                "donor_batchkl": donor_batchkl,
                "layer_conditioned_section_ilisi": float(layer_conditioned_ilisi),
            })
    metrics = pd.DataFrame(records)
    summary = {
        name: {
            metric: float(group[metric].mean())
            for metric in (
                "ari", "nmi", "accuracy", "section_ilisi", "section_batchkl",
                "donor_ilisi", "donor_batchkl", "layer_conditioned_section_ilisi",
            )
        }
        for name, group in metrics.groupby("representation")
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_parquet(metrics_path, index=False)
    with summary_path.open("w") as file:
        json.dump({"sections": list(sections), "representations": summary}, file, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--embeddings-npz", type=Path, required=True)
    parser.add_argument("--gene-npz", type=Path, required=True)
    parser.add_argument("--cross-modal-npz", type=Path, required=True)
    parser.add_argument("--sections", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    args = parser.parse_args()
    print(json.dumps(run(
        args.manifest_path, args.embeddings_npz, args.gene_npz,
        args.cross_modal_npz, tuple(args.sections), args.output_dir, tuple(args.seeds),
    ), indent=2))


if __name__ == "__main__":
    main()
