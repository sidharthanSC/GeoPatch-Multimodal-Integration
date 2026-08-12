"""Method 1 (unsupervised): bisector spot representation + clustering.

Interprets a spot's image and gene embeddings as two vectors forming a "V" -- the
representation is the unit vector exactly between them (the angle bisector): normalize
both to unit length, add, renormalize. One 128-d embedding per spot, then clustered
(KMeans, k=7 matching the known layer count -- STAIG/SpaGCN-style domain identification
commonly uses mclust or Leiden instead; KMeans is used here for simplicity/determinism,
easy to swap) into layer-like groups, entirely without using ground-truth labels.
Labels are only used afterward, to *report* accuracy/ARI/NMI -- never for clustering
itself.

Usage
-----
    python -m src.multimodal.bisector
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from src.multimodal.data import load_embeddings_and_labels
from src.multimodal.evaluation import clustering_metrics


def bisector_embedding(image_emb: np.ndarray, gene_emb: np.ndarray) -> np.ndarray:
    """Unit vector exactly between each spot's (unit-normalized) image and gene embedding."""
    image_unit = image_emb / np.linalg.norm(image_emb, axis=1, keepdims=True)
    gene_unit = gene_emb / np.linalg.norm(gene_emb, axis=1, keepdims=True)
    bisector = image_unit + gene_unit
    norm = np.maximum(np.linalg.norm(bisector, axis=1, keepdims=True), 1e-12)
    return bisector / norm


def run(checkpoint_path: Path, n_clusters: int, seed: int) -> dict:
    gene_emb, image_emb, labels, barcodes, section_ids = load_embeddings_and_labels(checkpoint_path)
    bisector = bisector_embedding(image_emb, gene_emb)

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    cluster_ids = kmeans.fit_predict(bisector)

    metrics = clustering_metrics(labels, cluster_ids)
    return {
        "metrics": metrics,
        "bisector": bisector,
        "cluster_ids": cluster_ids,
        "labels": labels,
        "barcodes": barcodes,
        "section_ids": section_ids,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, default=Path("checkpoints/dlpfc.pkl"))
    parser.add_argument("--n-clusters", type=int, default=7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/multimodal"))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = run(args.checkpoint_path, args.n_clusters, args.seed)

    print("Method 1 (bisector + KMeans clustering):", json.dumps(result["metrics"], indent=2))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "method1_bisector.npz",
        bisector=result["bisector"],
        cluster_ids=result["cluster_ids"],
        labels=result["labels"],
        barcodes=result["barcodes"],
        section_ids=result["section_ids"],
    )
    with (args.output_dir / "method1_bisector_metrics.json").open("w") as f:
        json.dump(result["metrics"], f, indent=2)
    print(f"Wrote {args.output_dir}/method1_bisector.npz and method1_bisector_metrics.json")


if __name__ == "__main__":
    main()
