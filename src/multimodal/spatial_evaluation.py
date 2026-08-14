"""Apply one common spatial refinement and evaluate spatial cluster continuity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors

from src.multimodal.evaluation import clustering_metrics


def spatial_neighbors(coordinates: np.ndarray, k: int = 6) -> np.ndarray:
    finder = NearestNeighbors(n_neighbors=k + 1).fit(coordinates)
    indices = finder.kneighbors(coordinates, return_distance=False)
    return indices[:, 1:]


def refine_strict_majority(
    cluster_ids: np.ndarray, neighbors: np.ndarray, minimum_votes: int = 4
) -> np.ndarray:
    refined = cluster_ids.copy()
    for row, neighbor_rows in enumerate(neighbors):
        labels, counts = np.unique(cluster_ids[neighbor_rows], return_counts=True)
        best = counts.max()
        winners = labels[counts == best]
        if best >= minimum_votes and len(winners) == 1:
            refined[row] = winners[0]
    return refined


def neighbor_agreement(cluster_ids: np.ndarray, neighbors: np.ndarray) -> float:
    return float(np.mean(cluster_ids[:, None] == cluster_ids[neighbors]))


def connectedness(cluster_ids: np.ndarray, neighbors: np.ndarray) -> float:
    rows = np.repeat(np.arange(len(cluster_ids)), neighbors.shape[1])
    cols = neighbors.reshape(-1)
    graph = csr_matrix(
        (np.ones(len(rows), dtype=np.uint8), (rows, cols)),
        shape=(len(cluster_ids), len(cluster_ids)),
    )
    graph = graph.maximum(graph.T)
    scores = []
    weights = []
    for cluster_id in np.unique(cluster_ids):
        mask = cluster_ids == cluster_id
        subgraph = graph[mask][:, mask]
        _, component_labels = connected_components(subgraph, directed=False)
        largest = np.bincount(component_labels).max()
        scores.append(largest / mask.sum())
        weights.append(mask.sum())
    return float(np.average(scores, weights=weights))


def evaluate_refinement(
    manifest_path: Path,
    assignments_path: Path,
    output_dir: Path,
    k: int = 6,
    minimum_votes: int = 4,
) -> dict:
    metrics_path = output_dir / "metrics.parquet"
    refined_path = output_dir / "assignments.parquet"
    summary_path = output_dir / "aggregate_summary.json"
    if any(path.exists() for path in (metrics_path, refined_path, summary_path)):
        raise FileExistsError(f"Refusing to overwrite artifacts under {output_dir}")

    manifest = pd.read_parquet(manifest_path)
    assignments = pd.read_parquet(assignments_path)
    metrics_records = []
    refined_records = []

    neighbor_by_section = {}
    label_by_section = {}
    barcode_order_by_section = {}
    for section_id in manifest["section_id"].drop_duplicates():
        section = manifest[
            (manifest["section_id"] == section_id) & manifest["ground_truth_valid"]
        ]
        neighbor_by_section[str(section_id)] = spatial_neighbors(
            section[["spatial_x", "spatial_y"]].to_numpy(), k=k
        )
        label_by_section[str(section_id)] = section["ground_truth"].to_numpy()
        barcode_order_by_section[str(section_id)] = section["barcode"].to_numpy()

    group_columns = ["section_id", "representation", "backend", "seed"]
    for keys, group in assignments.groupby(group_columns, sort=False):
        section_id, representation, backend, seed = keys
        section_id = str(section_id)
        order = pd.Index(group["barcode"]).get_indexer(
            barcode_order_by_section[section_id]
        )
        if np.any(order < 0):
            raise ValueError(f"Missing assignments for section {section_id}")
        cluster_ids = group.iloc[order]["cluster_id"].to_numpy()
        true_labels = label_by_section[section_id]
        neighbors = neighbor_by_section[section_id]
        refined = refine_strict_majority(cluster_ids, neighbors, minimum_votes)

        before = clustering_metrics(true_labels, cluster_ids)
        after = clustering_metrics(true_labels, refined)
        metrics_records.append(
            {
                "section_id": section_id,
                "representation": representation,
                "backend": backend,
                "seed": int(seed),
                "k": k,
                "minimum_votes": minimum_votes,
                **{f"before_{name}": value for name, value in before.items()},
                **{f"after_{name}": value for name, value in after.items()},
                "before_neighbor_agreement": neighbor_agreement(cluster_ids, neighbors),
                "after_neighbor_agreement": neighbor_agreement(refined, neighbors),
                "before_connectedness": connectedness(cluster_ids, neighbors),
                "after_connectedness": connectedness(refined, neighbors),
                "fraction_changed": float(np.mean(cluster_ids != refined)),
            }
        )
        refined_records.extend(
            {
                "section_id": section_id,
                "barcode": barcode,
                "representation": representation,
                "backend": backend,
                "seed": int(seed),
                "cluster_id_raw": int(raw),
                "cluster_id_refined": int(new),
            }
            for barcode, raw, new in zip(
                barcode_order_by_section[section_id], cluster_ids, refined
            )
        )

    metrics = pd.DataFrame(metrics_records)
    refined_frame = pd.DataFrame(refined_records)
    section_means = metrics.groupby(
        ["section_id", "representation"], as_index=False
    ).mean(numeric_only=True)
    summary = {}
    for representation, group in section_means.groupby("representation"):
        summary[representation] = {
            "median_before_ari": float(group["before_ari"].median()),
            "median_after_ari": float(group["after_ari"].median()),
            "median_before_nmi": float(group["before_nmi"].median()),
            "median_after_nmi": float(group["after_nmi"].median()),
            "median_neighbor_agreement_gain": float(
                (group["after_neighbor_agreement"] - group["before_neighbor_agreement"]).median()
            ),
            "median_fraction_changed": float(group["fraction_changed"].median()),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_parquet(metrics_path, index=False)
    refined_frame.to_parquet(refined_path, index=False)
    with summary_path.open("w") as file:
        json.dump(summary, file, indent=2)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--assignments-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--minimum-votes", type=int, default=4)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = evaluate_refinement(
        args.manifest_path,
        args.assignments_path,
        args.output_dir,
        args.k,
        args.minimum_votes,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
