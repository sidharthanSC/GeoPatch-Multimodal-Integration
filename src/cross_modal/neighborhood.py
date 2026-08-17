"""Coordinate-neighbor construction and label-free multimodal edge pruning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class NeighborhoodPruningConfig:
    coordinate_neighbors: int = 6
    retained_neighbors: int = 3
    min_gene_cosine: float | None = None
    min_image_cosine: float | None = None


@dataclass(frozen=True)
class SplitNeighborhoodGraph:
    """One split-isolated undirected graph using global spot row indices."""

    node_indices: np.ndarray
    candidate_edges: np.ndarray
    edges: np.ndarray
    neighbors: tuple[np.ndarray, ...]
    diagnostics: dict[str, float | int | str]


def _normalized(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0) or not np.isfinite(values).all():
        raise ValueError("edge-pruning embeddings must be finite and nonzero")
    return values / norms


def coordinate_candidate_edges(
    coordinates: np.ndarray,
    section_ids: np.ndarray,
    split: np.ndarray,
    split_name: str,
    k: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a symmetrized coordinate kNN graph independently inside one split."""
    if k < 1:
        raise ValueError("k must be positive")
    coordinates = np.asarray(coordinates)
    section_ids = np.asarray(section_ids).astype(str)
    split = np.asarray(split).astype(str)
    if not (len(coordinates) == len(section_ids) == len(split)):
        raise ValueError("coordinates, section_ids, and split must have equal length")

    node_indices = np.flatnonzero(split == split_name).astype(np.int64)
    edge_set: set[tuple[int, int]] = set()
    for section_id in np.unique(section_ids[node_indices]):
        rows = node_indices[section_ids[node_indices] == section_id]
        if len(rows) < 2:
            continue
        n_neighbors = min(k + 1, len(rows))
        local_coordinates = coordinates[rows]
        local_neighbors = NearestNeighbors(n_neighbors=n_neighbors).fit(
            local_coordinates
        ).kneighbors(local_coordinates, return_distance=False)
        for local_source, local_targets in enumerate(local_neighbors):
            source = int(rows[local_source])
            for local_target in local_targets:
                target = int(rows[local_target])
                if source != target:
                    edge_set.add((min(source, target), max(source, target)))
    edges = np.asarray(sorted(edge_set), dtype=np.int64)
    if edges.size == 0:
        edges = np.empty((0, 2), dtype=np.int64)
    return node_indices, edges


def prune_multimodal_edges(
    node_indices: np.ndarray,
    candidate_edges: np.ndarray,
    gene_embeddings: np.ndarray,
    image_embeddings: np.ndarray,
    section_ids: np.ndarray,
    split_name: str,
    config: NeighborhoodPruningConfig,
) -> SplitNeighborhoodGraph:
    """Keep mutually nominated coordinate edges supported by both modalities.

    Each endpoint separately ranks candidates in frozen gene and image space and only
    nominates the intersection of both top-neighbor lists. An undirected edge survives
    only if both endpoints nominate each other. This cuts modality-disagreement edges
    and never creates a relationship between non-neighboring coordinates.
    """
    if config.retained_neighbors < 1:
        raise ValueError("retained_neighbors must be positive")
    node_indices = np.asarray(node_indices, dtype=np.int64)
    candidate_edges = np.asarray(candidate_edges, dtype=np.int64).reshape(-1, 2)
    gene = _normalized(gene_embeddings)
    image = _normalized(image_embeddings)
    section_ids = np.asarray(section_ids).astype(str)
    node_set = set(node_indices.tolist())

    incident: dict[int, list[tuple[float, float, int]]] = {
        int(row): [] for row in node_indices
    }
    gene_scores: list[float] = []
    image_scores: list[float] = []
    valid_candidates: list[tuple[int, int]] = []
    for source, target in candidate_edges:
        source, target = int(source), int(target)
        if source == target or source not in node_set or target not in node_set:
            raise ValueError("candidate edges must connect distinct nodes in the split")
        if section_ids[source] != section_ids[target]:
            raise ValueError("candidate edge crosses tissue sections")
        gene_cosine = float(gene[source] @ gene[target])
        image_cosine = float(image[source] @ image[target])
        if config.min_gene_cosine is not None and gene_cosine < config.min_gene_cosine:
            continue
        if config.min_image_cosine is not None and image_cosine < config.min_image_cosine:
            continue
        incident[source].append((gene_cosine, image_cosine, target))
        incident[target].append((gene_cosine, image_cosine, source))
        valid_candidates.append((source, target))
        gene_scores.append(gene_cosine)
        image_scores.append(image_cosine)

    nominations: set[tuple[int, int]] = set()
    for source, candidates in incident.items():
        gene_ranked = sorted(candidates, key=lambda item: (-item[0], item[2]))
        image_ranked = sorted(candidates, key=lambda item: (-item[1], item[2]))
        gene_targets = {
            target for _, _, target in gene_ranked[: config.retained_neighbors]
        }
        image_targets = {
            target for _, _, target in image_ranked[: config.retained_neighbors]
        }
        for target in sorted(gene_targets & image_targets):
            nominations.add((source, target))

    retained = sorted(
        (source, target)
        for source, target in valid_candidates
        if (source, target) in nominations and (target, source) in nominations
    )
    edges = np.asarray(retained, dtype=np.int64)
    if edges.size == 0:
        edges = np.empty((0, 2), dtype=np.int64)

    neighbor_lists: dict[int, list[int]] = {int(row): [] for row in node_indices}
    for source, target in edges:
        neighbor_lists[int(source)].append(int(target))
        neighbor_lists[int(target)].append(int(source))
    neighbors = tuple(
        np.asarray(sorted(neighbor_lists.get(row, [])), dtype=np.int64)
        for row in range(len(section_ids))
    )
    degrees = np.asarray([len(neighbor_lists[int(row)]) for row in node_indices])
    diagnostics: dict[str, float | int | str] = {
        "split": split_name,
        "n_nodes": int(len(node_indices)),
        "candidate_edges": int(len(candidate_edges)),
        "eligible_edges": int(len(valid_candidates)),
        "retained_edges": int(len(edges)),
        "retention_fraction": float(len(edges) / max(len(candidate_edges), 1)),
        "mean_degree": float(degrees.mean()) if len(degrees) else 0.0,
        "max_degree": int(degrees.max()) if len(degrees) else 0,
        "isolated_fraction": float(np.mean(degrees == 0)) if len(degrees) else 0.0,
        "mean_eligible_gene_cosine": float(np.mean(gene_scores)) if gene_scores else 0.0,
        "mean_eligible_image_cosine": float(np.mean(image_scores)) if image_scores else 0.0,
    }
    return SplitNeighborhoodGraph(
        node_indices=node_indices,
        candidate_edges=candidate_edges,
        edges=edges,
        neighbors=neighbors,
        diagnostics=diagnostics,
    )


def build_split_neighborhood_graph(
    coordinates: np.ndarray,
    section_ids: np.ndarray,
    split: np.ndarray,
    split_name: str,
    gene_embeddings: np.ndarray,
    image_embeddings: np.ndarray,
    config: NeighborhoodPruningConfig,
) -> SplitNeighborhoodGraph:
    """Construct and prune one graph without using nodes from another split."""
    nodes, candidates = coordinate_candidate_edges(
        coordinates, section_ids, split, split_name, config.coordinate_neighbors
    )
    return prune_multimodal_edges(
        nodes,
        candidates,
        gene_embeddings,
        image_embeddings,
        section_ids,
        split_name,
        config,
    )
