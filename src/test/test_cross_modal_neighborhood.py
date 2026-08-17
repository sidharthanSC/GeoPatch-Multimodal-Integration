import numpy as np

from src.cross_modal.neighborhood import (
    NeighborhoodPruningConfig,
    build_split_neighborhood_graph,
    coordinate_candidate_edges,
    prune_multimodal_edges,
)
from src.cross_modal.train import make_anchor_batch


def test_coordinate_graph_never_crosses_section_or_split() -> None:
    coordinates = np.array(
        [[0, 0], [1, 0], [2, 0], [0, 0], [1, 0], [3, 0]], dtype=np.float32
    )
    sections = np.array(["A", "A", "A", "B", "B", "A"])
    split = np.array(["train", "train", "val", "train", "train", "val"])
    nodes, edges = coordinate_candidate_edges(
        coordinates, sections, split, "train", k=2
    )
    assert set(nodes.tolist()) == {0, 1, 3, 4}
    assert len(edges) == 2
    for source, target in edges:
        assert sections[source] == sections[target]
        assert split[source] == split[target] == "train"


def test_pruning_is_mutual_deterministic_and_bounded() -> None:
    nodes = np.arange(4)
    candidates = np.array([[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]])
    sections = np.array(["A"] * 4)
    gene = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]], dtype=np.float32)
    image = gene.copy()
    config = NeighborhoodPruningConfig(retained_neighbors=1)
    first = prune_multimodal_edges(
        nodes, candidates, gene, image, sections, "train", config
    )
    second = prune_multimodal_edges(
        nodes, candidates, gene, image, sections, "train", config
    )
    np.testing.assert_array_equal(first.edges, second.edges)
    assert set(map(tuple, first.edges)).issubset(set(map(tuple, candidates)))
    assert first.diagnostics["max_degree"] <= 1
    for source, target in first.edges:
        assert target in first.neighbors[source]
        assert source in first.neighbors[target]


def test_disagreeing_modality_can_cut_a_coordinate_edge() -> None:
    nodes = np.arange(3)
    candidates = np.array([[0, 1], [0, 2], [1, 2]])
    sections = np.array(["A"] * 3)
    gene = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    image = np.array([[1, 0], [0, 1], [1, 0]], dtype=np.float32)
    graph = prune_multimodal_edges(
        nodes,
        candidates,
        gene,
        image,
        sections,
        "train",
        NeighborhoodPruningConfig(retained_neighbors=1),
    )
    assert len(graph.edges) == 0
    assert (0, 2) not in set(map(tuple, graph.edges))
    assert (1, 2) not in set(map(tuple, graph.edges))


def test_split_graph_and_anchor_batch_keep_isolated_self_positive() -> None:
    coordinates = np.array([[0, 0], [1, 0], [10, 0]], dtype=np.float32)
    sections = np.array(["A", "A", "A"])
    split = np.array(["train"] * 3)
    embeddings = np.eye(3, dtype=np.float32)
    graph = build_split_neighborhood_graph(
        coordinates,
        sections,
        split,
        "train",
        embeddings,
        embeddings,
        NeighborhoodPruningConfig(coordinate_neighbors=2, retained_neighbors=1),
    )
    gathered, anchors, edges = make_anchor_batch(np.array([0, 1, 2]), graph.neighbors)
    np.testing.assert_array_equal(gathered[:3], np.array([0, 1, 2]))
    assert anchors.tolist() == [0, 1, 2]
    assert edges.shape[0] == 2
