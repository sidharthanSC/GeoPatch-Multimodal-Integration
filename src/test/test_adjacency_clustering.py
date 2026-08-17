import numpy as np
from scipy import sparse
from sklearn.metrics import adjusted_rand_score

from src.multimodal.adjacency_clustering import (
    cluster_adjacency,
    edge_cosine_distance_squared,
    median_kernel_affinity,
    sparse_adjacency,
)


def test_edge_affinity_supports_sparse_and_dense_features():
    values = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    edges = np.asarray([[0, 0], [1, 2]])
    dense = edge_cosine_distance_squared(values, edges)
    sparse_values = edge_cosine_distance_squared(sparse.csr_matrix(values), edges)
    np.testing.assert_allclose(dense, [0.0, 2.0])
    np.testing.assert_allclose(sparse_values, dense)
    weights, bandwidth = median_kernel_affinity(dense)
    assert bandwidth == 2.0
    assert weights[0] > weights[1] > 0


def test_sparse_adjacency_is_symmetric_without_self_loops():
    edges = np.asarray([[0, 1, 1], [1, 0, 2]])
    adjacency = sparse_adjacency(3, edges, np.asarray([0.5, 0.5, 0.8]))
    np.testing.assert_allclose(adjacency.toarray(), adjacency.toarray().T)
    np.testing.assert_allclose(adjacency.diagonal(), 0.0)
    assert adjacency[1, 2] == 0.8


def test_spectral_clustering_recovers_disconnected_blocks():
    first = np.ones((4, 4)) - np.eye(4)
    second = np.ones((4, 4)) - np.eye(4)
    adjacency = sparse.block_diag((first, second), format="csr")
    predicted = cluster_adjacency(adjacency, n_clusters=2, seed=0)
    assert adjusted_rand_score([0] * 4 + [1] * 4, predicted) == 1.0
