import numpy as np

from src.cross_modal.procrustes import (
    fit_graph_positive_procrustes,
    pool_spatial_neighbors,
    transform_procrustes,
)


def test_procrustes_maps_are_orthogonal_and_preserve_geometry() -> None:
    rng = np.random.default_rng(0)
    gene = rng.normal(size=(20, 5)).astype(np.float32)
    rotation, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    image = (gene @ rotation).astype(np.float32)
    before_gene = gene / np.linalg.norm(gene, axis=1, keepdims=True)
    before_image = image / np.linalg.norm(image, axis=1, keepdims=True)
    gene_map, image_map, _ = fit_graph_positive_procrustes(
        gene, image, np.arange(20), np.empty((0, 2), dtype=np.int64)
    )
    aligned_gene, aligned_image = transform_procrustes(
        gene, image, gene_map, image_map
    )
    np.testing.assert_allclose(gene_map.T @ gene_map, np.eye(5), atol=1e-5)
    np.testing.assert_allclose(image_map.T @ image_map, np.eye(5), atol=1e-5)
    np.testing.assert_allclose(aligned_gene @ aligned_gene.T, before_gene @ before_gene.T, atol=1e-5)
    np.testing.assert_allclose(aligned_image @ aligned_image.T, before_image @ before_image.T, atol=1e-5)
    assert np.mean(np.sum(aligned_gene * aligned_image, axis=1)) > 0.99


def test_neighbor_pooling_uses_only_listed_spots() -> None:
    values = np.eye(3, dtype=np.float32)
    neighbors = (np.array([1]), np.array([0]), np.array([], dtype=np.int64))
    pooled = pool_spatial_neighbors(values, neighbors, np.arange(3), 1.0)
    np.testing.assert_allclose(pooled[0], pooled[1], atol=1e-6)
    np.testing.assert_allclose(pooled[2], values[2], atol=1e-6)
