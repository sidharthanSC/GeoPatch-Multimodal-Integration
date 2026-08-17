import numpy as np
import torch

from src.prior_models.staig.data import (
    load_feature_bundle,
    build_spatial_graph,
    image_guided_edge_probabilities,
    sample_augmented_edges,
)
from src.prior_models.staig.evaluate import refine_labels, tied_gmm
from src.prior_models.staig.model import (
    StaigModel,
    neighbor_contrastive_loss,
    normalized_adjacency,
)
from src.prior_models.staig.multiscale_evaluation import _probe
from src.prior_models.staig.rich_evaluation import dimension_controlled_values
from src.prior_models.staig.published import SPAGCN_PUBLISHED_DLPFC, STAIG_PUBLISHED_DLPFC


def test_spatial_graph_is_symmetric_and_has_no_self_edges():
    coordinates = np.asarray([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=np.float32)
    edge_index = build_spatial_graph(coordinates, n_neighbors=1)
    edges = set(map(tuple, edge_index.T.tolist()))
    assert all(source != target for source, target in edges)
    assert all((target, source) in edges for source, target in edges)


def test_edge_probabilities_sum_to_one_per_source():
    coordinates = np.asarray([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=np.float32)
    edges = build_spatial_graph(coordinates, n_neighbors=2)
    probabilities = image_guided_edge_probabilities(edges, coordinates)
    for node in np.unique(edges[0]):
        assert np.isclose(probabilities[edges[0] == node].sum(), 1.0)


def test_sampled_edges_remain_symmetric():
    coordinates = np.asarray([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=np.float32)
    edges = build_spatial_graph(coordinates, n_neighbors=2)
    probabilities = image_guided_edge_probabilities(edges, coordinates)
    sampled = sample_augmented_edges(edges, probabilities, np.random.default_rng(0))
    pairs = set(map(tuple, sampled.T.tolist()))
    assert all((target, source) in pairs for source, target in pairs)


def test_model_and_neighbor_loss_are_finite():
    edges = torch.tensor([[0, 1, 1, 2, 2, 3, 1, 0, 2, 1, 3, 2]], dtype=torch.long).reshape(2, 6)
    adjacency = normalized_adjacency(edges, 4, torch.float32, torch.device("cpu"))
    model = StaigModel(input_dim=3, hidden_dim=4, projection_dim=4)
    first = model(torch.randn(4, 3), adjacency)
    second = model(torch.randn(4, 3), adjacency)
    loss = neighbor_contrastive_loss(
        first, second, edges, torch.tensor([0, 0, 1, 1]), temperature=10.0
    )
    assert first.shape == (4, 4)
    assert torch.isfinite(loss)
    loss.backward()


def test_refinement_uses_neighbor_plurality():
    labels = np.asarray([0, 1, 1, 1])
    coordinates = np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
    refined = refine_labels(labels, coordinates, n_neighbors=3)
    assert refined[0] == 1


def test_tied_gmm_handles_low_rank_float32_embeddings():
    rng = np.random.default_rng(0)
    latent = rng.normal(size=(100, 2)).astype(np.float32)
    embeddings = np.column_stack([latent, latent[:, :1], latent[:, :1]])
    predicted = tied_gmm(embeddings, n_clusters=3)
    assert predicted.shape == (100,)


def test_multiscale_probe_is_finite_and_dimension_matched():
    rng = np.random.default_rng(0)
    values = rng.normal(size=(60, 256)).astype(np.float32)
    labels = np.repeat(["a", "b", "c"], 20)
    train = np.concatenate([np.arange(0, 15), np.arange(20, 35), np.arange(40, 55)])
    test = np.setdiff1d(np.arange(60), train)
    result = _probe(values, labels, train, test, seed=0, pca_dim=8)
    assert set(result) == {"accuracy", "balanced_accuracy", "macro_f1"}
    assert all(np.isfinite(value) for value in result.values())


def test_rich_dimension_control_reduces_wide_inputs_before_gmm():
    values = np.random.default_rng(0).normal(size=(80, 300)).astype(np.float32)
    transformed, explained = dimension_controlled_values(values, target_dim=16, seed=0)
    assert transformed.shape == (80, 16)
    assert transformed.dtype == np.float64
    assert 0 < explained <= 1


def test_rich_dimension_control_does_not_rotate_matching_inputs():
    values = np.random.default_rng(0).normal(size=(20, 8)).astype(np.float32)
    transformed, explained = dimension_controlled_values(values, target_dim=8, seed=0)
    assert transformed.shape == values.shape
    assert explained == 1.0
    np.testing.assert_allclose(transformed.mean(axis=0), 0, atol=1e-6)


def test_published_tables_cover_the_same_twelve_sections():
    assert len(STAIG_PUBLISHED_DLPFC) == 12
    assert STAIG_PUBLISHED_DLPFC.keys() == SPAGCN_PUBLISHED_DLPFC.keys()
    assert np.isclose(np.mean([value[0] for value in STAIG_PUBLISHED_DLPFC.values()]), 0.6916666667)


def test_feature_bundle_reorders_by_section_and_barcode(tmp_path):
    path = tmp_path / "features.npz"
    np.savez(
        path,
        stage=np.asarray([[3, 30], [1, 10], [2, 20]], dtype=np.float32),
        section_ids=np.asarray(["b", "a", "a"]),
        barcodes=np.asarray(["bc1", "bc1", "bc2"]),
    )
    bundle = load_feature_bundle(path, "stage")
    np.testing.assert_array_equal(
        bundle.for_section("a", np.asarray(["bc2", "bc1"])),
        [[2, 20], [1, 10]],
    )
