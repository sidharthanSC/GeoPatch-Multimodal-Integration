import numpy as np
import torch
from scipy import sparse

from src.gene_encoder.model import (
    GeneEncoder,
    GeneEncoderConfig,
    GeneReconstructionDecoder,
    build_byol_learner,
    cosine_geometry_regularization,
    covariance_regularization,
    masked_reconstruction_loss,
)
from src.gene_encoder.rich_data import SparseExpressionStore, SparseMaskedView
from src.gene_encoder.rich_train import OnlineFeatureCapture


def _store() -> SparseExpressionStore:
    values = np.asarray(
        [
            [1, 0, 2, 0, 3, 0],
            [0, 2, 0, 1, 0, 3],
            [2, 0, 1, 0, 2, 0],
            [0, 1, 0, 3, 0, 2],
        ],
        dtype=np.float32,
    )
    return SparseExpressionStore(
        matrix=sparse.csr_matrix(values),
        gene_names=np.asarray([f"g{i}" for i in range(6)]),
        barcodes=np.asarray([f"b{i}" for i in range(4)]),
        section_ids=np.asarray(["151507"] * 4),
        gene_std=values.std(axis=0),
    )


def test_expression_residual_uses_fixed_components():
    encoder = GeneEncoder(
        GeneEncoderConfig(
            input_dim=6,
            hidden_dims=(8, 4),
            embedding_dim=3,
            dropout=0,
            expression_residual=True,
        )
    )
    components = torch.randn(3, 6)
    encoder.initialize_expression_residual(components)
    np.testing.assert_allclose(
        encoder.expression_residual.weight.detach().numpy(), components.numpy()
    )
    assert not encoder.expression_residual.weight.requires_grad
    assert encoder(torch.randn(5, 6)).shape == (5, 3)


def test_rich_regularizers_and_decoder_are_finite():
    embeddings = torch.randn(8, 12, requires_grad=True)
    geometry = torch.randn(8, 4)
    decoder = GeneReconstructionDecoder(12, 6)
    target = torch.rand(8, 6)
    mask = torch.rand(8, 6) < 0.4
    loss = (
        covariance_regularization(embeddings, max_dimensions=5)
        + cosine_geometry_regularization(embeddings, geometry)
        + masked_reconstruction_loss(decoder(embeddings), target, mask)
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(embeddings.grad).all()


def test_sparse_views_and_hooks_capture_exact_byol_batch():
    torch.manual_seed(0)
    store = _store()
    device = torch.device("cpu")
    first = SparseMaskedView(store, device, mask_rate=0.3, noise_std_fraction=0.0)
    second = SparseMaskedView(store, device, mask_rate=0.3, noise_std_fraction=0.0)
    config = GeneEncoderConfig(
        input_dim=6,
        hidden_dims=(8, 4),
        embedding_dim=3,
        projection_dim=3,
        projection_hidden_dim=8,
        dropout=0,
    )
    encoder = GeneEncoder(config)
    learner = build_byol_learner(encoder, config, first, second, device)
    capture = OnlineFeatureCapture(learner, encoder)
    indices = torch.tensor([0, 2, 3], dtype=torch.long)
    loss = learner(indices)
    features = capture.consume(expected_batch=3)
    capture.close()

    assert torch.isfinite(loss)
    assert np.array_equal(first.last_indices, indices.numpy())
    assert np.array_equal(second.last_indices, indices.numpy())
    assert features["encoder_stage_1"].shape == (6, 8)
    assert features["encoder_stage_2"].shape == (6, 4)
    assert features["embedding"].shape == (6, 3)
