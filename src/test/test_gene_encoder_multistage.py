import numpy as np
import torch

from src.gene_encoder.model import GeneEncoder, GeneEncoderConfig
from src.gene_encoder.train import export_multistage_embeddings


def test_multistage_encoder_shapes_and_final_contract():
    encoder = GeneEncoder(
        GeneEncoderConfig(
            input_dim=12,
            hidden_dims=(10, 8, 6),
            embedding_dim=4,
            dropout=0.0,
        )
    )
    inputs = torch.randn(5, 12)
    features = encoder.forward_features(inputs)

    assert list(features) == [
        "encoder_stage_1",
        "encoder_stage_2",
        "encoder_stage_3",
        "embedding",
    ]
    assert features["encoder_stage_1"].shape == (5, 10)
    assert features["encoder_stage_2"].shape == (5, 8)
    assert features["encoder_stage_3"].shape == (5, 6)
    assert features["embedding"].shape == (5, 4)
    assert torch.equal(encoder(inputs), features["embedding"])


def test_final_loss_reaches_every_hidden_stage():
    encoder = GeneEncoder(
        GeneEncoderConfig(input_dim=12, hidden_dims=(10, 8, 6), embedding_dim=4)
    )
    encoder(torch.randn(7, 12)).square().mean().backward()
    gradients = [block[0].weight.grad for block in encoder.hidden_blocks]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert all(torch.count_nonzero(gradient) > 0 for gradient in gradients)


def test_multistage_export_is_deterministic_and_float32():
    encoder = GeneEncoder(
        GeneEncoderConfig(
            input_dim=12,
            hidden_dims=(10, 8, 6),
            embedding_dim=4,
            dropout=0.5,
        )
    )
    expression = torch.randn(11, 12)
    first = export_multistage_embeddings(encoder, expression, batch_size=4)
    second = export_multistage_embeddings(encoder, expression, batch_size=5)

    assert first.keys() == second.keys()
    for key in first:
        assert first[key].dtype == np.float32
        np.testing.assert_allclose(first[key], second[key], rtol=1e-6, atol=1e-6)


def test_encoder_rejects_empty_or_nonpositive_hidden_dims():
    for hidden_dims in ((), (8, 0), (-1, 4)):
        try:
            GeneEncoder(GeneEncoderConfig(input_dim=12, hidden_dims=hidden_dims))
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected invalid hidden_dims to fail: {hidden_dims}")
