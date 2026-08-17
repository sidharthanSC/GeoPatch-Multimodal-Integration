"""Unit tests for MP-MNCA components."""

import numpy as np
import torch

from src.mp_mnca.config import MpMncaConfig
from src.mp_mnca.masking import (
    apply_mask,
    contiguous_block_mask,
    create_mask_tensor,
    scalar_mask,
)
from src.mp_mnca.model import (
    GeneEncoder,
    MorphologyPriorCrossAttention,
    MpMncaModel,
    covariance_regularization,
    latent_prediction_loss,
    variance_regularization,
)


def test_scalar_mask():
    """Test scalar feature masking."""
    x = torch.randn(4, 100)
    masked = scalar_mask(x, mask_rate=0.3)
    assert masked.shape == x.shape
    # Check that approximately 30% are zero
    zero_frac = (masked == 0).float().mean().item()
    assert 0.2 < zero_frac < 0.4


def test_contiguous_block_mask():
    """Test contiguous block masking."""
    x = torch.randn(4, 100)
    masked = contiguous_block_mask(x, mask_rate=0.3, block_size=10, num_blocks=3)
    assert masked.shape == x.shape
    # Check that some blocks are zeroed
    zero_frac = (masked == 0).float().mean().item()
    assert zero_frac > 0.1


def test_create_mask_tensor():
    """Test mask tensor creation."""
    mask = create_mask_tensor((4, 100), "scalar", 0.3)
    assert mask.shape == (4, 100)
    assert mask.dtype == torch.bool

    mask = create_mask_tensor((4, 100), "contiguous_block", 0.3, block_size=10, num_blocks=3)
    assert mask.shape == (4, 100)
    assert mask.dtype == torch.bool


def test_apply_mask():
    """Test apply_mask dispatcher."""
    x = torch.randn(4, 100)
    masked = apply_mask(x, "scalar", 0.3)
    assert masked.shape == x.shape

    masked = apply_mask(x, "contiguous_block", 0.3, block_size=10, num_blocks=3)
    assert masked.shape == x.shape


def test_gene_encoder():
    """Test GeneEncoder forward pass."""
    config = MpMncaConfig(gene_input_dim=100, gene_hidden_dims=(64, 32), gene_embedding_dim=16)
    encoder = GeneEncoder(config)
    x = torch.randn(8, 100)
    out = encoder(x)
    assert out.shape == (8, 16)


def test_morphology_prior_cross_attention():
    """Test MorphologyPriorCrossAttention forward pass."""
    config = MpMncaConfig(
        gene_embedding_dim=32,
        num_heads=2,
        image_embedding_dim=16,
        image_pca_dim=8,
    )
    attention = MorphologyPriorCrossAttention(config)

    batch_size = 4
    n_neighbors = 6
    embed_dim = 32
    image_dim = 16

    center_gene = torch.randn(batch_size, embed_dim)
    neighbor_genes = torch.randn(batch_size, n_neighbors, embed_dim)
    center_image = torch.randn(batch_size, image_dim)
    neighbor_images = torch.randn(batch_size, n_neighbors, image_dim)
    center_coords = torch.randn(batch_size, 2)
    neighbor_coords = torch.randn(batch_size, n_neighbors, 2)

    contextual, attn_weights, morph_prior = attention(
        center_gene, neighbor_genes, center_image, neighbor_images, center_coords, neighbor_coords
    )

    assert contextual.shape == (batch_size, embed_dim)
    assert attn_weights.shape == (batch_size, n_neighbors)
    assert morph_prior.shape == (batch_size, n_neighbors)
    # Attention weights should sum to 1 (mean across heads of softmax outputs)
    assert torch.allclose(attn_weights.sum(dim=-1), torch.ones(batch_size), atol=1e-3)


def test_mp_mnca_model():
    """Test full MP-MNCA model forward pass."""
    config = MpMncaConfig(
        gene_input_dim=100,
        gene_hidden_dims=(64, 32),
        gene_embedding_dim=16,
        num_heads=2,
        image_embedding_dim=16,
        image_pca_dim=8,
        use_ema_target=True,
    )
    model = MpMncaModel(config)

    batch_size = 4
    n_neighbors = 6
    gene_dim = 100
    image_dim = 16

    center_gene_masked = torch.randn(batch_size, gene_dim)
    neighbor_genes_masked = torch.randn(batch_size, n_neighbors, gene_dim)
    center_image = torch.randn(batch_size, image_dim)
    neighbor_images = torch.randn(batch_size, n_neighbors, image_dim)
    center_coords = torch.randn(batch_size, 2)
    neighbor_coords = torch.randn(batch_size, n_neighbors, 2)

    output = model.forward_online(
        center_gene_masked,
        neighbor_genes_masked,
        center_image,
        neighbor_images,
        center_coords,
        neighbor_coords,
    )

    assert output.online_embedding.shape == (batch_size, 16)
    assert output.target_embedding.shape == (batch_size, 16)
    assert output.contextual_embedding.shape == (batch_size, 16)
    assert output.attention_weights.shape == (batch_size, n_neighbors)
    assert output.morphology_prior.shape == (batch_size, n_neighbors)


def test_ema_update():
    """Test EMA target encoder update."""
    config = MpMncaConfig(
        gene_input_dim=100,
        gene_hidden_dims=(32,),
        gene_embedding_dim=16,
        use_ema_target=True,
        ema_decay=0.99,
    )
    model = MpMncaModel(config)

    # Get initial target params
    target_params_before = [p.clone() for p in model.target_encoder.parameters()]

    # Update online encoder
    with torch.no_grad():
        for p in model.online_encoder.parameters():
            p.add_(torch.randn_like(p) * 0.1)

    # Update target
    model.update_target_encoder()

    # Check that target params moved towards online params
    for target_before, target_after, online in zip(
        target_params_before, model.target_encoder.parameters(), model.online_encoder.parameters()
    ):
        # Target should be closer to online than before
        dist_before = (target_before - online).abs().mean()
        dist_after = (target_after - online).abs().mean()
        assert dist_after < dist_before


def test_variance_regularization():
    """Test variance regularization."""
    embeddings = torch.randn(32, 16)
    loss = variance_regularization(embeddings, gamma=1.0)
    assert loss.item() >= 0


def test_covariance_regularization():
    """Test covariance regularization."""
    embeddings = torch.randn(32, 16)
    loss = covariance_regularization(embeddings, max_dimensions=16)
    assert loss.item() >= 0


def test_latent_prediction_loss():
    """Test latent prediction loss."""
    predicted = torch.randn(16, 32)
    target = torch.randn(16, 32)
    loss = latent_prediction_loss(predicted, target)
    assert loss.item() >= 0


def test_smoke_train_step():
    """Smoke test: one forward+backward pass."""
    config = MpMncaConfig(
        gene_input_dim=100,
        gene_hidden_dims=(64, 32),
        gene_embedding_dim=16,
        num_heads=2,
        image_embedding_dim=16,
        image_pca_dim=8,
        epochs=1,
        batch_size=4,
    )
    model = MpMncaModel(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch_size = 4
    n_neighbors = 6
    gene_dim = 100
    image_dim = 16

    center_gene_masked = torch.randn(batch_size, gene_dim)
    center_gene_clean = torch.randn(batch_size, gene_dim)
    neighbor_genes_masked = torch.randn(batch_size, n_neighbors, gene_dim)
    neighbor_genes_clean = torch.randn(batch_size, n_neighbors, gene_dim)
    center_image = torch.randn(batch_size, image_dim)
    neighbor_images = torch.randn(batch_size, n_neighbors, image_dim)
    center_coords = torch.randn(batch_size, 2)
    neighbor_coords = torch.randn(batch_size, n_neighbors, 2)

    # Forward
    output = model.forward_online(
        center_gene_masked,
        neighbor_genes_masked,
        center_image,
        neighbor_images,
        center_coords,
        neighbor_coords,
    )

    target_emb = model.forward_target(center_gene_clean)
    predicted = model.predict(output.contextual_embedding)

    latent_loss = latent_prediction_loss(predicted, target_emb)
    var_loss = variance_regularization(output.online_embedding)
    cov_loss = covariance_regularization(output.online_embedding)

    loss = latent_loss + var_loss + cov_loss

    # Backward
    loss.backward()
    optimizer.step()

    assert loss.item() > 0


if __name__ == "__main__":
    test_scalar_mask()
    test_contiguous_block_mask()
    test_create_mask_tensor()
    test_apply_mask()
    test_gene_encoder()
    test_morphology_prior_cross_attention()
    test_mp_mnca_model()
    test_ema_update()
    test_variance_regularization()
    test_covariance_regularization()
    test_latent_prediction_loss()
    test_smoke_train_step()
    print("All tests passed!")