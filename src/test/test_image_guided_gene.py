import numpy as np
import pytest
import torch

from src.gene_encoder.image_guided import (
    ImageProjectionHead,
    build_local_identity_batch,
    initial_logit_scale,
    multi_positive_info_nce,
    neighborhood_cross_modal_loss,
)
from src.gene_encoder.model import GeneEncoder, GeneEncoderConfig


def test_local_identity_batch_contains_self_and_every_direct_neighbor():
    neighbors = np.asarray(
        [
            [1, 2],
            [0, 2],
            [0, 1],
            [1, 2],
        ],
        dtype=np.int64,
    )
    gathered, positives = build_local_identity_batch(np.asarray([0, 3]), neighbors)

    assert gathered[:2].tolist() == [0, 3]
    lookup = {index: position for position, index in enumerate(gathered)}
    assert {gathered[index] for index in torch.where(positives[0])[0].tolist()} == {0, 1, 2}
    assert {gathered[index] for index in torch.where(positives[1])[0].tolist()} == {1, 2, 3}
    assert positives[:, lookup[0]].tolist() == [True, False]
    assert positives[:, lookup[3]].tolist() == [False, True]


def test_multi_positive_info_nce_rewards_positive_similarity():
    positives = torch.tensor([[True, True, False], [False, True, True]])
    good = torch.tensor([[3.0, 2.0, -2.0], [-2.0, 3.0, 2.0]])
    bad = -good
    assert multi_positive_info_nce(good, positives) < multi_positive_info_nce(bad, positives)
    with pytest.raises(ValueError, match="at least one positive"):
        multi_positive_info_nce(good, torch.zeros_like(positives))


def test_neighborhood_alignment_updates_full_gene_encoder_and_image_head():
    torch.manual_seed(0)
    encoder = GeneEncoder(
        GeneEncoderConfig(
            input_dim=6,
            hidden_dims=(8, 4),
            embedding_dim=3,
            dropout=0.0,
        )
    )
    image_head = ImageProjectionHead(5, 7, 3, dropout=0.0)
    expression = torch.randn(4, 6)
    image = torch.randn(4, 5)
    positives = torch.tensor(
        [[True, True, False, False], [True, True, True, False]]
    )
    output = neighborhood_cross_modal_loss(
        encoder(expression),
        image_head(image),
        n_anchors=2,
        positives=positives,
        logit_scale=initial_logit_scale(),
    )
    output.total.backward()

    assert torch.isfinite(output.total)
    assert all(parameter.grad is not None for parameter in encoder.parameters())
    assert all(parameter.grad is not None for parameter in image_head.parameters())
    assert 0 <= output.neighborhood_retrieval <= 1
