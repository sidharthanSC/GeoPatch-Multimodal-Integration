import math

import pytest
import torch

from src.cross_modal.model import (
    CrossModalConfig,
    CrossModalModel,
    clip_loss,
    gradient_balanced_clip_loss,
    gradient_balanced_clip_loss_from_similarity,
    positive_neighborhood_loss,
)


def test_ratio_one_equals_clip_loss() -> None:
    torch.manual_seed(0)
    gene = torch.nn.functional.normalize(torch.randn(8, 16), dim=1)
    image = torch.nn.functional.normalize(torch.randn(8, 16), dim=1)
    logit_scale = torch.tensor(math.log(1 / 0.07))

    expected = clip_loss(gene, image, logit_scale)
    actual = gradient_balanced_clip_loss(gene, image, logit_scale, 1.0)

    torch.testing.assert_close(actual.total, expected)
    torch.testing.assert_close(actual.weighted_attraction, torch.tensor(0.0))


@pytest.mark.parametrize("ratio", [5.0, 10.0])
def test_gradient_ratio_and_negative_gradients(ratio: float) -> None:
    torch.manual_seed(1)
    similarity = torch.randn(6, 6, requires_grad=True)
    logit_scale = torch.tensor(math.log(1 / 0.2))

    baseline = gradient_balanced_clip_loss_from_similarity(
        similarity, logit_scale, 1.0
    ).total
    baseline_gradient = torch.autograd.grad(baseline, similarity, retain_graph=True)[0]

    weighted = gradient_balanced_clip_loss_from_similarity(
        similarity, logit_scale, ratio
    ).total
    weighted_gradient = torch.autograd.grad(weighted, similarity)[0]

    off_diagonal = ~torch.eye(similarity.shape[0], dtype=torch.bool)
    torch.testing.assert_close(
        weighted_gradient[off_diagonal], baseline_gradient[off_diagonal]
    )
    torch.testing.assert_close(
        weighted_gradient.diagonal(),
        ratio * baseline_gradient.diagonal(),
        rtol=1e-5,
        atol=1e-6,
    )
    assert torch.count_nonzero(weighted_gradient[off_diagonal]) > 0


def test_loss_is_finite_for_two_spots() -> None:
    gene = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=1)
    image = torch.nn.functional.normalize(torch.tensor([[0.9, 0.1], [0.1, 0.9]]), dim=1)
    result = gradient_balanced_clip_loss(gene, image, torch.tensor(0.0), 5.0)
    assert torch.isfinite(result.total)


@pytest.mark.parametrize("ratio", [0.0, 0.5, float("inf"), float("nan")])
def test_invalid_ratio_is_rejected(ratio: float) -> None:
    with pytest.raises(ValueError, match="same_spot_gradient_ratio"):
        gradient_balanced_clip_loss_from_similarity(
            torch.eye(2), torch.tensor(0.0), ratio
        )


def test_invalid_shapes_are_rejected() -> None:
    with pytest.raises(ValueError, match="square"):
        gradient_balanced_clip_loss_from_similarity(
            torch.ones(2, 3), torch.tensor(0.0), 5.0
        )
    with pytest.raises(ValueError, match="matching"):
        gradient_balanced_clip_loss(
            torch.ones(2, 3), torch.ones(2, 4), torch.tensor(0.0), 5.0
        )


def test_cross_modal_forward_remains_normalized() -> None:
    model = CrossModalModel(
        CrossModalConfig(gene_input_dim=4, image_input_dim=4, hidden_dim=8, output_dim=3)
    )
    gene, image = model(torch.randn(5, 4), torch.randn(5, 4))
    torch.testing.assert_close(gene.norm(dim=1), torch.ones(5))
    torch.testing.assert_close(image.norm(dim=1), torch.ones(5))


def test_positive_only_constant_outputs_are_penalized_for_collapse() -> None:
    constant = torch.ones(8, 4, requires_grad=True)
    output = positive_neighborhood_loss(
        constant,
        constant.clone(),
        torch.arange(8),
        torch.empty((2, 0), dtype=torch.long),
    )
    torch.testing.assert_close(output.self_alignment, torch.tensor(0.0))
    assert output.gene_variance > 0
    assert output.image_variance > 0
    assert output.total > 0


def test_radial_variation_does_not_hide_angular_collapse() -> None:
    direction = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    radial = torch.arange(1.0, 9.0)[:, None] * direction
    output = positive_neighborhood_loss(
        radial,
        radial.clone(),
        torch.arange(8),
        torch.empty((2, 0), dtype=torch.long),
    )
    assert output.gene_variance > 0.9
    assert output.image_variance > 0.9


def test_positive_loss_ignores_unreferenced_non_anchor_rows() -> None:
    torch.manual_seed(4)
    gene = torch.randn(4, 3)
    image = torch.randn(4, 3)
    anchors = torch.tensor([0, 1])
    edges = torch.tensor([[0], [2]])
    baseline = positive_neighborhood_loss(gene, image, anchors, edges).total
    gene_changed = gene.clone()
    image_changed = image.clone()
    gene_changed[3] = 1000
    image_changed[3] = -1000
    changed = positive_neighborhood_loss(
        gene_changed, image_changed, anchors, edges
    ).total
    torch.testing.assert_close(changed, baseline)
