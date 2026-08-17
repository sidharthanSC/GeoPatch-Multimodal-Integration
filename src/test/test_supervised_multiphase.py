import numpy as np
import torch

from src.train.model import ProjectorConfig, WeightedSliceContrastiveLoss
from src.train.supervised_multiphase import EMAMetricProjector, symmetric_ema_metric_loss


def test_ema_target_is_frozen_and_updates_after_online_change():
    torch.manual_seed(0)
    model = EMAMetricProjector(
        ProjectorConfig(input_dim=4, hidden_dim=6, projection_dim=3, dropout=0.0),
        ema_decay=0.5,
    )
    assert all(not parameter.requires_grad for parameter in model.target.parameters())
    target_before = [parameter.detach().clone() for parameter in model.target.parameters()]
    with torch.no_grad():
        for parameter in model.online.parameters():
            parameter.add_(1.0)
    model.update_target()
    for before, online, target in zip(target_before, model.online.parameters(), model.target.parameters()):
        torch.testing.assert_close(target, 0.5 * before + 0.5 * online)


def test_symmetric_metric_loss_backpropagates_only_online_and_predictor():
    torch.manual_seed(0)
    model = EMAMetricProjector(
        ProjectorConfig(input_dim=4, hidden_dim=8, projection_dim=3, dropout=0.0)
    )
    first = torch.randn(6, 4)
    second = torch.randn(6, 4)
    pair_kind = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.float32)
    loss, similarity = symmetric_ema_metric_loss(
        model, first, second, pair_kind, WeightedSliceContrastiveLoss(5.0, 1.0, 0.0)
    )
    loss.backward()
    assert similarity.shape == (6,)
    assert all(parameter.grad is not None for parameter in model.online.parameters())
    assert all(parameter.grad is not None for parameter in model.predictor.parameters())
    assert all(parameter.grad is None for parameter in model.target.parameters())


def test_positive_weight_exceeds_negative_weight_configuration():
    criterion = WeightedSliceContrastiveLoss(5.0, 1.0, 0.0)
    first = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    second = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    loss, _ = criterion(first, second, torch.tensor([0.0, 1.0]))
    np.testing.assert_allclose(float(loss), 3.0)
