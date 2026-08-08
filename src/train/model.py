"""Shared projection model for slice-aware metric learning.

The pretrained BYOL encoder is frozen outside this module. This model receives its
fixed image embeddings and learns a new normalized representation in which patches
from the same tissue slice are close and patches from different slices are separated.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class ProjectorConfig:
    """Architecture configuration for SliceProjectionModel."""

    input_dim: int = 128
    hidden_dim: int = 256
    projection_dim: int = 128
    dropout: float = 0.1


class SliceProjectionModel(nn.Module):
    """Shared MLP projector applied to both members of every pair.

    A single shared projector is preferable here because the objective is supervised
    metric learning over frozen embeddings, not standard augmentation-based BYOL.
    The returned embeddings are always L2-normalized, so their dot product is cosine
    similarity and the exported representation matches the training representation.
    """

    def __init__(self, config: ProjectorConfig) -> None:
        super().__init__()
        self.config = config

        self.projector = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return unit-normalized projected embeddings."""
        z = self.projector(x)
        return F.normalize(z, p=2, dim=-1)


class WeightedSliceContrastiveLoss(nn.Module):
    """Pairwise cosine loss for same-slice and different-slice pairs.

    eta = 0: same slice
        The squared attraction term pushes cosine similarity toward 1.

    eta = 1: different slices
        The squared hinge term penalizes similarities above `negative_margin`.

    Positive pairs are weighted more strongly by default because the earlier model
    reduced both positive and negative similarities. With balanced pair sampling,
    positive_weight=2 and negative_weight=1 explicitly prioritizes preserving and
    tightening intra-slice structure while still separating slices.
    """

    def __init__(
        self,
        positive_weight: float = 2.0,
        negative_weight: float = 1.0,
        negative_margin: float = 0.0, 
    ) -> None:
        super().__init__()

        if positive_weight <= 0:
            raise ValueError("positive_weight must be positive.")
        if negative_weight <= 0:
            raise ValueError("negative_weight must be positive.")
        if not -1.0 <= negative_margin <= 1.0:
            raise ValueError("negative_margin must lie in [-1, 1].")

        self.positive_weight = positive_weight
        self.negative_weight = negative_weight
        self.negative_margin = negative_margin

    def forward(
        self,
        z_a: torch.Tensor,
        z_b: torch.Tensor,
        eta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return mean loss and per-pair cosine similarities."""
        if z_a.shape != z_b.shape:
            raise ValueError(
                f"Projected pair shapes must match, got {z_a.shape} and {z_b.shape}."
            )

        eta = eta.to(device=z_a.device, dtype=z_a.dtype).view(-1)
        if eta.shape[0] != z_a.shape[0]:
            raise ValueError("eta length must equal batch size.")

        similarity = torch.sum(z_a * z_b, dim=-1).clamp(-1.0, 1.0)

        # Strongly pull same-slice pairs toward cosine similarity 1.
        positive_loss = (1.0 - similarity).pow(2)

        # Push different-slice pairs to margin or lower.
        negative_loss = torch.relu(
            similarity - self.negative_margin
        ).pow(2)

        pair_loss = (
            (1.0 - eta) * self.positive_weight * positive_loss
            + eta * self.negative_weight * negative_loss
        )

        return pair_loss.mean(), similarity
