"""CLIP-style cross-modal InfoNCE architecture between gene and image spot embeddings.

Both modalities are already 128-d (``gene_emb`` from ``src/gene_encoder/``, ``img_emb``/
``proj_emb`` from the BYOL image pipeline / ``src/train/``), but weren't trained to be
cross-modally aligned with each other. A small per-modality :class:`ProjectionHead`
(both trainable -- unlike a frozen-anchor setup, *both* sides adapt to find a shared
aligned space) maps each into a shared space, trained with the symmetric InfoNCE loss
from CLIP (Radford et al. 2021): in-batch negatives, a learnable temperature.

Unlike BYOL (``src/gene_encoder/model.py``), InfoNCE has no trivial collapse shortcut --
a constant projector output makes every pair equally similar, which is the *worst*
possible cross-entropy loss here, not the best -- so no auxiliary variance regularizer
is needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class CrossModalConfig:
    """Architecture and loss hyperparameters.

    Attributes
    ----------
    gene_input_dim, image_input_dim:
        Input dimensionality of the gene-side and image-side embeddings (128 for both,
        as currently stored, but kept independent in case that ever changes).
    hidden_dim:
        Hidden width of each projection head's MLP.
    output_dim:
        Shared cross-modal embedding dimensionality both heads project into.
    dropout:
        Dropout probability inside each projection head.
    logit_scale_init:
        Initial value of the learnable temperature, as ``log(1 / temperature)`` --
        ``log(1/0.07) ~= 2.6593``, matching CLIP's own initialization.
    logit_scale_max:
        Upper clamp on ``logit_scale`` after each optimizer step (``log(100)``,
        matching CLIP) -- prevents the temperature from collapsing to (near-)zero and
        destabilizing training.
    """

    gene_input_dim: int = 128
    image_input_dim: int = 128
    hidden_dim: int = 256
    output_dim: int = 128
    dropout: float = 0.1
    logit_scale_init: float = math.log(1 / 0.07)
    logit_scale_max: float = math.log(100)


class ProjectionHead(nn.Module):
    """``Linear -> LayerNorm -> GELU -> Dropout -> Linear``, mapping one modality into
    the shared cross-modal space. GELU (not ReLU) to match CLIP's own projection
    layers -- a smooth activation with no downside here, unlike heavier architectural
    changes (e.g. a graph-attention head) which would require redesigning batch
    sampling around a spatial graph."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossModalModel(nn.Module):
    """Gene-side and image-side projection heads, plus a learnable InfoNCE temperature."""

    def __init__(self, config: CrossModalConfig) -> None:
        super().__init__()
        self.config = config
        self.gene_projector = ProjectionHead(
            config.gene_input_dim, config.hidden_dim, config.output_dim, config.dropout
        )
        self.image_projector = ProjectionHead(
            config.image_input_dim, config.hidden_dim, config.output_dim, config.dropout
        )
        self.logit_scale = nn.Parameter(torch.tensor(config.logit_scale_init))

    def forward(self, gene_batch: torch.Tensor, image_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project and L2-normalize both modalities; does not apply the temperature."""
        gene_proj = F.normalize(self.gene_projector(gene_batch), dim=-1)
        image_proj = F.normalize(self.image_projector(image_batch), dim=-1)
        return gene_proj, image_proj

    @torch.no_grad()
    def clamp_logit_scale(self) -> None:
        self.logit_scale.data.clamp_(max=self.config.logit_scale_max)


def clip_loss(gene_proj: torch.Tensor, image_proj: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    """Symmetric InfoNCE loss over in-batch negatives (CLIP-style).

    ``gene_proj``/``image_proj`` must already be L2-normalized (see
    :meth:`CrossModalModel.forward`). Spot ``i``'s gene and image embeddings are the
    positive pair; every other spot in the batch is a negative, in both directions.
    """
    logits = logit_scale.exp() * gene_proj @ image_proj.T  # (B, B)
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_gene_to_image = F.cross_entropy(logits, labels)
    loss_image_to_gene = F.cross_entropy(logits.T, labels)
    return (loss_gene_to_image + loss_image_to_gene) / 2.0


@torch.no_grad()
def retrieval_accuracy(gene_proj: torch.Tensor, image_proj: torch.Tensor) -> float:
    """Mean top-1 in-batch retrieval accuracy, both directions (a monitoring metric,
    not part of the loss): does spot i's gene embedding's nearest image embedding in
    this batch belong to spot i, and vice versa."""
    logits = gene_proj @ image_proj.T
    labels = torch.arange(logits.shape[0], device=logits.device)
    acc_gene_to_image = (logits.argmax(dim=1) == labels).float().mean()
    acc_image_to_gene = (logits.T.argmax(dim=1) == labels).float().mean()
    return float((acc_gene_to_image + acc_image_to_gene).item() / 2.0)
