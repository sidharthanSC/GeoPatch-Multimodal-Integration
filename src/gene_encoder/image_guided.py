"""Image-guided local-identity objectives for rich gene representation learning."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class ImageProjectionHead(nn.Module):
    """Map fixed image features into the gene encoder's exported space."""

    def __init__(
        self, input_dim: int = 128, hidden_dim: int = 256, output_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


@dataclass(frozen=True)
class NeighborhoodAlignmentLoss:
    total: torch.Tensor
    gene_to_image: torch.Tensor
    image_to_gene: torch.Tensor
    positive_cosine: torch.Tensor
    negative_cosine: torch.Tensor
    neighborhood_retrieval: torch.Tensor


def build_local_identity_batch(
    anchors: np.ndarray, neighbor_rows: np.ndarray
) -> tuple[np.ndarray, torch.Tensor]:
    """Gather anchors and their direct KNN neighbors and build a positive mask.

    Rows in ``neighbor_rows`` are global spot indices. The returned gallery starts
    with the anchors in their original order, making anchor positions deterministic.
    """
    anchors = np.asarray(anchors, dtype=np.int64)
    neighbor_rows = np.asarray(neighbor_rows, dtype=np.int64)
    if anchors.ndim != 1 or len(anchors) == 0:
        raise ValueError("anchors must be a non-empty 1-D array")
    if len(np.unique(anchors)) != len(anchors):
        raise ValueError("anchors must be unique")
    if neighbor_rows.ndim != 2:
        raise ValueError("neighbor_rows must be a fixed-width 2-D array")
    if np.any(anchors < 0) or np.any(anchors >= len(neighbor_rows)):
        raise IndexError("anchor index is outside neighbor_rows")

    gathered = anchors.tolist()
    positions = {int(index): position for position, index in enumerate(gathered)}
    for anchor in anchors:
        for neighbor in neighbor_rows[int(anchor)]:
            neighbor = int(neighbor)
            if neighbor not in positions:
                positions[neighbor] = len(gathered)
                gathered.append(neighbor)

    positives = np.zeros((len(anchors), len(gathered)), dtype=bool)
    for row, anchor in enumerate(anchors):
        positive_indices = [int(anchor), *map(int, neighbor_rows[int(anchor)])]
        for index in positive_indices:
            positives[row, positions[index]] = True
    return np.asarray(gathered, dtype=np.int64), torch.from_numpy(positives)


def multi_positive_info_nce(
    logits: torch.Tensor, positives: torch.Tensor
) -> torch.Tensor:
    """Cross entropy where all members of one local spatial identity are positive."""
    if logits.ndim != 2 or positives.shape != logits.shape:
        raise ValueError("logits and positives must have matching 2-D shapes")
    positives = positives.bool()
    if not positives.any(dim=1).all():
        raise ValueError("every query must have at least one positive")
    negative_infinity = torch.finfo(logits.dtype).min
    positive_mass = torch.logsumexp(
        logits.masked_fill(~positives, negative_infinity), dim=1
    )
    all_mass = torch.logsumexp(logits, dim=1)
    return (all_mass - positive_mass).mean()


def neighborhood_cross_modal_loss(
    gene_gallery: torch.Tensor,
    image_gallery: torch.Tensor,
    n_anchors: int,
    positives: torch.Tensor,
    logit_scale: torch.Tensor,
) -> NeighborhoodAlignmentLoss:
    """Symmetric KNN multi-positive InfoNCE over gene and image galleries."""
    if gene_gallery.ndim != 2 or gene_gallery.shape != image_gallery.shape:
        raise ValueError("gene and image galleries must be matching 2-D tensors")
    if not 1 <= n_anchors <= len(gene_gallery):
        raise ValueError("n_anchors must index a non-empty gallery prefix")
    if positives.shape != (n_anchors, len(gene_gallery)):
        raise ValueError("positive mask does not match anchor/gallery dimensions")
    if not torch.isfinite(logit_scale):
        raise ValueError("logit_scale must be finite")

    gene = F.normalize(gene_gallery, dim=1)
    image = F.normalize(image_gallery, dim=1)
    similarity = gene[:n_anchors] @ image.T
    reverse_similarity = image[:n_anchors] @ gene.T
    scale = logit_scale.exp()
    positives = positives.to(device=similarity.device, dtype=torch.bool)
    gene_to_image = multi_positive_info_nce(scale * similarity, positives)
    image_to_gene = multi_positive_info_nce(scale * reverse_similarity, positives)

    pair_similarity = torch.cat([similarity, reverse_similarity], dim=0)
    pair_positives = positives.repeat(2, 1)
    positive_cosine = pair_similarity[pair_positives].mean()
    negative_cosine = (
        pair_similarity[~pair_positives].mean()
        if (~pair_positives).any()
        else pair_similarity.new_zeros(())
    )
    correct_forward = positives.gather(1, similarity.argmax(dim=1, keepdim=True)).float()
    correct_reverse = positives.gather(
        1, reverse_similarity.argmax(dim=1, keepdim=True)
    ).float()
    retrieval = torch.cat([correct_forward, correct_reverse]).mean()
    return NeighborhoodAlignmentLoss(
        total=(gene_to_image + image_to_gene) / 2.0,
        gene_to_image=gene_to_image.detach(),
        image_to_gene=image_to_gene.detach(),
        positive_cosine=positive_cosine.detach(),
        negative_cosine=negative_cosine.detach(),
        neighborhood_retrieval=retrieval.detach(),
    )


def initial_logit_scale(temperature: float = 0.07) -> nn.Parameter:
    if not 0 < temperature < 1:
        raise ValueError("temperature must be in (0, 1)")
    return nn.Parameter(torch.tensor(math.log(1.0 / temperature)))


@torch.no_grad()
def clamp_logit_scale(logit_scale: torch.Tensor) -> None:
    logit_scale.clamp_(max=math.log(100.0))
