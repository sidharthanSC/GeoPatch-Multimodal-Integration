"""MP-MNCA Phase 1: Gene Expression Cross-Attention (3000-dim throughout).

Replaces STAIG's GCN with cross-attention over spatial neighbors.
Uses raw 3000-dim gene expression throughout - no dimension reduction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class Phase1Output:
    """Output from Phase 1 cross-attention."""
    spot_embeddings: torch.Tensor       # (n_spots, gene_dim) - 3000-dim
    attention_weights: torch.Tensor     # (n_spots, n_neighbors)
    morphology_prior: torch.Tensor      # (n_spots, n_neighbors)


class GeneExpressionCrossAttention(nn.Module):
    """Cross-attention over 3000-dim gene expression with morphology prior."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        gene_dim = config.gene_dim
        num_heads = config.num_heads

        assert gene_dim % num_heads == 0
        self.head_dim = gene_dim // num_heads
        self.num_heads = num_heads
        self.scale = config.temperature ** -0.5

        # Q, K, V projections on gene_dim
        self.q_proj = nn.Linear(gene_dim, gene_dim, bias=False)
        self.k_proj = nn.Linear(gene_dim, gene_dim, bias=False)
        self.v_proj = nn.Linear(gene_dim, gene_dim, bias=False)
        self.out_proj = nn.Linear(gene_dim, gene_dim)

        # Learnable morphology weight (β)
        if config.learn_morphology_weight:
            self.morphology_weight = nn.Parameter(torch.tensor(config.morphology_prior_weight))
        else:
            self.register_buffer("morphology_weight", torch.tensor(config.morphology_prior_weight))

        # Learnable position weight (γ)
        if config.use_relative_position_bias and config.learn_position_weight:
            self.position_weight = nn.Parameter(torch.tensor(config.position_bias_weight))
        elif config.use_relative_position_bias:
            self.register_buffer("position_weight", torch.tensor(config.position_bias_weight))

        # Relative position bias MLP
        if config.use_relative_position_bias:
            self.position_bias_mlp = nn.Sequential(
                nn.Linear(2, config.relative_position_dim),
                nn.GELU(),
                nn.Linear(config.relative_position_dim, 1),
            )

        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.attention_norm = nn.LayerNorm(gene_dim)

    def compute_morphology_prior(
        self,
        center_image: torch.Tensor,      # (batch, image_dim)
        neighbor_images: torch.Tensor,   # (batch, n_neighbors, image_dim)
        center_gene: torch.Tensor = None,
        neighbor_genes: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute morphology prior from image or gene similarity."""
        if self.config.morphology_prior_type == "cosine":
            center_norm = F.normalize(center_image, dim=-1)
            neighbor_norm = F.normalize(neighbor_images, dim=-1)
            sim = torch.einsum("bd,bnd->bn", center_norm, neighbor_norm)
            prior = (sim + 1) / 2
        elif self.config.morphology_prior_type == "rbf":
            center_norm = F.normalize(center_image, dim=-1)
            neighbor_norm = F.normalize(neighbor_images, dim=-1)
            dist = torch.norm(center_norm.unsqueeze(1) - neighbor_norm, dim=-1)
            prior = torch.exp(-dist**2 / 2)
        elif self.config.morphology_prior_type == "gene_cosine":
            if center_gene is None or neighbor_genes is None:
                raise ValueError("gene_cosine prior requires raw gene expression inputs")
            center_norm = F.normalize(center_gene, dim=-1)
            neighbor_norm = F.normalize(neighbor_genes, dim=-1)
            sim = torch.einsum("bd,bnd->bn", center_norm, neighbor_norm)
            prior = (sim + 1) / 2
        else:
            raise ValueError(f"Unknown prior type: {self.config.morphology_prior_type}")
        return prior.clamp(min=1e-6)

    def compute_position_bias(
        self,
        center_coords: torch.Tensor,     # (batch, 2)
        neighbor_coords: torch.Tensor,   # (batch, n_neighbors, 2)
    ) -> torch.Tensor:
        if not self.config.use_relative_position_bias:
            return torch.zeros(
                center_coords.shape[0], neighbor_coords.shape[1],
                device=center_coords.device, dtype=center_coords.dtype
            )
        rel_pos = neighbor_coords - center_coords.unsqueeze(1)
        return self.position_bias_mlp(rel_pos).squeeze(-1)

    def forward(
        self,
        center_gene: torch.Tensor,       # (batch, gene_dim) - CLEAN
        neighbor_genes: torch.Tensor,    # (batch, n_neighbors, gene_dim) - MASKED
        center_image: torch.Tensor,      # (batch, image_dim)
        neighbor_images: torch.Tensor,   # (batch, n_neighbors, image_dim)
        center_coords: torch.Tensor,     # (batch, 2)
        neighbor_coords: torch.Tensor,   # (batch, n_neighbors, 2)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            contextual: (batch, gene_dim)
            attention_weights: (batch, n_neighbors)
            morphology_prior: (batch, n_neighbors)
        """
        batch_size, n_neighbors, gene_dim = neighbor_genes.shape

        # Morphology prior & position bias
        morphology_prior = self.compute_morphology_prior(
            center_image, neighbor_images, center_gene, neighbor_genes
        )
        position_bias = self.compute_position_bias(center_coords, neighbor_coords)

        # Q from clean center, K/V from masked neighbors
        q = self.q_proj(center_gene)                          # (batch, gene_dim)
        k = self.k_proj(neighbor_genes)                       # (batch, n_neighbors, gene_dim)
        v = self.v_proj(neighbor_genes)                       # (batch, n_neighbors, gene_dim)

        # Multi-head reshape
        q = q.view(batch_size, self.num_heads, self.head_dim)                          # (batch, heads, head_dim)
        k = k.view(batch_size, n_neighbors, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, heads, n_neighbors, head_dim)
        v = v.view(batch_size, n_neighbors, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, heads, n_neighbors, head_dim)

        # Attention scores
        attn_scores = torch.einsum("bhd,bhnd->bhn", q, k) * self.scale                  # (batch, heads, n_neighbors)

        # Add morphology prior (broadcast to heads)
        morph_logit = self.morphology_weight * torch.log(morphology_prior + 1e-6)       # (batch, n_neighbors)
        attn_scores = attn_scores + morph_logit.unsqueeze(1)

        # Add position bias
        if self.config.use_relative_position_bias:
            pos_logit = self.position_weight * position_bias
            attn_scores = attn_scores + pos_logit.unsqueeze(1)

        # Softmax over neighbors
        attention_weights = F.softmax(attn_scores, dim=-1)                              # (batch, heads, n_neighbors)
        attention_weights_raw = attention_weights
        attention_weights = self.attention_dropout(attention_weights)

        # Weighted sum of values
        contextual = torch.einsum("bhn,bhnd->bhd", attention_weights, v)                # (batch, heads, head_dim)
        contextual = contextual.reshape(batch_size, gene_dim)                           # (batch, gene_dim)

        # Output projection + residual + LayerNorm
        contextual = self.out_proj(contextual)
        contextual = self.attention_norm(center_gene + contextual)

        # Mean attention across heads for diagnostics
        mean_attention = attention_weights_raw.mean(dim=1)                              # (batch, n_neighbors)

        return contextual, mean_attention, morphology_prior


class Phase1Model(nn.Module):
    """Phase 1: Gene Expression Cross-Attention Model (3000-dim throughout)."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.cross_attention = GeneExpressionCrossAttention(config)

    def forward(
        self,
        center_gene: torch.Tensor,       # (batch, gene_dim) - CLEAN
        neighbor_genes: torch.Tensor,    # (batch, n_neighbors, gene_dim) - MASKED
        center_image: torch.Tensor,      # (batch, image_dim)
        neighbor_images: torch.Tensor,   # (batch, n_neighbors, image_dim)
        center_coords: torch.Tensor,     # (batch, 2)
        neighbor_coords: torch.Tensor,   # (batch, n_neighbors, 2)
    ) -> Phase1Output:
        spot_emb, attn_weights, morph_prior = self.cross_attention(
            center_gene, neighbor_genes, center_image, neighbor_images, center_coords, neighbor_coords
        )
        return Phase1Output(
            spot_embeddings=spot_emb,
            attention_weights=attn_weights,
            morphology_prior=morph_prior,
        )