"""MP-MNCA model architecture.

Implements:
1. Gene encoder (MLP with LayerNorm, GELU, Dropout)
2. Morphology-prior local cross-attention over spatial neighbors
3. EMA target encoder for latent prediction
4. Anti-collapse regularization (variance + covariance)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module

from .config import MpMncaConfig


@dataclass(frozen=True)
class MpMncaOutput:
    """Output from MP-MNCA forward pass."""
    online_embedding: torch.Tensor  # (batch, embed_dim) - online encoder output
    target_embedding: torch.Tensor  # (batch, embed_dim) - EMA target encoder output (stop-grad)
    contextual_embedding: torch.Tensor  # (batch, embed_dim) - after cross-attention
    attention_weights: torch.Tensor  # (batch, n_neighbors) - attention weights
    morphology_prior: torch.Tensor  # (batch, n_neighbors) - image similarity prior


class GeneEncoder(Module):
    """Gene encoder: either a deep MLP (for training from scratch) or a simple projection (for pre-trained embeddings)."""

    def __init__(self, config: MpMncaConfig) -> None:
        super().__init__()
        self.config = config
        self.use_pretrained = getattr(config, "use_pretrained_gene_emb", False)
        self.freeze_encoder = getattr(config, "freeze_gene_encoder", False)

        if self.use_pretrained:
            # Simple projection from pre-trained gene_emb (128) to embedding_dim (128)
            self.projection = nn.Sequential(
                nn.Linear(config.gene_embedding_dim, config.gene_embedding_dim),
                nn.LayerNorm(config.gene_embedding_dim),
                nn.GELU(),
                nn.Dropout(config.gene_dropout),
                nn.Linear(config.gene_embedding_dim, config.gene_embedding_dim),
            )
            if self.freeze_encoder:
                for param in self.projection.parameters():
                    param.requires_grad = False
        else:
            # Deep MLP from raw HVG expression (3000) to embedding_dim (128)
            dimensions = (config.gene_input_dim, *config.gene_hidden_dims)

            if config.gene_activation not in {"relu", "gelu"}:
                raise ValueError("gene_activation must be 'relu' or 'gelu'")
            activation = nn.ReLU if config.gene_activation == "relu" else nn.GELU

            self.hidden_blocks = nn.ModuleList(
                nn.Sequential(
                    nn.Linear(input_dim, output_dim),
                    nn.LayerNorm(output_dim),
                    activation(),
                    nn.Dropout(config.gene_dropout),
                )
                for input_dim, output_dim in zip(dimensions[:-1], dimensions[1:])
            )
            self.embedding = nn.Linear(config.gene_hidden_dims[-1], config.gene_embedding_dim)
            if self.freeze_encoder:
                for param in self.parameters():
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_pretrained:
            return self.projection(x)
        for block in self.hidden_blocks:
            x = block(x)
        return self.embedding(x)


class MorphologyPriorCrossAttention(Module):
    """Morphology-prior local cross-attention over spatial neighbors.

    For each center spot i, uses its representation as query and masked neighbor
    representations as keys/values. Attention logits combine:
    - Learned molecular compatibility (query-key dot product)
    - Morphology prior from image embeddings (cosine similarity or RBF)
    - Optional relative position bias
    """

    def __init__(self, config: MpMncaConfig) -> None:
        super().__init__()
        self.config = config
        embed_dim = config.gene_embedding_dim
        num_heads = config.num_heads

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.scale = config.temperature**-0.5  # 1/sqrt(d_k) * temperature scaling

        # Query, Key, Value projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Morphology prior weight (beta) - learnable scalar
        if config.learn_morphology_weight:
            self.morphology_weight = nn.Parameter(torch.tensor(config.morphology_prior_weight))
        else:
            self.register_buffer("morphology_weight", torch.tensor(config.morphology_prior_weight))

        # Position bias weight (gamma) - learnable scalar
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
        self.attention_norm = nn.LayerNorm(embed_dim)

    def compute_morphology_prior(
        self,
        center_image: torch.Tensor,  # (batch, image_dim)
        neighbor_images: torch.Tensor,  # (batch, n_neighbors, image_dim)
    ) -> torch.Tensor:
        """Compute morphology prior from image embeddings.

        Args:
            center_image: Image embedding for center spots.
            neighbor_images: Image embeddings for neighbor spots.

        Returns:
            Prior similarity of shape (batch, n_neighbors), non-negative.
        """
        if self.config.morphology_prior_type == "cosine":
            # Cosine similarity mapped to [0, 1]
            center_norm = F.normalize(center_image, dim=-1)  # (batch, dim)
            neighbor_norm = F.normalize(neighbor_images, dim=-1)  # (batch, n_neighbors, dim)
            sim = torch.einsum("bd,bnd->bn", center_norm, neighbor_norm)  # (batch, n_neighbors)
            prior = (sim + 1) / 2  # Map [-1, 1] -> [0, 1]
        elif self.config.morphology_prior_type == "rbf":
            # RBF kernel on normalized distance
            center_norm = F.normalize(center_image, dim=-1)
            neighbor_norm = F.normalize(neighbor_images, dim=-1)
            dist = torch.norm(center_norm.unsqueeze(1) - neighbor_norm, dim=-1)  # (batch, n_neighbors)
            prior = torch.exp(-dist**2 / 2)  # Gaussian RBF
        else:
            raise ValueError(f"Unknown morphology prior type: {self.config.morphology_prior_type}")

        return prior.clamp(min=1e-6)  # Avoid log(0)

    def compute_position_bias(
        self,
        center_coords: torch.Tensor,  # (batch, 2)
        neighbor_coords: torch.Tensor,  # (batch, n_neighbors, 2)
    ) -> torch.Tensor:
        """Compute relative position bias."""
        if not self.config.use_relative_position_bias:
            return torch.zeros(
                center_coords.shape[0], neighbor_coords.shape[1],
                device=center_coords.device, dtype=center_coords.dtype
            )
        rel_pos = neighbor_coords - center_coords.unsqueeze(1)  # (batch, n_neighbors, 2)
        bias = self.position_bias_mlp(rel_pos).squeeze(-1)  # (batch, n_neighbors)
        return bias

    def forward(
        self,
        center_gene: torch.Tensor,  # (batch, embed_dim)
        neighbor_genes: torch.Tensor,  # (batch, n_neighbors, embed_dim)
        center_image: torch.Tensor,  # (batch, image_dim)
        neighbor_images: torch.Tensor,  # (batch, n_neighbors, image_dim)
        center_coords: torch.Tensor,  # (batch, 2)
        neighbor_coords: torch.Tensor,  # (batch, n_neighbors, 2)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute cross-attention with morphology prior.

        Returns:
            contextual: Contextual embedding (batch, embed_dim)
            attention_weights: Softmax attention weights (batch, n_neighbors)
            morphology_prior: Image similarity prior (batch, n_neighbors)
        """
        batch_size, n_neighbors, embed_dim = neighbor_genes.shape

        # Compute morphology prior
        morphology_prior = self.compute_morphology_prior(center_image, neighbor_images)  # (batch, n_neighbors)

        # Compute position bias
        position_bias = self.compute_position_bias(center_coords, neighbor_coords)  # (batch, n_neighbors)

        # Project to Q, K, V
        q = self.q_proj(center_gene)  # (batch, embed_dim)
        k = self.k_proj(neighbor_genes)  # (batch, n_neighbors, embed_dim)
        v = self.v_proj(neighbor_genes)  # (batch, n_neighbors, embed_dim)

        # Reshape for multi-head attention
        q = q.view(batch_size, self.num_heads, self.head_dim)  # (batch, heads, head_dim)
        k = k.view(batch_size, n_neighbors, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, heads, n_neighbors, head_dim)
        v = v.view(batch_size, n_neighbors, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, heads, n_neighbors, head_dim)

        # Attention scores: Q @ K^T / sqrt(d_k)
        attn_scores = torch.einsum("bhd,bhnd->bhn", q, k) * self.scale  # (batch, heads, n_neighbors)

        # Add morphology prior (broadcast to heads)
        morph_logit = self.morphology_weight * torch.log(morphology_prior + 1e-6)  # (batch, n_neighbors)
        attn_scores = attn_scores + morph_logit.unsqueeze(1)  # (batch, heads, n_neighbors)

        # Add position bias
        if self.config.use_relative_position_bias:
            pos_logit = self.position_weight * position_bias  # (batch, n_neighbors)
            attn_scores = attn_scores + pos_logit.unsqueeze(1)

        # Softmax over neighbors
        attention_weights = F.softmax(attn_scores, dim=-1)  # (batch, heads, n_neighbors)
        # Store pre-dropout weights for diagnostics
        attention_weights_raw = attention_weights
        attention_weights = self.attention_dropout(attention_weights)

        # Weighted sum of values
        contextual = torch.einsum("bhn,bhnd->bhd", attention_weights, v)  # (batch, heads, head_dim)
        contextual = contextual.reshape(batch_size, embed_dim)  # (batch, embed_dim)

        # Output projection
        contextual = self.out_proj(contextual)

        # Residual connection + LayerNorm
        contextual = self.attention_norm(center_gene + contextual)

        # Average attention weights across heads for diagnostics (use raw weights)
        mean_attention = attention_weights_raw.mean(dim=1)  # (batch, n_neighbors)

        return contextual, mean_attention, morphology_prior


class MpMncaModel(Module):
    """Full MP-MNCA model with online and EMA target encoders."""

    def __init__(self, config: MpMncaConfig) -> None:
        super().__init__()
        self.config = config

        # Online encoder
        self.online_encoder = GeneEncoder(config)

        # Cross-attention module
        self.cross_attention = MorphologyPriorCrossAttention(config)

        # MLP for final contextual representation
        embed_dim = config.gene_embedding_dim
        self.context_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(config.gene_dropout),
            nn.Linear(embed_dim, embed_dim),
        )

        # Final LayerNorm
        self.final_norm = nn.LayerNorm(embed_dim)

        # EMA target encoder (initialized as copy of online)
        if config.use_ema_target:
            self.target_encoder = GeneEncoder(config)
            self.target_encoder.load_state_dict(self.online_encoder.state_dict())
            # Freeze target encoder parameters
            for param in self.target_encoder.parameters():
                param.requires_grad = False
        else:
            self.target_encoder = None

        # Predictor for latent prediction (BYOL-style)
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        """Update target encoder with EMA of online encoder."""
        if self.target_encoder is None:
            return
        decay = self.config.ema_decay
        for target_param, online_param in zip(
            self.target_encoder.parameters(), self.online_encoder.parameters()
        ):
            target_param.data.mul_(decay).add_(online_param.data, alpha=1 - decay)

    def forward_online(
        self,
        center_gene: torch.Tensor,  # (batch, gene_dim) - masked
        neighbor_genes: torch.Tensor,  # (batch, n_neighbors, gene_dim) - masked
        center_image: torch.Tensor,  # (batch, image_dim)
        neighbor_images: torch.Tensor,  # (batch, n_neighbors, image_dim)
        center_coords: torch.Tensor,  # (batch, 2)
        neighbor_coords: torch.Tensor,  # (batch, n_neighbors, 2)
    ) -> MpMncaOutput:
        """Forward pass through online encoder + cross-attention."""
        # Online encoder for center spot
        center_emb = self.online_encoder(center_gene)  # (batch, embed_dim)

        # Online encoder for neighbors (vectorized)
        batch_size, n_neighbors, gene_dim = neighbor_genes.shape
        neighbor_flat = neighbor_genes.view(batch_size * n_neighbors, gene_dim)
        neighbor_emb_flat = self.online_encoder(neighbor_flat)
        neighbor_emb = neighbor_emb_flat.view(batch_size, n_neighbors, -1)  # (batch, n_neighbors, embed_dim)

        # Cross-attention
        contextual, attention_weights, morphology_prior = self.cross_attention(
            center_emb, neighbor_emb, center_image, neighbor_images, center_coords, neighbor_coords
        )

        # Context MLP with residual
        combined = torch.cat([center_emb, contextual], dim=-1)  # (batch, 2*embed_dim)
        contextual = self.context_mlp(combined)
        contextual = self.final_norm(center_emb + contextual)

        # Target embedding (stop-gradient)
        if self.target_encoder is not None:
            with torch.no_grad():
                target_emb = self.target_encoder(center_gene)  # Use same (masked) input for target
        else:
            target_emb = center_emb.detach()

        return MpMncaOutput(
            online_embedding=center_emb,
            target_embedding=target_emb,
            contextual_embedding=contextual,
            attention_weights=attention_weights,
            morphology_prior=morphology_prior,
        )

    def forward_target(
        self,
        clean_center_gene: torch.Tensor,  # (batch, gene_dim) - clean/unmasked
    ) -> torch.Tensor:
        """Forward pass through target encoder (clean input)."""
        if self.target_encoder is None:
            return self.online_encoder(clean_center_gene).detach()
        with torch.no_grad():
            return self.target_encoder(clean_center_gene)

    def predict(self, contextual: torch.Tensor) -> torch.Tensor:
        """Predict target latent from contextual embedding."""
        return self.predictor(contextual)


def variance_regularization(embeddings: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    """VICReg-style variance term: penalize dimensions with std < gamma."""
    std = torch.sqrt(embeddings.var(dim=0) + eps)
    return torch.relu(gamma - std).mean()


def covariance_regularization(
    embeddings: torch.Tensor,
    max_dimensions: int = 256,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Mean squared off-diagonal correlation."""
    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        return embeddings.new_zeros(())
    if max_dimensions <= 1:
        return embeddings.new_zeros(())
    if embeddings.shape[1] > max_dimensions:
        indices = torch.linspace(
            0, embeddings.shape[1] - 1, max_dimensions, device=embeddings.device
        ).long()
        embeddings = embeddings[:, indices]
    embeddings = embeddings.float()
    centered = embeddings - embeddings.mean(dim=0)
    standardized = centered / torch.sqrt(centered.var(dim=0, unbiased=False) + eps)
    correlation = standardized.T @ standardized / standardized.shape[0]
    off_diagonal = correlation - torch.diag_embed(correlation.diagonal())
    return off_diagonal.square().sum() / (
        correlation.shape[0] * max(correlation.shape[0] - 1, 1)
    )


def contrastive_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    edge_index: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Symmetric neighbor contrastive loss WITHOUT pseudo-label negative masking.

    Positives: same-spot diagonal (inter.diag), spatial kNN neighbors
    (intra_positive + inter_positive). Negatives: every other spot (no mask).
    No labels of any kind are used.
    """
    first = _directional_loss_no_mask(z1, z2, edge_index, temperature)
    second = _directional_loss_no_mask(z2, z1, edge_index, temperature)
    return 0.5 * (first + second).mean()


def _directional_loss_no_mask(
    query: torch.Tensor,
    other: torch.Tensor,
    edge_index: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    query = F.normalize(query, dim=1)
    other = F.normalize(other, dim=1)
    intra = torch.exp((query @ query.T) / temperature)
    inter = torch.exp((query @ other.T) / temperature)

    src, dst = edge_index
    intra_positive = torch.zeros(query.shape[0], dtype=query.dtype, device=query.device)
    inter_positive = torch.zeros_like(intra_positive)
    intra_positive.scatter_add_(0, src, intra[src, dst])
    inter_positive.scatter_add_(0, src, inter[src, dst])
    neighbor_count = torch.zeros_like(intra_positive)
    neighbor_count.scatter_add_(0, src, torch.ones_like(src, dtype=query.dtype))

    numerator = inter.diag() + intra_positive + inter_positive
    denominator = intra.sum(1) + inter.sum(1) - intra.diag()
    ratio = numerator / (denominator.clamp_min(torch.finfo(query.dtype).tiny))
    ratio = ratio / (2 * neighbor_count + 1).clamp_min(1)
    return -torch.log(ratio.clamp_min(torch.finfo(query.dtype).tiny))


def contrastive_loss_masked(
    z1: torch.Tensor,
    z2: torch.Tensor,
    edge_index: torch.Tensor,
    pseudo_labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Symmetric neighbor contrastive loss with a gene-cluster negative mask.

    Same positives as ``contrastive_loss`` (same-spot diagonal + spatial kNN
    neighbors), but spots sharing a gene-KMeans pseudo-label are excluded from
    the negatives (STAIG-style debiasing). No ground truth is used.
    """
    first = _directional_loss_masked(z1, z2, edge_index, pseudo_labels, temperature)
    second = _directional_loss_masked(z2, z1, edge_index, pseudo_labels, temperature)
    return 0.5 * (first + second).mean()


def _directional_loss_masked(
    query: torch.Tensor,
    other: torch.Tensor,
    edge_index: torch.Tensor,
    pseudo_labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    query = F.normalize(query, dim=1)
    other = F.normalize(other, dim=1)
    intra = torch.exp((query @ query.T) / temperature)
    inter = torch.exp((query @ other.T) / temperature)
    negative_mask = pseudo_labels[:, None] != pseudo_labels[None, :]

    src, dst = edge_index
    intra_positive = torch.zeros(query.shape[0], dtype=query.dtype, device=query.device)
    inter_positive = torch.zeros_like(intra_positive)
    intra_positive.scatter_add_(0, src, intra[src, dst])
    inter_positive.scatter_add_(0, src, inter[src, dst])
    neighbor_count = torch.zeros_like(intra_positive)
    neighbor_count.scatter_add_(0, src, torch.ones_like(src, dtype=query.dtype))

    numerator = inter.diag() + intra_positive + inter_positive
    denominator = (
        (intra * negative_mask).sum(1)
        + (inter * negative_mask).sum(1)
        - intra.diag()
    )
    ratio = numerator / (denominator.clamp_min(torch.finfo(query.dtype).tiny))
    ratio = ratio / (2 * neighbor_count + 1).clamp_min(1)
    return -torch.log(ratio.clamp_min(torch.finfo(query.dtype).tiny))


class MpMncaContrastiveModel(Module):
    """MP-MNCA with contrastive loss (STAIG-style) for fair comparison."""

    def __init__(self, config: MpMncaConfig) -> None:
        super().__init__()
        self.config = config

        # Online encoder
        self.online_encoder = GeneEncoder(config)
        self.cross_attention = MorphologyPriorCrossAttention(config)

        # MLP for final contextual representation
        embed_dim = config.gene_embedding_dim
        self.context_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(config.gene_dropout),
            nn.Linear(embed_dim, embed_dim),
        )
        self.final_norm = nn.LayerNorm(embed_dim)

        # Projector for contrastive loss (like STAIG)
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, config.projection_dim if hasattr(config, 'projection_dim') else 64),
            nn.ELU(),
            nn.Linear(config.projection_dim if hasattr(config, 'projection_dim') else 64, embed_dim),
        )

    def forward(
        self,
        center_gene: torch.Tensor,
        neighbor_genes: torch.Tensor,
        center_image: torch.Tensor,
        neighbor_images: torch.Tensor,
        center_coords: torch.Tensor,
        neighbor_coords: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass returning contextual embeddings."""
        batch_size, n_neighbors, _ = neighbor_genes.shape

        # Encode center and neighbors
        center_emb = self.online_encoder(center_gene)
        neighbor_flat = neighbor_genes.view(batch_size * n_neighbors, -1)
        neighbor_emb_flat = self.online_encoder(neighbor_flat)
        neighbor_emb = neighbor_emb_flat.view(batch_size, n_neighbors, -1)

        # Cross-attention
        contextual, _, _ = self.cross_attention(
            center_emb, neighbor_emb, center_image, neighbor_images, center_coords, neighbor_coords
        )

        # Context MLP
        combined = torch.cat([center_emb, contextual], dim=-1)
        contextual = self.context_mlp(combined)
        contextual = self.final_norm(center_emb + contextual)

        return contextual

    def project(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Project embeddings for contrastive loss."""
        return self.projector(embeddings)