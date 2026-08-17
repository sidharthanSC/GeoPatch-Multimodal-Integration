"""MP-MNCA: Cross-Attention GNN Encoder (STAIG-style contrastive).

Replaces STAIG's GCN + binary edge dropping with:
- Cross-attention over k-NN neighbors
- Continuous attention weights: gene_sim + β·image_sim + γ·position_bias
- Structured masking on NEIGHBOR genes (keys/values)
- STAIG symmetric neighbor contrastive loss on output embeddings
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .config import MpMncaConfig


@dataclass(frozen=True)
class MpMncaOutput:
    """Output from MP-MNCA forward pass."""
    embeddings: torch.Tensor          # (n_spots, embed_dim) - final spot representations
    attention_weights: torch.Tensor   # (n_spots, n_neighbors) - mean across heads
    morphology_prior: torch.Tensor    # (n_spots, n_neighbors) - image similarity


class GeneEncoder(nn.Module):
    """Gene encoder: supports pre-trained projection, trainable MLP, or BYOL encoder."""

    def __init__(self, config: MpMncaConfig) -> None:
        super().__init__()
        self.config = config
        self.use_pretrained = getattr(config, "use_pretrained_gene_emb", True)
        self.freeze = getattr(config, "freeze_gene_encoder", True)
        self.use_byol = getattr(config, "use_byol_encoder", False)

        if self.use_byol:
            # BYOL encoder: 3000 -> 3000, then project to gene_embedding_dim (128)
            self.byol_encoder = GeneBYOLEncoder(config)
            self.byol_projection = nn.Linear(config.byol_gene_embedding_dim, config.gene_embedding_dim)
            if self.freeze:
                for p in self.byol_encoder.parameters():
                    p.requires_grad = False
                for p in self.byol_projection.parameters():
                    p.requires_grad = False
        elif self.use_pretrained:
            # Light projection from pre-trained 128-d gene_emb
            self.projection = nn.Sequential(
                nn.Linear(config.gene_embedding_dim, config.gene_embedding_dim),
                nn.LayerNorm(config.gene_embedding_dim),
                nn.GELU(),
                nn.Dropout(config.gene_dropout),
                nn.Linear(config.gene_embedding_dim, config.gene_embedding_dim),
            )
            if self.freeze:
                for p in self.projection.parameters():
                    p.requires_grad = False
        else:
            # Trainable MLP from raw HVG (not recommended for this task)
            dimensions = (config.gene_input_dim, *config.gene_hidden_dims)
            activation = nn.ReLU if config.gene_activation == "relu" else nn.GELU
            self.hidden_blocks = nn.ModuleList(
                nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.LayerNorm(out_dim),
                    activation(),
                    nn.Dropout(config.gene_dropout),
                )
                for in_dim, out_dim in zip(dimensions[:-1], dimensions[1:])
            )
            self.embedding = nn.Linear(config.gene_hidden_dims[-1], config.gene_embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_byol:
            x = self.byol_encoder.forward_encoder(x)  # 3000-d output
            return self.byol_projection(x)  # project to gene_embedding_dim (128)
        if self.use_pretrained:
            return self.projection(x)
        for block in self.hidden_blocks:
            x = block(x)
        return self.embedding(x)


class GeneBYOLEncoder(nn.Module):
    """BYOL-style gene encoder with 3000->3000 projection (no dim reduction).
    
    Uses the same architecture as the gene encoder in src/gene_encoder/model.py:
    - Encoder: 3000 -> 512 -> 512 -> 3000 (configurable hidden dims)
    - BYOL projector: 3000 -> 3000 -> 3000
    - Predictor: 3000 -> 3000 -> 3000
    - EMA target encoder
    
    For MP-MNCA, we use the encoder output (3000-d) as the gene representation.
    """

    def __init__(self, config: MpMncaConfig) -> None:
        super().__init__()
        self.config = config
        
        # Encoder: 3000 -> hidden_dims -> byol_gene_embedding_dim (3000)
        encoder_dims = (config.gene_input_dim, *config.gene_hidden_dims, config.byol_gene_embedding_dim)
        activation = nn.ReLU if config.gene_activation == "relu" else nn.GELU
        
        self.encoder = nn.ModuleList(
            nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                activation(),
                nn.Dropout(config.gene_dropout),
            )
            for in_dim, out_dim in zip(encoder_dims[:-1], encoder_dims[1:])
        )
        
        # BYOL projector (for training, not used at inference)
        self.projector = nn.Sequential(
            nn.Linear(config.byol_gene_embedding_dim, config.byol_projection_hidden_dim),
            nn.LayerNorm(config.byol_projection_hidden_dim),
            nn.GELU(),
            nn.Linear(config.byol_projection_hidden_dim, config.byol_projection_dim),
        )
        
        # Predictor
        self.predictor = nn.Sequential(
            nn.Linear(config.byol_projection_dim, config.byol_projection_hidden_dim),
            nn.LayerNorm(config.byol_projection_hidden_dim),
            nn.GELU(),
            nn.Linear(config.byol_projection_hidden_dim, config.byol_projection_dim),
        )
        
        # EMA target encoder
        self.target_encoder = None
        self._init_target_encoder()
        
    def _init_target_encoder(self):
        """Initialize EMA target encoder as copy of online encoder."""
        import copy
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
            
    @torch.no_grad()
    def update_target_encoder(self, decay: float = None):
        """EMA update of target encoder."""
        if decay is None:
            decay = self.config.byol_ema_decay
        for target_param, online_param in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            target_param.data.mul_(decay).add_(online_param.data, alpha=1 - decay)
            
    def forward_encoder(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder only (returns 3000-d representation)."""
        for block in self.encoder:
            x = block(x)
        return x
    
    def forward_online(self, x: torch.Tensor) -> torch.Tensor:
        """Online forward: encoder -> projector -> predictor."""
        z = self.forward_encoder(x)
        return self.predictor(self.projector(z))
    
    def forward_target(self, x: torch.Tensor) -> torch.Tensor:
        """Target forward: encoder -> projector (no grad)."""
        with torch.no_grad():
            z = self.forward_encoder(x)
            return self.projector(z)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Default forward: return encoder output (3000-d)."""
        return self.forward_encoder(x)
    
    def byol_loss(self, view1: torch.Tensor, view2: torch.Tensor) -> torch.Tensor:
        """BYOL loss: predict target projector output from online predictor."""
        pred1 = self.forward_online(view1)
        pred2 = self.forward_online(view2)
        target1 = self.forward_target(view2)
        target2 = self.forward_target(view1)
        loss1 = F.mse_loss(F.normalize(pred1, dim=-1), F.normalize(target1, dim=-1))
        loss2 = F.mse_loss(F.normalize(pred2, dim=-1), F.normalize(target2, dim=-1))
        return (loss1 + loss2) / 2


class MorphologyPriorCrossAttention(nn.Module):
    """Cross-attention over neighbors with morphology prior."""

    def __init__(self, config: MpMncaConfig) -> None:
        super().__init__()
        self.config = config
        embed_dim = config.gene_embedding_dim
        num_heads = config.num_heads

        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.scale = config.temperature ** -0.5

        # Q, K, V projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

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
        self.attention_norm = nn.LayerNorm(embed_dim)

    def compute_morphology_prior(
        self,
        center_image: torch.Tensor,      # (batch, image_dim)
        neighbor_images: torch.Tensor,   # (batch, n_neighbors, image_dim)
        center_gene_raw: torch.Tensor = None,  # (batch, gene_input_dim) - raw 3000-dim
        neighbor_genes_raw: torch.Tensor = None,  # (batch, n_neighbors, gene_input_dim)
    ) -> torch.Tensor:
        """Cosine similarity mapped to (0, 1]."""
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
            if center_gene_raw is None or neighbor_genes_raw is None:
                raise ValueError("gene_cosine prior requires raw gene expression inputs")
            center_norm = F.normalize(center_gene_raw, dim=-1)
            neighbor_norm = F.normalize(neighbor_genes_raw, dim=-1)
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
        center_gene: torch.Tensor,       # (batch, embed_dim) - CLEAN center
        neighbor_genes: torch.Tensor,    # (batch, n_neighbors, embed_dim) - MASKED neighbors
        center_image: torch.Tensor,      # (batch, image_dim)
        neighbor_images: torch.Tensor,   # (batch, n_neighbors, image_dim)
        center_coords: torch.Tensor,     # (batch, 2)
        neighbor_coords: torch.Tensor,   # (batch, n_neighbors, 2)
        center_gene_raw: torch.Tensor = None,  # (batch, gene_input_dim) - raw for gene_cosine prior
        neighbor_genes_raw: torch.Tensor = None,  # (batch, n_neighbors, gene_input_dim)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            contextual: (batch, embed_dim)
            attention_weights: (batch, n_neighbors)
            morphology_prior: (batch, n_neighbors)
        """
        batch_size, n_neighbors, embed_dim = neighbor_genes.shape

        # Morphology prior & position bias
        morphology_prior = self.compute_morphology_prior(
            center_image, neighbor_images, center_gene_raw, neighbor_genes_raw
        )
        position_bias = self.compute_position_bias(center_coords, neighbor_coords)

        # Q from clean center, K/V from masked neighbors
        q = self.q_proj(center_gene)                          # (batch, embed_dim)
        k = self.k_proj(neighbor_genes)                       # (batch, n_neighbors, embed_dim)
        v = self.v_proj(neighbor_genes)                       # (batch, n_neighbors, embed_dim)

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
        contextual = contextual.reshape(batch_size, embed_dim)                          # (batch, embed_dim)

        # Output projection + residual + LayerNorm
        contextual = self.out_proj(contextual)
        contextual = self.attention_norm(center_gene + contextual)

        # Mean attention across heads for diagnostics
        mean_attention = attention_weights_raw.mean(dim=1)                              # (batch, n_neighbors)

        return contextual, mean_attention, morphology_prior


class MpMncaEncoder(nn.Module):
    """Full MP-MNCA encoder: GeneEncoder + CrossAttention + Context MLP."""

    def __init__(self, config: MpMncaConfig) -> None:
        super().__init__()
        self.config = config
        self.gene_encoder = GeneEncoder(config)
        self.cross_attention = MorphologyPriorCrossAttention(config)

        embed_dim = config.gene_embedding_dim
        self.context_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(config.gene_dropout),
            nn.Linear(embed_dim, embed_dim),
        )
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        center_gene: torch.Tensor,           # (batch, gene_dim) - CLEAN (or raw if gene_cosine)
        neighbor_genes: torch.Tensor,        # (batch, n_neighbors, gene_dim) - MASKED (or raw if gene_cosine)
        center_image: torch.Tensor,          # (batch, image_dim)
        neighbor_images: torch.Tensor,       # (batch, n_neighbors, image_dim)
        center_coords: torch.Tensor,         # (batch, 2)
        neighbor_coords: torch.Tensor,       # (batch, n_neighbors, 2)
    ) -> MpMncaOutput:
        # For gene_cosine prior, we need raw gene expression (3000-dim)
        use_gene_cosine = self.config.morphology_prior_type == "gene_cosine"
        center_gene_raw = center_gene if use_gene_cosine else None
        neighbor_genes_raw = neighbor_genes if use_gene_cosine else None

        # Encode clean center
        center_emb = self.gene_encoder(center_gene)                         # (batch, embed_dim)

        # Encode masked neighbors (vectorized)
        batch_size, n_neighbors, gene_dim = neighbor_genes.shape
        neighbor_flat = neighbor_genes.view(batch_size * n_neighbors, gene_dim)
        neighbor_emb_flat = self.gene_encoder(neighbor_flat)
        neighbor_emb = neighbor_emb_flat.view(batch_size, n_neighbors, -1)   # (batch, n_neighbors, embed_dim)

        # Cross-attention
        contextual, attention_weights, morphology_prior = self.cross_attention(
            center_emb, neighbor_emb, center_image, neighbor_images, 
            center_coords, neighbor_coords,
            center_gene_raw, neighbor_genes_raw
        )

        # Context MLP + residual
        combined = torch.cat([center_emb, contextual], dim=-1)
        contextual = self.context_mlp(combined)
        final_emb = self.final_norm(center_emb + contextual)

        return MpMncaOutput(
            embeddings=final_emb,
            attention_weights=attention_weights,
            morphology_prior=morphology_prior,
        )