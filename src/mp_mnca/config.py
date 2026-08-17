"""Configuration for MP-MNCA (Morphology-Prior Masked Neighbour Cross-Attention)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class MpMncaConfig:
    """Configuration for MP-MNCA training and model architecture."""

    # Gene expression dimension (3000 HVGs)
    gene_dim: int = 3000

    # Cross-attention architecture (Phase 1)
    num_heads: int = 8
    attention_dropout: float = 0.1
    use_relative_position_bias: bool = True
    relative_position_dim: int = 16

    # Morphology prior
    image_embedding_dim: int = 128
    image_pca_dim: int = 16
    morphology_prior_type: Literal["cosine", "rbf", "gene_cosine"] = "cosine"
    morphology_prior_weight: float = 1.0  # beta
    position_bias_weight: float = 0.5  # gamma
    learn_morphology_weight: bool = True
    learn_position_weight: bool = True

    # Phase 1: Gene Expression Cross-Attention (3000-dim throughout)
    phase1_hidden_dim: int = 3000  # Q,K,V projection dim (can be 3000 or smaller)

    # Phase 2: BYOL Encoder 3000->3000
    byol_hidden_dims: tuple[int, ...] = (3000, 3000)  # encoder: 3000 -> 3000 -> 3000
    byol_projection_dim: int = 3000
    byol_projection_hidden_dim: int = 3000
    byol_predictor_hidden_dim: int = 3000
    byol_ema_decay: float = 0.996
    byol_mask_rate: float = 0.3
    byol_noise_std: float = 0.05
    gene_activation: str = "gelu"
    gene_dropout: float = 0.1

    # Training
    epochs: int = 200
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    temperature: float = 10.0
    dtype: str = "float32"
    mask_rate: float = 0.1  # feature masking rate for contrastive views

    # STAIG-specific (for contrastive baseline comparison)
    image_pseudo_clusters: int = 40
    refinement_neighbors: int = 15
    n_clusters: int | None = None

    # Reproducibility
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)