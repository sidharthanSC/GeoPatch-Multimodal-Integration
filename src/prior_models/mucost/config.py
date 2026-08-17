"""Configuration for the repository-local MuCoST reproduction.

MuCoST (Zhang et al., 2024, Briefings in Bioinformatics) learns a shared GCN
autoencoder across a spatial-radius graph and a gene co-expression k-NN graph,
and trains a three-view InfoNCE objective (spatial, masked-feature, shuffled)
plus expression reconstruction.  The defaults below match ``MuCoST/config.py``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class MuCostConfig:
    """Paper-default settings from the official MuCoST repository."""

    seed: int = 2023
    latent_dim: int = 50
    epochs: int = 1000
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    log_step: int = 10
    drop_feat_p: float = 0.2
    radius: float = 150.0
    rknn: int = 6
    knn: int = 6
    temperature: float = 0.05
    n_refine: int = 25
    refinement_neighbors: int = 15
    contrastive_weight: float = 0.2
    dtype: str = "float32"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)