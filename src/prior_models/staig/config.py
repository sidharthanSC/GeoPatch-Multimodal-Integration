"""Configuration for the repository-local STAIG reproduction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class StaigConfig:
    """Paper-default 10x Visium settings from the STAIG supplement."""

    n_neighbors: int = 5
    hidden_dim: int = 64
    projection_dim: int = 64
    n_layers: int = 1
    temperature: float = 10.0
    epochs: int = 400
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    feature_mask_rate_1: float = 0.1
    feature_mask_rate_2: float = 0.1
    image_pseudo_clusters: int = 40
    image_pca_dim: int = 16
    refinement_neighbors: int = 15
    seed: int = 0
    dtype: str = "float32"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
