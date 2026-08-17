"""Configuration for the repository-local GraphST reproduction.

GraphST (Long et al., 2023, Nature Communications) trains a GNN encoder with a
DGI-style graph-contrastive objective plus gene-expression reconstruction.
The defaults below match ``GraphST/GraphST.py`` (10X Visium branch): 600
epochs, 3,000-dim input, 64-dim output, learning rate 1e-3, alpha=10, beta=1,
random seed 41.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class GraphStConfig:
    """Paper-default 10x Visium settings from the official GraphST code."""

    n_neighbors: int = 3
    epochs: int = 600
    dim_input: int = 3000
    dim_output: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    alpha: float = 10.0
    beta: float = 1.0
    dropout: float = 0.0
    seed: int = 41
    refinement_neighbors: int = 15
    dtype: str = "float32"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)