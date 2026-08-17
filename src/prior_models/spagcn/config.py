"""Configuration for the repository-local SpaGCN reproduction.

SpaGCN (Hu et al., 2021, Nature Methods) learns a weighted spatial graph and
fits a single-layer GCN plus deep embedded clustering (DEC) head.  The
parameters below follow the official DLPFC tutorial for section ``151673``
(lr=0.05, max_epochs=200, tol=5e-3, 50 PCs, alpha=0.2).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SpaGcnConfig:
    """Paper-default 10x Visium settings from the official SpaGCN tutorial."""

    num_pcs: int = 50
    learning_rate: float = 0.05
    max_epochs: int = 200
    update_interval: int = 3
    weight_decay: float = 5e-4
    optimizer: str = "admin"  # 'sgd' or 'admin' (Adam)
    init_spa: bool = True
    n_clusters: int | None = None  # label-count-informed when None
    tol: float = 5e-3
    refinement_neighbors: int = 15
    seed: int = 100  # official tutorial sets seeds to 100
    dtype: str = "float32"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)