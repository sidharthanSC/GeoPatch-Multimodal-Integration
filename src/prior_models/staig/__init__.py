"""STAIG graph-contrastive architecture adapted to the repository data contract."""

from .config import StaigConfig
from .model import StaigModel, neighbor_contrastive_loss

__all__ = [
    "StaigConfig",
    "StaigModel",
    "neighbor_contrastive_loss",
]
