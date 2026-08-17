"""Repository-local SpaGCN reproduction for DLPFC domain identification."""

from .config import SpaGcnConfig
from .data import SpaGcnSectionData, prepare_section
from .model import GraphConvolution, SimpleGcDec
from .train import fit_spagcn, run_dlpfc

__all__ = [
    "SpaGcnConfig",
    "SpaGcnSectionData",
    "prepare_section",
    "GraphConvolution",
    "SimpleGcDec",
    "fit_spagcn",
    "run_dlpfc",
]