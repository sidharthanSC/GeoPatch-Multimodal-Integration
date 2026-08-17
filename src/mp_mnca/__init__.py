"""MP-MNCA: Morphology-Prior Masked Neighbour Cross-Attention.

A local graph transformer for spatial transcriptomics that replaces STAIG's
binary edge-dropping with continuous, morphology-prior, local cross-attention
and structured gene masking.
"""

from .config import MpMncaConfig
from .data import MpMncaSectionData, prepare_all_sections, prepare_section
from .evaluate import (
    attention_diagnostics,
    clustering_metrics,
    embedding_collapse_diagnostics,
    refine_labels,
    tied_gmm,
)
from .masking import (
    apply_mask,
    contiguous_block_mask,
    create_mask_tensor,
    gene_module_mask,
    load_gene_modules,
    neighbourhood_mask,
    scalar_mask,
)
from .model import (
    GeneEncoder,
    MorphologyPriorCrossAttention,
    MpMncaModel,
    MpMncaOutput,
    covariance_regularization,
    variance_regularization,
)
from .train import MpMncaResult, fit_mp_mnca, run_dlpfc

__all__ = [
    # Config
    "MpMncaConfig",
    # Data
    "MpMncaSectionData",
    "prepare_all_sections",
    "prepare_section",
    # Evaluate
    "attention_diagnostics",
    "clustering_metrics",
    "embedding_collapse_diagnostics",
    "refine_labels",
    "tied_gmm",
    # Masking
    "apply_mask",
    "contiguous_block_mask",
    "create_mask_tensor",
    "gene_module_mask",
    "load_gene_modules",
    "neighbourhood_mask",
    "scalar_mask",
    # Model
    "GeneEncoder",
    "MorphologyPriorCrossAttention",
    "MpMncaModel",
    "MpMncaOutput",
    "covariance_regularization",
    "latent_prediction_loss",
    "variance_regularization",
    # Train
    "MpMncaResult",
    "fit_mp_mnca",
    "run_dlpfc",
]

__version__ = "0.1.0"