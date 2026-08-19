"""SPARC concept-aligned sparse autoencoder for cross-modal gene/image alignment.

Implements Nasiri-Sarvi et al., "SPARC: Concept-Aligned Sparse Autoencoders for
Cross-Model and Cross-Modal Interpretability" (TMLR 3/2026) over the DLPFC spatial
transcriptomics streams used by :mod:`src.mp_mnca`, and evaluates the result through
the identical clustering protocol (PCA -> tied GMM -> spatial refinement -> ARI/NMI).
"""

from .config import SparcConfig
from .model import SparcModel, SparcOutput, global_topk, nmse

__all__ = ["SparcConfig", "SparcModel", "SparcOutput", "global_topk", "nmse"]
