"""Build the R1-R11 representation ablations from ``MODEL_BENCHMARK_REPORT.md``.

Every representation here is derived from data already sitting in
``checkpoints/dlpfc.pkl`` -- no BYOL/InfoNCE retraining. R12 (shuffled-pair InfoNCE
control), R13 (learned scalar-weighted sum), and R14 (``proj_emb`` variants) are out of
scope for this run (see ``MODEL_BENCHMARK_RESULTS.md``'s scope note) since the
"Recommended Execution Order" steps 1-5 only call for R1-R11.

Each ``build_*`` function operates on a single section's arrays (see
``src/benchmark/data.py``'s ``SectionData``) -- per the report's Table A protocol,
representations are produced "for each of the 12 DLPFC sections" independently, so PCA
steps (R1, R6, R11) are fit per-section, never pooling information across sections.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.decomposition import PCA

from src.benchmark.data import SectionData
from src.multimodal.bisector import bisector_embedding

REPRESENTATION_IDS = ("R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11")

_DESCRIPTIONS = {
    "R1": "Joint-HVG expression PCA, 128-d (classical gene-only baseline)",
    "R2": "gene_emb (effect of gene BYOL)",
    "R3": "img_emb (external image-BYOL baseline)",
    "R4": "Unaligned normalized mean of gene_emb and img_emb (fusion without InfoNCE)",
    "R5": "Unaligned concatenation, 256-d (information-preserving fusion control)",
    "R6": "Unaligned concatenation reduced to 128-d via per-section PCA (dimension-matched control)",
    "R7": "gene_emb_cm_img alone (gene-side effect of alignment)",
    "R8": "img_emb_cm alone (image-side effect of alignment)",
    "R9": "Aligned normalized mean/bisector (central hypothesis)",
    "R10": "Aligned concatenation, 256-d (tests whether the mean discards useful disagreement)",
    "R11": "Aligned concatenation reduced to 128-d via per-section PCA (dimension-matched fusion)",
}


def _unit_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def _unaligned_mean(gene: np.ndarray, image: np.ndarray) -> np.ndarray:
    return _unit_normalize(_unit_normalize(gene) + _unit_normalize(image))


def build_section_representations(section: SectionData, pca_seed: int = 0) -> Dict[str, np.ndarray]:
    """Return ``{representation_id: (n_spots, dim) array}`` for one section."""
    gene = section.embeddings["gene_emb"]
    image = section.embeddings["img_emb"]
    gene_cm = section.embeddings["gene_emb_cm_img"]
    image_cm = section.embeddings["img_emb_cm"]

    n_pca_components = min(128, section.hvg_expression.shape[0] - 1, section.hvg_expression.shape[1])
    r1 = PCA(n_components=n_pca_components, random_state=pca_seed).fit_transform(section.hvg_expression)

    unaligned_concat = np.concatenate([gene, image], axis=1)
    aligned_concat = np.concatenate([gene_cm, image_cm], axis=1)

    n_concat_pca = min(128, unaligned_concat.shape[0] - 1)

    representations = {
        "R1": r1.astype(np.float32),
        "R2": gene,
        "R3": image,
        "R4": _unaligned_mean(gene, image).astype(np.float32),
        "R5": unaligned_concat,
        "R6": PCA(n_components=n_concat_pca, random_state=pca_seed).fit_transform(unaligned_concat).astype(np.float32),
        "R7": gene_cm,
        "R8": image_cm,
        "R9": bisector_embedding(image_cm, gene_cm).astype(np.float32),
        "R10": aligned_concat,
        "R11": PCA(n_components=n_concat_pca, random_state=pca_seed).fit_transform(aligned_concat).astype(np.float32),
    }
    return representations
