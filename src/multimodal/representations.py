"""Reusable spot-representation constructions for common downstream evaluation."""

from __future__ import annotations

from typing import Mapping

import numpy as np


def l2_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Cannot normalize a representation containing zero vectors")
    return values / norms


def normalized_mean(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    if first.shape != second.shape:
        raise ValueError(f"Representation shapes differ: {first.shape} versus {second.shape}")
    return l2_normalize(l2_normalize(first) + l2_normalize(second))


def weighted_normalized_mean(
    gene: np.ndarray, image: np.ndarray, gene_weight: float
) -> np.ndarray:
    if not 0.0 <= gene_weight <= 1.0:
        raise ValueError("gene_weight must lie in [0, 1]")
    if gene.shape != image.shape:
        raise ValueError(f"Representation shapes differ: {gene.shape} versus {image.shape}")
    return l2_normalize(
        gene_weight * l2_normalize(gene)
        + (1.0 - gene_weight) * l2_normalize(image)
    )


def concatenate(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    if first.shape[0] != second.shape[0]:
        raise ValueError("Representations must have the same number of spots")
    return np.concatenate([l2_normalize(first), l2_normalize(second)], axis=1)


def existing_embedding_representations(
    embeddings: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Construct R2-R5 and R7-R10 from the audited checkpoint embedding arrays."""
    required = {"gene_emb", "img_emb", "gene_emb_cm_img", "img_emb_cm"}
    missing = sorted(required - embeddings.keys())
    if missing:
        raise KeyError(f"Missing source embedding keys: {missing}")

    gene = embeddings["gene_emb"]
    image = embeddings["img_emb"]
    aligned_gene = embeddings["gene_emb_cm_img"]
    aligned_image = embeddings["img_emb_cm"]
    representations = {
        "R2_gene_emb": l2_normalize(gene),
        "R3_img_emb": l2_normalize(image),
        "R4_unaligned_bisector": normalized_mean(gene, image),
        "R5_unaligned_concat": concatenate(gene, image),
        "R7_aligned_gene": l2_normalize(aligned_gene),
        "R8_aligned_image": l2_normalize(aligned_image),
        "R9_aligned_bisector": normalized_mean(aligned_gene, aligned_image),
        "R10_aligned_concat": concatenate(aligned_gene, aligned_image),
    }
    for gene_weight in (0.0, 0.25, 0.5, 0.75, 1.0):
        label = str(gene_weight).replace(".", "p")
        representations[f"R13_unaligned_weighted_gene_{label}"] = weighted_normalized_mean(
            gene, image, gene_weight
        )
        representations[f"R13_aligned_weighted_gene_{label}"] = weighted_normalized_mean(
            aligned_gene, aligned_image, gene_weight
        )
    return representations
