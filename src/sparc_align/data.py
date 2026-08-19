"""Stream preparation for SPARC on DLPFC sections.

The gene stream is byte-identical to what ``src.mp_mnca.train_phase1`` consumes:
``prepare_gene_expression(adata, use_feat_obsm=True)`` -> the 3000-d ``obsm['feat']``
HVG matrix. The image stream is the raw 128-d ``obsm['img_emb']`` rather than the
16-d PCA that MP-MNCA feeds its attention prior -- 16 dimensions cannot support a
dictionary of hundreds of latents. The 16-d PCA is still computed and carried
alongside, because the optional downstream cross-attention stage and the STAIG
pseudo-labels both need it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from anndata import AnnData
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.mp_mnca.data import prepare_gene_expression

from .config import SparcConfig


@dataclass(frozen=True)
class SparcSectionData:
    """Prepared streams and metadata for one DLPFC section."""

    section_id: str
    gene: np.ndarray  # (n_spots, 3000) standardized HVG expression
    image: np.ndarray  # (n_spots, 128) standardized img_emb
    gene_raw: np.ndarray  # (n_spots, 3000) unstandardized, for the attention stage
    image_pca: np.ndarray  # (n_spots, 16) morphology-prior features
    coordinates: np.ndarray  # (n_spots, 2)
    labels: np.ndarray  # (n_spots,) ground_truth, EVALUATION ONLY
    barcodes: np.ndarray  # (n_spots,)
    neighbor_indices: np.ndarray  # (n_spots, n_neighbors)
    pseudo_labels: np.ndarray  # (n_spots,) KMeans on image PCA -- unsupervised
    # (n_spots, n_neighbors) morphology prior s_ij = (cos(img_i, img_j) + 1)/2 over
    # the image PCA features, the same quantity MP-MNCA puts on its attention
    # logits. Static (image features are frozen), so precomputed once per section.
    neighbor_image_similarity: np.ndarray

    @property
    def stream_dims(self) -> dict[str, int]:
        return {"gene": self.gene.shape[1], "image": self.image.shape[1]}

    @property
    def streams(self) -> dict[str, np.ndarray]:
        return {"gene": self.gene, "image": self.image}


def prepare_section(
    adata: AnnData,
    section_id: str,
    config: SparcConfig,
) -> SparcSectionData:
    """Build both SPARC streams plus everything the downstream stages need."""
    gene_raw = prepare_gene_expression(
        adata, hvg_key="highly_variable", expected_hvg_count=config.gene_dim, use_feat_obsm=True
    )
    image_raw = np.asarray(adata.obsm["img_emb"], dtype=np.float32)
    if not np.isfinite(image_raw).all():
        raise ValueError(f"{section_id}: img_emb contains non-finite values")
    if image_raw.shape[1] != config.image_dim:
        raise ValueError(
            f"{section_id}: expected {config.image_dim}-d img_emb, found {image_raw.shape[1]}"
        )

    # Per-feature standardization. Mandatory here: img_emb spans roughly
    # [-131, 189] while feat spans [0, 10], so without it the Global TopK support
    # would be chosen by image-logit magnitude alone and gene would never vote.
    if config.standardize_streams:
        gene = StandardScaler().fit_transform(gene_raw).astype(np.float32)
        image = StandardScaler().fit_transform(image_raw).astype(np.float32)
    else:
        gene, image = gene_raw.copy(), image_raw.copy()

    # 16-d image PCA: MP-MNCA's morphology prior input and STAIG's pseudo-label source.
    standardized_image = StandardScaler().fit_transform(image_raw)
    image_pca = (
        PCA(n_components=min(config.image_pca_dim, standardized_image.shape[1]), random_state=42)
        .fit_transform(standardized_image)
        .astype(np.float32)
    )

    coordinates = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    labels = adata.obs["ground_truth"].astype(str).to_numpy()
    barcodes = adata.obs_names.astype(str).to_numpy()

    finder = NearestNeighbors(n_neighbors=config.n_neighbors + 1).fit(coordinates)
    neighbor_indices = finder.kneighbors(coordinates, return_distance=False)[:, 1:].astype(np.int64)

    # Unsupervised pseudo-labels from image morphology, exactly as STAIG derives
    # them (src/prior_models/staig/data.py:153). Ground truth is NEVER used here --
    # that substitution is precisely the label leak in mp_mnca/train_phase1.py:133.
    pseudo_labels = (
        KMeans(n_clusters=config.image_pseudo_clusters, random_state=0, n_init=10)
        .fit_predict(image_pca)
        .astype(np.int64)
    )

    # Morphology prior over the spatial neighbourhood, matching MP-MNCA's
    # compute_morphology_prior: cosine on image PCA features, mapped to [0, 1].
    normalized_image = image_pca / np.linalg.norm(image_pca, axis=1, keepdims=True).clip(1e-8)
    neighbor_cosine = np.einsum(
        "id,ind->in", normalized_image, normalized_image[neighbor_indices]
    )
    neighbor_image_similarity = np.clip((neighbor_cosine + 1.0) / 2.0, 1e-6, None).astype(np.float32)

    return SparcSectionData(
        section_id=section_id,
        gene=gene,
        image=image,
        gene_raw=gene_raw,
        image_pca=image_pca,
        coordinates=coordinates,
        labels=labels,
        barcodes=barcodes,
        neighbor_indices=neighbor_indices,
        pseudo_labels=pseudo_labels,
        neighbor_image_similarity=neighbor_image_similarity,
    )
