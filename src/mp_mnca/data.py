"""Data preparation for MP-MNCA.

Builds spatial graphs, prepares image features, and creates neighbor indices
for cross-attention.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from anndata import AnnData
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.datasets.dlpfc import DlpfcDataset


@dataclass(frozen=True)
class MpMncaSectionData:
    """Prepared data for one DLPFC section for MP-MNCA training."""
    section_id: str
    gene_expression: np.ndarray  # (n_spots, n_genes) - log1p normalized HVG expression
    image_features: np.ndarray  # (n_spots, image_pca_dim) - PCA-reduced img_emb
    coordinates: np.ndarray  # (n_spots, 2)
    labels: np.ndarray  # (n_spots,) - ground truth layer labels
    barcodes: np.ndarray  # (n_spots,)
    neighbor_indices: np.ndarray  # (n_spots, n_neighbors) - indices into this section
    gene_mask: np.ndarray  # (n_spots, n_genes) - boolean mask for structured masking (optional)
    n_neighbors: int
    train_mask: np.ndarray | None = None  # (n_spots,) bool; None = train on all spots
    pseudo_labels: np.ndarray | None = None  # (n_spots,) gene-KMeans clusters for negative masking


def gene_kmeans_pseudo_labels(
    gene_features: np.ndarray,
    n_clusters: int,
    seed: int = 0,
) -> np.ndarray:
    """KMeans-cluster gene expression and return the cluster assignments.

    These are the pseudo-labels used for the contrastive negative mask, so
    that spots in the same gene-expression cluster are not treated as
    negatives. Fully unsupervised (gene expression only, no ground truth).
    """
    from sklearn.cluster import KMeans

    return KMeans(
        n_clusters=n_clusters, random_state=seed, n_init=10
    ).fit_predict(gene_features).astype(np.int64)


def intersection_pseudo_mask(
    gene_features: np.ndarray,
    image_features: np.ndarray,
    n_clusters: int,
    seed: int = 0,
) -> np.ndarray:
    """Spots whose gene-KMeans and image-KMeans cluster assignments agree.

    Clusters gene expression and image embeddings separately with KMeans
    (same k, fixed seed), aligns the two cluster labelings with Hungarian
    matching on the contingency table, and returns a boolean mask marking
    spots whose aligned gene/image cluster IDs match. These are the spots to
    train on; the complement is held out for evaluation only.
    """
    from scipy.optimize import linear_sum_assignment
    from sklearn.cluster import KMeans

    gene_clusters = KMeans(
        n_clusters=n_clusters, random_state=seed, n_init=10
    ).fit_predict(gene_features).astype(np.int64)

    image_clusters = KMeans(
        n_clusters=n_clusters, random_state=seed, n_init=10
    ).fit_predict(image_features).astype(np.int64)

    contingency = np.zeros((n_clusters, n_clusters), dtype=np.int64)
    for g, im in zip(gene_clusters, image_clusters):
        contingency[g, im] += 1

    gene_row, image_col = linear_sum_assignment(-contingency)
    image_to_gene = np.full(n_clusters, -1, dtype=np.int64)
    image_to_gene[image_col] = gene_row

    aligned = image_to_gene[image_clusters]
    return aligned == gene_clusters


def build_spatial_knn_graph(
    coordinates: np.ndarray,
    n_neighbors: int = 6,
) -> np.ndarray:
    """Build k-NN spatial graph within a section.

    Args:
        coordinates: (n_spots, 2) spatial coordinates.
        n_neighbors: Number of neighbors per spot.

    Returns:
        neighbor_indices: (n_spots, n_neighbors) array of neighbor indices.
    """
    coords = np.asarray(coordinates, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[0] <= n_neighbors:
        raise ValueError("coordinates must be 2-D with more rows than n_neighbors")

    finder = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(coords)
    _, indices = finder.kneighbors(coords)  # (n_spots, n_neighbors+1), includes self
    neighbor_indices = indices[:, 1:]  # exclude self
    return neighbor_indices.astype(np.int64)


def prepare_image_features(
    adata: AnnData,
    image_pca_dim: int = 16,
    image_key: str = "img_emb",
) -> np.ndarray:
    """Prepare image features: standardize + PCA.

    Args:
        adata: AnnData object with obsm[image_key].
        image_pca_dim: Number of PCA components.
        image_key: Key in adata.obsm for image embeddings.

    Returns:
        Image features of shape (n_spots, image_pca_dim).
    """
    raw_image = np.asarray(adata.obsm[image_key], dtype=np.float32)
    if not np.isfinite(raw_image).all():
        raise ValueError("Image features contain non-finite values")

    standardized = StandardScaler().fit_transform(raw_image)
    pca_dim = min(image_pca_dim, standardized.shape[1])
    image_features = PCA(
        n_components=pca_dim, random_state=42
    ).fit_transform(standardized).astype(np.float32)

    return image_features


def prepare_gene_expression(
    adata: AnnData,
    hvg_key: str = "highly_variable",
    expected_hvg_count: int = 3000,
    use_feat_obsm: bool = True,
    use_pretrained_gene_emb: bool = False,
    feat_subset_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Extract gene expression or pre-trained gene embeddings from AnnData.

    Args:
        adata: AnnData with var[hvg_key] marking HVGs, or obsm['gene_emb'] for pre-trained.
        hvg_key: Column in adata.var for HVG selection.
        expected_hvg_count: Expected number of HVGs.
        use_feat_obsm: If True, use adata.obsm['feat'] (precomputed 3000 HVG expression)
                       instead of adata.X. This is needed when the checkpoint doesn't
                       contain the raw expression matrix.
        use_pretrained_gene_emb: If True, use adata.obsm['gene_emb'] (128-d pre-trained BYOL embeddings)
                                 instead of raw HVG expression.
        feat_subset_indices: Optional column indices into adata.obsm['feat'] to select
                             a reduced HVG subset (e.g. top-300 by variance) when the
                             requested gene_dim is smaller than the stored 3000.

    Returns:
        Gene expression matrix (n_spots, n_hvg) or gene embeddings (n_spots, 128) as float32.
    """
    if use_pretrained_gene_emb:
        if "gene_emb" not in adata.obsm:
            raise KeyError("adata.obsm['gene_emb'] not found for pre-trained gene embeddings")
        features = np.asarray(adata.obsm["gene_emb"], dtype=np.float32)
        if not np.isfinite(features).all():
            raise ValueError("Pre-trained gene embeddings contain non-finite values")
        return features

    if use_feat_obsm and "feat" in adata.obsm:
        features = np.asarray(adata.obsm["feat"], dtype=np.float32)
        if feat_subset_indices is not None:
            features = features[:, feat_subset_indices]
        if features.shape[1] != expected_hvg_count:
            raise ValueError(f"Expected {expected_hvg_count} HVGs in obsm['feat'], found {features.shape[1]}")
        if not np.isfinite(features).all():
            raise ValueError("Gene expression in obsm['feat'] contains non-finite values")
        return features

    # Fallback to original method using adata.X
    if hvg_key not in adata.var:
        raise KeyError(f"adata.var['{hvg_key}'] not found")

    hvg = adata.var[hvg_key].to_numpy(dtype=bool)
    actual_count = int(hvg.sum())
    if actual_count != expected_hvg_count:
        raise ValueError(f"Expected {expected_hvg_count} HVGs, found {actual_count}")

    matrix = adata[:, hvg].X
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    features = np.asarray(matrix, dtype=np.float32)

    if not np.isfinite(features).all():
        raise ValueError("Gene expression contains non-finite values")

    return features


def prepare_section(
    adata: AnnData,
    section_id: str,
    config,
    n_neighbors: int = 6,
    image_key: str = "img_emb",
    hvg_key: str = "highly_variable",
    use_feat_obsm: bool = True,
) -> MpMncaSectionData:
    """Prepare one DLPFC section for MP-MNCA training.

    Args:
        adata: AnnData for the section.
        section_id: Section identifier.
        config: MpMncaConfig object.
        n_neighbors: Number of spatial neighbors.
        image_key: Key for image embeddings in adata.obsm.
        hvg_key: Key for HVG selection in adata.var.
        use_feat_obsm: If True, use adata.obsm['feat'] for gene expression.

    Returns:
        MpMncaSectionData with all prepared arrays.
    """
    # Gene expression (or pre-trained gene embeddings)
    use_pretrained = getattr(config, "use_pretrained_gene_emb", False)
    gene_expression = prepare_gene_expression(
        adata, hvg_key, config.gene_input_dim, use_feat_obsm, use_pretrained
    )

    # Image features
    image_features = prepare_image_features(adata, config.image_pca_dim, image_key)

    # Coordinates
    coordinates = np.asarray(adata.obsm["spatial"], dtype=np.float32)

    # Labels
    labels = adata.obs["ground_truth"].astype(str).to_numpy()

    # Barcodes
    barcodes = adata.obs_names.astype(str).to_numpy()

    # Spatial neighbor graph
    neighbor_indices = build_spatial_knn_graph(coordinates, n_neighbors)

    return MpMncaSectionData(
        section_id=section_id,
        gene_expression=gene_expression,
        image_features=image_features,
        coordinates=coordinates,
        labels=labels,
        barcodes=barcodes,
        neighbor_indices=neighbor_indices,
        gene_mask=None,  # Will be generated per-batch
        n_neighbors=n_neighbors,
    )


def collate_mp_mnca_batch(
    section_data: MpMncaSectionData,
    batch_indices: np.ndarray,
    config,
    generator: Optional[torch.Generator] = None,
) -> dict[str, torch.Tensor]:
    """Collate a batch for MP-MNCA training.

    Args:
        section_data: Prepared section data.
        batch_indices: Indices of spots in this batch.
        config: MpMncaConfig.
        generator: Random generator for masking.

    Returns:
        Dictionary with tensors for model forward pass.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if config.dtype == "float32" else torch.float64

    batch_size = len(batch_indices)
    n_neighbors = section_data.n_neighbors

    # Center spot data
    center_gene = torch.as_tensor(
        section_data.gene_expression[batch_indices], dtype=dtype, device=device
    )
    center_image = torch.as_tensor(
        section_data.image_features[batch_indices], dtype=dtype, device=device
    )
    center_coords = torch.as_tensor(
        section_data.coordinates[batch_indices], dtype=dtype, device=device
    )

    # Neighbor indices for this batch
    neighbor_idx = section_data.neighbor_indices[batch_indices]  # (batch, n_neighbors)

    # Neighbor gene expression (clean for target, masked for online)
    neighbor_gene_clean = torch.as_tensor(
        section_data.gene_expression[neighbor_idx], dtype=dtype, device=device
    )
    neighbor_image = torch.as_tensor(
        section_data.image_features[neighbor_idx], dtype=dtype, device=device
    )
    neighbor_coords = torch.as_tensor(
        section_data.coordinates[neighbor_idx], dtype=dtype, device=device
    )

    # Create masked versions for online branch
    from .masking import apply_mask, create_mask_tensor

    # Gene masking for center
    center_mask = create_mask_tensor(
        (batch_size, config.gene_input_dim),
        config.masking_type,
        config.mask_rate,
        config.block_size,
        config.num_blocks,
        generator,
        device=str(device),
    )
    center_gene_masked = center_gene.clone()
    center_gene_masked[center_mask] = 0

    # Gene masking for neighbors - need to reshape to apply 2D mask
    neighbor_mask = create_mask_tensor(
        (batch_size * n_neighbors, config.gene_input_dim),
        config.masking_type,
        config.mask_rate,
        config.block_size,
        config.num_blocks,
        generator,
        device=str(device),
    )
    neighbor_gene_masked = neighbor_gene_clean.clone()
    neighbor_gene_masked_flat = neighbor_gene_masked.view(batch_size * n_neighbors, config.gene_input_dim)
    neighbor_gene_masked_flat[neighbor_mask] = 0

    return {
        "center_gene_masked": center_gene_masked,
        "center_gene_clean": torch.as_tensor(
            section_data.gene_expression[batch_indices], dtype=dtype, device=device
        ),
        "neighbor_genes_masked": neighbor_gene_masked,
        "neighbor_genes_clean": neighbor_gene_clean,
        "center_image": center_image,
        "neighbor_images": neighbor_image,
        "center_coords": center_coords,
        "neighbor_coords": neighbor_coords,
        "neighbor_indices": torch.as_tensor(neighbor_idx, dtype=torch.long, device=device),
    }


def prepare_all_sections(
    dataset: DlpfcDataset,
    config,
    n_neighbors: int = 6,
    image_key: str = "img_emb",
    hvg_key: str = "highly_variable",
) -> dict[str, MpMncaSectionData]:
    """Prepare all sections in the dataset."""
    sections = {}
    for section_id in dataset.section_ids():
        adata = dataset.get_section(section_id)
        sections[section_id] = prepare_section(
            adata, section_id, config, n_neighbors, image_key, hvg_key
        )
    return sections