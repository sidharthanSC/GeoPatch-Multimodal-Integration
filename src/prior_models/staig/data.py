"""Data preparation and graph augmentation used by the STAIG adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from anndata import AnnData
from scipy.special import softmax
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .config import StaigConfig


@dataclass(frozen=True)
class StaigSectionData:
    section_id: str
    features: np.ndarray
    image_features: np.ndarray
    coordinates: np.ndarray
    labels: np.ndarray
    barcodes: np.ndarray
    edge_index: np.ndarray
    edge_drop_probability: np.ndarray
    pseudo_labels: np.ndarray
    node_feature_name: str = "section_hvg_expression"
    image_feature_name: str = "img_emb"


@dataclass(frozen=True)
class PrecomputedFeatureBundle:
    values: np.ndarray
    section_ids: np.ndarray
    barcodes: np.ndarray
    source: str
    array_key: str

    def for_section(self, section_id: str, expected_barcodes: np.ndarray) -> np.ndarray:
        """Return rows in checkpoint barcode order, validating compound identities."""
        identities = list(zip(self.section_ids.tolist(), self.barcodes.tolist()))
        if len(set(identities)) != len(identities):
            raise ValueError(f"{self.source} contains duplicate (section_id, barcode) identities")
        lookup = {identity: index for index, identity in enumerate(identities)}
        expected = [(str(section_id), str(barcode)) for barcode in expected_barcodes]
        missing = [identity for identity in expected if identity not in lookup]
        if missing:
            raise ValueError(f"{self.source} is missing spot {missing[0]}")
        section_source = {identity for identity in lookup if identity[0] == str(section_id)}
        extras = section_source.difference(expected)
        if extras:
            raise ValueError(f"{self.source} has {len(extras)} unexpected spots in {section_id}")
        return self.values[np.asarray([lookup[identity] for identity in expected])]


def load_feature_bundle(path: Path, array_key: str) -> PrecomputedFeatureBundle:
    """Load one finite 2-D feature array plus compound spot identities."""
    with np.load(path, allow_pickle=True) as artifact:
        required = {array_key, "section_ids", "barcodes"}
        missing = required.difference(artifact.files)
        if missing:
            raise KeyError(f"Missing arrays in {path}: {sorted(missing)}")
        values = np.asarray(artifact[array_key], dtype=np.float32)
        section_ids = np.asarray(artifact["section_ids"]).astype(str)
        barcodes = np.asarray(artifact["barcodes"]).astype(str)
    if values.ndim != 2:
        raise ValueError(f"{path}:{array_key} must be 2-D, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{path}:{array_key} contains non-finite values")
    if not (len(values) == len(section_ids) == len(barcodes)):
        raise ValueError(f"{path} arrays have inconsistent row counts")
    return PrecomputedFeatureBundle(values, section_ids, barcodes, str(path), array_key)


def build_spatial_graph(coordinates: np.ndarray, n_neighbors: int) -> np.ndarray:
    """Return a symmetric directed edge list from a coordinate k-NN graph."""
    coordinates = np.asarray(coordinates, dtype=np.float32)
    if coordinates.ndim != 2 or coordinates.shape[0] <= n_neighbors:
        raise ValueError("coordinates must be 2-D with more rows than n_neighbors")
    indices = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(coordinates).kneighbors(
        coordinates, return_distance=False
    )[:, 1:]
    src = np.repeat(np.arange(coordinates.shape[0]), n_neighbors)
    dst = indices.reshape(-1)
    pairs = np.concatenate(
        [np.stack([src, dst], axis=1), np.stack([dst, src], axis=1)], axis=0
    )
    pairs = np.unique(pairs, axis=0)
    return pairs.T.astype(np.int64, copy=False)


def image_guided_edge_probabilities(
    edge_index: np.ndarray, image_features: np.ndarray
) -> np.ndarray:
    """Compute STAIG's row-wise softmax of log Euclidean edge distances."""
    image_features = np.asarray(image_features, dtype=np.float64)
    src, dst = edge_index
    distances = np.linalg.norm(image_features[src] - image_features[dst], axis=1)
    distances = np.maximum(distances, np.finfo(np.float64).tiny)
    probabilities = np.zeros(distances.shape[0], dtype=np.float32)
    for node in np.unique(src):
        mask = src == node
        probabilities[mask] = softmax(np.log(distances[mask])).astype(np.float32)
    return probabilities


def prepare_section(
    adata: AnnData,
    section_id: str,
    config: StaigConfig,
    node_features: np.ndarray | None = None,
    image_features: np.ndarray | None = None,
    node_feature_name: str = "section_hvg_expression",
    image_feature_name: str = "img_emb",
) -> StaigSectionData:
    """Prepare one already-normalized DLPFC section for STAIG training."""
    if node_features is None:
        if "highly_variable" not in adata.var:
            raise KeyError("STAIG reproduction requires adata.var['highly_variable']")
        hvg = adata.var["highly_variable"].to_numpy(dtype=bool)
        if int(hvg.sum()) != 3000:
            raise ValueError(f"Expected 3000 section-level HVGs, found {int(hvg.sum())}")
        matrix = adata[:, hvg].X
        if hasattr(matrix, "toarray"):
            matrix = matrix.toarray()
        features = np.asarray(matrix, dtype=np.float32)
    else:
        features = np.asarray(node_features, dtype=np.float32)
    if features.ndim != 2 or features.shape[0] != adata.n_obs:
        raise ValueError(f"Invalid node feature shape {features.shape} for {adata.n_obs} spots")
    if not np.isfinite(features).all():
        raise ValueError("Node features contain non-finite values")

    raw_image = np.asarray(
        adata.obsm["img_emb"] if image_features is None else image_features,
        dtype=np.float32,
    )
    if raw_image.shape != (adata.n_obs, 128):
        raise ValueError(f"Expected 128-d image features for {adata.n_obs} spots, got {raw_image.shape}")
    if not np.isfinite(raw_image).all():
        raise ValueError("Image features contain non-finite values")
    standardized = StandardScaler().fit_transform(raw_image)
    image_features = PCA(
        n_components=min(config.image_pca_dim, standardized.shape[1]), random_state=42
    ).fit_transform(standardized).astype(np.float32)

    coordinates = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    edge_index = build_spatial_graph(coordinates, config.n_neighbors)
    edge_probability = image_guided_edge_probabilities(edge_index, image_features)
    pseudo_labels = KMeans(
        n_clusters=config.image_pseudo_clusters, random_state=0, n_init=10
    ).fit_predict(image_features).astype(np.int64)

    return StaigSectionData(
        section_id=section_id,
        features=features,
        image_features=image_features,
        coordinates=coordinates,
        labels=adata.obs["ground_truth"].astype(str).to_numpy(),
        barcodes=adata.obs_names.astype(str).to_numpy(),
        edge_index=edge_index,
        edge_drop_probability=edge_probability,
        pseudo_labels=pseudo_labels,
        node_feature_name=node_feature_name,
        image_feature_name=image_feature_name,
    )


def sample_augmented_edges(
    edge_index: np.ndarray,
    drop_probability: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample one undirected graph using STAIG's image-guided edge deletion."""
    src, dst = edge_index
    upper = src < dst
    u_src, u_dst = src[upper], dst[upper]
    u_prob = drop_probability[upper]
    keep = rng.random(u_prob.shape[0]) >= u_prob
    kept_src, kept_dst = u_src[keep], u_dst[keep]
    return np.stack(
        [
            np.concatenate([kept_src, kept_dst]),
            np.concatenate([kept_dst, kept_src]),
        ],
        axis=0,
    ).astype(np.int64, copy=False)
