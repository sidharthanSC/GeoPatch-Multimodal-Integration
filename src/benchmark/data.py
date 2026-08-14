"""Per-section data loading for ``MODEL_BENCHMARK_REPORT.md``'s Table A (per-section
domain discovery) and the attention ablation suite.

Unlike ``src/multimodal/data.py`` (which pools every spot across all 12 sections into
flat arrays for a single global fit), the benchmark report's protocol evaluates each
section independently -- see ``MODEL_BENCHMARK_REPORT.md``'s "Required Common
Evaluations / Table A" section. This module keeps sections separate and additionally
loads the raw joint-HVG expression matrix needed for the R1 gene-only PCA baseline,
which nothing in ``src/multimodal/`` or ``src/cross_modal/`` needed before now.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.datasets.dlpfc import DlpfcDataset

_EMBEDDING_KEYS = (
    "gene_emb",
    "img_emb",
    "proj_emb",
    "gene_emb_cm_img",
    "img_emb_cm",
    "gene_emb_cm_proj",
    "proj_emb_cm",
)


@dataclass
class SectionData:
    section_id: str
    donor_id: int
    barcodes: np.ndarray
    labels: np.ndarray
    embeddings: Dict[str, np.ndarray]
    hvg_expression: np.ndarray  # (n_spots, len(shared_hvgs)), log1p-normalized
    spatial_neighbors: Dict[str, List[str]]  # barcode -> k nearest neighbor barcodes


def load_section_data(
    checkpoint_path: Path = Path("checkpoints/dlpfc.pkl"),
    hvg_path: Path = Path("outputs/gene_encoder/shared_hvgs.json"),
    spatial_graph_path: Path = Path("outputs/gene_encoder/spatial_knn_graph.pkl"),
) -> Dict[str, SectionData]:
    """Load every section's embeddings, raw HVG expression, and spatial-neighbor graph.

    Returns ``{section_id: SectionData}``, one entry per DLPFC section, so every
    downstream consumer (Table A evaluator, refinement pass) can process sections
    independently without ever concatenating across sections.
    """
    dataset = DlpfcDataset.from_checkpoint(checkpoint_path)

    with hvg_path.open() as f:
        shared_hvgs = json.load(f)

    with spatial_graph_path.open("rb") as f:
        all_spatial_graphs = pickle.load(f)

    sections: Dict[str, SectionData] = {}
    for section_id in dataset.section_ids():
        adata = dataset.get_section(section_id)
        hvg_mask = adata.var_names.isin(shared_hvgs)
        hvg_expr = adata.X[:, hvg_mask]
        hvg_expr = np.asarray(hvg_expr.todense() if hasattr(hvg_expr, "todense") else hvg_expr, dtype=np.float32)

        embeddings = {key: np.asarray(adata.obsm[key], dtype=np.float32) for key in _EMBEDDING_KEYS}

        sections[str(section_id)] = SectionData(
            section_id=str(section_id),
            donor_id=dataset.donor_id(section_id),
            barcodes=adata.obs_names.to_numpy(),
            labels=adata.obs["ground_truth"].to_numpy(dtype=object).astype(str),
            embeddings=embeddings,
            hvg_expression=hvg_expr,
            spatial_neighbors=all_spatial_graphs[str(section_id)],
        )
    return sections
