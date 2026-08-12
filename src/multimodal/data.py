"""Load the cross-modal embedding pair + ground-truth layer labels every method uses.

All four methods operate on the same pair -- ``gene_emb_cm_img`` and ``img_emb_cm``
(the more strongly cross-modally-aligned pair; see ``outputs/README.md``'s cross-modal
section for why) -- read directly from ``checkpoints/dlpfc.pkl``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from src.datasets.dlpfc import DlpfcDataset


def load_embeddings_and_labels(
    checkpoint_path: Path = Path("checkpoints/dlpfc.pkl"),
    gene_key: str = "gene_emb_cm_img",
    image_key: str = "img_emb_cm",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(gene_emb, image_emb, labels, barcodes, section_ids)`` for every spot,
    concatenated in ``dataset.section_ids()`` order."""
    dataset = DlpfcDataset.from_checkpoint(checkpoint_path)

    gene_blocks, image_blocks, label_blocks, barcode_blocks, section_id_blocks = [], [], [], [], []
    for section_id in dataset.section_ids():
        adata = dataset.get_section(section_id)
        gene_blocks.append(np.asarray(adata.obsm[gene_key], dtype=np.float64))
        image_blocks.append(np.asarray(adata.obsm[image_key], dtype=np.float64))
        label_blocks.append(adata.obs["ground_truth"].to_numpy(dtype=object).astype(str))
        barcode_blocks.append(adata.obs_names.to_numpy())
        section_id_blocks.append(np.full(adata.n_obs, str(section_id), dtype=object))

    return (
        np.concatenate(gene_blocks, axis=0),
        np.concatenate(image_blocks, axis=0),
        np.concatenate(label_blocks, axis=0),
        np.concatenate(barcode_blocks, axis=0),
        np.concatenate(section_id_blocks, axis=0),
    )
