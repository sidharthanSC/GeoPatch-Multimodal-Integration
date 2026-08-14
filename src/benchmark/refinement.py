"""One common spatial-neighbor majority-vote refinement, applied identically to every
representation's cluster assignments (``MODEL_BENCHMARK_REPORT.md``'s Table A step 5:
"Add a separate table where one identical spatial refinement is applied to every
method.").

Standard single-pass refinement used in the SpaGCN/STAGATE DLPFC workflows: for each
spot, look at its k spatial neighbors' cluster labels (``k=6``, the cached
``outputs/gene_encoder/spatial_knn_graph.pkl`` graph -- matches the 10x Visium hex-grid
neighbor count); if a strict majority of neighbors share one label different from the
spot's own, reassign the spot to that label. Applied once (not iterated to convergence),
uniformly, after clustering -- never influences the clustering itself.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List

import numpy as np


def refine_labels(
    barcodes: np.ndarray, labels: np.ndarray, spatial_neighbors: Dict[str, List[str]]
) -> np.ndarray:
    barcode_to_index = {barcode: i for i, barcode in enumerate(barcodes)}
    barcode_to_label = {barcode: labels[i] for i, barcode in enumerate(barcodes)}

    refined = labels.copy()
    for barcode, i in barcode_to_index.items():
        neighbor_barcodes = spatial_neighbors.get(barcode, [])
        if not neighbor_barcodes:
            continue
        neighbor_labels = [barcode_to_label[b] for b in neighbor_barcodes if b in barcode_to_label]
        if not neighbor_labels:
            continue

        counts = Counter(neighbor_labels)
        majority_label, majority_count = counts.most_common(1)[0]
        if majority_label != labels[i] and majority_count > len(neighbor_labels) / 2:
            refined[i] = majority_label
    return refined
