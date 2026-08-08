"""The two BYOL views for gene expression: masked corruption and spatial smoothing.

Both operate on the shared-HVG log1p expression matrix (see
``src/gene_encoder/hvg_selection.py``), producing two views with genuinely different
information content -- not just two noisy copies of the same vector:

- **Masked-corruption view** (``masked_corruption_view``): VIME/BYOL-tabular-style.
  Each call is stochastic -- a fresh random mask every time it's applied, so this is
  meant to be recomputed every training step, the way a random crop is redrawn every
  step for image BYOL. ~``mask_rate`` of genes are replaced with a random draw from
  that *gene's own* empirical min-max range (not simply zeroed -- a corrupted-but-
  plausible value is a stronger training signal than plain dropout), then Gaussian
  noise (std proportional to each gene's own range, via ``noise_std_fraction``) is
  added across every gene, not just the masked ones. Both knobs default higher than a
  first pass suggested: with 97.8% of any spot's expression already zero and same-layer
  neighbors already highly correlated, a gentle version of this view is too close to
  the spatially-smoothed view (below) for the online/target prediction task to be
  meaningfully hard -- confirmed empirically (a trained encoder's embeddings were
  *more* uniform across different spots than an untrained one's, the classic BYOL
  collapse signature) before these defaults were raised.
- **Spatial-smoothing view** (``spatial_mean_aggregation``): a one-hop, fixed-weight
  graph convolution -- the mean of a spot's own expression and its k physical spatial
  neighbors' (see ``src/gene_encoder/spatial_graph.py``). Unlike the masking view, this
  is **deterministic** given the data and graph, so it's meant to be precomputed once
  for the whole dataset rather than recomputed per training step.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def masked_corruption_view(
    expression: np.ndarray,
    gene_min: np.ndarray,
    gene_max: np.ndarray,
    mask_rate: float,
    rng: np.random.Generator,
    noise_std_fraction: float = 0.0,
) -> np.ndarray:
    """Randomly replace ``mask_rate`` of each row's genes with a value drawn from that
    gene's own ``[gene_min, gene_max]`` range, then add Gaussian noise everywhere.

    Parameters
    ----------
    expression:
        ``(..., n_genes)`` array.
    gene_min, gene_max:
        ``(n_genes,)`` per-gene empirical range (broadcasts against ``expression``).
    rng:
        Caller-owned generator, so callers control reproducibility/independence across
        calls (e.g. a fresh mask every training step).
    noise_std_fraction:
        Std of the additive Gaussian noise, as a fraction of each gene's own
        ``gene_max - gene_min`` range. ``0.0`` (default) disables it.
    """
    mask = rng.random(expression.shape) < mask_rate
    random_fill = gene_min + rng.random(expression.shape) * (gene_max - gene_min)
    corrupted = np.where(mask, random_fill, expression)

    if noise_std_fraction > 0:
        gene_range = gene_max - gene_min
        noise = rng.standard_normal(expression.shape) * (noise_std_fraction * gene_range)
        corrupted = corrupted + noise

    return corrupted


def spatial_mean_aggregation(
    expression: np.ndarray,
    barcodes: Sequence[str],
    section_graph: Dict[str, List[str]],
    include_self: bool = True,
) -> np.ndarray:
    """One section's spatially-smoothed expression: each row -> mean of (self +) its
    k spatial neighbors' rows.

    Parameters
    ----------
    expression:
        ``(n_spots, n_genes)``, row order matching ``barcodes``.
    barcodes:
        This section's spot barcodes, in the same order as ``expression``'s rows.
    section_graph:
        ``{barcode: [neighbor_barcode, ...]}`` for this section (see
        ``src/gene_encoder/spatial_graph.py``).
    include_self:
        If True (default), average includes the spot's own row alongside its neighbors.
    """
    barcode_to_row = {barcode: i for i, barcode in enumerate(barcodes)}
    smoothed = np.empty_like(expression)

    for i, barcode in enumerate(barcodes):
        neighbor_rows = [barcode_to_row[b] for b in section_graph[barcode]]
        if include_self:
            neighbor_rows.append(i)
        smoothed[i] = expression[neighbor_rows].mean(axis=0)

    return smoothed
