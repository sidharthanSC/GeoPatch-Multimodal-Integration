"""Shared cortical-layer label order and color map for DLPFC annotation plots.

Order and exact strings follow ``adata.obs["ground_truth"]`` (Layer_1..Layer_6
plus WM). A common color reference is used across every model so regions are
visually comparable.
"""

from __future__ import annotations

LAYER_LABELS: tuple[str, ...] = (
    "Layer_1",
    "Layer_2",
    "Layer_3",
    "Layer_4",
    "Layer_5",
    "Layer_6",
    "WM",
)

LAYER_COLORS: dict[str, str] = {
    "Layer_1": "#1f77b4",
    "Layer_2": "#ff7f0e",
    "Layer_3": "#2ca02c",
    "Layer_4": "#d62728",
    "Layer_5": "#9467bd",
    "Layer_6": "#8c564b",
    "WM": "#7f7f7f",
}
