"""Group spots into fixed-size random "slices" within each (section, layer) pair.

**Terminology note**: this introduces a new meaning of "slice" that supersedes the
informal usage of ``slice_ids`` in ``src/train/train.py`` (which means *tissue section
id*, e.g. ``"151507"``). From this module onward, a "slice" is a small, randomly
assembled group of spots that share one section AND one ground-truth cortical layer.
Tissue section identity continues to be called ``section_id`` (matching
``DlpfcDataset.section_ids()``) everywhere in this new pipeline, to keep the two
concepts unambiguous.

The full hierarchy this pipeline builds on:

    spot (barcode) -> slice (~20 spots, random, within one section+layer)
                    -> layer (all slices sharing a ground-truth layer, pooled across
                       every section that has that layer -- not per-section)
                    -> section (the 7 -- or fewer, some sections lack some layers --
                       layers of one tissue slide)
                    -> donor (4 sections per donor)

For each ``(section_id, layer)`` pair, spots are shuffled and partitioned into
consecutive groups of ``slice_size`` (default 20); any remainder smaller than
``slice_size`` is dropped rather than forming an undersized slice, so every slice has
exactly the same spot count (needed for the fixed-shape covariance matrices built in
``src/analysis/slice_covariance.py``).

Usage
-----
    python -m src.analysis.slice_grouping

Output: ``outputs/covariance/slice_groups.pkl`` -- a dict ``{slice_id: {"section_id":
str, "layer": str, "barcodes": List[str]}}``, where ``slice_id`` is
``f"{section_id}__{layer}__{local_index}"``.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.datasets.dlpfc import DlpfcDataset

SliceGroups = Dict[str, Dict[str, object]]


def build_slice_groups(
    dataset: DlpfcDataset, slice_size: int = 20, seed: int = 0
) -> SliceGroups:
    """Randomly partition every (section, layer) pair's spots into fixed-size slices.

    Parameters
    ----------
    dataset:
        A loaded ``DlpfcDataset`` (e.g. via ``DlpfcDataset.from_checkpoint``).
    slice_size:
        Spots per slice. Any remainder after dividing a (section, layer) group's spot
        count by this is dropped (not padded into an undersized slice).
    seed:
        Seed for the shuffle -- one ``np.random.default_rng`` per (section, layer) pair,
        derived deterministically from this seed and the pair's identity, so re-running
        with the same seed reproduces identical groupings.

    Returns
    -------
    Dict mapping ``slice_id`` (``f"{section_id}__{layer}__{local_index}"``) to
    ``{"section_id": str, "layer": str, "barcodes": List[str]}``.
    """
    slice_groups: SliceGroups = {}

    for section_id in dataset.section_ids():
        adata = dataset.get_section(section_id)
        ground_truth = adata.obs["ground_truth"]
        barcodes = adata.obs_names.to_numpy()

        for layer in sorted(ground_truth.dropna().unique()):
            layer_mask = (ground_truth == layer).to_numpy()
            layer_barcodes = barcodes[layer_mask]

            rng = np.random.default_rng(abs(hash((seed, section_id, layer))) % (2**32))
            shuffled = layer_barcodes.copy()
            rng.shuffle(shuffled)

            n_slices = len(shuffled) // slice_size
            for local_index in range(n_slices):
                start = local_index * slice_size
                chunk = shuffled[start:start + slice_size]
                slice_id = f"{section_id}__{layer}__{local_index}"
                slice_groups[slice_id] = {
                    "section_id": section_id,
                    "layer": layer,
                    "barcodes": chunk.tolist(),
                }

    return slice_groups


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, default=Path("checkpoints/dlpfc.pkl"))
    parser.add_argument("--slice-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs/covariance/slice_groups.pkl"))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    dataset = DlpfcDataset.from_checkpoint(args.checkpoint_path)
    slice_groups = build_slice_groups(dataset, slice_size=args.slice_size, seed=args.seed)

    by_layer: Dict[str, int] = {}
    by_section: Dict[str, int] = {}
    for meta in slice_groups.values():
        by_layer[meta["layer"]] = by_layer.get(meta["layer"], 0) + 1
        by_section[meta["section_id"]] = by_section.get(meta["section_id"], 0) + 1

    print(f"Built {len(slice_groups)} slices of size {args.slice_size}")
    print("Slices per layer:", dict(sorted(by_layer.items())))
    print("Slices per section:", dict(sorted(by_section.items())))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as f:
        pickle.dump(slice_groups, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
