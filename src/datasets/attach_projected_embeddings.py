"""Attach a run's projected embeddings to ``checkpoints/dlpfc.pkl`` as ``obsm["proj_emb"]``.

Reads the ``projected_embeddings`` array from a ``src/train/train.py``-exported
``.npz`` (see ``export_embeddings`` in that module) and writes it into each section's
``AnnData.obsm["proj_emb"]`` in the ``DlpfcDataset`` checkpoint, matched by the exact
spot barcode -- **not** by row position. Barcodes repeat across the twelve DLPFC
sections (10x Visium reuses one fixed barcode whitelist per array design: 47,329 spots
but only 4,941 distinct barcode strings), so matching is keyed on
``(section_id, barcode)`` pairs, recovered the same way
``src/analysis/knn_projection_analysis.py`` does, with the same alignment check against
the ``.npz``'s ``cortical_labels``/``slice_ids`` before trusting the barcode order.

This overwrites ``checkpoints/dlpfc.pkl`` (all other fields -- ``img_emb``, ``spatial``,
``ground_truth``, etc. -- are left untouched; only the new ``obsm`` key is added). The
write goes to a temporary file first, is reloaded and spot-checked, and only then
replaces the original, so a failure partway through never leaves a corrupted checkpoint.

Usage
-----
    python -m src.datasets.attach_projected_embeddings \\
        --embeddings-npz outputs/predictions/layer_projector_e_200_pw_5_similarity_gap_embeddings.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.analysis.knn_projection_analysis import load_barcodes_labels_and_slices
from src.datasets.dlpfc import DlpfcDataset


def attach_projected_embeddings(
    checkpoint_path: Path, embeddings_npz: Path, obsm_key: str
) -> DlpfcDataset:
    dataset = DlpfcDataset.from_checkpoint(checkpoint_path)

    data = np.load(embeddings_npz, allow_pickle=True)
    projected = data["projected_embeddings"]
    npz_labels = data["cortical_labels"].astype(str)
    npz_slice_ids = data["slice_ids"].astype(str)

    barcodes, labels, slice_ids = load_barcodes_labels_and_slices(checkpoint_path)

    if not (np.array_equal(labels, npz_labels) and np.array_equal(slice_ids, npz_slice_ids)):
        raise ValueError(
            f"Checkpoint-derived (labels, slice_ids) do not align positionally with "
            f"{embeddings_npz}. Re-export embeddings from the current checkpoint before "
            f"attaching them."
        )

    lookup = {
        (section_id, barcode): embedding
        for section_id, barcode, embedding in zip(slice_ids, barcodes, projected)
    }
    print(f"Built lookup for {len(lookup)} (section, barcode) pairs from {embeddings_npz}")

    for section_id in dataset.section_ids():
        adata = dataset.get_section(section_id)
        missing = [bc for bc in adata.obs_names if (section_id, bc) not in lookup]
        if missing:
            raise ValueError(
                f"Section '{section_id}': {len(missing)} of {adata.n_obs} spots have no "
                f"projected embedding in {embeddings_npz} (likely excluded from that export "
                f"by a ground_truth NaN filter). Cannot attach obsm['{obsm_key}'] for them."
            )
        section_proj = np.stack(
            [lookup[(section_id, bc)] for bc in adata.obs_names]
        ).astype(np.float32)
        adata.obsm[obsm_key] = section_proj
        print(f"  {section_id}: attached obsm['{obsm_key}'] shape={section_proj.shape}")

    return dataset


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, default=Path("checkpoints/dlpfc.pkl"))
    parser.add_argument("--embeddings-npz", type=Path, required=True)
    parser.add_argument("--obsm-key", type=str, default="proj_emb")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    dataset = attach_projected_embeddings(args.checkpoint_path, args.embeddings_npz, args.obsm_key)

    tmp_path = args.checkpoint_path.with_suffix(args.checkpoint_path.suffix + ".tmp")
    print(f"Saving to temporary path {tmp_path} ...")
    dataset.save_checkpoint(tmp_path)

    print("Reloading and spot-checking the temporary checkpoint ...")
    reloaded = DlpfcDataset.from_checkpoint(tmp_path)
    for section_id in reloaded.section_ids():
        adata = reloaded.get_section(section_id)
        if args.obsm_key not in adata.obsm:
            raise RuntimeError(f"Reload check failed: obsm['{args.obsm_key}'] missing for {section_id}")
        if adata.obsm[args.obsm_key].shape != (adata.n_obs, adata.obsm["img_emb"].shape[1]):
            raise RuntimeError(f"Reload check failed: obsm['{args.obsm_key}'] shape mismatch for {section_id}")

    tmp_path.replace(args.checkpoint_path)
    print(f"Verified and replaced {args.checkpoint_path}")


if __name__ == "__main__":
    main()
