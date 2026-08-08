"""Attach the trained gene-expression BYOL embeddings to ``checkpoints/dlpfc.pkl``.

Writes ``obsm["gene_emb"]`` into every section's ``AnnData``, matched by exact spot
barcode -- **not** by row position, since (as with
``src/datasets/attach_projected_embeddings.py``) barcodes repeat across sections and
only ``(section_id, barcode)`` pairs are unique. Unlike that earlier script, the
``.npz`` here (from ``src/gene_encoder/train.py``) already stores barcodes directly
(no need to reconstruct them from the checkpoint).

Writes to a temporary file first, reloads and spot-checks it, and only then replaces
the original checkpoint -- a failure partway through never leaves a corrupted file.

Usage
-----
    python -m src.gene_encoder.attach_gene_embeddings \\
        --embeddings-npz outputs/gene_encoder/predictions/gene_byol_embeddings.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.datasets.dlpfc import DlpfcDataset


def attach_gene_embeddings(
    checkpoint_path: Path, embeddings_npz: Path, obsm_key: str
) -> DlpfcDataset:
    dataset = DlpfcDataset.from_checkpoint(checkpoint_path)

    data = np.load(embeddings_npz, allow_pickle=True)
    embeddings = data["embeddings"]
    barcodes = data["barcodes"].astype(str)
    section_ids = data["section_ids"].astype(str)

    lookup = {
        (section_id, barcode): embedding
        for section_id, barcode, embedding in zip(section_ids, barcodes, embeddings)
    }
    print(f"Built lookup for {len(lookup)} (section, barcode) pairs from {embeddings_npz}")

    for section_id in dataset.section_ids():
        adata = dataset.get_section(section_id)
        missing = [bc for bc in adata.obs_names if (section_id, bc) not in lookup]
        if missing:
            raise ValueError(
                f"Section '{section_id}': {len(missing)} of {adata.n_obs} spots have no "
                f"gene embedding in {embeddings_npz}. Cannot attach obsm['{obsm_key}'] for them."
            )
        section_emb = np.stack(
            [lookup[(section_id, bc)] for bc in adata.obs_names]
        ).astype(np.float32)
        adata.obsm[obsm_key] = section_emb
        print(f"  {section_id}: attached obsm['{obsm_key}'] shape={section_emb.shape}")

    return dataset


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, default=Path("checkpoints/dlpfc.pkl"))
    parser.add_argument(
        "--embeddings-npz", type=Path,
        default=Path("outputs/gene_encoder/predictions/gene_byol_embeddings.npz"),
    )
    parser.add_argument("--obsm-key", type=str, default="gene_emb")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    dataset = attach_gene_embeddings(args.checkpoint_path, args.embeddings_npz, args.obsm_key)

    tmp_path = args.checkpoint_path.with_suffix(args.checkpoint_path.suffix + ".tmp")
    print(f"Saving to temporary path {tmp_path} ...")
    dataset.save_checkpoint(tmp_path)

    print("Reloading and spot-checking the temporary checkpoint ...")
    reloaded = DlpfcDataset.from_checkpoint(tmp_path)
    for section_id in reloaded.section_ids():
        adata = reloaded.get_section(section_id)
        if args.obsm_key not in adata.obsm:
            raise RuntimeError(f"Reload check failed: obsm['{args.obsm_key}'] missing for {section_id}")
        if adata.obsm[args.obsm_key].shape[0] != adata.n_obs:
            raise RuntimeError(f"Reload check failed: obsm['{args.obsm_key}'] row count mismatch for {section_id}")

    tmp_path.replace(args.checkpoint_path)
    print(f"Verified and replaced {args.checkpoint_path}")


if __name__ == "__main__":
    main()
