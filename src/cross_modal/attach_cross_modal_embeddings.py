"""Attach a cross-modal training run's two embeddings to ``checkpoints/dlpfc.pkl``.

Each ``src/cross_modal/train.py`` run trains both a gene-side and an image-side
projection head (see ``src/cross_modal/model.py``), so writes **two** new ``obsm``
keys per run -- matched by exact spot barcode, not row position (barcodes repeat
across sections; see ``src/datasets/attach_projected_embeddings.py`` for why). The
``.npz`` here already stores barcodes directly, same as
``src/gene_encoder/attach_gene_embeddings.py``.

Output key names are auto-derived from the ``.npz``'s recorded ``image_key`` unless
overridden:

- ``image_key="img_emb"``  -> ``gene_emb_cm_img``,  ``img_emb_cm``
- ``image_key="proj_emb"`` -> ``gene_emb_cm_proj``, ``proj_emb_cm``

Writes to a temporary file first, reloads and spot-checks it, and only then replaces
the original checkpoint.

Usage
-----
    python -m src.cross_modal.attach_cross_modal_embeddings \\
        --embeddings-npz outputs/cross_modal/predictions/cross_modal_img_emb_embeddings.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from src.datasets.dlpfc import DlpfcDataset

_DERIVED_KEYS = {
    "img_emb": ("gene_emb_cm_img", "img_emb_cm"),
    "proj_emb": ("gene_emb_cm_proj", "proj_emb_cm"),
}


def derive_output_keys(image_key: str) -> Tuple[str, str]:
    if image_key not in _DERIVED_KEYS:
        raise ValueError(
            f"No default output-key derivation for image_key={image_key!r}; "
            f"pass --gene-key-out/--image-key-out explicitly."
        )
    return _DERIVED_KEYS[image_key]


def attach_cross_modal_embeddings(
    checkpoint_path: Path,
    embeddings_npz: Path,
    gene_key_out: Optional[str],
    image_key_out: Optional[str],
) -> Tuple[DlpfcDataset, str, str]:
    dataset = DlpfcDataset.from_checkpoint(checkpoint_path)

    data = np.load(embeddings_npz, allow_pickle=True)
    gene_projected = data["gene_projected"]
    image_projected = data["image_projected"]
    barcodes = data["barcodes"].astype(str)
    section_ids = data["section_ids"].astype(str)
    image_key = str(data["image_key"])

    if gene_key_out is None or image_key_out is None:
        derived_gene, derived_image = derive_output_keys(image_key)
        gene_key_out = gene_key_out or derived_gene
        image_key_out = image_key_out or derived_image

    gene_lookup = {
        (sid, bc): emb for sid, bc, emb in zip(section_ids, barcodes, gene_projected)
    }
    image_lookup = {
        (sid, bc): emb for sid, bc, emb in zip(section_ids, barcodes, image_projected)
    }
    print(
        f"Built lookups for {len(gene_lookup)} (section, barcode) pairs from "
        f"{embeddings_npz} (image_key={image_key})"
    )

    for section_id in dataset.section_ids():
        adata = dataset.get_section(section_id)
        missing = [bc for bc in adata.obs_names if (section_id, bc) not in gene_lookup]
        if missing:
            raise ValueError(
                f"Section '{section_id}': {len(missing)} of {adata.n_obs} spots have no "
                f"cross-modal embedding in {embeddings_npz}."
            )
        adata.obsm[gene_key_out] = np.stack(
            [gene_lookup[(section_id, bc)] for bc in adata.obs_names]
        ).astype(np.float32)
        adata.obsm[image_key_out] = np.stack(
            [image_lookup[(section_id, bc)] for bc in adata.obs_names]
        ).astype(np.float32)
        print(
            f"  {section_id}: attached obsm['{gene_key_out}'] and obsm['{image_key_out}'], "
            f"shape={adata.obsm[gene_key_out].shape}"
        )

    return dataset, gene_key_out, image_key_out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, default=Path("checkpoints/dlpfc.pkl"))
    parser.add_argument("--embeddings-npz", type=Path, required=True)
    parser.add_argument(
        "--gene-key-out", type=str, default=None,
        help="Override the auto-derived gene-side output obsm key.",
    )
    parser.add_argument(
        "--image-key-out", type=str, default=None,
        help="Override the auto-derived image-side output obsm key.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    dataset, gene_key_out, image_key_out = attach_cross_modal_embeddings(
        args.checkpoint_path, args.embeddings_npz, args.gene_key_out, args.image_key_out
    )

    tmp_path = args.checkpoint_path.with_suffix(args.checkpoint_path.suffix + ".tmp")
    print(f"Saving to temporary path {tmp_path} ...")
    dataset.save_checkpoint(tmp_path)

    print("Reloading and spot-checking the temporary checkpoint ...")
    reloaded = DlpfcDataset.from_checkpoint(tmp_path)
    for section_id in reloaded.section_ids():
        adata = reloaded.get_section(section_id)
        for key in (gene_key_out, image_key_out):
            if key not in adata.obsm:
                raise RuntimeError(f"Reload check failed: obsm['{key}'] missing for {section_id}")
            if adata.obsm[key].shape[0] != adata.n_obs:
                raise RuntimeError(f"Reload check failed: obsm['{key}'] row count mismatch for {section_id}")

    tmp_path.replace(args.checkpoint_path)
    print(f"Verified and replaced {args.checkpoint_path}")


if __name__ == "__main__":
    main()
