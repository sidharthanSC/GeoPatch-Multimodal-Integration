"""Build per-section ``.h5ad`` files for the HER2+ breast tumour ST dataset.

Source data is the ``almaan/her2st`` GitHub repository
(https://github.com/almaan/her2st), whose ``data/`` tree holds, for each of the
36 sections (patients ``A``--``H``, up to six sections each):

- ``ST-cnts/<section>.tsv.gz``   -- raw counts, spots (rows, ``"<x>x<y>"``) x genes
  (columns, HGNC symbols). Gene sets differ per section (~15k of a 18,758-gene union).
- ``ST-spotfiles/<section>_selection.tsv`` -- array coordinates ``x``/``y`` and the
  matching full-resolution image coordinates ``pixel_x``/``pixel_y``.
- ``ST-imgs/<patient>/<section>/*.jpg`` -- the H&E image the pixel coordinates live in.
- ``ST-pat/lbl/<section>_labeled_coordinates.tsv`` -- pathologist annotations. These
  exist for only **8** of the 36 sections (A1, B1, C1, D1, E1, F1, G2, H1); every spot
  of every other section gets ``ground_truth = NaN``.

The output mirrors the DLPFC ``.h5ad`` files this repository already consumes (see
``src/datasets/dlpfc.py``), so that :class:`src.datasets.her2st.Her2stDataset` can
expose HER2ST in exactly the shape :class:`~src.datasets.dlpfc.DlpfcDataset` exposes
DLPFC:

- ``X``          -- STAIG's preprocessing, identical to the pipeline that produced the
  DLPFC ``X``: ``highly_variable_genes(flavor="seurat_v3", n_top_genes=3000)`` ->
  ``normalize_total(target_sum=1e4)`` -> ``log1p`` -> ``scale(zero_center=False,
  max_value=10)``, stored as a ``float32`` CSR matrix.
- ``layers["counts"]`` -- the raw integer counts (an addition over DLPFC, which ships
  no raw layer; HER2ST's counts exist nowhere else once the ``.tsv.gz`` is dropped).
- ``var``        -- one shared index across all 36 sections: the **union** of every
  section's genes, zero-filled where a gene is absent from a section's matrix. Carries
  ``gene_ids``/``feature_types`` plus the ``highly_variable*``/``means``/``variances``/
  ``variances_norm``/``mean``/``std`` columns the DLPFC files carry.
- ``obs``        -- ``in_tissue``, ``array_row``, ``array_col``, ``ground_truth``,
  ``patient``, ``section``.
- ``obsm``       -- ``spatial`` (full-resolution pixel coordinates), ``feat`` (the
  dense 3,000-HVG block, as in DLPFC).
- ``uns["spatial"][<section>]`` -- ``images`` (``hires``/``lowres`` downsamples as
  ``float32`` in ``[0, 1]``), ``scalefactors``, ``metadata``.

``obsm["img_emb"]`` is **not** written here -- it is produced separately by
``src/datasets/her2st_image_features.py``, which ports STAIG's BYOL image pipeline.

Scalefactors differ slightly from the 10x Visium convention because the ST platform
has no fiducial frame: ``fiducial_diameter_fullres`` is therefore omitted, and
``spot_pitch_fullres``/``pixels_per_micron`` are added. Both are measured from the
section's own spot grid (200 um centre-to-centre, 100 um spot diameter) rather than
assumed, since patient A's images are supplied at a lower resolution (~1.10 px/um)
than patients B--H (~1.46 px/um).

Usage::

    python -m src.datasets.her2st_build                       # all 36 sections
    python -m src.datasets.her2st_build --sections A1 B1      # a subset
"""

from __future__ import annotations

import argparse
import gzip
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from PIL import Image
from scipy import sparse

from src.datasets.her2st import GROUND_TRUTH_CATEGORIES

logger = logging.getLogger(__name__)

# The ST platform's fixed geometry: spots are 100 um across, on a 200 um grid.
SPOT_DIAMETER_UM: float = 100.0
SPOT_PITCH_UM: float = 200.0

# Patient letter -> integer donor id, mirroring DLPFC's 1-based donor ids.
PATIENT_IDS: Dict[str, int] = {letter: i for i, letter in enumerate("ABCDEFGH", start=1)}

_SECTION_RE = re.compile(r"^[A-H][1-9]$")


class Her2stSourceError(RuntimeError):
    """Raised when the her2st source tree is missing or internally inconsistent."""


@dataclass(frozen=True)
class SectionPaths:
    """Resolved source files for one HER2ST section."""

    section_id: str
    counts: Path
    selection: Path
    image: Path
    labels: Optional[Path]

    @property
    def patient(self) -> str:
        return self.section_id[0]


def discover_sections(her2st_root: Path) -> List[SectionPaths]:
    """Locate every section's count matrix, spot file, image and (optional) labels."""
    data_dir = her2st_root / "data"
    if not data_dir.is_dir():
        raise Her2stSourceError(
            f"Expected the her2st repository's 'data' directory at {data_dir}. Clone it with:\n"
            f"  git clone --filter=blob:none --sparse https://github.com/almaan/her2st.git {her2st_root}\n"
            f"  cd {her2st_root} && git sparse-checkout set data/ST-cnts data/ST-imgs "
            f"data/ST-spotfiles data/ST-pat/lbl"
        )

    sections: List[SectionPaths] = []
    for counts in sorted(data_dir.glob("ST-cnts/*.tsv.gz")):
        section_id = counts.name.split(".")[0]
        if not _SECTION_RE.match(section_id):
            logger.warning("Skipping unrecognised count file name %s", counts.name)
            continue

        selection = data_dir / "ST-spotfiles" / f"{section_id}_selection.tsv"
        if not selection.is_file():
            raise Her2stSourceError(f"Section '{section_id}': missing spot file {selection}.")

        image_dir = data_dir / "ST-imgs" / section_id[0] / section_id
        images = sorted(image_dir.glob("*.jpg"))
        if len(images) != 1:
            raise Her2stSourceError(
                f"Section '{section_id}': expected exactly one .jpg in {image_dir}, found {len(images)}."
            )

        labels = data_dir / "ST-pat" / "lbl" / f"{section_id}_labeled_coordinates.tsv"
        sections.append(
            SectionPaths(
                section_id=section_id,
                counts=counts,
                selection=selection,
                image=images[0],
                labels=labels if labels.is_file() else None,
            )
        )

    if not sections:
        raise Her2stSourceError(f"No 'ST-cnts/*.tsv.gz' count matrices found under {data_dir}.")
    return sections


def _read_counts(path: Path) -> pd.DataFrame:
    """Read one ``ST-cnts`` matrix as a spots x genes integer DataFrame."""
    with gzip.open(path, "rt") as handle:
        counts = pd.read_csv(handle, sep="\t", index_col=0)
    # A handful of sections carry duplicated gene symbol columns; keep the first.
    if counts.columns.duplicated().any():
        counts = counts.loc[:, ~counts.columns.duplicated()]
    return counts


def union_gene_index(sections: Sequence[SectionPaths]) -> pd.Index:
    """Return the sorted union of every section's gene symbols.

    DLPFC's twelve sections all share one 33,538-gene index because they came off the
    same CellRanger reference. HER2ST's matrices were filtered per section, so a shared
    index has to be constructed; genes absent from a section are genuinely
    zero-count there, so a zero-filled union loses nothing.
    """
    genes: set = set()
    for paths in sections:
        with gzip.open(paths.counts, "rt") as handle:
            header = handle.readline().rstrip("\n").split("\t")
        genes.update(g for g in header[1:] if g)
        logger.info("Section '%s': %d genes", paths.section_id, len(header) - 1)
    return pd.Index(sorted(genes), name=None)


def _spot_key(x: float, y: float) -> str:
    """Array coordinates -> the ``"<x>x<y>"`` key used as a count-matrix row name."""
    return f"{int(round(x))}x{int(round(y))}"


def _read_labels(path: Path) -> pd.Series:
    """Read a pathologist annotation file, keyed by ``"<x>x<y>"`` spot key.

    The label files carry their own ``pixel_x``/``pixel_y`` columns, but those live in
    the coordinate space of the annotated overlay images in ``ST-pat/img/``, not the
    H&E images in ``ST-imgs/``. Only the array coordinates are used for matching.
    """
    labels = pd.read_csv(path, sep="\t")
    labels = labels.dropna(subset=["x", "y", "label"])
    keys = [_spot_key(x, y) for x, y in zip(labels["x"], labels["y"])]
    series = pd.Series(labels["label"].to_numpy(), index=keys)
    return series[~series.index.duplicated(keep="first")]


def _measure_geometry(selection: pd.DataFrame) -> Tuple[float, float]:
    """Return ``(pixels_per_array_step, pixels_per_micron)`` for one section.

    The spot pitch is recovered by least-squares fitting pixel coordinates against
    array coordinates on both axes and averaging the two slopes, which is robust to
    the missing spots and the sub-pixel jitter present in the spot files.
    """
    slopes = []
    for array_col, pixel_col in (("x", "pixel_x"), ("y", "pixel_y")):
        slope = np.polyfit(selection[array_col].to_numpy(float), selection[pixel_col].to_numpy(float), 1)[0]
        slopes.append(abs(float(slope)))
    pitch_px = float(np.mean(slopes))
    return pitch_px, pitch_px / SPOT_PITCH_UM


def _downsample_image(image: Image.Image, max_px: int) -> np.ndarray:
    """Downsample ``image`` so its longest side is ``max_px``, as float32 in [0, 1].

    Matches the dtype and value range scanpy's ``read_visium`` produces for the DLPFC
    ``uns["spatial"][...]["images"]`` entries.
    """
    width, height = image.size
    scale = max_px / float(max(width, height))
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    resized = image.resize(size, Image.Resampling.LANCZOS)
    return (np.asarray(resized, dtype=np.float32) / 255.0).astype(np.float32)


def _staig_preprocess(adata: AnnData, n_top_genes: int) -> None:
    """Apply STAIG's exact preprocessing, in place.

    This is the pipeline that produced ``adata.X`` in the DLPFC ``.h5ad`` files
    (verified: their ``X`` maxes out at exactly 10.0 and their ``var`` carries the
    ``mean``/``std`` columns ``sc.pp.scale`` writes). ``highly_variable_genes`` runs
    first because ``flavor="seurat_v3"`` expects raw counts.
    """
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_top_genes)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)


def build_section(
    paths: SectionPaths,
    genes: pd.Index,
    n_top_genes: int = 3000,
    hires_px: int = 2000,
    lowres_px: int = 600,
) -> AnnData:
    """Build one section's :class:`~anndata.AnnData`, DLPFC-shaped, minus ``img_emb``."""
    counts = _read_counts(paths.counts)
    selection = pd.read_csv(paths.selection, sep="\t")
    selection.index = [_spot_key(x, y) for x, y in zip(selection["x"], selection["y"])]

    missing = counts.index.difference(selection.index)
    if len(missing):
        raise Her2stSourceError(
            f"Section '{paths.section_id}': {len(missing)} count-matrix spots have no spot-file "
            f"entry (e.g. {list(missing[:5])}), so their pixel coordinates are unknown."
        )
    # The count matrix is the authority on which spots exist: spot files list a few
    # extra selected positions that were dropped during count-matrix QC.
    selection = selection.loc[counts.index]

    dense = counts.reindex(columns=genes, fill_value=0).to_numpy(dtype=np.float32)
    raw_counts = sparse.csr_matrix(dense)

    adata = AnnData(X=raw_counts.copy())
    adata.obs_names = counts.index.astype(str)
    adata.var_names = genes.astype(str)
    adata.layers["counts"] = raw_counts

    adata.var["gene_ids"] = adata.var_names.to_numpy()
    adata.var["feature_types"] = "Gene Expression"
    adata.var["measured_in_section"] = np.asarray(genes.isin(counts.columns))

    adata.obs["in_tissue"] = np.ones(adata.n_obs, dtype=np.int64)
    adata.obs["array_row"] = selection["y"].to_numpy(dtype=np.float32)
    adata.obs["array_col"] = selection["x"].to_numpy(dtype=np.float32)
    adata.obs["patient"] = pd.Categorical([paths.patient] * adata.n_obs, categories=list(PATIENT_IDS))
    adata.obs["section"] = pd.Categorical([paths.section_id] * adata.n_obs)

    if paths.labels is not None:
        labels = _read_labels(paths.labels)
        matched = labels.reindex(adata.obs_names)
        n_matched = int(matched.notna().sum())
        if n_matched == 0:
            raise Her2stSourceError(
                f"Section '{paths.section_id}': annotation file {paths.labels} matched none of "
                f"its {adata.n_obs} spots."
            )
        logger.info(
            "Section '%s': %d/%d spots annotated", paths.section_id, n_matched, adata.n_obs
        )
        ground_truth = matched.to_numpy()
    else:
        ground_truth = np.full(adata.n_obs, np.nan, dtype=object)
    adata.obs["ground_truth"] = pd.Categorical(
        ground_truth, categories=list(GROUND_TRUTH_CATEGORIES)
    )

    adata.obsm["spatial"] = selection[["pixel_x", "pixel_y"]].to_numpy(dtype=np.float32)

    pitch_px, px_per_um = _measure_geometry(selection)
    with Image.open(paths.image) as image:
        image = image.convert("RGB")
        full_width, full_height = image.size
        hires = _downsample_image(image, hires_px)
        lowres = _downsample_image(image, lowres_px)
    longest = float(max(full_width, full_height))

    adata.uns["spatial"] = {
        paths.section_id: {
            "images": {"hires": hires, "lowres": lowres},
            "scalefactors": {
                # No fiducial frame exists on an ST array, so unlike 10x Visium there
                # is no 'fiducial_diameter_fullres'; the two extra keys below record
                # the measured geometry the diameter was derived from.
                "spot_diameter_fullres": float(px_per_um * SPOT_DIAMETER_UM),
                "spot_pitch_fullres": float(pitch_px),
                "pixels_per_micron": float(px_per_um),
                "tissue_hires_scalef": float(hires_px / longest),
                "tissue_lowres_scalef": float(lowres_px / longest),
            },
            "metadata": {
                "chemistry_description": "Spatial Transcriptomics (ST) 1k array",
                "source": "https://github.com/almaan/her2st",
                "image_file": paths.image.name,
                "fullres_width": int(full_width),
                "fullres_height": int(full_height),
                "patient": paths.patient,
            },
        }
    }

    _staig_preprocess(adata, n_top_genes=n_top_genes)
    highly_variable = adata.var["highly_variable"].to_numpy()
    feat = adata[:, highly_variable].X
    adata.obsm["feat"] = feat.toarray() if sparse.issparse(feat) else np.asarray(feat)
    return adata


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--her2st-root",
        type=Path,
        default=Path("Dataset/her2st_repo"),
        help="Clone of https://github.com/almaan/her2st (must contain a 'data' directory).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/her2st"),
        help="Directory to write one <section>.h5ad per section into.",
    )
    parser.add_argument("--sections", nargs="*", default=None, help="Subset of section ids to build.")
    parser.add_argument("--n-top-genes", type=int, default=3000, help="HVG count, as in STAIG.")
    parser.add_argument("--hires-px", type=int, default=2000, help="Longest side of the hires image.")
    parser.add_argument("--lowres-px", type=int, default=600, help="Longest side of the lowres image.")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild sections that already exist.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    all_sections = discover_sections(args.her2st_root)
    # The gene index is always the union over *all* 36 sections, even when only a
    # subset is being rebuilt, so every file on disk shares one var index.
    genes = union_gene_index(all_sections)
    logger.info("Shared gene index: %d genes across %d sections", len(genes), len(all_sections))

    selected = all_sections
    if args.sections:
        wanted = set(args.sections)
        selected = [s for s in all_sections if s.section_id in wanted]
        unknown = wanted - {s.section_id for s in all_sections}
        if unknown:
            raise SystemExit(f"Unknown section id(s): {sorted(unknown)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for paths in selected:
        out_path = args.out_dir / f"{paths.section_id}.h5ad"
        if out_path.is_file() and not args.overwrite:
            logger.info("Section '%s': %s exists, skipping (use --overwrite)", paths.section_id, out_path)
            continue
        adata = build_section(
            paths,
            genes,
            n_top_genes=args.n_top_genes,
            hires_px=args.hires_px,
            lowres_px=args.lowres_px,
        )
        adata.write_h5ad(out_path)
        logger.info(
            "Section '%s': wrote %s (%d spots x %d genes)",
            paths.section_id, out_path, adata.n_obs, adata.n_vars,
        )


if __name__ == "__main__":
    main()
