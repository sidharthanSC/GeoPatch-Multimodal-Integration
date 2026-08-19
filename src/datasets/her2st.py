"""PyTorch ``Dataset`` for the HER2+ breast tumour spatial transcriptomics benchmark.

HER2ST (Andersson et al., *Nature Communications* 2021; data at
https://github.com/almaan/her2st) consists of 36 Spatial Transcriptomics (ST) sections
from eight HER2-positive breast tumour patients, ``A``--``H``. This module exposes
:class:`Her2stDataset`, which is the HER2ST counterpart of
:class:`src.datasets.dlpfc.DlpfcDataset`: it discovers and loads all 36 ``.h5ad`` files,
keeps one :class:`anndata.AnnData` object per section (no concatenation), and exposes
every spot across every section as a single flat, constant-time-indexable dataset.

The sample dict, the field validation and the checkpoint format are deliberately
identical to :class:`~src.datasets.dlpfc.DlpfcDataset`'s, so downstream code written
against DLPFC works against HER2ST unchanged. Two differences are inherent to the
dataset rather than to this class:

- **``gt`` is missing for most spots.** Pathologist annotations exist for only 8 of the
  36 sections (:data:`ANNOTATED_SECTIONS`); every spot of the other 28 has
  ``gt = nan``. Use :meth:`annotated_indices` (or filter on ``notna``) when labels are
  required. DLPFC, by contrast, labels every spot.
- **``did`` runs 1--8**, one id per patient (``A`` -> 1 ... ``H`` -> 8), where DLPFC's
  donor ids run 1--3.

The ``.h5ad`` files themselves are produced by ``src/datasets/her2st_build.py``
(expression, coordinates, labels, images) and ``src/datasets/her2st_image_features.py``
(``obsm["img_emb"]``, via STAIG's BYOL pipeline); see those modules for exactly how each
field is derived.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from anndata import AnnData
from scipy import sparse
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Patient id -> that patient's sections, per the HER2ST study design. Fixed by the
# dataset itself (not derivable from any field inside the .h5ad files), so it is
# hard-coded here, exactly as DlpfcDataset hard-codes its donor/section table.
_PATIENT_SECTIONS: Dict[int, Tuple[str, ...]] = {
    1: ("A1", "A2", "A3", "A4", "A5", "A6"),
    2: ("B1", "B2", "B3", "B4", "B5", "B6"),
    3: ("C1", "C2", "C3", "C4", "C5", "C6"),
    4: ("D1", "D2", "D3", "D4", "D5", "D6"),
    5: ("E1", "E2", "E3"),
    6: ("F1", "F2", "F3"),
    7: ("G1", "G2", "G3"),
    8: ("H1", "H2", "H3"),
}

_SECTION_TO_PATIENT: Dict[str, int] = {
    section_id: patient_id
    for patient_id, section_ids in _PATIENT_SECTIONS.items()
    for section_id in section_ids
}

# The eight sections a pathologist annotated; every other section's ``ground_truth``
# is NaN for all of its spots.
ANNOTATED_SECTIONS: Tuple[str, ...] = ("A1", "B1", "C1", "D1", "E1", "F1", "G2", "H1")

# The seven pathologist annotation classes. Fixed here (rather than inferred per
# section) so every section shares one categorical dtype even though 28 of the 36
# carry no annotation at all; ``src/datasets/her2st_build.py`` builds against this.
GROUND_TRUTH_CATEGORIES: Tuple[str, ...] = (
    "adipose tissue",
    "breast glands",
    "cancer in situ",
    "connective tissue",
    "immune infiltrate",
    "invasive cancer",
    "undetermined",
)

_REQUIRED_OBSM_KEYS: Tuple[str, ...] = ("spatial", "img_emb")
_REQUIRED_OBS_COLUMNS: Tuple[str, ...] = ("ground_truth",)
_REQUIRED_UNS_KEYS: Tuple[str, ...] = ("spatial",)

# Bumped whenever the on-disk checkpoint schema changes, so stale checkpoints
# written by an older version of this class fail loudly instead of silently.
_CHECKPOINT_VERSION: int = 1


class Her2stSectionValidationError(ValueError):
    """Raised when a loaded HER2ST section is missing a required field.

    Distinguishing this from a plain ``ValueError`` lets callers catch
    dataset-integrity problems (e.g. a section whose ``img_emb`` was never
    generated) separately from ordinary argument-validation errors.
    """


@dataclass(frozen=True)
class _Section:
    """In-memory cache entry for one loaded HER2ST tissue section."""

    section_id: str
    patient_id: int
    adata: AnnData


class Her2stDataset(Dataset):
    """Flat, spot-level PyTorch dataset over all 36 HER2ST sections.

    On construction, every expected ``<section_id>.h5ad`` file under
    ``<root>/data/her2st/`` is read exactly once via :func:`scanpy.read_h5ad` and
    cached as an individual :class:`anndata.AnnData` object (sections are never
    concatenated). A global index of ``(section_id, local_spot_index)`` pairs is built
    so that ``__getitem__`` resolves any global sample index to its section and
    within-section row in O(1).

    Patient identity is assigned purely from the fixed HER2ST study design, not read
    from the AnnData objects:

    =======  ================================
    Patient  Sections
    =======  ================================
    1 (A)    A1, A2, A3, A4, A5, A6
    2 (B)    B1, B2, B3, B4, B5, B6
    3 (C)    C1, C2, C3, C4, C5, C6
    4 (D)    D1, D2, D3, D4, D5, D6
    5 (E)    E1, E2, E3
    6 (F)    F1, F2, F3
    7 (G)    G1, G2, G3
    8 (H)    H1, H2, H3
    =======  ================================

    Each sample returned by :meth:`__getitem__` is a dict with keys:

    - ``"x"``: gene expression vector for the spot (``torch.Tensor``, shape ``(n_genes,)``).
    - ``"sp"``: spatial coordinates of the spot (``torch.Tensor``, shape ``(2,)``).
    - ``"img"``: image embedding of the spot (``torch.Tensor``, shape ``(embed_dim,)``).
    - ``"gt"``: pathologist annotation for the spot, one of ``"invasive cancer"``,
      ``"cancer in situ"``, ``"breast glands"``, ``"connective tissue"``,
      ``"immune infiltrate"``, ``"adipose tissue"``, ``"undetermined"`` -- or ``nan``
      for every spot of the 28 unannotated sections.
    - ``"did"``: patient id, ``1`` (A) through ``8`` (H) (``int``).
    - ``"sid"``: section id, e.g. ``"A1"`` (``str``).
    - ``"img_meta"``: the section-level H&E image metadata dict, i.e.
      ``adata.uns["spatial"]`` for the spot's section (shared by every spot in
      that section).

    Only ``adata.X``, ``adata.obsm["spatial"]``, ``adata.obsm["img_emb"]``,
    ``adata.var``, ``adata.uns["spatial"]``, and ``adata.obs["ground_truth"]``
    are read. All other fields (e.g. ``adata.layers["counts"]``,
    ``adata.obsm["feat"]``, ``adata.obs["patient"]``) are ignored entirely.

    Building the dataset means reading and validating all 36 ``.h5ad`` files, which is
    comparatively slow. Once built, the fully-loaded dataset (every cached section plus
    the global spot index) can be persisted to a single file with
    :meth:`save_checkpoint` and restored without touching any ``.h5ad`` file via
    :meth:`from_checkpoint`, or in one call via :meth:`build_or_load`.

    Parameters
    ----------
    root:
        Path to the ``GEOPATCH`` project root. ``.h5ad`` files are expected at
        ``<root>/data/her2st/*.h5ad``.
    dtype:
        Floating-point dtype used for the ``"x"``, ``"sp"``, and ``"img"``
        tensors returned by :meth:`__getitem__`. Defaults to ``torch.float32``.

    Raises
    ------
    FileNotFoundError
        If ``<root>/data/her2st`` does not exist, or any of the 36 expected
        ``<section_id>.h5ad`` files is missing from it.
    Her2stSectionValidationError
        If a loaded section is missing a required field.
    """

    def __init__(self, root: Union[str, Path], dtype: torch.dtype = torch.float32) -> None:
        self.root = Path(root)
        self.data_dir = self.root / "data" / "her2st"
        self.dtype = dtype

        if not self.data_dir.is_dir():
            raise FileNotFoundError(
                f"Expected a 'data/her2st' directory at {self.data_dir}, but it does not exist. "
                f"Build it with 'python -m src.datasets.her2st_build'."
            )

        missing = [
            section_id
            for section_id in _SECTION_TO_PATIENT
            if not (self.data_dir / f"{section_id}.h5ad").is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing expected HER2ST .h5ad file(s) in {self.data_dir}: "
                f"{sorted(f'{s}.h5ad' for s in missing)}."
            )

        self._sections: Dict[str, _Section] = {}
        # Global sample index -> (section_id, local_spot_index), built in
        # section order so indexing is stable and constant-time.
        self._index: List[Tuple[str, int]] = []

        for section_id in sorted(_SECTION_TO_PATIENT):
            self._load_section(self.data_dir / f"{section_id}.h5ad", section_id)

    def _load_section(self, path: Path, section_id: str) -> None:
        """Read one ``.h5ad`` file, validate it, and register it in the index."""
        logger.info("Loading HER2ST section '%s' from %s", section_id, path)
        adata = sc.read_h5ad(path)
        self._validate_fields(adata, section_id)

        patient_id = _SECTION_TO_PATIENT[section_id]
        self._sections[section_id] = _Section(
            section_id=section_id, patient_id=patient_id, adata=adata
        )
        self._index.extend((section_id, local_idx) for local_idx in range(adata.n_obs))

    @staticmethod
    def _validate_fields(adata: AnnData, section_id: str) -> None:
        """Verify that ``adata`` carries every field this dataset relies on.

        Raises :class:`Her2stSectionValidationError` with a specific, actionable
        message identifying the section and the missing field.
        """
        if adata.X is None:
            raise Her2stSectionValidationError(
                f"Section '{section_id}': adata.X is missing (no gene expression matrix)."
            )
        if adata.var is None or adata.var.shape[0] == 0:
            raise Her2stSectionValidationError(
                f"Section '{section_id}': adata.var is missing or empty (no gene metadata)."
            )
        for key in _REQUIRED_OBSM_KEYS:
            if key not in adata.obsm:
                hint = (
                    " Run 'python -m src.datasets.her2st_image_features' to generate it."
                    if key == "img_emb"
                    else ""
                )
                raise Her2stSectionValidationError(
                    f"Section '{section_id}': adata.obsm['{key}'] is missing.{hint}"
                )
        for column in _REQUIRED_OBS_COLUMNS:
            if column not in adata.obs.columns:
                raise Her2stSectionValidationError(
                    f"Section '{section_id}': adata.obs['{column}'] is missing."
                )
        for key in _REQUIRED_UNS_KEYS:
            if key not in adata.uns:
                raise Her2stSectionValidationError(
                    f"Section '{section_id}': adata.uns['{key}'] is missing."
                )
        if adata.n_obs == 0:
            raise Her2stSectionValidationError(f"Section '{section_id}': contains zero spots.")

    def __len__(self) -> int:
        """Total number of spots across all 36 sections."""
        return len(self._index)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Return the sample at global index ``index``.

        Resolves ``index`` to ``(section_id, local_spot_index)`` via the
        precomputed global index (O(1)), then reads that single row out of the
        cached, already-loaded :class:`anndata.AnnData` object for that section.
        """
        if not isinstance(index, (int, np.integer)):
            raise TypeError(f"index must be an int, got {type(index).__name__}.")
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self)}.")

        section_id, local_idx = self._index[index]
        section = self._sections[section_id]
        adata = section.adata

        x = self._row_to_dense_array(adata.X, local_idx)
        sp = np.asarray(adata.obsm["spatial"][local_idx], dtype=np.float32).reshape(-1)
        img = np.asarray(adata.obsm["img_emb"][local_idx], dtype=np.float32).reshape(-1)
        gt = adata.obs["ground_truth"].iloc[local_idx]

        return {
            "x": torch.as_tensor(x, dtype=self.dtype),
            "sp": torch.as_tensor(sp, dtype=self.dtype),
            "img": torch.as_tensor(img, dtype=self.dtype),
            "gt": gt,
            "did": section.patient_id,
            "sid": section_id,
            "img_meta": adata.uns["spatial"],
        }

    @staticmethod
    def _row_to_dense_array(x: Any, local_idx: int) -> np.ndarray:
        """Extract one row of ``x`` as a dense, 1-D ``float32`` NumPy array.

        Handles both sparse (``scipy.sparse``) and dense (``numpy.ndarray`` /
        ``numpy.matrix``) representations of ``adata.X``.
        """
        row = x[local_idx]
        if sparse.issparse(row):
            row = row.toarray()
        return np.asarray(row, dtype=np.float32).reshape(-1)

    @property
    def n_genes(self) -> int:
        """Number of genes (features) shared by every section."""
        return next(iter(self._sections.values())).adata.n_vars

    def section_ids(self) -> List[str]:
        """Return the ids of all loaded sections, in load order."""
        return list(self._sections.keys())

    def patient_id(self, section_id: str) -> int:
        """Return the patient id (1-8) for a given section id."""
        return self._sections[section_id].patient_id

    # DLPFC calls this same lookup ``donor_id``; kept as an alias so code written
    # against DlpfcDataset's API runs unchanged against HER2ST.
    donor_id = patient_id

    @staticmethod
    def patient_letter(section_id: str) -> str:
        """Return the patient letter (``"A"``-``"H"``) for a given section id."""
        return section_id[0]

    def get_section(self, section_id: str) -> AnnData:
        """Return the cached :class:`anndata.AnnData` object for one section."""
        return self._sections[section_id].adata

    def gene_metadata(self, section_id: str) -> pd.DataFrame:
        """Return ``adata.var`` (gene metadata) for one section."""
        return self._sections[section_id].adata.var

    def spot_counts(self) -> Dict[str, int]:
        """Return the number of spots in each section, keyed by section id."""
        return {section_id: section.adata.n_obs for section_id, section in self._sections.items()}

    def annotated_indices(self) -> np.ndarray:
        """Return the global indices of every spot that carries a ground-truth label.

        Only the eight sections in :data:`ANNOTATED_SECTIONS` were annotated by a
        pathologist, so this is the subset to use for any supervised or
        label-evaluated task.
        """
        # The global index is built one whole section at a time, in ``section_ids()``
        # order, so each section occupies a contiguous block starting at ``offset``.
        labelled: List[np.ndarray] = []
        offset = 0
        for section_id in self.section_ids():
            adata = self._sections[section_id].adata
            mask = adata.obs["ground_truth"].notna().to_numpy()
            labelled.append(np.flatnonzero(mask) + offset)
            offset += adata.n_obs
        if not labelled:
            return np.zeros(0, dtype=np.int64)
        return np.concatenate(labelled).astype(np.int64)

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """Persist the fully-loaded dataset to a single pickle file.

        Serializes the cached per-section :class:`anndata.AnnData` objects and
        the global spot index, so a later call to :meth:`from_checkpoint` can
        reconstruct an equivalent dataset without re-reading or re-validating
        any ``.h5ad`` file.

        Parameters
        ----------
        path:
            Destination file path. Parent directories are created if needed.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _CHECKPOINT_VERSION,
            "root": str(self.root),
            "dtype": self.dtype,
            "sections": self._sections,
            "index": self._index,
        }
        with path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(
            "Saved Her2stDataset checkpoint (%d sections, %d spots) to %s",
            len(self._sections), len(self), path,
        )

    @classmethod
    def from_checkpoint(
        cls, path: Union[str, Path], dtype: Optional[torch.dtype] = None
    ) -> "Her2stDataset":
        """Reconstruct a :class:`Her2stDataset` from a file written by :meth:`save_checkpoint`.

        This bypasses :meth:`__init__` entirely: no ``.h5ad`` file is read or
        re-validated, so this is the fast path for repeated runs.

        Parameters
        ----------
        path:
            Checkpoint file previously written by :meth:`save_checkpoint`.
        dtype:
            Optional override for the tensor dtype used by :meth:`__getitem__`.
            Defaults to the dtype the checkpoint was saved with.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        ValueError
            If the checkpoint was written by an incompatible schema version.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        with path.open("rb") as f:
            payload = pickle.load(f)

        version = payload.get("version")
        if version != _CHECKPOINT_VERSION:
            raise ValueError(
                f"Unsupported Her2stDataset checkpoint version {version!r} in {path}; "
                f"expected {_CHECKPOINT_VERSION}."
            )

        instance = cls.__new__(cls)
        instance.root = Path(payload["root"])
        instance.data_dir = instance.root / "data" / "her2st"
        instance.dtype = dtype if dtype is not None else payload["dtype"]
        instance._sections = payload["sections"]
        instance._index = payload["index"]
        logger.info(
            "Loaded Her2stDataset checkpoint (%d sections, %d spots) from %s",
            len(instance._sections), len(instance._index), path,
        )
        return instance

    @classmethod
    def build_or_load(
        cls,
        root: Union[str, Path],
        checkpoint_path: Union[str, Path],
        dtype: torch.dtype = torch.float32,
        force_rebuild: bool = False,
    ) -> "Her2stDataset":
        """Load from ``checkpoint_path`` if present, else build and cache it.

        Convenience entry point for the common "build once, reuse many times"
        workflow: if ``checkpoint_path`` exists and ``force_rebuild`` is
        ``False``, delegates to :meth:`from_checkpoint`; otherwise builds a
        fresh dataset from the ``.h5ad`` files under ``root`` (i.e. the normal
        :meth:`__init__` path) and immediately writes it out via
        :meth:`save_checkpoint` so the next call is fast.

        Parameters
        ----------
        root:
            Path to the ``GEOPATCH`` project root, forwarded to ``__init__``
            when a rebuild is needed.
        checkpoint_path:
            File to load from / save to.
        dtype:
            Tensor dtype for :meth:`__getitem__`.
        force_rebuild:
            If ``True``, ignore any existing checkpoint and rebuild from the
            ``.h5ad`` files, overwriting ``checkpoint_path``.
        """
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.is_file() and not force_rebuild:
            return cls.from_checkpoint(checkpoint_path, dtype=dtype)

        instance = cls(root, dtype=dtype)
        instance.save_checkpoint(checkpoint_path)
        return instance

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(root={str(self.root)!r}, "
            f"n_sections={len(self._sections)}, n_spots={len(self)}, n_genes={self.n_genes})"
        )
