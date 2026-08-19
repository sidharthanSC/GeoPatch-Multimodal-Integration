# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`geopatch-crp` — Contrastive Riemannian Prototype (CRP) tissue classification for spatial
transcriptomics, evaluated against the GeoPatch DLPFC and 10x Visium (`v10x`) datasets. The
repo root contains `STAIG.pdf` (Yang et al., *Nature Communications* 2025, "STAIG: Spatial
transcriptomics analysis via image-aided graph contrastive learning for domain exploration and
alignment-free integration") as a reference paper — the CRP approach in this project is in the
same problem space (identifying spatial/histological domains from ST data) and is likely
benchmarked against it or a similar baseline set.

## Current repository state — read this before assuming code exists

**Read `PROGRESS.md` first** — it's the up-to-date, detailed log of the BYOL ->
cross-modal -> multimodal-representation pipeline (`src/gene_encoder/`,
`src/cross_modal/`, `src/multimodal/`) built on top of what's described below,
including key findings (e.g. a BYOL collapse investigation and its fix) and exactly
what's done vs. still open. The section below predates that work and is kept for the
`src/train/`/`src/datasets/` history it still accurately describes, but `src/` is no
longer "mostly a stub" — treat `PROGRESS.md` as the current source of truth for
overall status.

`src/` now exists but is still mostly a stub. What's actually implemented:

- `src/datasets/dlpfc.py` — `DlpfcDataset`, a `torch.utils.data.Dataset` over all twelve DLPFC
  sections (one spot per sample; one `AnnData` cached per section, never concatenated). Supports
  checkpointing the fully-loaded dataset to a single pickle file via `save_checkpoint` /
  `from_checkpoint` / `build_or_load`, since rebuilding from the twelve `.h5ad` files takes ~10s
  and a checkpoint reload takes ~1s. Each sample dict exposes `x`, `sp`, `img`, `gt`, `did`, `sid`,
  `img_meta` — see the class docstring for exactly which `AnnData` fields are read vs. ignored.
- `src/test/test_dlpfc_checkpoint.py` — pytest tests for the above. Tests are colocated under
  `src/test/`, not a top-level `tests/` directory (`pyproject.toml`'s `testpaths` is set
  accordingly). This test is skipped unless `checkpoints/dlpfc.pkl` already exists — build it
  first with `DlpfcDataset.build_or_load(root=".", checkpoint_path="checkpoints/dlpfc.pkl")`.
- `src/datasets/her2st.py` + `her2st_build.py` + `her2st_image_features.py` — the HER2ST
  (HER2-positive breast tumour) counterpart of the above, added so the pipeline has a second,
  non-cortical benchmark. `Her2stDataset` mirrors `DlpfcDataset`'s API, sample dict and
  checkpoint schema exactly (`checkpoints/her2st.pkl`, built the same way via `build_or_load`),
  so DLPFC-facing code runs against it unchanged. Two dataset-level differences: `did` is the
  patient id `1`–`8` (`A`–`H`), and **`gt` is `NaN` for most spots** — a pathologist annotated
  only 8 of the 36 sections (`ANNOTATED_SECTIONS` = A1, B1, C1, D1, E1, F1, G2, H1; 3,481 of
  13,620 spots), so use `annotated_indices()` for anything label-evaluated. The source data is
  a sparse clone of https://github.com/almaan/her2st at `Dataset/her2st_repo/` (raw `.tsv.gz`
  counts, spot files, H&E jpgs, annotations — not 10x Visium, so there is no `spaceranger`
  output to read); `her2st_build.py` turns it into DLPFC-shaped `data/her2st/<section>.h5ad`
  (shared 18,758-gene union index, STAIG's exact `seurat_v3` HVG → `normalize_total(1e4)` →
  `log1p` → `scale(max_value=10)` preprocessing, raw counts kept in `layers["counts"]`), and
  `her2st_image_features.py` generates `obsm["img_emb"]` by porting STAIG's BYOL histology
  pipeline (`example/Fig-3b.ipynb`), whose recipe was verified by reproducing the DLPFC
  `img_emb` from `Dataset/DLPFC/151507/embeddings.npy`. Both modules' docstrings list the
  deviations from STAIG; the load-bearing one is `--batch-size 8` (STAIG's 43 needs ~20 GB and
  swaps a 16 GB Mac to a standstill).
- `src/test/test_her2st_checkpoint.py` — pytest tests for `Her2stDataset`, skipped unless
  `checkpoints/her2st.pkl` exists.
- `src/train/` — a second-stage projection network trained on top of the frozen `img` (BYOL
  image-encoder) embeddings from `DlpfcDataset`, not on raw expression. Same-vs-different pairs
  are sampled in a class-balanced way and pulled together / pushed apart by a margin-based cosine
  loss with an EMA-updated target branch. Split across:
  - `model.py` — the projector/predictor network(s) and the loss module.
  - `pairs.py` — `BalancedPairSampler`, balanced same-vs-different pair sampling from any 1-D
    label array (label-agnostic by design).
  - `train.py` — CLI entry point (`python -m src.train.train`), training/validation loop,
    per-epoch checkpointing, and export of the trained projector + projected embeddings.

  The pairing key is ground-truth cortical layer (`adata.obs["ground_truth"]`,
  `Layer_1`..`Layer_6`/`WM`), passed into `BalancedPairSampler` — same-layer pairs get `eta=0`,
  different-layer pairs get `eta=1`. Tissue section id (`slice_ids`) is loaded and exported
  alongside as metadata only; it is **not** the pairing key. This was a real bug for one run: an
  earlier version of `train.py` defined its own duplicate inline sampler class and wired it to
  `slice_ids` instead of `cortical_labels`, so that run silently trained same-*section*/
  different-*section* separation instead of same-*layer*/different-*layer* separation — caught
  only by a k-NN-vs-ground-truth accuracy check (see `src/analysis/`) showing accuracy get
  *worse* after projection. Output artifacts from that run were named `slice_projector`; the
  corrected pairing key produces `layer_projector`. If you see `slice_projector`-named files
  anywhere (e.g. under `outputs/prelim/`), they're from the pre-fix run — don't treat them as
  the canonical result.

  **This module is under active, hands-on iteration** — model/loss class names and `TrainConfig`
  fields have changed multiple times across experiments, sometimes edited directly rather than
  through a request in chat. Don't assume any specific class or field name from memory;
  `grep`/read `src/train/model.py` and `src/train/train.py` fresh before relying on their
  contents.
- `src/plots/plot_metrics.py` — reads a `src/train/train.py` metrics CSV and renders one PNG per
  group of scale-comparable metrics (loss; same/diff-layer similarity; similarity gap; margin
  violation rate), train and validation overlaid on the same axes. `python -m src.plots.plot_metrics
  [--metrics-csv PATH]`; output filenames are derived from the input CSV's name, so it works
  unmodified against any run.
- `src/analysis/knn_projection_analysis.py` — quantifies what the projection changed: for every
  spot, a leave-one-out k-NN vote (self excluded) over ground-truth layer gives a "neighborhood
  label" in raw-embedding space and in projected-embedding space; spots where the two disagree,
  plus their barcodes, are written out. This is the check that caught the `slice_projector`
  pairing-key bug above. `python -m src.analysis.knn_projection_analysis [--embeddings-npz PATH]`.
- `checkpoints/` — git-ignored, holds `dlpfc.pkl` (~5 GB; contains the full cached `AnnData`
  objects, not just a handful of arrays) and `her2st.pkl` (~2 GB, same shape). Regenerate rather
  than expect either to be present in a fresh checkout.
- `outputs/` — git-ignored (`outputs/checkpoints/`, `outputs/metrics/`, `outputs/predictions/`,
  `outputs/plots/`, `outputs/analysis/`, plus `outputs/runs|tables|figures|splits/`), populated by
  the scripts above:
  `outputs/checkpoints/<run_name>/{best,last,projector_final}.pt`,
  `outputs/metrics/<run_name>_metrics.csv` (one row per epoch),
  `outputs/predictions/<run_name>_embeddings.npz` (projected embeddings for every spot, exported
  from the best/final checkpoint), `outputs/plots/<run_name>_*.png`,
  `outputs/analysis/{knn_projection_summary.json, knn_label_transitions.csv, changed_spots.csv,
  knn_predictions_full.csv}` (not run-name-namespaced — regenerating overwrites the previous
  analysis, so it always reflects whichever `.npz` it was last pointed at). `<run_name>` is
  currently `layer_projector` for the canonical run. Superseded experiment outputs get moved
  under `outputs/prelim/` (same subdirectory shape) rather than deleted, so past runs stay
  available for comparison — check there before assuming a past run's output is gone, but also
  don't assume everything under `outputs/checkpoints|metrics|predictions/` is current — verify
  against `src/train/train.py`'s actual output paths.

Everything else referenced in `pyproject.toml`'s intended layout (`packages.find` under `src*`)
is not yet built. A stale `.pytest_cache/v/cache/nodeids` file (last touched 2026-07-23,
predating the current `pyproject.toml`) lists a much larger set of test node IDs from a previous
state of the project — the only surviving record of that intended module structure. Treat it as
a design reference, not as code that currently exists; verify on the filesystem before relying on
any of it:

- `src/spd_ops.py` — core SPD/Riemannian geometry primitives: matrix log/exp, sqrt/inv-sqrt,
  symmetrize, `svec`, AIRM distance, log-Euclidean distance, batched vs. unbatched variants, with
  explicit MPS (Apple Silicon GPU) vs. CPU parity tests.
- `src/covariance.py` — turns per-spot/per-patch features into (regularized) SPD covariance
  matrices; includes shrinkage-alpha selection (`recommend_alpha`) for when `n` patches is small
  relative to feature dim `d`.
- `src/prototypes.py` — Riemannian prototype classifier: per-class prototypes, distance-based
  prediction, state-dict save/reload with reproducible distances.
- `src/supcon_loss.py` — supervised contrastive (SupCon) loss blended with a "PMDM"
  (prototype/minimum-distance-to-mean) loss term via a `lambda_c` weight, plus a balanced
  per-class batch sampler.
- `src/byol_encoder.py` — BYOL-style histology image encoder with two modes (frozen "mode A" /
  fine-tunable "mode B" with selective block-unfreezing) and provenance recording for
  reproducibility.
- `src/trainer.py` — two-stage training pipeline (Stage A / Stage B), auto-resolved `tau_p`,
  per-epoch geometric diagnostics, and `save_training_artifacts` (writes 7 required output files).
- `src/registry.py` — dataset registry that is expected to match the contents of `Dataset/`
  exactly (primary vs. secondary sample sets).
- `src/baselines/` (or similar) — a comparison suite: Euclidean prototype, frozen/fine-tuned
  mean-pool, frozen covariance-MDM, log-covariance + logistic regression, SPDNet (BiMap + ReEig
  layers), GCN, and GAT baselines, each sharing a common synthetic-task test harness.

## Setup and commands

A `.venv` already exists in the repo (Python 3.11). Third-party versions are pinned in
`requirements-lock.txt` for reference, but install the project itself via the extras in
`pyproject.toml`, not by pointing pip at the lock file:

```bash
pip install -e ".[dev]"
```

```bash
pytest                                                          # full suite (testpaths = src/test)
pytest src/test/test_dlpfc_checkpoint.py                        # one file
pytest src/test/test_dlpfc_checkpoint.py::test_checkpoint_reports_all_twelve_sections  # one test
pytest --cov                                                     # with coverage (pytest-cov is a dev dependency)
```

Run `src/train/` scripts as modules from the repo root, not as bare scripts, so `src` resolves
as a package (`pyproject.toml` also sets `pythonpath = ["."]` for the same reason under pytest):

```bash
python -m src.train.train                       # full training run, default TrainConfig
python -m src.train.train --num-epochs 1 --steps-per-epoch 5 --output-dir /tmp/smoke  # smoke test
python -m src.plots.plot_metrics                 # plot outputs/metrics/layer_projector_metrics.csv
python -m src.analysis.knn_projection_analysis   # before/after k-NN comparison, see src/analysis/ above
```

Rebuilding HER2ST from scratch (only needed if `data/her2st/` or `checkpoints/her2st.pkl` is
missing — the second step trains BYOL per section and takes ~45 min on an M-series Mac):

```bash
git clone --filter=blob:none --sparse https://github.com/almaan/her2st.git Dataset/her2st_repo
(cd Dataset/her2st_repo && git sparse-checkout set data/ST-cnts data/ST-imgs data/ST-spotfiles data/ST-pat/lbl)
python -m src.datasets.her2st_build              # -> data/her2st/<section>.h5ad
python -m src.datasets.her2st_image_features     # -> obsm["img_emb"], Dataset/her2st/<section>/embeddings.npy
python -c "from src.datasets.her2st import Her2stDataset; Her2stDataset.build_or_load(root='.', checkpoint_path='checkpoints/her2st.pkl')"
```

`TrainConfig` fields are exposed 1:1 as CLI flags (kebab-case, e.g. `--margin`, `--ema-decay`,
`--negative-margin` — the exact set depends on the current contents of `src/train/train.py`).

## Data layout

Two separate data trees exist and serve different purposes:

- `data/` — flat directory of standalone `.h5ad` (AnnData) files: the 12 DLPFC Visium slides
  (`151507`–`151676`), `Human_Breast_Cancer`, `Mouse_Brain_Anterior`/`Posterior`,
  `Mouse_Hippocampus_Tissue_slide-seqV2` (split into parts), `Mouse_Olfactory_Stereo-seq`,
  `Mouse_horizontal`, `starmap`, plus a few pre-built cross-slide `integration*.h5ad` /
  `partial_integration.h5ad` files. Its one subdirectory, `data/her2st/`, holds the 36 HER2ST
  sections (`A1`–`H3`) built by `src/datasets/her2st_build.py` — these are generated, not
  supplied, and are git-ignored along with the rest of `data/`.
- `Dataset/` — richer, per-sample directories, organized by platform (git-ignored — large
  binaries, not tracked):
  - `Dataset/DLPFC/<slide_id>/` — `filtered_feature_bc_matrix.h5`, `spatial/`, `truth.txt`
    (ground-truth domain labels), and (for some slides) precomputed `embeddings.npy` / `model.pt`
    / `imgmodel` artifacts from prior runs. `Dataset/DLPFC/DLPFC_annotations/` holds the same
    truth labels flattened into one file per slide.
  - `Dataset/v10x/<sample>/` — 10x Visium samples (`Human_Breast_Cancer`, `Mouse_Brain_Anterior`,
    `Mouse_Brain_Coronal`, `Mouse_Brain_Posterior`, `WS_PLA_*`, `GSM4838131_Visium_A/B`), same
    shape as the DLPFC entries.
  - `Dataset/other/` — additional platforms used for cross-platform evaluation: MERFISH,
    Stereo-seq, Slide-seqV2, STARmap, each with platform-specific raw/processed files.
  - `Dataset/her2st_repo/` — sparse clone of the upstream `almaan/her2st` repository (the
    *source* of HER2ST: `data/ST-cnts/`, `ST-spotfiles/`, `ST-imgs/`, `ST-pat/lbl/`), and
    `Dataset/her2st/<section>/` — its *derived* artifacts, laid out like the DLPFC entries:
    `embeddings.npy` (2,048-d BYOL features) plus the `clip_image_filter/` patches they came
    from. Both are regenerated by the two `src/datasets/her2st_*.py` scripts.

When `Dataset/DLPFC/<slide>/embeddings.npy` or `model.pt` already exists, treat it as a cached
artifact from a prior run, not something to regenerate unless the task requires it.
