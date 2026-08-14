# geopatch-crp

Contrastive Riemannian Prototype (CRP) tissue classification for spatial
transcriptomics, evaluated on the GeoPatch DLPFC and 10x Visium (`v10x`) datasets.

## Setup

A `.venv` (Python 3.11) is expected locally. Install the project via its extras, not
`requirements-lock.txt` directly:

```bash
pip install -e ".[dev]"
```

## Tests

```bash
pytest
```

## Layout

- `src/datasets/` — `DlpfcDataset`, checkpointed to `checkpoints/dlpfc.pkl` (git-ignored,
  regenerate from `data/`).
- `src/train/` — BYOL-style projection network trained on frozen histology-image
  embeddings, paired by ground-truth cortical layer.
- `src/gene_encoder/` — BYOL gene-expression encoder (masked-corruption + spatial-
  neighbor-smoothing views), producing `gene_emb`.
- `src/cross_modal/` — CLIP-style symmetric InfoNCE contrastive alignment between
  `gene_emb` and the image-side embedding, producing the `*_cm_*` embedding pairs.
- `src/multimodal/` — four ways of classifying cortical layer from the aligned
  gene/image pair (bisector clustering, per-spot Riemannian covariance clustering,
  Riemannian MDM, cross-attention + self-attention fusion), pooled across all 12
  sections.
- `src/benchmark/` — the per-section evaluation suite from `MODEL_BENCHMARK_REPORT.md`:
  representation ablations (R1-R11) clustered independently per section with
  KMeans/GaussianMixture/Leiden, one common spatial-refinement pass, and a supervised
  attention-fusion ablation suite (A1-A12).
- `src/plots/`, `src/analysis/` — metric plots and downstream analysis (k-NN
  before/after comparisons, Riemannian covariance/prototype construction) of trained
  embeddings.
- `outputs/`, `checkpoints/` — git-ignored, regenerable run artifacts (except
  `outputs/README.md` and `outputs/benchmark/`, tracked as documented exceptions);
  `outputs/README.md` indexes what each run produced.

`data/` and `Dataset/` (raw input data) are git-ignored — too large for normal git and
tracked separately. See `PROGRESS.md` for the up-to-date build log of the full
BYOL -> cross-modal -> multimodal pipeline, `MODEL_BENCHMARK_REPORT.md` /
`MODEL_BENCHMARK_RESULTS.md` for the per-section benchmark's methodology and findings,
and `CLAUDE.md` for full architecture notes and the project's history of design
decisions.
