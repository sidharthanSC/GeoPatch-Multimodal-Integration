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
  neighbor-smoothing views), producing `gene_emb` for the eventual cross-modal
  (image <-> gene expression) contrastive step.
- `src/plots/`, `src/analysis/` — metric plots and downstream analysis (k-NN
  before/after comparisons, Riemannian covariance/prototype construction) of trained
  embeddings.
- `outputs/`, `checkpoints/` — git-ignored, regenerable run artifacts; `outputs/README.md`
  indexes what each run produced.

`data/` and `Dataset/` (raw input data) are git-ignored — too large for normal git and
tracked separately. See `CLAUDE.md` for full architecture notes, current implementation
status, and the project's history of design decisions.
