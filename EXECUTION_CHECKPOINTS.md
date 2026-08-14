# GBSSA Training and Evaluation Checkpoints

## Purpose

This file is the resumable execution ledger for the training and evaluation program defined by `MODEL_BENCHMARK_REPORT.md`. Read this file after `AGENTS.md` and before resuming work. Update a phase only after its artifacts and verification are complete.

Status values: `PENDING`, `ACTIVE`, `BLOCKED`, `COMPLETED`, `PARTIAL`.

## Fixed Decisions

- Primary path: fixed external `img_emb` + newly trained `gene_emb`; `proj_emb` remains a separate label-assisted ablation.
- Cross-modal model: two symmetric, jointly optimized modality towers; no cross-modal EMA target branch.
- Objective: gradient-balanced same-spot CLIP (GBSSA), retaining all InfoNCE negatives.
- Primary ratio sweep: `1, 2, 5, 10`; ratio `1` is exactly standard symmetric CLIP.
- Initial representation seeds: `0, 1, 2` where compute permits.
- Evaluation authority: `MODEL_BENCHMARK_REPORT.md`.
- Generated result ledger: `MULTIMODAL_EVALUATION_RESULTS.md`.
- Do not edit `CLAUDE.md` or `PROGRESS.md`.
- Do not overwrite `checkpoints/dlpfc.pkl` or a prior named run.

## Environment Snapshot

Recorded 2026-08-13:

- Workspace branch: `SM/geopatch` at `6d101d1`.
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8,188 MiB total VRAM.
- Free workspace drive capacity at audit: approximately 429 GB.
- Base checkpoint: `checkpoints/dlpfc.pkl`, approximately 5.12 GB.
- Raw `data/` and `Dataset/` directories are absent.
- Active system Python was 3.13.5 and lacked `scanpy`, `anndata`, and `byol_pytorch`.
- A Python 3.11 environment must be created before dataset/training execution.

## Phase 0: Documentation And Reproducibility Contract

Status: `COMPLETED`

Deliverables:

- [x] Preserve the evaluation protocol in `MODEL_BENCHMARK_REPORT.md`.
- [x] Create this resumable execution ledger.
- [x] Create `MULTIMODAL_EVALUATION_RESULTS.md` for generated results.
- [x] Link the ledgers from `AGENTS.md`, `README.md`, and `outputs/README.md`.
- [x] Record the first verified environment and test command.

Artifacts:

- `EXECUTION_CHECKPOINTS.md`
- `MODEL_BENCHMARK_REPORT.md`
- `MULTIMODAL_EVALUATION_RESULTS.md`

Completion gate: documentation links exist and `git diff --check` passes.

## Phase 1: GBSSA Objective

Status: `COMPLETED`

Deliverables:

- [x] Preserve `src.cross_modal.model.clip_loss()` unchanged as the ratio-1 baseline.
- [x] Add an importable gradient-balanced same-spot CLIP objective.
- [x] Add `same_spot_gradient_ratio` to cross-modal configuration and CLI.
- [x] Log base CLIP loss, raw/weighted attraction, paired cosine, hardest-negative cosine, cosine margin, and temperature. The requested ratio is stored in run configuration; its exact score-gradient behavior is verified in tests.
- [x] Add checkpoint-independent unit tests for ratio 1, exact score-gradient ratio, unchanged off-diagonal gradients, invalid inputs, and finite batch-size-2 behavior.
- [x] Verify focused and full available tests.

Planned files:

- `src/cross_modal/model.py`
- `src/cross_modal/train.py`
- `src/test/test_cross_modal_model.py`
- `src/test/test_cross_modal_train.py` only if training plumbing requires it

Completion gate: ratio 1 numerically equals current CLIP and ratios 5/10 preserve nonzero, unchanged off-diagonal score gradients in tests.

## Phase 2: Python Environment

Status: `COMPLETED`

Deliverables:

- [x] Install or provision Python 3.11.
- [x] Create `.venv`.
- [x] Add and install the verified `byol-pytorch` dependency.
- [x] Install `pip install -e ".[dev]"`.
- [x] Record Python, PyTorch, CUDA, Scanpy, AnnData, BYOL, NumPy, SciPy, and scikit-learn versions.
- [x] Run checkpoint-independent tests.

Verified environment:

- Python `3.11.15`
- PyTorch `2.8.0+cu126`, TorchVision `0.23.0+cu126`
- CUDA `12.6`, NVIDIA GeForce RTX 4060 Laptop GPU visible
- Scanpy `1.11.5`, AnnData `0.12.19`, scikit-learn `1.9.0`
- `byol-pytorch==0.9.1`
- `pip check`: no broken requirements
- `python -m pytest -q`: 11 passed

Completion gate: project imports succeed under `.venv` and GBSSA tests pass on CPU and available CUDA where applicable.

## Phase 3: Checkpoint Audit And Compact Bundle

Status: `COMPLETED`

Deliverables:

- [x] Load `checkpoints/dlpfc.pkl` read-only.
- [x] Verify 3 donors, 12 sections, exact section mapping, labels, per-section spot counts, and total spots.
- [x] Verify `adata.X`, `obsm["spatial"]`, `obsm["img_emb"]`, and all documented embedding keys.
- [x] Reject missing/non-finite/zero-vector anomalies or record them explicitly.
- [x] Verify unique `(section_id, barcode)` identities.
- [x] Extract a compact float32 evaluation bundle with embeddings and metadata.
- [x] Reproduce the legacy pooled aligned-bisector result.

Verified audit run: `outputs/evaluation/20260813_checkpoint_audit_v1/`

- 47,329 spots, 12 sections, 3 donors, no missing labels.
- All documented embedding arrays have shape `(47329, 128)` after concatenation.
- Legacy pooled metrics reproduced: accuracy `0.3224239`, ARI `0.0904553`, NMI `0.1626040`.
- Sections `151669`-`151672` contain five observed classes (`Layer_3`-`Layer_6`, `WM`), not seven. Per-section `k` must use exact observed valid-label counts.

Planned artifacts:

- `outputs/evaluation/<run_id>/data/dataset_manifest.parquet`
- `outputs/evaluation/<run_id>/data/compact_embeddings.npz`
- `outputs/evaluation/<run_id>/data/embedding_manifest.json`
- `outputs/evaluation/<run_id>/audit.json`

Completion gate: legacy pooled bisector is consistent with accuracy `0.322`, ARI `0.090`, NMI `0.163`, or any discrepancy is explained before proceeding.

## Phase 4: Common Evaluation APIs

Status: `PARTIAL`

Deliverables:

- [x] Add a representation registry for existing-embedding R2-R11 variants. R1/R12-R14 remain pending.
- [x] Add per-section KMeans evaluation with explicit seeds and saved assignments.
- [ ] Add common GMM/mclust-compatible and Leiden sensitivity where dependencies permit.
- [ ] Add one identical label-free spatial refinement for all representations.
- [ ] Add spatial-neighbor agreement and connectedness metrics.
- [ ] Add blockwise full-gallery alignment metrics.
- [ ] Add frozen linear, nearest-centroid, k-NN, and MLP probes.
- [ ] Add complete provenance and long-form result schemas.

Preferred new modules:

- `src/multimodal/representations.py`
- `src/multimodal/benchmark.py`
- `src/cross_modal/evaluate.py`
- `src/splits/dlpfc.py`

Completion gate: existing saved embeddings can produce all primary per-section KMeans and alignment tables without mutating the 5 GB checkpoint.

Verified KMeans runs:

- `outputs/evaluation/20260813_existing_embeddings_kmeans_seed0_v1/`
- `outputs/evaluation/20260813_existing_embeddings_kmeans_10seeds_v1/`

Ten-seed median section-level results: aligned bisector ARI `0.14187`/NMI `0.23837`; raw image ARI `0.15245`/NMI `0.22667`; aligned image ARI `0.14793`/NMI `0.25274`; aligned concatenation ARI `0.13281`/NMI `0.23574`. The current aligned bisector does not beat the strongest single modality.

## Phase 5: Gene BYOL Retraining

Status: `COMPLETED`

Inputs:

- `checkpoints/dlpfc.pkl`
- `adata.X` log1p expression
- `adata.obsm["spatial"]`

Deliverables per seed:

- [x] Recompute 3,000 joint batch-aware HVGs using `flavor="seurat"`.
- [x] Rebuild within-section spatial KNN graphs with `k=6`.
- [x] Train the `3000 -> 512 -> 512 -> 128` GeneEncoder with BYOL and variance regularization.
- [x] Select `best.pt` using validation total loss while monitoring `embed_std`.
- [x] Export unaugmented raw encoder outputs for every spot.
- [x] Save complete configuration, metrics, barcodes, section IDs, and split provenance.

Initial run names:

- `gene_byol_gbssa_seed0`
- `gene_byol_gbssa_seed1`
- `gene_byol_gbssa_seed2`

Artifact template:

- `outputs/gene_encoder/checkpoints/<run>/{best,last}.pt`
- `outputs/gene_encoder/metrics/<run>_metrics.csv`
- `outputs/gene_encoder/predictions/<run>_embeddings.npz`

Completion gate: no collapse, finite validation metrics, exported `(n_spots, 128)` embeddings, and successful `(section_id, barcode)` validation.

Seed-0 verified artifacts:

- `outputs/gene_encoder/shared_hvgs_gbssa.json`
- `outputs/gene_encoder/spatial_knn_graph_gbssa.pkl`
- `outputs/gene_encoder/checkpoints/gene_byol_gbssa_seed0/{best,last}.pt`
- `outputs/gene_encoder/metrics/gene_byol_gbssa_seed0_metrics.csv`
- `outputs/gene_encoder/predictions/gene_byol_gbssa_seed0_embeddings.npz`
- `checkpoints/dlpfc_gbssa_seed0.pkl` (run-specific copy; canonical checkpoint untouched)

Seed 0 early-stopped at epoch 105 (best validation loss `0.1368`), seed 1 at epoch 122 (best `0.1648`), and seed 2 at epoch 117 (best `0.1468`). All remained non-collapsed. Unaligned-bisector median ARI across clustering seeds was `0.16323`, `0.14013`, and `0.13869`; the favorable seed-0 gain over raw image is not representation-seed robust.

## Phase 6: GBSSA Ratio Sweep

Status: `PARTIAL`

Primary image source: `img_emb`.

Run grid:

```text
ratio in {1, 2, 5, 10}
seed in {0, 1, 2}
```

Deliverables per run:

- [ ] Train two symmetric modality towers and learned temperature.
- [ ] Select within-run `best.pt` using validation GBSSA total loss.
- [ ] Export normalized gene and image projections for every spot.
- [ ] Compute full-gallery alignment diagnostics.
- [ ] Compute per-section biological clustering diagnostics without using labels for training or checkpoint selection.
- [ ] Record failures and negative results.

Run-name template:

```text
gbssa_img_r<ratio>_seed<seed>
```

Artifact template:

- `outputs/cross_modal/checkpoints/<run>/{best,last}.pt`
- `outputs/cross_modal/metrics/<run>_metrics.csv`
- `outputs/cross_modal/predictions/<run>_embeddings.npz`
- `outputs/evaluation/<run_id>/alignment/`

Completion gate: all feasible grid cells have either validated artifacts or an explicit failure record. No run is selected solely by layer ARI.

Seed-0 ratio results:

- Ratio 1 learned above-chance retrieval; one-seed bisector median ARI `0.14365`.
- Ratio 2 converged to chance retrieval; bisector median ARI `0.09225`.
- Ratio 5 converged to chance retrieval; bisector median ARI `0.04597`.
- Ratio 10 converged to chance retrieval; bisector median ARI `0.03790`.

The requested 5x/10x exact score-gradient weighting is therefore rejected for the current architecture. Repeating collapsed ratios across seeds is not scientifically justified. Transition ratios 1.1/1.25/1.5 are added as a diagnostic follow-up.

## Phase 7: Representation And Clustering Benchmark

Status: `PARTIAL`

Primary representations:

- R1 gene-expression PCA when expression extraction is available
- R2 `gene_emb`
- R3 `img_emb`
- R4 unaligned bisector
- R5/R6 unaligned concatenation and dimension-matched PCA
- R7/R8 aligned single modalities
- R9 aligned GBSSA bisector
- R10/R11 aligned concatenation and dimension-matched PCA
- R12 shuffled-pair control
- R13 learned scalar-weighted sum
- R14 label-assisted `proj_emb` variants in separate tables

Deliverables:

- [ ] Per-section KMeans, 10 clustering seeds, no refinement.
- [ ] Backend sensitivity.
- [ ] Common refinement table.
- [ ] Spatial-quality table.
- [ ] Pooled stress-test table.
- [ ] All 12 section values plus median/IQR and paired section-level differences.

Completion gate: the central R9 claim is judged against R2/R3, R4, R7/R8, and R11 under identical evaluation.

KMeans, GMM-surrogate, common spatial refinement, and a fixed angular-weight curve are complete for seed 0. Gene weight `0.25`/image weight `0.75` reached median ARI `0.16564`, but is an untuned sensitivity only. R1 expression PCA, R12 shuffled-pair retraining, Leiden, literal R mclust, and robust weighted-fusion selection remain pending.

## Phase 8: Supervised Probes And Attention Ablations

Status: `PARTIAL`

Deliverables:

- [ ] Frozen probe suite on immutable split manifests.
- [ ] Reproduce the legacy attention result where possible.
- [ ] Run A1-A12 from `MODEL_BENCHMARK_REPORT.md`.
- [ ] Match parameter counts for the concatenation MLP control.
- [ ] Save predictions, confusion matrices, label mapping, best epoch, history, and checkpoints.
- [ ] Report accuracy, balanced accuracy, macro-F1, and per-layer recall.

Completion gate: attention-specific benefit is claimed only if the full model beats the parameter-matched concatenation MLP under identical splits and seeds.

One-seed frozen logistic gate completed at `outputs/evaluation/20260814_seed0_random_spot_logistic_probe_gate_v2/`. Aligned concatenation achieved accuracy `0.5331`, balanced accuracy `0.4167`, and macro-F1 `0.4151`; aligned bisector reached `0.5245`/`0.4015`/`0.3968`. The first five-seed `lbfgs` configuration timed out after 20 minutes without artifacts and is recorded as failed. Attention A1-A12 and additional seeds remain pending.

Attention gate: parameter-matched concatenation MLP accuracy `0.52997`; self-attention-only `0.61251`/`0.60666` on seeds 0/1; full cross+self-attention `0.60532`/`0.60159`. Self-attention wins both seeds, so the length-one cross blocks are rejected as unnecessary. Additional modality-shuffle/dropout controls remain pending.

## Phase 9: Integration

Status: `PARTIAL`

Subsets:

- `151675 + 151676`
- `151507 + 151508 + 151675 + 151676`
- all 12 sections as a separate repository-specific experiment

Deliverables:

- [ ] ARI/NMI and per-section conservation.
- [ ] BatchKL and iLISI with exact implementation provenance.
- [ ] Per-layer section/donor mixing.
- [ ] Spatial connectedness.
- [ ] Section/donor predictability.

Completion gate: biological conservation and batch mixing are reported together; no batch-only metric is treated as success.

Completed the two STAIG-matched section subsets with common KMeans and local 30NN mixing surrogates. Same-donor aligned-bisector ARI `0.1120`; four-section unaligned-concatenation ARI `0.1509`. Both are far below STAIG's published `0.64` and `0.52`. All-12 integration and validated scIB-compatible mixing metrics remain pending.

All-12 integration v1 exceeded 20 minutes before writing artifacts due repeated global neighbor computation across every representation. Optimize/cache neighborhoods before retrying.

## Phase 10: Donor Generalization

Status: `PENDING`

Deliverables:

- [ ] Post-hoc donor transfer using transductively pretrained saved embeddings, clearly labeled.
- [ ] Shared `(section_id, barcode)` donor-fold manifests.
- [ ] Strict leave-one-donor-out HVG selection and expression statistics.
- [ ] Strict fold-specific gene BYOL and GBSSA training.
- [ ] Frozen held-out donor inference and evaluation.
- [ ] All three donor folds and feasible multi-seed variability.

Completion gate: no test donor contributes to HVGs, augmentation statistics, representation training, checkpoint selection, PCA, or supervised fitting. External image BYOL remains described as a fixed external extractor unless provenance proves otherwise.

## Phase 11: External Baselines

Status: `PENDING`

Deliverables:

- [ ] Run feasible gene-plus-spatial baselines from checkpoint-contained data.
- [ ] Run histology-dependent methods only if raw histology assets become available.
- [ ] Preserve each method's native result separately from the common-embedding evaluation.
- [ ] Never fabricate unavailable results; mark them `NOT RUN` with the blocker.

Completion gate: every reported external number has method version, input modalities, preprocessing, cluster backend/count, refinement, seed, and source artifact.

## Phase 12: Final Reporting

Status: `PENDING`

Deliverables:

- [ ] Fill `MULTIMODAL_EVALUATION_RESULTS.md` from immutable result artifacts.
- [ ] Update `outputs/README.md` with every completed run and artifact path.
- [ ] Update `README.md` commands and status.
- [ ] Update `AGENTS.md` only with stable contracts and canonical artifact locations.
- [ ] Record negative results, failed runs, and unresolved limitations.
- [ ] Run tests and `git diff --check`.

Completion gate: the final report keeps per-section clustering, integration, supervised probes, alignment, and donor generalization in separate comparable tables.

## Run Log

Append concise entries after verified milestones.

| Date | Phase | Status | Run/artifact | Verification or blocker |
|---|---|---|---|---|
| 2026-08-13 | 0 | ACTIVE | `EXECUTION_CHECKPOINTS.md` | Ledger created; documentation linking in progress |
| 2026-08-13 | 0 | COMPLETED | Documentation contract | Links added; `MODEL_BENCHMARK_REPORT.md` retained as fixed protocol |
| 2026-08-13 | 1 | COMPLETED | GBSSA objective and tests | 9 focused tests; ratio 1 equals CLIP and ratios 5/10 preserve off-diagonal gradients |
| 2026-08-13 | 2 | COMPLETED | `.venv` | Python 3.11.15, CUDA PyTorch 2.8.0; full suite 11 passed |
| 2026-08-13 | 3 | ACTIVE | Checkpoint audit | Read-only audit/export implementation started |
| 2026-08-13 | 3 | COMPLETED | `outputs/evaluation/20260813_checkpoint_audit_v1/` | 47,329 spots validated; legacy pooled metrics reproduced exactly |
| 2026-08-13 | 4 | ACTIVE | Common evaluation APIs | Existing-embedding per-section KMeans implementation started |
| 2026-08-13 | 4 | PARTIAL | `20260813_existing_embeddings_kmeans_10seeds_v1` | Existing R2-R11 KMeans complete; other backends/alignment/probes pending |
| 2026-08-13 | 5 | ACTIVE | Gene BYOL seed 0 | Shared-HVG and spatial-graph preparation starting |
| 2026-08-14 | 5 | PARTIAL | `gene_byol_gbssa_seed0` | Best val loss 0.1368; 47,329 non-collapsed embeddings exported and safely attached to run copy |
| 2026-08-14 | 6 | PARTIAL | `gbssa_img_r{01,02,05,10}_seed0` | Ratios >=2 collapsed retrieval and worsened section-level ARI; transition sweep added |
| 2026-08-14 | 5 | COMPLETED | `gene_byol_gbssa_seed{0,1,2}` | Three non-collapsed gene embeddings exported; unaligned fusion gain varied materially by seed |
| 2026-08-14 | 7 | PARTIAL | KMeans/GMM/refinement | Best common refined ARI 0.19937 for seed-0 unaligned bisector; representation-seed median lower |
| 2026-08-14 | 8 | ACTIVE | Frozen probes | Common random-spot probe implementation started |
| 2026-08-14 | 8 | PARTIAL | `20260814_seed0_random_spot_logistic_probe_gate_v2` | Aligned concatenation best linear probe; five-seed slow configuration failed without artifacts |
| 2026-08-14 | 8 | PARTIAL | Attention gate | Self-attention-only beat full cross+self-attention on seeds 0 and 1 |
| 2026-08-14 | 9 | ACTIVE | Integration subsets | Common joint-clustering and mixing evaluator started |
| 2026-08-14 | 9 | PARTIAL | Two STAIG subsets | Alignment improves mixing but biological ARI remains low; local mixing metrics labeled as surrogates |
| 2026-08-14 | 7 | PARTIAL | Weighted angular curve | Seed-0 gene/image weight 0.25/0.75 ARI 0.16564; sensitivity only |
| 2026-08-14 | 9 | BLOCKED | All-12 integration v1 | Timed out after 20 minutes with no artifacts; requires cached neighbors/filtering |
