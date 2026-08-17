# Rich Gene Expression Execution Ledger

## Objective

Test whether the prior gene BYOL pipeline lost cortical-domain information through both 3,000-HVG selection and its invariance-only representation objective. The superseding study preserves expression content with masked reconstruction, covariance decorrelation, expression-geometry preservation, a fixed expression-SVD residual path, and weak spatial consistency.

## Input Conditions

| ID | Input genes | Purpose | Status |
|---|---:|---|---|
| G-All | 33,538 common genes | Canonical complete expression space | COMPLETED |
| G-4096 | 4,096 joint batch-aware HVGs | Preliminary input-budget control | COMPLETED, NOT PRIMARY |
| G-3000 | 3,000 joint batch-aware HVGs | Historical input-budget control | COMPLETED, NOT PRIMARY |

All 12 sections contain the same 33,538 genes in the same native order. `adata.X` is log1p-normalized and sparse; no raw-count likelihood is claimed.

## Encoder

```text
33,538 genes -> 4096 -> 1024 -> 512 -> 128
```

- Hidden blocks: `Linear -> LayerNorm -> GELU -> Dropout`.
- Final output: 128 dimensions for fixed-image alignment.
- Fixed expression residual: train-free TruncatedSVD-128 components initialize and freeze an input-to-128 skip path; its normalized output is added to the nonlinear final branch.
- Every stage is exported separately and never concatenated for the primary stage comparison.
- Sparse expression remains on CPU; only requested rows are densified on the accelerator.

## Views And Loss

Two independently sampled expression views replace the deterministic neighbor-mean BYOL view. Each masks genes to zero and adds low-amplitude noise only to retained expressed entries.

```text
total = BYOL
      + reconstruction_weight * masked weighted-Huber reconstruction
      + variance_weight * stage-wise variance floors
      + covariance_weight * stage-wise sampled correlation decorrelation
      + geometry_weight * expression-SVD cosine-geometry preservation
      + spatial_weight * weak random-neighbor consistency
```

- Reconstruction is computed only at masked loci from the exported final 128-dimensional representation through a separate linear decoder.
- Nonzero masked targets receive explicit extra weight so sparse zeros cannot dominate.
- Variance, covariance, and geometry losses supervise all `4096/1024/512/128` representations.
- Covariance uses a fixed-size sampled coordinate subset for wide stages; a full 4,096-square covariance per batch is computationally unnecessary.
- Spatial consistency is a weak auxiliary term against one randomly sampled coordinate neighbor, not a replacement expression view.
- No cortical labels enter training or checkpoint selection.

## Cross-Modal Alignment

The canonical final gene-128 artifact is trained with one ratio-1 GBSSA/CLIP alignment against fixed `img_emb`. The gene projection receives masked expression reconstruction plus expression-geometry, variance, and covariance regularization. The decoder consumes the normalized gene projection so information hidden only in discarded vector norm cannot satisfy reconstruction.

## Evaluation Matrix

The feasible canonical evaluation contains:

- Direct PCA-128 tied-GMM controls for all-gene stages `4096/1024/512/128`, before and after 15-neighbor refinement.
- Ratio-1 alignment diagnostics and direct aligned gene/image/bisector/concatenation controls.
- Three independent STAIG runs: all-gene stage 4,096, all-gene final 128, and aligned final-128 plus aligned image guidance.

The proposed 21-condition matrix was cancelled as an unnecessary expansion. Published STAIG and SpaGCN remain native-pipeline context only.

## Artifact Policy

The earlier `gene_byol_multistage_1024_512_256_128_seed0` and S1-S6 artifacts remain immutable and are marked superseded, not overwritten. The canonical run is `rich_gene_v2_all33538_seed0`. `observations/MULTISCALE_GENE_STAIG_BENCHMARK.md` leads with the superseding results while retaining the prior ablation as historical evidence.

## Phases

### Phase 1: APIs And Tests

Status: `COMPLETED`

- [x] Sparse common-gene row store and independent masked views.
- [x] Rich encoder, fixed SVD residual, decoder, and all auxiliary losses.
- [x] Exact BYOL online-feature capture without regenerating views.
- [x] Reconstruction-regularized cross-modal objective.
- [x] Namespaced multi-source evaluation.
- [x] Checkpoint-independent tests.

### Phase 2: Input Preparation

Status: `COMPLETED`

- [x] Export the ordered 33,538 common-gene list.
- [x] Compute and export 4,096 joint batch-aware HVGs.
- [x] Validate the existing 3,000-HVG artifact.

The independently selected 3,000- and 4,096-HVG sets overlap in 2,695 genes. They are not nested because Scanpy's batch-aware selection recomputes each section's top-gene flags at the requested cutoff. This is an input-budget comparison, not a strict add-1,096-genes intervention.

### Phase 3: Gene Training

Status: `COMPLETED`

- [x] Train G-All seed 0.
- [x] Train preliminary G-4096 seed 0.
- [x] Train preliminary G-3000 seed 0.
- [x] Export and diagnose canonical `4096/1024/512/128` stages.

Canonical G-All used batch size 512, LR `4e-5`, a 50-epoch budget, and patience 10. It selected epoch 27 by validation total loss `1.29656` and stopped at epoch 37. Final validation embedding standard deviation was `0.99357` at the selected checkpoint.

### Phase 4: Cross-Modal Alignment

Status: `COMPLETED`

- [x] Train canonical reconstruction-regularized ratio-1 alignment.
- [x] Verify finite, non-contracted outputs and complete identities.

Run `rich_gene_v2_all33538_gbssa_r01_seed0` selected epoch 40 and stopped at epoch 55. Median per-section diagnostics: Recall@10 `0.07209`, FOSCTTM `0.11712`, cosine gap `0.11104`, gene/image effective rank `45.22/87.00`.

### Phase 5: STAIG And Controls

Status: `COMPLETED`

- [x] Run three targeted independent all-section STAIG conditions.
- [x] Run dimension-controlled direct clustering.
- [x] Record all 12 section values in the run artifacts.

Median refined direct ARI is `0.44860/0.42932/0.44446/0.43236` for stages `4096/1024/512/128`. Median refined STAIG ARI is `0.52293` for stage 4,096, `0.44533` for final 128, and `0.49365` for aligned final-128. The stage-4,096 result essentially matches the expression-input STAIG control (`0.51821`) under the repository protocol.

### Phase 6: Reporting And Verification

Status: `COMPLETED`

- [x] Supersede the prior ablation interpretation in `observations/MULTISCALE_GENE_STAIG_BENCHMARK.md`.
- [x] Update all canonical ledgers and output indexes.
- [x] Run the full test suite and artifact audit.

## Checkpoint Log

| Date | Phase | Status | Evidence |
|---|---|---|---|
| 2026-08-15 | Planning | COMPLETED | Three input conditions, six stages, rich loss, alignment, and 21-condition STAIG matrix fixed |
| 2026-08-15 | Phase 1 | ACTIVE | Sparse rich-gene implementation started |
| 2026-08-15 | Phase 1 | COMPLETED | 16 focused tests passed; rich BYOL, rich GBSSA, matrix runner, and evaluator verified |
| 2026-08-15 | Phase 2 | ACTIVE | Joint 4,096-HVG selection and input validation started |
| 2026-08-15 | Scope correction | COMPLETED | Dropped unnecessary 21-run matrix; fixed canonical all-gene `33538 -> 4096 -> 1024 -> 512 -> 128` study |
| 2026-08-15 | Phase 2 | COMPLETED | All 33,538 genes share exact section order; 4,096/3,000 controls validated |
| 2026-08-15 | Phase 3 | COMPLETED | All-gene best epoch 27; four stages exported for 47,329 spots |
| 2026-08-15 | Phase 4 | COMPLETED | Rich ratio-1 alignment substantially improves retrieval and effective rank |
| 2026-08-15 | Phase 5 | COMPLETED | Three targeted STAIG runs complete; stage-4,096 median refined ARI 0.52293 |
| 2026-08-15 | Phase 6 | ACTIVE | Canonical reports and verification in progress |
| 2026-08-15 | Phase 6 | COMPLETED | Full suite 51 passed; diff check clean; each targeted STAIG run has 12 checkpoints and embeddings |
