# Multi-Scale Gene STAIG Execution Ledger

## Objective

Test whether intermediate representations from one independently trained gene BYOL encoder improve STAIG-style spatial domain discovery when used separately as graph-node features.

The gene encoder is:

```text
3,000 shared HVGs -> 1,024 -> 512 -> 256 -> 128.
```

BYOL and encoder-output variance regularization optimize only the final 128-dimensional output. Post-block 1,024-, 512-, and 256-dimensional activations and the final 128-dimensional output are exported for every spot. They are evaluated separately and are never concatenated.

The fixed external 128-dimensional `img_emb` remains the image source. Raw image BYOL cannot be retrained because the repository lacks the source patches and image-training pipeline.

## Fixed Decisions

- One representation seed: `0`.
- At most two cross-modal objective variants: GBSSA ratios `1.0` and `1.1`.
- Shared input space: the existing 3,000 joint batch-aware HVGs.
- Hidden stage contract: output after `Linear -> LayerNorm -> ReLU`; dropout is disabled during export.
- Final stage contract: raw 128-dimensional final linear output.
- BYOL projector/predictor outputs are training-only and are not node features.
- Intermediate features remain immutable NPZ arrays; do not enlarge `checkpoints/dlpfc.pkl`.
- Match spots by `(section_id, barcode)`, never barcode alone.
- Coordinates define the STAIG 5-NN graph. Image features guide edge deletion and pseudo-label debiasing. Gene features are node inputs, not base-adjacency inputs.
- Each gene depth receives an independent STAIG model. No cross-depth concatenation or multi-branch fusion.
- Primary evaluation is label-count-informed per-section unsupervised clustering; frozen probes are secondary.
- Never overwrite an existing run directory.

## Experiment Matrix

| ID | STAIG node feature | Image guidance | Status |
|---|---|---|---|
| S0 | Section-specific 3,000-HVG expression | Raw `img_emb` | Existing canonical control |
| S1 | Gene encoder stage 1,024 | Raw `img_emb` | COMPLETED; median refined ARI 0.19672 |
| S2 | Gene encoder stage 512 | Raw `img_emb` | COMPLETED; median refined ARI 0.20898 |
| S3 | Gene encoder stage 256 | Raw `img_emb` | COMPLETED; median refined ARI 0.21495 |
| S4 | Gene encoder final 128 | Raw `img_emb` | COMPLETED; median refined ARI 0.21885 |
| S5 | GBSSA ratio-1 projected gene 128 | Ratio-1 projected image 128 | COMPLETED; median refined ARI 0.20331 |
| S6 | GBSSA ratio-1.1 projected gene 128 | Ratio-1.1 projected image 128 | COMPLETED NEGATIVE; median refined ARI 0.17796 |

Every STAIG row keeps coordinate `k=5`, image PCA-16, 40 image pseudo-clusters, 10%/10% feature masking, one 64-dimensional GCN layer, temperature 10, 400 epochs, seed 0, tied-covariance GMM, and 15-neighbor refinement.

## Phase 1: APIs And Contracts

Status: `COMPLETED`

- [x] Implement named gene stages `1024/512/256/128`.
- [x] Preserve tensor-valued `GeneEncoder.forward()` for BYOL compatibility.
- [x] Add deterministic multi-stage export with schema and architecture metadata.
- [x] Add validated external-NPZ loading for cross-modal training.
- [x] Add validated precomputed-node-feature loading for STAIG.
- [x] Persist source key, dimension, upstream checkpoint, identities, and configuration.
- [x] Add checkpoint-independent tests.

Completion gate: focused tests pass and malformed identities/dimensions are rejected.

## Phase 2: Multi-Stage Gene BYOL

Status: `COMPLETED`

Planned run: `gene_byol_multistage_1024_512_256_128_seed0`.

- [x] Train with explicit seed 0 and existing shared-HVG/spatial-view artifacts.
- [x] Select the checkpoint by validation total loss only.
- [x] Verify final-output effective rank, per-dimension std, and finite values.
- [x] Export all four encoder stages for all spots.
- [x] Verify exact `(section_id, barcode)` coverage and dimensions.

Run `gene_byol_multistage_1024_512_256_128_seed0` selected epoch 199 with validation total loss `0.443116`. The immutable export contains all 47,329 compound spot identities and arrays with dimensions 1,024/512/256/128. The final output is finite and non-collapsed (effective rank `10.7142`, mean/min dimension standard deviation `1.00016/0.68950`). ReLU hidden stages contain some globally inactive coordinates; this is retained and reported rather than filtered.

Completion gate: non-collapsed final 128 and complete immutable prediction artifact.

## Phase 3: Final-128 Cross-Modal Alignment

Status: `COMPLETED`

- [x] Train GBSSA ratio 1 using final gene-128 and fixed `img_emb`.
- [x] Train GBSSA ratio 1.1 as the requested lower-relative-negative-pressure sensitivity.
- [x] Evaluate retrieval, FOSCTTM, cosine gap, standard deviation, and effective rank.
- [x] Retain collapsed outputs as negative results; do not silently promote them.

Accepted run `multistage_gene128_gbssa_r01_seed0_v2` selected epoch 34 (`val_loss=5.761239`). Median per-section diagnostics are Recall@10 `0.01060`, FOSCTTM `0.30975`, cosine gap `0.05355`, gene/image effective rank `7.22/33.96`, and mean dimension standard deviation `0.04123/0.05640`.

Negative ablation `multistage_gene128_gbssa_r011_seed0_v2` selected epoch 51 (`val_loss=5.869197`). It has a smaller cosine gap (`0.01615`) and contracted gene/image outputs (effective rank `3.34/3.85`, mean dimension standard deviation `0.01676/0.01450`). It remains eligible only for S6 as an explicit collapsed sensitivity, not as a promoted alignment.

The first runs without the `_v2` suffix are invalid provenance artifacts: a fixed-width NumPy string allocation truncated every section ID to `"1"`. The loader now uses object strings and a regression test uses real six-character DLPFC section IDs. Invalid artifacts remain immutable and must not be evaluated or consumed downstream.

Completion gate: both runs have immutable checkpoints, embeddings, diagnostics, and explicit acceptance/rejection decisions.

## Phase 4: STAIG Node-Feature Study

Status: `COMPLETED`

- [x] Run S1-S4 independently across all 12 sections.
- [x] Run S5-S6 independently if their alignment artifacts are finite; preserve negative outcomes even when collapse diagnostics fail.
- [x] Save 12 section checkpoints and embedding files per row.
- [x] Record complete node/image feature provenance.

S1-S4 median refined ARI values increase modestly with depth (`0.19672`, `0.20898`, `0.21495`, `0.21885`) but all substantially underperform the canonical expression-input STAIG value (`0.51821`). Ratio-1 alignment decreases final-128 STAIG median refined ARI from `0.21885` to `0.20331`; collapsed ratio-1.1 decreases it further to `0.17796`. These runs do not support replacing expression with the tested BYOL stages or aligned final output in STAIG.

The first S4 attempt (`staig_multistage_gene128_img_emb_seed0_all12_v1`) stopped after training section 151507 because sklearn's float32 tied covariance was judged non-positive-definite. It is a partial, invalid run. `tied_gmm()` now converts embeddings to float64 before fitting while retaining tied covariance, seed, and `reg_covar=1e-6`; S4 was restarted immutably as `_v2`.

Completion gate: every planned row has all section artifacts or a documented blocker.

## Phase 5: Evaluation

Status: `COMPLETED`

- [x] Native tied-GMM ARI/NMI before and after 15-neighbor refinement.
- [x] Direct tied-GMM controls on each raw gene stage.
- [x] Dimension-matched PCA-128 frozen probes for raw gene stages.
- [x] Within-section frozen probes for independent STAIG embeddings.
- [x] Compare all 12 section values, mean, median, and IQR.
- [x] Compare against canonical expression STAIG, published STAIG, and SpaGCN rows.

Completion gate: generated evaluation files contain no label leakage into representation training or checkpoint selection.

## Phase 6: Documentation And Verification

Status: `COMPLETED`

- [x] Create `observations/MULTISCALE_GENE_STAIG_BENCHMARK.md`.
- [x] Update `MODEL_BENCHMARK_REPORT.md`.
- [x] Update `observations/MULTIMODAL_EVALUATION_RESULTS.md`.
- [x] Update `EXECUTION_CHECKPOINTS.md`.
- [x] Update `outputs/README.md` and `src/prior_models/README.md`.
- [x] Run the full test suite and `git diff --check`.
- [x] Verify every documented artifact exists and can be loaded.

Completion gate: all ledgers agree on run IDs, configurations, results, limitations, and artifact paths.

## Checkpoint Log

| Date | Phase | Status | Evidence |
|---|---|---|---|
| 2026-08-15 | Planning | COMPLETED | Architecture, experiment matrix, controls, and immutable artifact contract approved |
| 2026-08-15 | Phase 1 | ACTIVE | Implementation started |
| 2026-08-15 | Phase 1 | COMPLETED | Multi-stage encoder and validated NPZ boundaries; 15 focused tests passed |
| 2026-08-15 | Phase 2 | ACTIVE | Seed-0 multi-stage gene BYOL launch prepared |
| 2026-08-15 | Phase 2 | COMPLETED | Best epoch 199; complete 47,329-spot four-depth export; final effective rank 10.7142 |
| 2026-08-15 | Phase 3 | INVALIDATED | Initial alignment exports had truncated section IDs; artifacts retained but rejected |
| 2026-08-15 | Phase 3 | COMPLETED | Corrected `_v2` runs evaluated; ratio 1 accepted, ratio 1.1 retained as collapsed negative |
| 2026-08-15 | Phase 4 | ACTIVE | S1-S6 inputs validated; S1-S4 launch next |
| 2026-08-15 | Phase 4 | PARTIAL | S4 `_v1` stopped at first-section tied-GMM numerical failure; run retained as invalid partial artifact |
| 2026-08-15 | Phase 4 | COMPLETED | S1-S6 each produced 12 checkpoints/embeddings; all underperform canonical expression STAIG |
| 2026-08-15 | Phase 5 | ACTIVE | Direct clustering controls and frozen probes next |
| 2026-08-15 | Phase 5 | COMPLETED | Corrected `_v2` controls separate refinements and use convergent matched probes |
| 2026-08-15 | Phase 6 | ACTIVE | Reports and artifact indexes updated; final verification in progress |
| 2026-08-15 | Phase 6 | COMPLETED | Full suite 44 passed; diff check clean; all S1-S6 runs have 12 checkpoints and 12 embedding files |
