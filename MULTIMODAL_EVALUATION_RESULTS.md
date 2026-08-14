# GeoPatch Multimodal Evaluation Results

## Status

No new GBSSA training has completed yet. The read-only checkpoint audit and legacy-result reproduction are complete. This file is the generated-results ledger; the fixed protocol and literature context remain in `MODEL_BENCHMARK_REPORT.md`, and execution state remains in `EXECUTION_CHECKPOINTS.md`.

Never copy published reference values into a “repository run” column. Every repository result must link to an immutable artifact under `outputs/evaluation/<run_id>/` and include its representation seed, evaluator seed, split, embedding keys, clustering/probe configuration, and refinement policy.

## Provenance

| Field | Value |
|---|---|
| Dataset | DLPFC, 3 donors, 12 sections |
| Base checkpoint | `checkpoints/dlpfc.pkl` |
| Primary image source | `adata.obsm["img_emb"]`, fixed external BYOL feature |
| Primary gene source | Newly trained gene BYOL, pending |
| Alignment objective | Gradient-balanced same-spot CLIP (GBSSA), pending |
| Evaluation protocol | `MODEL_BENCHMARK_REPORT.md` |
| Execution ledger | `EXECUTION_CHECKPOINTS.md` |

## Legacy Results To Reproduce

These are historical values recorded in repository documentation, not yet reproduced in the new immutable evaluation bundle.

| Method | Regime | Accuracy | ARI | NMI | Status |
|---|---|---:|---:|---:|---|
| Aligned bisector + KMeans(7) | Pooled 12-section clustering | 0.322424 | 0.090455 | 0.162604 | Reproduced: `outputs/evaluation/20260813_checkpoint_audit_v1/legacy_pooled_bisector.json` |
| Log-Euclidean covariance + MiniBatchKMeans | Pooled 12-section clustering | 0.247 | Not recorded | Not recorded | Historical |
| Riemannian MDM | Random-spot supervised test | 0.3496 | Not applicable | Not applicable | Historical |
| Attention fusion | Random-spot supervised test | 0.608 | Not applicable | Not applicable | Historical |

## GBSSA Training Results

### Seed-0 Ratio Gate

| Run | Same-spot gradient ratio | Retrieval behavior | One-seed median bisector ARI | One-seed median bisector NMI | Decision |
|---|---:|---|---:|---:|---|
| `gbssa_img_r01_seed0` | 1 | Above chance | 0.14365 | 0.22207 | Baseline retained |
| `gbssa_img_r02_seed0` | 2 | Chance/collapsed alignment | 0.09225 | 0.14263 | Rejected |
| `gbssa_img_r05_seed0` | 5 | Chance/collapsed alignment | 0.04597 | 0.07499 | Rejected |
| `gbssa_img_r10_seed0` | 10 | Chance/collapsed alignment | 0.03790 | 0.06168 | Rejected |

Artifacts:

- `outputs/cross_modal/checkpoints/gbssa_img_r*_seed0/`
- `outputs/cross_modal/metrics/gbssa_img_r*_seed0_metrics.csv`
- `outputs/cross_modal/predictions/gbssa_img_r*_seed0_embeddings.npz`
- `outputs/evaluation/20260814_gbssa_r*_seed0_kmeans_seed0_v1/`

Conclusion: exact 5x/10x positive score-gradient weighting does not improve training. It drives retrieval to chance and damages biological clustering. These negative results are retained; they are not candidates for canonical attachment.

Transition-ratio follow-up found that ratio `1.1` still learned weak correspondence, ratio `1.25` was marginal, and ratio `1.5` converged to chance. None improved the one-seed aligned bisector over ratio 1; their bisector median ARIs were `0.11351`, `0.10825`, and `0.11243`, respectively.

## Cross-Modal Alignment

### Per-Section Complete-Gallery Diagnostics

Medians across 12 sections; all evaluated spots participated in representation training or checkpoint selection, so these are transductive diagnostics rather than held-out generalization.

| Run | R@1 | R@10 | Median rank | FOSCTTM lower is better | Cosine gap | Gene/image effective rank |
|---|---:|---:|---:|---:|---:|---:|
| Ratio 1 | 0.00162 | 0.01418 | 882.25 | 0.29177 | 0.06063 | 10.69 / 35.19 |
| Ratio 1.1 | 0.00067 | 0.00553 | 1025.75 | 0.31675 | 0.01840 | 1.88 / 2.47 |
| Ratio 5 | 0.00013 | 0.00286 | 1848.75 | 0.49113 | approximately 0 | nominal 50.33 / 53.74, but mean dimension std below 0.0002 |

Artifact: `outputs/evaluation/20260814_alignment_diagnostics_v1/`.

Interpretation: positive-heavy GBSSA contracts the representation. Ratio 1.1 already approaches low-rank geometry with nearly universal high cosine, and ratio 5 produces practically constant vectors. This explains both chance retrieval and degraded clustering.

## Per-Section Domain Discovery

### Existing Embeddings: KMeans, No Refinement

Ten explicit KMeans seeds per section. Each section uses its exact observed annotation count only to set `k`; assignments do not use labels. Values are medians over the 12 section-level seed means.

| Representation | Median ARI | ARI IQR | Median NMI | NMI IQR | Median Hungarian accuracy |
|---|---:|---:|---:|---:|---:|
| R2 `gene_emb` | 0.11132 | 0.07298 | 0.17152 | 0.04299 | 0.32054 |
| R3 `img_emb` | **0.15245** | 0.12944 | 0.22667 | 0.15591 | **0.39233** |
| R4 unaligned bisector | 0.12915 | 0.16663 | 0.22370 | 0.12625 | 0.38501 |
| R5 unaligned concatenation | 0.13138 | 0.16765 | 0.21803 | 0.12691 | 0.38259 |
| R6 unaligned concatenation PCA-128 | 0.13155 | 0.15969 | 0.21797 | 0.12519 | 0.38428 |
| R7 aligned gene | 0.10191 | 0.02780 | 0.17319 | 0.03425 | 0.30343 |
| R8 aligned image | 0.14793 | 0.07696 | **0.25274** | 0.11195 | 0.38924 |
| R9 aligned bisector | 0.14187 | 0.07145 | 0.23837 | 0.07014 | 0.36769 |
| R10 aligned concatenation | 0.13281 | 0.06677 | 0.23574 | 0.07786 | 0.36593 |
| R11 aligned concatenation PCA-128 | 0.13134 | 0.07611 | 0.24312 | 0.07658 | 0.36584 |

Artifact: `outputs/evaluation/20260813_existing_embeddings_kmeans_10seeds_v1/clustering/`.

Interpretation: the current aligned bisector does not outperform raw `img_emb` or aligned image alone on median ARI. It remains far below STAIG's published per-section median ARI `0.69`, though backend/refinement and method-training protocols are not yet harmonized. This motivates GBSSA retraining rather than supporting the current alignment.

GMM/mclust-compatible, Leiden, and refined tables remain pending.

### Newly Trained Gene BYOL: Corrected 10-Seed KMeans Gate

This table consistently uses `gene_byol_gbssa_seed0` for both unaligned and aligned rows. The aligned rows use standard CLIP ratio 1.

| Representation | Median ARI | ARI IQR | Median NMI | NMI IQR | Median Hungarian accuracy |
|---|---:|---:|---:|---:|---:|
| New `gene_emb` | 0.11656 | 0.06969 | 0.16780 | 0.04613 | 0.31229 |
| Raw `img_emb` | 0.15245 | 0.12944 | 0.22667 | 0.15591 | 0.39233 |
| **New-gene unaligned bisector** | **0.16323** | 0.15199 | 0.23697 | 0.12636 | **0.40558** |
| New-gene unaligned concatenation | 0.15922 | 0.14354 | 0.23654 | 0.12689 | 0.40146 |
| Unaligned concatenation PCA-128 | 0.15732 | 0.14235 | 0.24065 | 0.12609 | 0.39858 |
| Aligned gene | 0.10184 | 0.03261 | 0.17160 | 0.04024 | 0.31160 |
| Aligned image | 0.14623 | 0.07926 | **0.24439** | 0.12791 | 0.39358 |
| Standard-CLIP aligned bisector | 0.14706 | 0.07346 | 0.22358 | 0.07750 | 0.36835 |
| Aligned concatenation | 0.13681 | 0.07938 | 0.21891 | 0.08163 | 0.36143 |
| Aligned concatenation PCA-128 | 0.13915 | 0.07583 | 0.21734 | 0.08656 | 0.36340 |

Artifact: `outputs/evaluation/20260814_gbssa_r01_seed0_corrected_kmeans_10seeds_v1/clustering/`.

Interpretation: the newly trained gene BYOL improves the unaligned multimodal bisector over raw image on median ARI, but exact-spot InfoNCE removes that gain. The current best repository representation under this common KMeans protocol is the **unaligned bisector**, not GBSSA or standard-CLIP alignment.

Representation-seed check: unaligned-bisector median ARI was `0.16323`, `0.14013`, and `0.13869` for gene BYOL seeds 0, 1, and 2. The across-seed median `0.14013` is below fixed raw-image ARI `0.15245`; the seed-0 gain is not robust.

### Backend And Spatial Sensitivity

| Protocol | Best relevant representation | Median ARI | Median NMI | Notes |
|---|---|---:|---:|---|
| PCA-30 + sklearn full-covariance GMM, seed 0 | Standard-CLIP aligned bisector | 0.17200 | 0.26918 | Python GMM surrogate, not literal R `mclust` |
| KMeans, 10 seeds, strict-majority 6-NN refinement | New-gene unaligned bisector | **0.19937** | **0.28514** | Same label-free refinement applied to every representation |
| KMeans, 10 seeds, strict-majority 6-NN refinement | Standard-CLIP aligned bisector | 0.18346 | 0.26693 | Alignment remains below unaligned fusion |

Artifacts:

- `outputs/evaluation/20260814_gbssa_r01_seed0_corrected_gmm_seed0_v1/`
- `outputs/evaluation/20260814_gbssa_r01_seed0_corrected_kmeans_10seeds_v1/spatial_refinement/`

Spatial refinement improves continuity and ARI but does not close the large gap to STAIG's published native pipeline. Because this is a common repository refinement rather than STAIG's incompletely specified radius rule, the values remain a sensitivity analysis rather than a direct reproduction.

### Fixed Angular Weight Sensitivity, Gene Seed 0

The curve is reported in full; weights were not selected on a held-out unsupervised criterion.

| Gene weight | Image weight | Median ARI | Median NMI |
|---:|---:|---:|---:|
| 0.00 | 1.00 | 0.15245 | 0.22667 |
| 0.25 | 0.75 | **0.16564** | **0.24077** |
| 0.50 | 0.50 | 0.16323 | 0.23697 |
| 0.75 | 0.25 | 0.13329 | 0.18958 |
| 1.00 | 0.00 | 0.11656 | 0.16780 |

Artifact: `outputs/evaluation/20260814_weighted_bisector_seed0_kmeans_10seeds_v1/`.

The image-dominant mixture is a small seed-0 sensitivity gain, not a canonical tuned model. It requires train-only or label-free weight selection and representation-seed validation before any claim.

Audit note: sections `151669`-`151672` contain five observed annotation classes rather than seven. The common evaluator derives `k` from each section's exact observed valid labels, consistent with label-count-informed evaluation.

## Pooled Clustering Stress Test

Pending Phase 7. Do not compare this table directly with published per-section STAIG results.

## Spatial Quality

Pending Phase 7.

## Frozen Representation Probes

### Random-Spot Transductive Logistic Probe Gate, Seed 0

| Representation | Accuracy | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|
| New gene BYOL | 0.4524 | 0.2980 | 0.2636 |
| Raw image BYOL | 0.3907 | 0.2162 | 0.2106 |
| Unaligned bisector | 0.4655 | 0.3251 | 0.3152 |
| Unaligned concatenation | 0.4881 | 0.3585 | 0.3518 |
| Aligned gene | 0.4566 | 0.3007 | 0.2687 |
| Aligned image | 0.4967 | 0.3701 | 0.3654 |
| Aligned bisector | 0.5245 | 0.4015 | 0.3968 |
| Aligned concatenation | **0.5331** | **0.4167** | **0.4151** |

Artifact: `outputs/evaluation/20260814_seed0_random_spot_logistic_probe_gate_v2/`.

This shows that standard-CLIP alignment improves linear label separability under a random-spot transductive split, despite degrading unsupervised clustering. It does not establish unseen-section/donor generalization. The initial five-seed `lbfgs` grid exceeded 20 minutes and produced no artifacts; expanded probe seeds and other probe families remain pending.

## Attention Ablations

### Random-Spot Transductive Gate

All models use the same outer 70/30 split construction and internal validation rule. Seed-specific splits differ across seeds, as intended.

| Model | Parameters | Seed 0 accuracy | Seed 1 accuracy | Seed-0 balanced accuracy | Seed-0 macro-F1 |
|---|---:|---:|---:|---:|---:|
| Parameter-matched concatenation MLP | 270,343 | 0.52997 | Not run | 0.39933 | 0.39140 |
| **Two-token self-attention only** | 133,383 | **0.61251** | **0.60666** | **0.53056** | **0.53944** |
| Full length-one cross blocks + self-attention | approximately 266k | 0.60532 | 0.60159 | Not saved by legacy runner | Not saved by legacy runner |

Artifacts:

- `outputs/evaluation/20260814_gbssa_r01_seed0_attention_ablation_v1/`
- `outputs/evaluation/20260814_gbssa_r01_seed0_attention_legacy_v1/`
- `outputs/evaluation/20260814_gbssa_r01_seed0_attention_legacy_seed1_v1/`

Conclusion: the parameter-matched MLP confirms a benefit from token interaction, but the full cross-attention architecture does not beat self-attention alone. Because each cross block has only one key/value token, its softmax is always one; the experiments show no benefit from those blocks. The supported supervised architecture is therefore **two-token self-attention fusion**, not the original length-one cross-attention claim.

## Joint-Section Integration

### Adjacent Same-Donor Sections 151675 And 151676

Ten-seed mean joint KMeans results. Mixing metrics are repository-local cosine-30NN inverse-Simpson/KL surrogates and must not be numerically equated with STAIG's implementation.

| Representation | ARI | NMI | Section iLISI | Section BatchKL lower is better |
|---|---:|---:|---:|---:|
| New gene BYOL | 0.0915 | 0.1307 | 1.8179 | 0.0623 |
| Raw image BYOL | 0.0390 | 0.0645 | 1.1820 | 0.5181 |
| Unaligned bisector | 0.1052 | 0.1478 | 1.2253 | 0.4818 |
| Aligned gene | 0.0863 | 0.1482 | **1.8684** | **0.0424** |
| Aligned image | 0.0813 | 0.1323 | 1.3275 | 0.3823 |
| **Aligned bisector** | **0.1120** | **0.1823** | 1.4659 | 0.2771 |
| Aligned concatenation | 0.0899 | 0.1591 | 1.4396 | 0.2968 |

STAIG published native reference: ARI `0.64`, NMI `0.71`, BatchKL `0.14`, iLISI `1.88`. Only ARI/NMI task context is directly interpretable here; mixing implementations differ.

### Four Sections From Two Donors

Sections `151507`, `151508`, `151675`, `151676`.

| Representation | ARI | NMI | Section iLISI | Donor iLISI | Section BatchKL | Donor BatchKL |
|---|---:|---:|---:|---:|---:|---:|
| New gene BYOL | 0.0950 | 0.1359 | 3.0490 | **1.7303** | 0.2258 | 0.1348 |
| Raw image BYOL | 0.0794 | 0.1175 | 1.2897 | 1.1681 | 1.1148 | 0.5192 |
| Unaligned bisector | 0.1486 | 0.1997 | 1.4177 | 1.2299 | 1.0133 | 0.4623 |
| **Unaligned concatenation** | **0.1509** | **0.2018** | 1.4136 | 1.2269 | 1.0165 | 0.4647 |
| Aligned gene | 0.0883 | 0.1449 | **3.1392** | 1.7190 | **0.1796** | **0.1153** |
| Aligned image | 0.1306 | 0.1741 | 1.5467 | 1.2576 | 0.9102 | 0.4319 |
| Aligned bisector | 0.1242 | 0.1959 | 1.9009 | 1.3381 | 0.6832 | 0.3635 |
| Aligned concatenation | 0.1220 | 0.1917 | 1.8409 | 1.3304 | 0.7168 | 0.3696 |

STAIG published native reference: ARI `0.52`, NMI `0.62`, BatchKL `0.33`, iLISI `2.95`.

Artifacts:

- `outputs/evaluation/20260814_integration_151675_151676_v1/`
- `outputs/evaluation/20260814_integration_151507_151508_151675_151676_v1/`

Conclusion: alignment improves batch mixing, especially on the gene side, but does not retain enough cortical-domain structure. Unaligned fusion gives the best four-section biological ARI, while aligned-gene embeddings give the strongest mixing. This is not competitive with STAIG's reported biological conservation.

The all-12 integration run exceeded 20 minutes before writing artifacts because global 30-NN mixing metrics were recomputed for every representation. It is marked failed pending cached neighborhoods or a predeclared reduced representation set.

## Donor Transfer And Generalization

Pending Phase 10. Keep post-hoc transfer on transductive embeddings separate from strict end-to-end donor holdout.

## External Common-Protocol Baselines

Pending Phase 11. Missing methods must be marked `NOT RUN` with a reason.

## Negative And Failed Experiments

- Ratios `2`, `5`, and `10`: geometric/alignment collapse; artifacts retained under `outputs/cross_modal/` and ratio-specific evaluation directories.
- Transition ratios `1.1`, `1.25`, `1.5`: no clustering improvement; ratios 1.1 and above showed contraction, with 1.5 at chance retrieval.
- Five-seed `lbfgs` logistic grid: exceeded 20 minutes and produced no artifacts; replaced by a one-seed `saga` gate.
- All-12 integration v1: exceeded 20 minutes and produced no artifacts; implementation requires cached neighbor calculations or representation filtering.
