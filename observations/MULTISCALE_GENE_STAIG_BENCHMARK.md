# Multi-Scale Gene STAIG Benchmark

## Question

Does a hidden representation from one independently trained gene BYOL encoder provide a better STAIG node feature than section-level expression, and does same-spot cross-modal alignment improve that result?

The canonical encoder is `33,538 -> 4,096 -> 1,024 -> 512 -> 128`. It uses every common gene and is trained with BYOL, masked reconstruction, stage-wise variance/covariance, expression-SVD geometry preservation, a fixed SVD residual, and weak spatial consistency. The earlier `3,000 -> 1,024 -> 512 -> 256 -> 128` ablation is retained below as superseded historical evidence.

## Fixed Protocol

- Data: all 12 DLPFC sections, 47,329 spots; identity is `(section_id, barcode)`.
- Representation seed: 0.
- STAIG: one independent model per section, coordinate 5-NN, image PCA-16, 40 image pseudo-clusters, two 10% feature-masked views, one 64-dimensional GCN layer, temperature 10, 400 epochs, seed 0.
- Domain evaluation: label-count-informed tied-covariance GMM followed by optional 15-spatial-neighbor plurality refinement. Labels do not enter representation training.
- GMM is fitted in float64 for numerical stability; STAIG training remains float32.
- Frozen probes: identical within-section stratified 70/30 splits, seed 0, train-fitted scaling, `lbfgs` logistic regression. Gene stages above 128 dimensions receive train-fitted PCA-128; STAIG embeddings remain 64-dimensional.
- External image source: fixed repository `img_emb`. The unavailable filtered-patch STAIG image BYOL is not reproduced.

## Superseding All-Gene Result

### Training

Run `rich_gene_v2_all33538_seed0` uses all 33,538 genes and exports `gene_stage_4096`, `gene_stage_1024`, `gene_stage_512`, and `gene_embedding_128` for all 47,329 spots. It selected epoch 27 by validation total loss `1.29656` and stopped at epoch 37. The selected final embedding remained non-collapsed (`embed_std=0.99357`). No cortical labels entered training or checkpoint selection.

### Direct Stage Clustering

Each stage is standardized and reduced to PCA-128 within section when necessary, followed by tied GMM and the common 15-neighbor refinement.

| Representation | Median raw ARI/NMI | Median refined ARI/NMI |
|---|---:|---:|
| All-gene stage 4,096 | **0.34798/0.43079** | **0.44860/0.55147** |
| All-gene stage 1,024 | 0.33176/0.41886 | 0.42932/0.52808 |
| All-gene stage 512 | 0.32747/0.41642 | 0.44446/0.51680 |
| All-gene final 128 | 0.32552/0.39659 | 0.43236/0.52335 |

The final 128-dimensional bottleneck retains most of the all-gene domain structure. The earlier 3,000-HVG final-128 direct refined ARI was `0.14575`; the all-gene final-128 value is `0.43236`.

### Rich Alignment

Run `rich_gene_v2_all33538_gbssa_r01_seed0` aligns final gene-128 to fixed image-128 while reconstructing original expression from the normalized gene projection. It selected epoch 40 and stopped at epoch 55.

| Diagnostic | Previous multistage ratio 1 | All-gene rich ratio 1 |
|---|---:|---:|
| Recall@10 | 0.01060 | **0.07209** |
| FOSCTTM, lower is better | 0.30975 | **0.11712** |
| Cosine gap | 0.05355 | **0.11104** |
| Gene effective rank | 7.22 | **45.22** |
| Image effective rank | 33.96 | **87.00** |

| Aligned representation | Median raw ARI/NMI | Median refined ARI/NMI |
|---|---:|---:|
| Gene projection | 0.32628/0.46697 | 0.41347/0.57336 |
| Image projection | 0.16006/0.26185 | 0.19877/0.31914 |
| Bisector | 0.36260/0.50945 | 0.40034/0.58032 |
| Concatenation PCA-128 | **0.37563/0.51921** | **0.45084/0.59663** |

Alignment improves retrieval and effective rank, but the bisector is not the strongest fusion. Concatenation preserves complementary modality information and slightly exceeds the direct stage-4,096 ARI.

### Targeted STAIG Results

Only three requested downstream conditions were run; the unnecessary 21-condition expansion was cancelled.

| Node feature | Image guidance | Mean refined ARI | Median refined ARI | Median refined NMI |
|---|---|---:|---:|---:|
| Section 3,000-HVG expression control | Raw `img_emb` | 0.50693 | 0.51821 | **0.67427** |
| **All-gene stage 4,096** | Raw `img_emb` | 0.48164 | **0.52293** | 0.66682 |
| All-gene final 128 | Raw `img_emb` | 0.43867 | 0.44533 | 0.59228 |
| Aligned final gene-128 | Aligned image-128 | 0.49721 | 0.49365 | 0.66117 |

The all-gene stage-4,096 representation essentially matches expression-input STAIG: median refined ARI is slightly higher by `0.00472`, while median refined NMI is lower by `0.00745`. This one-seed difference is too small to claim superiority, but it establishes that the richer learned representation no longer suffers the severe information loss of the 3,000-HVG encoder.

Cells below are refined `ARI/NMI` for every biological section.

| Section | Expression STAIG | All-gene 4096 STAIG | All-gene 128 STAIG | Aligned 128 STAIG |
|---|---:|---:|---:|---:|
| 151507 | .585/.696 | .589/.721 | .520/.653 | .493/.680 |
| 151508 | .486/.642 | .585/.695 | .490/.598 | .498/.635 |
| 151509 | .583/.683 | .494/.673 | .486/.607 | .486/.665 |
| 151510 | .498/.665 | .464/.648 | .476/.621 | .494/.651 |
| 151669 | .353/.496 | .255/.498 | .359/.507 | .361/.500 |
| 151670 | .399/.501 | .263/.516 | .389/.489 | .306/.524 |
| 151671 | .480/.637 | .530/.674 | .460/.617 | .575/.703 |
| 151672 | .546/.677 | .574/.705 | .568/.643 | .606/.716 |
| 151673 | .553/.695 | .572/.698 | .431/.587 | .599/.728 |
| 151674 | .525/.681 | .523/.661 | .326/.458 | .475/.657 |
| 151675 | .564/.678 | .523/.660 | .407/.531 | .475/.623 |
| 151676 | .512/.671 | .407/.581 | .352/.517 | .599/.721 |

Sections, not spots, are the biological replicates. The new result remains a one-representation-seed transductive study, not donor-level generalization.

## Historical 3,000-HVG Ablation

The sections below document the superseded study. They explain why the all-gene reconstruction-preserving encoder was introduced and must not be presented as the current model.

## Gene BYOL

Run `gene_byol_multistage_1024_512_256_128_seed0` selected epoch 199 by validation total loss (`0.443116`). Its immutable NPZ contains complete finite 1,024/512/256/128-dimensional arrays for all 47,329 spots. Final-128 effective rank is `10.7142`; mean/min per-dimension standard deviation is `1.00016/0.68950`. Some ReLU hidden coordinates are globally inactive and were retained.

## Alignment Diagnostics

Values are medians across the 12 complete-section galleries.

| Run | R@10 | FOSCTTM | Cosine gap | Gene effective rank | Image effective rank | Decision |
|---|---:|---:|---:|---:|---:|---|
| GBSSA ratio 1 | 0.01060 | 0.30975 | 0.05355 | 7.22 | 33.96 | Accepted for S5 |
| GBSSA ratio 1.1 | 0.00800 | 0.30633 | 0.01615 | 3.34 | 3.85 | Contracted negative S6 |

The valid artifacts have the `_v2` suffix. Initial non-`v2` artifacts are invalid because a fixed-width NumPy string array truncated section IDs to `"1"`; exact manifest validation caught the defect before STAIG use.

## STAIG Results

### Aggregate Domain Discovery

| ID | Node feature | Image guidance | Mean refined ARI | Median refined ARI | Median refined NMI |
|---|---|---|---:|---:|---:|
| S0 | Section 3,000-HVG expression | Raw `img_emb` | **0.50693** | **0.51821** | **0.67427** |
| S1 | Gene stage 1,024 | Raw `img_emb` | 0.20010 | 0.19672 | 0.31947 |
| S2 | Gene stage 512 | Raw `img_emb` | 0.21617 | 0.20898 | 0.31342 |
| S3 | Gene stage 256 | Raw `img_emb` | 0.19941 | 0.21495 | 0.28885 |
| S4 | Final gene 128 | Raw `img_emb` | 0.22162 | **0.21885** | 0.27542 |
| S5 | Ratio-1 projected gene | Ratio-1 projected image | 0.20720 | 0.20331 | **0.32886** |
| S6 | Ratio-1.1 projected gene | Ratio-1.1 projected image | 0.19461 | 0.17796 | 0.32037 |

S4 has the highest median refined ARI among the multi-stage variants, while S5 has the highest median refined NMI. Neither approaches expression-input S0. Ratio-1 alignment reduces final-128 median refined ARI from `0.21885` to `0.20331`; ratio 1.1 reduces it to `0.17796`.

### All Section Values

Cells are refined `ARI/NMI`.

| Section | S0 expression | S1 1024 | S2 512 | S3 256 | S4 128 | S5 ratio 1 | S6 ratio 1.1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 151507 | .585/.696 | .202/.286 | .159/.267 | .217/.259 | .219/.258 | .159/.261 | .167/.256 |
| 151508 | .486/.642 | .208/.320 | .234/.312 | .213/.287 | .203/.267 | .177/.285 | .172/.281 |
| 151509 | .583/.683 | .282/.383 | .256/.330 | .249/.313 | .222/.298 | .274/.360 | .298/.366 |
| 151510 | .498/.665 | .210/.334 | .192/.315 | .254/.292 | .219/.283 | .194/.328 | .221/.325 |
| 151669 | .353/.496 | .168/.320 | .213/.318 | .200/.306 | .305/.328 | .209/.297 | .167/.297 |
| 151670 | .399/.501 | .132/.280 | .205/.290 | .226/.291 | .272/.307 | .219/.300 | .192/.291 |
| 151671 | .480/.637 | .186/.362 | .254/.333 | .240/.337 | .374/.337 | .242/.335 | .284/.344 |
| 151672 | .546/.677 | .309/.386 | .293/.350 | .273/.317 | .352/.343 | .202/.343 | .152/.328 |
| 151673 | .553/.695 | .192/.298 | .195/.307 | .122/.224 | .121/.220 | .205/.353 | .169/.315 |
| 151674 | .525/.681 | .178/.303 | .235/.370 | .144/.239 | .152/.254 | .195/.330 | .202/.336 |
| 151675 | .564/.678 | .208/.319 | .199/.306 | .163/.261 | .126/.230 | .237/.372 | .184/.344 |
| 151676 | .512/.671 | .126/.232 | .159/.242 | .092/.189 | .094/.185 | .173/.292 | .127/.267 |

The failure is consistent across sections rather than being caused by one outlier section. Sections remain biological replicates; no spot-level significance test is used.

## Direct Gene Controls

| Gene representation | Median raw GMM ARI/NMI | Median refined ARI/NMI |
|---|---:|---:|
| Stage 1,024 | .10225/.18409 | **.20599/.30219** |
| Stage 512 | .08822/.14976 | .14400/.22570 |
| Stage 256 | .09810/.14899 | .15231/.25146 |
| Final 128 | .09690/.14207 | .14575/.22758 |

Stage 1,024 is the strongest direct gene control after spatial refinement. STAIG does not improve its median ARI (`0.19672` versus `0.20599`), although its NMI is slightly higher (`0.31947` versus `0.30219`).

## Frozen Probes

Values are medians across 12 within-section probes.

| Representation | Accuracy | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|
| S0 expression STAIG | **0.89432** | **0.85665** | **0.86041** |
| S1 gene-1024 STAIG | 0.60868 | 0.41615 | 0.39508 |
| S2 gene-512 STAIG | 0.61377 | 0.42083 | 0.41479 |
| S3 gene-256 STAIG | 0.57177 | 0.36967 | 0.33333 |
| S4 gene-128 STAIG | 0.54907 | 0.34619 | 0.29565 |
| S5 ratio-1 STAIG | 0.58688 | **0.43179** | **0.42266** |
| S6 ratio-1.1 STAIG | 0.56245 | 0.40666 | 0.37990 |
| Raw stage 1,024 PCA-128 | 0.48815 | 0.35310 | 0.34822 |
| Raw stage 512 PCA-128 | 0.47079 | 0.33477 | 0.32690 |
| Raw stage 256 PCA-128 | 0.46714 | 0.31310 | 0.30679 |
| Raw final 128 | 0.48241 | 0.30968 | 0.28563 |

S5 improves linear separability over S4 despite worse clustering, reinforcing that label separability, exact-pair alignment, and unsupervised domain recovery are distinct claims. All multi-stage representations remain far below expression STAIG.

## External Context

- Repository-local expression STAIG adaptation: mean/median refined ARI `0.50693/0.51821`.
- Published STAIG displayed-table mean/median ARI: `0.69167/0.68000`.
- Published SpaGCN displayed-table mean/median ARI: `0.43833/0.43500`.

Published rows are native-pipeline context, not common-embedding results. The repository adaptation substitutes external `img_emb` and sklearn tied GMM, so direct implementation attribution is limited.

## Conclusion

The superseding study supports the user's diagnosis: 3,000-HVG selection and the old BYOL objective discarded substantial expression information. Using all 33,538 genes with masked reconstruction and geometry/covariance preservation raises stage-4,096 STAIG median refined ARI from the historical `0.19672` to `0.52293`, essentially matching expression STAIG (`0.51821`). Final-128 remains useful (`0.44533`), and reconstruction-regularized alignment improves retrieval dramatically, but alignment does not beat the unaligned stage-4,096 STAIG representation. Concatenation remains stronger than the bisector for direct multimodal clustering. These are one-seed transductive per-section results, not donor-generalization claims.

## Artifacts

- Canonical plan and ledger: `RICH_GENE_EXPRESSION_EXECUTION.md`.
- Canonical gene export: `outputs/gene_encoder/predictions/rich_gene_v2_all33538_seed0_embeddings.npz`.
- Canonical alignment: `outputs/cross_modal/predictions/rich_gene_v2_all33538_gbssa_r01_seed0_embeddings.npz`.
- Alignment diagnostics: `outputs/evaluation/20260815_rich_gene_all33538_alignment_v1.*`.
- Direct stage controls: `outputs/evaluation/20260815_rich_gene_all33538_direct_v1/`.
- Direct aligned controls: `outputs/evaluation/20260815_rich_gene_all33538_aligned_direct_v1/`.
- Targeted STAIG runs: `outputs/prior_models/staig_rich_gene_v2_all33538_*/`.
- Historical ledger: `MULTISCALE_GENE_STAIG_EXECUTION.md`.
- Reusable APIs: `src/gene_encoder/rich_train.py`, `src/cross_modal/rich_train.py`, and `src/prior_models/staig/rich_evaluation.py`.

The `_v1` control directory pooled refinements in its summary and used non-converged `saga` probes; it is retained but superseded by `_v2`. The partial S4 `_v1` run and invalid pre-`v2` alignment outputs are also non-canonical.
