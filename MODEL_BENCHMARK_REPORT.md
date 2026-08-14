# GeoPatch Multimodal Benchmark and Ablation Report

## Purpose

This report defines how to evaluate the repository's aligned gene-image bisector and supervised attention fusion against modern spatial-transcriptomics models without mixing incompatible tasks or metrics. It records published reference results, explains how each method obtains clusters, audits the repository's current results, and specifies common evaluations and ablations.

The central repository path is:

```text
external image BYOL -> img_emb
gene-expression BYOL -> gene_emb
same-spot symmetric InfoNCE -> img_emb_cm + gene_emb_cm_img
normalized mean / angular bisector -> one 128-d vector per spot
```

The central question is not only whether InfoNCE retrieves the matching modality, but whether the resulting spot vector preserves cortical-domain and spatial-neighborhood structure.

## Executive Assessment

- The current bisector result, pooled over all 47,329 spots, is accuracy `32.2%`, ARI `0.090`, and NMI `0.163`. This is a valid label-count-informed pooled clustering result, but it is not directly comparable with the standard DLPFC benchmark, which clusters each section independently and summarizes 12 section-level ARI/NMI values.
- The bisector's `32.2%` Hungarian-matched accuracy is below the pooled `37.2%` majority-class classifier baseline. More importantly, its low ARI and NMI show that the current pooled partition weakly matches cortical layers. Pooling may expose section/donor effects, but that explanation must be tested rather than assumed.
- STAIG reports median per-section ARI `0.69` and NMI `0.71`. These values cannot be ranked directly against the pooled bisector ARI `0.090`; the bisector must first be evaluated separately on the same 12 sections with a matched clustering and refinement protocol.
- The current attention result, `60.8%` test accuracy, is supervised random-spot classification. It is not comparable with unsupervised ARI/NMI from STAIG or other domain-clustering models. It also remains transductive because the upstream gene BYOL and InfoNCE representations used all sections before the classifier split.
- The attention result is promising relative to this repository's current supervised MDM result (`34.96%`), but it does not yet show that attention is responsible for the gain. It lacks single-modality, concatenation, linear, parameter-matched MLP, and attention-block ablations.
- The highest-priority benchmark is a common per-section evaluator. The highest-priority attention study is an equal-split, equal-capacity supervised ablation suite. Joint integration and leave-one-donor-out generalization must remain separate tables.

## Why Results Must Be Grouped

The following evaluations answer different questions and must not share one ranking table.

| Evaluation family | Training/evaluation unit | Main question | Appropriate metrics |
|---|---|---|---|
| Per-section domain discovery | One DLPFC section at a time | Does the embedding recover cortical domains within a tissue section? | Per-section ARI/NMI, secondary Hungarian accuracy, spatial metrics |
| Joint-section integration | Multiple sections jointly | Are sections mixed while cortical biology is retained? | ARI/NMI plus BatchKL, iLISI, kBET or equivalent |
| Pooled clustering | All spots concatenated without a complete integration protocol | Does one global partition survive section/donor variation? | Pooled ARI/NMI and section-conditioned diagnostics |
| Supervised representation probe | Labeled train/validation/test split | Can labels be predicted from a frozen or trainable representation? | Accuracy, balanced accuracy, macro-F1, per-layer recall |
| Cross-modal alignment | Held-out paired gene/image spots | Can one modality retrieve its true paired spot? | Recall@K, MRR, median rank, FOSCTTM, cosine gap |
| Inductive generalization | Entire held-out section or donor | Does the complete frozen pipeline transfer to unseen tissue? | All applicable metrics, reported by held-out group |

## What Produces A Cluster

Most modern methods do not directly guarantee that the learned embedding consists of seven cortical clusters. They generally have two stages:

1. Learn or construct a spot representation using gene expression, spatial adjacency, histology, or combinations of them.
2. Apply a separate clustering method such as `mclust`, Leiden, Louvain, KMeans, or a deep-clustering head.

Many DLPFC studies supply the known number of manual classes to the clustering stage. This is standard but should be described as **label-count-informed unsupervised clustering**, not fully label-free discovery.

### What ARI Establishes

Adjusted Rand Index compares whether pairs of spots are grouped together or separately in the predicted and annotated partitions, with adjustment for chance. It is invariant to cluster names. An ARI near `1` indicates strong partition agreement; an ARI near `0` indicates chance-level agreement under its random-partition model; negative values indicate worse-than-chance agreement.

A decent ARI does not reveal which component caused the result. It may reflect:

- A biologically informative learned embedding.
- Spatial graph smoothing or neighborhood contrast.
- A clustering backend well matched to the embedding geometry.
- Supplying the correct number of domains.
- Spatial label refinement after clustering.
- Hyperparameter selection using annotations.

Therefore, each benchmark must report representation, clustering backend, requested cluster count, refinement, seeds, and whether labels influenced any model-selection decision.

## Model Mechanisms and Native Clustering

The following table summarizes the models most relevant to this repository. “Known count” means that the annotated number of domains is supplied or a clustering resolution is searched until that count is obtained; it does not mean labels determine individual assignments.

| Method | Inputs | Main representation mechanism | Native/reported clustering | Known count? | Spatial refinement or smoothing |
|---|---|---|---|---|---|
| **GeoPatch bisector** | External image BYOL + gene BYOL; spatial graph enters the gene BYOL view | Same-spot symmetric InfoNCE, then normalized mean of aligned unit vectors | Current code: pooled KMeans | Yes, `k=7` | None |
| **STAIG** | Gene, spatial KNN, H&E image | Image BYOL guides graph-edge augmentation; GCN contrastive learning uses same-node and graph-neighbor positives | `mclust` on the mean of two augmented-view embeddings | Yes | Common DLPFC spatial refinement |
| **GraphST** | Gene + spatial graph | Self-supervised graph contrastive representation | Reported DLPFC workflow uses `mclust` | Yes | Refinement used/available in DLPFC workflows |
| **DeepST** | Gene + spatial coordinates + histology | Image features and augmented expression are encoded with graph convolution; integration can use domain-adversarial learning | Leiden resolution search | Yes | Optional refinement; protocol must state whether enabled |
| **conST** | Gene + spatial graph + optional histology | Local, context, and global multimodal contrastive objectives | Leiden in the native DLPFC workflow | Yes | No separate native refinement identified |
| **SpaGCN** | Gene + coordinates + histology RGB/pixel information | Weighted graph and graph convolution combine expression, spatial distance, and image information | SpaGCN/Louvain-initialized deep clustering | Yes | Optional neighborhood-majority refinement |
| **stLearn** | Gene + spatial coordinates + histology | Histology- and location-informed SME expression smoothing/normalization | PCA followed by Louvain or workflow-specific clustering | Usually | stSME modifies expression before clustering |
| **STAGATE** | Gene + spatial graph | Graph-attention autoencoder learns an adaptive spatial representation | Common DLPFC workflow uses `mclust` | Yes | Usually no separate histology refinement |
| **SEDR** | Gene + spatial graph | Expression autoencoder plus graph autoencoder/deep embedding | Revised workflow commonly uses `mclust`; older versions used Leiden/DEC | Yes | None in the main workflow |
| **MuCoST** | Gene + spatial graph; reported STAIG setting excludes histology | Shared multi-view graph-convolutional autoencoder with InfoNCE | `mclust` | Yes | None identified in headline workflow |
| **Seurat** | Gene only | PCA/neighborhood graph | `FindClusters()` resolution adjusted to desired count | Yes | None |

### Why STAIG's Objective Is More Cluster-Oriented Than Exact-Spot InfoNCE

GeoPatch cross-modal InfoNCE defines only the exact gene/image pair for spot `i` as positive. Every other spot in the batch is negative, including same-layer and spatial-neighbor spots.

STAIG instead treats the same spot across graph views and graph neighbors as positives. It also removes some image-similar pseudo-label spots from the negative set. This objective directly encourages local domain continuity. Its clustering strength can therefore come from the spatial graph and neighborhood-positive definition, not only from image-gene fusion.

This distinction motivates an ablation in which GeoPatch excludes immediate spatial neighbors from negatives or introduces label-free spatial-neighbor positives.

## Published Per-Section DLPFC Domain Results

### Aggregate Results That Are Explicitly Reported

These values use 12 DLPFC sections evaluated independently unless stated otherwise.

| Method | Aggregate statistic | ARI | NMI | Source status |
|---|---|---:|---:|---|
| STAIG | Median across 12 sections | **0.69** | **0.71** | Primary paper, exact prose value |
| GraphST | Median across 12 sections | **0.60** | Not explicitly printed | Primary paper |
| DeepST | Mean across 12 sections | **0.515 +/- 0.011** | Not explicitly printed | Primary paper; mean is not a median |
| MuCoST | Mean across 12 sections | **0.526** | Not explicitly printed | Primary paper |
| PRECAST | Median across 12 sections | **0.434** | Not listed here | Primary paper; joint model/integration-oriented protocol |

These rows are not a fully harmonized leaderboard. They use different aggregation statistics, preprocessing, clustering, and refinement. Missing exact aggregate values should not be estimated from boxplots and presented as official results.

### STAIG Rerun On The Common Slice 151673

STAIG Figure 2 provides directly readable values for one shared section. STAIG is the primary source for its own value and a secondary rerun source for the baselines.

| Method | ARI | NMI |
|---|---:|---:|
| STAIG | **0.68** | **0.74** |
| GraphST | 0.64 | 0.73 |
| MuCoST | 0.57 | 0.66 |
| SEDR | 0.55 | 0.69 |
| DeepST | 0.54 | 0.68 |
| SpaGCN | 0.51 | 0.69 |
| STAGATE | 0.50 | 0.68 |
| conST | 0.45 | 0.64 |
| stLearn | 0.39 | 0.59 |
| Seurat | 0.25 | 0.42 |

This is the cleanest published same-slice reference table, but a repository rerun on the exact same data is still preferable because implementation versions and preprocessing can differ.

### Additional Primary-Source Context

| Method | Primary-source DLPFC statement | Comparability note |
|---|---|---|
| conST | Section 151673 ARI `0.65` | Native workflow result differs from STAIG's rerun (`0.45`) |
| DeepST | Mean ARI `0.515 +/- 0.011`; best section 151671 ARI `0.798` | Native DeepST protocol |
| GraphST | Median ARI `0.60`; section 151673 ARI `0.64` | Closely aligned with standard per-slice evaluation |
| MuCoST | Mean ARI `0.526`; section 151673 ARI `0.61`, NMI `0.72` | Native result differs from STAIG rerun |
| STAGATE | Section 151676 ARI `0.60` | Exact all-section median not printed in prose |
| SpaGCN | Original article evaluates DLPFC, but no exact all-section headline ARI/NMI was verified in article text | Use a rerun for the common table |
| stLearn | No standard native 12-section DLPFC headline result | Treat later DLPFC values as third-party reruns |
| SEDR | Revised 2024 article reports broad median superiority but does not print one canonical median ARI in prose | Do not mix with the older preprint mean ARI `0.427` |

Different values for one method across papers are expected when preprocessing, image use, graph construction, clustering, refinement, or software version changes. They demonstrate why published numbers are context, not a substitute for a common rerun.

## Published Joint-Section Integration Results

These values belong in integration tables, not the per-section clustering table.

### Adjacent Sections 151675 And 151676, Same Donor

| Method | ARI | NMI | BatchKL lower is better | iLISI higher is better |
|---|---:|---:|---:|---:|
| STAIG | **0.64** | **0.71** | 0.14 | 1.88 |
| GraphST | 0.57 | 0.65 | **0.06** | 1.84 |
| STAGATE | 0.57 | 0.69 | 0.58 | 1.42 |
| DeepST | 0.51 | 0.66 | 0.17 | 1.83 |
| STAligner | 0.51 | 0.69 | 0.52 | 1.47 |
| SEDR | 0.46 | 0.64 | 0.17 | 1.83 |
| stLearn | 0.19 | 0.26 | 0.20 | **1.89** |

GraphST has the best BatchKL and stLearn the highest iLISI, while STAIG has the best biological ARI/NMI. This illustrates why batch mixing alone is not a successful integration result.

### Sections 151507, 151508, 151675, And 151676, Two Donors

| Method | ARI | NMI | BatchKL lower is better | iLISI higher is better |
|---|---:|---:|---:|---:|
| STAIG | **0.52** | 0.62 | **0.33** | **2.95** |
| STAGATE + Harmony | 0.47 | **0.63** | 1.10 | 1.57 |
| STAligner | 0.45 | 0.62 | 0.68 | 1.93 |
| STAGATE | 0.35 | 0.51 | 1.30 | 1.43 |
| DeepST | 0.28 | 0.43 | 1.09 | 1.81 |

STAIG's main text describes its NMI as highest, but the supplementary figure prints `0.63` for STAGATE + Harmony versus STAIG's `0.62`. The table retains the printed values.

## Current Repository Results

### Results Table

| Repository method | Learning type | Current protocol | Result | Valid interpretation |
|---|---|---|---|---|
| Aligned bisector + KMeans(7) | Unsupervised assignments; known label count | All 12 sections pooled | Accuracy `32.2%`, ARI `0.090`, NMI `0.163` | Weak pooled cortical-domain recovery |
| Log-Euclidean covariance + MiniBatchKMeans | Unsupervised assignments; known label count | All 12 sections pooled | Accuracy `24.7%` | Weak pooled result; rank-one covariance mostly encodes modality difference |
| Riemannian MDM | Supervised | Random pooled 70/30 spot split | Accuracy `34.96%` | Weak transductive supervised classification |
| Attention fusion | Supervised | Random pooled 70/30 outer spot split with internal validation | Accuracy `60.8%` | Strongest current transductive supervised result |
| Cross-modal `gene_emb <-> img_emb` | Self-supervised correspondence | Random spot train/validation, no test | Validation in-batch retrieval `1.02%`; batch chance about `0.20%` | Modest exact-pair alignment, not clustering evidence |
| Cross-modal `gene_emb <-> proj_emb` | Label-assisted upstream | Random spot train/validation, no test | Validation in-batch retrieval `0.72%` | Weaker exact-pair alignment; not a fully unsupervised path |

### Audit Of The Bisector Evaluation

What is correct:

- Cluster assignments do not use ground-truth labels.
- Labels are used after clustering for Hungarian accuracy, ARI, and NMI.
- The bisector is a deterministic, importable representation.
- The known `k=7` is disclosed in the source.

What limits the result:

- All sections and donors are pooled, unlike the primary DLPFC benchmark.
- KMeans is not matched to STAIG's `mclust` and common spatial refinement.
- Section/donor effects are not measured, so the cause of low pooled ARI is unknown.
- Only one reported pipeline seed is available.
- There is no unaligned bisector, aligned single-modality, concatenation, shuffled-pair, or dimensionality-controlled fusion comparison.
- There are no spatial-coherence or marker-gene metrics.
- Exact-spot InfoNCE treats biologically related spots as negatives, which can conflict with domain clustering.

Judgment:

The current result is an honest negative result for **pooled global clustering**. It should not be described as evidence that the bisector is worse than STAIG, GraphST, or DeepST until the same per-section protocol is run. It does establish that the present pipeline does not yet produce a section-invariant global layer partition.

### Why Accuracy Is Not A Literature Comparison

The bisector's `32.2%` is Hungarian-matched clustering accuracy. The attention model's `60.8%` is supervised test classification accuracy. Published STAIG/GraphST values are unsupervised ARI/NMI. These quantities are not interchangeable.

The pooled majority baseline of `37.2%` is useful for supervised accuracy. It is less informative for forced seven-cluster domain discovery because a one-cluster majority assignment does not satisfy the same clustering task. ARI/NMI and spatial metrics should remain primary for unsupervised evaluation.

### Audit Of Attention Fusion

The current model performs bidirectional cross-attention between one image token and one gene token, followed by self-attention over the two resulting tokens and a supervised classifier.

With exactly one key/value token, the cross-attention softmax is always `1`. Therefore, the query/key scores do not choose between alternatives. The block behaves as a learned value/output transformation of the opposite modality plus a residual; meaningful attention competition first appears in the later two-token self-attention block.

Current strengths:

- A held-out outer test set is not used for early stopping.
- Validation loss selects the best model.
- It clearly outperforms the current MDM classifier under the shared outer test split.

Current limitations:

- Random spots from the same sections and donors occur in train and test.
- Upstream gene BYOL and InfoNCE already saw all sections and test spots without labels.
- Only accuracy and loss are reported.
- The model has no equal-capacity concatenation MLP control.
- No ablation isolates cross-attention, self-attention, residual projections, or fusion.
- The MDM method uses all non-test spots, while attention withholds validation spots, so training counts differ.
- The saved artifacts omit predictions, split manifests, label mapping, complete configuration, best epoch, and input embedding keys.

Judgment:

The `60.8%` result demonstrates that a trainable supervised nonlinear model can extract label information from the aligned pair under a transductive random-spot protocol. It does not yet demonstrate that cross-attention is the cause, that it beats simple fusion, or that it generalizes to unseen sections or donors.

## Required Common Evaluations

### Table A: Per-Section Domain Discovery

This is the primary table for comparison with STAIG, GraphST, DeepST, conST, SpaGCN, STAGATE, SEDR, MuCoST, and stLearn.

For each of the 12 DLPFC sections:

1. Produce one embedding per spot without using layer labels.
2. Use the correct section-specific annotation count as `k`; record this as label-count-informed.
3. Run the same clustering backend for every embedding.
4. Use no refinement for any method in the primary common table.
5. Add a separate table where one identical spatial refinement is applied to every method.
6. Run at least 10 clustering seeds and at least 5 full representation-training seeds where feasible.
7. Report ARI and NMI as primary, Hungarian accuracy as secondary.
8. Report spatial-neighbor agreement, connectedness, and marker coherence separately.
9. Show all 12 section values and median/IQR; use paired section-level comparisons.

Recommended clustering sensitivity:

| Backend | Purpose |
|---|---|
| Gaussian mixture / `mclust`-compatible | Closest common comparison to STAIG, GraphST, STAGATE, SEDR, and MuCoST |
| KMeans or spherical KMeans | Deterministic geometry-focused baseline for the bisector |
| Leiden | Graph/community sensitivity comparison used by conST and DeepST workflows |

### Table B: Joint-Section Integration

First reproduce the two STAIG subsets:

- `151675 + 151676`.
- `151507 + 151508 + 151675 + 151676`.

Then report the repository-specific all-12-section experiment separately.

Required metrics:

- ARI and NMI for biological conservation.
- BatchKL and iLISI for section/donor mixing.
- Per-layer section mixing rather than unconditional mixing only.
- Graph connectedness and spatial coherence.
- Section/donor predictability from a frozen embedding.

### Table C: Supervised Frozen-Representation Probes

Use identical split manifests and classifiers for every representation:

- Multinomial logistic regression.
- Nearest centroid.
- k-NN, with `k` selected on training/validation only.
- Parameter-matched two-layer MLP.

Report:

- Accuracy.
- Balanced accuracy.
- Macro-F1.
- Per-layer precision/recall.
- Confusion matrix.
- Mean/standard deviation over seeds.

The current attention model belongs in a separate **trainable fusion** subsection of this table, not among frozen probes.

### Table D: Cross-Modal Alignment

Evaluate over the complete held-out section/donor gallery, not independent mini-batches:

- Bidirectional Recall@1, Recall@5, Recall@10, Recall@50.
- Mean reciprocal rank.
- Median and normalized median rank.
- FOSCTTM.
- Paired-minus-unpaired cosine similarity.
- Effective rank and per-dimension variance.
- Exact-pair, same-layer non-pair, spatial-neighbor, and distant-spot similarities.

The labels used for stratified diagnostics must not enter alignment training.

### Table E: Leave-One-Donor-Out Generalization

Use three outer folds. For each fold, fit the following using the two training donors only:

- HVG selection.
- Expression preprocessing statistics.
- Gene BYOL and its checkpoint selection.
- Cross-modal InfoNCE and its checkpoint selection.
- PCA or learned fusion.
- Supervised probes or attention model.

Freeze all components before processing the held-out donor. If the source image BYOL provenance is unknown, report it as a fixed external feature extractor rather than claiming end-to-end donor generalization.

## Core Representation Ablations

Every row should use the same spots, split manifest, clustering backend, requested cluster count, seeds, refinement policy, and probe capacity.

| ID | Representation | Purpose |
|---|---|---|
| R1 | Joint-HVG expression PCA, 128-d | Classical gene-only baseline |
| R2 | `gene_emb` | Effect of gene BYOL |
| R3 | `img_emb` | External image-BYOL baseline |
| R4 | Unaligned normalized mean of `gene_emb` and `img_emb` | Fusion without InfoNCE |
| R5 | Unaligned concatenation, 256-d | Information-preserving fusion control |
| R6 | Unaligned concatenation reduced to 128-d using train-fitted PCA | Dimension-matched control |
| R7 | `gene_emb_cm_img` alone | Gene-side effect of alignment |
| R8 | `img_emb_cm` alone | Image-side effect of alignment |
| R9 | Aligned normalized mean/bisector | Central hypothesis |
| R10 | Aligned concatenation, 256-d | Test whether the mean discards useful disagreement |
| R11 | Aligned concatenation reduced to 128-d | Dimension-matched fusion |
| R12 | Shuffled-pair InfoNCE followed by bisector | Correspondence sanity control |
| R13 | Learned scalar-weighted normalized sum | Test whether equal weighting is too restrictive |
| R14 | `proj_emb` and its cross-modal variants | Label-assisted ablations; separate table |

The central hypothesis is supported only if R9 consistently beats R2/R3, R4, R7/R8, and dimension-matched R11 under the same evaluator.

## Attention Ablations

Use the same train/validation/test manifests, optimizer budget, early-stopping rule, and as close to equal parameter count as possible.

| ID | Model | Question |
|---|---|---|
| A1 | Linear classifier on gene alone | Gene label information |
| A2 | Linear classifier on image alone | Image label information |
| A3 | Linear classifier on bisector | Linear separability of the central representation |
| A4 | Linear classifier on concatenation | Simple fusion baseline |
| A5 | Parameter-matched MLP on concatenation | Is nonlinear fusion enough? |
| A6 | Mean/bisector + parameter-matched MLP | Is attention better than a learned probe on the mean? |
| A7 | Two-token self-attention only | Contribution of non-degenerate attention |
| A8 | Bidirectional length-one cross blocks without self-attention | Contribution of value projections/residuals |
| A9 | Full current cross-attention + self-attention | Current model |
| A10 | Full model with one modality shuffled at test time | Modality dependence sanity check |
| A11 | Full model with modality dropout during training | Robustness to missing/weak modality |
| A12 | Full model on unaligned embeddings | Contribution of upstream InfoNCE |

If A9 does not beat A5 at matched capacity, there is no evidence that attention itself contributes. If A9 does not beat A12, cross-modal InfoNCE is not helping supervised fusion under that protocol.

## Alignment Objective Ablations

- No alignment.
- True same-spot pairs versus shuffled pairs.
- Gene head only trainable.
- Image head only trainable.
- Both heads trainable.
- Fixed versus learned temperature.
- Output dimensions 32, 64, and 128 with matched downstream capacity.
- Standard exact-spot negatives.
- Immediate spatial neighbors excluded from negatives.
- Same-spot positives plus label-free spatial-neighbor positives.

The final two variants test the main suspected conflict: exact-spot retrieval may disperse domain-related spots.

## Reporting Template

### Per-Section Domain Table

```text
method | representation | clustering | refinement | seed |
151507 ARI/NMI | ... | 151676 ARI/NMI | median | IQR
```

### Integration Table

```text
method | sections | ARI | NMI | BatchKL | iLISI |
graph connectedness | spatial agreement
```

### Supervised Table

```text
representation/model | split unit | accuracy | balanced accuracy |
macro-F1 | per-layer recall | parameters | seeds
```

### Alignment Table

```text
pair | test unit | R@1 | R@5 | R@10 | MRR | median rank |
FOSCTTM | cosine gap | effective rank
```

## Recommended Execution Order

1. Implement the common per-section evaluator and run R1-R11 on existing embeddings.
2. Add Gaussian-mixture, KMeans, and Leiden sensitivity without spatial refinement.
3. Add one common spatial refinement and rerun all representations.
4. Run the attention A1-A12 suite on the existing random-spot split to explain the current `60.8%` result.
5. Add balanced accuracy, macro-F1, per-layer recall, predictions, and split manifests.
6. Add global held-out cross-modal retrieval metrics and shuffled-pair controls.
7. Reproduce STAIG's two integration subsets.
8. Refit the complete pipeline under leave-one-donor-out folds.
9. Run external methods on the exact repository data for the final common benchmark.

## Sources

### Primary Method Papers

- Yang, Y. et al. “STAIG: Spatial transcriptomics analysis via image-aided graph contrastive learning for domain exploration and alignment-free integration.” *Nature Communications* 16, 1067 (2025). https://doi.org/10.1038/s41467-025-56276-0. Local source: `STAIG.pdf`. Supplement: https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41467-025-56276-0/MediaObjects/41467_2025_56276_MOESM1_ESM.pdf
- Long, Y. et al. “Spatially informed clustering, integration, and deconvolution of spatial transcriptomics with GraphST.” *Nature Communications* 14, 1155 (2023). https://doi.org/10.1038/s41467-023-36796-3
- Xu, C. et al. “DeepST: identifying spatial domains in spatial transcriptomics by deep learning.” *Nucleic Acids Research* 50, e131 (2022). https://doi.org/10.1093/nar/gkac901
- Dong, K. and Zhang, S. “Deciphering spatial domains from spatially resolved transcriptomics with an adaptive graph attention auto-encoder.” *Nature Communications* 13, 1739 (2022). https://doi.org/10.1038/s41467-022-29439-6
- Hu, J. et al. “SpaGCN: Integrating gene expression, spatial location and histology to identify spatial domains and spatially variable genes by graph convolutional network.” *Nature Methods* 18, 1342-1351 (2021). https://doi.org/10.1038/s41592-021-01255-8
- Zong, Y. et al. “conST: an interpretable multi-modal contrastive learning framework for spatial transcriptomics.” bioRxiv (2022). https://doi.org/10.1101/2022.01.14.476408
- Liu, W. et al. “Probabilistic embedding, clustering, and alignment for integrating spatial transcriptomics data with PRECAST.” *Nature Communications* 14, 296 (2023). https://doi.org/10.1038/s41467-023-35947-w
- MuCoST: https://doi.org/10.1093/bib/bbae255
- SEDR: https://doi.org/10.1186/s13073-024-01283-x
- stLearn: https://doi.org/10.1038/s41467-023-43120-6
- STAligner: https://doi.org/10.1038/s43588-023-00528-w

### Independent Benchmark Context

- Yuan, Z. et al. “Benchmarking spatial clustering methods with spatially resolved transcriptomics data.” *Nature Methods* 21, 712-722 (2024). https://doi.org/10.1038/s41592-024-02215-8. Code: https://github.com/zhaofangyuan98/SDMBench
- Singhal, V. et al. “BANKSY unifies cell typing and tissue domain segmentation for scalable spatial omics data analysis.” *Nature Genetics* 56, 431-441 (2024). https://doi.org/10.1038/s41588-024-01664-3
- Maynard, K. R. et al. “Transcriptome-scale spatial gene expression in the human dorsolateral prefrontal cortex.” *Nature Neuroscience* 24, 425-436 (2021). https://doi.org/10.1038/s41593-020-00787-0

## Final Position

The existing GeoPatch evidence supports modest same-spot cross-modal alignment and a promising supervised nonlinear fusion result. It does not yet establish successful unsupervised cortical-domain discovery, attention-specific gains, joint-section integration, or donor-level generalization.

The fair question is not whether pooled bisector ARI `0.090` is numerically below STAIG median ARI `0.69`; those values come from different protocols. The fair question is whether the aligned bisector improves over gene-only, image-only, unaligned fusion, aligned single modalities, and concatenation when every representation is clustered section-by-section with the same backend, cluster count, refinement, and seeds. That common experiment will locate the method relative to modern models and reveal which architectural component contributes to any gain.
