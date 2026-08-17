# Three-Phase Supervised Multimodal Classification

## Scope

This report evaluates a label-assisted three-phase representation pipeline for DLPFC
cortical-layer classification. Phases 1 and 2 use `ground_truth` cortical layers to
construct same-layer and different-layer pairs. They are therefore **supervised metric
learning with BYOL-style EMA mechanics**, not self-supervised BYOL.

The primary evaluation is three-fold leave-one-donor-out (LODO). Secondary evaluations
are 12-fold leave-one-section-out (LOSO), 12 within-donor section holdouts, three
within-donor random-spot folds, and one pooled mixed random-spot split. Grouped and
random-spot results answer different questions and are not combined into one score.

## Data And Splits

- Data: 47,329 spots, 12 sections, three donors.
- Inputs: fixed external image `img_emb` and repository gene `gene_emb`, both 128-d.
- Labels: `Layer_1` through `Layer_6` and `WM`.
- Donor 2 has only `Layer_3` through `Layer_6` and `WM`; absent-class recalls are `NA`.
- Primary metric: balanced accuracy, averaged equally over donor or section folds.
- Supporting metrics: accuracy, observed-test-class macro-F1, fixed-axis confusion
  matrices, and per-layer recall.
- Validation selects every metric projector, neural classifier, and GCN checkpoint.
- Outer-test labels are used only for final metrics.
- All spot identity joins use `(section_id, barcode)`.

Immutable split artifacts are under `outputs/splits/`. The five evaluated regimes are:

| Regime | Folds | Interpretation |
|---|---:|---|
| Leave one donor out | 3 | Primary donor transfer |
| Leave one section out | 12 | Section transfer with same donor represented in training |
| Within-donor section holdout | 12 | Transfer between sections of one donor |
| Within-donor random spot | 3 | Transductive interpolation within observed sections |
| Pooled mixed random spot | 1 | Transductive interpolation across all sections/donors |

Important limitation: the supervised phases are fold-trained, but the input `gene_emb`
was pretrained transductively across all sections. LODO/LOSO results are therefore
post-hoc transfer of fold-trained supervised heads over a transductive upstream gene
feature, not strict end-to-end donor-independent gene representation learning. External
`img_emb` is treated as a fixed feature extractor with unknown training provenance.

## Method

### Phases 1 And 2

Phase 1 receives fixed image vectors; Phase 2 receives fixed gene vectors. Each phase
has an online projector `f_theta`, online predictor `q_phi`, and stop-gradient target
projector `f_xi`. For pair `(a,b)`, both directions are optimized:

```text
z_a = normalize(q_phi(f_theta(a)))
t_b = stopgrad(normalize(f_xi(b)))
```

The target parameters follow the online projector after each optimizer step:

```text
xi <- 0.996 * xi + 0.004 * theta
```

For cosine similarity `s`, same-layer indicator `eta=0`, and different-layer indicator
`eta=1`, the supervised pair loss is:

```text
L_pair = (1-eta) * 5 * (1-s)^2 + eta * max(0, s)^2
```

Balanced batches contain equal same-layer and different-layer pairs. The positive term
has five times the negative coefficient, as requested. Fixed validation pairs select
the checkpoint with maximum same-minus-different similarity gap.

### Phase 3

Phase 3 aligns Phase-1 image and Phase-2 gene outputs using exact same-spot pairs from
the training role only. To avoid collapse or silently changing non-paired geometry, the
canonical Phase 3 uses orthogonal Procrustes:

```text
C = G_train^T I_train = U Sigma V^T
gene_phase3  = normalize(G U)
image_phase3 = normalize(I V)
```

Orthogonal maps preserve every within-modality inner product exactly. No different-spot
negative is used in Phase 3. Mean paired cosine rises from approximately zero in the raw
spaces to high positive values:

| Regime | Train paired cosine | Test paired cosine |
|---|---:|---:|
| Leave one donor out | 0.813 +/- 0.034 | 0.782 +/- 0.022 |
| Leave one section out | 0.747 +/- 0.056 | 0.692 +/- 0.069 |
| Within-donor section holdout | 0.723 +/- 0.053 | 0.568 +/- 0.121 |
| Within-donor random spot | 0.691 +/- 0.010 | 0.683 +/- 0.022 |
| Pooled mixed random spot | 0.674 | 0.667 |

### Fusion And Classification

The phase-3 angular bisector is:

```text
b_i = normalize(normalize(image_i) + normalize(gene_i))
```

Evaluated classifiers include class-balanced linear logistic SGD, nearest centroid,
single-modality MLPs, bisector MLP, concatenation MLP, genuine two-token self-attention,
and the historical directional-cross-plus-self model. Cross-attention over one key is
mathematically degenerate because its softmax is always one; that model is retained as
an ablation and described as directional cross-modal residual projection plus
self-attention.

### Log-Euclidean MDM

For image `x_i` and gene `g_i`, the two-observation covariance is:

```text
C_i = 0.5 * (x_i-g_i)(x_i-g_i)^T + epsilon_i I
```

For class `c`, training-only Log-Euclidean prototypes are:

```text
M_c = mean_{i in train,c}(log(C_i))
prediction_i = argmin_c ||log(C_i) - M_c||_F
```

No affine-invariant/Riemannian mean is used. The implementation evaluates the exact
Frobenius distance using the covariance's rank-one closed form.

### Spectral Bisector

Spectral clustering is run independently on test spots from each section using a
nearest-neighbor graph in phase-3 bisector space. The observed annotation count supplies
`k`; labels otherwise enter only after clustering. This is label-count-informed
unsupervised clustering, not supervised classification, so ARI/NMI are reported in a
separate table.

### Graph And Prior-Model Comparisons

- A repository-local two-layer supervised GCN uses coordinate K6 graphs and shared
  weights across sections. It is not called STAIG, SpaGCN, or GraphST.
- Canonical repository STAIG embeddings receive identical frozen linear/centroid
  probes. These embeddings were independently trained per section using all section
  spots, so random-spot results are transductive and grouped transfer is not a native
  STAIG generalization protocol.
- Native supervised STAIG is not a defined native task.
- Native SpaGCN is **NOT RUN**: the package is absent, raw UMI counts are unavailable,
  and no validated SpaGCN embedding artifact exists. Published unsupervised SpaGCN ARI
  is not substituted for supervised classification accuracy.
- Native GraphST is **NOT RUN** for the same dependency/input-provenance reason.

## Primary Donor-Holdout Results

Values are mean +/- standard deviation over three held-out donors.

| Method | Accuracy | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|
| **Supervised GCN, phase-3 gene** | 0.315 +/- 0.081 | **0.309 +/- 0.042** | **0.272 +/- 0.034** |
| Raw gene, linear | 0.315 +/- 0.024 | 0.267 +/- 0.025 | 0.248 +/- 0.010 |
| Supervised GCN, phase-3 concat | 0.269 +/- 0.064 | 0.254 +/- 0.034 | 0.234 +/- 0.023 |
| Phase-3 bisector MLP | 0.248 +/- 0.015 | 0.243 +/- 0.004 | 0.212 +/- 0.020 |
| Two-token self-attention | 0.210 +/- 0.052 | 0.238 +/- 0.011 | 0.194 +/- 0.002 |
| Phase-3 concat MLP | 0.239 +/- 0.061 | 0.237 +/- 0.022 | 0.211 +/- 0.015 |
| Phase-3 bisector, linear | 0.259 +/- 0.034 | 0.232 +/- 0.030 | 0.225 +/- 0.019 |
| Directional-cross plus self-attention | 0.227 +/- 0.020 | 0.231 +/- 0.015 | 0.206 +/- 0.017 |
| STAIG-64 frozen linear probe | 0.245 +/- 0.036 | 0.215 +/- 0.059 | 0.200 +/- 0.079 |
| Phase-3 covariance Log-Euclidean MDM | 0.136 +/- 0.046 | 0.168 +/- 0.057 | 0.130 +/- 0.025 |
| Raw image, linear | 0.191 +/- 0.008 | 0.166 +/- 0.009 | 0.165 +/- 0.011 |

The proposed multimodal pipeline does not beat raw gene features under donor holdout.
Phase 3 succeeds at exact-pair alignment but does not convert that alignment into donor-
generalizable cortical-layer classification. The strongest donor result is instead the
spatial GCN applied to the phase-3 gene branch.

Mean donor-fold per-layer recall further shows that no method solves every layer:

| Method | L1 | L2 | L3 | L4 | L5 | L6 | WM |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw gene linear | .314 | .056 | .364 | .120 | .438 | .062 | .480 |
| Phase-3 bisector MLP | .369 | .131 | .186 | .016 | .434 | .184 | .388 |
| GCN phase-3 gene | .361 | .139 | .202 | .078 | .470 | .212 | **.680** |

L1/L2 means use the two donor folds where those labels occur; donor 2 has no L1/L2.

## Representation-Phase Ablation

Cells are mean balanced accuracy from an identical class-balanced linear probe.

| Representation | LODO | LOSO | Within-donor section | Within-donor random | Pooled random |
|---|---:|---:|---:|---:|---:|
| Raw image | .166 | .168 | .155 | .338 | .240 |
| Phase-1 image | .172 | .193 | .155 | **.548** | **.505** |
| Raw gene | **.267** | **.314** | **.324** | .352 | **.317** |
| Phase-2 gene | .222 | .275 | .304 | .332 | .296 |
| Phase-1/2 bisector | .229 | .226 | .178 | .554 | .527 |
| Phase-1/2 concat | .219 | .238 | .184 | .543 | .513 |
| Phase-3 bisector | .232 | .248 | .198 | **.592** | **.553** |
| Phase-3 concat | .229 | .248 | .176 | .570 | .523 |

Phase 1 strongly improves image interpolation when train and test share sections, but
its gain nearly disappears for held sections/donors. Phase 2 decreases the raw-gene
linear score in every regime. Phase 3 improves multimodal random-spot performance and
paired cosine, but does not surpass raw gene in any grouped-transfer regime.

## Neural Fusion And Mathematical Results

Cells are mean balanced accuracy.

| Method | LODO | LOSO | Within-donor section | Within-donor random | Pooled random |
|---|---:|---:|---:|---:|---:|
| Bisector MLP | **.243** | .243 | .222 | .623 | .556 |
| Concat MLP | .237 | .248 | .218 | **.639** | **.566** |
| Two-token self-attention | .238 | **.255** | .212 | .618 | .561 |
| Directional-cross plus self | .231 | .238 | **.231** | .633 | .552 |
| Log-Euclidean MDM | .168 | .198 | .191 | .472 | .446 |

No attention model consistently beats the simpler concat MLP. Cross-attention over one
token provides no stable gain. Log-Euclidean MDM is substantially weaker, confirming
that the rank-one modality-difference covariance discards useful shared-mean content.

## Graph And STAIG Probe Results

Cells are mean balanced accuracy.

| Method | LODO | LOSO | Within-donor section | Within-donor random | Pooled random |
|---|---:|---:|---:|---:|---:|
| Supervised GCN, phase-3 gene | **.309** | **.360** | **.365** | .491 | .441 |
| Supervised GCN, phase-3 image | .179 | .144 | .162 | .747 | .626 |
| Supervised GCN, phase-3 concat | .254 | .258 | .180 | **.813** | **.685** |
| Frozen STAIG-64 linear probe | .215 | .249 | .204 | **.845** | **.801** |

STAIG's high random-spot probe scores do not transfer to held tissue because its
section-specific embedding spaces are independently fitted and transductive. The GCN
concat result similarly drops from `.813` within-donor random interpolation to `.254`
donor holdout and `.180` within-donor section transfer. Spatial message passing is very
useful when labels and test nodes share a tissue graph, but that is not evidence of
inductive tissue generalization.

## Spectral Bisector Clustering

Values summarize section-level test clustering.

| Regime | Hungarian accuracy | ARI | NMI |
|---|---:|---:|---:|
| Leave one donor out | .385 | .147 | .237 |
| Leave one section out | .402 | .157 | .232 |
| Within-donor section holdout | .396 | .129 | .206 |
| Within-donor random spot | .494 | .268 | .406 |
| Pooled mixed random spot | .423 | .188 | .321 |

Spectral clustering on the bisector does not approach repository Expression STAIG's
per-section refined ARI `0.518`. These metrics are not supervised accuracy and should
not be placed in the same ranking column as the classifiers.

## Main Conclusions

1. The requested image projector is valuable for random-spot interpolation but does not
   generalize its layer separation to unseen donors or sections.
2. The gene metric phase consistently underperforms the raw gene embedding under a
   fixed linear probe.
3. Exact same-spot Phase-3 alignment generalizes as an alignment operation but does not
   establish useful donor-level biological fusion.
4. Concatenation is the strongest neural fusion in random-spot regimes; no attention
   block has a consistent advantage over it.
5. The supervised spatial GCN is the strongest grouped-transfer model tested, using the
   gene branch, but balanced accuracy remains only `0.309` for donor holdout.
6. STAIG frozen embeddings are exceptionally strong for within-section random probes
   (`0.845` balanced accuracy) but weak for grouped transfer; this is expected from
   independent transductive per-section training.
7. Log-Euclidean MDM and spectral bisector clustering are mathematically valid controls,
   but neither is competitive.
8. Native supervised SpaGCN/GraphST results are unavailable and are not fabricated.

## Artifacts

- Aggregated tables: `outputs/evaluation/20260815_supervised_study_aggregate_v1/
- Regime-level balanced‑accuracy confidence summaries: `outputs/evaluation/20260815_supervised_study_aggregate_v1/regime_balanced_accuracy_summary.csv`

## Aggregate‑level confidence summaries

| Regime | n_methods | mean_balanced_accuracy | std_of_means | ci_lower | ci_upper | n_folds_per_fold |
|---|---:|---:|---:|---:|---:|---:|
| leave_one_donor_out | 36 | 0.2146 | 0.0337 | 0.2036 | 0.2256 | 3 |
| leave_one_section_out | 36 | 0.2371 | 0.0433 | 0.2230 | 0.2512 | 12 |
| pooled_mixed_random_spot | 36 | 0.4563 | 0.1363 | 0.4117 | 0.5008 | 1 |
| within_donor_random_spot | 36 | 0.5165 | 0.1486 | 0.4680 | 0.5651 | 3 |
| within_donor_section_holdout | 36 | 0.2237 | 0.0617 | 0.2035 | 0.2438 | 12 |

*Mean ± 95 % CI of balanced accuracy across methods within each regime. CI computed as mean ± 1.96·sd/√k where k = number of methods per regime.*

## Main Conclusions`
- Study source manifest: `outputs/evaluation/20260815_supervised_study_manifest_v1.json`
- Split manifests: `outputs/splits/20260815_dlpfc_*/`
- Phase models/embeddings: `outputs/supervised_multiphase/supervised_multiphase_*_v1/`
- Common evaluations: `outputs/evaluation/20260815_supervised_*_seed0_v1/`
- Supervised GCN: `outputs/prior_models/supervised_gcn_*_seed0_v1/`
- Reusable APIs: `src/train/supervised_multiphase.py`,
  `src/train/supervised_evaluation.py`,
  `src/train/aggregate_supervised_results.py`,
  `src/prior_models/supervised_gcn.py`, and `src/splits/dlpfc.py`.

All headline comparisons use seed 0. Donor and section folds quantify biological-unit
variation; random-spot optimization-seed variability remains a limitation and should be
expanded before a final publication claim.
