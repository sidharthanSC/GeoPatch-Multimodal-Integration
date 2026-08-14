# GeoPatch Multimodal Benchmark -- Results

This document reports the results of executing `MODEL_BENCHMARK_REPORT.md`'s
"Recommended Execution Order" steps 1-5 against this repository's existing frozen
embeddings (`checkpoints/dlpfc.pkl`). New code lives in `src/benchmark/`; new outputs
live in `outputs/benchmark/`.

## Scope of this run

Per explicit user decision, this run covers:

- **Table A -- per-section domain discovery**, R1-R11 (`src/benchmark/representations.py`),
  clustered independently per DLPFC section with KMeans, Gaussian-mixture (sklearn
  `GaussianMixture`, the Python substitute for R's `mclust`), and Leiden
  (`leidenalg`/`python-igraph`, newly added dependencies), across 10 clustering seeds
  each (`src/benchmark/clustering_backends.py`, `src/benchmark/per_section_eval.py`).
- **One common spatial refinement pass** (majority-vote over each spot's 6 spatial
  neighbors, applied identically to every representation/backend's cluster assignments;
  `src/benchmark/refinement.py`), reported as a second table alongside the raw one.
- **Attention ablations A1-A12** (`src/benchmark/attention_ablations.py`), using the
  same outer train/val/test split as methods 3/4, seed 0.

**Explicitly out of scope for this run** (per the user's chosen priority, not attempted):
R12 (shuffled-pair InfoNCE control) and R13/R14, Table B (joint-section integration),
Table D (cross-modal retrieval metrics), Table E (leave-one-donor-out refits), and
external-method reruns (STAIG/GraphST/etc. on this repository's exact data). These all
require either retraining upstream components or infrastructure this run did not set
up, and remain a documented follow-up. The attention ablation suite also ran at a
single seed (0), not the report's requested multi-seed protocol -- see the caveat in
"Interpretation" below.

## Design choices made for this run

- **Gaussian-mixture backend**: `sklearn.mixture.GaussianMixture(covariance_type="diag")`
  rather than R's `mclust` (no rpy2/R dependency added) and rather than `"full"`
  covariance (a full 128x128 covariance per component has far more free parameters
  than a ~600-spot cluster can support at a ~4200-spot section).
- **Leiden backend**: resolution is binary-searched once per (section, representation)
  to hit exactly `k=7` clusters, then reused across all 10 seeds (varying only Leiden's
  own `random_state`) -- avoids rebuilding the neighbor graph and re-searching
  resolution 10x per representation.
- **PCA-based representations (R1, R6, R11)**: fit per-section, matching Table A's
  per-section protocol ("Produce one embedding per spot without using layer labels" --
  independently per section, not pooling PCA fits across sections).
- **Spatial refinement**: single-pass majority vote (not iterated to convergence),
  applied post-hoc to every backend's cluster assignments uniformly -- the standard
  SpaGCN/STAGATE-style refinement, never used to influence clustering itself.
- **Attention ablations**: A10 (modality-shuffle sanity check) is evaluated post-hoc
  from the already-trained A9 model rather than as a separate training run, since
  shuffling is a test-time perturbation of the same model, not a different model.

## Table A -- Per-Section Domain Discovery (raw)

Median +/- IQR across the 12 DLPFC sections (each section's value is itself averaged
over 10 clustering seeds first). ARI/NMI are primary; accuracy is secondary
(Hungarian-matched).

| Representation | Backend | ARI (median) | ARI IQR | NMI (median) | NMI IQR | Accuracy (median) |
|---|---|---:|---:|---:|---:|---:|
| R1 | gmm | 0.094 | [0.053, 0.137] | 0.159 | [0.114, 0.204] | 0.318 |
| R2 | gmm | 0.093 | [0.059, 0.112] | 0.163 | [0.135, 0.171] | 0.282 |
| R3 | gmm | 0.108 | [0.098, 0.135] | 0.179 | [0.136, 0.212] | 0.347 |
| R4 | gmm | 0.101 | [0.063, 0.135] | 0.168 | [0.135, 0.184] | 0.308 |
| R5 | gmm | 0.094 | [0.061, 0.116] | 0.164 | [0.132, 0.177] | 0.289 |
| R6 | gmm | 0.114 | [0.107, 0.123] | 0.180 | [0.148, 0.212] | 0.345 |
| R7 | gmm | 0.097 | [0.077, 0.105] | 0.165 | [0.142, 0.179] | 0.286 |
| R8 | gmm | 0.144 | [0.098, 0.193] | 0.252 | [0.182, 0.288] | 0.368 |
| R9 | gmm | 0.143 | [0.118, 0.178] | 0.244 | [0.200, 0.273] | 0.355 |
| R10 | gmm | 0.114 | [0.085, 0.139] | 0.189 | [0.149, 0.220] | 0.314 |
| R11 | gmm | 0.105 | [0.086, 0.120] | 0.190 | [0.156, 0.221] | 0.328 |
| R1 | kmeans | 0.149 | [0.113, 0.159] | 0.195 | [0.182, 0.230] | 0.352 |
| R2 | kmeans | 0.103 | [0.059, 0.122] | 0.167 | [0.136, 0.174] | 0.292 |
| R3 | kmeans | 0.153 | [0.057, 0.241] | 0.240 | [0.122, 0.293] | 0.422 |
| R4 | kmeans | 0.131 | [0.095, 0.217] | 0.230 | [0.173, 0.304] | 0.363 |
| R5 | kmeans | **0.195** | [0.124, 0.237] | 0.251 | [0.192, 0.317] | 0.419 |
| R6 | kmeans | 0.190 | [0.124, 0.239] | 0.248 | [0.192, 0.315] | 0.417 |
| R7 | kmeans | 0.101 | [0.074, 0.112] | 0.168 | [0.148, 0.186] | 0.300 |
| R8 | kmeans | 0.146 | [0.077, 0.187] | 0.244 | [0.172, 0.290] | 0.369 |
| R9 | kmeans | 0.138 | [0.097, 0.184] | 0.236 | [0.199, 0.270] | 0.366 |
| R10 | kmeans | 0.132 | [0.090, 0.172] | 0.240 | [0.183, 0.278] | 0.338 |
| R11 | kmeans | 0.131 | [0.092, 0.172] | 0.240 | [0.182, 0.277] | 0.338 |
| R1 | leiden | 0.102 | [0.074, 0.119] | 0.120 | [0.114, 0.142] | 0.346 |
| R2 | leiden | 0.090 | [0.060, 0.109] | 0.162 | [0.134, 0.171] | 0.281 |
| R3 | leiden | 0.164 | [0.081, 0.232] | 0.252 | [0.148, 0.337] | 0.394 |
| R4 | leiden | 0.163 | [0.110, 0.256] | 0.240 | [0.183, 0.321] | 0.410 |
| R5 | leiden | 0.168 | [0.109, 0.247] | 0.266 | [0.201, 0.331] | 0.399 |
| R6 | leiden | 0.175 | [0.111, 0.241] | 0.271 | [0.204, 0.329] | 0.416 |
| R7 | leiden | 0.096 | [0.076, 0.107] | 0.161 | [0.141, 0.170] | 0.296 |
| R8 | leiden | 0.175 | [0.124, 0.231] | 0.262 | [0.207, 0.318] | 0.405 |
| R9 | leiden | 0.174 | [0.123, 0.247] | 0.262 | [0.209, 0.317] | 0.410 |
| R10 | leiden | 0.168 | [0.100, 0.246] | 0.259 | [0.211, 0.318] | 0.401 |
| R11 | leiden | 0.164 | [0.097, 0.235] | 0.254 | [0.205, 0.320] | 0.398 |

## Table A -- Per-Section Domain Discovery (spatially refined)

Same protocol, cluster assignments passed through the single-pass spatial-neighbor
majority-vote refinement before scoring.

| Representation | Backend | ARI (median) | ARI IQR | NMI (median) | NMI IQR | Accuracy (median) |
|---|---|---:|---:|---:|---:|---:|
| R1 | gmm | 0.123 | [0.070, 0.183] | 0.194 | [0.131, 0.256] | 0.361 |
| R2 | gmm | 0.110 | [0.071, 0.140] | 0.183 | [0.162, 0.195] | 0.309 |
| R3 | gmm | 0.124 | [0.111, 0.148] | 0.200 | [0.151, 0.236] | 0.365 |
| R4 | gmm | 0.127 | [0.077, 0.170] | 0.202 | [0.163, 0.215] | 0.334 |
| R5 | gmm | 0.113 | [0.076, 0.138] | 0.184 | [0.157, 0.201] | 0.316 |
| R6 | gmm | 0.135 | [0.127, 0.140] | 0.206 | [0.174, 0.230] | 0.361 |
| R7 | gmm | 0.119 | [0.100, 0.136] | 0.188 | [0.166, 0.203] | 0.320 |
| R8 | gmm | 0.164 | [0.112, 0.214] | 0.274 | [0.203, 0.317] | 0.384 |
| R9 | gmm | 0.178 | [0.140, 0.205] | 0.286 | [0.243, 0.310] | 0.382 |
| R10 | gmm | 0.146 | [0.114, 0.164] | 0.219 | [0.186, 0.250] | 0.345 |
| R11 | gmm | 0.131 | [0.117, 0.149] | 0.225 | [0.187, 0.259] | 0.365 |
| R1 | kmeans | 0.193 | [0.149, 0.218] | 0.244 | [0.220, 0.286] | 0.396 |
| R2 | kmeans | 0.127 | [0.072, 0.148] | 0.192 | [0.164, 0.202] | 0.318 |
| R3 | kmeans | 0.160 | [0.073, 0.257] | 0.253 | [0.144, 0.313] | 0.428 |
| R4 | kmeans | 0.157 | [0.108, 0.255] | 0.254 | [0.220, 0.349] | 0.384 |
| R5 | kmeans | 0.212 | [0.133, 0.279] | 0.293 | [0.225, 0.370] | 0.435 |
| R6 | kmeans | 0.212 | [0.132, 0.280] | 0.293 | [0.224, 0.370] | 0.433 |
| R7 | kmeans | 0.123 | [0.095, 0.132] | 0.200 | [0.175, 0.214] | 0.326 |
| R8 | kmeans | 0.168 | [0.088, 0.214] | 0.262 | [0.202, 0.320] | 0.396 |
| R9 | kmeans | 0.174 | [0.122, 0.208] | 0.276 | [0.243, 0.305] | 0.394 |
| R10 | kmeans | 0.161 | [0.111, 0.202] | 0.273 | [0.225, 0.312] | 0.367 |
| R11 | kmeans | 0.160 | [0.113, 0.201] | 0.273 | [0.225, 0.310] | 0.366 |
| R1 | leiden | 0.155 | [0.113, 0.184] | 0.183 | [0.174, 0.207] | 0.396 |
| R2 | leiden | 0.110 | [0.072, 0.130] | 0.184 | [0.154, 0.195] | 0.304 |
| R3 | leiden | 0.170 | [0.093, 0.249] | 0.266 | [0.165, 0.355] | 0.403 |
| R4 | leiden | 0.175 | [0.129, 0.314] | 0.266 | [0.227, 0.386] | 0.429 |
| R5 | leiden | 0.192 | [0.120, 0.289] | 0.291 | [0.235, 0.374] | 0.420 |
| R6 | leiden | 0.210 | [0.122, 0.280] | 0.294 | [0.231, 0.367] | 0.431 |
| R7 | leiden | 0.127 | [0.097, 0.137] | 0.194 | [0.168, 0.200] | 0.323 |
| R8 | leiden | 0.199 | [0.137, 0.245] | 0.283 | [0.233, 0.347] | 0.424 |
| R9 | leiden | **0.219** | [0.152, 0.276] | **0.310** | [0.243, 0.351] | **0.443** |
| R10 | leiden | 0.207 | [0.118, 0.269] | 0.301 | [0.250, 0.350] | 0.433 |
| R11 | leiden | 0.203 | [0.116, 0.255] | 0.299 | [0.244, 0.351] | 0.430 |

## Attention Ablations A1-A12

Single seed (0), shared outer 70/30 train-val/test split (stratified by layer),
identical optimizer/early-stopping settings. `n_test` = 14,199 for every row.

| Ablation | Representation | Accuracy | Loss | Params |
|---|---|---:|---:|---:|
| A1 -- linear, gene only | `gene_emb_cm_img` | 0.4548 | 1.4666 | 903 |
| A2 -- linear, image only | `img_emb_cm` | 0.4742 | 1.3976 | 903 |
| A3 -- linear, bisector | aligned bisector | 0.5022 | 1.3257 | 903 |
| A4 -- linear, concat | aligned, 256-d | 0.5139 | 1.2841 | 1,799 |
| A5 -- MLP, concat | aligned, 256-d | 0.5416 | 1.2075 | 67,591 |
| A6 -- MLP, bisector | aligned bisector | 0.5242 | 1.2509 | 34,823 |
| A7 -- self-attention only | aligned | 0.6192 | 1.0153 | 133,383 |
| A8 -- cross-attention only | aligned | 0.6110 | 1.0396 | 199,687 |
| A9 -- full model (cross + self) | aligned | 0.6081 | 1.0416 | 265,991 |
| A11 -- full model + modality dropout | aligned | 0.6169 | 1.0251 | 265,991 |
| A12 -- full model, unaligned | `gene_emb` / `img_emb` | **0.6560** | **0.9303** | 265,991 |
| A10 -- A9 model, image shuffled at test | aligned | 0.3014 | 2.5440 | (A9's) |

## Interpretation

### Table A: does the aligned bisector (R9) beat single-modality and dimension-matched
### fusion representations, per the report's central hypothesis?

**Only after spatial refinement, and then consistently.** In the refined table, R9 is
the single best cell in the entire table (ARI 0.219 / NMI 0.310 / accuracy 0.443,
Leiden) and beats every one of R2, R3, R4, R7, R8, R11 under all three backends. Before
refinement, R9 is competitive but does **not** clearly win: it is essentially tied with
R8 (`img_emb_cm` alone) under all three backends (e.g. Leiden: R9 0.174 vs. R8 0.175),
and both GMM and Leiden give R8 a marginal edge. So the raw central hypothesis --
that fusing the two modalities beats using image alone -- is not supported pre-
refinement; it is supported post-refinement. One plausible reading: the bisector
produces assignments that are locally noisier but more spatially coherent in aggregate
than single-modality clustering, so the same refinement pass recovers more signal from
it. That is a hypothesis, not something this run verified directly (would need a
spatial-autocorrelation metric on raw cluster assignments to confirm).

A second, unplanned finding cuts against the "alignment helps" framing directly:
**unaligned concatenation (R5) and its PCA reduction (R6) match or beat aligned
concatenation (R10/R11) under KMeans in both the raw and refined tables** (e.g. raw
KMeans: R5 0.195 vs. R10 0.132; refined KMeans: R6 0.212 vs. R11 0.160). Under Leiden
the gap narrows or reverses in refined (R9/R10 pull ahead of R5/R6). So whether
cross-modal InfoNCE alignment helps *concatenation-style* fusion is backend-dependent
and not a clean win either way in this run.

**Every representation in every cell remains far below the published per-section
DLPFC references** in `MODEL_BENCHMARK_REPORT.md` (STAIG median ARI 0.69, GraphST
0.60, DeepST mean 0.515, MuCoST mean 0.526, PRECAST median 0.434). The best cell found
here (R9, Leiden, refined, ARI 0.219) is roughly a third of STAIG's median. This
confirms the report's own framing: the fair comparison is the internal one (does
alignment help within this pipeline), not a claim of competitiveness with the
literature -- this pipeline's per-section domain discovery is still weak in absolute
terms.

Gene-only representations (R2, R7) are the weakest cells in almost every row of both
tables -- the gene BYOL encoder alone carries comparatively little cortical-layer
signal relative to the image side, consistent with the report's note that the image
modality was suspected to dominate.

### Attention ablations: does attention contribute, and does alignment help supervised fusion?

**Attention contributes over plain fusion**: A9 (0.6081) clearly beats A5, the
parameter-matched MLP on concatenation (0.5416), satisfying the report's first test
("if A9 does not beat A5 ... no evidence attention itself contributes" -- it does
beat it, by 6.6 points).

**Cross-modal InfoNCE alignment does *not* help this supervised fusion task** -- and
by a wide margin. A12 (full attention architecture on **unaligned** `gene_emb`/
`img_emb`) reached 0.6560 accuracy, beating A9's aligned version (0.6081) by 4.8
points and beating every other ablation in the table, aligned or not. Per the report's
own stated test ("if A9 does not beat A12, cross-modal InfoNCE is not helping
supervised fusion under that protocol"), this is a direct negative result for the
InfoNCE alignment step's value to downstream layer classification: the raw,
un-aligned BYOL embeddings are a *better* input to the attention classifier than the
aligned ones.

**An unplanned architectural finding**: A9 (both cross- and self-attention, 0.6081) is
*worse* than either sub-block alone -- A7 (self-attention only, 0.6192) and A8
(cross-attention only, 0.6110) both beat it, despite A9 having the most parameters
(265,991 vs. 133,383 / 199,687). A11 (A9's exact architecture plus modality dropout
during training) recovers most of that gap (0.6169), consistent with A9 mildly
overfitting its extra capacity rather than cross- and self-attention being
architecturally incompatible. This is a **single-seed** observation (seed 0 only, not
the report's requested multi-seed protocol -- see the scope caveat above); the A7/A8/A9
gaps (1-1.5 points) are close enough to plausibly be seed noise, and would need
repeated-seed runs to confirm as a real effect before acting on it (e.g. before
permanently dropping the combined architecture in favor of a single attention block).

**A10 (modality-shuffle sanity check) passes**: breaking the image/gene pairing at
test time collapses A9's accuracy from 0.6081 to 0.3014 -- below the ~0.372 pooled
majority-class baseline. This confirms the trained model genuinely relies on true
spot-level cross-modal correspondence rather than a single-modality shortcut, which is
a necessary (if not sufficient) sanity check before trusting the accuracy numbers above.

### Bottom line

1. The aligned bisector (R9) is the best unsupervised per-section representation found
   in this run, but only once spatial refinement is applied -- pre-refinement it is a
   tie with the image-only embedding, not a clear win, and all results remain well
   below published per-section DLPFC benchmarks.
2. Attention-based fusion clearly beats plain MLP fusion for supervised layer
   classification, but the specific combined cross+self architecture (A9, the
   repository's existing method 4) is *not* the best configuration found: both a
   self-attention-only variant and an unaligned-input variant beat it outright.
3. The central premise that cross-modal InfoNCE alignment improves downstream tasks is
   contradicted in the one place this run tested it directly (A9 vs. A12) -- alignment
   made the supervised classifier *worse*, not better. This is worth prioritizing as a
   follow-up investigation (why would the aligned embeddings underperform the raw ones
   here?) over further ablation coverage.
4. Steps 6-9 of the report's execution order (Table B/D/E, external-method reruns) are
   still untouched and are the natural next scope if this line of investigation
   continues.
