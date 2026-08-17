# Graph-Positive Multimodal Architecture

## Scope

This document specifies the seed-0 graph-pooled orthogonal positive-only candidate. It uses frozen external image-BYOL features and repository gene-BYOL features. It does not train image BYOL, use cortical labels during representation fitting, or construct different-spot negative pairs.

This is now a retained ablation. The newer primary candidate is documented in `SPATIAL_MULTIMODAL_CONTRASTIVE_ARCHITECTURE.md` and adds explicit intra-gene, intra-image, and cross-modal graph contrastive terms.

The canonical run is `graph_pooled_procrustes_img_k6r5_seed0`.

## Inputs

For spot `i` in section `s`, the model receives:

- Physical coordinate `p_i = (x_i, y_i)` from `adata.obsm["spatial"]`.
- Frozen gene embedding `g_i in R^128` from `adata.obsm["gene_emb"]`.
- Frozen image embedding `v_i in R^128` from `adata.obsm["img_emb"]`.

All spot identities are `(section_id, barcode)`. Graph edges never cross sections.

## 1. Coordinate Graph

Within each section, Euclidean coordinate distance is

```text
d_spatial(i,j) = ||p_i - p_j||_2.
```

Each spot nominates its six nearest physical neighbors. Directed nominations are converted to an undirected candidate set:

```text
E0 = {{i,j}: j in KNN_6(i) or i in KNN_6(j)}.
```

The canonical full graph contained 144,857 candidate edges.

## 2. Label-Free Edge Pruning

Normalize the frozen embeddings:

```text
g_hat_i = g_i / ||g_i||_2,
v_hat_i = v_i / ||v_i||_2.
```

For every coordinate candidate edge, compute within-modality similarities:

```text
s_g(i,j) = g_hat_i^T g_hat_j,
s_v(i,j) = v_hat_i^T v_hat_j.
```

For each endpoint `i`, independently rank incident coordinate candidates by `s_g` and `s_v`. Let `T_g(i)` and `T_v(i)` be the top five candidate sets. A directed nomination requires agreement:

```text
N(i) = T_g(i) intersection T_v(i).
```

An undirected edge survives only under mutual nomination:

```text
E = {{i,j} in E0: j in N(i) and i in N(j)}.
```

This process uses no annotation labels. It retained 72,089 edges (`49.77%`), mean degree `3.05`, maximum degree `5`, and isolated fraction `3.16%`.

## 3. Neighborhood Pooling

For each non-isolated spot, pool self and retained immediate neighbors separately in each modality:

```text
g_bar_i = normalize(g_hat_i + mean_{j in N_E(i)} g_hat_j),
v_bar_i = normalize(v_hat_i + mean_{j in N_E(i)} v_hat_j).
```

For isolated spots:

```text
g_bar_i = g_hat_i,
v_bar_i = v_hat_i.
```

Pooling makes the representation explicitly local and transductive. It can improve spatial continuity but may smooth true cortical boundaries.

## 4. Positive Set

The cross-modal positive set contains exact spots and both directions of every retained spatial edge:

```text
P = {(i,i)} union {(i,j),(j,i): {i,j} in E}.
```

No pair outside `P` is assigned a negative label or pushed apart.

## 5. Orthogonal Positive Alignment

Construct the positive cross-covariance:

```text
C = sum_i g_bar_i^T v_bar_i
    + sum_{{i,j} in E} (g_bar_i^T v_bar_j + g_bar_j^T v_bar_i).
```

Compute its singular value decomposition:

```text
C = U Sigma V^T.
```

The aligned outputs are

```text
z_g_i = normalize(g_bar_i U),
z_v_i = normalize(v_bar_i V).
```

Because `U^T U = I` and `V^T V = I`, orthogonal fitting cannot collapse distinct pooled vectors and preserves all within-modality inner products:

```text
(G_bar U)(G_bar U)^T = G_bar G_bar^T,
(V_bar V)(V_bar V)^T = V_bar V_bar^T.
```

The optimization is equivalent to maximizing total positive cross-modal similarity subject to exact geometry-preservation constraints. It has a closed-form solution and requires no optimizer, temperature, negative sampling, or early stopping.

## 6. Fusion Representations

Equal angular bisector:

```text
b_i = normalize(z_g_i + z_v_i).
```

Weighted angular mean with gene weight `alpha`:

```text
b_i(alpha) = normalize(alpha z_g_i + (1-alpha) z_v_i).
```

The full fixed sensitivity curve uses `alpha in {0, 0.25, 0.5, 0.75, 1}`.

Information-preserving concatenation:

```text
c_i = [z_g_i ; z_v_i] in R^256.
```

The current experiments show that concatenation is stronger than the equal bisector after common spatial refinement. Therefore, the bisector is retained as a central ablation, not declared the best model.

## 7. Domain Clustering

For each DLPFC section independently:

1. Derive the requested cluster count from the exact number of observed valid annotations. This is label-count-informed clustering.
2. Fit KMeans for ten explicit seeds, or PCA-30 plus full-covariance GMM as a sensitivity backend.
3. Use labels only after assignments for ARI, NMI, and Hungarian accuracy.

Common optional refinement uses coordinate 6-NN and a strict majority:

```text
c'_i = mode({c_j: j in KNN_6(i)})
```

only when one cluster receives at least four of six votes. Updates are synchronous.

## 8. Evaluation Regimes

Per-section domain discovery reports section-level ARI/NMI, Hungarian accuracy, median/IQR, clustering seeds, backend, and refinement.

Cross-modal diagnostics report complete-gallery Recall@K, MRR, median rank, FOSCTTM, cosine gap, and effective rank. Retrieval is diagnostic because the objective accepts neighborhoods, not only exact identity.

Frozen probes report accuracy, balanced accuracy, and macro-F1 under a random-spot transductive split. They are not donor-generalization evidence.

Integration reports biological ARI/NMI together with section/donor mixing. Repository iLISI and BatchKL values are local surrogates and are not numerically interchangeable with STAIG.

## 9. Current Seed-0 Results

The strongest per-section unrefined row is the aligned 25% gene weighted mean at ARI/NMI `0.18455/0.25694`. This weight is sensitivity-only because labels were not reserved for weight selection.

The strongest refined row is aligned concatenation/PCA-128 at approximately ARI/NMI `0.210/0.293`.

The equal aligned bisector reaches ARI/NMI `0.161/0.240`, or `0.180/0.273` after refinement.

The aligned-concatenation frozen logistic probe reaches accuracy/balanced accuracy/macro-F1 `0.5569/0.4498/0.4530`.

These improve repository controls but remain substantially below STAIG's published native per-section median ARI/NMI `0.69/0.71`.

## 10. Limitations

- The canonical representation uses the full section graph and is transductive.
- Edge pruning favors relationships already supported by both frozen modalities.
- Pooling can erase narrow layers or genuine boundaries.
- The external image-BYOL training provenance is unknown.
- Only seed 0 has a complete downstream evaluation for this candidate.
- No strict leave-one-donor-out refit has been performed.
- Better exact-pair retrieval is neither expected nor sufficient for domain discovery.
