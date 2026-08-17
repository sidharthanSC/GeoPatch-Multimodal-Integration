# Spatial Multimodal Contrastive Architecture

## Scope

This document defines `spatial_multimodal_contrastive_img_k6r5_seed0`, the current strongest domain-discovery candidate. It learns trainable projection heads over fixed gene and external image embeddings. Raw image BYOL cannot be retrained because image patches and image-BYOL training code are not available in this repository.

The run uses one model seed, one downstream clustering seed, 12 section-level biological evaluation units, and 30 fixed transductive epochs.

## Inputs

For spot `i`:

```text
p_i in R^2       physical coordinate from adata.obsm["spatial"]
g_i in R^128     frozen gene-BYOL embedding
v_i in R^128     fixed external image-BYOL embedding
```

## Coordinate Graph And Pruning

Within each section, create a Euclidean coordinate graph:

```text
E0 = {{i,j}: j in KNN_6(p_i) or i in KNN_6(p_j)}.
```

For coordinate candidates, calculate frozen within-modality cosine similarity:

```text
s_g(i,j) = cos(g_i,g_j),
s_v(i,j) = cos(v_i,v_j).
```

Each endpoint keeps the intersection of its top-five gene and image candidates. An edge is retained only under mutual nomination:

```text
E = {{i,j} in E0: j in Top5_g(i) intersection Top5_v(i)
                    and i in Top5_g(j) intersection Top5_v(j)}.
```

No annotation labels enter graph construction.

## Projection Heads

Each modality has an independent trainable head:

```text
h_g(g) = Linear -> LayerNorm -> GELU -> Dropout -> Linear
h_v(v) = Linear -> LayerNorm -> GELU -> Dropout -> Linear.
```

Outputs are normalized:

```text
z_g_i = h_g(g_i) / ||h_g(g_i)||_2,
z_v_i = h_v(v_i) / ||h_v(v_i)||_2.
```

## Positive Sets

For anchor `i`, retained graph neighbors are `N(i)`.

Cross-modal positives include the exact spot and retained neighbors:

```text
P_gv(i) = {i} union N(i).
```

Intra-modal positives contain retained neighbors:

```text
P_gg(i) = N(i),
P_vv(i) = N(i).
```

Unretained coordinate candidates are uncertain boundary cases and are excluded from negatives. Non-neighbor spots in the same section are negatives. Cross-section spots are not negatives.

## Multi-Positive InfoNCE

For query `q_i`, gallery keys `k_j`, positive set `P(i)`, and valid comparison set `A(i)`:

```text
L_MP(i) = -log [sum_{j in P(i)} exp(q_i^T k_j / tau)
                / sum_{j in A(i)} exp(q_i^T k_j / tau)].
```

The learned temperature is constrained by the existing CLIP logit-scale clamp.

The four losses are:

```text
L_gv = MPInfoNCE(z_g queries, z_v gallery, P_gv),
L_vg = MPInfoNCE(z_v queries, z_g gallery, P_gv),
L_gg = MPInfoNCE(z_g queries, z_g gallery, P_gg),
L_vv = MPInfoNCE(z_v queries, z_v gallery, P_vv).
```

The total canonical objective is

```text
L = 0.5(L_gv + L_vg) + L_gg + L_vv.
```

All weights are one. Unlike exact-spot CLIP, retained biological neighbors are positives rather than negatives.

## Graph Batching

Each batch begins with 256 random anchors and gathers their retained neighbors. Projection is performed once for every unique gathered spot. Masks identify exact positives, retained-neighbor positives, uncertain coordinate candidates, valid same-section negatives, and excluded cross-section pairs.

Every spot acts as an anchor once per epoch. The complete section graph is used, making this a transductive domain-discovery regime comparable in spirit, but not identical, to STAIG.

## Fusion

Equal angular bisector:

```text
b_i = normalize(z_g_i + z_v_i).
```

Concatenation:

```text
c_i = [z_g_i ; z_v_i].
```

Dimension-matched concatenation uses section-fitted PCA to 128 dimensions for clustering sensitivity.

## Evaluation

Per-section KMeans and GMM use the exact observed annotation count only to choose the number of clusters. Labels are used after clustering for ARI, NMI, and Hungarian accuracy. Common refinement changes a cluster only under a unique four-of-six spatial-neighbor majority.

The seed-0 KMeans results are:

```text
equal bisector: ARI/NMI 0.19783/0.29149
equal bisector refined: 0.21504/0.32875
concat PCA-128: 0.20622/0.28176
concat PCA-128 refined: 0.22356/0.31973
```

GMM gives the consistent bisector result `0.19627/0.27765`, refined to `0.21330/0.31494`.

## Limitations

- Spatial neighbors can cross true cortical boundaries; proximity does not prove shared layer identity.
- The graph is pruned using the same frozen modalities later trained, creating an agreement-selection bias.
- The gene projection has low effective rank (`3.31`) despite improved clustering.
- Raw image BYOL and raw-expression gene BYOL were not retrained by this objective.
- The run is transductive and does not establish donor-level generalization.
- Only one model and clustering seed were run. Twelve sections provide biological replicates but do not measure optimization-seed variance.
- Refined ARI near `0.22` remains far below STAIG's published native median `0.69`; the method does not perfectly align layers.
