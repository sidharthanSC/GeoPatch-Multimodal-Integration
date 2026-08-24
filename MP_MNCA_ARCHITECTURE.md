# MP-MNCA Architecture

## Scope And Source

This document describes the current repository implementation of **MP-MNCA**
(Morphology-Prior Masked Neighbour Cross-Attention), focusing on the Phase 1
model used by the active DLPFC convergence run. The source of truth is:

- `src/mp_mnca/config.py`
- `src/mp_mnca/data.py`
- `src/mp_mnca/phase1.py`
- `src/mp_mnca/model.py`
- `src/mp_mnca/train_phase1.py`

The Phase 1 source follows SM/GitHub commit `0027399`. The active convergence
job uses an external checkpointing wrapper in `src/convergence/mpmnca_reference.py`;
that wrapper imports the model and loss without changing MP-MNCA source code.

## What Phase 1 Is

Phase 1 is an unsupervised, per-section gene-expression cross-attention model.
For each spot, it:

1. finds six nearest spatial neighbours from the 2-D tissue coordinates;
2. uses the spot and neighbour 3,000-HVG gene-expression vectors in multi-head
   cross-attention;
3. biases attention using image-feature similarity and a learned relative-position
   function;
4. trains two masked gene-expression views with a same-spot plus spatial-neighbour
   contrastive objective.

The saved Phase 1 representation is 3,000-dimensional. There is no learned
low-dimensional bottleneck in this phase.

```text
coordinates ----------> 6-NN indices -------------------------------+
                                                                    |
img_emb (128) -> standardize -> PCA(16) -> morphology prior -------+---> attention logits
                                                                    |
gene expression (3,000 HVGs) -> masking -> Q/K/V cross-attention --+---> residual + LayerNorm
                                                                    |
two masked views -------------------------------------------------------> neighbour contrastive loss
```

## Inputs Per Section

Each DLPFC section is processed independently. No spot from another section is
used as a spatial neighbour.

| Input | Repository field | Shape | Use |
|---|---|---:|---|
| Gene expression | `adata.obsm["feat"]` by default | `n_spots x 3000` | Phase 1 center and neighbour gene tokens |
| Image embedding | `adata.obsm["img_emb"]` | `n_spots x 128` | Morphology prior after standardization and PCA |
| Spatial coordinates | `adata.obsm["spatial"]` | `n_spots x 2` | Coordinate 6-NN neighbours and position bias |
| Ground truth | `adata.obs["ground_truth"]` | `n_spots` | Evaluation only; not used by the default Phase 1 loss |

### Spatial Neighbourhood

`build_spatial_knn_graph()` fits `NearestNeighbors(n_neighbors=7)` on the
section coordinates and discards self-neighbours. Thus every spot has six
coordinate-nearest neighbours by default (`n_neighbors=6`). Those same
neighbour indices are used for:

- neighbour gene tokens in cross-attention;
- neighbour image tokens used by the morphology prior;
- relative coordinate offsets;
- the spatial-positive edge list in the contrastive loss.

## Image Preparation And Morphology Prior

The external image representation is not trained in this repository. The
checkpoint supplies a frozen 128-dimensional `img_emb` per spot.

For each section, `prepare_image_features()` applies:

```text
img_emb (128) -> StandardScaler -> PCA(16, random_state=42)
```

For center image vector `m_i` and neighbour image vector `m_j`, the default
cosine morphology prior is:

```text
r_ij = (cosine(m_i, m_j) + 1) / 2
```

It is clamped away from zero before the logarithm is taken. The model adds
`beta * log(r_ij + 1e-6)` to every attention head's score, where `beta` is a
learnable scalar initialized to `1.0`.

The alternative `rbf` and `gene_cosine` prior modes exist in configuration but
the active default is `cosine`.

## Relative Position Bias

For centre coordinate `c_i` and neighbour coordinate `c_j`, the relative offset
is `c_j - c_i`. A small MLP maps this 2-D offset to one scalar:

```text
[dx, dy] -> Linear(2, 16) -> GELU -> Linear(16, 1)
```

This scalar is multiplied by learnable `gamma`, initialized to `0.5`, and added

## Gene Cross-Attention Block

The default gene dimension is `D = 3000`, split into eight heads:

```text
heads = 8
head_dim = 3000 / 8 = 375
```

For center gene vector `x_i` and masked neighbour gene vectors `x_j`, the
implemented operations are:

```text
q_i = W_Q x_i
k_ij = W_K x_j
v_ij = W_V x_j

a_ij = (q_i . k_ij) * temperature^(-1/2)
       + beta * log(r_ij + 1e-6)
       + gamma * position_mlp(c_j - c_i)

alpha_ij = softmax_j(a_ij)
h_i = W_O [sum_j alpha_ij v_ij]
z_i = LayerNorm(x_i + h_i)
```

All `W_Q`, `W_K`, and `W_V` are `Linear(3000, 3000, bias=False)`. `W_O` is
`Linear(3000, 3000)`. Attention dropout is `0.1` during training. The output
`z_i` is the saved 3,000-dimensional Phase 1 spot embedding.

Implementation note: the score scale is `temperature^(-1/2)`. With the default
temperature `10`, this is `1/sqrt(10)`; the current implementation does not
also apply the conventional `1/sqrt(head_dim)` factor.

## Masked Two-View Training

Each optimizer update constructs two independently masked views of the complete
3,000-HVG matrix. `mask_features()` masks whole gene columns with probability
`mask_rate=0.1`; the same selected columns are zeroed for every spot in that
view.

The model produces embeddings `z1` and `z2` for the two masked views. During
evaluation, the unmasked gene matrix is used.

## Contrastive Objective

The default Phase 1 objective is `contrastive_loss()` from
`src/mp_mnca/model.py`. It is symmetric between the two views and has no
pseudo-label negative masking.

For each spot, positives include:

- the same spot across views;
- its six spatial neighbours within the same view;
- its six spatial neighbours across views.

Every other spot in the section is a negative. All embeddings are L2-normalized
inside the loss. The directional loss combines all same-view and cross-view
similarities, then the final objective averages both directions.

The default Phase 1 training loss does **not** consume `ground_truth` labels.

## Training Procedure

Default active convergence settings are:

| Setting | Value |
|---|---:|
| Epochs | 200 |
| Batch size | 256 |
| Optimizer | Adam |
| Learning rate | `3e-4` |
| Weight decay | `1e-5` |
| Temperature | `10.0` |
| Mask rate | `0.1` |
| Seed | `0` |

The implementation constructs two sampled spatial edge sets and their normalized
adjacency matrices every optimizer update. In the current Phase 1 source, these
adjacency matrices are not arguments to `Phase1Model.forward()`: the attention
block uses the fixed coordinate-neighbour index table instead. The unaugmented
spatial edge list is used by the contrastive loss for spatial positives.

The outer batch loop performs `ceil(n_spots / batch_size)` optimizer updates per
the current section before applying the full-section contrastive loss.

## KMeans And Pseudo-Label Options

The configuration contains `image_pseudo_clusters=40`,
`gene_pseudo_clusters`, and `intersection_clusters`. These should not be
described as part of the default Phase 1 training path:

- `image_pseudo_clusters=40` is retained for STAIG-comparison/ablation settings
  but is not consumed by default `train_phase1.py` optimization.
- `gene_pseudo_clusters=None` by default. If enabled in ablation code, it uses
  gene-expression KMeans labels to mask same-cluster negatives.
- `intersection_clusters=None` by default. If enabled in ablation code, it
  retains only spots whose aligned gene- and image-KMeans assignments agree.

Therefore, the default current Phase 1 model begins from coordinate 6-NN
neighbours plus raw gene/image features, not from an image-KMeans clustering.

## Evaluation

For representation evaluation only:

```text
unmasked Phase 1 embeddings (3000-D)
-> PCA(128, random_state=0)
-> tied-covariance Gaussian mixture
-> 15-nearest-spatial-neighbour plurality refinement
-> ARI / NMI against ground-truth layers
```

The cluster count is the observed number of layer labels in the evaluated
section. This is label-count-informed unsupervised clustering: labels are not
training targets, but their observed class count fixes the GMM component count.

The active 200-epoch convergence runner persists model state and metric
snapshots at epoch 1 and every five epochs. The per-section peak-ARI table is a
post-hoc diagnostic and must not be described as label-free checkpoint selection.

## Optional Phase 2

`src/mp_mnca/phase2.py` defines a separate optional gene BYOL/reconstruction
stage. It is not part of the active Phase 1 convergence run.

Phase 2 uses a 3,000-dimensional encoder with repeated
`Linear -> LayerNorm -> activation -> Dropout` blocks, a projector/predictor,
views and optimizes masked-position weighted Huber reconstruction together with
a BYOL-style normalized MSE loss.

## Practical Interpretation

The implemented Phase 1 architecture is best described as:

> Coordinate-neighbour gene cross-attention, routed by frozen image-embedding
> similarity and learned relative position bias, trained with masked two-view
> same-spot and spatial-neighbour contrastive learning.

It is not a GCN, does not train the external image encoder, and does not use
KMeans pseudo-clusters in its default Phase 1 optimization path.
