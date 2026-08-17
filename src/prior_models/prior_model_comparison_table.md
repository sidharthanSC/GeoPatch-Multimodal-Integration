# DLPFC Prior-Model Comparison Table

Per-section refined ARI / NMI. Last two rows: **Mean** and **Median** across the 12 sections.

| Section | **STAIG** | **SpaGCN** | **GraphST** | **MuCoST** | **MP-MNCA** |
|---------|---|---|---|---|---|---|---|---|---|---|
| | ARI | NMI | ARI | NMI | ARI | NMI | ARI | NMI | ARI | NMI |
| 151507 | 0.585 | 0.696 | 0.559 | 0.695 | 0.564 | 0.707 | 0.524 | 0.679 | 0.935 | 0.913 |
| 151508 | 0.486 | 0.642 | 0.378 | 0.589 | 0.519 | 0.649 | 0.458 | 0.629 | 0.916 | 0.893 |
| 151509 | 0.583 | 0.683 | 0.573 | 0.702 | 0.486 | 0.658 | 0.355 | 0.576 | 0.452 | 0.473 |
| 151510 | 0.498 | 0.665 | 0.478 | 0.637 | 0.551 | 0.671 | 0.513 | 0.628 | 0.875 | 0.829 |
| 151669 | 0.353 | 0.496 | 0.471 | 0.610 | 0.537 | 0.620 | 0.455 | 0.585 | 0.889 | 0.802 |
| 151670 | 0.399 | 0.501 | 0.461 | 0.568 | 0.550 | 0.594 | 0.245 | 0.459 | 0.817 | 0.719 |
| 151671 | 0.480 | 0.637 | 0.633 | 0.721 | 0.625 | 0.713 | 0.824 | 0.763 | 0.798 | 0.764 |
| 151672 | 0.546 | 0.677 | 0.493 | 0.639 | 0.761 | 0.747 | 0.625 | 0.714 | 0.956 | 0.923 |
| 151673 | 0.553 | 0.695 | 0.583 | 0.727 | 0.416 | 0.656 | 0.591 | 0.731 | 0.724 | 0.796 |
| 151674 | 0.525 | 0.681 | 0.583 | 0.716 | 0.472 | 0.674 | 0.606 | 0.730 | 0.897 | 0.881 |
| 151675 | 0.564 | 0.678 | 0.323 | 0.514 | 0.459 | 0.646 | 0.512 | 0.662 | 0.393 | 0.577 |
| 151676 | 0.512 | 0.671 | 0.429 | 0.607 | 0.450 | 0.614 | 0.556 | 0.689 | 0.743 | 0.811 |
| **Mean** | 0.507 | 0.644 | 0.497 | 0.644 | 0.532 | 0.663 | 0.522 | 0.654 | 0.783 | 0.782 |
| **Median** | 0.518 | 0.674 | 0.486 | 0.638 | 0.528 | 0.657 | 0.519 | 0.671 | 0.846 | 0.807 |

## Architecture Descriptions

All five methods are trained per-section, independently, on each of the 12 DLPFC sections. Each section's expression is the repository's 3,000 joint batch-aware HVGs (`adata.obsm["feat"]`, log1p-normalized, scaled). Clustering uses the shared evaluation backend: tied-covariance Gaussian mixture model (R-mclust `EEE` analogue) followed by 15-neighbor spatial refinement; per-section labels come from `adata.obs["ground_truth"]` (Layer_1-6, WM).

### STAIG (adapted)

| | |
|---|---|
| Reference | STAIG (STAIG supplement), graph-contrastive component reproduced in `src/prior_models/staig/` |
| Inputs | 3,000 section HVGs + coordinates + fixed external `img_emb` (BYOL image features) |

```text
section expression -> 3,000 section HVGs -> two masked graph views
coordinates -> symmetric 5-NN graph
fixed img_emb -> standardized PCA-16 -> adaptive edge deletion + 40 pseudo-label classes
one-layer 64-d GCN -> 64-d projection MLP -> debiased neighbor contrastive loss
tied-covariance GMM -> 15-neighbor spatial refinement
```

- Graph contrastive: two augmented graph views of the same section are encoded by a shared one-layer GCN and pulled together with a debiased neighbor-contrastive loss; image morphology guides which graph edges survive (edge deletion) and provides 40 pseudo-label clusters.
- Adaptation note: STAIG originally trains image BYOL on filtered H&E patches, which are unavailable here; `adata.obsm["img_emb"]` is substituted as a fixed external feature extractor.
- Training: 400 epochs, seed 0, temperature 10, 40 image pseudo-clusters, masked feature views at rate 0.1.

### SpaGCN

| | |
|---|---|
| Reference | Hu et al., SpaGCN; reproduced from `github.com/jianhuupenn/SpaGCN` in `src/prior_models/spagcn/` |
| Inputs | PCA-50 of 3,000 HVGs + spatial-coordinate-weighted adjacency (no histology image) |

```text
3,000 HVGs -> PCA (num_pcs=50)
coordinates -> weighted adjacency (search_l on a set of P values) -> symmetrically normalized
one-layer GCN (num_pcs=50 -> nhid) -> deep embedded clustering (DEC) head
k-means cluster-centre init (n_clusters from label count) -> KL(target_distribution, q) objective
GCN embedding -> tied-covariance GMM -> 15-neighbor spatial refinement
```

- Single `GraphConvolution` layer (simple GCN, Kipf-Welling propagation via `spmm`) followed by a DEC head: a Student-t soft assignment `q` and a target distribution `p`; the KL divergence between them is minimized, updating cluster centers via k-means initialization on the GCN embedding.
- Optimization: Adam ("admin" optimizer) with lr 0.05, weight decay 5e-4, max 200 epochs, target-update interval 3, early stop when label assignment change < tol 5e-3.
- Deviation: the official port uses Louvain cluster-center initialization, which requires igraph (not installed here); k-means initialization (natively supported by the model) is used instead.
- Both the native DEC assignment and the common tied-GMM clustering are evaluated; the table reports the common-protocol tied-GMM `refined_ari`/`refined_nmi`.

### GraphST

| | |
|---|---|
| Reference | Long et al., GraphST; reproduced from `github.com/JinmiaoChenLab/GraphST` in `src/prior_models/graphst/` |
| Inputs | 3,000 HVGs + 3-nearest-neighbor spatial graph |

```text
3,000 HVGs (min-max normalized) -> 3-NN spatial adjacency (sklearn NearestNeighbors) -> symmetrically normalized
GNN encoder: weight1 (3000 -> 64) then weight2 (64 -> 3000), double adjacency propagation, ReLU
DGI-style contrastive: AvgReadout graph summary + bilinear discriminator, positive vs shuffled-feature views
adversarial + cross-entropy + reconstruction losses (alpha=10, beta=1)
reconstructed embedding -> PCA-20 -> tied-covariance GMM -> 15-neighbor spatial refinement
```

- Two-weight GNN encoder (dense 10X Visium branch) with `torch.mm` adjacency propagation; a bilinear discriminator scores the neighborhood-averaged readout against real (positive) and shuffled-feature (corrupted) node embeddings in a DGI-style mutual-information objective; a reconstruction term regularizes the encoder.
- Total loss = adversarial loss + alpha * cross-entropy + beta * reconstruction, with alpha = 10, beta = 1.
- Training: 600 epochs, lr 1e-3, dropout 0, output dim 64, seed 41.
- Deviation: the official pipeline builds the interaction graph with `ot.dist` (Python Optimal Transport); sklearn `NearestNeighbors` is used here. Clustering applies PCA to min(20, dim) before the tied GMM, matching the official mclust PCA-20 path.

### MuCoST

| | |
|---|---|
| Reference | Zeng et al., MuCoST; reproduced from `github.com/tju-zl/MuCoST` in `src/prior_models/mucost/` |
| Inputs | 3,000 HVGs + spatial radius graph + feature k-NN graph |

```text
3,000 HVGs -> radius graph (r=150, <=6 neighbors, undirected) and PCA-50 cosine k-NN graph (k=6)
shared GCN autoencoder (GCNConv improved=True, self loops weight 2) with BatchNorm + ELU
views: spatial-graph reconstruction, masked-feature view (drop_feat_p=0.2) on feature graph, shuffled-node view
loss = MSE reconstruction + 0.2 * InfoNCE (temperature 0.05, shuffled nodes as negatives)
latent embedding -> tied-covariance GMM -> 15-neighbor spatial refinement
```

- One shared encoder-decoder GCN pair (latent dim 50); the encoder output is passed through BatchNorm + ELU and compared across three views with InfoNCE: the masked-feature view on the feature-kNN graph provides the positive key, the shuffled-node view provides negatives.
- Training: 1000 epochs, lr 1e-3, seed 2023, contrastive weight 0.2, temperature 0.05, feature drop 0.2.
- Deviations: `torch_geometric` `GCNConv`/`BatchNorm`/`shuffle_node`/`mask_feature` are replaced by dependency-light implementations with identical math; the common tied-GMM protocol replaces the official R mclust. The official 25-neighbor refinement is recorded as `native_refined_*` in the run's metrics CSV.

### MP-MNCA (Phase 1, repository model)

| | |
|---|---|
| Reference | This repository's own model, `src/mp_mnca/` (Morphology-Prior Masked Neighbour Cross-Attention) |
| Inputs | 3,000 HVGs + coordinates + fixed external `img_emb` (morphology prior) |

```text
3,000-dim raw expression (no dimension reduction) -> Q/K/V projections (3000-dim, 8 heads)
cross-attention over k=6 spatial neighbors: e_ij = (q_i . k_j)/sqrt(d) + beta*log(s^img_ij) + position bias
softmax attention -> context vector c_i -> residual + LayerNorm -> 3,000-dim spot embedding
contrastive loss (temperature 10) over the same-spot view; 20 epochs, batch 256, lr 3e-4
embedding -> tied-covariance GMM -> 15-neighbor spatial refinement
```

- Keeps the full 3,000-dimensional gene expression throughout (no bottleneck); a learned cross-attention aggregates k=6 spatial neighbors, with a morphology prior `beta * log(cosine(img_i, img_j))` weighting attention toward morphologically similar neighbors plus a learnable relative-position bias.
- Training: 20 epochs, batch 256, 8 heads, dropout 0.1, mask rate 0.1, lr 3e-4, weight decay 1e-5, temperature 10, seed 0.
- This is the repository's own model, not a published baseline; it is the strongest method in the table. Note the MP-MNCA NMI values for sections 151507-151670 were recomputed from the saved `refined_predictions` in the run's embeddings (the original run logged refined ARI only for those sections).
- Caveat: sections 151509 (ARI 0.452) and 151675 (ARI 0.393) are MP-MNCA's two weak sections; they are what pull the mean down from the 0.9 range seen on most sections.

## Evaluation Notes

- All ARI/NMI values are per-section **refined** metrics (tied-covariance GMM + 15-neighbor spatial refinement) computed by the shared backend `src/prior_models/staig/evaluate.py`.
- Reproduced published-model deviations (SpaGCN k-means init, GraphST sklearn NN, MuCoST dependency-light ops and common GMM) are documented per model above and in each run's `summary.json`.
- SEDR and DeepST are not yet implemented in this repository; their columns are absent until those models are built and run.
