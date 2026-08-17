# Spatial Adjacency Clustering Results

## Question

Can DLPFC cortical domains be recovered by clustering each section's spatial adjacency
matrix directly, and does image information remain useful when no GCN is trained?

## Protocol

- Data: all 12 DLPFC sections, evaluated independently.
- Candidate topology: symmetric coordinate KNN graph with `k=6`.
- Spatial-only condition: every candidate edge has weight one.
- Gene condition: candidate edges are weighted from cosine distance over all 33,538
  sparse log1p-normalized expression values.
- Image condition: candidate edges are weighted from cosine distance over fixed
  128-dimensional `img_emb`.
- Both feature-weighted conditions use the same label-free median edge-distance kernel.
- Clustering: normalized spectral clustering on each sparse precomputed adjacency,
  with the exact observed annotation count supplied as `k`.
- Evaluation: raw assignments and the same 15-neighbor plurality refinement used by
  the repository STAIG adaptation. Labels enter only after clustering.
- Seed: 0.

No neural network, image encoder training, dense spot-by-spot similarity matrix, or
cortical label is used to construct the adjacency matrices.

## Aggregate Results

| Condition | Mean refined ARI | Median refined ARI | Mean refined NMI | Median refined NMI |
|---|---:|---:|---:|---:|
| Spatial binary adjacency | 0.19174 | 0.19285 | 0.34852 | 0.35782 |
| Spatial + all-33,538-gene edge weights | 0.19083 | 0.19298 | 0.34769 | 0.35686 |
| Spatial + image-128 edge weights | **0.21186** | **0.20916** | **0.36808** | **0.36333** |
| Expression STAIG | 0.50693 | 0.51821 | 0.64353 | 0.67427 |

Direct adjacency clustering does not beat STAIG. Gene weighting is effectively equal
to the binary coordinate graph: its mean refined ARI is lower by `0.00091`, and it wins
on only 3 of 12 sections. Image weighting improves mean refined ARI by `0.02012` over
binary adjacency and wins on 10 of 12 sections, but remains far below Expression STAIG.

## Section Results

Cells are refined ARI.

| Section | Spatial binary | Gene-weighted | Image-weighted |
|---|---:|---:|---:|
| 151507 | .203 | .203 | **.215** |
| 151508 | **.176** | .176 | .166 |
| 151509 | .156 | .154 | **.168** |
| 151510 | .115 | .113 | **.168** |
| 151669 | .183 | **.184** | .176 |
| 151670 | .155 | .153 | **.203** |
| 151671 | .242 | .239 | **.249** |
| 151672 | .255 | .253 | **.262** |
| 151673 | .219 | .219 | **.228** |
| 151674 | .213 | .216 | **.242** |
| 151675 | .180 | .180 | **.186** |
| 151676 | .204 | .202 | **.279** |

## Interpretation

The coordinate graph alone contains some cortical structure because DLPFC layers are
spatially organized. Edge-local similarity over all genes adds almost nothing under
this spectral adjacency formulation. Image similarity provides a small, consistent
improvement, so the image embeddings are not useless, but image-weighted adjacency is
not sufficient to explain STAIG's much stronger result.

This experiment does not prove that images are required by the full STAIG model.
Expression STAIG also uses 3,000-gene node features, learned GCN propagation, stochastic
graph views, neighbor contrast, and image-derived pseudo-label debiasing. A strict image
necessity test would keep that complete model fixed and compare raw-image guidance with
uniform graph augmentation and gene-only guidance.

## Artifacts

- `outputs/evaluation/20260815_adjacency_clustering_seed0_v1/summary.json`
- `outputs/evaluation/20260815_adjacency_clustering_seed0_v1/section_metrics.csv`
- `outputs/evaluation/20260815_adjacency_clustering_seed0_v1/adjacency/`
- `outputs/evaluation/20260815_adjacency_clustering_seed0_v1/assignments/`
- Reusable API: `src/multimodal/adjacency_clustering.py`
