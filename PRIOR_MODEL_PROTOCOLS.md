# STAIG And SpaGCN DLPFC Protocols

## Result

The repository-local STAIG adaptation did not reproduce mean ARI `0.69`.

| Method/result source | Mean ARI | Median ARI | Mean NMI | Median NMI |
|---|---:|---:|---:|---:|
| STAIG adaptation, canonical seed 0 | 0.5069 | 0.5182 | 0.6435 | 0.6743 |
| STAIG, values printed in Supplementary Figs. S1-S2 | 0.6917 | 0.6800 | 0.7117 | 0.7150 |
| SpaGCN rerun, values printed by STAIG in Figs. S1-S2 | 0.4383 | 0.4350 | 0.5925 | 0.6050 |

The often-quoted STAIG `0.69` ARI is the arithmetic mean of the 12 two-decimal values printed in the figures, not their median. The displayed median is `0.68`.

Canonical artifacts:

- `outputs/prior_models/staig_paper_default_img_emb_seed0_all12_v3/`
- `outputs/evaluation/20260815_staig_adaptation_vs_published_v1/`

## Per-Section Comparison

All adaptation values below are after its 15-neighbor refinement. Published values are transcribed at the two-decimal precision printed in STAIG Supplementary Figs. S1-S2.

| Section | Adapted STAIG ARI | Published STAIG ARI | Published SpaGCN ARI |
|---|---:|---:|---:|
| 151507 | 0.5847 | 0.64 | 0.51 |
| 151508 | 0.4858 | 0.63 | 0.38 |
| 151509 | 0.5830 | 0.71 | 0.47 |
| 151510 | 0.4978 | 0.69 | 0.48 |
| 151669 | 0.3529 | 0.68 | 0.33 |
| 151670 | 0.3994 | 0.72 | 0.40 |
| 151671 | 0.4805 | 0.83 | 0.55 |
| 151672 | 0.5459 | 0.84 | 0.61 |
| 151673 | 0.5528 | 0.68 | 0.51 |
| 151674 | 0.5245 | 0.60 | 0.40 |
| 151675 | 0.5641 | 0.65 | 0.30 |
| 151676 | 0.5119 | 0.63 | 0.32 |

## Canonical Adaptation

Each DLPFC section is trained and clustered independently. The checkpoint already contains the same post-filter spot counts shown by the official `151673` notebook, exactly 3,000 section-level `seurat_v3` HVG flags, and expression values scaled to `[0,10]`.

Configuration:

- One model seed: `0`.
- Symmetric Euclidean coordinate 5-NN graph.
- Fixed external `img_emb`, standardized per section and reduced to PCA-16 with seed 42.
- Edge deletion probability: row-wise softmax of log Euclidean image-feature distance over coordinate edges.
- Image pseudo-label debiasing: KMeans with 40 clusters and seed 0.
- Two independently edge-dropped views.
- Feature-column masking: 10% in each view.
- Encoder: one normalized GCN layer, 3,000 to 64 dimensions, followed by PReLU.
- Projection: Linear 64-to-64, ELU, Linear 64-to-64.
- Temperature `10`, 400 epochs, Adam learning rate `5e-4`, weight decay `1e-5`.
- Complete transductive graph; no labels enter representation training.
- Labels determine the requested section-specific cluster count: five or seven according to the exact observed annotations.
- Clustering: sklearn tied-full-covariance Gaussian mixture as the available analogue of R `mclust(modelNames="EEE")`, seed 2023.
- Refinement: replace each assignment by the plurality label among its 15 nearest coordinate neighbors.
- Metrics: ARI and NMI against `adata.obs["ground_truth"]`, reported for every section and summarized over the 12 sections.

For graph views `a,b`, projected outputs `z_i^a,z_i^b`, and spatial neighbors `N(i)`, STAIG's numerator contains the paired node and first-order neighbors in both views:

```text
P_i = exp(sim(z_i^a,z_i^b)/tau)
    + sum_{j in N(i)} exp(sim(z_i^a,z_j^a)/tau)
    + sum_{j in N(i)} exp(sim(z_i^a,z_j^b)/tau).
```

The denominator uses all intra-view and cross-view comparisons except the intra-view self term. The debiased strategy removes comparisons whose image KMeans pseudo-label equals the anchor pseudo-label. The positive sum is divided by `2|N(i)|+1`; losses are averaged in both view directions.

## Exact STAIG Published Protocol

The supplement declares the following DLPFC defaults after tuning on average ARI/NMI across the same 12 annotated sections:

- Coordinate `k=5`.
- One GCN layer.
- Temperature `tau=10`.
- 400 epochs.
- 10% feature masking.
- 40 image pseudo-clusters for debiased negative removal.
- Euclidean distance for image-guided edge deletion.
- Per-section mclust followed by the same DLPFC refinement.

The image pipeline is a major component:

- Crop H&E patches with side length 3.5 times `fiducial_diameter_fullres`.
- Resize patches to 512 by 512.
- Apply Gaussian blur with a 7 by 7 kernel.
- Apply band-pass filtering retaining frequencies 245-275 in the stated 0-512 range.
- Train a ResNet50-based BYOL feature extractor on the filtered patches rather than using an off-the-shelf histology embedding.
- Standardize extracted image features and reduce them to 16 PCA components in the released loader.

The released `ARI0.68-Spatial_clustering-151673-img.ipynb` conflicts with the supplement defaults. It uses seed `39788` for NumPy/PyTorch, Python seed `12345`, deterministic CUDA, `tau=35`, 300 epochs, 10%/20% feature masking, and 80 image pseudo-clusters. It reports ARI/NMI `0.686394/0.730589` after R mclust EEE and 15-neighbor refinement. Running those released settings with repository `img_emb` produced refined ARI/NMI `0.536936/0.698293`.

Therefore no single fully disclosed parameter file reproduces all 12 published rows. The supplement provides one tuned default, while the repository provides one differently tuned section-specific configuration. This is a protocol-level reproducibility limitation, not a justification for selecting labels during this rerun.

## Why The Adaptation Is Not Exact STAIG

- The raw H&E patches used by STAIG are unavailable in the current workspace.
- `img_emb` was produced by an external image-BYOL pipeline with unknown training and preprocessing provenance, not STAIG's filtered-patch BYOL.
- R and `mclust` are unavailable. A tied-covariance sklearn GMM matches the EEE shared-covariance constraint conceptually but not R's initialization and numerical implementation.
- Canonical training used float32 for feasible local execution; the released STAIG implementation casts its model and inputs to float64.
- The supplement does not declare the seed that produced every published section result.
- One seed was used under the repository compute policy, so representation-seed variability is unmeasured.

These differences prevent calling the result a reproduction of STAIG's headline number. It is a tested adaptation of STAIG's graph architecture to the repository's available inputs.

## SpaGCN Native DLPFC Protocol

The official SpaGCN tutorial for section `151673` specifies:

- Input raw UMI counts, spot array/pixel coordinates, and the histology image.
- Remove genes detected in fewer than three spots and remove `ERCC*` and `MT-*` genes.
- Normalize per spot and apply `log1p`.
- Construct a histology-aware distance matrix from pixel coordinates and local color intensity with histology scale `alpha=1` and color neighborhood `beta=49`.
- Set neighborhood-expression contribution `p=0.5`; binary-search Gaussian length scale `l` over `[0.01,1000]` with tolerance `0.01`.
- Use 50 expression PCs by default in the model.
- Supply the known number of annotated domains and search Louvain resolution, starting at `0.7` with step `0.1`, until the requested count is obtained.
- Set Python, PyTorch, and NumPy seeds to `100`.
- Train SpaGCN's GCN plus deep embedded clustering model with spatial initialization, Louvain initialization, learning rate `0.05`, tolerance `5e-3`, and at most 200 epochs.
- Native optional refinement computes a coordinate-only distance matrix and, for Visium's hexagonal layout, examines six neighbors. A spot changes only if its current label has fewer than three votes and another label has more than three votes.
- Compute ARI/NMI against the manual layer annotations after clustering/refinement.

This protocol is label-count-informed and transductive. The resolution search uses the annotation-derived number of domains but not individual label assignments.

## SpaGCN In The STAIG Benchmark

STAIG states that SpaGCN was run with default parameters, that the representation immediately before SpaGCN's clustering layer was extracted, and that all DLPFC methods received a common refinement. STAIG Figs. S1-S2 print mean/median SpaGCN ARI `0.4383/0.4350` and mean/median NMI `0.5925/0.6050` after transcription of all 12 rows.

The paper does not fully disclose whether the extracted SpaGCN representation was passed to mclust EEE or returned to SpaGCN's native DEC clustering head, nor does it state in prose whether “common refinement” means STAIG's 15-neighbor unconditional plurality or SpaGCN's native six-neighbor strict-majority rule. Consequently, the exact third-party SpaGCN rerun protocol behind STAIG's figure cannot be reconstructed from the supplement alone. The native SpaGCN tutorial above is the exact public protocol; the STAIG baseline result must be labeled a third-party rerun with this unresolved clustering/refinement detail.

## Sources

- STAIG paper and `STAIG.pdf` in this repository.
- STAIG supplementary material, especially Notes pages 2-10 and Figs. S1, S2, S22, S23, S27-S31.
- Official STAIG repository: `https://github.com/y-itao/STAIG`, audited 2026-08-15.
- Official STAIG files: `staig/net.py`, `staig/staig.py`, `staig/adata_processing.py`, `staig/utils.py`, `train_img_config.yaml`, and `example/ARI0.68-Spatial_clustering-151673-img.ipynb`.
- Official SpaGCN repository and tutorial: `https://github.com/jianhuupenn/SpaGCN`, audited 2026-08-15.
