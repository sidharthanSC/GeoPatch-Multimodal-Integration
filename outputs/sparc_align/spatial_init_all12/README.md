# SPARC-align with spatial k-NN input smoothing

The best-performing SPARC-align configuration measured on the twelve DLPFC sections:
**0.4061 ± 0.0676 mean refined ARI** (median 0.4154), against 0.3740 for the same
configuration without smoothing and 0.2662 for the paper-faithful SPARC baseline.

Many other variants were tried — support-selection mechanisms (quota, rank fusion,
shared/private partitioning), nonlinear encoders, per-stream logit normalization,
dictionary capacity, batch size, training length, masked denoising, and a pooled
cross-section dictionary. All were flat or negative. Only input smoothing produced a
gain, and this document covers that configuration. The full ablation record is in
`../README.md` and `../CONSOLIDATED_12SECTION.csv`.

---

## Methodology

### Pipeline

```
gene 3000-d (obsm['feat'])  ─┐
                             ├─► spatial k-NN smoothing ─► Phase 1 ─► Phase 2 ─► PCA(128) ─► tied-GMM ─► 15-NN refinement ─► ARI/NMI
image 128-d (obsm['img_emb'])┘        (this document)      SPARC SAE   cross-attn
```

Every section is processed independently (per-section transductive). Training is fully
unsupervised: the Phase 2 contrastive objective takes pseudo-labels from `KMeans(40)`
on image PCA, never from `ground_truth`. Layer annotations enter only at evaluation,
and only their *count* fixes the GMM component number — this is label-count-informed
unsupervised clustering, not label-free model selection.

### The smoothing step

Before Phase 1, each spot's feature vector is replaced by a weighted blend of itself
and its six spatial nearest neighbours:

```
x_i  <-  (1 - a) * x_i  +  a * sum_j w_ij * x_j          a = 0.5
w_ij  =  softmax_j( beta * log s_ij )                    beta = 1
s_ij  =  ( cos(m_i, m_j) + 1 ) / 2
```

`s_ij` is the image-PCA cosine similarity between spot *i* and neighbour *j*, so
morphologically dissimilar neighbours contribute less and layer boundaries blur less
than under a uniform mean. Applied to **both** streams. On 151507 this reduces gene
std from 1.00 to 0.55 and image std from 1.00 to 0.68.

`a = 0.5` gives self and neighbourhood equal weight, by analogy to STAIG's normalized
adjacency with self-loops. It was **not** tuned. The 0.75 validated elsewhere in this
project applies to logit *selection*, a weaker intervention than rewriting the data,
and was deliberately not carried over.

### Why before Phase 1 rather than after

Gene self-NMSE is ~0.88 against image's ~0.28, so the Global TopK support is
image-dominated and the shared concept space carries little layer information.
Smoothing raises the gene stream's signal-to-noise so it competes in the vote.
Smoothing the *latents* after Phase 1 was also tested and helped less
(0.3856 vs 0.4061), consistent with it duplicating work Phase 2's cross-attention
already does over the same graph.

### Base configuration

Ablation arm `n_spatial_attention_plus_stage2`, the winner of the 14-arm ablation:

| | |
|---|---|
| Dictionary | 1024 latents, 128 active (Global TopK) |
| Spatial TopK | `spatial_topk_weight=0.75`, morphology-weighted |
| Phase 1 | 50 epochs, batch 256, Adam lr 1e-4, seed 42 |
| Phase 2 | 20 epochs cross-attention + neighbour contrastive |
| Evaluation | PCA(128) → tied-GMM (K = observed layer count) → 15-NN refinement |

### One caveat on the metrics

Smoothing changes the reconstruction target — an autoencoder reconstructs whatever it
is fed — so `self_nmse_*` is **not comparable** between smoothed and unsmoothed arms.
Only ARI and NMI are.

---

## Results

Per-section values: `arm_n_input_smoothing_per_section.csv`.

| Section | Donor | refined ARI | refined NMI |
|---|---|---|---|
| 151507 | 1 | 0.4214 | 0.5216 |
| 151508 | 1 | 0.3491 | 0.4635 |
| 151509 | 1 | **0.5159** | 0.5710 |
| 151510 | 1 | 0.4272 | 0.5267 |
| 151669 | 2 | 0.4287 | 0.5202 |
| 151670 | 2 | 0.3990 | 0.4546 |
| 151671 | 2 | 0.5131 | **0.5942** |
| 151672 | 2 | 0.4093 | 0.5112 |
| 151673 | 3 | 0.4235 | 0.5549 |
| 151674 | 3 | 0.3441 | 0.4616 |
| 151675 | 3 | 0.3626 | 0.5010 |
| 151676 | 3 | 0.2789 | 0.4289 |
| **mean ± std** | | **0.4061 ± 0.0676** | **0.5091 ± 0.0501** |
| **median** | | 0.4154 | 0.5157 |

Standard deviation uses `ddof=1`: the twelve sections are a sample, and sections
(not spots) are the biological replicates.

### Effect of smoothing

| | refined ARI | refined NMI |
|---|---|---|
| arm `n`, no smoothing | 0.3740 ± 0.0701 | 0.4823 ± 0.0494 |
| **arm `n` + input smoothing** | **0.4061 ± 0.0676** | **0.5091 ± 0.0501** |
| Δ | **+0.0320** | +0.0268 |

Wins 8 of 12 sections. Losses: 151508 (−0.105), 151510 (−0.066), 151674 (−0.029),
151676 (−0.000).

**Not statistically established.** Paired across the twelve sections: Wilcoxon
p = 0.233, paired-t p = 0.218. The effect should be reported as promising, not proven.
Gains concentrate where the sparse code was most volatile (151671 +0.207, 151673
+0.132, 151669 +0.100) — 151671 had the most unstable epoch trajectory in the
convergence study, so denoising helps most where the code was noisiest. That is a
coherent mechanism rather than scatter, but it does not substitute for significance.

Donor structure is visible: three of the four weakest sections (151674, 151675,
151676) are donor 3, so the ±0.0676 is not uniform noise.

### Standing against the baselines

| Method | mean refined ARI | median |
|---|---|---|
| MP-MNCA (collaborator's run, 20 epochs) | 0.5195 | 0.5185 |
| STAIG (this repo, 400 epochs) | 0.5092 | 0.5374 |
| **SPARC-align + smoothing** | **0.4061** | **0.4154** |
| SPARC-align, paper-faithful | 0.2662 | 0.2417 |

Smoothing takes SPARC-align from 0.2662 to 0.4061 (+53% over the published
configuration) but leaves it 0.103 below STAIG. The remaining gap appears structural
rather than a tuning shortfall: image→gene cross-reconstruction NMSE never falls below
~0.89 in any configuration tested, so morphology cannot predict expression in this
tissue and the shared concept space stays largely an image code, while cortical layer
identity lives in the gene channel.

A concept-agreement diagnostic makes the same point directly. Neighbouring spots share
4–5× more active concepts than random pairs (Jaccard 0.29 vs 0.06), so the space is
strongly spatial — but same-layer and cross-layer neighbours differ by only ~0.02
(AUC ≈ 0.56). The concept space captures spatial structure without capturing layer
boundaries, which is also what the annotation figures show: smoothing produces
laminar bands instead of blobs, yet Layer_4 stays too thick and WM/Layer_6 stay
entangled.

---

## Files

| File | Contents |
|---|---|
| `arm_n_input_smoothing_per_section.csv` | The table above, machine-readable |
| `spatial_init_metrics.csv` | All arms, 36 rows, with NMSE diagnostics |
| `summary.json` | Per-arm aggregates |
| `figures/spatial_init_ari.png` | Per-arm comparison |

Note `spatial_init_metrics.csv` stores the **refined** metrics in columns named `ari`
and `nmi`; the unrefined values are computed in `evaluate.py` but not persisted.
Refinement contributes very little here (+0.0006 ARI on 151507) because the clusters
are already spatially contiguous — it is retained only so the numbers stay comparable
to STAIG and MP-MNCA, which both use it.

## Reproduce

```bash
python -m src.sparc_align.spatial_init \
    --name spatial_init_all12 \
    --arms base input_smoothing latent_smoothing \
    --alpha 0.5 --epochs 50
```

Builds a prepared-section cache on first use (~219 MB, git-ignored) so no study
process pays the ~4 GB `dlpfc.pkl` load. Runs resume automatically: completed
`(arm, section)` pairs are skipped, so an interruption costs only the section in
flight. Roughly 3 h for all three arms on an M-series Mac.
