# SPARC cross-modal alignment — run index

SPARC (Nasiri-Sarvi et al., TMLR 3/2026) applied to DLPFC gene+histology streams,
evaluated through the identical protocol as `src/mp_mnca` and STAIG: PCA → tied GMM
→ 15-NN spatial refinement → ARI/NMI. Code: `src/sparc_align/`.

**All runs are unsupervised.** Contrastive pseudo-labels come from `KMeans(40)` on
image PCA, never `ground_truth`.

---

## Headline: 12-section results

Machine-readable: `CONSOLIDATED_12SECTION.csv`. Source: `ablation_all12_v1/`.

Full ranking across all three ablation runs (best cluster input per arm):

| Arm | input | **mean ARI** | median ARI | **mean NMI** |
|---|---|---|---|---|
| **STAIG baseline** | — | **0.5092** | **0.5374** | **0.6455** |
| **MP-MNCA, unsupervised** | — | **0.4836** | **0.5132** | **0.6384** |
| `n_spatial_attention_plus_stage2` | attn | **0.3740** | **0.3681** | 0.4823 |
| `e_spatial_plus_attention` | attn | 0.3629 | 0.3411 | 0.4780 |
| `i_partitioned_mlp1024` | support | 0.3163 | 0.3141 | 0.4397 |
| `j_best_plus_attention` | support | 0.3163 | 0.3141 | 0.4397 |
| `k_spatial_attention` | support | 0.3109 | 0.3000 | 0.4547 |
| `d_spatial_topk` | support | 0.2946 | 0.3137 | 0.4498 |
| `f_partitioned` | support | 0.2899 | 0.2809 | 0.4348 |
| `m_spatial_attention_partitioned` | support | 0.2801 | 0.2922 | 0.4348 |
| `l_spatial_attention_mlp1024` | support | 0.2775 | 0.2948 | 0.3931 |
| `g_mlp1024` | support | 0.2731 | 0.2747 | 0.3882 |
| `h_mlp2048` | support | 0.2679 | 0.2753 | 0.3895 |
| `a_paper_faithful` | support | 0.2662 | 0.2417 | 0.3570 |
| `c_mlp_encoder` | support | 0.2440 | 0.2717 | 0.3501 |
| `b_norm_logits` | support | 0.1925 | 0.1968 | 0.3143 |

**SPARC does not beat STAIG.** Best arm `n_spatial_attention_plus_stage2` reaches
0.3740 mean vs STAIG's 0.5092 and unsupervised MP-MNCA's 0.4836.

### v3: attention-style spatial weighting (`ablation_all12_v3_attention/`)

Replacing the uniform 6-neighbour mean with MP-MNCA's morphology kernel
`softmax(β log s_ij)` (learnable β) in the stage-1 support selection.

| Arm | mean ARI | vs uniform equivalent |
|---|---|---|
| `k_spatial_attention` | 0.3109 | **+0.0163** over `d_spatial_topk` (0.2946) |
| `n_spatial_attention_plus_stage2` | **0.3740** | **+0.0111** over `e_spatial_plus_attention` (0.3629) |
| `l_spatial_attention_mlp1024` | 0.2775 | below `k` — the MLP encoder still hurts |
| `m_spatial_attention_partitioned` | 0.2801 | below `k` — partitioning still neutral-to-negative |

**Attention weighting helps**, contradicting the single-section prediction
(151507 showed 0.3541 → 0.2667, i.e. a large *drop*). Section 151507 has now
misled three times — it also inflated the leak estimate and overstated SPARC's
absolute ARI. Treat single-section results here as indicative only.

### v2 capacity arms (`ablation_all12_v2_capacity/`)

| Arm | input | mean ARI | median ARI | mean NMI | self-NMSE gene | cross i→g |
|---|---|---|---|---|---|---|
| `f_partitioned` | support | 0.2899 | 0.2809 | 0.4348 | 0.881 | 0.940 |
| `g_mlp1024` | support | 0.2731 | 0.2747 | 0.3882 | 0.924 | 0.943 |
| `h_mlp2048` | support | 0.2679 | 0.2753 | 0.3895 | 0.929 | 0.947 |
| `i_partitioned_mlp1024` | support | 0.3163 | 0.3141 | 0.4397 | 0.920 | 0.957 |
| `j_best_plus_attention` | support | 0.3163 | 0.3141 | 0.4397 | 0.920 | 0.957 |

No v2 arm beat v1's `e_spatial_plus_attention` (0.3629). Arm `j` did run stage 2
(`attn_ari` mean 0.2902) but it scored below its own `support` input, so stage 2
*hurt* under the partitioned + MLP configuration while it *helped* under the
paper-faithful one — the two extensions interfere.

Wider encoders made gene reconstruction **worse** (0.924 / 0.929 vs the affine
encoder's 0.881): on a fixed 50-epoch budget the extra parameters are undertrained.
This refutes the "gene-encoder capacity is the bottleneck" hypothesis formed after
the TopK-mode sweep.

---

## Directories

| Path | Contents |
|---|---|
| `ablation_all12_v1/` | 5 arms × 12 sections. `ablation_metrics.csv` (60 rows), `summary.json` |
| `ablation_all12_v2_capacity/` | 5 capacity arms × 12 sections |
| `ablation_all12_v3_attention/` | 4 attention-weighting arms × 12 sections |
| `../mp_mnca/leak_ab_all12_v1/` | 12-section MP-MNCA label-leak A/B |
| `gonogo_151507_L1024_k32_v1/` | First feasibility run, single section |
| `exploratory_151507/` | Single-section sweep scripts + leak-test result |
| `logs/` | stdout of the long runs |
| `../prior_models/staig_baseline_seed0_all12_regen_v1/` | Regenerated STAIG baseline |

---

## Single-section exploratory results (151507)

Transcribed from run stdout; scripts saved in `exploratory_151507/` for
re-derivation. **151507 runs ≈ +0.06 above the 12-section median — do not read
these as representative.**

### Capacity sweep (`sweep.py`), refined ARI
| L | k | self-NMSE gene | cross i→g | ARI sum | ARI support |
|---|---|---|---|---|---|
| 1024 | 32 | 0.895 | 0.913 | 0.011 | 0.114 |
| 1024 | 128 | 0.855 | 0.922 | 0.148 | 0.260 |
| 2048 | 256 | 0.763 | 0.903 | 0.014 | 0.190 |
| 4096 | 512 | 0.601 | 0.861 | 0.186 | 0.224 |

### Spatial TopK weight (`spatial_sweep.py`), refined ARI
| w | ARI sum | ARI support |
|---|---|---|
| 0.00 (paper) | 0.148 | 0.260 |
| 0.25 | 0.159 | 0.295 |
| 0.50 | 0.154 | 0.262 |
| **0.75** | **0.207** | **0.354** |
| 0.90 | 0.160 | 0.261 |

### Stage-2 attention epochs (`stage2_epochs.py`), refined ARI
20 → 0.337 · 50 → 0.235 · 100 → 0.316 · 200 → 0.252 (non-monotonic; plateaued)

### Two-stage stacking (`combined.py`), refined ARI
| spatial w | stage1 support | stage2 sum | stage2 support | stage2 NMI |
|---|---|---|---|---|
| 0.00 | 0.2596 | 0.3368 | 0.2517 | 0.4416 |
| 0.75 | 0.3541 | 0.3923 | 0.4008 | 0.5071 |

### TopK selection modes (`topk_modes.py`), all with spatial w=0.75
| mode | ARI sum | ARI support | cross i→g | support Jaccard |
|---|---|---|---|---|
| `global_sum` (paper) | 0.2074 | 0.3541 | 0.930 | 0.650 |
| `quota` | 0.0102 | 0.2431 | 0.935 | 0.619 |
| `rank_fusion` | 0.0187 | 0.2174 | 0.939 | 0.574 |
| `partitioned` | 0.1960 | 0.3624 | 0.944 | 0.257 |

### Morphology-weighted spatial aggregation (`spatial_attention=True`)
| aggregation | ARI sum | ARI support | NMI support |
|---|---|---|---|
| uniform mean | 0.2074 | **0.3541** | 0.4796 |
| morphology-weighted `softmax(β log s_ij)` | 0.1863 | 0.2667 | 0.4690 |

On 151507 the weighting appears to **hurt** by −0.087. **This does not generalize** —
across 12 sections it *helps* (+0.0163 mean; see the v3 table above). A textbook case
of why single-section tuning is unreliable here.

---

## Findings

1. **MP-MNCA's 0.824 is label-leaked.** `train_phase1.py` fed `ground_truth` as the
   contrastive pseudo-labels, which build the negative mask, so the loss never
   separated same-layer spots. Controlled A/B on 151507 (`exploratory_151507/leak_test.json`):

   | pseudo-labels | refined ARI | refined NMI |
   |---|---|---|
   | `ground_truth` (as shipped) | 0.8591 | 0.8608 |
   | `image_kmeans` (unsupervised) | 0.5214 | 0.6647 |
   | STAIG | 0.5524 | 0.6799 |

   The leak is worth **+0.338 ARI**. Leak-free, MP-MNCA does *not* beat STAIG.
   `fit_phase1` now takes `pseudo_label_source`, defaulting to `image_kmeans`.

2. **Alignment is strongly asymmetric and never improved.** `cross_nmse_image_to_gene`
   stayed at 0.92–0.94 under every configuration; gene→image is 0.04–0.21.
   Expression predicts morphology; morphology cannot recover expression. The shared
   concept space is essentially an image code, and cortical layer lives in the gene
   channel. SPARC's premise — that both streams encode shared concepts — only
   half-holds here.

3. **The paper's own justification for Global TopK is inverted on this data.**
   SPARC §5 argues the shared support costs 0.030–0.060 self-NMSE but buys 2–3×
   larger cross-reconstruction gains. Here there is no cross gain to buy.

4. **Variance reduction helps; redistributing influence hurts.** Everything that
   worked (MLP encoder +0.030, uniform neighbour mean +0.072, stage-2 +0.027) either
   adds capacity or reduces variance. Everything that failed (`norm_logits` −0.045,
   `quota` −0.111, `rank_fusion` −0.137, morphology weighting −0.087) redistributes
   influence between streams. The modest signal SPARC produces here comes from having
   a stable low-variance sparse code, not from concept alignment.

5. **Reconstruction and clustering are anti-correlated.** Repeatedly, self-NMSE
   worsened while ARI improved. NMSE cannot be used for model selection on this task.

---

## Reproduce

All entry points write directly into `outputs/sparc_align/`; none require capturing
stdout.

```bash
# 12-section ablation arms (writes ablation_metrics.csv incrementally)
python -m src.sparc_align.ablation --output-dir outputs/sparc_align/<run> \
  --arms a_paper_faithful d_spatial_topk e_spatial_plus_attention --device mps

# Single-section parameter sweeps -> outputs/sparc_align/sweeps_<section>/
python -m src.sparc_align.sweeps --sweep all --section 151507
python -m src.sparc_align.sweeps --sweep spatial_attention --section 151507

# STAIG baseline
python -m src.prior_models.staig.train \
  --output-dir outputs/prior_models/<run> --epochs 400 --seed 0 --device cpu
```

## Persistence

`src/sparc_align/results_io.py` provides `IncrementalCsv`, which flushes after every
row, so an interrupted run leaves its completed rows on disk. `ablation.py` now
writes `ablation_metrics.csv` after each section and refreshes `summary.json` after
each arm, with a `complete` flag distinguishing partial from finished runs. It also
no longer treats an empty directory (left by a previous crash) as a conflict.

`src/sparc_align/sweeps.py` supersedes the ad-hoc scripts kept in
`exploratory_151507/`; those are retained only as provenance for the single-section
tables above, which were transcribed from stdout before persistence existed.
