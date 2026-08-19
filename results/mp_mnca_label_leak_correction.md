# CORRECTION: MP-MNCA Phase 1 results were label-leaked

**Date:** 2026-08-18
**Affects:** `results/mp_mnca_vs_staig.md`, `results/mp_mnca_vs_staig_corrected.md`,
`results/mp_mnca_architecture_paper.md`, `results/mp_mnca_phase1_complete_results.md`,
`outputs/README.md` (rows `20260816_mp_mnca_phase1_all12`, `20260816_mp_mnca_report`)

## What was wrong

`src/mp_mnca/train_phase1.py` fed the **ground-truth cortical layer** in as the
contrastive pseudo-labels:

```python
label_to_idx = {label: i for i, label in enumerate(np.unique(section_data.labels))}
pseudo_numeric = np.array([label_to_idx[l] for l in section_data.labels], dtype=np.int64)
pseudo_labels = torch.as_tensor(pseudo_numeric, dtype=torch.long, device=device)
```

`section_data.labels` is `adata.obs["ground_truth"]`. Those pseudo-labels go into
`neighbor_contrastive_loss`, where `src/prior_models/staig/model.py:69` builds the
negative mask from them:

```python
negative_mask = pseudo_labels[:, None] != pseudo_labels[None, :]
```

Every spot sharing the centre's ground-truth layer was therefore **excluded from the
negative set**. The loss was never asked to push apart two Layer_3 spots. That is
supervised contrastive learning on the exact label used for evaluation.

STAIG derives the same variable from `KMeans(n_clusters=40)` on image PCA
(`src/prior_models/staig/data.py:153`) — fully unsupervised. MP-MNCA's own
`prepare_section_phase1` computed that identical KMeans at `train_phase1.py:81-84`
**and then discarded it**, never storing it on `MpMncaSectionData`.

So the published comparison was **supervised MP-MNCA vs unsupervised STAIG**.

## Measured magnitude — all twelve sections

Controlled A/B: identical seed, config, architecture and epoch count (20); the *only*
difference is the pseudo-label source. Run: `outputs/mp_mnca/leak_ab_all12_v1/`
(`leak_ab_all_sections.csv`, `summary.json`); code `src/mp_mnca/leak_ab.py`.

| Section | leaked ARI | unsupervised ARI | ARI leak | STAIG | unsup − STAIG |
|---|---|---|---|---|---|
| 151507 | 0.9291 | 0.5245 | +0.4046 | 0.5524 | −0.0279 |
| 151508 | 0.9185 | 0.4902 | +0.4283 | 0.4815 | +0.0086 |
| 151509 | 0.4497 | 0.4323 | +0.0174 | 0.5698 | −0.1375 |
| 151510 | 0.8753 | 0.4937 | +0.3816 | 0.4930 | +0.0007 |
| 151669 | 0.6453 | 0.3023 | +0.3430 | 0.2797 | +0.0226 |
| 151670 | 0.6936 | 0.2922 | +0.4014 | 0.5213 | −0.2292 |
| 151671 | 0.7975 | 0.6061 | +0.1913 | 0.4693 | +0.1369 |
| 151672 | 0.9588 | 0.5815 | +0.3773 | 0.5309 | +0.0506 |
| 151673 | 0.8321 | 0.5098 | +0.3224 | 0.5571 | −0.0474 |
| 151674 | 0.8064 | 0.5292 | +0.2771 | 0.5557 | −0.0264 |
| **151675** | **0.2948** | **0.5242** | **−0.2294** | 0.5559 | −0.0316 |
| 151676 | 0.7397 | 0.5166 | +0.2231 | 0.5439 | −0.0273 |

**Leak: mean +0.2615 ARI, median +0.3327, range [−0.2294, +0.4283], positive on
11/12 sections.** NMI leak: mean +0.1219, median +0.1681.

Aggregates:

| Arm | mean ARI | median ARI | mean NMI | median NMI |
|---|---|---|---|---|
| MP-MNCA, `ground_truth` (leaked) | 0.7451 | 0.8019 | 0.7520 | 0.8172 |
| MP-MNCA, `image_kmeans` (unsupervised) | **0.4836** | **0.5132** | 0.6384 | 0.6626 |
| STAIG baseline | **0.5092** | **0.5374** | 0.6455 | 0.6751 |

**Corrected conclusion: unsupervised MP-MNCA does not beat STAIG.** It sits slightly
below on both mean (0.4836 vs 0.5092) and median (0.5132 vs 0.5374), winning 5 of 12
sections. The published claim of 0.824 mean / 0.896 median and "10/12 sections win"
is entirely attributable to the leak.

### Two qualifications

1. **The leak is not uniformly positive.** On 151675 the leaked arm scored *worse*
   (0.2948 vs 0.5242, a −0.2294 leak). Excluding same-layer spots from the negative
   set removes roughly 1/7 of all pairs, so the leak does not merely add signal — it
   changes the optimization problem, and on some sections leaves too few negatives for
   the contrastive objective to converge. Report the leak as a distribution, not a
   constant.

2. **Absolute values are sensitive to thread configuration.** This run used
   `OMP_NUM_THREADS=4`; an earlier single-section run at default threading gave 0.8591
   / 0.5214 for 151507 versus 0.9291 / 0.5245 here. `use_deterministic_algorithms(True)`
   guarantees reproducibility for a fixed thread count, not across counts, and BLAS
   reduction-order differences compound over 340 gradient steps. Note the unsupervised
   arm is far more stable (0.5214 → 0.5245) than the leaked arm (0.8591 → 0.9291),
   consistent with the leaked objective having a sharper, more perturbable optimum.
   Both arms share a process and thread config, so the *within-run delta* is sound.

## Corrected standing

| Claim as published | Status |
|---|---|
| "mean refined ARI 0.824, median 0.896" | Supervised. Unsupervised equivalent is **0.4836 / 0.5132**. |
| "DESTROYS STAIG (0.518 median)" | Withdrawn — unsupervised MP-MNCA is *below* STAIG (0.4836 vs 0.5092 mean). |
| "10/12 sections win" | Withdrawn — unsupervised wins **5/12**. |
| "20× faster / 20 epochs" | Unaffected, but see note below. |
| Phase 2, contrastive v1-v3, gene-cosine, BYOL rows | Unaffected — those used their own configurations and all scored 0.23-0.31, below STAIG regardless. |

The architectural claims in `mp_mnca_architecture_paper.md` (no bottleneck, adaptive
aggregation, continuous morphology prior) are **not supported** by the reported
numbers, because the numbers are explained by the leak rather than the architecture.

## Related corrections found in the same review

1. **The results are not reproducible.** `outputs/mp_mnca/`, `outputs/prior_models/`
   and `outputs/evaluation/` were all absent at the time of review, so no artifact
   backed any of the tables. A STAIG baseline was regenerated at
   `outputs/prior_models/staig_baseline_seed0_all12_regen_v1/` (mean refined ARI
   0.5092, median 0.5374; NMI 0.6455 / 0.6751), closely reproducing the recorded
   0.50693 / 0.64353.
2. **`mp_mnca_vs_staig_corrected.md` §4 lists a "frozen pre-trained gene encoder"** as
   a Phase 1 component. `Phase1Model` contains only `GeneExpressionCrossAttention` —
   there is no gene encoder.
3. **Dead code in `fit_phase1`:** `edges_1`, `edges_2`, `adjacency_1`, `adjacency_2`
   are computed every step and never used, so the STAIG-style image-guided edge
   dropping the report describes as "replaced by attention" is absent from the loss
   entirely. (Measured cost: ~0.2% of a step — a correctness/clarity issue, not a
   performance one.)
4. **Attention scale:** `phase1.py:37` sets `scale = temperature ** -0.5` ≈ 0.316
   with no `d_k` term, though `model.py:103` documents it as "1/sqrt(d_k) *
   temperature scaling". With `head_dim = 375`, standard scaling would be ≈ 0.052, so
   logits are ~6× larger than intended.
5. **Train/eval mismatch:** `phase1.py:112` documents a clean centre query with masked
   neighbours, but `train_phase1.py:187-188` passes masked features for both, while
   evaluation passes clean features for both.

## Fix applied

`fit_phase1` now takes `pseudo_label_source`, **defaulting to `"image_kmeans"`**
(unsupervised). `"ground_truth"` remains reachable so the leak stays measurable, and
its docstring states plainly that it leaks the evaluation label.

No prior artifacts were altered or deleted; this file records the correction.
