# SPARC-align: convergence, batch size, and standing vs the baselines

Twelve DLPFC sections. Stage 1 (the SPARC sparse autoencoder) is evaluated at epochs 10, 20, 50, 100, 200; **stage 2 (cross-attention + neighbour contrastive) is fixed at 20 epochs** at every checkpoint.

All checkpoints come from a **single 200-epoch run per section**, not five separate runs — separate runs would each reinitialize and would not be points on one trajectory, mixing convergence with seed noise.

Evaluation is the shared protocol used by MP-MNCA and STAIG: PCA → tied GMM (sklearn's approximation of R `mclust` EEE) → 15-NN spatial refinement → ARI/NMI. Fully unsupervised: contrastive pseudo-labels come from KMeans on image PCA, never `ground_truth`.

## Headline

| | |
|---|---|
| Best mean refined ARI | **0.3751** at **200 epochs** |
| Median at that epoch | 0.3639 |
| Sections peaking at 200 ep | 3/12 |
| Mean at 200 epochs | 0.3751 |
| Spread across the whole epoch range | 0.0306 |
| Distinct peak epochs across sections | 5 of 5 tested |

**The epoch budget barely matters.** The mean moves only 0.0306 ARI across a 20x range in training (0.3446 at 10 to 0.3751 at 200), against a between-section spread of roughly 0.27. There is a knee around 50 epochs and essentially nothing after it.

Peak epochs are scattered across sections (10 ep: 2, 20 ep: 2, 50 ep: 2, 100 ep: 3, 200 ep: 3), so no section-independent optimum exists — "train for N epochs" is not a setting that transfers here. The nominal best is 200 epochs, but it beats 50 epochs by only +0.0011, which is well inside noise.

## Refined ARI per section, by stage-1 epochs

| Section | 10 ep | 20 ep | 50 ep | 100 ep | 200 ep | best | @ epoch |
|---|---|---|---|---|---|---|---|
| 151507 | **0.4086** | 0.3872 | 0.3916 | 0.3911 | 0.3751 | **0.4086** | 10 |
| 151508 | 0.4333 | **0.4943** | 0.4541 | 0.2927 | 0.3618 | **0.4943** | 20 |
| 151509 | 0.4522 | 0.4357 | 0.4707 | 0.5221 | **0.5603** | **0.5603** | 200 |
| 151510 | 0.4284 | 0.4617 | **0.4936** | 0.4666 | 0.4478 | **0.4936** | 50 |
| 151669 | 0.3248 | 0.3565 | 0.3291 | 0.3505 | **0.3659** | **0.3659** | 200 |
| 151670 | 0.3542 | 0.3799 | 0.3922 | **0.4248** | 0.3947 | **0.4248** | 100 |
| 151671 | 0.1433 | 0.1305 | 0.3063 | 0.3529 | **0.3903** | **0.3903** | 200 |
| 151672 | 0.3186 | 0.3129 | **0.3634** | 0.3456 | 0.3308 | **0.3634** | 50 |
| 151673 | 0.2680 | 0.2866 | 0.2920 | **0.2996** | 0.2925 | **0.2996** | 100 |
| 151674 | 0.3394 | 0.3374 | 0.3728 | **0.3912** | 0.3408 | **0.3912** | 100 |
| 151675 | **0.3870** | 0.3539 | 0.3434 | 0.3789 | 0.3529 | **0.3870** | 10 |
| 151676 | 0.2769 | **0.2896** | 0.2792 | 0.2610 | 0.2887 | **0.2896** | 20 |
| **mean** | 0.3446 | 0.3522 | 0.3740 | 0.3731 | 0.3751 | | |
| **median** | 0.3468 | 0.3552 | 0.3681 | 0.3659 | 0.3639 | | |

## Refined NMI per section, by stage-1 epochs

| Section | 10 ep | 20 ep | 50 ep | 100 ep | 200 ep | best | @ epoch |
|---|---|---|---|---|---|---|---|
| 151507 | 0.4945 | 0.4938 | 0.5123 | **0.5138** | 0.4908 | **0.5138** | 100 |
| 151508 | 0.5055 | 0.5387 | **0.5543** | 0.4205 | 0.4937 | **0.5543** | 50 |
| 151509 | 0.5297 | 0.4731 | 0.5576 | 0.5498 | **0.5764** | **0.5764** | 200 |
| 151510 | 0.4930 | 0.5149 | **0.5475** | 0.5032 | 0.5170 | **0.5475** | 50 |
| 151669 | 0.4090 | 0.4618 | 0.4506 | **0.5068** | 0.4951 | **0.5068** | 100 |
| 151670 | 0.3931 | 0.4178 | 0.4152 | 0.4436 | **0.4596** | **0.4596** | 200 |
| 151671 | 0.2611 | 0.2460 | 0.4635 | 0.4693 | **0.5337** | **0.5337** | 200 |
| 151672 | 0.4432 | 0.4451 | 0.4869 | **0.5138** | 0.4837 | **0.5138** | 100 |
| 151673 | 0.4336 | **0.4419** | 0.4411 | 0.4395 | 0.4418 | **0.4419** | 20 |
| 151674 | 0.4491 | 0.4271 | 0.4648 | **0.5108** | 0.4358 | **0.5108** | 100 |
| 151675 | **0.5316** | 0.4753 | 0.4633 | 0.5190 | 0.4839 | **0.5316** | 10 |
| 151676 | 0.4331 | 0.4527 | 0.4312 | 0.4013 | **0.4548** | **0.4548** | 200 |
| **mean** | 0.4480 | 0.4490 | 0.4823 | 0.4826 | 0.4889 | | |
| **median** | 0.4461 | 0.4572 | 0.4641 | 0.5050 | 0.4873 | | |

## Which phase to use

SPARC-align has two trainable stages, so the operational question is whether the cross-attention stage earns its cost. Over all 60 (section, epoch) checkpoints:

| | two-stage | stage 1 only (`support`) |
|---|---|---|
| Mean refined ARI | **0.3638** | 0.2999 |
| Within-section volatility across epochs | **0.0312** | 0.0626 |
| Head-to-head wins | **46/60** | 14/60 |
| Oracle best-epoch mean | **0.4057** | 0.3910 |

**Use both stages.** The two-stage pipeline is ahead on mean, is about half as volatile across epochs, and stays ahead even under oracle per-section epoch selection. Stage 1 alone does beat it at individual checkpoints (14 of 60), but those wins are not predictable in advance — the peak epoch differs per section — so they are not exploitable.

Stage 2's main contribution here is **stabilization**: it roughly halves the epoch-to-epoch swing. Stage 1's sparse code is volatile (151671 alone swings 0.17 -> 0.15 -> 0.42 -> 0.16 -> 0.46 across checkpoints), and the contrastive stage smooths that into a usable representation.

## Standing vs the baselines

Baselines were run by the collaborator and are **not** re-run here; figures are their published 12-section values at each method's own best/native budget.

| Method | Epochs | Mean refined ARI | Median | Source |
|---|---|---|---|---|
| MP-MNCA Phase 1 | 20 | 0.5195 | 0.5185 | SM/geopatch, `20260819_mp_mnca_phase1_orig_20ep_all12` |
| STAIG (this repo, 400 ep) | 400 | 0.5092 | 0.5374 | `staig_baseline_seed0_all12_regen_v1` |
| SpaGCN | 200 | 0.4969 | 0.4856 | matched-epoch table, outputs/README.md |
| GraphST | 200 | 0.4765 | 0.4895 | `graphst_seed41_all12_200ep` (native 600) |
| STAIG | 200 | 0.4730 | 0.4850 | matched-epoch table (native 400) |
| MuCoST | 200 | 0.4555 | 0.4461 | `mucost_seed2023_all12_200ep` (native 1000) |
| **SPARC-align (ours)** | 200 | 0.3751 | 0.3639 | `convergence_all12` |

SPARC-align does not beat the baselines: 0.3751 against MP-MNCA's 0.5195 and STAIG's 0.4730–0.5092. The gap is structural rather than a tuning shortfall — cross-reconstruction NMSE from image to gene never falls below ~0.92 in any configuration, so morphology cannot predict expression in this tissue and the shared concept space ends up largely an image code, while cortical layer identity lives in the gene channel.

## Batch-size effect

At the best epoch budget (50 stage-1 epochs), all 12 sections.

| Batch size | Mean refined ARI | Median | vs 256 |
|---|---|---|---|
| 32 | 0.3944 **(best)** | 0.3761 | +0.0204 |
| 64 | 0.3712 | 0.3533 | -0.0029 |
| 128 | 0.3540 | 0.3520 | -0.0200 |
| 256 | 0.3740 | 0.3681 | +0.0000 |
| 512 | 0.3597 | 0.3484 | -0.0143 |

**No statistically detectable batch-size effect.** Batch 32 is nominally best (+0.0204 over the paper's 256, winning 8/12 sections), but paired across the 12 sections that gives Wilcoxon p=0.30 and paired-t p=0.23 — nowhere near significance at n=12. The curve is also non-monotonic (32 best, 128 worst, 256 second-best), which is the signature of noise rather than a trend.

Total spread across the whole 16x batch range is 0.0404 ARI. For scale, the mean within-section spread across batch sizes is 0.0281 against a between-section spread of 0.0586 — **which section you run on matters about twice as much as which batch size you pick.**

Per-section best batch size: 32: 7, 64: 1, 256: 2, 512: 2. The lean toward 32 is worth noting as a weak signal — if anything it suggests smaller batches, i.e. more gradient steps per epoch, help slightly at this dataset scale — but it should not be reported as a result on this evidence.

## Spatial k-NN smoothing: before vs after alignment

SPARC uses the spatial graph only to choose *which* latents activate. This tests pushing it further: smoothing the gene and image streams over the k=6 graph **before** encoding (what STAIG/GraphST do implicitly via graph convolution), versus smoothing the latent code **after** SPARC but before stage 2. Morphology kernel `softmax(beta*log s_ij)`, alpha=0.5, 12 sections at 50 epochs.

| Arm | Mean refined ARI | Median | vs baseline | Wins | Wilcoxon p |
|---|---|---|---|---|---|
| baseline | 0.3740 | 0.3681 | — | — | — |
| before | 0.4061 | 0.4154 | +0.0320 | 8/12 | 0.23 |
| after | 0.3856 | 0.3833 | +0.0116 | 8/12 | 0.62 |

**Before beats after** — roughly triple the mean gain (+0.0320 vs +0.0116) and the better median. That fits the mechanism: `after` is largely redundant because stage 2's cross-attention already aggregates over the same graph, whereas `before` changes what SPARC actually encodes.

**But it is not a significant improvement.** Wilcoxon p=0.23 at n=12, and the per-section spread is severe: 151671 gains +0.2068 while 151508 loses -0.1051, with 4 of 12 sections getting worse. The mean gain rests on three large winners.

It also does not close the gap: 0.4061 against MP-MNCA's 0.5195 and STAIG's 0.4730-0.5092. Recommended as the better of the two placements and worth keeping, but not reportable as an improvement on this evidence.

## Figures

- `convergence_all12/figures/convergence_per_section_ari.png` — one panel per section
- `convergence_all12/figures/convergence_mean_ari.png` — mean with IQR band
- `convergence_all12/figures/annotations/DLPFC_151507_epochs.png` — spatial domains vs ground truth
- `batch_size_all12/figures/batch_size_ari.png` — batch-size effect

## Reproduce

Both studies build a prepared-section cache on first use (~219 MB, git-ignored) so no study process pays the ~4 GB `dlpfc.pkl` load.

```bash
# Convergence: one 200-epoch run per section, scored at each checkpoint (~55 min)
python -m src.sparc_align.convergence --study convergence \
    --name convergence_all12 --save-predictions 151507

# Batch sweep, one size per process. Splitting this way keeps peak memory low
# enough to survive a 16 GB machine; a single process running all five was
# OOM-killed. Results resume automatically, so an eviction costs only the
# in-flight size rather than the whole sweep.
for bs in 32 64 128 256 512; do
    python -m src.sparc_align.convergence --study batch \
        --name batch_size_all12 --epochs 50 --batch-sizes $bs
done

# Figures and report
python -m src.sparc_align.plots --study convergence --metric ari
python -m src.sparc_align.plots --study convergence --metric nmi
python -m src.sparc_align.plots --study batch --metric ari
python -m src.sparc_align.annotation_plots --section-id 151507
python -m src.sparc_align.report
```
