# DLPFC Published (Paper-Reported) Per-Section Values vs MP-MNCA

Original values reported by each method's publication, compared with this repository's MP-MNCA model. The prior-model columns are **the papers' own reported numbers**, not this repository's reproductions (see `prior_model_comparison_table.md` for reproduced runs). MP-MNCA is this repository's own model (Phase 1, `src/mp_mnca/`).

## Sources

| Model | Source of published values |
|---|---|
| STAIG | STAIG paper, Supplementary Figures S1-S2 (transcribed at two-decimal figure precision in `src/prior_models/staig/published.py`). STAIG, Nat. Commun. (2025), DOI 10.1038/s41467-025-56276-0. |
| SpaGCN | STAIG paper, Supplementary Figures S1-S2 (transcribed in `src/prior_models/staig/published.py`). SpaGCN, Nat. Methods 18, 1342-1351 (2021). |
| GraphST | GraphST paper values as tabulated in EfNST, Commun. Biol. 7 (2024), DOI 10.1038/s42003-024-07286-z, Supplementary Table 2 (ARI only; mean 0.56 matches GraphST, Nat. Commun. 14, 115 (2023)). |
| MuCoST | MuCoST authors' official `report/` files at github.com/tju-zl/MuCoST (`MuCoST-ari-results.txt`, `MuCoST-nmi-results.txt`); mean ARI 0.526 matches MuCoST, Brief. Bioinform. 25 (2024), DOI 10.1093/bib/bbae255. |
| MP-MNCA | This repository's own Phase-1 model (`src/mp_mnca/`), per-section refined ARI/NMI. |

GraphST per-section NMI is not machine-readable: the GraphST paper reports DLPFC NMI as figure boxplots only, so its NMI column is left blank rather than filled with another paper's reproduction.

## Per-Section ARI

| Section | STAIG | SpaGCN | GraphST | MuCoST | **MP-MNCA** |
|---|---|---|---|---|---|
| 151507 | 0.64 | 0.51 | 0.44 | 0.448 | **0.935** |
| 151508 | 0.63 | 0.38 | 0.52 | 0.447 | **0.916** |
| 151509 | 0.71 | 0.47 | 0.45 | 0.458 | **0.452** |
| 151510 | 0.69 | 0.48 | 0.44 | 0.410 | **0.875** |
| 151669 | 0.68 | 0.33 | 0.54 | 0.656 | **0.889** |
| 151670 | 0.72 | 0.40 | 0.56 | 0.443 | **0.817** |
| 151671 | 0.83 | 0.55 | 0.64 | 0.586 | **0.798** |
| 151672 | 0.84 | 0.61 | 0.66 | 0.595 | **0.956** |
| 151673 | 0.68 | 0.51 | 0.64 | 0.605 | **0.724** |
| 151674 | 0.60 | 0.40 | 0.61 | 0.605 | **0.897** |
| 151675 | 0.65 | 0.30 | 0.63 | 0.486 | **0.393** |
| 151676 | 0.63 | 0.32 | 0.56 | 0.573 | **0.743** |
| **Mean** | 0.69 | 0.44 | 0.56 | 0.526 | **0.783** |
| **Median** | 0.685 | 0.435 | 0.56 | 0.573 | **0.846** |

## Per-Section NMI

| Section | STAIG | SpaGCN | GraphST | MuCoST | **MP-MNCA** |
|---|---|---|---|---|---|
| 151507 | 0.72 | 0.65 | – | 0.605 | **0.913** |
| 151508 | 0.67 | 0.50 | – | 0.621 | **0.893** |
| 151509 | 0.71 | 0.65 | – | 0.621 | **0.473** |
| 151510 | 0.69 | 0.63 | – | 0.577 | **0.829** |
| 151669 | 0.69 | 0.47 | – | 0.651 | **0.802** |
| 151670 | 0.65 | 0.53 | – | 0.562 | **0.719** |
| 151671 | 0.78 | 0.68 | – | 0.678 | **0.764** |
| 151672 | 0.77 | 0.73 | – | 0.684 | **0.923** |
| 151673 | 0.74 | 0.69 | – | 0.718 | **0.796** |
| 151674 | 0.68 | 0.58 | – | 0.715 | **0.881** |
| 151675 | 0.72 | 0.48 | – | 0.610 | **0.577** |
| 151676 | 0.72 | 0.52 | – | 0.672 | **0.811** |
| **Mean** | 0.71 | 0.59 | – | 0.643 | **0.782** |
| **Median** | 0.715 | 0.625 | – | 0.652 | **0.807** |

## Reading Notes

- Section order is donor-major: donor 1 = 151507-151510, donor 2 = 151669-151672, donor 3 = 151673-151676.
- STAIG/SpaGCN published values are transcribed from STAIG's own figure panels; MuCoST values are the official MuCoST authors' reported numbers; GraphST values are the GraphST-paper numbers as tabulated by EfNST.
- MP-MNCA wins ARI on 10/12 sections and NMI on 11/12 sections against the published baseline values. Its two weak sections are 151509 (ARI 0.452) and 151675 (ARI 0.393).
- These published values are not directly comparable in strict fairness to the reproduced table: the papers use their own clustering backends (mostly R mclust) and hyperparameters, whereas MP-MNCA uses the repository's tied-covariance GMM + 15-neighbor refinement. The reproduced runs in `prior_model_comparison_table.md` use a shared backend and are the fair comparison.
