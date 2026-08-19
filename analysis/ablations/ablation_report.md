# MP-MNCA Ablation Study — Per-Section Refined ARI / NMI

Each 3000-dim ablation keeps the canonical Phase-1 protocol fixed (20 epochs, seed 0, batch 256, lr 3e-4, mask rate 0.1, temperature 10, 8 heads, k=6 neighbors, image PCA-16, 40 image pseudo-clusters) and removes exactly one component. Evaluation is the shared tied-covariance GMM + 15-neighbor spatial refinement backend.

## Summary Table

| Variant | Input dim (genes) | Mean refined ARI | Mean refined NMI | Median ARI | Median NMI | Scientific question |
|---|---|---|---|---|---|---|
| Full MP-MNCA **(full model)** | 3000 | 0.783 | 0.782 | 0.846 | 0.807 | Complete model |
| no_morphology_prior | 3000 | 0.745 | 0.752 | 0.790 | 0.823 | Does histology-guided routing help? (beta = 0) |
| no_position_bias | 3000 | 0.661 | 0.692 | 0.742 | 0.698 | Does learned local geometry help? (gamma = 0) |
| gene_only_attention | 3000 | 0.681 | 0.709 | 0.769 | 0.723 | How strong is learned molecular routing alone? (beta = gamma = 0) |
| uniform_knn | 3000 | 0.775 | 0.775 | 0.785 | 0.780 | Is adaptive cross-attention better than fixed smoothing? (alpha_ij = 1/k) |
| embedding_only | 3000 | 0.465 | 0.599 | 0.470 | 0.589 | What is gained over direct clustering of spot embeddings? |
| contrastive_v3 | 3000 | 0.272 | 0.329 | 0.222 | 0.317 | Effect of strong representation compression (128-d bottleneck) |
| gene_cosine_v2 | 3000 | 0.243 | 0.324 | 0.222 | 0.322 | Earlier low-dimensional molecular variant |
| byol_v1 | 3000 | 0.234 | 0.294 | 0.234 | 0.294 | Earlier low-dimensional visual/self-supervised variant (1/12 sections) |
| STAIG | 3000 | 0.507 | 0.644 | 0.518 | 0.674 | Canonical reproduced graph-contrastive baseline |

## Per-Section Refined ARI

| Section | Full MP-MNCA | no_morphology_prior | no_position_bias | gene_only_attention | uniform_knn | embedding_only | contrastive_v3 | gene_cosine_v2 | byol_v1 | STAIG |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 151507 | 0.935 | 0.935 | 0.872 | 0.873 | 0.804 | 0.471 | 0.224 | 0.228 | 0.234 | 0.585 |
| 151508 | 0.916 | 0.903 | 0.905 | 0.892 | 0.861 | 0.347 | 0.264 | 0.244 | — | 0.486 |
| 151509 | 0.452 | 0.449 | 0.472 | 0.452 | 0.816 | 0.449 | 0.421 | 0.427 | — | 0.583 |
| 151510 | 0.875 | 0.877 | 0.873 | 0.874 | 0.840 | 0.446 | 0.353 | 0.357 | — | 0.498 |
| 151669 | 0.889 | 0.654 | 0.626 | 0.741 | 0.746 | 0.468 | 0.413 | 0.342 | — | 0.353 |
| 151670 | 0.817 | 0.669 | 0.689 | 0.686 | 0.795 | 0.497 | 0.460 | 0.221 | — | 0.399 |
| 151671 | 0.798 | 0.786 | 0.796 | 0.797 | 0.731 | 0.644 | 0.204 | 0.195 | — | 0.480 |
| 151672 | 0.956 | 0.960 | 0.848 | 0.964 | 0.786 | 0.522 | 0.152 | 0.155 | — | 0.546 |
| 151673 | 0.724 | 0.841 | 0.830 | 0.807 | 0.664 | 0.614 | 0.204 | 0.202 | — | 0.553 |
| 151674 | 0.897 | 0.793 | 0.315 | 0.314 | 0.784 | 0.330 | 0.209 | 0.197 | — | 0.525 |
| 151675 | 0.393 | 0.307 | 0.393 | 0.397 | 0.755 | 0.262 | 0.221 | 0.223 | — | 0.564 |
| 151676 | 0.743 | 0.768 | 0.309 | 0.378 | 0.722 | 0.533 | 0.136 | 0.125 | — | 0.512 |

## Per-Section Refined NMI

| Section | Full MP-MNCA | no_morphology_prior | no_position_bias | gene_only_attention | uniform_knn | embedding_only | contrastive_v3 | gene_cosine_v2 | byol_v1 | STAIG |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 151507 | 0.913 | 0.913 | 0.872 | 0.873 | 0.823 | 0.646 | 0.344 | 0.324 | 0.294 | 0.696 |
| 151508 | 0.893 | 0.886 | 0.887 | 0.874 | 0.841 | 0.574 | 0.357 | 0.328 | — | 0.642 |
| 151509 | 0.473 | 0.465 | 0.490 | 0.480 | 0.763 | 0.579 | 0.447 | 0.467 | — | 0.683 |
| 151510 | 0.829 | 0.831 | 0.829 | 0.830 | 0.792 | 0.518 | 0.411 | 0.407 | — | 0.665 |
| 151669 | 0.802 | 0.635 | 0.620 | 0.683 | 0.689 | 0.599 | 0.315 | 0.337 | — | 0.496 |
| 151670 | 0.719 | 0.626 | 0.636 | 0.633 | 0.706 | 0.567 | 0.340 | 0.297 | — | 0.501 |
| 151671 | 0.764 | 0.753 | 0.760 | 0.763 | 0.746 | 0.670 | 0.312 | 0.307 | — | 0.637 |
| 151672 | 0.923 | 0.929 | 0.849 | 0.933 | 0.778 | 0.633 | 0.275 | 0.275 | — | 0.677 |
| 151673 | 0.796 | 0.846 | 0.839 | 0.822 | 0.757 | 0.727 | 0.315 | 0.319 | — | 0.695 |
| 151674 | 0.881 | 0.841 | 0.466 | 0.468 | 0.817 | 0.579 | 0.317 | 0.317 | — | 0.681 |
| 151675 | 0.577 | 0.488 | 0.584 | 0.582 | 0.809 | 0.466 | 0.317 | 0.325 | — | 0.678 |
| 151676 | 0.811 | 0.816 | 0.470 | 0.572 | 0.782 | 0.632 | 0.197 | 0.189 | — | 0.671 |

## Notes

- `Full MP-MNCA` = `phase1_all12` + `phase1_all12_remaining` (verified from saved embeddings; mean refined ARI 0.783 / NMI 0.782).
- `contrastive_v3`, `gene_cosine_v2` are 12-section runs; `byol_v1` is a 1-section (151507) run and is flagged as such.
- `STAIG` = canonical reproduced baseline `staig_paper_default_img_emb_seed0_all12_v3`.
- The `Input dim (genes)` column is the gene-expression dimension fed to each encoder (3000-d joint HVGs for every method). Internal embedding dimensions differ: STAIG encodes to 64-d, the historical MP-MNCA variants to 128-d, and the Phase-1 family operates at the full 3000-d gene dimension.
- The ablation outputs live in `outputs/ablations/<variant>/` with checkpoints, embeddings, `section_metrics.csv` and `summary.json`.