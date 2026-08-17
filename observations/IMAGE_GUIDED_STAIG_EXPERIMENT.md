# Bisector-Guided STAIG Experiment

## Initial Observation

The first end-to-end image-guided K6 experiment fine-tuned the all-gene encoder and
then supplied its stage-4,096 representation to the unchanged STAIG adaptation with
raw `img_emb` guidance. The gene checkpoint selected epoch 4 by validation total loss
`3.62151`; training was stopped after epoch 10 when validation loss rose monotonically
while training loss continued to fall.

Direct stage-4,096 median refined ARI increased from `0.44860` to `0.46428`, but the
matched STAIG result was worse:

| STAIG condition | Mean refined ARI | Median refined ARI | Mean refined NMI | Median refined NMI |
|---|---:|---:|---:|---:|
| Expression STAIG | 0.50693 | 0.51821 | 0.64353 | 0.67427 |
| Previous all-gene stage 4,096 STAIG | 0.48164 | 0.52293 | 0.64409 | 0.66682 |
| End-to-end image-guided stage 4,096 STAIG | 0.45837 | 0.46331 | 0.62101 | 0.62437 |

This is a verified negative result. It does not establish that image-guided gene
fine-tuning improves STAIG.

Artifacts:

- `outputs/gene_encoder/checkpoints/rich_gene_v3_all33538_imgk6_seed0/best.pt`
- `outputs/gene_encoder/predictions/rich_gene_v3_all33538_imgk6_seed0_embeddings.npz`
- `outputs/evaluation/20260815_rich_gene_imgk6_direct_v1/`
- `outputs/prior_models/staig_rich_gene_v3_all33538_imgk6_stage4096_img_seed0_v1/`

## Corrected Experiment

The corrected experiment leaves STAIG's node features unchanged as section-level
3,000-HVG expression. It changes only STAIG's guidance representation:

```text
gene_emb_cm_img (128) + img_emb_cm (128)
    -> normalized angular bisector (128)
    -> STAIG edge-drop probabilities and pseudo-clusters
```

The STAIG GCN, coordinate 5-NN graph, contrastive objective, 400-epoch budget,
tied-covariance GMM, and 15-neighbor refinement remain unchanged. This isolates whether
the existing multimodal spot representation provides better graph guidance than raw
`img_emb`.

## Corrected Result

Run `staig_expression_aligned_bisector_guidance_seed0_all12_v1` completed all 12
sections. It uses section-level 3,000-HVG expression as the node matrix and the aligned
bisector only for STAIG edge-drop probabilities and 40-cluster guidance pseudo-labels.

| STAIG condition | Mean refined ARI | Median refined ARI | Mean refined NMI | Median refined NMI |
|---|---:|---:|---:|---:|
| Expression STAIG, raw `img_emb` guidance | **0.50693** | 0.51821 | **0.64353** | **0.67427** |
| All-gene stage 4,096 STAIG, raw `img_emb` guidance | 0.48164 | **0.52293** | 0.64409 | 0.66682 |
| Expression STAIG, aligned-bisector guidance | 0.49069 | 0.51526 | 0.63808 | 0.67119 |

Bisector guidance is close to Expression STAIG in median refined ARI (`-0.00295`) but
lower in mean refined ARI (`-0.01625`). It beats Expression STAIG on 4 of 12 sections.
Against stage-4,096 STAIG, it has higher mean refined ARI by `0.00905`, lower median by
`0.00767`, and a 6/6 section win/loss split. The experiment therefore does not beat
either control unambiguously.

Cells are refined ARI for every biological section.

| Section | Expression STAIG | Stage 4,096 STAIG | Bisector-guided Expression STAIG |
|---|---:|---:|---:|
| 151507 | .585 | .589 | .569 |
| 151508 | .486 | .585 | .474 |
| 151509 | .583 | .494 | **.592** |
| 151510 | .498 | .464 | **.504** |
| 151669 | **.353** | .255 | .347 |
| 151670 | **.399** | .263 | .304 |
| 151671 | .480 | **.530** | .451 |
| 151672 | .546 | **.574** | .546 |
| 151673 | .553 | **.572** | .542 |
| 151674 | **.525** | .523 | .513 |
| 151675 | **.564** | .523 | .528 |
| 151676 | .512 | .407 | **.517** |

The result suggests that the aligned bisector retains enough morphology-plus-expression
structure to replace raw image guidance with little median loss, but mixing gene signal
into the guidance does not consistently improve boundary-aware edge perturbations.
These are one-seed transductive results and do not establish donor generalization.

Artifacts:

- `outputs/multimodal/aligned_bisector_staig_guidance_v1.npz`
- `outputs/prior_models/staig_expression_aligned_bisector_guidance_seed0_all12_v1/`
