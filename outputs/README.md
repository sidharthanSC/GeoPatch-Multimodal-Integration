# outputs/ — run index

The fixed benchmark protocol is `MODEL_BENCHMARK_REPORT.md`, resumable execution state is
`EXECUTION_CHECKPOINTS.md`, and generated result tables are maintained in
`MULTIMODAL_EVALUATION_RESULTS.md`. New GBSSA and evaluation runs must be added here when
their artifacts are verified; a planned run is not a completed artifact.

## Evaluation artifacts

| run | status | artifacts | result |
|---|---|---|---|
| `20260813_checkpoint_audit_v1` | verified | `outputs/evaluation/20260813_checkpoint_audit_v1/{audit.json,legacy_pooled_bisector.json,data/}` | 47,329 spots and seven embedding keys validated; pooled bisector reproduced at accuracy 0.322424, ARI 0.090455, NMI 0.162604 |
| `20260813_existing_embeddings_kmeans_seed0_v1` | verified gate | `outputs/evaluation/20260813_existing_embeddings_kmeans_seed0_v1/clustering/` | one-seed per-section KMeans for R2-R11 |
| `20260813_existing_embeddings_kmeans_10seeds_v1` | verified | `outputs/evaluation/20260813_existing_embeddings_kmeans_10seeds_v1/clustering/` | 10-seed per-section KMeans; aligned bisector median ARI 0.14187/NMI 0.23837 versus raw image ARI 0.15245/NMI 0.22667 |
| `20260814_gbssa_r01_seed0_kmeans_seed0_v1` | verified gate | `outputs/evaluation/20260814_gbssa_r01_seed0_kmeans_seed0_v1/clustering/` | new gene BYOL + standard CLIP; bisector median ARI 0.14365/NMI 0.22207 |
| `20260814_gbssa_r02_seed0_kmeans_seed0_v1` | negative | `outputs/evaluation/20260814_gbssa_r02_seed0_kmeans_seed0_v1/clustering/` | ratio 2 chance retrieval; bisector ARI 0.09225 |
| `20260814_gbssa_r05_seed0_kmeans_seed0_v1` | negative | `outputs/evaluation/20260814_gbssa_r05_seed0_kmeans_seed0_v1/clustering/` | ratio 5 chance retrieval; bisector ARI 0.04597 |
| `20260814_gbssa_r10_seed0_kmeans_seed0_v1` | negative | `outputs/evaluation/20260814_gbssa_r10_seed0_kmeans_seed0_v1/clustering/` | ratio 10 chance retrieval; bisector ARI 0.03790 |
| `20260814_gbssa_r01_seed0_corrected_kmeans_10seeds_v1` | verified | `outputs/evaluation/20260814_gbssa_r01_seed0_corrected_kmeans_10seeds_v1/clustering/` | new-gene controls + standard CLIP; unaligned bisector median ARI 0.16323 versus aligned bisector 0.14706 |
| `20260814_gbssa_r01_seed0_corrected_gmm_seed0_v1` | verified sensitivity | `outputs/evaluation/20260814_gbssa_r01_seed0_corrected_gmm_seed0_v1/clustering/` | PCA-30 sklearn GMM surrogate; aligned bisector median ARI 0.17200/NMI 0.26918 |
| `20260814_gbssa_r01_seed0_corrected_kmeans_10seeds_v1/spatial_refinement` | verified sensitivity | `metrics.parquet`, `assignments.parquet`, `aggregate_summary.json` | common 6-NN strict-majority refinement; unaligned bisector ARI 0.19937, aligned bisector 0.18346 |
| `20260814_alignment_diagnostics_v1` | verified diagnostic | `outputs/evaluation/20260814_alignment_diagnostics_v1/` | ratio 1 cosine gap 0.06063; ratio 1.1 effective rank collapsed to 1.88/2.47; ratio 5 effectively constant |
| `20260814_seed0_random_spot_logistic_probe_gate_v2` | verified gate | `outputs/evaluation/20260814_seed0_random_spot_logistic_probe_gate_v2/` | aligned concatenation accuracy 0.5331, balanced accuracy 0.4167, macro-F1 0.4151 |
| `20260814_gbssa_r01_seed0_attention_ablation_v1` | verified gate | `outputs/evaluation/20260814_gbssa_r01_seed0_attention_ablation_v1/` | self-attention-only accuracy 0.61251 beats parameter-matched concat MLP 0.52997 and full model 0.60532 |
| `20260814_gbssa_r01_seed0_attention_legacy_seed1_v1` | verified gate | `outputs/evaluation/20260814_gbssa_r01_seed0_attention_legacy_seed1_v1/` | full model accuracy 0.60159 versus self-attention-only seed-1 accuracy 0.60666 |
| `20260814_integration_151675_151676_v1` | verified common integration | `outputs/evaluation/20260814_integration_151675_151676_v1/` | aligned bisector best ARI 0.1120; aligned gene best local mixing |
| `20260814_integration_151507_151508_151675_151676_v1` | verified common integration | `outputs/evaluation/20260814_integration_151507_151508_151675_151676_v1/` | unaligned concatenation best ARI 0.1509; aligned gene best local mixing |
| `20260814_weighted_bisector_seed0_kmeans_10seeds_v1` | verified sensitivity | `outputs/evaluation/20260814_weighted_bisector_seed0_kmeans_10seeds_v1/clustering/` | full fixed-weight curve; gene/image 0.25/0.75 median ARI 0.16564 |

## Gene BYOL GBSSA preparation

| run/artifact | status | result |
|---|---|---|
| `shared_hvgs_gbssa.json` | verified | 3,000 joint batch-aware HVGs |
| `spatial_knn_graph_gbssa.pkl` | verified | six within-section neighbors for 47,329 spots |
| `gene_byol_gbssa_seed0` | verified | early stop epoch 105, best val loss 0.1368; non-collapsed 128-d embeddings exported |
| `gene_byol_gbssa_seed1` | verified | early stop epoch 122, best val loss 0.1648; unaligned-bisector median ARI 0.14013 |
| `gene_byol_gbssa_seed2` | verified | early stop epoch 117, best val loss 0.1468; unaligned-bisector median ARI 0.13869 |

## GBSSA cross-modal runs

All use `gene_byol_gbssa_seed0` against raw external `img_emb`, hidden/output dimensions 256/128, batch size 512, LR `3e-4`, and seed 0.

| run | ratio | status | interpretation |
|---|---:|---|---|
| `gbssa_img_r01_seed0` | 1 | verified baseline | standard CLIP, above-chance retrieval |
| `gbssa_img_r02_seed0` | 2 | rejected | alignment retrieval converged to chance |
| `gbssa_img_r05_seed0` | 5 | rejected | alignment retrieval converged to chance |
| `gbssa_img_r10_seed0` | 10 | rejected | alignment retrieval converged to chance |

The runs immediately below (`layer_projector*`) were produced by
`python -m src.train.train --run-name <name> ...` (see `src/train/train.py`; run
`--help` for the full flag list). For a given `<name>`, its artifacts are:

- `checkpoints/<name>/{best.pt, last.pt, projector_final.pt}` — `best.pt` is selected by
  whichever `--best-metric` that run used (see table below); `last.pt` is simply the
  final epoch trained; `projector_final.pt` holds just the target projector's weights
  (no optimizer state) from `best.pt`, for lightweight reuse.
- `metrics/<name>_metrics.csv` — one row per epoch (train/val loss, same/diff-layer
  similarity, similarity gap, negative-margin violation rate).
- `predictions/<name>_embeddings.npz` — `raw_embeddings` / `projected_embeddings` (both
  128-d) for **all 47,329 spots** (train+val combined — there is no held-out test split
  anywhere in this pipeline), plus `cortical_labels` and `slice_ids` for each spot. No
  barcode field; see `src/analysis/knn_projection_analysis.py`'s
  `load_barcodes_labels_and_slices` to recover barcodes from `checkpoints/dlpfc.pkl`.
- `plots/<name>_{loss,similarity,similarity_gap,negative_margin_violation_rate}.png` —
  via `python -m src.plots.plot_metrics --metrics-csv metrics/<name>_metrics.csv`.
- `analysis/<name>_{knn_projection_summary.json, knn_label_transitions.csv,
  changed_spots.csv, knn_predictions_full.csv}` — via
  `python -m src.analysis.knn_projection_analysis --embeddings-npz predictions/<name>_embeddings.npz --k 49`.
  Compares a leave-one-out k-NN neighborhood-majority label (against ground-truth
  cortical layer) in the raw embedding space vs. the projected one, for every spot.
  `k=49` was chosen from `analysis/knn_k_elbow.csv` (see below) as the k that maximizes
  mean k-NN accuracy across the runs swept there — not an elbow-plot rule (elbow
  selection is a k-means/unsupervised-clustering technique and doesn't apply to
  supervised k-NN here), just direct accuracy-vs-k maximization against ground truth.

All pairing throughout is by ground-truth cortical layer (`adata.obs["ground_truth"]`,
`Layer_1`..`Layer_6`/`WM`), via `BalancedPairSampler` in `src/train/pairs.py` — same-layer
pairs get `eta=0`, different-layer pairs get `eta=1`. `slice_ids` (tissue section, e.g.
`"151507"`) rides along as metadata only and is never the pairing key (see `src/train/`
in `CLAUDE.md` for the history of a bug where an earlier script version got this backwards).

## Run configurations

| run | epoch budget | patience | best_metric | best epoch | positive_weight | negative_margin |
|---|---|---|---|---|---|---|
| `layer_projector` | 40 | 8 | `similarity_gap`* | 40 | 2.0 | 0.0 |
| `layer_projector_e200` | 200 | 30 | `loss` | 25 (early-stopped at 55) | 2.0 | 0.05 |
| `layer_projector_e_200_pw_5` | 200 | 30 | `loss` | 36 (early-stopped at 66) | 5.0 | 0.0 |
| `layer_projector_e200_similarity_gap` | 200 | 30 | `similarity_gap` | 102 (early-stopped at 132) | 2.0 | 0.0 |
| `layer_projector_e_200_pw_5_similarity_gap` | 200 | 30 | `similarity_gap` | 199 (ran full 200) | 5.0 | 0.0 |

\* `layer_projector` predates the `--best-metric` flag; checkpoint selection was
hardcoded to `similarity_gap` at the time.

All runs share: `hidden_dim=256`, `projection_dim=128`, `dropout=0.1`, `lr=3e-4`,
`val_fraction=0.2`, `seed=0`. `negative_weight=1.0` throughout — only `positive_weight`
varies (2.0 vs 5.0). `negative_margin=0.05` for `layer_projector_e200` was *not* a
deliberate experimental choice — it's whatever `TrainConfig`'s default happened to be in
`src/train/train.py` at the moment that run was launched (the file was hand-edited
between runs several times over the course of this project); every other run used the
`negative_margin=0.0` default. Worth knowing if comparing that run specifically.

## Results (k-NN accuracy vs. ground-truth layer, k=49)

| run | accuracy (raw) | accuracy (projected) | spots improved | spots regressed |
|---|---|---|---|---|
| `layer_projector` | 59.4% | 69.4% | 9,038 | 4,342 |
| `layer_projector_e200` | 59.4% | 65.6% | 7,712 | 4,773 |
| `layer_projector_e_200_pw_5` | 59.4% | 66.1% | 8,269 | 5,099 |
| `layer_projector_e200_similarity_gap` | 59.4% | **78.1%** | 11,978 | 3,164 |
| `layer_projector_e_200_pw_5_similarity_gap` | 59.4% | **78.5%** | 12,241 | 3,240 |

`accuracy_raw` is identical across all rows because `raw_embeddings` (the frozen,
pre-projection BYOL encoder output) is exactly the same array in every run's `.npz` —
only `projected_embeddings` differs. Selecting the checkpoint by `similarity_gap`
instead of `loss`, combined with a longer training budget (200 epochs vs. 40, with
early stopping actually engaging), is what drives the large accuracy jump for the two
`_similarity_gap` runs — not `positive_weight` on its own (compare `layer_projector_e200`
vs. `layer_projector_e200_similarity_gap`: same `positive_weight=2.0`, only
`best_metric` differs).

## Cross-modal runs (`gene_emb` <-> image embedding, InfoNCE)

Produced by `python -m src.cross_modal.train --image-key <img_emb|proj_emb> --run-name <name> ...`
(see `src/cross_modal/train.py`; run `--help` for the full flag list; `outputs/cross_modal/`
is a separate subtree, not `outputs/checkpoints|metrics|predictions/` used by the
`src/train/` runs above). Each run trains **two** trainable projection heads
(`src/cross_modal/model.py`) with a symmetric CLIP-style InfoNCE loss (in-batch
negatives, learnable temperature) -- one maps `gene_emb` into a shared space, the other
maps the chosen image-side embedding (`img_emb` or `proj_emb`) into the same space -- so
each run produces two new embeddings, not one. Artifacts per `<name>`:

- `checkpoints/<name>/{best.pt, last.pt}` -- `best.pt` selected by lowest `val_loss`.
- `metrics/<name>_metrics.csv` -- one row per epoch (train/val loss and retrieval accuracy).
- `predictions/<name>_embeddings.npz` -- `gene_projected` / `image_projected` (both 128-d)
  for all 47,329 spots, plus `barcodes`, `section_ids`, `split` (`"train"`/`"val"`/`"test"`),
  and `image_key` (which source this run used).

Both runs used `test_unit=none` (no held-out test set this round -- 37,863 train /
9,466 val, same split fraction as the gene encoder). `--test-unit {section,donor}` is
wired in for a later run testing generalization to entirely unseen tissue (whole
sections or whole 4-section donors, not just unseen spots) -- see the script's
docstring.

| run | image_key | best epoch | val_loss | val_retrieval_accuracy | train_retrieval_accuracy |
|---|---|---|---|---|---|
| `cross_modal_img_emb` | `img_emb` (raw, pre-projection) | 34 (early-stopped at 59) | 5.900 | 0.0102 | 0.0158 |
| `cross_modal_proj_emb` | `proj_emb` (post layer-projection) | 85 (early-stopped at 110) | 5.883 | 0.0072 | 0.0088 |

Both share: `hidden_dim=256`, `output_dim=128`, `dropout=0.1`, `batch_size=512`,
`lr=3e-4`, `weight_decay=1e-4`, `patience=25`, `seed=0`. Random-chance baseline at this
batch size is `loss=log(512)~=6.24`, `retrieval_accuracy~=1/512~=0.0020` -- both runs
learned real (if modest) cross-modal alignment, well above chance, with no collapse.

`img_emb` aligned noticeably better than `proj_emb` against `gene_emb` (train accuracy
0.016 vs. 0.009, roughly double) despite `proj_emb`'s own layer-projection training
being the "better" embedding by the k-NN-vs-layer measure above. Plausible reading:
`proj_emb` was optimized specifically to cluster tightly *within* a cortical layer,
which likely compresses away some of the per-spot distinguishing information a
one-to-one retrieval task (this spot's gene profile vs. this exact spot's image, out of
everyone else in the batch) needs -- squeezing out inter-layer spread doesn't
necessarily help distinguish spot A from spot B within the same layer. `img_emb`, being
less aggressively optimized toward one specific objective, apparently retained more of
that spot-level signal.

The four resulting embeddings live in `checkpoints/dlpfc.pkl`'s `obsm` (via
`python -m src.cross_modal.attach_cross_modal_embeddings --embeddings-npz <path>`,
same safe barcode-keyed write-verify-replace pattern as
`src/gene_encoder/attach_gene_embeddings.py`), auto-named from `image_key`:

| source run | gene-side key | image-side key |
|---|---|---|
| `cross_modal_img_emb` | `gene_emb_cm_img` | `img_emb_cm` |
| `cross_modal_proj_emb` | `gene_emb_cm_proj` | `proj_emb_cm` |

## Other things under `outputs/`

- `prelim/` — outputs from **before** the pairing-key fix described above (an early
  `train.py` paired by tissue section id instead of cortical layer): `byol_projector`
  and `byol_projector_margin_0.05`. Kept for historical comparison only; not directly
  comparable to the `layer_projector*` runs (different pairing key, different model/loss
  code entirely — see `src/train/model.py`'s git history via `CLAUDE.md` for context).
- `analysis/knn_k_elbow.{csv,json}` and `plots/knn_k_elbow.png` — the k-sweep
  (`python -m src.analysis.knn_k_selection`) used to choose `k=49` above. Note the
  `selected_k` field inside `knn_k_elbow.json` still says `91` — that was from a
  since-discarded elbow-plot-based selection method; the actual `k` used everywhere is
  `49`, chosen by direct accuracy maximization (see `knn_k_elbow.csv` for the full sweep
  table this was read off of).
- `plots.zip` — a manual export/backup of `plots/`, not regenerated by any script here.
