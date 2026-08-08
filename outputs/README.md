# outputs/ — run index

Every run below was produced by `python -m src.train.train --run-name <name> ...` (see
`src/train/train.py`; run `--help` for the full flag list). For a given `<name>`, its
artifacts are:

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
