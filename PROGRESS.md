# Progress log — read this before starting new work

Living status doc for the multi-stage BYOL -> cross-modal -> tissue-representation
pipeline built in `src/`. Kept up to date so a fresh session can pick up work without
re-deriving history. See `CLAUDE.md` for general repo navigation and
`outputs/README.md` for the `src/train/` run-by-run config/results index.

## Embedding keys currently in `checkpoints/dlpfc.pkl`

All 128-d, per spot, present in every section's `obsm`:

| key | source | trained how |
|---|---|---|
| `img_emb` | pre-existing | frozen BYOL image encoder (given, not built in this repo) |
| `proj_emb` | `src/train/` | MLP projector on `img_emb`, same-layer/diff-layer contrastive loss |
| `gene_emb` | `src/gene_encoder/` | BYOL on raw gene expression |
| `gene_emb_cm_img`, `img_emb_cm` | `src/cross_modal/` | InfoNCE alignment, `gene_emb` <-> `img_emb` |
| `gene_emb_cm_proj`, `proj_emb_cm` | `src/cross_modal/` | InfoNCE alignment, `gene_emb` <-> `proj_emb` |

## 1. Gene-expression BYOL encoder -- `src/gene_encoder/`

**Goal**: produce `gene_emb`, a 128-d self-supervised embedding of each spot's gene
expression, analogous to the pre-existing frozen `img_emb`.

**Data facts established**: `adata.X` is log1p-normalized (not raw counts), 97.8%
sparse, 33,538 genes. Per-section HVG flags only overlap ~35-40% between sections --
not usable directly for one shared encoder across sections.

**Pipeline, in order**:

1. `hvg_selection.py` -- 3,000 HVGs computed **jointly, batch-aware** across all 12
   sections (`flavor="seurat"`, not `seurat_v3`, since no raw counts exist) -> one
   consistent gene index space. Output: `outputs/gene_encoder/shared_hvgs.json`.
2. `spatial_graph.py` -- k=6 spatial nearest-neighbor graph per section (Visium hex
   grid). Output: `outputs/gene_encoder/spatial_knn_graph.pkl`.
3. `augmentations.py` -- the two BYOL views: **View A** = masked corruption
   (VIME-style: replace `mask_rate` fraction of genes with a draw from that gene's own
   min/max range) + Gaussian noise; **View B** = fixed spatial-neighbor mean
   (precomputed once, deterministic, not resampled per step).
4. `model.py` -- uses the actual `byol-pytorch` PyPI package, not a hand-rolled BYOL
   (explicit user instruction after an initial hand-rolled version). `GeneEncoder`
   (plain MLP) replaces the library's usual ResNet, wrapped via
   `byol_pytorch.BYOL(..., hidden_layer=-1, augment_fn=..., augment_fn2=...)`.
   Non-obvious trick: the library's `__init__` unconditionally probes itself with a
   mock *image* tensor to lazily size its projector, so `augment_fn`/`augment_fn2` are
   written to treat their input as spot row-*indices* (a 1-D LongTensor), not raw
   data -- real calls index into precomputed expression tensors; the mock probe (wrong
   dtype/shape) is detected and short-circuited to a dummy tensor.
5. **Collapse investigation -- don't repeat this**: plain BYOL (no regularizer)
   collapsed -- a *trained* encoder had **less** embedding diversity than an
   untrained/random one (pairwise cosine similarity ~0.99 trained vs. ~0.61 untrained,
   vs. ~0.28 for the raw input data itself). Lowering LR (3e-4 -> 1e-5) delayed
   collapse but didn't prevent it (still ~0.98 by epoch 25). Strengthening
   augmentation alone (`mask_rate` 0.3->0.5, added `noise_std_fraction=0.1`) barely
   changed anything (0.91 vs 0.88 at similar epoch counts -- same collapse
   trajectory). **Fix**: added a VICReg-style variance regularizer
   (`variance_regularization()` in `model.py`) directly on the encoder's own raw
   output (not the projector's), added on top of the BYOL loss. This worked --
   `embed_std` stabilized around ~1.4 instead of collapsing toward 0.
6. `train.py` -- three-way split (train/val used so far; `--test-fraction` /
   whole-group test wiring exists in the code but is unused by default). Validated
   config: `lr=1e-5`, `mask_rate=0.5`, `noise_std_fraction=0.1`, `variance_weight=5.0`,
   `variance_gamma=1.0`. **Final run**: early-stopped at epoch 130 (best epoch 110),
   `val_loss=0.174`, split 37,863 train / 9,466 val (0 test). Resulting embeddings:
   same-layer cosine similarity 0.57 vs. different-layer 0.34 -- real structure,
   learned entirely self-supervised, no ground-truth labels used during training.
7. `attach_gene_embeddings.py` -- safe barcode-keyed merge (write to `.tmp`, reload +
   verify, atomic replace) into `checkpoints/dlpfc.pkl` as `gene_emb`.

## 2. Cross-modal InfoNCE alignment -- `src/cross_modal/`

**Goal**: align `gene_emb` with the image-side embeddings via CLIP-style contrastive
learning, since they were never trained to share a space.

1. `model.py` -- `ProjectionHead` (Linear -> LayerNorm -> **GELU** -> Dropout ->
   Linear; GELU chosen over ReLU per explicit request, to match CLIP's own projection
   layers -- a graph-attention head was discussed and declined as unnecessary
   complexity for this step). **Both** a gene-side and image-side head are trainable
   (no frozen anchor) -- standard CLIP structure: `clip_loss()` (symmetric InfoNCE,
   in-batch negatives), learnable `logit_scale` (temperature, clamped to `log(100)`
   after each step), `retrieval_accuracy()` for monitoring.
2. `train.py` -- one run per `--image-key` (`img_emb` or `proj_emb`).
   `--test-unit {none,section,donor}` + `--test-sections`/`--test-donors` wired in for
   a *later* generalization-to-unseen-tissue run (whole-group holdout, not a random
   per-spot fraction), unused so far (`test_unit=none` in both runs to date). Sanity
   check confirmed: initial loss ~ log(512) ~ 6.24, exactly matching theoretical
   random-chance InfoNCE loss at that batch size -- confirms the loss implementation
   is correct.
3. **Two runs completed** (both `test_unit=none`, 37,863 train / 9,466 val):
   - `gene_emb` <-> `img_emb`: best epoch 34 (early-stopped 59), val_loss 5.90, train
     retrieval accuracy 0.016, val retrieval accuracy ~0.010.
   - `gene_emb` <-> `proj_emb`: best epoch 85 (early-stopped 110), val_loss 5.88,
     train retrieval accuracy **only 0.009** -- notably weaker. Plausible reading:
     `proj_emb`'s aggressive same-layer-clustering optimization compressed away the
     per-spot distinguishing signal InfoNCE's one-to-one retrieval task needs.
   - Neither run collapsed (InfoNCE structurally can't collapse the way BYOL did --
     constant output gives maximal, not minimal, loss).
4. `attach_cross_modal_embeddings.py` -- same safe-merge pattern; each run writes
   **two** new keys (gene-side + image-side), auto-derived from `image_key`:
   `img_emb` run -> `gene_emb_cm_img` + `img_emb_cm`; `proj_emb` run ->
   `gene_emb_cm_proj` + `proj_emb_cm`. All four verified present and correctly shaped
   in `checkpoints/dlpfc.pkl`.

## 3. Earlier work (image side): `src/train/`, `src/analysis/`, `src/plots/`

- `src/train/` -- projector on `img_emb` -> `proj_emb`. History includes a real bug
  (pairing key accidentally used tissue section id instead of ground-truth layer,
  because an inline sampler class was wired to the wrong array) that was found via a
  k-NN accuracy regression and fixed by switching to `src/train/pairs.py`'s
  `BalancedPairSampler`. Best runs (`layer_projector_e200_similarity_gap`,
  `layer_projector_e_200_pw_5_similarity_gap`) reached 78.1-78.5% k-NN accuracy
  against ground truth, vs. ~65-69% for `loss`-selected checkpoints of the same
  architecture -- checkpoint-selection *criterion* (`similarity_gap` vs `loss`)
  mattered more than epoch budget or `positive_weight` alone. Full run-by-run
  config/results table: `outputs/README.md`.
- `src/analysis/` -- k-NN before/after-projection comparisons
  (`knn_projection_analysis.py`, `k=49` chosen by direct accuracy maximization over a
  sweep, not an elbow method -- elbow selection is a k-means/unsupervised-clustering
  technique, doesn't apply to supervised k-NN). Also a **separate, earlier**
  covariance/Riemannian-mean pipeline (`slice_grouping.py` -> `slice_covariance.py` ->
  `slice_eigen_sort.py` -> `layer_riemannian_mean.py`): groups spots into ~20-spot
  random "slices" within one section+layer, builds a 20x20 SPD matrix per slice via
  softmax(cosine_similarity/tau) @ .T (temperature `tau=0.05`, needed after un-tempered
  softmax was found to collapse the spectrum to near rank-1), canonicalizes by sorted
  eigenvalues, then Riemannian/log-Euclidean means per layer, pooled across sections.
  Gene-embeddings-only, structurally unrelated to the multimodal covariance pipeline
  below despite similar terminology. Outputs: `outputs/covariance/`.
- `src/plots/` -- per-run metric plots (`plot_metrics.py`).

## 4. Multimodal tissue representation -- `src/multimodal/` (in progress)

Four ablations comparing ways to build one tissue/spot representation from the aligned
`gene_emb_cm_img` + `img_emb_cm` pair (the user's explicit choice -- the more
strongly cross-modally-aligned pair, see section 2). Ground-truth layer labels are
pooled across **all 12 sections / 3 donors** for evaluation (confirmed, not yet
reconsidered as a possible alternative: per-section evaluation would be an easier
task, since pooling requires cross-section invariance and we know sections carry real
technical variation, e.g. the HVG-overlap finding above). Majority-class baseline
(always predict `Layer_3`) = **37.2%** accuracy -- a real bar most of these methods
have not cleared.

| method | type | status | accuracy | notes |
|---|---|---|---|---|
| 1. Bisector embedding (`bisector.py`) + KMeans(7) | unsupervised | done | 32.2% | ARI 0.090, NMI 0.163 -- below baseline |
| 2a. Log-Euclidean K-Means on per-spot covariance (`covariance_clustering.py`) | unsupervised | done | 24.7% | below baseline |
| 2b. Spectral clustering on unit correlation vectors (`covariance_clustering.py`) | unsupervised | **stopped, unresolved** | -- | full 47k-spot run doesn't finish in 20+ min even with PCA(50 dims)+`n_jobs=-1` (sklearn has no GPU/MPS path; bottleneck is `scipy.sparse.linalg.eigsh` on the 47k-node graph Laplacian); `--spectral-subsample` flag (default 15000, stratified) is built and ready but not yet invoked; **awaiting a decision**: retry full-scale, use the subsample flag, drop 2b, or reconsider the pooled-vs-per-section evaluation scope first |
| 3. Riemannian MDM (`riemannian_mdm.py`) | **supervised** | done | 34.96% | per-layer log-Euclidean mean covariance from train spots only, nearest-mean classification on 14,199 test spots; shares its train/test split with method 4; below baseline |
| 4. Cross-attention + self-attention fusion (`attention_fusion.py`) | **supervised** | done | **60.8%** | by far the strongest result; bidirectional cross-attention (image<->gene, each a single-token sequence) then self-attention over the resulting 2-token pair (the "self-attention within them" design was an interpretation, flagged but not blocked on, since true self-attention over a length-1 single-modality sequence would be a no-op); early-stopped epoch 83 (best ~65) |

Shared infrastructure: `src/multimodal/data.py` (loads the embedding pair + labels,
all sections pooled), `src/multimodal/covariance.py` (per-spot 128x128 "covariance
between image and gene embedding" -- confirmed with the user to be the standard
2-observation covariance formula; closed-form as `0.5 * outer(diff, diff) + eps * I`
where `diff = image_vec - gene_vec`, validated to ~1e-11 against brute-force
`np.cov`/`scipy.linalg.logm`; matrix log/reconstruction also closed-form, so nothing is
ever materialized as a full `(47329, 128, 128)` array), `src/multimodal/evaluation.py`
(Hungarian-matched cluster accuracy + ARI + NMI for unsupervised methods; one shared
stratified train/test split reused by methods 3 and 4 for a fair comparison).

## Established engineering patterns (apply these to new steps)

- Every "attach embeddings to checkpoint" script: write to a `.tmp` path -> reload and
  verify shapes/keys -> atomic `Path.replace()` -- a failure partway through never
  leaves a corrupted checkpoint.
- Barcode-keyed merges always key on `(section_id, barcode)` pairs, never barcode
  alone -- confirmed barcodes repeat across the 12 sections (10x Visium reuses one
  fixed barcode whitelist per array design; only 4,941 distinct barcode strings across
  47,329 total spots).
- `checkpoints/dlpfc.pkl` (~5GB) and `outputs/` are git-ignored, except
  `outputs/README.md` (an explicit `.gitignore` exception -- `outputs/*` +
  `!outputs/README.md`), which is the canonical run-index for `src/train/` experiments.
- Long-running background jobs: always launch with `python -u` (unbuffered) --
  buffered stdout piped to a file can look like zero progress for many minutes even
  when a job is working normally, learned the hard way this session.
- sklearn has no GPU/MPS backend at all (PyTorch training scripts use MPS via
  `torch.backends.mps.is_available()`; sklearn algorithms -- KMeans, PCA,
  SpectralClustering -- are pure CPU/NumPy/SciPy, no device option exists for them).
- Repo is live on GitHub: `github.com/sidharthanSC/GeoPatch-Multimodal-Integration`
  (public, SSH remote). Pushed so far: the layer-projector fix/experiments and the
  `src/cross_modal/` addition. **Not yet pushed**: `src/gene_encoder/`,
  `src/multimodal/`, this file, and the current uncommitted state generally.
