# Prior Models

This package contains importable repository-local implementations of published prior models. Prior-model outputs must be written under `outputs/prior_models/<run_id>/`; they must not mutate `checkpoints/dlpfc.pkl`.

## STAIG

`src.prior_models.staig` implements the graph-contrastive portion of STAIG using the data available in this repository:

```text
section expression -> 3,000 section HVGs -> two masked graph views
coordinates -> symmetric 5-NN graph
fixed img_emb -> standardized PCA-16 -> adaptive edge deletion + 40 pseudo-label classes
one-layer 64-d GCN -> 64-d projection MLP -> debiased neighbor contrastive loss
tied-covariance GMM -> 15-neighbor spatial refinement
```

The implementation is an adaptation, not an exact image-pipeline reproduction. STAIG trains BYOL on 3.5-fiducial-diameter H&E patches after Gaussian and band-pass filtering. Those patches and features are unavailable here, so `adata.obsm["img_emb"]` is substituted.

Importable APIs:

- `src.prior_models.staig.config.StaigConfig`
- `src.prior_models.staig.data.prepare_section`
- `src.prior_models.staig.model.StaigModel`
- `src.prior_models.staig.model.neighbor_contrastive_loss`
- `src.prior_models.staig.train.fit_staig`
- `src.prior_models.staig.train.run_dlpfc`
- `src.prior_models.staig.evaluate.tied_gmm`
- `src.prior_models.staig.evaluate.refine_labels`
- `src.prior_models.staig.published.compare_with_published`

Canonical command:

```powershell
python -u -m src.prior_models.staig.train --checkpoint-path checkpoints/dlpfc.pkl --output-dir outputs/prior_models/staig_paper_default_img_emb_seed0_all12_v3 --epochs 400 --seed 0 --temperature 10 --image-pseudo-clusters 40 --feature-mask-rate-1 0.1 --feature-mask-rate-2 0.1 --dtype float32
```

Full protocol, deviations, and published comparisons are in `PRIOR_MODEL_PROTOCOLS.md`.

## Multi-Scale Gene Inputs

`src.prior_models.staig.data.PrecomputedFeatureBundle` allows immutable NPZ arrays to replace node or image features while retaining strict `(section_id, barcode)` matching. `src.prior_models.staig.multiscale_evaluation` provides matched direct clustering and within-section frozen probes. The completed S0-S6 study and negative result are documented in `observations/MULTISCALE_GENE_STAIG_BENCHMARK.md`.

The canonical superseding study uses all 33,538 genes through `src.gene_encoder.rich_train`, exports 4,096/1,024/512/128-dimensional stages, and evaluates three targeted STAIG conditions through `src.prior_models.staig.matrix`. Stage 4,096 reaches median refined ARI `0.52293`; see `RICH_GENE_EXPRESSION_EXECUTION.md`.
