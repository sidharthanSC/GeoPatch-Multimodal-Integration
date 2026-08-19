# outputs/ — run index

The fixed benchmark protocol is `MODEL_BENCHMARK_REPORT.md`, resumable execution state is
`EXECUTION_CHECKPOINTS.md`, and generated result tables are maintained in
`observations/MULTIMODAL_EVALUATION_RESULTS.md`. New GBSSA and evaluation runs must be added here when
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
| `20260814_graph_pooled_procrustes_img_k6r5_seed0_weighted_v1` | verified candidate | `clustering/`, `spatial_refinement/` | aligned concat ARI/NMI 0.17707/0.25652; refined 0.20904/0.29378; aligned 25% gene weighted mean unrefined ARI 0.18455 |
| `20260814_graph_pooled_procrustes_img_k6r5_seed0_gmm_v1` | verified sensitivity | `clustering/`, `spatial_refinement/` | aligned bisector GMM ARI/NMI 0.16663/0.25918; refined 0.18788/0.28917 |
| `20260814_graph_pooled_procrustes_img_k6r5_seed0_probe_v1` | verified gate | `metrics.parquet`, `predictions.parquet`, `summary.json` | aligned concat accuracy/balanced accuracy/macro-F1 0.55687/0.44982/0.45296 |
| `20260814_graph_pooled_procrustes_integration_151675_151676_v1` | verified integration | `metrics.parquet`, `summary.json` | aligned concat ARI/NMI 0.11635/0.17697 |
| `20260814_graph_pooled_procrustes_integration_4sections_v1` | verified integration | `metrics.parquet`, `summary.json` | aligned concat ARI/NMI 0.16758/0.22633; mixing remains weak |
| `20260814_spatial_multimodal_contrastive_seed0_v1` | verified candidate | `alignment/`, `clustering/`, `spatial_refinement/` | one-seed bisector ARI/NMI 0.19783/0.29149, refined 0.21504/0.32875; concat PCA refined ARI 0.22356 |
| `20260814_spatial_multimodal_contrastive_gmm_seed0_v1` | verified sensitivity | `clustering/`, `spatial_refinement/` | bisector GMM ARI/NMI 0.19627/0.27765, refined 0.21330/0.31494 |
| `20260814_spatial_multimodal_contrastive_probe_seed0_v1` | verified gate | `metrics.parquet`, `predictions.parquet`, `summary.json` | concat accuracy/balanced accuracy/macro-F1 0.52750/0.40424/0.40236 |
| `20260814_spatial_multimodal_contrastive_integration_151675_151676_v1` | verified integration | `metrics.parquet`, `summary.json` | bisector ARI/NMI 0.11278/0.18147 |
| `20260814_spatial_multimodal_contrastive_integration_4sections_v1` | verified integration | `metrics.parquet`, `summary.json` | bisector ARI/NMI 0.16418/0.23457; improved local mixing over graph-pooled candidate |
| `20260815_staig_adaptation_vs_published_v1` | verified comparison | `section_comparison.csv`, `summary.json` | adapted STAIG mean/median refined ARI 0.50693/0.51821 versus published STAIG 0.69167/0.68000 and published SpaGCN 0.43833/0.43500 |
| `20260815_multistage_gene128_gbssa_r01_v2_alignment` | verified diagnostic | parquet plus `summary.json` | median cosine gap 0.05355, FOSCTTM 0.30975, gene/image effective rank 7.22/33.96 |
| `20260815_multistage_gene128_gbssa_r011_v2_alignment` | negative diagnostic | parquet plus `summary.json` | ratio 1.1 contracted outputs: cosine gap 0.01615 and gene/image effective rank 3.34/3.85 |
| `20260815_multiscale_gene_staig_controls_seed0_v2` | verified multiscale evaluation | `direct_clustering.parquet`, `probe_metrics.parquet`, `summary.json`, `config.json` | stage-1024 direct refined median ARI 0.20599; expression STAIG probe median balanced accuracy 0.85665 versus best multistage 0.43179 |
| `20260815_rich_gene_all33538_direct_v1` | canonical rich-gene direct evaluation | `direct_clustering.parquet`, `preprocessing_diagnostics.parquet`, `summary.json`, `config.json` | median refined ARI: stage 4096 0.44860, stage 1024 0.42932, stage 512 0.44446, final 128 0.43236 |
| `20260815_rich_gene_all33538_alignment_v1` | canonical rich alignment diagnostic | parquet plus `summary.json` | median R@10 0.07209, FOSCTTM 0.11712, cosine gap 0.11104, gene/image effective rank 45.22/87.00 |
| `20260815_rich_gene_all33538_aligned_direct_v1` | canonical aligned clustering | `direct_clustering.parquet`, `summary.json`, `config.json` | median refined ARI/NMI: concat 0.45084/0.59663, bisector 0.40034/0.58032 |
| `20260815_adjacency_clustering_seed0_v1` | verified graph-only sensitivity | `summary.json`, `section_metrics.csv`, 36 sparse adjacency matrices, 36 assignment files | median refined ARI: spatial binary 0.19285, all-gene weighted 0.19298, image weighted 0.20916; all remain below Expression STAIG 0.51821 |
| `20260815_supervised_multiphase_study_v1` | verified conference-style supervised multimodal classification | `outputs/evaluation/20260815_supervised_study_aggregate_v1/`, `observations/SUPERVISED_MULTIPHASE_RESULTS.md` | regime‑wise balanced accuracy: LODO 0.2146 (CI 0.2036–0.2256), LOSO 0.2371 (0.2230–0.2512), within‑donor random 0.5165 (0.4680–0.5651), pooled random 0.4563 (0.4117–0.5008); primary LODO balanced‑accuracy 0.2146; does not beat raw gene; GCN strongest grouped‑transfer model at 0.309; see report for per‑method, per‑layer, spectral, and ablation tables. |
| `20260816_mp_mnca_phase1_all12` | ⚠️ **RETRACTED 2026-08-18 — LABEL-LEAKED, NOT A VALID RESULT** | artifacts **missing** (`outputs/mp_mnca/` absent) | Reported mean refined ARI 0.824 / median 0.896 was produced with `adata.obs["ground_truth"]` as the contrastive pseudo-labels, which build the negative mask — supervised training on the evaluation label. 12-section A/B (`leak_ab_all12_v1`): leaked **0.7451** mean → unsupervised **0.4836**, vs STAIG **0.5092**. Mean leak **+0.2615**. "DESTROYS STAIG" and "10/12 sections win" are withdrawn — unsupervised wins **5/12**. See `results/mp_mnca_label_leak_correction.md` |
| `leak_ab_all12_v1` | **verified 12-section label-leak A/B (2026-08-19)** | `outputs/mp_mnca/leak_ab_all12_v1/{leak_ab_all_sections.csv,summary.json,prepared/}` | Both arms per section, identical seed/config/epochs, only `pseudo_label_source` differs. Leak mean **+0.2615** ARI, median +0.3327, range [−0.2294, +0.4283], positive on 11/12. Negative on 151675 (−0.2294): removing same-layer negatives can starve the contrastive objective, so the leak is not purely additive. Absolute values are thread-config sensitive (`OMP_NUM_THREADS=4`); the within-run delta is the sound quantity |
| `20260816_mp_mnca_phase2_v1` | verified MP-MNCA Phase 2 (BYOL 3000→3000 reconstruction) | `outputs/mp_mnca/phase2_v1/` | 50 epochs, section 151507, refined ARI 0.278; reconstruction objective, weaker than contrastive |
| `20260816_mp_mnca_contrastive_v1` | verified MP-MNCA contrastive (50 ep, 1 section) | `outputs/mp_mnca/contrastive_v1/` | 50 epochs, section 151507, refined ARI 0.228; contrastive loss works |
| `20260816_mp_mnca_contrastive_v2` | verified MP-MNCA contrastive (100 ep, 4 sections) | `outputs/mp_mnca/contrastive_v2/` | 100 epochs, donor 1 sections, mean refined ARI 0.311; contrastive beats latent pred |
| `20260816_mp_mnca_contrastive_v3` | verified MP-MNCA contrastive (200 ep, 12 sections) | `outputs/mp_mnca/contrastive_v3/` | 200 epochs, all 12 sections, mean refined ARI 0.272, median 0.222; beats STAIG on 2 donor-2 sections |
| `20260816_mp_mnca_gene_cosine_v2` | verified MP-MNCA gene-cosine prior (200 ep, 12 sections) | `outputs/mp_mnca/gene_cosine_v2/` | 200 epochs, all 12 sections, mean refined ARI 0.243, median 0.222; gene_cosine similar to image_cosine |
| `20260816_mp_mnca_byol_v1` | verified MP-MNCA BYOL encoder (50 ep, 1 section) | `outputs/mp_mnca/byol_v1/` | 50 epochs, section 151507, refined ARI 0.234; random BYOL encoder, needs pre-training |
| `20260816_mp_mnca_report` | ⚠️ **RETRACTED 2026-08-18** — MP-MNCA vs STAIG comparison report | `results/mp_mnca_vs_staig.md` (banner added), `results/mp_mnca_label_leak_correction.md` | The "0.824 vs 0.507" comparison was supervised-vs-unsupervised and is withdrawn. The negative findings in the same report (latent prediction fails, 128-d variants fail) are unaffected — those arms scored 0.23–0.31 and were below STAIG regardless |

## Prior-model runs

| run | status | artifacts | result |
|---|---|---|---|
| `staig_paper_default_img_emb_seed0_all12_v3` | canonical adaptation | **artifacts missing as of 2026-08-18** (`outputs/prior_models/` was absent; regenerated as the row below) | 12 independent 400-epoch models; mean/median refined ARI 0.50693/0.51821, NMI 0.64353/0.67427 |
| `staig_baseline_seed0_all12_regen_v1` | **regenerated canonical STAIG baseline (2026-08-18)** | `outputs/prior_models/<run>/{config.json,summary.json,section_metrics.csv,checkpoints/,embeddings/}` | Same command/config as the row above (400 epochs, seed 0, temp 10.0, 40 pseudo-clusters). Mean/median refined ARI **0.50920/0.53740**, NMI **0.64550/0.67507** — closely reproduces the original. This is the live comparison bar for all `outputs/sparc_align/` runs |
| `spagcn_paper_default_seed100_all12_v1` | canonical SpaGCN reproduction | `outputs/prior_models/<run>/{config.json,summary.json,section_metrics.csv,checkpoints/,embeddings/}` | 12 independent 200-epoch GCN-DEC models; mean/median refined ARI 0.49711/0.48623, NMI 0.64393/0.63790; native DEC-refined mean/median ARI 0.425/0.412 |
| `graphst_paper_default_seed41_all12_v1` | canonical GraphST reproduction | `outputs/prior_models/<run>/{config.json,summary.json,section_metrics.csv,checkpoints/,embeddings/}` | 12 independent 600-epoch models; mean/median refined ARI 0.53238/0.52823, NMI 0.66317/0.65690; highest of the three reproduced published models |
| `mucost_paper_default_seed2023_all12_v1` | canonical MuCoST reproduction | `outputs/prior_models/<run>/{config.json,summary.json,section_metrics.csv,checkpoints/,embeddings/}` | 12 independent 1000-epoch models; mean/median refined ARI 0.52209/0.51859, NMI 0.65395/0.67062; native 25-neighbor refinement mean/median ARI 0.52472/0.52217 |
| `staig_official_151673_img_emb_seed39788_v1` | released-config sensitivity | config, embedding, metrics | official notebook hyperparameters with substituted `img_emb`; refined ARI/NMI 0.53694/0.69829 versus notebook 0.68639/0.73059 |
| `staig_multistage_gene1024_img_emb_seed0_all12_v1` | negative multistage S1 | complete 12-section run | mean/median refined ARI 0.20010/0.19672, NMI 0.31857/0.31947 |
| `staig_multistage_gene512_img_emb_seed0_all12_v1` | negative multistage S2 | complete 12-section run | mean/median refined ARI 0.21617/0.20898, NMI 0.31172/0.31342 |
| `staig_multistage_gene256_img_emb_seed0_all12_v1` | negative multistage S3 | complete 12-section run | mean/median refined ARI 0.19941/0.21495, NMI 0.27628/0.28885 |
| `staig_multistage_gene128_img_emb_seed0_all12_v2` | negative multistage S4 | complete 12-section run | mean/median refined ARI 0.22162/0.21885, NMI 0.27594/0.27542 |
| `staig_multistage_gbssa_r01_seed0_all12_v1` | negative multistage S5 | complete 12-section run | mean/median refined ARI 0.20720/0.20331, NMI 0.32120/0.32886 |
| `staig_multistage_gbssa_r011_seed0_all12_v1` | negative multistage S6 | complete 12-section collapsed sensitivity | mean/median refined ARI 0.19461/0.17796, NMI 0.31253/0.32037 |
| `staig_rich_gene_v2_all33538_stage4096_img_seed0_v1` | canonical rich-gene STAIG | complete 12-section run | mean/median refined ARI 0.48164/0.52293, NMI 0.64409/0.66682; essentially matches expression STAIG |
| `staig_rich_gene_v2_all33538_final128_img_seed0_v1` | rich bottleneck sensitivity | complete 12-section run | mean/median refined ARI 0.43867/0.44533, NMI 0.56895/0.59228 |
| `staig_rich_gene_v2_all33538_aligned128_seed0_v1` | rich aligned STAIG | complete 12-section run | mean/median refined ARI 0.49721/0.49365, NMI 0.65027/0.66117 |
| `staig_rich_gene_v3_all33538_imgk6_stage4096_img_seed0_v1` | negative end-to-end image-guided sensitivity | complete 12-section run | direct stage-4096 median refined ARI improved to 0.46428, but STAIG mean/median refined ARI fell to 0.45837/0.46331 |
| `staig_expression_aligned_bisector_guidance_seed0_all12_v1` | bisector edge-guidance sensitivity | complete 12-section run; guidance artifact `outputs/multimodal/aligned_bisector_staig_guidance_v1.npz` | 3,000-HVG Expression STAIG with aligned-bisector guidance; mean/median refined ARI 0.49069/0.51526, NMI 0.63808/0.67119; does not beat raw-image Expression STAIG |

Runs `staig_paper_default_img_emb_seed0_151673_v1`, `staig_paper_default_img_emb_seed0_all12_v1`, and `staig_paper_default_img_emb_seed0_all12_v2` are superseded implementation-audit artifacts. They remain immutable but are not canonical.

`staig_multistage_gene128_img_emb_seed0_all12_v1` is an invalid partial run: section 151507 training completed, but float32 tied-GMM covariance fitting failed before artifacts were saved. The corrected `_v2` run fits the same tied-GMM protocol in float64 for numerical stability.

`outputs/evaluation/20260815_multiscale_gene_staig_controls_seed0_v1/` is superseded: its summary pooled refined and unrefined clustering rows, and several `saga` probes reached their iteration limit. Corrected `_v2` separates refinement and uses convergent matched `lbfgs` probes.

## SPARC cross-modal alignment runs (`outputs/sparc_align/`)

SPARC (Nasiri-Sarvi et al., TMLR 3/2026) Global-TopK concept-aligned sparse
autoencoder over the DLPFC gene (3000-d `obsm['feat']`) and histology (128-d
`img_emb`) streams. Code `src/sparc_align/`. All runs unsupervised (contrastive
pseudo-labels from `KMeans(40)` on image PCA, never `ground_truth`). Evaluated
through the identical protocol as `src/mp_mnca` and STAIG. Full index and findings:
`outputs/sparc_align/README.md`.

| Run | Status | Artifacts | Result |
|---|---|---|---|
| `ablation_all12_v1` | verified 5-arm × 12-section ablation | `ablation_metrics.csv` (60 rows), `summary.json` | Best arm `e_spatial_plus_attention` mean/median refined ARI **0.3629/0.3411**, NMI 0.4780. Paper-faithful SPARC 0.2662. Spatially-aggregated TopK is the largest gain (+0.072 median) |
| `ablation_all12_v2_capacity` | verified 5-arm × 12-section capacity ablation | `ablation_metrics.csv` (60 rows), `summary.json` | Best `i_partitioned_mlp1024` **0.3163/0.3141**. No arm beat v1. Wider encoders *worsened* gene self-NMSE (0.924/0.929 vs affine 0.881) |
| `ablation_all12_v3_attention` | verified 4-arm × 12-section attention-weighting ablation | `ablation_metrics.csv` (48 rows), `summary.json` | Morphology-kernel `softmax(β log s_ij)` neighbour weighting replaces the uniform mean. **Best overall SPARC arm: `n_spatial_attention_plus_stage2` 0.3740/0.3681 ARI, 0.4823 NMI.** Beats the uniform-mean equivalents (+0.0163 for `k`, +0.0111 for `n`) — contradicting the single-section 151507 prediction of a large drop |
| `gonogo_151507_L1024_k32_v1` | feasibility probe, 1 section | `section_metrics.csv`, `summary.json`, latents, checkpoint | refined ARI 0.0166; established that cross-NMSE image→gene sits at 0.91 |
| `exploratory_151507` | single-section sweeps + leak A/B | sweep scripts, `leak_test.json` | Superseded by `src/sparc_align/sweeps.py`; retained as provenance |
| — | consolidated comparison | `CONSOLIDATED_12SECTION.csv` | All 10 SPARC arms + STAIG, mean/median ARI & NMI |

**Headline:** SPARC does not beat STAIG — best arm 0.3740 vs STAIG's 0.5092 and
unsupervised MP-MNCA's 0.4836 mean refined ARI. `cross_nmse_image_to_gene` never
moved below 0.92 in any of the 14 arms: morphology cannot predict expression in this
tissue, so the shared concept space is largely an image code while cortical layer
lives in the gene channel. Reported as a negative result.

## Gene BYOL GBSSA preparation

| run/artifact | status | result |
|---|---|---|
| `shared_hvgs_gbssa.json` | verified | 3,000 joint batch-aware HVGs |
| `spatial_knn_graph_gbssa.pkl` | verified | six within-section neighbors for 47,329 spots |
| `gene_byol_gbssa_seed0` | verified | early stop epoch 105, best val loss 0.1368; non-collapsed 128-d embeddings exported |
| `gene_byol_gbssa_seed1` | verified | early stop epoch 122, best val loss 0.1648; unaligned-bisector median ARI 0.14013 |
| `gene_byol_gbssa_seed2` | verified | early stop epoch 117, best val loss 0.1468; unaligned-bisector median ARI 0.13869 |
| `gene_byol_multistage_1024_512_256_128_seed0` | verified | best epoch 199, val total loss 0.443116; 47,329 finite 1024/512/256/128-d exports; final effective rank 10.7142 |
| `rich_gene_v1_hvg3000_seed0` | preliminary rich control | all 47,329 spots; 4096/2048/1024/512/256/128 exports; best epoch 72, stopped epoch 87 |
| `rich_gene_v1_hvg4096_seed0` | preliminary rich control | all 47,329 spots; 4096/2048/1024/512/256/128 exports; best epoch 78, stopped epoch 93 |
| `rich_gene_v2_all33538_seed0` | canonical rich gene encoder | all 33,538 genes -> 4096/1024/512/128; best epoch 27, val total loss 1.29656, stopped epoch 37; all 47,329 spots exported |
| `rich_gene_v3_all33538_imgk6_seed0` | negative image-guided K6 fine-tuning | initialized from canonical rich encoder; best epoch 4, manually stopped after epoch 10 validation divergence; finite 4096/1024/512/128 exports for all 47,329 spots |

## GBSSA cross-modal runs

All use `gene_byol_gbssa_seed0` against raw external `img_emb`, hidden/output dimensions 256/128, batch size 512, LR `3e-4`, and seed 0.

| run | ratio | status | interpretation |
|---|---:|---|---|
| `gbssa_img_r01_seed0` | 1 | verified baseline | standard CLIP, above-chance retrieval |
| `gbssa_img_r02_seed0` | 2 | rejected | alignment retrieval converged to chance |
| `gbssa_img_r05_seed0` | 5 | rejected | alignment retrieval converged to chance |
| `gbssa_img_r10_seed0` | 10 | rejected | alignment retrieval converged to chance |
| `multistage_gene128_gbssa_r01_seed0_v2` | 1 | verified multistage baseline | final gene-128 against `img_emb`; best epoch 34, val loss 5.761239; accepted for STAIG S5 |
| `multistage_gene128_gbssa_r011_seed0_v2` | 1.1 | negative multistage sensitivity | best epoch 51, val loss 5.869197; contracted effective rank; retained only for STAIG S6 |
| `rich_gene_v2_all33538_gbssa_r01_seed0` | 1 | canonical reconstruction-regularized alignment | all-gene final-128 against `img_emb`; best epoch 40, stopped epoch 55; median R@10 0.07209 and cosine gap 0.11104 |

`rich_gene_v1_hvg4096_gbssa_r01_seed0` is an incomplete exploratory alignment stopped after epoch 4 during scope clarification. It has no prediction export and must not be evaluated. The canonical replacement is `rich_gene_v2_all33538_gbssa_r01_seed0`.

The corresponding multistage runs without the `_v2` suffix are invalid and must not be consumed: their prediction NPZ files contain section IDs truncated to `"1"`. The corrected loader preserves full section strings, and the immutable `_v2` artifacts pass exact `(section_id, barcode)` manifest validation.

## Positive-neighborhood cross-modal runs

| run | status | interpretation |
|---|---|---|
| `positive_neighborhood_img_k6r5_seed0` | rejected | raw-output variance regularization allowed angular contraction |
| `positive_neighborhood_angular_img_k6r5_seed0` | rejected | angular diversity preserved but alignment weight was too weak |
| `positive_neighborhood_vicreg_img_k6r5_seed0` | rejected | stronger positive alignment contracted outputs to effective rank near 2 |
| `graph_positive_procrustes_img_k6r5_seed0` | control | collapse-free alignment preserved geometry but bisector did not improve |
| `graph_pooled_procrustes_img_k6r5_seed0` | retained candidate | 72,089 pruned edges; graph pooling plus orthogonal positive alignment; concatenation is strongest |
| `spatial_multimodal_contrastive_img_k6r5_seed0` | retained candidate | 30-epoch unified gene-gene, image-image, and cross-modal graph contrast; strongest current domain-discovery result |

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
