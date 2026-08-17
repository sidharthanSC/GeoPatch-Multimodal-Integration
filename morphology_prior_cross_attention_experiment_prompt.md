# Execution Prompt: Morphology-Prior Masked Neighbour Cross-Attention vs. Existing STAIG Runs

## Role and objective

Act as a careful ML research engineer working inside the current spatial-transcriptomics repository. Implement and evaluate a new spot-representation method that replaces STAIG's image-guided **binary edge-dropping augmentation** with **continuous, morphology-prior, local cross-attention**, while using structured masking of neighbouring gene information.

The primary objective is to test whether learned, continuous neighbour weighting produces better and more robust spot embeddings than the STAIG variants that have already been run in this repository.

Do not merely propose code or give implementation advice. Inspect the repository, implement the method, run the feasible experiments, and save the final comparison to a Markdown report.

## Important conceptual corrections

Use the following interpretation throughout the implementation and report:

1. Dropping an edge \((i,j)\) in STAIG does **not** delete or mask the entire neighbour spot \(j\). It only removes the message path between \(i\) and \(j\) in that augmented graph view.
2. STAIG already combines image-guided edge dropping with gene-feature masking. Therefore, the proposed method is not simply "STAIG with stronger masking."
3. Local cross-attention over coordinate-selected neighbours is still a graph-based message-passing method—more precisely, a local graph transformer. Do not claim that it is categorically "not a GNN."
4. The main methodological change is from stochastic binary edge retention to continuous learned neighbour contribution, with image morphology used as an explicit prior.
5. This is not, by itself, a sparse autoencoder. It becomes masked predictive representation learning only when the model must predict a clean target from corrupted inputs. Sparsity should be claimed only if an explicit sparsity penalty or sparse activation mechanism is implemented.

## Repository-first requirements

Before modifying code:

1. Inspect the repository structure, configuration files, training scripts, dataset loaders, saved checkpoints, logs, metrics, and result tables.
2. Find every completed STAIG or STAIG-like run relevant to this dataset. Record:
   - experiment/run name;
   - configuration and random seed;
   - dataset sections/donors;
   - split protocol;
   - preprocessing and HVG selection;
   - clustering procedure;
   - reported metrics;
   - checkpoint/result path.
3. Identify the strongest existing baseline and the exact evaluation protocol used for it.
4. Reuse the existing data pipeline, splits, preprocessing, clustering code, metric code, and seeds wherever possible.
5. Never invent missing baseline values. If a previous run cannot be verified from repository artifacts, mark it `not found` in the report.
6. Do not silently overwrite existing checkpoints, configurations, logs, or result files. Create a new experiment namespace/directory.
7. Check the current git status and preserve all unrelated user changes.

## Proposed model

Call the method **Morphology-Prior Masked Neighbour Cross-Attention (MP-MNCA)** unless the repository already contains a conflicting name.

### 1. Candidate graph

For every tissue section, construct or reuse the coordinate-based spatial neighbourhood graph. For centre spot \(i\), let \(\mathcal N(i)\) be its coordinate-selected neighbours.

For multiple sections, preserve the existing block-diagonal topology. Do not create cross-section spatial edges unless an existing experiment explicitly evaluates that as a separate ablation.

### 2. Input representations

Let:

- \(x_i\): preprocessed gene-expression vector for spot \(i\);
- \(m_i\): image embedding for spot \(i\);
- \(p_i\): spatial coordinate;
- \(h_i=E_g(x_i)\): gene representation;
- \(\tilde{x}_j\): corrupted/masked expression of neighbour \(j\);
- \(\tilde h_j=E_g(\tilde{x}_j)\): masked-neighbour representation.

Use the same image embeddings as the comparable STAIG run unless a change is necessary and explicitly documented.

### 3. Structured gene masking

Generate two independently corrupted views during training. Apply masking only to the online/input branch.

Prefer biologically meaningful masking over isolated scalar masking. Implement the best feasible option supported by the repository:

1. gene-module/pathway masking, if module definitions are already available;
2. contiguous latent/gene-token block masking;
3. whole-gene masking within a sampled neighbourhood;
4. independent scalar feature masking only as a fallback baseline.

Make the masking ratio configurable. At minimum test a moderate default and one lower/higher value if compute permits.

### 4. Morphology-prior local cross-attention

For each centre spot \(i\), use its representation as the query and masked neighbour representations as keys and values:

\[
q_i=W_qh_i,\qquad
k_j=W_k\tilde h_j,\qquad
v_j=W_v\tilde h_j.
\]

Calculate image similarity only for existing coordinate-based candidate edges. Use a stable normalized similarity such as cosine similarity mapped to a non-negative range, or an RBF kernel over normalized image-embedding distance.

Define the attention logit as:

\[
e_{ij}=
\frac{q_i^\top k_j}{\sqrt{d_h}}
+\beta\log(s^{\mathrm{img}}_{ij}+\epsilon)
+\gamma\,\phi(p_j-p_i),
\qquad j\in\mathcal N(i),
\]

where:

- the first term is learned molecular compatibility;
- \(s^{\mathrm{img}}_{ij}\) is the morphology prior;
- \(\phi(p_j-p_i)\) is an optional learned relative-position bias;
- \(\beta\) and \(\gamma\) are configurable or learnable scalars.

Then compute:

\[
\alpha_{ij}=\operatorname{softmax}_{j\in\mathcal N(i)}(e_{ij}),
\qquad
c_i=\sum_{j\in\mathcal N(i)}\alpha_{ij}v_j,
\]

and produce the final contextual representation using a residual connection:

\[
z_i=\operatorname{LayerNorm}\left(h_i+
\operatorname{MLP}([h_i\Vert c_i])\right).
\]

Use multi-head attention if it fits the existing architecture and compute budget. Preserve sparse/local computation; do not construct dense all-pairs attention over all spots.

### 5. Leakage prevention

This requirement is mandatory:

- Do not compute gene-attention keys, values, similarities, or gates from the complete unmasked gene vector when those same values are prediction targets.
- Keys and values on the online branch must be produced from masked/visible information only.
- A clean unmasked representation may be used only on a stop-gradient target branch.
- Fit normalization, PCA, HVG selection, gene modules, and any learned preprocessing using only the training data allowed by the existing evaluation protocol.
- Audit code explicitly for target-label and donor/test leakage.

## Training objective

### Preferred objective: EMA latent prediction

Use an online encoder with parameters \(\theta\) and an EMA target encoder with parameters \(\bar\theta\). The target encoder receives the clean, unmasked input and is updated without gradients:

\[
\bar\theta\leftarrow\tau\bar\theta+(1-\tau)\theta.
\]

For each spot, predict its clean target latent from a corrupted local view:

\[
\mathcal L_{\mathrm{latent}}=
\frac{1}{N}\sum_i
\left\|
P_\theta(z_i^{\mathrm{masked}})-
\operatorname{sg}(z_i^{\mathrm{target}})
\right\|_2^2.
\]

Apply appropriate normalization to prevent scale-based collapse. If the repository already has a tested BYOL/VICReg/JEPA-style implementation, reuse its stable projector, predictor, EMA, and variance controls.

### Anti-collapse regularization

Add the minimum necessary anti-collapse mechanism. Prefer an existing tested implementation, such as:

- variance and covariance regularization;
- normalized BYOL-style predictor/target loss;
- contrastive InfoNCE with well-defined negatives.

If variance/covariance regularization is used:

\[
\mathcal L =
\mathcal L_{\mathrm{latent}}
+\lambda_v\mathcal L_{\mathrm{var}}
+\lambda_c\mathcal L_{\mathrm{cov}}.
\]

Log all loss components separately.

### Optional auxiliary reconstruction

Only if useful and computationally feasible, add a low-weight auxiliary loss for masked gene reconstruction. Do not make raw count reconstruction the sole objective unless required for an ablation. Clearly distinguish latent prediction from gene-value reconstruction in the report.

## Required experimental matrix

Use identical data splits, seeds, preprocessing, clustering, label-count assumptions, and metric implementations for all comparable rows.

Run or recover the following experiments:

| ID | Experiment | Edge treatment | Image use | Gene corruption | Objective |
|---|---|---|---|---|---|
| B0 | Best verified prior STAIG run | Existing STAIG edge drop | Edge-drop probability | Existing | Existing STAIG loss |
| B1 | STAIG without image guidance, if already available or feasible | Fixed/random drop | None for edges | Existing | Existing STAIG loss |
| A0 | MP-MNCA gene-only attention | No image prior | None | Structured mask | Latent prediction |
| A1 | MP-MNCA image-prior attention | Continuous local attention | Logit prior | Structured mask | Latent prediction |
| A2 | A1 without structured masking | Continuous local attention | Logit prior | None/minimal | Latent prediction |
| A3 | A1 with small edge dropout | Attention plus mild stochastic drop | Logit prior | Structured mask | Latent prediction |
| A4 | A1 with direct post-softmax image×gene weighting | Multiplicative weighting | Multiplication | Structured mask | Latent prediction |

A4 is an ablation, not the preferred design. It tests whether additive logit fusion is better calibrated than multiplying independently normalized weights.

If compute is limited, prioritize `B0`, `A0`, `A1`, and `A3`. Clearly state which planned runs were not completed and why.

## Fair evaluation

Match the strongest verified STAIG protocol exactly. Where applicable, report:

- ARI;
- NMI;
- Hungarian-matched clustering accuracy;
- purity;
- silhouette score computed without labels where appropriate;
- supervised spot accuracy, balanced accuracy, and macro-F1 only if the existing pipeline contains a supervised downstream evaluation;
- mean and standard deviation over the same seeds used by the baseline;
- runtime, peak GPU memory, and parameter count if measurable.

For DLPFC, report both per-section values and the aggregate statistic used by the existing baseline. Do not pool spots across sections and report a single clustering score unless that is explicitly part of the verified protocol. If donor-level generalization is present, preserve the exact leave-one-donor-out split and report every held-out donor separately.

Do not tune hyperparameters using test labels. If the number of clusters is supplied from known annotations for compatibility with STAIG evaluation, state this explicitly—it is not fully label-free model selection.

## Diagnostics that must be saved

For each new method, save:

1. training and validation loss curves;
2. attention entropy by layer/head;
3. distribution of image-prior values and final attention weights;
4. correlation between image prior and learned attention;
5. average attention assigned to same-domain versus different-domain neighbours, using labels only after training for analysis;
6. per-section clustering maps if the repository already supports spatial plots;
7. embedding visualizations generated consistently across methods;
8. collapse diagnostics: per-dimension standard deviation, effective rank, and covariance off-diagonal magnitude;
9. configuration, seed, checkpoint, and machine/runtime metadata.

Interpret high image-attention correlation carefully: it may indicate successful prior use or that gene-dependent attention contributes little. Compare A0 and A1 before drawing conclusions.

## Implementation quality requirements

- Integrate with the repository's configuration system instead of hard-coding experiment values.
- Add clear type/shape assertions at module boundaries.
- Keep the attention computation sparse over neighbourhood edges.
- Add unit tests for masking, neighbourhood isolation across block-diagonal sections, attention normalization, EMA updates, and leakage prevention.
- Add a small smoke test that completes one forward/backward pass.
- Use deterministic seeds where supported.
- Run formatting/linting and relevant tests already configured by the repository.
- Do not refactor unrelated code.

## Final result report

Create or update the following file after the experiments finish:

`results/mp_mnca_vs_staig.md`

The report must contain:

### 1. Executive conclusion

State whether MP-MNCA outperformed the best verified STAIG run, on which metrics, and whether the difference is consistent across sections/seeds. Do not call the method superior based on a single favourable run.

### 2. Verified baseline provenance

List the paths to the original STAIG configurations, logs, checkpoints, and result files from which every baseline number was obtained.

### 3. Method summary

Explain the candidate coordinate graph, structured masking, morphology-prior attention, EMA target, loss, and inference procedure. Explicitly state that it remains a graph-based local-attention model.

### 4. Main comparison table

Use a table with one row per run and columns for:

- method;
- seed;
- split/section/donor;
- ARI;
- NMI;
- matched accuracy;
- purity;
- macro-F1 where applicable;
- runtime;
- parameter count;
- checkpoint/config path.

Include aggregate mean ± standard deviation in separate summary rows.

### 5. Ablation table

Compare image prior, masking, mild edge dropout, additive logit fusion, and multiplicative fusion.

### 6. Robustness and collapse diagnostics

Report masking-ratio sensitivity, attention entropy, effective rank, and any failed/collapsed runs.

### 7. Interpretation

Answer all of the following:

- Did continuous attention improve over binary edge dropping?
- Did image priors add information beyond gene-only attention?
- Was mild edge dropout still useful for robustness?
- Did structured masking help or merely make optimization harder?
- Did the model over-rely on neighbours?
- Were improvements consistent across tissue sections and donors?

### 8. Limitations and next experiment

Document incomplete runs, fairness limitations, label-count assumptions, compute limitations, and the single most informative next experiment.

### 9. Reproduction commands

Provide exact commands for every reported new run and for regenerating the tables/plots.

## Individual-spot representation extension

The contextual MP-MNCA embedding still depends on neighbouring spots. Do not claim neighbour-independent inference.

If the repository's research goal requires a standalone representation for each spot, implement this only as a clearly separated extension:

\[
z_i^{\mathrm{teacher}}=f(x_i,\mathcal N(i),m,p),
\qquad
z_i^{\mathrm{student}}=g(x_i,m_i),
\]

\[
\mathcal L_{\mathrm{distill}}=
\left\|z_i^{\mathrm{student}}-
\operatorname{sg}(z_i^{\mathrm{teacher}})\right\|_2^2.
\]

Evaluate the spot-only student separately from the contextual teacher. Do not mix their results in one method row.

## Completion criteria

The task is complete only when:

1. the existing STAIG results have been located and verified or explicitly marked missing;
2. the new method and prioritized ablations are implemented;
3. unit tests and a smoke test pass;
4. feasible full experiments have completed;
5. artifacts and configurations are saved without overwriting prior runs;
6. `results/mp_mnca_vs_staig.md` contains numerical comparisons, provenance, interpretation, limitations, and reproduction commands;
7. any unexecuted experiment is clearly marked `not run`, never filled with an estimate.

Begin by presenting a brief repository audit and execution plan, then proceed with implementation and experiments without waiting for confirmation unless a genuinely blocking ambiguity or destructive action is encountered.
