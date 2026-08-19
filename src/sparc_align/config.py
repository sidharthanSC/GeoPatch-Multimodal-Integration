"""Configuration for SPARC cross-modal alignment on DLPFC sections."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class SparcConfig:
    """SPARC architecture, training, and evaluation hyperparameters.

    Defaults follow the SPARC paper (Appendix A.2) except where the DLPFC scale
    forces a documented deviation; every such deviation is flagged in the field
    comment with "DEVIATION".
    """

    # --- Streams -------------------------------------------------------------
    # Gene stream is the 3000-d ``obsm['feat']`` HVG expression that
    # ``src.mp_mnca.train_phase1`` consumes; image stream is the raw 128-d
    # ``obsm['img_emb']``. Both are per-feature standardized before encoding --
    # required here (unlike the paper's already-comparable ViT features) because
    # img_emb spans roughly [-131, 189] while feat spans [0, 10], so an
    # unstandardized Global TopK would be decided entirely by the image stream.
    gene_dim: int = 3000
    image_dim: int = 128
    standardize_streams: bool = True

    # --- Sparse dictionary ---------------------------------------------------
    # DEVIATION: paper uses L=8192, k=64 over 1.7M Open Images samples. A DLPFC
    # section has ~3.5-4.2k spots, so an 8192-latent dictionary is wildly
    # over-parameterized (most latents would never fire). Swept over
    # {256, 512, 1024} x {16, 32, 64}.
    n_latents: int = 1024
    k_active: int = 32
    global_topk: bool = True  # False -> per-stream Local TopK ablation

    # --- Support-selection mechanism ----------------------------------------
    # How the shared active set is chosen. "global_sum" is Eq. 2-3 exactly.
    #
    # The paper justifies the shared support by a trade (Section 5): Global TopK
    # costs 0.030-0.060 self-reconstruction NMSE but buys cross-reconstruction
    # gains "2-3x larger". On DLPFC that trade is inverted -- cross-NMSE
    # image->gene stays at 0.92-0.93 under every configuration tried, so the
    # self-reconstruction penalty is paid for a benefit that does not exist. The
    # modes below relax the constraint in three increasingly structural ways.
    #
    # - "global_sum":  I = TopK(sum_s h_s, k).                        [paper]
    # - "quota":       reserve floor capacity per stream, so no single stream can
    #                  monopolize the support; remainder filled from the sum.
    # - "rank_fusion": aggregate reciprocal ranks rather than raw logits, making
    #                  selection invariant to per-stream scale AND distribution
    #                  shape. (Plain RMS normalization equalizes only scale and
    #                  measurably hurt: median ARI 0.2417 -> 0.1968.)
    # - "partitioned": split the dictionary into a shared block (Global TopK) plus
    #                  one private block per stream (Local TopK), so concepts with
    #                  no cross-modal counterpart stop competing for shared atoms.
    topk_mode: Literal["global_sum", "quota", "rank_fusion", "partitioned"] = "global_sum"

    # "quota": fraction of k reserved for guaranteed per-stream slots (split evenly).
    quota_fraction: float = 0.5
    # "rank_fusion": reciprocal-rank-fusion damping constant; 60 is the IR default.
    rank_fusion_constant: float = 60.0
    # "partitioned": fraction of L that is shared, and fraction of k spent private.
    shared_fraction: float = 0.5
    private_k_fraction: float = 0.5

    # --- Extensions beyond the paper (all default to paper-faithful OFF) ------
    # (1) Shallow nonlinear encoder. The paper keeps E_s affine and explicitly
    # leaves "shallow nonlinear mappings (e.g., two-layer MLPs)" to future work
    # (Section 3.2). Measured here: a linear encoder leaves self-NMSE on the
    # 3000-d gene stream at 0.86-0.90, i.e. it cannot compress sparse log1p
    # expression into k active latents. Decoders stay affine either way, so
    # dictionary atoms remain directions in input space and stay interpretable.
    encoder_hidden_dim: int | None = None

    # (2) Per-stream logit normalization before the Global TopK sum. Eq. 2 sums
    # raw h^s; with d_gene=3000 vs d_image=128 the larger-magnitude stream
    # decides the shared support alone. Measured symptom: cross-NMSE 0.04
    # gene->image but 0.91 image->gene, i.e. the shared space is an image code.
    normalize_stream_logits: bool = False

    # (4) Per-stream self-reconstruction weight. Equal weighting under-serves the
    # harder (gene) stream. Keys must match the stream names used in data.py.
    gene_loss_weight: float = 1.0
    image_loss_weight: float = 1.0

    # (3) Spatially-aggregated Global TopK -- the extension with no counterpart in
    # the paper, which treats samples as i.i.d. Measured on DLPFC: a spot's k=6
    # spatial neighbours share its cortical layer 92.3% of the time, so smoothing
    # the selection logits over the neighbourhood before TopK makes the shared
    # concept support layer-coherent rather than per-spot noisy. The support is
    # smoothed; z_s still gathers from the spot's OWN logits, so per-spot
    # reconstruction fidelity is untouched.
    #   h_select_i = (1 - w) * h_i + w * mean_{j in N(i)} h_j
    # 0.0 reproduces the paper exactly.
    spatial_topk_weight: float = 0.0

    # (5) Morphology-weighted spatial aggregation ("attention-style" weighting).
    # With spatial_attention=False the neighbourhood term above is a UNIFORM mean
    # over the k=6 neighbours, which treats every neighbour as equally informative
    # and therefore blurs supports across layer boundaries -- exactly where the
    # 92.3% same-layer agreement breaks down. Enabling this replaces the uniform
    # mean with MP-MNCA's morphology kernel, reusing the image-PCA cosine prior
    # s_ij = (cos + 1)/2 that its cross-attention puts on the attention logits:
    #
    #   w_ij = softmax_j( beta * log s_ij ),   h_nbr_i = sum_j w_ij * h_j
    #
    # so morphologically dissimilar neighbours are down-weighted rather than
    # averaged in. beta is learnable by default, matching MP-MNCA's
    # learn_morphology_weight=True.
    spatial_attention: bool = False
    morphology_prior_weight: float = 1.0
    learn_morphology_weight: bool = True

    # --- Loss ----------------------------------------------------------------
    cross_loss_weight: float = 1.0  # lambda; 0.0 ablates cross-reconstruction
    aux_k: int = 32  # k' for the AuxK dead-latent revival loss
    aux_loss_weight: float = 0.03125  # gamma = 1/32, per paper

    # DEVIATION: paper marks a latent dead after 1000 consecutive inactive steps.
    # One DLPFC section at batch 256 gives ~17 steps/epoch, so a 1000-step
    # threshold could never fire within a single-section run. Scaled to 200.
    dead_steps_threshold: int = 200
    dead_activation_threshold: float = 1e-3
    dead_reinit_std: float = 0.01

    # --- Training ------------------------------------------------------------
    epochs: int = 50
    batch_size: int = 256
    learning_rate: float = 1e-4
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8
    dtype: str = "float32"
    seed: int = 42

    # --- Downstream clustering (identical to src/mp_mnca/evaluate.py) --------
    # Which vector goes into PCA -> tied GMM.
    cluster_input: Literal["sum", "gene", "image", "concat", "support"] = "sum"
    pca_dim: int = 128
    n_clusters: int | None = None  # None -> count of distinct ground_truth labels
    refinement_neighbors: int = 15
    gmm_seed: int = 2023

    # --- Optional downstream MP-MNCA cross-attention stage -------------------
    use_cross_attention_stage: bool = False
    n_neighbors: int = 6
    num_heads: int = 8
    image_pca_dim: int = 16  # morphology-prior input, matches mp_mnca phase 1
    image_pseudo_clusters: int = 40  # KMeans on image PCA -> STAIG pseudo-labels
    temperature: float = 10.0
    mask_rate: float = 0.1
    attention_epochs: int = 20

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
