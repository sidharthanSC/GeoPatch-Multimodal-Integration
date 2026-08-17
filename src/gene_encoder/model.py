"""Gene-expression encoder wrapped in the ``byol-pytorch`` framework.

Uses the community `byol-pytorch <https://github.com/lucidrains/byol-pytorch>`_
package (``pip install byol-pytorch``) for the actual BYOL machinery (online/target
branches, projector/predictor, EMA update, loss) rather than a hand-rolled
reimplementation -- only the encoder network (:class:`GeneEncoder`, replacing the
package's usual ResNet) and the two augmented views are project-specific.

**Why ``x`` is spot row-indices, not raw expression, when calling the wrapped model**:
``byol_pytorch.BYOL.__init__`` unconditionally probes itself with
``torch.randn(2, 3, image_size, image_size)`` (a hardcoded *image-shaped* mock tensor)
to lazily size its internal projector -- this happens even when ``augment_fn``/
``augment_fn2`` are fully overridden, because the mock tensor is what gets passed
*into* those augment functions during that probe. Since our real augmentations
(``masked_corruption_view``, and looking up a precomputed spatially-smoothed vector)
both need to know *which spots* are in the batch -- not just be handed a raw tensor --
the augment functions here treat their input as a 1-D ``LongTensor`` of row-indices into
externally-held expression matrices, and fall back to a harmless dummy output whenever
they're called with something that isn't that shape/dtype (i.e. the library's one-time
mock probe). This keeps the encoder itself (:class:`GeneEncoder`) completely ordinary --
it only ever sees real ``(batch, n_hvg)`` float tensors, never the index trick.

``hidden_layer=-1`` is used when constructing the ``BYOL`` wrapper so it takes the
encoder's own output directly (no forward-hook interception needed, since
:class:`GeneEncoder` already returns exactly the ``embedding_dim``-d representation to
export as ``gene_emb``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import torch
from torch import nn

from byol_pytorch import BYOL


@dataclass(frozen=True)
class GeneEncoderConfig:
    """Architecture and EMA hyperparameters.

    Attributes
    ----------
    input_dim:
        Number of genes in the shared-HVG expression vector (see
        ``src/gene_encoder/hvg_selection.py``).
    hidden_dims:
        Ordered hidden widths of the encoder MLP. Each stage is exported for the
        multi-scale STAIG study, while BYOL still optimizes only the final embedding.
    embedding_dim:
        Encoder output dimensionality -- the actual ``gene_emb`` that gets stored.
        Matches ``img_emb``/``proj_emb`` (128) so the later cross-modal InfoNCE step
        doesn't need a dimension-reconciling projection.
    projection_dim:
        Output dimensionality of ``byol_pytorch``'s internal projector/predictor.
    projection_hidden_dim:
        Hidden width of ``byol_pytorch``'s internal projector/predictor MLPs.
    dropout:
        Dropout probability inside :class:`GeneEncoder`.
    ema_decay:
        EMA coefficient for the target encoder (``byol_pytorch``'s
        ``moving_average_decay``).
    """

    input_dim: int
    hidden_dims: Tuple[int, ...] = (1024, 512, 256)
    embedding_dim: int = 128
    projection_dim: int = 128
    projection_hidden_dim: int = 256
    dropout: float = 0.1
    ema_decay: float = 0.996
    activation: str = "relu"
    expression_residual: bool = False
    freeze_expression_residual: bool = True


class GeneEncoder(nn.Module):
    """Named hidden blocks followed by one final embedding projection.

    The network handed to ``byol_pytorch.BYOL`` in place of the package's usual
    ResNet -- sized for gene-expression vectors, not images.
    """

    def __init__(self, config: GeneEncoderConfig) -> None:
        super().__init__()
        if not config.hidden_dims or any(width <= 0 for width in config.hidden_dims):
            raise ValueError("hidden_dims must contain positive dimensions")
        self.config = config
        dimensions = (config.input_dim, *config.hidden_dims)
        if config.activation not in {"relu", "gelu"}:
            raise ValueError("activation must be 'relu' or 'gelu'")
        activation = nn.ReLU if config.activation == "relu" else nn.GELU
        self.hidden_blocks = nn.ModuleList(
            nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.LayerNorm(output_dim),
                activation(),
                nn.Dropout(config.dropout),
            )
            for input_dim, output_dim in zip(dimensions[:-1], dimensions[1:])
        )
        self.embedding = nn.Linear(config.hidden_dims[-1], config.embedding_dim)
        self.expression_residual = (
            nn.Linear(config.input_dim, config.embedding_dim, bias=False)
            if config.expression_residual
            else None
        )
        self.residual_norm = (
            nn.LayerNorm(config.embedding_dim, elementwise_affine=False)
            if config.expression_residual
            else None
        )

    def initialize_expression_residual(self, components: torch.Tensor) -> None:
        """Initialize the optional input skip from fixed SVD/PCA components."""
        if self.expression_residual is None:
            raise ValueError("expression_residual is disabled")
        expected = self.expression_residual.weight.shape
        if components.shape != expected:
            raise ValueError(f"Expected residual components {expected}, got {components.shape}")
        with torch.no_grad():
            self.expression_residual.weight.copy_(components)
        self.expression_residual.weight.requires_grad_(
            not self.config.freeze_expression_residual
        )

    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return every post-block hidden representation and the final embedding."""
        expression = x
        outputs: Dict[str, torch.Tensor] = {}
        for index, block in enumerate(self.hidden_blocks, start=1):
            x = block(x)
            outputs[f"encoder_stage_{index}"] = x
        embedding = self.embedding(x)
        if self.expression_residual is not None:
            residual = self.residual_norm(self.expression_residual(expression))
            embedding = (embedding + residual) / 2**0.5
        outputs["embedding"] = embedding
        return outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # BYOL(hidden_layer=-1) requires a single tensor, not a feature dictionary.
        return self.forward_features(x)["embedding"]


def variance_regularization(embeddings: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    """VICReg-style variance term: penalize any embedding dimension whose per-batch
    std falls below ``gamma``.

    Added on top of the BYOL loss because BYOL's own invariance-only objective turned
    out (confirmed empirically -- see ``src/gene_encoder/augmentations.py``'s
    module docstring) to let the encoder collapse to a near-constant output for this
    data: the BYOL loss alone only rewards the two views agreeing, never penalizes
    different *spots'* embeddings agreeing with each other too. This term does that
    directly, computed on :class:`GeneEncoder`'s own raw output (not the projector's),
    since that's what actually gets exported as ``gene_emb``.
    """
    std = torch.sqrt(embeddings.var(dim=0) + eps)
    return torch.relu(gamma - std).mean()


def covariance_regularization(
    embeddings: torch.Tensor,
    max_dimensions: int = 256,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Mean squared off-diagonal correlation on a bounded coordinate subset."""
    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        raise ValueError("covariance_regularization requires at least two 2-D rows")
    if max_dimensions <= 1:
        raise ValueError("max_dimensions must be greater than one")
    if embeddings.shape[1] > max_dimensions:
        # Evenly spaced coordinates are deterministic and cover the entire wide stage.
        indices = torch.linspace(
            0,
            embeddings.shape[1] - 1,
            max_dimensions,
            device=embeddings.device,
        ).long()
        embeddings = embeddings[:, indices]
    embeddings = embeddings.float()
    centered = embeddings - embeddings.mean(dim=0)
    standardized = centered / torch.sqrt(centered.var(dim=0, unbiased=False) + eps)
    correlation = standardized.T @ standardized / standardized.shape[0]
    off_diagonal = correlation - torch.diag_embed(correlation.diagonal())
    return off_diagonal.square().sum() / (
        correlation.shape[0] * max(correlation.shape[0] - 1, 1)
    )


def cosine_geometry_regularization(
    embeddings: torch.Tensor, geometry_target: torch.Tensor
) -> torch.Tensor:
    """Preserve batch-local pairwise cosine geometry from expression SVD/PCA."""
    if embeddings.ndim != 2 or geometry_target.ndim != 2:
        raise ValueError("embeddings and geometry_target must be 2-D")
    if embeddings.shape[0] != geometry_target.shape[0] or embeddings.shape[0] < 2:
        raise ValueError("embeddings and geometry_target need matching batches of size >= 2")
    embeddings = embeddings.float()
    geometry_target = geometry_target.float()
    embedding_similarity = nn.functional.normalize(embeddings, dim=1) @ nn.functional.normalize(
        embeddings, dim=1
    ).T
    target_similarity = nn.functional.normalize(
        geometry_target, dim=1
    ) @ nn.functional.normalize(geometry_target, dim=1).T
    mask = ~torch.eye(embeddings.shape[0], dtype=torch.bool, device=embeddings.device)
    return nn.functional.mse_loss(embedding_similarity[mask], target_similarity[mask])


class GeneReconstructionDecoder(nn.Module):
    """Training-only linear decoder from the exported embedding to expression."""

    def __init__(self, embedding_dim: int, expression_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(embedding_dim, expression_dim)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.linear(embeddings)


def masked_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    nonzero_weight: float = 5.0,
) -> torch.Tensor:
    """Weighted Huber reconstruction evaluated only at masked expression loci."""
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError("prediction, target, and mask must have identical shapes")
    if nonzero_weight < 0:
        raise ValueError("nonzero_weight must be non-negative")
    selected = mask.bool()
    if not selected.any():
        return prediction.new_zeros(())
    error = nn.functional.smooth_l1_loss(prediction, target, reduction="none")
    weights = 1.0 + nonzero_weight * (target > 0).to(target.dtype)
    return (error[selected] * weights[selected]).sum() / weights[selected].sum().clamp_min(1.0)


def make_view_augmentations(
    raw_expression: torch.Tensor,
    smoothed_expression: torch.Tensor,
    gene_min: torch.Tensor,
    gene_max: torch.Tensor,
    mask_rate: float,
    noise_std_fraction: float = 0.0,
) -> Tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]:
    """Build the ``augment_fn``/``augment_fn2`` closures for ``byol_pytorch.BYOL``.

    Parameters
    ----------
    raw_expression, smoothed_expression:
        ``(n_spots, n_hvg)`` tensors, row order shared with each other (row ``i`` is
        the same spot in both) -- see ``src/gene_encoder/train.py`` for how these are
        assembled from ``src/gene_encoder/augmentations.py``'s per-section functions.
    gene_min, gene_max:
        ``(n_hvg,)`` per-gene empirical range, for the masked-corruption view.
    mask_rate:
        Fraction of genes corrupted per call in the masked view.
    noise_std_fraction:
        Std of additive Gaussian noise (as a fraction of each gene's own range) applied
        across every gene in the masked view, on top of masking. See
        ``src/gene_encoder/augmentations.py``'s ``masked_corruption_view`` docstring
        for why this exists (without it, the two views collapsed the trained encoder's
        embeddings to near-uniformity, confirmed empirically).

    Returns
    -------
    ``(augment_view_a, augment_view_b)``: ``augment_view_a`` masks ``raw_expression``,
    ``augment_view_b`` looks up ``smoothed_expression`` -- both expect a 1-D
    ``LongTensor`` of row-indices as input (see module docstring for why).
    """
    n_hvg = raw_expression.shape[1]
    gene_range = gene_max - gene_min

    def _is_real_index_batch(x: torch.Tensor) -> bool:
        return x.dtype == torch.long and x.ndim == 1

    def augment_view_a(x: torch.Tensor) -> torch.Tensor:
        if not _is_real_index_batch(x):
            return torch.randn(x.shape[0], n_hvg, device=raw_expression.device)
        batch = raw_expression[x]
        mask = torch.rand_like(batch) < mask_rate
        random_fill = gene_min + torch.rand_like(batch) * gene_range
        corrupted = torch.where(mask, random_fill, batch)
        if noise_std_fraction > 0:
            corrupted = corrupted + torch.randn_like(corrupted) * (noise_std_fraction * gene_range)
        return corrupted

    def augment_view_b(x: torch.Tensor) -> torch.Tensor:
        if not _is_real_index_batch(x):
            return torch.randn(x.shape[0], n_hvg, device=smoothed_expression.device)
        return smoothed_expression[x]

    return augment_view_a, augment_view_b


def build_byol_learner(
    encoder: GeneEncoder,
    config: GeneEncoderConfig,
    augment_view_a: Callable[[torch.Tensor], torch.Tensor],
    augment_view_b: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
) -> BYOL:
    """Wrap :class:`GeneEncoder` in ``byol_pytorch.BYOL`` with our own augmentations.

    ``image_size=1`` is a required-but-otherwise-unused argument for a non-image
    encoder (see module docstring for how the mock-probe it triggers is handled).
    ``use_simplicial_embeddings=False`` keeps behavior as canonical BYOL rather than
    the package's newer, non-standard-by-default simplicial-embedding variant.
    """
    return BYOL(
        encoder,
        image_size=1,
        hidden_layer=-1,
        projection_size=config.projection_dim,
        projection_hidden_size=config.projection_hidden_dim,
        augment_fn=augment_view_a,
        augment_fn2=augment_view_b,
        moving_average_decay=config.ema_decay,
        use_simplicial_embeddings=False,
    ).to(device)
