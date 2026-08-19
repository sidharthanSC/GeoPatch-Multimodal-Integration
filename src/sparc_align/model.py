"""SPARC sparse autoencoder with Global TopK cross-stream concept alignment.

Faithful to Nasiri-Sarvi et al. (TMLR 3/2026) Sections 3.2-3.3 and Appendix A.2:

- affine per-stream encoders ``h_s = W_E^s (x_s - b_pre^s) + b_lat^s``     (Eq. 1)
- Global TopK index selection on the summed logits ``h_agg = sum_s h_s``   (Eq. 2-3)
- per-stream sparse codes sharing that support, ``z_s = ReLU(gather(h_s, I))`` (Eq. 4)
- affine per-stream decoders ``x_hat_s = W_D^s z_s + b_pre^s``            (Eq. 5)
- ``L_total = sum_s NMSE(x_s, D_s(z_s)) + lambda * sum_{s != t} NMSE(x_t, D_t(z_s))`` (Eq. 6)
- AuxK dead-latent revival, tied init, unit-norm decoder columns with parallel
  gradient projection, and dead-neuron reinitialization (Appendix A.2)

The paper's streams are two views of one image; here they are the two modalities of
one Visium spot: 3000-d HVG expression and 128-d histology BYOL features.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .config import SparcConfig


@dataclass(frozen=True)
class SparcOutput:
    """One SPARC forward pass over all streams."""

    latents: dict[str, torch.Tensor]  # z_s, (batch, n_latents), k-sparse
    logits: dict[str, torch.Tensor]  # h_s, (batch, n_latents), dense pre-activation
    self_reconstructions: dict[str, torch.Tensor]  # D_s(z_s)
    cross_reconstructions: dict[tuple[str, str], torch.Tensor]  # (s, t) -> D_t(z_s)
    # (batch, k) shared support I_global for the shared-support modes; under
    # "partitioned" the supports differ per stream, so this is their concatenation
    # (batch, k * n_streams) and is used only for dead-latent tracking.
    active_indices: torch.Tensor


def nmse(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    """Normalized MSE: squared error as a fraction of the target's own variance.

    ``1.0`` means the reconstruction is no better than predicting the batch mean,
    which is the diagnostic threshold for "this stream carries no transferable
    signal" -- the key quantity for cross-reconstruction on this dataset, where
    same-spot gene/image correspondence is known to be weak.
    """
    residual = (target - prediction).pow(2).sum()
    variance = (target - target.mean(dim=0, keepdim=True)).pow(2).sum()
    return residual / variance.clamp_min(torch.finfo(target.dtype).eps)


@torch.no_grad()
def _support_mask(logits: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Constant 0/1 mask marking the active support, built without autograd.

    Kept out of the graph on purpose: the mask carries no gradient, so multiplying
    by it is exactly gather+scatter of the selected logits while remaining
    deterministic on MPS.
    """
    mask = torch.zeros_like(logits)
    mask.scatter_(1, indices, 1.0)
    return mask


def global_topk(logits: dict[str, torch.Tensor], k: int) -> torch.Tensor:
    """Select the shared support from the summed per-stream logits (Eq. 2-3).

    Returns ``(batch, k)`` indices. Summing before selection is what forces every
    stream to activate identical latent dimensions for the same spot; this is the
    single mechanism separating SPARC from the Local TopK baseline.
    """
    aggregated = torch.stack(list(logits.values()), dim=0).sum(dim=0)
    return aggregated.topk(k, dim=-1).indices


@torch.no_grad()
def quota_topk(logits: dict[str, torch.Tensor], k: int, quota_fraction: float) -> torch.Tensor:
    """Global TopK with a guaranteed floor of slots per stream.

    Each stream is granted ``floor(k * quota_fraction / n_streams)`` slots from its
    own top-k; the union is then forced into the final selection by an additive
    boost, and the remaining slots are filled from the summed logits as usual.
    This makes single-stream monopolization structurally impossible rather than
    merely discouraged -- the failure mode measured on DLPFC, where the shared
    support was chosen almost entirely by the image stream.
    """
    names = list(logits)
    aggregated = torch.stack([logits[name] for name in names], dim=0).sum(dim=0)
    per_stream = int(k * quota_fraction) // len(names)
    if per_stream <= 0:
        return aggregated.topk(k, dim=-1).indices

    reserved = torch.zeros_like(aggregated)
    for name in names:
        reserved.scatter_(1, logits[name].topk(per_stream, dim=-1).indices, 1.0)

    # Boost must exceed the aggregate's dynamic range so reserved slots always win,
    # while ties among reserved entries still break by their aggregate value.
    boost = aggregated.abs().amax(dim=-1, keepdim=True) * 2.0 + 1.0
    return (aggregated + boost * reserved).topk(k, dim=-1).indices


@torch.no_grad()
def rank_fusion_topk(
    logits: dict[str, torch.Tensor], k: int, constant: float
) -> torch.Tensor:
    """Select the shared support by reciprocal-rank fusion across streams.

    ``score_j = sum_s 1 / (c + rank_s(j))`` with ``rank_s`` the 0-based descending
    rank of latent ``j`` within stream ``s``. Unlike summing logits (Eq. 2) this is
    invariant to each stream's scale *and* distribution shape, so a latent must
    rank highly in every stream to be chosen rather than merely being large in one.
    """
    score = None
    for logit in logits.values():
        order = logit.argsort(dim=-1, descending=True)
        ranks = torch.empty_like(order)
        arange = torch.arange(logit.shape[-1], device=logit.device).expand_as(order)
        ranks.scatter_(1, order, arange)
        contribution = 1.0 / (constant + ranks.to(logit.dtype))
        score = contribution if score is None else score + contribution
    return score.topk(k, dim=-1).indices


@torch.no_grad()
def partitioned_topk(
    logits: dict[str, torch.Tensor],
    k: int,
    n_latents: int,
    shared_fraction: float,
    private_k_fraction: float,
) -> dict[str, torch.Tensor]:
    """Shared Global-TopK block plus one private Local-TopK block per stream.

    The dictionary is split into a shared prefix and one contiguous private block
    per stream. Global TopK runs over the shared prefix only; each stream
    additionally selects its own atoms from its private block. Concepts with no
    cross-modal counterpart therefore stop competing for shared capacity, which is
    what the persistent 0.92 image->gene cross-NMSE says is happening on DLPFC.

    Returns per-stream supports; unlike the other modes these are NOT identical
    across streams -- they agree on the shared block and differ on the private one.
    """
    names = list(logits)
    n_shared = int(n_latents * shared_fraction)
    private_width = (n_latents - n_shared) // len(names)
    if private_width <= 0 or n_shared <= 0:
        raise ValueError("shared_fraction leaves no room for shared or private blocks")

    # Each stream must end up exactly k-sparse (||z_s||_0 <= k), matching every
    # other mode, so the private budget is carved out of k rather than added to it.
    k_private = min(int(k * private_k_fraction), private_width)
    k_shared = min(k - k_private, n_shared)
    if k_shared <= 0:
        raise ValueError("private_k_fraction consumes the entire sparsity budget")

    aggregated = torch.stack([logits[name][:, :n_shared] for name in names], dim=0).sum(dim=0)
    shared_indices = aggregated.topk(k_shared, dim=-1).indices

    supports: dict[str, torch.Tensor] = {}
    for position, name in enumerate(names):
        start = n_shared + position * private_width
        block = logits[name][:, start : start + private_width]
        private_indices = block.topk(k_private, dim=-1).indices + start
        supports[name] = torch.cat([shared_indices, private_indices], dim=-1)
    return supports


class StreamAutoencoder(nn.Module):
    """Encoder/decoder pair for one stream, with tied initialization.

    The decoder is always affine (Eq. 5) so its columns remain interpretable
    dictionary atoms in input space. The encoder is affine by default (Eq. 1); if
    ``hidden_dim`` is set, a single GELU hidden layer precedes the affine map --
    the paper's own suggested extension, needed here because a linear map cannot
    compress 3000-d sparse expression into k active latents.
    """

    def __init__(self, input_dim: int, n_latents: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.n_latents = n_latents
        self.pre_bias = nn.Parameter(torch.zeros(input_dim))
        self.latent_bias = nn.Parameter(torch.zeros(n_latents))

        self.hidden = (
            None
            if hidden_dim is None
            else nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
            )
        )
        # The affine head consumes the hidden width when one is present.
        encoder_in = input_dim if hidden_dim is None else hidden_dim
        self.encoder_weight = nn.Parameter(torch.empty(n_latents, encoder_in))
        nn.init.kaiming_uniform_(self.encoder_weight, a=5**0.5)

        # Tied init W_D = W_E^T is only meaningful when the encoder is affine, since
        # only then do the two live in the same space. With a hidden layer the
        # decoder gets its own init; unit-norm columns are enforced regardless.
        if hidden_dim is None:
            decoder_init = self.encoder_weight.data.T.clone()
        else:
            decoder_init = torch.empty(input_dim, n_latents)
            nn.init.kaiming_uniform_(decoder_init, a=5**0.5)
        self.decoder_weight = nn.Parameter(decoder_init)
        self.normalize_decoder()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        centered = x - self.pre_bias
        if self.hidden is not None:
            centered = self.hidden(centered)
        return F.linear(centered, self.encoder_weight, self.latent_bias)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return F.linear(z, self.decoder_weight) + self.pre_bias

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        """Rescale decoder columns to unit norm (Appendix A.2)."""
        self.decoder_weight.data /= self.decoder_weight.data.norm(dim=0, keepdim=True).clamp_min(1e-8)

    @torch.no_grad()
    def project_decoder_gradient(self) -> None:
        """Remove the gradient component parallel to each unit decoder column.

        Without this the optimizer fights the renormalization above: it would grow
        column norms that ``normalize_decoder`` immediately undoes, which shows up
        as a stalled loss rather than an error.
        """
        if self.decoder_weight.grad is None:
            return
        weight = self.decoder_weight.data
        grad = self.decoder_weight.grad
        parallel = (grad * weight).sum(dim=0, keepdim=True) * weight
        grad.sub_(parallel)


class SparcModel(nn.Module):
    """Multi-stream SPARC with Global TopK alignment and AuxK dead-latent revival."""

    def __init__(self, config: SparcConfig, stream_dims: dict[str, int]) -> None:
        super().__init__()
        if len(stream_dims) < 2:
            raise ValueError("SPARC requires at least two streams to align")
        self.config = config
        self.stream_names = list(stream_dims)
        self.streams = nn.ModuleDict(
            {
                name: StreamAutoencoder(dim, config.n_latents, config.encoder_hidden_dim)
                for name, dim in stream_dims.items()
            }
        )
        self.stream_loss_weights = {
            name: getattr(config, f"{name}_loss_weight", 1.0) for name in self.stream_names
        }

        # beta for the morphology-weighted neighbourhood aggregation. Learnable by
        # default, as in MP-MNCA's MorphologyPriorCrossAttention.
        if config.spatial_attention:
            beta = torch.tensor(float(config.morphology_prior_weight))
            if config.learn_morphology_weight:
                self.morphology_weight = nn.Parameter(beta)
            else:
                self.register_buffer("morphology_weight", beta)
        # Steps since each latent was last in the active support; drives both the
        # AuxK loss and dead-neuron reinitialization.
        self.register_buffer("steps_since_active", torch.zeros(config.n_latents, dtype=torch.long))

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        neighbor_batch: dict[str, torch.Tensor] | None = None,
        neighbor_similarity: torch.Tensor | None = None,
    ) -> SparcOutput:
        """Encode one batch of spots.

        ``neighbor_batch`` maps each stream to a ``(batch, n_neighbors, d_s)``
        tensor of that spot's spatial neighbours' features. It is used only when
        ``spatial_topk_weight > 0``, and only to smooth the support selection.
        ``neighbor_similarity`` is the ``(batch, n_neighbors)`` morphology prior,
        required only when ``spatial_attention`` is enabled.
        """
        logits = {name: self.streams[name].encode(batch[name]) for name in self.stream_names}

        # Only the support *selection* sees normalized logits; z_s keeps the raw
        # magnitudes so each stream can still activate a shared concept at its own
        # strength, which the paper (Section 3.2) treats as meaningful signal.
        selection_logits = (
            {name: logit / logit.pow(2).mean().sqrt().clamp_min(1e-8) for name, logit in logits.items()}
            if self.config.normalize_stream_logits
            else dict(logits)
        )

        weight = self.config.spatial_topk_weight
        if weight > 0.0:
            if neighbor_batch is None:
                raise ValueError("spatial_topk_weight > 0 requires neighbor_batch")

            # Attention-style neighbour weights from the morphology prior, shared
            # across streams (they describe the same spatial neighbourhood):
            #   w_ij = softmax_j(beta * log s_ij)
            # Uniform weights reduce this to the plain mean.
            attention_weights = None
            if self.config.spatial_attention:
                if neighbor_similarity is None:
                    raise ValueError("spatial_attention requires neighbor_similarity")
                attention_weights = F.softmax(
                    self.morphology_weight * torch.log(neighbor_similarity + 1e-6), dim=1
                ).unsqueeze(-1)  # (batch, n_neighbors, 1)

            for name in self.stream_names:
                neighbors = neighbor_batch[name]
                n_batch, n_neighbors, dim = neighbors.shape
                neighbor_logits = self.streams[name].encode(neighbors.reshape(-1, dim))
                neighbor_logits = neighbor_logits.view(n_batch, n_neighbors, -1)
                if attention_weights is None:
                    neighbor_aggregate = neighbor_logits.mean(dim=1)
                else:
                    neighbor_aggregate = (neighbor_logits * attention_weights).sum(dim=1)
                if self.config.normalize_stream_logits:
                    neighbor_aggregate = neighbor_aggregate / neighbor_aggregate.pow(
                        2
                    ).mean().sqrt().clamp_min(1e-8)
                selection_logits[name] = (
                    1.0 - weight
                ) * selection_logits[name] + weight * neighbor_aggregate

        k = self.config.k_active
        if not self.config.global_topk:
            # Local TopK ablation: each stream picks its own support independently,
            # so latent j can mean different things in different streams.
            supports = {
                name: logit.topk(k, dim=-1).indices for name, logit in selection_logits.items()
            }
            indices = supports[self.stream_names[0]]
        elif self.config.topk_mode == "global_sum":
            indices = global_topk(selection_logits, k)
            supports = {name: indices for name in self.stream_names}
        elif self.config.topk_mode == "quota":
            indices = quota_topk(selection_logits, k, self.config.quota_fraction)
            supports = {name: indices for name in self.stream_names}
        elif self.config.topk_mode == "rank_fusion":
            indices = rank_fusion_topk(selection_logits, k, self.config.rank_fusion_constant)
            supports = {name: indices for name in self.stream_names}
        elif self.config.topk_mode == "partitioned":
            supports = partitioned_topk(
                selection_logits,
                k,
                self.config.n_latents,
                self.config.shared_fraction,
                self.config.private_k_fraction,
            )
            # Supports differ per stream here, so the "active set" used for dead
            # tracking is their union rather than any single stream's.
            indices = torch.cat([supports[name] for name in self.stream_names], dim=-1)
        else:
            raise ValueError(f"unknown topk_mode: {self.config.topk_mode!r}")

        latents: dict[str, torch.Tensor] = {}
        for name in self.stream_names:
            # Build the support mask under no_grad and apply it multiplicatively.
            # Equivalent to gather+scatter, but scatter's backward on MPS
            # (scatter_reduce_mps) has no deterministic implementation, which
            # collides with the repo-wide use_deterministic_algorithms(True).
            latents[name] = F.relu(logits[name] * _support_mask(logits[name], supports[name]))

        self_reconstructions = {
            name: self.streams[name].decode(latents[name]) for name in self.stream_names
        }
        cross_reconstructions = {
            (source, target): self.streams[target].decode(latents[source])
            for source in self.stream_names
            for target in self.stream_names
            if source != target
        }
        return SparcOutput(
            latents=latents,
            logits=logits,
            self_reconstructions=self_reconstructions,
            cross_reconstructions=cross_reconstructions,
            active_indices=indices,
        )

    def compute_losses(
        self, batch: dict[str, torch.Tensor], output: SparcOutput
    ) -> dict[str, torch.Tensor]:
        """Self-reconstruction + lambda * cross-reconstruction + gamma * AuxK (Eq. 6)."""
        self_loss = sum(
            self.stream_loss_weights[name] * nmse(batch[name], output.self_reconstructions[name])
            for name in self.stream_names
        )
        cross_loss = (
            sum(nmse(batch[target], recon) for (_, target), recon in output.cross_reconstructions.items())
            if output.cross_reconstructions
            else self_loss.new_zeros(())
        )
        aux_loss = self._auxiliary_loss(batch, output)
        total = self_loss + self.config.cross_loss_weight * cross_loss + self.config.aux_loss_weight * aux_loss
        return {"total": total, "self": self_loss, "cross": cross_loss, "aux": aux_loss}

    def _auxiliary_loss(
        self, batch: dict[str, torch.Tensor], output: SparcOutput
    ) -> torch.Tensor:
        """AuxK: reconstruct the residual using top-k' currently-dead latents."""
        dead = self.steps_since_active >= self.config.dead_steps_threshold
        n_dead = int(dead.sum())
        if n_dead == 0:
            return output.logits[self.stream_names[0]].new_zeros(())

        k_aux = min(self.config.aux_k, n_dead)
        aggregated = torch.stack(list(output.logits.values()), dim=0).sum(dim=0)
        masked = aggregated.masked_fill(~dead, float("-inf"))
        aux_indices = masked.topk(k_aux, dim=-1).indices

        total = output.logits[self.stream_names[0]].new_zeros(())
        for name in self.stream_names:
            logit = output.logits[name]
            sparse = F.relu(logit * _support_mask(logit, aux_indices))
            residual = batch[name] - output.self_reconstructions[name].detach()
            total = total + nmse(
                residual, self.streams[name].decode(sparse) - self.streams[name].pre_bias
            )
        return total

    @torch.no_grad()
    def update_dead_latents(self, output: SparcOutput) -> None:
        """Age every latent, resetting those that fired in this batch."""
        self.steps_since_active += 1
        fired = torch.zeros_like(self.steps_since_active, dtype=torch.bool)
        fired[output.active_indices.reshape(-1)] = True
        self.steps_since_active[fired] = 0

    @torch.no_grad()
    def reinitialize_dead_latents(self) -> int:
        """Reinit latents dead past the threshold with N(0, dead_reinit_std) noise."""
        dead = self.steps_since_active >= self.config.dead_steps_threshold
        n_dead = int(dead.sum())
        if n_dead == 0:
            return 0
        for name in self.stream_names:
            stream = self.streams[name]
            stream.encoder_weight.data[dead] = (
                torch.randn_like(stream.encoder_weight.data[dead]) * self.config.dead_reinit_std
            )
            stream.latent_bias.data[dead] = 0.0
            stream.decoder_weight.data[:, dead] = (
                torch.randn_like(stream.decoder_weight.data[:, dead]) * self.config.dead_reinit_std
            )
            stream.normalize_decoder()
        self.steps_since_active[dead] = 0
        return n_dead

    @torch.no_grad()
    def post_step(self) -> None:
        """Renormalize decoder columns after the optimizer step."""
        for name in self.stream_names:
            self.streams[name].normalize_decoder()

    @torch.no_grad()
    def pre_step(self) -> None:
        """Project out decoder gradient components parallel to unit columns."""
        for name in self.stream_names:
            self.streams[name].project_decoder_gradient()
