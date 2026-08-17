"""MP-MNCA Phase 2: BYOL Encoder 3000->3000 with Reconstruction.

Encoder: 3000 -> 3000 -> 3000 (no dim reduction)
Mask gene values -> encode -> reconstruct full 3000-dim
Reconstruction loss (Huber on masked positions)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class Phase2Output:
    """Output from Phase 2 BYOL encoder."""
    spot_embeddings: torch.Tensor       # (n_spots, 3000) - encoder output
    reconstruction: torch.Tensor        # (batch, 3000) - decoder output
    mask: torch.Tensor                  # (batch, 3000) - which genes were masked


class GeneBYOLEncoder(nn.Module):
    """BYOL-style gene encoder: 3000 -> 3000 -> 3000 (no dim reduction)."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        gene_dim = config.gene_dim

        # Encoder: 3000 -> hidden -> 3000
        encoder_dims = (gene_dim, *config.byol_hidden_dims, gene_dim)
        activation = nn.ReLU if config.gene_activation == "relu" else nn.GELU

        self.encoder = nn.ModuleList(
            nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                activation(),
                nn.Dropout(config.gene_dropout),
            )
            for in_dim, out_dim in zip(encoder_dims[:-1], encoder_dims[1:])
        )

        # Projection head (for BYOL loss, not used at inference)
        self.projector = nn.Sequential(
            nn.Linear(gene_dim, config.byol_projection_hidden_dim),
            nn.LayerNorm(config.byol_projection_hidden_dim),
            nn.GELU(),
            nn.Linear(config.byol_projection_hidden_dim, config.byol_projection_dim),
        )

        # Predictor
        self.predictor = nn.Sequential(
            nn.Linear(config.byol_projection_dim, config.byol_predictor_hidden_dim),
            nn.LayerNorm(config.byol_predictor_hidden_dim),
            nn.GELU(),
            nn.Linear(config.byol_predictor_hidden_dim, config.byol_projection_dim),
        )

        # Decoder for reconstruction: 3000 -> 3000
        self.decoder = nn.Sequential(
            nn.Linear(gene_dim, gene_dim),
            nn.LayerNorm(gene_dim),
            nn.GELU(),
            nn.Dropout(config.gene_dropout),
            nn.Linear(gene_dim, gene_dim),
        )

        # EMA target encoder
        self.target_encoder = None
        self._init_target_encoder()

    def _init_target_encoder(self):
        """Initialize EMA target encoder as copy of online encoder."""
        import copy
        self.target_encoder = nn.ModuleList([
            copy.deepcopy(m) for m in self.encoder
        ])
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_target_encoder(self, decay: float = None):
        """EMA update of target encoder."""
        if decay is None:
            decay = self.config.byol_ema_decay
        for target_param, online_param in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            target_param.data.mul_(decay).add_(online_param.data, alpha=1 - decay)

    def forward_encoder(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder only (returns 3000-d representation)."""
        for block in self.encoder:
            x = block(x)
        return x

    def forward_online(self, x: torch.Tensor) -> torch.Tensor:
        """Online forward: encoder -> projector -> predictor."""
        z = self.forward_encoder(x)
        return self.predictor(self.projector(z))

    def forward_target(self, x: torch.Tensor) -> torch.Tensor:
        """Target forward: encoder -> projector (no grad)."""
        with torch.no_grad():
            z = self.forward_encoder(x)
            return self.projector(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Default forward: return encoder output (3000-d)."""
        return self.forward_encoder(x)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Encode and decode for reconstruction."""
        z = self.forward_encoder(x)
        return self.decoder(z)

    def byol_loss(self, view1: torch.Tensor, view2: torch.Tensor) -> torch.Tensor:
        """BYOL loss: predict target projector output from online predictor."""
        pred1 = self.forward_online(view1)
        pred2 = self.forward_online(view2)
        target1 = self.forward_target(view2)
        target2 = self.forward_target(view1)
        loss1 = F.mse_loss(F.normalize(pred1, dim=-1), F.normalize(target1, dim=-1))
        loss2 = F.mse_loss(F.normalize(pred2, dim=-1), F.normalize(target2, dim=-1))
        return (loss1 + loss2) / 2


class MaskedReconstructionLoss(nn.Module):
    """Weighted Huber reconstruction loss on masked positions."""

    def __init__(self, nonzero_weight: float = 5.0):
        super().__init__()
        self.nonzero_weight = nonzero_weight

    def forward(
        self,
        prediction: torch.Tensor,   # (batch, 3000)
        target: torch.Tensor,       # (batch, 3000)
        mask: torch.Tensor,         # (batch, 3000) bool
    ) -> torch.Tensor:
        if prediction.shape != target.shape or mask.shape != target.shape:
            raise ValueError("prediction, target, and mask must have identical shapes")
        selected = mask.bool()
        if not selected.any():
            return prediction.new_zeros(())
        error = F.smooth_l1_loss(prediction, target, reduction="none")
        weights = 1.0 + self.nonzero_weight * (target > 0).to(target.dtype)
        return (error[selected] * weights[selected]).sum() / weights[selected].sum().clamp_min(1.0)


class Phase2Model(nn.Module):
    """Phase 2: BYOL Encoder + Reconstruction."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.byol_encoder = GeneBYOLEncoder(config)
        self.recon_loss_fn = MaskedReconstructionLoss()

    def create_masked_view(
        self,
        x: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create masked view of gene expression."""
        mask_rate = self.config.byol_mask_rate
        noise_std = self.config.byol_noise_std

        if generator is not None:
            mask = torch.rand_like(x) < mask_rate
            mask = torch.bernoulli(torch.full_like(x, mask_rate), generator=generator).bool()
        else:
            mask = torch.rand_like(x) < mask_rate
        x_masked = x.clone()
        x_masked[mask] = 0

        if self.config.byol_noise_std > 0:
            x_masked = x_masked + torch.randn_like(x_masked) * noise_std * x.abs().mean()

        return x_masked, mask

    def forward(
        self,
        x: torch.Tensor,           # (batch, 3000) - clean gene expression
        generator: Optional[torch.Generator] = None,
    ) -> Phase2Output:
        # Create masked view
        x_masked, mask = self.create_masked_view(x, generator)

        # Encode masked
        z = self.byol_encoder.forward_encoder(x_masked)

        # Decode for reconstruction
        recon = self.byol_encoder.decoder(z)

        return Phase2Output(
            spot_embeddings=z,
            reconstruction=recon,
            mask=mask,
        )

    def compute_losses(
        self,
        output: Phase2Output,
        target: torch.Tensor,      # (batch, 3000) - clean gene expression
    ) -> dict[str, torch.Tensor]:
        """Compute reconstruction loss + BYOL loss."""
        # Reconstruction loss (only on masked positions)
        recon_loss = self.recon_loss_fn(output.reconstruction, target, output.mask)

        # BYOL loss (optional, for regularization)
        # Need two views - create second masked view
        x_masked2, mask2 = self.create_masked_view(target)
        byol_loss = self.byol_encoder.byol_loss(output.spot_embeddings, self.byol_encoder.forward_encoder(x_masked2))

        return {
            "reconstruction": recon_loss,
            "byol": byol_loss,
            "total": recon_loss + 0.1 * byol_loss,
        }