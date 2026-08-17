"""MuCoST model: shared GCN autoencoder with three-view InfoNCE.

Faithful port of ``MuCoST/model.py`` from ``https://github.com/tju-zl/MuCoST``
(audited 2026-08-17).  ``torch_geometric`` GCNConv/BatchNorm/shuffle_node/
mask_feature are replaced by dependency-light implementations with identical
math: GCNConv(improved=True) adds self loops with weight 2, the propagation
uses the symmetric GCN normalization, and InfoNCE follows the published loss
verbatim.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def add_self_loops(
    row: torch.Tensor,
    col: torch.Tensor,
    n_nodes: int,
    fill_value: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return rows, cols, weights with self loops appended (improved GCN)."""
    self_nodes = torch.arange(n_nodes, device=row.device)
    if row.numel():
        base_weight = torch.ones(row.shape[0], dtype=row.dtype, device=row.device)
    else:
        base_weight = torch.empty(0, dtype=row.dtype, device=row.device)
    self_weight = torch.full((n_nodes,), fill_value, dtype=row.dtype, device=row.device)
    return (
        torch.cat([row, self_nodes]),
        torch.cat([col, self_nodes]),
        torch.cat([base_weight, self_weight]),
    )


class GCNConv(nn.Module):
    """Dependency-light GCNConv reproducing ``improved=True`` semantics."""

    def __init__(self, in_dim: int, out_dim: int, improved: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=True)
        self.improved = improved

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n_nodes = x.shape[0]
        x = self.linear(x)
        row, col = edge_index
        n_edges = row.shape[0]
        edge_w = (
            edge_weight
            if edge_weight is not None
            else torch.ones(n_edges, dtype=x.dtype, device=x.device)
        )
        r, c, w = add_self_loops(row, col, n_nodes, fill_value=2.0 if self.improved else 1.0)
        w = torch.cat([edge_w, w[n_edges:]])
        deg = torch.zeros(n_nodes, dtype=x.dtype, device=x.device)
        deg.scatter_add_(0, r, w)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        norm = deg_inv_sqrt[r] * w * deg_inv_sqrt[c]
        out = torch.zeros_like(x)
        out.index_add_(0, c, norm.unsqueeze(1) * x[r])
        return out


class InfoNCE(nn.Module):
    """InfoNCE loss with a positive key and an unpaired negative-key set."""

    def __init__(self, reduction: str = "mean", negative_mode: str = "unpaired") -> None:
        super().__init__()
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(
        self,
        query: torch.Tensor,
        positive_key: torch.Tensor,
        negative_keys: torch.Tensor | None,
        temperature: float,
    ) -> torch.Tensor:
        return info_nce(
            query,
            positive_key,
            negative_keys,
            temperature=temperature,
            reduction=self.reduction,
            negative_mode=self.negative_mode,
        )


def info_nce(
    query: torch.Tensor,
    positive_key: torch.Tensor,
    negative_keys: torch.Tensor | None = None,
    temperature: float = 1.0,
    reduction: str = "mean",
    negative_mode: str = "unpaired",
) -> torch.Tensor:
    query = F.normalize(query, dim=-1)
    positive_key = F.normalize(positive_key, dim=-1)
    negative_keys = (
        None if negative_keys is None else F.normalize(negative_keys, dim=-1)
    )
    if negative_keys is not None:
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)
        if negative_mode == "unpaired":
            negative_logits = query @ negative_keys.transpose(-2, -1)
        elif negative_mode == "paired":
            query = query.unsqueeze(1)
            negative_logits = (query @ negative_keys.transpose(-2, -1)).squeeze(1)
        else:
            raise ValueError(negative_mode)
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        logits = query @ positive_key.transpose(-2, -1)
        labels = torch.arange(len(query), device=query.device)
    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def shuffle_nodes(x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Randomly permute node rows (torch_geometric ``shuffle_node``)."""
    perm = torch.randperm(x.shape[0], generator=generator, device=x.device)
    return x[perm]


def mask_features(
    x: torch.Tensor, p: float, generator: torch.Generator
) -> torch.Tensor:
    """Randomly zero whole node rows with probability ``p`` (nodewise masking)."""
    mask = torch.rand(x.shape[0], generator=generator, device=x.device) < p
    result = x.clone()
    result[mask] = 0.0
    return result


class Model(nn.Module):
    """Shared GCN autoencoder with spatial / masked-feature / shuffled views."""

    def __init__(self, in_dim: int, config: object) -> None:
        super().__init__()
        self.config = config
        self.encoder = GCNConv(in_dim, config.latent_dim, improved=True)
        self.decoder = GCNConv(config.latent_dim, in_dim, improved=True)
        self.norm = nn.BatchNorm1d(config.latent_dim)
        self.act = nn.ELU()
        self.act1 = nn.ReLU()
        self.info_nce = InfoNCE()

    def forward(
        self,
        x: torch.Tensor,
        g_s: torch.Tensor,
        g_f: torch.Tensor | None = None,
        w_h: torch.Tensor | None = None,
        w_h_a: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        generator = torch.Generator(device=x.device).manual_seed(self.config.seed)
        x_n = shuffle_nodes(x, generator)
        x_p = mask_features(x, self.config.drop_feat_p, generator)

        hi = self.encoder(x, g_s, edge_weight=w_h)
        h = self.decoder(hi, g_s, edge_weight=w_h)

        h0 = self.act(self.norm(hi))

        if g_f is not None:
            h1 = self.encoder(x_p, g_f, edge_weight=w_h_a)
            h1 = self.act(self.norm(h1))
        else:
            h1 = torch.zeros_like(h0)

        h2 = self.encoder(x_n, g_s, edge_weight=w_h)
        h2 = self.act(self.norm(h2))

        loss = self.compute_loss(x, h, h0, h1, h2)
        return hi, h, loss

    def compute_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        p: torch.Tensor,
        p1: torch.Tensor,
        p2: torch.Tensor,
    ) -> torch.Tensor:
        loss_rec = F.mse_loss(x, y)
        loss_ctr = self.info_nce(p, p1, p2, temperature=self.config.temperature)
        return loss_rec + self.config.contrastive_weight * loss_ctr