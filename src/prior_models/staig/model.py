"""Dependency-light implementation of STAIG's GCN and neighbor contrastive loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def normalized_adjacency(
    edge_index: torch.Tensor, n_nodes: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Build the symmetric GCN normalization with self loops."""
    self_nodes = torch.arange(n_nodes, device=device)
    indices = torch.cat(
        [edge_index, torch.stack([self_nodes, self_nodes], dim=0)], dim=1
    )
    values = torch.ones(indices.shape[1], dtype=dtype, device=device)
    adjacency = torch.sparse_coo_tensor(indices, values, (n_nodes, n_nodes)).coalesce()
    row, col = adjacency.indices()
    degree = torch.zeros(n_nodes, dtype=dtype, device=device)
    degree.scatter_add_(0, row, adjacency.values())
    weights = adjacency.values() * degree[row].rsqrt() * degree[col].rsqrt()
    return torch.sparse_coo_tensor(
        adjacency.indices(), weights, adjacency.shape, device=device
    ).coalesce()


class GraphConvolution(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(output_dim))
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(adjacency, self.linear(x)) + self.bias


class StaigModel(nn.Module):
    """One-layer PReLU GCN followed by STAIG's ELU MLP projection head."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, projection_dim: int = 64) -> None:
        super().__init__()
        self.gcn = GraphConvolution(input_dim, hidden_dim)
        self.activation = nn.PReLU()
        self.project_1 = nn.Linear(hidden_dim, projection_dim)
        self.project_2 = nn.Linear(projection_dim, hidden_dim)

    def encode(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return self.activation(self.gcn(x, adjacency))

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        encoded = self.encode(x, adjacency)
        return self.project_2(F.elu(self.project_1(encoded)))


def _directional_neighbor_loss(
    query: torch.Tensor,
    other: torch.Tensor,
    edge_index: torch.Tensor,
    pseudo_labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    query = F.normalize(query, dim=1)
    other = F.normalize(other, dim=1)
    intra = torch.exp((query @ query.T) / temperature)
    inter = torch.exp((query @ other.T) / temperature)
    negative_mask = pseudo_labels[:, None] != pseudo_labels[None, :]

    src, dst = edge_index
    intra_positive = torch.zeros(query.shape[0], dtype=query.dtype, device=query.device)
    inter_positive = torch.zeros_like(intra_positive)
    intra_positive.scatter_add_(0, src, intra[src, dst])
    inter_positive.scatter_add_(0, src, inter[src, dst])
    neighbor_count = torch.zeros_like(intra_positive)
    neighbor_count.scatter_add_(0, src, torch.ones_like(src, dtype=query.dtype))

    numerator = inter.diag() + intra_positive + inter_positive
    denominator = (
        (intra * negative_mask).sum(1)
        + (inter * negative_mask).sum(1)
        - intra.diag()
    )
    ratio = numerator / (denominator.clamp_min(torch.finfo(query.dtype).tiny))
    ratio = ratio / (2 * neighbor_count + 1).clamp_min(1)
    return -torch.log(ratio.clamp_min(torch.finfo(query.dtype).tiny))


def neighbor_contrastive_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    edge_index: torch.Tensor,
    pseudo_labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Symmetric debiased STAIG neighbor contrastive objective."""
    first = _directional_neighbor_loss(z1, z2, edge_index, pseudo_labels, temperature)
    second = _directional_neighbor_loss(z2, z1, edge_index, pseudo_labels, temperature)
    return 0.5 * (first + second).mean()


def mask_features(x: torch.Tensor, rate: float, generator: torch.Generator) -> torch.Tensor:
    """Mask complete gene columns, matching the official STAIG implementation."""
    mask = torch.rand(x.shape[1], generator=generator, device=x.device) < rate
    result = x.clone()
    result[:, mask] = 0
    return result
