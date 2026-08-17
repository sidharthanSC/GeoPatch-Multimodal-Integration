"""GraphST model: GNN encoder with DGI-style contrastive learning.

Faithful port of ``GraphST/model.py`` from
``https://github.com/JinmiaoChenLab/GraphST`` (audited 2026-08-17).  The dense
10X Visium branch is used; ``torch.mm`` reproduces the dense adjacency
propagation of the official ``Encoder``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter


class Discriminator(nn.Module):
    """Bilinear discriminator scoring positive vs corrupted node embeddings."""

    def __init__(self, n_h: int) -> None:
        super().__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        for module in self.modules():
            self.weights_init(module)

    def weights_init(self, m: nn.Module) -> None:
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(
        self,
        c: torch.Tensor,
        h_pl: torch.Tensor,
        h_mi: torch.Tensor,
        s_bias1: torch.Tensor | None = None,
        s_bias2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        c_x = c.expand_as(h_pl)
        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)
        if s_bias1 is not None:
            sc_1 = sc_1 + s_bias1
        if s_bias2 is not None:
            sc_2 = sc_2 + s_bias2
        return torch.cat((sc_1, sc_2), 1)


class AvgReadout(nn.Module):
    """Neighborhood-averaged, L2-normalized graph readout."""

    def forward(self, emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        vsum = torch.mm(mask, emb)
        row_sum = torch.sum(mask, 1)
        row_sum = row_sum.expand((vsum.shape[1], row_sum.shape[0])).T
        global_emb = vsum / row_sum
        return F.normalize(global_emb, p=2, dim=1)


class Encoder(nn.Module):
    """Two-weight GNN encoder with DGI discriminator (dense 10X branch)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        graph_neigh: torch.Tensor,
        dropout: float = 0.0,
        act: object = F.relu,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.graph_neigh = graph_neigh
        self.dropout = dropout
        self.act = act

        self.weight1 = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.weight2 = Parameter(torch.FloatTensor(self.out_features, self.in_features))
        self.reset_parameters()

        self.disc = Discriminator(self.out_features)
        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)

    def forward(
        self,
        feat: torch.Tensor,
        feat_a: torch.Tensor,
        adj: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z = F.dropout(feat, self.dropout, self.training)
        z = torch.mm(z, self.weight1)
        z = torch.mm(adj, z)
        hiden_emb = z

        h = torch.mm(z, self.weight2)
        h = torch.mm(adj, h)
        emb = self.act(z)

        z_a = F.dropout(feat_a, self.dropout, self.training)
        z_a = torch.mm(z_a, self.weight1)
        z_a = torch.mm(adj, z_a)
        emb_a = self.act(z_a)

        g = self.sigm(self.read(emb, self.graph_neigh))
        g_a = self.sigm(self.read(emb_a, self.graph_neigh))

        ret = self.disc(g, emb, emb_a)
        ret_a = self.disc(g_a, emb_a, emb)
        return hiden_emb, h, ret, ret_a