"""SpaGCN model: single GCN layer plus deep embedded clustering (DEC) head.

Faithful port of ``SpaGCN_package/SpaGCN/{layers,models}.py`` from
``https://github.com/jianhuupenn/SpaGCN`` (audited 2026-08-17).  ``louvain``
cluster-centre initialisation requires igraph, which is unavailable in this
repository, so ``kmeans`` initialisation is used with the label-count-informed
``n_clusters`` (the model supports ``init="kmeans"`` natively).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

from .config import SpaGcnConfig


class GraphConvolution(nn.Module):
    """Simple GCN layer, https://arxiv.org/abs/1609.02907."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        support = torch.mm(input, self.weight)
        if adj.is_sparse:
            output = torch.sparse.mm(adj, support)
        else:
            output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        return output


def to_sparse_adjacency(
    adjacency: np.ndarray,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Convert a dense thresholded adjacency to a coalesced sparse COO tensor."""
    adjacency = np.asarray(adjacency, dtype=np.float32)
    rows, cols = np.nonzero(adjacency)
    indices = np.stack([rows, cols], axis=0).astype(np.int64)
    values = adjacency[rows, cols].astype(np.float32)
    tensor = torch.sparse_coo_tensor(
        indices,
        torch.as_tensor(values),
        adjacency.shape,
        device=device,
    )
    return tensor.coalesce()


class SimpleGcDec(nn.Module):
    """Single-layer GCN with a DEC clustering head (alpha=0.2, student-t)."""

    def __init__(self, nfeat: int, nhid: int, alpha: float = 0.2) -> None:
        super().__init__()
        self.gc = GraphConvolution(nfeat, nhid)
        self.nhid = nhid
        self.mu: Parameter  # cluster centers; assigned during fit()
        self.alpha = alpha

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.gc(x, adj)
        q = 1.0 / (
            (1.0 + torch.sum((x.unsqueeze(1) - self.mu) ** 2, dim=2) / self.alpha) + 1e-8
        )
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = q / torch.sum(q, dim=1, keepdim=True)
        return x, q

    def loss_function(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        def kld(target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
            return torch.mean(torch.sum(target * torch.log(target / (pred + 1e-6)), dim=1))

        return kld(p, q)

    def target_distribution(self, q: torch.Tensor) -> torch.Tensor:
        p = q**2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p

    def fit(
        self,
        X: np.ndarray,
        adj: np.ndarray,
        lr: float = 0.05,
        max_epochs: int = 200,
        update_interval: int = 3,
        weight_decay: float = 5e-4,
        optimizer_name: str = "admin",
        n_clusters: int | None = None,
        init_spa: bool = True,
        tol: float = 5e-3,
        seed: int = 100,
        device: torch.device | str | None = None,
    ) -> list[float]:
        """Train the DEC objective with a k-means cluster-centre initialisation."""
        if optimizer_name == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        elif optimizer_name == "admin":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer {optimizer_name}")

        Xt = torch.FloatTensor(np.asarray(X, dtype=np.float32)).to(device)
        adjt = to_sparse_adjacency(adj, device=device)
        with torch.no_grad():
            features = self.gc(Xt, adjt)

        if n_clusters is None:
            raise ValueError("SpaGCN requires a known cluster count for k-means init.")
        self.n_clusters = n_clusters
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters, n_init=20, random_state=seed)
        y_pred = (
            kmeans.fit_predict(features.detach().cpu().numpy())
            if init_spa
            else kmeans.fit_predict(np.asarray(X, dtype=np.float32))
        )
        y_pred_last = y_pred

        self.mu = Parameter(torch.Tensor(self.n_clusters, self.nhid).to(device))
        centers = np.stack(
            [
                features.detach().cpu().numpy()[y_pred == c].mean(axis=0)
                for c in range(self.n_clusters)
            ]
        )
        self.mu.data.copy_(torch.Tensor(centers).to(device))

        self.train()
        losses: list[float] = []
        for epoch in range(max_epochs):
            if epoch % update_interval == 0:
                with torch.no_grad():
                    _, q = self.forward(Xt, adjt)
                p = self.target_distribution(q).data
            optimizer.zero_grad()
            _, q = self.forward(Xt, adjt)
            loss = self.loss_function(p, q)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

            y_pred = torch.argmax(q, dim=1).detach().cpu().numpy()
            delta_label = float(np.sum(y_pred != y_pred_last)) / Xt.shape[0]
            y_pred_last = y_pred
            if epoch > 0 and (epoch - 1) % update_interval == 0 and delta_label < tol:
                break
        return losses

    def predict(
        self, X: np.ndarray, adj: np.ndarray, device: torch.device | str | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (GCN embedding, DEC soft assignment argmax)."""
        self.eval()
        Xt = torch.FloatTensor(np.asarray(X, dtype=np.float32)).to(device)
        adjt = to_sparse_adjacency(adj, device=device)
        with torch.no_grad():
            z, q = self.forward(Xt, adjt)
        return z.detach().cpu().numpy(), torch.argmax(q, dim=1).detach().cpu().numpy()


def pca_embedding(
    features: np.ndarray, num_pcs: int, seed: int = 100
) -> np.ndarray:
    """Official wrapper reduces expression to ``num_pcs`` PCs before the GCN."""
    from sklearn.decomposition import PCA

    pca = PCA(n_components=num_pcs, random_state=seed)
    return pca.fit_transform(features).astype(np.float32)