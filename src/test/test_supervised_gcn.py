import torch

from src.prior_models.supervised_gcn import SupervisedGCN
from src.prior_models.staig.model import normalized_adjacency


def test_supervised_gcn_returns_node_logits_and_gradients():
    edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    adjacency = normalized_adjacency(edges, 3, dtype=torch.float32, device=torch.device("cpu"))
    model = SupervisedGCN(4, 6, 3, dropout=0.0)
    logits = model(torch.randn(3, 4), adjacency)
    assert logits.shape == (3, 3)
    logits.sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
