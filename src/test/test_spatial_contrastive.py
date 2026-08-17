import numpy as np
import torch

from src.cross_modal.spatial_contrastive import (
    build_graph_batch,
    multi_positive_info_nce,
)


def test_multi_positive_loss_rewards_positive_scores() -> None:
    logits = torch.tensor([[3.0, 2.0, -1.0], [2.0, 3.0, -1.0]])
    positives = torch.tensor([[True, True, False], [True, True, False]])
    valid = torch.ones_like(positives)
    good = multi_positive_info_nce(logits, positives, valid)
    bad = multi_positive_info_nce(-logits, positives, valid)
    assert good < bad


def test_graph_batch_marks_neighbors_and_masks_uncertain_candidates() -> None:
    retained = (np.array([1]), np.array([0]), np.array([], dtype=np.int64))
    candidates = (np.array([1, 2]), np.array([0]), np.array([0]))
    sections = np.array(["A", "A", "A"])
    gathered, cross_pos, neighbor_pos, cross_valid, intra_valid = build_graph_batch(
        np.array([0, 1]), retained, candidates, sections
    )
    positions = {row: i for i, row in enumerate(gathered)}
    assert cross_pos[0, positions[0]]
    assert cross_pos[0, positions[1]]
    assert neighbor_pos[0, positions[1]]
    if 2 in positions:
        assert not cross_valid[0, positions[2]]
    assert not intra_valid[0, positions[0]]
