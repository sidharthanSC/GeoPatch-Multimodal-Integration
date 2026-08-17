from pathlib import Path

import numpy as np
import torch
from scipy import sparse

from src.cross_modal.model import CrossModalConfig, CrossModalModel
from src.cross_modal.rich_train import (
    RichCrossModalConfig,
    load_geometry_scores,
    load_rich_provenance,
    run_epoch,
)
from src.gene_encoder.model import GeneReconstructionDecoder
from src.gene_encoder.rich_data import SparseExpressionStore


def test_rich_provenance_and_geometry_identity_validation(tmp_path):
    gene_path = tmp_path / "gene.npz"
    geometry_path = tmp_path / "geometry.npz"
    genes_path = tmp_path / "genes.json"
    np.savez(
        gene_path,
        feature_genes_path=np.asarray(str(genes_path)),
        geometry_path=np.asarray(str(geometry_path)),
    )
    np.savez(
        geometry_path,
        scores=np.ones((2, 3), dtype=np.float32),
        section_ids=np.asarray(["151507", "151508"]),
        barcodes=np.asarray(["a", "b"]),
    )
    assert load_rich_provenance(gene_path) == (genes_path, geometry_path)
    scores = load_geometry_scores(
        geometry_path,
        np.asarray(["151507", "151508"]),
        np.asarray(["a", "b"]),
    )
    assert scores.shape == (2, 3)


def test_rich_cross_modal_epoch_reports_finite_auxiliary_losses():
    rng = np.random.default_rng(0)
    expression = rng.random((8, 6), dtype=np.float32)
    store = SparseExpressionStore(
        matrix=sparse.csr_matrix(expression),
        gene_names=np.asarray([f"g{i}" for i in range(6)]),
        barcodes=np.asarray([f"b{i}" for i in range(8)]),
        section_ids=np.asarray(["151507"] * 8),
        gene_std=expression.std(axis=0),
    )
    model = CrossModalModel(
        CrossModalConfig(
            gene_input_dim=4,
            image_input_dim=4,
            hidden_dim=8,
            output_dim=4,
            dropout=0,
        )
    )
    decoder = GeneReconstructionDecoder(4, 6)
    config = RichCrossModalConfig(
        gene_npz_path=Path("unused"),
        run_name="test",
        hidden_dim=8,
        output_dim=4,
        dropout=0,
        batch_size=4,
        device="cpu",
    )
    metrics = run_epoch(
        model,
        decoder,
        torch.randn(8, 4),
        torch.randn(8, 4),
        store,
        torch.randn(8, 4),
        np.arange(8),
        config,
        rng,
        train=False,
    )
    assert set(metrics) == {
        "loss",
        "clip_loss",
        "reconstruction_loss",
        "geometry_loss",
        "variance_loss",
        "covariance_loss",
        "paired_cosine",
        "retrieval_accuracy",
    }
    assert all(np.isfinite(value) for value in metrics.values())
