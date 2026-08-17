"""Export multistage embeddings from a saved image-guided gene checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.datasets.dlpfc import DlpfcDataset
from src.gene_encoder.image_guided import ImageProjectionHead
from src.gene_encoder.image_guided_train import ImageGuidedTrainConfig, load_image_features
from src.gene_encoder.model import GeneEncoder, GeneEncoderConfig
from src.gene_encoder.rich_data import load_sparse_expression
from src.gene_encoder.rich_train import export_sparse_multistage
from src.gene_encoder.train import three_way_split


def export_image_guided_checkpoint(
    checkpoint_path: Path,
    predictions_path: Path | None = None,
    device: str = "cuda",
) -> Path:
    """Export one selected checkpoint without resuming its optimization."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ImageGuidedTrainConfig(**checkpoint["train_config"])
    encoder_config = GeneEncoderConfig(**checkpoint["encoder_config"])
    if predictions_path is None:
        predictions_path = (
            config.output_dir / "predictions" / f"{config.run_name}_embeddings.npz"
        )
    if predictions_path.exists():
        raise FileExistsError(f"Refusing to overwrite {predictions_path}")

    dataset = DlpfcDataset.from_checkpoint(config.checkpoint_path)
    genes = (
        json.loads(config.feature_genes_path.read_text(encoding="utf-8"))
        if config.feature_genes_path is not None
        else None
    )
    store = load_sparse_expression(dataset, genes)
    image_np = load_image_features(dataset, store, config.image_key)
    encoder = GeneEncoder(encoder_config).to(device)
    encoder.load_state_dict(checkpoint["encoder"])
    image_head = ImageProjectionHead(
        image_np.shape[1], config.image_hidden_dim, config.embedding_dim, config.dropout
    ).to(device)
    image_head.load_state_dict(checkpoint["image_head"])

    exported = export_sparse_multistage(
        encoder, store, torch.device(device), batch_size=256
    )
    image_head.eval()
    image_tensor = torch.from_numpy(image_np).to(device)
    image_blocks = []
    with torch.no_grad():
        for start in range(0, len(image_tensor), 4096):
            image_blocks.append(
                image_head(image_tensor[start : start + 4096])
                .cpu()
                .numpy()
                .astype(np.float32)
            )
    train_idx, val_idx, test_idx = three_way_split(
        store.section_ids, config.val_fraction, config.test_fraction, config.seed
    )
    split = np.full(store.matrix.shape[0], "train", dtype=object)
    split[val_idx] = "val"
    split[test_idx] = "test"
    stage_arrays = {
        f"gene_stage_{width}": exported[f"encoder_stage_{index}"]
        for index, width in enumerate(config.hidden_dims, start=1)
    }
    genes_path = (
        config.feature_genes_path
        or config.output_dir / "preprocessing" / f"{config.run_name}_genes.json"
    )
    geometry_path = (
        config.output_dir / "preprocessing" / f"{config.run_name}_geometry.npz"
    )
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        predictions_path,
        embeddings=exported["embedding"],
        gene_embedding_128=exported["embedding"],
        image_projected=np.concatenate(image_blocks),
        **stage_arrays,
        barcodes=store.barcodes,
        section_ids=store.section_ids,
        split=split,
        schema_version=np.asarray(1, dtype=np.int64),
        objective=np.asarray(checkpoint["objective"]),
        image_key=np.asarray(config.image_key),
        hidden_dims=np.asarray(config.hidden_dims, dtype=np.int64),
        embedding_dim=np.asarray(config.embedding_dim, dtype=np.int64),
        input_dim=np.asarray(store.matrix.shape[1], dtype=np.int64),
        feature_genes_path=np.asarray(str(genes_path)),
        geometry_path=np.asarray(str(geometry_path)),
        spatial_graph_path=np.asarray(str(config.spatial_graph_path)),
        initial_checkpoint=np.asarray(str(config.initial_checkpoint)),
        selected_checkpoint=np.asarray(str(checkpoint_path)),
        best_epoch=np.asarray(checkpoint["epoch"], dtype=np.int64),
        best_val_loss=np.asarray(
            checkpoint["metrics"]["val"]["total_loss"], dtype=np.float64
        ),
    )
    return predictions_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--predictions-path", type=Path)
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    path = export_image_guided_checkpoint(
        args.checkpoint, args.predictions_path, args.device
    )
    print(path)


if __name__ == "__main__":
    main()
