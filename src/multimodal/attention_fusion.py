"""Method 4 (supervised): cross-attention + self-attention multimodal fusion classifier.

Each modality is a single 128-d vector per spot, not a token sequence -- so plain
self-attention over one modality alone would be a no-op (a length-1 sequence has
nothing to attend to). Design used here instead:

1. **Cross-attention**, both directions: the image vector (as a length-1 "query"
   sequence) attends to the gene vector (as the length-1 "key/value" sequence), and
   vice versa. With only one key, the softmax weighting is trivially 1.0 -- but the
   learned Q/K/V projections still make this a genuine, non-identity transform of each
   modality conditioned on the other.
2. **Self-attention** over the *pair* of cross-attended outputs, stacked into a
   length-2 sequence -- this is where attention becomes non-degenerate (two real
   tokens, a real softmax distribution between them).
3. Pool (mean over the 2 tokens) -> a small FFN block -> linear classification head
   over the 7 ground-truth layers.

Uses the same stratified train/test split as method 3
(``src/multimodal/evaluation.py``'s ``shared_train_test_split``) so their accuracies
are directly comparable; the training split is further divided into train/val
internally, purely for this method's own early stopping -- the held-out test set is
never touched until final evaluation.

Usage
-----
    python -m src.multimodal.attention_fusion
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn

from src.multimodal.data import load_embeddings_and_labels
from src.multimodal.evaluation import shared_train_test_split


@dataclass(frozen=True)
class FusionConfig:
    dim: int = 128
    n_heads: int = 4
    ffn_dim: int = 256
    n_classes: int = 7
    dropout: float = 0.1


class AttentionFusionModel(nn.Module):
    def __init__(self, config: FusionConfig) -> None:
        super().__init__()
        dim = config.dim
        self.image_to_gene_attn = nn.MultiheadAttention(dim, config.n_heads, dropout=config.dropout, batch_first=True)
        self.gene_to_image_attn = nn.MultiheadAttention(dim, config.n_heads, dropout=config.dropout, batch_first=True)
        self.self_attn = nn.MultiheadAttention(dim, config.n_heads, dropout=config.dropout, batch_first=True)

        self.norm_image_ca = nn.LayerNorm(dim)
        self.norm_gene_ca = nn.LayerNorm(dim)
        self.norm_self_attn = nn.LayerNorm(dim)
        self.norm_ffn = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, config.ffn_dim), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(config.ffn_dim, dim)
        )
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(dim, config.n_classes)

    def forward(self, image_vec: torch.Tensor, gene_vec: torch.Tensor) -> torch.Tensor:
        image_tok = image_vec.unsqueeze(1)  # (B, 1, dim)
        gene_tok = gene_vec.unsqueeze(1)  # (B, 1, dim)

        image_ca, _ = self.image_to_gene_attn(query=image_tok, key=gene_tok, value=gene_tok)
        gene_ca, _ = self.gene_to_image_attn(query=gene_tok, key=image_tok, value=image_tok)
        image_ca = self.norm_image_ca(image_tok + self.dropout(image_ca))
        gene_ca = self.norm_gene_ca(gene_tok + self.dropout(gene_ca))

        seq = torch.cat([image_ca, gene_ca], dim=1)  # (B, 2, dim)
        self_attended, _ = self.self_attn(seq, seq, seq)
        seq = self.norm_self_attn(seq + self.dropout(self_attended))
        seq = self.norm_ffn(seq + self.dropout(self.ffn(seq)))

        pooled = seq.mean(dim=1)  # (B, dim)
        return self.classifier(pooled)


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_epoch(
    model: AttentionFusionModel,
    image_emb: torch.Tensor,
    gene_emb: torch.Tensor,
    label_ints: torch.Tensor,
    indices: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    train: bool,
    optimizer: torch.optim.Optimizer = None,
) -> Dict[str, float]:
    model.train() if train else model.eval()
    order = indices.copy()
    if train:
        rng.shuffle(order)

    total_loss, total_correct, n = 0.0, 0, 0
    for start in range(0, len(order), batch_size):
        batch_idx = order[start : start + batch_size]
        with torch.set_grad_enabled(train):
            logits = model(image_emb[batch_idx], gene_emb[batch_idx])
            loss = nn.functional.cross_entropy(logits, label_ints[batch_idx])

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item()) * len(batch_idx)
        total_correct += int((logits.argmax(dim=1) == label_ints[batch_idx]).sum().item())
        n += len(batch_idx)

    return {"loss": total_loss / n, "accuracy": total_correct / n}


def run(
    checkpoint_path: Path,
    test_fraction: float,
    val_fraction: float,
    num_epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: str,
) -> dict:
    gene_emb_np, image_emb_np, labels, barcodes, section_ids = load_embeddings_and_labels(checkpoint_path)
    train_val_idx, test_idx = shared_train_test_split(labels, test_fraction, seed)

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    label_names = sorted(np.unique(labels))
    label_to_int = {name: i for i, name in enumerate(label_names)}
    label_ints_np = np.array([label_to_int[l] for l in labels])

    from sklearn.model_selection import train_test_split as sk_split

    train_idx, val_idx = sk_split(
        train_val_idx, test_size=val_fraction, random_state=seed, stratify=labels[train_val_idx]
    )

    dev = torch.device(device)
    image_emb = torch.as_tensor(image_emb_np, dtype=torch.float32, device=dev)
    gene_emb = torch.as_tensor(gene_emb_np, dtype=torch.float32, device=dev)
    label_ints = torch.as_tensor(label_ints_np, dtype=torch.long, device=dev)

    model = AttentionFusionModel(FusionConfig(dim=image_emb_np.shape[1], n_classes=len(label_names))).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, num_epochs + 1):
        train_metrics = run_epoch(model, image_emb, gene_emb, label_ints, train_idx, batch_size, rng, True, optimizer)
        val_metrics = run_epoch(model, image_emb, gene_emb, label_ints, val_idx, batch_size, rng, False)

        if val_metrics["loss"] < best_val_loss - 1e-4:
            best_val_loss = val_metrics["loss"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch % 5 == 0 or epochs_without_improvement >= patience:
            print(
                f"epoch {epoch}/{num_epochs} | train loss {train_metrics['loss']:.4f} acc {train_metrics['accuracy']:.4f} "
                f"| val loss {val_metrics['loss']:.4f} acc {val_metrics['accuracy']:.4f}"
            )
        if epochs_without_improvement >= patience:
            print(f"Early stopping after {patience} epochs without improvement.")
            break

    model.load_state_dict(best_state)
    test_metrics = run_epoch(model, image_emb, gene_emb, label_ints, test_idx, batch_size, rng, False)

    return {
        "metrics": {
            "accuracy": test_metrics["accuracy"], "loss": test_metrics["loss"],
            "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
        },
        "model": model,
        "test_idx": test_idx,
        "barcodes": barcodes,
        "section_ids": section_ids,
        "labels": labels,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, default=Path("checkpoints/dlpfc.pkl"))
    parser.add_argument("--test-fraction", type=float, default=0.3)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=default_device())
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/multimodal"))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = run(
        args.checkpoint_path, args.test_fraction, args.val_fraction, args.num_epochs,
        args.patience, args.batch_size, args.lr, args.seed, args.device,
    )

    print("Method 4 (attention fusion, supervised):", json.dumps(result["metrics"], indent=2))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "method4_attention_fusion_metrics.json").open("w") as f:
        json.dump(result["metrics"], f, indent=2)
    torch.save(result["model"].state_dict(), args.output_dir / "method4_attention_fusion_model.pt")
    print(f"Wrote {args.output_dir}/method4_attention_fusion_metrics.json and model checkpoint")


if __name__ == "__main__":
    main()
