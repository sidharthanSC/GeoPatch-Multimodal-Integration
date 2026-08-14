"""Parameter-matched supervised fusion ablations on a shared spot split."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch import nn

from src.multimodal.data import load_embeddings_and_labels
from src.multimodal.evaluation import shared_train_test_split


class ConcatMLP(nn.Module):
    def __init__(self, dim: int, n_classes: int, hidden_dim: int = 1024) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, image: torch.Tensor, gene: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([image, gene], dim=1))


class SelfAttentionFusion(nn.Module):
    def __init__(self, dim: int, n_classes: int, n_heads: int = 4) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, n_heads, dropout=0.1, batch_first=True)
        self.norm_attention = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, dim)
        )
        self.norm_ffn = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, n_classes)

    def forward(self, image: torch.Tensor, gene: torch.Tensor) -> torch.Tensor:
        tokens = torch.stack([image, gene], dim=1)
        attended, _ = self.attention(tokens, tokens, tokens)
        tokens = self.norm_attention(tokens + attended)
        tokens = self.norm_ffn(tokens + self.ffn(tokens))
        return self.classifier(tokens.mean(dim=1))


def run(
    checkpoint_path: Path,
    model_name: str,
    output_dir: Path,
    seed: int = 0,
    epochs: int = 100,
    patience: int = 15,
    batch_size: int = 256,
    lr: float = 1e-4,
    device: str = "cuda",
) -> dict:
    metrics_path = output_dir / f"{model_name}_metrics.json"
    checkpoint_path_out = output_dir / f"{model_name}_best.pt"
    if metrics_path.exists() or checkpoint_path_out.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")

    gene_np, image_np, labels, barcodes, section_ids = load_embeddings_and_labels(checkpoint_path)
    train_val_idx, test_idx = shared_train_test_split(labels, 0.3, seed)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.15,
        random_state=seed,
        stratify=labels[train_val_idx],
    )
    label_names = sorted(np.unique(labels))
    label_to_int = {label: index for index, label in enumerate(label_names)}
    encoded = np.array([label_to_int[label] for label in labels])

    torch.manual_seed(seed)
    dev = torch.device(device)
    gene = torch.as_tensor(gene_np, dtype=torch.float32, device=dev)
    image = torch.as_tensor(image_np, dtype=torch.float32, device=dev)
    targets = torch.as_tensor(encoded, dtype=torch.long, device=dev)
    if model_name == "concat_mlp":
        model = ConcatMLP(image.shape[1], len(label_names)).to(dev)
    elif model_name == "self_attention":
        model = SelfAttentionFusion(image.shape[1], len(label_names)).to(dev)
    else:
        raise ValueError("model_name must be concat_mlp or self_attention")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0

    def epoch(indices: np.ndarray, train: bool) -> tuple[float, np.ndarray]:
        model.train(train)
        order = indices.copy()
        if train:
            rng.shuffle(order)
        loss_total = 0.0
        predictions = []
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            with torch.set_grad_enabled(train):
                logits = model(image[batch], gene[batch])
                loss = nn.functional.cross_entropy(logits, targets[batch])
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            loss_total += float(loss.item()) * len(batch)
            predictions.append(logits.argmax(dim=1).detach().cpu().numpy())
        return loss_total / len(order), np.concatenate(predictions)

    for current_epoch in range(1, epochs + 1):
        epoch(train_idx, True)
        val_loss, _ = epoch(val_idx, False)
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = current_epoch
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    model.load_state_dict(best_state)
    test_loss, predicted = epoch(test_idx, False)
    true = encoded[test_idx]
    result = {
        "model": model_name,
        "seed": seed,
        "best_epoch": best_epoch,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "test_loss": test_loss,
        "accuracy": accuracy_score(true, predicted),
        "balanced_accuracy": balanced_accuracy_score(true, predicted),
        "macro_f1": f1_score(true, predicted, average="macro"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "result": result, "label_names": label_names},
        checkpoint_path_out,
    )
    with metrics_path.open("w") as file:
        json.dump(result, file, indent=2)
    np.savez_compressed(
        output_dir / f"{model_name}_predictions.npz",
        test_idx=test_idx,
        predicted=predicted,
        true=true,
        barcodes=barcodes[test_idx],
        section_ids=section_ids[test_idx],
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--model", choices=["concat_mlp", "self_attention"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(run(
        args.checkpoint_path, args.model, args.output_dir, args.seed, args.epochs,
        args.patience, args.batch_size, args.lr, args.device,
    ), indent=2))


if __name__ == "__main__":
    main()
