"""Attention ablation suite A1-A12 from ``MODEL_BENCHMARK_REPORT.md``.

All variants share one training/eval harness (``run``, adapted from
``src/multimodal/attention_fusion.py``'s loop) and the same ``shared_train_test_split``
outer split used by methods 3/4, so every ablation's test accuracy is directly
comparable. Two axes vary per ablation:

- **Representation**: aligned (``gene_emb_cm_img`` / ``img_emb_cm``, the pair every
  other ablation uses) vs. unaligned (``gene_emb`` / ``img_emb``, A12 only).
- **Model**: a linear/MLP probe on some fixed combination of the two vectors (A1-A6),
  or ``FlexibleAttentionModel`` with cross-/self-attention toggled on or off (A7-A9,
  A11, A12).

A10 (modality-shuffle-at-test sanity check) and the reported parameter counts are
computed post-hoc from the already-trained A9 model rather than as separate training
runs -- shuffling is an evaluation-time perturbation, not a different model.

Usage
-----
    python -u -m src.benchmark.attention_ablations
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn

from src.multimodal.attention_fusion import default_device
from src.multimodal.data import load_embeddings_and_labels
from src.multimodal.evaluation import shared_train_test_split

FeatureFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def bisector_torch(image_vec: torch.Tensor, gene_vec: torch.Tensor) -> torch.Tensor:
    image_unit = image_vec / image_vec.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    gene_unit = gene_vec / gene_vec.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    bisector = image_unit + gene_unit
    return bisector / bisector.norm(dim=-1, keepdim=True).clamp_min(1e-12)


class FeatureProbe(nn.Module):
    """Linear or two-layer-MLP classifier on top of a fixed ``feature_fn`` of the two
    input vectors -- covers A1-A6 (gene-only, image-only, bisector, concatenation,
    each either linear or with one GELU hidden layer)."""

    def __init__(self, feature_fn: FeatureFn, in_dim: int, n_classes: int, hidden_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        self.feature_fn = feature_fn
        if hidden_dim is None:
            self.net = nn.Linear(in_dim, n_classes)
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, n_classes)
            )

    def forward(self, image_vec: torch.Tensor, gene_vec: torch.Tensor) -> torch.Tensor:
        return self.net(self.feature_fn(image_vec, gene_vec))


@dataclass(frozen=True)
class FlexibleFusionConfig:
    dim: int = 128
    n_heads: int = 4
    ffn_dim: int = 256
    n_classes: int = 7
    dropout: float = 0.1
    use_cross_attention: bool = True
    use_self_attention: bool = True


class FlexibleAttentionModel(nn.Module):
    """``src.multimodal.attention_fusion.AttentionFusionModel`` with cross-/self-
    attention independently toggleable -- A7 (self-only), A8 (cross-only), A9 (both,
    == method 4's architecture), A11 (both, + modality dropout at train time), A12
    (both, on unaligned embeddings) all instantiate this with different flags/inputs."""

    def __init__(self, config: FlexibleFusionConfig) -> None:
        super().__init__()
        dim = config.dim
        self.config = config

        if config.use_cross_attention:
            self.image_to_gene_attn = nn.MultiheadAttention(dim, config.n_heads, dropout=config.dropout, batch_first=True)
            self.gene_to_image_attn = nn.MultiheadAttention(dim, config.n_heads, dropout=config.dropout, batch_first=True)
            self.norm_image_ca = nn.LayerNorm(dim)
            self.norm_gene_ca = nn.LayerNorm(dim)

        if config.use_self_attention:
            self.self_attn = nn.MultiheadAttention(dim, config.n_heads, dropout=config.dropout, batch_first=True)
            self.norm_self_attn = nn.LayerNorm(dim)

        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, config.ffn_dim), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(config.ffn_dim, dim)
        )
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(dim, config.n_classes)

    def forward(self, image_vec: torch.Tensor, gene_vec: torch.Tensor) -> torch.Tensor:
        image_tok = image_vec.unsqueeze(1)
        gene_tok = gene_vec.unsqueeze(1)

        if self.config.use_cross_attention:
            image_ca, _ = self.image_to_gene_attn(query=image_tok, key=gene_tok, value=gene_tok)
            gene_ca, _ = self.gene_to_image_attn(query=gene_tok, key=image_tok, value=image_tok)
            image_ca = self.norm_image_ca(image_tok + self.dropout(image_ca))
            gene_ca = self.norm_gene_ca(gene_tok + self.dropout(gene_ca))
        else:
            image_ca, gene_ca = image_tok, gene_tok

        seq = torch.cat([image_ca, gene_ca], dim=1)

        if self.config.use_self_attention:
            self_attended, _ = self.self_attn(seq, seq, seq)
            seq = self.norm_self_attn(seq + self.dropout(self_attended))

        seq = self.norm_ffn(seq + self.dropout(self.ffn(seq)))
        pooled = seq.mean(dim=1)
        return self.classifier(pooled)


def run_epoch(
    model: nn.Module,
    image_emb: torch.Tensor,
    gene_emb: torch.Tensor,
    label_ints: torch.Tensor,
    indices: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    train: bool,
    optimizer: Optional[torch.optim.Optimizer] = None,
    modality_dropout_p: float = 0.0,
) -> Dict[str, float]:
    model.train() if train else model.eval()
    order = indices.copy()
    if train:
        rng.shuffle(order)

    total_loss, total_correct, n = 0.0, 0, 0
    for start in range(0, len(order), batch_size):
        batch_idx = order[start : start + batch_size]
        image_batch = image_emb[batch_idx]
        gene_batch = gene_emb[batch_idx]

        if train and modality_dropout_p > 0:
            drop_image = torch.as_tensor(rng.random(len(batch_idx)) < modality_dropout_p / 2, device=image_batch.device)
            drop_gene = torch.as_tensor(
                (rng.random(len(batch_idx)) < modality_dropout_p / 2) & (~drop_image.cpu().numpy()), device=gene_batch.device
            )
            image_batch = image_batch.clone()
            gene_batch = gene_batch.clone()
            image_batch[drop_image] = 0.0
            gene_batch[drop_gene] = 0.0

        with torch.set_grad_enabled(train):
            logits = model(image_batch, gene_batch)
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
    model_factory: Callable[[int], nn.Module],
    gene_key: str,
    image_key: str,
    checkpoint_path: Path,
    test_fraction: float,
    val_fraction: float,
    num_epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: str,
    modality_dropout_p: float = 0.0,
) -> dict:
    gene_emb_np, image_emb_np, labels, barcodes, section_ids = load_embeddings_and_labels(
        checkpoint_path, gene_key=gene_key, image_key=image_key
    )
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

    model = model_factory(len(label_names)).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, num_epochs + 1):
        run_epoch(model, image_emb, gene_emb, label_ints, train_idx, batch_size, rng, True, optimizer, modality_dropout_p)
        val_metrics = run_epoch(model, image_emb, gene_emb, label_ints, val_idx, batch_size, rng, False)

        if val_metrics["loss"] < best_val_loss - 1e-4:
            best_val_loss = val_metrics["loss"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break

    model.load_state_dict(best_state)
    test_metrics = run_epoch(model, image_emb, gene_emb, label_ints, test_idx, batch_size, rng, False)

    return {
        "metrics": {"accuracy": test_metrics["accuracy"], "loss": test_metrics["loss"], "n_params": n_params,
                     "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx), "best_epoch": epoch - epochs_without_improvement},
        "model": model,
        "test_idx": test_idx,
        "image_emb": image_emb,
        "gene_emb": gene_emb,
        "label_ints": label_ints,
        "device": dev,
    }


def modality_shuffle_eval(trained: dict, batch_size: int, seed: int) -> Dict[str, float]:
    """A10: evaluate the already-trained model with the test set's image embeddings
    permuted relative to gene embeddings (breaking spot correspondence), as a sanity
    check that the model uses genuine cross-modal pairing rather than one modality alone."""
    rng = np.random.default_rng(seed)
    test_idx = trained["test_idx"]
    permuted_image = trained["image_emb"].clone()
    perm = rng.permutation(len(test_idx))
    permuted_image[test_idx] = trained["image_emb"][test_idx][perm]

    return run_epoch(
        trained["model"], permuted_image, trained["gene_emb"], trained["label_ints"],
        test_idx, batch_size, rng, train=False,
    )


ALIGNED_KEYS = dict(gene_key="gene_emb_cm_img", image_key="img_emb_cm")
UNALIGNED_KEYS = dict(gene_key="gene_emb", image_key="img_emb")


def build_ablations(n_heads: int, ffn_dim: int, dropout: float) -> Dict[str, dict]:
    """Each entry: {"model_factory": callable(n_classes)->nn.Module, keys..., "modality_dropout_p": float}."""
    def linear_gene(n_classes): return FeatureProbe(lambda i, g: g, 128, n_classes)
    def linear_image(n_classes): return FeatureProbe(lambda i, g: i, 128, n_classes)
    def linear_bisector(n_classes): return FeatureProbe(bisector_torch, 128, n_classes)
    def linear_concat(n_classes): return FeatureProbe(lambda i, g: torch.cat([i, g], dim=-1), 256, n_classes)
    def mlp_concat(n_classes): return FeatureProbe(lambda i, g: torch.cat([i, g], dim=-1), 256, n_classes, hidden_dim=256, dropout=dropout)
    def mlp_bisector(n_classes): return FeatureProbe(bisector_torch, 128, n_classes, hidden_dim=256, dropout=dropout)

    def self_only(n_classes): return FlexibleAttentionModel(FlexibleFusionConfig(n_heads=n_heads, ffn_dim=ffn_dim, n_classes=n_classes, dropout=dropout, use_cross_attention=False, use_self_attention=True))
    def cross_only(n_classes): return FlexibleAttentionModel(FlexibleFusionConfig(n_heads=n_heads, ffn_dim=ffn_dim, n_classes=n_classes, dropout=dropout, use_cross_attention=True, use_self_attention=False))
    def full_model(n_classes): return FlexibleAttentionModel(FlexibleFusionConfig(n_heads=n_heads, ffn_dim=ffn_dim, n_classes=n_classes, dropout=dropout, use_cross_attention=True, use_self_attention=True))

    return {
        "A1_linear_gene": {"model_factory": linear_gene, **ALIGNED_KEYS},
        "A2_linear_image": {"model_factory": linear_image, **ALIGNED_KEYS},
        "A3_linear_bisector": {"model_factory": linear_bisector, **ALIGNED_KEYS},
        "A4_linear_concat": {"model_factory": linear_concat, **ALIGNED_KEYS},
        "A5_mlp_concat": {"model_factory": mlp_concat, **ALIGNED_KEYS},
        "A6_mlp_bisector": {"model_factory": mlp_bisector, **ALIGNED_KEYS},
        "A7_self_attention_only": {"model_factory": self_only, **ALIGNED_KEYS},
        "A8_cross_attention_only": {"model_factory": cross_only, **ALIGNED_KEYS},
        "A9_full_current_model": {"model_factory": full_model, **ALIGNED_KEYS},
        "A11_modality_dropout": {"model_factory": full_model, **ALIGNED_KEYS, "modality_dropout_p": 0.15},
        "A12_unaligned_full_model": {"model_factory": full_model, **UNALIGNED_KEYS},
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, default=Path("checkpoints/dlpfc.pkl"))
    parser.add_argument("--test-fraction", type=float, default=0.3)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--num-epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", type=str, default=default_device())
    parser.add_argument("--ablations", type=str, nargs="*", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/benchmark/attention_ablations"))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    ablations = build_ablations(args.n_heads, args.ffn_dim, args.dropout)
    names = args.ablations or list(ablations.keys())

    results = {}
    trained_a9 = None
    for name in names:
        spec = ablations[name]
        print(f"Running {name} ...", flush=True)
        trained = run(
            spec["model_factory"], spec["gene_key"], spec["image_key"], args.checkpoint_path,
            args.test_fraction, args.val_fraction, args.num_epochs, args.patience,
            args.batch_size, args.lr, args.seed, args.device,
            modality_dropout_p=spec.get("modality_dropout_p", 0.0),
        )
        results[name] = trained["metrics"]
        print(f"  {name}: {json.dumps(trained['metrics'])}", flush=True)
        if name == "A9_full_current_model":
            trained_a9 = trained

    if trained_a9 is not None:
        print("Running A10_modality_shuffle_sanity_check ...", flush=True)
        shuffle_metrics = modality_shuffle_eval(trained_a9, args.batch_size, args.seed)
        results["A10_modality_shuffle_sanity_check"] = shuffle_metrics
        print(f"  A10_modality_shuffle_sanity_check: {json.dumps(shuffle_metrics)}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "ablation_results.json").open("w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {args.output_dir}/ablation_results.json", flush=True)


if __name__ == "__main__":
    main()
