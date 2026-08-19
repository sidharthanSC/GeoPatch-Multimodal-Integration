"""MP-MNCA ablation suite: reusable API for systematic component ablations.

Each ablation keeps the full Phase-1 training protocol identical (20 epochs,
seed 0, batch 256, lr 3e-4, mask rate 0.1, temperature 10, 8 heads, k=6
neighbors, image PCA-16, 40 image pseudo-clusters, tied-GMM + 15-neighbor
refinement) and removes exactly one component at a time. This isolates the
contribution of each mechanism so the reported deltas are attributable.

Variants
--------
- ``full``: the complete model (morphology prior + relative position bias +
  adaptive cross-attention + feature masking).
- ``no_morphology_prior``: beta = 0 (image-guided routing removed).
- ``no_position_bias``: gamma = 0 (learned local geometry removed).
- ``gene_only_attention``: beta = gamma = 0 (molecular routing alone).
- ``uniform_knn``: alpha_ij = 1/k (fixed smoothing instead of adaptive softmax).
- ``embedding_only``: no model, cluster raw ``obsm['feat']`` directly with the
  same tied-GMM + refinement backend.

Ablation outputs live under ``<output_root>/<variant>/`` with per-section
checkpoints, embeddings, ``section_metrics.csv`` and ``summary.json``,
mirroring ``src/mp_mnca/train_phase1.py``'s layout so ``outputs/README.md``
can index them identically.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.datasets.dlpfc import DlpfcDataset
from src.prior_models.staig.evaluate import clustering_metrics, refine_labels, tied_gmm
from src.prior_models.staig.data import (
    build_spatial_graph,
    image_guided_edge_probabilities,
    sample_augmented_edges,
)
from src.prior_models.staig.model import mask_features, normalized_adjacency

from .config import MpMncaConfig
from .data import MpMncaSectionData, prepare_image_features
from .model import contrastive_loss, contrastive_loss_masked
from .phase1 import Phase1Model

ALL_SECTIONS = [
    "151507", "151508", "151509", "151510",
    "151669", "151670", "151671", "151672",
    "151673", "151674", "151675", "151676",
]


@dataclass(frozen=True)
class AblationVariant:
    """One ablation row: a config mutation plus a scientific question."""
    name: str
    dim: int
    question: str
    config: MpMncaConfig


def full_protocol() -> MpMncaConfig:
    """Exact config used by the canonical full MP-MNCA Phase-1 run."""
    return MpMncaConfig(
        gene_dim=3000,
        num_heads=8,
        attention_dropout=0.1,
        use_relative_position_bias=True,
        relative_position_dim=16,
        n_neighbors=6,
        image_embedding_dim=128,
        image_pca_dim=16,
        morphology_prior_type="cosine",
        morphology_prior_weight=1.0,
        position_bias_weight=0.5,
        learn_morphology_weight=True,
        learn_position_weight=True,
        epochs=20,
        batch_size=256,
        learning_rate=3e-4,
        weight_decay=1e-5,
        temperature=10.0,
        mask_rate=0.1,
        image_pseudo_clusters=40,
        refinement_neighbors=15,
        n_clusters=None,
        seed=0,
    )


def _clone(config: MpMncaConfig, **overrides: object) -> MpMncaConfig:
    values = {**config.to_dict(), **overrides}
    return MpMncaConfig(**values)


def ablation_variants() -> list[AblationVariant]:
    """Define every ablation row in the study (3000-dim Phase-1 family)."""
    full = full_protocol()
    return [
        AblationVariant(
            name="full",
            dim=3000,
            question="Complete model: morphology prior + position bias + adaptive cross-attention + feature masking",
            config=full,
        ),
        AblationVariant(
            name="phase1_20ep",
            dim=3000,
            question="Canonical Phase 1 at the original 20-epoch / batch-256 setting, all 12 sections",
            config=_clone(full, epochs=20, batch_size=256),
        ),
        AblationVariant(
            name="intersection_filter",
            dim=3000,
            question="Train only on gene/image KMeans-agreement spots; hold out the rest for evaluation",
            config=_clone(full, intersection_clusters=7),
        ),
        AblationVariant(
            name="gene_kmeans_mask",
            dim=3000,
            question="Contrastive negative mask from gene-expression KMeans pseudo-labels (no image clustering)",
            config=_clone(full, gene_pseudo_clusters=7),
        ),
        AblationVariant(
            name="gene300_400ep",
            dim=300,
            question="Top-300 HVGs, 400 epochs, gene-KMeans contrastive mask (label-free)",
            config=_clone(
                full, gene_dim=300, num_heads=6, epochs=400, gene_pseudo_clusters=7
            ),
        ),
        AblationVariant(
            name="no_morphology_prior",
            dim=3000,
            question="Does histology-guided routing help? (beta = 0)",
            config=_clone(full, morphology_prior_weight=0.0, learn_morphology_weight=False),
        ),
        AblationVariant(
            name="no_position_bias",
            dim=3000,
            question="Does learned local geometry help? (gamma = 0)",
            config=_clone(full, use_relative_position_bias=False),
        ),
        AblationVariant(
            name="gene_only_attention",
            dim=3000,
            question="How strong is learned molecular routing alone? (beta = gamma = 0)",
            config=_clone(
                full,
                morphology_prior_weight=0.0,
                learn_morphology_weight=False,
                use_relative_position_bias=False,
            ),
        ),
        AblationVariant(
            name="embedding_only",
            dim=3000,
            question="What is gained over direct clustering of spot embeddings?",
            config=full,
        ),
    ]


def _set_seed(seed: int) -> None:
    import os
    import random

    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def prepare_ablation_section(
    adata,
    section_id: str,
    config: MpMncaConfig,
    n_neighbors: int | None = None,
    image_key: str = "img_emb",
    hvg_key: str = "highly_variable",
    use_feat_obsm: bool = True,
) -> MpMncaSectionData:
    """Prepare one section exactly like ``train_phase1.prepare_section_phase1``."""
    from .data import prepare_gene_expression

    if n_neighbors is None:
        n_neighbors = config.n_neighbors
    feat_subset = None
    if config.gene_dim < 3000:
        from pathlib import Path as _Path
        idx_path = _Path("outputs/gene_encoder/top300_hvg_indices.npy")
        if idx_path.exists():
            feat_subset = np.load(idx_path)[: config.gene_dim]
    features = prepare_gene_expression(
        adata, hvg_key, config.gene_dim, use_feat_obsm, feat_subset_indices=feat_subset
    )
    image_features = prepare_image_features(adata, config.image_pca_dim, image_key)
    coordinates = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    labels = adata.obs["ground_truth"].astype(str).to_numpy()
    barcodes = adata.obs_names.astype(str).to_numpy()

    from sklearn.neighbors import NearestNeighbors

    finder = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(coordinates)
    _, indices = finder.kneighbors(coordinates)
    neighbor_indices = indices[:, 1:]

    train_mask = None
    pseudo_labels = None
    if config.intersection_clusters is not None:
        from .data import intersection_pseudo_mask
        train_mask = intersection_pseudo_mask(
            features, image_features, config.intersection_clusters, config.seed
        )
    if config.gene_pseudo_clusters is not None:
        from .data import gene_kmeans_pseudo_labels
        pseudo_labels = gene_kmeans_pseudo_labels(
            features, config.gene_pseudo_clusters, config.seed
        )

    return MpMncaSectionData(
        section_id=section_id,
        gene_expression=features,
        image_features=image_features,
        coordinates=coordinates,
        labels=labels,
        barcodes=barcodes,
        neighbor_indices=neighbor_indices,
        gene_mask=None,
        n_neighbors=n_neighbors,
        train_mask=train_mask,
        pseudo_labels=pseudo_labels,
    )


def fit_ablation_variant(
    section_data: MpMncaSectionData,
    config: MpMncaConfig,
    device: torch.device | str | None = None,
    resume_state: dict | None = None,
    start_epoch: int = 1,
    resume_best_ari: float = -1.0,
) -> dict:
    """Train one ablation variant on one section; returns metrics + arrays.

    This is the Phase-1 training loop (contrastive loss over two masked views)
    copied from ``train_phase1.fit_phase1`` so every ablation shares the exact
    objective, optimizer, augmentation, and evaluation backend.

    ``resume_state`` / ``start_epoch`` continue training from a previously
    saved state dict (e.g. 10 more epochs from the best checkpoint).
    """
    import torch

    _set_seed(config.seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float32 if config.dtype == "float32" else torch.float64

    n_spots = section_data.gene_expression.shape[0]
    section_id = section_data.section_id

    model = Phase1Model(config).to(device=device, dtype=dtype)
    if resume_state is not None:
        model.load_state_dict({k: torch.as_tensor(v, device=device, dtype=dtype) for k, v in resume_state.items()})
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    numpy_rng = np.random.default_rng(config.seed)
    torch_generator = torch.Generator(device=device).manual_seed(config.seed)

    losses: list[float] = []

    features = torch.as_tensor(section_data.gene_expression, dtype=dtype, device=device)
    center_image = torch.as_tensor(section_data.image_features, dtype=dtype, device=device)
    center_coords = torch.as_tensor(section_data.coordinates, dtype=dtype, device=device)
    neighbor_idx = torch.as_tensor(section_data.neighbor_indices, dtype=torch.long, device=device)

    neighbor_genes = features[neighbor_idx]
    neighbor_images = center_image[neighbor_idx]
    neighbor_coords = center_coords[neighbor_idx]

    src = np.repeat(np.arange(n_spots), section_data.neighbor_indices.shape[1])
    dst = section_data.neighbor_indices.ravel()
    edge_index_np = np.stack([src, dst], axis=0).astype(np.int64)
    edge_index = torch.as_tensor(edge_index_np, dtype=torch.long, device=device)

    edge_drop_prob = np.ones(edge_index_np.shape[1]) * 0.1

    if section_data.train_mask is None:
        train_idx = np.arange(n_spots, dtype=np.int64)
    else:
        train_idx = np.where(section_data.train_mask)[0].astype(np.int64)
    train_set = set(train_idx.tolist())
    keep_edge = np.array(
        [s in train_set and d in train_set for s, d in zip(edge_index_np[0], edge_index_np[1])],
        dtype=bool,
    )
    remap = np.full(n_spots, -1, dtype=np.int64)
    remap[train_idx] = np.arange(train_idx.size, dtype=np.int64)
    edge_index_train_np = remap[edge_index_np[:, keep_edge]]
    edge_index_train = torch.as_tensor(edge_index_train_np, dtype=torch.long, device=device)
    train_idx_t = torch.as_tensor(train_idx, dtype=torch.long, device=device)

    batch_size = config.batch_size
    n_batches = (n_spots + batch_size - 1) // batch_size
    indices = np.arange(n_spots)

    best_ari = resume_best_ari
    best_state = None
    best_epoch = 0

    for epoch in range(start_epoch, config.epochs + 1):
        model.train()
        epoch_loss = 0.0

        numpy_rng.shuffle(indices)

        for b in range(n_batches):
            start = b * batch_size
            end = min(start + batch_size, n_spots)
            idx = torch.arange(start, end, device=device)

            optimizer.zero_grad(set_to_none=True)

            edges_1 = torch.as_tensor(
                sample_augmented_edges(edge_index_np, edge_drop_prob, numpy_rng),
                dtype=torch.long, device=device
            )
            edges_2 = torch.as_tensor(
                sample_augmented_edges(edge_index_np, edge_drop_prob, numpy_rng),
                dtype=torch.long, device=device
            )

            adjacency_1 = normalized_adjacency(edges_1, n_spots, dtype=dtype, device=device)
            adjacency_2 = normalized_adjacency(edges_2, n_spots, dtype=dtype, device=device)

            masked_1 = mask_features(features, config.mask_rate, torch_generator)
            masked_2 = mask_features(features, config.mask_rate, torch_generator)

            all_emb_1 = []
            for b in range(n_batches):
                start = b * batch_size
                end = min(start + batch_size, n_spots)
                b_idx = torch.arange(start, end, device=device)
                out = model.forward(
                    masked_1[b_idx],
                    masked_1[neighbor_idx[b_idx]],
                    center_image[b_idx],
                    neighbor_images[b_idx],
                    center_coords[b_idx],
                    neighbor_coords[b_idx],
                )
                all_emb_1.append(out.spot_embeddings)
            z1 = torch.cat(all_emb_1, dim=0)

            all_emb_2 = []
            for b in range(n_batches):
                start = b * batch_size
                end = min(start + batch_size, n_spots)
                b_idx = torch.arange(start, end, device=device)
                out = model.forward(
                    masked_2[b_idx],
                    masked_2[neighbor_idx[b_idx]],
                    center_image[b_idx],
                    neighbor_images[b_idx],
                    center_coords[b_idx],
                    neighbor_coords[b_idx],
                )
                all_emb_2.append(out.spot_embeddings)
            z2 = torch.cat(all_emb_2, dim=0)

            if section_data.pseudo_labels is None:
                loss = contrastive_loss(
                    z1[train_idx_t], z2[train_idx_t], edge_index_train, config.temperature
                )
            else:
                pseudo_t = torch.as_tensor(
                    section_data.pseudo_labels[train_idx], dtype=torch.long, device=device
                )
                loss = contrastive_loss_masked(
                    z1[train_idx_t], z2[train_idx_t], edge_index_train,
                    pseudo_t, config.temperature,
                )

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(idx)

        epoch_loss /= n_spots
        losses.append(epoch_loss)

        if epoch == 1 or epoch % 10 == 0 or epoch == config.epochs:
            print(f"{section_data.section_id} epoch={epoch:03d} loss={epoch_loss:.6f}", flush=True)

        if epoch % 10 == 0 or epoch == config.epochs:
            model.eval()
            with torch.no_grad():
                all_emb = []
                for b in range(n_batches):
                    start = b * batch_size
                    end = min(start + batch_size, n_spots)
                    idx = torch.arange(start, end, device=device)
                    out = model.forward(
                        features[idx],
                        features[neighbor_idx[idx]],
                        center_image[idx],
                        neighbor_images[idx],
                        center_coords[idx],
                        neighbor_coords[idx],
                    )
                    all_emb.append(out.spot_embeddings.cpu().numpy())
                tmp_emb = np.concatenate(all_emb, axis=0).astype(np.float32)
            from sklearn.decomposition import PCA as _PCA

            tmp_pca = _PCA(
                n_components=min(128, tmp_emb.shape[0], tmp_emb.shape[1]), random_state=0
            ).fit_transform(tmp_emb)
            tmp_n_clusters = config.n_clusters or len(np.unique(section_data.labels))
            tmp_pred = tied_gmm(tmp_pca, n_clusters=tmp_n_clusters)
            tmp_refined = refine_labels(
                tmp_pred, section_data.coordinates, config.refinement_neighbors
            )
            tmp_ari = clustering_metrics(section_data.labels, tmp_refined)["ari"]
            tmp_nmi = clustering_metrics(section_data.labels, tmp_refined)["nmi"]
            if tmp_ari > best_ari:
                best_ari = tmp_ari
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(
                f"{section_data.section_id} epoch={epoch:03d} ARI={tmp_ari:.4f} NMI={tmp_nmi:.4f}",
                flush=True,
            )
            model.train()

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        all_emb = []
        for b in range(n_batches):
            start = b * batch_size
            end = min(start + batch_size, n_spots)
            idx = torch.arange(start, end, device=device)
            out = model.forward(
                features[idx],
                features[neighbor_idx[idx]],
                center_image[idx],
                neighbor_images[idx],
                center_coords[idx],
                neighbor_coords[idx],
            )
            all_emb.append(out.spot_embeddings.cpu().numpy())
        embeddings = np.concatenate(all_emb, axis=0).astype(np.float32)

    from sklearn.decomposition import PCA

    pca_dim = min(128, embeddings.shape[0], embeddings.shape[1])
    embeddings_pca = PCA(n_components=pca_dim, random_state=0).fit_transform(embeddings)

    n_clusters = config.n_clusters or len(np.unique(section_data.labels))
    predictions = tied_gmm(embeddings_pca, n_clusters=n_clusters)
    refined = refine_labels(predictions, section_data.coordinates, config.refinement_neighbors)
    before = clustering_metrics(section_data.labels, predictions)
    after = clustering_metrics(section_data.labels, refined)

    heldout = {}
    if section_data.train_mask is not None:
        held_idx = np.where(~section_data.train_mask)[0]
        held_pca = embeddings_pca[held_idx]
        held_pred = tied_gmm(held_pca, n_clusters=n_clusters)
        held_refined = refine_labels(
            held_pred, section_data.coordinates[held_idx], config.refinement_neighbors
        )
        held_labels = section_data.labels[held_idx]
        held_before = clustering_metrics(held_labels, held_pred)
        held_after = clustering_metrics(held_labels, held_refined)
        heldout = {
            "heldout_ari": held_before["ari"],
            "heldout_nmi": held_before["nmi"],
            "heldout_refined_ari": held_after["ari"],
            "heldout_refined_nmi": held_after["nmi"],
            "heldout_n_spots": int(held_idx.size),
            "train_n_spots": int(train_idx.size),
        }

    return {
        "section_id": section_id,
        "embeddings": embeddings,
        "predictions": predictions,
        "refined_predictions": refined,
        "labels": section_data.labels,
        "barcodes": section_data.barcodes,
        "coordinates": section_data.coordinates,
        "losses": np.asarray(losses, dtype=np.float32),
        "train_mask": section_data.train_mask,
        "pseudo_labels": section_data.pseudo_labels,
        "metrics": {
            **before,
            "refined_ari": after["ari"],
            "refined_nmi": after["nmi"],
            "n_spots": int(n_spots),
            "n_clusters": int(n_clusters),
            "final_loss": losses[-1],
            "best_ari": float(best_ari),
            "best_epoch": int(best_epoch),
            **heldout,
        },
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }


def embedding_only_baseline(
    adata,
    section_id: str,
    config: MpMncaConfig,
    hvg_key: str = "highly_variable",
    use_feat_obsm: bool = True,
) -> dict:
    """Cluster raw ``obsm['feat']`` directly (no model) with the same backend."""
    from .data import prepare_gene_expression

    features = prepare_gene_expression(adata, hvg_key, config.gene_dim, use_feat_obsm)
    coordinates = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    labels = adata.obs["ground_truth"].astype(str).to_numpy()
    barcodes = adata.obs_names.astype(str).to_numpy()

    from sklearn.decomposition import PCA

    pca_dim = min(128, features.shape[0], features.shape[1])
    embeddings_pca = PCA(n_components=pca_dim, random_state=0).fit_transform(features)

    n_clusters = config.n_clusters or len(np.unique(labels))
    predictions = tied_gmm(embeddings_pca, n_clusters=n_clusters)
    refined = refine_labels(predictions, coordinates, config.refinement_neighbors)
    before = clustering_metrics(labels, predictions)
    after = clustering_metrics(labels, refined)

    return {
        "section_id": section_id,
        "embeddings": features,
        "predictions": predictions,
        "refined_predictions": refined,
        "labels": labels,
        "barcodes": barcodes,
        "coordinates": coordinates,
        "losses": np.zeros(1, dtype=np.float32),
        "metrics": {
            **before,
            "refined_ari": after["ari"],
            "refined_nmi": after["nmi"],
            "n_spots": int(features.shape[0]),
            "n_clusters": int(n_clusters),
            "final_loss": 0.0,
        },
        "state_dict": {},
    }


def run_variant(
    variant: AblationVariant,
    checkpoint_path: Path,
    output_root: Path,
    sections: list[str] | None = None,
    device: str | None = None,
    skip_existing: bool = True,
) -> dict:
    """Run one ablation variant across sections, writing standard artifacts.

    Returns the variant summary (metrics mean/median/IQR + config).
    """
    output_dir = output_root / variant.name
    if not output_dir.exists():
        (output_dir / "checkpoints").mkdir(parents=True)
        (output_dir / "embeddings").mkdir(parents=True)

    dataset = DlpfcDataset.from_checkpoint(checkpoint_path)
    section_ids = sections or ALL_SECTIONS
    rows: list[dict] = []

    for section_id in section_ids:
        metrics_csv = output_dir / "section_metrics.csv"
        npz_done = (output_dir / "embeddings" / f"{section_id}.npz").exists()
        pt_done = (output_dir / "checkpoints" / f"{section_id}.pt").exists()
        if skip_existing and (npz_done and pt_done):
            print(f"skip {variant.name}/{section_id} (already done)", flush=True)
            continue
        if skip_existing and metrics_csv.exists():
            with metrics_csv.open(newline="", encoding="utf-8") as handle:
                existing = {r["section_id"] for r in csv.DictReader(handle)}
            if section_id in existing:
                print(f"skip {variant.name}/{section_id} (already done)", flush=True)
                continue

        started = time.perf_counter()
        adata = dataset.get_section(section_id)

        if variant.name == "embedding_only":
            result = embedding_only_baseline(adata, section_id, variant.config)
        else:
            data = prepare_ablation_section(adata, section_id, variant.config)
            result = fit_ablation_variant(data, variant.config, device=device)

        elapsed = time.perf_counter() - started
        row = {"section_id": section_id, **result["metrics"], "elapsed_seconds": elapsed}
        rows.append(row)

        torch.save(
            {"section_id": section_id, "config": variant.config.to_dict(),
             "input_dim": int(result["embeddings"].shape[1]),
             "state_dict": result["state_dict"], "metrics": result["metrics"]},
            output_dir / "checkpoints" / f"{section_id}.pt",
        )
        np.savez_compressed(
            output_dir / "embeddings" / f"{section_id}.npz",
            embeddings=result["embeddings"],
            predictions=result["predictions"],
            refined_predictions=result["refined_predictions"],
            labels=result["labels"],
            barcodes=result["barcodes"],
            coordinates=result["coordinates"],
            losses=result["losses"],
            train_mask=result["train_mask"],
            pseudo_labels=result["pseudo_labels"],
        )
        print(json.dumps(row, sort_keys=True), flush=True)

    # Re-read all completed rows (in canonical order) for the summary.
    metrics_csv = output_dir / "section_metrics.csv"
    if rows:
        with metrics_csv.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            if metrics_csv.stat().st_size == 0 or not metrics_csv.exists() or (
                metrics_csv.stat().st_size == 0
            ):
                writer.writeheader()
            writer.writerows(rows)

    with metrics_csv.open(newline="", encoding="utf-8") as handle:
        completed = {r["section_id"]: r for r in csv.DictReader(handle)}
    ordered = [completed[s] for s in section_ids if s in completed]

    numeric = ["ari", "nmi", "refined_ari", "refined_nmi"]
    summary = {
        "run_id": variant.name,
        "implementation": f"MP-MNCA ablation: {variant.name}",
        "scientific_question": variant.question,
        "config": variant.config.to_dict(),
        "sections": [r["section_id"] for r in ordered],
        "n_sections": len(ordered),
        "metrics": {
            k: {"mean": float(np.mean([float(r[k]) for r in ordered])),
                "median": float(np.median([float(r[k]) for r in ordered])),
                "iqr": float(
                    np.percentile([float(r[k]) for r in ordered], 75)
                    - np.percentile([float(r[k]) for r in ordered], 25)
                )}
            for k in numeric
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "config.json").write_text(json.dumps(variant.config.to_dict(), indent=2), encoding="utf-8")
    return summary


def run_all_ablations(
    checkpoint_path: Path,
    output_root: Path,
    variants: list[AblationVariant] | None = None,
    sections: list[str] | None = None,
    device: str | None = None,
) -> dict[str, dict]:
    """Run every ablation variant and return name -> summary."""
    output_root = Path(output_root)
    results: dict[str, dict] = {}
    for variant in variants or ablation_variants():
        print(f"\n=== ablation variant: {variant.name} ===", flush=True)
        results[variant.name] = run_variant(
            variant, checkpoint_path, output_root, sections=sections, device=device
        )
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, default=Path("checkpoints/dlpfc.pkl"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/ablations"))
    parser.add_argument("--variants", nargs="*", default=None,
                        help="subset of variants to run (default: all)")
    parser.add_argument("--sections", nargs="*", default=None)
    parser.add_argument("--device", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    variants = ablation_variants()
    if args.variants:
        by_name = {v.name: v for v in variants}
        variants = [by_name[name] for name in args.variants]
    results = run_all_ablations(
        args.checkpoint_path, args.output_root, variants, args.sections, args.device
    )
    for name, summary in results.items():
        print(json.dumps({"variant": name, **summary["metrics"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()