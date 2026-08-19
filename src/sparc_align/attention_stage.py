"""Stage 2: MP-MNCA morphology-prior cross-attention over SPARC latents.

SPARC (stage 1) aligns the gene and image streams into a shared sparse concept
space but is a pure reconstruction objective with no spatial term, so its latents
cluster poorly on their own. This stage supplies what SPARC structurally lacks:
the k=6 spatial neighbour graph, the continuous morphology prior on the attention
logits, and the STAIG neighbour-contrastive objective that creates inter-spot
discrimination.

Two deliberate differences from ``src.mp_mnca.train_phase1.fit_phase1``:

1. **Pseudo-labels come from KMeans over image PCA**, never ``ground_truth``. The
   negative mask in ``staig/model.py`` is built from these, so sourcing them from
   the annotation makes the objective supervised on the reported metric.
2. **The training loop is not nested.** ``fit_phase1`` reuses the loop variable
   ``b`` for both its outer batch loop and its two inner full-graph loops, so every
   outer iteration recomputes the identical full-graph loss and the per-epoch cost
   is ``n_batches`` times larger than intended. Here each step does exactly one
   full-graph forward per view.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.prior_models.staig.model import mask_features, neighbor_contrastive_loss

from .config import SparcConfig
from .data import SparcSectionData
from .evaluate import cluster_sparc_latents


@dataclass(frozen=True)
class AttentionStageResult:
    embeddings: np.ndarray
    predictions: np.ndarray
    refined_predictions: np.ndarray
    metrics: dict[str, float]
    losses: list[float]


def _full_graph_forward(
    model,
    features: torch.Tensor,
    neighbor_idx: torch.Tensor,
    image: torch.Tensor,
    coords: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """One pass over every spot, batched only to bound memory."""
    outputs = []
    for start in range(0, features.shape[0], batch_size):
        end = min(start + batch_size, features.shape[0])
        index = torch.arange(start, end, device=features.device)
        neighbors = neighbor_idx[index]
        outputs.append(
            model.forward(
                features[index],
                features[neighbors],
                image[index],
                image[neighbors],
                coords[index],
                coords[neighbors],
            ).spot_embeddings
        )
    return torch.cat(outputs, dim=0)


def fit_attention_stage(
    section_data: SparcSectionData,
    latent_matrix: np.ndarray,
    config: SparcConfig,
    device: torch.device | str | None = None,
    verbose: bool = True,
) -> AttentionStageResult:
    """Train morphology-prior cross-attention + neighbour contrastive on SPARC latents.

    ``latent_matrix`` is the stage-1 output already reduced to one matrix per spot
    (see :func:`src.sparc_align.evaluate.build_cluster_input`). Its width becomes the
    attention model's ``gene_dim`` and must be divisible by ``num_heads``.
    """
    from src.mp_mnca.config import MpMncaConfig
    from src.mp_mnca.phase1 import Phase1Model

    latent_dim = latent_matrix.shape[1]
    if latent_dim % config.num_heads != 0:
        raise ValueError(
            f"latent dim {latent_dim} must be divisible by num_heads {config.num_heads}"
        )

    device = torch.device(device or "cpu")
    dtype = torch.float32 if config.dtype == "float32" else torch.float64

    attention_config = MpMncaConfig(
        gene_dim=latent_dim,
        num_heads=config.num_heads,
        temperature=config.temperature,
        mask_rate=config.mask_rate,
        image_pca_dim=config.image_pca_dim,
        image_pseudo_clusters=config.image_pseudo_clusters,
        refinement_neighbors=config.refinement_neighbors,
        epochs=config.attention_epochs,
        batch_size=config.batch_size,
        seed=config.seed,
        dtype=config.dtype,
    )

    torch.manual_seed(config.seed)
    model = Phase1Model(attention_config).to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=attention_config.learning_rate,
        weight_decay=attention_config.weight_decay,
    )

    features = torch.as_tensor(latent_matrix, dtype=dtype, device=device)
    image = torch.as_tensor(section_data.image_pca, dtype=dtype, device=device)
    coords = torch.as_tensor(section_data.coordinates, dtype=dtype, device=device)
    neighbor_idx = torch.as_tensor(section_data.neighbor_indices, dtype=torch.long, device=device)

    n_spots = features.shape[0]
    source = np.repeat(np.arange(n_spots), section_data.neighbor_indices.shape[1])
    destination = section_data.neighbor_indices.ravel()
    edge_index = torch.as_tensor(
        np.stack([source, destination], axis=0).astype(np.int64), dtype=torch.long, device=device
    )
    # Unsupervised: KMeans over image morphology, matching STAIG.
    pseudo_labels = torch.as_tensor(section_data.pseudo_labels, dtype=torch.long, device=device)

    generator = torch.Generator(device=device).manual_seed(config.seed)
    losses: list[float] = []

    for epoch in range(1, config.attention_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        view_1 = mask_features(features, config.mask_rate, generator)
        view_2 = mask_features(features, config.mask_rate, generator)
        z1 = _full_graph_forward(model, view_1, neighbor_idx, image, coords, config.batch_size)
        z2 = _full_graph_forward(model, view_2, neighbor_idx, image, coords, config.batch_size)

        loss = neighbor_contrastive_loss(z1, z2, edge_index, pseudo_labels, config.temperature)
        loss.backward()
        optimizer.step()
        losses.append(float(loss))

        if verbose and (epoch == 1 or epoch % 5 == 0 or epoch == config.attention_epochs):
            print(f"{section_data.section_id} attn epoch={epoch:03d} loss={float(loss):.6f}", flush=True)

    model.eval()
    with torch.no_grad():
        embeddings = (
            _full_graph_forward(model, features, neighbor_idx, image, coords, config.batch_size)
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    clustered = cluster_sparc_latents(
        {"attention": embeddings}, section_data.labels, section_data.coordinates, config
    )
    metrics = {
        **clustered["metrics"],
        "n_spots": int(n_spots),
        "latent_dim": int(latent_dim),
        "final_loss": losses[-1],
    }
    return AttentionStageResult(
        embeddings=embeddings,
        predictions=clustered["predictions"],
        refined_predictions=clustered["refined_predictions"],
        metrics=metrics,
        losses=losses,
    )
