"""Structured gene masking for MP-MNCA.

Implements multiple masking strategies:
1. Gene-module/pathway masking (biologically meaningful)
2. Contiguous latent/gene-token block masking
3. Whole-gene masking within a sampled neighbourhood
4. Independent scalar feature masking (fallback baseline)
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class GeneModules:
    """Gene module/pathway definitions.

    Attributes:
        module_names: List of module names.
        gene_to_module: Array of shape (n_genes,) mapping each gene index to module index.
        module_to_genes: List of arrays, each containing gene indices for that module.
    """
    module_names: list[str]
    gene_to_module: np.ndarray
    module_to_genes: list[np.ndarray]

    @classmethod
    def from_json(cls, path: Path) -> "GeneModules":
        with path.open("r") as f:
            data = json.load(f)
        module_names = data["module_names"]
        gene_to_module = np.asarray(data["gene_to_module"], dtype=np.int64)
        module_to_genes = [np.asarray(genes, dtype=np.int64) for genes in data["module_to_genes"]]
        return cls(module_names, gene_to_module, module_to_genes)

    def save(self, path: Path) -> None:
        data = {
            "module_names": self.module_names,
            "gene_to_module": self.gene_to_module.tolist(),
            "module_to_genes": [genes.tolist() for genes in self.module_to_genes],
        }
        with path.open("w") as f:
            json.dump(data, f)


def load_gene_modules(path: Optional[Path]) -> Optional[GeneModules]:
    """Load gene modules from JSON file if provided."""
    if path is None:
        return None
    return GeneModules.from_json(path)


def scalar_mask(
    x: Tensor,
    mask_rate: float,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Independent scalar feature masking (baseline).

    Args:
        x: Input tensor of shape (batch, n_genes).
        mask_rate: Fraction of features to mask per sample.
        generator: Optional random generator for reproducibility.

    Returns:
        Masked tensor with masked entries set to 0.
    """
    mask = torch.rand(x.shape, generator=generator, device=x.device) < mask_rate
    result = x.clone()
    result[mask] = 0
    return result


def contiguous_block_mask(
    x: Tensor,
    mask_rate: float,
    block_size: int = 64,
    num_blocks: int = 1,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Contiguous latent/gene-token block masking.

    Masks contiguous blocks of genes. Each block has `block_size` genes.
    Total masked fraction is approximately `mask_rate`.

    Args:
        x: Input tensor of shape (batch, n_genes).
        mask_rate: Fraction of genes to mask.
        block_size: Size of each contiguous block.
        num_blocks: Number of blocks to mask per sample.
        generator: Optional random generator.

    Returns:
        Masked tensor with masked blocks set to 0.
    """
    batch_size, n_genes = x.shape
    result = x.clone()

    # Calculate number of blocks needed to achieve mask_rate
    total_masked = int(n_genes * mask_rate)
    blocks_needed = max(1, total_masked // block_size)
    blocks_needed = min(blocks_needed, n_genes // block_size)
    blocks_per_sample = min(num_blocks, blocks_needed)

    for b in range(batch_size):
        for _ in range(blocks_per_sample):
            start = torch.randint(
                0, n_genes - block_size + 1, (1,), generator=generator, device=x.device
            ).item()
            end = start + block_size
            result[b, start:end] = 0

    return result


def gene_module_mask(
    x: Tensor,
    gene_modules: GeneModules,
    mask_rate: float,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Gene-module/pathway masking.

    Masks entire gene modules (pathways) rather than individual genes.
    This is biologically meaningful as genes in the same pathway tend to be co-regulated.

    Args:
        x: Input tensor of shape (batch, n_genes).
        gene_modules: GeneModules object containing module definitions.
        mask_rate: Fraction of modules to mask.
        generator: Optional random generator.

    Returns:
        Masked tensor with masked modules set to 0.
    """
    batch_size, n_genes = x.shape
    result = x.clone()

    n_modules = len(gene_modules.module_to_genes)
    num_modules_to_mask = max(1, int(n_modules * mask_rate))

    for b in range(batch_size):
        module_indices = torch.randperm(n_modules, generator=generator, device=x.device)[
            :num_modules_to_mask
        ]
        for mod_idx in module_indices:
            gene_indices = gene_modules.module_to_genes[mod_idx]
            result[b, gene_indices] = 0

    return result


def neighbourhood_mask(
    x: Tensor,
    neighbor_indices: Tensor,
    mask_rate: float,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Whole-gene masking within a sampled neighbourhood.

    For each spot, masks genes in its spatial neighbours. This encourages the model
    to predict a spot's expression from its neighbours' (masked) expression.

    Args:
        x: Input tensor of shape (batch, n_genes).
        neighbor_indices: Tensor of shape (batch, n_neighbors) containing neighbor indices.
        mask_rate: Fraction of neighbour genes to mask.
        generator: Optional random generator.

    Returns:
        Masked tensor.
    """
    # This is more complex - for each spot, we mask genes in its neighbours
    # For now, fall back to scalar masking as a placeholder
    # Full implementation would require the neighbour graph structure
    return scalar_mask(x, mask_rate, generator)


def apply_mask(
    x: Tensor,
    mask_type: Literal["scalar", "contiguous_block", "gene_module", "neighbourhood"],
    mask_rate: float,
    **kwargs,
) -> Tensor:
    """Apply the specified masking strategy.

    Args:
        x: Input tensor of shape (batch, n_genes).
        mask_type: Type of masking to apply.
        mask_rate: Fraction of features to mask.
        **kwargs: Additional arguments for specific masking strategies:
            - block_size, num_blocks for contiguous_block
            - gene_modules for gene_module
            - neighbor_indices for neighbourhood
            - generator for all

    Returns:
        Masked tensor.
    """
    generator = kwargs.get("generator", None)

    if mask_type == "scalar":
        return scalar_mask(x, mask_rate, generator)
    elif mask_type == "contiguous_block":
        return contiguous_block_mask(
            x, mask_rate, kwargs.get("block_size", 64), kwargs.get("num_blocks", 1), generator
        )
    elif mask_type == "gene_module":
        gene_modules = kwargs.get("gene_modules")
        if gene_modules is None:
            raise ValueError("gene_modules required for gene_module masking")
        return gene_module_mask(x, gene_modules, mask_rate, generator)
    elif mask_type == "neighbourhood":
        neighbor_indices = kwargs.get("neighbor_indices")
        if neighbor_indices is None:
            raise ValueError("neighbor_indices required for neighbourhood masking")
        return neighbourhood_mask(x, neighbor_indices, mask_rate, generator)
    else:
        raise ValueError(f"Unknown mask type: {mask_type}")


def create_mask_tensor(
    shape: tuple[int, int],
    mask_type: Literal["scalar", "contiguous_block"],
    mask_rate: float,
    block_size: int = 64,
    num_blocks: int = 1,
    generator: Optional[torch.Generator] = None,
    device: str = "cpu",
) -> Tensor:
    """Create a boolean mask tensor for the specified strategy.

    Args:
        shape: (batch_size, n_genes)
        mask_type: Type of masking.
        mask_rate: Fraction to mask.
        block_size: For contiguous block masking.
        num_blocks: Number of blocks per sample.
        generator: Random generator.
        device: Device to create the mask on.

    Returns:
        Boolean mask tensor of shape (batch_size, n_genes).
    """
    batch_size, n_genes = shape
    mask = torch.zeros(shape, dtype=torch.bool, device=device)

    if mask_type == "scalar":
        mask = torch.rand(shape, generator=generator, device=device) < mask_rate
    elif mask_type == "contiguous_block":
        total_masked = int(n_genes * mask_rate)
        blocks_needed = max(1, total_masked // block_size)
        blocks_needed = min(blocks_needed, n_genes // block_size)
        blocks_per_sample = min(num_blocks, blocks_needed)

        for b in range(batch_size):
            for _ in range(blocks_per_sample):
                start = torch.randint(
                    0, n_genes - block_size + 1, (1,), generator=generator, device=device
                ).item()
                end = start + block_size
                mask[b, start:end] = True
    else:
        raise ValueError(f"Mask tensor creation not implemented for: {mask_type}")

    return mask