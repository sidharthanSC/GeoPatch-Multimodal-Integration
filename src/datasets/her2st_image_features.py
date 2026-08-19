"""STAIG's BYOL histology pipeline, ported to the HER2ST sections.

Produces ``adata.obsm["img_emb"]`` for the ``.h5ad`` files written by
``src/datasets/her2st_build.py``, using the same procedure that produced the DLPFC
``img_emb`` this repository already consumes. That procedure lives in STAIG's
``example/Fig-3b.ipynb`` (https://github.com/y-itao/STAIG) and its
``staig/adata_processing.py``; it was confirmed against the shipped DLPFC data by
reproducing ``data/151507.h5ad``'s ``obsm["img_emb"]`` from
``Dataset/DLPFC/151507/embeddings.npy`` to within float32 round-off (component signs
aside) via ``StandardScaler`` -> ``PCA(n_components=128, random_state=42)``.

Per section:

1. **Crop** a square patch of the full-resolution H&E image centred on each spot, then
   resize it to 512x512.
2. **Filter** it the way STAIG does: greyscale -> Gaussian blur -> 2-D FFT -> keep only
   the central ``[lower:upper]`` box of the shifted spectrum (a low-pass filter) ->
   inverse FFT -> magnitude -> Gaussian blur -> back to three channels. This throws
   away stain colour and fine texture, leaving coarse morphology.
3. **Train** ``byol_pytorch.BYOL`` over that section's own patches, on an
   ImageNet-pretrained ResNet-50 (``hidden_layer='avgpool'``, 256x256 inputs, Adam at
   3e-4) for a single epoch -- STAIG's ``epoch_num = 1``.
4. **Embed** every patch with the fine-tuned encoder, giving 2,048-d features, saved to
   ``<derived-dir>/<section>/embeddings.npy`` (mirroring
   ``Dataset/DLPFC/<slide>/embeddings.npy``).
5. **Reduce** the 2,048-d features to 128-d with ``StandardScaler`` +
   ``PCA(n_components=128, random_state=42)`` and write them back into the section's
   ``.h5ad`` as ``obsm["img_emb"]``.

Three deliberate deviations from STAIG's notebook, all forced by this environment or by
the difference in platform, are:

- **Patch size is measured, not hard-coded.** STAIG uses ``3.5 * 144`` = 504 px, which
  at DLPFC's 1.752 px/um works out to 287.6 um of tissue. HER2ST images are supplied at
  ~1.09 px/um (patient A) or ~1.46 px/um (patients B--H), so a fixed pixel size would
  cover a different amount of tissue per patient. ``--patch-um`` (default 287.6)
  reproduces DLPFC's *physical* field of view on every section instead.
- **float32, not float64.** The notebook calls ``.double()``; Apple's MPS backend has no
  float64. Patches near the image border are edge-padded rather than silently truncated
  to a smaller array, which STAIG's raw slicing would do.
- **Batch size 8, not 43** (see :data:`DEFAULT_BATCH_SIZE`), which is a memory
  accommodation rather than a change to the objective: BYOL has no batch-negative term,
  so batch size influences only its BatchNorm statistics.
- **Only the filtered patches are written to disk** (``--keep-raw-patches`` also writes
  the unfiltered crops), since the raw crops are an intermediate the BYOL stage never
  reads.

Usage::

    python -m src.datasets.her2st_image_features                  # all sections
    python -m src.datasets.her2st_image_features --sections A1 B1 # a subset
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import random
from multiprocessing import Pool
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import anndata as ad
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from tqdm import tqdm

logger = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = None

# STAIG's DLPFC settings: a 3.5 * 144 px patch at 96.375 px / 55 um.
DLPFC_PATCH_UM: float = 3.5 * 144.0 * 55.0 / 96.375  # ~287.6 um
PATCH_RESIZE_PX: int = 512  # size patches are stored at, as in STAIG
BYOL_IMAGE_PX: int = 256  # size BYOL consumes
# STAIG uses 43. BYOL concatenates both views into one forward pass, so 43 means 86
# 256x256 images through ResNet-50 twice (online + target) -- ~20 GB of activations,
# which thrashes a 16 GB unified-memory Mac into swap (measured: 110 s for a single
# forward, versus 0.9 s per full step at 8). BYOL's objective has no batch-negative
# term, so batch size affects only BatchNorm statistics here, not what is optimised.
DEFAULT_BATCH_SIZE: int = 8
# Bounds of the retained box in the 512x512 shifted FFT spectrum.
FFT_LOWER: int = 245
FFT_UPPER: int = 275


def resolve_device(name: str) -> torch.device:
    """Resolve ``"auto"`` to MPS or CUDA when available, else CPU."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def crop_patch(image: np.ndarray, x: float, y: float, patch_px: int) -> np.ndarray:
    """Crop one ``patch_px``-square patch centred on ``(x, y)``, resized to 512x512.

    A patch that would run off the edge of the image is edge-padded, so every patch
    comes back the same size regardless of how close its spot sits to the border.
    (STAIG's raw slicing instead returns a smaller, off-centre array there.)
    """
    height, width = image.shape[:2]
    half = patch_px // 2
    left, top = int(round(x)) - half, int(round(y)) - half
    right, bottom = left + patch_px, top + patch_px

    pad_left, pad_top = max(0, -left), max(0, -top)
    pad_right, pad_bottom = max(0, right - width), max(0, bottom - height)
    patch = image[max(0, top): min(height, bottom), max(0, left): min(width, right)]
    if pad_left or pad_top or pad_right or pad_bottom:
        patch = cv2.copyMakeBorder(
            patch, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REPLICATE
        )
    if patch.shape[0] != PATCH_RESIZE_PX or patch.shape[1] != PATCH_RESIZE_PX:
        patch = cv2.resize(
            patch, (PATCH_RESIZE_PX, PATCH_RESIZE_PX), interpolation=cv2.INTER_LINEAR
        )
    return patch


def crop_patches(image: np.ndarray, coordinates: np.ndarray, patch_px: int) -> List[np.ndarray]:
    """Crop one patch per spot. See :func:`crop_patch`."""
    return [crop_patch(image, x, y, patch_px) for x, y in coordinates]


def _custom_mask(shape: Sequence[int], lower: int, upper: int) -> np.ndarray:
    """STAIG's spectral mask: keep only the ``[lower:upper]`` box, zero elsewhere."""
    mask = np.zeros(shape, np.uint8)
    mask[lower:upper, lower:upper] = 1
    return mask


def fourier_filter(patch: np.ndarray, lower: int = FFT_LOWER, upper: int = FFT_UPPER) -> np.ndarray:
    """Apply STAIG's low-pass morphology filter to one 512x512 BGR patch."""
    grey = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    grey = cv2.GaussianBlur(grey, (5, 5), 0)

    spectrum = np.fft.fftshift(np.fft.fft2(grey))
    spectrum = spectrum * _custom_mask(grey.shape, lower, upper)
    filtered = np.abs(np.fft.ifft2(np.fft.ifftshift(spectrum)))
    filtered = cv2.GaussianBlur(filtered, (15, 15), 0)
    return cv2.cvtColor(np.float32(filtered), cv2.COLOR_GRAY2RGB)


def _filter_and_write(task: Tuple[np.ndarray, str, Optional[str]]) -> str:
    """Pool worker: filter one cropped patch and write it (and optionally the crop)."""
    patch, filtered_path, raw_path = task
    if raw_path is not None:
        cv2.imwrite(raw_path, patch)
    cv2.imwrite(filtered_path, fourier_filter(patch))
    return filtered_path


def write_filtered_patches(
    image: np.ndarray,
    coordinates: np.ndarray,
    patch_px: int,
    filtered_dir: Path,
    raw_dir: Optional[Path] = None,
    processes: Optional[int] = None,
    desc: str = "patches",
) -> List[Path]:
    """Crop, filter and write every spot's patch, filtering across a process pool.

    Patches are cropped lazily in this process and streamed to the workers, so a
    section's worth of 512x512 crops is never all resident at once. STAIG parallelises
    the same step with ``multiprocessing.Pool``.
    """
    processes = processes or max(1, (os.cpu_count() or 2) - 1)
    paths = [filtered_dir / f"{index}.png" for index in range(len(coordinates))]

    def tasks():
        for index, (x, y) in enumerate(coordinates):
            raw_path = str(raw_dir / f"{index}.png") if raw_dir is not None else None
            yield crop_patch(image, x, y, patch_px), str(paths[index]), raw_path

    with Pool(processes=processes) as pool:
        for _ in tqdm(
            pool.imap_unordered(_filter_and_write, tasks(), chunksize=4),
            total=len(paths), desc=desc, leave=False,
        ):
            pass
    return paths


class _PatchDataset(Dataset):
    """The filtered patches of one section, as 256x256 float tensors in [0, 1]."""

    def __init__(self, patch_paths: Sequence[Path]) -> None:
        self.patch_paths = list(patch_paths)
        self.transform = T.Compose([T.Resize((BYOL_IMAGE_PX, BYOL_IMAGE_PX)), T.ToTensor()])

    def __len__(self) -> int:
        return len(self.patch_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.patch_paths[index]) as image:
            return self.transform(image.convert("RGB"))


class _RandomApply(nn.Module):
    """STAIG's own probabilistic wrapper (``byol_pytorch`` ships an equivalent)."""

    def __init__(self, fn: nn.Module, p: float) -> None:
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return x
        return self.fn(x)


def _augmentation() -> nn.Module:
    """STAIG's BYOL view augmentation, unchanged."""
    return nn.Sequential(
        _RandomApply(T.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.3),
        T.RandomGrayscale(p=0.2),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        _RandomApply(T.GaussianBlur((3, 3), (1.0, 2.0)), p=0.2),
        T.RandomRotation(degrees=(0, 360)),
        T.RandomResizedCrop((BYOL_IMAGE_PX, BYOL_IMAGE_PX)),
        T.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225])),
    )


def byol_embeddings(
    patch_paths: Sequence[Path],
    device: torch.device,
    epochs: int = 1,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = 3e-4,
    seed: int = 42,
) -> np.ndarray:
    """Fine-tune BYOL on one section's patches and return its 2,048-d embeddings."""
    from byol_pytorch import BYOL  # imported lazily: heavyweight, and optional for tests

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    dataset = _PatchDataset(patch_paths)
    # BYOL refuses a batch of one (its projector is BatchNorm'd), which a section whose
    # spot count is 1 mod batch_size would otherwise hand it. Dropping that remainder
    # costs nothing: the embedding pass below is a separate, non-dropping loader, so
    # every patch is still embedded.
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=len(dataset) > batch_size,
    )

    backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    learner = BYOL(
        backbone,
        image_size=BYOL_IMAGE_PX,
        hidden_layer="avgpool",
        augment_fn=_augmentation(),
        # STAIG pinned byol-pytorch 0.6.0, which predates simplicial embeddings; this
        # repository pins 0.9.1, where they default to on. Disabled for fidelity.
        use_simplicial_embeddings=False,
    )
    learner = learner.to(device)
    optimizer = torch.optim.Adam(learner.parameters(), lr=learning_rate)

    learner.train()
    for epoch in range(epochs):
        for images in tqdm(loader, desc=f"BYOL epoch {epoch + 1}/{epochs}", leave=False):
            loss = learner(images.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            learner.update_moving_average()
        logger.info("  epoch %d/%d done (last batch loss %.4f)", epoch + 1, epochs, loss.detach().item())

    learner.eval()
    embeddings = []
    eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    with torch.no_grad():
        for images in tqdm(eval_loader, desc="embedding", leave=False):
            _, embedding = learner(images.to(device), return_embedding=True)
            embeddings.append(embedding.detach().cpu().numpy())
    stacked = np.concatenate(embeddings, axis=0).astype(np.float32)

    # A run over all 36 sections builds one learner per section; without releasing the
    # previous one (two ResNet-50s plus Adam state) the caching allocator holds onto
    # every section's memory, which a 16 GB unified-memory machine cannot absorb.
    del learner, optimizer, backbone, loader, eval_loader, dataset
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    return stacked


def reduce_to_img_emb(embeddings: np.ndarray, n_components: int = 128) -> np.ndarray:
    """STAIG's 2,048-d -> 128-d reduction: standardize, then PCA with ``random_state=42``."""
    flat = embeddings.reshape(embeddings.shape[0], -1)
    standardized = StandardScaler().fit_transform(flat)
    return PCA(n_components=n_components, random_state=42).fit_transform(standardized).astype(np.float32)


def process_section(
    h5ad_path: Path,
    her2st_root: Path,
    derived_dir: Path,
    device: torch.device,
    patch_um: float = DLPFC_PATCH_UM,
    epochs: int = 1,
    batch_size: int = DEFAULT_BATCH_SIZE,
    n_components: int = 128,
    keep_raw_patches: bool = False,
    reuse_embeddings: bool = True,
    processes: Optional[int] = None,
) -> None:
    """Run the whole pipeline for one section and write ``img_emb`` back into its file."""
    section_id = h5ad_path.stem
    adata = ad.read_h5ad(h5ad_path)
    spatial = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    scalefactors = adata.uns["spatial"][section_id]["scalefactors"]
    metadata = adata.uns["spatial"][section_id]["metadata"]

    section_dir = derived_dir / section_id
    section_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = section_dir / "embeddings.npy"

    if reuse_embeddings and embeddings_path.is_file():
        embeddings = np.load(embeddings_path)
        if embeddings.shape[0] != adata.n_obs:
            raise ValueError(
                f"Section '{section_id}': cached {embeddings_path} has {embeddings.shape[0]} rows "
                f"but the section has {adata.n_obs} spots. Delete it or pass --force."
            )
        logger.info("Section '%s': reusing cached %s", section_id, embeddings_path)
    else:
        image_path = her2st_root / "data" / "ST-imgs" / section_id[0] / section_id / metadata["image_file"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Section '{section_id}': could not read image {image_path}.")

        patch_px = int(round(patch_um * float(scalefactors["pixels_per_micron"])))
        logger.info(
            "Section '%s': %d spots, %.3f px/um -> %d px patches (%.1f um)",
            section_id, adata.n_obs, scalefactors["pixels_per_micron"], patch_px, patch_um,
        )

        filtered_dir = section_dir / "clip_image_filter"
        filtered_dir.mkdir(parents=True, exist_ok=True)
        raw_dir = section_dir / "clip_image"
        if keep_raw_patches:
            raw_dir.mkdir(parents=True, exist_ok=True)

        patch_paths = write_filtered_patches(
            image,
            spatial,
            patch_px,
            filtered_dir,
            raw_dir if keep_raw_patches else None,
            processes=processes,
            desc=f"{section_id} patches",
        )
        del image

        embeddings = byol_embeddings(
            patch_paths, device=device, epochs=epochs, batch_size=batch_size
        )
        np.save(embeddings_path, embeddings)
        logger.info("Section '%s': wrote %s %s", section_id, embeddings_path, embeddings.shape)

    adata.obsm["img_emb"] = reduce_to_img_emb(embeddings, n_components=n_components)
    # Write via a temporary file so an interrupted write cannot leave a section's
    # .h5ad truncated; the rebuild from source is expensive.
    tmp_path = h5ad_path.with_suffix(".h5ad.tmp")
    adata.write_h5ad(tmp_path)
    tmp_path.replace(h5ad_path)
    logger.info("Section '%s': attached img_emb %s to %s", section_id, adata.obsm["img_emb"].shape, h5ad_path)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--h5ad-dir", type=Path, default=Path("data/her2st"), help="Directory of <section>.h5ad files.")
    parser.add_argument("--her2st-root", type=Path, default=Path("Dataset/her2st_repo"), help="Clone of almaan/her2st.")
    parser.add_argument(
        "--derived-dir",
        type=Path,
        default=Path("Dataset/her2st"),
        help="Where patches and embeddings.npy are written (mirrors Dataset/DLPFC/<slide>/).",
    )
    parser.add_argument("--sections", nargs="*", default=None, help="Subset of section ids to process.")
    parser.add_argument("--patch-um", type=float, default=DLPFC_PATCH_UM, help="Patch field of view in microns.")
    parser.add_argument("--epochs", type=int, default=1, help="BYOL epochs per section (STAIG uses 1).")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"BYOL batch size (default {DEFAULT_BATCH_SIZE}; STAIG uses 43, which needs ~20 GB).",
    )
    parser.add_argument("--processes", type=int, default=None, help="Worker processes for patch filtering.")
    parser.add_argument("--pca-components", type=int, default=128, help="img_emb dimensionality.")
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', 'mps' or 'cuda'.")
    parser.add_argument("--keep-raw-patches", action="store_true", help="Also write the unfiltered crops.")
    parser.add_argument("--force", action="store_true", help="Recompute embeddings even if cached.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    paths = sorted(args.h5ad_dir.glob("*.h5ad"))
    if args.sections:
        wanted = set(args.sections)
        paths = [p for p in paths if p.stem in wanted]
        unknown = wanted - {p.stem for p in paths}
        if unknown:
            raise SystemExit(f"No .h5ad found for section(s): {sorted(unknown)} in {args.h5ad_dir}")
    if not paths:
        raise SystemExit(f"No .h5ad files found in {args.h5ad_dir}; run src.datasets.her2st_build first.")

    device = resolve_device(args.device)
    logger.info("Processing %d section(s) on %s", len(paths), device)
    for path in paths:
        process_section(
            path,
            her2st_root=args.her2st_root,
            derived_dir=args.derived_dir,
            device=device,
            patch_um=args.patch_um,
            epochs=args.epochs,
            batch_size=args.batch_size,
            n_components=args.pca_components,
            keep_raw_patches=args.keep_raw_patches,
            reuse_embeddings=not args.force,
            processes=args.processes,
        )


if __name__ == "__main__":
    main()
