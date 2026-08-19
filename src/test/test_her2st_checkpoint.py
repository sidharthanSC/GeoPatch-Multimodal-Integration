"""Quick smoke test: load the Her2stDataset checkpoint and walk every sample.

Skipped automatically if no checkpoint exists yet — run
``Her2stDataset.build_or_load(root=".", checkpoint_path="checkpoints/her2st.pkl")``
once to produce one (which in turn needs ``python -m src.datasets.her2st_build`` and
``python -m src.datasets.her2st_image_features`` to have populated ``data/her2st/``).
"""

from pathlib import Path

import pandas as pd
import pytest
import torch

from src.datasets.her2st import ANNOTATED_SECTIONS, GROUND_TRUTH_CATEGORIES, Her2stDataset

CHECKPOINT_PATH = Path(__file__).resolve().parent.parent.parent / "checkpoints" / "her2st.pkl"

pytestmark = pytest.mark.skipif(
    not CHECKPOINT_PATH.is_file(),
    reason=f"No checkpoint found at {CHECKPOINT_PATH}.",
)

_EXPECTED_KEYS = {"x", "sp", "img", "gt", "did", "sid", "img_meta"}


@pytest.fixture(scope="module")
def dataset() -> Her2stDataset:
    return Her2stDataset.from_checkpoint(CHECKPOINT_PATH)


def test_checkpoint_reports_all_thirty_six_sections(dataset: Her2stDataset) -> None:
    assert len(dataset.section_ids()) == 36
    assert len(dataset) == sum(dataset.spot_counts().values())


def test_every_sample_dict_has_expected_keys_and_shapes(dataset: Her2stDataset) -> None:
    """Iterate every spot in the dataset and sanity-check its sample dict."""
    n_genes = dataset.n_genes
    seen_sections = set()

    for idx in range(len(dataset)):
        sample = dataset[idx]

        assert set(sample.keys()) == _EXPECTED_KEYS

        assert isinstance(sample["x"], torch.Tensor)
        assert sample["x"].shape == (n_genes,)
        assert torch.isfinite(sample["x"]).all()

        assert isinstance(sample["sp"], torch.Tensor)
        assert sample["sp"].shape == (2,)

        assert isinstance(sample["img"], torch.Tensor)
        assert sample["img"].ndim == 1

        assert sample["did"] in range(1, 9)
        assert sample["sid"] in dataset.section_ids()
        assert dataset.patient_id(sample["sid"]) == sample["did"]
        assert dataset.donor_id(sample["sid"]) == sample["did"]

        assert isinstance(sample["img_meta"], dict)

        seen_sections.add(sample["sid"])

    assert seen_sections == set(dataset.section_ids())


def test_image_embeddings_are_128_dimensional(dataset: Her2stDataset) -> None:
    """STAIG's PCA reduction fixes ``img_emb`` at 128-d, as it is for DLPFC."""
    for section_id in dataset.section_ids():
        adata = dataset.get_section(section_id)
        assert adata.obsm["img_emb"].shape == (adata.n_obs, 128)


def test_ground_truth_exists_exactly_for_the_annotated_sections(dataset: Her2stDataset) -> None:
    """Only 8 of the 36 sections were annotated by a pathologist."""
    for section_id in dataset.section_ids():
        labels = dataset.get_section(section_id).obs["ground_truth"]
        n_labelled = int(labels.notna().sum())
        if section_id in ANNOTATED_SECTIONS:
            assert n_labelled == len(labels), f"{section_id} should be fully annotated"
            assert set(labels.dropna().unique()) <= set(GROUND_TRUTH_CATEGORIES)
        else:
            assert n_labelled == 0, f"{section_id} is not an annotated section"


def test_annotated_indices_match_the_annotated_sections(dataset: Her2stDataset) -> None:
    indices = dataset.annotated_indices()
    expected = sum(dataset.spot_counts()[s] for s in ANNOTATED_SECTIONS)
    assert len(indices) == expected
    for idx in (indices[0], indices[len(indices) // 2], indices[-1]):
        sample = dataset[int(idx)]
        assert sample["sid"] in ANNOTATED_SECTIONS
        assert pd.notna(sample["gt"])


def test_all_sections_share_one_gene_index(dataset: Her2stDataset) -> None:
    """A single var index across sections is what makes the flat dataset coherent."""
    reference = dataset.get_section(dataset.section_ids()[0]).var_names
    for section_id in dataset.section_ids()[1:]:
        assert dataset.get_section(section_id).var_names.equals(reference)


def test_spatial_coordinates_fall_inside_their_section_image(dataset: Her2stDataset) -> None:
    """Guards the coordinate-space assumption: spot pixels index the H&E image itself."""
    for section_id in dataset.section_ids():
        adata = dataset.get_section(section_id)
        metadata = adata.uns["spatial"][section_id]["metadata"]
        spatial = adata.obsm["spatial"]
        assert spatial[:, 0].min() >= 0 and spatial[:, 1].min() >= 0
        assert spatial[:, 0].max() <= metadata["fullres_width"]
        assert spatial[:, 1].max() <= metadata["fullres_height"]
