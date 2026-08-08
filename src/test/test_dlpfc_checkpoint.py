"""Quick smoke test: load the DlpfcDataset checkpoint and walk every sample.

Skipped automatically if no checkpoint exists yet — run
``DlpfcDataset.build_or_load(root=".", checkpoint_path="checkpoints/dlpfc.pkl")``
once to produce one.
"""

from pathlib import Path

import pytest
import torch

from src.datasets.dlpfc import DlpfcDataset

CHECKPOINT_PATH = Path(__file__).resolve().parent.parent.parent / "checkpoints" / "dlpfc.pkl"

pytestmark = pytest.mark.skipif(
    not CHECKPOINT_PATH.is_file(),
    reason=f"No checkpoint found at {CHECKPOINT_PATH}.",
)

_EXPECTED_KEYS = {"x", "sp", "img", "gt", "did", "sid", "img_meta"}


@pytest.fixture(scope="module")
def dataset() -> DlpfcDataset:
    return DlpfcDataset.from_checkpoint(CHECKPOINT_PATH)


def test_checkpoint_reports_all_twelve_sections(dataset: DlpfcDataset) -> None:
    assert len(dataset.section_ids()) == 12
    assert len(dataset) == sum(dataset.spot_counts().values())


def test_every_sample_dict_has_expected_keys_and_shapes(dataset: DlpfcDataset) -> None:
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

        assert sample["did"] in (1, 2, 3)
        assert sample["sid"] in dataset.section_ids()
        assert dataset.donor_id(sample["sid"]) == sample["did"]

        assert isinstance(sample["img_meta"], dict)

        seen_sections.add(sample["sid"])

    assert seen_sections == set(dataset.section_ids())
