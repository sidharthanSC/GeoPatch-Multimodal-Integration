from types import SimpleNamespace

import numpy as np
import pytest

from src.cross_modal.train import load_external_gene_embeddings


class _Dataset:
    def __init__(self):
        self.sections = {
            "151507": SimpleNamespace(
                n_obs=2,
                obs_names=np.asarray(["bc1", "bc2"]),
                obsm={"img_emb": np.ones((2, 128), dtype=np.float32)},
            ),
            "151508": SimpleNamespace(
                n_obs=1,
                obs_names=np.asarray(["bc1"]),
                obsm={"img_emb": np.full((1, 128), 2, dtype=np.float32)},
            ),
        }

    def section_ids(self):
        return list(self.sections)

    def get_section(self, section_id):
        return self.sections[section_id]


def test_external_gene_loader_reorders_compound_identities(tmp_path):
    path = tmp_path / "features.npz"
    np.savez(
        path,
        gene_stage_128=np.asarray([[30, 31], [10, 11], [20, 21]], dtype=np.float32),
        section_ids=np.asarray(["151508", "151507", "151507"]),
        barcodes=np.asarray(["bc1", "bc1", "bc2"]),
    )
    gene, image, barcodes, sections = load_external_gene_embeddings(
        _Dataset(), path, "gene_stage_128", "img_emb"
    )
    np.testing.assert_array_equal(gene, [[10, 11], [20, 21], [30, 31]])
    assert image.shape == (3, 128)
    np.testing.assert_array_equal(barcodes, ["bc1", "bc2", "bc1"])
    np.testing.assert_array_equal(sections, ["151507", "151507", "151508"])


@pytest.mark.parametrize("problem", ["duplicate", "missing", "nonfinite"])
def test_external_gene_loader_rejects_invalid_artifacts(tmp_path, problem):
    path = tmp_path / f"{problem}.npz"
    gene = np.asarray([[10, 11], [20, 21], [30, 31]], dtype=np.float32)
    sections = np.asarray(["151507", "151507", "151508"])
    barcodes = np.asarray(["bc1", "bc2", "bc1"])
    if problem == "duplicate":
        sections[1], barcodes[1] = "151507", "bc1"
    elif problem == "missing":
        sections[2], barcodes[2] = "c", "bc9"
    else:
        gene[0, 0] = np.nan
    np.savez(path, values=gene, section_ids=sections, barcodes=barcodes)
    with pytest.raises(ValueError):
        load_external_gene_embeddings(_Dataset(), path, "values", "img_emb")
