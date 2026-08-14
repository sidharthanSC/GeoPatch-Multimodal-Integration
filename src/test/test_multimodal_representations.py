import numpy as np
import pytest

from src.multimodal.representations import (
    l2_normalize,
    normalized_mean,
    weighted_normalized_mean,
)


def test_normalized_mean_is_unit_length() -> None:
    first = np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    second = np.array([[0.0, 1.0], [1.0, -1.0]], dtype=np.float32)
    result = normalized_mean(first, second)
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1.0, atol=1e-6)


def test_zero_vectors_are_rejected() -> None:
    with pytest.raises(ValueError, match="zero vectors"):
        l2_normalize(np.zeros((2, 3), dtype=np.float32))


def test_weighted_mean_endpoints_match_modalities() -> None:
    gene = np.array([[1.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    image = np.array([[1.0, -1.0], [0.0, 1.0]], dtype=np.float32)
    np.testing.assert_allclose(weighted_normalized_mean(gene, image, 1.0), l2_normalize(gene))
    np.testing.assert_allclose(weighted_normalized_mean(gene, image, 0.0), l2_normalize(image))
