"""Balanced same-layer / different-layer embedding pair sampling.

Each sampled pair ``(a, b)`` carries an indicator ``eta``: ``eta=0`` if ``a`` and ``b``
share the same ground-truth cortical layer, ``eta=1`` otherwise. Batches are built with
as close to an even split between the two as possible, so the training signal is not
dominated by the (typically much larger) pool of different-layer pairs.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


class BalancedPairSampler:
    """Draws balanced same-layer / different-layer index pairs from a label pool.

    Pairs are sampled with replacement across batches (indices may repeat both
    within and across calls to :meth:`sample_batch`), which is standard for
    contrastive-style training where "an epoch" is a fixed number of sampled
    batches rather than one pass over a fixed set of pairs.

    Parameters
    ----------
    labels:
        1-D array of per-sample labels (e.g. ground-truth layer strings). Indices
        into this array are what :meth:`sample_batch` returns.
    seed:
        Seed for the sampler's own random generator, independent of any global
        RNG state, for reproducible pair sequences.

    Raises
    ------
    ValueError
        If there are fewer than two samples, no label has at least two members
        (so no same-layer pair can ever be formed), or fewer than two distinct
        labels are present (so no different-layer pair can ever be formed).
    """

    def __init__(self, labels: np.ndarray, seed: int = 0) -> None:
        labels = np.asarray(labels)
        if labels.shape[0] < 2:
            raise ValueError("Need at least two labeled samples to form pairs.")

        self._rng = np.random.default_rng(seed)
        self._label_to_indices: Dict[object, np.ndarray] = {
            label: np.where(labels == label)[0] for label in np.unique(labels)
        }
        self._same_layer_labels: List[object] = [
            label for label, idx in self._label_to_indices.items() if len(idx) >= 2
        ]
        self._all_labels: List[object] = list(self._label_to_indices.keys())

        if not self._same_layer_labels:
            raise ValueError("No label has at least two samples; cannot form same-layer pairs.")
        if len(self._all_labels) < 2:
            raise ValueError("Need at least two distinct labels to form different-layer pairs.")

    def sample_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample one balanced batch of pairs.

        Parameters
        ----------
        batch_size:
            Total number of pairs to draw. Split as evenly as possible between
            same-layer and different-layer pairs (the same-layer half gets the
            extra pair when ``batch_size`` is odd).

        Returns
        -------
        Tuple of ``(idx_a, idx_b, eta)``, each a length-``batch_size`` NumPy array:
        ``idx_a``/``idx_b`` (``int64``) index into the ``labels`` array passed to
        the constructor, and ``eta`` (``float32``) is ``0.0`` for same-layer pairs
        and ``1.0`` for different-layer pairs. The batch order is shuffled.
        """
        n_same = batch_size // 2
        n_diff = batch_size - n_same

        idx_a = np.empty(batch_size, dtype=np.int64)
        idx_b = np.empty(batch_size, dtype=np.int64)
        eta = np.empty(batch_size, dtype=np.float32)

        for i in range(n_same):
            label = self._rng.choice(self._same_layer_labels)
            pool = self._label_to_indices[label]
            a, b = self._rng.choice(pool, size=2, replace=False)
            idx_a[i], idx_b[i], eta[i] = a, b, 0.0

        for i in range(n_diff):
            label_a, label_b = self._rng.choice(self._all_labels, size=2, replace=False)
            a = self._rng.choice(self._label_to_indices[label_a])
            b = self._rng.choice(self._label_to_indices[label_b])
            j = n_same + i
            idx_a[j], idx_b[j], eta[j] = a, b, 1.0

        perm = self._rng.permutation(batch_size)
        return idx_a[perm], idx_b[perm], eta[perm]
