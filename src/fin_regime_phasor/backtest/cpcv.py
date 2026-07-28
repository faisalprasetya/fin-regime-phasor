"""Purged K-fold and combinatorial purged cross-validation with embargo (AFML ch. 7, 12).

Used for any hyperparameter selection (alphabet size, k, hidden-state count) so the
search doesn't leak information across the serial correlation in overlapping bars.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np


def purged_kfold_splits(
    n_samples: int, n_splits: int, embargo_frac: float = 0.01
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Contiguous-block K-fold; train excludes an embargo window around each test block."""
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    embargo = int(n_samples * embargo_frac)
    indices = np.arange(n_samples)
    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=np.int64)
    fold_sizes[: n_samples % n_splits] += 1

    splits = []
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_idx = indices[start:stop]
        purge_start = max(0, start - embargo)
        purge_stop = min(n_samples, stop + embargo)
        train_idx = np.concatenate([indices[:purge_start], indices[purge_stop:]])
        splits.append((train_idx, test_idx))
        current = stop
    return splits


def combinatorial_purged_splits(
    n_samples: int, n_groups: int, n_test_groups: int, embargo_frac: float = 0.01
) -> list[tuple[np.ndarray, np.ndarray]]:
    """CPCV (Bailey & Lopez de Prado): all C(n_groups, n_test_groups) ways of picking
    test blocks out of `n_groups` contiguous blocks, purging an embargo window around
    each test block from the corresponding training set."""
    if not (0 < n_test_groups < n_groups):
        raise ValueError("0 < n_test_groups < n_groups required")

    indices = np.arange(n_samples)
    blocks = np.array_split(indices, n_groups)
    embargo = int(n_samples * embargo_frac)

    splits = []
    for test_group_ids in combinations(range(n_groups), n_test_groups):
        test_idx = np.concatenate([blocks[g] for g in test_group_ids])
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[test_idx] = False
        for g in test_group_ids:
            block = blocks[g]
            if len(block) == 0:
                continue
            lo, hi = int(block[0]), int(block[-1])
            purge_lo = max(0, lo - embargo)
            purge_hi = min(n_samples, hi + embargo + 1)
            train_mask[purge_lo:purge_hi] = False
        train_idx = indices[train_mask]
        splits.append((train_idx, test_idx))
    return splits
