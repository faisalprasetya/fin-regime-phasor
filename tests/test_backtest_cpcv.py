import numpy as np
import pytest

from fin_regime_phasor.backtest.cpcv import combinatorial_purged_splits, purged_kfold_splits


def test_purged_kfold_splits_cover_all_test_indices_exactly_once():
    n = 100
    splits = purged_kfold_splits(n, n_splits=5, embargo_frac=0.0)
    all_test = np.concatenate([test for _, test in splits])
    assert sorted(all_test) == list(range(n))


def test_purged_kfold_splits_train_excludes_test_and_embargo():
    n = 100
    embargo_frac = 0.05
    splits = purged_kfold_splits(n, n_splits=5, embargo_frac=embargo_frac)
    embargo = int(n * embargo_frac)
    for train_idx, test_idx in splits:
        test_start, test_stop = test_idx.min(), test_idx.max()
        forbidden = set(range(max(0, test_start - embargo), min(n, test_stop + embargo + 1)))
        assert not (set(train_idx.tolist()) & forbidden)


def test_purged_kfold_rejects_too_few_splits():
    with pytest.raises(ValueError):
        purged_kfold_splits(100, n_splits=1)


def test_combinatorial_purged_splits_count_matches_binomial():
    from math import comb

    splits = combinatorial_purged_splits(200, n_groups=6, n_test_groups=2, embargo_frac=0.0)
    assert len(splits) == comb(6, 2)


def test_combinatorial_purged_splits_train_test_disjoint():
    splits = combinatorial_purged_splits(200, n_groups=6, n_test_groups=2, embargo_frac=0.02)
    for train_idx, test_idx in splits:
        assert not (set(train_idx.tolist()) & set(test_idx.tolist()))


def test_combinatorial_purged_splits_rejects_bad_group_counts():
    with pytest.raises(ValueError):
        combinatorial_purged_splits(100, n_groups=4, n_test_groups=4)
    with pytest.raises(ValueError):
        combinatorial_purged_splits(100, n_groups=4, n_test_groups=0)
