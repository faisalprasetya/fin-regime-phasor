import numpy as np
import pytest

from fin_regime_phasor.model_selection import aic, bic


def test_aic_matches_definition():
    assert aic(log_likelihood=-100.0, n_params=5) == pytest.approx(2 * 100.0 + 2 * 5)


def test_bic_matches_definition():
    ll, k, n = -100.0, 5, 200
    assert bic(ll, k, n) == pytest.approx(2 * 100.0 + k * np.log(n))


def test_bic_penalizes_more_than_aic_for_large_n():
    ll, k, n = -100.0, 5, 1000
    assert bic(ll, k, n) > aic(ll, k)
