import numpy as np
import pytest

from fin_regime_phasor.fracdiff import ffd_weights, frac_diff_ffd, minimum_ffd_d


def test_ffd_weights_d1_is_simple_difference():
    weights = ffd_weights(d=1.0, thres=1e-5)
    # binomial(1, k) = 0 for k >= 2, so only two nonzero terms: [-1, 1] (oldest-lag-first).
    np.testing.assert_allclose(weights, np.array([-1.0, 1.0]))


def test_frac_diff_ffd_d1_matches_np_diff():
    rng = np.random.default_rng(0)
    log_p = np.cumsum(rng.normal(scale=0.01, size=200))
    diffed = frac_diff_ffd(log_p, d=1.0)

    expected = np.diff(log_p)
    np.testing.assert_allclose(diffed[1:], expected, atol=1e-10)


def test_frac_diff_ffd_prefix_is_nan():
    series = np.arange(50.0)
    diffed = frac_diff_ffd(series, d=0.4)
    width = len(ffd_weights(0.4))
    assert np.all(np.isnan(diffed[: width - 1]))
    assert np.all(~np.isnan(diffed[width - 1 :]))


def test_ffd_weights_shrink_with_larger_thres():
    loose = ffd_weights(d=0.5, thres=1e-2)
    tight = ffd_weights(d=0.5, thres=1e-6)
    assert len(loose) <= len(tight)


def test_minimum_ffd_d_on_stationary_white_noise_is_near_zero():
    rng = np.random.default_rng(1)
    noise = rng.normal(size=2000)
    result = minimum_ffd_d(noise, d_grid=np.linspace(0.0, 1.0, 11))
    assert result.d_star <= 0.2


def test_minimum_ffd_d_on_random_walk_requires_larger_d():
    rng = np.random.default_rng(2)
    random_walk = np.cumsum(rng.normal(size=2000))
    result = minimum_ffd_d(random_walk, d_grid=np.linspace(0.0, 1.0, 11))
    assert result.d_star >= 0.3


def test_minimum_ffd_d_returns_monotonic_step_grid():
    rng = np.random.default_rng(3)
    random_walk = np.cumsum(rng.normal(size=1000))
    result = minimum_ffd_d(random_walk, d_grid=np.linspace(0.0, 1.0, 6))
    ds = [s.d for s in result.steps]
    assert ds == sorted(ds)


@pytest.mark.parametrize("d", [0.0, 0.3, 0.7, 1.0])
def test_frac_diff_ffd_output_length_matches_input(d):
    series = np.linspace(1.0, 2.0, 100)
    diffed = frac_diff_ffd(series, d=d)
    assert diffed.shape == series.shape
