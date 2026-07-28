import numpy as np
import pytest

from fin_regime_phasor.phasor import (
    build_phasor,
    log_return,
    parkinson_sigma,
    phase_direct,
    phase_weierstrass_components,
    phasor_closed_form,
    phasor_from_components,
)


def test_parkinson_sigma_matches_closed_form():
    high = np.array([101.0, 110.0, 100.5])
    low = np.array([99.0, 100.0, 100.0])
    sigma = parkinson_sigma(high, low)
    expected = np.sqrt((np.log(high / low)) ** 2 / (4.0 * np.log(2.0)))
    np.testing.assert_allclose(sigma, expected)


def test_parkinson_sigma_nonnegative_and_zero_at_equal_high_low():
    sigma = parkinson_sigma(np.array([100.0]), np.array([100.0]))
    assert sigma[0] == pytest.approx(0.0)


def test_log_return_first_is_nan_and_matches_diff():
    price = np.array([100.0, 105.0, 103.0, 110.0])
    r = log_return(price)
    assert np.isnan(r[0])
    np.testing.assert_allclose(r[1:], np.diff(np.log(price)))


@pytest.mark.parametrize("seed", range(5))
def test_weierstrass_components_match_direct_trig(seed):
    rng = np.random.default_rng(seed)
    w = rng.normal(scale=3.0, size=1000)

    theta_direct = phase_direct(w)
    cos_expected, sin_expected = np.cos(theta_direct), np.sin(theta_direct)

    cos_theta, sin_theta = phase_weierstrass_components(w)

    np.testing.assert_allclose(cos_theta, cos_expected, atol=1e-10)
    np.testing.assert_allclose(sin_theta, sin_expected, atol=1e-10)


@pytest.mark.parametrize("seed", range(5))
def test_closed_form_matches_components_and_direct_trig(seed):
    rng = np.random.default_rng(seed)
    sigma = rng.uniform(0.0, 5.0, size=1000)
    w = rng.normal(scale=3.0, size=1000)

    z_closed = phasor_closed_form(sigma, w)

    cos_theta, sin_theta = phase_weierstrass_components(w)
    z_components = phasor_from_components(sigma, cos_theta, sin_theta)

    theta_direct = phase_direct(w)
    z_direct = sigma * np.exp(1j * theta_direct)

    np.testing.assert_allclose(z_closed, z_components, atol=1e-10)
    np.testing.assert_allclose(z_closed, z_direct, atol=1e-8)


@pytest.mark.parametrize("seed", range(5))
def test_phasor_magnitude_equals_sigma_exactly(seed):
    rng = np.random.default_rng(seed)
    sigma = rng.uniform(0.0, 5.0, size=1000)
    w = rng.normal(scale=10.0, size=1000)

    z = phasor_closed_form(sigma, w)
    np.testing.assert_allclose(np.abs(z), sigma, atol=1e-9)


def test_phase_bounded_in_open_interval():
    w = np.array([-1e6, -1.0, 0.0, 1.0, 1e6])
    theta = phase_direct(w)
    assert np.all(theta > -np.pi) and np.all(theta < np.pi)


def test_phase_odd_and_zero_at_zero():
    theta = phase_direct(np.array([0.0]))
    assert theta[0] == pytest.approx(0.0)

    w = np.array([0.3, 1.7, 5.0])
    np.testing.assert_allclose(phase_direct(-w), -phase_direct(w))


def test_build_phasor_end_to_end():
    high = np.array([101.0, 103.0, 99.0])
    low = np.array([100.0, 100.0, 98.0])
    price = np.array([100.5, 102.0, 98.5])
    k = 50.0

    sigma = parkinson_sigma(high, low)
    r_star = log_return(price)
    # first bar has no prior return; drop it for phase construction as the pipeline would.
    z = build_phasor(sigma[1:], r_star[1:], k)

    assert z.shape == (2,)
    np.testing.assert_allclose(np.abs(z), sigma[1:], atol=1e-9)
