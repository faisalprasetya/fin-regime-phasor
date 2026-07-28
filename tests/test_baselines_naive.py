import numpy as np
import pytest

from fin_regime_phasor.baselines.naive import fit_unconditional_gaussian


def test_fit_unconditional_gaussian_recovers_mean_and_std():
    rng = np.random.default_rng(0)
    log_returns = rng.normal(loc=0.001, scale=0.02, size=5000)
    result = fit_unconditional_gaussian(log_returns)

    assert result.mean == pytest.approx(0.001, abs=0.005)
    assert result.std == pytest.approx(0.02, abs=0.005)
    assert result.n_params == 2


def test_fit_unconditional_gaussian_log_likelihood_matches_normal_pdf():
    log_returns = np.array([0.01, -0.01, 0.02, -0.02, 0.0])
    result = fit_unconditional_gaussian(log_returns)

    expected = np.sum(
        -0.5 * np.log(2 * np.pi * result.std**2)
        - 0.5 * ((log_returns - result.mean) / result.std) ** 2
    )
    assert result.log_likelihood == pytest.approx(expected)
