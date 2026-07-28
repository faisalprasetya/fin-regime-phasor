import numpy as np

from fin_regime_phasor.baselines.hamilton import fit_hamilton_markov_switching


def test_fit_hamilton_markov_switching_on_two_regime_returns():
    rng = np.random.default_rng(0)
    regime_a = rng.normal(loc=0.002, scale=0.005, size=200)
    regime_b = rng.normal(loc=-0.003, scale=0.02, size=200)
    log_returns = np.concatenate([regime_a, regime_b])

    result = fit_hamilton_markov_switching(log_returns, n_states=2)

    assert result.states.shape == log_returns.shape
    assert np.isfinite(result.log_likelihood)
    assert np.isfinite(result.aic)
    assert np.isfinite(result.bic)


def test_fit_hamilton_markov_switching_separates_high_low_vol_segments():
    rng = np.random.default_rng(1)
    regime_a = rng.normal(loc=0.0, scale=0.002, size=250)
    regime_b = rng.normal(loc=0.0, scale=0.03, size=250)
    log_returns = np.concatenate([regime_a, regime_b])
    labels = np.concatenate([np.zeros(250), np.ones(250)])

    result = fit_hamilton_markov_switching(log_returns, n_states=2)
    agreement = max(
        np.mean(result.states == labels),
        np.mean(result.states == (1 - labels)),
    )
    assert agreement > 0.8
