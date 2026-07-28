import numpy as np
import pytest

from fin_regime_phasor.synthetic import generate_regime_switching_gbm


def test_generate_regime_switching_gbm_shapes():
    transition_matrix = np.array([[0.95, 0.05], [0.05, 0.95]])
    mu = np.array([0.001, -0.001])
    sigma = np.array([0.01, 0.05])

    dataset = generate_regime_switching_gbm(
        n_bars=300, transition_matrix=transition_matrix, mu=mu, sigma=sigma, seed=0
    )

    assert dataset.bars.height == 300
    assert dataset.regimes.shape == (300,)
    assert set(np.unique(dataset.regimes)).issubset({0, 1})


def test_generate_regime_switching_gbm_ohlc_consistency():
    transition_matrix = np.array([[0.9, 0.1], [0.1, 0.9]])
    mu = np.array([0.0, 0.0])
    sigma = np.array([0.02, 0.02])

    dataset = generate_regime_switching_gbm(
        n_bars=100, transition_matrix=transition_matrix, mu=mu, sigma=sigma, seed=1
    )
    bars = dataset.bars
    assert (bars["high"] >= bars["low"]).all()
    assert (bars["high"] >= bars["open"]).all()
    assert (bars["high"] >= bars["close"]).all()
    assert (bars["low"] <= bars["open"]).all()
    assert (bars["low"] <= bars["close"]).all()


def test_generate_regime_switching_gbm_high_vol_regime_has_larger_range():
    transition_matrix = np.array([[0.0, 1.0], [1.0, 0.0]])  # strict alternation
    mu = np.array([0.0, 0.0])
    sigma = np.array([0.001, 0.2])  # regime 1 much more volatile

    dataset = generate_regime_switching_gbm(
        n_bars=200, transition_matrix=transition_matrix, mu=mu, sigma=sigma, seed=2
    )
    bars = dataset.bars
    log_range = np.log(bars["high"] / bars["low"]).to_numpy()

    mean_range_regime0 = log_range[dataset.regimes == 0].mean()
    mean_range_regime1 = log_range[dataset.regimes == 1].mean()
    assert mean_range_regime1 > mean_range_regime0


def test_generate_regime_switching_gbm_deterministic_with_seed():
    transition_matrix = np.array([[0.9, 0.1], [0.1, 0.9]])
    mu = np.array([0.0, 0.0])
    sigma = np.array([0.01, 0.03])

    d1 = generate_regime_switching_gbm(
        n_bars=50, transition_matrix=transition_matrix, mu=mu, sigma=sigma, seed=7
    )
    d2 = generate_regime_switching_gbm(
        n_bars=50, transition_matrix=transition_matrix, mu=mu, sigma=sigma, seed=7
    )

    np.testing.assert_array_equal(d1.regimes, d2.regimes)
    assert d1.bars.equals(d2.bars)


def test_generate_regime_switching_gbm_rejects_mismatched_params():
    transition_matrix = np.array([[0.9, 0.1], [0.1, 0.9]])
    with pytest.raises(ValueError):
        generate_regime_switching_gbm(
            n_bars=10,
            transition_matrix=transition_matrix,
            mu=np.array([0.0]),
            sigma=np.array([0.01, 0.02]),
        )
