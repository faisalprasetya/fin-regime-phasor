import numpy as np

from fin_regime_phasor.benchmark import (
    cusum_agreement,
    regime_change_points,
    run_ablation_grid,
    run_full_benchmark,
)
from fin_regime_phasor.phasor import log_return, parkinson_sigma, phase_direct
from fin_regime_phasor.synthetic import generate_regime_switching_gbm


def _synthetic_features(seed=0, n_bars=150):
    transition_matrix = np.array([[0.95, 0.05], [0.05, 0.95]])
    mu = np.array([0.0005, -0.0005])
    sigma_regime = np.array([0.01, 0.04])
    dataset = generate_regime_switching_gbm(
        n_bars=n_bars, transition_matrix=transition_matrix, mu=mu, sigma=sigma_regime, seed=seed
    )
    bars = dataset.bars
    sigma = parkinson_sigma(bars["high"].to_numpy(), bars["low"].to_numpy())
    r = log_return(bars["close"].to_numpy())
    # drop the first bar (no prior return)
    sigma, r, regimes = sigma[1:], r[1:], dataset.regimes[1:]
    theta = phase_direct(50.0 * r)
    return r, sigma, theta, regimes


def test_run_ablation_grid_has_all_four_cells():
    log_returns, sigma, theta, _ = _synthetic_features()
    grid = run_ablation_grid(
        log_returns, sigma, theta, n_states=2, n_symbols=4, seed=0, hqmm_steps=40, hqmm_restarts=1
    )
    assert set(grid.keys()) == {
        ("raw", "classical"),
        ("phasor", "classical"),
        ("raw", "quantum"),
        ("phasor", "quantum"),
    }
    for cell in grid.values():
        assert np.isfinite(cell.log_likelihood)
        assert np.isfinite(cell.aic)
        assert np.isfinite(cell.bic)
        assert cell.states.shape == log_returns.shape


def test_run_full_benchmark_includes_hamilton_and_naive():
    log_returns, sigma, theta, _ = _synthetic_features()
    result = run_full_benchmark(
        log_returns, sigma, theta, n_states=2, n_symbols=4, seed=0, hqmm_steps=40, hqmm_restarts=1
    )
    assert np.isfinite(result.hamilton.log_likelihood)
    assert np.isfinite(result.naive.log_likelihood)
    assert result.cusum_breaks.dtype == np.int64


def test_regime_change_points_detects_transitions():
    states = np.array([0, 0, 0, 1, 1, 0, 0])
    change_points = regime_change_points(states)
    np.testing.assert_array_equal(change_points, np.array([3, 5]))


def test_regime_change_points_empty_for_constant_states():
    states = np.zeros(10, dtype=int)
    assert regime_change_points(states).size == 0


def test_cusum_agreement_range():
    states = np.array([0, 0, 1, 1, 1, 0, 0])
    cusum_breaks = np.array([2, 5])
    rate = cusum_agreement(states, cusum_breaks, tolerance=1)
    assert 0.0 <= rate <= 1.0
    assert rate == 1.0  # both change points (index 2, 5) are within tolerance of a break
