import numpy as np
import pytest

from fin_regime_phasor.backtest.overlay import (
    apply_overlay,
    base_position,
    regime_risk_multiplier,
    strategy_returns_with_costs,
    trend_signal,
    vol_target_size,
)


def test_trend_signal_causal_no_lookahead():
    # a single huge return at the very last bar must not affect any earlier signal
    log_returns = np.zeros(50)
    log_returns[-1] = 100.0
    signal_a = trend_signal(log_returns, window=5)

    log_returns_b = log_returns.copy()
    log_returns_b[-1] = -100.0
    signal_b = trend_signal(log_returns_b, window=5)

    np.testing.assert_array_equal(signal_a[:-1], signal_b[:-1])


def test_trend_signal_detects_uptrend():
    log_returns = np.full(30, 0.01)
    signal = trend_signal(log_returns, window=5)
    assert np.all(signal[5:] == 1.0)


def test_trend_signal_detects_downtrend():
    log_returns = np.full(30, -0.01)
    signal = trend_signal(log_returns, window=5)
    assert np.all(signal[5:] == -1.0)


def test_vol_target_size_inversely_proportional_to_sigma():
    sigma = np.array([0.01, 0.02, 0.04])
    size = vol_target_size(sigma, target_vol=0.02, leverage_cap=100.0)
    np.testing.assert_allclose(size, np.array([2.0, 1.0, 0.5]))


def test_vol_target_size_respects_leverage_cap():
    sigma = np.array([0.0001])
    size = vol_target_size(sigma, target_vol=0.02, leverage_cap=3.0)
    assert size[0] == pytest.approx(3.0)


def test_base_position_combines_direction_and_size():
    log_returns = np.full(30, 0.01)
    sigma = np.full(30, 0.02)
    position = base_position(log_returns, sigma, window=5, target_vol=0.02, leverage_cap=10.0)
    assert np.all(position[5:] == pytest.approx(1.0))


def test_regime_risk_multiplier_shrinks_in_high_risk_regime():
    regime_probs = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    multiplier = regime_risk_multiplier(
        regime_probs, high_risk_state=1, low_multiplier=0.2, high_multiplier=1.0
    )
    np.testing.assert_allclose(multiplier, np.array([1.0, 0.6, 0.2]))


def test_apply_overlay_respects_leverage_cap():
    base = np.array([2.0, -2.0])
    multiplier = np.array([2.0, 2.0])
    overlaid = apply_overlay(base, multiplier, leverage_cap=3.0)
    np.testing.assert_allclose(overlaid, np.array([3.0, -3.0]))


def test_strategy_returns_with_costs_zero_position_gives_zero_return():
    returns = np.array([0.01, -0.02, 0.03, 0.01])
    positions = np.zeros(4)
    strat_returns = strategy_returns_with_costs(returns, positions, cost_bps=1.0)
    np.testing.assert_allclose(strat_returns, np.zeros(3))


def test_strategy_returns_with_costs_matches_gross_minus_cost():
    returns = np.array([0.0, 0.01, -0.01])
    positions = np.array([1.0, 1.0, 0.0])
    strat_returns = strategy_returns_with_costs(returns, positions, cost_bps=10.0)
    # trades: [1.0, 0.0, -1.0]; costs = |trade|*10bps: [1e-3, 0, 1e-3]
    # gross = positions[:-1]*returns[1:] = [1.0*0.01, 1.0*-0.01] = [0.01, -0.01]
    # net = gross - costs[:-1] = [0.01 - 1e-3, -0.01 - 0]
    np.testing.assert_allclose(strat_returns, np.array([0.009, -0.01]))


def test_strategy_returns_with_costs_penalizes_higher_turnover():
    returns = np.array([0.0, 0.0, 0.0])
    low_turnover = np.array([1.0, 1.0, 1.0])
    high_turnover = np.array([1.0, -1.0, 1.0])
    low_cost = strategy_returns_with_costs(returns, low_turnover, cost_bps=10.0)
    high_cost = strategy_returns_with_costs(returns, high_turnover, cost_bps=10.0)
    assert high_cost.sum() < low_cost.sum()
