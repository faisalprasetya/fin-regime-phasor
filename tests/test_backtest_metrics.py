import numpy as np
import pytest

from fin_regime_phasor.backtest.metrics import (
    deflated_sharpe_ratio,
    expected_max_sharpe_under_null,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
    sharpe_ratio,
)


def test_sharpe_ratio_zero_for_zero_mean_returns():
    returns = np.array([0.01, -0.01, 0.01, -0.01])
    assert sharpe_ratio(returns) == pytest.approx(0.0, abs=1e-10)


def test_sharpe_ratio_positive_for_positive_drift():
    rng = np.random.default_rng(0)
    returns = rng.normal(loc=0.001, scale=0.01, size=1000)
    assert sharpe_ratio(returns) > 0


def test_sharpe_ratio_zero_std_returns_zero():
    returns = np.full(10, 0.01)
    assert sharpe_ratio(returns) == 0.0


def test_expected_max_sharpe_under_null_increases_with_more_trials():
    rng = np.random.default_rng(0)
    few_trials = rng.normal(scale=0.5, size=10)
    many_trials = rng.normal(scale=0.5, size=1000)
    assert expected_max_sharpe_under_null(many_trials) > expected_max_sharpe_under_null(few_trials)


def test_probabilistic_sharpe_ratio_higher_when_observed_sr_higher():
    low = probabilistic_sharpe_ratio(
        observed_sr=0.5, benchmark_sr=0.0, skew=0.0, kurtosis=3.0, n_obs=200
    )
    high = probabilistic_sharpe_ratio(
        observed_sr=1.5, benchmark_sr=0.0, skew=0.0, kurtosis=3.0, n_obs=200
    )
    assert high > low


def test_deflated_sharpe_ratio_bounded_in_unit_interval():
    rng = np.random.default_rng(0)
    returns = rng.normal(loc=0.001, scale=0.01, size=500)
    sr_trials = rng.normal(scale=0.3, size=50)
    dsr = deflated_sharpe_ratio(returns, sr_trials)
    assert 0.0 <= dsr <= 1.0


def test_deflated_sharpe_ratio_lower_with_more_trials_searched():
    rng = np.random.default_rng(0)
    returns = rng.normal(loc=0.001, scale=0.01, size=500)
    few_trials = rng.normal(scale=0.3, size=5)
    many_trials = rng.normal(scale=0.3, size=500)
    dsr_few = deflated_sharpe_ratio(returns, few_trials)
    dsr_many = deflated_sharpe_ratio(returns, many_trials)
    assert dsr_many <= dsr_few


def test_pbo_high_when_is_and_oos_performance_uncorrelated():
    rng = np.random.default_rng(0)
    n_splits, n_periods, n_trials = 20, 100, 10
    is_oos_pairs = []
    for _ in range(n_splits):
        is_returns = rng.normal(size=(n_periods, n_trials)) * 0.01
        oos_returns = (
            rng.normal(size=(n_periods, n_trials)) * 0.01
        )  # independent of IS -> no real skill
        is_oos_pairs.append((is_returns, oos_returns))
    pbo = probability_of_backtest_overfitting(is_oos_pairs)
    assert 0.0 <= pbo <= 1.0
    # with no true skill, the IS-winner should underperform OOS median roughly half the time
    assert 0.25 <= pbo <= 0.75


def test_pbo_low_when_skill_is_persistent():
    n_splits, n_periods, n_trials = 20, 100, 10
    rng = np.random.default_rng(1)
    true_skill = np.linspace(-0.02, 0.02, n_trials)  # trial n_trials-1 is consistently best
    is_oos_pairs = []
    for _ in range(n_splits):
        noise_is = rng.normal(scale=0.01, size=(n_periods, n_trials))
        noise_oos = rng.normal(scale=0.01, size=(n_periods, n_trials))
        is_returns = true_skill + noise_is
        oos_returns = true_skill + noise_oos
        is_oos_pairs.append((is_returns, oos_returns))
    pbo = probability_of_backtest_overfitting(is_oos_pairs)
    assert pbo < 0.3


def test_pbo_empty_input_returns_nan():
    assert np.isnan(probability_of_backtest_overfitting([]))
