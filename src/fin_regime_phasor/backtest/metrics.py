"""Deflated Sharpe ratio and probability of backtest overfitting (Bailey & Lopez de
Prado, 2014) -- reported instead of a single in-sample Sharpe, given how many knobs
this pipeline has (PLAN.md "Backtest discipline").
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis as _kurtosis
from scipy.stats import norm
from scipy.stats import skew as _skew

_EULER_MASCHERONI = 0.5772156649015329


def sharpe_ratio(returns: np.ndarray, periods_per_year: float = 1.0) -> float:
    returns = np.asarray(returns, dtype=np.float64)
    std = returns.std(ddof=1)
    if std < 1e-15:
        return 0.0
    return float(returns.mean() / std * np.sqrt(periods_per_year))


def expected_max_sharpe_under_null(sr_trials: np.ndarray) -> float:
    """SR0*: expected max Sharpe ratio across N independent zero-skill trials
    (Bailey & Lopez de Prado, 2014), used as the deflation benchmark."""
    sr_trials = np.asarray(sr_trials, dtype=np.float64)
    n = len(sr_trials)
    if n < 2:
        return 0.0
    sigma_sr = sr_trials.std(ddof=1)
    z1 = norm.ppf(1.0 - 1.0 / n)
    z2 = norm.ppf(1.0 - 1.0 / (n * np.e))
    return float(sigma_sr * ((1.0 - _EULER_MASCHERONI) * z1 + _EULER_MASCHERONI * z2))


def probabilistic_sharpe_ratio(
    observed_sr: float, benchmark_sr: float, skew: float, kurtosis: float, n_obs: int
) -> float:
    """PSR(SR*): probability the true Sharpe ratio exceeds `benchmark_sr`, adjusting
    for the non-normality of returns via their skew/kurtosis."""
    numerator = (observed_sr - benchmark_sr) * np.sqrt(n_obs - 1)
    denominator = np.sqrt(1.0 - skew * observed_sr + (kurtosis - 1.0) / 4.0 * observed_sr**2)
    if denominator <= 0:
        return float("nan")
    return float(norm.cdf(numerator / denominator))


def deflated_sharpe_ratio(returns: np.ndarray, sr_trials: np.ndarray) -> float:
    """DSR: PSR evaluated at the expected-max-Sharpe-under-null benchmark, deflating
    the observed Sharpe by the number of hyperparameter trials searched."""
    returns = np.asarray(returns, dtype=np.float64)
    observed_sr = sharpe_ratio(returns)
    benchmark_sr = expected_max_sharpe_under_null(sr_trials)
    skew = float(_skew(returns))
    kurt = float(_kurtosis(returns, fisher=False))
    return probabilistic_sharpe_ratio(observed_sr, benchmark_sr, skew, kurt, len(returns))


def probability_of_backtest_overfitting(
    is_oos_returns: list[tuple[np.ndarray, np.ndarray]],
) -> float:
    """PBO via CSCV (Bailey, Borwein, Lopez de Prado & Zhu, 2014).

    `is_oos_returns` is a list of (in_sample_returns_matrix, out_of_sample_returns_matrix)
    pairs, one per CPCV combination, each matrix shaped (n_periods, n_trials) -- the
    return series of every hyperparameter configuration ("trial") on that split.
    For each split: rank trials by IS Sharpe, take the IS-best trial, find its
    relative rank on OOS performance; PBO is the fraction of splits where that OOS
    rank falls at or below the median (the IS-winner doesn't even beat median OOS).
    """
    below_median_count = 0
    n_splits = len(is_oos_returns)
    if n_splits == 0:
        return float("nan")

    for is_returns, oos_returns in is_oos_returns:
        n_trials = is_returns.shape[1]
        is_sharpes = np.array([sharpe_ratio(is_returns[:, j]) for j in range(n_trials)])
        oos_sharpes = np.array([sharpe_ratio(oos_returns[:, j]) for j in range(n_trials)])

        best_trial = int(np.argmax(is_sharpes))
        oos_rank = (
            oos_sharpes < oos_sharpes[best_trial]
        ).sum() + 1  # 1-indexed rank, ties broken low
        relative_rank = oos_rank / (n_trials + 1)
        logit = np.log(relative_rank / (1.0 - relative_rank)) if 0 < relative_rank < 1 else 0.0
        if logit <= 0:
            below_median_count += 1

    return below_median_count / n_splits
