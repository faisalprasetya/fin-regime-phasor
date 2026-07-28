"""Synthetic regime-switching GBM generator with ground-truth labels.

PLAN.md "Sanity check before touching real data": financial data has no ground
truth, so this is the only way to validate the HQMM/baseline fitting procedures
actually recover planted regimes before trusting output on real market data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass(frozen=True)
class SyntheticDataset:
    bars: pl.DataFrame  # open, high, low, close per bar
    regimes: np.ndarray  # ground-truth regime index per bar, shape (n_bars,)


def _sample_regime_path(
    n_bars: int, transition_matrix: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    n_states = transition_matrix.shape[0]
    regimes = np.empty(n_bars, dtype=np.int64)
    regimes[0] = rng.integers(n_states)
    for i in range(1, n_bars):
        regimes[i] = rng.choice(n_states, p=transition_matrix[regimes[i - 1]])
    return regimes


def generate_regime_switching_gbm(
    n_bars: int,
    transition_matrix: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    n_subticks: int = 20,
    s0: float = 100.0,
    seed: int = 0,
) -> SyntheticDataset:
    """Simulate a Markov regime-switching GBM at bar-level OHLC resolution.

    `transition_matrix` is (n_states, n_states) row-stochastic; `mu`/`sigma` are
    per-regime drift/volatility of log-price over one bar. Each bar's OHLC is
    built from a mini intrabar GBM path (`n_subticks` sub-steps) so the Parkinson
    estimator applied downstream recovers a noisy but consistent proxy for the
    planted regime `sigma`, rather than handing the estimator the true value directly.
    """
    transition_matrix = np.asarray(transition_matrix, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    n_states = transition_matrix.shape[0]
    if mu.shape != (n_states,) or sigma.shape != (n_states,):
        raise ValueError("mu/sigma must have one entry per regime")

    rng = np.random.default_rng(seed)
    regimes = _sample_regime_path(n_bars, transition_matrix, rng)

    log_price = np.log(s0)
    rows = []
    for i in range(n_bars):
        state = regimes[i]
        step_mu = mu[state] / n_subticks
        step_sigma = sigma[state] / np.sqrt(n_subticks)
        increments = rng.normal(loc=step_mu, scale=step_sigma, size=n_subticks)
        path = log_price + np.concatenate([[0.0], np.cumsum(increments)])

        open_price = np.exp(path[0])
        close_price = np.exp(path[-1])
        high_price = np.exp(path.max())
        low_price = np.exp(path.min())

        rows.append(
            {
                "open": float(open_price),
                "high": float(high_price),
                "low": float(low_price),
                "close": float(close_price),
            }
        )
        log_price = path[-1]

    bars = pl.DataFrame(
        rows,
        schema={"open": pl.Float64, "high": pl.Float64, "low": pl.Float64, "close": pl.Float64},
    )
    return SyntheticDataset(bars=bars, regimes=regimes)
