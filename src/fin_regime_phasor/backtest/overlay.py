"""Regime probability as a risk-sizing overlay, not a standalone alpha signal
(PLAN.md "From regime detection to a trading strategy"): theta informs the base
trend signal, sigma feeds vol-target sizing, and the regime posterior scales
exposure up/down around that base position.
"""

from __future__ import annotations

import numpy as np


def trend_signal(log_returns: np.ndarray, window: int) -> np.ndarray:
    """Causal sign-of-trailing-sum momentum signal: signal[i] uses only
    log_returns[i-window:i], never the return realized at bar i itself."""
    log_returns = np.asarray(log_returns, dtype=np.float64)
    n = len(log_returns)
    cumsum = np.concatenate([[0.0], np.cumsum(log_returns)])
    signal = np.zeros(n)
    if n > window:
        window_sum = cumsum[window:] - cumsum[:-window]
        signal[window:] = np.sign(window_sum[:-1])
    return signal


def vol_target_size(sigma: np.ndarray, target_vol: float, leverage_cap: float = 3.0) -> np.ndarray:
    """Base position magnitude: inverse-vol scaling to a target volatility, capped."""
    sigma = np.asarray(sigma, dtype=np.float64)
    size = target_vol / np.maximum(sigma, 1e-12)
    return np.clip(size, 0.0, leverage_cap)


def base_position(
    log_returns: np.ndarray,
    sigma: np.ndarray,
    window: int,
    target_vol: float,
    leverage_cap: float = 3.0,
) -> np.ndarray:
    """Direction from `trend_signal`, magnitude from `vol_target_size`."""
    direction = trend_signal(log_returns, window)
    size = vol_target_size(sigma, target_vol, leverage_cap)
    return direction * size


def regime_risk_multiplier(
    regime_probs: np.ndarray,
    high_risk_state: int,
    low_multiplier: float = 0.2,
    high_multiplier: float = 1.0,
) -> np.ndarray:
    """Exposure multiplier in [low_multiplier, high_multiplier]: shrinks toward
    `low_multiplier` as posterior probability of the high-risk regime rises."""
    regime_probs = np.asarray(regime_probs, dtype=np.float64)
    p_high_risk = regime_probs[:, high_risk_state]
    return high_multiplier - p_high_risk * (high_multiplier - low_multiplier)


def apply_overlay(
    base_positions: np.ndarray, risk_multiplier: np.ndarray, leverage_cap: float = 3.0
) -> np.ndarray:
    return np.clip(base_positions * risk_multiplier, -leverage_cap, leverage_cap)


def strategy_returns_with_costs(
    returns: np.ndarray, positions: np.ndarray, cost_bps: float = 1.0
) -> np.ndarray:
    """Net per-bar strategy returns: position[i] (decided at bar i's close) earns
    returns[i+1], less transaction cost proportional to the bar's turnover."""
    positions = np.asarray(positions, dtype=np.float64)
    returns = np.asarray(returns, dtype=np.float64)
    trades = np.diff(positions, prepend=0.0)
    costs = np.abs(trades) * (cost_bps * 1e-4)
    gross = positions[:-1] * returns[1:]
    return gross - costs[:-1]
