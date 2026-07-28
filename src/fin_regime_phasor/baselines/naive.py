"""No-regime baseline: single unconditional Gaussian on log-returns.

Sanity check (PLAN.md "Metrics") that any regime model beats "no regimes at all".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fin_regime_phasor.model_selection import aic, bic


@dataclass(frozen=True)
class NaiveFitResult:
    mean: float
    std: float
    log_likelihood: float
    n_params: int
    aic: float
    bic: float


def fit_unconditional_gaussian(log_returns: np.ndarray) -> NaiveFitResult:
    log_returns = np.asarray(log_returns, dtype=np.float64)
    mean = float(log_returns.mean())
    std = float(log_returns.std(ddof=0))
    n_obs = log_returns.shape[0]

    log_likelihood = float(
        np.sum(-0.5 * np.log(2.0 * np.pi * std**2) - 0.5 * ((log_returns - mean) / std) ** 2)
    )
    n_params = 2  # mean, std
    return NaiveFitResult(
        mean=mean,
        std=std,
        log_likelihood=log_likelihood,
        n_params=n_params,
        aic=aic(log_likelihood, n_params),
        bic=bic(log_likelihood, n_params, n_obs),
    )
