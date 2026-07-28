"""Hamilton (1989) Markov-switching model baseline -- the canonical econometric
regime-switching reference (PLAN.md "Benchmark design"), included so the comparison
isn't only against other ML models.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

from fin_regime_phasor.model_selection import aic, bic


@dataclass(frozen=True)
class HamiltonFitResult:
    result: object
    states: np.ndarray
    log_likelihood: float
    n_params: int
    aic: float
    bic: float


def fit_hamilton_markov_switching(log_returns: np.ndarray, n_states: int) -> HamiltonFitResult:
    """Fit a switching-mean, switching-variance Markov regression on log-returns."""
    log_returns = np.asarray(log_returns, dtype=np.float64)
    model = MarkovRegression(log_returns, k_regimes=n_states, switching_variance=True)
    fit_result = model.fit()

    smoothed_probs = np.asarray(fit_result.smoothed_marginal_probabilities)
    states = smoothed_probs.argmax(axis=1)
    log_likelihood = float(fit_result.llf)
    n_params = int(fit_result.params.shape[0])
    n_obs = log_returns.shape[0]

    return HamiltonFitResult(
        result=fit_result,
        states=states,
        log_likelihood=log_likelihood,
        n_params=n_params,
        aic=aic(log_likelihood, n_params),
        bic=bic(log_likelihood, n_params, n_obs),
    )
