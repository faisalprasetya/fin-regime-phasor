"""Shared model-selection metrics (AIC/BIC) for the benchmark grid.

BIC-style penalties are known to under-select regime count in Markov-switching
models (Psaradakis & Spagnolo, 2003); PLAN.md caps the state-count search to
n in {2,3,4} rather than trusting BIC over an unbounded grid.
"""

from __future__ import annotations

import numpy as np


def aic(log_likelihood: float, n_params: int) -> float:
    return -2.0 * log_likelihood + 2.0 * n_params


def bic(log_likelihood: float, n_params: int, n_obs: int) -> float:
    return -2.0 * log_likelihood + n_params * np.log(n_obs)
