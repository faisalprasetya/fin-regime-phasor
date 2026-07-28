"""Classical HMM baselines (`hmmlearn`) for the representation x mechanism ablation grid.

- `fit_gaussian_hmm`: Baseline A, raw features (R, sigma) as two real numbers with
  Gaussian emissions.
- `fit_categorical_hmm`: Baseline B, phasor features (sigma, theta) discretized to
  a finite alphabet (see `discretize.py`) with categorical emissions.

`hmmlearn` has no notion of complex amplitudes or Kraus/unitary evolution (PLAN.md
"Why not hmmlearn") -- these are strictly the classical-mechanism half of the grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from hmmlearn.hmm import CategoricalHMM, GaussianHMM

from fin_regime_phasor.model_selection import aic, bic


@dataclass(frozen=True)
class HMMFitResult:
    model: GaussianHMM | CategoricalHMM
    states: np.ndarray
    log_likelihood: float
    n_params: int
    aic: float
    bic: float


def _gaussian_hmm_n_params(n_states: int, n_dims: int) -> int:
    startprob = n_states - 1
    transmat = n_states * (n_states - 1)
    means = n_states * n_dims
    covars = n_states * n_dims * (n_dims + 1) // 2  # full covariance
    return startprob + transmat + means + covars


def _categorical_hmm_n_params(n_states: int, n_symbols: int) -> int:
    startprob = n_states - 1
    transmat = n_states * (n_states - 1)
    emissions = n_states * (n_symbols - 1)
    return startprob + transmat + emissions


def fit_gaussian_hmm(
    features: np.ndarray,
    n_states: int,
    random_state: int = 0,
    n_iter: int = 200,
    n_restarts: int = 5,
) -> HMMFitResult:
    """Baseline A: continuous 2D Gaussian-emission HMM on raw (R, sigma) features.

    Baum-Welch is only guaranteed to find a local optimum, so this restarts from
    `n_restarts` distinct initializations and keeps the best-log-likelihood fit.
    """
    features = np.asarray(features, dtype=np.float64)
    rng = np.random.default_rng(random_state)

    best_model, best_ll = None, -np.inf
    for seed in rng.integers(0, np.iinfo(np.int32).max, size=n_restarts):
        model = GaussianHMM(
            n_components=n_states, covariance_type="full", n_iter=n_iter, random_state=int(seed)
        )
        model.fit(features)
        log_likelihood = float(model.score(features))
        if log_likelihood > best_ll:
            best_model, best_ll = model, log_likelihood

    states = best_model.predict(features)
    n_params = _gaussian_hmm_n_params(n_states, features.shape[1])
    n_obs = features.shape[0]
    return HMMFitResult(
        model=best_model,
        states=states,
        log_likelihood=best_ll,
        n_params=n_params,
        aic=aic(best_ll, n_params),
        bic=bic(best_ll, n_params, n_obs),
    )


def fit_categorical_hmm(
    symbols: np.ndarray,
    n_states: int,
    n_symbols: int,
    random_state: int = 0,
    n_iter: int = 200,
    n_restarts: int = 5,
) -> HMMFitResult:
    """Baseline B: categorical-emission HMM on VQ-discretized phasor symbols."""
    symbols = np.asarray(symbols, dtype=np.int64).reshape(-1, 1)
    rng = np.random.default_rng(random_state)

    best_model, best_ll = None, -np.inf
    for seed in rng.integers(0, np.iinfo(np.int32).max, size=n_restarts):
        model = CategoricalHMM(
            n_components=n_states, n_features=n_symbols, n_iter=n_iter, random_state=int(seed)
        )
        model.fit(symbols)
        log_likelihood = float(model.score(symbols))
        if log_likelihood > best_ll:
            best_model, best_ll = model, log_likelihood

    states = best_model.predict(symbols)
    n_params = _categorical_hmm_n_params(n_states, n_symbols)
    n_obs = symbols.shape[0]
    return HMMFitResult(
        model=best_model,
        states=states,
        log_likelihood=best_ll,
        n_params=n_params,
        aic=aic(best_ll, n_params),
        bic=bic(best_ll, n_params, n_obs),
    )
