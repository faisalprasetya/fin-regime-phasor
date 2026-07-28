import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fin_regime_phasor.hqmm.model import (
    filter_regime_posteriors,
    kraus_operators,
    sequence_log_likelihood,
    train_hqmm,
)


@pytest.mark.parametrize("seed", range(4))
def test_kraus_operators_satisfy_completeness_relation(seed):
    n_states, n_symbols = 3, 4
    dim = n_states * n_symbols
    key = jax.random.PRNGKey(seed)
    key_re, key_im = jax.random.split(key)
    generator = jax.random.normal(key_re, (dim, dim)) + 1j * jax.random.normal(key_im, (dim, dim))

    kraus_ops = kraus_operators(generator, n_states, n_symbols)
    completeness = sum(jnp.conj(k).T @ k for k in kraus_ops)

    np.testing.assert_allclose(np.asarray(completeness), np.eye(n_states), atol=1e-8)


def test_sequence_log_likelihood_is_finite_real():
    n_states, n_symbols = 2, 3
    dim = n_states * n_symbols
    key = jax.random.PRNGKey(0)
    key_re, key_im = jax.random.split(key)
    generator = 0.1 * (
        jax.random.normal(key_re, (dim, dim)) + 1j * jax.random.normal(key_im, (dim, dim))
    )

    symbols = jnp.array([0, 1, 2, 0, 1])
    ll = sequence_log_likelihood(generator, symbols, n_states, n_symbols)
    assert np.isfinite(float(ll))
    assert float(ll) <= 0.0 + 1e-6  # log-probabilities of a properly normalized channel are <= 0


def test_regime_posteriors_are_valid_probability_distributions():
    n_states, n_symbols = 2, 3
    dim = n_states * n_symbols
    key = jax.random.PRNGKey(1)
    key_re, key_im = jax.random.split(key)
    generator = 0.1 * (
        jax.random.normal(key_re, (dim, dim)) + 1j * jax.random.normal(key_im, (dim, dim))
    )

    symbols = jnp.array([0, 1, 2, 0, 1, 2, 0])
    posteriors = np.asarray(filter_regime_posteriors(generator, symbols, n_states, n_symbols))

    assert posteriors.shape == (7, n_states)
    np.testing.assert_allclose(posteriors.sum(axis=1), np.ones(7), atol=1e-6)
    assert np.all(posteriors >= -1e-8)


def _symbols_from_two_regimes(seed=0, n_per_regime=60):
    """Two regimes with cleanly separated symbol-emission tendencies, alternating in
    long runs so the sequence carries real temporal structure to recover."""
    rng = np.random.default_rng(seed)
    regime_a_symbols = rng.choice([0, 1], size=n_per_regime, p=[0.9, 0.1])
    regime_b_symbols = rng.choice([0, 1], size=n_per_regime, p=[0.1, 0.9])
    symbols = np.concatenate(
        [regime_a_symbols, regime_b_symbols, regime_a_symbols, regime_b_symbols]
    )
    labels = np.concatenate(
        [
            np.zeros(n_per_regime),
            np.ones(n_per_regime),
            np.zeros(n_per_regime),
            np.ones(n_per_regime),
        ]
    )
    return symbols, labels


def test_train_hqmm_decreases_loss():
    symbols, _ = _symbols_from_two_regimes()
    result = train_hqmm(symbols, n_states=2, n_symbols=2, seed=0, n_steps=150, learning_rate=0.1)
    assert result.loss_curve[-1] < result.loss_curve[0]


def test_train_hqmm_recovers_planted_regimes():
    symbols, labels = _symbols_from_two_regimes(seed=2)
    result = train_hqmm(symbols, n_states=2, n_symbols=2, seed=2, n_steps=300, learning_rate=0.1)

    agreement = max(
        np.mean(result.states == labels),
        np.mean(result.states == (1 - labels)),
    )
    assert agreement > 0.75


def test_train_hqmm_result_metrics_are_finite():
    symbols, _ = _symbols_from_two_regimes()
    result = train_hqmm(symbols, n_states=2, n_symbols=2, seed=0, n_steps=50, learning_rate=0.1)
    assert np.isfinite(result.log_likelihood)
    assert np.isfinite(result.aic)
    assert np.isfinite(result.bic)
    assert result.n_params == (2 * 2) ** 2
