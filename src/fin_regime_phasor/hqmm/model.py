"""HQMM: Kraus-operator quantum channel over a density-matrix hidden state.

PLAN.md "HQMM formalism": state rho in C^{n x n} (n = number of regimes), one
Kraus operator K_a per discrete observation symbol a, updated as
rho -> K_a rho K_a^dagger / Tr(K_a rho K_a^dagger), with Tr(...) = P(a | rho).

Kraus operators are sliced from one big unitary (Stinespring dilation) fit by
unconstrained gradient descent, sidestepping the completeness constraint
sum_a K_a^dagger K_a = I at every optimizer step (Srinivasan, Gordon & Boots, 2018).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import optax

from fin_regime_phasor.model_selection import aic, bic


def _skew_hermitian(g: jnp.ndarray) -> jnp.ndarray:
    """S = (G - G^dagger)/2 is skew-Hermitian for any square complex G, so expm(S) is
    always unitary -- this is what lets the generator be fit unconstrained."""
    return (g - jnp.conj(g).T) / 2.0


def _expm_skew_hermitian(s: jnp.ndarray) -> jnp.ndarray:
    """expm(S) for skew-Hermitian S via eigh(-i*S) rather than `jax.scipy.linalg.expm`.

    `jax.scipy.linalg.expm`'s scaling-and-squaring Pade approximant needs a complex128
    `lu`/`triangular_solve`, whose TPU lowering hits an internal XLA RET_CHECK (layout
    mismatch in the double-precision-emulation custom call) -- CPU/GPU LAPACK/cuSolver
    paths don't have this bug, but `eigh` is exact for Hermitian input and its TPU
    fallback is a pure-matmul/QR algorithm with no such custom call, so it sidesteps
    the bug on every backend instead of only working around it on two of three.
    S skew-Hermitian => H = -i*S is Hermitian, and expm(S) = expm(i*H) = V exp(i*lambda) V^dagger.
    """
    h = -1j * s
    eigvals, eigvecs = jnp.linalg.eigh(h)
    return (eigvecs * jnp.exp(1j * eigvals)) @ jnp.conj(eigvecs).T


def kraus_operators(generator: jnp.ndarray, n_states: int, n_symbols: int) -> jnp.ndarray:
    """Slice `n_symbols` (n_states x n_states) Kraus operators out of the dilation unitary.

    Row-block a, first n_states columns (ancilla input fixed at |0>): this is exactly
    a Stinespring dilation, so sum_a K_a^dagger K_a = I holds by construction.
    """
    dim = n_states * n_symbols
    s = _skew_hermitian(generator.reshape(dim, dim))
    u = _expm_skew_hermitian(s)
    return u[:, :n_states].reshape(n_symbols, n_states, n_states)


def _step(
    rho: jnp.ndarray, symbol: jnp.ndarray, kraus_ops: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    k = kraus_ops[symbol]
    unnorm = k @ rho @ jnp.conj(k).T
    p = jnp.clip(jnp.real(jnp.trace(unnorm)), 1e-12, None)
    rho_next = unnorm / p
    return rho_next, p


def sequence_log_likelihood(
    generator: jnp.ndarray, symbols: jnp.ndarray, n_states: int, n_symbols: int
) -> jnp.ndarray:
    """Total log P(symbols) = sum_t log Tr(K_{a_t} rho_{t-1} K_{a_t}^dagger)."""
    kraus_ops = kraus_operators(generator, n_states, n_symbols)
    rho0 = jnp.eye(n_states, dtype=jnp.complex128) / n_states

    def scan_fn(rho, symbol):
        rho_next, p = _step(rho, symbol, kraus_ops)
        return rho_next, jnp.log(p)

    _, log_probs = jax.lax.scan(scan_fn, rho0, symbols)
    return jnp.sum(log_probs)


def filter_regime_posteriors(
    generator: jnp.ndarray, symbols: jnp.ndarray, n_states: int, n_symbols: int
) -> jnp.ndarray:
    """diag(rho_t) after each observation -- the natural classical regime posterior,
    since the diagonal of a density matrix in the computational basis is exactly a
    probability distribution over the `n_states` basis states.
    """
    kraus_ops = kraus_operators(generator, n_states, n_symbols)
    rho0 = jnp.eye(n_states, dtype=jnp.complex128) / n_states

    def scan_fn(rho, symbol):
        rho_next, _ = _step(rho, symbol, kraus_ops)
        return rho_next, jnp.real(jnp.diag(rho_next))

    _, diag_probs = jax.lax.scan(scan_fn, rho0, symbols)
    return diag_probs


@dataclass(frozen=True)
class HQMMFitResult:
    generator: np.ndarray
    n_states: int
    n_symbols: int
    regime_posteriors: np.ndarray  # (T, n_states)
    states: np.ndarray  # (T,) argmax regime per bar
    log_likelihood: float
    n_params: int
    aic: float
    bic: float
    loss_curve: list[float]


def _train_single_run(
    symbols_arr: jnp.ndarray,
    n_states: int,
    n_symbols: int,
    seed: int,
    n_steps: int,
    learning_rate: float,
) -> tuple[jnp.ndarray, list[float]]:
    dim = n_states * n_symbols
    key = jax.random.PRNGKey(seed)
    key_re, key_im = jax.random.split(key)
    generator = jax.random.normal(key_re, (dim, dim), dtype=jnp.float64) + 1j * jax.random.normal(
        key_im, (dim, dim), dtype=jnp.float64
    )
    generator = generator * 0.1

    n_obs = symbols_arr.shape[0]

    def loss_fn(g):
        return -sequence_log_likelihood(g, symbols_arr, n_states, n_symbols) / n_obs

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(generator)

    @jax.jit
    def train_step(g, opt_state):
        loss, grad = jax.value_and_grad(loss_fn)(g)
        updates, opt_state = optimizer.update(grad, opt_state, g)
        g = optax.apply_updates(g, updates)
        return g, opt_state, loss

    loss_curve = []
    for _ in range(n_steps):
        generator, opt_state, loss = train_step(generator, opt_state)
        loss_curve.append(float(loss))

    return generator, loss_curve


def train_hqmm(
    symbols: np.ndarray,
    n_states: int,
    n_symbols: int,
    seed: int = 0,
    n_steps: int = 500,
    learning_rate: float = 0.05,
    n_restarts: int = 5,
) -> HQMMFitResult:
    """Fit Kraus operators by minimizing per-step negative log-likelihood via Adam.

    No closed-form M-step exists for Kraus-operator parameters (PLAN.md "Training
    objective"), so this uses gradient descent rather than classical Baum-Welch EM.
    Gradient descent on this non-convex objective is as prone to local optima as
    classical Baum-Welch, so this restarts from `n_restarts` seeds and keeps the
    best-log-likelihood fit (mirrors `baselines.classical_hmm`'s multi-restart).
    """
    symbols_arr = jnp.asarray(np.asarray(symbols, dtype=np.int64))
    dim = n_states * n_symbols
    n_obs = symbols_arr.shape[0]
    rng = np.random.default_rng(seed)

    best_generator, best_loss_curve, best_ll = None, None, -np.inf
    for run_seed in rng.integers(0, np.iinfo(np.int32).max, size=n_restarts):
        generator, loss_curve = _train_single_run(
            symbols_arr, n_states, n_symbols, int(run_seed), n_steps, learning_rate
        )
        log_likelihood = float(sequence_log_likelihood(generator, symbols_arr, n_states, n_symbols))
        if log_likelihood > best_ll:
            best_generator, best_loss_curve, best_ll = generator, loss_curve, log_likelihood

    posteriors = np.asarray(
        filter_regime_posteriors(best_generator, symbols_arr, n_states, n_symbols)
    )
    states = posteriors.argmax(axis=1)

    n_params = (
        dim * dim
    )  # dimension of the u(dim) Lie algebra: the effective dof of expm(skew-Hermitian(.))
    return HQMMFitResult(
        generator=np.asarray(best_generator),
        n_states=n_states,
        n_symbols=n_symbols,
        regime_posteriors=posteriors,
        states=states,
        log_likelihood=best_ll,
        n_params=n_params,
        aic=aic(best_ll, n_params),
        bic=bic(best_ll, n_params, n_obs),
        loss_curve=best_loss_curve,
    )
