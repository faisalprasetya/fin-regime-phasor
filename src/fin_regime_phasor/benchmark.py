"""2x2 ablation grid (representation x mechanism) + external-validity baselines.

PLAN.md "Benchmark design": isolate whether a win is attributable to the phasor
representation, the quantum mechanism, or both, by crossing raw/phasor features
against classical-HMM/HQMM mechanism, rather than reporting one classical-vs-quantum
number that conflates the two.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fin_regime_phasor.baselines.classical_hmm import fit_categorical_hmm, fit_gaussian_hmm
from fin_regime_phasor.baselines.hamilton import HamiltonFitResult, fit_hamilton_markov_switching
from fin_regime_phasor.baselines.naive import NaiveFitResult, fit_unconditional_gaussian
from fin_regime_phasor.discretize import apply_vq_codebook, fit_vq_codebook
from fin_regime_phasor.hqmm.model import HQMMFitResult, train_hqmm
from fin_regime_phasor.structural_breaks import agreement_rate, cusum_filter

REPRESENTATIONS = ("raw", "phasor")
MECHANISMS = ("classical", "quantum")


@dataclass(frozen=True)
class AblationCell:
    representation: str  # "raw" | "phasor"
    mechanism: str  # "classical" | "quantum"
    states: np.ndarray
    log_likelihood: float
    n_params: int
    aic: float
    bic: float


@dataclass(frozen=True)
class BenchmarkResult:
    grid: dict[tuple[str, str], AblationCell]
    hamilton: HamiltonFitResult
    naive: NaiveFitResult
    cusum_breaks: np.ndarray


def _cell_from_hmm(representation: str, result) -> AblationCell:
    return AblationCell(
        representation=representation,
        mechanism="classical",
        states=result.states,
        log_likelihood=result.log_likelihood,
        n_params=result.n_params,
        aic=result.aic,
        bic=result.bic,
    )


def _cell_from_hqmm(representation: str, result: HQMMFitResult) -> AblationCell:
    return AblationCell(
        representation=representation,
        mechanism="quantum",
        states=result.states,
        log_likelihood=result.log_likelihood,
        n_params=result.n_params,
        aic=result.aic,
        bic=result.bic,
    )


def run_ablation_grid(
    log_returns: np.ndarray,
    sigma: np.ndarray,
    theta: np.ndarray,
    n_states: int,
    n_symbols: int,
    seed: int = 0,
    hqmm_steps: int = 300,
    hqmm_restarts: int = 5,
) -> dict[tuple[str, str], AblationCell]:
    """Fit all four grid cells: (raw, classical)=A, (phasor, classical)=B,
    (raw, quantum)=C, (phasor, quantum)=Target. Codebooks are fit on the same
    data passed in -- callers are responsible for restricting this to the
    training fold (leakage control, PLAN.md)."""
    log_returns = np.asarray(log_returns, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)

    raw_features = np.column_stack([log_returns, sigma])
    raw_codebook = fit_vq_codebook(log_returns, sigma, n_symbols=n_symbols, random_state=seed)
    raw_symbols = apply_vq_codebook(raw_codebook, log_returns, sigma)

    phasor_codebook = fit_vq_codebook(sigma, theta, n_symbols=n_symbols, random_state=seed)
    phasor_symbols = apply_vq_codebook(phasor_codebook, sigma, theta)

    baseline_a = fit_gaussian_hmm(raw_features, n_states=n_states, random_state=seed)
    baseline_b = fit_categorical_hmm(
        phasor_symbols, n_states=n_states, n_symbols=n_symbols, random_state=seed
    )
    baseline_c = train_hqmm(
        raw_symbols,
        n_states=n_states,
        n_symbols=n_symbols,
        seed=seed,
        n_steps=hqmm_steps,
        n_restarts=hqmm_restarts,
    )
    target = train_hqmm(
        phasor_symbols,
        n_states=n_states,
        n_symbols=n_symbols,
        seed=seed,
        n_steps=hqmm_steps,
        n_restarts=hqmm_restarts,
    )

    return {
        ("raw", "classical"): _cell_from_hmm("raw", baseline_a),
        ("phasor", "classical"): _cell_from_hmm("phasor", baseline_b),
        ("raw", "quantum"): _cell_from_hqmm("raw", baseline_c),
        ("phasor", "quantum"): _cell_from_hqmm("phasor", target),
    }


def run_full_benchmark(
    log_returns: np.ndarray,
    sigma: np.ndarray,
    theta: np.ndarray,
    n_states: int,
    n_symbols: int,
    seed: int = 0,
    hqmm_steps: int = 300,
    hqmm_restarts: int = 5,
    cusum_threshold: float = 3.0,
) -> BenchmarkResult:
    """Full grid plus the Hamilton and no-regime external-validity baselines, plus
    an independent CUSUM structural-break series to cross-check regime timestamps."""
    grid = run_ablation_grid(
        log_returns,
        sigma,
        theta,
        n_states,
        n_symbols,
        seed=seed,
        hqmm_steps=hqmm_steps,
        hqmm_restarts=hqmm_restarts,
    )
    hamilton = fit_hamilton_markov_switching(log_returns, n_states=n_states)
    naive = fit_unconditional_gaussian(log_returns)
    cusum_breaks = cusum_filter(np.cumsum(log_returns), threshold=cusum_threshold)

    return BenchmarkResult(grid=grid, hamilton=hamilton, naive=naive, cusum_breaks=cusum_breaks)


def regime_change_points(states: np.ndarray) -> np.ndarray:
    """Bar indices where the decoded regime label changes from the previous bar."""
    states = np.asarray(states)
    if len(states) < 2:
        return np.array([], dtype=np.int64)
    return np.where(np.diff(states) != 0)[0] + 1


def cusum_agreement(states: np.ndarray, cusum_breaks: np.ndarray, tolerance: int = 5) -> float:
    """Fraction of a model's decoded regime switches confirmed by the independent CUSUM
    detector within `tolerance` bars (PLAN.md "Metrics")."""
    change_points = regime_change_points(states)
    return agreement_rate(cusum_breaks, change_points, tolerance=tolerance)
