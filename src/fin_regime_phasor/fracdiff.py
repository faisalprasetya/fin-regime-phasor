"""Fixed-width-window fractional differentiation (AFML sec. 5.4) + minimum-d search.

Applied independently to ln(P) (feeds phase, PLAN.md "Fractional differentiation:
where does it belong?") and to sigma (feeds magnitude) -- never to the combined
complex phasor z.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from statsmodels.tsa.stattools import adfuller


def ffd_weights(d: float, thres: float = 1e-5, max_size: int = 10_000) -> np.ndarray:
    """Fixed-width FFD weights, oldest lag last: w[0]=1, w[k] = -w[k-1]*(d-k+1)/k.

    Truncated once |w_k| < thres (fixed window), per AFML sec. 5.4, rather than the
    unbounded expanding-window variant (not viable for a causal bar-by-bar pipeline).
    """
    weights = [1.0]
    k = 1
    while k < max_size:
        next_w = -weights[-1] * (d - k + 1) / k
        if abs(next_w) < thres:
            break
        weights.append(next_w)
        k += 1
    return np.array(weights[::-1])


def frac_diff_ffd(series: np.ndarray, d: float, thres: float = 1e-5) -> np.ndarray:
    """Apply FFD(d) to `series`; first `len(weights)-1` entries are NaN (insufficient window)."""
    series = np.asarray(series, dtype=np.float64)
    weights = ffd_weights(d, thres=thres)
    width = len(weights)
    out = np.full(series.shape, np.nan)
    if width > len(series):
        return out
    for i in range(width - 1, len(series)):
        window = series[i - width + 1 : i + 1]
        out[i] = np.dot(weights, window)
    return out


@dataclass(frozen=True)
class DStepResult:
    d: float
    adf_stat: float
    p_value: float
    n_obs: int


@dataclass(frozen=True)
class FracDiffSearchResult:
    d_star: float
    steps: list[DStepResult]


def minimum_ffd_d(
    series: np.ndarray,
    d_grid: np.ndarray | None = None,
    thres: float = 1e-5,
    p_value_threshold: float = 0.05,
) -> FracDiffSearchResult:
    """Minimal d in `d_grid` clearing the ADF stationarity threshold, fit on `series` alone.

    `series` must already be restricted to the training fold by the caller -- this
    function performs no train/test split itself (leakage control, PLAN.md).
    """
    if d_grid is None:
        d_grid = np.linspace(0.0, 1.0, 21)

    steps: list[DStepResult] = []
    d_star: float | None = None
    for d in d_grid:
        diffed = frac_diff_ffd(series, d, thres=thres)
        valid = diffed[~np.isnan(diffed)]
        if len(valid) < 20:
            continue
        adf_stat, p_value = adfuller(valid, maxlag=1, autolag=None)[:2]
        steps.append(
            DStepResult(
                d=float(d), adf_stat=float(adf_stat), p_value=float(p_value), n_obs=len(valid)
            )
        )
        if d_star is None and p_value <= p_value_threshold:
            d_star = float(d)

    if d_star is None:
        d_star = float(d_grid[-1])

    return FracDiffSearchResult(d_star=d_star, steps=steps)
