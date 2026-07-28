"""Phasor construction: magnitude (Parkinson volatility) + phase (2*arctan of scaled log return).

See PLAN.md "Constructing the Phasor" for the derivations this module implements.
"""

from __future__ import annotations

import numpy as np

_PARKINSON_SCALE = 1.0 / (2.0 * np.sqrt(np.log(2.0)))


def parkinson_sigma(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """Single-bar Parkinson volatility: sigma = ln(P_H/P_L) / (2*sqrt(ln 2))."""
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    return _PARKINSON_SCALE * np.abs(np.log(high / low))


def log_return(price: np.ndarray) -> np.ndarray:
    """R_i = ln(P_i) - ln(P_{i-1}); first element is NaN (no prior bar)."""
    log_p = np.log(np.asarray(price, dtype=np.float64))
    r = np.empty_like(log_p)
    r[0] = np.nan
    r[1:] = np.diff(log_p)
    return r


def phase_direct(scaled_return: np.ndarray) -> np.ndarray:
    """theta = 2*arctan(k*R), computed directly via `arctan` (reference implementation).

    `scaled_return` is `k*R` (already scaled by the sensitivity constant `k`).
    """
    w = np.asarray(scaled_return, dtype=np.float64)
    return 2.0 * np.arctan(w)


def phase_weierstrass_components(scaled_return: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """cos(theta), sin(theta) for theta = 2*arctan(w) via the tangent half-angle identity.

    Rational function of `w` alone (PLAN.md line 155) — no `arctan`/`sin`/`cos` evaluation.
    """
    w = np.asarray(scaled_return, dtype=np.float64)
    denom = 1.0 + w**2
    cos_theta = (1.0 - w**2) / denom
    sin_theta = (2.0 * w) / denom
    return cos_theta, sin_theta


def phasor_from_components(
    sigma: np.ndarray, cos_theta: np.ndarray, sin_theta: np.ndarray
) -> np.ndarray:
    """z = sigma * (cos(theta) + i*sin(theta)), the rectangular-form phasor."""
    sigma = np.asarray(sigma, dtype=np.float64)
    return sigma * (cos_theta + 1j * sin_theta)


def phasor_closed_form(sigma: np.ndarray, scaled_return: np.ndarray) -> np.ndarray:
    """z = sigma * (1 + i*w) / (1 - i*w), the Mobius closed form (PLAN.md line 169).

    Equivalent to `phasor_from_components` with the Weierstrass cos/sin, but computed
    with a single complex division instead of two rational real expressions — kept as
    a separate implementation so tests can cross-check the two derivations agree.
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    w = np.asarray(scaled_return, dtype=np.float64)
    numerator = 1.0 + 1j * w
    denominator = 1.0 - 1j * w
    return sigma * (numerator / denominator)


def build_phasor(sigma: np.ndarray, log_return_star: np.ndarray, k: float) -> np.ndarray:
    """z_i = sigma_i * exp(i * 2*arctan(k * R_i*)), via the closed-form Mobius transform.

    `log_return_star` is the (possibly frac-diffed) return series feeding phase; see
    PLAN.md "Fractional differentiation: where does it belong?" — magnitude and phase
    must each already be fully formed (frac-diffed on their own native series) before
    this call, since frac-diffing `z` directly would mix magnitude/phase history.
    """
    w = k * np.asarray(log_return_star, dtype=np.float64)
    return phasor_closed_form(sigma, w)
