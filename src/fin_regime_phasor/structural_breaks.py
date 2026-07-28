"""Symmetric CUSUM filter (AFML ch. 17) -- a structural-break detector independent of
either regime model, used as an external check on detected regime-switch timestamps
(PLAN.md "Metrics").
"""

from __future__ import annotations

import numpy as np


def cusum_filter(series: np.ndarray, threshold: float) -> np.ndarray:
    """Indices where the cumulative sum of deviations from the running mean step
    exceeds +/- `threshold`, resetting the accumulator to 0 after each event.
    """
    series = np.asarray(series, dtype=np.float64)
    diffs = np.diff(series)
    mean_step = diffs.mean() if len(diffs) else 0.0

    events = []
    pos, neg = 0.0, 0.0
    for i, d in enumerate(diffs):
        pos = max(0.0, pos + (d - mean_step))
        neg = min(0.0, neg + (d - mean_step))
        if pos > threshold:
            events.append(i + 1)  # +1: diffs[i] is the step from series[i] to series[i+1]
            pos = 0.0
        elif neg < -threshold:
            events.append(i + 1)
            neg = 0.0
    return np.array(events, dtype=np.int64)


def agreement_rate(
    detected_breaks: np.ndarray, regime_change_points: np.ndarray, tolerance: int
) -> float:
    """Fraction of `regime_change_points` with a CUSUM break within `tolerance` bars,
    used to cross-check model-detected regime switches against an independent detector.
    """
    if len(regime_change_points) == 0:
        return 1.0
    if len(detected_breaks) == 0:
        return 0.0
    matched = 0
    for cp in regime_change_points:
        if np.any(np.abs(detected_breaks - cp) <= tolerance):
            matched += 1
    return matched / len(regime_change_points)
