import numpy as np

from fin_regime_phasor.structural_breaks import agreement_rate, cusum_filter


def test_cusum_filter_detects_single_level_shift():
    series = np.concatenate([np.zeros(50), np.full(50, 10.0)])
    breaks = cusum_filter(series, threshold=2.0)
    assert len(breaks) >= 1
    assert any(45 <= b <= 55 for b in breaks)


def test_cusum_filter_no_breaks_on_flat_series():
    series = np.zeros(100)
    breaks = cusum_filter(series, threshold=1.0)
    assert len(breaks) == 0


def test_cusum_filter_higher_threshold_detects_fewer_breaks():
    rng = np.random.default_rng(0)
    series = np.cumsum(rng.normal(scale=1.0, size=500))
    low_thresh_breaks = cusum_filter(series, threshold=2.0)
    high_thresh_breaks = cusum_filter(series, threshold=20.0)
    assert len(high_thresh_breaks) <= len(low_thresh_breaks)


def test_agreement_rate_perfect_match():
    detected = np.array([50, 100, 150])
    change_points = np.array([49, 101, 149])
    assert agreement_rate(detected, change_points, tolerance=2) == 1.0


def test_agreement_rate_partial_match():
    detected = np.array([50])
    change_points = np.array([50, 200])
    assert agreement_rate(detected, change_points, tolerance=1) == 0.5


def test_agreement_rate_no_detected_breaks_is_zero():
    assert agreement_rate(np.array([]), np.array([10, 20]), tolerance=1) == 0.0


def test_agreement_rate_no_change_points_is_one():
    assert agreement_rate(np.array([10]), np.array([]), tolerance=1) == 1.0
