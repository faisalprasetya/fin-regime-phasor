import polars as pl
import pytest

from fin_regime_phasor.bars import calibrate_dollar_threshold, dollar_bars


def _trades(prices, quantities, timestamps=None):
    n = len(prices)
    timestamps = timestamps if timestamps is not None else list(range(n))
    return pl.DataFrame({"timestamp": timestamps, "price": prices, "quantity": quantities})


def test_dollar_bars_closes_on_threshold():
    trades = _trades(prices=[10.0, 10.0, 10.0, 10.0], quantities=[3.0, 3.0, 3.0, 3.0])
    # dollar value per trade = 30; threshold 50 -> first bar closes after 2 trades (60 >= 50)
    bars = dollar_bars(trades, threshold=50.0)
    assert bars["n_trades"].to_list() == [2, 2]


def test_dollar_bars_ohlc_correct():
    trades = _trades(prices=[100.0, 105.0, 95.0, 102.0], quantities=[1.0, 1.0, 1.0, 1.0])
    bars = dollar_bars(trades, threshold=200.0)
    assert bars["open"][0] == pytest.approx(100.0)
    assert bars["high"][0] == pytest.approx(105.0)
    assert bars["low"][0] == pytest.approx(100.0)
    assert bars["close"][0] == pytest.approx(105.0)


def test_dollar_bars_drops_incomplete_trailing_bar():
    trades = _trades(prices=[10.0, 10.0], quantities=[1.0, 1.0])
    bars = dollar_bars(trades, threshold=1000.0)
    assert bars.height == 0


def test_dollar_bars_rejects_nonpositive_threshold():
    trades = _trades(prices=[10.0], quantities=[1.0])
    with pytest.raises(ValueError):
        dollar_bars(trades, threshold=0.0)


def test_dollar_bars_dollar_volume_sums_correctly():
    trades = _trades(prices=[10.0, 20.0], quantities=[2.0, 3.0])
    # dollar values: 20, 60 -> single bar closes at trade 2 (cum 80 >= threshold 50)
    bars = dollar_bars(trades, threshold=50.0)
    assert bars["dollar_volume"][0] == pytest.approx(80.0)


def test_calibrate_dollar_threshold_hits_target_bar_count():
    trades = _trades(prices=[100.0] * 1000, quantities=[1.0] * 1000)
    threshold = calibrate_dollar_threshold(trades, target_bars_per_day=10.0, day_span=5.0)
    bars = dollar_bars(trades, threshold=threshold)
    # total dollar volume 100_000 / threshold should give ~50 bars (10/day * 5 days)
    assert bars.height == pytest.approx(50, abs=2)


def test_calibrate_dollar_threshold_rejects_bad_inputs():
    trades = _trades(prices=[10.0], quantities=[1.0])
    with pytest.raises(ValueError):
        calibrate_dollar_threshold(trades, target_bars_per_day=0.0, day_span=1.0)
