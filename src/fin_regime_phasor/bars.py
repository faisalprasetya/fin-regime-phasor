"""Dollar-bar construction from raw trades (AFML ch. 2).

Dollar bars, not fixed-time bars: PLAN.md "Data pipeline" -- this is also why the
magnitude comparison (V collapses under dollar bars by construction) mattered for
choosing Parkinson volatility over raw volume as magnitude.
"""

from __future__ import annotations

import polars as pl

TRADE_SCHEMA = {"timestamp": pl.Int64, "price": pl.Float64, "quantity": pl.Float64}


def dollar_bars(trades: pl.DataFrame, threshold: float) -> pl.DataFrame:
    """Aggregate raw trades into dollar bars, closing a bar once cumulative dollar
    volume (price*quantity) reaches `threshold`.

    `trades` must be sorted ascending by `timestamp` and have columns
    `timestamp`, `price`, `quantity`. Returns one row per bar with OHLC, dollar
    volume, trade count, and open/close timestamps -- the `high`/`low` columns
    feed `phasor.parkinson_sigma`, `close` feeds `phasor.log_return`.
    """
    if threshold <= 0:
        raise ValueError("threshold must be positive")

    prices = trades["price"].to_numpy()
    quantities = trades["quantity"].to_numpy()
    timestamps = trades["timestamp"].to_numpy()
    dollar_value = prices * quantities

    bars = []
    cum_dollar = 0.0
    bar_start = 0
    for i in range(len(trades)):
        cum_dollar += dollar_value[i]
        if cum_dollar >= threshold:
            segment_price = prices[bar_start : i + 1]
            segment_qty = quantities[bar_start : i + 1]
            bars.append(
                {
                    "open_time": int(timestamps[bar_start]),
                    "close_time": int(timestamps[i]),
                    "open": float(segment_price[0]),
                    "high": float(segment_price.max()),
                    "low": float(segment_price.min()),
                    "close": float(segment_price[-1]),
                    "dollar_volume": float(segment_qty @ segment_price),
                    "n_trades": i + 1 - bar_start,
                }
            )
            cum_dollar = 0.0
            bar_start = i + 1

    return pl.DataFrame(
        bars,
        schema={
            "open_time": pl.Int64,
            "close_time": pl.Int64,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "dollar_volume": pl.Float64,
            "n_trades": pl.Int64,
        },
    )


def calibrate_dollar_threshold(
    trades: pl.DataFrame, target_bars_per_day: float, day_span: float
) -> float:
    """Pick a dollar threshold hitting ~`target_bars_per_day` bars/day over `day_span` days.

    Calibrate on the training period only (PLAN.md "Bars"): total dollar volume
    over the sample, divided by the target total bar count.
    """
    if target_bars_per_day <= 0 or day_span <= 0:
        raise ValueError("target_bars_per_day and day_span must be positive")
    total_dollar = float((trades["price"] * trades["quantity"]).sum())
    target_total_bars = target_bars_per_day * day_span
    return total_dollar / target_total_bars
