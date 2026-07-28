"""Download and parse Binance aggTrades archives from data.binance.vision.

PLAN.md "Dataset": BTC/USDT perpetual futures (Binance USDS-M) is the primary
asset, sourced from the public `data.binance.vision` archive rather than a
rate-limited REST API, since it ships trade-level granularity needed for real
dollar/volume bars (not just resampled OHLC).

The archive layout, CSV column order, and checksum format below were confirmed
against the live endpoints rather than assumed from docs, since the format has
changed over time: pre-2024 dumps ship with no CSV header row, later ones do,
so `parse_aggtrades_csv` detects rather than assumes.
"""

from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from collections.abc import Callable, Iterator
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from fin_regime_phasor.bars import TRADE_SCHEMA

_BASE_URL = "https://data.binance.vision/data"
_MARKET_PATHS = {"futures-um": "futures/um", "spot": "spot"}
_AGGTRADES_COLUMNS = [
    "agg_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "transact_time",
    "is_buyer_maker",
]

Downloader = Callable[[str], bytes]


def _http_get(url: str) -> bytes:
    with urllib.request.urlopen(url) as response:
        return response.read()


def archive_url(symbol: str, period: str, market: str = "futures-um") -> str:
    """`period` is `YYYY-MM` for a monthly archive or `YYYY-MM-DD` for a daily one."""
    if market not in _MARKET_PATHS:
        raise ValueError(f"unknown market {market!r}, expected one of {sorted(_MARKET_PATHS)}")
    frequency = "monthly" if len(period) == len("YYYY-MM") else "daily"
    return (
        f"{_BASE_URL}/{_MARKET_PATHS[market]}/{frequency}/aggTrades/"
        f"{symbol}/{symbol}-aggTrades-{period}.zip"
    )


def _iter_months(start: str, end: str) -> Iterator[str]:
    start_year, start_month = (int(part) for part in start.split("-"))
    end_year, end_month = (int(part) for part in end.split("-"))
    if (end_year, end_month) < (start_year, start_month):
        raise ValueError(f"end {end!r} precedes start {start!r}")
    year, month = start_year, start_month
    while (year, month) <= (end_year, end_month):
        yield f"{year:04d}-{month:02d}"
        month += 1
        if month == 13:
            month, year = 1, year + 1


def _iter_days(start: str, end: str) -> Iterator[str]:
    start_date, end_date = date.fromisoformat(start), date.fromisoformat(end)
    if end_date < start_date:
        raise ValueError(f"end {end!r} precedes start {start!r}")
    current = start_date
    while current <= end_date:
        yield current.isoformat()
        current += timedelta(days=1)


def iter_periods(frequency: str, start: str, end: str) -> Iterator[str]:
    if frequency == "monthly":
        yield from _iter_months(start, end)
    elif frequency == "daily":
        yield from _iter_days(start, end)
    else:
        raise ValueError(f"frequency must be 'monthly' or 'daily', got {frequency!r}")


def parse_aggtrades_csv(raw_csv: bytes) -> pl.DataFrame:
    """Parse one archive's CSV bytes into the `timestamp`/`price`/`quantity` schema
    `bars.dollar_bars` expects, sorted ascending (Binance dumps already are).
    """
    first_field = raw_csv.split(b",", 1)[0]
    has_header = not first_field.isdigit()
    df = pl.read_csv(
        io.BytesIO(raw_csv),
        has_header=has_header,
        new_columns=None if has_header else _AGGTRADES_COLUMNS,
    )
    return df.select(
        pl.col("transact_time").alias("timestamp"),
        pl.col("price"),
        pl.col("quantity"),
    ).cast(TRADE_SCHEMA)


def _verify_checksum(raw_zip: bytes, checksum_text: str, filename: str) -> None:
    expected = checksum_text.split()[0]
    actual = hashlib.sha256(raw_zip).hexdigest()
    if actual != expected:
        raise ValueError(f"checksum mismatch for {filename}: expected {expected}, got {actual}")


def fetch_period(
    symbol: str,
    period: str,
    market: str = "futures-um",
    cache_dir: Path | None = None,
    verify_checksum: bool = True,
    downloader: Downloader = _http_get,
) -> pl.DataFrame:
    """Fetch (or reuse a cached copy of) one monthly/daily aggTrades archive.

    Cached archives are trusted as already-verified and are not re-checksummed on
    every read; verification only happens on first download.
    """
    url = archive_url(symbol, period, market)
    filename = url.rsplit("/", 1)[-1]
    cache_path = None if cache_dir is None else Path(cache_dir) / filename

    if cache_path is not None and cache_path.exists():
        raw_zip = cache_path.read_bytes()
    else:
        raw_zip = downloader(url)
        if verify_checksum:
            _verify_checksum(raw_zip, downloader(f"{url}.CHECKSUM").decode(), filename)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(raw_zip)

    with zipfile.ZipFile(io.BytesIO(raw_zip)) as archive:
        (member,) = archive.namelist()
        raw_csv = archive.read(member)

    return parse_aggtrades_csv(raw_csv)


def fetch_aggtrades(
    symbol: str,
    start: str,
    end: str,
    frequency: str = "monthly",
    market: str = "futures-um",
    cache_dir: Path | None = None,
    verify_checksum: bool = True,
    downloader: Downloader = _http_get,
) -> pl.DataFrame:
    """Download every period in `[start, end]` and concatenate into one
    `timestamp`/`price`/`quantity` DataFrame sorted ascending, ready for
    `bars.dollar_bars`.
    """
    frames = [
        fetch_period(
            symbol,
            period,
            market=market,
            cache_dir=cache_dir,
            verify_checksum=verify_checksum,
            downloader=downloader,
        )
        for period in iter_periods(frequency, start, end)
    ]
    return pl.concat(frames).sort("timestamp")
