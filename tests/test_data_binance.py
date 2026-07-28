import pytest

from fin_regime_phasor.data.binance import (
    _verify_checksum,
    archive_url,
    fetch_aggtrades,
    fetch_period,
    iter_periods,
    parse_aggtrades_csv,
)

# Column order/values confirmed against live data.binance.vision archives:
# pre-2024 dumps ship with no header row, later ones do (see module docstring).
_NO_HEADER_CSV = (
    b"20539386,7817.91,0.001,28272968,28272968,1578614400254,true\n"
    b"20539387,7817.90,0.192,28272969,28272970,1578614400300,false\n"
)
_HEADER_CSV = (
    b"agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker\n"
    b"2199728897,67577.9,0.029,5056514276,5056514280,1717200000131,true\n"
    b"2199728898,67577.9,0.002,5056514281,5056514281,1717200000254,true\n"
)


def test_archive_url_monthly_futures_um():
    url = archive_url("BTCUSDT", "2024-01", market="futures-um")
    assert url == (
        "https://data.binance.vision/data/futures/um/monthly/aggTrades/"
        "BTCUSDT/BTCUSDT-aggTrades-2024-01.zip"
    )


def test_archive_url_daily_spot():
    url = archive_url("BTCUSDT", "2024-01-15", market="spot")
    assert url == (
        "https://data.binance.vision/data/spot/daily/aggTrades/"
        "BTCUSDT/BTCUSDT-aggTrades-2024-01-15.zip"
    )


def test_archive_url_rejects_unknown_market():
    with pytest.raises(ValueError):
        archive_url("BTCUSDT", "2024-01", market="dex")


def test_iter_periods_monthly_spans_year_boundary():
    assert list(iter_periods("monthly", "2023-11", "2024-02")) == [
        "2023-11",
        "2023-12",
        "2024-01",
        "2024-02",
    ]


def test_iter_periods_daily_spans_month_boundary():
    assert list(iter_periods("daily", "2024-01-30", "2024-02-02")) == [
        "2024-01-30",
        "2024-01-31",
        "2024-02-01",
        "2024-02-02",
    ]


def test_iter_periods_rejects_end_before_start():
    with pytest.raises(ValueError):
        list(iter_periods("monthly", "2024-02", "2024-01"))


def test_iter_periods_rejects_unknown_frequency():
    with pytest.raises(ValueError):
        list(iter_periods("weekly", "2024-01", "2024-02"))


def test_parse_aggtrades_csv_without_header():
    df = parse_aggtrades_csv(_NO_HEADER_CSV)
    assert df.columns == ["timestamp", "price", "quantity"]
    assert df["timestamp"].to_list() == [1578614400254, 1578614400300]
    assert df["price"].to_list() == pytest.approx([7817.91, 7817.90])
    assert df["quantity"].to_list() == pytest.approx([0.001, 0.192])


def test_parse_aggtrades_csv_with_header():
    df = parse_aggtrades_csv(_HEADER_CSV)
    assert df.columns == ["timestamp", "price", "quantity"]
    assert df["timestamp"].to_list() == [1717200000131, 1717200000254]


def test_verify_checksum_accepts_matching_hash():
    import hashlib

    raw = b"some zip bytes"
    digest = hashlib.sha256(raw).hexdigest()
    _verify_checksum(
        raw, f"{digest}  BTCUSDT-aggTrades-2024-01.zip\n", "BTCUSDT-aggTrades-2024-01.zip"
    )


def test_verify_checksum_rejects_mismatched_hash():
    with pytest.raises(ValueError):
        _verify_checksum(b"some zip bytes", "0" * 64 + "  file.zip", "file.zip")


def _fake_downloader(archives: dict[str, bytes]):
    def _download(url: str) -> bytes:
        if url.endswith(".CHECKSUM"):
            import hashlib

            raw = archives[url[: -len(".CHECKSUM")]]
            return f"{hashlib.sha256(raw).hexdigest()}  ignored.zip".encode()
        return archives[url]

    return _download


def _zip_bytes(name: str, content: bytes) -> bytes:
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(name, content)
    return buf.getvalue()


def test_fetch_period_downloads_verifies_and_parses(tmp_path):
    url = archive_url("BTCUSDT", "2024-01-15", market="futures-um")
    zip_content = _zip_bytes("BTCUSDT-aggTrades-2024-01-15.csv", _HEADER_CSV)
    downloader = _fake_downloader({url: zip_content})

    df = fetch_period(
        "BTCUSDT",
        "2024-01-15",
        cache_dir=tmp_path,
        downloader=downloader,
    )
    assert df.height == 2
    assert (tmp_path / "BTCUSDT-aggTrades-2024-01-15.zip").exists()


def test_fetch_period_rejects_bad_checksum(tmp_path):
    zip_content = _zip_bytes("BTCUSDT-aggTrades-2024-01-15.csv", _HEADER_CSV)

    def bad_downloader(request_url: str) -> bytes:
        if request_url.endswith(".CHECKSUM"):
            return ("0" * 64 + "  ignored.zip").encode()
        return zip_content

    with pytest.raises(ValueError):
        fetch_period("BTCUSDT", "2024-01-15", cache_dir=tmp_path, downloader=bad_downloader)


def test_fetch_period_reuses_cache_without_redownloading(tmp_path):
    zip_content = _zip_bytes("BTCUSDT-aggTrades-2024-01-15.csv", _HEADER_CSV)
    (tmp_path / "BTCUSDT-aggTrades-2024-01-15.zip").write_bytes(zip_content)

    def _fail(_: str) -> bytes:
        raise AssertionError("should not hit the network when the cache is warm")

    df = fetch_period("BTCUSDT", "2024-01-15", cache_dir=tmp_path, downloader=_fail)
    assert df.height == 2


def test_fetch_aggtrades_concatenates_and_sorts_across_periods(tmp_path):
    url_a = archive_url("BTCUSDT", "2024-01-15", market="futures-um")
    url_b = archive_url("BTCUSDT", "2024-01-16", market="futures-um")
    archives = {
        url_a: _zip_bytes("a.csv", _NO_HEADER_CSV),
        url_b: _zip_bytes("b.csv", _HEADER_CSV),
    }
    downloader = _fake_downloader(archives)

    df = fetch_aggtrades(
        "BTCUSDT",
        "2024-01-15",
        "2024-01-16",
        frequency="daily",
        cache_dir=tmp_path,
        downloader=downloader,
    )
    assert df.height == 4
    assert df["timestamp"].to_list() == sorted(df["timestamp"].to_list())
