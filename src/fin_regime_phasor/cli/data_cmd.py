from pathlib import Path

import typer

from fin_regime_phasor.data.binance import fetch_aggtrades

app = typer.Typer(name="data", help="Download real market trade data (data.binance.vision).")


@app.command("fetch-binance")
def fetch_binance(
    symbol: str = typer.Option(..., help="e.g. BTCUSDT."),
    start: str = typer.Option(
        ..., help="First period inclusive (YYYY-MM, or YYYY-MM-DD if --frequency daily)."
    ),
    end: str = typer.Option(..., help="Last period inclusive, same format as --start."),
    out: Path = typer.Option(
        ..., help="Output parquet (timestamp, price, quantity) -- feeds `bars build`."
    ),
    frequency: str = typer.Option(
        "monthly",
        help="'monthly' (large, per PLAN.md's dataset spec) or 'daily' (smaller archives).",
    ),
    market: str = typer.Option(
        "futures-um", help="'futures-um' (USDT-M perpetual, primary asset) or 'spot'."
    ),
    cache_dir: Path = typer.Option(
        Path(".cache/binance"), help="Where raw archives are cached across runs."
    ),
    verify_checksum: bool = typer.Option(
        True, help="Verify each archive's published SHA256 checksum on first download."
    ),
) -> None:
    trades = fetch_aggtrades(
        symbol,
        start,
        end,
        frequency=frequency,
        market=market,
        cache_dir=cache_dir,
        verify_checksum=verify_checksum,
    )
    trades.write_parquet(out)
    typer.echo(f"wrote {trades.height} trades ({start}..{end}, {market}) -> {out}")
