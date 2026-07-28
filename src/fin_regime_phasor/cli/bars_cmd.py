from pathlib import Path

import polars as pl
import typer

from fin_regime_phasor.bars import calibrate_dollar_threshold, dollar_bars

app = typer.Typer(name="bars", help="Build dollar bars from raw trades.")


@app.command("build")
def build(
    trades: Path = typer.Option(..., help="Parquet file with columns timestamp, price, quantity."),
    out: Path = typer.Option(..., help="Output parquet path for the resulting bars."),
    threshold: float = typer.Option(None, help="Dollar-volume threshold per bar."),
    target_bars_per_day: float = typer.Option(
        None, help="Alternative to --threshold: calibrate threshold to hit this many bars/day."
    ),
    day_span: float = typer.Option(
        None, help="Number of days spanned by `trades` (required with --target-bars-per-day)."
    ),
) -> None:
    trades_df = pl.read_parquet(trades)

    if threshold is None:
        if target_bars_per_day is None or day_span is None:
            raise typer.BadParameter(
                "provide either --threshold or both --target-bars-per-day and --day-span"
            )
        threshold = calibrate_dollar_threshold(trades_df, target_bars_per_day, day_span)
        typer.echo(f"calibrated threshold: {threshold:.4f}")

    bars = dollar_bars(trades_df, threshold=threshold)
    bars.write_parquet(out)
    typer.echo(f"wrote {bars.height} bars to {out}")
