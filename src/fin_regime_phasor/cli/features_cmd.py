import json
from pathlib import Path

import numpy as np
import polars as pl
import typer

from fin_regime_phasor.fracdiff import frac_diff_ffd, minimum_ffd_d
from fin_regime_phasor.phasor import build_phasor, parkinson_sigma, phase_direct

app = typer.Typer(name="features", help="Frac-diff minimum-d search + phasor construction.")


def _load_log_price_and_sigma(bars_path: Path) -> tuple[np.ndarray, np.ndarray]:
    bars = pl.read_parquet(bars_path)
    log_price = np.log(bars["close"].to_numpy())
    sigma = parkinson_sigma(bars["high"].to_numpy(), bars["low"].to_numpy())
    return log_price, sigma


@app.command("fracdiff-search")
def fracdiff_search(
    bars: Path = typer.Option(..., help="Parquet file of bars (needs high, low, close)."),
    series: str = typer.Option(..., help="Which series to search: 'log_price' or 'sigma'."),
    out: Path = typer.Option(..., help="Output JSON path for d_star and the search grid."),
    p_value_threshold: float = typer.Option(0.05),
) -> None:
    log_price, sigma = _load_log_price_and_sigma(bars)
    target = {"log_price": log_price, "sigma": sigma}.get(series)
    if target is None:
        raise typer.BadParameter("series must be 'log_price' or 'sigma'")

    result = minimum_ffd_d(target, p_value_threshold=p_value_threshold)
    payload = {
        "series": series,
        "d_star": result.d_star,
        "steps": [s.__dict__ for s in result.steps],
    }
    out.write_text(json.dumps(payload, indent=2))
    typer.echo(f"d*_{series} = {result.d_star:.3f}")


@app.command("build-phasor")
def build_phasor_cmd(
    bars: Path = typer.Option(...),
    d_r: float = typer.Option(..., help="Minimum-d found for ln(P) (feeds phase)."),
    d_sigma: float = typer.Option(..., help="Minimum-d found for sigma (feeds magnitude)."),
    k: float = typer.Option(..., help="Phase sensitivity constant."),
    out: Path = typer.Option(...),
) -> None:
    log_price, sigma = _load_log_price_and_sigma(bars)

    r_star = frac_diff_ffd(log_price, d_r)
    sigma_star = frac_diff_ffd(sigma, d_sigma)

    valid = ~(np.isnan(r_star) | np.isnan(sigma_star))
    r_star, sigma_star = r_star[valid], sigma_star[valid]

    theta = phase_direct(k * r_star)
    z = build_phasor(sigma_star, r_star, k)

    out_df = pl.DataFrame(
        {
            "sigma_star": sigma_star,
            "r_star": r_star,
            "theta": theta,
            "z_real": z.real,
            "z_imag": z.imag,
        }
    )
    out_df.write_parquet(out)
    typer.echo(f"wrote {out_df.height} phasor rows to {out}")
