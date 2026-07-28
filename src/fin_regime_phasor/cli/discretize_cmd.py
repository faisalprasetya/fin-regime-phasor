from pathlib import Path

import numpy as np
import polars as pl
import typer

from fin_regime_phasor.discretize import VQCodebook, apply_vq_codebook, fit_vq_codebook

app = typer.Typer(name="discretize", help="Vector-quantize (sigma, theta) into a finite alphabet.")


@app.command("fit")
def fit(
    phasor: Path = typer.Option(..., help="Parquet with sigma_star, theta columns."),
    n_symbols: int = typer.Option(...),
    out: Path = typer.Option(..., help="Output .npz codebook path."),
    random_state: int = typer.Option(0),
) -> None:
    df = pl.read_parquet(phasor)
    codebook = fit_vq_codebook(
        df["sigma_star"].to_numpy(), df["theta"].to_numpy(), n_symbols, random_state
    )
    np.savez(out, centers=codebook.centers, mean=codebook.feature_mean, std=codebook.feature_std)
    typer.echo(f"fit {n_symbols}-symbol codebook -> {out}")


@app.command("apply")
def apply_cmd(
    phasor: Path = typer.Option(...),
    codebook: Path = typer.Option(...),
    out: Path = typer.Option(...),
) -> None:
    df = pl.read_parquet(phasor)
    npz = np.load(codebook)
    cb = VQCodebook(centers=npz["centers"], feature_mean=npz["mean"], feature_std=npz["std"])
    symbols = apply_vq_codebook(cb, df["sigma_star"].to_numpy(), df["theta"].to_numpy())
    pl.DataFrame({"symbol": symbols}).write_parquet(out)
    typer.echo(f"wrote {len(symbols)} symbols to {out}")
