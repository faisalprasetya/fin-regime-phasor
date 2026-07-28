import json
from pathlib import Path

import numpy as np
import polars as pl
import typer

from fin_regime_phasor.baselines.classical_hmm import fit_categorical_hmm, fit_gaussian_hmm
from fin_regime_phasor.baselines.hamilton import fit_hamilton_markov_switching
from fin_regime_phasor.baselines.naive import fit_unconditional_gaussian

app = typer.Typer(name="baselines", help="Classical regime-detection baselines.")


def _write_metrics(out: Path, result) -> None:
    out.write_text(
        json.dumps(
            {
                "log_likelihood": result.log_likelihood,
                "n_params": result.n_params,
                "aic": result.aic,
                "bic": result.bic,
            },
            indent=2,
        )
    )


@app.command("gaussian-hmm")
def gaussian_hmm_cmd(
    features: Path = typer.Option(
        ..., help="Parquet with columns r_star, sigma_star (raw features)."
    ),
    n_states: int = typer.Option(...),
    out: Path = typer.Option(
        ..., help="Output JSON metrics path; states saved alongside as .states.npy."
    ),
    seed: int = typer.Option(0),
) -> None:
    df = pl.read_parquet(features)
    x = np.column_stack([df["r_star"].to_numpy(), df["sigma_star"].to_numpy()])
    result = fit_gaussian_hmm(x, n_states=n_states, random_state=seed)
    np.save(out.with_suffix(".states.npy"), result.states)
    _write_metrics(out, result)
    typer.echo(f"gaussian HMM fit -> {out}")


@app.command("categorical-hmm")
def categorical_hmm_cmd(
    symbols: Path = typer.Option(..., help="Parquet with a 'symbol' column."),
    n_states: int = typer.Option(...),
    n_symbols: int = typer.Option(...),
    out: Path = typer.Option(...),
    seed: int = typer.Option(0),
) -> None:
    df = pl.read_parquet(symbols)
    result = fit_categorical_hmm(
        df["symbol"].to_numpy(), n_states=n_states, n_symbols=n_symbols, random_state=seed
    )
    np.save(out.with_suffix(".states.npy"), result.states)
    _write_metrics(out, result)
    typer.echo(f"categorical HMM fit -> {out}")


@app.command("hamilton")
def hamilton_cmd(
    returns: Path = typer.Option(..., help="Parquet with an 'r_star' column of log-returns."),
    n_states: int = typer.Option(...),
    out: Path = typer.Option(...),
) -> None:
    df = pl.read_parquet(returns)
    result = fit_hamilton_markov_switching(df["r_star"].to_numpy(), n_states=n_states)
    np.save(out.with_suffix(".states.npy"), result.states)
    _write_metrics(out, result)
    typer.echo(f"Hamilton Markov-switching fit -> {out}")


@app.command("naive")
def naive_cmd(returns: Path = typer.Option(...), out: Path = typer.Option(...)) -> None:
    df = pl.read_parquet(returns)
    result = fit_unconditional_gaussian(df["r_star"].to_numpy())
    out.write_text(
        json.dumps(
            {
                "mean": result.mean,
                "std": result.std,
                "log_likelihood": result.log_likelihood,
                "n_params": result.n_params,
                "aic": result.aic,
                "bic": result.bic,
            },
            indent=2,
        )
    )
    typer.echo(f"naive unconditional Gaussian fit -> {out}")
