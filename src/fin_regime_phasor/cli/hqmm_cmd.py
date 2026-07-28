import json
from pathlib import Path

import numpy as np
import polars as pl
import typer

from fin_regime_phasor.hqmm.model import train_hqmm

app = typer.Typer(name="hqmm", help="Train the Kraus-operator HQMM on discretized symbols.")


@app.command("train")
def train(
    symbols: Path = typer.Option(..., help="Parquet with a 'symbol' column."),
    n_states: int = typer.Option(...),
    n_symbols: int = typer.Option(...),
    out: Path = typer.Option(..., help="Output .npz with generator/posteriors/states."),
    seed: int = typer.Option(0),
    n_steps: int = typer.Option(500),
    learning_rate: float = typer.Option(0.05),
    n_restarts: int = typer.Option(5),
) -> None:
    df = pl.read_parquet(symbols)
    result = train_hqmm(
        df["symbol"].to_numpy(),
        n_states=n_states,
        n_symbols=n_symbols,
        seed=seed,
        n_steps=n_steps,
        learning_rate=learning_rate,
        n_restarts=n_restarts,
    )
    np.savez(
        out,
        generator=result.generator,
        regime_posteriors=result.regime_posteriors,
        states=result.states,
    )
    metrics_path = out.with_suffix(".json")
    metrics_path.write_text(
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
    typer.echo(f"trained HQMM -> {out}, metrics -> {metrics_path}")
