import json
from pathlib import Path

import polars as pl
import typer

from fin_regime_phasor.benchmark import run_full_benchmark

app = typer.Typer(name="benchmark", help="Run the 2x2 representation x mechanism ablation grid.")


@app.command("grid")
def grid(
    phasor: Path = typer.Option(..., help="Parquet with r_star, sigma_star, theta columns."),
    n_states: int = typer.Option(...),
    n_symbols: int = typer.Option(...),
    out: Path = typer.Option(..., help="Output JSON summary of the grid + baselines."),
    seed: int = typer.Option(0),
    hqmm_steps: int = typer.Option(300),
    hqmm_restarts: int = typer.Option(5),
) -> None:
    df = pl.read_parquet(phasor)
    result = run_full_benchmark(
        df["r_star"].to_numpy(),
        df["sigma_star"].to_numpy(),
        df["theta"].to_numpy(),
        n_states=n_states,
        n_symbols=n_symbols,
        seed=seed,
        hqmm_steps=hqmm_steps,
        hqmm_restarts=hqmm_restarts,
    )
    summary = {
        f"{rep}_{mech}": {
            "log_likelihood": cell.log_likelihood,
            "n_params": cell.n_params,
            "aic": cell.aic,
            "bic": cell.bic,
        }
        for (rep, mech), cell in result.grid.items()
    }
    summary["hamilton"] = {
        "log_likelihood": result.hamilton.log_likelihood,
        "aic": result.hamilton.aic,
        "bic": result.hamilton.bic,
    }
    summary["naive"] = {
        "log_likelihood": result.naive.log_likelihood,
        "aic": result.naive.aic,
        "bic": result.naive.bic,
    }
    out.write_text(json.dumps(summary, indent=2))
    typer.echo(f"wrote benchmark summary -> {out}")
