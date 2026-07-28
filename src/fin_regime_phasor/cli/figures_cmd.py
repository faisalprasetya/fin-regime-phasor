import json
from pathlib import Path

import matplotlib
import numpy as np
import polars as pl
import typer

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fin_regime_phasor.plotting.figures import (
    plot_ablation_grid_bic,
    plot_loss_curve,
    plot_phasor_scatter,
    plot_regime_timeline,
)

app = typer.Typer(
    name="figures", help="Generate paper figures (PDF, publication style) from pipeline outputs."
)

_STYLE_PATH = Path(__file__).resolve().parent.parent / "plotting" / "paper.mplstyle"


@app.command("phasor-scatter")
def phasor_scatter(
    phasor: Path = typer.Option(..., help="Parquet with sigma_star, theta columns."),
    out: Path = typer.Option(...),
    regimes: Path = typer.Option(None, help="Optional .npy of regime labels, one per row."),
) -> None:
    plt.style.use(_STYLE_PATH)
    df = pl.read_parquet(phasor)
    regime_labels = np.load(regimes) if regimes is not None else None
    fig = plot_phasor_scatter(df["sigma_star"].to_numpy(), df["theta"].to_numpy(), regime_labels)
    fig.savefig(out, format="pdf", bbox_inches="tight")
    typer.echo(f"wrote {out}")


@app.command("regime-timeline")
def regime_timeline(
    bars: Path = typer.Option(..., help="Parquet with a 'close' column."),
    hqmm_result: Path = typer.Option(
        ..., help=".npz produced by `hqmm train` (needs regime_posteriors)."
    ),
    out: Path = typer.Option(...),
) -> None:
    plt.style.use(_STYLE_PATH)
    bars_df = pl.read_parquet(bars)
    npz = np.load(hqmm_result)
    fig = plot_regime_timeline(bars_df["close"].to_numpy(), npz["regime_posteriors"])
    fig.savefig(out, format="pdf", bbox_inches="tight")
    typer.echo(f"wrote {out}")


@app.command("hqmm-loss-curve")
def hqmm_loss_curve(
    loss_curve: Path = typer.Option(..., help="JSON list of per-step loss values."),
    out: Path = typer.Option(...),
) -> None:
    plt.style.use(_STYLE_PATH)
    values = json.loads(loss_curve.read_text())
    fig = plot_loss_curve(values)
    fig.savefig(out, format="pdf", bbox_inches="tight")
    typer.echo(f"wrote {out}")


@app.command("ablation-grid-bic")
def ablation_grid_bic(
    summary: Path = typer.Option(..., help="JSON produced by `benchmark grid`."),
    out: Path = typer.Option(...),
) -> None:
    plt.style.use(_STYLE_PATH)
    grid_summary = json.loads(summary.read_text())
    fig = plot_ablation_grid_bic(grid_summary)
    fig.savefig(out, format="pdf", bbox_inches="tight")
    typer.echo(f"wrote {out}")
