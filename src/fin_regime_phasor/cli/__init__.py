"""Root Typer app; every runnable pipeline step registers as a sub-command here."""

import typer

app = typer.Typer(
    name="fin-regime-phasor",
    help="Phasor feature engineering + HQMM regime detection pipeline.",
    no_args_is_help=True,
)

from fin_regime_phasor.cli import (
    bars_cmd,
    baselines_cmd,
    benchmark_cmd,
    discretize_cmd,
    features_cmd,
    figures_cmd,
    hqmm_cmd,
    synthetic_cmd,
)

for module in (
    bars_cmd,
    features_cmd,
    discretize_cmd,
    synthetic_cmd,
    hqmm_cmd,
    baselines_cmd,
    benchmark_cmd,
    figures_cmd,
):
    app.add_typer(module.app)


if __name__ == "__main__":
    app()
