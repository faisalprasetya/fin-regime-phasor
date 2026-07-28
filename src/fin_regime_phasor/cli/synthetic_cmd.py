import json
from pathlib import Path

import numpy as np
import typer

from fin_regime_phasor.synthetic import generate_regime_switching_gbm

app = typer.Typer(name="synthetic", help="Generate synthetic regime-switching GBM ground truth.")


@app.command("generate")
def generate(
    n_bars: int = typer.Option(...),
    config: Path = typer.Option(
        ..., help="JSON with transition_matrix, mu, sigma (per-regime arrays)."
    ),
    out_bars: Path = typer.Option(...),
    out_regimes: Path = typer.Option(...),
    seed: int = typer.Option(0),
) -> None:
    payload = json.loads(config.read_text())
    dataset = generate_regime_switching_gbm(
        n_bars=n_bars,
        transition_matrix=np.array(payload["transition_matrix"]),
        mu=np.array(payload["mu"]),
        sigma=np.array(payload["sigma"]),
        seed=seed,
    )
    dataset.bars.write_parquet(out_bars)
    np.save(out_regimes, dataset.regimes)
    typer.echo(
        f"generated {dataset.bars.height} synthetic bars -> {out_bars}, regimes -> {out_regimes}"
    )
