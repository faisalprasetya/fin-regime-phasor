import json

import numpy as np
from typer.testing import CliRunner

from fin_regime_phasor.cli import app

runner = CliRunner()


def test_root_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0


def test_synthetic_generate_then_full_pipeline(tmp_path):
    config = {
        "transition_matrix": [[0.95, 0.05], [0.05, 0.95]],
        "mu": [0.0005, -0.0005],
        "sigma": [0.01, 0.04],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    bars_path = tmp_path / "bars.parquet"
    regimes_path = tmp_path / "regimes.npy"
    result = runner.invoke(
        app,
        [
            "synthetic",
            "generate",
            "--n-bars",
            "200",
            "--config",
            str(config_path),
            "--out-bars",
            str(bars_path),
            "--out-regimes",
            str(regimes_path),
            "--seed",
            "0",
        ],
    )
    assert result.exit_code == 0, result.output
    assert bars_path.exists()
    assert regimes_path.exists()

    d_r_path = tmp_path / "d_r.json"
    result = runner.invoke(
        app,
        [
            "features",
            "fracdiff-search",
            "--bars",
            str(bars_path),
            "--series",
            "log_price",
            "--out",
            str(d_r_path),
        ],
    )
    assert result.exit_code == 0, result.output
    d_r = json.loads(d_r_path.read_text())["d_star"]

    d_sigma_path = tmp_path / "d_sigma.json"
    result = runner.invoke(
        app,
        [
            "features",
            "fracdiff-search",
            "--bars",
            str(bars_path),
            "--series",
            "sigma",
            "--out",
            str(d_sigma_path),
        ],
    )
    assert result.exit_code == 0, result.output
    d_sigma = json.loads(d_sigma_path.read_text())["d_star"]

    phasor_path = tmp_path / "phasor.parquet"
    result = runner.invoke(
        app,
        [
            "features",
            "build-phasor",
            "--bars",
            str(bars_path),
            "--d-r",
            str(d_r),
            "--d-sigma",
            str(d_sigma),
            "--k",
            "50.0",
            "--out",
            str(phasor_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert phasor_path.exists()

    codebook_path = tmp_path / "codebook.npz"
    result = runner.invoke(
        app,
        [
            "discretize",
            "fit",
            "--phasor",
            str(phasor_path),
            "--n-symbols",
            "4",
            "--out",
            str(codebook_path),
        ],
    )
    assert result.exit_code == 0, result.output

    symbols_path = tmp_path / "symbols.parquet"
    result = runner.invoke(
        app,
        [
            "discretize",
            "apply",
            "--phasor",
            str(phasor_path),
            "--codebook",
            str(codebook_path),
            "--out",
            str(symbols_path),
        ],
    )
    assert result.exit_code == 0, result.output

    hqmm_out = tmp_path / "hqmm_result.npz"
    result = runner.invoke(
        app,
        [
            "hqmm",
            "train",
            "--symbols",
            str(symbols_path),
            "--n-states",
            "2",
            "--n-symbols",
            "4",
            "--out",
            str(hqmm_out),
            "--n-steps",
            "20",
            "--n-restarts",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert hqmm_out.exists()
    npz = np.load(hqmm_out)
    assert "regime_posteriors" in npz

    figure_out = tmp_path / "phasor_scatter.pdf"
    result = runner.invoke(
        app,
        ["figures", "phasor-scatter", "--phasor", str(phasor_path), "--out", str(figure_out)],
    )
    assert result.exit_code == 0, result.output
    assert figure_out.exists()


def test_baselines_naive_cli(tmp_path):
    import polars as pl

    rng = np.random.default_rng(0)
    returns_path = tmp_path / "returns.parquet"
    pl.DataFrame({"r_star": rng.normal(scale=0.01, size=200)}).write_parquet(returns_path)

    out_path = tmp_path / "naive.json"
    result = runner.invoke(
        app, ["baselines", "naive", "--returns", str(returns_path), "--out", str(out_path)]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(out_path.read_text())
    assert "log_likelihood" in payload
