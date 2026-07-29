# fin-regime-phasor

Financial Phasor Feature Engineering for Quantum Regime-Switching using HQMM

Research project that builds complex-valued "phasor" features from financial bar data — magnitude from Parkinson volatility, phase from the `2*arctan(k*R)` mapping of a fractionally-differenced log return — and feeds them into a Hidden Quantum Markov Model (HQMM, JAX Kraus-operator formulation) to detect latent market regimes. Results are benchmarked against classical baselines (`hmmlearn`, Hamilton Markov-switching) via a representation x mechanism ablation grid, and detected regimes are used as a risk-sizing overlay on a trading strategy.

See [PLAN.md](PLAN.md) for the full research design: math derivations, candidate comparisons, benchmark grid, dataset/backtest discipline, and open decisions. See [CLAUDE.md](CLAUDE.md) for repo conventions.

## Status

Core pipeline implemented and tested: real market data ingestion (`data.binance.vision` aggTrades archives), phasor math, frac-diff + minimum-*d* search, dollar bars, VQ discretization, synthetic regime-switching data generator, classical baselines (Gaussian/categorical HMM, Hamilton, no-regime), HQMM (JAX Kraus operators), CUSUM structural breaks, 2x2 ablation-grid benchmark, backtest overlay (CPCV/embargo, deflated Sharpe, PBO), full Typer CLI, and a matplotlib paper style + figure commands.

Not yet done: the paper itself, and running the pipeline end-to-end on real BTC/USDT data at PLAN.md's full 2020-2025 scale — the ingestion path is implemented and tested, but the rest of the pipeline so far has only been validated on synthetic ground truth plus unit/property tests (per PLAN.md's research discipline: validate on synthetic ground truth before trusting output on real market data).

## Install

Managed with [`uv`](https://docs.astral.sh/uv/) (not pip/poetry/conda directly), Python >= 3.12.

```bash
uv sync
```

## Usage

Every runnable pipeline step is a Typer sub-command on one root app:

```bash
uv run fin-regime-phasor --help
```

| Sub-command group | Purpose |
|---|---|
| `data fetch-binance` | Download real BTC/USDT trade archives from `data.binance.vision` |
| `bars build` | Build dollar bars from raw trades |
| `features fracdiff-search` | AFML minimum-*d* search on `ln(P)` and `sigma`, independently |
| `features build-phasor` | Construct the phasor `z = sigma * exp(i * theta)` from frac-diffed inputs |
| `discretize fit` / `discretize apply` | Vector-quantize `(sigma, theta)` into a finite alphabet for the HQMM |
| `synthetic generate` | Generate synthetic regime-switching GBM data with ground-truth labels |
| `hqmm train` | Train the Kraus-operator HQMM (JAX) on discretized symbols |
| `baselines gaussian-hmm` / `categorical-hmm` / `hamilton` / `naive` | Classical regime-detection baselines |
| `benchmark grid` | Run the 2x2 representation x mechanism ablation grid |
| `figures phasor-scatter` / `regime-timeline` / `hqmm-loss-curve` / `ablation-grid-bic` | Publication-quality figures (matplotlib, paper style) |

## Development

```bash
uv run ruff check .      # lint
uv run ruff format .     # format
uv run pytest            # tests
```

## Running on Google Colab

The only step in this pipeline that benefits from an accelerator is `hqmm train` (JAX gradient descent over Kraus operators). Bar building (`polars`), frac-diff search, discretization, and the classical baselines (`hmmlearn`/`statsmodels`) are CPU-bound and gain nothing from a GPU/TPU runtime — pick a CPU runtime for those steps if you're conserving Colab quota, and only switch to GPU/TPU before `hqmm train`.

Each block below is one Colab cell.

### 1. Clone the repo

```python
!git clone https://github.com/faisalprasetya/fin-regime-phasor.git
%cd fin-regime-phasor
```

### 2. Install `uv`

```python
!curl -LsSf https://astral.sh/uv/install.sh | sh
import os
os.environ["PATH"] = f"/root/.local/bin:{os.environ['PATH']}"
!uv --version
```

`os.environ` persists across cells in the same Colab session, so later `!uv ...` cells pick it up without re-exporting `PATH`.

### 3. Install dependencies

```python
!uv sync
```

This creates a `.venv` in the repo and installs the CPU build of `jax` by default — enough for everything except accelerated `hqmm train`.

### 4. GPU support (optional, for `hqmm train`)

Runtime > Change runtime type > Hardware accelerator > GPU (T4/L4/A100), then:

```python
!nvidia-smi                      # confirm the GPU + driver are visible
!uv add "jax[cuda12]"            # replaces the CPU jaxlib with a CUDA 12 build
```

Verify JAX sees the GPU:

```python
!uv run python -c "import jax; print(jax.devices())"
```

Expect something like `[CudaDevice(id=0)]` instead of `[CpuDevice(id=0)]`.

### 5. TPU support (optional, for `hqmm train`)

Runtime > Change runtime type > Hardware accelerator > TPU, then:

```python
!uv add "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

```python
!uv run python -c "import jax; print(jax.devices())"
```

Expect `[TpuDevice(...)]`. JAX's TPU install command shifts between releases more often than the GPU one — if this fails, check the current snippet at [docs.jax.dev/en/latest/installation.html](https://docs.jax.dev/en/latest/installation.html).

Switching between GPU and TPU (or back to CPU) requires reconnecting to a fresh runtime with the new accelerator selected — `uv add` alone won't move you between them.

### 6. Persist outputs (optional)

Colab VMs are ephemeral; mount Drive if you want `data/`, codebooks, or trained-model artifacts to survive a disconnect:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Then point the `--out`/`--out-bars`/`--out-regimes` flags below at a path under `/content/drive/MyDrive/...` instead of the repo directory.

### 7. Smoke test: fetch a narrow slice of real data and run the pipeline end-to-end

`data fetch-binance` downloads real trade archives from `data.binance.vision` (BTC/USDT perpetual futures, per PLAN.md's dataset choice) — no synthetic data needed. Each archive's SHA256 checksum is verified against the one Binance publishes alongside it. This demo uses a narrow 3-day daily-archive window (~5MB/day) to keep the Colab download quick, and fixed `--n-states`/`--n-symbols` placeholders rather than the BIC/alphabet sweep described in Section 8 — the point here is just to confirm every stage of the pipeline runs, not to produce a result worth reading.

```python
%%bash
mkdir -p out

uv run fin-regime-phasor data fetch-binance \
  --symbol BTCUSDT --start 2024-06-01 --end 2024-06-03 \
  --frequency daily --out out/trades.parquet

uv run fin-regime-phasor bars build \
  --trades out/trades.parquet --out out/bars.parquet \
  --target-bars-per-day 50 --day-span 3

uv run fin-regime-phasor features fracdiff-search \
  --bars out/bars.parquet --series log_price --out out/dstar_logprice.json
uv run fin-regime-phasor features fracdiff-search \
  --bars out/bars.parquet --series sigma --out out/dstar_sigma.json

# read the two d* values back into the phasor-construction step
D_R=$(python -c "import json; print(json.load(open('out/dstar_logprice.json'))['d_star'])")
D_SIGMA=$(python -c "import json; print(json.load(open('out/dstar_sigma.json'))['d_star'])")

uv run fin-regime-phasor features build-phasor \
  --bars out/bars.parquet --d-r "$D_R" --d-sigma "$D_SIGMA" --k 50.0 --out out/phasor.parquet

uv run fin-regime-phasor discretize fit \
  --phasor out/phasor.parquet --n-symbols 8 --out out/codebook.npz
uv run fin-regime-phasor discretize apply \
  --phasor out/phasor.parquet --codebook out/codebook.npz --out out/symbols.parquet

uv run fin-regime-phasor hqmm train \
  --symbols out/symbols.parquet --n-states 2 --n-symbols 8 --out out/hqmm.npz

uv run fin-regime-phasor baselines gaussian-hmm \
  --features out/phasor.parquet --n-states 2 --out out/gaussian_hmm.json
uv run fin-regime-phasor baselines categorical-hmm \
  --symbols out/symbols.parquet --n-states 2 --n-symbols 8 --out out/categorical_hmm.json
uv run fin-regime-phasor baselines hamilton \
  --returns out/phasor.parquet --n-states 2 --out out/hamilton.json
uv run fin-regime-phasor baselines naive \
  --returns out/phasor.parquet --out out/naive.json

uv run fin-regime-phasor benchmark grid \
  --phasor out/phasor.parquet --n-states 2 --n-symbols 8 --out out/benchmark_summary.json
```

`--n-states`/`--n-symbols` above (2, 8) are hardcoded placeholders — good enough to smoke-test, not to trust. Also per PLAN.md's research discipline, validate the HQMM against **synthetic** ground-truth regimes before trusting its output on real data like this — swap the first two commands above for `synthetic generate` (see the CLI table's `synthetic generate` entry, or `uv run fin-regime-phasor synthetic generate --help`) to run the identical rest of the pipeline against data with known planted regimes.

### 8. Full run: select `n-states`/`n-symbols` on training data, then run at scale

PLAN.md's actual research sample is 2020-01-01 to 2025-12-31 via monthly archives (`--frequency monthly`), which is multiple GB — pull it once you're past the smoke test above. Everything here runs on the training range only; the 2026-01-01-to-present window stays untouched until a single final evaluation pass (PLAN.md "Held-out test").

```python
%%bash
mkdir -p out

uv run fin-regime-phasor data fetch-binance \
  --symbol BTCUSDT --start 2020-01 --end 2025-12 \
  --frequency monthly --out out/trades_full.parquet

uv run fin-regime-phasor bars build \
  --trades out/trades_full.parquet --out out/bars_full.parquet \
  --target-bars-per-day 50 --day-span 2191

uv run fin-regime-phasor features fracdiff-search \
  --bars out/bars_full.parquet --series log_price --out out/dstar_logprice_full.json
uv run fin-regime-phasor features fracdiff-search \
  --bars out/bars_full.parquet --series sigma --out out/dstar_sigma_full.json

D_R=$(python -c "import json; print(json.load(open('out/dstar_logprice_full.json'))['d_star'])")
D_SIGMA=$(python -c "import json; print(json.load(open('out/dstar_sigma_full.json'))['d_star'])")

uv run fin-regime-phasor features build-phasor \
  --bars out/bars_full.parquet --d-r "$D_R" --d-sigma "$D_SIGMA" --k 50.0 --out out/phasor_full.parquet
```

**Pick `n-symbols` (VQ-alphabet size)** — per PLAN.md ("VQ-alphabet size: own small sweep over alphabet size, tuned on training folds only"), fit a codebook per candidate size, discretize, then score each via a fixed-`n-states` categorical-HMM's BIC as a proxy (`discretize fit` itself emits no intrinsic distortion score to sweep on):

```python
%%bash
for m in 8 12 16 24; do
  uv run fin-regime-phasor discretize fit \
    --phasor out/phasor_full.parquet --n-symbols "$m" --out "out/codebook_m${m}.npz"
  uv run fin-regime-phasor discretize apply \
    --phasor out/phasor_full.parquet --codebook "out/codebook_m${m}.npz" --out "out/symbols_m${m}.parquet"
  uv run fin-regime-phasor baselines categorical-hmm \
    --symbols "out/symbols_m${m}.parquet" --n-states 2 --n-symbols "$m" \
    --out "out/categorical_hmm_m${m}.json"
done

python -c "
import json, glob
scores = {f: json.load(open(f))['bic'] for f in glob.glob('out/categorical_hmm_m*.json')}
for f, b in sorted(scores.items(), key=lambda kv: kv[1]):
    print(f'{f}: bic={b:.2f}')
print('best:', min(scores, key=scores.get))
"
```

Set `N_SYMBOLS` to whichever alphabet size had the lowest BIC, then apply the matching codebook as `out/symbols_full.parquet` (copy or re-run `discretize apply` with that codebook).

**Pick `n-states` (regime count)** — per PLAN.md ("Regime count n: decided by BIC, restricted to the interpretable range n in {2,3,4}"), not an unbounded search, and BIC alone is known to under-select regime count in Markov-switching models (Psaradakis & Spagnolo, 2003) — cross-check the winner qualitatively against the `figures regime-timeline` plot and a CUSUM structural-break pass before committing:

```python
%%bash
N_SYMBOLS=8   # from the sweep above

for n in 2 3 4; do
  uv run fin-regime-phasor baselines categorical-hmm \
    --symbols out/symbols_full.parquet --n-states "$n" --n-symbols "$N_SYMBOLS" \
    --out "out/categorical_hmm_n${n}.json"
done

python -c "
import json, glob
scores = {f: json.load(open(f))['bic'] for f in glob.glob('out/categorical_hmm_n*.json')}
for f, b in sorted(scores.items(), key=lambda kv: kv[1]):
    print(f'{f}: bic={b:.2f}')
print('best:', min(scores, key=scores.get))
"
```

With `N_STATES`/`N_SYMBOLS` fixed to the winners of both sweeps, run the full pipeline (HQMM, all classical baselines, and the ablation grid) on the full-scale data:

```python
%%bash
N_STATES=2      # from the BIC sweep above
N_SYMBOLS=8     # from the alphabet sweep above

uv run fin-regime-phasor hqmm train \
  --symbols out/symbols_full.parquet --n-states "$N_STATES" --n-symbols "$N_SYMBOLS" --out out/hqmm_full.npz

uv run fin-regime-phasor baselines gaussian-hmm \
  --features out/phasor_full.parquet --n-states "$N_STATES" --out out/gaussian_hmm_full.json
uv run fin-regime-phasor baselines categorical-hmm \
  --symbols out/symbols_full.parquet --n-states "$N_STATES" --n-symbols "$N_SYMBOLS" --out out/categorical_hmm_full.json
uv run fin-regime-phasor baselines hamilton \
  --returns out/phasor_full.parquet --n-states "$N_STATES" --out out/hamilton_full.json
uv run fin-regime-phasor baselines naive \
  --returns out/phasor_full.parquet --out out/naive_full.json

uv run fin-regime-phasor benchmark grid \
  --phasor out/phasor_full.parquet --n-states "$N_STATES" --n-symbols "$N_SYMBOLS" --out out/benchmark_summary_full.json
```

Both sweeps above use `hmmlearn`-backed classical baselines (cheap) as a proxy for hyperparameter selection, per PLAN.md's leakage-control discipline (anything fit on data must fit on training folds only) — `hqmm train` itself is the expensive GPU/TPU step and is run once, after `N_STATES`/`N_SYMBOLS` are already fixed.

### 9. Generate figures

```python
%%bash
uv run fin-regime-phasor figures phasor-scatter \
  --phasor out/phasor.parquet --out out/phasor_scatter.pdf
uv run fin-regime-phasor figures regime-timeline \
  --bars out/bars.parquet --hqmm-result out/hqmm.npz --out out/regime_timeline.pdf
uv run fin-regime-phasor figures ablation-grid-bic \
  --summary out/benchmark_summary.json --out out/ablation_grid_bic.pdf
```

(`figures phasor-scatter --regimes <path.npy>` overlays ground-truth regime labels when you have them, e.g. from `synthetic generate`'s `--out-regimes` — real market data has no such ground truth, hence it's omitted above.)

`figures hqmm-loss-curve` needs a JSON list of per-step loss values, which `hqmm train` doesn't currently persist to disk (only the final metrics). Extract it directly from the library instead of the CLI:

```python
%%bash
uv run python -c "
import json, polars as pl
from fin_regime_phasor.hqmm.model import train_hqmm

df = pl.read_parquet('out/symbols.parquet')
result = train_hqmm(df['symbol'].to_numpy(), n_states=2, n_symbols=8)
json.dump(result.loss_curve, open('out/loss_curve.json', 'w'))
"
uv run fin-regime-phasor figures hqmm-loss-curve \
  --loss-curve out/loss_curve.json --out out/hqmm_loss_curve.pdf
```

PDFs land in `out/` — download them with `files.download("out/phasor_scatter.pdf")` (`from google.colab import files`) or view them inline via `IPython.display.IFrame`.

## Paper

LaTeX source lives in `paper/` (`main.tex` as root), compiled with `latexmk`:

```bash
latexmk -pdf -cd paper/main.tex
```
