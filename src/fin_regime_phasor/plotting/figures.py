"""Figure-generating functions, one per named CLI sub-command (see `cli.figures_cmd`),
so every figure is reproducible byte-for-byte from `data/` + code (CLAUDE.md
"Plotting standards").
"""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

# Placeholder single-column width (paper/main.tex doesn't exist yet); update to the
# document's actual \textwidth once it does, per CLAUDE.md "Sizing".
TEXTWIDTH_IN = 6.5

# Okabe-Ito, colorblind-safe and distinguishable in grayscale.
REGIME_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]


def _figsize(width_frac: float = 1.0, aspect: float = 0.65) -> tuple[float, float]:
    width = TEXTWIDTH_IN * width_frac
    return width, width * aspect


def plot_phasor_scatter(
    sigma: np.ndarray, theta: np.ndarray, regimes: np.ndarray | None = None
) -> Figure:
    """Phasor points in the complex plane (x = sigma*cos(theta), y = sigma*sin(theta)),
    colored by regime label if provided."""
    x = sigma * np.cos(theta)
    y = sigma * np.sin(theta)

    fig = Figure(figsize=_figsize(aspect=1.0))
    ax = fig.add_subplot(111)

    if regimes is None:
        ax.scatter(x, y, s=4, alpha=0.5, color=REGIME_COLORS[0])
    else:
        for i, state in enumerate(np.unique(regimes)):
            mask = regimes == state
            ax.scatter(
                x[mask],
                y[mask],
                s=4,
                alpha=0.5,
                color=REGIME_COLORS[i % len(REGIME_COLORS)],
                label=f"regime {state}",
            )
        ax.legend(loc="upper right", markerscale=3)

    ax.set_xlabel(r"$\mathrm{Re}(z) = \sigma\cos\theta$")
    ax.set_ylabel(r"$\mathrm{Im}(z) = \sigma\sin\theta$")
    ax.set_aspect("equal", adjustable="datalim")
    return fig


def plot_regime_timeline(close_price: np.ndarray, regime_posteriors: np.ndarray) -> Figure:
    """Price series (top) and per-regime posterior probability (bottom, stacked area)."""
    n_states = regime_posteriors.shape[1]
    t = np.arange(len(close_price))

    fig = Figure(figsize=_figsize(aspect=0.8))
    ax_price, ax_regime = fig.subplots(2, 1, sharex=True, height_ratios=[2, 1])

    ax_price.plot(t, close_price, color=REGIME_COLORS[0], linewidth=0.8)
    ax_price.set_ylabel("price")

    ax_regime.stackplot(
        t,
        regime_posteriors.T,
        colors=[REGIME_COLORS[i % len(REGIME_COLORS)] for i in range(n_states)],
        labels=[f"regime {i}" for i in range(n_states)],
    )
    ax_regime.set_ylabel("P(regime)")
    ax_regime.set_xlabel("bar index")
    ax_regime.set_ylim(0, 1)
    ax_regime.legend(loc="upper right", ncol=n_states)
    return fig


def plot_loss_curve(loss_curve: list[float]) -> Figure:
    """HQMM training loss (per-step negative log-likelihood) over optimizer steps."""
    fig = Figure(figsize=_figsize())
    ax = fig.add_subplot(111)
    ax.plot(np.arange(len(loss_curve)), loss_curve, color=REGIME_COLORS[0])
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("negative log-likelihood / bar")
    return fig


def plot_ablation_grid_bic(grid_summary: dict[str, dict[str, float]]) -> Figure:
    """Bar chart of BIC across the four (representation, mechanism) ablation cells."""
    cells = [k for k in grid_summary if "_" in k and k.split("_")[0] in ("raw", "phasor")]
    bics = [grid_summary[c]["bic"] for c in cells]

    fig = Figure(figsize=_figsize())
    ax = fig.add_subplot(111)
    ax.bar(
        range(len(cells)),
        bics,
        color=[REGIME_COLORS[i % len(REGIME_COLORS)] for i in range(len(cells))],
    )
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels(cells, rotation=20, ha="right")
    ax.set_ylabel("BIC (lower is better)")
    return fig
