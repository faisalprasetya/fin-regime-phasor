# Financial Phasor Feature Engineering for Quantum Regime-Switching using HQMM

## The Idea

The idea is to construct phasor from financial market variables set as magnitude and phase.

$$z = x + iy$$
$$z = re^{i\theta}$$
$$z = r(\cos{\theta} + i\sin{\theta})$$

- Where magnitude: $r = |z| = \sqrt{x^2 + y^2}$
- and phase: $\theta = \arctan{\dfrac{y}{x}}$

In financial term, magnitude should represent the market intensity and the phase should represent market direction.

### In Search for Magnitude

We can use volume or dollar volume $V$ to represent to represent magnitude, but it'll collapse if we're using aggregated bars like volume bars and dollar bars as the volume or dollar value per data point is relatively static to the threshold we set.

Other option is to use inverse duration $\dfrac{1}{\Delta t}$ to indicate market velocity (i.e., how fast the market to accummulate the threshold volume/dollar).

Another option is to represent market intensity with volatility. There are different types of volatilities in finance. We use single bar parkinson volatility $\sigma$ as the magnitude.

$$\sigma = \sqrt{\dfrac{\left(\ln\dfrac{P_H}{P_L}\right)^2}{4\ln{2}}}$$

$$\sigma = \dfrac{1}{2\sqrt{\ln{2}}} \cdot \ln\dfrac{P_H}{P_L}$$

$$\sigma = \dfrac{\sqrt{\ln{2}}}{2\ln{2}} \cdot \ln\dfrac{P_H}{P_L}$$

#### Comparing the candidate magnitude measures

The three candidates, $V$ (volume/dollar volume), $\frac{1}{\Delta t}$ (inverse duration), and $\sigma$ (Parkinson volatility), differ most in how they behave under a chosen bar-sampling scheme, which is decisive here since financial ML practice commonly samples bars on time, volume, or dollar thresholds (López de Prado, 2018).

Under **volume or dollar bars**, each bar is constructed to accumulate a fixed threshold of $V$ by definition. This makes $V$ per bar approximately constant across the sample, equal to the threshold, so it degenerates into a near-flat magnitude signal, the exact collapse noted above. $V$ therefore only carries usable information under time bars, where accumulated volume/dollar value genuinely varies bar to bar; it is not portable across sampling schemes.

$\frac{1}{\Delta t}$ resolves the degeneracy: under volume/dollar bars, $\Delta t$ (the time needed to accumulate the threshold) is precisely the quantity that varies, so $1/\Delta t$ is a legitimate proxy for the rate of information arrival, the same intuition behind the "volume clock" (Easley, López de Prado & O'Hara, 2012). However, it has two structural weaknesses: (1) it is unbounded above and heavy-tailed, a burst of trades accumulating the threshold almost instantaneously drives $\Delta t \to 0$ and $1/\Delta t \to \infty$, with no natural ceiling; and (2) it measures *how fast* the threshold filled, not *how much the price moved* while it did, so two bars with identical duration but very different intrabar price dispersion are assigned the same magnitude.

$\sigma$ (Parkinson volatility) is computed purely from the intrabar high/low range and is agnostic to how the bar boundary was chosen, it is equally well-defined under time, volume, tick, or dollar bars, so it does not inherit the collapse that afflicts $V$. It is also a well-studied estimator: under the assumption of a driftless geometric Brownian motion, the Parkinson (1980) range estimator is roughly 5 times more statistically efficient than the close-to-close realized-variance estimator for the same number of observations, because it uses the full intrabar path information rather than only the bar's endpoints.

| Property | $V$ (volume/dollar) | $1/\Delta t$ (inverse duration) | $\sigma$ (Parkinson volatility) |
|---|---|---|---|
| What it captures | Traded volume/turnover | Rate of threshold accumulation ("clock speed") | Intrabar price dispersion |
| Behavior under time bars | Varies meaningfully | Roughly constant (bar length fixed) | Varies meaningfully |
| Behavior under volume/dollar bars | Collapses to ≈constant (the threshold) | Varies meaningfully | Varies meaningfully |
| Boundedness | Bounded below by 0, no natural upper bound | Unbounded, blows up as $\Delta t \to 0$ | Bounded below by 0, grows smoothly with range |
| Statistical grounding | Microstructure proxy, no distributional assumption | Ad hoc "clock" proxy, no distributional assumption | Efficient estimator under GBM (Parkinson, 1980), ~5x efficiency of close-to-close variance |
| Captures price movement directly | No | No | Yes |
| Portable across bar-sampling schemes | No | Yes | Yes |
| Known biases | Structural collapse under volume/dollar bars | Downward-biased duration drives magnitude blow-up; ignores price path | Downward-biased under jumps/discreteness; assumes no drift |

**Justification.** Because the phasor construction should not be structurally tied to one bar-sampling scheme, a magnitude measure that degenerates under volume/dollar bars ($V$) is disqualified outright for this purpose. Between the remaining two, $1/\Delta t$ is a legitimate but noisier and unbounded signal that reflects order-flow speed rather than price movement, the actual quantity "intensity" is meant to track. $\sigma$ (Parkinson volatility) directly measures how much the price moved within the bar, is well-behaved (bounded below by zero, smooth, no blow-up), is portable across any bar type, and is backed by a well-established efficiency result in the range-volatility literature. It is therefore the preferred choice of magnitude for this construction.

### In Search for Phase

Market direction can be represented by the price change. It can go up and down representing the market direction.

$$R = \dfrac{P_i - P_{i-1}}{P}$$
$$R = \dfrac{\Delta P}{P}$$
$$R = \dfrac{1}{P} \cdot \Delta P$$

In continuous notation

$$R = \dfrac{1}{P} \,dP$$
$$\int{R} = \int{\dfrac{1}{P} \,dP}$$
$$\int{R} = \ln{P} + C$$
$$d\int{R} = d(\ln{P} + C)$$
$$R = d(\ln{P})$$

In discrete notation we get

$$R = \Delta(\ln{P})$$
$$R = \ln{P_i} - \ln{P_{i-1}}$$
$$R = \ln{\dfrac{P_i}{P_{i-1}}}$$

But the phase in a complex number will be evaluated under sine and cosine which are cyclical operators. This can cause large price change be treated the same way as small price change if they're fell under the same period in the cycle. To avoid this, we need to map the log return with function $f: \mathbb{R} \to (-\pi, \pi)$. There are several functions that we can use.

Suppose $k$ is a sensitivity (scale) constant that sets how steeply $\theta$ responds to small values of $R$, larger $k$ makes the mapping saturate toward $\pm\pi$ faster for small log returns, while smaller $k$ keeps the mapping closer to linear over a wider range of $R$.

#### Scaled sigmoid

$$\theta = \pi \left(2 \cdot \dfrac{1}{1+e^{-kR}} - 1 \right)$$
$$\theta = \pi \cdot \dfrac{2-(1+e^{-kR})}{1+e^{-kR}}$$
$$\theta = \pi \cdot \dfrac{1-e^{-kR}}{1+e^{-kR}}$$

#### Scaled tanh

$$\theta = \pi \cdot \dfrac{e^{kR} - e^{-kR}}{e^{kR} + e^{-kR}}$$

But wait?, if we do this:

$$\theta = \pi \cdot \dfrac{e^{kR} - e^{-kR}}{e^{kR} + e^{-kR}} \cdot \dfrac{e^{-kR}}{e^{-kR}}$$

$$\theta = \pi \cdot \dfrac{1 - e^{-2kR}}{1 + e^{-2kR}}$$

It essentially the same as sigmoid with doubled constant $k$. We can use one of them and ignore the other.

#### 2 arctan

$$\theta = 2\arctan{(kR)}$$

#### Scaled softsign

$$\theta = \pi \cdot \dfrac{kR}{1+|kR|}$$

#### Comparing the candidate mappings

Scaled sigmoid and scaled tanh are the same family, as shown above, the sigmoid form collapses to tanh under a rescaled constant. So there are really three distinct candidates: $\pi\tanh(kR)$, $2\arctan(kR)$, and the softsign form $\pi \cdot \frac{kR}{1+|kR|}$.

Near $R = 0$ (small-return regime), all three are locally linear with slope proportional to $k$, consistent with $k$'s role as a sensitivity constant:

$$\pi\tanh(kR) \approx \pi kR, \qquad 2\arctan(kR) \approx 2kR, \qquad \pi \cdot \dfrac{kR}{1+|kR|} \approx \pi kR$$

As $R \to \infty$ (large-return / saturation regime), the three diverge sharply — this is the more consequential comparison:

$$\pi - \pi\tanh(kR) \sim 2\pi e^{-2kR} \quad \text{(exponential decay)}$$
$$\pi - 2\arctan(kR) \sim \dfrac{2}{kR} \quad \text{(polynomial decay)}$$
$$\pi - \pi \cdot \dfrac{kR}{1+kR} \sim \dfrac{\pi}{kR} \quad \text{(polynomial decay)}$$

| Property | $\pi\tanh(kR)$ | $2\arctan(kR)$ | $\pi \cdot \frac{kR}{1+\lvert kR\rvert}$ |
|---|---|---|---|
| Range | $(-\pi,\pi)$ | $(-\pi,\pi)$ | $(-\pi,\pi)$ |
| Odd (sign-preserving) | Yes | Yes | Yes |
| Smoothness | $C^\infty$ | $C^\infty$ | $C^1$ only, curvature jumps at $R=0$ |
| Slope at $R=0$ | $\pi k$ | $2k$ | $\pi k$ |
| Saturation rate | Exponential | $\sim 1/R$ | $\sim 1/R$ (heavier than arctan by factor $\pi/2$) |
| Tail information retained | Low to extreme returns collapse together fast | High | High |
| Cost per evaluation | 1 transcendental (`exp`) | 1 transcendental (`atan`) | 0 transcendental (`abs`, divide) |
| Consistent w/ $\theta=\arctan(y/x)$ definition (line 12) | No | Yes | No |
| Outlier robustness | High (hard clip) | Moderate | Moderate |

**Justification.** Financial log-returns are well documented as leptokurtic/fat-tailed (Mandelbrot 1963; Cont 2001), so large moves are both more frequent and more informationally significant than a thin-tailed assumption would suggest. $\pi\tanh(kR)$'s exponential saturation compresses any sufficiently large $|R|$ to nearly the same phase near $\pm\pi$ almost immediately, it cannot distinguish "large" from "extreme," which is exactly the regime-discriminating signal an HQMM would want to preserve. The polynomially-decaying candidates keep resolving magnitude differences much further into the tail.

Between $2\arctan(kR)$ and softsign, arctan is preferred on two grounds: it is $C^\infty$-smooth, whereas softsign has a curvature discontinuity at $R=0$ (a property that matters if $\theta$ or its derivatives propagate further, e.g. through unitary phase operators in the HQMM, or through gradient-based fitting of $k$); and it reuses the same function already defining the phasor's phase, $\theta = \arctan(y/x)$, keeping one functional family across the construction rather than mixing in an unrelated nonlinearity.

$$\theta = 2\arctan(kR)$$

is therefore the recommended mapping. Softsign remains worth revisiting only if profiling shows `atan` is a throughput bottleneck at tick-level scale.

## Constructing the Phasor

With magnitude and phase settled, the single-bar phasor $z$ combines the Parkinson volatility $\sigma$ (line 22) as magnitude and the doubled-arctan of the scaled log return $\theta = 2\arctan(kR)$ (line 99) as phase. Indexing by bar $i$ makes explicit that this is a per-bar quantity:

$$r_i = \sigma_i, \qquad \theta_i = 2\arctan(kR_i)$$

**Euler form**

$$z_i = \sigma_i \, e^{i \cdot 2\arctan(kR_i)}$$

**Rectangular form**

$$z_i = \sigma_i \cos\!\big(2\arctan(kR_i)\big) + i\,\sigma_i \sin\!\big(2\arctan(kR_i)\big)$$

Because $\theta_i$ is exactly *twice* an arctangent, the Weierstrass (tangent half-angle) identities collapse the rectangular components to a rational function of $kR_i$ alone, with no trigonometric or inverse-trigonometric evaluation required:

$$\cos\big(2\arctan(w)\big) = \dfrac{1-w^2}{1+w^2}, \qquad \sin\big(2\arctan(w)\big) = \dfrac{2w}{1+w^2}, \qquad w = kR_i$$

$$z_i = \sigma_i \cdot \dfrac{\big(1-(kR_i)^2\big) + i\cdot 2kR_i}{1+(kR_i)^2}$$

<!-- which is equivalent to the closed-form Möbius (Cayley) transform -->

Let $m = ikR_i$ thus $m^2 = -(kR_i)^2$

$$z_i = \sigma_i \cdot \dfrac{\big(1+m^2\big) + 2m}{1-m^2}$$
$$z_i = \sigma_i \cdot \dfrac{(1+m)^2}{(1+m)(1-m)}$$
$$z_i = \sigma_i \cdot \dfrac{1+m}{1-m}$$

Substitute $m = ikR_i$ back:

$$z_i = \sigma_i \cdot \dfrac{1+ikR_i}{1-ikR_i}$$

This is more than a notational convenience: it means $x_i = \mathrm{Re}(z_i)$ and $y_i = \mathrm{Im}(z_i)$ can be computed directly from $\sigma_i$, $k$, and $R_i$ using only arithmetic, without ever calling `arctan`, `sin`, or `cos`, a meaningful saving at tick-level throughput, and a clean confirmation that $2\arctan(kR)$ composes naturally with the phasor's own polar-to-rectangular machinery.
 
## The Research

### Objective

Feed the per-bar phasor $z_i = \sigma_i e^{i\theta_i}$ into a Hidden Quantum Markov Model (HQMM) to detect latent market regimes, benchmark the result against a rigorous set of classical baselines, and demonstrate practical value by using the detected regimes as a risk-management overlay on a trading strategy.

### Why not `hmmlearn`

`hmmlearn` only supports real-valued emission distributions (Gaussian, GMM, Multinomial, Poisson) evolved by a classical stochastic transition matrix. It has no notion of a complex probability amplitude, unitary/Kraus evolution, or quantum interference between hidden-state hypotheses — all of which are the actual mechanism by which an HQMM can, for the same number of hidden states, represent strictly more expressive process classes than a classical HMM (Monras, Beige & Wiesner, 2010). Since the phasor is explicitly built as a complex number to exploit that structure, we need a library that supports complex linear algebra and gradient-based fitting — hence JAX (`jax.numpy` complex dtypes, `jax.grad`, `jax.scipy.linalg.expm` for unitary parameterization, `optax` for optimization).

### HQMM formalism (planning notes)

- **State:** represented as a density matrix $\rho \in \mathbb{C}^{n\times n}$ (mixed-state), rather than a classical probability vector over $n$ regimes, to allow superposition between hypothesized regimes.
- **Transition + emission:** modeled jointly via a set of Kraus operators $\{K_a\}_{a\in\mathcal{A}}$, one per observation symbol $a$, updating the state as $\rho \mapsto \dfrac{K_a \rho K_a^\dagger}{\mathrm{Tr}(K_a \rho K_a^\dagger)}$, with $\mathrm{Tr}(K_a\rho K_a^\dagger)$ giving $P(a\mid\rho)$ (Monras et al., 2010; Srinivasan, Gordon & Boots, 2018).
- **Constraint:** completeness $\sum_a K_a^\dagger K_a = I$ must hold for $\{K_a\}$ to be a valid quantum channel. Following Srinivasan et al. (2018), parameterize each $K_a$ implicitly through one larger isometry/unitary (Stinespring dilation) fit by unconstrained gradient descent, then sliced out — this sidesteps having to project onto the completeness constraint at every optimizer step. JAX's autodiff plus `expm` makes this practical without hand-deriving update rules.
- **Training objective:** negative log-likelihood of the observed bar sequence, matching classical Baum–Welch's objective, but optimized by gradient descent rather than EM, since no closed-form M-step exists for Kraus-operator parameters.

### Discretizing the phasor for a discrete-observation HQMM

All practical HQMM training methods in the literature (Monras et al., 2010; Srinivasan et al., 2018) assume a **finite observation alphabet** $\mathcal{A}$ — there is no established continuous-observation HQMM analogous to a continuous-emission classical HMM. $z_i$ is complex-valued and continuous, so it must be discretized before it can index a Kraus operator.

Planned approach: vector-quantize $(\sigma_i, \theta_i)$ jointly (not $z_i$'s raw real/imaginary parts, to avoid distorting the polar semantics) via k-means or quantile binning into a modest alphabet (e.g. 8–16 symbols), fit on the training split only to avoid look-ahead leakage into the codebook itself. A finer alphabet gives the HQMM more resolution but costs more Kraus-operator parameters ($O(n^2\times|\mathcal{A}|)$ real parameters); this trade-off needs a small sweep.

Stretch goal (explicitly out of scope for the first pass): a continuous-observable POVM model (e.g. treating $\theta_i$ as a continuous phase-estimation observable) would avoid discretization entirely, but is a materially larger research undertaking with no settled reference implementation — flagged here so it isn't silently dropped, not committed to.

### Benchmark design

For the comparison to be academically defensible, a win (if any) has to be attributed to the right cause: is it the phasor feature representation, or the quantum mechanism? A single classical-vs-quantum comparison on the phasor alone can't answer that, so the plan uses a small ablation grid crossing **representation** against **model class**:

| | Classical HMM (`hmmlearn`, Gaussian emissions) | HQMM (JAX, Kraus operators) |
|---|---|---|
| Raw features ($R_i$, $\sigma_i$ as two real numbers) | Baseline A — plain classical regime model | Baseline C — quantum mechanism, no phasor |
| Phasor features ($z_i$, discretized) | Baseline B — phasor representation, classical mechanism | **Target model** |

- **A vs. Target** isolates the combined effect (what most papers report, but it's underdetermined on its own).
- **A vs. B** isolates whether wrapping $(\sigma, R)$ into the bounded-phase polar form helps a classical model.
- **A vs. C / B vs. Target** isolates whether the quantum (Kraus/interference) mechanism helps, holding the feature representation fixed.

Additional baselines for external validity:
- **Hamilton (1989) Markov-switching model** on log-returns — the canonical econometric regime-switching reference, included so the comparison isn't only against other ML models.
- **No-regime baseline** — a single unconditional Gaussian / buy-and-hold, to sanity-check that any regime model beats "no regimes at all."

**Sanity check before touching real data:** first fit the HQMM and Baseline A on **synthetic data generated from a known regime-switching process** (e.g. a simulated 2–3 state Markov-switching GBM) with ground-truth regime labels, and confirm the models actually recover the planted regimes. Financial data has no ground truth, so this is the only way to validate that the JAX HQMM implementation and training procedure work correctly before trusting its output on market data.

**Metrics:** held-out log-likelihood / perplexity; AIC/BIC (with the caveat that BIC-style penalties are known to under-select the true number of regimes in Markov-switching models, e.g. Psaradakis & Spagnolo, 2003, so state count won't be chosen by information criterion alone); qualitative alignment of detected regime-switch timestamps against known structural events (e.g. Mar 2020 COVID crash, May 2022 Terra/Luna collapse, Nov 2022 FTX collapse, for the crypto sample); and agreement with an independent structural-break detector (CUSUM, per López de Prado AFML ch. 17) as an external check that isn't derived from either model.

### From regime detection to a trading strategy

Framing chosen: **regime probability as a risk-sizing overlay**, not a standalone alpha signal — this is both more defensible (regime detection is a weak/noisy prior on risk, not a strong directional edge) and lets the phasor's two channels do distinct jobs: $\theta_i$ (direction) informs which regime looks trend-persistent vs. choppy, and $\sigma_i$ (intensity) feeds a volatility-target position size. Concretely:

- Base strategy: a simple trend-following or vol-targeting position sizer on the same instrument used for feature construction.
- Overlay: scale/gate exposure by the HQMM's posterior regime probability — e.g. shrink or flatten exposure when the model assigns high probability to a historically high-drawdown regime, and scale up (to a leverage cap) in low-risk regimes.
- Compare against: the same base strategy (a) unconditioned, (b) gated by the classical-HMM baseline instead of the HQMM, and (c) buy-and-hold.

**Backtest discipline (López de Prado):** walk-forward evaluation using combinatorial purged cross-validation (CPCV) with embargo (AFML ch. 12) for any hyperparameter selection (alphabet size, $k$, number of hidden states, vol-target level); report the deflated Sharpe ratio and an estimate of the probability of backtest overfitting (PBO) (Bailey & López de Prado, 2014) rather than a single in-sample Sharpe, given how many knobs this pipeline has; account for transaction costs and realistic slippage given bar-triggered (not fixed-time) rebalancing.

### Dataset

| | Recommendation | Rationale |
|---|---|---|
| Primary asset | BTC/USDT perpetual futures, Binance | Deepest liquidity and longest continuous history among crypto derivatives; 24/7 trading avoids overnight-gap artifacts in bar construction |
| Data source | `data.binance.vision` public archive (monthly `aggTrades` dumps) | Free, no rate limits, trade-level granularity required to build genuine dollar/volume bars (not just resampled OHLC) |
| Primary sample | 2020-01-01 to 2025-12-31 | Spans COVID crash, 2021 bull, 2022 bear (Luna, FTX), 2023–24 recovery — structurally distinct regimes to validate against |
| Held-out test | 2026-01-01 to present (2026-07-27) | Kept fully untouched until the final, single evaluation pass — no CV, no tuning, touched once |
| Secondary/robustness check (stretch goal) | SPY or ES futures via a paid tick-data source (Polygon.io, Databento) | Tests whether any edge is crypto-idiosyncratic or generalizes to a different market microstructure; free sources (Yahoo/Stooq) only offer time-bar OHLCV and can't support proper dollar-bar construction, so this step is gated on budget/access, not committed to for v1 |

### Data pipeline (López de Prado best practices)

- **Bars:** dollar bars (AFML ch. 2) built from raw trades, not fixed-time bars — this is also *why* the magnitude comparison earlier in this document mattered: $V$ collapses under dollar bars by construction, which is precisely the failure mode dollar bars are chosen to avoid measuring with. Threshold picked via a target-average-bars-per-day heuristic, calibrated on the training period only.
- **Leakage control:** any fitting step that touches the future relative to its evaluation window (VQ codebook for discretization, frac-diff order search, $k$, hidden-state count, HMM/HQMM parameters) is fit strictly on training folds; hyperparameter search uses purged K-fold / CPCV with embargo around each fold boundary to remove leakage from serial correlation in overlapping bar windows (AFML ch. 7, 12).
- **Sample uniqueness / sequential bootstrap:** relevant only if a downstream supervised layer is later trained on overlapping-outcome labels (e.g. a meta-labeling classifier consuming the HQMM regime posterior); not required for the HQMM's own unsupervised sequence-likelihood fitting, since it consumes the bar sequence directly rather than overlapping labeled windows.

### Fractional differentiation: where does it belong?

The goal of frac-diff (AFML ch. 5) is to reach stationarity while discarding as little memory as possible. The original framing below treated $R_i=\ln(P_i/P_{i-1})$ as fixed and asked only whether the *mapped* phase $\theta_i$ needed further treatment. That skipped a prior question: is $d=1$ (the standard log return) even the right amount of differencing to feed the phase map in the first place? López de Prado's own point in AFML ch. 5 is that $d=1$ is usually *more* differencing than the minimum required for stationarity — his examples typically find a minimum stationary order $d^{*}$ well below 1 (e.g. $\approx 0.35$–$0.5$ for E-mini S&P), with correlation to the original price series collapsing to near-zero by $d=1$. So the standard log return isn't "correctly differenced," it's "maximally differenced for simplicity" — and it likely throws away more memory than necessary before phase is even computed.

**The fix: search for the minimal $d$ on $\ln P$ itself, not on $R$.** The fractional differencing operator applied to $\ln P$,

$$\mathrm{FFD}_d(\ln P)_i = \sum_{k=0}^{\infty} w_k \ln P_{i-k}, \qquad w_k = (-1)^k \binom{d}{k}$$

has the log return as an exact special case: at $d=1$, $w_0=1,\ w_1=-1,\ w_k=0$ for $k\geq 2$, so $\mathrm{FFD}_1(\ln P)_i = \ln P_i - \ln P_{i-1} = R_i$ — recovering exactly the definition derived earlier (lines 55–73). Rather than assuming $d=1$, run the standard AFML minimum-$d$ search (ADF statistic vs. $d\in[0,1]$) on $\ln P$ directly, take the minimal $d^{*}_R$ that clears the stationarity threshold, and use that as the phase input:

$$R_i^{*} = \mathrm{FFD}_{d^{*}_R}(\ln P)_i, \qquad \theta_i = 2\arctan\!\big(k R_i^{*}\big)$$

This strictly subsumes the original definition — if the search happens to find $d^{*}_R\approx 1$ for this price series and bar type, $R_i^{*}$ collapses back to the plain log return; if it finds $d^{*}_R<1$, phase is built from a more memory-preserving signal, which matters more here than it might elsewhere, since phase is explicitly meant to represent *directional persistence*, and a maximally-differenced, near-white-noise return is exactly the kind of signal that discards it.

**The mapped angle $\theta_i$ itself still should not be frac-diffed.** This part of the original argument holds regardless of which $d^{*}_R$ the search finds: $\theta_i$ is bounded, angular, and already downstream of whatever stationarity the search on $\ln P$ established. Running a second differencing pass on the angle itself would risk producing linear combinations of angles across bars that no longer have a clean directional meaning, for no additional stationarity benefit — the only differencing decision on the return channel belongs on its *input* ($\ln P$), not its *output* ($\theta$).

**Magnitude $\sigma_i$ still needs its own, separate check.** Volatility is a levels variable with well-documented long memory (slowly decaying autocorrelation, near-unit-root behavior in persistent-vol regimes — the stylized fact behind FIGARCH and long-memory realized-volatility models), and there's no reason its minimum stationary order $d^{*}_\sigma$ should match $d^{*}_R$ — they are different series with potentially different memory structure, so each gets its own independent ADF-vs-$d$ search.

**Architecture decision, revised:**

1. Run the AFML minimum-$d$ search independently on $\ln P$ (yielding $d^{*}_R$, feeding phase) and on $\sigma_i$ (yielding $d^{*}_\sigma$, feeding magnitude) — two separate searches, since there's no reason to expect the same order for both.
2. Compute $R_i^{*} = \mathrm{FFD}_{d^{*}_R}(\ln P)_i$ and $\theta_i = 2\arctan(kR_i^{*})$.
3. Compute $\sigma_i^{*} = \mathrm{FFD}_{d^{*}_\sigma}(\sigma)_i$.
4. Reconstruct $z_i^{*} = \sigma_i^{*}\, e^{i\theta_i}$ — magnitude and phase are each frac-diffed on their own native series *before* combining into the phasor, for the same reason as before: frac-diffing the complex $z_i$ directly would mix magnitude and phase history across lags into both components of the result and destroy the intensity/direction decomposition.

Use the fixed-width-window (FFD) variant in both searches (AFML §5.4), not the expanding-window version, since it has unbounded lookback and isn't viable for a causal bar-by-bar pipeline.

**Downstream check to add.** The earlier choice of $2\arctan(kR)$ over $\pi\tanh(kR)$ (see "Comparing the candidate mappings") leaned partly on log returns being fat-tailed, favoring a polynomial- over exponential-tail mapping. $\mathrm{FFD}_{d^{*}_R}$ is a weighted average over many lags, which tends to mildly smooth tails relative to a raw one-bar return. Once $R_i^{*}$ replaces $R_i$, re-check empirically that it's still fat-tailed enough for the arctan-over-tanh argument to hold — this doesn't invalidate the earlier reasoning, but it is now a claim about a different series and should be reconfirmed rather than assumed to carry over.

**Verification plan.** Both $d^{*}_R$ and $d^{*}_\sigma$ are outputs of the search, not assumptions to bake in — there is no longer a predicted value for $d^{*}_R$ the way the original plan predicted $d^{*}_\theta\approx 0$; the point of this section is precisely to let the data determine how much memory the phase input can retain while staying stationary. $d^{*}_\sigma > 0$ remains the one directional prediction worth stating explicitly, per the long-memory argument above.

### Open questions / risks

- Number of hidden regimes ($n$) is a real design choice, not just a value to grid-search: information criteria are known to under-select regime count in Markov-switching contexts, so an economically motivated choice (e.g., 2–4 interpretable regimes) may be preferable to picking $n$ by BIC alone.
- The VQ-alphabet size for discretizing $(\sigma_i,\theta_i)$ trades resolution against Kraus-operator parameter count ($O(n^2\times|\mathcal{A}|)$) — needs its own small sweep, tuned only on training folds.
- $k$ (phase sensitivity constant) interacts with the discretization step (it reshapes the distribution $\theta$ is drawn from before binning) — it may need to be tuned jointly with the VQ codebook rather than fixed in isolation.
- Continuous-observable (POVM) HQMM avoids discretization entirely but has no settled reference implementation; treated as future work, not part of the initial plan.

