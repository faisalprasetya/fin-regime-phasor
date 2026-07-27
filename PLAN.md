# Financial Phasor Feature Engineering for Quantum Regime-Switching using HQMM

The idea is to construct phasor from financial market variables set as magnitude and phase.

$$z = x + iy$$
$$z = re^{i\theta}$$
$$z = r(\cos{\theta} + i\sin{\theta})$$

- Where magnitude: $r = |z| = \sqrt{x^2 + y^2}$
- and phase: $\theta = \arctan{\dfrac{y}{x}}$

In financial term, magnitude should represent the market intensity and the phase should represent market direction.

We can use volume or dellar volume ($V$) to represent to represent magnitude, but it'll collapse if we're using aggregated bars like volume bars and dollar bars as the volume or dollar value per data point is relatively static to the threshold we set.

Other option is to use inverse duration ($\dfrac{1}{\Delta t}$) to indicate market velocity (i.e., how fast the market to accummulate the threshold volume/dollar).

Another option is to use single bar parkinson volatility ($\sigma$) as the magnitude.
