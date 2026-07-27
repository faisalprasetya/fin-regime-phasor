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

Another option is to use single bar parkinson volatility $\sigma$ as the magnitude.

$$\sigma = \sqrt{\dfrac{\left(\ln\dfrac{P_H}{P_L}\right)^2}{4\ln{2}}}$$

$$\sigma = \dfrac{1}{2\sqrt{\ln{2}}} \cdot \ln\dfrac{P_H}{P_L}$$

$$\sigma = \dfrac{\sqrt{\ln{2}}}{2\ln{2}} \cdot \ln\dfrac{P_H}{P_L}$$

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

Assuming $k$ is a constant to determine how aggresive the curvature of the function on early input.

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

