# Options Pricing

The `quantflow.options` subpackage provides two analytic option pricers that share a common
parameterisation:

- {class}`~quantflow.BlackScholes` -- the closed-form Black-Scholes-Merton model for
  **European** options, with a full Greeks suite and an implied-volatility solver.
- {class}`~quantflow.BinomialTree` -- a Cox-Ross-Rubinstein (CRR) lattice that prices
  both **European** and **American** options and converges to the Black-Scholes value as the
  number of steps grows.

Both classes take the same core inputs: spot price `S`, strike `K`, time to expiry `T` (in
years), the annualised risk-free rate `r`, annualised volatility `sigma`, an `option_type`
of `'call'` or `'put'`, and an optional continuous dividend yield `q`.

:::{note}
All sensitivities follow the package conventions: **vega** and **rho** are reported
**per 1 percentage point** change (i.e. already divided by 100), and **theta** is reported
**per year** (divide by 365 for a calendar-day estimate). These conventions are identical in
both pricers, so results are directly comparable.
:::

## Black-Scholes-Merton pricing

The {class}`~quantflow.BlackScholes` model assumes constant volatility and interest rates and
a lognormal terminal price. With a continuous dividend yield $q$, the price of a European call
and put are

$$
C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2), \qquad
P = K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1),
$$

$$
d_1 = \frac{\ln(S/K) + (r - q + \tfrac{1}{2}\sigma^2) T}{\sigma\sqrt{T}}, \qquad
d_2 = d_1 - \sigma\sqrt{T},
$$

where $N(\cdot)$ is the standard normal CDF. Setting `q=0.0` (the default) recovers the classic
non-dividend Black-Scholes formula.

```python
from quantflow import BlackScholes

# A 3-month, out-of-the-money European call (no dividends)
call = BlackScholes(S=100, K=105, T=0.25, r=0.05, sigma=0.20, option_type="call")
put = BlackScholes(S=100, K=105, T=0.25, r=0.05, sigma=0.20, option_type="put")

print(repr(call))
print(f"Call: {call.price():.4f}")
print(f"Put:  {put.price():.4f}")
```

```text
BlackScholes(Call: S=100.0, K=105.0, T=0.250, r=0.050, σ=0.200, q=0.000)
Call: 2.4779
Put:  6.1736
```

### The dividend yield `q`

A positive dividend yield lowers the effective drift of the underlying, which **reduces call
values** and **raises put values**. Use `q` for dividend-paying equities, equity indices, or
the foreign rate in an FX option (the Garman-Kohlhagen convention).

```python
call_nodiv = BlackScholes(S=100, K=105, T=0.25, r=0.05, sigma=0.20, q=0.00)
call_div = BlackScholes(S=100, K=105, T=0.25, r=0.05, sigma=0.20, q=0.03)

print(f"q = 0%:  {call_nodiv.price():.4f}")
print(f"q = 3%:  {call_div.price():.4f}")
```

```text
q = 0%:  2.4779
q = 3%:  2.2066
```

### Put-call parity

For European options the call and put prices are linked by put-call parity,

$$
C - P = S e^{-qT} - K e^{-rT}.
$$

This is a model-free no-arbitrage identity; the package satisfies it to machine precision.

```python
import numpy as np

S, K, T, r, q = 100, 100, 1.0, 0.05, 0.0
c = BlackScholes(S, K, T, r, 0.20, "call")
p = BlackScholes(S, K, T, r, 0.20, "put")

lhs = c.price() - p.price()
rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
print(f"C - P            = {lhs:.10f}")
print(f"Se^-qT - Ke^-rT  = {rhs:.10f}")
```

```text
C - P            = 4.8770575499
Se^-qT - Ke^-rT  = 4.8770575499
```

:::{tip}
Parity is a quick sanity check when wiring up your own calculations: if `call.price() -
put.price()` does not equal $S e^{-qT} - K e^{-rT}$ for two options sharing the same `S, K, T,
r, q`, one of the inputs differs.
:::

## The Greeks

The Greeks measure the option price's sensitivity to each input. Compute them individually
({meth}`~quantflow.BlackScholes.delta`, {meth}`~quantflow.BlackScholes.gamma`,
{meth}`~quantflow.BlackScholes.theta`, {meth}`~quantflow.BlackScholes.vega`,
{meth}`~quantflow.BlackScholes.rho`) or all at once with
{meth}`~quantflow.BlackScholes.greeks`.

| Greek | Symbol | Measures | Package units |
|-------|--------|----------|---------------|
| Delta | $\Delta = \partial V / \partial S$ | price change per \$1 move in spot | per \$1 |
| Gamma | $\Gamma = \partial^2 V / \partial S^2$ | change in delta per \$1 move in spot | per \$1 |
| Theta | $\Theta = \partial V / \partial t$ | time decay | **per year** |
| Vega  | $\nu = \partial V / \partial \sigma$ | sensitivity to volatility | **per 1% vol** |
| Rho   | $\rho = \partial V / \partial r$ | sensitivity to the rate | **per 1% rate** |

The closed-form expressions (with dividend yield $q$) are

$$
\Delta_{\text{call}} = e^{-qT} N(d_1), \qquad
\Delta_{\text{put}} = e^{-qT}\bigl(N(d_1) - 1\bigr), \qquad
\Gamma = \frac{e^{-qT}\varphi(d_1)}{S\sigma\sqrt{T}},
$$

$$
\nu = \frac{S e^{-qT}\varphi(d_1)\sqrt{T}}{100}, \qquad
\rho_{\text{call}} = \frac{K T e^{-rT} N(d_2)}{100}, \qquad
\rho_{\text{put}} = -\frac{K T e^{-rT} N(-d_2)}{100},
$$

where $\varphi(\cdot)$ is the standard normal PDF. Vega and rho carry the $/100$ factor so that
the returned number is already the change for a **1 percentage point** move.

```python
call = BlackScholes(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type="call")
for name, value in call.greeks().items():
    print(f"{name:>6}: {value:>10.6f}")
```

```text
 price:  10.450584
 delta:   0.636831
 gamma:   0.018762
 theta:  -6.414028
  vega:   0.375240
   rho:   0.532325
```

Reading these for the at-the-money call above: a \$1 rise in spot adds about \$0.64 to the
option (delta); holding it for one year costs roughly \$6.41 of time value if nothing else
moves (theta); a 1-point rise in implied volatility (20% to 21%) adds about \$0.38 (vega); and
a 1-point rise in rates (5% to 6%) adds about \$0.53 (rho).

:::{note}
Gamma and vega are **identical for a call and put** with the same parameters, since both
derive from $\varphi(d_1)$ and do not depend on the option type. Delta, theta, and rho differ
in sign and magnitude between calls and puts.
:::

For a quick human-readable view of price and all Greeks together, use
{meth}`~quantflow.BlackScholes.summary`:

```python
print(BlackScholes(S=100, K=105, T=0.25, r=0.05, sigma=0.20, option_type="call").summary())
```

```text
Black-Scholes Call Option Summary
==================================================
Parameters:
  Spot Price (S):     $  100.00
  Strike Price (K):   $  105.00
  Time to Expiry (T):    0.250 years
  Risk-free Rate (r):    5.00%
  Dividend Yield (q):    0.00%
  Volatility (σ):       20.00%

Valuation:
  Option Price:       $    2.48

Greeks:
  Delta (Δ):            0.3772
  Gamma (Γ):            0.0380
  Theta (Θ):          $   -9.36 /year
  Vega (ν):           $    0.19 /1% vol
  Rho (ρ):            $    0.09 /1% rate
```

## Implied volatility

Implied volatility is the value of $\sigma$ that makes the model price equal an observed market
price. Because the Black-Scholes price is strictly increasing in $\sigma$, the inverse is well
defined. {meth}`~quantflow.BlackScholes.implied_volatility` first validates that the supplied
price lies within the option's no-arbitrage bounds, then solves with a robust bracketing
root-finder (Brent's method), which avoids the divergence that an unbracketed Newton iteration
can suffer in low-vega regimes.

A round-trip confirms correctness: price an option at a known $\sigma$, then recover that
$\sigma$ from the price. The `sigma` you pass when constructing the solver object is only a
starting context -- the solved volatility does not depend on it.

```python
# Price at a known sigma = 0.20
true_vol = 0.20
market_price = BlackScholes(100, 100, 1.0, 0.05, true_vol, "call").price()

# Recover it from the price (note the deliberately wrong seed sigma=0.50)
solver = BlackScholes(100, 100, 1.0, 0.05, 0.50, "call")
iv = solver.implied_volatility(market_price)

print(f"Market price: {market_price:.6f}")
print(f"Implied vol:  {iv:.6f}")
```

```text
Market price: 10.450584
Implied vol:  0.199999
```

The solver works directly from a quoted price as well:

```python
opt = BlackScholes(100, 100, 1.0, 0.05, 0.30, "call")
iv = opt.implied_volatility(12.50)
print(f"IV for a 12.50 quote: {iv:.6f}")
print(f"Reprice check:        {BlackScholes(100, 100, 1.0, 0.05, iv, 'call').price():.6f}")
```

```text
IV for a 12.50 quote: 0.254333
Reprice check:        12.500000
```

:::{warning}
A price outside the no-arbitrage bounds has no implied volatility, and the method raises a
`ValueError` rather than returning a meaningless number. For a call those bounds are
$[\,\max(S e^{-qT} - K e^{-rT},\,0),\; S e^{-qT}\,]$.

```python
opt = BlackScholes(100, 100, 1.0, 0.05, 0.20, "call")
opt.implied_volatility(0.50)
# ValueError: market_price 0.500000 violates the no-arbitrage bounds
#             [4.877058, 100.000000] for this option
```
:::

## Binomial trees (CRR)

{class}`~quantflow.BinomialTree` builds a Cox-Ross-Rubinstein lattice. With $\mathrm{dt} = T /
N$ for `steps` $= N$, the up/down factors and risk-neutral probability are

$$
u = e^{\sigma\sqrt{\mathrm{dt}}}, \qquad d = \frac{1}{u}, \qquad
p = \frac{e^{(r-q)\,\mathrm{dt}} - d}{u - d}.
$$

The tree is solved by backward induction, discounting each step by $e^{-r\,\mathrm{dt}}$. At
each node an **American** option takes $\max(\text{continuation},\ \text{intrinsic})$, whereas a
**European** option only ever takes the continuation value. The `exercise` argument selects the
style.

### European vs American

An American option is worth at least as much as the otherwise-identical European one, because
early exercise is an added right. For a put, that right has real value:

```python
from quantflow import BinomialTree

euro = BinomialTree(100, 100, 1.0, 0.05, 0.20, "put", "european", steps=500)
amer = BinomialTree(100, 100, 1.0, 0.05, 0.20, "put", "american", steps=500)

print(f"European put: {euro.price():.6f}")
print(f"American put: {amer.price():.6f}")
print(f"Early-exercise premium: {amer.early_exercise_premium():.6f}")
```

```text
European put: 5.569528
American put: 6.088810
Early-exercise premium: 0.519283
```

{meth}`~quantflow.BinomialTree.early_exercise_premium` returns the American price minus the
European price (always $\ge 0$) by repricing the same option under both exercise styles.

:::{tip}
**An American call on a non-dividend-paying stock is never exercised early**, so its price
equals the European call (and the Black-Scholes value). The early-exercise premium is exactly
zero:

```python
amer_call = BinomialTree(100, 100, 1.0, 0.05, 0.20, "call", "american", steps=500)
euro_call = BinomialTree(100, 100, 1.0, 0.05, 0.20, "call", "european", steps=500)
print(amer_call.price(), euro_call.price())          # 10.446585 10.446585
print(amer_call.early_exercise_premium())            # 0.0
```

A dividend yield changes this: with `q` large enough, early exercise of a call can become
optimal and the premium turns positive (e.g. `q=0.06` above gives a premium of about `0.18`).
:::

### Convergence to Black-Scholes

For European options the CRR price oscillates around and converges to the Black-Scholes value as
`steps` increases. The table below reprices a single call at a range of step counts against the
closed-form reference (`S=100, K=105, T=0.5, r=0.05, sigma=0.25`):

```python
from quantflow import BlackScholes, BinomialTree

reference = BlackScholes(100, 105, 0.5, 0.05, 0.25, "call").price()
print(f"Black-Scholes reference: {reference:.6f}\n")

for n in [10, 25, 50, 100, 250, 500, 1000]:
    price = BinomialTree(100, 105, 0.5, 0.05, 0.25, "call",
                         "european", steps=n).price()
    print(f"steps = {n:>5d}   price = {price:.6f}   error = {price - reference:+.6f}")
```

```text
Black-Scholes reference: 5.988490

steps =    10   price = 6.146968   error = +0.158477
steps =    25   price = 5.998673   error = +0.010183
steps =    50   price = 5.957422   error = -0.031068
steps =   100   price = 6.002719   error = +0.014229
steps =   250   price = 5.989283   error = +0.000793
steps =   500   price = 5.986909   error = -0.001581
steps =  1000   price = 5.989858   error = +0.001368
```

The error shrinks roughly like $O(1/N)$ but with a characteristic sawtooth: it does not decrease
monotonically, because whether a node lands exactly at the strike depends on the parity of $N$.
The default of `steps=500` is a good accuracy/cost trade-off for most uses.

:::{note}
The {class}`~quantflow.BinomialTree` Greeks are computed numerically: delta and gamma are read
from the early lattice layers, theta from the lattice time step, and vega and rho by a
central-difference bump-and-reprice (each scaled per 1% to match the
{class}`~quantflow.BlackScholes` conventions). For European options these agree with the
analytic Black-Scholes Greeks to within lattice resolution.
:::

A complete picture, including the early-exercise premium and all Greeks, is available from
{meth}`~quantflow.BinomialTree.summary`:

```python
print(BinomialTree(100, 100, 1.0, 0.05, 0.20, "put", "american", steps=500).summary())
```

```text
Binomial Tree American Put Option Summary
==================================================
Parameters:
  Spot Price (S):     $  100.00
  Strike Price (K):   $  100.00
  Time to Expiry (T):    1.000 years
  Risk-free Rate (r):    5.00%
  Dividend Yield (q):    0.00%
  Volatility (σ):       20.00%
  Steps:                   500

Valuation:
  Option Price:       $    6.09
  Early Ex. Premium:  $  0.5193

Greeks:
  Delta (Δ):          -0.4112
  Gamma (Γ):           0.0230
  Theta (Θ):          $   -2.24 /year
  Vega (ν):           $    0.37 /1% vol
  Rho (ρ):            $   -0.30 /1% rate
```

## See also

- {class}`~quantflow.BlackScholes` -- closed-form European pricing, Greeks, and implied volatility.
- {class}`~quantflow.BinomialTree` -- CRR lattice for European and American options.
- {class}`~quantflow.RiskMetrics` and {class}`~quantflow.Portfolio` -- portfolio-level risk and optimisation built on returns.
