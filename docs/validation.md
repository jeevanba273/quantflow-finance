# Validation

This page compares quantflow with values that do not come from its own code:
published worked examples, exact identities, closed-form solutions, known
distributions and two other libraries. The same comparisons are assertions in
the test suite (`tests/test_reference_*.py`); the tables below are printed by
the code shown, which the test suite also runs.

## Published worked examples

Hull, *Options, Futures, and Other Derivatives*: the Black-Scholes-Merton
example (S=42, K=40, r=10%, sigma=20%, T=0.5) and the American put tree
(S=K=50, r=10%, sigma=40%, T=5 months). Haug, *The Complete Guide to Option
Pricing Formulas* (2nd ed.), chapters 1 and 2; Haug's cost of carry $b$ is
$r - q$ here.

```python
from quantflow import BinomialTree, BlackScholes

rows = [
    ("Hull BSM call", BlackScholes(42, 40, 0.5, 0.10, 0.20).price(), 4.76, 2),
    ("Hull BSM put", BlackScholes(42, 40, 0.5, 0.10, 0.20, "put").price(), 0.81, 2),
    ("Haug BS call", BlackScholes(60, 65, 0.25, 0.08, 0.30).price(), 2.1334, 4),
    ("Haug Merton put", BlackScholes(100, 95, 0.5, 0.10, 0.20, "put", 0.05).price(), 2.4648, 4),
    ("Haug Black-76 call", BlackScholes(19, 19, 0.75, 0.10, 0.28, "call", 0.10).price(), 1.7011, 4),
    ("Haug Garman-Kohlhagen", BlackScholes(1.56, 1.60, 0.5, 0.06, 0.12, "call", 0.08).price(), 0.0291, 4),
    ("Haug call delta", BlackScholes(105, 100, 0.5, 0.10, 0.36, "call", 0.10).delta(), 0.5946, 4),
    ("Haug put delta", BlackScholes(105, 100, 0.5, 0.10, 0.36, "put", 0.10).delta(), -0.3566, 4),
    ("Haug gamma", BlackScholes(55, 60, 0.75, 0.10, 0.30).gamma(), 0.0278, 4),
    ("Haug put theta", BlackScholes(430, 405, 0.0833, 0.07, 0.20, "put", 0.05).theta(), -31.1924, 4),
    ("Haug call rho x100", BlackScholes(72, 75, 1.0, 0.09, 0.19).rho() * 100, 38.7325, 4),
]
for steps, published in [(5, 4.49), (30, 4.263), (50, 4.272), (100, 4.278), (500, 4.283)]:
    tree = BinomialTree(50, 50, 5 / 12, 0.10, 0.40, "put", "american", steps=steps)
    rows.append((f"Hull American put, {steps} steps", tree.price(), published, 3 if steps > 5 else 2))

print(f"{'case':<30} {'published':>10} {'quantflow':>12}  match")
for label, value, published, digits in rows:
    match = round(value, digits) == published
    print(f"{label:<30} {published:>10} {value:>12.6f}  {match}")
```

```text
case                            published    quantflow  match
Hull BSM call                        4.76     4.759422  True
Hull BSM put                         0.81     0.808599  True
Haug BS call                       2.1334     2.133368  True
Haug Merton put                    2.4648     2.464788  True
Haug Black-76 call                 1.7011     1.701051  True
Haug Garman-Kohlhagen              0.0291     0.029099  True
Haug call delta                    0.5946     0.594629  True
Haug put delta                    -0.3566    -0.356601  True
Haug gamma                         0.0278     0.027821  True
Haug put theta                   -31.1924   -31.192367  True
Haug call rho x100                38.7325    38.732505  True
Hull American put, 5 steps           4.49     4.488459  True
Hull American put, 30 steps         4.263     4.263427  True
Hull American put, 50 steps         4.272     4.272021  True
Hull American put, 100 steps        4.278     4.278059  True
Hull American put, 500 steps        4.283     4.283021  True
```

"match" means quantflow's value rounded to the published number of decimals
equals the published value.

## Identities and derivatives

Put-call parity, $C - P = S e^{-qT} - K e^{-rT}$, on 648 parameter sets
(spot 50-180, strike 60-150, expiry one day to five years, rates -1% to 5%,
volatility 5-120%, dividend yield 0-4%), and every Greek against a central
finite difference of the price on 72 parameter sets:

```python
import itertools
import math

import numpy as np

from quantflow import black_scholes_greeks, black_scholes_price


def bound(x):
    """Smallest power of ten above x (3.6e-14 -> 1e-13): rounding-level
    errors vary in their last digits between platforms and versions."""
    return f"1e{math.floor(math.log10(x)) + 1}"


grid = np.array(list(itertools.product([50.0, 100.0, 180.0], [60.0, 100.0, 150.0],
                [1 / 365, 0.25, 1.0, 5.0], [-0.01, 0.0, 0.05], [0.05, 0.3, 1.2], [0.0, 0.04])))
S, K, T, r, v, q = grid.T
gap = (black_scholes_price(S, K, T, r, v, "call", q) - black_scholes_price(S, K, T, r, v, "put", q)
       - (S * np.exp(-q * T) - K * np.exp(-r * T)))
print(f"put-call parity: {len(grid)} cases, max |gap| below {bound(np.abs(gap).max())}")

cases = np.array(list(itertools.product([80.0, 100.0, 120.0], [0.1, 1.0, 3.0], [0.15, 0.45], [0.0, 0.03])))
worst = {}
for kind in ("call", "put"):
    S, T, v, q = cases.T
    K, r, e = 100.0, 0.04, 1e-5
    g = black_scholes_greeks(S, K, T, r, v, kind, q)
    h = 1e-4 * S
    P = lambda **kw: black_scholes_price(**{**dict(S=S, K=K, T=T, r=r, sigma=v, option_type=kind, q=q), **kw})
    fd = {"delta": (P(S=S + h) - P(S=S - h)) / (2 * h),
          "gamma": (P(S=S + h) - 2 * g["price"] + P(S=S - h)) / h**2,
          "theta": -(P(T=T + e) - P(T=T - e)) / (2 * e),
          "vega": (P(sigma=v + e) - P(sigma=v - e)) / (2 * e) / 100,
          "rho": (P(r=r + e) - P(r=r - e)) / (2 * e) / 100}
    for name, approx in fd.items():
        diff = np.abs(g[name] - approx)
        big = np.abs(approx) >= 1e-3
        a, r_ = worst.get(name, (0.0, 0.0))
        worst[name] = (max(a, diff.max()), max(r_, (diff[big] / np.abs(approx[big])).max()))
for name, (a, rel) in worst.items():
    print(f"{name:<5} vs central difference: {2 * len(cases)} cases, "
          f"max abs diff below {bound(a)}, max rel diff below {bound(rel)} where |value| >= 1e-3")
```

```text
put-call parity: 648 cases, max |gap| below 1e-13
delta vs central difference: 72 cases, max abs diff below 1e-7, max rel diff below 1e-6 where |value| >= 1e-3
gamma vs central difference: 72 cases, max abs diff below 1e-7, max rel diff below 1e-6 where |value| >= 1e-3
theta vs central difference: 72 cases, max abs diff below 1e-7, max rel diff below 1e-7 where |value| >= 1e-3
vega  vs central difference: 72 cases, max abs diff below 1e-9, max rel diff below 1e-8 where |value| >= 1e-3
rho   vs central difference: 72 cases, max abs diff below 1e-8, max rel diff below 1e-8 where |value| >= 1e-3
```

The differences are the truncation and rounding error of the finite
differences themselves (largest for the second difference in gamma).

## QuantLib and vollib

Produced by `python benchmarks/run_benchmarks.py` (see {doc}`benchmarks` for
the machine and versions) on 6,048 European options: spot 80-120, strike
70-130, expiry 7-730 days, rate 0-6%, dividend yield 0-2%, volatility
10-100%, calls and puts, with $T$ = days / 365 in every library. Relative
errors use reference values above $10^{-6}$; far out-of-the-money prices
below that are differences of nearly equal small numbers in every library.

| Quantity | Reference | Max abs difference | Max rel difference |
|---|---|---|---|
| price | QuantLib AnalyticEuropeanEngine | 1.5e-13 | 2.9e-09 |
| delta | QuantLib AnalyticEuropeanEngine | 8.8e-15 | 4.4e-11 |
| gamma | QuantLib AnalyticEuropeanEngine | 6.1e-16 | 4.1e-13 |
| theta | QuantLib AnalyticEuropeanEngine | 1.3e-12 | 2.8e-09 |
| vega | QuantLib AnalyticEuropeanEngine | 1.2e-15 | 3.3e-13 |
| rho | QuantLib AnalyticEuropeanEngine | 2.1e-15 | 1.9e-11 |
| price | vollib black_scholes_merton | 4.3e-14 | 3.6e-13 |

Implied volatility recovered from exact model prices for the 5,186 of those
options with vega of at least $10^{-3}$ (the others have too little time
value for any solver):

| Solver | Max abs vol error | Median abs vol error | Failures |
|---|---|---|---|
| quantflow (Brent, tol 1e-10) | 3.4e-11 | 7.2e-14 | 0 |
| vollib (Let's Be Rational) | 1.7e-11 | 3.3e-16 | 0 |
| QuantLib impliedVolatility (accuracy 1e-12) | 1.9e-11 | 1.6e-15 | 0 |

The optional tests in `tests/test_reference_libraries.py` assert agreement
with QuantLib to $10^{-10}$ relative for European prices and Greeks, with
QuantLib's CRR tree to $0.05/N$ for American options, with QuantLib's finite
difference engine to $3 \times 10^{-3}$ for 2000-step American trees, and with
vollib's implied volatilities to $10^{-9}$.

## Trees

The European tree converges to the closed form at order $1/N$. The test
suite requires $|\text{error}| \cdot N < 5$ for $N$ from 50 to 3200 on 48
contracts, and an error below $2 \times 10^{-3}$ at 3200 steps. An American
put is never below the European put or its intrinsic value, and an American
call without dividends equals the European call to $10^{-12}$.

## Monte Carlo

The closed-form price must lie inside the 99% confidence interval for four
contracts and each of the four estimator settings (16 seeded runs), the
standard error must halve when the paths quadruple, and every variance
reduction method must lower the standard error at a fixed number of paths.

```python
from quantflow import BlackScholes, MonteCarlo

contracts = [(100, 100, 1.0, 0.05, 0.2, "call", 0.0), (100, 120, 0.5, 0.03, 0.3, "call", 0.02),
             (100, 90, 0.25, 0.05, 0.25, "put", 0.0), (100, 110, 2.0, 0.02, 0.4, "put", 0.01)]
inside, runs, worst = 0, 0, 0.0
for S, K, T, r, sigma, kind, q in contracts:
    exact = BlackScholes(S, K, T, r, sigma, kind, q).price()
    for anti, cv in [(False, False), (True, False), (False, True), (True, True)]:
        res = MonteCarlo(S, K, T, r, sigma, kind, q=q, n_paths=100_000, antithetic=anti,
                         control_variate=cv, seed=11).simulate(confidence=0.99)
        inside += res.ci_low <= exact <= res.ci_high
        runs += 1
        worst = max(worst, abs(res.price - exact) / res.std_error)
print(f"{inside} of {runs} 99% intervals contain the closed form; "
      f"largest |error| / std error {worst:.2f}")
```

```text
16 of 16 99% intervals contain the closed form; largest |error| / std error 1.18
```

## Risk metrics

Historical VaR and expected shortfall on 400,000 draws from a normal and a
Student-t(5) distribution, against the exact quantile and tail mean (computed
with SciPy). The tests allow five asymptotic standard errors.

```python
import numpy as np
from scipy import stats

from quantflow import RiskMetrics

n = 400_000
samples = {
    "normal": (np.random.default_rng(1).normal(0.0004, 0.012, n),
               stats.norm(0.0004, 0.012)),
    "t(5)": (0.0003 + 0.009 * np.random.default_rng(2).standard_t(5, n),
             stats.t(5, 0.0003, 0.009)),
}
print(f"{'sample':<7} {'alpha':>5} {'VaR':>9} {'exact':>9} {'ES':>9} {'exact':>9}")
for name, (x, dist) in samples.items():
    rm = RiskMetrics(x)
    for alpha in (0.05, 0.01):
        q = dist.ppf(alpha)
        es = dist.expect(lambda v: v, ub=q, conditional=True)
        print(f"{name:<7} {alpha:>5} {rm.var_historical(alpha):9.5f} {q:9.5f} "
              f"{rm.expected_shortfall(alpha):9.5f} {es:9.5f}")
```

```text
sample  alpha       VaR     exact        ES     exact
normal   0.05  -0.01933  -0.01934  -0.02441  -0.02435
normal   0.01  -0.02772  -0.02752  -0.03175  -0.03158
t(5)     0.05  -0.01778  -0.01784  -0.02566  -0.02571
t(5)     0.01  -0.02999  -0.02998  -0.03962  -0.03977
```

The parametric VaR and ES match the normal closed forms to $10^{-12}$ and
$10^{-8}$ relative (the latter limited by SciPy's numerical integration), and
Sharpe, Sortino, Calmar, Omega, beta, alpha and the information ratio are
recomputed from their written definitions in `tests/test_reference_risk.py`.

## Portfolio optimisation

With short sales allowed, the minimum-variance and target-return weights
from SLSQP are compared with the closed forms of Merton (1972). (The
tangency portfolio with short sales is computed from its closed form, so
comparing it with that formula would test nothing.) Long-only problems have
no closed form; there the optimiser must do at least as well as the best of
200,000 random long-only portfolios. Simulated returns of four assets:

```python
import numpy as np
import pandas as pd

from quantflow import Portfolio

rng = np.random.default_rng(0)
vols = np.linspace(0.008, 0.02, 4)
cov = (0.3 + 0.7 * np.eye(4)) * np.outer(vols, vols)
draws = np.linspace(0.0002, 0.0009, 4) + rng.standard_normal((1500, 4)) @ np.linalg.cholesky(cov).T
frame = pd.DataFrame(draws)
mu, S = frame.mean().to_numpy() * 252, frame.cov().to_numpy() * 252
one = np.ones(4)
inv_one, inv_mu = np.linalg.solve(S, one), np.linalg.solve(S, mu)
A, B, C = one @ inv_mu, mu @ inv_mu, one @ inv_one
p = Portfolio(frame)

w_target = ((B - A * 0.15) * inv_one + (C * 0.15 - A) * inv_mu) / (B * C - A * A)
for label, got, exact in [
    ("global minimum variance", p.min_variance(allow_short=True)["weights"], inv_one / C),
    ("target return 15%", p.optimize("target_return", 0.15, allow_short=True)["weights"], w_target),
]:
    err = np.abs(got - exact).max()
    print(f"{label:<24} max |w - w_exact| below {bound(err)}  |sum - 1| below 1e-12: {abs(got.sum() - 1) < 1e-12}")

W = np.random.default_rng(1).dirichlet(np.ones(4), 200_000)
var = np.einsum("ij,jk,ik->i", W, S, W)
best_sharpe = ((W @ mu - 0.02) / np.sqrt(var)).max()
print(f"long-only max Sharpe     SLSQP {p.max_sharpe(0.02)['sharpe']:.6f}  best random {best_sharpe:.6f}")
print(f"long-only min volatility SLSQP {p.min_variance()['volatility']:.6f}  best random {np.sqrt(var.min()):.6f}")
```

```text
global minimum variance  max |w - w_exact| below 1e-6  |sum - 1| below 1e-12: True
target return 15%        max |w - w_exact| below 1e-6  |sum - 1| below 1e-12: True
long-only max Sharpe     SLSQP 0.820501  best random 0.820434
long-only min volatility SLSQP 0.115378  best random 0.115408
```

The tests also check the two-asset minimum-variance weight
$(\sigma_2^2 - \sigma_{12}) / (\sigma_1^2 + \sigma_2^2 - 2\sigma_{12})$, the
long-only bounds, the frontier variance $(C\mu^2 - 2A\mu + B)/(BC - A^2)$ at
six points, and that `max_sharpe` with short sales raises an error when the
tangency portfolio does not exist.
