# Getting Started

This guide installs **quantflow-finance** and walks through every part of the public API in a single, self-contained tour. All examples are deterministic (seeded synthetic data) so you can copy, run, and reproduce the exact numbers shown.

## Installation

quantflow-finance is published on PyPI. The distribution name is `quantflow-finance`, but the import name is `quantflow`:

```bash
pip install quantflow-finance
```

```python
import quantflow
print(quantflow.__version__)
# 0.2.1
```

:::{note}
quantflow-finance requires **Python 3.9 or newer** and builds on the scientific Python
stack (NumPy, SciPy, pandas, matplotlib). The market-data layer
({class}`~quantflow.MarketData`) uses [yfinance](https://pypi.org/project/yfinance/) for
live price downloads. All of these dependencies are installed automatically.
:::

:::{tip}
For a fully reproducible research environment, pin the version:
`pip install quantflow-finance==0.2.1`.
:::

## 5-minute tour

The public API exposes five classes:

| Class | Purpose |
| --- | --- |
| {class}`~quantflow.BlackScholes` | Closed-form European option pricing and Greeks |
| {class}`~quantflow.BinomialTree` | Cox-Ross-Rubinstein lattice for European **and** American options |
| {class}`~quantflow.RiskMetrics` | VaR, expected shortfall, and performance ratios on a return series |
| {class}`~quantflow.Portfolio` | Covariance, optimization, and the efficient frontier |
| {class}`~quantflow.MarketData` | Fetch prices and convert them to returns |

The sections below touch all five, end to end.

### 1. Price an option with Greeks

{class}`~quantflow.BlackScholes` prices a European option under the Black-Scholes-Merton model. Pass spot `S`, strike `K`, time to expiry `T` (in years), risk-free rate `r`, volatility `sigma`, and optionally a continuous dividend yield `q`.

```python
from quantflow import BlackScholes

opt = BlackScholes(S=100, K=105, T=0.5, r=0.04, sigma=0.20,
                   option_type='call', q=0.0)

print(opt.price())     # 4.37699932701166
print(opt.greeks())
```

```text
{'price': 4.37699932701166,
 'delta': 0.44714947064142396,
 'gamma': 0.027961576664115848,
 'theta': -7.205833242308398,
 'vega': 0.2796157666411585,
 'rho': 0.20168973868565365}
```

{meth}`~quantflow.BlackScholes.greeks` returns the full sensitivity set in one call, or you can request individual Greeks via {meth}`~quantflow.BlackScholes.delta`, {meth}`~quantflow.BlackScholes.gamma`, {meth}`~quantflow.BlackScholes.theta`, {meth}`~quantflow.BlackScholes.vega`, and {meth}`~quantflow.BlackScholes.rho`.

:::{note}
**Conventions.** `theta` is quoted **per year**, while `vega` and `rho` are scaled **per 1%** move in volatility / interest rate (i.e. divided by 100). So a vega of `0.28` means the option gains roughly \$0.28 for a one-percentage-point rise in implied volatility.
:::

A formatted overview is available from {meth}`~quantflow.BlackScholes.summary`:

```python
print(opt.summary())
```

```text
Black-Scholes Call Option Summary
==================================================
Parameters:
  Spot Price (S):     $  100.00
  Strike Price (K):   $  105.00
  Time to Expiry (T):    0.500 years
  Risk-free Rate (r):    4.00%
  Dividend Yield (q):    0.00%
  Volatility (σ):       20.00%

Valuation:
  Option Price:       $    4.38

Greeks:
  Delta (Δ):            0.4471
  Gamma (Γ):            0.0280
  Theta (Θ):          $   -7.21 /year
  Vega (ν):           $    0.28 /1% vol
  Rho (ρ):            $    0.20 /1% rate
```

Given a market price, recover the implied volatility with {meth}`~quantflow.BlackScholes.implied_volatility`:

```python
market_price = 4.37699932701166
iv = BlackScholes(S=100, K=105, T=0.5, r=0.04, sigma=0.30,
                  option_type='call').implied_volatility(market_price)
print(iv)   # 0.19999999300662794
```

The solver recovers the 20% volatility we priced with, to within the default tolerance of `1e-6`.

### 2. Price an American option with the binomial tree

{class}`~quantflow.BinomialTree` builds a Cox-Ross-Rubinstein lattice. Set `exercise='american'` to allow early exercise; for European options it converges to the closed-form Black-Scholes value as `steps` grows.

```python
from quantflow import BinomialTree, BlackScholes

american = BinomialTree(S=100, K=105, T=0.5, r=0.04, sigma=0.20,
                        option_type='put', exercise='american', steps=500)
european = BinomialTree(S=100, K=105, T=0.5, r=0.04, sigma=0.20,
                        option_type='put', exercise='european', steps=500)

print(american.price())                    # 7.6366753908692795
print(european.price())                    # 7.297551320382295
print(american.early_exercise_premium())   # 0.33912407048698423

# European lattice matches the analytic Black-Scholes price:
print(BlackScholes(S=100, K=105, T=0.5, r=0.04, sigma=0.20,
                   option_type='put').price())   # 7.297860024220967
```

The American put is worth more than its European counterpart; the difference is the **early-exercise premium** reported by {meth}`~quantflow.BinomialTree.early_exercise_premium`. The European lattice price (`7.2976`) agrees with the analytic Black-Scholes value (`7.2979`) to four decimal places, confirming convergence.

{class}`~quantflow.BinomialTree` also exposes Greeks and a {meth}`~quantflow.BinomialTree.summary` in the same style as {class}`~quantflow.BlackScholes`:

```python
print(american.summary())
```

```text
Binomial Tree American Put Option Summary
==================================================
Parameters:
  Spot Price (S):     $  100.00
  Strike Price (K):   $  105.00
  Time to Expiry (T):    0.500 years
  Risk-free Rate (r):    4.00%
  Dividend Yield (q):    0.00%
  Volatility (σ):       20.00%
  Steps:                   500

Valuation:
  Option Price:       $    7.64
  Early Ex. Premium:  $  0.3391

Greeks:
  Delta (Δ):           -0.5910
  Gamma (Γ):            0.0318
  Theta (Θ):          $   -3.70 /year
  Vega (ν):           $    0.27 /1% vol
  Rho (ρ):            $   -0.21 /1% rate
```

### 3. Compute risk metrics on a return series

{class}`~quantflow.RiskMetrics` accepts a pandas `Series`, `DataFrame`, or NumPy `ndarray` of **periodic returns** and computes risk and performance statistics. Annualization uses `periods_per_year` (252 trading days by default).

```python
import numpy as np
from quantflow import RiskMetrics

rng = np.random.default_rng(0)
returns = rng.normal(0.0007, 0.012, 252)   # one year of daily returns

rm = RiskMetrics(returns, periods_per_year=252)

print(rm.var_historical(0.05))       # -0.01713366971238655
print(rm.expected_shortfall(0.05))   # -0.02450628627077234
print(rm.sharpe_ratio(0.02))         # 0.7269357633684007
print(rm.sortino_ratio(0.02))        # 1.0758015752219874
print(rm.volatility())               # 0.19340285084602765 (annualized)
print(rm.max_drawdown())             # -0.134604566248705
```

VaR and expected shortfall are reported as (negative) return thresholds at the given confidence level `cl=0.05` — here, the worst expected loss on a typical bad day is about `-1.71%` historically, with an average tail loss of `-2.45%`.

{meth}`~quantflow.RiskMetrics.summary_stats` bundles the headline numbers into one dictionary:

```python
print(rm.summary_stats())
```

```text
{'count': 252,
 'periods_per_year': 252,
 'mean_return': 0.0006372676548308811,
 'volatility': 0.19340285084602765,
 'sharpe_ratio': 0.7269357633684007,
 'sortino_ratio': 1.0758015752219874,
 'calmar_ratio': 1.1329084750215086,
 'var_95': -0.01713366971238655,
 'var_99': -0.027437175309698275,
 'expected_shortfall_95': -0.02450628627077234,
 'max_drawdown': -0.134604566248705,
 'skewness': -0.020173822089832052,
 'kurtosis': -0.051172546326759605}
```

The class also offers parametric and Cornish-Fisher VaR, the Omega and Calmar ratios, and benchmark-relative measures (beta, alpha, CAPM, information ratio). See the {doc}`guides/risk-portfolio` guide for the full set.

### 4. Build and optimize a Portfolio

{class}`~quantflow.Portfolio` takes a `DataFrame` of asset returns (one column per asset) and provides covariance estimation, optimization, and the efficient frontier. With no `weights`, methods default to an equal-weight allocation.

```python
import numpy as np
import pandas as pd
from quantflow import Portfolio

rng = np.random.default_rng(3)
n = 252
asset_returns = pd.DataFrame({
    'AAA': rng.normal(0.0006, 0.013, n),
    'BBB': rng.normal(0.0005, 0.010, n),
    'CCC': rng.normal(0.0009, 0.018, n),
})

pf = Portfolio(asset_returns, periods_per_year=252)

print(pf.portfolio_return())       # 0.32487397298441123  (equal-weight, annualized)
print(pf.portfolio_volatility())   # 0.13028396627135563
print(pf.portfolio_sharpe())       # 2.3400728555455492
```

:::{warning}
These statistics come from a short synthetic sample, so the Sharpe ratios are unrealistically high. Treat them as illustrations of the API, not of attainable performance.
:::

Optimize for the maximum-Sharpe (tangency) portfolio with {meth}`~quantflow.Portfolio.max_sharpe`, or the global minimum-variance portfolio with {meth}`~quantflow.Portfolio.min_variance`. Both default to long-only weights (`allow_short=False`) and return a result dictionary:

```python
print(pf.max_sharpe(risk_free_rate=0.02))
```

```text
{'weights': array([0.13708568, 0.66237824, 0.20053608]),
 'weights_by_asset': {'AAA': 0.1370856789962887,
                      'BBB': 0.6623782389772281,
                      'CCC': 0.20053608202648324},
 'return': 0.36830096850142907,
 'volatility': 0.12231962552699845,
 'sharpe': 2.847465948336736,
 'success': True}
```

```python
print(pf.min_variance())
```

```text
{'weights': array([0.29061184, 0.54294422, 0.16644395]),
 'weights_by_asset': {'AAA': 0.2906118372265905,
                      'BBB': 0.5429442150789872,
                      'CCC': 0.16644394769442233},
 'return': 0.33536154371531196,
 'volatility': 0.11639201351188319,
 'sharpe': 2.7094775165403826,
 'success': True}
```

:::{note}
Several methods take a `risk_free_rate` argument (default `0.02`). The unified {meth}`~quantflow.Portfolio.optimize` entry point dispatches on `objective='max_sharpe' | 'min_variance' | 'target_return'`, and {meth}`~quantflow.Portfolio.efficient_frontier` returns a `DataFrame` of frontier points for plotting. You can also extract a {class}`~quantflow.RiskMetrics` object for any weighting via {meth}`~quantflow.Portfolio.risk_metrics`.
:::

### 5. Turn prices into returns with MarketData

{class}`~quantflow.MarketData` exposes static methods for fetching and transforming price data. {meth}`~quantflow.MarketData.fetch_stock_data` downloads live prices via yfinance, but the transform helpers work on any price series — here we use a synthetic one so the result is reproducible offline.

```python
import numpy as np
import pandas as pd
from quantflow import MarketData

rng = np.random.default_rng(123)
prices = pd.Series(100 * np.cumprod(1 + rng.normal(0.0005, 0.01, 6)), name='PRICE')

simple = MarketData.calculate_returns(prices, method='simple')
log = MarketData.calculate_returns(prices, method='log')

print(prices.round(4).to_list())
# [99.0609, 98.7461, 100.0672, 100.3114, 101.2846, 101.9198]

print(simple.round(6).to_list())
# [-0.003178, 0.013379, 0.00244, 0.009702, 0.006271]

print(log.round(6).to_list())
# [-0.003183, 0.013291, 0.002437, 0.009656, 0.006251]
```

`calculate_returns` drops the first (undefined) observation, so an input of 6 prices yields 5 returns. Simple returns ($P_t/P_{t-1} - 1$) and log returns ($\ln(P_t/P_{t-1})$) nearly coincide for small moves, diverging only as returns grow. The output is a pandas `Series`, ready to feed straight into {class}`~quantflow.RiskMetrics` or {class}`~quantflow.Portfolio`.

:::{tip}
To pull real data, swap the synthetic series for `MarketData.fetch_stock_data('AAPL', period='1y')`. Network access is required, and results vary by date — see {doc}`guides/market-data` for caching and multi-ticker patterns.
:::

## Next steps

You have now priced options, measured risk, and optimized a portfolio. To go deeper:

- {doc}`guides/options` — vanilla and American option pricing, the full Greek suite, and implied-volatility solving with {class}`~quantflow.BlackScholes` and {class}`~quantflow.BinomialTree`.
- {doc}`guides/risk-portfolio` — VaR families, performance ratios, benchmark-relative metrics, covariance estimation, and the efficient frontier.
- {doc}`guides/market-data` — fetching single and multiple tickers, intervals and periods, and converting prices to returns.
- {doc}`theory` — the mathematical background: Black-Scholes-Merton, the CRR lattice, and the risk and optimization formulas.
- {doc}`api/index` — the complete API reference for every class and method.
