# Risk & Portfolio Analytics

This guide covers the two analytics classes: {class}`~quantflow.RiskMetrics` for
single-series risk and performance statistics, and {class}`~quantflow.Portfolio` for
multi-asset analysis and mean-variance optimization.

All examples use seeded synthetic data, so the numbers are reproducible.

## RiskMetrics

{class}`~quantflow.RiskMetrics` takes a series of **periodic returns** (a pandas
`Series`, `DataFrame`, or NumPy `ndarray`) and computes risk and performance measures.
Annualization is controlled by `periods_per_year` (252 trading days by default; use 12
for monthly data, 52 for weekly).

```python
import numpy as np
import pandas as pd
from quantflow import RiskMetrics

rng = np.random.default_rng(0)
benchmark = pd.Series(rng.normal(0.0004, 0.010, 252), name="benchmark")
returns = pd.Series(0.9 * benchmark.to_numpy() + rng.normal(0.0003, 0.006, 252),
                    name="strategy")

rm = RiskMetrics(returns, periods_per_year=252)
```

### Value at Risk and Expected Shortfall

VaR answers "how bad is a bad period?" at a confidence level (`cl=0.05` is the 5% / 95%
VaR). Three estimators are available, plus their tail-average counterparts (Expected
Shortfall / Conditional VaR):

```python
rm.var_historical(0.05)                 # -0.015827  empirical 5th percentile
rm.var_parametric(0.05)                 # -0.017539  Gaussian (mean + z * std)
rm.var_cornish_fisher(0.05)             # -0.017731  skew/kurtosis-adjusted
rm.expected_shortfall(0.05)             # -0.021712  mean loss beyond historical VaR
rm.expected_shortfall_parametric(0.05)  # -0.022073  Gaussian closed form
```

All are reported as (negative) return thresholds. Expected Shortfall is always at least
as severe as the corresponding VaR.

:::{tip}
Use {meth}`~quantflow.RiskMetrics.var_cornish_fisher` when returns are visibly skewed or
fat-tailed — it adjusts the Gaussian quantile using the sample skewness and excess
kurtosis, so it does not understate tail risk the way the plain normal VaR can.
:::

### Performance ratios

```python
rm.sharpe_ratio(0.02)     # 0.334793  excess return per unit of total volatility
rm.sortino_ratio(0.02)    # 0.483182  excess return per unit of downside deviation
rm.calmar_ratio()         # 0.599804  annualized return divided by |max drawdown|
rm.omega_ratio()          # 1.072996  probability-weighted gains vs losses (>1 is good)
rm.volatility()           # 0.172248  annualized standard deviation
rm.max_drawdown()         # -0.108200 worst peak-to-trough decline
```

The `risk_free_rate` arguments are **annual** rates; they are de-annualized internally
using `periods_per_year`. {meth}`~quantflow.RiskMetrics.sortino_ratio` penalizes only
downside deviation (returns below `target`, default 0), so it rewards strategies whose
volatility is mostly on the upside.

### Benchmark-relative metrics (CAPM)

Pass a benchmark return series to get the market-model regression. Because the strategy
above was constructed as `0.9 * benchmark + noise`, the recovered beta is close to 0.9:

```python
rm.beta(benchmark)        # 0.883622
rm.alpha(benchmark)       # -0.002089  annualized Jensen's alpha
rm.capm(benchmark)        # {'beta': 0.883622, 'alpha': -0.002089, 'r_squared': 0.683578}
rm.information_ratio(benchmark)
```

{meth}`~quantflow.RiskMetrics.capm` returns beta, annualized alpha, and the regression
$R^2$ in one call.

:::{note}
A (near-)constant benchmark has no variance, so beta is undefined;
{meth}`~quantflow.RiskMetrics.beta` raises a `ValueError` in that case rather than
returning a meaningless number.
:::

### One-shot summary

{meth}`~quantflow.RiskMetrics.summary_stats` returns the headline figures as a dict:

```python
rm.summary_stats()
```

```text
{'count': 252, 'periods_per_year': 252, 'mean_return': 0.000308,
 'volatility': 0.172248, 'sharpe_ratio': 0.334793, 'sortino_ratio': 0.483182,
 'calmar_ratio': 0.599804, 'var_95': -0.015827, 'var_99': -0.025124,
 'expected_shortfall_95': -0.021712, 'max_drawdown': -0.1082,
 'skewness': -0.059582, 'kurtosis': -0.040884}
```

## Portfolio

{class}`~quantflow.Portfolio` works on a returns **matrix** (observations x assets) and
adds covariance analysis and mean-variance optimization. It needs at least two assets.

```python
import numpy as np
import pandas as pd
from quantflow import Portfolio

rng = np.random.default_rng(7)
data = pd.DataFrame({
    "EQ":  rng.normal(0.0006, 0.011, 400),   # equities
    "BD":  rng.normal(0.0002, 0.004, 400),   # bonds (low volatility)
    "GLD": rng.normal(0.0004, 0.009, 400),   # gold
    "RE":  rng.normal(0.0005, 0.013, 400),   # real estate
}, index=pd.bdate_range("2022-01-03", periods=400))

port = Portfolio(data)
```

### Covariance and correlation

```python
port.correlation_matrix().round(3)
```

```text
        EQ     BD    GLD     RE
EQ   1.000  0.020  0.046  0.030
BD   0.020  1.000  0.106 -0.040
GLD  0.046  0.106  1.000 -0.110
RE   0.030 -0.040 -0.110  1.000
```

{meth}`~quantflow.Portfolio.covariance_matrix` returns the (optionally annualized)
covariance; {meth}`~quantflow.Portfolio.portfolio_return`,
{meth}`~quantflow.Portfolio.portfolio_volatility`, and
{meth}`~quantflow.Portfolio.portfolio_sharpe` evaluate any weight vector:

```python
import numpy as np
equal_weight = np.repeat(0.25, 4)
port.portfolio_return(equal_weight)      # 0.0894 (annualized)
port.portfolio_volatility(equal_weight)  # 0.0775
port.portfolio_sharpe(equal_weight)      # 0.8948
```

### Mean-variance optimization

Three optimizers are built on `scipy.optimize` (SLSQP), each returning a result dict with
`weights`, `weights_by_asset`, `return`, `volatility`, `sharpe`, and `success`:

```python
mv = port.min_variance()    # lowest-variance fully-invested portfolio
ms = port.max_sharpe()      # tangency portfolio (highest Sharpe)
```

```text
min_variance: return=0.0708  volatility=0.0536  sharpe=0.948
  weights: {'EQ': 0.091, 'BD': 0.728, 'GLD': 0.099, 'RE': 0.082}

max_sharpe:  return=0.1198  volatility=0.0751  sharpe=1.329
  weights: {'EQ': 0.009, 'BD': 0.676, 'GLD': 0.005, 'RE': 0.311}
```

The minimum-variance portfolio correctly tilts toward the low-volatility bond (`BD`),
and its volatility (5.36%) is below the equal-weight portfolio's (7.75%). The
maximum-Sharpe portfolio lifts the Sharpe ratio from 0.89 to 1.33. By default weights are
long-only and sum to 1; pass `allow_short=True` to permit negative weights.

:::{note}
The minimum-variance solution matches the analytical closed form
$w = \Sigma^{-1}\mathbf{1} / (\mathbf{1}^\top \Sigma^{-1} \mathbf{1})$ at any return
frequency. (0.2.0 had a solver-scaling bug here that returned equal weights on daily
data; it is fixed in 0.2.1.)
:::

### The efficient frontier

{meth}`~quantflow.Portfolio.efficient_frontier` sweeps target returns from the
minimum-variance point to the highest-return asset, returning a `DataFrame` of
risk/return/Sharpe and the weights at each point:

```python
port.efficient_frontier(n_points=6)[["return", "volatility", "sharpe"]].round(4)
```

```text
 return  volatility  sharpe
 0.0708      0.0536  0.9480
 0.1060      0.0656  1.3116
 0.1412      0.0931  1.3022
 0.1764      0.1282  1.2205
 0.2116      0.1663  1.1522
 0.2469      0.2059  1.1020
```

Return rises monotonically and volatility is minimized at the bottom of the frontier; the
Sharpe ratio peaks near the tangency portfolio. {meth}`~quantflow.Portfolio.optimize`
dispatches to any objective, including a return target:

```python
port.optimize(objective="target_return", target_return=0.12)
```

### Bridge to RiskMetrics

{meth}`~quantflow.Portfolio.risk_metrics` returns a {class}`~quantflow.RiskMetrics` object
for the weighted portfolio series, so every metric above is available on an optimized
portfolio:

```python
rm = port.risk_metrics(ms["weights"])
rm.sortino_ratio()    # 2.1121
```

This makes it easy to run the full risk panel (VaR, drawdown, Sortino, ...) on whatever
allocation the optimizer produces.
```
