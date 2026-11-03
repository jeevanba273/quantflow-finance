# Risk metrics

{class}`~quantflow.RiskMetrics` computes risk and performance statistics of
one series of periodic simple returns. The examples on this page use five
years of simulated daily returns with Student-t(5) shocks, so the results can
be compared with known values.

```python
import numpy as np
import pandas as pd

from quantflow import RiskMetrics

rng = np.random.default_rng(2024)
dates = pd.bdate_range("2020-01-01", periods=1250)
returns = pd.Series(0.0003 + 0.009 * rng.standard_t(5, len(dates)), index=dates)
rm = RiskMetrics(returns, periods_per_year=252)
print(f"{len(rm.returns)} returns")
```

```text
1250 returns
```

## Conventions

- `periods_per_year` (default 252) annualises: means times $m$, standard
  deviations times $\sqrt{m}$, and an annual risk-free rate $r_f$ becomes
  $r_f / m$ per period.
- Standard deviations are sample standard deviations (`ddof=1`).
- VaR and expected shortfall are **signed returns**: a loss is negative.
  `confidence_level` is the tail probability, so `0.05` is the 95% level.
- NaNs are dropped; infinite values and multi-column frames are rejected.

## Value at Risk and expected shortfall

```python
for alpha in (0.05, 0.01):
    print(f"{1 - alpha:.0%}: historical {rm.var_historical(alpha):+.4f}  "
          f"normal {rm.var_parametric(alpha):+.4f}  "
          f"Cornish-Fisher {rm.var_cornish_fisher(alpha):+.4f}  "
          f"ES {rm.expected_shortfall(alpha):+.4f}")
```

```text
95%: historical -0.0172  normal -0.0178  Cornish-Fisher -0.0167  ES -0.0245
99%: historical -0.0300  normal -0.0255  Cornish-Fisher -0.0328  ES -0.0371
```

- `var_historical` is the empirical quantile (linear interpolation).
- `var_parametric` is $\hat\mu + z_\alpha \hat\sigma$, assuming normal returns.
- `var_cornish_fisher` adjusts $z_\alpha$ for sample skewness and excess
  kurtosis.
- `expected_shortfall` is the mean of the returns at or below the historical
  VaR; `expected_shortfall_parametric` is the normal closed form
  $\hat\mu - \hat\sigma \varphi(z_\alpha)/\alpha$.

With these fat-tailed returns the normal VaR is too mild at 99%. The true 1%
quantile of the simulated distribution is known:

```python
from scipy import stats

print(f"true 1% quantile {0.0003 + 0.009 * stats.t.ppf(0.01, 5):+.4f}")
```

```text
true 1% quantile -0.0300
```

The {doc}`../validation` page checks the historical estimators against exact
normal and Student-t quantiles and tail means on 400,000-draw samples.

## Performance ratios

```python
print(f"volatility {rm.volatility():.4f}")
print(f"Sharpe     {rm.sharpe_ratio(0.02):.4f}")
print(f"Sortino    {rm.sortino_ratio(0.02):.4f}")
print(f"Calmar     {rm.calmar_ratio():.4f}")
print(f"Omega      {rm.omega_ratio():.4f}")
print(f"max drawdown {rm.max_drawdown():.4f}")
```

```text
volatility 0.1795
Sharpe     1.0665
Sortino    1.5998
Calmar     0.7222
Omega      1.2228
max drawdown -0.2986
```

The definitions are:

- Sharpe: $(\hat\mu - r_f/m)/\hat\sigma \cdot \sqrt{m}$.
- Sortino: $(\hat\mu - r_f/m)/\sigma_d \cdot \sqrt{m}$ with
  $\sigma_d = \sqrt{\tfrac1n \sum \min(r_i - \tau, 0)^2}$ and target $\tau$
  (default 0). The numerator uses $r_f$ and the downside deviation uses
  $\tau$; pass `target=rf / periods_per_year` to use one threshold for both.
- Calmar: compound annual growth rate over the whole sample divided by the
  absolute maximum drawdown.
- Omega: sum of gains above the threshold over the sum of losses below it.
- Maximum drawdown: the largest fall of the compounded wealth index from a
  running peak, where the starting value counts as a peak.

Sample ratios carry large estimation error. The process simulated above has
a true annual Sharpe ratio near 0.30, while five years of it gave the value
printed above. The standard error of an annualised Sharpe ratio from $n$
daily returns of an independent series is roughly $\sqrt{252/n}$ (Lo 2002):

```python
true_sharpe = (0.0003 * 252 - 0.02) / (0.009 * np.sqrt(5 / 3) * np.sqrt(252))
print(f"true Sharpe {true_sharpe:.2f}, standard error {np.sqrt(252 / 1250):.2f}")
```

```text
true Sharpe 0.30, standard error 0.45
```

## Benchmark-relative measures

`beta`, `alpha` (Jensen's, annualised), `capm` (beta, alpha and $R^2$) and
`information_ratio` take a benchmark series. Two date-indexed series are
matched on their common dates; plain arrays are matched by position from the
end.

```python
benchmark = pd.Series(0.0002 + 0.01 * rng.standard_normal(len(dates)), index=dates)
strategy = 0.7 * benchmark + 0.5 * returns
capm = RiskMetrics(strategy).capm(benchmark, risk_free_rate=0.02)
print({k: round(v, 4) for k, v in capm.items()})
print(f"information ratio {RiskMetrics(strategy).information_ratio(benchmark):.4f}")
```

```text
{'beta': 0.7056, 'alpha': 0.0993, 'r_squared': 0.6069}
information ratio 0.7555
```

## What it does not do

There is no time-varying volatility (GARCH, EWMA), no Monte Carlo VaR, no
multi-day scaling of VaR, and no backtesting of VaR exceptions. For the
Monte Carlo approach to VaR see the article listed in the README.
