# Getting started

## Install

QuantFlow Finance needs Python 3.10 or newer. The distribution is called
`quantflow-finance`; the import name is `quantflow_finance`.

```bash
pip install quantflow-finance
```

The core depends only on NumPy, SciPy and pandas. Two optional extras exist:

```bash
pip install "quantflow-finance[data]"        # yfinance, for MarketData.fetch_*
pip install "quantflow-finance[validation]"  # QuantLib and vollib, for the cross-checks
```

### Import name

Up to 0.3.1 the import name was `quantflow`. Version 0.3.2 added
`quantflow_finance` and kept `quantflow` with a `FutureWarning`; 0.4.0
removed `quantflow`, because the PyPI project
[quantflow](https://github.com/quantmind/quantflow), which is not related to
QuantFlow Finance, installs a top-level module with the same name. To upgrade
code written for 0.3.x, replace `quantflow` with `quantflow_finance` in your
import lines:

| Before | After |
|---|---|
| `from quantflow import BlackScholes` | `from quantflow_finance import BlackScholes` |
| `from quantflow.risk.metrics import RiskMetrics` | `from quantflow_finance.risk.metrics import RiskMetrics` |
| `import quantflow` | `import quantflow_finance`, and `quantflow.X` becomes `quantflow_finance.X` |

Code that must keep the old name can pin `quantflow-finance<0.4`.

## A first example

Price the call and put from Hull's textbook example (spot 42, strike 40,
six months, 10% rate, 20% volatility), check put-call parity and recover the
volatility from the call price:

```python
import math

from quantflow_finance import BlackScholes

call = BlackScholes(S=42, K=40, T=0.5, r=0.10, sigma=0.20, option_type="call")
put = BlackScholes(S=42, K=40, T=0.5, r=0.10, sigma=0.20, option_type="put")

print(f"call {call.price():.4f}")
print(f"put  {put.price():.4f}")
print(f"parity gap below 1e-12: {abs(call.price() - put.price() - (42 - 40 * math.exp(-0.05))) < 1e-12}")
print(f"implied vol {call.implied_volatility(call.price()):.10f}")
```

```text
call 4.7594
put  0.8086
parity gap below 1e-12: True
implied vol 0.2000000000
```

Hull quotes 4.76 and 0.81 for these two options.

## What is in the package

| Object | What it does |
| --- | --- |
| {class}`~quantflow_finance.BlackScholes` | Closed-form Black-Scholes-Merton price, Greeks and implied volatility for one European option |
| {func}`~quantflow_finance.black_scholes_price`, {func}`~quantflow_finance.black_scholes_greeks` | The same formulas on NumPy arrays |
| {class}`~quantflow_finance.BinomialTree` | Cox-Ross-Rubinstein tree for European and American options |
| {class}`~quantflow_finance.MonteCarlo` | Monte Carlo price of a European option with its standard error |
| {class}`~quantflow_finance.RiskMetrics` | VaR, expected shortfall, drawdown and performance ratios of one return series |
| {class}`~quantflow_finance.Portfolio` | Covariance, minimum-variance, maximum-Sharpe and target-return portfolios |
| {class}`~quantflow_finance.MarketData` | Prices from Yahoo Finance (optional extra) and price-to-return conversion |

## Conventions

These apply everywhere unless a page says otherwise.

- Rates (`r`, `q`) are continuously compounded and annual; volatility is
  annualised.
- `T` is a year fraction you supply. No day count or calendar is applied;
  `days / 365` is the usual choice.
- `theta` is per year, `vega` and `rho` are per 1 percentage point.
- Return statistics are annualised with `periods_per_year`, 252 by default.
- VaR and expected shortfall are signed returns (a loss is negative), and
  `confidence_level=0.05` means the 95% level.

The {doc}`user guide <guide/index>` has one page per area, {doc}`methods`
gives the models and their limits, and {doc}`validation` shows how the
results were checked.
