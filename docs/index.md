# QuantFlow Finance

```{toctree}
:hidden:
:maxdepth: 2

getting-started
guides/options
guides/risk-portfolio
guides/market-data
theory
api/index
changelog
contributing
```

**Professional-grade quantitative finance tools for Python.**

QuantFlow Finance provides validated, industry-standard implementations of options
pricing, risk analytics, and portfolio optimization, built on the scientific Python
stack (NumPy, SciPy, pandas). Every model is checked against analytical references and
known benchmarks.

```{code-block} bash
pip install quantflow-finance
```

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`rocket` Getting Started
:link: getting-started
:link-type: doc

Install the package and tour all five core classes in five minutes.
:::

:::{grid-item-card} {octicon}`graph` Options Pricing
:link: guides/options
:link-type: doc

Black-Scholes-Merton with dividends, the full Greeks suite, implied volatility,
and a binomial tree for American options.
:::

:::{grid-item-card} {octicon}`shield-check` Risk & Portfolio
:link: guides/risk-portfolio
:link-type: doc

VaR, Expected Shortfall, performance ratios, CAPM, and mean-variance optimization.
:::

:::{grid-item-card} {octicon}`database` Market Data
:link: guides/market-data
:link-type: doc

Fetch prices from Yahoo Finance and compute simple or log returns.
:::

:::{grid-item-card} {octicon}`book` Mathematical Background
:link: theory
:link-type: doc

The formulas behind every model, with references.
:::

:::{grid-item-card} {octicon}`code` API Reference
:link: api/index
:link-type: doc

Complete reference for every public class and method.
:::

::::

## Core capabilities

- **Options** — {class}`~quantflow.BlackScholes` (Black-Scholes-Merton with a continuous
  dividend yield, all five Greeks, and implied volatility) and {class}`~quantflow.BinomialTree`
  (Cox-Ross-Rubinstein, European and American exercise).
- **Risk analytics** — {class}`~quantflow.RiskMetrics`: historical / parametric /
  Cornish-Fisher VaR, Expected Shortfall, Sharpe, Sortino, Calmar, Omega, maximum
  drawdown, information ratio, and CAPM beta/alpha.
- **Portfolio optimization** — {class}`~quantflow.Portfolio`: covariance/correlation
  analysis and mean-variance optimization (minimum-variance, maximum-Sharpe, and the
  efficient frontier).
- **Market data** — {class}`~quantflow.MarketData`: Yahoo Finance integration and return
  calculation.

## Why QuantFlow

This project bridges academic financial theory and practical implementation with a clean,
typed, well-documented API. It is designed for quantitative analysts, portfolio managers,
researchers, and graduate students in financial engineering.
```
