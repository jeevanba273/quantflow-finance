<div align="center">

# QUANTFLOW FINANCE

<em>Option pricing, risk metrics and portfolio optimisation in Python.</em>

<!-- BADGES -->
<img src="https://img.shields.io/github/license/jeevanba273/quantflow-finance?style=flat&logo=opensourceinitiative&logoColor=white&color=0080ff&cache=bust&v=1" alt="license">
<img src="https://img.shields.io/github/last-commit/jeevanba273/quantflow-finance?style=flat&logo=git&logoColor=white&color=0080ff" alt="last-commit">
<img src="https://img.shields.io/github/languages/top/jeevanba273/quantflow-finance?style=flat&color=0080ff" alt="repo-top-language">
<img src="https://img.shields.io/github/languages/count/jeevanba273/quantflow-finance?style=flat&color=0080ff" alt="repo-language-count">
<a href="https://pypi.org/project/quantflow-finance/"><img src="https://img.shields.io/pypi/v/quantflow-finance?style=flat&logo=pypi&logoColor=white&color=0080ff&cache=bust" alt="pypi-version"></a>
<img src="https://static.pepy.tech/badge/quantflow-finance" alt="total-downloads">
<a href="https://github.com/jeevanba273/quantflow-finance/actions/workflows/ci.yml"><img src="https://github.com/jeevanba273/quantflow-finance/actions/workflows/ci.yml/badge.svg" alt="ci-status"></a>
<a href="https://quantflow-finance.readthedocs.io"><img src="https://readthedocs.org/projects/quantflow-finance/badge/?version=latest" alt="docs-status"></a>

<em>Built with the tools and technologies:</em>

<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat&logo=Python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/NumPy-013243.svg?style=flat&logo=NumPy&logoColor=white" alt="NumPy">
<img src="https://img.shields.io/badge/SciPy-8CAAE6.svg?style=flat&logo=SciPy&logoColor=white" alt="SciPy">
<img src="https://img.shields.io/badge/Pandas-150458.svg?style=flat&logo=pandas&logoColor=white" alt="Pandas">

</div>
<br>

QuantFlow Finance (`pip install quantflow-finance`, imported as `quantflow`)
is a small Python library for option pricing (Black-Scholes-Merton in closed
form and on NumPy arrays, Cox-Ross-Rubinstein trees for American options,
Monte Carlo with standard errors), risk statistics of return series (VaR,
expected shortfall, drawdown, Sharpe and related ratios) and mean-variance
portfolio optimisation. It is written for students, researchers and anyone
prototyping who wants short, documented formulas whose results have been
checked against textbooks, QuantLib and vollib. It depends only on NumPy,
SciPy and pandas.

## Install

```bash
pip install quantflow-finance            # Python 3.10+
pip install "quantflow-finance[data]"    # adds yfinance for MarketData.fetch_*
```

## Examples

The output under each example is what it prints; the test suite runs every
example in this file.

A European call and put (Hull's textbook example), the call's Greeks and its
implied volatility:

```python
from quantflow import BlackScholes

call = BlackScholes(S=42, K=40, T=0.5, r=0.10, sigma=0.20, option_type="call")
put = BlackScholes(S=42, K=40, T=0.5, r=0.10, sigma=0.20, option_type="put")
print(f"call {call.price():.4f}  put {put.price():.4f}")
print({k: round(v, 4) for k, v in call.greeks().items()})
print(f"implied vol {call.implied_volatility(4.70):.6f}")
```

```text
call 4.7594  put 0.8086
{'price': 4.7594, 'delta': 0.7791, 'gamma': 0.05, 'theta': -4.5591, 'vega': 0.0881, 'rho': 0.1398}
implied vol 0.193200
```

Theta is per year; vega and rho are per percentage point.

An American put on a binomial tree, the same contract as a European option in
closed form, and a Monte Carlo estimate with its standard error:

```python
from quantflow import BinomialTree, BlackScholes, MonteCarlo

american = BinomialTree(S=50, K=50, T=5 / 12, r=0.10, sigma=0.40, option_type="put",
                        exercise="american", steps=500)
european = BlackScholes(S=50, K=50, T=5 / 12, r=0.10, sigma=0.40, option_type="put")
mc = MonteCarlo(S=50, K=50, T=5 / 12, r=0.10, sigma=0.40, option_type="put",
                n_paths=1_000_000, antithetic=True, control_variate=True, seed=1)
res = mc.simulate(confidence=0.99)
print(f"American (tree, 500 steps) {american.price():.4f}")
print(f"European (closed form)     {european.price():.4f}")
print(f"European (Monte Carlo)     {res.price:.4f}, std error {res.std_error:.4f}, "
      f"99% interval [{res.ci_low:.4f}, {res.ci_high:.4f}]")
print(f"Monte Carlo minus closed form: {(res.price - european.price()) / res.std_error:+.1f} std errors")
```

```text
American (tree, 500 steps) 4.2830
European (closed form)     4.0760
European (Monte Carlo)     4.0716, std error 0.0019, 99% interval [4.0668, 4.0764]
Monte Carlo minus closed form: -2.3 std errors
```

The closed form lies inside the 99% interval. With this seed the estimate
happens to sit 2.3 standard errors low, a deviation that large or larger
occurs for about 2% of seeds; the
[Monte Carlo guide](https://quantflow-finance.readthedocs.io/en/latest/guide/monte-carlo.html)
shows the error shrinking with the number of paths.

Risk statistics of five years of simulated daily returns with fat tails
(Student-t, 5 degrees of freedom), where the true 1% quantile is known:

```python
import numpy as np
from scipy import stats

from quantflow import RiskMetrics

returns = 0.0003 + 0.009 * np.random.default_rng(2024).standard_t(5, 1250)
rm = RiskMetrics(returns)  # annualised with 252 periods per year
print(f"99% VaR: historical {rm.var_historical(0.01):.4f}, "
      f"normal {rm.var_parametric(0.01):.4f}, "
      f"true {0.0003 + 0.009 * stats.t.ppf(0.01, 5):.4f}")
print(f"99% expected shortfall {rm.expected_shortfall(0.01):.4f}")
print(f"volatility {rm.volatility():.4f}, max drawdown {rm.max_drawdown():.4f}")
```

```text
99% VaR: historical -0.0300, normal -0.0255, true -0.0300
99% expected shortfall -0.0371
volatility 0.1795, max drawdown -0.2986
```

VaR and expected shortfall are signed returns (a loss is negative). The
normal VaR is milder than the true quantile here because the returns have
fat tails.

## Validation

Every row is a test in `tests/`; the
[validation page](https://quantflow-finance.readthedocs.io/en/latest/validation.html)
prints the full tables.

| Check | Reference | Result |
|---|---|---|
| Black-Scholes call and put | Hull, S=42, K=40: 4.76 and 0.81 | 4.7594 and 0.8086 |
| 10 values from 9 worked examples (Merton, Black-76, Garman-Kohlhagen, Greeks) | Haug (2007), chapters 1 and 2 | all match to the 4 published decimals |
| American put, S=K=50, 5 to 500 steps | Hull: 4.49, 4.263, 4.272, 4.278, 4.283 | all match to the published decimals |
| Put-call parity, 648 parameter sets | exact identity | max gap below 1e-13 |
| Price and five Greeks, 6,048 options | QuantLib AnalyticEuropeanEngine | max abs difference 1.5e-13 (price), 1.3e-12 (theta) |
| Implied volatility, 5,186 options | true sigma; vollib and QuantLib | max error 3.4e-11 (vollib 1.7e-11, QuantLib 1.9e-11) |
| Monte Carlo, 16 seeded runs | closed form | inside every 99% confidence interval |
| Historical VaR and ES, 400,000 draws | exact normal and Student-t(5) values | within 5 asymptotic standard errors |
| Minimum-variance and frontier weights | Merton (1972) closed forms | max weight difference below 1e-6 |

## Benchmarks

Measured with `python benchmarks/run_benchmarks.py` on an Intel Core Ultra 9
185H laptop (22 logical cores, 31.5 GB RAM, Windows 11) with Python 3.13.5,
NumPy 2.5.4, SciPy 1.18.1, QuantLib 1.44 and vollib 1.0.11. Medians of 7
repetitions; the
[benchmarks page](https://quantflow-finance.readthedocs.io/en/latest/benchmarks.html)
has the spread, the method and more rows.

| Task | quantflow | Baseline |
|---|---|---|
| Price 1,000,000 options, vectorised | 5,691,834 options/s | plain NumPy/SciPy, no input checks: 9,059,514 options/s |
| Price one option at a time (Python loop) | 90,568 options/s | vollib (pure Python): 460,384; QuantLib (C++, objects reused): 272,509 |
| Implied volatility, one at a time | 36,366 solves/s | vollib (Let's Be Rational): 56,190; QuantLib: 71,124 |

quantflow is slower than QuantLib and vollib for one option at a time and
for implied volatility, and slower than a bare NumPy implementation because
it validates inputs and evaluates puts directly rather than through parity.

## Limitations

- Black-Scholes-Merton assumptions throughout: constant volatility, rates and
  dividend yield, lognormal prices, no jumps, no transaction costs. No
  volatility smile, no term structures, no stochastic volatility.
- `T` is a year fraction you supply; there are no calendars, day counts or
  settlement conventions. Dividends are a continuous yield only.
- Trees handle American calls and puts; there are no barrier, Asian or other
  exotic payoffs, and Monte Carlo covers European payoffs only.
- Implied volatility is solved one option at a time. For deep in-the-money or
  far out-of-the-money options very close to expiry the volatility is not
  determined to many digits, and prices with no time value raise an error.
- Risk statistics treat the returns as one stationary sample: no GARCH or
  EWMA volatility, no multi-day VaR scaling, no VaR backtests. Sample Sharpe
  ratios from a few years of data have standard errors of several tenths.
- Portfolio optimisation uses sample means and covariances as if they were
  known; the weights are sensitive to estimation error, and there is no
  shrinkage, transaction-cost or turnover control.
- `MarketData` relies on Yahoo Finance through yfinance; the data are not
  checked by this library.

Do not use the results for trading or risk decisions without your own
checks.

## Documentation

<https://quantflow-finance.readthedocs.io>: getting started, a user guide
for each area, the API reference, the models and their assumptions,
validation and benchmarks.

## Further reading

- [Hedging a Portfolio with Index Options: Protective Puts, Collars, and What Insurance Really Costs](https://financefluentblog.com/blog/hedging-portfolio/)
- [Butterfly Option Structures: Long, Iron and Broken-Wing, and When Each One Earns Its Place](https://financefluentblog.com/blog/neutral-options/)
- [Credit Spreads and Iron Condors: What You Are Actually Being Paid For](https://financefluentblog.com/blog/income-oriented-spreads/)
- [Monte Carlo Value-at-Risk: Simulating the Tail, and Knowing How Far to Trust It](https://financefluentblog.com/blog/monte-carlo-var/)
- [Risk Management Essentials: Drawdowns, Value at Risk, and the Limits of Both](https://financefluentblog.com/blog/risk-management/)
- [Portfolio Construction: The Theory, the Process, and Why Optimisers Give Bad Answers](https://financefluentblog.com/blog/portfolio-construction/)

## Citing and licence

Citation metadata are in
[CITATION.cff](https://github.com/jeevanba273/quantflow-finance/blob/main/CITATION.cff).
MIT licence; see
[LICENSE](https://github.com/jeevanba273/quantflow-finance/blob/main/LICENSE).
Contributions:
[CONTRIBUTING.md](https://github.com/jeevanba273/quantflow-finance/blob/main/CONTRIBUTING.md).

For research and education; not investment advice.
