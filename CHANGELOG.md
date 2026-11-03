# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-11-03

### Fixed
- `RiskMetrics.max_drawdown` ignored a loss from the starting value: the
  running peak began at the first period's wealth instead of 1.0, so returns
  of -10% then +5% gave a drawdown of 0 instead of -10%. `calmar_ratio` was
  affected the same way.
- `RiskMetrics.beta`, `alpha`, `capm` and `information_ratio` paired the
  portfolio and benchmark by position. Date-indexed series with different
  coverage were silently misaligned (dropping five portfolio days turned a
  beta of 1.22 into -0.01). They are now joined on common dates.
- `Portfolio.max_sharpe(allow_short=True)` returned `success=True` with
  weights in the thousands when the minimum-variance return was below the
  risk-free rate, where no tangency portfolio exists. It now uses the closed
  form and raises a `ValueError` in that case.
- `BinomialTree.gamma` returned 47.0 for an at-the-money call with
  `steps=2`, and `theta` returned NaN for fewer than 3 steps. Both now use
  the lattice formulas for any `steps >= 2` and raise for `steps=1`.
- `BlackScholes.implied_volatility` returned 1e-6 for a price with no time
  value above the lower bound. It now raises a `ValueError` saying the
  volatility is not determined; prices outside the bounds name the bound.
- `summary()` and `repr()` of the option classes raised
  `UnicodeEncodeError` on Windows cp1252 streams; their output is now ASCII.

### Added
- `black_scholes_price` and `black_scholes_greeks`: vectorised
  Black-Scholes-Merton functions that broadcast over NumPy arrays.
- `MonteCarlo` and `MonteCarloResult`: European options by exact GBM
  sampling, with standard error, confidence interval, antithetic variates
  and a control variate.
- Tests against independent references: Hull's and Haug's worked examples,
  put-call parity, finite-difference Greeks, Hull's American put values,
  normal and Student-t VaR and ES, Merton's (1972) closed-form portfolios,
  and optional cross-checks against QuantLib and vollib. `pytest` now
  collects 396 tests (81 before), including doctests, the documented
  examples and the example scripts; 395 run offline with the `validation`
  extra installed.
- `benchmarks/run_benchmarks.py`, a validation page and a benchmarks page
  in the documentation, and `CITATION.cff`.
- `[data]` and `[validation]` extras.

### Changed
- Python 3.10 or newer is required.
- yfinance is now an optional extra (`pip install "quantflow-finance[data]"`)
  imported only by the `MarketData` fetch methods; matplotlib, which was
  never used, is no longer a dependency.
- `BlackScholes` now shares the vectorised implementation and the
  `OptionPricer` validation. `implied_volatility` evaluates a scalar price
  function instead of building an object at every Brent step (current
  timings are on the benchmarks page), and its default tolerance is 1e-10
  instead of 1e-6.
- `RiskMetrics` rejects multi-column DataFrames and infinite returns;
  option classes reject array and non-numeric inputs with clear messages;
  `BinomialTree` rejects a non-string `exercise` and boolean `steps`.
- Warnings are issued with `warnings.warn` and fetch progress goes to the
  `quantflow.data.fetcher` logger instead of `print`. A failed fetch raises
  `RuntimeError` (a subclass of the `Exception` raised before).
- Methods return plain Python floats rather than NumPy scalars.
- `Portfolio.efficient_frontier` takes `risk_free_rate` for its Sharpe
  column (previously fixed at 2%).
- `BinomialTree.greeks()` builds five trees instead of eight.
- Docstrings state formulas, units and conventions and cite sources; the
  documentation was rewritten, and every example in it and in the README is
  executed by the tests.
- Lint and formatting use ruff; CI also tests the minimum dependency
  versions, runs the reference cross-checks and builds the docs with
  warnings as errors.

### Removed
- `examples/comprehensive_demo.py` (live data, an incorrect diversification
  ratio, and a crash on Windows consoles) and the two live-data examples,
  replaced by offline examples.

## [0.2.1] - 2026-06-29

Input validation and bug fixes. All changes are backward-compatible; correct
prior usage is unaffected.

### Fixed
- **Portfolio optimization:** `Portfolio.min_variance`, the
  minimum-variance end of `efficient_frontier`, and `optimize('target_return')`
  could silently return the equal-weight starting guess for typical daily-frequency
  returns. The per-period variance objective (on the order of 1e-5 for daily data)
  fell below SLSQP's default tolerance, so the optimizer reported `success=True`
  without actually optimizing. The variance objective is now normalized to order 1
  and the solver tolerance tightened, so results match the analytical closed-form
  solution at any return frequency. `max_sharpe` was not affected.
- **`BlackScholes.implied_volatility`:** now validates `market_price` (must be
  positive and within the option's no-arbitrage bounds) and solves with a
  bracketing method (Brent), instead of an unbracketed Newton-Raphson that could
  silently return its initial guess in low-vega regimes.
- **Constructor validation:** `BlackScholes` and `BinomialTree` now reject
  non-finite `S`, `K`, `T`, `sigma`, and `r` (previously only `q` was checked, so
  `NaN`/`inf` slipped through and silently poisoned pricing). A non-string
  `option_type` now raises `ValueError` rather than `AttributeError`.
- **`RiskMetrics.beta` / `alpha` / `capm`:** the zero-variance benchmark guard now
  uses a scale-aware tolerance, so a (near-)constant benchmark reliably raises
  instead of returning a meaningless beta from floating-point residual variance.
- **`RiskMetrics.var_parametric`:** now validates the confidence level
  (`0 < cl < 1`) like the other VaR/ES methods.
- **`MarketData.calculate_returns(method='simple')`:** uses
  `pct_change(fill_method=None)`, so interior gaps no longer create spurious 0%
  returns (now consistent with log returns) and the pandas `FutureWarning` is gone.
- **`theta()` docstring:** corrected the one-line label to `-dV/dT` (time decay per
  year); the formula block was already correct.

## [0.2.0] - 2026-06-29

All changes are additive and backward-compatible. Existing `BlackScholes`,
`RiskMetrics`, and `MarketData` calls keep working unchanged.

### Added
- **Dividend yield (`q`)** on `BlackScholes` (Black-Scholes-Merton). New optional
  keyword `q=0.0`; the price, delta, gamma, theta, vega, and implied volatility
  are all dividend-aware. `q=0.0` reproduces the previous results exactly.
- **`BinomialTree`**: Cox-Ross-Rubinstein lattice pricer supporting European and
  American exercise, dividends, a full Greeks suite (delta/gamma/theta from the
  lattice; vega/rho by bump-and-reprice), and `early_exercise_premium()`.
- **`OptionPricer`** abstract base (internal) centralizing shared parameter
  validation and terminal payoff helpers for lattice/Monte-Carlo pricers.
- **`RiskMetrics`** additions: `sortino_ratio`, `calmar_ratio`, `omega_ratio`,
  CAPM `beta`/`alpha`/`capm`, `expected_shortfall_parametric`, and
  `var_cornish_fisher`. New configurable `periods_per_year` constructor argument
  (default 252) replacing the previously hardcoded trading-day count.
- **`Portfolio`** class: covariance/correlation analysis, portfolio
  return/volatility/Sharpe, mean-variance optimization (minimum-variance,
  maximum-Sharpe, efficient frontier), and a bridge to `RiskMetrics` for the
  optimized portfolio.

### Changed
- Packaging migrated from `setup.py` to a PEP 621 `pyproject.toml`. The package
  version is now single-sourced from `quantflow.__version__`.
- Test suite migrated to a proper `pytest` layout with fixtures and an offline
  yfinance mock; live-network tests are marked `@pytest.mark.network`.
- `comprehensive_test.py` moved to `examples/comprehensive_demo.py`.
- Removed emojis from the README and example scripts.

### Removed
- Dropped Python 3.8 support (end-of-life since October 2024; the `yfinance`
  dependency chain and current numpy/pandas no longer support it). Minimum
  supported version is now Python 3.9.

### Added (project infrastructure)
- `.gitignore`, `.flake8`, `CHANGELOG.md`, `CONTRIBUTING.md`, and GitHub Actions
  workflows for CI (lint + test matrix across Python 3.8-3.13) and PyPI release.
- Stopped tracking build artifacts (`dist/`, `*.egg-info/`, `*.pyc`).

## [0.1.10] - 2025-06-12
### Changed
- Beta status classifier and additional package metadata.

## [0.1.9] - 2025-06-12
### Fixed
- yfinance session compatibility (removed custom curl_cffi sessions).

## [0.1.8] - 2025-06-12
### Fixed
- yfinance compatibility (removed the deprecated `show_errors` parameter).

## [0.1.7] - 2025-06-12
### Fixed
- Data fetching and risk metrics: added browser headers and retry logic.

## [0.1.6] - 2025-06-12
### Fixed
- Package structure: added missing `__init__.py` files.

## [0.1.0] - 2025-06-09
### Added
- Initial release: Black-Scholes pricing with Greeks, core risk metrics, and
  Yahoo Finance market data integration.

[0.3.0]: https://github.com/jeevanba273/quantflow-finance/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/jeevanba273/quantflow-finance/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/jeevanba273/quantflow-finance/compare/v0.1.10...v0.2.0
[0.1.10]: https://github.com/jeevanba273/quantflow-finance/releases/tag/v0.1.10
