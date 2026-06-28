# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2026-06-29

Robustness and input-validation hardening following a deep audit. All changes are
backward-compatible; correct prior usage is unaffected.

### Fixed
- **Portfolio optimization (critical):** `Portfolio.min_variance`, the
  minimum-variance end of `efficient_frontier`, and `optimize('target_return')`
  could silently return the equal-weight starting guess for typical daily-frequency
  returns. The per-period variance objective (on the order of 1e-5 for daily data)
  fell below SLSQP's default tolerance, so the optimizer reported `success=True`
  without actually optimizing. The variance objective is now normalized to order 1
  and the solver tolerance tightened, so results match the analytical closed-form
  solution at any return frequency. `max_sharpe` was not affected.
- **`BlackScholes.implied_volatility`:** now validates `market_price` (must be
  positive and within the option's no-arbitrage bounds) and solves with a robust
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
- **`theta()` docstring:** corrected the one-line label to `-∂V/∂T` (time decay per
  year); the formula block was already correct.

## [0.2.0] - 2026-06-29

All changes are additive and backward-compatible. Existing `BlackScholes`,
`RiskMetrics`, and `MarketData` calls keep working unchanged.

### Added
- **Dividend yield (`q`)** on `BlackScholes` (Black-Scholes-Merton). New optional
  keyword `q=0.0`; the price, delta, gamma, theta, vega, and implied volatility
  are all dividend-aware. `q=0.0` reproduces the previous results exactly.
- **`BinomialTree`** — Cox-Ross-Rubinstein lattice pricer supporting European and
  American exercise, dividends, a full Greeks suite (delta/gamma/theta from the
  lattice; vega/rho by bump-and-reprice), and `early_exercise_premium()`.
- **`OptionPricer`** abstract base (internal) centralizing shared parameter
  validation and terminal payoff helpers for lattice/Monte-Carlo pricers.
- **`RiskMetrics`** additions: `sortino_ratio`, `calmar_ratio`, `omega_ratio`,
  CAPM `beta`/`alpha`/`capm`, `expected_shortfall_parametric`, and
  `var_cornish_fisher`. New configurable `periods_per_year` constructor argument
  (default 252) replacing the previously hardcoded trading-day count.
- **`Portfolio`** class — covariance/correlation analysis, portfolio
  return/volatility/Sharpe, mean-variance optimization (minimum-variance,
  maximum-Sharpe, efficient frontier), and a bridge to `RiskMetrics` for the
  optimized portfolio.

### Changed
- Packaging migrated from `setup.py` to a PEP 621 `pyproject.toml`. The package
  version is now single-sourced from `quantflow.__version__`.
- Test suite migrated to a proper `pytest` layout with fixtures and an offline
  yfinance mock; live-network tests are marked `@pytest.mark.network`.
- `comprehensive_test.py` moved to `examples/comprehensive_demo.py`.
- Removed decorative emojis from the README and example scripts for a more
  professional presentation.

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
- Upgraded to Beta status with enhanced classifiers, audiences, and professional metadata.

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

[0.2.1]: https://github.com/jeevanba273/quantflow-finance/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/jeevanba273/quantflow-finance/compare/v0.1.10...v0.2.0
[0.1.10]: https://github.com/jeevanba273/quantflow-finance/releases/tag/v0.1.10
