# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - Unreleased

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

[0.2.0]: https://github.com/jeevanba273/quantflow-finance/compare/v0.1.10...HEAD
[0.1.10]: https://github.com/jeevanba273/quantflow-finance/releases/tag/v0.1.10
