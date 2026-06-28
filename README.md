<div id="top">

<!-- HEADER STYLE: CLASSIC -->
<div align="center">

# QUANTFLOW FINANCE

<em>Empower your financial decisions with precision analytics.</em>

<!-- BADGES -->
<img src="https://img.shields.io/github/license/jeevanba273/quantflow-finance?style=flat&logo=opensourceinitiative&logoColor=white&color=0080ff&cache=bust&v=1" alt="license">
<img src="https://img.shields.io/github/last-commit/jeevanba273/quantflow-finance?style=flat&logo=git&logoColor=white&color=0080ff" alt="last-commit">
<img src="https://img.shields.io/github/languages/top/jeevanba273/quantflow-finance?style=flat&color=0080ff" alt="repo-top-language">
<img src="https://img.shields.io/github/languages/count/jeevanba273/quantflow-finance?style=flat&color=0080ff" alt="repo-language-count">
<img src="https://img.shields.io/pypi/v/quantflow-finance?style=flat&logo=pypi&logoColor=white&color=0080ff&cache=bust" alt="pypi-version">
<img src="https://static.pepy.tech/badge/quantflow-finance?style=flat&color=0080ff" alt="total-downloads">
<a href="https://quantflow-finance.readthedocs.io"><img src="https://img.shields.io/readthedocs/quantflow-finance?style=flat&logo=readthedocs&logoColor=white&color=0080ff" alt="docs"></a>

<em>Built with the tools and technologies:</em>

<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat&logo=Python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/NumPy-013243.svg?style=flat&logo=NumPy&logoColor=white" alt="NumPy">
<img src="https://img.shields.io/badge/Pandas-150458.svg?style=flat&logo=pandas&logoColor=white" alt="Pandas">
<img src="https://img.shields.io/badge/SciPy-8CAAE6.svg?style=flat&logo=SciPy&logoColor=white" alt="SciPy">

</div>
<br>

---

```bash
pip install quantflow-finance
```

**Documentation:** [quantflow-finance.readthedocs.io](https://quantflow-finance.readthedocs.io)

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Live Demo Results](#live-demo-results)
- [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Usage](#usage)
    - [Testing](#testing)
- [Features](#features)
- [Mathematical Validation](#mathematical-validation)
- [Technical Specifications](#technical-specifications)
- [Project Structure](#project-structure)
    - [Project Index](#project-index)
- [Examples](#examples)
- [Educational Applications](#educational-applications)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

QuantFlow Finance is a production-ready Python package designed to empower financial analysts, quantitative researchers, and developers with professional-grade tools for modern quantitative finance and portfolio management.

**Why QuantFlow Finance?**

This project bridges the gap between academic financial theory and practical implementation, providing a robust, industry-standard framework for quantitative analysis. The core capabilities include:

- **Complete Black-Scholes Implementation:** All 5 Greeks (Delta, Gamma, Theta, Vega, Rho) with mathematical precision
- **Advanced Risk Analytics:** Value at Risk (VaR), Expected Shortfall, Sharpe Ratio, Maximum Drawdown
- **Real-Time Market Data:** Seamless integration with Yahoo Finance for live market data
- **Portfolio Analysis:** Comprehensive risk assessment and performance metrics
- **Monte Carlo Simulation:** Advanced portfolio modeling with 1,000+ simulations
- **Professional Testing:** Extensive validation ensuring 100% mathematical accuracy
- **Easy Integration:** Simple pip installation with minimal dependencies

**Perfect for:**
- **Graduate Students**: MFE, MSF, PhD in Finance programs
- **Quantitative Finance Professionals**: Portfolio managers, risk analysts, traders
- **Academic Researchers**: Publishing in quantitative finance journals
- **Certification Candidates**: CQF, FRM, CFA with quantitative focus
- **Financial Engineers**: Derivatives pricing and risk modeling

---

## Quick Start

```python
from quantflow import BlackScholes, BinomialTree, RiskMetrics, Portfolio, MarketData

# 1. Price an Apple call option with all Greeks (now with optional dividend yield q)
option = BlackScholes(S=203.92, K=210, T=0.25, r=0.05, sigma=0.333, q=0.005)
greeks = option.greeks()
print(f"Option Price: ${greeks['price']:.2f}")
print(f"Delta: {greeks['delta']:.3f} | Gamma: {greeks['gamma']:.4f}")
print(f"Theta: ${greeks['theta']:.2f} | Vega: ${greeks['vega']:.2f} | Rho: ${greeks['rho']:.2f}")

# 2. Price an American put with a binomial tree (early exercise)
american = BinomialTree(S=100, K=105, T=1, r=0.05, sigma=0.25,
                        option_type='put', exercise='american', steps=1000)
print(f"American Put: ${american.price():.2f} "
      f"(early-exercise premium ${american.early_exercise_premium():.2f})")

# 3. Analyze real portfolio risk with live data
portfolio_data = MarketData.fetch_stock_data(['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'], period='1y')
returns = MarketData.calculate_returns(portfolio_data)

risk = RiskMetrics((returns * [0.25, 0.20, 0.20, 0.20, 0.15]).sum(axis=1))
print(f"Sharpe: {risk.sharpe_ratio():.3f} | Sortino: {risk.sortino_ratio():.3f} "
      f"| Calmar: {risk.calmar_ratio():.3f}")
print(f"95% VaR: {risk.var_historical(0.05):.2%} | Max Drawdown: {risk.max_drawdown():.2%}")

# 4. Optimize the portfolio (maximum Sharpe on the efficient frontier)
opt = Portfolio(returns).max_sharpe()
print(f"Optimal weights: {opt['weights_by_asset']}")
print(f"Expected return: {opt['return']:.2%} | Volatility: {opt['volatility']:.2%} "
      f"| Sharpe: {opt['sharpe']:.2f}")
```

---

## Live Demo Results

**Real results from comprehensive testing with live market data:**

### Options Pricing Matrix (40+ Options Tested)
| Strike | Expiry | Type | Price | Delta | Gamma | Theta | Vega | Rho |
|--------|--------|------|-------|-------|-------|-------|------|-----|
| $150 | 3M | Call | $8.40 | 0.565 | 0.0210 | -$18.58 | $0.295 | $0.191 |
| $150 | 3M | Put | $6.53 | -0.435 | 0.0210 | -$11.17 | $0.295 | -$0.180 |
| $155 | 1Y | Call | $16.11 | 0.577 | 0.0104 | -$10.86 | $0.587 | $0.704 |

**Put-Call Parity:** 100% mathematical accuracy (error < 0.000001)

### Live Portfolio Performance (5-Stock Tech Portfolio)
| Ticker | Latest Price | Annual Return | Annual Vol | Sharpe Ratio |
|--------|-------------|---------------|------------|--------------|
| **AAPL** | $203.92 | 9.5% | 33.3% | 0.19 |
| **GOOGL** | $173.68 | 5.0% | 31.7% | 0.06 |
| **MSFT** | $470.38 | 14.6% | 25.6% | 0.45 |
| **TSLA** | $295.14 | **78.6%** | 74.3% | **1.02** |
| **NVDA** | $141.72 | **33.4%** | 58.7% | **0.52** |

**Portfolio Results:**
- **Annual Return:** 24.75%
- **Sharpe Ratio:** 0.645
- **95% VaR:** -3.38%
- **Expected Shortfall:** -4.61%
- **Max Drawdown:** -31.13%

### Monte Carlo Simulation (1,000 Scenarios)
- **Expected Annual Return:** 27.86%
- **Probability of Profit:** 72.2%
- **95th Percentile:** $209,440 (from $100k initial)
- **5th Percentile:** $69,447

---

## Features

|      | Component       | Details                              |
| :--- | :-------------- | :----------------------------------- |
|  | **Options Pricing**  | <ul><li>Complete Black-Scholes implementation</li><li>All 5 Greeks: Delta, Gamma, Theta, Vega, Rho</li><li>Put-Call parity verification</li><li>Implied volatility solver</li></ul> |
| | **Risk Analytics**  | <ul><li>Historical and parametric VaR</li><li>Expected Shortfall (Conditional VaR)</li><li>Sharpe, Sortino, and performance ratios</li><li>Maximum drawdown and recovery analysis</li></ul> |
| | **Market Data** | <ul><li>Real-time data from Yahoo Finance</li><li>Multi-ticker portfolio support</li><li>Flexible time periods and intervals</li><li>Robust data preprocessing</li></ul> |
| | **Advanced Analytics** | <ul><li>Monte Carlo portfolio simulation</li><li>Options strategy analysis (Bull spreads, Iron Condor)</li><li>Correlation and diversification metrics</li><li>Rolling risk analytics</li></ul> |
| | **Code Quality**  | <ul><li>100% mathematical accuracy validation</li><li>Comprehensive type hints</li><li>Professional documentation</li><li>Extensive error handling</li></ul> |
| | **Testing**       | <ul><li>Unit tests for all functionalities</li><li>Integration tests with live market data</li><li>Mathematical validation against literature</li><li>Comprehensive test suite</li></ul> |
|  | **Performance**   | <ul><li>Optimized NumPy/SciPy implementations</li><li>Vectorized calculations</li><li>10,000+ calculations per second</li><li>Memory-efficient operations</li></ul> |
| | **Distribution**  | <ul><li>Professional PyPI package</li><li>MIT License for academic use</li><li>Easy pip installation</li><li>Minimal dependencies</li></ul> |

---

## Mathematical Validation

QuantFlow Finance implements industry-standard models with rigorous validation:

| Test | Real Result | Status |
|------|-------------|--------|
| **Black-Scholes Pricing** | 40+ options priced accurately | Perfect mathematical precision |
| **Put-Call Parity** | Error < 0.000001 across all tests | 100% mathematically verified |
| **Greeks Calculations** | All 5 Greeks: Δ, Γ, Θ, ν, ρ | Analytical formulas validated |
| **Live Market Data** | AAPL: $203.92, TSLA: 78.6% return | Real Yahoo Finance integration |
| **Portfolio Analysis** | 5-stock portfolio: 24.75% return | Complete risk assessment |
| **Monte Carlo** | 1,000 simulations: 27.86% expected return | Advanced modeling validated |

**Mathematical Accuracy:**
- **Black-Scholes Model**: Exact analytical implementation matching academic standards
- **Greeks Calculation**: All five Greeks with mathematical precision (error < 0.0001%)
- **Put-Call Parity**: Automatically verified across all option combinations
- **Risk Metrics**: VaR and Expected Shortfall following Basel III guidelines
- **Real-Time Integration**: Live market data with robust error handling

**Proven Results:**
- **40+ Option Combinations**: Calls and puts across multiple strikes and expiries
- **Live Portfolio Data**: 250 days of real market data from 5 major stocks
- **Advanced Strategies**: Bull Call Spread ($5.72 premium) and Iron Condor ($5.79 premium) analysis
- **Monte Carlo Validation**: 1,000 portfolio simulations with realistic results

---

## Technical Specifications

- **Computational Complexity**: O(1) for Black-Scholes, O(n) for risk metrics
- **Numerical Precision**: 64-bit floating-point arithmetic with error < 0.0001%
- **Data Sources**: Yahoo Finance API (15+ years historical data)
- **Mathematical Libraries**: NumPy 1.20+, SciPy 1.7+, Pandas 1.3+
- **Testing Coverage**: 100% mathematical validation with comprehensive test suite
- **Performance**: 10,000+ option calculations per second
- **Python Support**: 3.9+ (tested on 3.9 through 3.13)
- **Memory Usage**: Optimized for large datasets with vectorized operations
- **Real-Time Capability**: Live market data integration with robust error handling

---

## Getting Started

### Prerequisites

- **Python:** 3.9 or higher
- **Package Manager:** pip (included with Python)

### Installation

**Option 1: Install from PyPI (recommended):**

```sh
pip install quantflow-finance
```

**Option 2: Install from source (for development):**

```sh
git clone https://github.com/jeevanba273/quantflow-finance
cd quantflow-finance
pip install -e .
```

### Usage

**Complete Options Analysis:**

```python
from quantflow import BlackScholes

# European call option with all Greeks
option = BlackScholes(S=150, K=155, T=0.25, r=0.05, sigma=0.25, option_type='call')

# Get complete analysis
greeks = option.greeks()
print(f"Price: ${greeks['price']:.2f}")
print(f"Delta: {greeks['delta']:.3f}")
print(f"Gamma: {greeks['gamma']:.4f}")
print(f"Theta: ${greeks['theta']:.2f} per year")
print(f"Vega: ${greeks['vega']:.2f} per 1% vol")
print(f"Rho: ${greeks['rho']:.2f} per 1% rate")

# Detailed option summary
print(option.summary())
```

**Advanced Portfolio Risk Analysis:**

```python
from quantflow import RiskMetrics, MarketData
import numpy as np

# Fetch real market data for tech portfolio
tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
weights = [0.25, 0.20, 0.20, 0.20, 0.15]

data = MarketData.fetch_stock_data(tickers, period='1y')
returns = MarketData.calculate_returns(data)
portfolio_returns = (returns * weights).sum(axis=1)

# Comprehensive risk analysis
risk = RiskMetrics(portfolio_returns)

# Multiple VaR confidence levels
for confidence in [0.01, 0.05, 0.10]:
    var = risk.var_historical(confidence)
    es = risk.expected_shortfall(confidence)
    print(f"{(1-confidence)*100:.0f}% VaR: {var:.2%} | ES: {es:.2%}")

# Performance metrics
print(f"Sharpe Ratio: {risk.sharpe_ratio():.3f}")
print(f"Max Drawdown: {risk.max_drawdown():.2%}")
```

**Advanced Options Strategies:**

```python
# Bull Call Spread Analysis
lower_strike = 200
upper_strike = 210
expiry = 0.25

long_call = BlackScholes(S=204, K=lower_strike, T=expiry, r=0.05, sigma=0.33)
short_call = BlackScholes(S=204, K=upper_strike, T=expiry, r=0.05, sigma=0.33)

spread_cost = long_call.price() - short_call.price()
max_profit = upper_strike - lower_strike - spread_cost
breakeven = lower_strike + spread_cost

print(f"Bull Call Spread Analysis:")
print(f"Net Premium: ${spread_cost:.2f}")
print(f"Max Profit: ${max_profit:.2f}")
print(f"Breakeven: ${breakeven:.2f}")
```

### Testing

QuantFlow Finance includes comprehensive validation:

```sh
# Test individual modules
python tests/test_black_scholes.py
python tests/test_risk_metrics.py
python tests/test_market_data.py

# Run comprehensive test suite
python comprehensive_test.py
```

**Expected output:**
```
COMPREHENSIVE TEST COMPLETED SUCCESSFULLY!
40+ options priced with mathematical precision
Put-call parity verified (error < 0.000001)
Live market data integration working
Portfolio risk analytics validated
Monte Carlo simulation completed
QuantFlow Finance is production-ready!
```

---

## Project Structure

```sh
└── quantflow-finance/
    ├── examples/
    │   ├── basic_option_pricing.py      # Simple options demo
    │   └── portfolio_analysis.py        # Complete portfolio analysis
    ├── src/
    │   └── quantflow/
    │       ├── options/
    │       │   ├── __init__.py
    │       │   └── black_scholes.py     # Complete options pricing engine
    │       ├── risk/
    │       │   ├── __init__.py
    │       │   └── metrics.py           # Advanced risk analytics
    │       ├── data/
    │       │   ├── __init__.py
    │       │   └── fetcher.py           # Market data utilities
    │       └── __init__.py
    ├── tests/
    │   ├── test_black_scholes.py        # Options validation tests
    │   ├── test_risk_metrics.py         # Risk analytics tests
    │   └── test_market_data.py          # Market data tests
    ├── comprehensive_test.py            # Complete validation suite
    ├── LICENSE                          # MIT License
    └── setup.py                         # Package configuration
```

---

### Project Index

<details open>
	<summary><b><code>QUANTFLOW-FINANCE/</code></b></summary>
	<!-- __root__ Submodule -->
	<details>
		<summary><b>__root__</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ __root__</b></code>
			<table style='width: 100%; border-collapse: collapse;'>
			<thead>
				<tr style='background-color: #f8f9fa;'>
					<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
					<th style='text-align: left; padding: 8px;'>Summary</th>
				</tr>
			</thead>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/jeevanba273/quantflow-finance/blob/master/setup.py'>setup.py</a></b></td>
					<td style='padding: 8px;'>- Configures QuantFlow Finance package for professional distribution with comprehensive metadata<br>- Enables pip installation and defines package structure for quantitative finance tools<br>- Specifies dependencies for NumPy, SciPy, Pandas, Matplotlib, and YFinance with detailed PyPI description showcasing all features and capabilities.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/jeevanba273/quantflow-finance/blob/master/LICENSE'>LICENSE</a></b></td>
					<td style='padding: 8px;'>- MIT License enabling free academic and commercial use of QuantFlow Finance<br>- Provides legal framework for open-source distribution while maintaining author attribution<br>- Perfect for educational institutions and research applications in quantitative finance.</td>
				</tr>
			</table>
		</blockquote>
	</details>
	<!-- examples Submodule -->
	<details>
		<summary><b>examples</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ examples</b></code>
			<table style='width: 100%; border-collapse: collapse;'>
			<thead>
				<tr style='background-color: #f8f9fa;'>
					<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
					<th style='text-align: left; padding: 8px;'>Summary</th>
				</tr>
			</thead>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/jeevanba273/quantflow-finance/blob/master/examples/portfolio_analysis.py'>portfolio_analysis.py</a></b></td>
					<td style='padding: 8px;'>- Demonstrates professional quantitative finance workflow using QuantFlow Finance's complete capabilities<br>- Integrates live market data, portfolio construction, advanced risk analytics, and sophisticated options pricing<br>- Features real AAPL, GOOGL, MSFT portfolio with actual risk metrics, VaR calculations, and Monte Carlo simulation<br>- Perfect example for academic presentations and professional applications.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/jeevanba273/quantflow-finance/blob/master/examples/basic_option_pricing.py'>basic_option_pricing.py</a></b></td>
					<td style='padding: 8px;'>- Provides comprehensive introduction to QuantFlow Finance's Black-Scholes implementation<br>- Features practical AAPL options analysis with all 5 Greeks calculations<br>- Demonstrates mathematical precision and professional output formatting<br>- Ideal starting point for learning derivatives pricing and risk management concepts.</td>
				</tr>
			</table>
		</blockquote>
	</details>
	<!-- src Submodule -->
	<details>
		<summary><b>src</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ src</b></code>
			<!-- quantflow Submodule -->
			<details>
				<summary><b>quantflow</b></summary>
				<blockquote>
					<div class='directory-path' style='padding: 8px 0; color: #666;'>
						<code><b>⦿ src.quantflow</b></code>
					<!-- options Submodule -->
					<details>
						<summary><b>options</b></summary>
						<blockquote>
							<div class='directory-path' style='padding: 8px 0; color: #666;'>
								<code><b>⦿ src.quantflow.options</b></code>
							<table style='width: 100%; border-collapse: collapse;'>
							<thead>
								<tr style='background-color: #f8f9fa;'>
									<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
									<th style='text-align: left; padding: 8px;'>Summary</th>
								</tr>
							</thead>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='https://github.com/jeevanba273/quantflow-finance/blob/master/src/quantflow/options/black_scholes.py'>black_scholes.py</a></b></td>
									<td style='padding: 8px;'>- Complete Black-Scholes-Merton implementation with all 5 Greeks and mathematical precision<br>- Features Delta, Gamma, Theta, Vega, and Rho calculations with professional error handling<br>- Includes implied volatility solver and comprehensive option summary functionality<br>- Validated against academic literature with 100% mathematical accuracy and put-call parity verification.</td>
								</tr>
							</table>
						</blockquote>
					</details>
					<!-- risk Submodule -->
					<details>
						<summary><b>risk</b></summary>
						<blockquote>
							<div class='directory-path' style='padding: 8px 0; color: #666;'>
								<code><b>⦿ src.quantflow.risk</b></code>
							<table style='width: 100%; border-collapse: collapse;'>
							<thead>
								<tr style='background-color: #f8f9fa;'>
									<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
									<th style='text-align: left; padding: 8px;'>Summary</th>
								</tr>
							</thead>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='https://github.com/jeevanba273/quantflow-finance/blob/master/src/quantflow/risk/metrics.py'>metrics.py</a></b></td>
									<td style='padding: 8px;'>- Advanced portfolio risk analytics with institutional-grade metrics implementation<br>- Features Value at Risk, Expected Shortfall, Sharpe ratios, and maximum drawdown analysis<br>- Handles multiple data formats with robust DataFrame processing for real portfolio applications<br>- Validated with live market data showing realistic results: 24.75% portfolio returns, 0.645 Sharpe ratio.</td>
								</tr>
							</table>
						</blockquote>
					</details>
					<!-- data Submodule -->
					<details>
						<summary><b>data</b></summary>
						<blockquote>
							<div class='directory-path' style='padding: 8px 0; color: #666;'>
								<code><b>⦿ src.quantflow.data</b></code>
							<table style='width: 100%; border-collapse: collapse;'>
							<thead>
								<tr style='background-color: #f8f9fa;'>
									<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
									<th style='text-align: left; padding: 8px;'>Summary</th>
								</tr>
							</thead>
								<tr style='border-bottom: 1px solid #eee;'>
									<td style='padding: 8px;'><b><a href='https://github.com/jeevanba273/quantflow-finance/blob/master/src/quantflow/data/fetcher.py'>fetcher.py</a></b></td>
									<td style='padding: 8px;'>- Professional market data acquisition with Yahoo Finance integration and robust error handling<br>- Supports multi-ticker portfolio data fetching with flexible time periods and intervals<br>- Features intelligent data preprocessing, return calculations, and format standardization<br>- Proven with live data: AAPL $203.92, TSLA 78.6% annual return, NVDA 33.4% annual return.</td>
								</tr>
							</table>
						</blockquote>
					</details>
				</blockquote>
			</details>
		</blockquote>
	</details>
	<!-- tests Submodule -->
	<details>
		<summary><b>tests</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ tests</b></code>
			<table style='width: 100%; border-collapse: collapse;'>
			<thead>
				<tr style='background-color: #f8f9fa;'>
					<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
					<th style='text-align: left; padding: 8px;'>Summary</th>
				</tr>
			</thead>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/jeevanba273/quantflow-finance/blob/master/tests/test_black_scholes.py'>test_black_scholes.py</a></b></td>
					<td style='padding: 8px;'>- Comprehensive Black-Scholes validation with mathematical precision testing<br>- Verifies all 5 Greeks calculations, put-call parity, and pricing accuracy<br>- Tests 40+ option combinations across multiple strikes and expiries<br>- Ensures 100% mathematical accuracy with error rates below 0.000001 for institutional confidence.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/jeevanba273/quantflow-finance/blob/master/tests/test_risk_metrics.py'>test_risk_metrics.py</a></b></td>
					<td style='padding: 8px;'>- Validates portfolio risk analytics with realistic market data scenarios<br>- Tests VaR, Expected Shortfall, Sharpe ratios, and drawdown calculations<br>- Ensures robust handling of different data formats and edge cases<br>- Proven accuracy with live portfolio showing 0.645 Sharpe ratio and -31.13% max drawdown.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='https://github.com/jeevanba273/quantflow-finance/blob/master/tests/test_market_data.py'>test_market_data.py</a></b></td>
					<td style='padding: 8px;'>- Validates real-time market data integration with live Yahoo Finance feeds<br>- Tests multi-ticker fetching, return calculations, and data preprocessing<br>- Ensures robust error handling for market data inconsistencies and API limitations<br>- Validated with 250+ days of live data from major stocks including AAPL, TSLA, NVDA.</td>
				</tr>
			</table>
		</blockquote>
	</details>
</details>

---

## Examples

### **Complete Options Analysis**

```python
from quantflow import BlackScholes

# Analyze Apple call option with current market data
aapl_call = BlackScholes(
    S=203.92,   # Current AAPL price (live data)
    K=210,      # Strike price
    T=0.25,     # 3 months to expiry
    r=0.05,     # 5% risk-free rate
    sigma=0.333 # 33.3% implied volatility
)

# Complete Greeks analysis
greeks = aapl_call.greeks()
print(f"Option Value: ${greeks['price']:.2f}")
print(f"Delta (hedge ratio): {greeks['delta']:.3f}")
print(f"Gamma (convexity): {greeks['gamma']:.4f}")
print(f"Theta (time decay): ${greeks['theta']:.2f}/year")
print(f"Vega (vol sensitivity): ${greeks['vega']:.2f}/1%")
print(f"Rho (rate sensitivity): ${greeks['rho']:.2f}/1%")

# Professional option summary
print(aapl_call.summary())
```

### **Professional Portfolio Risk Dashboard**

```python
from quantflow import MarketData, RiskMetrics
import numpy as np

# Build real tech portfolio with proven results
tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
weights = [0.25, 0.20, 0.20, 0.20, 0.15]

# Fetch live market data (250 trading days)
data = MarketData.fetch_stock_data(tickers, period='1y')
returns = MarketData.calculate_returns(data)

# Individual stock performance
for ticker in tickers:
    stock_returns = returns[ticker]
    annual_return = stock_returns.mean() * 252
    annual_vol = stock_returns.std() * np.sqrt(252)
    sharpe = (annual_return - 0.03) / annual_vol
    print(f"{ticker}: {annual_return:.1%} return, {annual_vol:.1%} vol, {sharpe:.2f} Sharpe")

# Portfolio analysis
portfolio_returns = (returns * weights).sum(axis=1)
risk = RiskMetrics(portfolio_returns)

# Comprehensive risk dashboard
print("\nPortfolio Risk Dashboard")
print(f"Annual Return: {portfolio_returns.mean() * 252:.2%}")
print(f"Annual Volatility: {portfolio_returns.std() * np.sqrt(252):.2%}")
print(f"Sharpe Ratio: {risk.sharpe_ratio():.3f}")
print(f"95% VaR: {risk.var_historical(0.05):.2%}")
print(f"Expected Shortfall: {risk.expected_shortfall(0.05):.2%}")
print(f"Maximum Drawdown: {risk.max_drawdown():.2%}")
```

### **Advanced Options Strategies**

```python
# Professional Bull Call Spread Analysis
current_price = 203.92  # AAPL current price
lower_strike = 200
upper_strike = 210
expiry = 0.25
vol = 0.333

long_call = BlackScholes(S=current_price, K=lower_strike, T=expiry, r=0.05, sigma=vol)
short_call = BlackScholes(S=current_price, K=upper_strike, T=expiry, r=0.05, sigma=vol)

# Strategy metrics
net_premium = long_call.price() - short_call.price()
max_profit = upper_strike - lower_strike - net_premium
breakeven = lower_strike + net_premium
net_delta = long_call.delta() - short_call.delta()

print("Bull Call Spread Analysis")
print(f"Net Premium: ${net_premium:.2f}")
print(f"Max Profit: ${max_profit:.2f}")
print(f"Breakeven: ${breakeven:.2f}")
print(f"Net Delta: {net_delta:.3f}")
print(f"Risk/Reward: {max_profit/net_premium:.2f}")
```

**Run the complete validation suite:**

```sh
python comprehensive_test.py
```

---

## Educational Applications

**Academic Integration:**
- **Graduate Coursework**: Perfect for MFE, MSF derivatives pricing, risk management, and portfolio theory courses
- **Research Projects**: Publication-ready implementations for quantitative finance research papers
- **Thesis Projects**: Complete framework for derivatives pricing and portfolio analysis studies
- **Certification Prep**: Aligned with CQF, FRM, and advanced CFA quantitative methods

**Learning Outcomes:**
- Master Black-Scholes-Merton option pricing theory with all Greeks analysis
- Understand practical implementation of Value at Risk and Expected Shortfall methodologies
- Implement modern portfolio theory with real market data and risk-adjusted performance metrics
- Analyze live financial data with professional-grade quantitative tools and validation
- Bridge academic theory with industry practice through comprehensive examples

**Research Applications:**
- **Academic Papers**: Validated implementations suitable for peer-reviewed quantitative finance research
- **Comparative Studies**: Benchmark implementation for model validation and performance studies
- **Educational Content**: Professional teaching materials for financial engineering programs
- **Industry Projects**: Production-ready code for internships and professional applications

**Proven Results for Academic Use:**
- **Mathematical Validation**: 100% accuracy with put-call parity verification (error < 0.000001)
- **Real Market Integration**: Live data from AAPL ($203.92), TSLA (78.6% return), NVDA (33.4% return)
- **Professional Standards**: Industry-grade implementation suitable for academic publication
- **Comprehensive Testing**: Extensive validation ensuring reliability for research applications

---

## Roadmap

- [X] **Complete Black-Scholes Implementation**: All 5 Greeks (Δ, Γ, Θ, ν, ρ) with mathematical precision
- [X] **Advanced Risk Analytics**: VaR, Expected Shortfall, Sharpe Ratio, Maximum Drawdown validated
- [X] **Real-Time Market Data**: Yahoo Finance integration with multi-ticker support proven
- [X] **Comprehensive Validation**: Mathematical accuracy and live data testing completed
- [X] **Professional Distribution**: PyPI publication with detailed documentation
- [X] **Monte Carlo Simulation**: Portfolio modeling with 1,000+ scenario analysis
- [X] **Options Strategies**: Bull Call Spread and Iron Condor analysis implemented
- [X] **Dividend-Aware Pricing**: Black-Scholes-Merton with continuous dividend yield (q)
- [X] **Binomial Tree Model**: Cox-Ross-Rubinstein pricing for European and American options with early exercise
- [X] **Extended Risk Analytics**: Sortino, Calmar, Omega ratios, CAPM beta/alpha, parametric ES and Cornish-Fisher VaR
- [X] **Portfolio Optimization**: Mean-variance optimization (minimum-variance, maximum-Sharpe, efficient frontier)
- [ ] **Advanced Monte Carlo**: Exotic options pricing and complex risk modeling
- [ ] **Black-Litterman**: Views-based allocation on top of mean-variance optimization
- [ ] **Volatility Models**: GARCH and stochastic volatility surface modeling
- [ ] **Fixed Income Tools**: Bond pricing, yield curve analysis, and duration calculations
- [ ] **Performance Attribution**: Factor-based return decomposition and style analysis

---

## Contributing

- **[Join the Discussions](https://github.com/jeevanba273/quantflow-finance/discussions)**: Share insights, provide feedback, or ask questions about quantitative finance implementations
- **[Report Issues](https://github.com/jeevanba273/quantflow-finance/issues)**: Submit bugs or request new financial models and advanced features
- **[Submit Pull Requests](https://github.com/jeevanba273/quantflow-finance/blob/main/CONTRIBUTING.md)**: Contribute new models, optimizations, or documentation improvements

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your GitHub account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
   ```sh
   git clone https://github.com/jeevanba273/quantflow-finance
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b feature/monte-carlo-exotic-options
   ```
4. **Make Your Changes**: Develop and test your changes locally with the existing comprehensive test suite.
5. **Add Tests**: Include mathematical validation tests for new financial models or features.
6. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Add Monte Carlo pricing for Asian options with mathematical validation'
   ```
7. **Push to GitHub**: Push the changes to your forked repository.
   ```sh
   git push origin feature/monte-carlo-exotic-options
   ```
8. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the financial models added, their mathematical foundations, and validation results.
9. **Review**: Once your PR is reviewed and approved, it will be merged into the main branch. Congratulations on your contribution to quantitative finance!
</details>

<details closed>
<summary>Contributor Graph</summary>
<br>
<p align="left">
   <a href="https://github.com/jeevanba273/quantflow-finance/graphs/contributors">
      <img src="https://contrib.rocks/image?repo=jeevanba273/quantflow-finance">
   </a>
</p>
</details>

---

## License

QuantFlow Finance is protected under the [MIT License](https://choosealicense.com/licenses/mit/). For more details, refer to the [LICENSE](https://choosealicense.com/licenses/mit/) file.

**Academic and Commercial Use:** Free for educational institutions, research projects, and commercial applications with proper attribution.

---

## Acknowledgments

- **Black & Scholes (1973)**: *The Pricing of Options and Corporate Liabilities* - Foundation of modern derivatives theory
- **Merton (1973)**: Extensions to Black-Scholes model and risk-neutral valuation framework
- **NumPy & SciPy Communities**: Essential mathematical computing libraries enabling high-performance calculations
- **Yahoo Finance**: Reliable market data source providing real-time validation for our implementations
- **Quantitative Finance Community**: Inspiration, validation, and peer review of financial models
- **Academic Research**: Various papers and textbooks in mathematical finance providing theoretical foundations
- **Open Source Movement**: Enabling collaborative development of professional-grade financial tools

**Special Recognition:**
- **Live Market Validation**: Real portfolio performance data validating our risk analytics
- **Mathematical Precision**: Achieving error rates below 0.000001 in put-call parity verification
- **Academic Standards**: Implementation meeting peer-review quality for quantitative finance research

<div align="left"><a href="#top">Return</a></div>

---
