"""Download index ETF prices and compute risk metrics.

Needs network access and the optional dependency:
    pip install "quantflow-finance[data]"
Run with:  python examples/market_data.py

The output depends on the day it runs. It is not part of the test suite.
"""

from quantflow import MarketData, Portfolio, RiskMetrics

prices = MarketData.fetch_stock_data(["SPY", "TLT", "GLD"], period="5y")
returns = MarketData.calculate_returns(prices)
print(f"{len(returns)} daily returns from {returns.index[0].date()}")

for ticker in returns.columns:
    rm = RiskMetrics(returns[ticker])
    print(
        f"{ticker}: vol {rm.volatility():.3f}  95% VaR {rm.var_historical(0.05):.4f}"
        f"  max drawdown {rm.max_drawdown():.3f}"
    )

print(Portfolio(returns).correlation_matrix().round(2))
