"""
Complete Portfolio Analysis Example - QuantFlow Package

This example demonstrates the full power of QuantFlow by:
1. Fetching real market data
2. Calculating portfolio risk metrics
3. Pricing options on the portfolio
"""

import sys
import os

# Add the src folder to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from quantflow.data.fetcher import MarketData
from quantflow.risk.metrics import RiskMetrics
from quantflow.options.black_scholes import BlackScholes
import pandas as pd
import numpy as np


def main():
    print("=" * 60)
    print("QuantFlow Complete Portfolio Analysis")
    print("=" * 60)

    # Step 1: Fetch real market data for a tech portfolio
    print("\nStep 1: Fetching Market Data")
    tickers = ["AAPL", "GOOGL", "MSFT"]
    portfolio_weights = [0.4, 0.3, 0.3]

    print(f"Portfolio: {tickers}")
    print(f"Weights: {portfolio_weights}")

    # Fetch 1 year of data
    prices = MarketData.fetch_stock_data(tickers, period="1y")
    returns = MarketData.calculate_returns(prices)

    print(f"Data period: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"Number of trading days: {len(prices)}")

    # Step 2: Calculate portfolio returns
    print("\nStep 2: Portfolio Construction")
    portfolio_returns = (returns * portfolio_weights).sum(axis=1)

    print(
        f"Portfolio daily return: {portfolio_returns.mean():.4f} ({portfolio_returns.mean()*100:.2f}%)"
    )
    print(
        f"Portfolio daily volatility: {portfolio_returns.std():.4f} ({portfolio_returns.std()*100:.2f}%)"
    )

    # Step 3: Risk Analysis
    print("\n Step 3: Risk Analysis")
    risk_metrics = RiskMetrics(portfolio_returns)

    var_5 = risk_metrics.var_historical(0.05)
    es_5 = risk_metrics.expected_shortfall(0.05)
    sharpe = risk_metrics.sharpe_ratio()
    max_dd = risk_metrics.max_drawdown()

    print(f"5% Value at Risk: {var_5:.4f} ({var_5*100:.2f}%)")
    print(f"5% Expected Shortfall: {es_5:.4f} ({es_5*100:.2f}%)")
    print(f"Sharpe Ratio: {sharpe:.3f}")
    print(f"Maximum Drawdown: {max_dd:.4f} ({max_dd*100:.2f}%)")

    # Step 4: Options Analysis
    print("\nStep 4: Options Analysis")
    # Use AAPL as example for options pricing
    current_price = prices["AAPL"].iloc[-1]
    annual_vol = returns["AAPL"].std() * np.sqrt(252)

    print(f"AAPL Current Price: ${current_price:.2f}")
    print(f"AAPL Annual Volatility: {annual_vol:.2%}")

    # Price call options at different strikes
    strikes = [current_price * 0.95, current_price, current_price * 1.05]

    print(f"\nOption Prices (3-month expiry, 5% risk-free rate):")
    print("-" * 50)

    for strike in strikes:
        option = BlackScholes(
            S=current_price, K=strike, T=0.25, r=0.05, sigma=annual_vol  # 3 months
        )

        greeks = option.greeks()
        moneyness = (
            "ITM"
            if strike < current_price
            else "ATM" if abs(strike - current_price) < 1 else "OTM"
        )

        print(f"Strike ${strike:.0f} ({moneyness}):")
        print(f"  Price: ${greeks['price']:.2f}")
        print(f"  Delta: {greeks['delta']:.3f}")
        print(f"  Gamma: {greeks['gamma']:.4f}")
        print(f"  Theta: ${greeks['theta']:.2f}/year")
        print(f"  Vega: ${greeks['vega']:.2f}")
        print()

    print("=" * 60)
    print("Analysis Complete!")
    print("\nThis demonstrates QuantFlow's capabilities:")
    print("Real market data integration")
    print("Professional risk analytics")
    print("Advanced options pricing")
    print("Complete quantitative workflow")


if __name__ == "__main__":
    main()
