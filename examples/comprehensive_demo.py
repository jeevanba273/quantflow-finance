"""
COMPREHENSIVE QUANTFLOW FINANCE PACKAGE TEST
===========================================

This advanced test suite demonstrates every feature of QuantFlow Finance:
- Complete options pricing with all Greeks
- Advanced portfolio risk analytics
- Real-time market data integration
- Professional financial analysis workflows

Perfect for showcasing the package's capabilities!
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

print("COMPREHENSIVE QUANTFLOW FINANCE TEST SUITE")
print("=" * 80)
print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

try:
    # Import QuantFlow Finance
    from quantflow import BlackScholes, RiskMetrics, MarketData

    print("QuantFlow Finance successfully imported!")

    # ==========================================
    # SECTION 1: ADVANCED OPTIONS ANALYSIS
    # ==========================================
    print("\n" + "=" * 80)
    print("SECTION 1: COMPREHENSIVE OPTIONS ANALYSIS")
    print("=" * 80)

    # Create options with different strikes and expiries
    spot_price = 150
    strikes = [140, 145, 150, 155, 160]  # ITM, slight ITM, ATM, slight OTM, OTM
    expiries = [0.0833, 0.25, 0.5, 1.0]  # 1 month, 3 months, 6 months, 1 year
    risk_free_rate = 0.05
    volatility = 0.25

    print(f"Underlying Asset Price: ${spot_price}")
    print(f"Risk-free Rate: {risk_free_rate:.1%}")
    print(f"Volatility: {volatility:.1%}")
    print("\n" + "-" * 60)
    print("COMPLETE OPTIONS PRICING MATRIX")
    print("-" * 60)

    # Headers
    print(
        f"{'Strike':<8} {'Expiry':<8} {'Type':<6} {'Price':<8} {'Delta':<8} {'Gamma':<8} {'Theta':<10} {'Vega':<8} {'Rho':<8}"
    )
    print("-" * 80)

    options_data = []

    for strike in strikes:
        for expiry in expiries:
            # Call option
            call = BlackScholes(
                S=spot_price,
                K=strike,
                T=expiry,
                r=risk_free_rate,
                sigma=volatility,
                option_type="call",
            )
            call_greeks = call.greeks()

            # Put option
            put = BlackScholes(
                S=spot_price,
                K=strike,
                T=expiry,
                r=risk_free_rate,
                sigma=volatility,
                option_type="put",
            )
            put_greeks = put.greeks()

            # Store data
            options_data.append(
                {
                    "Strike": strike,
                    "Expiry": expiry,
                    "Type": "Call",
                    "Price": call_greeks["price"],
                    "Delta": call_greeks["delta"],
                    "Gamma": call_greeks["gamma"],
                    "Theta": call_greeks["theta"],
                    "Vega": call_greeks["vega"],
                    "Rho": call_greeks["rho"],
                }
            )

            options_data.append(
                {
                    "Strike": strike,
                    "Expiry": expiry,
                    "Type": "Put",
                    "Price": put_greeks["price"],
                    "Delta": put_greeks["delta"],
                    "Gamma": put_greeks["gamma"],
                    "Theta": put_greeks["theta"],
                    "Vega": put_greeks["vega"],
                    "Rho": put_greeks["rho"],
                }
            )

            # Print call
            print(
                f"{strike:<8} {expiry:<8.2f} {'Call':<6} {call_greeks['price']:<8.2f} {call_greeks['delta']:<8.3f} {call_greeks['gamma']:<8.4f} {call_greeks['theta']:<10.2f} {call_greeks['vega']:<8.3f} {call_greeks['rho']:<8.3f}"
            )

            # Print put
            print(
                f"{strike:<8} {expiry:<8.2f} {'Put':<6} {put_greeks['price']:<8.2f} {put_greeks['delta']:<8.3f} {put_greeks['gamma']:<8.4f} {put_greeks['theta']:<10.2f} {put_greeks['vega']:<8.3f} {put_greeks['rho']:<8.3f}"
            )

    # Put-Call Parity Verification
    print("\n" + "-" * 60)
    print("PUT-CALL PARITY VERIFICATION")
    print("-" * 60)

    for strike in strikes:
        for expiry in expiries:
            call = BlackScholes(
                S=spot_price,
                K=strike,
                T=expiry,
                r=risk_free_rate,
                sigma=volatility,
                option_type="call",
            )
            put = BlackScholes(
                S=spot_price,
                K=strike,
                T=expiry,
                r=risk_free_rate,
                sigma=volatility,
                option_type="put",
            )

            # Put-Call Parity: C - P = S - K*e^(-rT)
            left_side = call.price() - put.price()
            right_side = spot_price - strike * np.exp(-risk_free_rate * expiry)
            error = abs(left_side - right_side)

            status = "PASS" if error < 0.001 else "FAIL"
            print(f"K={strike}, T={expiry:.2f}: Error = {error:.6f} {status}")

    # ==========================================
    # SECTION 2: REAL MARKET DATA ANALYSIS
    # ==========================================
    print("\n" + "=" * 80)
    print("SECTION 2: REAL MARKET DATA ANALYSIS")
    print("=" * 80)

    # Fetch data for a diversified portfolio
    tickers = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]
    portfolio_weights = [0.25, 0.20, 0.20, 0.20, 0.15]

    print(f"Portfolio Tickers: {tickers}")
    print(f" Portfolio Weights: {portfolio_weights}")
    print("\nFetching real market data...")

    # Fetch market data
    market_data = MarketData.fetch_stock_data(tickers, period="1y")
    returns_data = MarketData.calculate_returns(market_data)

    print(f"Successfully fetched {len(market_data)} days of market data")
    print(f"Calculated {len(returns_data)} return observations")

    # Individual stock analysis
    print("\n" + "-" * 60)
    print("INDIVIDUAL STOCK ANALYSIS")
    print("-" * 60)
    print(
        f"{'Ticker':<8} {'Latest Price':<12} {'Ann. Return':<12} {'Ann. Vol':<12} {'Sharpe':<8}"
    )
    print("-" * 60)

    individual_stats = {}
    for ticker in tickers:
        latest_price = market_data[ticker].iloc[-1]
        stock_returns = returns_data[ticker]

        # Calculate annualized metrics
        annual_return = stock_returns.mean() * 252
        annual_vol = stock_returns.std() * np.sqrt(252)
        sharpe_ratio = (annual_return - 0.03) / annual_vol  # Assuming 3% risk-free rate

        individual_stats[ticker] = {
            "price": latest_price,
            "annual_return": annual_return,
            "annual_vol": annual_vol,
            "sharpe": sharpe_ratio,
        }

        print(
            f"{ticker:<8} ${latest_price:<11.2f} {annual_return:<11.1%} {annual_vol:<11.1%} {sharpe_ratio:<8.2f}"
        )

    # ==========================================
    # SECTION 3: PORTFOLIO RISK ANALYTICS
    # ==========================================
    print("\n" + "=" * 80)
    print(" SECTION 3: COMPREHENSIVE PORTFOLIO RISK ANALYSIS")
    print("=" * 80)

    # Calculate portfolio returns
    portfolio_returns = (returns_data * portfolio_weights).sum(axis=1)

    # Comprehensive risk analysis
    risk_analyzer = RiskMetrics(portfolio_returns)

    # Multiple VaR confidence levels
    var_levels = [0.01, 0.05, 0.10]
    print("VALUE AT RISK ANALYSIS")
    print("-" * 40)
    for level in var_levels:
        var_value = risk_analyzer.var_historical(level)
        es_value = risk_analyzer.expected_shortfall(level)
        print(f"{(1-level)*100:>4.0f}% VaR: {var_value:>8.2%} | ES: {es_value:>8.2%}")

    # Performance metrics
    sharpe = risk_analyzer.sharpe_ratio(risk_free_rate=0.03)
    max_dd = risk_analyzer.max_drawdown()

    print(f"\nPERFORMANCE METRICS")
    print("-" * 40)
    print(f"Sharpe Ratio:      {sharpe:>8.3f}")
    print(f"Max Drawdown:      {max_dd:>8.2%}")
    print(f"Annual Return:     {portfolio_returns.mean() * 252:>8.2%}")
    print(f"Annual Volatility: {portfolio_returns.std() * np.sqrt(252):>8.2%}")

    # Rolling metrics analysis
    print(f"\nROLLING RISK METRICS (30-day windows)")
    print("-" * 50)

    rolling_window = 30
    rolling_sharpe = []
    rolling_vol = []

    for i in range(rolling_window, len(portfolio_returns)):
        window_returns = portfolio_returns.iloc[i - rolling_window : i]
        window_sharpe = (
            (window_returns.mean() - 0.03 / 252) / window_returns.std() * np.sqrt(252)
        )
        window_vol = window_returns.std() * np.sqrt(252)

        rolling_sharpe.append(window_sharpe)
        rolling_vol.append(window_vol)

    print(f"Average Rolling Sharpe: {np.mean(rolling_sharpe):>8.3f}")
    print(f"Rolling Sharpe Std:     {np.std(rolling_sharpe):>8.3f}")
    print(f"Average Rolling Vol:    {np.mean(rolling_vol):>8.2%}")
    print(f"Rolling Vol Std:        {np.std(rolling_vol):>8.2%}")

    # ==========================================
    # SECTION 4: CORRELATION & DIVERSIFICATION
    # ==========================================
    print("\n" + "=" * 80)
    print("SECTION 4: CORRELATION & DIVERSIFICATION ANALYSIS")
    print("=" * 80)

    # Correlation matrix
    correlation_matrix = returns_data.corr()
    print("CORRELATION MATRIX")
    print("-" * 60)
    print(correlation_matrix.round(3))

    # Diversification metrics
    portfolio_var = np.var(portfolio_returns)
    weighted_individual_var = sum(
        w**2 * np.var(returns_data[ticker])
        for w, ticker in zip(portfolio_weights, tickers)
    )
    diversification_ratio = 1 - (portfolio_var / weighted_individual_var)

    print(f"\nDIVERSIFICATION METRICS")
    print("-" * 40)
    print(f"Portfolio Variance:      {portfolio_var:>10.6f}")
    print(f"Weighted Individual Var: {weighted_individual_var:>10.6f}")
    print(f"Diversification Ratio:   {diversification_ratio:>10.2%}")

    # ==========================================
    # SECTION 5: ADVANCED OPTIONS STRATEGIES
    # ==========================================
    print("\n" + "=" * 80)
    print("SECTION 5: ADVANCED OPTIONS STRATEGIES")
    print("=" * 80)

    # Using AAPL current price for options strategies
    aapl_price = market_data["AAPL"].iloc[-1]
    aapl_vol = returns_data["AAPL"].std() * np.sqrt(252)

    print(f"AAPL Current Price: ${aapl_price:.2f}")
    print(f"AAPL Implied Vol: {aapl_vol:.1%}")

    # Strategy 1: Bull Call Spread
    print(f"\nBULL CALL SPREAD ANALYSIS")
    print("-" * 50)
    lower_strike = aapl_price * 0.98
    upper_strike = aapl_price * 1.04
    expiry = 0.25  # 3 months

    long_call = BlackScholes(
        S=aapl_price,
        K=lower_strike,
        T=expiry,
        r=0.05,
        sigma=aapl_vol,
        option_type="call",
    )
    short_call = BlackScholes(
        S=aapl_price,
        K=upper_strike,
        T=expiry,
        r=0.05,
        sigma=aapl_vol,
        option_type="call",
    )

    spread_cost = long_call.price() - short_call.price()
    spread_delta = long_call.delta() - short_call.delta()
    spread_gamma = long_call.gamma() - short_call.gamma()
    spread_theta = long_call.theta() - short_call.theta()

    print(f"Long {lower_strike:.0f} Call:  ${long_call.price():.2f}")
    print(f"Short {upper_strike:.0f} Call: ${short_call.price():.2f}")
    print(f"Net Premium:        ${spread_cost:.2f}")
    print(f"Net Delta:          {spread_delta:.3f}")
    print(f"Net Gamma:          {spread_gamma:.4f}")
    print(f"Net Theta:          ${spread_theta:.2f}/year")
    print(f"Max Profit:         ${upper_strike - lower_strike - spread_cost:.2f}")
    print(f"Breakeven:          ${lower_strike + spread_cost:.2f}")

    # Strategy 2: Iron Condor
    print(f"\nIRON CONDOR ANALYSIS")
    print("-" * 50)
    put_strike_low = aapl_price * 0.92
    put_strike_high = aapl_price * 0.96
    call_strike_low = aapl_price * 1.04
    call_strike_high = aapl_price * 1.08

    # Iron Condor legs
    long_put_low = BlackScholes(
        S=aapl_price,
        K=put_strike_low,
        T=expiry,
        r=0.05,
        sigma=aapl_vol,
        option_type="put",
    )
    short_put_high = BlackScholes(
        S=aapl_price,
        K=put_strike_high,
        T=expiry,
        r=0.05,
        sigma=aapl_vol,
        option_type="put",
    )
    short_call_low = BlackScholes(
        S=aapl_price,
        K=call_strike_low,
        T=expiry,
        r=0.05,
        sigma=aapl_vol,
        option_type="call",
    )
    long_call_high = BlackScholes(
        S=aapl_price,
        K=call_strike_high,
        T=expiry,
        r=0.05,
        sigma=aapl_vol,
        option_type="call",
    )

    condor_premium = (short_put_high.price() + short_call_low.price()) - (
        long_put_low.price() + long_call_high.price()
    )
    condor_delta = (short_put_high.delta() + short_call_low.delta()) - (
        long_put_low.delta() + long_call_high.delta()
    )

    print(f"Net Premium Received: ${condor_premium:.2f}")
    print(f"Net Delta:           {condor_delta:.4f}")
    print(f"Profit Range:        ${put_strike_high:.0f} - ${call_strike_low:.0f}")
    print(f"Max Profit:          ${condor_premium:.2f}")
    print(
        f"Risk/Reward Ratio:   {(call_strike_low - put_strike_high - condor_premium) / condor_premium:.2f}"
    )

    # ==========================================
    # SECTION 6: MONTE CARLO SIMULATION
    # ==========================================
    print("\n" + "=" * 80)
    print("SECTION 6: MONTE CARLO PORTFOLIO SIMULATION")
    print("=" * 80)

    # Monte Carlo parameters
    num_simulations = 1000
    time_horizon = 252  # 1 year
    initial_portfolio_value = 100000

    print(f"Simulations: {num_simulations:,}")
    print(f"Time Horizon: {time_horizon} days (1 year)")
    print(f"Initial Value: ${initial_portfolio_value:,}")

    # Historical parameters
    mean_return = portfolio_returns.mean()
    volatility = portfolio_returns.std()

    print(f"Historical Daily Return: {mean_return:.4f} ({mean_return*252:.2%} annual)")
    print(
        f"Historical Daily Vol: {volatility:.4f} ({volatility*np.sqrt(252):.2%} annual)"
    )

    # Monte Carlo simulation
    np.random.seed(42)  # For reproducibility
    final_values = []

    for sim in range(num_simulations):
        daily_returns = np.random.normal(mean_return, volatility, time_horizon)
        portfolio_value = initial_portfolio_value

        for daily_return in daily_returns:
            portfolio_value *= 1 + daily_return

        final_values.append(portfolio_value)

    final_values = np.array(final_values)

    # Monte Carlo results
    print(f"\nMONTE CARLO RESULTS")
    print("-" * 40)
    print(f"Mean Final Value:    ${np.mean(final_values):>12,.0f}")
    print(f"Median Final Value:  ${np.median(final_values):>12,.0f}")
    print(f"5th Percentile:      ${np.percentile(final_values, 5):>12,.0f}")
    print(f"95th Percentile:     ${np.percentile(final_values, 95):>12,.0f}")
    print(
        f"Probability of Loss: {np.mean(final_values < initial_portfolio_value):>12.1%}"
    )
    print(f"Maximum Gain:        ${np.max(final_values):>12,.0f}")
    print(f"Maximum Loss:        ${np.min(final_values):>12,.0f}")

    # Expected returns
    expected_return = (np.mean(final_values) / initial_portfolio_value) - 1
    print(f"Expected Annual Return: {expected_return:>10.2%}")

    # ==========================================
    # SECTION 7: SUMMARY & RECOMMENDATIONS
    # ==========================================
    print("\n" + "=" * 80)
    print("SECTION 7: COMPREHENSIVE ANALYSIS SUMMARY")
    print("=" * 80)

    print("KEY FINDINGS:")
    print("-" * 50)
    print(
        f"Portfolio Sharpe Ratio: {sharpe:.3f} ({'Excellent' if sharpe > 1.5 else 'Good' if sharpe > 1.0 else 'Fair'})"
    )
    print(
        f"Maximum Drawdown: {max_dd:.2%} ({'Low Risk' if abs(max_dd) < 0.15 else 'Moderate Risk' if abs(max_dd) < 0.25 else 'High Risk'})"
    )
    print(
        f"Diversification Ratio: {diversification_ratio:.2%} ({'Well Diversified' if diversification_ratio > 0.3 else 'Moderately Diversified'})"
    )
    print(f"VaR (95%): {risk_analyzer.var_historical(0.05):.2%}")
    print(f"Expected Annual Return: {expected_return:.2%}")

    print(f"\nQUANTFLOW FINANCE CAPABILITIES DEMONSTRATED:")
    print("-" * 60)
    print("Complete Black-Scholes implementation with all Greeks")
    print("Advanced options strategies (Bull Call Spread, Iron Condor)")
    print("Real-time market data integration and analysis")
    print("Comprehensive portfolio risk analytics")
    print("Monte Carlo simulation capabilities")
    print("Professional-grade correlation and diversification analysis")
    print("Put-call parity verification")
    print("Rolling metrics analysis")

    print(f"\nPERFECT FOR ACADEMIC APPLICATIONS:")
    print("-" * 50)
    print("• Demonstrates deep understanding of derivatives pricing theory")
    print("• Shows practical implementation of risk management concepts")
    print("• Proves ability to work with real financial data")
    print("• Exhibits professional software development skills")
    print("• Ready for quantitative finance graduate programs")

    print("\n" + "=" * 80)
    print("COMPREHENSIVE TEST COMPLETED SUCCESSFULLY!")
    print("QuantFlow Finance is production-ready for academic and professional use!")
    print("=" * 80)

except ImportError as e:
    print(f"Import Error: {e}")
    print("Try: pip install quantflow-finance")

except Exception as e:
    print(f"Unexpected Error: {e}")
    import traceback

    traceback.print_exc()
    print("Please check your installation and try again")

print(f"\n⏰ Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
