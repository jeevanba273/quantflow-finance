"""
Basic Option Pricing Example - QuantFlow Package Demo

This example shows how to use the QuantFlow package to price options
and calculate risk metrics (Greeks).
"""

import sys
import os

# Add the src folder to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from quantflow.options.black_scholes import BlackScholes


def main():
    print("=" * 50)
    print("QuantFlow Package - Option Pricing Demo")
    print("=" * 50)

    # Example: Apple stock option
    print("\nAAPL Call Option Example:")
    print("Current Stock Price: $150")
    print("Strike Price: $155")
    print("Time to Expiry: 3 months (0.25 years)")
    print("Risk-free Rate: 5%")
    print("Volatility: 25%")

    # Create the option
    aapl_call = BlackScholes(
        S=150,  # Current stock price
        K=155,  # Strike price
        T=0.25,  # 3 months
        r=0.05,  # 5% risk-free rate
        sigma=0.25,  # 25% volatility
        option_type="call",
    )

    # Calculate price and delta
    price = aapl_call.price()
    delta = aapl_call.delta()

    print(f"\nResults:")
    print(f"Option Price: ${price:.2f}")
    print(f"Delta: {delta:.3f}")
    print(f"\nInterpretation:")
    print(f"• For every $1 increase in AAPL stock, this option gains ~${delta:.2f}")
    print(f"• This option is worth ${price:.2f} according to Black-Scholes model")


if __name__ == "__main__":
    main()
