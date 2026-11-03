"""Black-Scholes-Merton prices, Greeks, put-call parity and implied volatility.

Run with:  python examples/option_pricing.py
"""

import math

import numpy as np

from quantflow import BlackScholes, black_scholes_greeks, black_scholes_price

# Hull's textbook example: S=42, K=40, r=10%, sigma=20%, six months.
call = BlackScholes(S=42, K=40, T=0.5, r=0.10, sigma=0.20, option_type="call")
put = BlackScholes(S=42, K=40, T=0.5, r=0.10, sigma=0.20, option_type="put")
print(f"call {call.price():.4f}  put {put.price():.4f}")

parity_gap = (call.price() - put.price()) - (42 - 40 * math.exp(-0.10 * 0.5))
print(f"put-call parity gap {parity_gap:.1e}")

print("Greeks of the call (theta per year, vega and rho per 1%):")
for name, value in call.greeks().items():
    print(f"  {name:>5}  {value:9.4f}")

# Recover the volatility from the price.
iv = call.implied_volatility(call.price())
print(f"implied volatility {iv:.10f}")

# A strip of strikes in one vectorised call.
strikes = np.array([35.0, 40.0, 45.0, 50.0])
prices = black_scholes_price(42, strikes, 0.5, 0.10, 0.20, "call")
deltas = black_scholes_greeks(42, strikes, 0.5, 0.10, 0.20, "call")["delta"]
for k, p, d in zip(strikes, prices, deltas, strict=True):
    print(f"K={k:5.1f}  price {p:7.4f}  delta {d:6.4f}")
