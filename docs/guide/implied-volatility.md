# Implied volatility

{meth}`BlackScholes.implied_volatility() <quantflow_finance.BlackScholes.implied_volatility>`
finds the volatility at which the Black-Scholes-Merton price equals a given
price. The `sigma` the object was built with is ignored.

```python
from quantflow_finance import BlackScholes

option = BlackScholes(S=100, K=110, T=0.5, r=0.03, sigma=0.30, option_type="call", q=0.01)
iv = option.implied_volatility(4.25)
print(f"implied vol {iv:.10f}")
print(f"repriced    {BlackScholes(100, 110, 0.5, 0.03, iv, 'call', q=0.01).price():.10f}")
```

```text
implied vol 0.2701228772
repriced    4.2500000000
```

## Method

The price is strictly increasing in volatility, so a solution is unique when
it exists. The method first checks the no-arbitrage bounds, for a call
$\max(S e^{-qT} - K e^{-rT}, 0) < C < S e^{-qT}$ and for a put
$\max(K e^{-rT} - S e^{-qT}, 0) < P < K e^{-rT}$. It then runs Brent's method
on $[10^{-6}, 10]$ (the upper end widens up to 1000 if needed) with an
absolute volatility tolerance of $10^{-10}$.

## When there is no answer

Each of these raises a `ValueError` with a message saying which case it is:

- the price is zero, negative or not finite;
- the price is at or above the upper bound;
- the price is below the lower bound;
- the price has no time value above the lower bound in double precision.

The last case matters for deep in-the-money options close to expiry: their
model price can equal the discounted intrinsic value to every digit for a
wide range of volatilities, so no volatility can be recovered.

```python
from quantflow_finance import BlackScholes

deep_itm = BlackScholes(S=100, K=50, T=1 / 365, r=0.03, sigma=0.20, option_type="call")
try:
    deep_itm.implied_volatility(deep_itm.price())
except ValueError as err:
    print(err)
```

```text
market_price 50.004109 has no time value above the lower bound 50.004109; the implied volatility is not determined
```

## Accuracy

The recovered volatility is limited by the conditioning of the problem: its
error is roughly the rounding error of the price divided by vega. When vega
is tiny (far from the money, very short expiry) the volatility can be wrong in
the fifth decimal even though it reproduces the price exactly. The
{doc}`../validation` page compares the solver with vollib (Jaeckel's "Let's
Be Rational") and QuantLib on 5,186 options; the largest volatility error was
$3.4 \times 10^{-11}$, against $1.7 \times 10^{-11}$ for vollib and
$1.9 \times 10^{-11}$ for QuantLib. Brent's method needs more price
evaluations than Let's Be Rational, which is designed to converge in two
iterations; the {doc}`../benchmarks` page shows QuantFlow Finance's solver is slower
than both libraries.

There is no vectorised implied-volatility function; solve options one at a
time.
