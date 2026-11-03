# Black-Scholes-Merton pricing

{class}`~quantflow.BlackScholes` prices one European option;
{func}`~quantflow.black_scholes_price` prices many at once from NumPy arrays.
Both use the same code.

## One option

```python
from quantflow import BlackScholes

option = BlackScholes(S=100, K=105, T=0.25, r=0.05, sigma=0.20, option_type="call")
print(option)
print(f"price {option.price():.6f}")
```

```text
BlackScholes(Call: S=100.0, K=105.0, T=0.250, r=0.050, sigma=0.200, q=0.000)
price 2.477902
```

`S` and `K` are in currency units, `T` is a year fraction, `r` is the
continuously compounded rate and `sigma` the annualised volatility. The
constructor rejects non-positive `S`, `K`, `T` or `sigma`, non-finite values
and arrays, with a `ValueError` that names the argument.

## Dividends, FX and futures

The continuous yield `q` covers three common cases (Haug 2007, chapter 1):

- an equity or index with dividend yield `q` (Merton 1973);
- an FX option, with `q` set to the foreign interest rate
  (Garman-Kohlhagen 1983);
- an option on a future, with `S` the futures price and `q = r`
  (Black 1976).

```python
from quantflow import BlackScholes

index_put = BlackScholes(100, 95, 0.5, 0.10, 0.20, "put", q=0.05)
fx_call = BlackScholes(1.56, 1.60, 0.5, 0.06, 0.12, "call", q=0.08)
futures_call = BlackScholes(19, 19, 0.75, 0.10, 0.28, "call", q=0.10)
for label, opt in [("index put", index_put), ("FX call", fx_call),
                   ("futures call", futures_call)]:
    print(f"{label:<13} {opt.price():.4f}")
```

```text
index put     2.4648
FX call       0.0291
futures call  1.7011
```

These are worked examples from Haug's book, which gives 2.4648, 0.0291 and
1.7011. Discrete cash dividends are not modelled; approximating them by a
yield is the user's choice.

## Put-call parity

For European options $C - P = S e^{-qT} - K e^{-rT}$ for any model. The test
suite checks it to $10^{-10}$ on a grid of 648 parameter sets.

```python
import math

from quantflow import black_scholes_price

S, K, T, r, sigma, q = 100, 110, 2.0, 0.03, 0.35, 0.01
c = black_scholes_price(S, K, T, r, sigma, "call", q)
p = black_scholes_price(S, K, T, r, sigma, "put", q)
print(f"{c - p:.12f}")
print(f"{S * math.exp(-q * T) - K * math.exp(-r * T):.12f}")
```

```text
-5.574231363592
-5.574231363592
```

## Many options at once

{func}`~quantflow.black_scholes_price` broadcasts its arguments with NumPy's
rules, so strikes along one axis and expiries along another give a price
grid. `option_type` can also be an array.

```python
import numpy as np

from quantflow import black_scholes_price

strikes = np.array([90.0, 100.0, 110.0])
expiries = np.array([[0.25], [1.0]])
print(black_scholes_price(100, strikes, expiries, 0.05, 0.2, "call").round(4))
print(black_scholes_price(100, 100, 1.0, 0.05, 0.2, ["call", "put"]).round(4))
```

```text
[[11.6701  4.615   1.1911]
 [16.6994 10.4506  6.0401]]
[10.4506  5.5735]
```

All inputs are validated before pricing; one bad element raises a
`ValueError` for the whole call. The {doc}`../benchmarks` page has throughput
numbers.

## Limits of the model

The price assumes constant volatility, rates and yield, lognormal prices and
no transaction costs. Market option prices show a volatility smile, so a
single `sigma` will not reproduce a whole chain; the usual practice is to
quote each option by its own implied volatility (see
{doc}`implied-volatility`). {doc}`../methods` lists the assumptions in full.
