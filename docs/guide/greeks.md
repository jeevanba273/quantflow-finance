# Greeks

{meth}`BlackScholes.greeks() <quantflow.BlackScholes.greeks>` and
{func}`~quantflow.black_scholes_greeks` return the price with delta, gamma,
theta, vega and rho from their closed forms.

## Units

| Greek | Returned value | To get the raw derivative |
| --- | --- | --- |
| delta | $\partial V / \partial S$ | as is |
| gamma | $\partial^2 V / \partial S^2$ | as is |
| theta | $-\partial V / \partial T$, per year | as is; divide by 365 for one calendar day |
| vega | change for +1 volatility point | multiply by 100 |
| rho | change for +1 rate point | multiply by 100 |

QuantLib and most textbooks quote vega and rho per unit (multiply quantflow's
values by 100 to compare); vollib quotes them per point, like quantflow, and
quotes theta per day.

```python
from quantflow import BlackScholes

option = BlackScholes(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type="call")
for name, value in option.greeks().items():
    print(f"{name:>5} {value:10.6f}")
```

```text
price  10.450584
delta   0.636831
gamma   0.018762
theta  -6.414028
 vega   0.375240
  rho   0.532325
```

Read this as: the call gains about 0.64 per unit rise in spot, loses about
6.41 per year of time decay with nothing else moving, gains about 0.38 if
volatility goes from 20% to 21%, and about 0.53 if the rate goes from 5% to
6%.

## Checking a Greek by finite differences

Each Greek is the derivative of the price, so a central difference of
{func}`~quantflow.black_scholes_price` should agree with it. The test suite
does this for every Greek on 72 parameter sets, calls and puts, with and
without a dividend yield.

```python
from quantflow import BlackScholes, black_scholes_price

S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
h = 1e-5
fd_vega = (black_scholes_price(S, K, T, r, sigma + h) -
           black_scholes_price(S, K, T, r, sigma - h)) / (2 * h) / 100
fd_theta = -(black_scholes_price(S, K, T + h, r, sigma) -
             black_scholes_price(S, K, T - h, r, sigma)) / (2 * h)
option = BlackScholes(S, K, T, r, sigma)
print(f"vega  {option.vega():.8f}  finite difference {fd_vega:.8f}")
print(f"theta {option.theta():.8f}  finite difference {fd_theta:.8f}")
```

```text
vega  0.37524035  finite difference 0.37524035
theta -6.41402755  finite difference -6.41402755
```

## Greeks of a position

Greeks add across positions with the same underlying. A bull call spread,
long the 100 call and short the 110 call:

```python
from quantflow import black_scholes_greeks

g = black_scholes_greeks(100, [100.0, 110.0], 0.5, 0.04, 0.25, "call")
weights = [1, -1]
for name in ("price", "delta", "gamma", "vega"):
    print(f"{name:>5} {sum(w * x for w, x in zip(weights, g[name])):8.4f}")
```

```text
price   3.9481
delta   0.2120
gamma   0.0008
 vega   0.0100
```

## Tree and Monte Carlo Greeks

{class}`~quantflow.BinomialTree` also has `delta`, `gamma`, `theta`, `vega` and
`rho` in the same units: the first three come from the first two layers of
the tree, vega and rho from repricing with bumped inputs. For European
options they approach the analytic values as the number of steps grows; the
tests require agreement within 0.2% for an at-the-money call and put at 2000
steps. {class}`~quantflow.MonteCarlo` has no Greeks.
