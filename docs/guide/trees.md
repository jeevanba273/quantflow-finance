# Binomial trees

{class}`~quantflow.BinomialTree` builds a Cox-Ross-Rubinstein (CRR) tree.
It prices European and American calls and puts with a continuous dividend
yield.

## American options

Hull's example: an American put with spot and strike 50, five months, 10%
rate and 40% volatility. Hull reports 4.49 for a 5-step tree and 4.263,
4.272, 4.278 and 4.283 for 30, 50, 100 and 500 steps.

```python
from quantflow import BinomialTree, BlackScholes

for steps in (5, 30, 50, 100, 500):
    tree = BinomialTree(50, 50, 5 / 12, 0.10, 0.40, "put", "american", steps=steps)
    print(f"steps {steps:>3}  {tree.price():.4f}")
european = BlackScholes(50, 50, 5 / 12, 0.10, 0.40, "put").price()
print(f"European (Black-Scholes) {european:.4f}")
```

```text
steps   5  4.4885
steps  30  4.2634
steps  50  4.2720
steps 100  4.2781
steps 500  4.2830
European (Black-Scholes) 4.0760
```

The difference between the American and European prices is the
early-exercise premium:

```python
from quantflow import BinomialTree

put = BinomialTree(50, 50, 5 / 12, 0.10, 0.40, "put", "american", steps=500)
print(f"{put.early_exercise_premium():.4f}")
```

```text
0.2096
```

An American call on a stock without dividends is never exercised early
(Merton 1973), so its tree price equals the European one. With a dividend
yield the premium can be positive:

```python
from quantflow import BinomialTree

for q in (0.0, 0.08):
    call = BinomialTree(100, 100, 1.0, 0.05, 0.2, "call", "american", steps=500, q=q)
    print(f"q={q:.2f}  premium {call.early_exercise_premium():.6f}")
```

```text
q=0.00  premium 0.000000
q=0.08  premium 0.401033
```

## Convergence

For European options the tree converges to the Black-Scholes-Merton price.
The error is of order $1/N$ but changes sign from one step count to the next,
because the position of the strike between the final nodes changes with $N$.
Averaging the prices from two neighbouring step counts is a common way to
damp the oscillation; the library does not do it for you.

```python
from quantflow import BinomialTree, BlackScholes

exact = BlackScholes(100, 105, 0.5, 0.05, 0.25, "call").price()
for steps in (50, 100, 200, 400, 800, 1600):
    tree = BinomialTree(100, 105, 0.5, 0.05, 0.25, "call", "european", steps)
    error = tree.price() - exact
    print(f"steps {steps:>4}  error {error:+.6f}  error x steps {error * steps:+.3f}")
```

```text
steps   50  error -0.031068  error x steps -1.553
steps  100  error +0.014229  error x steps +1.423
steps  200  error -0.006194  error x steps -1.239
steps  400  error +0.001711  error x steps +0.685
steps  800  error -0.000821  error x steps -0.657
steps 1600  error +0.001016  error x steps +1.626
```

American prices also settle as $N$ grows. For a one-year put, a 2000-step
tree was 0.0011 above a QuantLib finite-difference reference (see
{doc}`../benchmarks`).

## Cost

Each step of backward induction is one NumPy operation on a layer of nodes,
so the work grows as $N^2/2$ node updates. Each method call rebuilds the
tree: `greeks()` builds five (one for the price, delta, gamma and theta, two
each for vega and rho) and `early_exercise_premium()` two. A few hundred
steps is a common compromise between accuracy and time.

The lattice probability $p = (e^{(r-q)\Delta t} - d)/(u - d)$ must lie in
$[0, 1]$. With few steps and a large $|r - q|$ relative to $\sigma$ it may
not; the class then emits a warning.
