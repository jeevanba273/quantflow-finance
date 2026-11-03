# Monte Carlo

{class}`~quantflow.MonteCarlo` estimates the price of a European call or put
by simulating the terminal price under geometric Brownian motion. It exists
to show Monte Carlo error and variance reduction on a problem whose answer is
known: for European options the closed form is exact and much cheaper.

## Price and standard error

```python
from quantflow import BlackScholes, MonteCarlo

mc = MonteCarlo(S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type="call",
                n_paths=200_000, seed=7)
res = mc.simulate(confidence=0.95)
print(f"price      {res.price:.4f}")
print(f"std error  {res.std_error:.4f}")
print(f"95% CI     [{res.ci_low:.4f}, {res.ci_high:.4f}]")
print(f"closed form {BlackScholes(100, 100, 1.0, 0.05, 0.2).price():.4f}")
```

```text
price      10.4479
std error  0.0329
95% CI     [10.3834, 10.5123]
closed form 10.4506
```

The terminal price is sampled exactly,
$S_T = S \exp\bigl((r - q - \sigma^2/2) T + \sigma \sqrt{T} Z\bigr)$, so there is
no discretisation bias. The standard error is the sample standard deviation
of the discounted payoffs divided by $\sqrt{n}$, and the interval is
price $\pm z \times$ standard error with $z$ the normal quantile. With a seed,
every call to `simulate()` returns the same numbers; `seed=None` draws fresh
numbers each time.

## Error against the number of paths

The standard error falls as $1/\sqrt{n}$: four times the paths halve it.

```python
from quantflow import BlackScholes, MonteCarlo

exact = BlackScholes(100, 100, 1.0, 0.05, 0.2).price()
print("paths        price  std err  error/std err")
for n in (1_000, 10_000, 100_000, 1_000_000):
    res = MonteCarlo(100, 100, 1.0, 0.05, 0.2, n_paths=n, seed=42).simulate()
    print(f"{n:>9,}  {res.price:8.4f}  {res.std_error:7.4f}  {(res.price - exact) / res.std_error:+6.2f}")
```

```text
paths        price  std err  error/std err
    1,000    9.9120   0.4483   -1.20
   10,000   10.3452   0.1477   -0.71
  100,000   10.4205   0.0468   -0.64
1,000,000   10.4532   0.0147   +0.18
```

## Variance reduction

Two methods are available, alone or together.

- **Antithetic variates** (`antithetic=True`) pair every draw $Z$ with $-Z$
  and average the two discounted payoffs. The pair averages are the
  independent samples.
- **Control variate** (`control_variate=True`) uses the discounted terminal
  price, whose mean $S e^{-qT}$ is known, and subtracts
  $\hat\beta\,(X - S e^{-qT})$ from the payoff with $\hat\beta$ the sample
  regression coefficient (Glasserman 2003, section 4.1).

Comparisons below use the same number of simulated terminal prices.

```python
from quantflow import MonteCarlo

base = dict(S=100, K=100, T=1.0, r=0.05, sigma=0.2, n_paths=100_000, seed=42)
plain = MonteCarlo(**base).simulate().std_error
for label, kw in [("antithetic", dict(antithetic=True)),
                  ("control variate", dict(control_variate=True)),
                  ("both", dict(antithetic=True, control_variate=True))]:
    se = MonteCarlo(**base, **kw).simulate().std_error
    print(f"{label:<16} std error {se:.5f}  ratio to plain {se / plain:.3f}")
```

```text
antithetic       std error 0.03308  ratio to plain 0.707
control variate  std error 0.01788  ratio to plain 0.382
both             std error 0.00880  ratio to plain 0.188
```

The gain depends on the contract. These are the four contracts used in
`tests/test_monte_carlo.py`:

```python
from quantflow import MonteCarlo

contracts = [(100, 100, 1.0, 0.05, 0.2, "call", 0.0), (100, 120, 0.5, 0.03, 0.3, "call", 0.02),
             (100, 90, 0.25, 0.05, 0.25, "put", 0.0), (100, 110, 2.0, 0.02, 0.4, "put", 0.01)]
print("contract                       antithetic  control  both")
for S, K, T, r, sigma, kind, q in contracts:
    def se(**kw):
        return MonteCarlo(S, K, T, r, sigma, kind, q=q, n_paths=100_000, seed=11,
                          **kw).simulate().std_error
    plain = se()
    print(f"{kind:<4} K={K:<3} T={T:<4} sigma={sigma:<4}      "
          f"{se(antithetic=True) / plain:.3f}      {se(control_variate=True) / plain:.3f}"
          f"    {se(antithetic=True, control_variate=True) / plain:.3f}")
```

```text
contract                       antithetic  control  both
call K=100 T=1.0  sigma=0.2       0.707      0.382    0.187
call K=120 T=0.5  sigma=0.3       0.950      0.708    0.176
put  K=90  T=0.25 sigma=0.25      0.927      0.804    0.195
put  K=110 T=2.0  sigma=0.4       0.337      0.651    0.161
```

The out-of-the-money call (K=120) gains little from antithetic sampling,
because its payoff is zero on most paths whatever the sign of $Z$. The
{doc}`../benchmarks` page also measures run time, so efficiency (variance
times time) can be compared.

## Limits

Only European calls and puts on one GBM underlying are supported; there is
no path simulation, so no Asian, barrier or American payoffs, and no Greeks.
All paths are held in memory at once, in several NumPy arrays of length
`n_paths`.
