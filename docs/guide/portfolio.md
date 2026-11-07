# Portfolio optimisation

{class}`~quantflow_finance.Portfolio` takes a matrix of periodic returns (one column
per asset) and computes portfolio statistics and mean-variance optimal
weights. Expected returns and covariances are the sample mean and sample
covariance, annualised with `periods_per_year` (252 by default). Weights
always sum to 1.

```python
import numpy as np
import pandas as pd

from quantflow_finance import Portfolio

rng = np.random.default_rng(7)
vols = np.array([0.012, 0.004, 0.009, 0.014])
corr = np.array([[1.0, -0.2, 0.1, 0.6], [-0.2, 1.0, 0.2, -0.1],
                 [0.1, 0.2, 1.0, 0.2], [0.6, -0.1, 0.2, 1.0]])
mu = np.array([0.0004, 0.00012, 0.00025, 0.0005])
chol = np.linalg.cholesky(corr * np.outer(vols, vols))
returns = pd.DataFrame(mu + rng.standard_normal((2000, 4)) @ chol.T,
                       columns=["EQ", "BOND", "GOLD", "EM"])
p = Portfolio(returns)
print(p.correlation_matrix().round(2))
```

```text
        EQ  BOND  GOLD    EM
EQ    1.00 -0.17  0.11  0.60
BOND -0.17  1.00  0.20 -0.10
GOLD  0.11  0.20  1.00  0.22
EM    0.60 -0.10  0.22  1.00
```

## Optimal portfolios

```python
def show(label, res):
    w = " ".join(f"{a}={x:.3f}" for a, x in res["weights_by_asset"].items())
    print(f"{label:<14} ret {res['return']:.4f} vol {res['volatility']:.4f} {w}")

show("min variance", p.min_variance())
show("max Sharpe", p.max_sharpe(risk_free_rate=0.02))
show("target 4%", p.optimize("target_return", target_return=0.04))
show("min var, short", p.min_variance(allow_short=True))
```

```text
min variance   ret 0.0048 vol 0.0548 EQ=0.115 BOND=0.813 GOLD=0.056 EM=0.016
max Sharpe     ret 0.0550 vol 0.1198 EQ=0.424 BOND=0.000 GOLD=0.576 EM=0.000
target 4%      ret 0.0400 vol 0.0922 EQ=0.309 BOND=0.241 GOLD=0.450 EM=0.000
min var, short ret 0.0048 vol 0.0548 EQ=0.115 BOND=0.813 GOLD=0.056 EM=0.016
```

Each method returns a dict with `weights`, `weights_by_asset`, annualised
`return` and `volatility`, `sharpe` and the optimiser's `success` flag.

- Long-only problems (the default) are solved with SLSQP from equal
  weights, with bounds $0 \le w_i \le 1$.
- With `allow_short=True`, `max_sharpe` uses the closed form
  $w \propto \Sigma^{-1}(\mu - r_f \mathbf 1)$. If the minimum-variance
  portfolio's return is below $r_f$ the normalising sum is negative, no
  fully invested portfolio maximises the Sharpe ratio, and the method raises
  a `ValueError` instead of returning extreme weights.
- An infeasible target (for example a long-only target below the lowest
  asset mean) makes SLSQP fail; the result has `success=False` and a
  `RuntimeWarning` is issued.

The {doc}`../validation` page compares these weights with the closed-form
solutions of Merton (1972).

## Efficient frontier

```python
frontier = p.efficient_frontier(n_points=6, risk_free_rate=0.02)
print(frontier[["return", "volatility", "sharpe"]].round(4).to_string(index=False))
```

```text
 return  volatility  sharpe
 0.0048      0.0548 -0.2769
 0.0160      0.0594 -0.0667
 0.0272      0.0721  0.1003
 0.0384      0.0896  0.2058
 0.0496      0.1095  0.2706
 0.0608      0.1885  0.2165
```

Targets run from the minimum-variance return to the highest single-asset
mean. Targets the solver cannot reach are dropped, so the frame can have
fewer rows than requested (the last, all-in-one-asset point often is).

## Risk metrics of a portfolio

{meth}`~quantflow_finance.Portfolio.risk_metrics` returns a
{class}`~quantflow_finance.RiskMetrics` for the portfolio's return series with
constant weights (rebalanced every period).

```python
rm = p.risk_metrics(p.max_sharpe()["weights"])
print(f"95% VaR {rm.var_historical(0.05):.4f}  max drawdown {rm.max_drawdown():.4f}")
```

```text
95% VaR -0.0123  max drawdown -0.3003
```

## Cautions

Mean-variance weights are very sensitive to the expected returns, and sample
means are noisy. The standard error of each annualised sample mean above is
$\hat\sigma \sqrt{252 / n}$:

```python
se = returns.std() * np.sqrt(252 / len(returns))
print(pd.DataFrame({"annual mean": returns.mean() * 252, "std error": se}).round(4))
```

```text
      annual mean  std error
EQ         0.0608     0.0042
BOND      -0.0067     0.0014
GOLD       0.0507     0.0031
EM         0.0257     0.0050
```

Small changes in the inputs can move the weights a lot (Michaud 1989). This
class has no shrinkage of the estimates and no turnover or transaction-cost
constraints; the articles on portfolio construction and hedging listed in
the README discuss these issues.
