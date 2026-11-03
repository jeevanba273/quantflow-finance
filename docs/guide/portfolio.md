# Portfolio optimisation

{class}`~quantflow.Portfolio` takes a matrix of periodic returns (one column
per asset) and computes portfolio statistics and mean-variance optimal
weights. Expected returns and covariances are the sample mean and sample
covariance, annualised with `periods_per_year` (252 by default). Weights
always sum to 1.

```python
import numpy as np
import pandas as pd

from quantflow import Portfolio

rng = np.random.default_rng(7)
vols = np.array([0.012, 0.004, 0.009, 0.014])
corr = np.array([[1.0, -0.2, 0.1, 0.6], [-0.2, 1.0, 0.2, -0.1],
                 [0.1, 0.2, 1.0, 0.2], [0.6, -0.1, 0.2, 1.0]])
mu = np.array([0.0004, 0.00012, 0.00025, 0.0005])
returns = pd.DataFrame(rng.multivariate_normal(mu, corr * np.outer(vols, vols), 2000),
                       columns=["EQ", "BOND", "GOLD", "EM"])
p = Portfolio(returns)
print(p.correlation_matrix().round(2))
```

```text
        EQ  BOND  GOLD    EM
EQ    1.00 -0.17  0.12  0.60
BOND -0.17  1.00  0.19 -0.10
GOLD  0.12  0.19  1.00  0.23
EM    0.60 -0.10  0.23  1.00
```

## Optimal portfolios

```python
def show(label, res):
    w = " ".join(f"{a}={x:.3f}" for a, x in res["weights_by_asset"].items())
    print(f"{label:<14} ret {res['return']:.4f} vol {res['volatility']:.4f} {w}")

show("min variance", p.min_variance())
show("max Sharpe", p.max_sharpe(risk_free_rate=0.02))
show("target 12%", p.optimize("target_return", target_return=0.12))
show("min var, short", p.min_variance(allow_short=True))
```

```text
min variance   ret 0.0753 vol 0.0549 EQ=0.117 BOND=0.811 GOLD=0.056 EM=0.016
max Sharpe     ret 0.1051 vol 0.0681 EQ=0.005 BOND=0.597 GOLD=0.224 EM=0.173
target 12%     ret 0.1200 vol 0.0821 EQ=0.000 BOND=0.447 GOLD=0.321 EM=0.232
min var, short ret 0.0753 vol 0.0549 EQ=0.117 BOND=0.811 GOLD=0.056 EM=0.016
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
 0.0753      0.0549  1.0072
 0.0984      0.0631  1.2416
 0.1215      0.0836  1.2130
 0.1446      0.1112  1.1199
 0.1677      0.1438  1.0267
```

Targets run from the minimum-variance return to the highest single-asset
mean. Targets the solver cannot reach are dropped, so the frame can have
fewer rows than requested (the last, all-in-one-asset point often is).

## Risk metrics of a portfolio

{meth}`~quantflow.Portfolio.risk_metrics` returns a
{class}`~quantflow.RiskMetrics` for the portfolio's return series with
constant weights (rebalanced every period).

```python
rm = p.risk_metrics(p.max_sharpe()["weights"])
print(f"95% VaR {rm.var_historical(0.05):.4f}  max drawdown {rm.max_drawdown():.4f}")
```

```text
95% VaR -0.0066  max drawdown -0.0978
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
EQ         0.0928     0.0042
BOND       0.0657     0.0014
GOLD       0.1445     0.0032
EM         0.1908     0.0049
```

Small changes in the inputs can move the weights a lot (Michaud 1989). This
class has no shrinkage of the estimates and no turnover or transaction-cost
constraints; the articles on portfolio construction and hedging listed in
the README discuss these issues.
