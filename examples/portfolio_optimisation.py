"""Mean-variance optimisation on simulated returns of four assets.

Run with:  python examples/portfolio_optimisation.py
"""

import numpy as np
import pandas as pd

from quantflow import Portfolio

rng = np.random.default_rng(7)
vols = np.array([0.012, 0.004, 0.009, 0.014])  # daily
corr = np.array(
    [
        [1.0, -0.2, 0.1, 0.6],
        [-0.2, 1.0, 0.2, -0.1],
        [0.1, 0.2, 1.0, 0.2],
        [0.6, -0.1, 0.2, 1.0],
    ]
)
cov = corr * np.outer(vols, vols)
mu = np.array([0.0004, 0.00012, 0.00025, 0.0005])
# Correlated normals through a Cholesky factor, which is reproducible across
# NumPy versions (multivariate_normal goes through an SVD and is not).
draws = mu + rng.standard_normal((2000, 4)) @ np.linalg.cholesky(cov).T
returns = pd.DataFrame(draws, columns=["EQ", "BOND", "GOLD", "EM"])

p = Portfolio(returns)
for label, res in [
    ("min variance", p.min_variance()),
    ("max Sharpe", p.max_sharpe(risk_free_rate=0.02)),
    ("target 4%", p.optimize("target_return", target_return=0.04)),
]:
    weights = " ".join(f"{a}={w:.3f}" for a, w in res["weights_by_asset"].items())
    print(
        f"{label:<13} ret {res['return']:.4f} vol {res['volatility']:.4f} "
        f"Sharpe {res['sharpe']:.3f}  {weights}"
    )

frontier = p.efficient_frontier(n_points=5)
print(frontier[["return", "volatility", "sharpe"]].round(4).to_string(index=False))
