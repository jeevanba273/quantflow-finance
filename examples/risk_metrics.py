"""Risk metrics on simulated daily returns (Student-t, 5 degrees of freedom).

Run with:  python examples/risk_metrics.py
"""

import numpy as np
import pandas as pd

from quantflow_finance import RiskMetrics

rng = np.random.default_rng(2024)
dates = pd.bdate_range("2020-01-01", periods=1250)
returns = pd.Series(0.0003 + 0.009 * rng.standard_t(5, len(dates)), index=dates)

rm = RiskMetrics(returns)  # 252 periods per year
print(f"annualised volatility  {rm.volatility():.4f}")
print(f"Sharpe (rf 2%)         {rm.sharpe_ratio(0.02):.4f}")
print(f"Sortino (rf 2%)        {rm.sortino_ratio(0.02):.4f}")
print(f"max drawdown           {rm.max_drawdown():.4f}")
for alpha in (0.05, 0.01):
    print(
        f"{1 - alpha:.0%} VaR  historical {rm.var_historical(alpha):.4f}"
        f"  normal {rm.var_parametric(alpha):.4f}"
        f"  Cornish-Fisher {rm.var_cornish_fisher(alpha):.4f}"
        f"  ES {rm.expected_shortfall(alpha):.4f}"
    )

benchmark = pd.Series(0.0002 + 0.01 * rng.standard_normal(len(dates)), index=dates)
strategy = 0.7 * benchmark + returns * 0.5
capm = RiskMetrics(strategy).capm(benchmark)
print(
    f"beta {capm['beta']:.4f}  alpha {capm['alpha']:.4f}  R^2 {capm['r_squared']:.4f}"
)
