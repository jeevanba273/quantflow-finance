"""
QuantFlow Finance: option pricing, return-based risk metrics and mean-variance
portfolio optimisation on NumPy, SciPy and pandas.
"""

__version__ = "0.3.2"
__author__ = "Jeevan B A"
__email__ = "jeevanba273@gmail.com"

from .data.fetcher import MarketData
from .options.analytic import black_scholes_greeks, black_scholes_price
from .options.binomial import BinomialTree
from .options.black_scholes import BlackScholes
from .options.monte_carlo import MonteCarlo, MonteCarloResult
from .risk.metrics import RiskMetrics
from .risk.portfolio import Portfolio

__all__ = [
    "BlackScholes",
    "BinomialTree",
    "MonteCarlo",
    "MonteCarloResult",
    "black_scholes_price",
    "black_scholes_greeks",
    "RiskMetrics",
    "Portfolio",
    "MarketData",
]
