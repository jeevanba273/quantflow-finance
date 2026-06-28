"""
QuantFlow: Essential quantitative finance tools for modern portfolio management.

A comprehensive Python package for options pricing, risk analytics, and market data processing.
"""

__version__ = "0.2.1"
__author__ = "JEEVAN B A"
__email__ = "jeevanba273@gmail.com"

# Import main classes for easy access
from .options.black_scholes import BlackScholes
from .options.binomial import BinomialTree
from .risk.metrics import RiskMetrics
from .risk.portfolio import Portfolio
from .data.fetcher import MarketData

__all__ = ["BlackScholes", "BinomialTree", "RiskMetrics", "Portfolio", "MarketData"]
