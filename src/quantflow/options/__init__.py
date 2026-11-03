"""
Option pricing.

* :class:`BlackScholes` and the vectorised :func:`black_scholes_price` /
  :func:`black_scholes_greeks`: closed-form Black-Scholes-Merton prices and
  Greeks for European options.
* :class:`BinomialTree`: Cox-Ross-Rubinstein lattice for European and
  American options.
* :class:`MonteCarlo`: Monte Carlo price of European options with a standard
  error, antithetic variates and a control variate.
"""

from .analytic import black_scholes_greeks, black_scholes_price
from .binomial import BinomialTree
from .black_scholes import BlackScholes
from .monte_carlo import MonteCarlo, MonteCarloResult

__all__ = [
    "BlackScholes",
    "BinomialTree",
    "MonteCarlo",
    "MonteCarloResult",
    "black_scholes_price",
    "black_scholes_greeks",
]
