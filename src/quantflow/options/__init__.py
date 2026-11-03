"""
Option pricing.

* :class:`BlackScholes` and the vectorised :func:`black_scholes_price` /
  :func:`black_scholes_greeks`: closed-form Black-Scholes-Merton prices and
  Greeks for European options.
* :class:`BinomialTree`: Cox-Ross-Rubinstein lattice for European and
  American options.
"""

from .analytic import black_scholes_greeks, black_scholes_price
from .binomial import BinomialTree
from .black_scholes import BlackScholes

__all__ = [
    "BlackScholes",
    "BinomialTree",
    "black_scholes_price",
    "black_scholes_greeks",
]
