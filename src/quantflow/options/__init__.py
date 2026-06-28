"""
Options pricing and Greeks calculation module.

This module provides implementations of various option pricing models including
the Black-Scholes-Merton model for European options and a Cox-Ross-Rubinstein
binomial tree for European and American options.
"""

from .black_scholes import BlackScholes
from .binomial import BinomialTree

__all__ = ["BlackScholes", "BinomialTree"]
