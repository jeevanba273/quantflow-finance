"""
Options pricing and Greeks calculation module.

This module provides implementations of various option pricing models including
the Black-Scholes-Merton model for European options and a Cox-Ross-Rubinstein
binomial tree for European and American options.
"""

from .binomial import BinomialTree
from .black_scholes import BlackScholes

__all__ = ["BlackScholes", "BinomialTree"]
