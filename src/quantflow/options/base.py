"""
Shared base class and helpers for option pricing models.

This internal module centralizes the parameter validation common to every
pricer (analytic, lattice, Monte-Carlo) along with the terminal payoff
functions, so the individual pricers stay small and consistent.

``OptionPricer`` is an abstract base class; it is not part of the public API and
``BlackScholes`` deliberately keeps its own standalone implementation for
backward compatibility. New pricers such as :class:`~quantflow.options.binomial.BinomialTree`
subclass ``OptionPricer``.
"""

from abc import ABC, abstractmethod
import warnings

import numpy as np


def _validate_params(S, K, T, r, sigma, q, option_type):
    """Validate the core option parameters shared by every pricer."""
    if S <= 0:
        raise ValueError("Stock price (S) must be positive")
    if K <= 0:
        raise ValueError("Strike price (K) must be positive")
    if T <= 0:
        raise ValueError("Time to expiration (T) must be positive")
    if sigma <= 0:
        raise ValueError("Volatility (sigma) must be positive")
    if not np.isfinite(q):
        raise ValueError("Dividend yield (q) must be a finite number")
    if option_type.lower() not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")


def _call_payoff(S_T, K):
    """Terminal payoff of a call: max(S_T - K, 0). Vectorized."""
    return np.maximum(S_T - K, 0.0)


def _put_payoff(S_T, K):
    """Terminal payoff of a put: max(K - S_T, 0). Vectorized."""
    return np.maximum(K - S_T, 0.0)


class OptionPricer(ABC):
    """
    Abstract base class for European/American option pricing models.

    Centralizes parameter validation and terminal payoff logic. Subclasses must
    implement :meth:`price`.

    Parameters
    ----------
    S, K, T, r, sigma : float
        Spot, strike, time-to-expiry (years), risk-free rate, volatility.
    q : float, default 0.0
        Continuous dividend yield.
    option_type : {'call', 'put'}, default 'call'
        Option type.
    """

    def __init__(self, S, K, T, r, sigma, q=0.0, option_type="call"):
        _validate_params(S, K, T, r, sigma, q, option_type)
        self.S = float(S)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.sigma = float(sigma)
        self.q = float(q)
        self.option_type = option_type.lower()

        if self.T > 5:
            warnings.warn(
                "Time to expiration > 5 years may produce unrealistic results"
            )
        if self.sigma > 2:
            warnings.warn("Volatility > 200% may produce unrealistic results")
        if self.q > 1:
            warnings.warn("Dividend yield (q) > 100% may produce unrealistic results")
        elif self.q < 0:
            warnings.warn("Negative dividend yield (q) is unusual but allowed")

    @abstractmethod
    def price(self) -> float:
        """Return the theoretical option price."""
        raise NotImplementedError

    def _intrinsic(self, S_T):
        """Intrinsic value at the given underlying price(s) for this option type."""
        if self.option_type == "call":
            return _call_payoff(S_T, self.K)
        return _put_payoff(S_T, self.K)
