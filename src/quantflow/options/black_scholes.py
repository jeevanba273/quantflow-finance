"""
Black-Scholes-Merton pricing of a single European option.

:class:`BlackScholes` wraps the closed-form formulas in
:mod:`quantflow.options.analytic` for one option and adds an implied
volatility solver. For many options at once, call
:func:`~quantflow.black_scholes_price` or
:func:`~quantflow.black_scholes_greeks` on arrays instead.
"""

from typing import Literal

import numpy as np
from scipy.optimize import brentq

from .analytic import _d1_d2, _greeks_core, _price_core, _price_scalar
from .base import OptionPricer


class BlackScholes(OptionPricer):
    r"""
    European option under the Black-Scholes-Merton model.

    The underlying follows geometric Brownian motion with constant volatility
    ``sigma``, a constant continuously compounded rate ``r`` and a continuous
    dividend yield ``q``. With ``q = 0`` this is the original Black-Scholes
    (1973) model; ``q > 0`` is Merton's (1973) extension. Setting ``q`` to the
    foreign rate gives Garman-Kohlhagen FX options, and ``q = r`` gives Black
    (1976) options on futures.

    Parameters
    ----------
    S : float
        Spot price of the underlying, > 0.
    K : float
        Strike price, > 0.
    T : float
        Time to expiry as a year fraction, > 0. No day count is applied;
        ``days / 365`` is the usual choice.
    r : float
        Continuously compounded risk-free rate per year (0.05 = 5%).
    sigma : float
        Annualised volatility of log returns, > 0 (0.2 = 20%).
    option_type : {'call', 'put'}, default 'call'
        Option type.
    q : float, default 0.0
        Continuous dividend yield per year.

    Attributes
    ----------
    S, K, T, r, sigma, q : float
        Model parameters.
    option_type : str
        ``'call'`` or ``'put'``.

    Notes
    -----
    Greek conventions: ``theta`` is the calendar-time derivative per year
    (``-dV/dT``; divide by 365 for a one-day figure when ``T`` is ACT/365),
    ``vega`` is per 1 percentage point of volatility and ``rho`` per
    1 percentage point of ``r``. ``delta`` and ``gamma`` are per unit of the
    underlying.

    The model assumes no transaction costs, continuous trading, constant
    ``r``, ``q`` and ``sigma``, and lognormal prices. It has no volatility
    smile: one ``sigma`` prices every strike.

    Examples
    --------
    >>> from quantflow import BlackScholes
    >>> option = BlackScholes(S=42, K=40, T=0.5, r=0.10, sigma=0.20)
    >>> print(f"{option.price():.4f}")
    4.7594
    >>> print(f"{option.delta():.4f}")
    0.7791

    References
    ----------
    Black, F. and Scholes, M. (1973). The Pricing of Options and Corporate
    Liabilities. Journal of Political Economy, 81(3), 637-654.

    Merton, R. C. (1973). Theory of Rational Option Pricing. Bell Journal of
    Economics and Management Science, 4(1), 141-183.

    Hull, J. C. Options, Futures, and Other Derivatives (10th ed.), chapters
    15 and 19. Pearson.
    """

    def __init__(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal["call", "put"] = "call",
        q: float = 0.0,
    ):
        super().__init__(S, K, T, r, sigma, q=q, option_type=option_type)

    def _args(self, sigma=None):
        sigma = self.sigma if sigma is None else sigma
        return (self.S, self.K, self.T, self.r, sigma, self.q)

    def _d1(self) -> float:
        """d1 of the Black-Scholes-Merton formula."""
        return float(_d1_d2(*self._args())[0])

    def _d2(self) -> float:
        """d2 of the Black-Scholes-Merton formula, ``d1 - sigma * sqrt(T)``."""
        return float(_d1_d2(*self._args())[1])

    def _greek(self, name: str) -> float:
        greeks = _greeks_core(*self._args(), self.option_type == "call")
        return float(greeks[name])

    def price(self) -> float:
        r"""
        Option price.

        Returns
        -------
        float
            Present value in the same currency units as ``S`` and ``K``.

        Notes
        -----
        .. math::

            C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2), \qquad
            P = K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1).
        """
        return float(_price_core(*self._args(), self.option_type == "call"))

    def delta(self) -> float:
        r"""
        Delta, :math:`\partial V / \partial S`.

        Calls: :math:`e^{-qT} N(d_1)`; puts: :math:`-e^{-qT} N(-d_1)`.
        """
        return self._greek("delta")

    def gamma(self) -> float:
        r"""
        Gamma, :math:`\partial^2 V / \partial S^2`, the same for calls and puts.

        :math:`\Gamma = e^{-qT} \varphi(d_1) / (S \sigma \sqrt{T})`, with
        :math:`\varphi` the standard normal density.
        """
        return self._greek("gamma")

    def theta(self) -> float:
        r"""
        Theta per year, :math:`-\partial V / \partial T`.

        Divide by 365 for an approximate one-day change when ``T`` is measured
        in calendar years.

        Notes
        -----
        .. math::

            \Theta_C = -\frac{S e^{-qT} \varphi(d_1) \sigma}{2\sqrt{T}}
                       - r K e^{-rT} N(d_2) + q S e^{-qT} N(d_1),

            \Theta_P = -\frac{S e^{-qT} \varphi(d_1) \sigma}{2\sqrt{T}}
                       + r K e^{-rT} N(-d_2) - q S e^{-qT} N(-d_1).
        """
        return self._greek("theta")

    def vega(self) -> float:
        r"""
        Vega per 1 percentage point of volatility.

        :math:`S e^{-qT} \varphi(d_1) \sqrt{T} / 100`, the same for calls and
        puts. Multiply by 100 for :math:`\partial V / \partial \sigma`.
        """
        return self._greek("vega")

    def rho(self) -> float:
        r"""
        Rho per 1 percentage point of the risk-free rate.

        Calls: :math:`K T e^{-rT} N(d_2) / 100`; puts:
        :math:`-K T e^{-rT} N(-d_2) / 100`. Multiply by 100 for
        :math:`\partial V / \partial r`.
        """
        return self._greek("rho")

    def greeks(self) -> dict[str, float]:
        """
        Price and all Greeks in one call.

        Returns
        -------
        dict
            Keys ``price``, ``delta``, ``gamma``, ``theta`` (per year),
            ``vega`` (per 1% vol) and ``rho`` (per 1% rate).
        """
        greeks = _greeks_core(*self._args(), self.option_type == "call")
        return {name: float(value) for name, value in greeks.items()}

    def implied_volatility(
        self, market_price: float, tolerance: float = 1e-10, max_iterations: int = 100
    ) -> float:
        """
        Volatility at which the model price equals ``market_price``.

        The price is strictly increasing in volatility, so the solution is
        unique when it exists. The method checks the no-arbitrage bounds
        (for a call, ``max(S e^{-qT} - K e^{-rT}, 0) < C < S e^{-qT}``) and
        then runs Brent's method on ``[1e-6, 10]``, widening the upper end
        up to 1000 if needed. The ``sigma`` this object was built with is not
        used.

        Parameters
        ----------
        market_price : float
            Observed option price, in the same units as ``S`` and ``K``.
        tolerance : float, default 1e-10
            Absolute tolerance on the volatility (Brent ``xtol``).
        max_iterations : int, default 100
            Maximum number of Brent iterations.

        Returns
        -------
        float
            Annualised implied volatility.

        Raises
        ------
        ValueError
            If ``market_price`` is not a positive finite number, lies outside
            the no-arbitrage bounds, has no time value above the lower bound
            (the volatility is then not determined), or the solver cannot
            bracket or converge.

        Notes
        -----
        The accuracy is limited by the conditioning of the problem: the
        volatility error is roughly the price rounding error divided by
        vega. For deep in-the-money or far out-of-the-money options with very
        short expiries vega is tiny, so the recovered volatility can be off in
        the fifth decimal even though it reproduces the price exactly.

        Brent, R. P. (1973). Algorithms for Minimization Without Derivatives,
        chapter 4. Prentice-Hall.
        """
        if not np.isfinite(market_price) or market_price <= 0:
            raise ValueError("market_price must be a positive, finite number")

        disc_q = np.exp(-self.q * self.T)
        disc_r = np.exp(-self.r * self.T)
        if self.option_type == "call":
            lower = max(self.S * disc_q - self.K * disc_r, 0.0)
            upper = self.S * disc_q
        else:
            lower = max(self.K * disc_r - self.S * disc_q, 0.0)
            upper = self.K * disc_r

        if market_price >= upper:
            raise ValueError(
                f"market_price {market_price:.6f} is at or above the no-arbitrage "
                f"upper bound {upper:.6f}; no volatility reproduces it"
            )
        # Prices are computed to about 1e-13 of the larger discounted leg, so a
        # time value smaller than that cannot identify a volatility.
        noise = 1e-13 * max(self.S * disc_q, self.K * disc_r) if lower > 0 else 0.0
        time_value = market_price - lower
        if time_value < -noise:
            raise ValueError(
                f"market_price {market_price:.6f} is below the no-arbitrage lower "
                f"bound {lower:.6f}; no volatility reproduces it"
            )
        if time_value <= noise:
            raise ValueError(
                f"market_price {market_price:.6f} has no time value above the "
                f"lower bound {lower:.6f}; the implied volatility is not determined"
            )

        S, K, T, r, q = self.S, self.K, self.T, self.r, self.q
        is_call = self.option_type == "call"

        def price_gap(sigma):
            return _price_scalar(S, K, T, r, sigma, q, is_call) - market_price

        sigma_low, sigma_high = 1e-6, 10.0
        # Expand the upper bracket for unusually high-volatility prices.
        tries = 0
        while price_gap(sigma_high) < 0 and sigma_high < 1e3 and tries < 20:
            sigma_high *= 2.0
            tries += 1

        if price_gap(sigma_low) > 0 or price_gap(sigma_high) < 0:
            raise ValueError(
                "Failed to bracket implied volatility for the given market price"
            )

        try:
            return float(
                brentq(
                    price_gap,
                    sigma_low,
                    sigma_high,
                    xtol=tolerance,
                    maxiter=max_iterations,
                )
            )
        except (ValueError, RuntimeError) as exc:
            raise ValueError(f"Implied volatility solver failed: {exc}") from exc

    def __repr__(self) -> str:
        return (
            f"BlackScholes({self.option_type.capitalize()}: "
            f"S={self.S}, K={self.K}, T={self.T:.3f}, "
            f"r={self.r:.3f}, sigma={self.sigma:.3f}, q={self.q:.3f})"
        )

    def summary(self) -> str:
        """
        Text summary of the parameters, price and Greeks.

        Returns
        -------
        str
            Multi-line ASCII text.
        """
        greeks = self.greeks()

        summary = f"""
Black-Scholes {self.option_type.capitalize()} Option Summary
{"=" * 50}
Parameters:
  Spot Price (S):     ${self.S:>8.2f}
  Strike Price (K):   ${self.K:>8.2f}
  Time to Expiry (T): {self.T:>8.3f} years
  Risk-free Rate (r): {self.r:>8.2%}
  Dividend Yield (q): {self.q:>8.2%}
  Volatility (sigma): {self.sigma:>8.2%}

Valuation:
  Option Price:       ${greeks["price"]:>8.2f}

Greeks:
  Delta:              {greeks["delta"]:>8.4f}
  Gamma:              {greeks["gamma"]:>8.4f}
  Theta:              ${greeks["theta"]:>8.2f} /year
  Vega:               ${greeks["vega"]:>8.2f} /1% vol
  Rho:                ${greeks["rho"]:>8.2f} /1% rate
        """
        return summary.strip()
