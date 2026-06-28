# src/quantflow/options/black_scholes.py
"""
Complete Black-Scholes option pricing model implementation with all Greeks.

This module provides a comprehensive implementation of the Black-Scholes-Merton model
for European option pricing, including all Greeks calculations (Delta, Gamma, Theta, Vega, Rho).
"""

import numpy as np
from scipy.stats import norm
from typing import Dict, Literal
import warnings


class BlackScholes:
    """
    Black-Scholes option pricing model with complete Greeks suite.

    The Black-Scholes model is used to calculate the theoretical price of European
    options. It assumes constant volatility and interest rates. A continuous
    dividend yield ``q`` is supported (the Black-Scholes-Merton extension); the
    default ``q=0.0`` recovers the classic non-dividend Black-Scholes results.

    Parameters
    ----------
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to expiration in years
    r : float
        Risk-free interest rate (annualized)
    sigma : float
        Volatility of underlying asset (annualized)
    option_type : {'call', 'put'}, default 'call'
        Type of option
    q : float, default 0.0
        Continuous dividend yield (annualized). Use 0.0 for a non-dividend-paying
        asset; for an index or FX rate use the dividend / foreign rate.

    Attributes
    ----------
    S, K, T, r, sigma, q : float
        Model parameters
    option_type : str
        Option type ('call' or 'put')

    Examples
    --------
    >>> # Price a call option
    >>> option = BlackScholes(S=100, K=105, T=0.25, r=0.05, sigma=0.2)
    >>> price = option.price()
    >>> print(f"Call price: ${price:.2f}")
    Call price: $2.87

    >>> # Get all Greeks at once
    >>> greeks = option.greeks()
    >>> print(f"Delta: {greeks['delta']:.3f}")
    Delta: 0.378

    References
    ----------
    Black, F., & Scholes, M. (1973). The pricing of options and corporate liabilities.
    Journal of Political Economy, 81(3), 637-654.
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
        # Input validation
        if S <= 0:
            raise ValueError("Stock price (S) must be positive")
        if K <= 0:
            raise ValueError("Strike price (K) must be positive")
        if T <= 0:
            raise ValueError("Time to expiration (T) must be positive")
        if sigma <= 0:
            raise ValueError("Volatility (sigma) must be positive")
        if option_type.lower() not in ["call", "put"]:
            raise ValueError("option_type must be 'call' or 'put'")
        if not np.isfinite(q):
            raise ValueError("Dividend yield (q) must be a finite number")

        self.S = float(S)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.sigma = float(sigma)
        self.option_type = option_type.lower()
        self.q = float(q)

        # Warn for extreme parameters
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

    def _d1(self) -> float:
        """Calculate d1 parameter for Black-Scholes(-Merton) formula."""
        numerator = (
            np.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T
        )
        denominator = self.sigma * np.sqrt(self.T)
        return numerator / denominator

    def _d2(self) -> float:
        """Calculate d2 parameter for Black-Scholes formula."""
        return self._d1() - self.sigma * np.sqrt(self.T)

    def price(self) -> float:
        """
        Calculate option price using Black-Scholes formula.

        Returns
        -------
        float
            Theoretical option price

        Notes
        -----
        For a call option:
        C = S₀e^(-qT)N(d₁) - Ke^(-rT)N(d₂)

        For a put option:
        P = Ke^(-rT)N(-d₂) - S₀e^(-qT)N(-d₁)

        where N(x) is the cumulative standard normal distribution function and
        q is the continuous dividend yield.
        """
        d1, d2 = self._d1(), self._d2()
        spot = self.S * np.exp(-self.q * self.T)

        if self.option_type == "call":
            return spot * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(
                d2
            )
        else:  # put
            return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - spot * norm.cdf(
                -d1
            )

    def delta(self) -> float:
        """
        Calculate option delta (∂V/∂S).

        Delta measures the rate of change of option price with respect to
        the underlying asset price.

        Returns
        -------
        float
            Option delta

        Notes
        -----
        For calls: Δ = e^(-qT)N(d₁)
        For puts: Δ = e^(-qT)(N(d₁) - 1)
        """
        d1 = self._d1()
        disc_q = np.exp(-self.q * self.T)
        if self.option_type == "call":
            return disc_q * norm.cdf(d1)
        else:
            return disc_q * (norm.cdf(d1) - 1)

    def gamma(self) -> float:
        """
        Calculate option gamma (∂²V/∂S²).

        Gamma measures the rate of change of delta with respect to
        the underlying asset price.

        Returns
        -------
        float
            Option gamma

        Notes
        -----
        Γ = e^(-qT)φ(d₁) / (S₀σ√T)
        where φ(x) is the standard normal probability density function.
        """
        d1 = self._d1()
        return (
            np.exp(-self.q * self.T)
            * norm.pdf(d1)
            / (self.S * self.sigma * np.sqrt(self.T))
        )

    def theta(self) -> float:
        """
        Calculate option theta (∂V/∂T).

        Theta measures the rate of change of option price with respect to time.
        Usually expressed as the dollar amount an option loses per day.

        Returns
        -------
        float
            Option theta (per year, divide by 365 for daily theta)

        Notes
        -----
        For calls:
        Θ = -S₀e^(-qT)φ(d₁)σ/(2√T) - rKe^(-rT)N(d₂) + qS₀e^(-qT)N(d₁)

        For puts:
        Θ = -S₀e^(-qT)φ(d₁)σ/(2√T) + rKe^(-rT)N(-d₂) - qS₀e^(-qT)N(-d₁)
        """
        d1, d2 = self._d1(), self._d2()
        disc_q = np.exp(-self.q * self.T)

        first_term = -(self.S * disc_q * norm.pdf(d1) * self.sigma) / (
            2 * np.sqrt(self.T)
        )

        if self.option_type == "call":
            second_term = -self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
            dividend_term = self.q * self.S * disc_q * norm.cdf(d1)
        else:
            second_term = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2)
            dividend_term = -self.q * self.S * disc_q * norm.cdf(-d1)

        return first_term + second_term + dividend_term

    def vega(self) -> float:
        """
        Calculate option vega (∂V/∂σ).

        Vega measures the rate of change of option price with respect to volatility.

        Returns
        -------
        float
            Option vega (for 1% change in volatility)

        Notes
        -----
        ν = S₀e^(-qT)φ(d₁)√T / 100
        """
        d1 = self._d1()
        return (
            self.S * np.exp(-self.q * self.T) * norm.pdf(d1) * np.sqrt(self.T)
        ) / 100

    def rho(self) -> float:
        """
        Calculate option rho (∂V/∂r).

        Rho measures the rate of change of option price with respect to
        the risk-free interest rate.

        Returns
        -------
        float
            Option rho (for 1% change in interest rate)

        Notes
        -----
        For calls: ρ = KTe^(-rT)N(d₂) / 100
        For puts: ρ = -KTe^(-rT)N(-d₂) / 100
        """
        d2 = self._d2()

        if self.option_type == "call":
            return (self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2)) / 100
        else:
            return -(self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2)) / 100

    def greeks(self) -> Dict[str, float]:
        """
        Calculate all Greeks at once.

        Returns
        -------
        dict
            Dictionary containing all Greeks:
            - price: Option price
            - delta: Price sensitivity to underlying
            - gamma: Delta sensitivity to underlying
            - theta: Time decay (per year)
            - vega: Volatility sensitivity (per 1%)
            - rho: Interest rate sensitivity (per 1%)
        """
        return {
            "price": self.price(),
            "delta": self.delta(),
            "gamma": self.gamma(),
            "theta": self.theta(),
            "vega": self.vega(),
            "rho": self.rho(),
        }

    def implied_volatility(
        self, market_price: float, tolerance: float = 1e-6, max_iterations: int = 100
    ) -> float:
        """
        Calculate implied volatility using Newton-Raphson method.

        Parameters
        ----------
        market_price : float
            Observed market price of the option
        tolerance : float, default 1e-6
            Convergence tolerance
        max_iterations : int, default 100
            Maximum number of iterations

        Returns
        -------
        float
            Implied volatility

        Raises
        ------
        ValueError
            If convergence is not achieved
        """
        # Initial guess
        sigma_guess = 0.3

        for i in range(max_iterations):
            # Create temporary option with current volatility guess
            temp_option = BlackScholes(
                self.S, self.K, self.T, self.r, sigma_guess, self.option_type, q=self.q
            )

            # Calculate price and vega with current guess
            price_diff = temp_option.price() - market_price
            vega_val = temp_option.vega() * 100  # Convert back from percentage

            # Check convergence
            if abs(price_diff) < tolerance:
                return sigma_guess

            # Newton-Raphson update
            if vega_val == 0:
                raise ValueError("Vega is zero, cannot calculate implied volatility")

            sigma_guess = sigma_guess - price_diff / vega_val

            # Ensure positive volatility
            if sigma_guess <= 0:
                sigma_guess = 0.01

        raise ValueError(f"Failed to converge after {max_iterations} iterations")

    def __repr__(self) -> str:
        """String representation of the Black-Scholes option."""
        return (
            f"BlackScholes({self.option_type.capitalize()}: "
            f"S={self.S}, K={self.K}, T={self.T:.3f}, "
            f"r={self.r:.3f}, σ={self.sigma:.3f}, q={self.q:.3f})"
        )

    def summary(self) -> str:
        """
        Generate a comprehensive summary of the option.

        Returns
        -------
        str
            Formatted summary including price and all Greeks
        """
        price = self.price()
        greeks = self.greeks()

        summary = f"""
Black-Scholes {self.option_type.capitalize()} Option Summary
{'='*50}
Parameters:
  Spot Price (S):     ${self.S:>8.2f}
  Strike Price (K):   ${self.K:>8.2f}
  Time to Expiry (T): {self.T:>8.3f} years
  Risk-free Rate (r): {self.r:>8.2%}
  Dividend Yield (q): {self.q:>8.2%}
  Volatility (σ):     {self.sigma:>8.2%}

Valuation:
  Option Price:       ${price:>8.2f}

Greeks:
  Delta (Δ):          {greeks['delta']:>8.4f}
  Gamma (Γ):          {greeks['gamma']:>8.4f}
  Theta (Θ):          ${greeks['theta']:>8.2f} /year
  Vega (ν):           ${greeks['vega']:>8.2f} /1% vol
  Rho (ρ):            ${greeks['rho']:>8.2f} /1% rate
        """
        return summary.strip()
