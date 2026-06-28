"""
Cox-Ross-Rubinstein (CRR) binomial tree option pricing model.

Supports both European and American exercise, a continuous dividend yield, and
a full Greeks suite. European prices converge to the Black-Scholes-Merton value
as ``steps`` increases; American options are priced with an early-exercise check
at every node.
"""

from typing import Dict, Literal
import warnings

import numpy as np

from .base import OptionPricer


class BinomialTree(OptionPricer):
    """
    CRR binomial tree pricer for European and American options.

    Parameters
    ----------
    S : float
        Current stock price.
    K : float
        Strike price.
    T : float
        Time to expiration in years.
    r : float
        Risk-free interest rate (annualized).
    sigma : float
        Volatility of the underlying (annualized).
    option_type : {'call', 'put'}, default 'call'
        Option type.
    exercise : {'european', 'american'}, default 'european'
        Exercise style. American options allow early exercise.
    steps : int, default 500
        Number of time steps in the lattice. More steps increase accuracy and cost.
    q : float, default 0.0
        Continuous dividend yield (annualized).

    Examples
    --------
    >>> # American put has an early-exercise premium over the European put
    >>> amer = BinomialTree(100, 100, 1, 0.05, 0.2, 'put', 'american', steps=500)
    >>> euro = BinomialTree(100, 100, 1, 0.05, 0.2, 'put', 'european', steps=500)
    >>> amer.price() >= euro.price()
    True

    Notes
    -----
    With ``dt = T / steps``, the up/down factors are ``u = exp(sigma*sqrt(dt))``
    and ``d = 1/u``, and the risk-neutral up-probability is
    ``p = (exp((r - q) * dt) - d) / (u - d)``. Backward induction discounts by
    ``exp(-r * dt)`` each step; American nodes take ``max(continuation, intrinsic)``.
    """

    def __init__(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal["call", "put"] = "call",
        exercise: Literal["european", "american"] = "european",
        steps: int = 500,
        q: float = 0.0,
    ):
        super().__init__(S, K, T, r, sigma, q=q, option_type=option_type)
        if not isinstance(steps, (int, np.integer)) or steps < 1:
            raise ValueError("steps must be a positive integer")
        if exercise.lower() not in ("european", "american"):
            raise ValueError("exercise must be 'european' or 'american'")
        self.steps = int(steps)
        self.exercise = exercise.lower()
        if self.steps > 5000:
            warnings.warn("steps > 5000 may be slow or memory-intensive")

    def _lattice(self):
        """
        Run the backward induction and return ``(price, nodes)``.

        ``nodes`` retains the early lattice values used for analytic Greeks:
        ``step1`` = [V_u, V_d], ``step2`` = [V_uu, V_ud, V_dd], plus ``u``, ``d``,
        ``dt``.
        """
        N = self.steps
        dt = self.T / N
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1.0 / u
        disc = np.exp(-self.r * dt)
        p = (np.exp((self.r - self.q) * dt) - d) / (u - d)

        if not (0.0 <= p <= 1.0):
            warnings.warn(
                "Risk-neutral probability p outside [0, 1]; increase steps or "
                "check inputs (results may be unreliable)"
            )

        # Terminal underlying prices and payoffs (j = number of down-moves).
        j = np.arange(N + 1)
        S_T = self.S * u ** (N - j) * d**j
        values = self._intrinsic(S_T)

        nodes = {"u": u, "d": d, "dt": dt}
        for i in range(N - 1, -1, -1):
            values = disc * (p * values[:-1] + (1.0 - p) * values[1:])
            if self.exercise == "american":
                jj = np.arange(i + 1)
                S_i = self.S * u ** (i - jj) * d**jj
                values = np.maximum(values, self._intrinsic(S_i))
            if i == 2:
                nodes["step2"] = values.copy()
            elif i == 1:
                nodes["step1"] = values.copy()

        price = float(values[0])
        nodes["price"] = price
        return price, nodes

    def price(self) -> float:
        """Return the option price from the CRR lattice."""
        return self._lattice()[0]

    def delta(self) -> float:
        """Delta read from the first lattice layer (finite difference)."""
        price, n = self._lattice()
        if "step1" not in n:
            return self._bump_spot_delta()
        v_u, v_d = n["step1"]
        s_u, s_d = self.S * n["u"], self.S * n["d"]
        return (v_u - v_d) / (s_u - s_d)

    def gamma(self) -> float:
        """Gamma read from the second lattice layer (finite difference)."""
        price, n = self._lattice()
        if "step2" not in n:
            return self._bump_spot_gamma()
        v_uu, v_ud, v_dd = n["step2"]
        u, d = n["u"], n["d"]
        s_uu, s_ud, s_dd = self.S * u * u, self.S * u * d, self.S * d * d
        delta_up = (v_uu - v_ud) / (s_uu - s_ud)
        delta_dn = (v_ud - v_dd) / (s_ud - s_dd)
        return (delta_up - delta_dn) / (0.5 * (s_uu - s_dd))

    def theta(self) -> float:
        """Theta (per year) from the lattice: (V_ud - V0) / (2·dt)."""
        price, n = self._lattice()
        if "step2" not in n:
            return float("nan")
        v_ud = n["step2"][1]
        return (v_ud - price) / (2.0 * n["dt"])

    def vega(self, d_sigma: float = 1e-4) -> float:
        """Vega via central-difference bump-and-reprice, scaled per 1% vol."""
        up = self._reprice(sigma=self.sigma + d_sigma)
        dn = self._reprice(sigma=self.sigma - d_sigma)
        return (up - dn) / (2.0 * d_sigma) / 100.0

    def rho(self, d_r: float = 1e-4) -> float:
        """Rho via central-difference bump-and-reprice, scaled per 1% rate."""
        up = self._reprice(r=self.r + d_r)
        dn = self._reprice(r=self.r - d_r)
        return (up - dn) / (2.0 * d_r) / 100.0

    def greeks(self) -> Dict[str, float]:
        """Return price and all Greeks in a single dict."""
        return {
            "price": self.price(),
            "delta": self.delta(),
            "gamma": self.gamma(),
            "theta": self.theta(),
            "vega": self.vega(),
            "rho": self.rho(),
        }

    def early_exercise_premium(self) -> float:
        """American price minus the otherwise-identical European price (>= 0)."""
        amer = self._reprice(exercise="american")
        euro = self._reprice(exercise="european")
        return amer - euro

    def _reprice(self, **overrides) -> float:
        """Build a sibling tree with overridden parameters and return its price."""
        params = dict(
            S=self.S,
            K=self.K,
            T=self.T,
            r=self.r,
            sigma=self.sigma,
            option_type=self.option_type,
            exercise=self.exercise,
            steps=self.steps,
            q=self.q,
        )
        params.update(overrides)
        return BinomialTree(**params).price()

    def _bump_spot_delta(self, h: float = 1e-4) -> float:
        up = BinomialTree(
            self.S + h,
            self.K,
            self.T,
            self.r,
            self.sigma,
            self.option_type,
            self.exercise,
            self.steps,
            self.q,
        ).price()
        dn = BinomialTree(
            self.S - h,
            self.K,
            self.T,
            self.r,
            self.sigma,
            self.option_type,
            self.exercise,
            self.steps,
            self.q,
        ).price()
        return (up - dn) / (2.0 * h)

    def _bump_spot_gamma(self, h: float = 1e-2) -> float:
        up = BinomialTree(
            self.S + h,
            self.K,
            self.T,
            self.r,
            self.sigma,
            self.option_type,
            self.exercise,
            self.steps,
            self.q,
        ).price()
        mid = self.price()
        dn = BinomialTree(
            self.S - h,
            self.K,
            self.T,
            self.r,
            self.sigma,
            self.option_type,
            self.exercise,
            self.steps,
            self.q,
        ).price()
        return (up - 2.0 * mid + dn) / (h * h)

    def __repr__(self) -> str:
        return (
            f"BinomialTree({self.exercise.capitalize()} "
            f"{self.option_type.capitalize()}: S={self.S}, K={self.K}, "
            f"T={self.T:.3f}, r={self.r:.3f}, σ={self.sigma:.3f}, "
            f"q={self.q:.3f}, steps={self.steps})"
        )

    def summary(self) -> str:
        """Return a formatted summary including price and all Greeks."""
        g = self.greeks()
        premium = self.early_exercise_premium()
        return (
            f"Binomial Tree {self.exercise.capitalize()} "
            f"{self.option_type.capitalize()} Option Summary\n"
            f"{'=' * 50}\n"
            f"Parameters:\n"
            f"  Spot Price (S):     ${self.S:>8.2f}\n"
            f"  Strike Price (K):   ${self.K:>8.2f}\n"
            f"  Time to Expiry (T): {self.T:>8.3f} years\n"
            f"  Risk-free Rate (r): {self.r:>8.2%}\n"
            f"  Dividend Yield (q): {self.q:>8.2%}\n"
            f"  Volatility (σ):     {self.sigma:>8.2%}\n"
            f"  Steps:              {self.steps:>8d}\n\n"
            f"Valuation:\n"
            f"  Option Price:       ${g['price']:>8.2f}\n"
            f"  Early Ex. Premium:  ${premium:>8.4f}\n\n"
            f"Greeks:\n"
            f"  Delta (Δ):          {g['delta']:>8.4f}\n"
            f"  Gamma (Γ):          {g['gamma']:>8.4f}\n"
            f"  Theta (Θ):          ${g['theta']:>8.2f} /year\n"
            f"  Vega (ν):           ${g['vega']:>8.2f} /1% vol\n"
            f"  Rho (ρ):            ${g['rho']:>8.2f} /1% rate"
        )
