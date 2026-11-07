"""
Cox-Ross-Rubinstein (CRR) binomial tree for European and American options.

European prices converge to the Black-Scholes-Merton value as ``steps``
grows, with an error of order 1/steps that oscillates in sign. American
options are priced by checking early exercise at every node.
"""

import warnings
from typing import Literal

import numpy as np

from .base import OptionPricer


class BinomialTree(OptionPricer):
    r"""
    Cox-Ross-Rubinstein binomial tree pricer.

    Parameters
    ----------
    S : float
        Spot price, > 0.
    K : float
        Strike, > 0.
    T : float
        Time to expiry as a year fraction, > 0.
    r : float
        Continuously compounded risk-free rate per year.
    sigma : float
        Annualised volatility, > 0.
    option_type : {'call', 'put'}, default 'call'
        Option type.
    exercise : {'european', 'american'}, default 'european'
        Exercise style. American options may be exercised at any node.
    steps : int, default 500
        Number of time steps. Cost grows as ``steps**2`` for American
        options and the lattice Greeks need ``steps >= 2``.
    q : float, default 0.0
        Continuous dividend yield per year. Discrete dividends are not
        modelled.

    Notes
    -----
    With :math:`\Delta t = T / N`, :math:`u = e^{\sigma\sqrt{\Delta t}}`,
    :math:`d = 1/u` and
    :math:`p = (e^{(r-q)\Delta t} - d) / (u - d)`, values are rolled back with
    :math:`V = e^{-r\Delta t}[p V_u + (1-p) V_d]`; American nodes take
    :math:`\max(V, \text{intrinsic})`. A warning is raised if :math:`p`
    falls outside [0, 1] (too few steps for the rate and volatility).

    Greeks follow the :class:`~quantflow_finance.BlackScholes` units. Delta, gamma
    and theta are read from the first two layers of the tree (Hull's
    formulas); vega and rho are central differences of repriced trees with
    bumps of 1e-4.

    Examples
    --------
    >>> from quantflow_finance import BinomialTree
    >>> put = BinomialTree(50, 50, 5 / 12, 0.10, 0.40, "put", "american", steps=5)
    >>> print(f"{put.price():.2f}")
    4.49

    References
    ----------
    Cox, J. C., Ross, S. A. and Rubinstein, M. (1979). Option Pricing: A
    Simplified Approach. Journal of Financial Economics, 7(3), 229-263.

    Hull, J. C. Options, Futures, and Other Derivatives (10th ed.), chapter
    on basic numerical procedures. Pearson.
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
        if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)):
            raise ValueError("steps must be a positive integer")
        if steps < 1:
            raise ValueError("steps must be a positive integer")
        if not isinstance(exercise, str) or exercise.lower() not in (
            "european",
            "american",
        ):
            raise ValueError("exercise must be 'european' or 'american'")
        self.steps = int(steps)
        self.exercise = exercise.lower()
        if self.steps > 5000:
            warnings.warn("steps > 5000 may be slow or memory-intensive", stacklevel=2)

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
                "check inputs (results may be unreliable)",
                stacklevel=2,
            )

        # Terminal underlying prices and payoffs (j = number of down-moves).
        j = np.arange(N + 1)
        S_T = self.S * u ** (N - j) * d**j
        values = self._intrinsic(S_T)

        nodes = {"u": u, "d": d, "dt": dt}
        # With one or two steps the terminal layer is itself layer 1 or 2.
        if N <= 2:
            nodes[f"step{N}"] = values.copy()
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

    def _delta_from(self, n):
        v_u, v_d = n["step1"]
        s_u, s_d = self.S * n["u"], self.S * n["d"]
        return float((v_u - v_d) / (s_u - s_d))

    def _gamma_from(self, n):
        if "step2" not in n:
            raise ValueError("Lattice gamma needs steps >= 2")
        v_uu, v_ud, v_dd = n["step2"]
        u, d = n["u"], n["d"]
        s_uu, s_ud, s_dd = self.S * u * u, self.S * u * d, self.S * d * d
        delta_up = (v_uu - v_ud) / (s_uu - s_ud)
        delta_dn = (v_ud - v_dd) / (s_ud - s_dd)
        return float((delta_up - delta_dn) / (0.5 * (s_uu - s_dd)))

    def _theta_from(self, n):
        if "step2" not in n:
            raise ValueError("Lattice theta needs steps >= 2")
        return float((n["step2"][1] - n["price"]) / (2.0 * n["dt"]))

    def delta(self) -> float:
        """Delta from the first lattice layer: (V_u - V_d) / (S u - S d)."""
        return self._delta_from(self._lattice()[1])

    def gamma(self) -> float:
        """Gamma from the second lattice layer (difference of layer deltas)."""
        return self._gamma_from(self._lattice()[1])

    def theta(self) -> float:
        """Theta per year from the lattice: (V_ud - V_0) / (2 dt)."""
        return self._theta_from(self._lattice()[1])

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

    def greeks(self) -> dict[str, float]:
        """
        Price and all Greeks in one dict.

        Price, delta, gamma and theta come from one backward induction; vega
        and rho each need two bumped trees, so five trees are built in all.
        """
        price, n = self._lattice()
        return {
            "price": price,
            "delta": self._delta_from(n),
            "gamma": self._gamma_from(n),
            "theta": self._theta_from(n),
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

    def __repr__(self) -> str:
        return (
            f"BinomialTree({self.exercise.capitalize()} "
            f"{self.option_type.capitalize()}: S={self.S}, K={self.K}, "
            f"T={self.T:.3f}, r={self.r:.3f}, sigma={self.sigma:.3f}, "
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
            f"  Volatility (sigma): {self.sigma:>8.2%}\n"
            f"  Steps:              {self.steps:>8d}\n\n"
            f"Valuation:\n"
            f"  Option Price:       ${g['price']:>8.2f}\n"
            f"  Early Ex. Premium:  ${premium:>8.4f}\n\n"
            f"Greeks:\n"
            f"  Delta:              {g['delta']:>8.4f}\n"
            f"  Gamma:              {g['gamma']:>8.4f}\n"
            f"  Theta:              ${g['theta']:>8.2f} /year\n"
            f"  Vega:               ${g['vega']:>8.2f} /1% vol\n"
            f"  Rho:                ${g['rho']:>8.2f} /1% rate"
        )
