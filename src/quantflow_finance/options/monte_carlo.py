"""
Monte Carlo pricing of European options under geometric Brownian motion.

The terminal price is sampled exactly,

    S_T = S exp((r - q - sigma^2 / 2) T + sigma sqrt(T) Z),   Z ~ N(0, 1),

so there is no time-discretisation bias; the only error is statistical and is
reported as a standard error. For European vanilla options the closed form in
:class:`~quantflow_finance.BlackScholes` is exact and much cheaper. This pricer exists
to show and test Monte Carlo convergence and variance reduction on a problem
with a known answer.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import norm

from .base import OptionPricer


@dataclass(frozen=True)
class MonteCarloResult:
    """
    Outcome of a Monte Carlo run.

    Attributes
    ----------
    price : float
        Estimated option price.
    std_error : float
        Standard error of ``price`` (sample standard deviation of the
        independent estimates divided by the square root of their number).
    ci_low, ci_high : float
        Two-sided normal-approximation confidence interval for the price.
    confidence : float
        Confidence level of the interval, e.g. 0.95.
    n_paths : int
        Number of simulated terminal prices.
    """

    price: float
    std_error: float
    ci_low: float
    ci_high: float
    confidence: float
    n_paths: int


class MonteCarlo(OptionPricer):
    r"""
    European option price by Monte Carlo simulation.

    Parameters
    ----------
    S, K, T, r, sigma, option_type, q
        As for :class:`~quantflow_finance.BlackScholes`: spot, strike, year fraction
        to expiry, continuously compounded rate, annualised volatility,
        ``'call'``/``'put'`` and continuous dividend yield.
    n_paths : int, default 100_000
        Number of simulated terminal prices. Must be even when
        ``antithetic=True``. All paths are held in memory at once.
    antithetic : bool, default False
        Pair every normal draw ``Z`` with ``-Z`` and average the two
        discounted payoffs. The pairs are the independent samples used for
        the standard error.
    control_variate : bool, default False
        Use the discounted terminal price :math:`e^{-rT} S_T`, whose mean
        :math:`S e^{-qT}` is known, as a control variate. The coefficient is
        the sample regression slope, which adds a bias of order
        :math:`1/n` (negligible next to the standard error).
    seed : int or None, default None
        Seed for :func:`numpy.random.default_rng`. With a seed every call to
        :meth:`simulate` returns the same result.

    Examples
    --------
    >>> from quantflow_finance import MonteCarlo
    >>> mc = MonteCarlo(100, 100, 1.0, 0.05, 0.2, n_paths=200_000, seed=7)
    >>> res = mc.simulate()
    >>> print(f"{res.price:.4f} +/- {res.std_error:.4f}")
    10.4479 +/- 0.0329

    References
    ----------
    Glasserman, P. (2003). Monte Carlo Methods in Financial Engineering,
    sections 3.2 (exact GBM sampling) and 4.1-4.2 (control variates,
    antithetic variates). Springer.
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
        n_paths: int = 100_000,
        antithetic: bool = False,
        control_variate: bool = False,
        seed: int | None = None,
    ):
        super().__init__(S, K, T, r, sigma, q=q, option_type=option_type)
        if isinstance(n_paths, bool) or not isinstance(n_paths, (int, np.integer)):
            raise ValueError("n_paths must be an integer")
        if n_paths < 2:
            raise ValueError("n_paths must be at least 2")
        if antithetic and n_paths % 2:
            raise ValueError("n_paths must be even when antithetic=True")
        if antithetic and n_paths < 4:
            raise ValueError("n_paths must be at least 4 when antithetic=True")
        self.n_paths = int(n_paths)
        self.antithetic = bool(antithetic)
        self.control_variate = bool(control_variate)
        self.seed = seed

    def simulate(self, confidence: float = 0.95) -> MonteCarloResult:
        """
        Run the simulation.

        Parameters
        ----------
        confidence : float, default 0.95
            Confidence level of the reported interval, in (0, 1).

        Returns
        -------
        MonteCarloResult
            Price, standard error and confidence interval.
        """
        if not 0 < confidence < 1:
            raise ValueError("confidence must be between 0 and 1")
        rng = np.random.default_rng(self.seed)
        n_draws = self.n_paths // 2 if self.antithetic else self.n_paths
        z = rng.standard_normal(n_draws)

        drift = (self.r - self.q - 0.5 * self.sigma**2) * self.T
        vol = self.sigma * np.sqrt(self.T)
        disc = np.exp(-self.r * self.T)

        def discounted(z_):
            s_t = self.S * np.exp(drift + vol * z_)
            return disc * self._intrinsic(s_t), disc * s_t

        y, x = discounted(z)
        if self.antithetic:
            y_anti, x_anti = discounted(-z)
            y = 0.5 * (y + y_anti)
            x = 0.5 * (x + x_anti)

        if self.control_variate:
            x_mean = self.S * np.exp(-self.q * self.T)
            x_c = x - x.mean()
            beta = float(np.dot(x_c, y - y.mean()) / np.dot(x_c, x_c))
            y = y - beta * (x - x_mean)

        price = float(y.mean())
        std_error = float(y.std(ddof=1) / np.sqrt(y.size))
        half_width = float(norm.ppf(0.5 + confidence / 2)) * std_error
        return MonteCarloResult(
            price=price,
            std_error=std_error,
            ci_low=price - half_width,
            ci_high=price + half_width,
            confidence=confidence,
            n_paths=self.n_paths,
        )

    def price(self) -> float:
        """Estimated option price (``simulate().price``)."""
        return self.simulate().price

    def __repr__(self) -> str:
        return (
            f"MonteCarlo({self.option_type.capitalize()}: S={self.S}, K={self.K}, "
            f"T={self.T:.3f}, r={self.r:.3f}, sigma={self.sigma:.3f}, "
            f"q={self.q:.3f}, n_paths={self.n_paths}, "
            f"antithetic={self.antithetic}, control_variate={self.control_variate})"
        )
