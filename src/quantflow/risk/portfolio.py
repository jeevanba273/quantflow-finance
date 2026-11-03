"""
Mean-variance analysis of a portfolio of several assets.

Expected returns and the covariance matrix are the sample mean and sample
covariance (``ddof=1``) of the historical returns, annualised with
``m = periods_per_year`` (mean times ``m``, covariance times ``m``). Weights
always sum to 1. Long-only problems are solved numerically with SLSQP; with
short sales allowed the tangency portfolio uses its closed form.
"""

import warnings

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.optimize import OptimizeResult, minimize

from .metrics import RiskMetrics


class Portfolio:
    r"""
    Portfolio statistics and mean-variance optimisation.

    Parameters
    ----------
    returns : pandas.DataFrame or array_like
        Periodic simple returns, shape (observations, assets), at least two
        assets. Rows containing a NaN are dropped.
    weights : array_like, optional
        Default weights for the analytics methods; must sum to 1. Equal
        weights if omitted.
    periods_per_year : int, default 252
        Periods per year used for annualisation.

    Notes
    -----
    For weights :math:`w`, annualised mean vector :math:`\mu` and covariance
    :math:`\Sigma`, the portfolio return is :math:`w^\top \mu`, the
    volatility :math:`\sqrt{w^\top \Sigma w}` and the Sharpe ratio
    :math:`(w^\top \mu - r_f) / \sqrt{w^\top \Sigma w}`.

    Sample means are noisy estimates of expected returns, and optimised
    weights are very sensitive to them (Michaud, 1989). Treat the output as
    a function of the inputs, not as a forecast.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from quantflow import Portfolio
    >>> rng = np.random.default_rng(0)
    >>> rets = pd.DataFrame(rng.normal(0.0004, 0.01, (500, 3)), columns=list("ABC"))
    >>> w = Portfolio(rets).min_variance()["weights"]
    >>> print(np.round(w, 3), round(w.sum(), 12))
    [0.337 0.346 0.318] 1.0

    References
    ----------
    Markowitz, H. (1952). Portfolio Selection. Journal of Finance, 7(1), 77-91.

    Merton, R. C. (1972). An Analytic Derivation of the Efficient Portfolio
    Frontier. Journal of Financial and Quantitative Analysis, 7(4), 1851-1872.

    Michaud, R. O. (1989). The Markowitz Optimization Enigma: Is 'Optimized'
    Optimal? Financial Analysts Journal, 45(1), 31-42.
    """

    def __init__(
        self,
        returns: ArrayLike,
        weights: ArrayLike | None = None,
        periods_per_year: int = 252,
    ):
        if isinstance(returns, pd.DataFrame):
            self.returns = returns.dropna()
        else:
            arr = np.asarray(returns, dtype=float)
            if arr.ndim != 2:
                raise ValueError("returns must be 2-D (observations x assets)")
            self.returns = pd.DataFrame(arr).dropna()

        if self.returns.empty:
            raise ValueError("No valid return data provided")
        if self.returns.shape[1] < 2:
            raise ValueError(
                "Portfolio requires at least 2 assets; use RiskMetrics for a single asset"
            )
        if periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive")

        self.periods_per_year = int(periods_per_year)
        self.assets = list(self.returns.columns)
        self._mu = self.returns.mean().to_numpy()
        self._cov = self.returns.cov().to_numpy()
        # Normalizer that keeps the variance objective O(1) regardless of return
        # frequency, so SLSQP's tolerance is meaningful (daily variances are tiny).
        self._var_scale = 1.0 / max(float(np.mean(np.diag(self._cov))), 1e-300)

        if weights is None:
            self.weights = np.repeat(1.0 / self.n_assets, self.n_assets)
        else:
            self.weights = self._validate_weights(weights)

        if len(self.returns) < self.n_assets:
            warnings.warn(
                f"Only {len(self.returns)} observations for {self.n_assets} "
                "assets; the covariance matrix may be unstable",
                stacklevel=2,
            )

    @property
    def n_assets(self) -> int:
        """Number of assets in the portfolio."""
        return self.returns.shape[1]

    # ------------------------------------------------------------------ helpers
    def _validate_weights(self, weights):
        w = np.asarray(weights, dtype=float).flatten()
        if len(w) != self.n_assets:
            raise ValueError(f"weights must have length {self.n_assets}")
        if not np.all(np.isfinite(w)):
            raise ValueError("weights must be finite")
        if not np.isclose(w.sum(), 1.0, atol=1e-6):
            raise ValueError("weights must sum to 1")
        return w

    def _resolve(self, weights):
        if weights is None:
            return self.weights
        return self._validate_weights(weights)

    def _ret(self, w, annualized=True):
        """Raw (unvalidated) portfolio return for optimization."""
        r = float(w @ self._mu)
        return r * self.periods_per_year if annualized else r

    def _vol(self, w, annualized=True):
        """Raw (unvalidated) portfolio volatility for optimization."""
        var = float(w @ self._cov @ w)
        var = max(var, 0.0)
        vol = np.sqrt(var)
        return float(vol * np.sqrt(self.periods_per_year) if annualized else vol)

    def _sharpe(self, w, risk_free_rate):
        vol = self._vol(w)
        if vol == 0:
            return -1e9
        return (self._ret(w) - risk_free_rate) / vol

    def _variance_obj(self, w):
        """Scaled portfolio variance, kept O(1) for SLSQP."""
        return self._var_scale * float(w @ self._cov @ w)

    def _bounds(self, allow_short):
        return (
            [(None, None)] * self.n_assets
            if allow_short
            else [(0.0, 1.0)] * self.n_assets
        )

    def _solve(self, objective, constraints, allow_short):
        x0 = np.repeat(1.0 / self.n_assets, self.n_assets)
        return minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=self._bounds(allow_short),
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 1000},
        )

    def _result(self, res, risk_free_rate=0.02):
        w = res.x
        if not res.success:
            warnings.warn(
                f"Optimizer did not converge: {res.message}",
                RuntimeWarning,
                stacklevel=3,
            )
        return {
            "weights": w,
            "weights_by_asset": {
                a: float(wi) for a, wi in zip(self.assets, w, strict=True)
            },
            "return": self._ret(w),
            "volatility": self._vol(w),
            "sharpe": self._sharpe(w, risk_free_rate),
            "success": bool(res.success),
        }

    # ------------------------------------------------------------- analytics
    def covariance_matrix(self, annualized: bool = True) -> pd.DataFrame:
        """Sample covariance matrix (``ddof=1``), times ``m`` if annualised."""
        factor = self.periods_per_year if annualized else 1
        return self.returns.cov() * factor

    def correlation_matrix(self) -> pd.DataFrame:
        """Sample correlation matrix."""
        return self.returns.corr()

    def portfolio_return(
        self, weights: ArrayLike | None = None, annualized: bool = True
    ) -> float:
        """Expected return ``w' mu`` (annualised by default)."""
        return self._ret(self._resolve(weights), annualized)

    def portfolio_volatility(
        self, weights: ArrayLike | None = None, annualized: bool = True
    ) -> float:
        """Volatility ``sqrt(w' Sigma w)`` (annualised by default)."""
        return self._vol(self._resolve(weights), annualized)

    def portfolio_sharpe(
        self, weights: ArrayLike | None = None, risk_free_rate: float = 0.02
    ) -> float:
        """Annualised Sharpe ratio ``(w' mu - rf) / sqrt(w' Sigma w)``."""
        return self._sharpe(self._resolve(weights), risk_free_rate)

    def portfolio_returns_series(self, weights: ArrayLike | None = None) -> pd.Series:
        """Per-period portfolio returns with constant weights (rebalanced
        every period)."""
        w = self._resolve(weights)
        return pd.Series(self.returns.to_numpy() @ w, index=self.returns.index)

    def risk_metrics(self, weights: ArrayLike | None = None) -> RiskMetrics:
        """:class:`RiskMetrics` of :meth:`portfolio_returns_series`."""
        return RiskMetrics(
            self.portfolio_returns_series(weights),
            periods_per_year=self.periods_per_year,
        )

    # ----------------------------------------------------------- optimization
    def min_variance(self, allow_short: bool = False) -> dict:
        r"""
        Minimum-variance portfolio.

        With ``allow_short=True`` this is the global minimum-variance
        portfolio :math:`\Sigma^{-1}\mathbf 1 / (\mathbf 1^\top \Sigma^{-1}
        \mathbf 1)`; otherwise weights are also bounded to [0, 1].

        Returns
        -------
        dict
            ``weights`` (ndarray), ``weights_by_asset``, annualised
            ``return`` and ``volatility``, ``sharpe`` (at rf = 2%) and
            ``success``.
        """
        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
        res = self._solve(self._variance_obj, constraints, allow_short)
        return self._result(res)

    def max_sharpe(
        self, risk_free_rate: float = 0.02, allow_short: bool = False
    ) -> dict:
        r"""
        Maximum-Sharpe (tangency) portfolio.

        With ``allow_short=True`` the weights are the closed form
        :math:`\Sigma^{-1}(\mu - r_f \mathbf 1)` normalised to sum to 1. Long
        only, the ratio is maximised numerically (SLSQP from equal weights).

        Parameters
        ----------
        risk_free_rate : float, default 0.02
            Annual risk-free rate.
        allow_short : bool, default False
            Allow negative weights.

        Returns
        -------
        dict
            Same keys as :meth:`min_variance`.

        Raises
        ------
        ValueError
            With ``allow_short=True``, if the minimum-variance portfolio's
            return is below ``risk_free_rate``: no fully invested portfolio
            then attains the supremum of the Sharpe ratio.
        """
        if allow_short:
            return self._tangency(risk_free_rate)
        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
        res = self._solve(
            lambda w: -self._sharpe(w, risk_free_rate), constraints, allow_short
        )
        return self._result(res, risk_free_rate)

    def _tangency(self, risk_free_rate):
        """Closed-form tangency portfolio, w proportional to S^-1 (mu - rf)."""
        m = self.periods_per_year
        try:
            x = np.linalg.solve(self._cov * m, self._mu * m - risk_free_rate)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Covariance matrix is singular") from exc
        if x.sum() <= 0:
            raise ValueError(
                "No maximum-Sharpe portfolio with weights summing to 1: the "
                "minimum-variance portfolio's return is below the risk-free "
                "rate, so with short sales the Sharpe ratio can only be "
                "approached with unbounded long-short positions"
            )
        res = OptimizeResult(x=x / x.sum(), success=True, message="closed form")
        return self._result(res, risk_free_rate)

    def efficient_frontier(
        self,
        n_points: int = 50,
        allow_short: bool = False,
        risk_free_rate: float = 0.02,
    ) -> pd.DataFrame:
        """
        Minimum-variance portfolios for a grid of target returns.

        Targets run in equal steps from the minimum-variance portfolio's
        return to the highest single-asset mean return. Targets the solver
        cannot reach are left out, so the frame can have fewer than
        ``n_points`` rows.

        Parameters
        ----------
        n_points : int, default 50
            Number of target returns.
        allow_short : bool, default False
            Allow negative weights.
        risk_free_rate : float, default 0.02
            Annual rate used for the ``sharpe`` column.

        Returns
        -------
        pandas.DataFrame
            Columns ``return``, ``volatility``, ``sharpe`` (annualised) and
            ``w_<asset>`` for each asset.
        """
        if n_points < 2:
            raise ValueError("n_points must be at least 2")

        r_min = self.min_variance(allow_short)["return"]
        r_max = self.periods_per_year * float(self._mu.max())
        targets = np.linspace(r_min, r_max, n_points)

        rows = []
        for target in targets:
            constraints = [
                {"type": "eq", "fun": lambda w: w.sum() - 1.0},
                {"type": "eq", "fun": lambda w, t=target: self._ret(w) - t},
            ]
            res = self._solve(self._variance_obj, constraints, allow_short)
            if not res.success:
                continue
            w = res.x
            row = {
                "return": self._ret(w),
                "volatility": self._vol(w),
                "sharpe": self._sharpe(w, risk_free_rate),
            }
            row.update(
                {f"w_{a}": float(wi) for a, wi in zip(self.assets, w, strict=True)}
            )
            rows.append(row)

        return pd.DataFrame(rows)

    def optimize(
        self,
        objective: str = "max_sharpe",
        target_return: float | None = None,
        risk_free_rate: float = 0.02,
        allow_short: bool = False,
    ) -> dict:
        """
        Run one of the optimisers by name.

        Parameters
        ----------
        objective : {'max_sharpe', 'min_variance', 'target_return'}
            ``'target_return'`` minimises variance subject to an annualised
            expected return of ``target_return``.
        target_return : float, optional
            Required for ``objective='target_return'``.
        risk_free_rate : float, default 0.02
            Annual rate for the Sharpe ratio.
        allow_short : bool, default False
            Allow negative weights.

        Returns
        -------
        dict
            Same keys as :meth:`min_variance`.
        """
        if objective == "min_variance":
            return self.min_variance(allow_short)
        if objective == "max_sharpe":
            return self.max_sharpe(risk_free_rate, allow_short)
        if objective == "target_return":
            if target_return is None:
                raise ValueError(
                    "target_return is required for objective='target_return'"
                )
            constraints = [
                {"type": "eq", "fun": lambda w: w.sum() - 1.0},
                {"type": "eq", "fun": lambda w: self._ret(w) - target_return},
            ]
            res = self._solve(self._variance_obj, constraints, allow_short)
            return self._result(res, risk_free_rate)
        raise ValueError(
            "objective must be 'max_sharpe', 'min_variance', or 'target_return'"
        )

    def __repr__(self) -> str:
        return (
            f"Portfolio(n_assets={self.n_assets}, "
            f"observations={len(self.returns)}, "
            f"periods_per_year={self.periods_per_year})"
        )
