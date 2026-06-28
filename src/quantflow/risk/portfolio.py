"""
Multi-asset portfolio analytics and mean-variance optimization.

Provides covariance/correlation analysis, portfolio return/volatility/Sharpe
given weights, and mean-variance optimization (minimum-variance, maximum-Sharpe,
and the efficient frontier) using ``scipy.optimize`` only. Bridges to
:class:`~quantflow.risk.metrics.RiskMetrics` so every risk metric is available
for an optimized portfolio.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .metrics import RiskMetrics


class Portfolio:
    """
    Multi-asset portfolio analytics and optimization.

    Parameters
    ----------
    returns : pandas.DataFrame or numpy.ndarray
        Asset returns with shape (observations, assets). At least 2 assets.
    weights : array-like, optional
        Portfolio weights (must sum to 1). Defaults to equal weighting.
    periods_per_year : int, default 252
        Periods per year for annualization (252 daily, 12 monthly, ...).

    Examples
    --------
    >>> import pandas as pd
    >>> from quantflow import MarketData, Portfolio
    >>> prices = MarketData.fetch_stock_data(['AAPL', 'MSFT', 'GOOGL'], period='1y')
    >>> rets = MarketData.calculate_returns(prices)
    >>> port = Portfolio(rets)
    >>> result = port.max_sharpe()
    >>> result['weights_by_asset']  # doctest: +SKIP
    {'AAPL': 0.41, 'MSFT': 0.39, 'GOOGL': 0.20}
    """

    def __init__(self, returns, weights=None, periods_per_year: int = 252):
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

        if weights is None:
            self.weights = np.repeat(1.0 / self.n_assets, self.n_assets)
        else:
            self.weights = self._validate_weights(weights)

        if len(self.returns) < self.n_assets:
            print(
                f"Warning: only {len(self.returns)} observations for "
                f"{self.n_assets} assets; covariance may be unstable"
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
        return vol * np.sqrt(self.periods_per_year) if annualized else vol

    def _sharpe(self, w, risk_free_rate):
        vol = self._vol(w)
        if vol == 0:
            return -1e9
        return (self._ret(w) - risk_free_rate) / vol

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
        )

    def _result(self, res, risk_free_rate=0.02):
        w = res.x
        if not res.success:
            print(f"Warning: optimizer did not converge: {res.message}")
        return {
            "weights": w,
            "weights_by_asset": {a: float(wi) for a, wi in zip(self.assets, w)},
            "return": self._ret(w),
            "volatility": self._vol(w),
            "sharpe": self._sharpe(w, risk_free_rate),
            "success": bool(res.success),
        }

    # ------------------------------------------------------------- analytics
    def covariance_matrix(self, annualized=True) -> pd.DataFrame:
        """Return the (optionally annualized) covariance matrix."""
        factor = self.periods_per_year if annualized else 1
        return self.returns.cov() * factor

    def correlation_matrix(self) -> pd.DataFrame:
        """Return the asset correlation matrix."""
        return self.returns.corr()

    def portfolio_return(self, weights=None, annualized=True) -> float:
        """Portfolio expected return for the given weights."""
        return self._ret(self._resolve(weights), annualized)

    def portfolio_volatility(self, weights=None, annualized=True) -> float:
        """Portfolio volatility for the given weights."""
        return self._vol(self._resolve(weights), annualized)

    def portfolio_sharpe(self, weights=None, risk_free_rate=0.02) -> float:
        """Annualized Sharpe ratio for the given weights."""
        return self._sharpe(self._resolve(weights), risk_free_rate)

    def portfolio_returns_series(self, weights=None) -> pd.Series:
        """Weighted portfolio return series (one value per observation)."""
        w = self._resolve(weights)
        return pd.Series(self.returns.to_numpy() @ w, index=self.returns.index)

    def risk_metrics(self, weights=None) -> RiskMetrics:
        """
        Return a :class:`RiskMetrics` object for the weighted portfolio series.

        Lets every RiskMetrics measure (Sortino, Calmar, VaR, drawdown, ...) be
        computed on an optimized portfolio for free.
        """
        return RiskMetrics(
            self.portfolio_returns_series(weights),
            periods_per_year=self.periods_per_year,
        )

    # ----------------------------------------------------------- optimization
    def min_variance(self, allow_short=False) -> dict:
        """Minimum-variance portfolio (weights sum to 1)."""
        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
        res = self._solve(
            lambda w: self._vol(w, annualized=False) ** 2, constraints, allow_short
        )
        return self._result(res)

    def max_sharpe(self, risk_free_rate=0.02, allow_short=False) -> dict:
        """Maximum-Sharpe (tangency) portfolio (weights sum to 1)."""
        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
        res = self._solve(
            lambda w: -self._sharpe(w, risk_free_rate), constraints, allow_short
        )
        return self._result(res, risk_free_rate)

    def efficient_frontier(self, n_points=50, allow_short=False) -> pd.DataFrame:
        """
        Trace the efficient frontier.

        Returns a DataFrame with columns ``return``, ``volatility``, ``sharpe``,
        and one weight column per asset (``w_<asset>``), sweeping target returns
        from the minimum-variance return to the best single-asset return.
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
            res = self._solve(
                lambda w: self._vol(w, annualized=False) ** 2, constraints, allow_short
            )
            if not res.success:
                continue
            w = res.x
            row = {
                "return": self._ret(w),
                "volatility": self._vol(w),
                "sharpe": self._sharpe(w, 0.02),
            }
            row.update({f"w_{a}": float(wi) for a, wi in zip(self.assets, w)})
            rows.append(row)

        return pd.DataFrame(rows)

    def optimize(
        self,
        objective="max_sharpe",
        target_return=None,
        risk_free_rate=0.02,
        allow_short=False,
    ) -> dict:
        """
        Dispatch to an optimization objective.

        Parameters
        ----------
        objective : {'max_sharpe', 'min_variance', 'target_return'}
        target_return : float, optional
            Required when ``objective='target_return'`` (annualized).
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
            res = self._solve(
                lambda w: self._vol(w, annualized=False) ** 2, constraints, allow_short
            )
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
