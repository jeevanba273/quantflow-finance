"""
Risk and performance statistics for one return series.

Conventions used by every method:

* Inputs are periodic simple returns (0.01 = +1%), one per period.
* VaR and expected shortfall are returned as signed returns: a loss is a
  negative number. The argument ``confidence_level`` is the tail probability
  alpha, so ``0.05`` gives the 95% VaR.
* Annualisation uses ``m = periods_per_year`` (default 252 trading days):
  means are multiplied by ``m``, standard deviations by ``sqrt(m)``, and an
  annual risk-free rate ``rf`` is converted to ``rf / m`` per period.
* Standard deviations are sample standard deviations (``ddof=1``).
"""

import warnings

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats


class RiskMetrics:
    """
    Risk and performance statistics of a return series.

    Parameters
    ----------
    returns : pandas.Series, single-column pandas.DataFrame or array_like
        Periodic simple returns. NaNs are dropped; infinite values are
        rejected.
    periods_per_year : int, default 252
        Periods per year used for annualisation (252 for daily data, 52 for
        weekly, 12 for monthly).

    Raises
    ------
    ValueError
        If no finite returns remain, a DataFrame has more than one column,
        or ``periods_per_year`` is not positive.

    Examples
    --------
    >>> import numpy as np
    >>> from quantflow_finance import RiskMetrics
    >>> rm = RiskMetrics(np.random.default_rng(0).normal(0.0005, 0.01, 1000))
    >>> print(f"{rm.var_historical(0.05):.4f}")
    -0.0151
    """

    def __init__(self, returns: ArrayLike, periods_per_year: int = 252):
        if isinstance(returns, pd.DataFrame):
            if returns.shape[1] != 1:
                raise ValueError(
                    "returns must have exactly one column; select one series or "
                    "use Portfolio for several assets"
                )
            self.returns = returns.iloc[:, 0].dropna()
        elif isinstance(returns, pd.Series):
            self.returns = returns.dropna()
        else:
            returns_flat = np.array(returns, dtype=float).flatten()
            self.returns = pd.Series(returns_flat).dropna()

        if len(self.returns) == 0:
            raise ValueError(
                "No valid return data provided. Cannot calculate risk metrics."
            )
        if not np.all(np.isfinite(self.returns.to_numpy(dtype=float))):
            raise ValueError("returns must not contain infinite values")
        if periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive")
        self.periods_per_year = int(periods_per_year)

        if len(self.returns) < 30:
            warnings.warn(
                f"Only {len(self.returns)} observations; risk metrics may be "
                "unreliable",
                stacklevel=2,
            )

    @staticmethod
    def _check_alpha(confidence_level):
        if not (0 < confidence_level < 1):
            raise ValueError("Confidence level must be between 0 and 1")

    def var_historical(self, confidence_level: float = 0.05) -> float:
        """
        Historical VaR: the empirical ``confidence_level`` quantile of returns.

        Parameters
        ----------
        confidence_level : float, default 0.05
            Tail probability alpha (0.05 for 95% VaR, 0.01 for 99%).

        Returns
        -------
        float
            The alpha-quantile of the returns (negative for a loss), using
            linear interpolation between order statistics
            (``numpy.percentile`` default).
        """
        self._check_alpha(confidence_level)
        return float(np.percentile(self.returns, confidence_level * 100))

    def var_parametric(self, confidence_level: float = 0.05) -> float:
        r"""
        Gaussian VaR, :math:`\hat\mu + z_\alpha \hat\sigma` with
        :math:`z_\alpha = N^{-1}(\alpha)`.

        Assumes normally distributed returns, so it understates the tail of
        fat-tailed returns (see the validation page).

        Parameters
        ----------
        confidence_level : float, default 0.05
            Tail probability alpha.

        Returns
        -------
        float
            Per-period VaR as a signed return.
        """
        self._check_alpha(confidence_level)
        z_score = stats.norm.ppf(confidence_level)
        return float(self.returns.mean() + z_score * self.returns.std())

    def sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        r"""
        Annualised Sharpe ratio.

        .. math::

            \text{Sharpe} = \frac{\hat\mu - r_f / m}{\hat\sigma} \sqrt{m}

        with :math:`\hat\mu` and :math:`\hat\sigma` the per-period sample mean
        and standard deviation.

        Parameters
        ----------
        risk_free_rate : float, default 0.02
            Annual risk-free rate, converted to ``rf / m`` per period.

        Returns
        -------
        float

        References
        ----------
        Sharpe, W. F. (1994). The Sharpe Ratio. Journal of Portfolio
        Management, 21(1), 49-58.
        """
        excess_return = self.returns.mean() - risk_free_rate / self.periods_per_year
        return float(
            excess_return / self.returns.std() * np.sqrt(self.periods_per_year)
        )

    def expected_shortfall(self, confidence_level: float = 0.05) -> float:
        """
        Historical expected shortfall: mean of the returns at or below the
        historical VaR.

        Parameters
        ----------
        confidence_level : float, default 0.05
            Tail probability alpha.

        Returns
        -------
        float
            Mean tail return (negative for a loss); never above the VaR.
        """
        var = self.var_historical(confidence_level)
        tail_returns = self.returns[self.returns <= var]
        if len(tail_returns) == 0:
            return var
        return float(tail_returns.mean())

    def max_drawdown(self) -> float:
        r"""
        Maximum drawdown of the compounded wealth index.

        With :math:`W_0 = 1` and :math:`W_t = \prod_{i \le t} (1 + r_i)`,
        :math:`\text{MDD} = \min_t \bigl(W_t / \max_{s \le t} W_s - 1\bigr)`,
        where the running maximum includes :math:`W_0`.

        Returns
        -------
        float
            Largest peak-to-trough fall as a fraction (``-0.25`` = -25%), or
            0.0 if wealth never falls.
        """
        wealth = (1 + self.returns).cumprod().to_numpy()
        peak = np.maximum.accumulate(np.concatenate(([1.0], wealth)))[1:]
        return float(np.min(wealth / peak - 1.0))

    def volatility(self, annualized: bool = True) -> float:
        r"""
        Sample standard deviation of returns, times :math:`\sqrt{m}` if
        ``annualized``.
        """
        vol = self.returns.std()
        if annualized:
            vol *= np.sqrt(self.periods_per_year)
        return float(vol)

    def _align(self, benchmark_returns):
        """
        Pair portfolio and benchmark returns observation by observation.

        If both series carry a DatetimeIndex they are joined on their common
        dates. Otherwise both are trimmed to their last ``min_length``
        observations and paired by position. Returns ``(portfolio, benchmark)``
        as numpy arrays.
        """
        if isinstance(benchmark_returns, pd.DataFrame):
            if benchmark_returns.shape[1] != 1:
                raise ValueError("benchmark_returns must have exactly one column")
            benchmark_returns = benchmark_returns.iloc[:, 0]
        if (
            isinstance(benchmark_returns, pd.Series)
            and isinstance(self.returns.index, pd.DatetimeIndex)
            and isinstance(benchmark_returns.index, pd.DatetimeIndex)
        ):
            joined = pd.concat(
                [self.returns, benchmark_returns], axis=1, join="inner"
            ).dropna()
            if len(joined) < 2:
                raise ValueError(
                    "Need at least 2 common dates between portfolio and benchmark"
                )
            return (
                joined.iloc[:, 0].to_numpy(dtype=float),
                joined.iloc[:, 1].to_numpy(dtype=float),
            )
        if isinstance(benchmark_returns, pd.Series):
            benchmark_returns = benchmark_returns.to_numpy()
        benchmark_returns = np.asarray(benchmark_returns, dtype=float).flatten()

        min_length = min(len(self.returns), len(benchmark_returns))
        if min_length < 2:
            raise ValueError("Need at least 2 aligned observations for this metric")

        portfolio_rets = self.returns.iloc[-min_length:].to_numpy()
        benchmark_rets = benchmark_returns[-min_length:]
        return portfolio_rets, benchmark_rets

    def information_ratio(self, benchmark_returns: ArrayLike) -> float:
        r"""
        Annualised information ratio against a benchmark.

        With active returns :math:`a_i = r_i - b_i`,
        :math:`\text{IR} = \bar a / s_a \cdot \sqrt{m}`, where :math:`s_a` is
        the sample standard deviation (tracking error).

        Parameters
        ----------
        benchmark_returns : pandas.Series or array_like
            Benchmark returns. Date-indexed series are matched on common
            dates; arrays are matched by position from the end.

        Returns
        -------
        float
            0.0 if the tracking error is zero.
        """
        portfolio_rets, benchmark_rets = self._align(benchmark_returns)
        excess_returns = portfolio_rets - benchmark_rets
        tracking_error = excess_returns.std(ddof=1)
        if tracking_error == 0:
            return 0.0
        return float(
            excess_returns.mean() / tracking_error * np.sqrt(self.periods_per_year)
        )

    def sortino_ratio(self, risk_free_rate: float = 0.02, target: float = 0.0) -> float:
        r"""
        Annualised Sortino ratio.

        .. math::

            \sigma_d = \sqrt{\frac{1}{n} \sum_{i=1}^{n} \min(r_i - \tau, 0)^2},
            \qquad
            \text{Sortino} = \frac{\hat\mu - r_f / m}{\sigma_d} \sqrt{m}

        The downside deviation :math:`\sigma_d` is a root mean square over
        all ``n`` observations (returns above the target count as zero).
        Note that the numerator subtracts the risk-free rate while the
        downside deviation is measured against ``target``; pass
        ``target=risk_free_rate / periods_per_year`` to use the same
        threshold in both, as in Sortino and Price (1994).

        Parameters
        ----------
        risk_free_rate : float, default 0.02
            Annual risk-free rate.
        target : float, default 0.0
            Per-period minimum acceptable return for the downside deviation.

        Returns
        -------
        float
            ``inf`` if no return is below ``target``.

        References
        ----------
        Sortino, F. A. and Price, L. N. (1994). Performance Measurement in a
        Downside Risk Framework. Journal of Investing, 3(3), 59-64.
        """
        excess_return = self.returns.mean() - risk_free_rate / self.periods_per_year
        downside = np.sqrt(np.mean(np.minimum(self.returns - target, 0.0) ** 2))
        if downside == 0:
            return np.inf
        return float(excess_return / downside * np.sqrt(self.periods_per_year))

    def calmar_ratio(self, risk_free_rate: float = 0.0) -> float:
        r"""
        Calmar ratio over the whole sample.

        :math:`\text{CAGR} = \prod (1 + r_i)^{m/n} - 1` and
        :math:`\text{Calmar} = (\text{CAGR} - r_f) / |\text{MDD}|`. The
        original definition uses a trailing 36-month window; this method uses
        all observations.

        Parameters
        ----------
        risk_free_rate : float, default 0.0
            Annual rate subtracted from the CAGR.

        Returns
        -------
        float
            ``inf`` if there is no drawdown.

        References
        ----------
        Young, T. W. (1991). Calmar Ratio: A Smoother Tool. Futures, 20(1).
        """
        n = len(self.returns)
        cagr = (1 + self.returns).prod() ** (self.periods_per_year / n) - 1
        mdd = abs(self.max_drawdown())
        if mdd == 0:
            return np.inf
        return float((cagr - risk_free_rate) / mdd)

    def omega_ratio(self, threshold: float = 0.0) -> float:
        r"""
        Omega ratio at a per-period threshold :math:`\tau` (not annualised).

        :math:`\Omega = \sum \max(r_i - \tau, 0) / \sum \max(\tau - r_i, 0)`.

        Returns
        -------
        float
            ``inf`` if there are no returns below the threshold, 0.0 if none
            above it.

        References
        ----------
        Keating, C. and Shadwick, W. F. (2002). A Universal Performance
        Measure. Journal of Performance Measurement, 6(3), 59-84.
        """
        gains = (self.returns - threshold).clip(lower=0).sum()
        losses = (threshold - self.returns).clip(lower=0).sum()
        if losses == 0:
            return np.inf
        if gains == 0:
            return 0.0
        return float(gains / losses)

    def beta(self, benchmark_returns: ArrayLike) -> float:
        r"""
        Beta against a benchmark, :math:`\widehat{\text{Cov}}(r, b) /
        \widehat{\text{Var}}(b)` with sample (``ddof=1``) moments.

        Parameters
        ----------
        benchmark_returns : pandas.Series or array_like
            Benchmark returns, matched as in :meth:`information_ratio`.

        Raises
        ------
        ValueError
            If the benchmark has (near-)zero variance.
        """
        port, bench = self._align(benchmark_returns)
        var_bench = np.var(bench, ddof=1)
        # Use a scale-aware tolerance: a constant benchmark can leave a tiny
        # floating-point residual variance that an exact == 0 test would miss.
        scale = max(bench.mean() ** 2, 1.0)
        if np.ptp(bench) == 0 or var_bench <= 1e-15 * scale:
            raise ValueError("Benchmark has (near-)zero variance; beta is undefined")
        return float(np.cov(port, bench, ddof=1)[0, 1] / var_bench)

    def alpha(
        self, benchmark_returns: ArrayLike, risk_free_rate: float = 0.02
    ) -> float:
        r"""
        Annualised Jensen's alpha.

        :math:`\alpha = m \bigl[(\bar r - r_f/m) - \beta (\bar b - r_f/m)\bigr]`,
        a simple (not compounded) annualisation of the per-period alpha.

        References
        ----------
        Jensen, M. C. (1968). The Performance of Mutual Funds in the Period
        1945-1964. Journal of Finance, 23(2), 389-416.
        """
        port, bench = self._align(benchmark_returns)
        b = self.beta(benchmark_returns)
        rf_per = risk_free_rate / self.periods_per_year
        per_period_alpha = (port.mean() - rf_per) - b * (bench.mean() - rf_per)
        return float(per_period_alpha * self.periods_per_year)

    def capm(
        self, benchmark_returns: ArrayLike, risk_free_rate: float = 0.02
    ) -> dict[str, float]:
        """
        Beta, annualised alpha and R-squared against a benchmark.

        Returns
        -------
        dict
            ``{'beta', 'alpha', 'r_squared'}``, where ``r_squared`` is the
            squared correlation of the paired returns.
        """
        port, bench = self._align(benchmark_returns)
        b = self.beta(benchmark_returns)
        a = self.alpha(benchmark_returns, risk_free_rate)
        corr = np.corrcoef(port, bench)[0, 1]
        return {"beta": b, "alpha": a, "r_squared": float(corr**2)}

    def expected_shortfall_parametric(self, confidence_level: float = 0.05) -> float:
        r"""
        Gaussian expected shortfall,
        :math:`\hat\mu - \hat\sigma \, \varphi(z_\alpha) / \alpha`.

        Parameters
        ----------
        confidence_level : float, default 0.05
            Tail probability alpha.
        """
        self._check_alpha(confidence_level)
        mu = self.returns.mean()
        sigma = self.returns.std()
        z = stats.norm.ppf(confidence_level)
        return float(mu - sigma * stats.norm.pdf(z) / confidence_level)

    def var_cornish_fisher(self, confidence_level: float = 0.05) -> float:
        r"""
        Cornish-Fisher (modified) VaR.

        Replaces the normal quantile :math:`z` by

        .. math::

            z_{cf} = z + \frac{(z^2 - 1) S}{6} + \frac{(z^3 - 3z) K}{24}
                     - \frac{(2z^3 - 5z) S^2}{36}

        with :math:`S` the sample skewness and :math:`K` the sample excess
        kurtosis (pandas' bias-corrected estimators), and returns
        :math:`\hat\mu + z_{cf} \hat\sigma`. The expansion is an
        approximation; for strongly fat-tailed returns at small alpha it can
        overshoot the true quantile.

        References
        ----------
        Cornish, E. A. and Fisher, R. A. (1938). Moments and Cumulants in the
        Specification of Distributions. Revue de l'Institut International de
        Statistique, 5(4), 307-320.
        """
        self._check_alpha(confidence_level)
        if len(self.returns) < 50:
            warnings.warn(
                "Cornish-Fisher VaR is unreliable with fewer than 50 observations",
                stacklevel=2,
            )

        mu = self.returns.mean()
        sigma = self.returns.std()
        skew = self.returns.skew()
        kurt = self.returns.kurtosis()  # pandas returns excess kurtosis
        z = stats.norm.ppf(confidence_level)
        z_cf = (
            z
            + (z**2 - 1) * skew / 6
            + (z**3 - 3 * z) * kurt / 24
            - (2 * z**3 - 5 * z) * skew**2 / 36
        )
        return float(mu + z_cf * sigma)

    def summary_stats(self) -> dict[str, float]:
        """
        Headline statistics in one dict.

        Ratios use their defaults: ``risk_free_rate=0.02`` for Sharpe and
        Sortino, 0 for Calmar. ``var_95``/``var_99`` and
        ``expected_shortfall_95`` are historical. ``kurtosis`` is excess
        kurtosis.
        """
        return {
            "count": len(self.returns),
            "periods_per_year": self.periods_per_year,
            "mean_return": float(self.returns.mean()),
            "volatility": self.volatility(),
            "sharpe_ratio": self.sharpe_ratio(),
            "sortino_ratio": self.sortino_ratio(),
            "calmar_ratio": self.calmar_ratio(),
            "var_95": self.var_historical(0.05),
            "var_99": self.var_historical(0.01),
            "expected_shortfall_95": self.expected_shortfall(0.05),
            "max_drawdown": self.max_drawdown(),
            "skewness": float(self.returns.skew()),
            "kurtosis": float(self.returns.kurtosis()),
        }
