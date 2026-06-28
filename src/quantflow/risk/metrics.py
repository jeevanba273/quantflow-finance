"""
Portfolio risk metrics and analytics.

This module provides implementations of common risk measures including
Value at Risk (VaR), Expected Shortfall, and performance ratios.
"""

import numpy as np
import pandas as pd
from scipy import stats


class RiskMetrics:
    """
    Portfolio risk analysis tools.

    Parameters:
    returns: pandas Series or numpy array of portfolio returns
    periods_per_year: int - Number of return periods per year used for
        annualization (default 252 for daily data; use 12 for monthly,
        52 for weekly, etc.)
    """

    def __init__(self, returns, periods_per_year: int = 252):
        if isinstance(returns, pd.DataFrame):
            # If DataFrame, take the first column or flatten
            self.returns = returns.iloc[:, 0].dropna()
        elif isinstance(returns, pd.Series):
            self.returns = returns.dropna()
        else:
            # Handle numpy arrays
            returns_flat = np.array(returns).flatten()
            self.returns = pd.Series(returns_flat).dropna()

        # Validate we have data
        if len(self.returns) == 0:
            raise ValueError(
                "No valid return data provided. Cannot calculate risk metrics."
            )

        if periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive")
        self.periods_per_year = int(periods_per_year)

        # Warn if we have very little data
        if len(self.returns) < 30:
            print(
                f"Warning: Only {len(self.returns)} data points available. Risk metrics may be unreliable."
            )

    def var_historical(self, confidence_level=0.05):
        """
        Calculate Historical Value at Risk.

        Parameters:
        confidence_level: float - Confidence level (e.g., 0.05 for 95% VaR)

        Returns:
        float - VaR value (negative number representing loss)
        """
        if len(self.returns) == 0:
            raise ValueError("No return data available for VaR calculation")

        if not (0 < confidence_level < 1):
            raise ValueError("Confidence level must be between 0 and 1")

        return np.percentile(self.returns, confidence_level * 100)

    def var_parametric(self, confidence_level=0.05):
        """
        Calculate Parametric VaR (assumes normal distribution).

        Parameters:
        confidence_level: float - Confidence level

        Returns:
        float - Parametric VaR value
        """
        if len(self.returns) == 0:
            raise ValueError("No return data available for parametric VaR calculation")

        if not (0 < confidence_level < 1):
            raise ValueError("Confidence level must be between 0 and 1")

        mean_return = self.returns.mean()
        std_return = self.returns.std()
        z_score = stats.norm.ppf(confidence_level)

        return mean_return + z_score * std_return

    def sharpe_ratio(self, risk_free_rate=0.02):
        """
        Calculate Sharpe ratio (assuming daily returns).

        Parameters:
        risk_free_rate: float - Annual risk-free rate (default 2%)

        Returns:
        float - Annualized Sharpe ratio
        """
        if len(self.returns) == 0:
            raise ValueError("No return data available for Sharpe ratio calculation")

        excess_return = self.returns.mean() - risk_free_rate / self.periods_per_year
        return excess_return / self.returns.std() * np.sqrt(self.periods_per_year)

    def expected_shortfall(self, confidence_level=0.05):
        """
        Calculate Expected Shortfall (Conditional VaR).

        Parameters:
        confidence_level: float - Confidence level

        Returns:
        float - Expected Shortfall value
        """
        if len(self.returns) == 0:
            raise ValueError(
                "No return data available for Expected Shortfall calculation"
            )

        var = self.var_historical(confidence_level)
        # Average of returns that are worse than VaR
        tail_returns = self.returns[self.returns <= var]

        if len(tail_returns) == 0:
            # If no returns are worse than VaR, return the VaR itself
            return var

        return tail_returns.mean()

    def max_drawdown(self):
        """
        Calculate maximum drawdown.

        Returns:
        float - Maximum drawdown as a percentage (negative value)
        """
        if len(self.returns) == 0:
            raise ValueError(
                "No return data available for maximum drawdown calculation"
            )

        # Convert returns to cumulative wealth
        cumulative = (1 + self.returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        return drawdown.min()

    def volatility(self, annualized=True):
        """
        Calculate volatility.

        Parameters:
        annualized: bool - Whether to annualize the volatility

        Returns:
        float - Volatility (standard deviation of returns)
        """
        if len(self.returns) == 0:
            raise ValueError("No return data available for volatility calculation")

        vol = self.returns.std()
        if annualized:
            vol *= np.sqrt(self.periods_per_year)
        return vol

    def _align(self, benchmark_returns):
        """
        Align portfolio and benchmark returns to a common length.

        Flattens Series/DataFrame/array benchmarks and trims both series to their
        last ``min_length`` observations. Returns ``(portfolio, benchmark)`` as
        numpy arrays.
        """
        if isinstance(benchmark_returns, (pd.Series, pd.DataFrame)):
            benchmark_returns = benchmark_returns.values.flatten()
        benchmark_returns = np.asarray(benchmark_returns, dtype=float)

        min_length = min(len(self.returns), len(benchmark_returns))
        if min_length < 2:
            raise ValueError("Need at least 2 aligned observations for this metric")

        portfolio_rets = self.returns.iloc[-min_length:].to_numpy()
        benchmark_rets = benchmark_returns[-min_length:]
        return portfolio_rets, benchmark_rets

    def information_ratio(self, benchmark_returns):
        """
        Calculate Information Ratio.

        Parameters:
        benchmark_returns: array-like - Benchmark returns for comparison

        Returns:
        float - Information ratio
        """
        if len(self.returns) == 0:
            raise ValueError(
                "No return data available for Information Ratio calculation"
            )

        portfolio_rets, benchmark_rets = self._align(benchmark_returns)
        excess_returns = portfolio_rets - benchmark_rets
        tracking_error = excess_returns.std(ddof=1)

        if tracking_error == 0:
            return 0

        return excess_returns.mean() / tracking_error * np.sqrt(self.periods_per_year)

    def sortino_ratio(self, risk_free_rate=0.02, target=0.0):
        """
        Calculate the Sortino ratio (downside-risk-adjusted return).

        Like the Sharpe ratio but penalizes only downside deviation (returns
        below ``target``) instead of total volatility.

        Parameters:
        risk_free_rate: float - Annual risk-free rate (default 2%)
        target: float - Minimum acceptable per-period return (default 0)

        Returns:
        float - Annualized Sortino ratio (np.inf if there is no downside risk)
        """
        excess_return = self.returns.mean() - risk_free_rate / self.periods_per_year
        downside = np.sqrt(np.mean(np.minimum(self.returns - target, 0.0) ** 2))
        if downside == 0:
            return np.inf
        return excess_return / downside * np.sqrt(self.periods_per_year)

    def calmar_ratio(self, risk_free_rate=0.0):
        """
        Calculate the Calmar ratio (CAGR divided by maximum drawdown).

        Parameters:
        risk_free_rate: float - Annual risk-free rate subtracted from CAGR (default 0)

        Returns:
        float - Calmar ratio (np.inf if there is no drawdown)
        """
        n = len(self.returns)
        cagr = (1 + self.returns).prod() ** (self.periods_per_year / n) - 1
        mdd = abs(self.max_drawdown())
        if mdd == 0:
            return np.inf
        return (cagr - risk_free_rate) / mdd

    def omega_ratio(self, threshold=0.0):
        """
        Calculate the Omega ratio at the given threshold return.

        Ratio of probability-weighted gains to losses relative to ``threshold``.

        Parameters:
        threshold: float - Per-period threshold return (default 0)

        Returns:
        float - Omega ratio (np.inf if there are no losses, 0.0 if no gains)
        """
        gains = (self.returns - threshold).clip(lower=0).sum()
        losses = (threshold - self.returns).clip(lower=0).sum()
        if losses == 0:
            return np.inf
        if gains == 0:
            return 0.0
        return gains / losses

    def beta(self, benchmark_returns):
        """
        Calculate CAPM beta versus a benchmark.

        Parameters:
        benchmark_returns: array-like - Benchmark returns

        Returns:
        float - Beta (slope of returns regressed on benchmark)
        """
        port, bench = self._align(benchmark_returns)
        var_bench = np.var(bench, ddof=1)
        # Use a scale-aware tolerance: a constant benchmark can leave a tiny
        # floating-point residual variance that an exact == 0 test would miss.
        scale = max(bench.mean() ** 2, 1.0)
        if np.ptp(bench) == 0 or var_bench <= 1e-15 * scale:
            raise ValueError("Benchmark has (near-)zero variance; beta is undefined")
        return np.cov(port, bench, ddof=1)[0, 1] / var_bench

    def alpha(self, benchmark_returns, risk_free_rate: float = 0.02):
        """
        Calculate annualized Jensen's alpha versus a benchmark.

        Parameters:
        benchmark_returns: array-like - Benchmark returns
        risk_free_rate: float - Annual risk-free rate (default 2%)

        Returns:
        float - Annualized alpha
        """
        port, bench = self._align(benchmark_returns)
        b = self.beta(benchmark_returns)
        rf_per = risk_free_rate / self.periods_per_year
        per_period_alpha = (port.mean() - rf_per) - b * (bench.mean() - rf_per)
        return per_period_alpha * self.periods_per_year

    def capm(self, benchmark_returns, risk_free_rate: float = 0.02):
        """
        Calculate CAPM beta, alpha, and R-squared versus a benchmark.

        Returns:
        dict - {'beta', 'alpha', 'r_squared'}
        """
        port, bench = self._align(benchmark_returns)
        b = self.beta(benchmark_returns)
        a = self.alpha(benchmark_returns, risk_free_rate)
        corr = np.corrcoef(port, bench)[0, 1]
        return {"beta": b, "alpha": a, "r_squared": corr**2}

    def expected_shortfall_parametric(self, confidence_level=0.05):
        """
        Calculate parametric (Gaussian) Expected Shortfall.

        Parameters:
        confidence_level: float - Confidence level (e.g. 0.05 for 95% ES)

        Returns:
        float - Parametric Expected Shortfall (negative number)
        """
        if not (0 < confidence_level < 1):
            raise ValueError("Confidence level must be between 0 and 1")
        mu = self.returns.mean()
        sigma = self.returns.std()
        z = stats.norm.ppf(confidence_level)
        return mu - sigma * stats.norm.pdf(z) / confidence_level

    def var_cornish_fisher(self, confidence_level=0.05):
        """
        Calculate Cornish-Fisher (modified) VaR, adjusting for skew and kurtosis.

        Parameters:
        confidence_level: float - Confidence level (e.g. 0.05 for 95% VaR)

        Returns:
        float - Modified VaR value (negative number representing loss)
        """
        if not (0 < confidence_level < 1):
            raise ValueError("Confidence level must be between 0 and 1")
        if len(self.returns) < 50:
            print(
                "Warning: Cornish-Fisher VaR is unreliable with fewer than 50 observations"
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
        return mu + z_cf * sigma

    def summary_stats(self):
        """
        Calculate comprehensive risk statistics.

        Returns:
        dict - Dictionary containing various risk metrics
        """
        if len(self.returns) == 0:
            raise ValueError("No return data available for summary statistics")

        return {
            "count": len(self.returns),
            "periods_per_year": self.periods_per_year,
            "mean_return": self.returns.mean(),
            "volatility": self.volatility(),
            "sharpe_ratio": self.sharpe_ratio(),
            "sortino_ratio": self.sortino_ratio(),
            "calmar_ratio": self.calmar_ratio(),
            "var_95": self.var_historical(0.05),
            "var_99": self.var_historical(0.01),
            "expected_shortfall_95": self.expected_shortfall(0.05),
            "max_drawdown": self.max_drawdown(),
            "skewness": self.returns.skew(),
            "kurtosis": self.returns.kurtosis(),
        }
