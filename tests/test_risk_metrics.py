"""Tests for the RiskMetrics risk analytics class."""

import numpy as np
import pandas as pd
import pytest

from quantflow import RiskMetrics


@pytest.mark.parametrize("confidence_level", [0.01, 0.05, 0.10])
def test_var_negative(sample_returns, confidence_level):
    assert RiskMetrics(sample_returns).var_historical(confidence_level) < 0


def test_es_worse_than_var(sample_returns):
    rm = RiskMetrics(sample_returns)
    assert rm.expected_shortfall(0.05) <= rm.var_historical(0.05)


def test_es_parametric_worse_than_var(sample_returns):
    rm = RiskMetrics(sample_returns)
    assert rm.expected_shortfall_parametric(0.05) < rm.var_historical(0.05)


def test_input_types_supported(sample_returns):
    for data in (sample_returns.to_numpy(), sample_returns.to_frame(), sample_returns):
        assert np.isfinite(RiskMetrics(data).volatility())


def test_max_drawdown_nonpositive(sample_returns):
    assert RiskMetrics(sample_returns).max_drawdown() <= 0


def test_annualization_increases_vol(sample_returns):
    rm = RiskMetrics(sample_returns)
    assert rm.volatility(annualized=True) > rm.volatility(annualized=False)


def test_periods_per_year_configurable(sample_returns):
    daily = RiskMetrics(sample_returns, periods_per_year=252)
    monthly = RiskMetrics(sample_returns, periods_per_year=12)
    assert daily.volatility() > monthly.volatility()


def test_periods_per_year_validation(sample_returns):
    with pytest.raises(ValueError):
        RiskMetrics(sample_returns, periods_per_year=0)


def test_backward_compat_default(sample_returns):
    """Default construction must keep the 252-day annualization behavior."""
    rm = RiskMetrics(sample_returns)
    expected = rm.returns.std() * np.sqrt(252)
    assert np.isclose(rm.volatility(), expected)


def test_sortino_ratio(sample_returns):
    assert np.isfinite(RiskMetrics(sample_returns).sortino_ratio())


def test_calmar_ratio(sample_returns):
    assert np.isfinite(RiskMetrics(sample_returns).calmar_ratio())


def test_omega_ratio_positive(sample_returns):
    assert RiskMetrics(sample_returns).omega_ratio() > 0


def test_capm_beta_recovery():
    rng = np.random.default_rng(1)
    bench = pd.Series(rng.normal(0.0004, 0.01, 300))
    port = 1.5 * bench + pd.Series(rng.normal(0, 0.0008, 300))
    cap = RiskMetrics(port).capm(bench)
    assert np.isclose(cap["beta"], 1.5, atol=0.1)
    assert 0.0 <= cap["r_squared"] <= 1.0


def test_beta_zero_variance_raises(sample_returns):
    with pytest.raises(ValueError):
        RiskMetrics(sample_returns).beta(pd.Series(np.zeros(len(sample_returns))))


def test_cornish_fisher_var(sample_returns):
    assert RiskMetrics(sample_returns).var_cornish_fisher(0.05) < 0


def test_nan_values_dropped():
    series = pd.Series([0.01, np.nan, -0.02, 0.0, np.nan, 0.015])
    assert np.isfinite(RiskMetrics(series).volatility())


def test_summary_stats_keys(sample_returns):
    stats = RiskMetrics(sample_returns).summary_stats()
    assert {"periods_per_year", "sortino_ratio", "var_95", "max_drawdown"}.issubset(
        stats
    )
