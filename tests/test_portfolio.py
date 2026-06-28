"""Tests for the Portfolio analytics and optimization class."""

import numpy as np
import pandas as pd
import pytest

from quantflow import Portfolio, RiskMetrics


def test_construction(sample_multi_returns):
    p = Portfolio(sample_multi_returns)
    assert p.n_assets == 3
    assert len(p.weights) == 3
    assert np.isclose(p.weights.sum(), 1.0)


def test_requires_two_assets(sample_returns):
    with pytest.raises(ValueError):
        Portfolio(sample_returns.to_frame())


def test_covariance_and_correlation(sample_multi_returns):
    p = Portfolio(sample_multi_returns)
    corr = p.correlation_matrix().to_numpy()
    assert np.allclose(np.diag(corr), 1.0)
    assert np.allclose(corr, corr.T)
    assert p.covariance_matrix().shape == (3, 3)


def test_portfolio_return_vol_hand_check():
    # Two uncorrelated assets, equal weights: known closed-form values.
    rng = np.random.default_rng(3)
    a = rng.normal(0.001, 0.01, 500)
    b = rng.normal(0.002, 0.02, 500)
    df = pd.DataFrame({"A": a, "B": b})
    p = Portfolio(df, periods_per_year=1)
    w = np.array([0.5, 0.5])
    expected_ret = 0.5 * a.mean() + 0.5 * b.mean()
    assert np.isclose(p.portfolio_return(w, annualized=False), expected_ret)
    cov = np.cov(np.vstack([a, b]), ddof=1)
    expected_var = w @ cov @ w
    assert np.isclose(
        p.portfolio_volatility(w, annualized=False), np.sqrt(expected_var)
    )


def test_min_variance(sample_multi_returns):
    p = Portfolio(sample_multi_returns)
    mv = p.min_variance()
    ew_vol = p.portfolio_volatility(np.repeat(1 / 3, 3))
    assert mv["success"]
    assert mv["volatility"] <= ew_vol + 1e-9
    assert np.isclose(sum(mv["weights"]), 1.0)
    assert all(w >= -1e-9 for w in mv["weights"])


def test_min_variance_matches_analytical():
    """Unconstrained min-variance must equal the closed-form Sigma^-1 1 / (1' Sigma^-1 1).

    Regression guard: with daily-scale returns the variance objective is tiny, and a
    loose optimizer tolerance previously returned the equal-weight starting guess.
    """
    rng = np.random.default_rng(5)
    data = pd.DataFrame(
        rng.normal(0.0005, 0.01, (800, 3)) * np.array([1.0, 1.3, 0.8]),
        columns=["A", "B", "C"],
    )
    p = Portfolio(data)
    cov = data.cov().to_numpy()
    inv = np.linalg.inv(cov)
    one = np.ones(3)
    w_analytical = inv @ one / (one @ inv @ one)
    w_pkg = p.min_variance(allow_short=True)["weights"]
    assert np.max(np.abs(w_pkg - w_analytical)) < 1e-4
    # Must have actually optimized, not returned the equal-weight starting guess.
    assert np.max(np.abs(w_pkg - np.repeat(1 / 3, 3))) > 1e-3


def test_max_sharpe(sample_multi_returns):
    p = Portfolio(sample_multi_returns)
    ms = p.max_sharpe()
    ew_sharpe = p.portfolio_sharpe(np.repeat(1 / 3, 3))
    assert ms["success"]
    assert ms["sharpe"] >= ew_sharpe - 1e-6


def test_efficient_frontier_monotone(sample_multi_returns):
    ef = Portfolio(sample_multi_returns).efficient_frontier(n_points=12)
    assert len(ef) >= 2
    assert (ef["return"].diff().dropna() >= -1e-6).all()


def test_risk_metrics_bridge(sample_multi_returns):
    p = Portfolio(sample_multi_returns)
    rm = p.risk_metrics(p.max_sharpe()["weights"])
    assert isinstance(rm, RiskMetrics)
    assert np.isfinite(rm.sortino_ratio())


def test_bad_weights_raise(sample_multi_returns):
    p = Portfolio(sample_multi_returns)
    with pytest.raises(ValueError):
        p.portfolio_return([0.5, 0.5])  # wrong length
    with pytest.raises(ValueError):
        p.portfolio_return([0.5, 0.3, 0.1])  # does not sum to 1


def test_optimize_dispatch(sample_multi_returns):
    p = Portfolio(sample_multi_returns)
    assert p.optimize("min_variance")["success"]
    with pytest.raises(ValueError):
        p.optimize("nonsense")
