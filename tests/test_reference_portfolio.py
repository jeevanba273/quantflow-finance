"""
Mean-variance optimisation checked against closed-form solutions.

With short sales allowed and only equality constraints, the frontier has
closed forms (Merton, R. C. (1972). An Analytic Derivation of the Efficient
Portfolio Frontier. Journal of Financial and Quantitative Analysis, 7(4),
1851-1872). With A = 1'S^-1 mu, B = mu'S^-1 mu, C = 1'S^-1 1, D = BC - A^2:

* minimum variance:  w = S^-1 1 / C
* target return m:   w = [(B - A m) S^-1 1 + (C m - A) S^-1 mu] / D
* tangency (rf):     w proportional to S^-1 (mu - rf 1)

For two assets the minimum-variance weight on asset 1 is
(s2^2 - s12) / (s1^2 + s2^2 - 2 s12).
"""

import numpy as np
import pandas as pd
import pytest

from quantflow import Portfolio


def _returns(seed, n_assets=4, n_obs=1500):
    rng = np.random.default_rng(seed)
    vols = np.linspace(0.008, 0.02, n_assets)
    corr = 0.3 * np.ones((n_assets, n_assets)) + 0.7 * np.eye(n_assets)
    cov = corr * np.outer(vols, vols)
    mu = np.linspace(0.0002, 0.0009, n_assets)
    data = rng.multivariate_normal(mu, cov, size=n_obs)
    return pd.DataFrame(data, columns=[f"A{i}" for i in range(n_assets)])


def _moments(frame, m=252):
    return frame.mean().to_numpy() * m, frame.cov().to_numpy() * m


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_two_asset_minimum_variance_closed_form(seed):
    frame = _returns(seed, n_assets=2)
    cov = frame.cov().to_numpy()
    w1 = (cov[1, 1] - cov[0, 1]) / (cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    assert 0 < w1 < 1  # interior solution, so long-only gives the same answer
    for allow_short in (False, True):
        w = Portfolio(frame).min_variance(allow_short=allow_short)["weights"]
        assert w == pytest.approx([w1, 1 - w1], abs=1e-7)


def test_two_asset_minimum_variance_corner():
    # Asset B is riskier and perfectly correlated with A: long-only puts all
    # weight in A, while the unconstrained solution shorts B.
    rng = np.random.default_rng(3)
    a = rng.normal(0.0005, 0.01, 1000)
    frame = pd.DataFrame({"A": a, "B": 2.0 * a + rng.normal(0, 1e-4, 1000)})
    w = Portfolio(frame).min_variance()["weights"]
    assert w == pytest.approx([1.0, 0.0], abs=1e-7)


@pytest.mark.parametrize("seed", [0, 1])
def test_global_minimum_variance_closed_form(seed):
    frame = _returns(seed)
    _, cov = _moments(frame)
    inv_one = np.linalg.solve(cov, np.ones(len(cov)))
    expected = inv_one / inv_one.sum()
    w = Portfolio(frame).min_variance(allow_short=True)["weights"]
    assert w == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("target", [0.08, 0.12, 0.2])
def test_target_return_closed_form(target):
    frame = _returns(4)
    mu, cov = _moments(frame)
    one = np.ones(len(mu))
    inv_one, inv_mu = np.linalg.solve(cov, one), np.linalg.solve(cov, mu)
    A, B, C = one @ inv_mu, mu @ inv_mu, one @ inv_one
    expected = ((B - A * target) * inv_one + (C * target - A) * inv_mu) / (
        B * C - A * A
    )
    res = Portfolio(frame).optimize(
        "target_return", target_return=target, allow_short=True
    )
    assert res["success"]
    assert res["return"] == pytest.approx(target, abs=1e-8)
    assert res["weights"] == pytest.approx(expected, abs=1e-5)


def test_tangency_closed_form():
    frame = _returns(0)
    mu, cov = _moments(frame)
    rf = 0.02
    w = np.linalg.solve(cov, mu - rf)
    assert w.sum() > 0  # the tangency portfolio exists for this sample
    expected = w / w.sum()
    res = Portfolio(frame).max_sharpe(risk_free_rate=rf, allow_short=True)
    assert res["weights"] == pytest.approx(expected, abs=1e-5)
    assert res["sharpe"] == pytest.approx(
        (expected @ mu - rf) / np.sqrt(expected @ cov @ expected), rel=1e-8
    )


@pytest.mark.parametrize("seed", [4, 5])
def test_max_sharpe_with_shorting_raises_when_no_tangency_exists(seed):
    # Here 1' S^-1 (mu - rf) < 0: the minimum-variance return is below rf, so
    # among fully invested portfolios the Sharpe ratio has no maximum; it is
    # only approached with unbounded long-short positions.
    frame = _returns(seed)
    mu, cov = _moments(frame)
    assert np.linalg.solve(cov, mu - 0.02).sum() < 0
    with pytest.raises(ValueError, match="risk-free rate"):
        Portfolio(frame).max_sharpe(risk_free_rate=0.02, allow_short=True)


@pytest.mark.parametrize("seed", [6, 7, 8])
def test_long_only_constraints_hold(seed):
    p = Portfolio(_returns(seed, n_assets=5))
    results = [p.min_variance(), p.max_sharpe(), p.optimize("target_return", 0.1)]
    frontier = p.efficient_frontier(n_points=8)
    weight_rows = frontier.filter(like="w_").to_numpy()
    for w in [r["weights"] for r in results] + list(weight_rows):
        assert w.sum() == pytest.approx(1.0, abs=1e-9)
        assert np.all(w >= -1e-9)
        assert np.all(w <= 1 + 1e-9)


def test_frontier_matches_closed_form_with_shorting():
    frame = _returns(9)
    mu, cov = _moments(frame)
    one = np.ones(len(mu))
    inv_one, inv_mu = np.linalg.solve(cov, one), np.linalg.solve(cov, mu)
    A, B, C = one @ inv_mu, mu @ inv_mu, one @ inv_one
    D = B * C - A * A
    frontier = Portfolio(frame).efficient_frontier(n_points=6, allow_short=True)
    for ret, vol in zip(frontier["return"], frontier["volatility"], strict=True):
        exact_var = (C * ret * ret - 2 * A * ret + B) / D
        assert vol == pytest.approx(np.sqrt(exact_var), rel=1e-6)


@pytest.mark.parametrize("seed", [0, 3])
def test_long_only_optimisers_beat_random_search(seed):
    # An independent check of SLSQP where no closed form exists: no portfolio
    # among 200,000 random long-only weight vectors may do better.
    frame = _returns(seed)
    mu, cov = _moments(frame)
    w = np.random.default_rng(seed + 100).dirichlet(np.ones(len(mu)), 200_000)
    var = np.einsum("ij,jk,ik->i", w, cov, w)
    sharpe = (w @ mu - 0.02) / np.sqrt(var)
    p = Portfolio(frame)
    assert p.max_sharpe(0.02)["sharpe"] >= sharpe.max() - 1e-9
    assert p.min_variance()["volatility"] <= np.sqrt(var.min()) + 1e-9
