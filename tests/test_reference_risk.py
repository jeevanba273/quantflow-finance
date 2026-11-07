"""
Risk metrics checked against distributions with known quantiles and against
their written definitions.

VaR and expected shortfall are reported as signed returns (a loss is
negative) at tail probability ``alpha`` (0.05 for 95% VaR). Annualisation
uses ``m = periods_per_year`` (default 252 trading days) throughout.

The tolerances for sample estimates come from the asymptotic standard error
of a sample quantile, sqrt(alpha (1 - alpha) / n) / f(q) (for example
Serfling, R. J. (1980). Approximation Theorems of Mathematical Statistics,
section 2.3), and of a tail mean. Seeds are fixed, so the tests are
deterministic.
"""

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from quantflow_finance import RiskMetrics

N = 400_000


def _quantile_se(alpha, density_at_q, n=N):
    return math.sqrt(alpha * (1 - alpha) / n) / density_at_q


@pytest.mark.parametrize("alpha", [0.01, 0.05])
def test_historical_var_and_es_on_normal_sample(alpha):
    mu, sd = 0.0004, 0.012
    x = np.random.default_rng(1).normal(mu, sd, N)
    rm = RiskMetrics(x)
    q = stats.norm.ppf(alpha, mu, sd)
    es = stats.norm.expect(lambda v: v, loc=mu, scale=sd, ub=q, conditional=True)
    se_q = _quantile_se(alpha, stats.norm.pdf(q, mu, sd))
    assert rm.var_historical(alpha) == pytest.approx(q, abs=5 * se_q)
    tail = x[x <= q]
    se_es = tail.std(ddof=1) / math.sqrt(tail.size) + se_q
    assert rm.expected_shortfall(alpha) == pytest.approx(es, abs=5 * se_es)


@pytest.mark.parametrize("alpha", [0.01, 0.05])
def test_historical_var_and_es_on_student_t_sample(alpha):
    nu, loc, scale = 5, 0.0003, 0.009
    x = loc + scale * np.random.default_rng(2).standard_t(nu, N)
    rm = RiskMetrics(x)
    q = stats.t.ppf(alpha, nu, loc, scale)
    es = stats.t.expect(
        lambda v: v, args=(nu,), loc=loc, scale=scale, ub=q, conditional=True
    )
    se_q = _quantile_se(alpha, stats.t.pdf(q, nu, loc, scale))
    assert rm.var_historical(alpha) == pytest.approx(q, abs=5 * se_q)
    tail = x[x <= q]
    se_es = tail.std(ddof=1) / math.sqrt(tail.size) + se_q
    assert rm.expected_shortfall(alpha) == pytest.approx(es, abs=5 * se_es)


def test_gaussian_var_understates_fat_tail():
    # Documented limitation: with Student-t(5) returns the normal VaR at 1%
    # is less severe than the true 1% quantile.
    nu, scale = 5, 0.009
    x = scale * np.random.default_rng(3).standard_t(nu, N)
    rm = RiskMetrics(x)
    true_q = stats.t.ppf(0.01, nu, scale=scale)
    assert rm.var_parametric(0.01) > true_q + 0.0005
    assert rm.var_historical(0.01) < rm.var_parametric(0.01)


@pytest.mark.parametrize("alpha", [0.01, 0.025, 0.05])
def test_parametric_var_and_es_closed_forms(alpha, sample_returns):
    rm = RiskMetrics(sample_returns)
    mu, sd = sample_returns.mean(), sample_returns.std(ddof=1)
    q = stats.norm.ppf(alpha, mu, sd)
    es = stats.norm.expect(lambda v: v, loc=mu, scale=sd, ub=q, conditional=True)
    assert rm.var_parametric(alpha) == pytest.approx(q, rel=1e-12)
    assert rm.expected_shortfall_parametric(alpha) == pytest.approx(es, rel=1e-8)


def test_cornish_fisher_reduces_to_normal_without_skew_or_kurtosis():
    # Sample skew and excess kurtosis of a large normal sample are near zero,
    # so the Cornish-Fisher quantile is close to the Gaussian one.
    x = np.random.default_rng(4).normal(0.0, 0.01, N)
    rm = RiskMetrics(x)
    assert rm.var_cornish_fisher(0.05) == pytest.approx(
        rm.var_parametric(0.05), rel=5e-3
    )


def test_cornish_fisher_formula():
    x = pd.Series(np.random.default_rng(5).gamma(2.0, 0.01, 5000) - 0.02)
    rm = RiskMetrics(x)
    z = stats.norm.ppf(0.05)
    s = stats.skew(x, bias=False)
    k = stats.kurtosis(x, bias=False)  # excess kurtosis
    z_cf = (
        z
        + (z**2 - 1) * s / 6
        + (z**3 - 3 * z) * k / 24
        - (2 * z**3 - 5 * z) * s**2 / 36
    )
    assert rm.var_cornish_fisher(0.05) == pytest.approx(
        x.mean() + z_cf * x.std(ddof=1), rel=1e-12
    )


# ---------------------------------------------------------------- definitions
# Each ratio is recomputed here from the formula in its docstring.


@pytest.fixture
def series():
    return pd.Series(np.random.default_rng(6).normal(0.0006, 0.011, 750))


def test_volatility_definition(series):
    rm = RiskMetrics(series)
    assert rm.volatility() == pytest.approx(series.std(ddof=1) * math.sqrt(252))
    assert rm.volatility(annualized=False) == pytest.approx(series.std(ddof=1))


@pytest.mark.parametrize("m", [252, 52, 12])
def test_sharpe_definition(series, m):
    rf = 0.03
    expected = (series.mean() - rf / m) / series.std(ddof=1) * math.sqrt(m)
    rm = RiskMetrics(series, periods_per_year=m)
    assert rm.sharpe_ratio(rf) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("target", [0.0, 0.001])
def test_sortino_definition(series, target):
    rf, m = 0.03, 252
    downside = math.sqrt(np.mean(np.minimum(series - target, 0.0) ** 2))
    expected = (series.mean() - rf / m) / downside * math.sqrt(m)
    rm = RiskMetrics(series)
    assert rm.sortino_ratio(rf, target) == pytest.approx(expected, rel=1e-12)


def test_calmar_definition(series):
    m, n = 252, len(series)
    cagr = (1 + series).prod() ** (m / n) - 1
    wealth = np.concatenate(([1.0], (1 + series).cumprod()))
    mdd = np.min(wealth / np.maximum.accumulate(wealth) - 1)
    assert RiskMetrics(series).calmar_ratio() == pytest.approx(cagr / abs(mdd))


def test_omega_definition(series):
    gains = np.clip(series, 0, None).sum()
    losses = np.clip(-series, 0, None).sum()
    assert RiskMetrics(series).omega_ratio() == pytest.approx(gains / losses)


def test_beta_alpha_information_ratio_definitions(series):
    rng = np.random.default_rng(7)
    bench = pd.Series(rng.normal(0.0004, 0.01, len(series)))
    port = 0.8 * bench + series * 0.3
    rm = RiskMetrics(port)
    rf, m = 0.02, 252
    beta = np.cov(port, bench, ddof=1)[0, 1] / np.var(bench, ddof=1)
    alpha = m * ((port.mean() - rf / m) - beta * (bench.mean() - rf / m))
    active = port - bench
    ir = active.mean() / active.std(ddof=1) * math.sqrt(m)
    assert rm.beta(bench) == pytest.approx(beta, rel=1e-12)
    assert rm.alpha(bench, rf) == pytest.approx(alpha, rel=1e-10)
    assert rm.information_ratio(bench) == pytest.approx(ir, rel=1e-12)
    assert rm.capm(bench, rf)["r_squared"] == pytest.approx(
        np.corrcoef(port, bench)[0, 1] ** 2
    )
