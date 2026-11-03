"""Tests for the Monte Carlo European option pricer."""

import numpy as np
import pytest

from quantflow import BlackScholes, MonteCarlo

CASES = [
    # S, K, T, r, sigma, option_type, q
    (100, 100, 1.0, 0.05, 0.2, "call", 0.0),
    (100, 120, 0.5, 0.03, 0.3, "call", 0.02),
    (100, 90, 0.25, 0.05, 0.25, "put", 0.0),
    (100, 110, 2.0, 0.02, 0.4, "put", 0.01),
]
METHODS = [(False, False), (True, False), (False, True), (True, True)]


def _run(case, antithetic=False, control_variate=False, n_paths=100_000, seed=11):
    S, K, T, r, sigma, kind, q = case
    mc = MonteCarlo(
        S, K, T, r, sigma, kind, q=q, n_paths=n_paths,
        antithetic=antithetic, control_variate=control_variate, seed=seed,
    )  # fmt: skip
    return mc.simulate(confidence=0.99)


def test_same_seed_same_result():
    a = _run(CASES[0], seed=3)
    b = _run(CASES[0], seed=3)
    assert a == b


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("antithetic,control_variate", METHODS)
def test_analytic_price_inside_99pct_interval(case, antithetic, control_variate):
    S, K, T, r, sigma, kind, q = case
    exact = BlackScholes(S, K, T, r, sigma, kind, q=q).price()
    res = _run(case, antithetic, control_variate)
    assert res.ci_low <= exact <= res.ci_high


def test_interval_is_price_plus_minus_z_times_se():
    res = _run(CASES[0])
    assert res.ci_high - res.price == pytest.approx(2.5758293 * res.std_error)
    assert res.price - res.ci_low == pytest.approx(2.5758293 * res.std_error)


def test_standard_error_halves_when_paths_quadruple():
    small = _run(CASES[0], n_paths=50_000).std_error
    large = _run(CASES[0], n_paths=200_000).std_error
    assert large / small == pytest.approx(0.5, rel=0.05)


@pytest.mark.parametrize("case", CASES)
def test_variance_reduction_lowers_standard_error(case):
    plain = _run(case).std_error
    anti = _run(case, antithetic=True).std_error
    cv = _run(case, control_variate=True).std_error
    both = _run(case, antithetic=True, control_variate=True).std_error
    # Same number of simulated terminal prices in every run.
    assert anti < plain
    assert cv < plain
    assert both < 0.3 * plain


def test_variance_reduction_at_the_money():
    plain = _run(CASES[0]).std_error
    assert _run(CASES[0], antithetic=True).std_error < 0.8 * plain
    assert _run(CASES[0], control_variate=True).std_error < 0.5 * plain


def test_price_method_matches_simulate():
    mc = MonteCarlo(100, 100, 1, 0.05, 0.2, n_paths=10_000, seed=1)
    assert mc.price() == mc.simulate().price


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(n_paths=1),
        dict(n_paths=10.5),
        dict(n_paths=True),
        dict(n_paths=1001, antithetic=True),
    ],
)
def test_invalid_n_paths(kwargs):
    with pytest.raises(ValueError):
        MonteCarlo(100, 100, 1, 0.05, 0.2, **kwargs)


@pytest.mark.parametrize("confidence", [0, 1, 1.5])
def test_invalid_confidence(confidence):
    with pytest.raises(ValueError):
        MonteCarlo(100, 100, 1, 0.05, 0.2, n_paths=100, seed=0).simulate(confidence)


def test_inherits_parameter_validation():
    with pytest.raises(ValueError):
        MonteCarlo(100, 100, 1, 0.05, -0.2)
    assert np.isfinite(MonteCarlo(100, 100, 1, 0.05, 0.2, n_paths=100, seed=0).price())
