"""Tests for the Black-Scholes-Merton option pricing model."""

import math

import pytest

from quantflow import BlackScholes


@pytest.mark.parametrize("option_type,lo,hi", [("call", 10.0, 11.0), ("put", 5.0, 6.0)])
def test_price_range(option_type, lo, hi):
    price = BlackScholes(100, 100, 1, 0.05, 0.2, option_type).price()
    assert lo < price < hi


def test_atm_call_benchmark(atm_call):
    # Classic ATM benchmark value
    assert math.isclose(atm_call.price(), 10.4506, abs_tol=1e-3)


def test_put_call_parity():
    c = BlackScholes(100, 100, 1, 0.05, 0.2, "call").price()
    p = BlackScholes(100, 100, 1, 0.05, 0.2, "put").price()
    assert math.isclose(c - p, 100 - 100 * math.exp(-0.05), abs_tol=1e-8)


@pytest.mark.parametrize("option_type,positive", [("call", True), ("put", False)])
def test_delta_sign(option_type, positive):
    d = BlackScholes(100, 100, 1, 0.05, 0.2, option_type).delta()
    assert (d > 0) if positive else (d < 0)


def test_gamma_vega_type_invariant():
    c = BlackScholes(100, 100, 1, 0.05, 0.2, "call")
    p = BlackScholes(100, 100, 1, 0.05, 0.2, "put")
    assert math.isclose(c.gamma(), p.gamma())
    assert math.isclose(c.vega(), p.vega())


def test_implied_vol_roundtrip(atm_call):
    iv = atm_call.implied_volatility(atm_call.price())
    assert math.isclose(iv, 0.2, abs_tol=1e-4)


def test_greeks_keys(atm_call):
    assert {"price", "delta", "gamma", "theta", "vega", "rho"} == set(atm_call.greeks())


# ----------------------------------------------------------------- dividends


def test_dividend_default_unchanged():
    """q=0.0 must reproduce the non-dividend result exactly (backward-compat gate)."""
    base = BlackScholes(100, 105, 0.25, 0.05, 0.2)
    assert base.price() == BlackScholes(100, 105, 0.25, 0.05, 0.2, q=0.0).price()


def test_dividend_lowers_call_raises_put():
    c0 = BlackScholes(100, 100, 1, 0.05, 0.2, "call").price()
    p0 = BlackScholes(100, 100, 1, 0.05, 0.2, "put").price()
    cq = BlackScholes(100, 100, 1, 0.05, 0.2, "call", q=0.03).price()
    pq = BlackScholes(100, 100, 1, 0.05, 0.2, "put", q=0.03).price()
    assert cq < c0
    assert pq > p0


def test_put_call_parity_with_dividends():
    c = BlackScholes(100, 100, 1, 0.05, 0.2, "call", q=0.03).price()
    p = BlackScholes(100, 100, 1, 0.05, 0.2, "put", q=0.03).price()
    lhs = c - p
    rhs = 100 * math.exp(-0.03) - 100 * math.exp(-0.05)
    assert math.isclose(lhs, rhs, abs_tol=1e-6)


def test_implied_vol_with_dividends():
    opt = BlackScholes(100, 105, 0.5, 0.04, 0.25, "call", q=0.02)
    iv = opt.implied_volatility(opt.price())
    assert math.isclose(iv, 0.25, abs_tol=1e-4)


# ---------------------------------------------------------------- validation


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(S=-1, K=100, T=1, r=0.05, sigma=0.2),
        dict(S=100, K=0, T=1, r=0.05, sigma=0.2),
        dict(S=100, K=100, T=-0.5, r=0.05, sigma=0.2),
        dict(S=100, K=100, T=1, r=0.05, sigma=0),
    ],
)
def test_invalid_params_raise(kwargs):
    with pytest.raises(ValueError):
        BlackScholes(**kwargs)


def test_bad_option_type_raises():
    with pytest.raises(ValueError):
        BlackScholes(100, 100, 1, 0.05, 0.2, "straddle")


def test_extreme_param_warnings():
    with pytest.warns(UserWarning):
        BlackScholes(100, 100, 6, 0.05, 0.2)
    with pytest.warns(UserWarning):
        BlackScholes(100, 100, 1, 0.05, 2.5)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(S=float("nan"), K=100, T=1, r=0.05, sigma=0.2),
        dict(S=100, K=float("inf"), T=1, r=0.05, sigma=0.2),
        dict(S=100, K=100, T=float("nan"), r=0.05, sigma=0.2),
        dict(S=100, K=100, T=1, r=float("nan"), sigma=0.2),
        dict(S=100, K=100, T=1, r=0.05, sigma=float("inf")),
    ],
)
def test_non_finite_params_raise(kwargs):
    with pytest.raises(ValueError):
        BlackScholes(**kwargs)


def test_non_string_option_type_raises():
    with pytest.raises(ValueError):
        BlackScholes(100, 100, 1, 0.05, 0.2, None)


def test_iv_rejects_non_positive_price(atm_call):
    with pytest.raises(ValueError):
        atm_call.implied_volatility(0.0)
    with pytest.raises(ValueError):
        atm_call.implied_volatility(-1.0)


def test_iv_rejects_arbitrage_violation():
    # market price at/above the S*e^{-qT} upper bound is not achievable
    option = BlackScholes(100, 100, 1, 0.05, 0.2, "call")
    with pytest.raises(ValueError):
        option.implied_volatility(100.0)
    with pytest.raises(ValueError):
        option.implied_volatility(150.0)
