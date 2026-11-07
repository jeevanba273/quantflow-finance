"""Tests for the vectorised Black-Scholes-Merton functions."""

import itertools

import numpy as np
import pytest

from quantflow_finance import BlackScholes, black_scholes_greeks, black_scholes_price
from quantflow_finance.options.analytic import _price_scalar

GRID = list(
    itertools.product(
        [80.0, 100.0, 125.0],  # S
        [90.0, 100.0, 110.0],  # K
        [0.05, 0.5, 2.0],  # T
        [-0.01, 0.04],  # r
        [0.1, 0.35],  # sigma
        [0.0, 0.03],  # q
        ["call", "put"],
    )
)


def test_vectorised_matches_class_elementwise():
    S, K, T, r, sigma, q, kind = (np.array(col) for col in zip(*GRID, strict=True))
    vec = black_scholes_greeks(S, K, T, r, sigma, kind, q)
    for i, args in enumerate(GRID):
        s, k, t, rr, v, qq, ot = args
        one = BlackScholes(s, k, t, rr, v, ot, q=qq).greeks()
        for name, value in one.items():
            assert vec[name][i] == pytest.approx(value, rel=1e-13, abs=1e-15)


def test_broadcasting_shape():
    strikes = np.linspace(80, 120, 5)
    expiries = np.array([0.25, 0.5, 1.0])[:, None]
    calls = black_scholes_price(100, strikes, expiries, 0.03, 0.2, "call")
    assert calls.shape == (3, 5)
    # Without dividends a call gains value with expiry and loses it with strike.
    assert np.all(np.diff(calls, axis=0) > 0)
    assert np.all(np.diff(calls, axis=1) < 0)


def test_scalar_inputs_return_float():
    assert isinstance(black_scholes_price(100, 100, 1, 0.05, 0.2), float)
    greeks = black_scholes_greeks(100, 100, 1, 0.05, 0.2)
    assert all(isinstance(v, float) for v in greeks.values())


def test_option_type_case_insensitive():
    assert black_scholes_price(100, 100, 1, 0.05, 0.2, "PUT") == pytest.approx(
        black_scholes_price(100, 100, 1, 0.05, 0.2, "put")
    )


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(S=[100, -1]), "S"),
        (dict(K=[100, 0]), "K"),
        (dict(T=[1, 0]), "T"),
        (dict(sigma=[0.2, -0.1]), "sigma"),
        (dict(r=[0.05, np.nan]), "r must be a finite"),
        (dict(q=np.inf), "q must be a finite"),
        (dict(option_type=["call", "straddle"]), "option_type"),
        (dict(option_type=1), "option_type"),
    ],
)
def test_validation(kwargs, match):
    args = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        black_scholes_price(**args)


def test_scalar_solver_price_matches_vectorised():
    for s, k, t, rr, v, qq, ot in GRID:
        expected = black_scholes_price(s, k, t, rr, v, ot, qq)
        got = _price_scalar(s, k, t, rr, v, qq, ot == "call")
        assert got == pytest.approx(expected, rel=1e-12, abs=1e-14)


def test_implied_vol_round_trip_default_tolerance():
    opt = BlackScholes(100, 110, 0.5, 0.03, 0.25, "call")
    assert opt.implied_volatility(opt.price()) == pytest.approx(0.25, abs=1e-10)
