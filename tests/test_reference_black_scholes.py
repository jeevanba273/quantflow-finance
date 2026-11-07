"""
Black-Scholes-Merton checked against values that do not come from this code.

Sources
-------
Hull, J. C. Options, Futures, and Other Derivatives (10th ed.), worked
example in the Black-Scholes-Merton chapter: S0=42, K=40, r=10%, sigma=20%,
T=0.5 gives c=4.76 and p=0.81.

Haug, E. G. (2007). The Complete Guide to Option Pricing Formulas (2nd ed.),
McGraw-Hill. Worked examples in chapter 1 (Black-Scholes-Merton and its
Merton, Black-76 and Garman-Kohlhagen variants) and chapter 2 (Greeks).
Haug writes the models with a cost-of-carry b; here q = r - b. The same cases
are repriced with QuantLib in test_reference_libraries.py.
"""

import itertools
import math

import pytest

from quantflow_finance import BlackScholes, black_scholes_price


def test_hull_textbook_example():
    call = BlackScholes(42, 40, 0.5, 0.10, 0.20, "call").price()
    put = BlackScholes(42, 40, 0.5, 0.10, 0.20, "put").price()
    assert round(call, 2) == 4.76
    assert round(put, 2) == 0.81


# (description, kwargs, method, Haug value). Haug prints four decimals.
HAUG = [
    ("BS call", dict(S=60, K=65, T=0.25, r=0.08, sigma=0.30), "price", 2.1334),
    (
        "Merton put, dividend yield",
        dict(S=100, K=95, T=0.5, r=0.10, sigma=0.20, option_type="put", q=0.05),
        "price",
        2.4648,
    ),
    (
        "Black-76 call on a future (b=0)",
        dict(S=19, K=19, T=0.75, r=0.10, sigma=0.28, q=0.10),
        "price",
        1.7011,
    ),
    (
        "Black-76 put on a future (b=0)",
        dict(S=19, K=19, T=0.75, r=0.10, sigma=0.28, option_type="put", q=0.10),
        "price",
        1.7011,
    ),
    (
        "Garman-Kohlhagen FX call",
        dict(S=1.56, K=1.60, T=0.5, r=0.06, sigma=0.12, q=0.08),
        "price",
        0.0291,
    ),
    (
        "call delta, future (b=0)",
        dict(S=105, K=100, T=0.5, r=0.10, sigma=0.36, q=0.10),
        "delta",
        0.5946,
    ),
    (
        "put delta, future (b=0)",
        dict(S=105, K=100, T=0.5, r=0.10, sigma=0.36, option_type="put", q=0.10),
        "delta",
        -0.3566,
    ),
    ("gamma (b=r)", dict(S=55, K=60, T=0.75, r=0.10, sigma=0.30), "gamma", 0.0278),
    (
        "put theta per year (b=0.02)",
        dict(S=430, K=405, T=0.0833, r=0.07, sigma=0.20, option_type="put", q=0.05),
        "theta",
        -31.1924,
    ),
    # quantflow_finance's rho is per 1 percentage point; Haug's is per unit of r.
    ("call rho (b=r)", dict(S=72, K=75, T=1.0, r=0.09, sigma=0.19), "rho100", 38.7325),
]


@pytest.mark.parametrize("label,kwargs,method,expected", HAUG, ids=[h[0] for h in HAUG])
def test_haug_reference_values(label, kwargs, method, expected):
    option = BlackScholes(**kwargs)
    if method == "rho100":
        value = option.rho() * 100
    else:
        value = getattr(option, method)()
    assert round(value, 4) == pytest.approx(expected, abs=1e-12)


PARITY_GRID = list(
    itertools.product(
        [50.0, 100.0, 180.0],  # S
        [60.0, 100.0, 150.0],  # K
        [1 / 365, 0.25, 1.0, 5.0],  # T
        [-0.01, 0.0, 0.05],  # r
        [0.05, 0.3, 1.2],  # sigma
        [0.0, 0.04],  # q
    )
)


def test_put_call_parity_to_1e10():
    worst = 0.0
    for S, K, T, r, sigma, q in PARITY_GRID:
        c = black_scholes_price(S, K, T, r, sigma, "call", q)
        p = black_scholes_price(S, K, T, r, sigma, "put", q)
        worst = max(worst, abs((c - p) - (S * math.exp(-q * T) - K * math.exp(-r * T))))
    assert worst < 1e-10


GREEK_CASES = list(
    itertools.product(
        [80.0, 100.0, 120.0],  # S
        [0.1, 1.0, 3.0],  # T
        [0.15, 0.45],  # sigma
        [0.0, 0.03],  # q
        ["call", "put"],
    )
)


def _price(S, K, T, r, sigma, kind, q):
    return black_scholes_price(S, K, T, r, sigma, kind, q)


@pytest.mark.parametrize("S,T,sigma,q,kind", GREEK_CASES)
def test_greeks_match_central_finite_differences(S, T, sigma, q, kind):
    K, r = 100.0, 0.04
    option = BlackScholes(S, K, T, r, sigma, kind, q=q)
    g = option.greeks()
    h = 1e-4 * S  # spot bump
    e = 1e-5  # bump for T, sigma and r
    fd = {
        "delta": (_price(S + h, K, T, r, sigma, kind, q) - _price(S - h, K, T, r, sigma, kind, q)) / (2 * h),
        "gamma": (
            _price(S + h, K, T, r, sigma, kind, q)
            - 2 * g["price"]
            + _price(S - h, K, T, r, sigma, kind, q)
        ) / (h * h),
        "theta": -(_price(S, K, T + e, r, sigma, kind, q) - _price(S, K, T - e, r, sigma, kind, q)) / (2 * e),
        "vega": (_price(S, K, T, r, sigma + e, kind, q) - _price(S, K, T, r, sigma - e, kind, q)) / (2 * e) / 100,
        "rho": (_price(S, K, T, r + e, sigma, kind, q) - _price(S, K, T, r - e, sigma, kind, q)) / (2 * e) / 100,
    }  # fmt: skip
    for name, approx in fd.items():
        # Central differences carry O(h^2) truncation and rounding error; the
        # gamma second difference is the noisiest.
        tol = 1e-5 if name == "gamma" else 1e-7
        assert g[name] == pytest.approx(approx, rel=tol, abs=tol), name


IV_GRID = list(
    itertools.product(
        [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 2.0],  # K / S
        [1 / 365, 7 / 365, 30 / 365, 0.25, 1.0, 3.0],  # T
        [0.05, 0.2, 0.5, 1.0],  # sigma
        ["call", "put"],
    )
)


def test_implied_vol_round_trip_across_moneyness_and_expiry():
    S, r, q = 100.0, 0.03, 0.01
    solved = 0
    for moneyness, T, sigma, kind in IV_GRID:
        option = BlackScholes(S, S * moneyness, T, r, sigma, kind, q=q)
        price = option.price()
        disc_k = S * moneyness * math.exp(-r * T)
        disc_s = S * math.exp(-q * T)
        lower = (
            max(disc_s - disc_k, 0.0) if kind == "call" else max(disc_k - disc_s, 0.0)
        )
        noise = 1e-13 * max(disc_s, disc_k) if lower > 0 else 0.0
        if price - lower <= noise:
            # No time value left in double precision (or a price that
            # underflowed to zero): no unique volatility.
            with pytest.raises(ValueError):
                option.implied_volatility(price)
            continue
        iv = option.implied_volatility(price)
        solved += 1
        repriced = BlackScholes(S, S * moneyness, T, r, iv, kind, q=q).price()
        assert repriced == pytest.approx(price, abs=1e-8)
        # Volatility error is bounded by the solver tolerance plus price
        # rounding divided by vega (the problem's condition number).
        vega = option.vega() * 100
        assert abs(iv - sigma) <= 1e-9 + 1e-11 * S / vega
    assert solved > 250


@pytest.mark.parametrize(
    "price,match",
    [
        (0.0, "positive"),
        (-1.0, "positive"),
        (float("nan"), "positive"),
        (100.0, "upper bound"),
        (5.0, "lower bound"),
    ],
)
def test_implied_vol_errors_when_no_solution(price, match):
    # Call with S=100, K=90, T=1, r=5%: bounds are (100 - 90e^-0.05, 100).
    option = BlackScholes(100, 90, 1.0, 0.05, 0.2, "call")
    with pytest.raises(ValueError, match=match):
        option.implied_volatility(price)
