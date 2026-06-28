"""Tests for the CRR BinomialTree option pricing model."""

import pytest

from quantflow import BinomialTree, BlackScholes


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_european_converges_to_bs(option_type):
    bs = BlackScholes(100, 105, 1, 0.05, 0.2, option_type, q=0.02).price()
    bt = BinomialTree(
        100, 105, 1, 0.05, 0.2, option_type, "european", steps=1000, q=0.02
    ).price()
    assert abs(bs - bt) < 1e-2


def test_american_call_no_div_equals_european():
    amer = BinomialTree(100, 100, 1, 0.05, 0.2, "call", "american", steps=1000).price()
    euro = BinomialTree(100, 100, 1, 0.05, 0.2, "call", "european", steps=1000).price()
    assert abs(amer - euro) < 1e-6


def test_american_put_premium_positive():
    amer = BinomialTree(100, 100, 1, 0.05, 0.2, "put", "american", steps=1000)
    euro = BinomialTree(100, 100, 1, 0.05, 0.2, "put", "european", steps=1000)
    assert amer.price() >= euro.price()
    assert amer.early_exercise_premium() > 0


def test_delta_matches_bs():
    bt = BinomialTree(100, 100, 1, 0.05, 0.2, "call", "european", steps=1000)
    bs = BlackScholes(100, 100, 1, 0.05, 0.2, "call")
    assert abs(bt.delta() - bs.delta()) < 1e-2


def test_greeks_keys():
    g = BinomialTree(100, 100, 1, 0.05, 0.2, steps=200).greeks()
    assert {"price", "delta", "gamma", "theta", "vega", "rho"} == set(g)


def test_convergence_improves_with_steps():
    bs = BlackScholes(100, 100, 1, 0.05, 0.2, "call").price()
    err_coarse = abs(
        BinomialTree(100, 100, 1, 0.05, 0.2, "call", "european", steps=100).price() - bs
    )
    err_fine = abs(
        BinomialTree(100, 100, 1, 0.05, 0.2, "call", "european", steps=2000).price()
        - bs
    )
    assert err_fine < err_coarse


def test_invalid_steps_raises():
    with pytest.raises(ValueError):
        BinomialTree(100, 100, 1, 0.05, 0.2, steps=0)


def test_invalid_exercise_raises():
    with pytest.raises(ValueError):
        BinomialTree(100, 100, 1, 0.05, 0.2, exercise="bermudan")
