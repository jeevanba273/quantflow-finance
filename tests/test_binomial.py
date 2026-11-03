"""Tests for the CRR BinomialTree option pricing model."""

import math

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


def test_non_finite_params_raise():
    with pytest.raises(ValueError):
        BinomialTree(float("nan"), 100, 1, 0.05, 0.2)
    with pytest.raises(ValueError):
        BinomialTree(100, 100, 1, float("inf"), 0.2)


def test_two_step_lattice_greeks_match_hand_computation():
    """Hull's lattice delta, gamma and theta formulas on an explicit 2-step tree."""
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    dt = T / 2
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    p = (math.exp(r * dt) - d) / (u - d)
    disc = math.exp(-r * dt)
    f_uu, f_ud, f_dd = max(S * u * u - K, 0), max(S - K, 0), max(S * d * d - K, 0)
    f_u = disc * (p * f_uu + (1 - p) * f_ud)
    f_d = disc * (p * f_ud + (1 - p) * f_dd)
    f_0 = disc * (p * f_u + (1 - p) * f_d)
    delta = (f_u - f_d) / (S * u - S * d)
    gamma = ((f_uu - f_ud) / (S * u * u - S) - (f_ud - f_dd) / (S - S * d * d)) / (
        0.5 * (S * u * u - S * d * d)
    )
    theta = (f_ud - f_0) / (2 * dt)

    tree = BinomialTree(S, K, T, r, sigma, "call", "european", steps=2)
    assert tree.price() == pytest.approx(f_0, rel=1e-12)
    assert tree.delta() == pytest.approx(delta, rel=1e-12)
    assert tree.gamma() == pytest.approx(gamma, rel=1e-12)
    assert tree.theta() == pytest.approx(theta, rel=1e-12)


def test_one_step_tree_has_no_lattice_gamma_or_theta():
    tree = BinomialTree(100, 100, 1, 0.05, 0.2, steps=1)
    assert math.isfinite(tree.delta())
    with pytest.raises(ValueError, match="steps >= 2"):
        tree.gamma()
    with pytest.raises(ValueError, match="steps >= 2"):
        tree.theta()


@pytest.mark.parametrize("kwargs", [dict(exercise=None), dict(steps=True)])
def test_bad_exercise_or_steps_type_raises_value_error(kwargs):
    with pytest.raises(ValueError):
        BinomialTree(100, 100, 1, 0.05, 0.2, **kwargs)
