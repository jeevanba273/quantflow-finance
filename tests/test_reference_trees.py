"""
Cox-Ross-Rubinstein trees checked against published values and theory.

Sources
-------
Hull, J. C. Options, Futures, and Other Derivatives (10th ed.), chapter
"Basic Numerical Procedures": American put with S0=50, K=50, r=10%,
sigma=40%, T=5 months. A 5-step tree gives 4.49; 30, 50, 100 and 500 steps
give 4.263, 4.272, 4.278 and 4.283.

Merton, R. C. (1973): an American call on a non-dividend-paying stock is
never exercised early, so it is worth the same as the European call.
"""

import itertools

import pytest

from quantflow_finance import BinomialTree, BlackScholes

HULL_PUT = dict(S=50, K=50, T=5 / 12, r=0.10, sigma=0.40, option_type="put")


def test_hull_five_step_american_put():
    tree = BinomialTree(**HULL_PUT, exercise="american", steps=5)
    assert round(tree.price(), 2) == 4.49


@pytest.mark.parametrize(
    "steps,expected", [(30, 4.263), (50, 4.272), (100, 4.278), (500, 4.283)]
)
def test_hull_american_put_by_step_count(steps, expected):
    tree = BinomialTree(**HULL_PUT, exercise="american", steps=steps)
    assert round(tree.price(), 3) == expected


CASES = list(
    itertools.product(
        [80.0, 100.0, 125.0],  # S
        [0.25, 1.0],  # T
        [0.15, 0.4],  # sigma
        [0.0, 0.03],  # q
        ["call", "put"],
    )
)


@pytest.mark.parametrize("S,T,sigma,q,kind", CASES)
def test_european_tree_converges_to_black_scholes(S, T, sigma, q, kind):
    K, r = 100.0, 0.05
    exact = BlackScholes(S, K, T, r, sigma, kind, q=q).price()
    errors = {
        n: abs(BinomialTree(S, K, T, r, sigma, kind, "european", n, q).price() - exact)
        for n in (50, 200, 800, 3200)
    }
    # CRR error is O(1/N) with an oscillating constant; |error| * N stays
    # bounded and the finest tree is within 2e-3 of the closed form.
    assert max(err * n for n, err in errors.items()) < 5.0
    assert errors[3200] < 2e-3
    assert errors[3200] < errors[50]


@pytest.mark.parametrize("S,T,sigma,q", [(c[0], c[1], c[2], c[3]) for c in CASES[::2]])
def test_american_put_never_below_european_put(S, T, sigma, q):
    args = (S, 100.0, T, 0.05, sigma, "put")
    amer = BinomialTree(*args, "american", 400, q).price()
    euro_tree = BinomialTree(*args, "european", 400, q).price()
    assert amer >= euro_tree
    assert amer >= max(100.0 - S, 0.0)


@pytest.mark.parametrize(
    "S,T,sigma", [(80.0, 1.0, 0.2), (100.0, 0.5, 0.3), (125.0, 2.0, 0.4)]
)
def test_american_call_without_dividends_equals_european(S, T, sigma):
    args = (S, 100.0, T, 0.05, sigma, "call")
    amer = BinomialTree(*args, "american", 1000).price()
    euro = BinomialTree(*args, "european", 1000).price()
    assert amer == pytest.approx(euro, abs=1e-12)
    assert amer == pytest.approx(BlackScholes(*args).price(), abs=5e-3)


def test_american_call_with_high_dividend_has_premium():
    args = (100.0, 100.0, 1.0, 0.05, 0.2, "call")
    amer = BinomialTree(*args, "american", 500, q=0.08)
    assert amer.early_exercise_premium() > 0


@pytest.mark.parametrize("kind", ["call", "put"])
def test_european_tree_greeks_close_to_black_scholes(kind):
    tree = BinomialTree(100, 100, 1.0, 0.05, 0.2, kind, "european", steps=2000)
    exact = BlackScholes(100, 100, 1.0, 0.05, 0.2, kind)
    for name in ("delta", "gamma", "theta", "vega", "rho"):
        assert getattr(tree, name)() == pytest.approx(getattr(exact, name)(), rel=2e-3)
