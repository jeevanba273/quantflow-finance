"""
Cross-checks against QuantLib and vollib (Jaeckel's "Let's Be Rational").

Both libraries are optional; each test is skipped when its library is not
installed. Install them with ``pip install "quantflow-finance[validation]"``.

QuantLib: https://www.quantlib.org (AnalyticEuropeanEngine, BinomialVanillaEngine
with the "crr" tree, FdBlackScholesVanillaEngine).

vollib: https://github.com/vollib/vollib, implied volatility by
Jaeckel, P. (2015). Let's Be Rational. Wilmott, 2015(75), 40-53.
"""

import importlib
import itertools

import pytest

from quantflow_finance import BinomialTree, BlackScholes

EURO_GRID = list(
    itertools.product(
        [80.0, 100.0, 120.0],  # S
        [90.0, 110.0],  # K
        [30, 182, 730],  # days to expiry
        [0.01, 0.06],  # r
        [0.0, 0.03],  # q
        [0.15, 0.6],  # sigma
        ["call", "put"],
    )
)


@pytest.fixture(scope="module")
def ql():
    return pytest.importorskip("QuantLib")


class _QL:
    """Minimal QuantLib set-up: flat curves, ACT/365F, so T = days / 365."""

    def __init__(self, ql):
        self.ql = ql
        self.today = ql.Date(15, 1, 2025)
        ql.Settings.instance().evaluationDate = self.today
        self.dc = ql.Actual365Fixed()

    def process(self, S, r, q, sigma):
        ql, today, dc = self.ql, self.today, self.dc
        return ql.BlackScholesMertonProcess(
            ql.QuoteHandle(ql.SimpleQuote(S)),
            ql.YieldTermStructureHandle(ql.FlatForward(today, q, dc)),
            ql.YieldTermStructureHandle(ql.FlatForward(today, r, dc)),
            ql.BlackVolTermStructureHandle(
                ql.BlackConstantVol(today, ql.NullCalendar(), sigma, dc)
            ),
        )

    def option(self, K, days, kind, american=False):
        ql = self.ql
        maturity = self.today + days
        flag = ql.Option.Call if kind == "call" else ql.Option.Put
        exercise = (
            ql.AmericanExercise(self.today, maturity)
            if american
            else ql.EuropeanExercise(maturity)
        )
        option = ql.VanillaOption(ql.PlainVanillaPayoff(flag, K), exercise)
        return option, self.dc.yearFraction(self.today, maturity)


def test_european_price_and_greeks_match_quantlib(ql):
    env = _QL(ql)
    for S, K, days, r, q, sigma, kind in EURO_GRID:
        option, T = env.option(K, days, kind)
        option.setPricingEngine(ql.AnalyticEuropeanEngine(env.process(S, r, q, sigma)))
        ours = BlackScholes(S, K, T, r, sigma, kind, q=q).greeks()
        theirs = {
            "price": option.NPV(),
            "delta": option.delta(),
            "gamma": option.gamma(),
            "theta": option.theta(),
            "vega": option.vega() / 100,  # QuantLib: per unit of vol
            "rho": option.rho() / 100,  # QuantLib: per unit of rate
        }
        for name, value in theirs.items():
            assert ours[name] == pytest.approx(value, rel=1e-10, abs=1e-12), name


def test_implied_vol_matches_quantlib(ql):
    env = _QL(ql)
    compared = 0
    for S, K, days, r, q, sigma, kind in EURO_GRID:
        option, T = env.option(K, days, kind)
        process = env.process(S, r, q, sigma)
        option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
        price = option.NPV()
        ours_option = BlackScholes(S, K, T, r, sigma, kind, q=q)
        if ours_option.vega() * 100 < 1e-3:
            continue  # no usable time value: the volatility is not determined
        theirs = option.impliedVolatility(price, process, 1e-12, 1000, 1e-4, 5.0)
        assert ours_option.implied_volatility(price) == pytest.approx(theirs, abs=1e-9)
        compared += 1
    assert compared > 120


@pytest.mark.parametrize("steps", [100, 500, 2000])
@pytest.mark.parametrize("kind,q", [("put", 0.0), ("put", 0.03), ("call", 0.05)])
def test_american_tree_matches_quantlib_crr(ql, steps, kind, q):
    # QuantLib's "crr" tree uses a log-space probability
    # p = 1/2 + (r - q - sigma^2/2) dt / (2 sigma sqrt(dt)) rather than
    # (e^{(r-q)dt} - d) / (u - d), so the two trees differ at O(1/N).
    env = _QL(ql)
    S, K, r, sigma = 100.0, 105.0, 0.04, 0.25
    option, T = env.option(K, 365, kind, american=True)
    option.setPricingEngine(
        ql.BinomialVanillaEngine(env.process(S, r, q, sigma), "crr", steps)
    )
    ours = BinomialTree(S, K, T, r, sigma, kind, "american", steps, q).price()
    assert ours == pytest.approx(option.NPV(), abs=0.05 / steps)


@pytest.mark.parametrize("kind,q", [("put", 0.0), ("call", 0.05)])
def test_american_tree_close_to_quantlib_finite_differences(ql, kind, q):
    env = _QL(ql)
    S, K, r, sigma = 100.0, 105.0, 0.04, 0.25
    option, T = env.option(K, 365, kind, american=True)
    option.setPricingEngine(
        ql.FdBlackScholesVanillaEngine(env.process(S, r, q, sigma), 2000, 2000)
    )
    ours = BinomialTree(S, K, T, r, sigma, kind, "american", 2000, q).price()
    assert ours == pytest.approx(option.NPV(), abs=3e-3)


def _lets_be_rational():
    for name in (
        "vollib.black_scholes_merton.implied_volatility",
        "py_vollib.black_scholes_merton.implied_volatility",
    ):
        try:
            return importlib.import_module(name).implied_volatility
        except ImportError:
            continue
    pytest.skip("vollib / py_vollib not installed")


IV_GRID = list(
    itertools.product(
        [0.6, 0.8, 0.95, 1.0, 1.05, 1.25, 1.6],  # K / S
        [7 / 365, 30 / 365, 0.25, 1.0, 2.0],  # T
        [0.08, 0.25, 0.6],  # sigma
        ["call", "put"],
    )
)


def test_implied_vol_matches_lets_be_rational():
    lbr = _lets_be_rational()
    S, r, q = 100.0, 0.03, 0.01
    compared = 0
    for moneyness, T, sigma, kind in IV_GRID:
        option = BlackScholes(S, S * moneyness, T, r, sigma, kind, q=q)
        price = option.price()
        if option.vega() * 100 < 1e-3:
            continue  # ill-conditioned: both solvers lose digits here
        ours = option.implied_volatility(price)
        theirs = lbr(price, S, S * moneyness, T, r, q, kind[0])
        assert ours == pytest.approx(theirs, abs=1e-9)
        compared += 1
    assert compared > 150
