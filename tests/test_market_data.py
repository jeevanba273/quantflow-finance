"""Tests for the MarketData fetcher (offline via mocked yfinance)."""

import numpy as np
import pandas as pd
import pytest

from quantflow import MarketData


def test_fetch_single(mock_yfinance):
    data = MarketData.fetch_stock_data("AAPL", period="6mo")
    assert isinstance(data, pd.DataFrame)
    assert "AAPL" in data.columns
    assert (data["AAPL"] > 0).all()


def test_fetch_multiple(mock_yfinance):
    data = MarketData.fetch_stock_data(["AAPL", "MSFT"], period="1y")
    assert {"AAPL", "MSFT"}.issubset(data.columns)


@pytest.mark.parametrize("method", ["simple", "log"])
def test_returns_length_and_no_nan(mock_yfinance, method):
    prices = MarketData.fetch_stock_data("MSFT", period="3mo")
    returns = MarketData.calculate_returns(prices, method=method)
    assert len(returns) == len(prices) - 1
    assert not returns.isna().any().any()


def test_bad_return_method(sample_prices):
    with pytest.raises(ValueError):
        MarketData.calculate_returns(sample_prices, method="quadratic")


def test_empty_returns_raises():
    with pytest.raises(ValueError):
        MarketData.calculate_returns(pd.DataFrame())


def test_calculate_returns_interior_nan_consistent():
    # An interior NaN must not produce a spurious 0% simple return; simple and
    # log returns should agree on the surviving observation.
    prices = pd.Series([100.0, np.nan, 110.0, 121.0])
    simple = MarketData.calculate_returns(prices, "simple")
    log = MarketData.calculate_returns(prices, "log")
    assert len(simple) == len(log) == 1
    assert np.allclose(simple.values, np.expm1(log.values))


def test_fetch_empty_raises(mock_yfinance_empty):
    with pytest.raises(Exception):
        MarketData.fetch_stock_data("FAKE", period="1mo", max_tries=1)


@pytest.mark.network
def test_live_smoke():
    """Optional live smoke test; deselected in CI with -m 'not network'."""
    data = MarketData.fetch_stock_data("AAPL", period="5d")
    assert not data.empty
