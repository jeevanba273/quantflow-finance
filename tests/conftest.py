"""
Shared pytest fixtures for the QuantFlow test suite.

Includes deterministic data fixtures and an offline mock of yfinance so the
market-data tests never touch the network.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def atm_call():
    from quantflow import BlackScholes

    return BlackScholes(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="call")


@pytest.fixture
def atm_put():
    from quantflow import BlackScholes

    return BlackScholes(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="put")


@pytest.fixture
def sample_returns():
    """A reproducible daily return series (252 observations)."""
    rng = np.random.default_rng(42)
    idx = pd.bdate_range("2023-01-02", periods=252)
    return pd.Series(rng.normal(0.0005, 0.012, 252), index=idx, name="asset")


@pytest.fixture
def sample_prices():
    """A reproducible single-asset price frame (252 observations)."""
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2023-01-02", periods=252)
    px = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, 252)))
    return pd.DataFrame({"AAPL": px}, index=idx)


@pytest.fixture
def sample_multi_returns():
    """A reproducible multi-asset return frame (300 observations, 3 assets)."""
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2022-01-03", periods=300)
    return pd.DataFrame(
        {
            "A": rng.normal(0.0006, 0.011, 300),
            "B": rng.normal(0.0004, 0.009, 300),
            "C": rng.normal(0.0008, 0.015, 300),
        },
        index=idx,
    )


def _fake_history(n=180, start=150.0, seed=1):
    """Build a deterministic OHLCV frame resembling yfinance output."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n)
    close = start * np.exp(np.cumsum(rng.normal(0.0004, 0.01, n)))
    return pd.DataFrame(
        {
            "Open": close * 0.99,
            "High": close * 1.01,
            "Low": close * 0.98,
            "Close": close,
            "Adj Close": close,
            "Volume": rng.integers(1_000_000, 5_000_000, n),
        },
        index=idx,
    )


@pytest.fixture
def mock_yfinance(monkeypatch):
    """Patch the yfinance alias used inside the fetcher with offline stand-ins."""
    import quantflow.data.fetcher as fetcher

    class _Ticker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, period="1y", interval="1d"):
            return _fake_history(seed=abs(hash(self.symbol)) % 1000)

    def _download(
        tickers, period="1y", interval="1d", progress=False, threads=False, **kwargs
    ):
        if isinstance(tickers, str):
            tickers = [tickers]
        frames = {t: _fake_history(seed=i)["Adj Close"] for i, t in enumerate(tickers)}
        adj = pd.concat(frames, axis=1)
        adj.columns = pd.MultiIndex.from_product([["Adj Close"], list(tickers)])
        return adj

    monkeypatch.setattr(fetcher.yf, "Ticker", _Ticker)
    monkeypatch.setattr(fetcher.yf, "download", _download)


@pytest.fixture
def mock_yfinance_empty(monkeypatch):
    """Patch yfinance so every fetch returns empty data."""
    import quantflow.data.fetcher as fetcher

    class _EmptyTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, period="1y", interval="1d"):
            return pd.DataFrame()

    monkeypatch.setattr(fetcher.yf, "Ticker", _EmptyTicker)
