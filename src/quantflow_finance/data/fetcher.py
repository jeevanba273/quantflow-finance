"""
Market data fetching and preprocessing utilities.

This module provides tools for downloading stock data and calculating returns.
Downloading needs the optional ``yfinance`` dependency
(``pip install "quantflow-finance[data]"``); it is imported only when a fetch
method is called, so the rest of the package never depends on it.
"""

import logging
import random
import time

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _yfinance():
    """Import and return the optional yfinance module, with an install hint."""
    try:
        import yfinance
    except ImportError as exc:
        raise ImportError(
            "Fetching market data needs the optional 'yfinance' package. "
            'Install it with: pip install "quantflow-finance[data]"'
        ) from exc
    return yfinance


class MarketData:
    """
    Download prices from Yahoo Finance and convert prices to returns.

    All methods are static. The fetch methods need the optional ``yfinance``
    package and network access; their output depends on the day they run and
    on Yahoo's data, which this library does not check. Yahoo's terms of use
    apply to the data. :meth:`calculate_returns` works offline on any price
    series.
    """

    @staticmethod
    def fetch_stock_data(
        tickers: str | list[str],
        period: str = "1y",
        interval: str = "1d",
        max_tries: int = 3,
        delay_range: tuple[float, float] = (1, 3),
    ) -> pd.DataFrame:
        """
        Download closing prices for one or more tickers, with retries.

        Parameters
        ----------
        tickers : str or list of str
            Yahoo Finance symbols, e.g. ``"SPY"`` or ``["SPY", "TLT"]``.
        period : str, default '1y'
            History length accepted by yfinance ('1mo', '1y', '5y', 'max', ...).
        interval : str, default '1d'
            Bar size ('1d', '1wk', '1mo', ...).
        max_tries : int, default 3
            Attempts before giving up.
        delay_range : tuple of float, default (1, 3)
            Range in seconds of the random pause between attempts.

        Returns
        -------
        pandas.DataFrame
            One column per ticker, indexed by date. Values are the "Adj
            Close" column when yfinance provides it, otherwise "Close" (which
            yfinance adjusts for splits and dividends by default).

        Raises
        ------
        ImportError
            If yfinance is not installed.
        RuntimeError
            If every attempt fails or returns no data.
        """
        yf = _yfinance()
        if isinstance(tickers, str):
            tickers = [tickers]

        last_exception = None

        for attempt in range(max_tries):
            try:
                logger.info(
                    "Fetching %s (attempt %d/%d)", tickers, attempt + 1, max_tries
                )

                # Add random delay to avoid rate limiting
                if attempt > 0:
                    delay = random.uniform(*delay_range)
                    logger.info("Waiting %.1f seconds before retrying", delay)
                    time.sleep(delay)

                # Let yfinance handle sessions automatically (no custom session)
                if len(tickers) == 1:
                    ticker_obj = yf.Ticker(tickers[0])
                    data = ticker_obj.history(period=period, interval=interval)
                else:
                    # For multiple tickers, use download
                    data = yf.download(
                        tickers,
                        period=period,
                        interval=interval,
                        progress=False,
                        threads=False,
                    )

                # Check if data is empty
                if data.empty:
                    raise ValueError(f"No data returned for tickers: {tickers}")

                # Handle different data structures from yfinance
                if len(tickers) == 1:
                    # Single ticker - data might be a simple DataFrame
                    if "Adj Close" in data.columns:
                        result = data[["Adj Close"]].rename(
                            columns={"Adj Close": tickers[0]}
                        )
                    elif "Close" in data.columns:
                        # Fallback to Close if Adj Close not available
                        result = data[["Close"]].rename(columns={"Close": tickers[0]})
                    else:
                        raise ValueError(f"No price data found for {tickers[0]}")
                else:
                    # Multiple tickers
                    if "Adj Close" in data.columns:
                        result = data["Adj Close"]
                    elif "Close" in data.columns:
                        result = data["Close"]
                    else:
                        raise ValueError(f"No price data found for {tickers}")

                # Validate result has data
                if result.empty or result.isna().all().all():
                    raise ValueError(f"All data is empty or NaN for tickers: {tickers}")

                logger.info("Fetched %d rows", len(result))
                return result

            except Exception as e:
                last_exception = e
                logger.warning("Attempt %d failed: %s", attempt + 1, e)

                # If it's a rate limit error, wait longer before next attempt
                if (
                    "rate limit" in str(e).lower()
                    or "too many requests" in str(e).lower()
                ):
                    if attempt < max_tries - 1:  # Don't wait on the last attempt
                        longer_delay = random.uniform(10, 20)
                        logger.warning(
                            "Rate limited; waiting %.1f seconds", longer_delay
                        )
                        time.sleep(longer_delay)

        # If all attempts failed
        raise RuntimeError(
            f"Failed to fetch data after {max_tries} attempts. Last error: {last_exception}"
        )

    @staticmethod
    def fetch_single_stock_batch(
        tickers: list[str],
        period: str = "1y",
        interval: str = "1d",
        batch_delay: float = 2,
    ) -> pd.DataFrame:
        """
        Download tickers one at a time, skipping those that fail.

        Slower than :meth:`fetch_stock_data` but less likely to be rate
        limited. A ticker that fails is logged and left out.

        Parameters
        ----------
        tickers : list of str
            Yahoo Finance symbols.
        period, interval : str
            As for :meth:`fetch_stock_data`.
        batch_delay : float, default 2
            Pause in seconds between requests.

        Returns
        -------
        pandas.DataFrame
            One column per ticker that was fetched.

        Raises
        ------
        RuntimeError
            If no ticker could be fetched.
        """
        yf = _yfinance()
        all_data = {}

        for i, ticker in enumerate(tickers):
            try:
                logger.info("Fetching %s (%d/%d)", ticker, i + 1, len(tickers))

                # Add delay between requests
                if i > 0:
                    time.sleep(batch_delay)

                # Fetch single stock (let yfinance handle sessions)
                ticker_obj = yf.Ticker(ticker)
                stock_data = ticker_obj.history(period=period, interval=interval)

                if not stock_data.empty:
                    if "Adj Close" in stock_data.columns:
                        all_data[ticker] = stock_data["Adj Close"]
                    elif "Close" in stock_data.columns:
                        all_data[ticker] = stock_data["Close"]
                    else:
                        logger.warning("No price column for %s", ticker)
                else:
                    logger.warning("No data for %s", ticker)

            except Exception as e:
                logger.warning("Failed to fetch %s: %s", ticker, e)
                continue

        if not all_data:
            raise RuntimeError("Failed to fetch data for any ticker")

        # Combine all data into single DataFrame
        result = pd.DataFrame(all_data)
        logger.info("Fetched data for %d tickers", len(result.columns))
        return result

    @staticmethod
    def calculate_returns(
        prices: pd.Series | pd.DataFrame, method: str = "simple"
    ) -> pd.Series | pd.DataFrame:
        r"""
        Period returns from a price series.

        Simple returns are :math:`P_t / P_{t-1} - 1`; log returns are
        :math:`\ln(P_t / P_{t-1})`. Rows with a missing return in any
        column (including the first row) are dropped. Gaps are not
        forward-filled, so no artificial zero returns appear.

        Parameters
        ----------
        prices : pandas.Series or pandas.DataFrame
            Price levels, one column per asset.
        method : {'simple', 'log'}, default 'simple'
            Return definition. Use simple returns for :class:`Portfolio`,
            which weights returns across assets.

        Returns
        -------
        pandas.Series or pandas.DataFrame
            Same type as ``prices``, with at least one row fewer.
        """
        if prices.empty:
            raise ValueError("Cannot calculate returns from empty price data")

        if method == "simple":
            # fill_method=None avoids forward-filling interior gaps (which would
            # emit spurious 0% returns) and the pandas pct_change FutureWarning.
            returns = prices.pct_change(fill_method=None).dropna()
        elif method == "log":
            returns = np.log(prices / prices.shift(1)).dropna()
        else:
            raise ValueError("Method must be 'simple' or 'log'")

        if returns.empty:
            raise ValueError("No valid returns calculated - check your price data")

        return returns
