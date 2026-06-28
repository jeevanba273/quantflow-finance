"""
Market data fetching and preprocessing utilities.

This module provides tools for downloading stock data and calculating returns.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import time
import random
from typing import Union, List


class MarketData:
    """
    Market data utilities for fetching and processing financial data.
    """

    @staticmethod
    def fetch_stock_data(
        tickers: Union[str, List[str]],
        period="1y",
        interval="1d",
        max_tries=3,
        delay_range=(1, 3),
    ):
        """
        Fetch stock data from Yahoo Finance with retry logic.

        Parameters:
        tickers: str or list of str - Stock symbols
        period: str - Time period ('1y', '2y', '5y', etc.)
        interval: str - Data interval ('1d', '1wk', '1mo')
        max_tries: int - Maximum number of retry attempts
        delay_range: tuple - Random delay range between retries (min, max seconds)

        Returns:
        pandas DataFrame with adjusted close prices

        Raises:
        Exception: If all retry attempts fail
        """
        if isinstance(tickers, str):
            tickers = [tickers]

        last_exception = None

        for attempt in range(max_tries):
            try:
                print(
                    f"Fetching data for {tickers} (attempt {attempt + 1}/{max_tries})"
                )

                # Add random delay to avoid rate limiting
                if attempt > 0:
                    delay = random.uniform(*delay_range)
                    print(f"Waiting {delay:.1f} seconds before retry...")
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

                print(f"Successfully fetched {len(result)} rows of data")
                return result

            except Exception as e:
                last_exception = e
                print(f"Attempt {attempt + 1} failed: {str(e)}")

                # If it's a rate limit error, wait longer before next attempt
                if (
                    "rate limit" in str(e).lower()
                    or "too many requests" in str(e).lower()
                ):
                    if attempt < max_tries - 1:  # Don't wait on the last attempt
                        longer_delay = random.uniform(10, 20)
                        print(
                            f"Rate limit detected. Waiting {longer_delay:.1f} seconds..."
                        )
                        time.sleep(longer_delay)

        # If all attempts failed
        raise Exception(
            f"Failed to fetch data after {max_tries} attempts. Last error: {last_exception}"
        )

    @staticmethod
    def fetch_single_stock_batch(
        tickers: List[str], period="1y", interval="1d", batch_delay=2
    ):
        """
        Fetch multiple stocks one by one to avoid rate limiting.

        Parameters:
        tickers: list of str - Stock symbols
        period: str - Time period
        interval: str - Data interval
        batch_delay: float - Delay between individual stock requests

        Returns:
        pandas DataFrame with all stocks' adjusted close prices
        """
        all_data = {}

        for i, ticker in enumerate(tickers):
            try:
                print(f"Fetching {ticker} ({i+1}/{len(tickers)})")

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
                        print(f"Warning: No price data for {ticker}")
                else:
                    print(f"Warning: No data for {ticker}")

            except Exception as e:
                print(f"Failed to fetch {ticker}: {str(e)}")
                continue

        if not all_data:
            raise Exception("Failed to fetch data for any ticker")

        # Combine all data into single DataFrame
        result = pd.DataFrame(all_data)
        print(f"Successfully fetched data for {len(result.columns)} stocks")
        return result

    @staticmethod
    def calculate_returns(prices, method="simple"):
        """
        Calculate returns from price data.

        Parameters:
        prices: pandas DataFrame or Series
        method: str - 'simple' or 'log' returns

        Returns:
        pandas DataFrame or Series of returns
        """
        if prices.empty:
            raise ValueError("Cannot calculate returns from empty price data")

        if method == "simple":
            returns = prices.pct_change().dropna()
        elif method == "log":
            returns = np.log(prices / prices.shift(1)).dropna()
        else:
            raise ValueError("Method must be 'simple' or 'log'")

        if returns.empty:
            raise ValueError("No valid returns calculated - check your price data")

        return returns
