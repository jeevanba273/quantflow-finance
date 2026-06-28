# Market Data

The {class}`~quantflow.MarketData` class collects the utilities QuantFlow uses to obtain raw price history and turn it into the return series that feed every other module — {class}`~quantflow.RiskMetrics`, {class}`~quantflow.Portfolio`, and the option pricers. All of its members are static methods, so you never instantiate the class:

```python
from quantflow import MarketData

prices = MarketData.fetch_stock_data("AAPL", period="2y")
returns = MarketData.calculate_returns(prices, method="log")
```

There are three entry points:

| Method | Purpose | Network |
| --- | --- | --- |
| {meth}`~quantflow.MarketData.fetch_stock_data` | Download one or many tickers in a single call | Yes |
| {meth}`~quantflow.MarketData.fetch_single_stock_batch` | Download many tickers sequentially, rate-limit friendly | Yes |
| {meth}`~quantflow.MarketData.calculate_returns` | Convert a price frame into simple or log returns | No |

:::{note}
Both fetch methods depend on [`yfinance`](https://pypi.org/project/yfinance/) and live Yahoo Finance endpoints, so they require network access and their output is not deterministic across days. The examples in this guide that involve fetching are marked as such; everywhere else we use synthetic price data so the results are reproducible. {meth}`~quantflow.MarketData.calculate_returns` is pure and offline — you can run it on any pandas `Series` or `DataFrame` of prices regardless of where they came from.
:::

## Fetching price data

### `fetch_stock_data`

```python
MarketData.fetch_stock_data(
    tickers,            # str or list[str]
    period="1y",        # "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"
    interval="1d",      # "1d", "1wk", "1mo" (and intraday "1h", "5m", ... for short periods)
    max_tries=3,        # retry attempts on transient failures
    delay_range=(1, 3), # random back-off (seconds) between retries
)
```

The return value is always a pandas `DataFrame` indexed by date. For a **single ticker** the frame has one column named after the symbol; for **multiple tickers** there is one column per symbol. The values are the adjusted close where Yahoo provides it, falling back to the raw close otherwise.

```python
# --- requires network access (live Yahoo Finance) ---
from quantflow import MarketData

# Single ticker -> one-column DataFrame named "AAPL"
aapl = MarketData.fetch_stock_data("AAPL", period="1y", interval="1d")

# Multiple tickers -> one column per symbol, aligned on a shared DatetimeIndex
prices = MarketData.fetch_stock_data(
    ["AAPL", "MSFT", "GOOGL"], period="2y", interval="1wk"
)
print(prices.tail())
```

A representative shape of the multi-ticker result (values will differ by run):

```text
                  AAPL        MSFT       GOOGL
Date
2025-05-26  198.420013  462.970001  171.210007
2025-06-02  201.000000  470.380005  173.680008
2025-06-09  196.580002  474.960007  176.299999
2025-06-16  199.200005  480.240005  178.050003
2025-06-23  201.560001  486.000000  180.770004
```

:::{note}
The frame returned by the fetch methods holds the **adjusted/close price** level, not returns. Always pass it through {meth}`~quantflow.MarketData.calculate_returns` (or your own transform) before handing it to {class}`~quantflow.RiskMetrics` or {class}`~quantflow.Portfolio`, both of which expect a *returns* series/frame.
:::

:::{tip}
`fetch_stock_data` retries up to `max_tries` times with a randomized back-off and waits longer when it detects a rate-limit response. If every attempt fails it raises an exception that includes the last underlying error — wrap the call in `try/except` if you need to fall back to cached data.
:::

### `fetch_single_stock_batch`

When you request a large universe at once, Yahoo Finance may throttle the bundled request. `fetch_single_stock_batch` sidesteps this by downloading each ticker **sequentially**, pausing `batch_delay` seconds between requests, and assembling the columns into a single `DataFrame`:

```python
MarketData.fetch_single_stock_batch(
    tickers,          # list[str]
    period="1y",
    interval="1d",
    batch_delay=2,    # seconds to wait between consecutive ticker requests
)
```

```python
# --- requires network access (live Yahoo Finance) ---
from quantflow import MarketData

universe = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
prices = MarketData.fetch_single_stock_batch(universe, period="1y", batch_delay=2)
print(prices.shape)   # e.g. (251, 5)
```

Unlike `fetch_stock_data`, this method is fault-tolerant per ticker: if one symbol fails to download it prints a warning, skips it, and continues, raising only if **no** ticker could be fetched. It is slower (one HTTP round trip per symbol plus the inter-request delay) but markedly more robust for batches of more than a handful of names.

:::{tip}
Prefer `fetch_stock_data` for one to a few symbols and `fetch_single_stock_batch` for larger baskets or whenever you hit `"Too Many Requests"` errors. Both produce the same column-per-ticker layout, so the rest of your pipeline does not change.
:::

## Computing returns

{meth}`~quantflow.MarketData.calculate_returns` converts a price level series into a return series. It accepts either a `Series` (one asset) or a `DataFrame` (one column per asset) and supports two conventions.

**Simple (arithmetic) returns** — the relative change from one period to the next:

$$
r_t^{\text{simple}} = \frac{P_t}{P_{t-1}} - 1
$$

**Log (continuously compounded) returns** — the natural log of the price ratio:

$$
r_t^{\text{log}} = \ln\!\left(\frac{P_t}{P_{t-1}}\right)
$$

The two are related by $r^{\text{log}} = \ln(1 + r^{\text{simple}})$, so they nearly coincide for small moves and diverge as moves grow. Log returns are time-additive (they sum across periods), which makes them convenient for aggregation; simple returns are weight-additive across assets, which is why they are the natural input for {class}`~quantflow.Portfolio`.

The first observation has no predecessor, so it is dropped: a length-$n$ price series yields $n-1$ returns. Interior gaps are left as `NaN` rather than forward-filled, which avoids fabricating spurious zero-return days.

### Verified example

The following runs entirely offline on a deterministic synthetic price path (`numpy` default RNG, fixed seed), so you can reproduce the exact numbers below:

```python
import numpy as np
import pandas as pd
from quantflow import MarketData

# Deterministic synthetic price path (no network).
rng = np.random.default_rng(42)
dates = pd.date_range("2024-01-01", periods=6, freq="B")
shocks = rng.normal(0.0005, 0.01, size=len(dates))
prices = pd.Series(100.0 * np.exp(np.cumsum(shocks)), index=dates, name="ACME")
print(prices.round(4))
```

```text
2024-01-01    100.3553
2024-01-02     99.3667
2024-01-03    100.1653
2024-01-04    101.1625
2024-01-05     99.2575
2024-01-08     98.0224
Freq: B, Name: ACME, dtype: float64
```

```python
simple = MarketData.calculate_returns(prices, method="simple")
print(simple.round(6))
```

```text
2024-01-02   -0.009851
2024-01-03    0.008037
2024-01-04    0.009955
2024-01-05   -0.018831
2024-01-08   -0.012444
Freq: B, Name: ACME, dtype: float64
```

```python
log = MarketData.calculate_returns(prices, method="log")
print(log.round(6))
```

```text
2024-01-02   -0.009900
2024-01-03    0.008005
2024-01-04    0.009906
2024-01-05   -0.019010
2024-01-08   -0.012522
Freq: B, Name: ACME, dtype: float64
```

Note how the six prices produce five returns, how the simple and log values agree to roughly three decimals on these small moves, and how the first simple return reproduces the formula directly: $99.3667 / 100.3553 - 1 = -0.009851$.

:::{warning}
`calculate_returns` raises `ValueError` on empty input, on an unrecognized `method` (only `"simple"` and `"log"` are accepted), and when the price data yields no valid returns (for example a single row). Make sure your fetched frame has at least two rows before calling it.
:::

## End-to-end: prices to risk and portfolio analytics

A typical workflow fetches prices, converts them to returns, and feeds those returns into the analytics classes. Below, the fetch step is marked as network-dependent; the analytics are demonstrated on a synthetic price frame whose **shape matches a fetched frame** (a `DatetimeIndex` with one column per ticker) so the output is reproducible.

```python
# --- Step 1: fetch prices (requires network access; live Yahoo Finance) ---
# from quantflow import MarketData
# prices = MarketData.fetch_stock_data(["AAPL", "MSFT", "GOOGL"], period="1y")

# --- Step 1 (offline stand-in): a frame with the same shape as a fetched one ---
import numpy as np
import pandas as pd
from quantflow import MarketData, RiskMetrics, Portfolio

rng = np.random.default_rng(0)
dates = pd.date_range("2023-01-02", periods=252, freq="B")
tickers = ["AAPL", "MSFT", "GOOGL"]
drift = np.array([0.0006, 0.0007, 0.0005])
vol = np.array([0.014, 0.012, 0.016])
shocks = rng.normal(drift, vol, size=(len(dates), len(tickers)))
prices = pd.DataFrame(
    100.0 * np.exp(np.cumsum(shocks, axis=0)), index=dates, columns=tickers
)

# --- Step 2: prices -> returns (offline, deterministic) ---
returns = MarketData.calculate_returns(prices, method="simple")
print("returns shape:", returns.shape)   # (251, 3): 252 prices -> 251 returns

# --- Step 3a: single-asset risk analytics on one column ---
rm = RiskMetrics(returns["MSFT"])
print("MSFT VaR(95%):", round(rm.var_historical(), 6))
print("MSFT Sharpe:  ", round(rm.sharpe_ratio(), 4))
print("MSFT ann vol: ", round(rm.volatility(), 4))

# --- Step 3b: portfolio optimisation across the whole frame ---
port = Portfolio(returns)
opt = port.max_sharpe()
print("weights:", {k: round(v, 4) for k, v in opt["weights_by_asset"].items()})
print("return: ", round(opt["return"], 4))
print("vol:    ", round(opt["volatility"], 4))
print("sharpe: ", round(opt["sharpe"], 4))
```

```text
returns shape: (251, 3)
MSFT VaR(95%): -0.016949
MSFT Sharpe:   1.7279
MSFT ann vol:  0.1907
weights: {'AAPL': 0.0, 'MSFT': 0.9692, 'GOOGL': 0.0308}
return:  0.3402
vol:     0.1851
sharpe:  1.7294
```

The key takeaways for wiring market data into the rest of QuantFlow:

- Select a single column (`returns["MSFT"]`) to drive {class}`~quantflow.RiskMetrics`, or pass the full returns frame straight to {class}`~quantflow.Portfolio`.
- {meth}`~quantflow.Portfolio.max_sharpe` returns a result dict with `weights_by_asset` keyed by your original ticker names, because the column labels flow through from the fetched frame untouched.
- Use the same `period`/`interval` for every ticker so the rows align on a common `DatetimeIndex`; the fetch methods already enforce this when you pass a list.

## See also

- {class}`~quantflow.RiskMetrics` — VaR, expected shortfall, Sharpe/Sortino, drawdowns, and CAPM statistics computed on a return series.
- {class}`~quantflow.Portfolio` — covariance estimation, mean-variance optimisation, and efficient frontiers built from a returns frame.
