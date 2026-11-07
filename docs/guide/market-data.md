# Market data

{class}`~quantflow_finance.MarketData` downloads prices from Yahoo Finance through the
optional `yfinance` package and turns prices into returns. The rest of the
library does not depend on it.

```bash
pip install "quantflow-finance[data]"
```

## Downloading prices

The fetch methods need network access, and what they return depends on the
day and on Yahoo's data, which this library does not check. They are not
run by the test suite.

```python
from quantflow_finance import MarketData

prices = MarketData.fetch_stock_data(["SPY", "TLT", "GLD"], period="5y")
batch = MarketData.fetch_single_stock_batch(["SPY", "TLT"], period="1y", batch_delay=2)
```

`fetch_stock_data` retries up to `max_tries` times with a random pause and
raises `RuntimeError` if every attempt fails. `fetch_single_stock_batch`
downloads one ticker at a time and leaves out the ones that fail. Progress
and failures go to the `quantflow_finance.data.fetcher` logger; enable it with
`logging.basicConfig(level=logging.INFO)`. Without yfinance installed both
methods raise an `ImportError` that says how to install it.

## Prices to returns

{meth}`~quantflow_finance.MarketData.calculate_returns` works offline on any price
series or frame:

```python
import numpy as np
import pandas as pd

from quantflow_finance import MarketData

prices = pd.Series([100.0, 101.0, np.nan, 99.0, 100.5],
                   index=pd.bdate_range("2024-01-01", periods=5))
print(MarketData.calculate_returns(prices, "simple").round(6))
print(MarketData.calculate_returns(prices, "log").round(6))
```

```text
2024-01-02    0.010000
2024-01-05    0.015152
dtype: float64
2024-01-02    0.009950
2024-01-05    0.015038
dtype: float64
```

Rows with a missing return are dropped rather than forward-filled, so a gap
does not create an artificial zero return. Use simple returns for
{class}`~quantflow_finance.Portfolio`, because portfolio returns are weighted sums of
simple returns.
