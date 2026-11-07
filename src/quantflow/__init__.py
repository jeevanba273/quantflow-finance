"""Old import name of QuantFlow Finance, kept for the 0.3.x series only.

QuantFlow Finance 0.3.2 renamed the import package to ``quantflow_finance``,
because the unrelated PyPI project quantflow (quantmind/quantflow) installs a
top-level module with the same name. This module makes ``import quantflow``
and ``from quantflow.<submodule> import ...`` return the same objects as
``quantflow_finance`` and emits a FutureWarning. It is removed in 0.4.0.
"""

import importlib
import pkgutil
import sys
import warnings

import quantflow_finance as _package
from quantflow_finance import (  # noqa: F401
    BinomialTree,
    BlackScholes,
    MarketData,
    MonteCarlo,
    MonteCarloResult,
    Portfolio,
    RiskMetrics,
    __all__,
    __version__,
    black_scholes_greeks,
    black_scholes_price,
    data,
    options,
    risk,
)

warnings.warn(
    "QuantFlow Finance is now imported as quantflow_finance; the old name "
    "quantflow is removed in 0.4.0 because another PyPI project uses it. "
    "Replace quantflow with quantflow_finance in your import lines (for "
    "example: from quantflow_finance import BlackScholes), or pin "
    "quantflow-finance<0.4.",
    FutureWarning,
    stacklevel=2,
)

# Map every submodule, so that for example quantflow.options.black_scholes is
# the module quantflow_finance.options.black_scholes itself.
for _info in pkgutil.walk_packages(_package.__path__, prefix="quantflow_finance."):
    _old_name = "quantflow" + _info.name[len("quantflow_finance") :]
    sys.modules[_old_name] = importlib.import_module(_info.name)
