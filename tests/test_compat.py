"""The old import name quantflow, kept with a FutureWarning until 0.4.0.

Each check runs in a fresh interpreter, because the warning is emitted only
the first time the module is imported.
"""

import os
import pathlib
import subprocess
import sys

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"


def _run(code):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(SRC), env.get("PYTHONPATH")) if p
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )


def test_old_name_warns_at_the_import_line():
    result = _run("import quantflow")
    assert result.returncode == 0, result.stderr
    assert "<string>:1: FutureWarning" in result.stderr
    assert "from quantflow_finance import BlackScholes" in result.stderr
    assert "quantflow-finance<0.4" in result.stderr


def test_old_name_gives_the_same_objects():
    code = "\n".join(
        [
            "import warnings",
            "warnings.simplefilter('ignore', FutureWarning)",
            "import quantflow",
            "import quantflow_finance",
            "from quantflow import Portfolio",
            "from quantflow.options.black_scholes import BlackScholes",
            "from quantflow.risk import metrics",
            "assert Portfolio is quantflow_finance.Portfolio",
            "assert BlackScholes is quantflow_finance.BlackScholes",
            "assert metrics is quantflow_finance.risk.metrics",
            "assert quantflow.options.binomial is quantflow_finance.options.binomial",
            "assert quantflow.data.fetcher is quantflow_finance.data.fetcher",
            "assert quantflow.__version__ == quantflow_finance.__version__",
            "assert quantflow.__all__ == quantflow_finance.__all__",
            "for name in quantflow_finance.__all__:",
            "    assert getattr(quantflow, name) is getattr(quantflow_finance, name)",
        ]
    )
    result = _run(code)
    assert result.returncode == 0, result.stderr


def test_new_name_does_not_warn():
    code = "\n".join(
        [
            "import warnings",
            "with warnings.catch_warnings(record=True) as caught:",
            "    warnings.simplefilter('always')",
            "    import quantflow_finance",
            "assert not [w for w in caught if 'quantflow' in str(w.message)], caught",
        ]
    )
    result = _run(code)
    assert result.returncode == 0, result.stderr
