"""Run every offline example script; warnings are treated as errors."""

import pathlib
import subprocess
import sys

import pytest

EXAMPLES = pathlib.Path(__file__).resolve().parents[1] / "examples"
OFFLINE = sorted(p for p in EXAMPLES.glob("*.py") if p.name != "market_data.py")


@pytest.mark.parametrize("script", OFFLINE, ids=[p.name for p in OFFLINE])
def test_example_runs(script):
    result = subprocess.run(
        [sys.executable, "-W", "error", str(script)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()
