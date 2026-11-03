"""
Execute the Python examples in the README and the documentation.

A ```python block followed directly by a ```text block must print exactly
that text (trailing whitespace ignored). Blocks in one file run in order in a
shared namespace. Blocks that download data (they call MarketData.fetch_*)
are skipped because they need the network.
"""

import contextlib
import io
import pathlib
import re
import warnings

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
FILES = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
FENCE = re.compile(r"^```(\w+)[^\n]*\n(.*?)^```[ \t]*$", re.M | re.S)
NETWORK = ("fetch_stock_data", "fetch_single_stock_batch")


def code_blocks(text):
    """Yield (line, code, expected_output_or_None) for each python block."""
    matches = list(FENCE.finditer(text))
    for i, m in enumerate(matches):
        if m.group(1) != "python":
            continue
        expected = None
        if i + 1 < len(matches):
            nxt = matches[i + 1]
            if nxt.group(1) == "text" and not text[m.end() : nxt.start()].strip():
                expected = nxt.group(2)
        yield text[: m.start()].count("\n") + 1, m.group(2), expected


def _normalise(s):
    return "\n".join(line.rstrip() for line in s.strip("\n").splitlines())


DOC_FILES = [
    f for f in FILES if any(True for _ in code_blocks(f.read_text(encoding="utf-8")))
]


@pytest.mark.parametrize(
    "path", DOC_FILES, ids=[str(f.relative_to(ROOT)) for f in DOC_FILES]
)
def test_documented_examples(path):
    namespace = {"__name__": "__docs__"}
    for line, code, expected in code_blocks(path.read_text(encoding="utf-8")):
        if any(name in code for name in NETWORK):
            continue
        out = io.StringIO()
        with contextlib.redirect_stdout(out), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exec(compile(code, f"{path.name}:{line}", "exec"), namespace)
        if expected is not None:
            assert _normalise(out.getvalue()) == _normalise(expected), (
                f"{path.name} line {line}"
            )
