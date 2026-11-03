# Contributing

Bug reports, corrections and small, tested improvements are welcome. A
numerical bug report is most useful with the inputs, the value you got, the
value you expected and where the expected value comes from.

## Setup

The package uses a `src/` layout and `pyproject.toml`. Python 3.10 or newer.

```sh
git clone https://github.com/jeevanba273/quantflow-finance
cd quantflow-finance
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -e ".[dev,docs,validation]"
```

## Checks

```sh
ruff check .                  # lint
ruff format --check .         # formatting
pytest -m "not network"       # tests, doctests, documented examples, example scripts
python -m sphinx -W -b html docs docs/_build/html   # docs, warnings are errors
python benchmarks/run_benchmarks.py --quick         # benchmark script still runs
```

`pytest` also runs the examples in the docstrings, every Python block in the
README and the docs (`tests/test_docs.py` compares their printed output with
the page), and the scripts in `examples/`. The QuantLib and vollib
cross-checks in `tests/test_reference_libraries.py` are skipped unless the
`validation` extra is installed; CI installs it. The one live-data test is
marked `network` and is not run by default.

## What a change needs

- A test. New numerical code needs a comparison with something independent
  of the code: a published value, a closed form, an identity, a finite
  difference, or another library. Say where the reference comes from.
- Deterministic tests: seed every random generator and never use the
  network.
- Docstrings in NumPy style that state the formula, units and conventions
  and cite the source. Public signatures need type hints.
- Clear errors: raise `ValueError` naming the argument for invalid input;
  use `warnings.warn` for unusual but valid input. Library code does not
  print.
- Documentation updated in the same change. If you change an output that a
  docs page shows, rerun the example and paste what it prints.
- An entry in `CHANGELOG.md`.

Numbers in the docs, README and changelog must come from code in this
repository, with the command that produces them.

## Compatibility

The package is on PyPI, so changes to public names, argument order or
default results need a reason and a changelog entry under "Changed". Bug
fixes that change results are listed under "Fixed" with the old and new
behaviour.

## Reporting bugs

Open an issue at
<https://github.com/jeevanba273/quantflow-finance/issues> with a minimal
reproduction, expected and actual output, and your Python and package
versions.
