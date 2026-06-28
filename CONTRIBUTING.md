# Contributing to QuantFlow Finance

Thanks for your interest in contributing! This guide covers how to set up a
development environment, run the tests, and submit changes.

## Development setup

QuantFlow uses a `src/` layout and a PEP 621 `pyproject.toml`. Install the
package in editable mode with the development extras:

```sh
git clone https://github.com/jeevanba273/quantflow-finance
cd quantflow-finance
python -m venv .venv
# Windows: .venv\Scripts\activate    |    macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
```

## Running the tests

The test suite is pytest-based and runs fully offline (Yahoo Finance is mocked):

```sh
pytest                       # run everything except live-network tests by default config
pytest -m "not network"      # offline only (what CI runs)
pytest -m network            # opt in to the live yfinance smoke test
pytest --cov=quantflow       # with coverage
```

New tests must be deterministic and offline. Use the fixtures in
`tests/conftest.py` (`atm_call`, `sample_returns`, `sample_multi_returns`,
`mock_yfinance`, ...) rather than hitting the network, and seed any randomness
(`numpy.random.default_rng(seed)`).

## Code style

```sh
black .          # format (line length 88)
flake8 .         # lint
```

- Follow the existing NumPy-style docstrings with a `Notes` math block and an
  `Examples` section for public methods.
- Add type hints to public signatures.
- Validate inputs and raise `ValueError` with a clear message for bad arguments;
  use `warnings.warn(UserWarning)` for unusual-but-valid inputs.

## Backward compatibility

This package is published on PyPI. Changes must be **additive**:

- Do not rename or remove public classes, methods, or their import paths.
- Do not reorder existing positional parameters. New parameters are appended last
  with backward-compatible defaults.
- Existing calls with default arguments must return identical results.

## Pull request checklist

1. Branch from `main` with a descriptive name (e.g. `feature/monte-carlo`).
2. Keep changes additive and backward-compatible (see above).
3. Add offline tests covering the new behavior; ensure `pytest -m "not network"`
   passes locally.
4. Run `black .` and `flake8 .`.
5. Update `CHANGELOG.md` under `[Unreleased]`/the next version.
6. Ensure CI is green across the supported Python versions (3.8-3.13).

## Commit messages

Write clear, concise, professional commit messages as a human developer would.
Do not include AI/assistant attribution or automation trailers.

## Reporting bugs

Open an issue at
<https://github.com/jeevanba273/quantflow-finance/issues> with a minimal
reproduction, the expected vs. actual behavior, and your Python / package
versions.
