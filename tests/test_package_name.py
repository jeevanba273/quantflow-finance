"""QuantFlow Finance 0.4.0 removed the old import name quantflow.

The unrelated PyPI project quantflow (quantmind/quantflow) installs a
top-level module with that name, so this package must not ship one.
"""

import pathlib

import quantflow_finance


def test_old_import_name_is_not_shipped():
    source_root = pathlib.Path(quantflow_finance.__file__).resolve().parents[1]
    old_init = source_root / "quantflow" / "__init__.py"
    # A quantflow module from the other project may sit next to this package
    # in site-packages; only an alias of this package would be an error.
    assert not (
        old_init.exists() and "quantflow_finance" in old_init.read_text("utf-8")
    )
