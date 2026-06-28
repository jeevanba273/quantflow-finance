"""Sphinx configuration for the QuantFlow Finance documentation."""

import importlib.metadata

# -- Project information -----------------------------------------------------
project = "QuantFlow Finance"
author = "Jeevan B A"
copyright = "2026, Jeevan B A"

try:
    release = importlib.metadata.version("quantflow-finance")
except importlib.metadata.PackageNotFoundError:  # pragma: no cover
    release = "0.0.0"
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "numpydoc",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}
root_doc = "index"

# -- Autodoc / numpydoc ------------------------------------------------------
autodoc_member_order = "bysource"
autodoc_typehints = "signature"
# yfinance is only needed at call time; mock it so the build never touches the
# network or its heavy import chain.
autodoc_mock_imports = ["yfinance"]
autosummary_generate = True

numpydoc_show_class_members = False
numpydoc_class_members_toctree = False
numpydoc_xref_param_type = True

# -- MyST --------------------------------------------------------------------
myst_enable_extensions = [
    "dollarmath",
    "colon_fence",
    "deflist",
    "fieldlist",
    "substitution",
    "tasklist",
    "attrs_inline",
]
myst_heading_anchors = 3

# -- Intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- HTML output (pydata-sphinx-theme) ---------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = f"{project} {release}"
html_short_title = project

html_theme_options = {
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/jeevanba273/quantflow-finance",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/quantflow-finance/",
            "icon": "fa-brands fa-python",
        },
    ],
    "show_prev_next": True,
    "show_toc_level": 2,
    "header_links_before_dropdown": 6,
}

html_context = {
    "github_user": "jeevanba273",
    "github_repo": "quantflow-finance",
    "github_version": "main",
    "doc_path": "docs",
    "default_mode": "auto",
}
