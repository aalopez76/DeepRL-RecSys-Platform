# Sphinx configuration for DeepRL-RecSys-Platform docs

import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))

project = "DeepRL-RecSys-Platform"
copyright = "2026, AALP"
author = "AALP"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True
