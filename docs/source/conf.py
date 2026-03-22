import os
import sys

# Path
__location__ = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(__location__, "../.."))

# Project information
project = 'Fishy RL'
copyright = '2026, Noah Cohen Kalafut'
author = 'Noah Cohen Kalafut'

# Take the version from pyproject.toml
with open(os.path.join(__location__, "../..", "pyproject.toml"), "r") as f:
    for line in f:
        if line.startswith('version = '):
            release = line.split('version = ')[1].strip().strip('"')
            break

# Configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
]

templates_path = ['_templates']
exclude_patterns = []

# HTML
html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
