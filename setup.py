"""Compatibility shim.

All package metadata lives in ``pyproject.toml``. This file exists only so that
legacy tooling / ``python setup.py`` style installs keep working.
"""

from setuptools import setup

if __name__ == "__main__":
    setup()
