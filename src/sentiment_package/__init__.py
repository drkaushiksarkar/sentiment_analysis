"""Utilities for training sentiment and sarcasm classifiers."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sentiment-platform")
except PackageNotFoundError:  # pragma: no cover - local editable install
    __version__ = "0.0.0"

__all__ = ["__version__"]
