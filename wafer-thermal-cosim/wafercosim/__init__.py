"""Trimmed thermal-core package for joint co-simulation integration."""

__version__ = "0.2.0"

from .thermal import Block, LiquidRC
from .backends import create_thermal, close_thermal

__all__ = ["Block", "LiquidRC", "create_thermal", "close_thermal", "__version__"]
