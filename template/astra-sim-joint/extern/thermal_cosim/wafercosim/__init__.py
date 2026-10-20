"""Trimmed thermal-core package embedded inside the joint repository.

The component exposes one long-lived :class:`~wafercosim.session.ThermalSession`
per joint run: the host supplies activity windows, the physical mapping and the
parsed hardware snapshot, and the session returns per-block power, thermal
feedback and (in the host) the throttled resource envelope.
"""

__version__ = "0.2.0"

from .thermal import Block, LiquidRC
from .backends import create_thermal, close_thermal
from .contracts import (
    ACTION_DESTINATIONS,
    ACTION_UNITS,
    ActivityWindow,
    ContractError,
    HardwareSnapshot,
    MappingSnapshot,
    OperatingPoint,
    PowerWindow,
    ResourceEnvelope,
    ThermalFeedback,
    block_kinds,
    build_mapping,
)
from .paths import CONFIGS_DIR, config_path, configs_dir
from .session import ThermalSession

__all__ = [
    "Block", "LiquidRC", "create_thermal", "close_thermal",
    "ACTION_DESTINATIONS", "ACTION_UNITS", "ActivityWindow", "ContractError",
    "HardwareSnapshot", "MappingSnapshot", "OperatingPoint", "PowerWindow",
    "ResourceEnvelope", "ThermalFeedback", "block_kinds", "build_mapping",
    "CONFIGS_DIR", "config_path", "configs_dir", "ThermalSession",
    "__version__",
]
