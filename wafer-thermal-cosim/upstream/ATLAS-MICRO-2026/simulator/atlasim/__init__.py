import sys
from .atlasim import *

# Helper function to inject kwargs support into C++ classes
def _inject_kwargs_init(cls):
    original_init = cls.__init__
    
    def __init__(self, **kwargs):
        # Call original (empty) init
        original_init(self)
        # Set attributes from kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise TypeError(f"'{cls.__name__}' object has no attribute '{key}'")
    
    cls.__init__ = __init__

# List of classes to patch
_classes_to_patch = [
    ControllerConfig,
    MatrixConfig,
    VectorConfig,
    BufferConfig,
    DRAMConfig,
    NoCConfig,
    ChipConfig,
    AttnInput,
    DAttnInput,
    DAttnCoreInput,
    Stats,
    Performance
]

# Apply patches
for cls in _classes_to_patch:
    _inject_kwargs_init(cls)

# Expose all classes and functions from the C++ extension
__all__ = [
    "Chip",
    "ChipConfig",
    "ControllerConfig",
    "MatrixConfig",
    "VectorConfig",
    "BufferConfig",
    "DRAMConfig",
    "NoCConfig",
    "AttnInput",
    "DAttnInput",
    "DAttnCoreInput",
    "Stats",
    "Performance",
    "SimulatorOperatorType",
]

# Apply patches to SimulatorOperatorType
SimulatorOperatorType.__str__ = SimulatorOperatorType.to_string
