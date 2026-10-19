"""Construct one temperature-state owner per run."""
from pathlib import Path


def create_thermal(config,out):
    settings=config['thermal'];backend=settings.get('backend','rc_liquid')
    if backend=='rc_liquid':return None
    if backend=='hotspot':
        from .native import HotSpotThermal
        return HotSpotThermal(config,settings.get('hotspot_bridge','build/hotspot_bridge'),
                              settings.get('hotspot_config'),out)
    if backend=='3dice':
        from .ice import ThreeDiceThermal
        return ThreeDiceThermal(config,settings.get('ice_bridge','build/ice_bridge'),out)
    raise ValueError('unknown thermal backend: '+str(backend))


def close_thermal(thermal):
    if thermal is not None and hasattr(thermal,'close'):thermal.close()
