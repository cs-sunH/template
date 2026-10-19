import os
import warnings
from typing import Dict, List, Optional, Tuple

from .architecture import AtlasArch
from pyta.hotspot_manager import HotspotManager
from pyta.plot_thermal_map import plot_hotspot_thermal_map


def run_atlas_hotspot_simulation(
    name: str,
    tile_array_size: Tuple[int, int],
    tile_shape: Tuple[float, float],
    num_memory_layer: int,
    cooling_style: str,
    interposer_size: float,
    logic_power: float,
    dram_power: float,
    output_dir: str,
    num_grid: int = 64,
    tile_block_list: Optional[List[str]] = None,
    hotspot_manager: Optional[HotspotManager] = None,
) -> Dict[str, float]:
    arch = AtlasArch(
        name=name,
        tile_array_size=tile_array_size,
        tile_shape=tile_shape,
        num_memory_layer=num_memory_layer,
        cooling_style=cooling_style,
        interposer_size=interposer_size,
        tile_block_list=tile_block_list if tile_block_list is not None else ["logic"],
    )

    config = arch.get_pyta_config()
    nx, ny = tile_array_size
    for row_id in range(ny):
        for col_id in range(nx):
            arch.set_logic_tile_power(config, power_list=[logic_power], row_id=row_id, col_id=col_id)
            arch.set_dram_tile_power(config, power_list=[dram_power], row_id=row_id, col_id=col_id)

    config.dump_hotspot_config(output_dir)
    if hotspot_manager is None:
        hotspot_manager = HotspotManager()
    return hotspot_manager.run(output_dir, num_grid=num_grid, force_rerun=True)


def thermal_evaluator(
    name: str,
    # Chip physical information
    tile_array_size: Tuple[int, int],
    tile_shape: Tuple[float, float],
    num_memory_layer: int,
    cooling_style: str,
    interposer_size: float,
    # Power information
    base_frequency: float,
    logic_power: float,
    dram_power: float,
    # Frequency scaling information
    temperature_threshold: float, # Celsius
    frequency_power_dict: Optional[Dict[float, float]] = None,
    # Output directory
    output_dir: str = "",
):
    if output_dir == "":
        output_dir = os.path.join(os.path.dirname(__file__), "temp", name)
    os.makedirs(output_dir, exist_ok=True)

    hotspot_manager = HotspotManager()
    cur_frequency = base_frequency

    r = run_atlas_hotspot_simulation(
        name=name,
        tile_array_size=tile_array_size,
        tile_shape=tile_shape,
        num_memory_layer=num_memory_layer,
        cooling_style=cooling_style,
        interposer_size=interposer_size,
        logic_power=logic_power,
        dram_power=dram_power,
        output_dir=output_dir,
        num_grid=64,
        tile_block_list=["logic"],
        hotspot_manager=hotspot_manager,
    )

    max_temperature_c = r['max_temperature'] - 273.15
    if max_temperature_c > temperature_threshold:
        frequency_power_dict = frequency_power_dict or {}
        frequency_list = sorted(frequency_power_dict.keys(), reverse=True)
        for cur_frequency in frequency_list:
            cur_logic_power = frequency_power_dict[cur_frequency]
            r = run_atlas_hotspot_simulation(
                name=name,
                tile_array_size=tile_array_size,
                tile_shape=tile_shape,
                num_memory_layer=num_memory_layer,
                cooling_style=cooling_style,
                interposer_size=interposer_size,
                logic_power=cur_logic_power,
                dram_power=dram_power,
                output_dir=output_dir,
                num_grid=64,
                tile_block_list=["logic"],
                hotspot_manager=hotspot_manager,
            )
            max_temperature_c = r['max_temperature'] - 273.15
            if max_temperature_c <= temperature_threshold:
                break
        if max_temperature_c > temperature_threshold:
            warnings.warn(
                f"[{name}] No frequency in frequency_power_dict satisfies the temperature threshold. "
                f"Final frequency: {cur_frequency} MHz, "
                f"max_temperature: {r['max_temperature']:.2f} K, "
                f"{max_temperature_c:.2f} ℃, "
                f"threshold: {temperature_threshold:.2f} ℃",
                RuntimeWarning,
            )

    plot_hotspot_thermal_map(
        output_dir,
        n_row=64,
        n_col=64,
        t_delta=273.15,
        tmin=45,
        tmax=85,
        floorplan_path=os.path.join(output_dir, "L0_logic_die.flp"),
        layer_idx=0,
    )

    print(
        f"[{name}] "
        f"base_frequency: {base_frequency} MHz, "
        f"current frequency: {cur_frequency} MHz, "
        f"max_temperature: {r['max_temperature']:.2f} K, "
        f"{max_temperature_c:.2f} ℃"
    )

    return {
        **r,
        "max_temperature_c": max_temperature_c,
        "frequency_MHz": cur_frequency,
        "output_dir": output_dir,
        "thermal_map_png": os.path.join(output_dir, "thermal_map.png"),
        "thermal_map_pdf": os.path.join(output_dir, "thermal_map.pdf"),
    }
