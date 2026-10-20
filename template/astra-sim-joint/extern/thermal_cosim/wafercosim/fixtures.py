"""Named synthetic fixtures for engineering acceptance of the embedded thermal core.

These fixtures bind a synthetic hardware snapshot to the packaged 4x2 thermal
*profile*.  They exist so the closed-loop plumbing can be exercised end to end
before target floorplan/cooling data is available.  Every fixture is explicitly
labelled synthetic (``target_calibrated=False``) and must never be presented as
a target-device power or thermal result.
"""
from __future__ import annotations

import json

from .action_energy import EnergyModel
from .contracts import HardwareSnapshot, build_mapping
from .paths import config_path
from .session import ThermalSession
from .thermal import LiquidRC

ENERGY_TABLE = "configs/energy/research-v1.json"
MESH = (9, 6)  # 54 workload-capable ranks (joint default), a fixture choice
PRECISION = "int4"


def reference_thermal_config() -> dict:
    """The packaged 4x2 reference thermal profile (geometry/materials/cooling)."""
    return json.loads(config_path("thermal", "3dice_8die_reference.json").read_text())


def energy_table() -> dict:
    return json.loads(config_path("energy", "research-v1.json").read_text())


def synthetic_hardware_facts() -> dict:
    """Host facts matching the packaged research-v1 table domain (synthetic)."""
    table = energy_table()
    domain = table["hardware_domain"]
    sram = table["background"]["sram_leak"]
    noc = domain["noc"]
    topology = noc["supported_topologies"][0]
    return {
        "sram_capacity_bytes_per_die": int(sram["macro_capacity_bytes"] * sram["macro_count_per_die"]),
        "pe_ops_s_per_die": domain["compute"]["pe_ops_s_per_die"],
        "vec_ops_s_per_die": domain["compute"]["vec_ops_s_per_die"],
        "hbm_capacity_bytes_per_die": domain["hbm"]["configured_capacity_bytes_per_die"],
        "dram_capacity_bytes": domain["dram"]["configured_capacity_bytes"],
        "compute": {
            "weight_bytes": domain["compute"]["weight_bytes"],
            "activation_bytes": domain["compute"]["activation_bytes"],
        },
        "noc": {
            "nx": topology["nx"],
            "ny": topology["ny"],
            "flit_width_bits": noc["flit_width_bits"],
            "num_vcs": noc["num_vcs"],
            "packet_size_flits": noc["packet_size_flits"],
            "router": noc["router"],
            "routing": noc["routing"],
        },
        "hbm": {"protocol": domain["hbm"]["protocol"], "config": domain["hbm"]["config"]},
        "dram": {"protocol": domain["dram"]["protocol"], "config": domain["dram"]["config"]},
        "sram": {
            "macro_capacity_bytes": sram["macro_capacity_bytes"],
            "macro_count_per_die": sram["macro_count_per_die"],
            "word_width_bits": sram["source"]["word_bits"],
            "access_bytes": sram["source"]["access_bytes"],
        },
    }


def synthetic_hardware(thermal_config: dict | None = None) -> HardwareSnapshot:
    thermal_config = thermal_config or reference_thermal_config()
    geometry = thermal_config["geometry"]
    die_count = geometry["die_columns"] * geometry["die_rows"]
    facts = synthetic_hardware_facts()
    dies = [
        {
            "die": die,
            "cores_x": 8,
            "cores_y": 11,
            "core_count": 88,
            "die_width_m": geometry["module_width_m"],
            "die_height_m": geometry["module_height_m"],
            "clock_hz": 1.0e9,
            "precision": PRECISION,
        }
        for die in range(die_count)
    ]
    return HardwareSnapshot.build(
        source_path="<synthetic-fixture>",
        units={"sim_time": "ns", "thermal_time": "s", "frequency": "Hz",
               "energy": "J", "power": "W", "temperature": "K", "geometry": "m"},
        mesh_rows=MESH[0],
        mesh_cols=MESH[1],
        dies=dies,
        resource_peaks={
            "pe_ops_s_per_die": facts["pe_ops_s_per_die"],
            "vec_ops_s_per_die": facts["vec_ops_s_per_die"],
            "sram_capacity_bytes_per_die": facts["sram_capacity_bytes_per_die"],
            "hbm_capacity_bytes_per_die": facts["hbm_capacity_bytes_per_die"],
            "dram_capacity_bytes": facts["dram_capacity_bytes"],
        },
        sram=facts["sram"],
        hbm=facts["hbm"],
        dram=facts["dram"],
        noc=facts["noc"],
        d2d_bandwidth_bytes_s=0.0,
        clock_hz=1.0e9,
        precision=PRECISION,
        energy_hardware_facts=facts,
        target_calibrated=False,
        notes=["synthetic fixture; not a target hardware export"],
    )


def synthetic_mapping(hardware: HardwareSnapshot, thermal_config: dict | None = None):
    thermal_config = thermal_config or reference_thermal_config()
    thermal = LiquidRC(thermal_config)
    die_count = thermal_config["geometry"]["die_columns"] * thermal_config["geometry"]["die_rows"]
    rank_to_die = {rank: rank % die_count for rank in range(MESH[0] * MESH[1])}
    return build_mapping(
        thermal.blocks,
        mapping_epoch="fixture-mapping-v1",
        hardware_hash=hardware.content_hash,
        rank_to_die=rank_to_die,
        notes=["synthetic fixture mapping"],
    )


def energy_model() -> EnergyModel:
    return EnergyModel(
        {
            "calibration": {
                "energy_table": ENERGY_TABLE,
                "strict": True,
                "accept_assumed_energy": True,
                "memory_organization_sensitivity_accept": {"hbm": True, "dram": True},
            },
            "thermal": {"initial_k": 300.0},
        },
        hardware_facts=synthetic_hardware_facts(),
    )


def reference_session(run_id: str = "fixture-run") -> ThermalSession:
    thermal_config = reference_thermal_config()
    hardware = synthetic_hardware(thermal_config)
    mapping = synthetic_mapping(hardware, thermal_config)
    return ThermalSession(
        run_id=run_id,
        thermal=LiquidRC(thermal_config),
        energy=energy_model(),
        hardware=hardware,
        mapping=mapping,
    )


def synthetic_resolver_exports(thermal_config: dict | None = None) -> tuple[dict, dict]:
    """Hardware/mapping audit JSON, in the shape ``HardwareSnapshot.from_json`` reads.

    Used by tests that exercise the run-config loader, which consumes the
    resolver's exported files rather than the in-memory fixtures.
    """
    thermal_config = thermal_config or reference_thermal_config()
    hardware = synthetic_hardware(thermal_config)
    mapping = synthetic_mapping(hardware, thermal_config)
    return hardware.to_json(), mapping.to_json()


def synthetic_thermal_facts_export(thermal_config: dict | None = None) -> dict:
    """A hardware-facts export in the joint resolver's schema, for bridge tests."""
    hardware = synthetic_hardware(thermal_config)
    return {
        "schema_version": "joint.thermal-hardware-facts.v1",
        "profile": "synthetic_fixture",
        "source_path": hardware.source_path,
        "capacity_profile": "synthetic",
        "target_calibrated": False,
        "mesh": {"rows": MESH[0], "columns": MESH[1],
                 "npus_count": MESH[0] * MESH[1], "rank_layout": "row_major"},
        "clock_hz": 1.0e9,
        "precision": PRECISION,
        "dies": [dict(row) for row in hardware.dies],
        "resource_peaks": dict(hardware.resource_peaks),
        "sram": dict(hardware.sram),
        "hbm": dict(hardware.hbm),
        "dram": dict(hardware.dram),
        "noc": dict(hardware.noc),
        "d2d_bytes_s": 0.0,
        "energy_hardware_facts": synthetic_hardware_facts(),
        "missing_host_facts": [],
        "notes": ["synthetic fixture"],
    }


__all__ = [
    "ENERGY_TABLE", "MESH", "PRECISION", "reference_thermal_config",
    "energy_table", "synthetic_hardware_facts", "synthetic_hardware",
    "synthetic_mapping", "energy_model", "reference_session",
    "synthetic_thermal_facts_export",
]
