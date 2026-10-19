"""Validate thermal geometry independently of the host execution model."""
from __future__ import annotations

import math
from collections.abc import Mapping


def thermal_geometry(config: Mapping) -> Mapping:
    """Return explicit v2 geometry; execution inputs belong to the joint host.

    Die placement and thermal subdivisions are physical model inputs. They are
    deliberately distinct from the host's Core grid and network topology.
    """
    if config.get("schema_version") != 2:
        raise ValueError("thermal config requires schema_version=2; migrate hardware to geometry")
    obsolete = {"hardware", "model", "workload", "simulation", "compute"}.intersection(config)
    if obsolete:
        raise ValueError(
            "execution inputs belong to the joint host, not thermal config: "
            + ", ".join(sorted(obsolete))
        )
    allowed = {
        "schema_version", "geometry", "thermal", "thermal_calibration",
        "energy", "calibration", "metadata",
    }
    unexpected_config = config.keys() - allowed
    if unexpected_config:
        raise ValueError("unsupported thermal config fields: " + ", ".join(sorted(unexpected_config)))
    geometry = config.get("geometry")
    if not isinstance(geometry, Mapping):
        raise ValueError("thermal config requires a geometry object")
    required = {
        "die_columns", "die_rows", "module_width_m", "module_height_m",
        "compute_side_m", "compute_groups_per_side", "hbm", "router",
    }
    missing = required - geometry.keys()
    unexpected = geometry.keys() - required
    if missing or unexpected:
        raise ValueError(
            f"invalid geometry fields: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    for name in ("die_columns", "die_rows", "compute_groups_per_side"):
        value = geometry[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"geometry.{name} must be a positive integer")

    def dimension(value, name, *, allow_zero=False):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a finite dimension in metres")
        if not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
            raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")

    for name in ("module_width_m", "module_height_m", "compute_side_m"):
        dimension(geometry[name], f"geometry.{name}")

    def within_module(extent, limit, name):
        if extent > limit and not math.isclose(extent, limit, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError(f"geometry.{name} extends beyond its module")

    within_module(geometry["compute_side_m"], geometry["module_width_m"], "compute_side_m")
    within_module(geometry["compute_side_m"], geometry["module_height_m"], "compute_side_m")
    for kind in ("hbm", "router"):
        rectangle = geometry[kind]
        if not isinstance(rectangle, Mapping) or set(rectangle) != {"x_m", "y_m", "width_m", "height_m"}:
            raise ValueError(f"geometry.{kind} requires x_m, y_m, width_m and height_m")
        for name, value in rectangle.items():
            dimension(value, f"geometry.{kind}.{name}", allow_zero=name in {"x_m", "y_m"})
        within_module(rectangle["x_m"] + rectangle["width_m"], geometry["module_width_m"], kind)
        within_module(rectangle["y_m"] + rectangle["height_m"], geometry["module_height_m"], kind)
    obsolete_controls = {"throttle_k", "resume_k", "throttle_ratio", "idle_cooldown_s"}
    if obsolete_controls.intersection(config.get("thermal", {})):
        raise ValueError("unused throttle settings were removed; the joint host owns execution control")
    return geometry
