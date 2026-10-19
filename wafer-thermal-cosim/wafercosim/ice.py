"""Persistent 3D-ICE 4RM liquid reference backend.

The native model is one regular source plane coupled to a 3D-ICE 4RM channel
stack. Project thermal blocks are rotated so LiquidRC's west-to-east row flow
maps to 3D-ICE's south-to-north (+y) channel flow. Project watts are deposited
by exact planar area overlap; returned block temperatures are area-weighted
means over that same active source-layer footprint.
"""

from __future__ import annotations

import math
import selectors
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from .thermal import LiquidRC


class ThreeDiceThermal(LiquidRC):
    """3D-ICE 4RM backend implementing the LiquidRC thermal interface.

    The 3D-ICE native state and sparse factorization remain alive in one bridge
    process. This first adapter supports a single regular 4RM channel region,
    one active source layer, uniform y cells, and a fixed stack timestep.
    """

    name = "three_dice_4rm_reference"

    def __init__(self, config: dict[str, Any], bridge: str | Path, workdir: str | Path):
        super().__init__(config)
        self.name = "three_dice_4rm_reference"
        self.energy_residual_j = None
        self.workdir = Path(workdir).resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.options = dict(config.get("thermal", {}).get("three_dice", {}))
        self.initial_k = float(config["thermal"]["initial_k"])
        self.inlet_k = float(config["thermal"]["inlet_k"])
        self.step_s = float(self.options.get("step_s", config["thermal"]["window_s"]))
        if not math.isfinite(self.step_s) or self.step_s <= 0:
            raise ValueError("3D-ICE stack step_s must be finite and positive")

        self.channel_height_um = float(self.options.get("channel_height_um", 100.0))
        self.channel_length_um = float(self.options.get("channel_length_um", 50.0))
        self.wall_length_um = float(self.options.get("wall_length_um", 50.0))
        self.first_wall_length_um = float(self.options.get("first_wall_length_um", 25.0))
        self.last_wall_length_um = float(self.options.get("last_wall_length_um", 25.0))
        self.grid_cell_width_um = float(self.options.get("grid_cell_width_um", 1000.0))
        self.active_layer_um = float(self.options.get("active_layer_um", 50.0))
        self.passive_die_um = float(self.options.get("passive_die_um", 50.0))
        self.flow_ml_min = float(self.options.get("flow_rate_ml_min", self._flow_ml_min_from_rc_config(config)))
        self.htc_side_w_m2k = float(self.options.get("htc_side_w_m2k", 27132.0))
        self.htc_top_w_m2k = float(self.options.get("htc_top_w_m2k", 47132.0))
        self.htc_bottom_w_m2k = float(self.options.get("htc_bottom_w_m2k", 57132.0))
        self.coolant_vhc_j_um3k = float(
            self.options.get(
                "coolant_vhc_j_um3k",
                config["thermal"]["coolant_density_kg_m3"] * config["thermal"]["coolant_cp_j_kgk"] / 1e18,
            )
        )
        self.bridge = Path(bridge).resolve()
        if not self.bridge.is_file():
            raise FileNotFoundError(f"3D-ICE bridge does not exist: {self.bridge}")

        self.ice_block_rects = self._rotated_block_rects()
        self.chip_length_um, self.chip_width_um, self.n_channels = self._domain_geometry()
        self.flow_area_m2 = (
            self.n_channels * self.channel_length_um * 1e-6 * self.channel_height_um * 1e-6
        )
        self.flow_lanes = int(config["hardware"]["ny"])
        if self.flow_lanes <= 0:
            raise ValueError("hardware.ny must be positive to map RC coolant rows")
        self.total_coolant_mass_flow_kg_s = (
            self.flow_ml_min * 1e-6 / 60.0 * float(config["thermal"]["coolant_density_kg_m3"])
        )
        self.rc_coolant_segment_volume_m3 = (
            self.flow_area_m2 / self.flow_lanes * float(config["hardware"]["module_width_m"])
        )
        self.native_cells_per_source_layer = None
        self.native_rows = None
        self.native_columns = None
        self.native_source_layer = None
        self.native_step_count = 0
        self.time_s = 0.0
        self.last_input_power_w = 0.0
        self.block_temperature_semantics = "area-weighted mean over active 3D-ICE source-layer cells"
        self.die_max_semantics = "maximum project-block mean temperature for the die"

        self._validate_options()
        stack_text, source_floorplan, manifest = self._input_files()
        self.stack_path = self._write_same_or_new("stack.stk", stack_text)
        self.source_floorplan_path = self._write_same_or_new("source.flp", source_floorplan)
        self.manifest_path = self._write_same_or_new("project-blocks.map", manifest)
        self.export_floorplan(self.workdir / "project-floorplan.flp")

        self.bridge_log_path = self.workdir / "ice_bridge.stderr.log"
        try:
            self.log = self.bridge_log_path.open("x", encoding="utf-8")
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite existing 3D-ICE bridge log: {self.bridge_log_path}"
            ) from exc

        self.process = subprocess.Popen(
            [str(self.bridge), "--stack", self.stack_path.name, "--blocks", self.manifest_path.name],
            cwd=self.workdir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            bufsize=1,
        )
        self.selector = selectors.DefaultSelector()
        assert self.process.stdout is not None
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        try:
            header = self._readline()
            if not header.startswith("# time_s") or len(header.split(",")) != self.nsolid + 1:
                raise RuntimeError(f"3D-ICE bridge protocol mismatch; see {self.bridge_log_path}")
        except Exception:
            self.close()
            raise

        # .T remains the per-project-block solid temperature vector expected by
        # runtime.py. The complete persistent solid/coolant state lives inside
        # the retained 3D-ICE ThermalData_t in the child process.
        self.T = np.full(self.nsolid, self.initial_k, dtype=float)
        self.energy_residual_j = None

    @staticmethod
    def _flow_ml_min_from_rc_config(config: dict[str, Any]) -> float:
        thermal = config["thermal"]
        row_flows = thermal.get("coolant_mass_flow_kg_s_by_row")
        if row_flows is not None:
            row_flows = [float(value) for value in row_flows]
            if len(row_flows) != int(config["hardware"]["ny"]):
                raise ValueError("coolant_mass_flow_kg_s_by_row must have one value per hardware row")
            if any(not math.isclose(value, row_flows[0], rel_tol=1e-9, abs_tol=1e-12) for value in row_flows[1:]):
                raise ValueError(
                    "3D-ICE 4RM uses one total flow shared uniformly across parallel channels; "
                    "nonuniform per-row flow is outside this adapter's domain"
                )
            mass_flow = sum(row_flows)
        else:
            mass_flow = float(thermal["coolant_mass_flow_kg_s_per_row"]) * int(config["hardware"]["ny"])
        density = float(thermal["coolant_density_kg_m3"])
        if not math.isfinite(mass_flow) or mass_flow <= 0:
            raise ValueError("total RC coolant mass flow must be finite and positive")
        return mass_flow / density * 60.0 * 1e6

    def _validate_options(self) -> None:
        positive = {
            "channel_height_um": self.channel_height_um,
            "channel_length_um": self.channel_length_um,
            "wall_length_um": self.wall_length_um,
            "first_wall_length_um": self.first_wall_length_um,
            "last_wall_length_um": self.last_wall_length_um,
            "grid_cell_width_um": self.grid_cell_width_um,
            "active_layer_um": self.active_layer_um,
            "passive_die_um": self.passive_die_um,
            "flow_rate_ml_min": self.flow_ml_min,
            "coolant_vhc_j_um3k": self.coolant_vhc_j_um3k,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"3D-ICE option {name} must be finite and positive")
        for name, value in [
            ("htc_side_w_m2k", self.htc_side_w_m2k),
            ("htc_top_w_m2k", self.htc_top_w_m2k),
            ("htc_bottom_w_m2k", self.htc_bottom_w_m2k),
        ]:
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"3D-ICE option {name} must be finite and nonnegative")
        if not np.all(np.isfinite(self.T[: self.nsolid])):
            raise ValueError("LiquidRC geometry shell produced invalid initial temperatures")

    def _rotated_block_rects(self) -> list[tuple[float, float, float, float]]:
        # Project RC flow is +x within each row. Rotating x/y makes the same
        # order enter 3D-ICE's documented +y channel direction.
        return [
            (b.y * 1e6, b.x * 1e6, b.height * 1e6, b.width * 1e6)
            for b in self.blocks
        ]

    def _domain_geometry(self) -> tuple[int, int, int]:
        hardware = self.config["hardware"]
        # Preserve the full declared module plane, including unpowered space:
        # its channels still carry coolant and define lane area/residence time.
        module_transverse_um = int(hardware["ny"]) * float(hardware["module_height_m"]) * 1e6
        module_flow_length_um = int(hardware["nx"]) * float(hardware["module_width_m"]) * 1e6
        max_x = max(
            module_transverse_um,
            max(x + width for x, _y, width, _height in self.ice_block_rects),
        )
        max_y = max(
            module_flow_length_um,
            max(y + height for _x, y, _width, height in self.ice_block_rects),
        )
        base = self.first_wall_length_um + self.last_wall_length_um + self.channel_length_um
        pitch = self.channel_length_um + self.wall_length_um
        channels_after_first = max(0, math.ceil((max_x - base - 1e-8) / pitch))
        chip_length = base + channels_after_first * pitch
        rows = max(1, math.ceil((max_y - 1e-8) / self.grid_cell_width_um))
        chip_width = rows * self.grid_cell_width_um
        # The parser computes NChannels=(NColumns-1)/2, where the 4RM grid has
        # 2*n+3 columns and n=channels_after_first.
        n_channels = channels_after_first + 1
        if chip_length + 1e-8 < max_x or chip_width + 1e-8 < max_y:
            raise AssertionError("derived 3D-ICE domain does not cover project blocks")
        return int(round(chip_length)), int(round(chip_width)), n_channels

    def _input_files(self) -> tuple[str, str, str]:
        h = self.config["thermal"]
        silicon_k_w_umk = float(h["silicon_k_w_mk"]) / 1e6
        silicon_vhc_j_um3k = float(h["silicon_volumetric_heat_capacity_j_m3k"]) / 1e18

        def number(value: float) -> str:
            return f"{float(value):.12g}"

        floorplan = (
            "project_source:\n"
            "  position 0, 0 ;\n"
            f"  dimension {self.chip_length_um}, {self.chip_width_um} ;\n"
            "  power values 0.0 ;\n"
        )
        stack = f"""material silicon :
   thermal conductivity     {number(silicon_k_w_umk)} ;
   volumetric heat capacity {number(silicon_vhc_j_um3k)} ;

microchannel 4rm :
   height {number(self.channel_height_um)} ;
   channel length {number(self.channel_length_um)} ;
   wall length {number(self.wall_length_um)} ;
   first wall length {number(self.first_wall_length_um)} ;
   last wall length {number(self.last_wall_length_um)} ;
   wall material silicon ;
   coolant flow rate {number(self.flow_ml_min)} ;
   coolant heat transfer coefficient side {number(self.htc_side_w_m2k / 1e12)} ,
                                      top {number(self.htc_top_w_m2k / 1e12)} ,
                                   bottom {number(self.htc_bottom_w_m2k / 1e12)} ;
   coolant volumetric heat capacity {number(self.coolant_vhc_j_um3k)} ;
   coolant incoming temperature {number(self.inlet_k)} ;

dimensions :
   chip length {self.chip_length_um}, width {self.chip_width_um} ;
   cell length {number(self.channel_length_um)}, width {number(self.grid_cell_width_um)} ;
   non-uniform false ;

layer passive_backing :
   height {number(self.passive_die_um)} ;
   material silicon ;

die active_die :
   source {number(self.active_layer_um)} silicon ;

stack:
   die active active_die floorplan "source.flp" ;
   channel channel1 ;
   layer passive passive_backing ;

solver:
   transient step {number(self.step_s)}, slot {number(self.step_s)} ;
   initial temperature {number(self.initial_k)} ;
   numofcores 1 ;

output:
   T ( active, 0, 0, "bridge-unused.txt", step ) ;
"""
        manifest_lines = [f"ICEBLOCKS 1 {self.nsolid}"]
        manifest_lines.extend(" ".join(number(v) for v in rect) for rect in self.ice_block_rects)
        manifest = "\n".join(manifest_lines) + "\n"
        return stack, floorplan, manifest

    def _write_same_or_new(self, name: str, content: str) -> Path:
        path = self.workdir / name
        if path.exists():
            if path.read_text(encoding="utf-8") != content:
                raise FileExistsError(f"refusing to replace different existing 3D-ICE input: {path}")
            return path
        path.write_text(content, encoding="utf-8")
        return path

    def _readline(self) -> str:
        assert self.process.stdout is not None
        if not self.selector.select(timeout=60):
            self.process.kill()
            raise TimeoutError(f"3D-ICE bridge did not reply within 60 s; see {self.bridge_log_path}")
        line = self.process.stdout.readline()
        if not line:
            code = self.process.poll()
            raise RuntimeError(f"3D-ICE bridge exited ({code}); see {self.bridge_log_path}")
        return line.strip()

    def step(self, power, dt):
        if not math.isfinite(float(dt)) or float(dt) <= 0:
            raise ValueError("thermal dt must be finite and positive")
        if abs(float(dt) - self.step_s) > max(1e-12, self.step_s * 1e-9):
            raise ValueError(
                f"3D-ICE factorization uses fixed step {self.step_s:.12g} s; "
                f"received {float(dt):.12g} s. Rebuild a stack for the requested dt."
            )
        values = np.asarray(power, dtype=float)
        if values.shape != (self.nsolid,) or np.any(values < 0) or not np.all(np.isfinite(values)):
            raise ValueError("nonnegative finite W required for every project solid block")
        assert self.process.stdin is not None
        self.process.stdin.write(",".join([f"{self.step_s:.12g}", *(f"{v:.12g}" for v in values)]) + "\n")
        self.process.stdin.flush()
        row = np.asarray([float(x) for x in self._readline().split(",")], dtype=float)
        if row.shape != (self.nsolid + 1,) or not np.all(np.isfinite(row)):
            raise RuntimeError(f"3D-ICE bridge returned invalid row; see {self.bridge_log_path}")
        if abs(row[0] - (self.time_s + self.step_s)) > max(1e-10, self.step_s * 1e-7):
            raise RuntimeError("3D-ICE bridge time does not match the supplied step sequence")
        self.time_s = float(row[0])
        self.T[: self.nsolid] = row[1:]
        self.native_step_count += 1
        self.last_input_power_w = float(values.sum())
        self.energy_residual_j = None
        return self.T[: self.nsolid].copy()

    def close(self):
        process = getattr(self, "process", None)
        if process is not None:
            if process.poll() is None:
                if process.stdin is not None:
                    process.stdin.close()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if process.stdout is not None:
                process.stdout.close()
        selector = getattr(self, "selector", None)
        if selector is not None:
            selector.close()
        log = getattr(self, "log", None)
        if log is not None and not log.closed:
            log.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
