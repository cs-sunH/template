"""Small CLI for the trimmed thermal-core package.

Request/KV/performance simulation was deliberately removed from this package.
The joint host supplies PowerWindow data in the future; this CLI only provides
repeatable thermal smoke steps and backend inspection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .backends import close_thermal, create_thermal
from .thermal import LiquidRC


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def _thermal(config: dict, backend: str | None, output: Path):
    config = json.loads(json.dumps(config))
    if backend:
        config.setdefault("thermal", {})["backend"] = backend
    selected = config.get("thermal", {}).get("backend", "rc_liquid")
    if selected == "rc_liquid":
        return LiquidRC(config)
    return create_thermal(config, output)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Trimmed wafer thermal-core smoke CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    step = sub.add_parser("thermal-step", help="advance one backend with uniform power")
    step.add_argument("--config", required=True)
    step.add_argument("--out", default="local-results/thermal-step")
    step.add_argument("--backend", choices=["rc_liquid", "hotspot", "3dice"])
    step.add_argument("--dt-s", type=float, default=None)
    step.add_argument("--steps", type=int, default=1)
    step.add_argument("--power-w", type=float, default=0.0)

    describe = sub.add_parser("describe", help="print backend geometry and capabilities")
    describe.add_argument("--config", required=True)
    describe.add_argument("--backend", choices=["rc_liquid", "hotspot", "3dice"])
    describe.add_argument("--out", default="local-results/thermal-describe")

    args = parser.parse_args(argv)
    config = _load(args.config)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    backend = _thermal(config, args.backend, output)
    elapsed = 0.0
    try:
        dt = getattr(args, "dt_s", None) or float(config["thermal"].get("window_s", 0.01))
        if dt <= 0 or getattr(args, "steps", 1) < 1 or getattr(args, "power_w", 0.0) < 0:
            raise ValueError("dt, steps and power must be positive/nonnegative")
        if args.command == "describe":
            result = {
                "backend": backend.name,
                "nsolid": backend.nsolid,
                "ndie": backend.ndie,
                "blocks": len(backend.blocks),
                "capabilities": {
                    "persistent_state": True,
                    "snapshot": False,
                    "fluid_observation": backend.name == "rc_liquid",
                },
            }
        else:
            power = np.full(backend.nsolid, args.power_w, dtype=float)
            temperatures = None
            for _ in range(args.steps):
                temperatures = backend.step(power, dt)
                elapsed += dt
            result = {
                "backend": backend.name,
                "steps": args.steps,
                "dt_s": dt,
                "elapsed_s": elapsed,
                "nsolid": backend.nsolid,
                "ndie": backend.ndie,
                "peak_solid_temperature_k": float(np.max(temperatures)),
                "die_max_k": backend.die_max().tolist(),
                "energy_residual_j": getattr(backend, "energy_residual_j", None),
            }
        (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, ensure_ascii=False))
        return 0
    finally:
        close_thermal(backend)
