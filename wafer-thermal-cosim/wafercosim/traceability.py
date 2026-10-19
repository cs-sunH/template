"""Small provenance helpers for the trimmed thermal-core package."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy
import scipy


def _describe(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.is_file():
        return {"path": str(path), "status": "not_found"}
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def run_manifest(config: dict, summary: dict | None = None) -> dict:
    root = Path(__file__).resolve().parents[1]
    summary = summary or {}
    inputs = {}
    for name, value in {
        "energy_table": config.get("calibration", {}).get("energy_table"),
        "cacti_table": config.get("energy", {}).get("cacti_calibration"),
    }.items():
        if value:
            inputs[name] = _describe(value)
    source = {}
    for path in sorted((root / "wafercosim").glob("*.py")):
        source[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "schema_version": 2,
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "python": {"executable": sys.executable, "version": sys.version},
        "packages": {"numpy": numpy.__version__, "scipy": scipy.__version__},
        "inputs": inputs,
        "source_sha256": source,
        "thermal_backend": summary.get("backend"),
        "model_status": "thermal_core_only; joint_host_not_connected",
    }
