#!/usr/bin/env python3
"""Create isolated tiny Online inputs for the active JOINT checkout.

All generated Python bindings, runtime configs, manifests, queues, and plans
live under work/joint-reproduction.  Source trees are copied read-only as an
import/config staging area so the repo's normal resolver cannot write into
astra-sim-joint/sh_test_mesh/generated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def _repo_root() -> Path:
    override = os.environ.get("JOINT_REPRO_REPO_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return Path(
        subprocess.check_output(
            ["git", "-C", str(HERE), "rev-parse", "--show-toplevel"],
            text=True,
        ).strip()
    ).resolve()


ROOT = _repo_root()
SOURCE = Path(os.environ.get("ASTRA_SIM_ROOT", ROOT / "astra-sim-joint")).resolve()
REPRO_WORK = Path(
    os.environ.get("JOINT_REPRO_WORKDIR", ROOT / "work" / "joint-reproduction")
).resolve()
STAGE = REPRO_WORK / "stage"
SH_TEST = STAGE / "sh_test_mesh"
WORKLOAD = SH_TEST / "workload" / "llama2_7b_inference"
GENERATED = SH_TEST / "generated"
PROTO_SOURCE = SOURCE / "extern" / "graph_frontend" / "chakra" / "schema" / "protobuf"
PROTO_STAGE = STAGE / "extern" / "graph_frontend" / "chakra" / "schema" / "protobuf"


def _copy_source_stage() -> None:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    shutil.copytree(
        SOURCE / "sh_test_mesh",
        SH_TEST,
        ignore=shutil.ignore_patterns(
            "generated", "runs", "__pycache__", ".pytest_cache", "*.pyc", "*.pyo"
        ),
    )
    PROTO_STAGE.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PROTO_SOURCE / "et_def.proto", PROTO_STAGE / "et_def.proto")
    coarse_template = SH_TEST / "system" / "llama2_7b_coarse_intended_template.json"
    template = json.loads(
        (SH_TEST / "system" / "llama2_7b_die_template.json").read_text(
            encoding="utf-8"
        )
    )
    template["die-model-profile"] = "coarse_die_v2"
    coarse_template.write_text(json.dumps(template, indent=2) + "\n", encoding="utf-8")
    subprocess.run(
        [
            "protoc",
            f"--proto_path={PROTO_SOURCE}",
            f"--python_out={PROTO_STAGE}",
            str(PROTO_SOURCE / "et_def.proto"),
        ],
        check=True,
    )


def _write_request_queue(path: Path) -> None:
    fields = (
        "session_id",
        "turn_index",
        "request_id",
        "prefill_length",
        "decode_length",
        "session_arrival_time_ns",
        "inter_request_interval_ns",
        "next_trigger_type",
        "description",
    )
    with path.open("w", newline="", encoding="utf-8") as sink:
        writer = csv.DictWriter(sink, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "session_id": "joint_tiny_s0",
                "turn_index": 0,
                "request_id": "joint_tiny_s0_r0",
                "prefill_length": 4,
                "decode_length": 2,
                "session_arrival_time_ns": 0,
                "inter_request_interval_ns": "",
                "next_trigger_type": "human",
                "description": "one-request production Online smoke input",
            }
        )


def _write_queue_provenance(path: Path) -> None:
    """Write the same fail-closed provenance contract as trace materializer."""
    payload = path.read_bytes()
    fnv = 14695981039346656037
    for byte in payload:
        fnv = ((fnv ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF

    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    sessions: set[str] = set()
    closed_sessions: set[str] = set()
    current_session = ""
    arrivals: list[int] = []
    for row in rows:
        session = row["session_id"]
        if session != current_session:
            if current_session:
                closed_sessions.add(current_session)
            if session in closed_sessions:
                raise ValueError("tiny request queue has non-contiguous sessions")
            current_session = session
        sessions.add(session)
        if row["session_arrival_time_ns"]:
            arrivals.append(int(row["session_arrival_time_ns"]))

    inversions = sum(left > right for left, right in zip(arrivals, arrivals[1:]))
    provenance = {
        "schema": 1,
        "generator_version": "joint-reproduction tiny request fixture v1",
        "csv_sha256": hashlib.sha256(payload).hexdigest(),
        "csv_fnv1a64": fnv,
        "csv_bytes": len(payload),
        "data_rows": len(rows),
        "sessions": len(sessions),
        "turn0_count": len(arrivals),
        "turn0_arrival_min_ns": min(arrivals, default=0),
        "turn0_arrival_max_ns": max(arrivals, default=0),
        "turn0_adjacent_inversions": inversions,
        "session_blocks_contiguous": True,
    }
    path.with_suffix(path.suffix + ".provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_trace_config(path: Path, variant: str, queue_path: Path) -> None:
    source_config = SOURCE / "sh_test_mesh" / "workload" / "llama2_7b_inference" / "trace_config.csv"
    with source_config.open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    scalar_updates = {
        "layers": "1",
        # The fixed FACE mesh has nine TP=6 instances. Keep dimensions
        # divisible across the real topology so the tiny graph has no empty
        # rank shards.
        "hidden_size": "384",
        "ffn_size": "768",
        "num_heads": "12",
        "vocab_size": "384",
        "bytes_per_elem": "2",
        "model_name": f"joint_tiny_{variant}",
        "output_prefix": f"joint_tiny_{variant}",
        "output_dir": "",
        "request_queue_csv": str(queue_path),
        "request_queue_session_limit": "0",
        "trace_granularity": "request_aggregated",
        "prefill_chunk_size": "4",
        "kv_reserve_context_tokens": "64",
        "remote_operand_loads": "false",
        "position_encoding": "rope",
    }
    kept_rows = []
    for row in rows:
        if row["kind"] == "config" and row["key"] in scalar_updates:
            row["value"] = scalar_updates[row["key"]]
        if (
            variant == "coarse_intended"
            and row["kind"] == "config"
            and row["key"] == "system_template"
        ):
            row["value"] = "system/llama2_7b_coarse_intended_template.json"
        kept_rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as sink:
        writer = csv.DictWriter(sink, fieldnames=kept_rows[0].keys())
        writer.writeheader()
        writer.writerows(kept_rows)


def _run_materializer(config_path: Path, die_cabi: Path) -> Path:
    sys.path.insert(0, str(WORKLOAD))
    sys.path.insert(0, str(SH_TEST))
    sys.path.insert(0, str(STAGE))
    import generate_face_trace as gft  # type: ignore[import-not-found]
    import plan_materializer as materializer  # type: ignore[import-not-found]

    gft.SH_TEST_DIR = SH_TEST
    materializer.SH_TEST_DIR = SH_TEST
    materializer.GENERATED_ROOT = GENERATED
    materializer.load_face_trace_config = lambda: gft.load_face_trace_config(config_path)

    import os

    os.environ["DIE_MODEL_CABI_PATH"] = str(die_cabi)
    materializer.main()
    cfg8 = hashlib.sha256(config_path.read_bytes()).hexdigest()[:8]
    plan_dir = GENERATED / f"llama2_7b_inference_54npus_plan_{cfg8}"
    if not (plan_dir / "manifest.json").is_file():
        raise RuntimeError(f"plan materializer did not create {plan_dir}")
    return plan_dir


def _clone_profile_runtime(plan_dir: Path, profile: str) -> tuple[Path, Path]:
    manifest_path = plan_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_resolved = Path(manifest["die-model-config"])
    base_rc = base_resolved.parent
    profiles_root = GENERATED / "runtime_config" / "tiny_profiles"
    profile_rc = profiles_root / profile
    if profile_rc.exists():
        shutil.rmtree(profile_rc)
    shutil.copytree(base_rc, profile_rc)

    if profile == "coarse_intended":
        coarse_source = SOURCE / "extern" / "die_model" / "coarse" / "configs" / "example_hardware.json"
        coarse_config = profile_rc / "resolved_coarse_die.json"
        shutil.copyfile(coarse_source, coarse_config)
        system_path = profile_rc / "system.json"
        system = json.loads(system_path.read_text(encoding="utf-8"))
        system["die-model-profile"] = "coarse_die_v2"
        system["compute-cost-model"] = "die"
        system["die-model-config"] = str(coarse_config.resolve())
        system_path.write_text(json.dumps(system, indent=2) + "\n", encoding="utf-8")

    resolved_path = (
        profile_rc / "resolved_coarse_die.json"
        if profile == "coarse_intended"
        else profile_rc / "resolved_die_model.json"
    )
    if profile == "legacy":
        system_path = profile_rc / "system.json"
        system = json.loads(system_path.read_text(encoding="utf-8"))
        system["die-model-config"] = str(resolved_path.resolve())
        system_path.write_text(json.dumps(system, indent=2) + "\n", encoding="utf-8")
    manifest["compute-cost-model"] = "die"
    manifest["die-model-config"] = str(resolved_path.resolve())
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    metrics_path = plan_dir / "metrics_manifest.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["compute-cost-model"] = "die"
    metrics["die-model-config"] = str(resolved_path.resolve())
    metrics_path.write_text(json.dumps(metrics, separators=(",", ":")) + "\n", encoding="utf-8")
    return profile_rc, base_rc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--die-cabi", type=Path, required=True)
    args = parser.parse_args()
    die_cabi = args.die_cabi.resolve()
    if not die_cabi.is_file():
        parser.error(f"missing JOINT legacy C ABI: {die_cabi}")

    _copy_source_stage()
    queue_path = WORKLOAD / "joint_tiny_requests.csv"
    _write_request_queue(queue_path)
    _write_queue_provenance(queue_path)

    plan_dirs: dict[str, Path] = {}
    runtime_dirs: dict[str, Path] = {}
    active_runtime_dirs: dict[str, Path] = {}
    for profile, config_name in (
        ("legacy", "trace_config_legacy.csv"),
        ("coarse_intended", "trace_config_coarse_intended.csv"),
    ):
        config_path = WORKLOAD / config_name
        _write_trace_config(config_path, profile, queue_path)
        plan_dir = _run_materializer(config_path, die_cabi)
        runtime_dir, active_runtime_dir = _clone_profile_runtime(plan_dir, profile)
        plan_dirs[profile] = plan_dir
        runtime_dirs[profile] = runtime_dir
        active_runtime_dirs[profile] = active_runtime_dir

    (REPRO_WORK / "tiny-inputs.json").write_text(
        json.dumps(
            {
                "source_head": subprocess.check_output(
                    ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
                ).strip(),
                "request_csv": str(queue_path),
                "request_count": 1,
                "prefill_tokens": 4,
                "decode_tokens": 2,
                "configured_instances": 9,
                "tp_ranks_per_instance": 6,
                "configured_ranks": list(range(54)),
                "legacy": {
                    "trace_config": str(WORKLOAD / "trace_config_legacy.csv"),
                    "plan_dir": str(plan_dirs["legacy"]),
                    "runtime_config_dir": str(runtime_dirs["legacy"]),
                    "active_runtime_config_dir": str(active_runtime_dirs["legacy"]),
                },
                "coarse_intended": {
                    "trace_config": str(WORKLOAD / "trace_config_coarse_intended.csv"),
                    "plan_dir": str(plan_dirs["coarse_intended"]),
                    "runtime_config_dir": str(runtime_dirs["coarse_intended"]),
                    "active_runtime_config_dir": str(active_runtime_dirs["coarse_intended"]),
                    "scope": "legacy FACE resolver materializes mesh inputs; only the per-rank die descriptor/profile are changed to the coarse example so the actual JOINT provider guard is reached. This is a fail-closed integration probe, not a hardware-mapping or performance run.",
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print((REPRO_WORK / "tiny-inputs.json").read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
