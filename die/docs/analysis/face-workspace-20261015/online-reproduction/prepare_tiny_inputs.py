#!/usr/bin/env python3
"""Prepare isolated tiny or workspace-overlap FACE/FACE-LRU Online inputs.

The helper copies each selected repository's real ``sh_test_mesh`` and
``et_def.proto`` into work/, invokes the FACE config resolver and plan
materializer, and leaves all generated plans/configs outside the source tree.
It intentionally uses the FACE queue, manifest, model dimensions, and 9x6/TP6
topology as one matched input set.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = Path(
    subprocess.check_output(
        ["git", "-C", str(HERE), "rev-parse", "--show-toplevel"], text=True
    ).strip()
).resolve()
SOURCE_VARIANTS = {
    "face": "astra-sim-face",
    "face-lru": "astra-sim-face-LRU",
}
REQUEST_FIELDS = (
    "session_id",
    "turn_index",
    "request_id",
    "prefill_length",
    "decode_length",
    "session_arrival_time_ns",
    "inter_request_interval_ns",
    "description",
)
MODEL = {
    "layers": 1,
    "hidden_size": 384,
    "ffn_size": 768,
    "num_heads": 12,
    "vocab_size": 384,
    "bytes_per_elem": 2,
    "prefill_tokens": 4,
    "decode_tokens": 2,
    "tp_ranks_per_instance": 6,
    "configured_instances": 9,
    "configured_ranks": 54,
}
QUEUE_SCENARIOS = {
    "tiny": (
        {
            "session_id": "face_tiny_session_0",
            "turn_index": 0,
            "request_id": "face_tiny_request_0",
            "prefill_length": MODEL["prefill_tokens"],
            "decode_length": MODEL["decode_tokens"],
            "session_arrival_time_ns": 0,
            "inter_request_interval_ns": "",
            "description": "one-request real FACE Online reproduction",
        },
    ),
    "workspace-overlap": (
        {
            "session_id": "face_overlap_session_0",
            "turn_index": 0,
            "request_id": "face_overlap_request_0",
            "prefill_length": 32,
            "decode_length": MODEL["decode_tokens"],
            "session_arrival_time_ns": 0,
            "inter_request_interval_ns": "",
            "description": "long real FACE prefill held open while a second session arrives",
        },
        {
            "session_id": "face_overlap_session_1",
            "turn_index": 0,
            "request_id": "face_overlap_request_1",
            "prefill_length": MODEL["prefill_tokens"],
            "decode_length": MODEL["decode_tokens"],
            "session_arrival_time_ns": 1_000,
            "inter_request_interval_ns": "",
            "description": "second real FACE session arriving during the first long prefill",
        },
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_file_hashes(tree: Path) -> dict[str, str]:
    ignored = {"generated", "runs", "__pycache__", ".pytest_cache"}
    hashes: dict[str, str] = {}
    for path in sorted(tree.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(tree)
        if any(part in ignored for part in relative.parts):
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        hashes[relative.as_posix()] = _sha256(path)
    return hashes


def _copy_stage(source: Path, stage: Path) -> tuple[dict[str, str], Path, str]:
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    sh_source = source / "sh_test_mesh"
    sh_stage = stage / "sh_test_mesh"
    shutil.copytree(
        sh_source,
        sh_stage,
        ignore=shutil.ignore_patterns(
            "generated", "runs", "__pycache__", ".pytest_cache", "*.pyc", "*.pyo"
        ),
    )
    source_hashes = _source_file_hashes(sh_source)
    staged_hashes = _source_file_hashes(sh_stage)
    if source_hashes != staged_hashes:
        missing = sorted(set(source_hashes) - set(staged_hashes))
        extra = sorted(set(staged_hashes) - set(source_hashes))
        changed = sorted(
            name for name in set(source_hashes) & set(staged_hashes)
            if source_hashes[name] != staged_hashes[name]
        )
        raise RuntimeError(
            "staged sh_test_mesh differs from source: "
            f"missing={missing[:5]} extra={extra[:5]} changed={changed[:5]}"
        )

    proto_source = source / "extern/graph_frontend/chakra/schema/protobuf/et_def.proto"
    if not proto_source.is_file():
        raise FileNotFoundError(f"missing source schema: {proto_source}")
    proto_stage_dir = stage / "extern/graph_frontend/chakra/schema/protobuf"
    proto_stage_dir.mkdir(parents=True, exist_ok=True)
    proto_stage = proto_stage_dir / "et_def.proto"
    shutil.copyfile(proto_source, proto_stage)
    if _sha256(proto_source) != _sha256(proto_stage):
        raise RuntimeError("staged et_def.proto is not byte-identical to source")

    protoc = shutil.which("protoc")
    if not protoc:
        raise RuntimeError("protoc is required to generate staged et_def_pb2.py")
    version = subprocess.check_output([protoc, "--version"], text=True).strip()
    subprocess.run(
        [
            protoc,
            f"--proto_path={proto_source.parent}",
            f"--python_out={proto_stage_dir}",
            str(proto_source),
        ],
        check=True,
    )
    generated_pb2 = proto_stage_dir / "et_def_pb2.py"
    if not generated_pb2.is_file() or generated_pb2.stat().st_size == 0:
        raise RuntimeError("protoc did not produce a non-empty et_def_pb2.py")
    return source_hashes, proto_stage, version


def _write_queue(path: Path, scenario: str, overlap_arrival_ns: int) -> int:
    rows = [dict(row) for row in QUEUE_SCENARIOS[scenario]]
    if scenario == "workspace-overlap":
        if overlap_arrival_ns < 0:
            raise ValueError("workspace-overlap arrival time must be nonnegative")
        rows[1]["session_arrival_time_ns"] = overlap_arrival_ns
        rows[1]["description"] = (
            "second real FACE session arriving during the first long prefill "
            f"at {overlap_arrival_ns} ns"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as sink:
        writer = csv.DictWriter(sink, fieldnames=REQUEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _write_queue_provenance(
    queue: Path,
    archive: Path,
    source_materializer: Path,
    *,
    scenario: str,
    expected_request_count: int,
) -> None:
    spec = importlib.util.spec_from_file_location(
        "face_tiny_queue_provenance", source_materializer
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load FACE queue provenance helper: {source_materializer}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    record = module.queue_provenance(
        str(queue), f"FACE workspace-port {scenario} queue v1"
    )
    if (
        record.get("data_rows") != expected_request_count
        or record.get("sessions") != expected_request_count
    ):
        raise RuntimeError(
            "queue provenance request/session counts do not match the selected scenario"
        )
    sidecar = Path(str(queue) + ".provenance.json")
    encoded = json.dumps(record, indent=2, sort_keys=True) + "\n"
    sidecar.write_text(encoded, encoding="utf-8")
    archive.write_text(encoded, encoding="utf-8")


def _read_config(config_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with config_path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ValueError(f"empty FACE trace config: {config_path}")
        return list(reader.fieldnames), list(reader)


def _write_trace_config(
    source_config: Path,
    destination: Path,
    *,
    queue: Path,
    model_name: str | None,
    hardware_path: str | None = None,
    capacity_profile: str | None = None,
    system_template: str | None = None,
) -> None:
    fields, rows = _read_config(source_config)
    updates = {
        "layers": str(MODEL["layers"]),
        "hidden_size": str(MODEL["hidden_size"]),
        "ffn_size": str(MODEL["ffn_size"]),
        "num_heads": str(MODEL["num_heads"]),
        "vocab_size": str(MODEL["vocab_size"]),
        "bytes_per_elem": str(MODEL["bytes_per_elem"]),
        "request_queue_csv": str(queue.resolve()),
        "request_queue_session_limit": "0",
        "trace_granularity": "request_aggregated",
        "prefill_chunk_size": str(MODEL["prefill_tokens"]),
        "kv_reserve_context_tokens": "64",
        "remote_operand_loads": "false",
        "position_encoding": "rope",
    }
    if model_name is not None:
        updates["model_name"] = model_name
    if hardware_path is not None:
        updates["hardware_config"] = hardware_path
    if capacity_profile is not None:
        updates["local_hbm_capacity_profile"] = capacity_profile
    if system_template is not None:
        updates["system_template"] = system_template

    found: set[str] = set()
    for row in rows:
        if row.get("kind", "").strip().lower() == "config":
            key = row.get("key", "").strip()
            if key in updates:
                row["value"] = updates[key]
                found.add(key)
    missing = sorted(set(updates) - found)
    if missing:
        raise RuntimeError(
            f"source trace_config lacks required keys for tiny input: {missing}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as sink:
        writer = csv.DictWriter(sink, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _make_coarse_hw_inputs(
    source_variant: Path,
    sh_test: Path,
    *,
    resolver: Any,
) -> tuple[Path, str, int, str]:
    example_path = source_variant / "extern/die_model/coarse/configs/example_hardware.json"
    base_hw_path = sh_test / "hardware/face_case5_config_c.json"
    if not example_path.is_file() or not base_hw_path.is_file():
        raise FileNotFoundError("FACE source is missing coarse or Case-5 hardware input")
    descriptor = json.loads(example_path.read_text(encoding="utf-8"))
    # The coarse ABI is the only source of derived physical capacity here.
    summary = resolver.derive_coarse_die_hardware(descriptor)
    derived = summary.get("derived", {}).get("hardware_summary", {})
    raw_capacity = derived.get("hbm_capacity_bytes")
    if isinstance(raw_capacity, bool) or not isinstance(raw_capacity, (int, float)):
        raise RuntimeError(f"coarse C ABI returned invalid HBM capacity: {raw_capacity!r}")
    capacity = int(raw_capacity)
    if capacity <= 0 or float(capacity) != float(raw_capacity):
        raise RuntimeError(f"coarse C ABI capacity is not a positive integer: {raw_capacity!r}")

    hardware = json.loads(base_hw_path.read_text(encoding="utf-8"))
    hardware.pop("die", None)
    hardware["coarse-die"] = descriptor
    hardware["slug"] = "face_case5_coarse_die_v2"
    hardware["label"] = "C / FACE Case 5 with coarse_die_v2 hardware"
    profile = f"coarse-die-{capacity}-bytes"
    hardware["local-hbm"]["capacity-profiles"] = {
        profile: {
            "bytes": capacity,
            "label": f"coarse_die_v2 C ABI derived capacity ({capacity} bytes)",
            "note": "Generated reproduction profile; exact value came from libcoarse_die_cabi.so.",
        }
    }
    hardware["notes"] = list(hardware.get("notes", [])) + [
        "Generated tiny Online input: coarse-die descriptor and local HBM capacity are resolved together through FACE config_resolver and the coarse C ABI."
    ]
    hardware_path = sh_test / "hardware/face_tiny_coarse_die_v2.json"
    hardware_path.write_text(json.dumps(hardware, indent=2) + "\n", encoding="utf-8")

    base_template = sh_test / "system/llama2_7b_die_template.json"
    template = json.loads(base_template.read_text(encoding="utf-8"))
    template["die-model-profile"] = "coarse_die_v2"
    template_path = sh_test / "system/face_tiny_coarse_die_v2_template.json"
    template_path.write_text(json.dumps(template, indent=2) + "\n", encoding="utf-8")
    return hardware_path, profile, capacity, template_path.relative_to(sh_test).as_posix()


def _run_materializer(
    stage: Path,
    config_path: Path,
    *,
    die_cabi: Path,
    coarse_cabi: Path,
    expected_request_count: int,
) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    sh_test = stage / "sh_test_mesh"
    workload = sh_test / "workload/llama2_7b_inference"
    sys.path[:0] = [str(workload), str(sh_test), str(stage)]

    # These imports must resolve from the byte-checked stage, never the source
    # checkout. The FACE loader/materializer stay on their production path.
    import generate_face_trace as gft  # type: ignore[import-not-found]
    import plan_materializer as materializer  # type: ignore[import-not-found]
    import config_resolver as resolver  # type: ignore[import-not-found]

    if Path(gft.__file__).resolve().parents[3] != stage.resolve():
        raise RuntimeError(f"FACE config loader imported outside stage: {gft.__file__}")
    if Path(materializer.__file__).resolve().parents[3] != stage.resolve():
        raise RuntimeError(f"FACE plan materializer imported outside stage: {materializer.__file__}")
    gft.SH_TEST_DIR = sh_test
    materializer.SH_TEST_DIR = sh_test
    materializer.GENERATED_ROOT = sh_test / "generated"
    materializer.load_face_trace_config = lambda: gft.load_face_trace_config(config_path)

    os.environ["DIE_MODEL_CABI_PATH"] = str(die_cabi)
    os.environ["DIE_COARSE_CABI_PATH"] = str(coarse_cabi)
    config = gft.load_face_trace_config(config_path)
    if len(config.request_queue) != expected_request_count:
        raise RuntimeError(
            f"FACE loader selected {len(config.request_queue)} requests, "
            f"expected {expected_request_count}"
        )
    if config.npus_count != MODEL["configured_ranks"]:
        raise RuntimeError(f"FACE hardware resolved {config.npus_count} ranks, expected 54")
    if config.layers != MODEL["layers"] or config.hidden_size != MODEL["hidden_size"]:
        raise RuntimeError("resolved FACE model dimensions do not match tiny request model")
    if config.ffn_size != MODEL["ffn_size"] or config.num_heads != MODEL["num_heads"]:
        raise RuntimeError("resolved FACE FFN/head dimensions do not match tiny request model")
    all_ranks = [rank for group in config.inference_groups for rank in group.ranks]
    if sorted(all_ranks) != list(range(config.npus_count)) or len(all_ranks) != len(set(all_ranks)):
        raise RuntimeError("FACE inference groups must cover every physical rank exactly once")
    if any(len(group.ranks) != MODEL["tp_ranks_per_instance"] for group in config.inference_groups):
        raise RuntimeError("tiny FACE topology requires TP=6 in every inference group")

    materializer_stdout = io.StringIO()
    with contextlib.redirect_stdout(materializer_stdout):
        result = materializer.main()
    if result not in (None, 0):
        raise RuntimeError(f"FACE plan_materializer returned {result}")
    printed = materializer_stdout.getvalue().strip().splitlines()
    if not printed:
        raise RuntimeError("FACE plan_materializer emitted no provenance record")
    provenance = json.loads(printed[-1])
    plan_dir = Path(provenance["plan_dir"]).resolve()
    manifest_path = plan_dir / "manifest.json"
    metrics_path = plan_dir / "metrics_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if (
        len(manifest.get("requests", [])) != expected_request_count
        or manifest.get("selected_request_count") != expected_request_count
    ):
        raise RuntimeError("materialized FACE manifest does not match the queue request count")
    if len(metrics.get("requests", [])) != expected_request_count:
        raise RuntimeError("FACE metrics manifest does not match the queue request count")
    queue_ids = [request.request_id for request in config.request_queue]
    manifest_ids = [request["request_id"] for request in manifest["requests"]]
    if manifest_ids != queue_ids:
        raise RuntimeError("FACE queue and manifest request identities differ")
    system = json.loads(config.system_config.read_text(encoding="utf-8"))
    if manifest.get("compute-cost-model") != "die" or system.get("compute-cost-model") != "die":
        raise RuntimeError("resolved FACE input does not declare compute-cost-model=die")
    if manifest.get("die-model-config") != system.get("die-model-config"):
        raise RuntimeError("FACE manifest and resolved system config name different Die inputs")
    die_config = Path(system["die-model-config"])
    if not die_config.is_absolute() or not die_config.is_file():
        raise RuntimeError("FACE resolved die-model-config is not an existing absolute file")

    profile = system.get("die-model-profile")
    if profile == "coarse_die_v2":
        if "coarse-die" not in json.loads(config.hardware_config.read_text(encoding="utf-8")):
            raise RuntimeError("coarse profile lacks a coarse-die source descriptor")
        descriptor = json.loads(config.hardware_config.read_text(encoding="utf-8"))["coarse-die"]
        typed = resolver.derive_coarse_die_hardware(descriptor)
        typed_capacity = int(typed["derived"]["hardware_summary"]["hbm_capacity_bytes"])
        if config.hardware.local_hbm_capacity_bytes != typed_capacity:
            raise RuntimeError(
                "FACE coarse HBM capacity is not aligned with typed coarse hardware: "
                f"{config.hardware.local_hbm_capacity_bytes} != {typed_capacity}"
            )
        if typed_capacity <= 0:
            raise RuntimeError("FACE coarse resolver returned a non-positive capacity")
    else:
        if profile != "mesh_die_v1":
            raise RuntimeError(f"legacy FACE input has unexpected profile {profile!r}")
        hardware = json.loads(config.hardware_config.read_text(encoding="utf-8"))
        if "die" not in hardware or "coarse-die" in hardware:
            raise RuntimeError("legacy FACE input must retain its native die descriptor")
    runtime_config_dir = config.system_config.parent.resolve()
    required_runtime = ("system.json", "comm_group.json", "network.yml")
    missing_runtime = [name for name in required_runtime if not (runtime_config_dir / name).is_file()]
    if missing_runtime:
        raise RuntimeError(f"FACE runtime config is incomplete: {missing_runtime}")

    (stage / "materializer-last.log").write_text(
        materializer_stdout.getvalue(), encoding="utf-8"
    )
    return plan_dir, runtime_config_dir, manifest, {
        "profile": profile,
        "system_config": str(config.system_config.resolve()),
        "hardware_config": str(config.hardware_config.resolve()),
        "hardware_capacity_bytes": config.hardware.local_hbm_capacity_bytes,
        "inference_group_count": len(config.inference_groups),
        "configured_rank_count": config.npus_count,
        "request_count": len(config.request_queue),
        "request_ids": queue_ids,
        "model": {
            "layers": config.layers,
            "hidden_size": config.hidden_size,
            "ffn_size": config.ffn_size,
            "num_heads": config.num_heads,
            "vocab_size": config.vocab_size,
            "bytes_per_elem": config.bytes_per_elem,
        },
        "die_model_config": str(die_config),
    }


def prepare(
    source_name: str,
    die_cabi: Path,
    coarse_cabi: Path,
    profile_selection: str,
    scenario: str,
    overlap_arrival_ns: int,
) -> Path:
    repo_name = SOURCE_VARIANTS[source_name]
    source = (ROOT / repo_name).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"missing FACE source variant: {source}")
    for label, path in (("die C ABI", die_cabi), ("coarse C ABI", coarse_cabi)):
        if not path.is_file():
            raise FileNotFoundError(f"missing {label}: {path}")

    work = HERE
    stage = work / "stage" / source_name
    input_dir = work / "inputs" / source_name
    input_dir.mkdir(parents=True, exist_ok=True)
    source_hashes, proto_stage, protoc_version = _copy_stage(source, stage)
    sh_test = stage / "sh_test_mesh"
    workload = sh_test / "workload/llama2_7b_inference"
    queue = workload / "face_tiny_requests.csv"
    expected_request_count = _write_queue(queue, scenario, overlap_arrival_ns)
    _write_queue_provenance(
        queue,
        input_dir / "queue-provenance.json",
        workload / "traces/derive_20_first_30_seconds.py",
        scenario=scenario,
        expected_request_count=expected_request_count,
    )

    # Import staged FACE modules so descriptor derivation and all subsequent
    # materialization use the exact source copy that the Online service will
    # load.
    sys.path[:0] = [
        str(workload),
        str(sh_test),
        str(stage),
    ]
    import generate_face_trace as gft  # type: ignore[import-not-found]
    import config_resolver as resolver  # type: ignore[import-not-found]

    if Path(gft.__file__).resolve().parents[3] != stage.resolve():
        raise RuntimeError(f"FACE config loader did not come from the staged source: {gft.__file__}")
    gft.SH_TEST_DIR = sh_test
    os.environ["DIE_MODEL_CABI_PATH"] = str(die_cabi)
    os.environ["DIE_COARSE_CABI_PATH"] = str(coarse_cabi)
    coarse_hw, coarse_capacity_profile, coarse_capacity, coarse_template = _make_coarse_hw_inputs(
        source, sh_test, resolver=resolver
    )

    trace_config_source = workload / "trace_config.csv"
    config_specs = {
        "legacy": {
            "hardware_path": None,
            "capacity_profile": None,
            "system_template": None,
        },
        "coarse": {
            "hardware_path": coarse_hw.relative_to(sh_test).as_posix(),
            "capacity_profile": coarse_capacity_profile,
            "system_template": coarse_template,
        },
    }
    profiles: dict[str, dict[str, Any]] = {}
    selected_profiles = (
        tuple(config_specs)
        if profile_selection == "all"
        else (profile_selection,)
    )
    for profile_name in selected_profiles:
        spec = config_specs[profile_name]
        config_path = workload / f"face_tiny_{profile_name}_trace_config.csv"
        _write_trace_config(
            trace_config_source,
            config_path,
            queue=queue,
            model_name=(
                f"face_tiny_{source_name}_{profile_name}"
                if source_name == "face"
                else None
            ),
            hardware_path=spec["hardware_path"],
            capacity_profile=spec["capacity_profile"],
            system_template=spec["system_template"],
        )
        plan_dir, runtime_dir, manifest, resolved = _run_materializer(
            stage,
            config_path,
            die_cabi=die_cabi,
            coarse_cabi=coarse_cabi,
            expected_request_count=expected_request_count,
        )
        if profile_name == "coarse" and resolved["profile"] != "coarse_die_v2":
            raise RuntimeError("coarse trace did not resolve to coarse_die_v2")
        if profile_name == "legacy" and resolved["profile"] == "coarse_die_v2":
            raise RuntimeError("legacy trace unexpectedly resolved to coarse_die_v2")
        profiles[profile_name] = {
            "trace_config": str(config_path.resolve()),
            "request_queue": str(queue.resolve()),
            "plan_dir": str(plan_dir),
            "runtime_config_dir": str(runtime_dir),
            "manifest": str((plan_dir / "manifest.json").resolve()),
            "metrics_manifest": str((plan_dir / "metrics_manifest.json").resolve()),
            "manifest_backend": manifest.get("compute-cost-model"),
            "manifest_profile": resolved["profile"],
            "manifest_request_count": len(manifest.get("requests", [])),
            "resolved": resolved,
        }

    staged_hashes = _source_file_hashes(sh_test)
    if any(
        name not in staged_hashes or staged_hashes[name] != digest
        for name, digest in source_hashes.items()
    ):
        changed = sorted(
            name for name in set(source_hashes) & set(staged_hashes)
            if source_hashes[name] != staged_hashes[name]
        )
        missing = sorted(set(source_hashes) - set(staged_hashes))
        raise RuntimeError(
            "source copy changed during preparation: "
            f"missing={missing[:5]} changed={changed[:5]}"
        )
    proto_source = source / "extern/graph_frontend/chakra/schema/protobuf/et_def.proto"
    if _sha256(proto_source) != _sha256(proto_stage):
        raise RuntimeError("staged protobuf source changed during preparation")

    stage_identity = {
        "source_variant": repo_name,
        "source_head": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_worktree_copy_sha256": source_hashes,
        "source_worktree_copy_file_count": len(source_hashes),
        "et_def_proto_sha256": _sha256(proto_source),
        "staged_et_def_proto_sha256": _sha256(proto_stage),
        "protoc": shutil.which("protoc"),
        "protoc_version": protoc_version,
        "generated_et_def_pb2_sha256": _sha256(
            proto_stage.parent / "et_def_pb2.py"
        ),
        "die_cabi": {"path": str(die_cabi), "sha256": _sha256(die_cabi)},
        "coarse_die_cabi": {"path": str(coarse_cabi), "sha256": _sha256(coarse_cabi)},
    }
    identity_path = input_dir / "source-stage-identity.json"
    identity_path.write_text(json.dumps(stage_identity, indent=2, sort_keys=True) + "\n")
    output = {
        "source_variant": repo_name,
        "source_head": stage_identity["source_head"],
        "scenario": scenario,
        "workspace_overlap_arrival_ns": (
            overlap_arrival_ns if scenario == "workspace-overlap" else None
        ),
        "stage_dir": str(stage.resolve()),
        "source_stage_identity": str(identity_path.resolve()),
        "request_count": expected_request_count,
        "request_queue": str(queue.resolve()),
        "queue_provenance": str((input_dir / "queue-provenance.json").resolve()),
        "model": MODEL,
        "topology": {
            "mesh_rows": 9,
            "mesh_columns": 6,
            "instance_count": MODEL["configured_instances"],
            "tp_ranks_per_instance": MODEL["tp_ranks_per_instance"],
            "ranks": list(range(MODEL["configured_ranks"])),
        },
        "coarse_capacity_bytes_from_cabi": coarse_capacity,
        "coarse_capacity_profile": coarse_capacity_profile,
        "coarse_hardware_config": str(coarse_hw.resolve()),
        "profiles": profiles,
    }
    summary_path = input_dir / "tiny-inputs.json"
    summary_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(summary_path.read_text(encoding="utf-8"), end="")
    return summary_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-variant", choices=tuple(SOURCE_VARIANTS), required=True)
    parser.add_argument("--profile", choices=("all", "legacy", "coarse"), default="all")
    parser.add_argument(
        "--scenario", choices=tuple(QUEUE_SCENARIOS), default="tiny"
    )
    parser.add_argument("--overlap-arrival-ns", type=int, default=1_000)
    parser.add_argument("--die-cabi", type=Path, required=True)
    parser.add_argument("--coarse-die-cabi", type=Path, required=True)
    args = parser.parse_args()
    prepare(
        args.source_variant,
        args.die_cabi.resolve(),
        args.coarse_die_cabi.resolve(),
        args.profile,
        args.scenario,
        args.overlap_arrival_ns,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
