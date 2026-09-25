"""Resolve canonical hardware descriptions into Astra-Sim runtime files."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Sequence


_MANAGED_SYSTEM_FIELDS = frozenset(
    {
        "peak-perf",
        "peak-perf-note",
        "local-mem-bw",
        "local-mem-latency",
        "local-mem-capacity-bytes",
        "local-mem-capacity-note",
        "remote-mem-bw",
        "remote-mem-latency",
    }
)
_SUPPORTED_TOPOLOGIES = frozenset(
    {"Line", "Mesh", "Ring", "Switch", "FullyConnected"}
)


@dataclass(frozen=True)
class ResolvedHardware:
    source_path: Path
    capacity_profile: str
    slug: str
    label: str
    paper_case: str
    mesh_rows: int
    mesh_cols: int
    topology_by_network_dimension: tuple[str, ...]
    local_hbm_capacity_bytes: int
    local_hbm_bandwidth_gbps: float
    local_hbm_latency_ns: int
    d2d_bandwidth_gbps: float
    d2d_latency_ns: int
    remote_memory_type: str
    remote_memory_bandwidth_gbps: float
    remote_memory_latency_ns: int | None
    remote_memory_npu_selection: str | None
    peak_perf_tflops: float
    metadata: dict[str, object]

    @property
    def npus_count(self) -> int:
        return self.mesh_rows * self.mesh_cols

    @property
    def remote_memory_runtime_label(self) -> str:
        if self.remote_memory_type == "NO_MEMORY_EXPANSION":
            return "no_memory_expansion"
        if self.remote_memory_npu_selection == "mesh-boundary":
            return "edge_remote_memory_pool"
        return "remote_memory_expansion"


@dataclass(frozen=True)
class RuntimeConfigPaths:
    system: Path
    network: Path
    remote_memory: Path
    comm_group: Path


def _read_json(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    try:
        with path.open(encoding="utf-8") as source:
            value = json.load(source)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {description} {path}: {error.msg}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a JSON object: {path}")
    return value


def _require_object(container: dict[str, Any], key: str, context: str) -> dict[str, Any]:
    value = container.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{context}.{key} must be an object")
    return value


def _require_string(container: dict[str, Any], key: str, context: str) -> str:
    value = container.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}.{key} must be a non-empty string")
    return value


def _require_int(value: Any, context: str, *, positive: bool = False, nonnegative: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} must be an integer")
    if positive and value <= 0:
        raise ValueError(f"{context} must be positive")
    if nonnegative and value < 0:
        raise ValueError(f"{context} must be non-negative")
    return value


def _require_number(value: Any, context: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{context} must be finite")
    if positive and numeric <= 0:
        raise ValueError(f"{context} must be positive")
    if nonnegative and numeric < 0:
        raise ValueError(f"{context} must be non-negative")
    return numeric


def _require_exact_keys(container: dict[str, Any], expected: set[str], context: str) -> None:
    actual = set(container)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if extra:
            details.append(f"unexpected {', '.join(extra)}")
        raise ValueError(f"{context} has invalid keys ({'; '.join(details)})")


def load_hardware_config(path: Path, capacity_profile: str) -> ResolvedHardware:
    """Load and validate a canonical hardware source for one capacity profile."""
    source_path = Path(path)
    data = _read_json(source_path, "Hardware configuration")
    if "npus-count" in data:
        raise ValueError(
            "Hardware configuration must not contain legacy npus-count; "
            "derive it from mesh"
        )
    _require_exact_keys(
        data,
        {
            "schema-version",
            "slug",
            "label",
            "paper-case",
            "mesh",
            "local-hbm",
            "d2d",
            "remote-memory",
            "compute",
            "notes",
        },
        "Hardware configuration",
    )
    if _require_int(data["schema-version"], "Hardware configuration.schema-version") != 1:
        raise ValueError("Hardware configuration.schema-version must be 1")
    slug = _require_string(data, "slug", "Hardware configuration")
    label = _require_string(data, "label", "Hardware configuration")
    paper_case = _require_string(data, "paper-case", "Hardware configuration")

    mesh = _require_object(data, "mesh", "Hardware configuration")
    _require_exact_keys(mesh, {"rows", "columns", "topology-by-network-dimension"}, "Hardware configuration.mesh")
    rows = _require_int(mesh["rows"], "Hardware configuration.mesh.rows", positive=True)
    columns = _require_int(mesh["columns"], "Hardware configuration.mesh.columns", positive=True)
    topology = mesh["topology-by-network-dimension"]
    if not isinstance(topology, list) or len(topology) != 2:
        raise ValueError("Hardware configuration.mesh.topology-by-network-dimension must contain two names")
    if any(not isinstance(name, str) or not name.strip() or name not in _SUPPORTED_TOPOLOGIES for name in topology):
        raise ValueError(
            "Hardware configuration.mesh.topology-by-network-dimension contains an unsupported or empty topology"
        )

    local_hbm = _require_object(data, "local-hbm", "Hardware configuration")
    _require_exact_keys(
        local_hbm,
        {"bandwidth-gbps", "latency-ns", "capacity-profiles"},
        "Hardware configuration.local-hbm",
    )
    local_bandwidth = _require_number(
        local_hbm["bandwidth-gbps"], "Hardware configuration.local-hbm.bandwidth-gbps", positive=True
    )
    local_latency = _require_int(
        local_hbm["latency-ns"], "Hardware configuration.local-hbm.latency-ns", nonnegative=True
    )
    profiles = local_hbm["capacity-profiles"]
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("Hardware configuration.local-hbm.capacity-profiles must be a non-empty object")
    if capacity_profile not in profiles:
        raise ValueError(f"Unknown local HBM capacity profile {capacity_profile!r}")
    profile = profiles[capacity_profile]
    if not isinstance(profile, dict):
        raise ValueError(f"Capacity profile {capacity_profile!r} must be an object")
    _require_exact_keys(profile, {"bytes", "label", "note"}, f"Capacity profile {capacity_profile!r}")
    capacity = _require_int(profile["bytes"], f"Capacity profile {capacity_profile!r}.bytes", positive=True)
    profile_label = _require_string(profile, "label", f"Capacity profile {capacity_profile!r}")
    profile_note = _require_string(profile, "note", f"Capacity profile {capacity_profile!r}")

    d2d = _require_object(data, "d2d", "Hardware configuration")
    _require_exact_keys(d2d, {"bandwidth-gbps", "latency-ns"}, "Hardware configuration.d2d")
    d2d_bandwidth = _require_number(d2d["bandwidth-gbps"], "Hardware configuration.d2d.bandwidth-gbps", positive=True)
    d2d_latency = _require_int(d2d["latency-ns"], "Hardware configuration.d2d.latency-ns", nonnegative=True)

    remote_memory = _require_object(
        data,
        "remote-memory",
        "Hardware configuration",
    )
    remote_memory_type = _require_string(
        remote_memory,
        "memory-type",
        "Hardware configuration.remote-memory",
    )
    if remote_memory_type == "NO_MEMORY_EXPANSION":
        _require_exact_keys(
            remote_memory,
            {"memory-type", "bandwidth-gbps"},
            "Hardware configuration.remote-memory",
        )
        remote_memory_latency = None
        remote_memory_npu_selection = None
    elif remote_memory_type == "PER_NPU_MEMORY_EXPANSION":
        _require_exact_keys(
            remote_memory,
            {
                "memory-type",
                "bandwidth-gbps",
                "latency-ns",
                "npu-selection",
                "logical-pool",
            },
            "Hardware configuration.remote-memory",
        )
        remote_memory_latency = _require_int(
            remote_memory["latency-ns"],
            "Hardware configuration.remote-memory.latency-ns",
            nonnegative=True,
        )
        remote_memory_npu_selection = _require_string(
            remote_memory,
            "npu-selection",
            "Hardware configuration.remote-memory",
        )
        if remote_memory_npu_selection != "mesh-boundary":
            raise ValueError(
                "Hardware configuration.remote-memory.npu-selection must be "
                "'mesh-boundary'"
            )
        # P11 死键清除（2026-09-23 深挖审计；2026-09-25 字段退役）：源侧
        # "logical-pool" 键的值不再解析——它曾写入 remote_memory.json，
        # 但 C++ 远端内存后端零读取（纯审计键——开关清单 §10.7
        # remote_memory.json 键集表口径），写入链连同其专属 fail-closed
        # 守卫一并退役。键本身仍由上方 _require_exact_keys 键集校验
        # （canonical 源缺键/多键照常 fail-closed），ResolvedHardware 不
        # 再携带该死字段。
    else:
        raise ValueError(
            "Hardware configuration.remote-memory.memory-type is unsupported"
        )
    remote_memory_bandwidth = _require_number(
        remote_memory["bandwidth-gbps"],
        "Hardware configuration.remote-memory.bandwidth-gbps",
        positive=True,
    )

    compute = _require_object(data, "compute", "Hardware configuration")
    _require_exact_keys(compute, {"peak-perf-tflops"}, "Hardware configuration.compute")
    peak_perf = _require_number(compute["peak-perf-tflops"], "Hardware configuration.compute.peak-perf-tflops", positive=True)
    notes = data["notes"]
    if not isinstance(notes, list) or not all(isinstance(note, str) and note.strip() for note in notes):
        raise ValueError("Hardware configuration.notes must be a list of non-empty strings")

    metadata: dict[str, object] = deepcopy(data)
    metadata["selected-capacity-profile"] = capacity_profile
    metadata["selected-capacity-note"] = profile_note
    return ResolvedHardware(
        source_path=source_path,
        capacity_profile=capacity_profile,
        slug=slug,
        label=f"{label} / {profile_label}",
        paper_case=paper_case,
        mesh_rows=rows,
        mesh_cols=columns,
        topology_by_network_dimension=tuple(topology),
        local_hbm_capacity_bytes=capacity,
        local_hbm_bandwidth_gbps=local_bandwidth,
        local_hbm_latency_ns=local_latency,
        d2d_bandwidth_gbps=d2d_bandwidth,
        d2d_latency_ns=d2d_latency,
        remote_memory_type=remote_memory_type,
        remote_memory_bandwidth_gbps=remote_memory_bandwidth,
        remote_memory_latency_ns=remote_memory_latency,
        remote_memory_npu_selection=remote_memory_npu_selection,
        peak_perf_tflops=peak_perf,
        metadata=metadata,
    )


def _write_text_atomically(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False) as temporary:
        temporary.write(contents)
        temporary_name = temporary.name
    try:
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _serialize_json(value: object) -> str:
    return json.dumps(value, indent=2) + "\n"


def _prepare_remote_memory(hardware: ResolvedHardware) -> dict[str, Any]:
    remote: dict[str, Any] = {
        "memory-type": hardware.remote_memory_type,
        "remote-mem-bw": hardware.remote_memory_bandwidth_gbps,
    }
    if hardware.remote_memory_type == "NO_MEMORY_EXPANSION":
        return remote

    if hardware.remote_memory_latency_ns is None:
        raise ValueError("Remote-memory expansion requires a configured latency")
    if hardware.remote_memory_npu_selection != "mesh-boundary":
        raise ValueError("Remote-memory expansion requires mesh-boundary selection")
    # P11 死键清除（2026-09-23 深挖审计）：logical-pool 曾写入
    # remote_memory.json，但 C++ 远端内存后端零读取（纯审计键——开关清单
    # §10.7 remote_memory.json 键集表口径）——写入链已退役，此处不落该
    # 键；源侧键值解析与字段亦已随 2026-09-25 清理退役（键集校验仍由
    # load_hardware_config 的 _require_exact_keys 承担）。

    boundary_ranks = []
    for row in range(hardware.mesh_rows):
        for column in range(hardware.mesh_cols):
            if row in {0, hardware.mesh_rows - 1} or column in {
                0,
                hardware.mesh_cols - 1,
            }:
                boundary_ranks.append(row * hardware.mesh_cols + column)
    remote["remote-mem-latency"] = hardware.remote_memory_latency_ns
    remote["npu-ids"] = boundary_ranks
    return remote


def _prepare_comm_groups(
    inference_groups: Sequence[tuple[str, Sequence[int]]], hardware: ResolvedHardware
) -> dict[str, dict[str, object]]:
    groups: dict[str, dict[str, object]] = {}
    seen_ids: set[int] = set()
    for pg_name, ranks_source in inference_groups:
        if isinstance(pg_name, bool) or not isinstance(pg_name, (str, int)):
            raise ValueError(f"Communicator group name {pg_name!r} must parse as a positive integer")
        try:
            pg_id = int(pg_name)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Communicator group name {pg_name!r} must parse as a positive integer") from error
        if pg_id <= 0:
            raise ValueError(f"Communicator group name {pg_name!r} must parse as a positive integer")
        if pg_id in seen_ids:
            raise ValueError(f"Duplicate communicator group ID {pg_id}")
        seen_ids.add(pg_id)
        if isinstance(ranks_source, (str, bytes)) or not isinstance(ranks_source, Sequence):
            raise ValueError(f"Communicator group {pg_id} ranks must be a sequence")
        ranks = list(ranks_source)
        if not ranks:
            raise ValueError(f"Communicator group {pg_id} ranks must not be empty")
        seen_ranks: set[int] = set()
        for rank in ranks:
            validated_rank = _require_int(rank, f"Communicator group {pg_id} rank")
            if not 0 <= validated_rank < hardware.npus_count:
                raise ValueError(f"Communicator group {pg_id} rank {validated_rank} is outside the physical mesh")
            if validated_rank in seen_ranks:
                raise ValueError(f"Communicator group {pg_id} has duplicate rank {validated_rank}")
            seen_ranks.add(validated_rank)
        rows = {rank // hardware.mesh_cols for rank in ranks}
        columns = {rank % hardware.mesh_cols for rank in ranks}
        expected = {
            row * hardware.mesh_cols + column
            for row in range(min(rows), max(rows) + 1)
            for column in range(min(columns), max(columns) + 1)
        }
        if seen_ranks != expected:
            raise ValueError(f"Communicator group {pg_id} ranks must form a contiguous row-major rectangle")
        groups[str(pg_id)] = {
            "ranks": ranks,
            "dimensions": [len(columns), len(rows)],
        }
    return groups


def _network_yaml(hardware: ResolvedHardware) -> str:
    topology = ", ".join(hardware.topology_by_network_dimension)
    dimensions = len(hardware.topology_by_network_dimension)
    bandwidth = ", ".join(str(hardware.d2d_bandwidth_gbps) for _ in range(dimensions))
    latency = ", ".join(str(hardware.d2d_latency_ns) for _ in range(dimensions))
    return (
        f"# Generated from {hardware.source_path} (capacity profile: {hardware.capacity_profile})\n"
        f"topology: [ {topology} ]\n"
        f"npus_count: [ {hardware.mesh_cols}, {hardware.mesh_rows} ]\n"
        f"bandwidth: [ {bandwidth} ]\n"
        f"latency: [ {latency} ]\n"
    )


def materialize_runtime_configs(
    *,
    hardware: ResolvedHardware,
    system_template_path: Path,
    inference_groups: Sequence[tuple[str, Sequence[int]]],
    output_dir: Path,
) -> RuntimeConfigPaths:
    """Materialize deterministic Astra-Sim-native files from canonical sources."""
    system_template = _read_json(Path(system_template_path), "System template")
    managed_fields = sorted(_MANAGED_SYSTEM_FIELDS.intersection(system_template))
    if managed_fields:
        raise ValueError(f"System template contains managed hardware field(s): {', '.join(managed_fields)}")
    remote_memory = _prepare_remote_memory(hardware)
    comm_groups = _prepare_comm_groups(inference_groups, hardware)

    system = dict(system_template)
    system["local-mem-bw"] = hardware.local_hbm_bandwidth_gbps
    system["local-mem-latency"] = hardware.local_hbm_latency_ns
    # P11 死键清除（2026-09-23 深挖审计）：local-mem-capacity-bytes 曾在此
    # 写入 system.json，而 C++ 侧零读取（全仓 grep 无读者）——写入链退役；
    # _MANAGED_SYSTEM_FIELDS 仍保留该键（模板禁写守卫，防死键回流）。
    system["remote-mem-bw"] = hardware.remote_memory_bandwidth_gbps
    if "remote-mem-latency" in remote_memory:
        system["remote-mem-latency"] = remote_memory["remote-mem-latency"]
    system["peak-perf"] = hardware.peak_perf_tflops

    destination = Path(output_dir)
    paths = RuntimeConfigPaths(
        system=destination / "system.json",
        network=destination / "network.yml",
        remote_memory=destination / "remote_memory.json",
        comm_group=destination / "comm_group.json",
    )
    _write_text_atomically(paths.system, _serialize_json(system))
    _write_text_atomically(paths.network, _network_yaml(hardware))
    _write_text_atomically(paths.remote_memory, _serialize_json(remote_memory))
    _write_text_atomically(paths.comm_group, _serialize_json(comm_groups))
    return paths
