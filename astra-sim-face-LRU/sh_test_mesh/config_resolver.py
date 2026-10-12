"""Resolve canonical hardware descriptions into Astra-Sim runtime files."""

from __future__ import annotations

import ctypes
from copy import deepcopy
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Optional, Sequence


# 键分两类：materialize_runtime_configs 实际写入 system.json 的只有
# peak-perf / local-mem-bw / local-mem-latency / remote-mem-bw /
# remote-mem-latency 五键（C++ Sys 亦只读这五键）；三个 *-note /
# local-mem-capacity-bytes 键从不写入，仅作为"模板禁入"哨兵参与
# 交集 fail-closed 检查（防系统模板自带硬件派生值绕过 resolver），
# 与原仓同款防御，勿删。
# 阶段 C(2026-10-03,契约 §3.2/§7 裁定 1;face-LRU 随迁):"die-model-config"
# 加入 managed 集——system 模板出现即拒,由 resolver 注入(resolved_die_
# model.json 绝对路径)。"compute-cost-model"/"die-model-profile" 来自
# system 模板(die 模板自带),不在 managed 集。
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
        "die-model-config",
    }
)
# sync-py-E12 (2026-09-30, adapted from the joint repo's M-28 tightening
# after verifying THIS repo's C++/Python split): the topology whitelist =
# the planar-grid values on which the Python full stack (face_scheduler
# instance adjacency / session_kv_manager.physical_edge_ranks /
# generate_face_trace._xy_route -- all row-major mesh manhattan) and the
# C++ backend agree.  NetworkParser::parse_topology_name maps "Line" and
# "Mesh" to the same Mesh building block (row-major grid), matching the
# Python manhattan semantics; everything else silently forks -- Ring wraps
# around the row (Ring::route), FullyConnected is single-hop, Switch hops
# through a star switch device -- so over-advertising them here would let
# a config whose Python-side edge-rank/XY-route derivations are wrong
# reach the run.  Oversold names fail closed at this earliest boundary.
_SUPPORTED_TOPOLOGIES = frozenset({"Line", "Mesh"})

# ---------------------------------------------------------------------------
# die 成本库 C ABI(阶段 C;契约 §3.3/§7 裁定 1;face-LRU 随迁 2026-10-03)。
# resolver 通过 ctypes 调 libdie_model_cabi.so 取硬件派生值——禁止在 Python
# 复写第二套派生公式。发现序(env DIE_MODEL_CABI_PATH -> 仓内
# build/astra_analytical/build_congestion_aware/{lib,bin}/)失败即
# fail-closed,绝不静默回退 roofline。
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
_CABI_LIBRARY_NAME = "libdie_model_cabi.so"
_CABI_LIBRARY_ENV = "DIE_MODEL_CABI_PATH"
_DIE_NODE_KEYS = frozenset(
    {
        "sram-mb-per-core",
        "core-type",
        "compute-area-mm2",
        "hbm-per-side",
        "hbm-sides",
        "noc-width-bits",
    }
)
_LEGAL_NOC_WIDTH_BITS = frozenset({64, 128, 256, 512, 1024})
_CACHED_CABI: dict[str, Any] = {}


def cabi_library_candidates(repo_root: Optional[Path] = None) -> tuple[Path, ...]:
    """契约 §7 裁定 1 的仓内发现路径序(测试/诊断也用它列候选)。"""
    root = Path(repo_root) if repo_root is not None else _REPO_ROOT
    build = root / "build" / "astra_analytical" / "build_congestion_aware"
    return (build / "lib" / _CABI_LIBRARY_NAME, build / "bin" / _CABI_LIBRARY_NAME)


def discover_cabi_library() -> Path:
    """发现序:env DIE_MODEL_CABI_PATH -> 仓内 build 路径 -> fail-closed。"""
    candidates: list[Path] = []
    env_value = os.environ.get(_CABI_LIBRARY_ENV)
    if env_value:
        candidates.append(Path(env_value))
    candidates.extend(cabi_library_candidates())
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError(
        "die hardware node present but {} not found; build the "
        "die_model_cabi target first (contract ruling 1: candidates = "
        "env {} -> {})".format(
            _CABI_LIBRARY_NAME, _CABI_LIBRARY_ENV,
            " -> ".join(str(path) for path in cabi_library_candidates())))


def _load_cabi_library() -> ctypes.CDLL:
    """加载并缓存 libdie_model_cabi.so(进程内单例;签名按 DieCApi.h)。"""
    library_path = discover_cabi_library()
    cache_key = str(library_path)
    library = _CACHED_CABI.get(cache_key)
    if library is not None:
        return library
    library = ctypes.CDLL(str(library_path))
    library.die_create_model.argtypes = [ctypes.c_char_p]
    library.die_create_model.restype = ctypes.c_void_p
    library.die_hardware_summary.argtypes = [ctypes.c_void_p]
    library.die_hardware_summary.restype = ctypes.c_void_p
    library.die_free_result.argtypes = [
        ctypes.c_char_p]
    library.die_free_result.restype = None
    library.die_destroy_model.argtypes = [ctypes.c_void_p]
    library.die_destroy_model.restype = None
    library.die_last_error.argtypes = []
    library.die_last_error.restype = ctypes.c_char_p
    _CACHED_CABI[cache_key] = library
    return library


def _cabi_last_error(library: ctypes.CDLL) -> str:
    raw = library.die_last_error()
    if not raw:
        return ""
    return raw.decode("utf-8", errors="replace")


def derive_die_hardware(die_node: dict[str, Any]) -> dict[str, Any]:
    """调 die_create_model + die_hardware_summary 取派生硬件(契约 §3.3)。

    die 六参数的值域校验(合法 (面积,每侧,侧数) 布局表、位宽集合、
    (sram,核型) 命中核表)全部由库内 make_hardware 承担——本函数只做
    结构检查(六键齐全/基础类型),库报错原文透传(ValueError)。返回
    库的 §3.3 摘要 dict(schema_version/profile/die/derived)。"""
    library = _load_cabi_library()
    config_json = json.dumps(
        {"schema_version": 1, "die": die_node}, sort_keys=True)
    model = library.die_create_model(config_json.encode("utf-8"))
    if not model:
        raise ValueError(
            "die hardware node rejected by the die_model cost library: "
            "{}".format(_cabi_last_error(library)))
    try:
        summary_ptr = library.die_hardware_summary(model)
        if not summary_ptr:
            raise ValueError(
                "die_hardware_summary failed: {}".format(
                    _cabi_last_error(library)))
        summary_cchar = ctypes.cast(summary_ptr, ctypes.c_char_p)
        raw = summary_cchar.value
        library.die_free_result(summary_cchar)
        if raw is None:
            raise ValueError("die_hardware_summary returned an empty buffer")
        return json.loads(raw.decode("utf-8"))
    finally:
        library.die_destroy_model(model)


def _validate_die_node_structure(die: dict[str, Any]) -> None:
    """die 节六键的结构检查(键集/基础类型/枚举;值域由库校验)。"""
    context = "Hardware configuration.die"
    if not isinstance(die, dict):
        raise ValueError(f"{context} must be an object")
    actual = set(die)
    if actual != set(_DIE_NODE_KEYS):
        missing = sorted(_DIE_NODE_KEYS - actual)
        extra = sorted(actual - _DIE_NODE_KEYS)
        raise ValueError(
            f"{context} has invalid keys (missing {', '.join(missing)}; "
            f"unexpected {', '.join(extra)})")
    _require_number(die["sram-mb-per-core"], f"{context}.sram-mb-per-core", positive=True)
    core_type = die["core-type"]
    if not isinstance(core_type, str) or core_type not in {"small", "large"}:
        raise ValueError(f"{context}.core-type must be 'small' or 'large'")
    _require_number(die["compute-area-mm2"], f"{context}.compute-area-mm2", positive=True)
    _require_int(die["hbm-per-side"], f"{context}.hbm-per-side", positive=True)
    _require_int(die["hbm-sides"], f"{context}.hbm-sides", positive=True)
    noc_width = _require_int(die["noc-width-bits"], f"{context}.noc-width-bits", positive=True)
    if noc_width not in _LEGAL_NOC_WIDTH_BITS:
        raise ValueError(
            f"{context}.noc-width-bits must be one of "
            f"{sorted(_LEGAL_NOC_WIDTH_BITS)}, got {noc_width}")

# sync-py-E12 (2026-09-30): system-template key whitelist, derived from the
# keys THIS repo's C++ actually reads out of the system JSON:
# Sys::initialize_sys (Sys.cc: scheduling-policy .. local-mem-trace-
# filename) plus CollectiveImplLookup::setup_collective_impl_from_config
# (CollectiveImplLookup.cc: the *-implementation keys), which Sys calls on
# the SAME parsed system.json.  A typo'd template key used to pass this
# resolver freely, flow into the generated system.json, and be silently
# ignored by the C++ reader (fail-open: the feature the key was meant to
# configure stayed off with zero diagnostics).  The joint repo's list is
# NOT copied verbatim: it carries joint-specific keys and omits the
# -custom implementation keys this repo's CollectiveImplLookup reads.
_KNOWN_SYSTEM_TEMPLATE_KEYS = frozenset(
    {
        "scheduling-policy",
        "collective-optimization",
        "local-reduction-delay",
        "active-chunks-per-dimension",
        "L",
        "o",
        "g",
        "G",
        "endpoint-delay",
        "model-shared-bus",
        "preferred-dataset-splits",
        "peak-perf",
        "local-mem-bw",
        "local-mem-latency",
        "remote-mem-bw",
        "remote-mem-latency",
        "pipeline-tile-fraction",
        "hbm-kv-restore-bandwidth-sharing",
        "hbm-bandwidth-contention",
        "roofline-enabled",
        "trace-enabled",
        "replay-only",
        "track-local-mem",
        "local-mem-trace-filename",
        "all-reduce-implementation",
        "reduce-scatter-implementation",
        "all-gather-implementation",
        "all-to-all-implementation",
        "all-reduce-implementation-custom",
        "reduce-scatter-implementation-custom",
        "all-gather-implementation-custom",
        "all-to-all-implementation-custom",
        "per-node-custom-implementation",
        "compute-cost-model",
        "die-model-profile",
        "die-model-config",
    }
)


@dataclass(frozen=True)
class ResolvedHardware:
    source_path: Path
    capacity_profile: str
    slug: str
    label: str
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
    # 阶段 C(die 后端,契约 §3.1/§3.3;face-LRU 随迁 2026-10-03):die 节
    # 存在时非 None——die 六参数原文 + 库派生摘要(derived 全字段);
    # peak_perf_tflops / local_hbm_bandwidth_gbps 此时来自 derived
    # (PE_peak_die/1e12、HBM 总带宽/1e9),legacy 路径两值为空/沿用
    # compute 节。本仓的 remote-memory/pool 字段与 die 节正交,照常装载。
    die_node: Optional[dict[str, Any]] = None
    die_summary: Optional[dict[str, Any]] = None

    @property
    def die_derived(self) -> Optional[dict[str, Any]]:
        if self.die_summary is None:
            return None
        return self.die_summary.get("derived")

    @property
    def die_profile(self) -> str:
        if self.die_summary is None:
            return ""
        return str(self.die_summary.get("profile", ""))

    @property
    def npus_count(self) -> int:
        return self.mesh_rows * self.mesh_cols

    # P3(face-LRU 审计,2026-09-29):原 mesh_label property 删除
    # (全仓 grep 零引用;标签口径由 slug/label 承载)。

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
    # 阶段 C:die 后端时为 resolved_die_model.json 的路径(与 runtime
    # 四小件同目录),legacy 时为 None。
    die_model: Optional[Path] = None


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
    # 阶段 C(契约 §3.1):顶层键集 = 核心键 + 计算描述二选一("compute"
    # legacy 或 "die" 新后端;互斥判定在下方 compute 段,并存/全无均报错)。
    _required_top_level = {
        "schema-version",
        "slug",
        "label",
        "mesh",
        "local-hbm",
        "d2d",
        "remote-memory",
        "notes",
    }
    actual_top_level = set(data)
    missing_top_level = _required_top_level - actual_top_level
    extra_top_level = actual_top_level - _required_top_level - {"compute", "die"}
    if missing_top_level or extra_top_level:
        details = []
        if missing_top_level:
            details.append(f"missing {', '.join(sorted(missing_top_level))}")
        if extra_top_level:
            details.append(f"unexpected {', '.join(sorted(extra_top_level))}")
        raise ValueError(
            "Hardware configuration has invalid keys (" + "; ".join(details) + ")")
    if _require_int(data["schema-version"], "Hardware configuration.schema-version") != 1:
        raise ValueError("Hardware configuration.schema-version must be 1")
    slug = _require_string(data, "slug", "Hardware configuration")
    label = _require_string(data, "label", "Hardware configuration")

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
    else:
        raise ValueError(
            "Hardware configuration.remote-memory.memory-type is unsupported"
        )
    remote_memory_bandwidth = _require_number(
        remote_memory["bandwidth-gbps"],
        "Hardware configuration.remote-memory.bandwidth-gbps",
        positive=True,
    )

    # 阶段 C(契约 §3.1):die 节与 compute.peak-perf-tflops 互斥出现——
    # 并存即报错(新后端不得静默沿用旧算力);二者全无同样报错。本仓的
    # remote-memory/pool 节与计算描述正交,不受影响。
    die_node = data.get("die")
    compute_node = data.get("compute")
    if die_node is not None and compute_node is not None:
        raise ValueError(
            "Hardware configuration must not declare both 'die' and "
            "'compute.peak-perf-tflops' (mutually exclusive compute "
            "descriptions, contract section 3.1)")
    die_summary: Optional[dict[str, Any]] = None
    if die_node is not None:
        if not isinstance(die_node, dict):
            raise ValueError("Hardware configuration.die must be an object")
        _validate_die_node_structure(die_node)
        # 值域校验(合法布局组合/核表命中)与派生全部经 ctypes 同源
        # (契约 §3.3:禁止 Python 复写第二套派生公式)。
        die_summary = derive_die_hardware(die_node)
        derived = die_summary.get("derived") or {}
        try:
            derived_peak = float(derived["pe_peak_tflops"])
            derived_bandwidth = float(derived["hbm_bandwidth_bytes"]) / 1e9
            derived_capacity = int(float(derived["hbm_capacity_bytes"]))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "die cost library returned an incomplete derived block: "
                f"{derived!r}") from error
        # 契约 §3.4 容量对齐:die 节存在时所选 capacity-profile 的 bytes
        # 必须等于 HBM 派生容量,不等即报错(错误信息给出两边数值)。
        if capacity != derived_capacity:
            raise ValueError(
                f"Capacity profile {capacity_profile!r}.bytes ({capacity}) "
                "does not match the die-derived HBM capacity "
                f"({derived_capacity} = hbm-sides x hbm-per-side x 16 GiB); "
                "select a die-aligned capacity profile (contract "
                "section 3.4)")
        peak_perf = derived_peak
        local_bandwidth = derived_bandwidth
    elif compute_node is not None:
        if not isinstance(compute_node, dict):
            raise ValueError("Hardware configuration.compute must be an object")
        _require_exact_keys(compute_node, {"peak-perf-tflops"}, "Hardware configuration.compute")
        peak_perf = _require_number(compute_node["peak-perf-tflops"], "Hardware configuration.compute.peak-perf-tflops", positive=True)
    else:
        raise ValueError(
            "Hardware configuration must declare exactly one of 'die' or "
            "'compute.peak-perf-tflops' (contract section 3.1)")
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
        die_node=(dict(die_node) if die_node is not None else None),
        die_summary=die_summary,
    )


def _write_text_atomically(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False) as temporary:
        try:
            temporary.write(contents)
        except BaseException:
            Path(temporary.name).unlink(missing_ok=True)
            raise
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
        # The C++ GeneralComplexTopology consumes the ranks list as a
        # dim0-fastest row-major ring; validating only the rank SET would let
        # an out-of-order list silently build a transposed ring and misalign
        # the per-rank custom ET files.  A4-10 (2026-09-26): the former
        # separate set-equality check ("must form a contiguous row-major
        # rectangle") is fully subsumed by the position-by-position
        # comparison below -- any non-rectangle set also fails the ordered
        # comparison -- so only one fail-closed check remains.
        expected_order = [
            row * hardware.mesh_cols + column
            for row in range(min(rows), max(rows) + 1)
            for column in range(min(columns), max(columns) + 1)
        ]
        if ranks != expected_order:
            mismatched = ", ".join(
                f"position {index}: got {actual}, expected {wanted}"
                for index, (actual, wanted) in enumerate(zip(ranks, expected_order))
                if actual != wanted
            )
            raise ValueError(
                f"Communicator group {pg_id} ranks must be listed in "
                f"dim0-fastest row-major order (the C++ "
                f"GeneralComplexTopology consumes them in that order); "
                f"expected head {expected_order[:4]}, got {ranks[:4]} "
                f"({mismatched})"
            )
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
    # sync-py-E12 (2026-09-30): reject unknown template keys at this
    # earliest boundary -- a typo would otherwise flow into the generated
    # system.json and be silently ignored by the C++ readers (the feature
    # stays off with zero diagnostics).  Whitelist = the keys this repo's
    # C++ actually reads (Sys.cc Sys::initialize_sys + CollectiveImplLookup
    # setup_collective_impl_from_config); keep it in sync with those two
    # readers.
    unknown_template_keys = sorted(set(system_template) - _KNOWN_SYSTEM_TEMPLATE_KEYS)
    if unknown_template_keys:
        raise ValueError(
            "System template contains unknown key(s) (typo? see "
            "_KNOWN_SYSTEM_TEMPLATE_KEYS / the C++ system-json readers in "
            "Sys.cc and CollectiveImplLookup.cc): "
            + ", ".join(unknown_template_keys)
        )
    remote_memory = _prepare_remote_memory(hardware)
    comm_groups = _prepare_comm_groups(inference_groups, hardware)

    system = dict(system_template)
    system["local-mem-bw"] = hardware.local_hbm_bandwidth_gbps
    system["local-mem-latency"] = hardware.local_hbm_latency_ns
    system["remote-mem-bw"] = hardware.remote_memory_bandwidth_gbps
    if "remote-mem-latency" in remote_memory:
        system["remote-mem-latency"] = remote_memory["remote-mem-latency"]
    system["peak-perf"] = hardware.peak_perf_tflops

    destination = Path(output_dir)
    die_model_path: Optional[Path] = None
    if hardware.die_summary is not None:
        # 阶段 C(die 后端,契约 §3.2/§3.3):resolved_die_model.json 落
        # runtime_config 目录(库的 §3.3 摘要原文,可被 make_hardware
        # round-trip);system.json 注入 die-model-config(绝对路径,
        # managed 字段)并做冲突 fail-closed(die + roofline-enabled /
        # compute-cost-model != die)。peak-perf/local-mem-bw 已在上方由
        # ResolvedHardware 的派生值填充(§3.4 回流口径)。
        cost_model = system.get("compute-cost-model")
        if cost_model is not None and cost_model != "die":
            raise ValueError(
                "System template declares compute-cost-model "
                f"{cost_model!r} but the hardware resolves the die backend; "
                "use a die system template (contract section 3.2)")
        if system.get("roofline-enabled"):
            raise ValueError(
                "die backend conflicts with roofline-enabled=1 in the "
                "system template (contract section 3.2 conflict matrix)")
        die_model_path = destination / "resolved_die_model.json"
        _write_text_atomically(
            die_model_path, _serialize_json(hardware.die_summary))
        system["die-model-config"] = str(die_model_path.resolve())

    paths = RuntimeConfigPaths(
        system=destination / "system.json",
        network=destination / "network.yml",
        remote_memory=destination / "remote_memory.json",
        comm_group=destination / "comm_group.json",
        die_model=die_model_path,
    )
    _write_text_atomically(paths.system, _serialize_json(system))
    _write_text_atomically(paths.network, _network_yaml(hardware))
    _write_text_atomically(paths.remote_memory, _serialize_json(remote_memory))
    _write_text_atomically(paths.comm_group, _serialize_json(comm_groups))
    return paths
