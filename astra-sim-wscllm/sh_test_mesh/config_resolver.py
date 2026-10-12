"""Resolve canonical hardware descriptions into Astra-Sim runtime files."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import ctypes
import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Optional, Sequence


# die 后端(wscllm 推广,契约 §3.2/§7 裁定 1):"die-model-config" 加入
# managed 集——system 模板出现即拒,由 resolver 注入(resolved_die_
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

# 3.12 system-template whitelist (2026-09-29, same contract as the C++
# unknown-key gate in Sys::initialize_sys_fields): a typo'd template key
# used to pass this resolver freely, reach the generated system.json, and
# be silently ignored by the C++ reader (fail-open: the feature the key
# was meant to configure stayed off with zero diagnostics). Validate the
# template keys HERE, at the earliest boundary. Keep this set in sync with
# known_system_keys in Sys.cc (which stays as the last line of defense).
_KNOWN_SYSTEM_TEMPLATE_KEYS = frozenset(
    {
        "scheduling-policy",
        "inter-dimension-scheduling",
        "intra-dimension-scheduling",
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
        "roofline-enabled",
        "hbm-bandwidth-contention",
        "trace-enabled",
        "all-reduce-implementation",
        "reduce-scatter-implementation",
        "all-gather-implementation",
        "all-to-all-implementation",
        "compute-cost-model",
        "die-model-profile",
    }
)
# Retired custom-collective keys (2026-09-30 sync with Sys.cc
# known_system_keys / CollectiveImplLookup reject_retired_custom_key): the
# C++ side keeps them in its known set purely to give the specific
# retirement diagnosis; this earliest boundary mirrors that instead of
# lumping them into the generic unknown-key message.
_RETIRED_SYSTEM_TEMPLATE_KEYS = frozenset(
    {
        "all-reduce-implementation-custom",
        "all-gather-implementation-custom",
        "reduce-scatter-implementation-custom",
        "all-to-all-implementation-custom",
        "per-node-custom-implementation",
    }
)
_SUPPORTED_TOPOLOGIES = frozenset(
    {"Line", "Mesh", "Ring", "Switch", "FullyConnected"}
)

# ---------------------------------------------------------------------------
# die 成本库 C ABI(wscllm 推广;契约 §3.3/§7 裁定 1)。resolver 通过
# ctypes 调 libdie_model_cabi.so 取硬件派生值——禁止在 Python 复写第二套
# 派生公式。发现序(env DIE_MODEL_CABI_PATH -> 仓内 build/astra_analytical/
# build_congestion_aware/{lib,bin}/)失败即 fail-closed,绝不静默回退
# roofline。
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
    peak_perf_tflops: float
    metadata: dict[str, object]
    # die 后端(契约 §3.1/§3.3):die 节存在时非 None——die 六参数原文 +
    # 库派生摘要(derived 全字段);peak_perf_tflops /
    # local_hbm_bandwidth_gbps 此时来自 derived(PE_peak_die/1e12、HBM
    # 总带宽/1e9),legacy 路径两值为空/沿用 compute 节。
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

    @property
    def remote_memory_runtime_label(self) -> str:
        # face A.2 alignment (wscllm removal 2026-09-24): the remote memory
        # backend was removed; the parse layer fails closed unless the
        # hardware declares NO_MEMORY_EXPANSION, so this label is constant
        # and only keeps runtime dir naming stable.
        return "no_memory_expansion"


@dataclass(frozen=True)
class RuntimeConfigPaths:
    system: Path
    network: Path
    comm_group: Path
    # die 后端时为 resolved_die_model.json 的路径(与三件套同目录),
    # legacy 时为 None。
    die_model: Optional[Path] = None


def _read_json(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    try:
        with path.open(encoding="utf-8") as source:
            value = json.load(source)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {description} {path}: {error.msg}") from error
    # E2 (2026-09-28): a non-UTF-8 byte sequence used to escape as a bare
    # UnicodeDecodeError with no filename context; read() OSError likewise
    # escaped without context. Both now carry description + path.
    except UnicodeDecodeError as error:
        raise ValueError(
            f"{description} {path} is not valid UTF-8: {error.reason} at byte {error.start}"
        ) from error
    except OSError as error:
        raise OSError(f"Cannot read {description} {path}: {error}") from error
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


def _require_number(value: Any, context: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{context} must be finite")
    if positive and numeric <= 0:
        raise ValueError(f"{context} must be positive")
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
    # die 后端(契约 §3.1):顶层键集 = 核心键 + 计算描述二选一
    # ("compute" legacy 或 "die" 新后端,互斥判定在下方 compute 段;
    # 并存/全无均报错)。
    _required_top_level = {
        "schema-version",
        "slug",
        "label",
        "paper-case",
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
    # face A.2 alignment (wscllm removal 2026-09-24): the remote memory
    # backend was removed from this repository; the guard moved here from
    # the materialization validator so a hardware config declaring any
    # expansion fails closed at parse time.
    if remote_memory_type != "NO_MEMORY_EXPANSION":
        raise ValueError(
            "Hardware configuration.remote-memory.memory-type must be "
            "'NO_MEMORY_EXPANSION' (the remote memory backend was removed "
            f"from this repository), got {remote_memory_type!r}"
        )
    # A4-13/M7-7/global-7 (2026-09-26, face alignment): "bandwidth-gbps" is
    # deleted from the remote-memory schema -- the remote memory backend was
    # removed, the key was required but its value was never read or validated.
    _require_exact_keys(
        remote_memory,
        {"memory-type"},
        "Hardware configuration.remote-memory",
    )

    # die 后端(契约 §3.1):die 节与 compute.peak-perf-tflops 互斥出现——
    # 并存即报错(新后端不得静默沿用旧算力);二者全无同样报错。
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
        # die 节是唯一真值源:local-hbm.bandwidth-gbps 被派生值取代
        # (不校验两者相等;目标点上两者同为 1640.0——严格连续点)。
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
        peak_perf_tflops=peak_perf,
        metadata=metadata,
        die_node=(dict(die_node) if die_node is not None else None),
        die_summary=die_summary,
    )


def _stage_text(path: Path, contents: str) -> Path:
    """Write contents to an adjacent temp file without publishing it.

    E2 (2026-09-28): split from the former _write_text_atomically so
    materialize_runtime_configs can stage all derived files before the
    rename phase (transactional publish). The old single-file helper had no
    remaining callers and was removed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False) as temporary:
        temporary.write(contents)
        return Path(temporary.name)


def _serialize_json(value: object) -> str:
    return json.dumps(value, indent=2) + "\n"


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
    retired_template_keys = sorted(
        _RETIRED_SYSTEM_TEMPLATE_KEYS.intersection(system_template))
    if retired_template_keys:
        raise ValueError(
            "System template carries retired custom-collective key(s): "
            + ", ".join(retired_template_keys)
            + " (custom collectives are retired in this repository; remove "
            "them and use the native implementation keys instead)")
    # Value-combination gate (2026-09-30, mirror of the C++
    # CollectiveImplLookup key gate): all-to-all supports only the
    # direct/oneDirect family, doubleBinaryTree only all-reduce -- any other
    # name under these keys used to pass here and die inside the simulator
    # at config-load time (or at stream init before the C++ gate existed)
    # with a far-away message. Reject at this earliest boundary instead.
    def _impl_allowed(impl: str, key: str) -> bool:
        direct_family = impl.startswith("direct") or impl.startswith("oneDirect")
        if key == "all-to-all-implementation":
            return direct_family
        if impl == "doubleBinaryTree":
            return key == "all-reduce-implementation"
        return True

    for _impl_key in ("all-reduce-implementation",
                      "reduce-scatter-implementation",
                      "all-gather-implementation",
                      "all-to-all-implementation"):
        _value = system_template.get(_impl_key)
        if isinstance(_value, list):
            for _impl in _value:
                if isinstance(_impl, str) and not _impl_allowed(_impl, _impl_key):
                    raise ValueError(
                        "System template key '" + _impl_key + "' carries '"
                        + _impl + "': not a legal combination (all-to-all "
                        "supports only the direct/oneDirect family, "
                        "doubleBinaryTree only all-reduce)")

    unknown_template_keys = sorted(set(system_template) - _KNOWN_SYSTEM_TEMPLATE_KEYS)
    if unknown_template_keys:
        # 3.12: a typo'd template key would otherwise flow into the
        # generated system.json and be silently ignored by the C++ reader
        # (the feature stays off with zero diagnostics).
        raise ValueError(
            "System template contains unknown key(s) (typo? see "
            "_KNOWN_SYSTEM_TEMPLATE_KEYS / Sys.cc known_system_keys): "
            + ", ".join(unknown_template_keys)
        )
    comm_groups = _prepare_comm_groups(inference_groups, hardware)

    system = dict(system_template)
    system["local-mem-bw"] = hardware.local_hbm_bandwidth_gbps
    system["local-mem-latency"] = hardware.local_hbm_latency_ns
    # P11 死键清除（2026-09-23 深挖审计；本仓 2026-09-27 对齐姊妹仓执行）：
    # local-mem-capacity-bytes 曾在此写入 system.json，而 C++ 侧零读取
    # （Sys.cc 只解析 local-mem-bw/local-mem-latency）——写入链退役；
    # _MANAGED_SYSTEM_FIELDS 仍保留该键（模板禁写守卫，防死键回流）。
    system["peak-perf"] = hardware.peak_perf_tflops

    destination = Path(output_dir)
    die_model_path: Optional[Path] = None
    if hardware.die_summary is not None:
        # die 后端(契约 §3.2/§3.3):resolved_die_model.json 落
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
        system["die-model-config"] = str(die_model_path.resolve())
    paths = RuntimeConfigPaths(
        system=destination / "system.json",
        network=destination / "network.yml",
        comm_group=destination / "comm_group.json",
        die_model=die_model_path,
    )
    # E2 (2026-09-28): the three derived files are now published as one
    # transaction (aligned with traces/derive_20_first_30_seconds.py's
    # stage-all-then-rename-all pattern). Previously each file was
    # written+renamed in turn, so a crash between the writes left a mixed
    # old/new configuration set in place. All three temp files are staged
    # first; only then does the rename loop publish. POSIX has no multi-file
    # atomic rename, so a crash inside the rename loop itself can still
    # partially publish -- but the window shrinks from full-write duration
    # to the rename loop, and any failure before the first rename leaves the
    # previous set byte-identical.
    staged: list[tuple[Path, Path]] = []
    try:
        derived_files: list[tuple[Path, str]] = [
            (paths.system, _serialize_json(system)),
            (paths.network, _network_yaml(hardware)),
            (paths.comm_group, _serialize_json(comm_groups)),
        ]
        if die_model_path is not None:
            derived_files.append(
                (die_model_path, _serialize_json(hardware.die_summary)))
        for path, contents in derived_files:
            staged.append((path, _stage_text(path, contents)))
        for path, temporary in staged:
            os.replace(temporary, path)
    except BaseException:
        for _, temporary in staged:
            Path(temporary).unlink(missing_ok=True)
        raise
    return paths
