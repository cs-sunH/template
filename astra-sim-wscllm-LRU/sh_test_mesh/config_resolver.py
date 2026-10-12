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


# 阶段 C(2026-10-03,契约 §3.2/§7 裁定 1):"die-model-config" 加入
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
# M-28 收紧(2026-09-30):拓扑白名单 = Python 全栈(邻接/曼哈顿距离/XY
# 路由)与 C++ 后端实际一致的平面网格取值。C++ parser 把 "Line"/"Mesh"
# 同映射为二维 Mesh;Ring 环绕/FullyConnected 单跳与 Python 曼哈顿静默
# 分叉、Switch 在 C++ 构造期 exit(-1)——超售即 fail-closed。
_SUPPORTED_TOPOLOGIES = frozenset({"Line", "Mesh"})

# ---------------------------------------------------------------------------
# die 成本库 C ABI(阶段 C;契约 §3.3/§7 裁定 1)。resolver 通过 ctypes
# 调 libdie_model_cabi.so 取硬件派生值——禁止在 Python 复写第二套派生
# 公式。发现序(env DIE_MODEL_CABI_PATH -> 仓内 build/astra_analytical/
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

# 3.12 system-template whitelist (ported from the wscllm mother repo,
# 2026-10-01; same contract as the C++ unknown-key gate in
# Sys::initialize_sys): a typo'd template key used to pass this resolver
# freely, reach the generated system.json, and be silently ignored by the
# C++ reader (fail-open: the feature the key was meant to configure stayed
# off with zero diagnostics). Validate the template keys HERE, at the
# earliest boundary. Keep this set in sync with known_system_keys in
# astra-sim/system/Sys.cc (which stays as the last line of defense). Same
# tree-specific deltas as the C++ set: the five custom-collective keys are
# LIVE in this repository (the custom path is retained), the five
# remote-memory/KV keys are read by this repo's Sys.cc, and
# "intra-dimension-scheduling" (known to the mother repo) is absent because
# this repo has no reader for it. Managed hardware fields listed here as
# well stay unreachable in templates: the _MANAGED_SYSTEM_FIELDS gate
# above rejects them first with its own message.
_KNOWN_SYSTEM_TEMPLATE_KEYS = frozenset(
    {
        "scheduling-policy",
        "inter-dimension-scheduling",
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
        "roofline-enabled",
        "hbm-kv-restore-bandwidth-sharing",
        "hbm-bandwidth-contention",
        "trace-enabled",
        "replay-only",
        "compute-cost-model",
        "die-model-profile",
        "die-model-config",
        "all-reduce-implementation",
        "reduce-scatter-implementation",
        "all-gather-implementation",
        "all-to-all-implementation",
        "all-reduce-implementation-custom",
        "all-gather-implementation-custom",
        "reduce-scatter-implementation-custom",
        "all-to-all-implementation-custom",
        "per-node-custom-implementation",
    }
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
    # 阶段 C(die 后端,契约 §3.1/§3.3):die 节存在时非 None——die 六参数
    # 原文 + 库派生摘要(derived 全字段);peak_perf_tflops /
    # local_hbm_bandwidth_gbps 此时来自 derived(PE_peak_die、HBM 总带宽/
    # 1e9),legacy 路径两值为空/沿用 compute 节。本仓 KV 账本消费的
    # local_hbm_capacity_bytes 在 die 节下等于派生 HBM 容量(容量对齐
    # 硬校验,契约 §3.4)——准入/逐出行为随容量变化属正常反馈,远程/
    # 迁移机制照常兜底。
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
        # M-28(2026-09-30):NO_MEMORY_EXPANSION 分支随 loader 白名单收紧
        # 删除(本仓 KV 三态冷热管理依赖远端池,无池配置超售会在首个远端
        # 访问时 C++ sys_panic)。
        if self.remote_memory_npu_selection == "mesh-boundary":
            return "edge_remote_memory_pool"
        # （2026-10-01 注记）该回退臂生产不可达：load_hardware_config 已
        # 把 npu-selection fail-closed 到 mesh-boundary（直构 dataclass
        # 的测试也走 mesh-boundary）——防御性保留。
        return "remote_memory_expansion"


@dataclass(frozen=True)
class RuntimeConfigPaths:
    system: Path
    network: Path
    remote_memory: Path
    comm_group: Path
    # 阶段 C:die 后端时为 resolved_die_model.json 的路径(与三件套同
    # 目录),legacy 时为 None。
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
    # 阶段 C:顶层键集 = 核心八键 + "notes" + 计算描述二选一("compute"
    # legacy 或 "die" 新后端,契约 §3.1;互斥/缺一判定在下方 compute 段)。
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
    # M-28 收紧(2026-09-30):NO_MEMORY_EXPANSION 超售拒绝——Python 侧
    # KV 三态管理无"无远端池"概念,首个远端访问会在 C++ sys_panic;唯一
    # 实际支持取值为 PER_NPU_MEMORY_EXPANSION(mesh-boundary)。
    if remote_memory_type != "PER_NPU_MEMORY_EXPANSION":
        raise ValueError(
            "Hardware configuration.remote-memory.memory-type is unsupported "
            f"(this stack requires the remote-memory pool): {remote_memory_type}"
        )
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
    remote_memory_bandwidth = _require_number(
        remote_memory["bandwidth-gbps"],
        "Hardware configuration.remote-memory.bandwidth-gbps",
        positive=True,
    )

    # 阶段 C(契约 §3.1):die 节与 compute.peak-perf-tflops 互斥出现——
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
        # 本仓容量与 KV 账本耦合:对齐后容量=派生值,准入/逐出行为变化
        # 属正常反馈(远程池/迁移机制照常兜底,不判请求失败)。
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
        die_node=(dict(die_node) if die_node is not None else None),
        die_summary=die_summary,
        metadata=metadata,
    )


def _write_text_atomically(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False) as temporary:
        # S4-1(2026-10-01,face 同款):write 本体失败(如 ENOSPC)时临时文件
        # 已以 delete=False 落盘——不清理会残留 .tmp 伴生文件。正常路径
        # 字节不变(os.replace 原子性不受影响,replace 侧清理原已在位)。
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
    # M-28(2026-09-30):NO_MEMORY_EXPANSION 早退分支随 loader 白名单收紧
    # 删除——唯一支持取值 PER_NPU_MEMORY_EXPANSION 恒携带 latency 与
    # mesh-boundary 边缘 rank 集。
    remote: dict[str, Any] = {
        "memory-type": hardware.remote_memory_type,
        # 单位约定（RB-01, 2026-10-01）：本键与 system.json 的同名键
        # "remote-mem-bw" 均写入同一源值 remote_memory_bandwidth_gbps
        # （hardware json remote-memory.bandwidth-gbps，十进制 GB/s）。
        # 两个消费方约定相反：system.json 侧 Sys.cc 按 GB/s 消费（×1e9 →
        # B/s）；本文件（remote_memory.json）侧 AnalyticalRemoteMemory.cc
        # 按原始数值 = B/ns 消费（无换算）。数值恒等成立：1 GB/s ≡ 1 B/ns
        # （10^9 进制恒等），故同一数值两处正确——改值/改语义时两侧必须
        # 同步，单一侧手改会静默偏 1e9 倍。
        "remote-mem-bw": hardware.remote_memory_bandwidth_gbps,
        "remote-mem-latency": hardware.remote_memory_latency_ns,
    }
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
    remote["npu-ids"] = boundary_ranks
    return remote


def _prepare_comm_groups(
    inference_groups: Sequence[tuple[str, Sequence[int]]], hardware: ResolvedHardware
) -> dict[str, dict[str, object]]:
    groups: dict[str, dict[str, object]] = {}
    seen_ids: set[int] = set()
    # M-29(2026-09-30):跨组 rank 全局账本——组内重复由 seen_ranks 拦截,
    # 跨组重叠与整网覆盖缺口此前均不设防(物化看似成功、online_service
    # 启动才崩),与本仓"loader 侧 fail-closed"纪律对齐。
    claimed_ranks: set[int] = set()
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
        # M-29:跨组 rank 重叠 fail-closed(共享 rank 会令 C++ Communicator
        # 组与 Python 实例账本对同一 rank 双重归属)。
        overlap = claimed_ranks & seen_ranks
        if overlap:
            raise ValueError(
                f"Communicator group {pg_id} ranks overlap earlier groups at "
                f"{sorted(overlap)}; every NPU must belong to exactly one group"
            )
        claimed_ranks |= seen_ranks
        rows = {rank // hardware.mesh_cols for rank in ranks}
        columns = {rank % hardware.mesh_cols for rank in ranks}
        # A4-10(2026-09-26 face 审查修复;2026-10-01 同款移植):C++ 侧按
        # 列表位次消费 ranks——CommunicatorGroup 的
        # communicator_rank_position/get_position_in_group 以输入序定位
        # (ET 文件选择与 algorithm-rank 翻译都按位次索引),RingTopology
        # 环邻居按 index 相接。只校验 rank 集合会放过乱序列表:集合等价但
        # 顺序错位的组会静默建出转置环。position-by-position 有序比较完全
        # 蕴含原"contiguous row-major rectangle"集合等价校验(任何非矩形
        # 集合同样过不了有序比较),故二者合一,只保留这一个 fail-closed 门。
        # 本仓 M-29 三门(组内去重/跨组重叠/整网覆盖)与 >=2x2 维度下限门
        # 为本仓独有强化,原样保留、不受影响。
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
                f"dim0-fastest row-major order (the C++ CommunicatorGroup/"
                f"RingTopology consume them in that order); "
                f"expected head {expected_order[:4]}, got {ranks[:4]} "
                f"({mismatched})"
            )
        # RingTopology (the per-dimension collective backend) fail-closes on
        # any dimension with fewer than two ranks, so a 1xN / Nx1 group would
        # pass resolution here and only explode at the first collective on
        # the C++ side. Reject it at configuration time with the same
        # fail-closed contract.
        if len(rows) < 2 or len(columns) < 2:
            raise ValueError(
                f"Communicator group {pg_id} spans {len(rows)} row(s) x "
                f"{len(columns)} column(s); every group must cover at least "
                "2 rows and 2 columns (a ring dimension needs >= 2 ranks)"
            )
        groups[str(pg_id)] = {
            "ranks": ranks,
            "dimensions": [len(columns), len(rows)],
        }
    # M-29:整网覆盖 fail-closed——组集必须恰好覆盖全部物理 NPU(与
    # 调度器 build_instances 的"every configured NPU exactly once"不变
    # 量同源,缺口在 loader 侧拦截而非物化后启动期)。
    if claimed_ranks != set(range(hardware.npus_count)):
        missing = sorted(set(range(hardware.npus_count)) - claimed_ranks)
        raise ValueError(
            f"Communicator groups must cover every physical NPU exactly "
            f"once; uncovered ranks: {missing}"
        )
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
    # 值组合门（2026-10-01 移植，C++ CollectiveImplLookup 键门 / Ring.cc
    # plain-Ring 拒驱动 All_to_All 的配置侧镜像）：all-to-all 只支持
    # direct/oneDirect 族、doubleBinaryTree 只支持 all-reduce——这些键下的
    # 其它名字以前会在这里放行、再死在仿真器内（config 装载期或流初始化，
    # 报错远离根因）。改在最早的边界拒绝。
    def _impl_allowed(impl: str, key: str) -> bool:
        # direct 族词法镜像 C++ CollectiveImplLookup（R2 复核收紧）：
        # 'direct'/'oneDirect' 裸名合法，否则后缀必须是一个十进制正整数
        #（窗口 >= 1，占满余串）——'directX'/'direct0' 之类此前经
        # startswith 放行、漏到 C++ 装载期才炸。
        def _direct_family_ok(token: str) -> bool:
            if token == "direct" or token == "oneDirect":
                return True
            for prefix in ("direct", "oneDirect"):
                if token.startswith(prefix):
                    suffix = token[len(prefix):]
                    return suffix.isascii() and suffix.isdigit()                         and int(suffix) >= 1
            return False

        if key == "all-to-all-implementation":
            return _direct_family_ok(impl)
        if impl == "doubleBinaryTree":
            return key == "all-reduce-implementation"
        return True

    for _impl_key in ("all-reduce-implementation",
                      "reduce-scatter-implementation",
                      "all-gather-implementation",
                      "all-to-all-implementation"):
        _value = system_template.get(_impl_key)
        # 形态门（R3 复核补强）：这四个键的 C++ 消费面（Sys 构造期的实现
        # 表解析）只接受数组——标量/对象形态此前不经任何门直接写入
        # system.json、死在 C++ 装载期，正是本门要消灭的"报错远离根因"。
        # 显式 null 与非 list 形态同拒（R4 补强）：键缺席（get 缺省 None）
        # 与键显式为 null 要区分——前者合法，后者写入 system.json 后死在
        # C++ 装载期（报错远离根因），正是本门要消灭的形态。
        if _impl_key in system_template and not isinstance(_value, list):
            raise ValueError(
                "System template key '" + _impl_key + "' must be an array "
                "of implementation names (got " + repr(_value) + ")")
        if isinstance(_value, list):
            for _impl in _value:
                # 非字符串元素（R2 复核收紧）此前被静默跳过、写进
                # system.json 后到 C++ 装载期 json 类型错才炸——在最早的
                # 边界点名拒绝。
                if not isinstance(_impl, str):
                    raise ValueError(
                        "System template key '" + _impl_key + "' carries a "
                        "non-string implementation entry " + repr(_impl))
                if not _impl_allowed(_impl, _impl_key):
                    raise ValueError(
                        "System template key '" + _impl_key + "' carries '"
                        + _impl + "': not a legal combination (all-to-all "
                        "supports only the direct/oneDirect family with an "
                        "optional >=1 decimal window, doubleBinaryTree only "
                        "all-reduce)")
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
    remote_memory = _prepare_remote_memory(hardware)
    comm_groups = _prepare_comm_groups(inference_groups, hardware)

    system = dict(system_template)
    system["local-mem-bw"] = hardware.local_hbm_bandwidth_gbps
    system["local-mem-latency"] = hardware.local_hbm_latency_ns
    # P11 死键清除（2026-09-23 深挖审计）：local-mem-capacity-bytes 曾在此
    # 写入 system.json，而 C++ 侧零读取（全仓 grep 无读者）——写入链退役；
    # _MANAGED_SYSTEM_FIELDS 仍保留该键（模板禁写守卫，防死键回流）。
    system["remote-mem-bw"] = hardware.remote_memory_bandwidth_gbps
    # 单位约定（RB-01, 2026-10-01）：本键按 GB/s 写入（Sys.cc 读取时 ×1e9
    # → B/s）；remote_memory.json 的同名键写入同一源值但按 B/ns 原始数值
    # 消费（AnalyticalRemoteMemory.cc 无换算）。GB/s 数值 ≡ B/ns 数值
    # （10^9 进制恒等），两文件数值必须始终来自同一源值——详见
    # _prepare_remote_memory 内的完整约定注释。
    # M-28 后残留清理（2026-10-01）：_prepare_remote_memory 恒写
    # remote-mem-latency（PER_NPU_MEMORY_EXPANSION 唯一支持取值恒携带），
    # 此处的成员存在性判断恒真，直接赋值。
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
