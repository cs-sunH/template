#!/usr/bin/env python3
"""Shared WSC-LLM configuration and GraphBatch-emission primitives for online runs."""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

if TYPE_CHECKING:
    # 仅注解用:在线侧 builder 才具备 KV 发射原语全集(comm_recv 的
    # hbm_charge/mem_store 的 hbm_access_mode/local_hbm_kv_restore);
    # 离线 TraceBuilder 传入即 TypeError/AttributeError(py-metrics-C3
    # 修复:注解改指真实契约)。运行时不 import,避免与
    # online.graph_batch_builder 循环。
    from online.graph_batch_builder import OnlineTraceBuilder


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

SH_TEST_DIR = MODULE_DIR.parents[1]
if str(SH_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(SH_TEST_DIR))

from wsc_llm_scheduler import (  # noqa: E402
    DECODE_ROLE,
    PREFILL_ROLE,
    WscLlmHardware,
    WscLlmModel,
    # 再导出（非死 import，2026-09-30 复核回置）：生产消费方是
    # online/graph_batch_builder.py 的多行 import 列表；曾误判死删除。
    kv_cache_bytes_for_tokens,
)
from session_kv_manager import (  # noqa: E402
    KVTransfer,
    KVTransferShard,
    physical_edge_ranks,
)
from generate_trace import (  # noqa: E402
    RequestSpec,
    clean_csv_row,
    load_request_queue,
    parse_bool,
    parse_int,
    parse_nonnegative_int,
    parse_rank_spec,
    sanitize_node_prefix,
    shard_extent,
    transformer_pass_aggregated,
)
from config_resolver import (  # noqa: E402
    ResolvedHardware,
    load_hardware_config,
    materialize_runtime_configs,
)


CONFIG_CSV_PATH = MODULE_DIR / "trace_config.csv"
CONFIG_COLUMNS = (
    "kind",
    "key",
    "value",
    "group_name",
    "pg_name",
    "ranks",
    "phase_role",
)
REQUIRED_CONFIG_KEYS = (
    "layers",
    "hidden_size",
    "ffn_size",
    "num_heads",
    "vocab_size",
    "bytes_per_elem",
    "model_name",
    "mlp_variant",
    "output_dir",
    "request_queue_csv",
    "hardware_config",
    "local_hbm_capacity_profile",
    "system_template",
    "remote_operand_loads",
)
OPTIONAL_CONFIG_DEFAULTS = {
    "trace_granularity": "request_aggregated",
    "prefill_chunk_size": "512",
    "kv_cache_policy": "session_lru_tiered",
    # kv_reserve_context_tokens 为遗留观测键:loader/生成管线零消费(仅键
    # 校验),只在 trace_config.csv 携带、由 slo_tools/hbm_watermark.py 原始
    # 读 CSV 作 SLO 观测口径。record_planning_iterations 键已随 CSV 行退役
    # (S3-5,对齐 BASE 退役批);output_dir 键在本仓仍为 loader REQUIRED
    # 键,与 BASE 不同,不得删行——但其值"只验不存"(2026-10-01 审查
    # 修正:值恒弃,输出目录由 plan_materializer 自行派生,见 parsed
    # 组装处注)。
    "kv_reserve_context_tokens": "0",
    # 阶段 C(die 后端):位置编码声明(rope|none;缺省 none 向后兼容)。
    # 模型启用 RoPE 时新 Q/K 旋转折入 qkv_rope 壳末段(die_model_adapter
    # 构造 spec 的 model_dims.position_encoding;设计文档 §7.1)。
    "position_encoding": "none",
}
SUPPORTED_CONFIG_KEYS = set(REQUIRED_CONFIG_KEYS) | set(OPTIONAL_CONFIG_DEFAULTS)
INT_CONFIG_KEYS = {
    "layers",
    "hidden_size",
    "ffn_size",
    "num_heads",
    "vocab_size",
    "bytes_per_elem",
}
# output_dir 不入 PATH_CONFIG_KEYS(2026-10-01 审查修正):该键"只验不存"
# ——存在性仍由 REQUIRED_CONFIG_KEYS 门校验(兼容外部 CSV,不得删行),
# 值在本仓恒弃(输出目录由 plan_materializer 自行派生),不再经
# _parse_config_value 解析进 parsed。
PATH_CONFIG_KEYS = {
    "request_queue_csv",
    "hardware_config",
    "system_template",
}
@dataclass(frozen=True)
class WscLlmInferenceGroup:
    name: str
    pg_name: str
    ranks: tuple[int, ...]
    phase_role: str


@dataclass(frozen=True)
class WscLlmTraceConfig:
    config_csv: Path
    npus_count: int
    layers: int
    hidden_size: int
    ffn_size: int
    num_heads: int
    vocab_size: int
    bytes_per_elem: int
    model_name: str
    mlp_variant: str
    request_queue_csv: Path
    request_queue: tuple[RequestSpec, ...]
    hardware_config: Path
    hardware_capacity_profile: str
    hardware: WscLlmHardware
    system_template: Path
    system_config: Path
    network_config: Path
    comm_group_config: Path
    remote_operand_loads: bool
    inference_groups: tuple[WscLlmInferenceGroup, ...]
    trace_granularity: str
    prefill_chunk_size: int
    kv_cache_policy: str
    position_encoding: str = "none"

    @property
    def model(self) -> WscLlmModel:
        return WscLlmModel(
            layers=self.layers,
            hidden_size=self.hidden_size,
            ffn_size=self.ffn_size,
            num_heads=self.num_heads,
            vocab_size=self.vocab_size,
            bytes_per_elem=self.bytes_per_elem,
            mlp_variant=self.mlp_variant,
        )


# ---------------------------------------------------------------------------
# B3(2026-09-06) KV 转移发射三件套(照抄 sh_2.0 generate_face_trace.py
# 承袭的类形与语义;契约 §6:tag 基址与 _stage_tag 的
# queue_index*10000+{category} 段错开。boundary-C1 措辞修正
# (2026-09-26):现役段仅 category=3000(P→D joiner 配对迁移一个调用点);
# noc_migrate/history 恢复全部走分配器段,{1000,1900} 无运行时产生者)。
# H8(2026-09-24)修正:基址自 10_000_000 上移至 100_000_000——30s 窗口
# 源 trace 实测 1177 行,queue_index≥1000 的 _stage_tag 已进入旧基址的
# 分配器独占段;新基址下队列 ≤9999 行时两段保持不相交,且 _stage_tag
# 侧补了越界 fail-closed 守卫(见 _stage_tag)。
# boundary-E1 修正(2026-09-26):分配器上限自 0xFFFFFFFF 收紧至 C++
# NATIVE 段上界 500_000_000——Sys 对 NATIVE 类型无条件 tag%5e8,越过
# 5e8 的 tag 会静默别名回低位段(而非 fail-closed),>INT32_MAX 还先遭
# uint32→int 收窄;守卫改在 Python 分配侧闭合(上限语义 = 半开区间)。
# ---------------------------------------------------------------------------


@dataclass
class PendingHistoryGate:
    """跨请求 history 门:上一 turn 完成 → 下一 turn 准入之间的会话
    KV 位置账本(timer_gates 在下一 turn 发射时由 interval gate 节点 id
    填充;location 由 sync_pending_history_after_evictions 镜像逐出)。"""

    source_instance_index: int
    timer_gates: tuple[Optional[int], ...]
    location: str


@dataclass(frozen=True)
class TransferTriggerGate:
    """逐出发射的触发门(control 实例各 rank 的节点门;control rank 与
    数据源 rank 不同实例时以 1B p2p 触发,同 rank 直接 arm)。"""

    control_instance_index: int
    node_gates: tuple[Optional[int], ...]


class TransferTagAllocator:
    """KV 转移通信 tag 单调分配器(基址 100_000_000;上限 500_000_000)。

    契约 §6 硬约束:sh_2.0 原版从 1 起会撞进本仓 ``queue_index*10000+
    {category}`` 的 _stage_tag 段,统一错开到安全高位段。H8(2026-09-24):
    基址 10_000_000 只能容纳 <1000 行队列(30s 窗口实测 1177 行已重叠),
    上移至 100_000_000(队列 ≤9999 行不相交)。
    boundary-E1(2026-09-26):上限 = C++ NATIVE 段半开上界 500_000_000
    (Sys.hh FrontEndSendRecvType;Sys 对 NATIVE 无条件 tag%5e8,≥5e8 即
    静默别名),分配越界在 Python 侧 fail-closed,不再放行至 0xFFFFFFFF。"""

    _TAG_BASE = 100_000_000
    _NATIVE_TAG_LIMIT = 500_000_000  # 半开区间上界:tag 必须 < 5e8

    def __init__(self) -> None:
        self._next_tag = self._TAG_BASE

    def take(self) -> int:
        if self._next_tag >= self._NATIVE_TAG_LIMIT:
            raise OverflowError(
                "KV transfer communication tag exhausted the C++ NATIVE "
                f"segment [0, {self._NATIVE_TAG_LIMIT}) (Sys wraps NATIVE "
                "tags modulo 5e8; larger tags would silently alias)")
        tag = self._next_tag
        self._next_tag += 1
        return tag


def _resolve_from_sh_test(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (SH_TEST_DIR / path).resolve()


def _resolve_request_queue(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (SH_TEST_DIR / "workload" / path).resolve()


def _frontier_neutral(builder, emit_fn, *arm_ids):
    """死锁真修（方案①，2026-10-11）：把一段发射包成**不推进主链
    frontier** 的中立段——checkpoint → 暂存/清空 pending 依赖 →
    显式 arm 前驱节点（排序语义由显式边承担，不再靠链位）→ 发射
    （段内节点自行接续成链）→ restore 主链。ack 对 fork/restore
    idiom 的统一载体（generate_wsc_llm_trace 与 graph_batch_builder
    共用）；与 _emit_side_branch 的差别：单 builder 粒度、arm 显式化。

    - arm_ids：进入本段前显式 arm 的同 rank 前驱节点 id（None 跳过）；
      由本段首节点消费。
    - 返回值：段尾节点 id（emit_fn 末个发射节点；调用方显式边落账用，
      与 _emit_train_marker 取 previous_id 同款口径）。emit_fn 必须
      至少发射一个节点（arm 未消费即触发上面的泄漏 raise）。
    - 泄漏 fail-closed：emit_fn 若未消费全部显式 arm、或留下新的未消费
      pending 依赖（触发门泄漏），raise——与逐出旁路支链同款纪律。
    """
    checkpoint = builder.chain_checkpoint()
    if builder.pending_extra_dependencies:
        builder.pending_extra_dependencies = []
    for arm_id in arm_ids:
        if arm_id is not None:
            builder.arm_dependency(arm_id)
    try:
        emit_fn()
        if builder.pending_extra_dependencies:
            raise RuntimeError(
                "frontier-neutral emission left unconsumed pending "
                "dependencies on rank {}".format(builder.rank))
        tail = builder.previous_id
        if tail is None or tail == checkpoint[0]:
            raise RuntimeError(
                "frontier-neutral emission produced no node on rank "
                "{}".format(builder.rank))
    finally:
        builder.restore_chain(checkpoint)
    return tail


def _parse_config_value(key: str, value: str) -> object:
    if key in INT_CONFIG_KEYS:
        return parse_int(value, key)
    if key == "remote_operand_loads":
        return parse_bool(value, key)
    if key == "prefill_chunk_size":
        return parse_int(value, key)
    if key == "kv_reserve_context_tokens":
        return parse_nonnegative_int(value, key)
    if key == "kv_cache_policy":
        # 唯一取值 session_lru_tiered(session 级二态冷热管理:完整本地/
        # 完整远端、整体 LRU 逐出 + 远端池全量恢复;2026-09-25 session 级
        # Tiered-LRU 批起原 PARTIAL 半层化三态与两段式逐出已物理移除);
        # 旧值别名与其同调度器(无档位分支),
        # 已随 2026-09-25 命名卫生直接清除。其余取值 fail-closed。
        if value != "session_lru_tiered":
            raise ValueError(
                "config key kv_cache_policy must be session_lru_tiered, "
                f"got {value!r}"
            )
        return value
    if key == "position_encoding":
        # 阶段 C:值域 rope|none(fail-closed);缺省 none 向后兼容。
        if value not in ("rope", "none"):
            raise ValueError(
                "config key position_encoding must be rope or none, "
                f"got {value!r}"
            )
        return value
    if key == "trace_granularity":
        # 值域收紧为单值 request_aggregated(S3-4):token_expanded 发射面
        # 已物理删除,在线唯一调用链 fail-closed 限定 request_aggregated;
        # 保留双值域会让缺省值(token_expanded)直接撞 fail-closed。
        if value != "request_aggregated":
            raise ValueError(
                "config key trace_granularity must be request_aggregated, "
                f"got {value!r}"
            )
        return value
    if key == "mlp_variant":
        if value not in {"gelu", "swiglu"}:
            raise ValueError("config key mlp_variant must be gelu or swiglu")
        return value
    if key in PATH_CONFIG_KEYS:
        if not value:
            raise ValueError(f"config key {key} must not be empty")
        return Path(value)
    if not value:
        raise ValueError(f"config key {key} must not be empty")
    return value


def _to_wsc_llm_hardware(hardware: ResolvedHardware) -> WscLlmHardware:
    return WscLlmHardware(
        mesh_rows=hardware.mesh_rows,
        mesh_cols=hardware.mesh_cols,
        local_hbm_capacity_bytes=hardware.local_hbm_capacity_bytes,
        local_hbm_bandwidth_gbps=hardware.local_hbm_bandwidth_gbps,
        d2d_bandwidth_gbps=hardware.d2d_bandwidth_gbps,
        peak_perf_tflops=hardware.peak_perf_tflops,
        d2d_latency_ns=hardware.d2d_latency_ns,
        local_hbm_latency_ns=hardware.local_hbm_latency_ns,
        label=hardware.label,
    )


def load_wsc_llm_trace_config(config_csv: Path = CONFIG_CSV_PATH) -> WscLlmTraceConfig:
    if not config_csv.exists():
        raise FileNotFoundError(f"trace config CSV not found: {config_csv}")
    values: dict[str, str] = {}
    groups: list[WscLlmInferenceGroup] = []
    with config_csv.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"trace config CSV is empty: {config_csv}")
        columns = {column.strip() for column in reader.fieldnames if column is not None}
        missing_columns = [column for column in CONFIG_COLUMNS if column not in columns]
        if missing_columns:
            raise ValueError(
                f"trace config CSV is missing columns: {', '.join(missing_columns)}"
            )
        for line_number, raw_row in enumerate(reader, start=2):
            row = clean_csv_row(raw_row)
            kind = row.get("kind", "").lower()
            if not kind or kind.startswith("#") or kind == "comment":
                continue
            if kind == "config":
                key = row.get("key", "")
                if key not in SUPPORTED_CONFIG_KEYS:
                    raise ValueError(f"line {line_number}: unsupported config key {key!r}")
                if key in values:
                    raise ValueError(f"line {line_number}: duplicate config key {key!r}")
                values[key] = row.get("value", "")
                continue
            if kind == "inference_group":
                name = row.get("group_name", "")
                pg_name = row.get("pg_name", "")
                phase_role = row.get("phase_role", "").lower()
                if not name or not pg_name or not phase_role:
                    raise ValueError(
                        f"line {line_number}: inference_group requires name, pg_name, "
                        "and phase_role"
                    )
                if phase_role not in {PREFILL_ROLE, DECODE_ROLE}:
                    raise ValueError(
                        f"line {line_number}: phase_role must be prefill or decode"
                    )
                groups.append(
                    WscLlmInferenceGroup(
                        name=name,
                        pg_name=pg_name,
                        ranks=parse_rank_spec(row.get("ranks", "")),
                        phase_role=phase_role,
                    )
                )
                continue
            raise ValueError(f"line {line_number}: unsupported row kind {kind!r}")

    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in values]
    if missing:
        raise ValueError(f"trace config CSV is missing keys: {', '.join(missing)}")
    if not groups:
        raise ValueError("trace config requires at least one inference_group")
    # output_dir"只验不存"(2026-10-01 审查修正):REQUIRED 门已校验键
    # 存在,值恒弃——parsed["output_dir"] 改前即"解析后即弃"的死数据
    # (WscLlmTraceConfig 无该字段、组装不传入、零读者),故不再解析进
    # parsed;键存在性校验行为与改前一致。
    parsed = {
        key: _parse_config_value(key, values.get(key, default_value))
        for key, default_value in {
            **{
                key: values[key]
                for key in REQUIRED_CONFIG_KEYS
                if key != "output_dir"
            },
            **OPTIONAL_CONFIG_DEFAULTS,
        }.items()
    }

    request_queue_csv = _resolve_request_queue(parsed["request_queue_csv"])
    if not request_queue_csv.is_file():
        print(
            f"missing request queue: {request_queue_csv};"
            "请按 traces/derive_20_first_30_seconds.py 物化输入"
            "(运行 stdout 即权威 provenance 记录)",
            file=sys.stderr,
        )
        sys.exit(1)
    request_queue = load_request_queue(request_queue_csv)
    hardware_path = _resolve_from_sh_test(parsed["hardware_config"])
    hardware_capacity_profile = str(parsed["local_hbm_capacity_profile"])
    resolved_hardware = load_hardware_config(
        hardware_path,
        hardware_capacity_profile,
    )
    hardware = _to_wsc_llm_hardware(resolved_hardware)
    npus_count = resolved_hardware.npus_count
    system_template = _resolve_from_sh_test(parsed["system_template"])
    runtime_config_dir = (
        SH_TEST_DIR
        / "generated"
        / "runtime_config"
        / (
            f"{resolved_hardware.slug}__{hardware_capacity_profile}__"
            f"{resolved_hardware.remote_memory_runtime_label}"
        )
    )
    runtime_configs = materialize_runtime_configs(
        hardware=resolved_hardware,
        system_template_path=system_template,
        inference_groups=tuple((group.pg_name, group.ranks) for group in groups),
        output_dir=runtime_config_dir,
    )
    return WscLlmTraceConfig(
        config_csv=config_csv.resolve(),
        npus_count=npus_count,
        layers=int(parsed["layers"]),
        hidden_size=int(parsed["hidden_size"]),
        ffn_size=int(parsed["ffn_size"]),
        num_heads=int(parsed["num_heads"]),
        vocab_size=int(parsed["vocab_size"]),
        bytes_per_elem=int(parsed["bytes_per_elem"]),
        model_name=str(parsed["model_name"]),
        mlp_variant=str(parsed["mlp_variant"]),
        request_queue_csv=request_queue_csv,
        request_queue=request_queue,
        hardware_config=hardware_path,
        hardware_capacity_profile=hardware_capacity_profile,
        hardware=hardware,
        system_template=system_template,
        system_config=runtime_configs.system,
        network_config=runtime_configs.network,
        comm_group_config=runtime_configs.comm_group,
        remote_operand_loads=bool(parsed["remote_operand_loads"]),
        inference_groups=tuple(groups),
        trace_granularity=str(parsed["trace_granularity"]),
        prefill_chunk_size=int(parsed["prefill_chunk_size"]),
        kv_cache_policy=str(parsed["kv_cache_policy"]),
        position_encoding=str(parsed["position_encoding"]),
    )


def _stage_tag(queue_index: int, category: int, relative_rank: int) -> int:
    tag = queue_index * 10000 + category + relative_rank
    if tag >= TransferTagAllocator._TAG_BASE:
        raise OverflowError(
            f"stage tag {tag} (queue_index={queue_index}) would enter the "
            "KV transfer allocator's exclusive tag segment "
            f"[{TransferTagAllocator._TAG_BASE}, "
            f"{TransferTagAllocator._NATIVE_TAG_LIMIT}); shrink the "
            "request queue or raise the allocator base")
    return tag


def _xy_route(hardware: WscLlmHardware, source: int, target: int) -> list[int]:
    row, col = divmod(source, hardware.mesh_cols)
    target_row, target_col = divmod(target, hardware.mesh_cols)
    route = [source]
    while col != target_col:
        col += 1 if target_col > col else -1
        route.append(row * hardware.mesh_cols + col)
    while row != target_row:
        row += 1 if target_row > row else -1
        route.append(row * hardware.mesh_cols + col)
    return route


def _paired_transfer(
    *,
    config: WscLlmTraceConfig,
    builders: dict[int, OnlineTraceBuilder],
    queue_index: int,
    category: int,
    name: str,
    source_group: WscLlmInferenceGroup,
    target_group: WscLlmInferenceGroup,
    total_bytes: int,
    timer_gates: Optional[dict[int, Optional[int]]] = None,
) -> list[dict[str, object]]:
    if len(source_group.ranks) != len(target_group.ranks):
        raise ValueError("WSC-LLM direct KV shard pairing requires equal TP degree")
    if source_group.name == target_group.name:
        if timer_gates is not None:
            for rank in target_group.ranks:
                builders[rank].arm_timer_gate(timer_gates.get(rank))
        return []

    # Full KV transfers retain the configured whole-head ownership.  An
    # arbitrary capacity-split remainder falls back to exact byte partitioning
    # so the total transfer volume is never rounded up.
    if total_bytes % config.num_heads == 0:
        bytes_per_head = total_bytes // config.num_heads
        bytes_by_relative_rank = [
            bytes_per_head
            * shard_extent(config.num_heads, len(source_group.ranks), index)
            for index in range(len(source_group.ranks))
        ]
    else:
        bytes_by_relative_rank = [
            shard_extent(total_bytes, len(source_group.ranks), index)
            for index in range(len(source_group.ranks))
        ]
    routes: list[dict[str, object]] = []
    for relative_index, (source, target) in enumerate(
        zip(source_group.ranks, target_group.ranks)
    ):
        # 死锁真修（方案①，2026-10-11）：send/recv 中立化——不占主链
        # frontier；门与前驱改显式 arm；recv 节点 id 随 route 落账
        # （消费方显式边用：WL join 标记 C5=arm(xfer_recv)，FL 同构）。
        send_gate = None
        if timer_gates is not None:
            send_gate = timer_gates.get(source)
        tag = _stage_tag(queue_index, category, relative_index)
        _frontier_neutral(
            builders[source],
            lambda source=source, target=target, tag=tag:
                builders[source].comm_send(
                    f"{name}_send_rank{source}_to_rank{target}",
                    src=source,
                    dst=target,
                    comm_size=bytes_by_relative_rank[relative_index],
                    comm_tag=tag),
            send_gate)
        recv_node_id = _frontier_neutral(
            builders[target],
            lambda source=source, target=target, tag=tag:
                builders[target].comm_recv(
                    f"{name}_recv_rank{source}_to_rank{target}",
                    src=source,
                    dst=target,
                    comm_size=bytes_by_relative_rank[relative_index],
                    comm_tag=tag))
        path = _xy_route(config.hardware, source, target)
        routes.append(
            {
                "relative_shard": relative_index,
                "source_rank": source,
                "target_rank": target,
                "bytes": bytes_by_relative_rank[relative_index],
                "noc_path": path,
                "noc_hops": len(path) - 1,
                "recv_node_id": recv_node_id,
            }
        )
    return routes


def _emit_prefill_stage(
    *,
    config: WscLlmTraceConfig,
    builders: dict[int, OnlineTraceBuilder],
    group: WscLlmInferenceGroup,
    prefix: str,
    stage: str,
    tokens: int,
    initial_context_tokens: int,
    die_ctx=None,
) -> dict[int, tuple[int, int]]:
    """Emit one chunked prefill stage; returns, per rank, the ids of the first
    and last *real* operator/collective nodes (the artificial one-byte
    ``*_end_barrier`` collectives are excluded), or an empty mapping when the
    stage has no tokens.  Used only for metrics boundary recording (doc
    sec.6.2/6.3); the emitted nodes are unchanged.

    分 chunk 聚合发射(trace chunking)是合法物理机制,完整保留;首 chunk
    层段拆分真流水(B3,只为 PARTIAL 部分恢复服务)已随该恢复路径删除。
    token_expanded 逐 token 分支同步删除:唯一生产调用链
    (online/graph_batch_builder)在进入本函数前已对
    trace_granularity != "request_aggregated" fail-closed,聚合路径即
    唯一路径。
    """

    if tokens == 0:
        return {}
    bounds: dict[int, list[int]] = {}
    spans: list[tuple[int, int]] = []
    processed = 0
    while processed < tokens:
        chunk = min(config.prefill_chunk_size, tokens - processed)
        spans.append((chunk, initial_context_tokens + processed + chunk))
        processed += chunk
    tp = len(group.ranks)
    for relative_rank, rank in enumerate(group.ranks):
        rank_die_ctx = None
        if die_ctx is not None:
            # die 后端(阶段 C):prefill 体发射语境(phase=prefill,span
            # 记录带绝对位置——本函数的 chunk span 满足恒等式
            # kv == (initial_context + processed) + chunk)。
            die_ctx.begin_emission(
                phase="prefill",
                tensor_parallel=tp,
                tensor_parallel_rank=relative_rank,
                span_namespace=f"{prefix}_{stage}",
                pass_spans=spans,
            )
            rank_die_ctx = die_ctx
        first_node_id = builders[rank].next_id
        pass_count = transformer_pass_aggregated(
            builders[rank], phase=f"{prefix}_{stage}_request_aggregated",
            pass_spans=spans, layers=config.layers, hidden_size=config.hidden_size,
            ffn_size=config.ffn_size, tensor_parallel=tp, pg_name=group.pg_name,
            vocab_size=config.vocab_size, bytes_per_elem=config.bytes_per_elem,
            num_heads=config.num_heads, tensor_parallel_rank=relative_rank,
            mlp_variant=config.mlp_variant,
            die_ctx=rank_die_ctx,
        )
        bounds[rank] = [first_node_id, builders[rank].previous_id]
        builders[rank].all_reduce(
            f"{prefix}_{stage}_chunks_aggregated_end_barrier", pass_count, group.pg_name
        )
    return {rank: (pair[0], pair[1]) for rank, pair in bounds.items()}


def _validate_transfer_shard(
    config: WscLlmTraceConfig,
    transfer: KVTransfer,
    shard: KVTransferShard,
) -> None:
    """KV 转移 shard 发射前校验(照抄 sh_2.0（按函数名对应）;边缘成员资格按
    本仓 hardware 的网格边界 rank 集合判定——B2 构造期 nearest_edge
    与此同源)。"""
    if shard.bytes <= 0:
        raise ValueError("emitted KV transfer shards must contain positive bytes")
    if (
        shard.layer_start != transfer.layer_start
        or shard.layer_end != transfer.layer_end
    ):
        raise ValueError("KV transfer shard layer range does not match its action")
    if transfer.kind == "noc_migrate":
        if shard.source_rank is None or shard.target_rank is None:
            raise ValueError("NoC migration shard requires source and target ranks")
        if shard.edge_rank is not None:
            raise ValueError("NoC migration shard must not define an edge rank")
    elif transfer.kind == "remote_store":
        if (
            shard.source_rank is None
            or shard.target_rank is None
            or shard.edge_rank is None
            or shard.target_rank != shard.edge_rank
        ):
            raise ValueError("remote store shard must route source to its edge rank")
    elif transfer.kind == "remote_load":
        if (
            shard.source_rank is None
            or shard.target_rank is None
            or shard.edge_rank is None
            or shard.source_rank != shard.edge_rank
        ):
            raise ValueError("remote load shard must route its edge rank to target")

    if shard.edge_rank is not None and shard.edge_rank not in (
            physical_edge_ranks(config.hardware)):
        raise ValueError(
            f"rank {shard.edge_rank} is not a configured remote-memory edge"
        )
    if shard.noc_path:
        if shard.source_rank is None or shard.target_rank is None:
            raise ValueError("NoC path requires source and target ranks")
        if (
            shard.noc_path[0] != shard.source_rank
            or shard.noc_path[-1] != shard.target_rank
        ):
            raise ValueError("KV transfer NoC path endpoints do not match its shard")


def _kv_transfer_dict(
    transfer: KVTransfer,
    shard_records: Sequence[dict[str, object]],
) -> dict[str, object]:
    """发射侧 KVTransfer 记录(照抄 sh_2.0（按函数名对应）;含发射产生的
    tag/依赖元数据 shard_records)。"""
    return {
        "kind": transfer.kind,
        "phase": transfer.phase,
        "reason": transfer.reason,
        "session_id": transfer.session_id,
        "trigger_request_id": transfer.trigger_request_id,
        "source_instance_index": transfer.source_instance_index,
        "target_instance_index": transfer.target_instance_index,
        "total_bytes": transfer.total_bytes,
        "model_layers": transfer.model_layers,
        "layer_start": transfer.layer_start,
        "layer_end": transfer.layer_end,
        "resident_prefix_layers_before": (
            transfer.resident_prefix_layers_before
        ),
        "resident_prefix_layers_after": transfer.resident_prefix_layers_after,
        "shards": list(shard_records),
    }


def _history_control(
    *,
    group_by_index: dict[int, WscLlmInferenceGroup],
    pending_gate: PendingHistoryGate,
    relative_index: int,
) -> tuple[int, Optional[int]]:
    """history 门控制通道:pending 门源实例的 relative_index rank 与其
    timer gate(照抄 sh_2.0（按函数名对应）)。"""
    source_group = group_by_index[pending_gate.source_instance_index]
    if relative_index >= len(source_group.ranks):
        raise ValueError("history control rank index exceeds source TP degree")
    return source_group.ranks[relative_index], pending_gate.timer_gates[relative_index]


def _emit_transfer_trigger(
    *,
    builders: dict[int, OnlineTraceBuilder],
    group_by_index: dict[int, WscLlmInferenceGroup],
    tag_allocator: TransferTagAllocator,
    trigger_gate: Optional[TransferTriggerGate],
    relative_index: int,
    source_rank: int,
    action_name: str,
    shard_index: int,
) -> dict[str, object]:
    """逐出动作的触发门发射(照抄 sh_2.0（按函数名对应）):control rank 与
    数据源 rank 同 rank → 直接 arm 节点门;跨 rank → 1B p2p 触发
    (trigger 按字节正常计费,量级可忽略)。

    死锁真修（方案①，2026-10-11）：trigger send/recv 为簿记握手节点，
    改经 _frontier_neutral 中立化——不占主链 frontier、排序改显式边
    （send 段显式 arm trigger 门；跨 rank 时数据源侧的数据 send 由调用方
    显式 arm trigger_recv_node_id）。门在 control==source 时不在此
    arm，由返回的 trigger_dependency_node_id 交给调用方传入数据 send
    的显式 arm（pending 门不可留在主链等下一节点——中立化会暂存恢复）。"""
    if trigger_gate is None:
        return {}
    control_group = group_by_index[trigger_gate.control_instance_index]
    if len(control_group.ranks) != len(trigger_gate.node_gates):
        raise ValueError("KV transfer trigger gate does not match its TP degree")
    if relative_index >= len(control_group.ranks):
        raise ValueError("KV transfer trigger rank index exceeds its TP degree")
    control_rank = control_group.ranks[relative_index]
    node_gate = trigger_gate.node_gates[relative_index]
    if control_rank == source_rank:
        return {
            "trigger_control_rank": control_rank,
            "trigger_dependency_node_id": node_gate,
            "trigger_tag": None,
        }

    trigger_tag = tag_allocator.take()
    _frontier_neutral(
        builders[control_rank],
        lambda: builders[control_rank].comm_send(
            f"{action_name}_shard{shard_index}_trigger_to_rank{source_rank}",
            src=control_rank,
            dst=source_rank,
            comm_size=1,
            comm_tag=trigger_tag),
        node_gate)
    trigger_recv_id = _frontier_neutral(
        builders[source_rank],
        lambda: builders[source_rank].comm_recv(
            f"{action_name}_shard{shard_index}_trigger_from_rank{control_rank}",
            src=control_rank,
            dst=source_rank,
            comm_size=1,
            comm_tag=trigger_tag))
    return {
        "trigger_control_rank": control_rank,
        "trigger_dependency_node_id": node_gate,
        "trigger_tag": trigger_tag,
        "trigger_recv_node_id": trigger_recv_id,
    }


def _emit_kv_transfer(
    *,
    config: WscLlmTraceConfig,
    builders: dict[int, OnlineTraceBuilder],
    group_by_index: dict[int, WscLlmInferenceGroup],
    tag_allocator: TransferTagAllocator,
    transfer: KVTransfer,
    action_name: str,
    pending_gate: Optional[PendingHistoryGate] = None,
    trigger_gate: Optional[TransferTriggerGate] = None,
    extra_arm_ids: Optional[dict] = None,
) -> dict[str, object]:
    """发射一个 KV 转移动作(照抄 sh_2.0（按函数名对应）,适配本仓 config/组
    类型;HBM 计费三险逐节点核对见策略文档 §9 地图):

    extra_arm_ids（死锁真修 方案①,2026-10-11）:store→restore 补偿边的
    显式传参形态 {rank: [节点 id]};remote_load 链在对应 rank 的首节点
    显式 arm（数据面中立化后主链 pending 不再被传输链消费）。

    - local_hit:无数据 shard,仅按 pending 门 arm 到达/interval gates;
    - noc_migrate:shard send/recv(两端默认 charged)+ 1B ack 对(源端
      收 ack 后才物理释放);
    - remote_store 链 A(source≠edge):源 comm_send(源端 COMM_READ
      唯一数据计费)→ edge comm_recv(hbm_charge=False 过路零计费)→
      edge mem_store(池写,本地零计费)→ 1B ack 双端各 1B;
    - remote_store 链 B(source==edge 直连):mem_store(hbm_access_mode=1)
      = POOL_READ 唯一计费 + 池端口流体计时双异步 join(2026-09-24 起
      边缘端口为并发在途＋流平分流体端口,旧严格 FIFO 串行队列已删);
    - remote_load:1B arm/trigger → edge mem_load(仅池端口流体计时)→
      edge comm_send(hbm_charge=False)→ target comm_recv(hbm_charge=False,
      目标写由 restore 承担)→ target local_hbm_kv_restore(RESTORE
      唯一数据计费)+ code-7 anchor 观测不计费。

    计费单次性:每字节每链恰好一次本地 HBM 计费 + 一次池端口流体计时;
    禁止漏 hbm_charge=False / 直连漏 hbm_access_mode=1 / restore 退化
    普通 mem_load(契约 §12.5 红线)。"""
    if transfer.kind == "local_hit":
        if transfer.shards:
            raise ValueError("local-hit transfer must not contain data shards")
        if pending_gate is not None:
            if transfer.target_instance_index is None:
                raise ValueError("history local hit requires a target instance")
            target_group = group_by_index[transfer.target_instance_index]
            if len(target_group.ranks) != len(pending_gate.timer_gates):
                raise ValueError("history local-hit TP degree does not match its gate")
            for relative_index, target_rank in enumerate(target_group.ranks):
                control_rank, timer_gate = _history_control(
                    group_by_index=group_by_index,
                    pending_gate=pending_gate,
                    relative_index=relative_index,
                )
                if control_rank != target_rank:
                    raise ValueError(
                        "local history hit control rank does not match target rank"
                    )
                builders[target_rank].arm_timer_gate(timer_gate)
        return _kv_transfer_dict(transfer, ())

    if sum(shard.bytes for shard in transfer.shards) != transfer.total_bytes:
        raise ValueError("KV transfer shards do not preserve the action byte total")

    source_group = (
        None
        if transfer.source_instance_index is None
        else group_by_index[transfer.source_instance_index]
    )
    target_group = (
        None
        if transfer.target_instance_index is None
        else group_by_index[transfer.target_instance_index]
    )
    shard_records: list[dict[str, object]] = []
    for shard_index, shard in enumerate(transfer.shards):
        _validate_transfer_shard(config, transfer, shard)
        record: dict[str, object] = {
            "shard_index": shard_index,
            "source_rank": shard.source_rank,
            "target_rank": shard.target_rank,
            "edge_rank": shard.edge_rank,
            "bytes": shard.bytes,
            "layer_start": shard.layer_start,
            "layer_end": shard.layer_end,
            "noc_path": list(shard.noc_path),
            "noc_hops": max(0, len(shard.noc_path) - 1),
        }

        if transfer.kind == "noc_migrate":
            assert shard.source_rank is not None
            assert shard.target_rank is not None
            if source_group is None or target_group is None:
                raise ValueError("NoC migration requires source and target instances")
            relative_index = source_group.ranks.index(shard.source_rank)
            if target_group.ranks[relative_index] != shard.target_rank:
                raise ValueError("NoC migration does not preserve relative TP rank")
            history_gate = None
            if pending_gate is not None:
                control_rank, timer_gate = _history_control(
                    group_by_index=group_by_index,
                    pending_gate=pending_gate,
                    relative_index=relative_index,
                )
                if control_rank != shard.source_rank:
                    raise ValueError(
                        "history NoC migration control rank is not its source rank"
                    )
                # 门不留在主链 pending：随 data send 进中立段显式 arm（T1）。
                history_gate = timer_gate
            data_tag = tag_allocator.take()
            # 死锁真修（方案①，2026-10-11）：数据面对（簿记排序之外的
            # 传输节点）中立化——不占主链 frontier，门/序改显式边；
            # 数据到达 recv 是历史就绪的唯一载体，其节点 id 记入
            # history_arrival_node_ids 供 tp_ready_barrier 显式 arm（C4）。
            _frontier_neutral(
                builders[shard.source_rank],
                lambda: builders[shard.source_rank].comm_send(
                    f"{action_name}_shard{shard_index}_send",
                    src=shard.source_rank,
                    dst=shard.target_rank,
                    comm_size=shard.bytes,
                    comm_tag=data_tag),
                history_gate)
            recv_kv_id = _frontier_neutral(
                builders[shard.target_rank],
                lambda: builders[shard.target_rank].comm_recv(
                    # 名字含 kv 子串 = C++ code-7 memory anchor 契约
                    # (main_online.cc 按名字匹配"每 (request,rank) 最后一个 kv
                    # 节点"):目标侧数据到达节点即本迁移的 transfer-arrival
                    # 锚点(py-slo-E1 修复:锚点经节点名后缀由 C++ 采集器
                    # 直接消费,无需额外观测接线)。
                    f"{action_name}_shard{shard_index}_recv_kv",
                    src=shard.source_rank,
                    dst=shard.target_rank,
                    comm_size=shard.bytes,
                    comm_tag=data_tag))
            ack_tag = tag_allocator.take()
            # 2026-10-11 死锁真修(裁决8 第二轮,run5 census 六边环):
            # ack 对是簿记节点(仓内无消费者),裸挂主链 frontier 会与集合
            # 通信/gpu_comm 槽闭成跨 rank 等待环。改为中立段:ack send
            # 显式 arm data recv(排序语义=链位边不变),ack recv 纯叶;
            # 双方主链 frontier 不动。
            _frontier_neutral(
                builders[shard.target_rank],
                lambda: builders[shard.target_rank].comm_send(
                    f"{action_name}_shard{shard_index}_ack_to_rank{shard.source_rank}",
                    src=shard.target_rank,
                    dst=shard.source_rank,
                    comm_size=1,
                    comm_tag=ack_tag),
                recv_kv_id)
            _frontier_neutral(
                builders[shard.source_rank],
                lambda: builders[shard.source_rank].comm_recv(
                    f"{action_name}_shard{shard_index}_ack_from_rank{shard.target_rank}",
                    src=shard.target_rank,
                    dst=shard.source_rank,
                    comm_size=1,
                    comm_tag=ack_tag))
            # C3 写后不读登记(2026-09-30 审查):本 record 的 tag 类字段
            # (data_tag/ack_tag/request_tag/trigger_tag 等)与依赖描述字段
            # 在仓内无生产消费者(决策日志走调度器 _kv_transfer_dict,不含
            # tag;graph_batch_builder._register_store_tails 只读 kind/
            # session_id/shards)——保留为发射审计/对拍诊断面,非死码清理
            # 对象(与 capacity_result.insufficient_ranks 的登记同款)。
            record.update(
                {
                    "data_tag": data_tag,
                    "ack_tag": ack_tag,
                    "source_release_dependency": "noc_migration_ack_recv",
                    "history_arrival_node_ids": {
                        shard.target_rank: recv_kv_id},
                }
            )

        elif transfer.kind == "remote_store":
            assert shard.source_rank is not None
            assert shard.edge_rank is not None
            source_rank = shard.source_rank
            edge_rank = shard.edge_rank
            if source_group is None:
                raise ValueError("remote store requires a source instance")
            relative_index = source_group.ranks.index(source_rank)
            trigger_info = _emit_transfer_trigger(
                builders=builders,
                group_by_index=group_by_index,
                tag_allocator=tag_allocator,
                trigger_gate=trigger_gate,
                relative_index=relative_index,
                source_rank=source_rank,
                action_name=action_name,
                shard_index=shard_index,
            )
            record.update(trigger_info)
            # 死锁真修（方案①，2026-10-11）：跨 rank trigger 时数据 send
            # 显式 arm trigger recv；同 rank 时显式 arm trigger 门。
            data_gate = trigger_info.get("trigger_recv_node_id")
            if data_gate is None:
                data_gate = trigger_info.get("trigger_dependency_node_id")
            if source_rank == edge_rank:
                # 池存储端点即边缘rank(单 mem_store 路径):边缘rank从本地
                # HBM 读出该分片再经 SerDes 端口写入池——HBM 读计费
                # (hbm_access_mode=1,bytes=tensor_size),MEM 完成=join(端口
                # 事务, HBM 作业)。
                edge_store_node_id = _frontier_neutral(
                    builders[edge_rank],
                    lambda: builders[edge_rank].mem_store(
                        f"{action_name}_shard{shard_index}_edge_store",
                        shard.bytes,
                        hbm_access_mode=1),
                    data_gate)
                # 逐出旁路支链尾部观测(2026-09-13):直连路径的池写完成
                # 节点即源端释放节点。中立化后 previous_id 不再前进,
                # id 改取中立段尾节点(同一节点,取法变更)。
                record.update(
                    {
                        "direct_edge_access": True,
                        "source_release_dependency": "mem_store_completion",
                        "edge_store_node_id": edge_store_node_id,
                    }
                )
            else:
                data_tag = tag_allocator.take()
                ack_tag = tag_allocator.take()
                # 源rank的 comm_send 自动计费(发送端 HBM 读);数据经 NoC
                # 到达边缘rank后由 SerDes 直通写池——边缘rank的 comm_recv
                # 与 mem_store 均为直通,不占其本地 HBM(hbm_charge=False /
                # 不标注)。
                _frontier_neutral(
                    builders[source_rank],
                    lambda: builders[source_rank].comm_send(
                        f"{action_name}_shard{shard_index}_send_to_edge{edge_rank}",
                        src=source_rank,
                        dst=edge_rank,
                        comm_size=shard.bytes,
                        comm_tag=data_tag),
                    data_gate)
                data_recv_id = _frontier_neutral(
                    builders[edge_rank],
                    lambda: builders[edge_rank].comm_recv(
                        f"{action_name}_shard{shard_index}_recv_from_rank{source_rank}",
                        src=source_rank,
                        dst=edge_rank,
                        comm_size=shard.bytes,
                        comm_tag=data_tag,
                        hbm_charge=False))
                edge_store_node_id = _frontier_neutral(
                    builders[edge_rank],
                    lambda: builders[edge_rank].mem_store(
                        f"{action_name}_shard{shard_index}_remote_store",
                        shard.bytes),
                    data_recv_id)
                # 逐出旁路支链尾部观测(2026-09-13):池写完成节点(消费方
                # 唯一:builder 登记 pending_store_tails 的 store→restore
                # 前递依赖);取法改中立段尾节点。
                # 2026-10-11 死锁真修(裁决8 第二轮):ack 对簿记节点改
                # 中立段(显式 arm mem_store,排序语义=链位边不变),
                # 双方主链 frontier 不再被 ack 推进。
                _frontier_neutral(
                    builders[edge_rank],
                    lambda: builders[edge_rank].comm_send(
                        f"{action_name}_shard{shard_index}_ack_to_rank{source_rank}",
                        src=edge_rank,
                        dst=source_rank,
                        comm_size=1,
                        comm_tag=ack_tag),
                    edge_store_node_id)
                _frontier_neutral(
                    builders[source_rank],
                    lambda: builders[source_rank].comm_recv(
                        f"{action_name}_shard{shard_index}_ack_from_edge{edge_rank}",
                        src=edge_rank,
                        dst=source_rank,
                        comm_size=1,
                        comm_tag=ack_tag))
                # 源端 ack recv 节点 id 此前曾落盘 shard 行,全仓零读者
                # (2026-09-25 审计删除);ack 发射链本体不变。
                record.update(
                    {
                        "direct_edge_access": False,
                        "data_tag": data_tag,
                        "ack_tag": ack_tag,
                        "source_release_dependency": "remote_store_ack_recv",
                        "edge_store_node_id": edge_store_node_id,
                    }
                )

        elif transfer.kind == "remote_load":
            assert shard.target_rank is not None
            assert shard.edge_rank is not None
            if target_group is None or pending_gate is None:
                raise ValueError("history remote load requires target and timer gate")
            target_rank = shard.target_rank
            edge_rank = shard.edge_rank
            relative_index = target_group.ranks.index(target_rank)
            control_rank, timer_gate = _history_control(
                group_by_index=group_by_index,
                pending_gate=pending_gate,
                relative_index=relative_index,
            )
            # 死锁真修（方案①，2026-10-11）：remote_load 数据面全链中立化
            # （request 握手/mem_load/data/restore），门与前驱改显式 arm；
            # restore 节点 id 仍经 target_hbm_completion_node_id 落账（C3）。
            request_tag: Optional[int] = None
            pool_read_gate = None
            # store→restore 补偿边的显式形态（方案①）：本 rank 的 arm
            # 表由 remote_load 链首节点（request_recv 或 mem_load）消费。
            restore_arms = tuple((extra_arm_ids or {}).get(edge_rank) or ())
            if control_rank == edge_rank:
                pool_read_gate = timer_gate
            else:
                request_tag = tag_allocator.take()
                _frontier_neutral(
                    builders[control_rank],
                    lambda: builders[control_rank].comm_send(
                        f"{action_name}_shard{shard_index}_request_to_edge{edge_rank}",
                        src=control_rank,
                        dst=edge_rank,
                        comm_size=1,
                        comm_tag=request_tag),
                    timer_gate)
                request_recv_id = _frontier_neutral(
                    builders[edge_rank],
                    lambda: builders[edge_rank].comm_recv(
                        f"{action_name}_shard{shard_index}_request_from_rank{control_rank}",
                        src=control_rank,
                        dst=edge_rank,
                        comm_size=1,
                        comm_tag=request_tag),
                    *restore_arms)
                pool_read_gate = request_recv_id
                restore_arms = ()
            mem_load_id = _frontier_neutral(
                builders[edge_rank],
                lambda: builders[edge_rank].mem_load(
                    f"{action_name}_shard{shard_index}_remote_load",
                    shard.bytes),
                *((pool_read_gate,)
                  if pool_read_gate is not None else ()),
                *restore_arms)
            data_tag: Optional[int] = None
            restore_gate = mem_load_id
            if edge_rank != target_rank:
                data_tag = tag_allocator.take()
                # 边缘rank的 mem_load(SerDes->NoC 直通)与 comm_send
                # (NoC 发射端直通)均不占其本地 HBM;目标rank的 comm_recv
                # 也不计费——其 HBM 写由紧随的 restore 节点(串行语义)承担,
                # 每字节流恰好计费一次。
                _frontier_neutral(
                    builders[edge_rank],
                    lambda: builders[edge_rank].comm_send(
                        f"{action_name}_shard{shard_index}_send_to_rank{target_rank}",
                        src=edge_rank,
                        dst=target_rank,
                        comm_size=shard.bytes,
                        comm_tag=data_tag,
                        hbm_charge=False),
                    mem_load_id)
                data_recv_id = _frontier_neutral(
                    builders[target_rank],
                    lambda: builders[target_rank].comm_recv(
                        f"{action_name}_shard{shard_index}_recv_from_edge{edge_rank}",
                        src=edge_rank,
                        dst=target_rank,
                        comm_size=shard.bytes,
                        comm_tag=data_tag,
                        hbm_charge=False))
                restore_gate = data_recv_id
            target_hbm_completion_node_id = _frontier_neutral(
                builders[target_rank],
                lambda: builders[target_rank].local_hbm_kv_restore(
                    # 名字含 kv 子串 = code-7 anchor 契约:restore(HBM 写完成)
                    # 节点是 REMOTE_RESTORE 的还原完成锚点(py-slo-E1 修复;
                    # WP5 三段分解的 restore 窗口据此取真值)。
                    f"{action_name}_shard{shard_index}_target_hbm_write_kv",
                    shard.bytes),
                restore_gate)
            if target_hbm_completion_node_id is None:
                raise RuntimeError("target HBM restore did not generate a node")
            record.update(
                {
                    "control_rank": control_rank,
                    "request_tag": request_tag,
                    "data_tag": data_tag,
                    "direct_edge_access": edge_rank == target_rank,
                    "target_hbm_completion_node_id": (
                        target_hbm_completion_node_id
                    ),
                    "target_hbm_bandwidth_policy": (
                        "n_way_equal_split_with_all_active_hbm_users"
                    ),
                    "control_dependencies": [
                        "previous_decode_interval_gate",
                        "control_rank_prior_operations_including_remote_store_ack",
                    ],
                    "history_arrival_node_ids": {
                        target_rank: target_hbm_completion_node_id},
                }
            )

        shard_records.append(record)

    return _kv_transfer_dict(transfer, shard_records)


def main(argv=None) -> None:  # noqa: ARG001
    """fail-closed 拒绝桩(2026-08-18 起生效):离线静态全管线入口已删除。

    本模块保留的仅是③④在线路径只读 import 的符号(config 装载/发射辅助/
    估算函数)。③④ 的输入物化入口是
    plan_materializer.py(manifest/metrics/runtime_config);
    静态 ET 生成入口不再存在。
    """
    raise SystemExit(
        "path-1 (offline static full pipeline) was removed on 2026-08-18; "
        "use plan_materializer.py for the online routes' plan-dir inputs")


if __name__ == "__main__":
    main()
