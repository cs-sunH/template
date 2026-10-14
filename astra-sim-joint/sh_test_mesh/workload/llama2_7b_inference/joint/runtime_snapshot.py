"""runtime_snapshot.py -- W1 因果一致性快照（schema ``snapshot-v1-20260929``）。

《joint仓库当下+历史修改方案》§4.1 DecisionSnapshot 最小数据契约的
Python 侧落地；版本常量从 ``joint.schema_registry`` 引用（W0 冻结）。

设计要点（盘点文档 inv_w1.md 的 §4.2/§4.4——基底仓外部证据文档，本仓未随附）：

* **白名单投影**（§4.2 白名单投影条）：后端可以知道真实 trace 长度和未来事件，
  但导出预测视图必须经白名单投影——只导出已实际提交/暴露的工作，不
  透传未来 decode 终点（``decode_length``/``final_context_tokens``）、
  真实返回间隔（``inter_request_interval_ns``）或未到达队列全表。
  投影在**快照构造处**执行（Python 侧白名单，而非依赖 C++ 裁剪——
  双保险：C++ 侧 ``work_ledger`` 状态本身不含这些字段）；每次过滤记入
  ``projection_log``（记字段名与条数，不记值——日志本身不得成为泄漏面）。
* **V1 因果验证锚**：改隐藏后缀长度（decode_length / CSV 后继行）不改
  快照——快照的每个字节都只由因果可见输入派生。
* **resource_state 四分列**（W1 任务原文）：仲裁状态（active set + 每
  成员剩余量 + 当前份额）≠ 吞吐观测（窗口差分遥测）≠ slot gate
  （HardwareResource 在途计数）≠ 物理容量（bytes/带宽标量）。四者来
  源与语义不同，混用即漂移（§5.1a）。
* **版本失效校验接口**（W3 消费）：``cache_key()`` 覆盖全部会改变结果
  的版本（runtime/layout/reservation/history/hardware/model/ledger），
  失效则刷新，失败不准沿用刚才的容量承诺（§4.2:164）。
* ``existing_unknown_work``（§4.1 ExistingUnknownWork）：已到达请求身份、
  已暴露进度、删失标记——标记为估计而非真实承诺。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

from joint.schema_registry import SNAPSHOT_SCHEMA_VERSION

#: arrivals[] 白名单（§4.4 投影点表第 1 行）：只取因果可见字段。
ARRIVAL_WHITELIST: Tuple[str, ...] = (
    "request_id", "session_id", "turn_index", "prefill_length",
    "arrival_world_ns", "ingress_seq", "queue_index")

#: 全域禁止字段（改它们不得改变快照——V1 因果锚）。
FORBIDDEN_FIELDS: Tuple[str, ...] = (
    "decode_length",               # 未来 decode 终点（真实 trace 长度）
    "final_context_tokens",        # 同上（manifest 侧字段名）
    "inter_request_interval_ns",   # 后继真实返回间隔（执行真值）
    "future_arrivals",             # 未到达队列全表
    "future_alarms",               # 预排定的真实返回 tick
)

#: resource_state 四列（W1 任务原文"区分吞吐观测、仲裁状态、slot gate
#: 和物理容量"）。
RESOURCE_STATE_COLUMNS: Tuple[str, ...] = (
    "arbitration", "throughput_observation", "slot_gate", "physical_capacity")

#: 版本面键（§4.1 DecisionSnapshot.versions；directory 为 W4 预留位）。
VERSION_KEYS: Tuple[str, ...] = (
    "runtime", "layout", "reservation", "history", "hardware", "model",
    "ledger")


class SnapshotError(ValueError):
    """fail-closed 快照构造/校验错误（禁止字段泄漏/版本面残缺）。"""


# ---------------------------------------------------------------- 投影 --


def project_arrivals(arrivals: Any) -> Tuple[List[Dict[str, Any]], List[str]]:
    """arrivals[] 白名单投影；返回 (投影后列表, projection_log 条目)。

    log 条目只记字段名与条数（"arrivals[].decode_length filtered (N
    entries)"），不记被过滤的值——投影日志自身不得成为泄漏面。
    """
    if arrivals is None:
        return [], []
    if not isinstance(arrivals, list):
        raise SnapshotError("arrivals must be a list")
    projected: List[Dict[str, Any]] = []
    filtered: Dict[str, int] = {}
    for arrival in arrivals:
        if not isinstance(arrival, Mapping):
            raise SnapshotError("arrival entries must be mappings")
        row: Dict[str, Any] = {}
        for key in ARRIVAL_WHITELIST:
            if key in arrival:
                row[key] = arrival[key]
        for key in arrival:
            if key in FORBIDDEN_FIELDS:
                filtered[key] = filtered.get(key, 0) + 1
        projected.append(row)
    log = [
        "arrivals[].{} filtered ({} entries)".format(key, count)
        for key, count in sorted(filtered.items())
    ]
    return projected, log


def project_mapping(
    payload: Any, *, context: str,
) -> Tuple[Dict[str, Any], List[str]]:
    """递归剔除禁止字段（请求队列冻结视图等调用方数据的防御性投影）。"""
    if payload is None:
        return {}, []
    if isinstance(payload, Mapping):
        out: Dict[str, Any] = {}
        log: List[str] = []
        for key, value in payload.items():
            if key in FORBIDDEN_FIELDS:
                log.append("{}.{} filtered".format(context, key))
                continue
            child, child_log = project_mapping(
                value, context="{}.{}".format(context, key))
            out[key] = child
            log.extend(child_log)
        return out, log
    if isinstance(payload, (list, tuple)):
        items = []
        log: List[str] = []
        for index, item in enumerate(payload):
            child, child_log = project_mapping(
                item, context="{}[{}]".format(context, index))
            items.append(child)
            log.extend(child_log)
        return (items if isinstance(payload, list) else tuple(items)), log
    return payload, []


# ---------------------------------------------------------------- 快照 --


@dataclass(frozen=True)
class DecisionSnapshot:
    """§4.1 DecisionSnapshot：决策边界的只读因果背景结构（§4.4）。"""

    snapshot_id: str
    now_ns: int
    delivery_sequence: int
    event_watermark: int
    versions: Mapping[str, Any]
    observed_tick: int
    observation_lag_ns: int
    arrivals: Tuple[Mapping[str, Any], ...]
    request_queues: Mapping[str, Any]
    work_index: Mapping[str, Any]
    resource_state: Mapping[str, Any]
    existing_unknown_work: Mapping[str, Any]
    capacity_view: Optional[Mapping[str, Any]]
    projection_log: Tuple[str, ...]
    schema_version: str = SNAPSHOT_SCHEMA_VERSION
    # Runtime-only immutable ledger rows/indexes. Kept outside to_dict() so
    # the frozen snapshot wire representation stays schema-compatible.
    ledger_projection: Any = field(default=None, repr=False, compare=False)
    pipeline_ledger_state: Any = field(default=None, repr=False, compare=False)
    cache_dimensions: Tuple[Any, ...] = field(
        default=(), repr=False, compare=False)

    def to_json(self) -> str:
        """规范化序列化（V1 因果锚：改隐藏后缀长度本串逐字节不变）。"""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "snapshot_id": self.snapshot_id,
            "now_ns": self.now_ns,
            "delivery_sequence": self.delivery_sequence,
            "event_watermark": self.event_watermark,
            "versions": dict(self.versions),
            "observed_tick": self.observed_tick,
            "observation_lag_ns": self.observation_lag_ns,
            "arrivals": [dict(row) for row in self.arrivals],
            "request_queues": dict(self.request_queues),
            "work_index": dict(self.work_index),
            "resource_state": dict(self.resource_state),
            "existing_unknown_work": dict(self.existing_unknown_work),
            "capacity_view": (
                dict(self.capacity_view) if self.capacity_view is not None
                else None),
            "projection_log": list(self.projection_log),
        }

    # ------------------------------------------------------------ 版本 --

    def cache_key(self) -> Tuple[str, ...]:
        """版本失效校验接口（§4.2:164：快照复制、预测和选中提交通过
        版本校验连接，失效则刷新）。键序冻结 = VERSION_KEYS 顺序。"""
        result = []
        for key in VERSION_KEYS:
            value = _hashable(self.versions.get(key))
            if key == "ledger":
                # Preserve the frozen seven-key version surface while
                # invalidating on time and resource/directory/capacity epochs.
                value = (value, int(self.now_ns),
                         _hashable(self.cache_dimensions))
            result.append((key, value))
        return tuple(result)

    def is_valid_against(self, other: "DecisionSnapshot") -> bool:
        """对端（如 W3 结算层）持的版本面与本快照是否仍然一致。"""
        return self.cache_key() == other.cache_key()


def _hashable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(sorted((k, _hashable(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(item) for item in value)
    return value


def build_work_index(
    ledger_state: Mapping[str, Any],
) -> Dict[str, Any]:
    """按 resource_leg 分组的只读工作索引（§4.1:145 "只读资源索引"）。

    输入是 C++ ``work_ledger.state``（或全量装载后的等价结构），输出按
    资源腿分组：每链路 active flow 集、每 HBM rank 活跃 job 集、每 pool
    port streaming/queued 集、每 rank 在途节点集。节点视图与资源腿视图
    按 (work_id, leg) 分键——同一物理腿不重复计费。
    """
    index: Dict[str, Any] = {
        "by_link": {},
        "by_hbm_rank": {},
        "by_pool_port": {},
        "by_rank_nodes": {},
    }
    for rank_block in ledger_state.get("nodes", ()):
        rank = int(rank_block.get("rank", 0))
        index["by_rank_nodes"][rank] = [
            int(node.get("node_id", 0))
            for node in rank_block.get("nodes", ())]
    noc = ledger_state.get("noc") or {}
    for flow in noc.get("flows", ()):
        # A completed NoC flow remains in FluidScheduler during its
        # propagation tail, but its link memberships were already removed.
        # Keep the runtime index aligned with link.active_flow_count and only
        # index the active arbitration set.
        if flow.get("state", "Active") not in ("Active", 0, "0"):
            continue
        for link in flow.get("route", ()):
            index["by_link"].setdefault(int(link), []).append(
                int(flow.get("flow_id", 0)))
    for hbm_block in ledger_state.get("hbm", ()):
        rank = int(hbm_block.get("rank", 0))
        index["by_hbm_rank"][rank] = [
            {
                "node_id": int(job.get("node_id", 0)),
                "kind": job.get("kind"),
                "remaining_bytes": job.get("remaining_bytes"),
                "remaining_ops": job.get("remaining_ops"),
                # C++ names this residual remaining_latency_ns.  Keep the
                # old internal alias for consumers written before the wire
                # spelling was aligned, but derive both from the same field.
                "remaining_latency_ns": job.get(
                    "remaining_latency_ns", job.get("memory_latency_ns")),
                "memory_latency_ns": job.get(
                    "remaining_latency_ns", job.get("memory_latency_ns")),
            }
            for job in hbm_block.get("jobs", ())]
    for port_block in ledger_state.get("pool", ()):
        port = int(port_block.get("port_index", 0))
        index["by_pool_port"][port] = [
            {
                "issue_seq": int(job.get("issue_seq", 0)),
                "rank": int(job.get("rank", job.get("sys_id", 0))),
                "sys_id": int(job.get("sys_id", job.get("rank", 0))),
                "node_id": int(job.get("node_id", 0)),
                "bytes": job.get("bytes"),
                "remaining_bytes": job.get("remaining_bytes"),
                "state": job.get("state"),
            }
            for job in port_block.get("jobs", ())]
    return index


def build_resource_state(
    delta: Mapping[str, Any],
    ledger_state: Mapping[str, Any],
) -> Dict[str, Any]:
    """resource_state 四分列（W1 任务原文：区分四类资源状态语义）。

    * arbitration：仲裁状态——active set + 每成员剩余量（来自账本状态，
      执行真值；NoC 仲裁单位是 active flow，HBM 是 active byte-streaming
      job，池端口是 latency 已结束的 streaming job，§4.1:154）。
    * throughput_observation：吞吐观测——相邻 delivery epoch 的积分平均
      （delta.link_telemetry[]，可选通道；缺席 = None，不补零——§4.2:168
      "缺必要输入应显式拒绝该模式或进入已登记降级"）。
    * slot_gate：发射门——HardwareResource 五类在途计数（drain/release
      检测用；不构成有限容量或带宽分母，§4.1:154）。
    * physical_capacity：物理容量——链路带宽/HBM 带宽/池端口带宽标量。
    """
    arbitration: Dict[str, Any] = {
        "noc_links": {},
        "hbm_ranks": {},
        "pool_ports": {},
    }
    noc = ledger_state.get("noc") or {}
    for link in noc.get("links", ()):
        arbitration["noc_links"][int(link.get("link_id", 0))] = {
            "active_flow_count": int(link.get("active_flow_count", 0)),
            "capacity_Bpns": link.get("capacity_Bpns"),
        }
    for hbm_block in ledger_state.get("hbm", ()):
        jobs = hbm_block.get("jobs", ())
        streaming = [
            job for job in jobs
            if not job.get(
                    "remaining_latency_ns",
                    job.get("memory_latency_ns"))]
        arbitration["hbm_ranks"][int(hbm_block.get("rank", 0))] = {
            "active_jobs": len(jobs),
            "streaming_jobs": len(streaming),
            "as_of_tick": hbm_block.get("as_of_tick"),
        }
    for port_block in ledger_state.get("pool", ()):
        arbitration["pool_ports"][int(port_block.get("port_index", 0))] = {
            "latency_waiting": int(port_block.get("latency_waiting", 0)),
            "streaming": int(port_block.get("streaming", 0)),
            "completion_waiting": int(
                port_block.get("completion_waiting", 0)),
            "as_of_ns": port_block.get("as_of_ns"),
        }
    telemetry = delta.get("link_telemetry")
    throughput = (
        [dict(sample) for sample in telemetry] if telemetry is not None
        else None)
    slot_gate = {
        int(block.get("rank", 0)): {
            key: int(block.get(key, 0)) for key in (
                "cpu", "gpu_comp", "gpu_comm", "hbm_dma", "remote_mem")
        }
        for block in ledger_state.get("slots", ())
    }
    capacity: Dict[str, Any] = {
        "noc_link_Bpns": {
            int(link.get("link_id", 0)): link.get("capacity_Bpns")
            for link in noc.get("links", ())
        },
    }
    for hbm_block in ledger_state.get("hbm", ()):
        if "bandwidth_Bpns" in hbm_block:
            capacity.setdefault(
                "hbm_Bpns", {})[int(hbm_block.get("rank", 0))] = \
                hbm_block.get("bandwidth_Bpns")
    for port_block in ledger_state.get("pool", ()):
        if "bandwidth_Bpns" in port_block:
            capacity.setdefault(
                "pool_Bpns", {})[int(port_block.get("port_index", 0))] = \
                port_block.get("bandwidth_Bpns")
    return {
        "arbitration": arbitration,
        "throughput_observation": throughput,
        "slot_gate": slot_gate,
        "physical_capacity": capacity,
    }


def build_existing_unknown_work(
    request_queues: Mapping[str, Any],
    arrivals: Tuple[Mapping[str, Any], ...],
) -> Dict[str, Any]:
    """ExistingUnknownWork（§4.1:147）：已到达请求的删失观测。

    每活跃请求：``exposed_progress``（已消费 decode 步数等已暴露进度）、
    ``censoring`` 恒 True（未结束证据）、``source`` 恒 "estimate"——标记
    为估计而非真实承诺（不得读取 decode_length 构造"精确剩余"）。
    """
    unknown: Dict[str, Any] = {}
    exposed_by_queue = request_queues.get("exposed_decode_steps") or {}
    for arrival in arrivals:
        request_id = arrival.get("request_id")
        if request_id is None:
            continue
        unknown[request_id] = {
            "exposed_progress": int(exposed_by_queue.get(request_id, 0)),
            "censoring": True,
            "source": "estimate",
        }
    for request_id, steps in exposed_by_queue.items():
        unknown.setdefault(
            request_id,
            {"exposed_progress": int(steps), "censoring": True,
             "source": "estimate"})
    return unknown


def build_decision_snapshot(
    delta: Mapping[str, Any],
    *,
    ledger_versions: Optional[Mapping[str, Any]] = None,
    ledger_state: Optional[Mapping[str, Any]] = None,
    event_watermark: int = 0,
    versions: Optional[Mapping[str, Any]] = None,
    request_queues: Optional[Mapping[str, Any]] = None,
    observation_lag_ns: int = 0,
    capacity_view: Optional[Mapping[str, Any]] = None,
    now_ns: Optional[int] = None,
    ledger_projection: Any = None,
    resource_version_vector: Optional[Tuple[Tuple[str, int], ...]] = None,
    directory_version: Any = None,
    capacity_version: Any = None,
    pipeline_ledger_state: Any = None,
    pricing_only: bool = False,
) -> DecisionSnapshot:
    """从 delivery delta + 账本状态构造 DecisionSnapshot（唯一入口）。

    * ``delta``：桥 request JSON（含可选 additive 字段 ``work_ledger``；
      缺席 = 旧对端——快照的 work_index/resource_state 为空骨架并记入
      projection_log，不补零冒充观测）。显式离线调用可传
      ``ledger_state`` 构造完整只读索引；在线计价传 ``pricing_only=True``，
      不从 full/delta envelope 重建整份账本，当前资源前沿保存在运行时
      ``ledger_projection`` 上。
    * ``versions``：调用方提供的版本面（runtime/layout/reservation/
      history/hardware/model）；``ledger`` 维度由账本版本填充。
    * ``request_queues``：冻结队列视图（调用方构造；仍经防御性投影）。
    * ``capacity_view``：§4.1 CapacityView（逐 rank 容量账 + 摘要版本/
      误差来源）——None 时 projection_log 登记"容量面缺席"（无生产
      者不虚构零容量）。
    """
    if not isinstance(delta, Mapping):
        raise SnapshotError("delta must be a mapping")
    delivery_sequence = int(delta["delivery_sequence"])
    tick = int(delta["tick"])
    snapshot_id = "{}:{}".format(delivery_sequence, event_watermark)
    arrivals, arrival_log = project_arrivals(delta.get("arrivals"))
    queues, queue_log = project_mapping(
        request_queues or {}, context="request_queues")
    ledger_payload = delta.get("work_ledger")
    delta_frame_without_state = False
    if pricing_only:
        # The online pricing caller supplies a compact runtime-only frontier
        # below. Do not materialize a full cumulative WorkLedger export into
        # the serialized decision snapshot on every delivery.
        ledger_state = {}
    elif ledger_state is None:
        if (isinstance(ledger_payload, Mapping)
                and ledger_payload.get("snapshot_kind") == "full"):
            ledger_state = ledger_payload.get("state")
        else:
            ledger_state = {}
            # delta 帧不携带 state（桥接协议：首交付 full、其后恒 delta）
            # 且调用方未提供累积账本状态——work_index/resource_state 空面
            # 必须显式登记降级，不得静默空缺（§4.1:144 最小数据契约 +
            # §4.2:168 "缺必要输入应显式拒绝该模式或进入已登记降级"）。
            if isinstance(ledger_payload, Mapping):
                delta_frame_without_state = True
    if isinstance(ledger_payload, Mapping):
        watermark = int(ledger_payload.get("event_watermark", 0))
        if watermark > event_watermark:
            event_watermark = watermark
    projection_log = list(arrival_log) + list(queue_log)
    if pricing_only:
        projection_log.append(
            "hot pricing snapshot omits full work_index/resource_state; "
            "current pricing rows remain available on runtime ledger_projection")
    elif not isinstance(ledger_payload, Mapping):
        projection_log.append(
            "work_ledger field absent (legacy peer); work_index/resource_state "
            "left empty rather than fabricated")
    elif delta_frame_without_state:
        projection_log.append(
            "work_ledger delta frame carries events only and no "
            "ledger_state was provided; work_index/resource_state left "
            "empty rather than fabricated")
    version_map: Dict[str, Any] = {key: 0 for key in VERSION_KEYS}
    if versions:
        unknown = set(versions) - set(VERSION_KEYS)
        if unknown:
            raise SnapshotError(
                "unknown version keys {!r} (known: {})".format(
                    sorted(unknown), list(VERSION_KEYS)))
        version_map.update(versions)
    if ledger_versions:
        version_map["ledger"] = dict(ledger_versions)
    resource_state = build_resource_state(delta, ledger_state)
    work_index = build_work_index(ledger_state)
    unknown_work = build_existing_unknown_work(queues, tuple(arrivals))
    if capacity_view is None:
        # CapacityView 无生产者注册——容量面缺席显式登记（§4.1:149 逐
        # rank 容量账分量（容量/权重/KV 驻留/预约）与逐出摘要不得静默
        # None 冒充空观测），不虚构零容量。
        projection_log.append(
            "capacity_view absent (no producer registered); per-rank "
            "capacity face unavailable rather than fabricated")
    projected_capacity, cap_log = project_mapping(
        capacity_view or {}, context="capacity_view")
    projection_log.extend(cap_log)
    return DecisionSnapshot(
        snapshot_id=snapshot_id,
        now_ns=int(now_ns if now_ns is not None else tick),
        delivery_sequence=delivery_sequence,
        event_watermark=event_watermark,
        versions=version_map,
        observed_tick=tick,
        observation_lag_ns=int(observation_lag_ns),
        arrivals=tuple(arrivals),
        request_queues=queues,
        work_index=work_index,
        resource_state=resource_state,
        existing_unknown_work=unknown_work,
        capacity_view=projected_capacity or None,
        projection_log=tuple(projection_log),
        ledger_projection=ledger_projection,
        pipeline_ledger_state=pipeline_ledger_state,
        cache_dimensions=(
            tuple(resource_version_vector or ()),
            directory_version,
            capacity_version,
            (pipeline_ledger_state.get("state_version")
             if isinstance(pipeline_ledger_state, Mapping) else None),
        ),
    )
