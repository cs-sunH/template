#!/usr/bin/env python3
"""graph_batch_builder.py -- sh_2.0 在线 GraphBatch 构图器（方案 §4 步骤 1-8 操作 4）。

阶段 1 最关键的对齐点：复用共享发射原语的节点结构。per-request 发射
（generate_face_trace.py 模块级函数）由共享
助手函数组成（_emit_kv_transfer / _emit_tp_readiness_barrier /
_emit_tp_point_to_point_readiness_barrier / transformer_pass_aggregated）——本模块
直接 import 它们（红线：只读 import），用 OnlineTraceBuilder（与 TraceBuilder
同构的在线侧 builder）驱动，保证：

  - 节点属性、插入顺序、rank ownership 跨 request 链结构一致（节点级审计口径）；
  - per-rank 节点 id 跨批次全局递增，保持运行时 GraphBatch 的连续 ID 契约；
  - interval gate 的 after_node_id 指向上一 request 的 decode end barrier
    节点 id（completion 段账本，跨批次解析）。

在线语义差异（刻意，注释标注；与蓝本 wscllm 同构）：
  - timer gate 始终发射节点（结构保留）但 runtime_ns=0：到达时间由 C++ 的
    arrival alarm（future_alarms）替代，gate 不再等待——保持时长会双重等待；
  - 每 request 拆三段发射：ARRIVAL 边界发射 prefill 整段（gates/history
    逐出/恢复/prefill），PREFILL_DRAIN 边界发射 decode 整段（decode 逐出/
    prefill→decode 迁移/decode/end barrier），REQUEST_COMPLETE 边界发射
    completion 逐出 + 下一 turn 的 interval gates；三段共享同一 per-rank ID
    序列；
  - watch 锚点遵循保留的调度语义：PREFILL_DRAIN = 每 rank 末个真实
    prefill 节点（prefill_last_node_by_rank，排除 end barrier；含 PARTIAL
    恢复流水 suffix 尾段——suffix restore 节点先于 prefill 算子发射，其
    完成被 first-chunk-suffix 的 arm_dependency 依赖覆盖，合同④边界映射）；
    DECODE_COMPLETION = end barrier 前每 rank 的 decode 末节点
    （decode_last_node_by_rank，含迁移 transfer 尾）；
  - strategy 模式保持物理跨 request 链。
"""

import os
import sys

# --------------------------------------------------------------------------
# import 路径：本文件位于 workload/llama2_7b_inference/online/，共享发射与
# 配置模块在上一级。路径只做 import 用途（红线：generate_face_trace.py /
# face_scheduler.py 只读 import 与注释）。
# --------------------------------------------------------------------------
_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _path in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from face_scheduler import KVTransfer, KVTransferShard  # noqa: E402
from generate_face_trace import (  # noqa: E402
    PendingHistoryGate,
    TransferTagAllocator,
    TransferTriggerGate,
    _emit_kv_transfer,
    _emit_tp_point_to_point_readiness_barrier,
    _emit_tp_readiness_barrier,
    sanitize_node_prefix,
)
from generate_trace import (  # noqa: E402
    ALL_REDUCE,
    COMM_COLL_NODE,
    COMM_RECV_NODE,
    COMM_SEND_NODE,
    COMP_NODE,
    MEM_LOAD_NODE,
    MEM_STORE_NODE,
    transformer_pass_aggregated,
)


class OnlineTraceBuilder:
    """与共享 TraceBuilder 接口同构的在线侧每-rank builder。

    同一接口面：timer_gate / arm_timer_gate / arm_dependency /
    chain_checkpoint / restore_chain / mem_store / mem_load /
    local_hbm_kv_restore / comp / all_reduce / comm_send / comm_recv /
    next_id / previous_id / node_count——共享助手函数（_emit_kv_transfer 等）
    可直接驱动。与 TraceBuilder 的差异：节点发射为 GraphBatch nodes[] dict
    （而非 ChakraNode 对象），依赖记录为 parent_edges[]（共享 builder 的
    data_deps 内联在节点里，在线按边列表携带）；timer_gate 忽略 duration
    （runtime_ns=0，alarm 替代等待）。

    per-rank id 自 0 起全局递增，跨批次保持连续。
    """

    def __init__(self, rank: int, *, remote_operand_loads: bool):
        self.rank = rank
        self.remote_operand_loads = remote_operand_loads
        self.next_id = 0
        self.previous_id = None
        self.pending_extra_dependencies = []
        self.nodes = []   # 本 rank 全部已发射节点 dict（发射序）
        self.edges = []   # 本 rank 全部 parent edges {"rank","from","to","kind"}
        self.node_count = 0
        # 当前 request-stage 反向索引上下文（每次 per-request 段发射前设置）。
        self.request_id = ""
        self.stage = ""
        self.generation = 0

    def set_context(self, request_id: str, stage: str, generation: int) -> None:
        self.request_id = request_id
        self.stage = stage
        self.generation = generation

    # ------------------------------------------------------------- 节点发射 --

    def _new_node(self, name: str, node_type: int, *, is_cpu_op: bool = False,
                  is_timer_op: bool = False) -> dict:
        node = {
            "rank": self.rank,
            "id": self.next_id,
            "name": name,
            "type": node_type,
            "is_cpu_op": is_cpu_op,
            "is_timer_op": is_timer_op,
            "inputs_values": "",
            "request_id": self.request_id,
            "stage": self.stage,
            "generation": self.generation,
            "compute": {"num_ops": 0, "tensor_size": 0, "runtime_ns": 0},
            "comm": {"bytes": 0, "src": 0, "dst": 0, "tag": 0},
            "coll": {"comm_type": 0, "bytes": 0, "priority": 0,
                     "pg_name": "", "involved_dim": []},
        }
        dependency_ids = []
        if self.previous_id is not None:
            dependency_ids.append(self.previous_id)
        dependency_ids.extend(self.pending_extra_dependencies)
        for dependency_id in dict.fromkeys(dependency_ids):
            self.edges.append({
                "rank": self.rank,
                "from": dependency_id,
                "to": self.next_id,
                "kind": "data",
            })
        self.pending_extra_dependencies.clear()
        self.previous_id = self.next_id
        self.next_id += 1
        self.nodes.append(node)
        self.node_count += 1
        return node

    @staticmethod
    def _uint64(value: int) -> int:
        return max(1, int(value))

    def arm_timer_gate(self, timer_node_id) -> None:
        if timer_node_id is not None:
            self.pending_extra_dependencies.append(int(timer_node_id))

    def arm_dependency(self, node_id) -> None:
        if node_id is not None:
            self.pending_extra_dependencies.append(int(node_id))

    def chain_checkpoint(self):
        return self.previous_id, tuple(self.pending_extra_dependencies)

    def restore_chain(self, checkpoint) -> None:
        self.previous_id, dependencies = checkpoint
        self.pending_extra_dependencies = list(dependencies)

    def timer_gate(self, name: str, duration_ns: int, *, after_node_id=None):
        """在线 timer gate：发射节点（runtime_ns=0，is_timer_op）。

        与共享 TraceBuilder.timer_gate 完全同构：
        - duration == 0 → 直接返回 after_node_id，不发射节点
            （共享接口同款；interval==0 的 request 不产生 interval gate
            节点）；
          - 否则直接创建节点（不经 _new_node）——不链 previous_id、不消费
            pending_extra_dependencies、不更新 previous_id；仅 after_node_id
            依赖（interval gate 依赖上一 request 完成 barrier）。
        调度参考语义（duration = 到达时刻/interval，gate 等待）由 C++ arrival
        alarm（future_alarms）替代——gate 只保留结构与依赖，保持时长会双重
        等待（蓝本步骤 1-8 设计分析，同样适用本仓）。
        """
        if duration_ns < 0 or duration_ns % 1000 != 0:
            raise ValueError(
                "timer duration must be a non-negative whole number of microseconds"
            )
        if duration_ns == 0:
            return after_node_id
        node = {
            "rank": self.rank,
            "id": self.next_id,
            "name": name,
            "type": COMP_NODE,
            "is_cpu_op": True,
            "is_timer_op": True,
            "inputs_values": "",
            "request_id": self.request_id,
            "stage": self.stage,
            "generation": self.generation,
            "compute": {"num_ops": 0, "tensor_size": 0, "runtime_ns": 0},
            "comm": {"bytes": 0, "src": 0, "dst": 0, "tag": 0},
            "coll": {"comm_type": 0, "bytes": 0, "priority": 0,
                     "pg_name": "", "involved_dim": []},
        }
        if after_node_id is not None:
            self.edges.append({
                "rank": self.rank,
                "from": int(after_node_id),
                "to": self.next_id,
                "kind": "data",
            })
        self.next_id += 1
        self.nodes.append(node)
        self.node_count += 1
        return node["id"]

    def mem_store(self, name: str, tensor_size: int, hbm_access_mode: int = 0) -> None:
        node = self._new_node(name, MEM_STORE_NODE)
        node["compute"]["tensor_size"] = self._uint64(tensor_size)
        if hbm_access_mode:
            # sh_2.0 N-way HBM contention pool endpoint charging（与离线
            # TraceBuilder.mem_store 语义 parity：absent = 无本地 HBM 访问；
            # 在线 snake 键驻留 compute 段，与离线 kebab ET attr 同义）。
            node["compute"]["hbm_access_mode"] = int(hbm_access_mode)

    def mem_load(self, name: str, tensor_size: int, hbm_access_mode: int = 0) -> None:
        node = self._new_node(name, MEM_LOAD_NODE)
        node["compute"]["tensor_size"] = self._uint64(tensor_size)
        if hbm_access_mode:
            node["compute"]["hbm_access_mode"] = int(hbm_access_mode)

    def local_hbm_kv_restore(self, name: str, tensor_size: int) -> None:
        """目标 HBM DMA 写（可与推理计算重叠）——sh_2.0 特有路由位。"""
        node = self._new_node(name, MEM_LOAD_NODE)
        node["compute"]["tensor_size"] = self._uint64(tensor_size)
        node["is_local_hbm_kv_restore"] = True

    def comp(self, name: str, num_ops: int, tensor_size: int,
             remote_read_size: int = 0) -> None:
        node = self._new_node(name, COMP_NODE)
        node["compute"]["num_ops"] = self._uint64(num_ops)
        node["compute"]["tensor_size"] = self._uint64(tensor_size)
        if self.remote_operand_loads and remote_read_size:
            node["compute"]["remote_weight_bytes"] = self._uint64(remote_read_size)

    def all_reduce(self, name: str, comm_size: int, pg_name: str) -> None:
        node = self._new_node(name, COMM_COLL_NODE)
        node["coll"]["comm_type"] = ALL_REDUCE
        node["coll"]["bytes"] = self._uint64(comm_size)
        node["coll"]["priority"] = 0
        node["coll"]["pg_name"] = pg_name
        node["coll"]["involved_dim"] = [True, True]

    def comm_send(self, name: str, *, src: int, dst: int, comm_size: int,
                  comm_tag: int, hbm_charge: bool = True) -> None:
        node = self._new_node(name, COMM_SEND_NODE)
        node["comm"]["src"] = int(src)
        node["comm"]["dst"] = int(dst)
        node["comm"]["bytes"] = self._uint64(comm_size)
        node["comm"]["tag"] = int(comm_tag)
        if not hbm_charge:
            # sh_2.0 N-way HBM contention：直通流量不建本端 HBM 作业（默认
            # true=发送端 HBM 读，absent = charged；与离线
            # TraceBuilder.comm_send 语义 parity，在线 snake 键与离线
            # kebab ET attr 同义）。
            node["comm"]["hbm_charge"] = False

    def comm_recv(self, name: str, *, src: int, dst: int, comm_size: int,
                  comm_tag: int, hbm_charge: bool = True) -> None:
        node = self._new_node(name, COMM_RECV_NODE)
        node["comm"]["src"] = int(src)
        node["comm"]["dst"] = int(dst)
        node["comm"]["bytes"] = self._uint64(comm_size)
        node["comm"]["tag"] = int(comm_tag)
        if not hbm_charge:
            node["comm"]["hbm_charge"] = False

    # ------------------------------------------------------------- 只读属性 --

    @property
    def node_count_total(self) -> int:
        return self.node_count


class _HistorySnapshot:
    """history_location_before 的轻量重建（离线为 SessionKVSnapshot；在线
    从 decision_log 的 completion 记录 + partial transfer 层界重建）。"""

    def __init__(self, location: str, instance_index, resident_prefix_layers: int):
        self.location = location
        self.instance_index = instance_index
        self.resident_prefix_layers = resident_prefix_layers


def kv_transfer_from_log(record) -> KVTransfer:
    """从 decision_log 的 KVTransfer dict 重建 KVTransfer 对象。

    字段集 = generate_face_trace._kv_transfer_log_dict 的输出（含 shards）。
    strategy 模式传入的已是 KVTransfer 对象（face_scheduler 只读复用），
    原样返回。
    """
    if isinstance(record, KVTransfer):
        return record
    return KVTransfer(
        kind=record["kind"],
        phase=record["phase"],
        reason=record["reason"],
        session_id=record["session_id"],
        trigger_request_id=record["trigger_request_id"],
        source_instance_index=record["source_instance_index"],
        target_instance_index=record["target_instance_index"],
        total_bytes=record["total_bytes"],
        shards=tuple(
            KVTransferShard(
                source_rank=shard["source_rank"],
                target_rank=shard["target_rank"],
                edge_rank=shard["edge_rank"],
                bytes=shard["bytes"],
                noc_path=tuple(shard["noc_path"]),
                layer_start=shard["layer_start"],
                layer_end=shard["layer_end"],
            )
            for shard in record.get("shards", ())
        ),
        model_layers=record["model_layers"],
        layer_start=record["layer_start"],
        layer_end=record["layer_end"],
        resident_prefix_layers_before=record["resident_prefix_layers_before"],
        resident_prefix_layers_after=record["resident_prefix_layers_after"],
    )


def history_snapshot_from_log(plan: dict):
    """重建 history_location_before：location 取 prefill 决策上下文记录的
    history_location_before（决策日志已含）；resident 层数 = partial
    history_transfer 的 layer_start（共享调度语义要求两者相等）。
    strategy 模式的 plan["history_location_before"] 直接是
    SessionKVSnapshot 对象（字段同名），原样返回。"""
    snapshot = plan.get("history_location_before")
    if snapshot is not None and not isinstance(snapshot, dict):
        return snapshot
    location = plan.get("history_location_before_location")
    if location is None:
        return None
    transfer = plan.get("history_transfer")
    resident = (
        transfer["layer_start"]
        if transfer is not None and transfer["kind"] == "remote_load"
        and location == "partial_hbm_remote"
        else plan.get("history_resident_prefix_layers", 0)
    )
    return _HistorySnapshot(
        location=location,
        instance_index=plan.get("history_location_before_instance_index"),
        resident_prefix_layers=int(resident),
    )


class GraphBatchBuilder:
    """在线构图器：持有 per-rank OnlineTraceBuilder（状态跨批次），按决策
    边界发射 prefill 整段 / decode 整段 / completion 段，并维护
    pending_history 与 completion gate 账本（共享调度语义同构）。"""

    def __init__(self, config, *, digest_sink=None):
        self.config = config
        # strategy 保持物理跨 request 链。
        self.builders = {
            rank: OnlineTraceBuilder(
                rank, remote_operand_loads=config.remote_operand_loads)
            for rank in range(config.npus_count)
        }
        self.group_by_index = dict(enumerate(config.inference_groups))
        # 跨请求 history/completion gate 账本：
        self.pending_history = {}          # request_id -> PendingHistoryGate
        self.pending_request_by_session = {}
        self.deferred_session_locations = {}
        self._prefill_block_ends = {}       # request_id -> {rank: prev_id|None}
        self._prefill_completion_nodes = {}  # request_id -> {rank: node_id}
        self._decode_completion_nodes = {}   # request_id -> {rank: node_id}
        self._tag_allocator = TransferTagAllocator()
        # 每 request 的 action_sequence 账本（action 名含 _action{seq:03d}_，
        # 跨该 request 的全部 transfer 递增）——在线三段发射共享同一计数器，保证节点
        # 名跨 request 逐字节稳定（canonical 命名 key）。
        self._action_sequence = {}
        self.batch = None

    # ------------------------------------------------------------- 批次 --

    def begin_batch(self) -> None:
        self.batch = {
            "nodes": [],
            "parent_edges": [],
            "watches": [],
            "assignments": [],
            "kv_actions": [],
            "future_alarms": [],
        }

    def _collect(self, marker: dict) -> None:
        for rank, builder in self.builders.items():
            node_mark, edge_mark = marker[rank]
            self.batch["nodes"].extend(builder.nodes[node_mark:])
            self.batch["parent_edges"].extend(builder.edges[edge_mark:])

    def _mark(self) -> dict:
        return {
            rank: (len(builder.nodes), len(builder.edges))
            for rank, builder in self.builders.items()
        }

    def _set_context(self, request_plan: dict, stage: str,
                     generation: int) -> None:
        request_id = request_plan["request_id"]
        for builder in self.builders.values():
            builder.set_context(request_id, stage, generation)

    # ------------------------------------------------------- 相位计时校准 --

    def emit_prefill_batch(self, request_plan: dict) -> dict:
        """发射 request 的 prefill 整段（ARRIVAL 决策的图）。

        strategy 恒走 roofline 物理时钟。

        返回 PREFILL_DRAIN watch 成员：{rank: 末个真实 prefill 节点 id}
        （离线 prefill_last_node_by_rank 口径，排除 end barrier）。
        """
        self._set_context(request_plan, "prefill", 0)
        marker = self._mark()
        members = self._emit_prefill(request_plan)
        self._collect(marker)
        return members

    def emit_decode_batch(self, request_plan: dict) -> dict:
        """发射 decode 整段（decode 逐出 + prefill→decode 迁移 + decode
        readiness barrier + decode 整段 + end barrier；PREFILL_DRAIN 决策）。

        返回 DECODE_COMPLETION watch 成员：{rank: end barrier 前的 decode
        末节点 id}（离线 decode_last_node_by_rank 口径）。
        """
        self._set_context(request_plan, "decode", 1)
        marker = self._mark()
        members = self._emit_decode(request_plan)
        self._collect(marker)
        return members

    def emit_completion_batch(self, request_plan: dict,
                              following_plan: dict = None) -> None:
        """发射 completion 逐出 + 下一 turn 的 interval gates
        （REQUEST_COMPLETE 决策；共享完成语义的在线实现）。"""
        # GraphBatch 校验的 stage/generation 闭集 = {prefill:0, decode:1}
        # （GraphBatchCommitter :174/:420）；completion 段节点（completion
        # 逐出与 interval gates）归 decode 阶段，generation 1。
        self._set_context(request_plan, "decode", 1)
        marker = self._mark()
        self._emit_completion(request_plan, following_plan)
        self._collect(marker)

    # --------------------------------------------------- per-request 发射 --

    def _mark_pending_history_store(self, transfer: KVTransfer) -> None:
        # 与共享 history-gate 账本一致。
        if transfer.resident_prefix_layers_after == 0:
            location = "remote_memory"
        elif transfer.resident_prefix_layers_after < transfer.model_layers:
            location = "partial_hbm_remote"
        else:
            raise RuntimeError("remote store did not reduce resident KV layers")
        session_id = transfer.session_id
        pending_request_id = self.pending_request_by_session.get(session_id)
        if pending_request_id is None:
            self.deferred_session_locations[session_id] = location
            return
        gate = self.pending_history.get(pending_request_id)
        if gate is None:
            # Backport 2026-08-16 (对比报告 §5.2, verified in the 3-min
            # test copies): an in-flight turn-0 request has its session key
            # registered in pending_request_by_session at prefill emission
            # WITHOUT a pending_history gate (turn-0's gate is built inline
            # as "new_session" and never stored; turn>0 gates are popped at
            # arrival). When another request's completion eviction
            # (remote_store) targets this session, the old direct subscript
            # raised KeyError (20/50 档 delivery 3575/1553 两起历史事故,
            # sh_2.0 改造实录登记). Defer to
            # session completion instead -- mirroring the turn>0 no-entry
            # semantics above (offline planner equivalence: the eviction
            # applies to the session state; the next turn's
            # history_location_before reflects the post-eviction location).
            self.deferred_session_locations[session_id] = location
            return
        gate.location = location

    def _emit_prefill(self, request_plan: dict) -> dict:
        builders = self.builders
        config = self.config
        prefill_group = self.group_by_index[
            request_plan["prefill_instance_index"]]
        prefix = _prefix_of(request_plan)
        request = config.request_queue[request_plan["queue_index"]]

        # ---- 到达/history gate（turn-0 预置到达 timer；turn>0 消费跨请求
        #      pending gate）----
        if request_plan["turn_index"] == 0:
            arrival = request.session_arrival_time_ns
            if arrival is None:
                raise RuntimeError("first request lost its session arrival time")
            # turn-0 gate duration 先做 µs 下取整：admission_time_ns 是准入
            # 时刻的虚拟 tick（Roofline 任意 ns 粒度，如 153516157242），
            # timer_gate 的离线同构校验要求整 µs（duration_ns % 1000 != 0
            # 即 raise）。该 duration 不进节点（runtime_ns=0、不存储），仅
            # 驱动校验与 0 跳过，下取整对既有通过路径零影响；消除
            # 2026-08-22 30s 窗 delivery seq=1145 确定性崩溃（阻塞 turn-0
            # 在非 µs 对齐 tick 准入时触发）。
            duration = request_plan.get("admission_time_ns", arrival)
            duration -= duration % 1000
            short_prefix = (
                f"q{request_plan['queue_index']:04d}_"
                f"{sanitize_node_prefix(request_plan['request_id'])}"
            )
            timers = tuple(
                builders[rank].timer_gate(
                    f"{short_prefix}_global_arrival_timer_gate",
                    duration,
                )
                for rank in prefill_group.ranks
            )
            pending_gate = PendingHistoryGate(
                source_instance_index=request_plan["prefill_instance_index"],
                timer_gates=timers,
                location="new_session",
            )
        else:
            pending_gate = self.pending_history.pop(request_plan["request_id"], None)
            if pending_gate is None:
                raise RuntimeError(
                    f"request {request_plan['request_id']} has no arrival/history gate")
            # 消费该 request 的跨请求 history gate。
            self.pending_request_by_session.pop(request_plan["session_id"], None)

        history_before = history_snapshot_from_log(request_plan)
        request_plan["_history_before"] = history_before
        if history_before is not None and pending_gate.location not in (
                history_before.location, "new_session") and (
                history_before.location == "partial_hbm_remote"):
            # partial 恢复：pending gate 的 location 经 remote_store 折算后
            # 应与计划一致（离线 reconcile 同款校验）。
            if pending_gate.location != history_before.location:
                raise RuntimeError(
                    f"history gate location {pending_gate.location!r} does not "
                    f"match planned location {history_before.location!r}")

        def emit_transfer(transfer: KVTransfer, stage: str, *,
                          gate=None, trigger_gate=None) -> dict:
            action_name = (
                f"{prefix}_{stage}_action"
                f"{self._next_action_sequence(request_plan):03d}_"
                f"{sanitize_node_prefix(transfer.session_id)}_"
                f"{transfer.kind}"
            )
            record = _emit_kv_transfer(
                config=config,
                builders=builders,
                group_by_index=self.group_by_index,
                tag_allocator=self._tag_allocator,
                transfer=transfer,
                action_name=action_name,
                pending_gate=gate,
                trigger_gate=trigger_gate,
                transfer_anchor_sink=None,
            )
            if transfer.kind == "remote_store":
                self._mark_pending_history_store(transfer)
            return record

        # ---- history 逐出（trigger = 到达/history gate）----
        history_eviction_trigger = TransferTriggerGate(
            control_instance_index=pending_gate.source_instance_index,
            node_gates=pending_gate.timer_gates,
        )
        for transfer in request_plan["history_evictions"]:
            emit_transfer(kv_transfer_from_log(transfer), "history_evictions",
                          trigger_gate=history_eviction_trigger)

        partial_history_restore = (
            history_before is not None
            and history_before.location == "partial_hbm_remote"
        )
        suffix_ready_nodes_by_rank = {}

        history_transfer = request_plan.get("history_transfer")
        history_transfer = (
            kv_transfer_from_log(history_transfer)
            if history_transfer is not None else None
        )
        history_prefix_transfer = request_plan.get("history_prefix_transfer")
        history_prefix_transfer = (
            kv_transfer_from_log(history_prefix_transfer)
            if history_prefix_transfer is not None else None
        )

        if history_transfer is None:
            if request_plan["turn_index"] != 0:
                raise RuntimeError("later request is missing its history transfer action")
            if history_prefix_transfer is not None:
                raise RuntimeError("new history unexpectedly has a prefix migration")
            source_group = self.group_by_index[pending_gate.source_instance_index]
            if source_group.ranks != prefill_group.ranks:
                raise RuntimeError("first-request arrival gate is not on its Prefill ranks")
            for relative_index, rank in enumerate(prefill_group.ranks):
                builders[rank].arm_timer_gate(
                    pending_gate.timer_gates[relative_index])
        elif not partial_history_restore:
            if history_prefix_transfer is not None:
                raise RuntimeError("non-partial history has a prefix migration")
            if history_before is None:
                raise RuntimeError(
                    "history transfer is missing its location snapshot")
            if pending_gate.location != history_before.location:
                raise RuntimeError(
                    f"history gate location {pending_gate.location!r} does not "
                    f"match planned location {history_before.location!r}")
            emit_transfer(history_transfer, "history_transfer", gate=pending_gate)

        for transfer in request_plan["prefill_evictions"]:
            emit_transfer(kv_transfer_from_log(transfer), "prefill_evictions")

        # ---- PARTIAL 恢复流水 / 全量 readiness barrier ----
        if partial_history_restore:
            if history_transfer is None:
                raise RuntimeError("partial history is missing its suffix load")
            if (history_transfer.kind != "remote_load"
                    or history_transfer.layer_start != history_before.resident_prefix_layers
                    or history_transfer.layer_end != config.layers):
                raise RuntimeError("partial history load does not match its suffix")
            if history_before.instance_index is None:
                raise RuntimeError("partial history has no resident source instance")
            if history_prefix_transfer is None:
                if request_plan["prefill_instance_index"] != history_before.instance_index:
                    raise RuntimeError(
                        "cross-instance partial history is missing its prefix migration"
                    )
                for relative_index, rank in enumerate(prefill_group.ranks):
                    builders[rank].arm_timer_gate(
                        pending_gate.timer_gates[relative_index])
                _emit_tp_readiness_barrier(
                    builders=builders,
                    group=prefill_group,
                    name=f"{prefix}_prefill_resident_prefix_ready_barrier",
                )
                prefix_ready_nodes = tuple(
                    builders[rank].previous_id for rank in prefill_group.ranks
                )
            else:
                resident_prefix_layers = history_before.resident_prefix_layers
                if (
                    history_prefix_transfer.kind != "noc_migrate"
                    or history_prefix_transfer.phase != "history"
                    or history_prefix_transfer.reason
                    != "history_partial_prefix_migrate"
                    or history_prefix_transfer.source_instance_index
                    != history_before.instance_index
                    or history_prefix_transfer.target_instance_index
                    != request_plan["prefill_instance_index"]
                    or history_prefix_transfer.source_instance_index
                    == history_prefix_transfer.target_instance_index
                    or history_prefix_transfer.layer_start != 0
                    or history_prefix_transfer.layer_end != resident_prefix_layers
                    or history_prefix_transfer.resident_prefix_layers_before
                    != resident_prefix_layers
                    or history_prefix_transfer.resident_prefix_layers_after
                    != resident_prefix_layers
                ):
                    raise RuntimeError("partial history prefix migration is invalid")
                emit_transfer(
                    history_prefix_transfer,
                    "history_prefix_transfer",
                    gate=pending_gate,
                )
                prefix_readiness = _emit_tp_point_to_point_readiness_barrier(
                    builders=builders,
                    group=prefill_group,
                    tag_allocator=self._tag_allocator,
                    name=f"{prefix}_prefill_prefix_ready_barrier",
                )
                prefix_ready_nodes = tuple(
                    int(node_id)
                    for _, node_id in prefix_readiness["node_ids_by_rank"]
                )
            checkpoints = {
                rank: builders[rank].chain_checkpoint()
                for rank in prefill_group.ranks
            }
            branch_gate = PendingHistoryGate(
                source_instance_index=request_plan["prefill_instance_index"],
                timer_gates=prefix_ready_nodes,
                location=history_before.location,
            )
            history_record = emit_transfer(
                history_transfer,
                "history_transfer",
                gate=branch_gate,
            )
            for shard_record in history_record["shards"]:
                target_rank = shard_record.get("target_rank")
                completion_node = shard_record.get("target_hbm_completion_node_id")
                if not isinstance(target_rank, int) or not isinstance(completion_node, int):
                    raise RuntimeError("suffix restore is missing a target HBM gate")
                suffix_ready_nodes_by_rank[target_rank] = completion_node
            if set(suffix_ready_nodes_by_rank) != set(prefill_group.ranks):
                raise RuntimeError(
                    "suffix restore did not cover every Prefill rank; "
                    f"request={request_plan['request_id']}")
            suffix_readiness = _emit_tp_point_to_point_readiness_barrier(
                builders=builders,
                group=prefill_group,
                tag_allocator=self._tag_allocator,
                name=f"{prefix}_prefill_suffix_ready_barrier",
            )
            suffix_ready_nodes_by_rank = {
                int(rank): int(node_id)
                for rank, node_id in suffix_readiness["node_ids_by_rank"]
            }
            for rank in prefill_group.ranks:
                builders[rank].restore_chain(checkpoints[rank])
        else:
            _emit_tp_readiness_barrier(
                builders=builders,
                group=prefill_group,
                name=f"{prefix}_prefill_kv_ready_barrier",
            )

        # ---- prefill 主体（request_aggregated；token_expanded 在线 fail-closed）--
        if config.trace_granularity != "request_aggregated":
            raise RuntimeError(
                "online emission supports request_aggregated granularity only "
                f"(got {config.trace_granularity!r})")
        tensor_parallel = len(prefill_group.ranks)
        prefill_tokens_to_process = (
            request_plan["prefill_context_tokens"]
            - request_plan["history_tokens_before"]
        )
        if prefill_tokens_to_process <= 0:
            raise ValueError(
                f"request {request_plan['request_id']} has no Prefill work tokens"
            )
        prefill_spans = []
        processed = 0
        while processed < prefill_tokens_to_process:
            chunk_tokens = min(
                self.config_p_chunk(), prefill_tokens_to_process - processed)
            prefill_spans.append(
                (chunk_tokens,
                 request_plan["history_tokens_before"] + processed + chunk_tokens))
            processed += chunk_tokens
        prefill_last_node_by_rank = {}
        pass_count_total = 0
        for relative_rank, rank in enumerate(prefill_group.ranks):
            aggregate_arguments = {
                "layers": config.layers,
                "hidden_size": config.hidden_size,
                "ffn_size": config.ffn_size,
                "tensor_parallel": tensor_parallel,
                "pg_name": prefill_group.pg_name,
                "vocab_size": config.vocab_size,
                "bytes_per_elem": config.bytes_per_elem,
                "num_heads": config.num_heads,
                "tensor_parallel_rank": relative_rank,
                "mlp_variant": config.mlp_variant,
            }
            if partial_history_restore:
                suffix_start = history_before.resident_prefix_layers
                transformer_pass_aggregated(
                    builders[rank],
                    phase=f"{prefix}_prefill_first_chunk_prefix",
                    pass_spans=(prefill_spans[0],),
                    layer_start=0,
                    layer_end=suffix_start,
                    include_output=False,
                    **aggregate_arguments,
                )
                builders[rank].arm_dependency(suffix_ready_nodes_by_rank[rank])
                transformer_pass_aggregated(
                    builders[rank],
                    phase=f"{prefix}_prefill_first_chunk_suffix",
                    pass_spans=(prefill_spans[0],),
                    layer_start=suffix_start,
                    layer_end=config.layers,
                    include_output=True,
                    **aggregate_arguments,
                )
                if len(prefill_spans) > 1:
                    transformer_pass_aggregated(
                        builders[rank],
                        phase=f"{prefix}_prefill_remaining_chunks_aggregated",
                        pass_spans=prefill_spans[1:],
                        **aggregate_arguments,
                    )
                pass_count = len(prefill_spans)
            else:
                pass_count = transformer_pass_aggregated(
                    builders[rank],
                    phase=f"{prefix}_prefill_request_aggregated",
                    pass_spans=prefill_spans,
                    **aggregate_arguments,
                )
            pass_count_total = pass_count
            prefill_last_node_by_rank[rank] = builders[rank].previous_id
            builders[rank].all_reduce(
                f"{prefix}_prefill_chunks_aggregated_end_barrier",
                pass_count,
                prefill_group.pg_name,
            )

        # end barrier 之后的 previous_id = decode 逐出 trigger 的 node_gates。
        prefill_completion_nodes = {
            rank: builders[rank].previous_id for rank in prefill_group.ranks
        }
        if any(node_id is None for node_id in prefill_completion_nodes.values()):
            raise RuntimeError("Prefill completion node IDs were not generated")
        self._prefill_completion_nodes[request_plan["request_id"]] = (
            prefill_completion_nodes)
        # [previous_id 链修复，蓝本同款] 记录 per-rank prefill 块末 previous_id
        # （emitted-ranks-only 精确语义：decode 段发射前恢复；decode 组
        # rank 不在 prefill 组内即 None，与本 request 的 decode 决策无关，
        # 故此处无需 decode_instance_index——strategy 模式在 prefill 段
        # 发射时 decode 实例尚未决策）。
        self._prefill_block_ends[request_plan["request_id"]] = {
            rank: (builders[rank].previous_id if rank in prefill_group.ranks
                   else None)
            for rank in range(self.config.npus_count)
        }
        # turn>0 的 request 在本段发射后成为本 session 的 pending request
        # （remote_store 折算目标）；turn-0 在 gate 创建时已登记。
        if request_plan["turn_index"] == 0:
            self.pending_request_by_session[request_plan["session_id"]] = (
                request_plan["request_id"])
        return prefill_last_node_by_rank

    def _emit_decode(self, request_plan: dict) -> dict:
        builders = self.builders
        config = self.config
        prefill_group = self.group_by_index[
            request_plan["prefill_instance_index"]]
        decode_group = self.group_by_index[
            request_plan["decode_instance_index"]]
        prefix = _prefix_of(request_plan)
        request = config.request_queue[request_plan["queue_index"]]

        # [frontier 接续裁决，strategy 死锁修复 2026-08-16]
        # strategy 模式（真实物理）**不恢复**：per-rank previous_id 保持
        # 接续到当前 frontier（= 共享跨 request 物理链语义），
        # 使 per-rank 发行序 = 全局发射序——任意两个发射段在所有共享
        # rank 上的相对次序一致，跨实例 P2P（noc_migrate send/recv/ack）
        # 与 collective 参与序不可能反转（死锁机理见实录：sdg1 waits-for
        # 证据，rank0/rank6 槽位互持 + ack 反压成环）。decode 组 rank
        # 首节点因此链到该 rank 当前 frontier（跨 request 边），保持连续的
        # per-rank previous_id 链（差分归因类别②的既有口径）。
        prefill_completion_nodes = self._prefill_completion_nodes[
            request_plan["request_id"]]

        def emit_transfer(transfer: KVTransfer, stage: str, *,
                          gate=None, trigger_gate=None) -> dict:
            action_name = (
                f"{prefix}_{stage}_action"
                f"{self._next_action_sequence(request_plan):03d}_"
                f"{sanitize_node_prefix(transfer.session_id)}_"
                f"{transfer.kind}"
            )
            record = _emit_kv_transfer(
                config=config,
                builders=builders,
                group_by_index=self.group_by_index,
                tag_allocator=self._tag_allocator,
                transfer=transfer,
                action_name=action_name,
                pending_gate=gate,
                trigger_gate=trigger_gate,
                transfer_anchor_sink=None,
            )
            if transfer.kind == "remote_store":
                self._mark_pending_history_store(transfer)
            return record

        decode_eviction_trigger = TransferTriggerGate(
            control_instance_index=request_plan["prefill_instance_index"],
            node_gates=tuple(
                prefill_completion_nodes[rank] for rank in prefill_group.ranks),
        )
        for transfer in request_plan["decode_evictions"]:
            emit_transfer(kv_transfer_from_log(transfer), "decode_evictions",
                          trigger_gate=decode_eviction_trigger)
        prefill_decode_transfer = request_plan.get("prefill_decode_transfer")
        if prefill_decode_transfer is None:
            raise RuntimeError("request is missing its Prefill-to-Decode KV action")
        emit_transfer(kv_transfer_from_log(prefill_decode_transfer),
                      "prefill_decode_transfer")

        _emit_tp_readiness_barrier(
            builders=builders,
            group=decode_group,
            name=f"{prefix}_decode_kv_ready_barrier",
        )

        tensor_parallel = len(decode_group.ranks)
        decode_spans = tuple(
            (1, request_plan["prefill_context_tokens"] + step + 1)
            for step in range(request.decode_length)
        )
        decode_last_node_by_rank = {}
        for relative_rank, rank in enumerate(decode_group.ranks):
            transformer_pass_aggregated(
                builders[rank],
                phase=f"{prefix}_decode_request_aggregated",
                pass_spans=decode_spans,
                layers=config.layers,
                hidden_size=config.hidden_size,
                ffn_size=config.ffn_size,
                tensor_parallel=tensor_parallel,
                pg_name=decode_group.pg_name,
                vocab_size=config.vocab_size,
                bytes_per_elem=config.bytes_per_elem,
                num_heads=config.num_heads,
                tensor_parallel_rank=relative_rank,
                mlp_variant=config.mlp_variant,
            )
            # 离线 doc sec.6.4：completion candidate = end barrier 前每 rank
            # 的 decode 末节点（DECODE_COMPLETION watch 成员）。
            decode_last_node_by_rank[rank] = builders[rank].previous_id
            builders[rank].all_reduce(
                f"{prefix}_decode_request_end_barrier",
                1,
                decode_group.pg_name,
            )
        decode_completion_nodes = {
            rank: builders[rank].previous_id for rank in decode_group.ranks
        }
        if any(node_id is None for node_id in decode_completion_nodes.values()):
            raise RuntimeError("Decode completion node IDs were not generated")
        self._decode_completion_nodes[request_plan["request_id"]] = (
            decode_completion_nodes)
        return decode_last_node_by_rank

    def _emit_completion(self, request_plan: dict,
                         following_plan: dict) -> None:
        builders = self.builders
        decode_group = self.group_by_index[
            request_plan["decode_instance_index"]]
        prefix = _prefix_of(request_plan)

        decode_completion_nodes = self._decode_completion_nodes[
            request_plan["request_id"]]

        def emit_transfer(transfer: KVTransfer, stage: str, *,
                          gate=None, trigger_gate=None) -> dict:
            action_name = (
                f"{prefix}_{stage}_action"
                f"{self._next_action_sequence(request_plan):03d}_"
                f"{sanitize_node_prefix(transfer.session_id)}_"
                f"{transfer.kind}"
            )
            record = _emit_kv_transfer(
                config=self.config,
                builders=builders,
                group_by_index=self.group_by_index,
                tag_allocator=self._tag_allocator,
                transfer=transfer,
                action_name=action_name,
                pending_gate=gate,
                trigger_gate=trigger_gate,
                transfer_anchor_sink=None,
            )
            if transfer.kind == "remote_store":
                self._mark_pending_history_store(transfer)
            return record

        completion_eviction_trigger = TransferTriggerGate(
            control_instance_index=request_plan["decode_instance_index"],
            node_gates=tuple(
                decode_completion_nodes[rank] for rank in decode_group.ranks),
        )
        for transfer in request_plan["completion_evictions"]:
            emit_transfer(kv_transfer_from_log(transfer), "completion_evictions",
                          trigger_gate=completion_eviction_trigger)

        # ---- 下一 turn 的 interval gates（duration == 0 → 无节点，
        #      after_node_id 直接作依赖；实际等待由 C++ arrival alarm 替代）----
        if following_plan is None:
            self.deferred_session_locations.pop(request_plan["session_id"], None)
            self.pending_request_by_session.pop(
                request_plan["session_id"], None)
            return
        following_request = self.config.request_queue[
            following_plan["queue_index"]]
        interval = following_request.inter_request_interval_ns
        if interval is None:
            raise RuntimeError("later request lost its inter-request interval")
        timers = tuple(
            builders[rank].timer_gate(
                f"q{following_plan['queue_index']:04d}_"
                f"{sanitize_node_prefix(following_plan['request_id'])}_"
                f"history_rank{rank}_interval_gate",
                interval + following_plan.get("hbm_wait_ns", 0),
                after_node_id=decode_completion_nodes[rank],
            )
            for rank in decode_group.ranks
        )
        completion_location = request_plan.get("kv_location_after_completion")
        if request_plan["session_id"] in self.deferred_session_locations:
            completion_location = self.deferred_session_locations.pop(
                request_plan["session_id"])
        if completion_location not in {
            "local_hbm", "partial_hbm_remote", "remote_memory",
        }:
            raise RuntimeError("completed request has no valid KV location")
        self.pending_history[following_plan["request_id"]] = PendingHistoryGate(
            source_instance_index=request_plan["decode_instance_index"],
            timer_gates=timers,
            location=completion_location,
        )
        self.pending_request_by_session[request_plan["session_id"]] = (
            following_plan["request_id"])

    def _next_action_sequence(self, request_plan: dict) -> int:
        key = request_plan["request_id"]
        value = self._action_sequence.get(key, 0)
        self._action_sequence[key] = value + 1
        return value

    def config_p_chunk(self) -> int:
        return self.config.prefill_chunk_size


def _prefix_of(request_plan: dict) -> str:
    return (
        f"q{request_plan['queue_index']:04d}_"
        f"{sanitize_node_prefix(request_plan['session_id'])}_"
        f"turn{request_plan['turn_index']}_"
        f"{sanitize_node_prefix(request_plan['request_id'])}"
    )
