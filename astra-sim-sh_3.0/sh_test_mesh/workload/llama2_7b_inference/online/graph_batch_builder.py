#!/usr/bin/env python3
"""graph_batch_builder.py -- sh_3.0 在线 GraphBatch 构图器（方案 §4 步骤 1-8 操作 4）。

阶段 1 最关键的对齐点：复用共享发射原语的节点结构。per-request 发射
（generate_face_trace.py 模块级发射助手函数）组成
（_emit_kv_transfer / _emit_tp_readiness_barrier /
_emit_tp_point_to_point_readiness_barrier / transformer_pass_aggregated）——
本模块直接 import 它们，用 OnlineTraceBuilder（与 TraceBuilder 同构的在线侧
builder）驱动，保证节点属性、插入顺序、rank ownership 跨 request 链一致
（节点级审计口径）。

与纯调度参考的刻意差异（合同① 两段式发射边界）：
  - prefill 段（ARRIVAL 边界提交）= 到达/interval gate → history_evictions →
    history_transfer（含 partial 流水恢复全部节点）→ prefill_evictions →
    prefill readiness barrier → prefill 段（含 end barrier）；
  - decode 段（PREFILL_DRAIN 边界提交）= decode_evictions →
    prefill_decode_transfer（local_hit 无节点）→ decode readiness barrier →
    decode 段（含 end barrier）；
  - completion 批（DECODE_COMPLETION/REQUEST_COMPLETE 边界提交）=
    completion_evictions + 下一 turn interval gate 的依赖登记。

在线语义差异（刻意，注释标注）：
  - timer gate 始终发射节点（结构保留）但 runtime_ns=0：到达时间由 C++ 的
    arrival alarm（future_alarms）替代；
  - timer gate 的 duration 语义与共享参考同参（duration=admission_time_ns /
    interval + hbm_wait_ns；0 时不发射 gate 节点，在线同款跳过）；
  - pending_history / deferred_session_locations 账本与共享调度语义同构
    （跨请求的 arrival/interval gate 关联）；
  - partial 流水恢复的 chain_checkpoint/restore_chain 段内分支并行机制
    原样支持；
  - strategy 保持物理链与真实 MEM/HBM 物理时长。
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
from generate_face_trace import (  # noqa: E402
    PendingHistoryGate,
    TransferTagAllocator,
    TransferTriggerGate,
    _emit_kv_transfer,
    _emit_tp_point_to_point_readiness_barrier,
    _emit_tp_readiness_barrier,
    reconcile_pending_history_location,
    sanitize_node_prefix,
)


class OnlineTraceBuilder:
    """与共享 TraceBuilder 接口同构的在线侧每-rank builder。

    同一接口面：timer_gate / arm_timer_gate / arm_dependency /
    chain_checkpoint / restore_chain / comp / all_reduce / comm_send /
    comm_recv / mem_store / mem_load / local_hbm_kv_restore / next_id /
    previous_id / node_count——共享助手函数可直接驱动。

    与 TraceBuilder 的差异：节点发射为 GraphBatch nodes[] dict（而非
    ChakraNode 对象），依赖记录为 parent_edges[]（共享 builder 的 data_deps
    内联在节点里，在线按边列表携带）；timer_gate 忽略 duration
    （runtime_ns=0，alarm 替代等待）。

    per-rank id 自 0 起全局递增，跨批次保持连续。
    """

    def __init__(self, rank: int, *, remote_operand_loads: bool):
        self.rank = rank
        self.remote_operand_loads = remote_operand_loads
        self.next_id = 0
        self.previous_id = None
        self.pending_extra_dependencies = []
        self._checkpoints = {}
        self.nodes = []   # 本 rank 全部已发射节点 dict（发射序）
        self.edges = []   # 本 rank 全部 parent edges {"rank","from","to","kind"}
        self.node_count = 0
        # 当前 request-stage 反向索引上下文（每次 per-request 发射前设置）。
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
            "mem": {"tensor_size": 0, "is_local_hbm_kv_restore": False,
                    "hbm_access_mode": 0},
            "comm": {"bytes": 0, "src": 0, "dst": 0, "tag": 0,
                     "hbm_charge": True},
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

    # 与共享 TraceBuilder.arm_dependency 同构。
    def arm_dependency(self, node_id) -> None:
        if node_id is not None:
            self.pending_extra_dependencies.append(int(node_id))

    # 与共享 TraceBuilder.chain_checkpoint/restore_chain
    # 同构（sh_2.0 在线版同款完整恢复）：
    # partial 流水恢复的段内分支并行机制（prefix 计算与 suffix 恢复并行）。
    # checkpoint 同时捕获 previous_id 与 pending_extra_dependencies，
    # restore 两者一并回滚——分支内新 arm 的依赖不泄漏到恢复点之后。
    # 返回 checkpoint 句柄。（2026-08-20 前在线版只存/回 previous_id，
    # 因唯一调用点 checkpoint 时 pending_extra_dependencies 恒被
    # readiness barrier 清空而无实际差异；补齐为完整同构以消除潜伏
    # 分叉，对现行行为零变化。）
    def chain_checkpoint(self):
        handle = len(self._checkpoints)
        self._checkpoints[handle] = (
            self.previous_id, tuple(self.pending_extra_dependencies))
        return handle

    def restore_chain(self, handle) -> None:
        self.previous_id, dependencies = self._checkpoints.pop(handle)
        self.pending_extra_dependencies = list(dependencies)

    def timer_gate(self, name: str, duration_ns: int, *,
                   after_node_id=None):
        """在线 timer gate：发射节点（runtime_ns=0，is_timer_op）。

        与共享 TraceBuilder.timer_gate 同构：
        - duration_ns == 0 → 直接返回 after_node_id，不发射节点
            （共享接口同款；interval==0 的 request 不产生 interval gate
            节点）；
          - 否则直接创建节点（不经 _new_node）——不链 previous_id、不消费
            pending_extra_dependencies、不更新 previous_id；仅
            after_node_id 依赖（interval gate 依赖上一 request 完成
            barrier 节点）。
        调度参考语义（duration = admission/interval 等待）由 C++ arrival alarm
        （future_alarms）替代——gate 只保留结构与依赖，保持时长会双重等待。
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
            "mem": {"tensor_size": 0, "is_local_hbm_kv_restore": False,
                    "hbm_access_mode": 0},
            "comm": {"bytes": 0, "src": 0, "dst": 0, "tag": 0,
                     "hbm_charge": True},
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
        node["comm"]["hbm_charge"] = bool(hbm_charge)

    def comm_recv(self, name: str, *, src: int, dst: int, comm_size: int,
                  comm_tag: int, hbm_charge: bool = True) -> None:
        node = self._new_node(name, COMM_RECV_NODE)
        node["comm"]["src"] = int(src)
        node["comm"]["dst"] = int(dst)
        node["comm"]["bytes"] = self._uint64(comm_size)
        node["comm"]["tag"] = int(comm_tag)
        node["comm"]["hbm_charge"] = bool(hbm_charge)

    def mem_store(self, name: str, tensor_size: int, *,
                  hbm_access_mode: int = 0) -> None:
        node = self._new_node(name, MEM_STORE_NODE)
        node["mem"]["tensor_size"] = self._uint64(tensor_size)
        node["mem"]["hbm_access_mode"] = int(hbm_access_mode)
        node["compute"]["tensor_size"] = node["mem"]["tensor_size"]

    def mem_load(self, name: str, tensor_size: int, *,
                 hbm_access_mode: int = 0) -> None:
        node = self._new_node(name, MEM_LOAD_NODE)
        node["mem"]["tensor_size"] = self._uint64(tensor_size)
        node["mem"]["hbm_access_mode"] = int(hbm_access_mode)
        node["compute"]["tensor_size"] = node["mem"]["tensor_size"]

    def local_hbm_kv_restore(self, name: str, tensor_size: int) -> None:
        """发射目标-HBM DMA 写节点（is_local_hbm_kv_restore 路由）。"""
        node = self._new_node(name, MEM_LOAD_NODE)
        node["mem"]["tensor_size"] = self._uint64(tensor_size)
        node["mem"]["is_local_hbm_kv_restore"] = True
        node["compute"]["tensor_size"] = node["mem"]["tensor_size"]

    # ------------------------------------------------------------- 只读属性 --

    @property
    def node_count_total(self) -> int:
        return self.node_count


class GraphBatchBuilder:
    """sh_3.0 在线构图器：持有 per-rank OnlineTraceBuilder（状态跨批次），
    按决策边界发射 prefill 整段 / decode 整段 / completion 批，并维护
    pending_history / completion gates 账本（与离线 writer 同构）。"""

    def __init__(self, config, *, digest_sink=None):
        self.config = config
        # strategy 模式保持物理跨 request 链。
        self.builders = {
            rank: OnlineTraceBuilder(
                rank, remote_operand_loads=config.remote_operand_loads)
            for rank in range(config.npus_count)
        }
        self.group_by_index = dict(enumerate(config.inference_groups))
        self.tag_allocator = TransferTagAllocator()
        # 跨请求 history gate 账本（request_id → PendingHistoryGate）。
        self.pending_history = {}
        self.pending_request_by_session = {}
        self.deferred_session_locations = {}
        # request_id -> {rank: prefill 块末 previous_id}（decode 段
        # decode_eviction_trigger 的 node_gates 来源；frontier 接续裁决
        # 2026-08-19 起不再作恢复用，见 _emit_decode）。
        self._prefill_block_ends = {}
        # request_id -> decode 完成节点 per rank（completion 批的
        # completion_eviction trigger gate 与 interval gate after_node_id）。
        self._decode_completion_nodes = {}
        self.batch = None  # 当前批次累加器（由 begin_batch 建立）
        # request_id -> 跨阶段连续的 action 序号（离线 writer 的 per-
        # request action_sequence 闭包在两段式发射下的等价物：prefill/
        # decode/completion 三批共享同一计数器，保证 actionNNN 命名与
        # 逐字节稳定——canonical 命名对照前提）。
        self._action_seq = {}

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

    # ------------------------------------------------------------- 发射 --

    def emit_prefill_batch(self, request_plan: dict) -> dict:
        """发射 request 的 prefill 整段（ARRIVAL 决策的图；两段式发射的
        第一段）。返回 PREFILL_DRAIN watch 成员：{rank: prefill 末个真实
        节点 id}（离线 EVENT_PREFILL_END 锚点口径，排除 end barrier）。

        strategy 恒走 roofline 物理时钟。"""
        self._set_context(request_plan, "prefill", 0)
        marker = self._mark()
        members = self._emit_prefill(request_plan)
        self._collect(marker)
        return members

    def emit_decode_batch(self, request_plan: dict) -> dict:
        """发射 request 的 decode 整段（PREFILL_DRAIN 决策的图；两段式
        发射的第二段）。返回 DECODE_COMPLETION watch 成员：{rank: end
        barrier 前的 decode 末节点 id}（离线 EVENT_DECODE_END 锚点口径）。"""
        self._set_context(request_plan, "decode", 1)
        marker = self._mark()
        members = self._emit_decode(request_plan)
        self._collect(marker)
        return members

    def emit_completion_batch(self, request_plan: dict) -> dict:
        """completion 批（DECODE_COMPLETION/REQUEST_COMPLETE 边界）：
        completion_evictions + 下一 turn interval gate 的依赖登记。
        返回空成员（无 watch）。"""
        self._set_context(request_plan, "completion", 1)
        marker = self._mark()
        self._emit_completion(request_plan)
        self._collect(marker)
        return {}

    def _set_context(self, request_plan: dict, stage: str,
                     generation: int) -> None:
        request_id = request_plan["request_id"]
        for builder in self.builders.values():
            builder.set_context(request_id, stage, generation)

    # ------------------------------------------------- per-request 发射主体 --

    def _mark_pending_history_store(self, transfer) -> None:
        """remote_store 发射后的会话位置登记
        （resident_prefix_layers_after 折算 location）。"""
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
        """turn-gates / history / prefill 块的在线发射。
        request_plan 为 dict（strategy 自在线账本；
        字段与 FaceRequestPlan 同名）。"""
        builders = self.builders
        config = self.config
        group_by_index = self.group_by_index
        prefill_group = group_by_index[request_plan["prefill_instance_index"]]
        request = config.request_queue[request_plan["queue_index"]]
        prefix = _prefix_of(request_plan)
        action_state = self._action_seq.setdefault(
            request_plan["request_id"], [0])

        def emit_transfer(transfer, stage: str, *, gate=None,
                          trigger_gate=None) -> dict:
            action_sequence = action_state[0]
            action_name = (
                f"{prefix}_{stage}_action{action_sequence:03d}_"
                f"{sanitize_node_prefix(transfer.session_id)}_{transfer.kind}"
            )
            record = _emit_kv_transfer(
                config=config,
                builders=builders,
                group_by_index=group_by_index,
                tag_allocator=self.tag_allocator,
                transfer=transfer,
                action_name=action_name,
                pending_gate=gate,
                trigger_gate=trigger_gate,
            )
            record["sequence_stage"] = stage
            record["action_sequence"] = action_sequence
            action_state[0] = action_sequence + 1
            if transfer.kind == "remote_store":
                self._mark_pending_history_store(transfer)
            return record

        # ---- arrival / interval gate（turn-0 到达 timer / turn>0 pending gate）----
        if request_plan["turn_index"] == 0:
            # 在线：turn-0 arrival timer gate 在本批次发射（runtime=0，
            # 到达时间由 C++ arrival alarm 替代；duration 语义与离线同参：
            # admission_time_ns，0 时离线也不发射 gate 节点）。
            arrival = request_plan.get("admission_time_ns")
            if arrival is None:
                arrival = request.session_arrival_time_ns
            if arrival is None:
                raise RuntimeError("first request lost its session arrival")
            # turn-0 gate 命名用短前缀（q{queue:04d}_{request_id}，
            # 无 session/turn 段）——canonical 命名 name 键逐字节一致前提。
            short_prefix = (
                f"q{request_plan['queue_index']:04d}_"
                f"{sanitize_node_prefix(request_plan['request_id'])}")
            timers = tuple(
                builders[rank].timer_gate(
                    f"{short_prefix}_global_arrival_timer_gate", arrival)
                for rank in prefill_group.ranks
            )
            pending_gate = PendingHistoryGate(
                source_instance_index=request_plan["prefill_instance_index"],
                timer_gates=timers,
                location="new_session",
            )
        else:
            pending_gate = self.pending_history.pop(
                request_plan["request_id"], None)
            if pending_gate is None:
                raise RuntimeError(
                    f"request {request_plan['request_id']} has no "
                    "arrival/history gate")
            self.pending_request_by_session.pop(request_plan["session_id"],
                                                None)
        # sh_3.0 裁决（登记合同⑦/§13，两模式统一）：pending-gate location
        # 是 writer 侧的派生缓存（历史静态实现曾用全局 KV 因果预排序
        # 保持与 planner 一致）；当前在线路径以 kv_manager 权威账本/图依赖为准，
        # 触发 request 的准入/完成逐出（reserve/prepare/expand/enforce）与
        # gate 注册/消费的相对顺序不同，缓存会滞后于权威账本（strategy 权威
        # = kv_manager 快照,即 history_location_before 本身）。故弹出 gate 时统一归一化到
        # history_location_before；物理时序由图依赖保证（store 节点挂触发
        # request 链），gate location 仅是发射期元数据。结构性错误（缺
        # gate/缺 location）仍 fail-closed。
        if request_plan["history_location_before"] is not None:
            pending_gate.location = (
                request_plan["history_location_before"].location)
        else:
            reconcile_pending_history_location(
                pending_gate, _as_plan(request_plan))

        # ---- history_evictions ----
        history_eviction_trigger = TransferTriggerGate(
            control_instance_index=pending_gate.source_instance_index,
            node_gates=pending_gate.timer_gates,
        )
        for transfer in request_plan["history_evictions"]:
            emit_transfer(transfer, "history_evictions",
                          trigger_gate=history_eviction_trigger)

        partial_history_restore = (
            request_plan["history_location_before"] is not None
            and request_plan["history_location_before"].location
            == "partial_hbm_remote"
        )
        suffix_ready_nodes_by_rank = {}

        # ---- history_transfer ----
        if _as_plan(request_plan).history_transfer is None:
            if request_plan["turn_index"] != 0:
                raise RuntimeError(
                    "later request is missing its history transfer action")
            source_group = group_by_index[pending_gate.source_instance_index]
            if source_group.ranks != prefill_group.ranks:
                raise RuntimeError(
                    "first-request arrival gate is not on its Prefill ranks")
            for relative_index, rank in enumerate(prefill_group.ranks):
                builders[rank].arm_timer_gate(
                    pending_gate.timer_gates[relative_index])
        elif not partial_history_restore:
            # gate location 已在弹出时归一化到权威快照（见上方裁决注释）。
            emit_transfer(_as_plan(request_plan).history_transfer,
                          "history_transfer", gate=pending_gate)

        # ---- prefill_evictions ----
        for transfer in request_plan["prefill_evictions"]:
            emit_transfer(transfer, "prefill_evictions")

        # ---- readiness barrier / partial 流水恢复 ----
        if partial_history_restore:
            history_transfer = _as_plan(request_plan).history_transfer
            history_before = request_plan["history_location_before"]
            if (history_transfer.kind != "remote_load"
                    or history_transfer.layer_start
                    != history_before.resident_prefix_layers
                    or history_transfer.layer_end != config.layers):
                raise RuntimeError(
                    "partial history load does not match its suffix")
            if (request_plan["prefill_instance_index"]
                    != history_before.instance_index):
                raise RuntimeError(
                    "partial history lost Prefill instance affinity")
            if pending_gate.location != history_before.location:
                raise RuntimeError(
                    f"history gate location {pending_gate.location!r} does "
                    f"not match planned {history_before.location!r}")
            for relative_index, rank in enumerate(prefill_group.ranks):
                builders[rank].arm_timer_gate(
                    pending_gate.timer_gates[relative_index])
            _emit_tp_readiness_barrier(
                builders=builders, group=prefill_group,
                name=f"{prefix}_prefill_resident_prefix_ready_barrier")
            checkpoints = {
                rank: builders[rank].chain_checkpoint()
                for rank in prefill_group.ranks
            }
            prefix_barrier_nodes = tuple(
                builders[rank].previous_id for rank in prefill_group.ranks)
            branch_gate = PendingHistoryGate(
                source_instance_index=request_plan["prefill_instance_index"],
                timer_gates=prefix_barrier_nodes,
                location=history_before.location,
            )
            history_record = emit_transfer(history_transfer,
                                           "history_transfer", gate=branch_gate)
            suffix_restore_nodes_by_rank = {}
            for shard_record in history_record["shards"]:
                target_rank = shard_record.get("target_rank")
                completion_node = shard_record.get(
                    "target_hbm_completion_node_id")
                if not isinstance(target_rank, int) or not isinstance(
                        completion_node, int):
                    raise RuntimeError(
                        "suffix restore is missing a target HBM gate")
                suffix_restore_nodes_by_rank[target_rank] = completion_node
            if set(suffix_restore_nodes_by_rank) != set(prefill_group.ranks):
                raise RuntimeError(
                    "suffix restore did not cover every Prefill rank")
            suffix_readiness = _emit_tp_point_to_point_readiness_barrier(
                builders=builders, group=prefill_group,
                tag_allocator=self.tag_allocator,
                name=f"{prefix}_prefill_suffix_ready_barrier")
            suffix_ready_nodes_by_rank = {
                int(rank): int(node_id)
                for rank, node_id in suffix_readiness["node_ids_by_rank"]
            }
            for rank in prefill_group.ranks:
                builders[rank].restore_chain(checkpoints[rank])
        else:
            _emit_tp_readiness_barrier(
                builders=builders, group=prefill_group,
                name=f"{prefix}_prefill_kv_ready_barrier")

        # ---- prefill 段（request_aggregated）----
        tensor_parallel = len(prefill_group.ranks)
        prefill_tokens_to_process = (
            request_plan["prefill_context_tokens"]
            - request_plan["history_tokens_before"]
        )
        if prefill_tokens_to_process <= 0:
            raise ValueError(
                f"request {request_plan['request_id']} has no Prefill work tokens"
            )
        if config.trace_granularity != "request_aggregated":
            raise RuntimeError(
                "online emission supports request_aggregated granularity "
                "only (production config)")
        prefill_spans = []
        processed = 0
        p_chunk = self.p_chunk
        while processed < prefill_tokens_to_process:
            chunk_tokens = min(p_chunk,
                               prefill_tokens_to_process - processed)
            prefill_spans.append(
                (chunk_tokens,
                 request_plan["history_tokens_before"] + processed
                 + chunk_tokens))
            processed += chunk_tokens
        prefill_last_node_by_rank = {}
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
                suffix_start = (
                    request_plan["history_location_before"]
                    .resident_prefix_layers)
                transformer_pass_aggregated(
                    builders[rank],
                    phase=f"{prefix}_prefill_first_chunk_prefix",
                    pass_spans=(prefill_spans[0],),
                    layer_start=0, layer_end=suffix_start,
                    include_output=False, **aggregate_arguments)
                builders[rank].arm_dependency(
                    suffix_ready_nodes_by_rank[rank])
                transformer_pass_aggregated(
                    builders[rank],
                    phase=f"{prefix}_prefill_first_chunk_suffix",
                    pass_spans=(prefill_spans[0],),
                    layer_start=suffix_start, layer_end=config.layers,
                    include_output=True, **aggregate_arguments)
                if len(prefill_spans) > 1:
                    transformer_pass_aggregated(
                        builders[rank],
                        phase=f"{prefix}_prefill_remaining_chunks_aggregated",
                        pass_spans=prefill_spans[1:], **aggregate_arguments)
                pass_count = len(prefill_spans)
            else:
                pass_count = transformer_pass_aggregated(
                    builders[rank],
                    phase=f"{prefix}_prefill_request_aggregated",
                    pass_spans=prefill_spans, **aggregate_arguments)
            # PREFILL_DRAIN watch 成员 = 每 rank 末个真实 prefill 节点
            # （离线 EVENT_PREFILL_END 锚点，排除 end barrier）。
            prefill_last_node_by_rank[rank] = builders[rank].previous_id
            builders[rank].all_reduce(
                f"{prefix}_prefill_chunks_aggregated_end_barrier",
                pass_count, prefill_group.pg_name)

        # 记录本 request 的 per-rank prefill 块末 previous_id（emitted-ranks-
        # only；仅作 decode 段 decode_eviction_trigger 的 node_gates 来源，
        # 不再恢复进 builders——frontier 接续裁决 2026-08-19，见 _emit_decode）。
        decode_group = group_by_index[
            request_plan["decode_instance_index"]]
        self._prefill_block_ends[request_plan["request_id"]] = {
            rank: (builders[rank].previous_id
                   if rank in prefill_group.ranks else None)
            for rank in sorted(
                set(prefill_group.ranks) | set(decode_group.ranks))
        }
        return prefill_last_node_by_rank

    def _emit_decode(self, request_plan: dict) -> dict:
        """decode_evictions / pd transfer / decode 段块的在线发射
        （request_aggregated）。"""
        builders = self.builders
        config = self.config
        group_by_index = self.group_by_index
        prefill_group = group_by_index[request_plan["prefill_instance_index"]]
        decode_group = group_by_index[request_plan["decode_instance_index"]]
        request = config.request_queue[request_plan["queue_index"]]
        prefix = _prefix_of(request_plan)
        action_state = self._action_seq.setdefault(
            request_plan["request_id"], [0])

        def emit_transfer(transfer, stage: str, *, trigger_gate=None) -> dict:
            action_sequence = action_state[0]
            action_name = (
                f"{prefix}_{stage}_action{action_sequence:03d}_"
                f"{sanitize_node_prefix(transfer.session_id)}_{transfer.kind}"
            )
            record = _emit_kv_transfer(
                config=config, builders=builders,
                group_by_index=group_by_index,
                tag_allocator=self.tag_allocator, transfer=transfer,
                action_name=action_name, trigger_gate=trigger_gate)
            record["sequence_stage"] = stage
            record["action_sequence"] = action_sequence
            action_state[0] = action_sequence + 1
            if transfer.kind == "remote_store":
                self._mark_pending_history_store(transfer)
            return record

        # [frontier 接续裁决,strategy 死锁修复统一(2026-08-19,对齐 sh_1.0/
        # sh_2.0)] strategy **不做任何块末恢复/段内清链**:per-rank
        # previous_id 无条件接续当前 frontier(= 离线 writer 跨 request
        # 物理链同构),per-rank 发行序 = 全局发射序——任意两个发射段在
        # 所有共享 rank 上的相对次序一致,跨请求 P2P 与 collective 参与
        # 序不可能反转成环。decode 段首节点(decode_evictions 的首个
        # transfer)因此链到该 rank 当前 frontier(跨 request 边);本
        # request 的 prefill 块末依赖经 per-rank 全序传递性保持(prefill
        # 先于 decode 发射)。(蓝本裁决 3/7/9 的 own-prefill-end 恢复
        # 自此废止;块末账本仅存 trigger gate 用途。)

        prefill_completion_nodes = tuple(
            self._prefill_block_ends.get(request_plan["request_id"], {})
            .get(rank)
            for rank in prefill_group.ranks)
        if any(node_id is None for node_id in prefill_completion_nodes):
            # 段 1 发射后 prefill 组必有块末;缺即账本损坏(fail-closed)。
            raise RuntimeError(
                "segment-1 block ends missing on prefill ranks")
        decode_eviction_trigger = TransferTriggerGate(
            control_instance_index=request_plan["prefill_instance_index"],
            node_gates=prefill_completion_nodes,
        )
        for transfer in request_plan["decode_evictions"]:
            emit_transfer(transfer, "decode_evictions",
                          trigger_gate=decode_eviction_trigger)
        pd_transfer = _as_plan(request_plan).prefill_decode_transfer
        if pd_transfer is None:
            raise RuntimeError(
                "request is missing its Prefill-to-Decode KV action")
        emit_transfer(pd_transfer, "prefill_decode_transfer")

        _emit_tp_readiness_barrier(
            builders=builders, group=decode_group,
            name=f"{prefix}_decode_kv_ready_barrier")

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
            # DECODE_COMPLETION watch 成员 = end barrier 前每 rank 的
            # decode 末节点（离线 EVENT_DECODE_END 锚点口径）。
            decode_last_node_by_rank[rank] = builders[rank].previous_id
            builders[rank].all_reduce(
                f"{prefix}_decode_request_end_barrier", 1,
                decode_group.pg_name)

        decode_completion_nodes = tuple(
            builders[rank].previous_id for rank in decode_group.ranks)
        if any(node_id is None for node_id in decode_completion_nodes):
            raise RuntimeError("Decode completion node IDs were not generated")
        self._decode_completion_nodes[request_plan["request_id"]] = (
            request_plan["decode_instance_index"],
            decode_completion_nodes,
        )
        return decode_last_node_by_rank

    def _emit_completion(self, request_plan: dict) -> None:
        """completion_evictions + 下一 turn interval gate 段的在线发射。"""
        builders = self.builders
        config = self.config
        request = config.request_queue[request_plan["queue_index"]]
        prefix = _prefix_of(request_plan)
        entry = self._decode_completion_nodes.get(request_plan["request_id"])
        if entry is None:
            raise RuntimeError("completion batch before decode emission")
        decode_instance_index, decode_completion_nodes = entry
        decode_group = self.group_by_index[decode_instance_index]
        action_state = self._action_seq.setdefault(
            request_plan["request_id"], [0])

        def emit_transfer(transfer, stage: str) -> dict:
            action_sequence = action_state[0]
            action_name = (
                f"{prefix}_{stage}_action{action_sequence:03d}_"
                f"{sanitize_node_prefix(transfer.session_id)}_{transfer.kind}"
            )
            record = _emit_kv_transfer(
                config=config, builders=builders,
                group_by_index=self.group_by_index,
                tag_allocator=self.tag_allocator, transfer=transfer,
                action_name=action_name,
                trigger_gate=TransferTriggerGate(
                    control_instance_index=decode_instance_index,
                    node_gates=decode_completion_nodes,
                ))
            record["sequence_stage"] = stage
            record["action_sequence"] = action_sequence
            action_state[0] = action_sequence + 1
            if transfer.kind == "remote_store":
                self._mark_pending_history_store(transfer)
            return record

        for transfer in request_plan["completion_evictions"]:
            emit_transfer(transfer, "completion_evictions")

        following = self.next_plan.get(request_plan["request_id"])
        if isinstance(following, str):
            # strategy 模式：值是下一 turn 的 request_id，经 resolver 取
            # 实时 plan dict（hbm_wait_ns 在准入时才落账本）。
            following = self._plan_resolver(following)
        if following is not None:
            following_request = config.request_queue[following["queue_index"]]
            interval = following_request.inter_request_interval_ns
            if interval is None:
                raise RuntimeError(
                    "later request lost its inter-request interval")
            timers = tuple(
                builders[rank].timer_gate(
                    f"q{following['queue_index']:04d}_"
                    f"{sanitize_node_prefix(following['request_id'])}_"
                    f"history_rank{rank}_interval_gate",
                    interval + following.get("hbm_wait_ns", 0),
                    after_node_id=decode_completion_nodes[relative_index],
                )
                for relative_index, rank in enumerate(decode_group.ranks)
            )
            completion_location = request_plan.get(
                "kv_location_after_completion")
            if request_plan["session_id"] in self.deferred_session_locations:
                completion_location = (
                    self.deferred_session_locations.pop(
                        request_plan["session_id"]))
            if completion_location not in {
                "local_hbm", "partial_hbm_remote", "remote_memory",
            }:
                raise RuntimeError(
                    "completed request has no valid KV location")
            self.pending_history[following["request_id"]] = PendingHistoryGate(
                source_instance_index=decode_instance_index,
                timer_gates=timers,
                location=completion_location,
            )
            self.pending_request_by_session[request_plan["session_id"]] = (
                following["request_id"])
        else:
            self.deferred_session_locations.pop(request_plan["session_id"],
                                                None)

    # ------------------------------------------------------------- 属性 --

    @property
    def p_chunk(self) -> int:
        return self.config.prefill_chunk_size

    next_plan = {}
    _plan_resolver = staticmethod(lambda request_id: None)

    def set_next_plan(self, next_plan: dict) -> None:
        """request_id -> 下一 turn 的 plan dict
        （interval gate 发射用，由 scheduler 初始化时按 (session_id,
        turn_index) 构建）。"""
        self.next_plan = next_plan

    def set_plan_resolver(self, resolver) -> None:
        """strategy 模式：request_id -> 实时 plan dict 的解析器
        （下一 turn 的 hbm_wait_ns 等在准入时才落账本）。"""
        self._plan_resolver = resolver


def _prefix_of(request_plan: dict) -> str:
    return (
        f"q{request_plan['queue_index']:04d}_"
        f"{sanitize_node_prefix(request_plan['session_id'])}_"
        f"turn{request_plan['turn_index']}_"
        f"{sanitize_node_prefix(request_plan['request_id'])}"
    )


class _PlanShim:
    """把 request_plan dict 包装为离线 helper 期望的 FaceRequestPlan 属性面
    （history_transfer 等 KVTransfer 对象在 dict 中直接携带）。"""

    __slots__ = ("_plan",)

    def __init__(self, plan: dict):
        self._plan = plan

    def __getattr__(self, name):
        try:
            return self._plan[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _as_plan(request_plan: dict):
    # _PlanShim 为纯代理（仅 _plan 字段 + __getattr__ 转发，无状态），每
    # 次直接构造（每请求约 5 次调用的对象构造开销可忽略）——不用 id() 做
    # 模块级缓存：id 复用隐患 + 强引用把每请求 plan 永久钉在内存（O(N)
    # 驻留，R4-11）。
    return _PlanShim(request_plan)
