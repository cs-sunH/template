#!/usr/bin/env python3
"""graph_batch_builder.py -- 在线 GraphBatch 构图器(方案 §4 步骤 1-8 操作 4)。

阶段 1 最关键的对齐点:复用 generator 发射逻辑的节点结构。per-request 发射
(generate_wsc_llm_trace.py 模块级函数)由
助手函数组成(_emit_control_trigger / _paired_transfer / _emit_prefill_stage /
transformer_pass_aggregated)——本模块直接 import 它们,用 OnlineTraceBuilder
(与 TraceBuilder 同构的在线侧 builder)驱动,保证:

  - 节点属性、插入顺序、rank ownership 跨 request 链结构一致(节点级审计口径);
  - per-rank 节点 id 跨批次全局递增(与 generator per-rank id 序列一致);
  - interval gate 的 after_node_id 指向上一 request 的完成 barrier 节点 id
    (completion_gates 账本,跨批次解析)。

在线语义差异(刻意,注释标注):
  - timer gate 始终发射节点(结构保留)但 runtime_ns=0:到达时间由 C++ 的
    arrival alarm(future_alarms)替代,gate 不再等待——保持时长会双重等待;
  - 每 request 拆两次发射:ARRIVAL 边界发射 prefill 整段(含 gates/history/
    recompute/barrier),PREFILL_DRAIN 边界发射 decode 整段(含 transfer 3000/
    decode/end barrier)——离线同 request 一次写完,但 per-rank id 序列一致;
  - watch 锚点与离线 metrics 锚点一致:PREFILL_DRAIN = prefill_bounds 每 rank
    末节点(排除 end barrier),DECODE_COMPLETION = end barrier 前每 rank 的
    decode 末节点(离线 doc sec.6.2/6.3/6.4 口径)。
"""

import os
import sys

# --------------------------------------------------------------------------
# import 路径:本文件位于 workload/llama2_7b_inference/online/,离线写出模块在
# 上一级。路径只做 import 用途(红线:generate_wsc_llm_trace.py / wsc_llm_
# scheduler.py 只读 import 与注释)。
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
    transformer_pass_aggregated,
)
from generate_wsc_llm_trace import (  # noqa: E402
    _emit_control_trigger,
    _emit_prefill_stage,
    _paired_transfer,
    kv_cache_bytes_for_tokens,
    sanitize_node_prefix,
)
from generate_wsc_llm_trace import NOC_MIGRATE, RECOMPUTE  # noqa: E402


class OnlineTraceBuilder:
    """与离线 TraceBuilder 同构的在线侧每-rank builder。

    同一接口面:timer_gate / arm_timer_gate / comp / all_reduce / comm_send /
    comm_recv / next_id / previous_id / node_count——离线助手函数可直接驱动。
    与 TraceBuilder 的差异:节点发射为 GraphBatch nodes[] dict(而非 ChakraNode
    字节流),依赖记录为 parent_edges[](离线 .et 的 data_deps 内联在节点里,
    在线按边列表携带);timer_gate 忽略 duration(runtime_ns=0,alarm 替代等待)。

    per-rank id 自 0 起全局递增(跨批次),与离线 .et 每 rank id 序列一致。
    """

    def __init__(self, rank: int, *, remote_operand_loads: bool):
        self.rank = rank
        self.remote_operand_loads = remote_operand_loads
        self.next_id = 0
        self.previous_id = None
        self.pending_extra_dependencies = []
        self.nodes = []   # 本 rank 全部已发射节点 dict(发射序)
        self.edges = []   # 本 rank 全部 parent edges {"rank","from","to","kind"}
        self.node_count = 0
        # 当前 request-stage 反向索引上下文(每次 per-request 发射前设置)。
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

    def timer_gate(self, name: str, duration_ns: int, *,
                   after_node_id=None):
        """在线 timer gate:发射节点(runtime_ns=0,is_timer_op)。

        与离线 TraceBuilder.timer_gate(generate_trace.py:697-719)完全同构:
          - duration_ns == 0 → 直接返回 after_node_id,不发射节点
            (离线 :707-709 同款;inter_request_interval_ns==0 的 request
            在离线 .et 中没有 interval gate 节点,在线也必须没有);
          - 否则直接创建节点(不经 _new_node)——不链 previous_id、不消费
            pending_extra_dependencies、不更新 previous_id;仅
            after_node_id 依赖(interval gate 依赖上一 request 完成
            barrier)。
        离线语义(duration = 到达时刻/interval,gate 等待)由 C++ arrival
        alarm(future_alarms)替代——gate 只保留结构与依赖,保持时长会双重
        等待(见步骤 1-8 设计分析)。
        """
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
                  comm_tag: int) -> None:
        node = self._new_node(name, COMM_SEND_NODE)
        node["comm"]["src"] = int(src)
        node["comm"]["dst"] = int(dst)
        node["comm"]["bytes"] = self._uint64(comm_size)
        node["comm"]["tag"] = int(comm_tag)

    def comm_recv(self, name: str, *, src: int, dst: int, comm_size: int,
                  comm_tag: int) -> None:
        node = self._new_node(name, COMM_RECV_NODE)
        node["comm"]["src"] = int(src)
        node["comm"]["dst"] = int(dst)
        node["comm"]["bytes"] = self._uint64(comm_size)
        node["comm"]["tag"] = int(comm_tag)

    # ------------------------------------------------------------- 只读属性 --

    @property
    def node_count_total(self) -> int:
        return self.node_count


class GraphBatchBuilder:
    """在线构图器:持有 per-rank OnlineTraceBuilder(状态跨批次),按决策边界
    发射 prefill 整段 / decode 整段,并维护 completion_gates 账本。"""

    def __init__(self, config, *, digest_sink=None):
        self.config = config
        # strategy 模式保持物理跨 request 链。
        self.builders = {
            rank: OnlineTraceBuilder(
                rank, remote_operand_loads=config.remote_operand_loads)
            for rank in range(config.npus_count)
        }
        self.group_by_index = dict(enumerate(config.inference_groups))
        # session_id -> (decode_instance_index, {rank: end_barrier_id});
        # 与离线 writer 的 completion_gates 账本同构(interval gate 的
        # after_node_id 来源)。
        self.completion_gates = {}
        # request_id -> {rank: prefill 块末 previous_id}。在线两阶段发射:
        # decode 段(PREFILL_DRAIN)发射时,prefill 段之后可能已插入其他 request
        # 的节点,transfer3000 首节点的 previous_id 依赖会串到别的 request。
        # 离线 writer 逐 request 连续发射,transfer3000 首节点依赖是本 request
        # 的 prefill 末节点(end barrier / interval gate 等)。本账本在 decode 段
        # 发射前恢复 per-rank previous_id,使 within-request 依赖与本 request
        # 的 prefill 块末一致(主控裁决 2026-08-15 批准;恢复值在
        # PREFILL_DRAIN 边界触发时必已发射,无死锁风险)。
        self._prefill_block_ends = {}
        self.batch = None  # 当前批次累加器(由 begin_batch 建立)

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
        """把各 builder 自 marker 起新增的节点/边并入本批次。"""
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
        """发射 request 的 prefill 整段(ARRIVAL 决策的图)。

        strategy 恒走 roofline 物理时钟。

        返回 PREFILL_DRAIN watch 成员:{rank: 末个真实 prefill 节点 id}
        (与离线 EVENT_PREFILL_END 锚点一致,排除 end barrier)。
        """
        self._set_context(request_plan, "prefill", 0)
        # strategy 模式保持物理跨 request 链(根因 #5 裁决):prefill 组各
        # rank 的 previous_id 不清空,本 request 的 prefill 链首可链上一
        # request 在本 rank 的链尾(共享 rank 上的链串行化)。
        # 保留:within-request 串行化;decode 段 own-prefill-end 恢复(previous_id 链);
        # 同 session interval gate。
        marker = self._mark()
        members = self._emit_prelim(request_plan)
        self._collect(marker)
        return members

    def emit_decode_batch(self, request_plan: dict) -> dict:
        """发射 request 的 decode 整段(含 prefill_to_decode transfer 3000 /
        decode 整段 / decode_request_end_barrier)(PREFILL_DRAIN 决策的图)。

        strategy 恒走 roofline 物理时钟。

        返回 DECODE_COMPLETION watch 成员:{rank: end barrier 前的 decode
        末节点 id}(离线 doc sec.6.4 口径)。
        """
        self._set_context(request_plan, "decode", 1)
        marker = self._mark()
        members = self._emit_decode(request_plan)
        self._collect(marker)
        return members

    def _set_context(self, request_plan: dict, stage: str,
                     generation: int) -> None:
        request_id = request_plan["request_id"]
        for builder in self.builders.values():
            builder.set_context(request_id, stage, generation)

    # ------------------------------------------------- per-request 发射主体 --

    def _emit_prelim(self, request_plan: dict) -> dict:
        """离线 writer 的 turn-gates/history/barrier/current_prefill 块
        (generate_wsc_llm_trace.py:1093-1213 的在线复刻)。"""
        builders = self.builders
        prefill_group = self.group_by_index[request_plan["prefill_instance_index"]]
        prefix = _prefix_of(request_plan)
        request = _request_spec(self.config, request_plan)
        if request_plan["turn_index"] == 0:
            if request.session_arrival_time_ns is None:
                raise RuntimeError("first request lost its session arrival")
            # 在线:timer gate runtime=0,到达时间由 C++ arrival alarm 替代。
            # duration 传真实 session_arrival_time_ns(离线同参):0 时离线
            # 不发射 gate 节点,在线同款跳过。
            timers = {
                rank: builders[rank].timer_gate(
                    "{}_arrival_timer_gate".format(prefix),
                    request.session_arrival_time_ns)
                for rank in prefill_group.ranks
            }
            for rank in prefill_group.ranks:
                builders[rank].arm_timer_gate(timers[rank])
        else:
            previous = self.completion_gates.get(request_plan["session_id"])
            if previous is None or request.inter_request_interval_ns is None:
                raise RuntimeError(
                    "later request has no completion interval gate")
            previous_index, previous_nodes = previous
            previous_group = self.group_by_index[previous_index]
            # duration 传真实 inter_request_interval_ns(离线同参):0 时离线
            # 不发射 interval gate 节点(返回 after_node_id 直接作依赖),
            # 在线同款跳过——与离线 .et 节点数/依赖逐字节一致。
            timers = {
                rank: builders[rank].timer_gate(
                    "{}_interval_timer_gate".format(prefix),
                    request.inter_request_interval_ns,
                    after_node_id=previous_nodes.get(rank))
                for rank in previous_group.ranks
            }
            if request_plan["history_action"] == NOC_MIGRATE:
                source_index = request_plan["history_source_instance_index"]
                if source_index is None:
                    raise RuntimeError("NoC history action has no source")
                source_group = self.group_by_index[source_index]
                if source_group.name != previous_group.name:
                    _emit_control_trigger(
                        builders=builders,
                        queue_index=request_plan["queue_index"],
                        name="{}_history_source_control".format(prefix),
                        source_group=previous_group,
                        target_group=source_group,
                        timer_gates=timers,
                    )
                    timers = {rank: None for rank in source_group.ranks}
                _paired_transfer(
                    config=self.config, builders=builders,
                    queue_index=request_plan["queue_index"], category=1000,
                    name="{}_history_kv".format(prefix),
                    source_group=source_group, target_group=prefill_group,
                    total_bytes=request_plan["history_transfer_bytes"],
                    timer_gates=timers,
                )
            else:
                _emit_control_trigger(
                    builders=builders, queue_index=request_plan["queue_index"],
                    name="{}_interval_control".format(prefix),
                    source_group=previous_group, target_group=prefill_group,
                    timer_gates=timers,
                )
        if request_plan["history_action"] == RECOMPUTE:
            _emit_prefill_stage(
                config=self.config, builders=builders, group=prefill_group,
                prefix=prefix, stage="history_recompute",
                tokens=request_plan["history_recompute_tokens"],
                initial_context_tokens=0,
            )
        for rank in prefill_group.ranks:
            builders[rank].all_reduce(
                "{}_history_tp_ready_barrier".format(prefix), 1,
                prefill_group.pg_name)
        prefill_bounds = _emit_prefill_stage(
            config=self.config, builders=builders, group=prefill_group,
            prefix=prefix, stage="current_prefill",
            tokens=request_plan["prefill_length"],
            initial_context_tokens=request_plan["history_tokens_before"],
        )
        # PREFILL_DRAIN watch 成员 = 每 rank 末个真实 prefill 节点
        # (与离线 EVENT_PREFILL_END 锚点一致,排除 end barrier)。
        members = {rank: bounds[1] for rank, bounds in prefill_bounds.items()}
        # [previous_id 链修复] 记录本 request 的 per-rank prefill 块末 previous_id。
        # transfer3000 的 comm_send 在 prefill rank 上发射、comm_recv 在
        # decode rank 上发射。prefill 侧首节点(send)链本块末节点(离线同款,
        # 实测 .et send data_deps=[本 request prefill 末节点]);decode 侧首
        # 节点(recv)在离线 .et 中链"上一完整 request 块末"(跨 request 串行
        # 化边,方案 §4 步骤 1-8 裁决明言不可复现、差分归因)——本 request 的
        # prefill 块在 decode 组 rank 上没有发射任何节点,块末 = None:
        # 恢复 None 使 recv 无父(与离线首个 decode 的 recv 同构,实测 .et
        # session_0 recv data_deps=[]),decode 块即到即跑,不串行等待上一
        # request 的 decode 段(实测 2026-08-15:若把 stale 的跨 request
        # previous_id 记入,recv 会等待上一 decode 的 end barrier——该 barrier
        # 在 PREFILL_DRAIN 触发时仍在执行,与方案 :1148 的"恢复值彼时必然
        # 已完成"断言相悖,rank 上 decode 全部串行化,完成边界被整体推后)。恢复值
        # 只作用于 transfer 首节点,块内其余节点正常续链。
        decode_group = self.group_by_index[
            request_plan["decode_instance_index"]]
        self._prefill_block_ends[request_plan["request_id"]] = {
            rank: (builders[rank].previous_id if rank in prefill_group.ranks
                   else None)
            for rank in sorted(
                set(prefill_group.ranks) | set(decode_group.ranks))
        }
        return members

    def _emit_decode(self, request_plan: dict) -> dict:
        """离线 writer 的 transfer 3000 + decode 整段 + end barrier 块
        (generate_wsc_llm_trace.py:1215-1275 的在线复刻)。"""
        builders = self.builders
        prefill_group = self.group_by_index[request_plan["prefill_instance_index"]]
        decode_group = self.group_by_index[request_plan["decode_instance_index"]]
        prefix = _prefix_of(request_plan)
        # [previous_id 链修复] 恢复 prefill 块末 per-rank previous_id,使 transfer3000 /
        # decode 块的首节点依赖与离线逐边一致(own end barrier / own interval
        # gate)。仅恢复本 request 覆盖的 rank;restore 后本块内部继续自然链式。
        block_ends = self._prefill_block_ends.get(request_plan["request_id"])
        if block_ends is not None:
            for rank, end_id in block_ends.items():
                builders[rank].previous_id = end_id
        _paired_transfer(
            config=self.config, builders=builders,
            queue_index=request_plan["queue_index"], category=3000,
            name="{}_prefill_to_decode_kv".format(prefix),
            source_group=prefill_group, target_group=decode_group,
            total_bytes=kv_cache_bytes_for_tokens(
                self.config.model, request_plan["prefill_context_tokens"]),
        )
        tp = len(decode_group.ranks)
        spans = tuple(
            (1, request_plan["prefill_context_tokens"] + step + 1)
            for step in range(request_plan["decode_length"])
        )
        members = {}
        for relative_rank, rank in enumerate(decode_group.ranks):
            transformer_pass_aggregated(
                builders[rank],
                phase="{}_decode_request_aggregated".format(prefix),
                pass_spans=spans, layers=self.config.layers,
                hidden_size=self.config.hidden_size,
                ffn_size=self.config.ffn_size, tensor_parallel=tp,
                pg_name=decode_group.pg_name, vocab_size=self.config.vocab_size,
                bytes_per_elem=self.config.bytes_per_elem,
                num_heads=self.config.num_heads,
                tensor_parallel_rank=relative_rank,
                mlp_variant=self.config.mlp_variant,
            )
            # 离线 doc sec.6.4: completion candidate = end barrier 前每 rank
            # 的 decode 末节点(DECODE_COMPLETION watch 成员)。
            members[rank] = builders[rank].previous_id
            builders[rank].all_reduce(
                "{}_decode_request_end_barrier".format(prefix), 1,
                decode_group.pg_name)
        # completion gate 账本: end barrier 后的 previous_id 是 barrier 自身
        # (下一 turn 的 interval gate after_node_id 指向它)。
        self.completion_gates[request_plan["session_id"]] = (
            request_plan["decode_instance_index"],
            {rank: builders[rank].previous_id for rank in decode_group.ranks},
        )
        return members


def _prefix_of(request_plan: dict) -> str:
    return (
        "q{:04d}_{}_turn{}_{}".format(
            request_plan["queue_index"],
            sanitize_node_prefix(request_plan["session_id"]),
            request_plan["turn_index"],
            sanitize_node_prefix(request_plan["request_id"]),
        )
    )


def _request_spec(config, request_plan: dict):
    """从 config.request_queue 取 RequestSpec(按 manifest 的 queue_index)。"""
    return config.request_queue[request_plan["queue_index"]]
