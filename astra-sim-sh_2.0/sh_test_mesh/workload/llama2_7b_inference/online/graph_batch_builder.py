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
  - interval gate 的 after_node_id 指向上一 request 退出列车 end barrier
    节点 id（completion 段账本，跨批次解析）。

sh_2.0 发射结构（拼 batch 改造,2026-08-22;原三段式的列车化重构,
设计文档《层次 B Continuous Batching 改造》§3.2;母本 sh_1.0 定型版同构）:
  - 准入动作(ARRIVAL 边界,emit_admission_batch)= 到达/interval timer
    gates + history_evictions + history_transfer(含 sh_2.0 特有的 partial
    前缀两段式迁移:prefix noc_migrate → prefix ready barrier → suffix
    remote_load 恢复 → suffix ready barrier,chain checkpoint/restore 保持
    恢复分支与主链并行)+ prefill_evictions + prefill 屏障;prefill 主体
    不再在此发射;
  - 迭代列车(各决策边界,emit_iteration_train)= joiner 的
    decode_evictions(触发门 = drain 列车 barrier)+ prefill→decode 迁移
    + 共享 readiness barrier + 折叠列车体(成员×迭代 span,weight_passes
    =迭代数:权重每迭代只读一次;partial 恢复请求的首 chunk 仍按
    prefix/suffix 层段拆分发射,suffix 层段 arm 依赖 admission 记录的
    suffix ready 节点——两段式流水保留)+ drain/exit 标记节点(挂
    PREFILL_DRAIN / DECODE_COMPLETION watch)+ 每列车一个共享 end
    barrier;
  - 完成段(REQUEST_COMPLETE 边界,emit_completion_batch)= completion_
    evictions(触发门 = 退出列车 barrier)+ 下一同 session turn 的
    interval timer gates(after_node_id=退出列车 barrier)。

在线语义差异（刻意，注释标注；蓝本裁决 3/7/9 的三段推广）：
  - timer gate 始终发射节点（结构保留）但 runtime_ns=0：到达时间由 C++ 的
    arrival alarm（future_alarms）替代，gate 不再等待——保持时长会双重等待；
  - watch 锚点与共享指标口径一致（拼 batch 改造起由列车标记承载）：
    PREFILL_DRAIN = drain 标记节点（列车体后、end barrier 前，与
    EVENT_PREFILL_END 锚点同款"barrier 前末节点"口径）；
    DECODE_COMPLETION/REQUEST_COMPLETE = exit 标记节点（同位置）。
    触发门角色（joiner decode_evictions / completion_evictions 的
    node_gates、下一 turn interval gate 的 after_node_id、_block_ends
    账本）仍用 post-barrier 节点（列车 end barrier；五仓一致口径，
    不随 watch 锚点变化）。决策边界因此比 post-barrier 口径早一个
    all_reduce——这是五仓统一的预期时间线变化；
  - strategy 模式保持物理跨 request 链（列车整体接续 per-rank frontier）。
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
    边界发射准入动作 / 实例迭代列车 / completion 段，并维护 pending_history
    与列车块末账本（共享调度语义同构；拼 batch 改造 2026-08-22）。"""

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
        # request_id -> {"seg1": {rank: node_id|None}, "seg2": {rank: node_id|None}}
        # 列车块末账本（拼 batch 改造,2026-08-22;emitted-ranks-only 语义由
        # 发射侧保证——seg1/seg2 均只记列车组 rank 的实际 end barrier 节点）。
        # post-barrier 口径（五仓一致,不随 watch 锚点变化）:seg1 = drain 列车
        # barrier（joiner decode_evictions 触发门）,seg2 = 退出列车 barrier
        # （completion_evictions 触发门 + 下一 turn interval gate after_node_id）。
        self._block_ends = {}
        # request_id -> partial 恢复两段式流水信息（admission 发射 suffix
        # 恢复分支时登记,该请求首 chunk 所在列车消费后弹出）:
        #   suffix_start            驻留前缀层数（层段拆分界）
        #   suffix_ready_nodes_by_rank  suffix 恢复完成门（首 chunk suffix
        #                           层段的 arm_dependency 目标）
        self._partial_first_chunk = {}
        self._tag_allocator = TransferTagAllocator()
        # 每 request 的 action_sequence 账本（action 名含 _action{seq:03d}_，
        # 跨该 request 的全部 transfer 递增）——在线各段发射共享同一计数器，保证节点
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

    def emit_admission_batch(self, request_plan: dict) -> None:
        """准入动作发射（ARRIVAL 边界；拼 batch 改造,2026-08-22）：
        到达/间隔 timer gates + history_evictions + history_transfer
        （或 partial 前缀两段式迁移：prefix noc_migrate → prefix ready
        barrier → suffix remote_load 恢复 → suffix ready barrier）+
        prefill_evictions + prefill 屏障。

        prefill 主体（chunk 序列）与 PREFILL_DRAIN watch 不再在此发射——
        移入实例迭代列车（emit_iteration_train 的折叠体与 drain 标记）；
        本方法无 watch 返回值（调度器在列车发射处注册）。[frontier 接续
        裁决,strategy 死锁修复统一（2026-08-16）] 的无条件接续语义不变：
        准入动作链到该 rank 当前 frontier（实例列车在飞时物理排在列车后）。
        partial 恢复分支沿用 chain checkpoint/restore：suffix 恢复分支自
        checkpoint 起并行悬挂，主链（以及后续列车体）不受其阻塞；首
        chunk 的 suffix 层段在列车内 arm 依赖 suffix ready 节点（跨批次
        边，经持久 (rank,id) 映射解析）——两段式流水语义原样保留。"""
        if self.config.trace_granularity != "request_aggregated":
            raise RuntimeError(
                "online emission supports request_aggregated granularity only "
                f"(got {self.config.trace_granularity!r})")
        self._set_context(request_plan, "prefill", 0)
        marker = self._mark()
        self._emit_admission(request_plan)
        self._collect(marker)

    def emit_iteration_train(self, train_plan: dict) -> dict:
        """发射一趟实例迭代列车（拼 batch 改造核心,2026-08-22;设计
        文档 §3.2"迭代列车聚合发射";母本 sh_1.0 定型版同构）。

        train_plan（调度器冻结的成员快照）字段：
          train_id          批命名空间 id（"batch_train_i<实例>_<序号>"；
                            共享体节点归属，物理完成与逻辑请求完成分离）
          instance_index    列车所在实例
          stage             "decode"（有 decode 成员，含混合迭代）或
                            "prefill"（纯 prefill 列车）
          joiners           新成员 request_plan 列表（含 decode_evictions/
                            prefill_decode_transfer/prefill_drain_block_ends
                            {rank: drain 列车 barrier 节点 id}）——迁移
                            节点先于列车体，经共享 readiness barrier 栅栏
          pass_spans        成员×迭代展开的 (tokens, kv) 平铺列表（partial
                            恢复列车 layout：[首 chunk] + [各成员第 1 迭代
                            span] + [其余 chunk + 成员剩余 span]）
          iterations        迭代数（= weight_passes：权重字节 ×迭代数，
                            与批成员数无关；激活/KV/AR 逐 span 精确）
          partial_first_chunk_count  partial 恢复列车的前缀组 span 数
                            （1 + 成员数；缺省 None = 非 partial 列车）
          drain_members     本列车内完成最后 prefill chunk 的请求 plan 列表
          exit_members      本列车内退出 decode 的成员 plan 列表

        每实例 rank 上的结构（链序）：
          [joiner 迁移 ...] → [共享 readiness barrier] → 17 类聚合体节点
          （transformer_pass_aggregated, weight_passes=iterations;partial
          恢复列车首 chunk 拆 prefix/suffix 层段两段发射）→
          [drain 标记 ...] → [exit 标记 ...] → 共享 end barrier。

        返回 {"drain_members": {request_id: {rank: 标记节点 id}},
              "exit_members": {request_id: {rank: 标记节点 id}},
              "block_ends": {rank: end barrier 节点 id}}——标记即
        PREFILL_DRAIN / DECODE_COMPLETION watch 成员（barrier 前末节点
        口径：标记是列车体后、end barrier 前的真实节点）；块末账本
        _block_ends[req]["seg1"]/["seg2"] = 本列车 post-barrier 节点
        （decode_evictions/completion_evictions 触发门与下一 turn
        interval gate after_node_id 的来源，五仓一致口径不随 watch 锚点
        变化）。"""
        if self.config.trace_granularity != "request_aggregated":
            raise RuntimeError(
                "online emission supports request_aggregated granularity only "
                f"(got {self.config.trace_granularity!r})")
        instance_index = train_plan["instance_index"]
        group = self.group_by_index[instance_index]
        train_id = train_plan["train_id"]
        stage = train_plan["stage"]
        generation = 1 if stage == "decode" else 0
        iterations = int(train_plan["iterations"])
        marker = self._mark()

        # ---- joiner 迁移（触发门 = 该成员 drain 列车的 post-barrier
        #      块末；上下文 (joiner, decode, 1) = decode_start 指标锚点） ----
        joiners = list(train_plan.get("joiners", ()))
        for joiner in joiners:
            self._set_context(joiner, "decode", 1)
            prefix = _prefix_of(joiner)
            drain_gates = joiner.get("prefill_drain_block_ends") or {}
            prefill_group = self.group_by_index[
                joiner["prefill_instance_index"]]
            decode_eviction_trigger = TransferTriggerGate(
                control_instance_index=joiner["prefill_instance_index"],
                node_gates=tuple(
                    drain_gates[rank] for rank in prefill_group.ranks),
            )
            for transfer in (joiner.get("decode_evictions") or ()):
                self._emit_plan_transfer(
                    joiner, kv_transfer_from_log(transfer),
                    "decode_evictions", trigger_gate=decode_eviction_trigger)
            prefill_decode_transfer = joiner.get("prefill_decode_transfer")
            if prefill_decode_transfer is None:
                raise RuntimeError(
                    "joiner is missing its Prefill-to-Decode KV action")
            self._emit_plan_transfer(
                joiner, kv_transfer_from_log(prefill_decode_transfer),
                "prefill_decode_transfer")

        # ---- 共享 readiness barrier（仅在有 joiner 时发射；无 joiner 的
        #      列车成员 KV 已就绪，无需再栅栏） ----
        if joiners:
            _emit_tp_readiness_barrier(
                builders=self.builders,
                group=group,
                name=f"{train_id}_decode_kv_ready_barrier",
            )

        # ---- 起始标记节点（指标锚点；§3.5：decode_start = 请求加入后
        #      第一个迭代所在列车节点；prefill_start = 请求首个 chunk
        #      所在列车的首节点。joiner 迁移零节点（如 local_hit）时这是
        #      唯一锚点；迁移有节点时 min-tick 语义取更早者，不冲突） ----
        for joiner in joiners:
            self._set_context(joiner, "decode", 1)
            for rank in group.ranks:
                self._emit_train_marker(
                    rank, f"{train_id}_join_"
                    f"{sanitize_node_prefix(joiner['request_id'])}")
        prefill_start_member = train_plan.get("prefill_start_member")
        if prefill_start_member is not None:
            self._set_context(prefill_start_member, "prefill", 0)
            for rank in group.ranks:
                self._emit_train_marker(
                    rank, f"{train_id}_pstart_"
                    f"{sanitize_node_prefix(prefill_start_member['request_id'])}")

        # ---- 折叠列车体（17 类聚合节点；weight_passes = 迭代数）----
        # sh_2.0 特性保留（拼 batch 改造,2026-08-22）：partial 前缀两段式
        # 迁移的请求，其首 chunk 所在列车仍按层段拆分——prefix 层段
        # （resident 前缀层,已在目标 HBM）不等 suffix 恢复，suffix 层段
        # arm 依赖 admission 记录的 suffix ready 节点（两段式流水）；
        # 首 chunk + 各成员第 1 迭代 span 进前缀组（weight_passes=1 的
        # 两段层发射，激活/KV 按层段分列、权重恰一份），其余 span 进
        # 剩余段（weight_passes = iterations-1）。
        pass_spans = list(train_plan["pass_spans"])
        drain_plan_members = list(train_plan.get("drain_members", ()))
        # partial 账本按"首 chunk 所在列车"弹出(与 plan 的 partial 判据
        # head_first_chunk 同源):T_max 截断下首 chunk 列车与 drain 列车
        # 可能分属不同列车,按 drain 弹出会漏配(2026-08-22 增量修正)。
        partial_info = None
        if prefill_start_member is not None:
            partial_info = self._partial_first_chunk.pop(
                prefill_start_member["request_id"], None)
        partial_count = train_plan.get("partial_first_chunk_count")
        if (partial_info is None) != (partial_count is None):
            raise RuntimeError(
                "partial train plan and admission ledger disagree on the "
                "first-chunk split")
        for builder in self.builders.values():
            builder.set_context(train_id, stage, generation)
        tensor_parallel = len(group.ranks)
        aggregate_arguments = {
            "layers": self.config.layers,
            "hidden_size": self.config.hidden_size,
            "ffn_size": self.config.ffn_size,
            "tensor_parallel": tensor_parallel,
            "pg_name": group.pg_name,
            "vocab_size": self.config.vocab_size,
            "bytes_per_elem": self.config.bytes_per_elem,
            "num_heads": self.config.num_heads,
            "mlp_variant": self.config.mlp_variant,
        }
        for relative_rank, rank in enumerate(group.ranks):
            arguments = dict(
                aggregate_arguments, tensor_parallel_rank=relative_rank)
            if partial_info is not None:
                if not pass_spans:
                    raise RuntimeError(
                        "partial train carries no first-chunk span")
                suffix_start = partial_info["suffix_start"]
                suffix_ready = partial_info["suffix_ready_nodes_by_rank"]
                if rank not in suffix_ready:
                    raise RuntimeError(
                        "partial train rank missing its suffix ready gate")
                first_group = pass_spans[:partial_count]
                rest_spans = pass_spans[partial_count:]
                transformer_pass_aggregated(
                    self.builders[rank],
                    phase=f"{train_id}_first_chunk_prefix",
                    pass_spans=first_group,
                    layer_start=0,
                    layer_end=suffix_start,
                    include_output=False,
                    weight_passes=1,
                    **arguments,
                )
                self.builders[rank].arm_dependency(suffix_ready[rank])
                transformer_pass_aggregated(
                    self.builders[rank],
                    phase=f"{train_id}_first_chunk_suffix",
                    pass_spans=first_group,
                    layer_start=suffix_start,
                    layer_end=self.config.layers,
                    include_output=True,
                    weight_passes=1,
                    **arguments,
                )
                if rest_spans:
                    transformer_pass_aggregated(
                        self.builders[rank],
                        phase=f"{train_id}_remaining_aggregated",
                        pass_spans=rest_spans,
                        weight_passes=iterations - 1,
                        **arguments,
                    )
            else:
                transformer_pass_aggregated(
                    self.builders[rank],
                    phase=train_id,
                    pass_spans=pass_spans,
                    weight_passes=iterations,
                    **arguments,
                )

        # ---- drain / exit 标记（列车体后、end barrier 前；每成员每 rank
        #      1 个小节点，承载该请求的 PREFILL_DRAIN / DECODE_COMPLETION
        #      watch 与指标 end 锚点；物理完成时刻 = 标记完成时刻） ----
        drain_members = {}
        for member in drain_plan_members:
            request_id = member["request_id"]
            self._set_context(member, "prefill", 0)
            drain_members[request_id] = {
                rank: self._emit_train_marker(
                    rank, f"{train_id}_drain_"
                    f"{sanitize_node_prefix(request_id)}")
                for rank in group.ranks
            }
        exit_members = {}
        for member in train_plan.get("exit_members", ()):
            request_id = member["request_id"]
            self._set_context(member, "decode", 1)
            exit_members[request_id] = {
                rank: self._emit_train_marker(
                    rank, f"{train_id}_exit_"
                    f"{sanitize_node_prefix(request_id)}")
                for rank in group.ranks
            }

        # ---- 共享 end barrier（每列车一个，替代每请求一个） ----
        for builder in self.builders.values():
            builder.set_context(train_id, stage, generation)
        # 哨兵标记(T_max 截断且无自然 drain/exit 标记的列车):
        #      request_id = train_id(批命名空间),固定 (prefill, 0) 单事件
        #      通道;fire 后经 PREFILL_DRAIN 通道送回,调度器按 train_id 核销。
        sentinel_members = {}
        if train_plan.get("sentinel"):
            for builder in self.builders.values():
                builder.set_context(train_id, "prefill", 0)
            sentinel_members = {
                rank: self._emit_train_marker(
                    rank, f"{train_id}_sentinel")
                for rank in group.ranks
            }

        for rank in group.ranks:
            self.builders[rank].all_reduce(
                f"{train_id}_end_barrier",
                iterations,
                group.pg_name,
            )
        block_ends = {
            rank: self.builders[rank].previous_id for rank in group.ranks}
        if any(node_id is None for node_id in block_ends.values()):
            raise RuntimeError("train end barrier node IDs were not generated")

        # ---- 块末账本（post-barrier 口径，五仓一致）：drain 成员写 seg1
        #      （decode_evictions 触发门），exit 成员写 seg2（completion_
        #      evictions 触发门 + 下一 turn interval gate after_node_id） ----
        for member in drain_plan_members:
            self._block_ends.setdefault(member["request_id"], {})[
                "seg1"] = dict(block_ends)
        for member in train_plan.get("exit_members", ()):
            self._block_ends.setdefault(member["request_id"], {})[
                "seg2"] = dict(block_ends)

        self._collect(marker)
        return {
            "drain_members": drain_members,
            "exit_members": exit_members,
            "sentinel_members": sentinel_members,
            "block_ends": block_ends,
        }

    def _emit_train_marker(self, rank: int, name: str) -> int:
        """列车标记节点（每 rank 1 个小 COMP 节点；上下文由调用方设置）。"""
        self.builders[rank].comp(name, 1, 1)
        return self.builders[rank].previous_id

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

    def _emit_plan_transfer(self, request_plan: dict, transfer: KVTransfer,
                            stage: str, *, gate=None,
                            trigger_gate=None) -> dict:
        """发射一个 KV 迁移动作（准入/列车 joiner/completion 段共用；
        action 序号跨该 request 的全部段共享，保证 canonical 命名稳定）。"""
        prefix = _prefix_of(request_plan)
        action_name = (
            f"{prefix}_{stage}_action"
            f"{self._next_action_sequence(request_plan):03d}_"
            f"{sanitize_node_prefix(transfer.session_id)}_"
            f"{transfer.kind}"
        )
        record = _emit_kv_transfer(
            config=self.config,
            builders=self.builders,
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

    def _emit_admission(self, request_plan: dict) -> None:
        """准入动作发射（ARRIVAL 边界；原 _emit_prefill 的动作部分,
        拼 batch 改造 2026-08-22 起 prefill 主体移入实例迭代列车）。"""
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
            # turn-0 request 在本段发射后成为本 session 的 pending request
            # （remote_store 折算目标）。
            self.pending_request_by_session[request_plan["session_id"]] = (
                request_plan["request_id"])
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

        # ---- history 逐出（trigger = 到达/history gate）----
        history_eviction_trigger = TransferTriggerGate(
            control_instance_index=pending_gate.source_instance_index,
            node_gates=pending_gate.timer_gates,
        )
        for transfer in request_plan["history_evictions"]:
            self._emit_plan_transfer(
                request_plan, kv_transfer_from_log(transfer),
                "history_evictions", trigger_gate=history_eviction_trigger)

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
            self._emit_plan_transfer(
                request_plan, history_transfer, "history_transfer",
                gate=pending_gate)

        for transfer in request_plan["prefill_evictions"]:
            self._emit_plan_transfer(
                request_plan, kv_transfer_from_log(transfer),
                "prefill_evictions")

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
                self._emit_plan_transfer(
                    request_plan, history_prefix_transfer,
                    "history_prefix_transfer", gate=pending_gate)
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
            history_record = self._emit_plan_transfer(
                request_plan, history_transfer, "history_transfer",
                gate=branch_gate)
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
            # 拼 batch 改造（2026-08-22）：两段式流水信息登记——首 chunk
            # 所在列车（emit_iteration_train）按层段拆分发射，suffix 层段
            # arm 依赖 suffix ready 节点；列车消费后弹出（恰一次）。
            self._partial_first_chunk[request_plan["request_id"]] = {
                "suffix_start": history_before.resident_prefix_layers,
                "suffix_ready_nodes_by_rank": dict(suffix_ready_nodes_by_rank),
            }
        else:
            _emit_tp_readiness_barrier(
                builders=builders,
                group=prefill_group,
                name=f"{prefix}_prefill_kv_ready_barrier",
            )
        # prefill 主体（chunk spans + end barrier）自拼 batch 改造
        # （2026-08-22）起移入 emit_iteration_train 的折叠体与 drain 标记；
        # 此处止于准入动作（到达 gates/历史迁移/逐出/屏障）。

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

    def _emit_completion(self, request_plan: dict,
                         following_plan: dict) -> None:
        builders = self.builders
        decode_group = self.group_by_index[
            request_plan["decode_instance_index"]]

        # 拼 batch 改造（2026-08-22）：decode 完成块末 = 退出列车 end
        # barrier（_block_ends["seg2"] 的 post-barrier 口径，触发门与
        # 下一 turn interval gate 的 after_node_id 同源；五仓一致）。
        decode_completion_nodes = self._block_ends.get(
            request_plan["request_id"], {}).get("seg2", {})
        if not decode_completion_nodes:
            raise RuntimeError("completed request has no decode train block end")

        completion_eviction_trigger = TransferTriggerGate(
            control_instance_index=request_plan["decode_instance_index"],
            node_gates=tuple(
                decode_completion_nodes[rank] for rank in decode_group.ranks),
        )
        for transfer in request_plan["completion_evictions"]:
            self._emit_plan_transfer(
                request_plan, kv_transfer_from_log(transfer),
                "completion_evictions",
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
