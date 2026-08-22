#!/usr/bin/env python3
"""graph_batch_builder.py -- 在线 GraphBatch 构图器(方案 §4 步骤 1-8 操作 4)。

阶段 1 最关键的对齐点:复用共享发射原语的节点结构。per-request 发射
(generate_face_trace.py 模块级函数)由
助手函数组成(_emit_control_trigger / _paired_transfer / _emit_prefill_stage /
transformer_pass_aggregated)——本模块直接 import 它们,用 OnlineTraceBuilder
(与 TraceBuilder 同构的在线侧 builder)驱动,保证:

  - 节点属性、插入顺序、rank ownership 和跨 request 链结构稳定(节点级审计口径);
  - per-rank 节点 id 跨批次全局递增;
  - interval gate 的 after_node_id 指向上一同 session request 所在列车 end
    barrier 节点 id(completion_gates 账本,跨批次解析)。

拼 batch 改造(2026-08-22,设计文档《层次 B Continuous Batching 改造》§3.2;
sh_1.0 定型版为母本,face 发射原语适配):层次 B 从"请求级大段串行"重构为
"实例迭代级列车"——decode 互拼、decode 与 prefill chunk 混拼、chunk 之间
不拼、批成员只在迭代(列车)边界变化。strategy 路径的两段发射:
  - 准入动作(emit_admission_batch)= 到达/interval timer gates + history
    迁移(NOC_MIGRATE)或 control 触发链 + history readiness 屏障;prefill
    主体(recompute 段 + 当前段 chunk)不再在此发射——移入实例迭代列车;
  - 迭代列车(emit_iteration_train)= joiner 的 prefill→decode 迁移(3000
    类 paired transfer,face 统一实例同实例时零节点)+ join/pstart 起始
    标记(指标锚点)+ 折叠列车体(成员×迭代 span,weight_passes = 迭代数:
    权重每迭代只读一次)+ drain/exit/哨兵标记节点(挂 PREFILL_DRAIN /
    DECODE_COMPLETION watch)+ 每列车一个共享 end barrier;
  - completion_gates 账本:exit 成员所在列车的 end barrier 即下一同 session
    turn 的 interval gate after_node_id 来源。

在线语义差异(刻意,注释标注):
  - timer gate 始终发射节点(结构保留)但 runtime_ns=0:到达时间由 C++ 的
    arrival alarm(future_alarms)替代,gate 不再等待——保持时长会双重等待;
  - watch 锚点(拼 batch 改造起由列车标记承载):PREFILL_DRAIN = drain 标记
    节点(列车体后、end barrier 前),DECODE_COMPLETION/REQUEST_COMPLETE =
    exit 标记节点(同位置);C++ watch fire 自动同时推 DECODE_COMPLETION +
    REQUEST_COMPLETE(共享机制不变);
  - legacy 变体(10.6)不经列车:emit_prefill_batch_legacy / emit_decode_batch
    保留旧 request-aggregated 结构(legacy 语义保留对象,不得抹平)。
"""

import os
import sys

# --------------------------------------------------------------------------
# import 路径:本文件位于 workload/llama2_7b_inference/online/,共享配置与
# 发射模块在上一级。路径只做 import 用途(红线:generate_face_trace.py / face_
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
from generate_face_trace import (  # noqa: E402
    _emit_control_trigger,
    _paired_transfer,
    kv_cache_bytes_for_tokens,
    sanitize_node_prefix,
)
from generate_face_trace import NOC_MIGRATE, RECOMPUTE  # noqa: E402


class OnlineTraceBuilder:
    """与共享 TraceBuilder API 同构的在线侧每-rank builder。

    同一接口面:timer_gate / arm_timer_gate / comp / all_reduce / comm_send /
    comm_recv / next_id / previous_id / node_count——共享助手函数可直接驱动。
    与 TraceBuilder 的差异:节点发射为 GraphBatch nodes[] dict(而非 ChakraNode
    对象),依赖记录为 parent_edges[](而非节点内联 data_deps);timer_gate
    忽略 duration(runtime_ns=0,alarm 替代等待)。

    per-rank id 自 0 起全局递增并跨批次保留。
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

        与共享 TraceBuilder.timer_gate 的结构契约一致:
          - duration_ns == 0 → 直接返回 after_node_id,不发射节点
            (inter_request_interval_ns==0 的 request 不应多出 interval gate);
          - 否则直接创建节点(不经 _new_node)——不链 previous_id、不消费
            pending_extra_dependencies、不更新 previous_id;仅
            after_node_id 依赖(interval gate 依赖上一 request 完成
            barrier)。
        到达时刻/interval 的等待由 C++ arrival alarm(future_alarms)承担——
        gate 只保留结构与依赖,保持时长会双重
        等待(见步骤 1-8 设计分析)。
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
        # completion_gates 保存动态跨 turn interval gate 的 after_node_id 来源。
        self.completion_gates = {}
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

    def emit_admission_batch(self, request_plan: dict) -> None:
        """发射 request 的准入动作(ARRIVAL 边界;拼 batch 改造,2026-08-22):
        到达/interval timer gates + history 迁移(NOC_MIGRATE paired transfer)
        或 control 触发链 + history readiness 屏障。

        prefill 主体(recompute 段 + 当前段 chunk 序列)与 PREFILL_DRAIN
        watch 不再在此发射——移入实例迭代列车(emit_iteration_train 的折叠
        体与 drain 标记);本方法无 watch 返回值(调度器在列车发射处注册)。
        [根因 #5 裁决,strategy 死锁修复统一(2026-08-19)] 的无条件 frontier
        接续语义不变:准入动作链到该 rank 当前 frontier(实例列车在飞时物理
        排在列车后)。"""
        if self.config.trace_granularity != "request_aggregated":
            raise RuntimeError(
                "online emission supports request_aggregated granularity only "
                f"(got {self.config.trace_granularity!r})")
        self._set_context(request_plan, "prefill", 0)
        marker = self._mark()
        self._emit_admission_actions(request_plan)
        self._collect(marker)

    def emit_iteration_train(self, train_plan: dict) -> dict:
        """发射一趟实例迭代列车(拼 batch 改造核心,2026-08-22;设计文档
        §3.2"迭代列车聚合发射";sh_1.0 母本同构,face 发射原语适配)。

        train_plan(调度器冻结的成员快照)字段:
          train_id          批命名空间 id("batch_train_i<实例>_<序号>";
                            共享体节点归属,物理完成与逻辑请求完成分离)
          instance_index    列车所在实例
          stage             "decode"(有 decode 成员,含混合迭代)或
                            "prefill"(纯 prefill 列车)
          joiners           新成员 request_plan 列表(prefill→decode 迁移
                            3000 类 paired transfer,face 统一实例同实例时
                            零节点)——迁移节点先于列车体
          pass_spans        成员×迭代展开的 (tokens, kv) 平铺列表
          iterations        迭代数(= weight_passes:权重字节 ×迭代数,
                            与批成员数无关;激活/KV/AR 逐 span 精确)
          prefill_start_member 队列头首个 chunk 在本列车时的成员 plan
                            (prefill_start 指标锚点)
          drain_members     本列车内完成最后 prefill chunk 的请求 plan 列表
          exit_members      本列车内退出 decode 的成员 plan 列表
          sentinel          T_max 截断且无自然 drain/exit 标记时为真
                            (哨兵标记承载完成信号)

        每实例 rank 上的结构(链序;face 与母本的差异点:joiner 迁移用
        face 的 _paired_transfer(3000)而非母本 _emit_kv_transfer,且无
        共享 readiness 屏障——face 统一实例同实例迁移零节点,跨实例迁移
        经 per-rank 物理链天然栅栏,face 旧 decode 发射同款结构):
          [joiner 迁移 ...] → [join/pstart 起始标记 ...] → 17 类聚合体节点
          (transformer_pass_aggregated, weight_passes=iterations) →
          [drain 标记 ...] → [exit 标记 ...] → [哨兵标记 ...] → 共享 end
          barrier。

        返回 {"drain_members": {request_id: {rank: 标记节点 id}},
              "exit_members": {request_id: {rank: 标记节点 id}},
              "sentinel_members": {rank: 标记节点 id} 或 {},
              "block_ends": {rank: end barrier 节点 id}}——标记即
        PREFILL_DRAIN / DECODE_COMPLETION watch 成员(标记是列车体后、
        end barrier 前的真实节点);block_ends = exit 成员所在列车 end
        barrier(completion_gates 账本写入口径,下一同 session turn 的
        interval gate after_node_id 来源)。"""
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

        # ---- joiner 迁移(face 3000 类 paired transfer;上下文
        #      (joiner, decode, 1) = decode_start 指标锚点) ----
        joiners = list(train_plan.get("joiners", ()))
        for joiner in joiners:
            self._set_context(joiner, "decode", 1)
            prefill_group = self.group_by_index[
                joiner["prefill_instance_index"]]
            decode_group = self.group_by_index[
                joiner["decode_instance_index"]]
            _paired_transfer(
                config=self.config, builders=self.builders,
                queue_index=joiner["queue_index"], category=3000,
                name="{}_prefill_to_decode_kv".format(_prefix_of(joiner)),
                source_group=prefill_group, target_group=decode_group,
                total_bytes=kv_cache_bytes_for_tokens(
                    self.config.model, joiner["prefill_context_tokens"]),
            )

        # ---- 起始标记节点(指标锚点;§3.5:decode_start = 请求加入后
        #      第一个迭代所在列车节点;prefill_start = 请求首个 chunk
        #      所在列车的首节点。joiner 迁移零节点(face 统一实例)时这是
        #      唯一锚点;迁移有节点时 min-tick 语义取更早者,不冲突) ----
        for joiner in joiners:
            self._set_context(joiner, "decode", 1)
            for rank in group.ranks:
                self._emit_train_marker(
                    rank, "{}_join_{}".format(
                        train_id, sanitize_node_prefix(
                            joiner["request_id"])))
        prefill_start_member = train_plan.get("prefill_start_member")
        if prefill_start_member is not None:
            self._set_context(prefill_start_member, "prefill", 0)
            for rank in group.ranks:
                self._emit_train_marker(
                    rank, "{}_pstart_{}".format(
                        train_id, sanitize_node_prefix(
                            prefill_start_member["request_id"])))

        # ---- 折叠列车体(17 类聚合节点;weight_passes = 迭代数) ----
        for builder in self.builders.values():
            builder.set_context(train_id, stage, generation)
        tensor_parallel = len(group.ranks)
        for relative_rank, rank in enumerate(group.ranks):
            transformer_pass_aggregated(
                self.builders[rank],
                phase=train_id,
                pass_spans=train_plan["pass_spans"],
                layers=self.config.layers,
                hidden_size=self.config.hidden_size,
                ffn_size=self.config.ffn_size,
                tensor_parallel=tensor_parallel,
                pg_name=group.pg_name,
                vocab_size=self.config.vocab_size,
                bytes_per_elem=self.config.bytes_per_elem,
                num_heads=self.config.num_heads,
                tensor_parallel_rank=relative_rank,
                mlp_variant=self.config.mlp_variant,
                weight_passes=iterations,
            )

        # ---- drain / exit 标记(列车体后、end barrier 前;每成员每 rank
        #      1 个小节点,承载该请求的 PREFILL_DRAIN / DECODE_COMPLETION
        #      watch 与指标 end 锚点;物理完成时刻 = 标记完成时刻) ----
        drain_members = {}
        for member in train_plan.get("drain_members", ()):
            request_id = member["request_id"]
            self._set_context(member, "prefill", 0)
            drain_members[request_id] = {
                rank: self._emit_train_marker(
                    rank, "{}_drain_{}".format(
                        train_id, sanitize_node_prefix(request_id)))
                for rank in group.ranks
            }
        exit_members = {}
        for member in train_plan.get("exit_members", ()):
            request_id = member["request_id"]
            self._set_context(member, "decode", 1)
            exit_members[request_id] = {
                rank: self._emit_train_marker(
                    rank, "{}_exit_{}".format(
                        train_id, sanitize_node_prefix(request_id)))
                for rank in group.ranks
            }

        # ---- 哨兵标记(T_max 截断且无自然 drain/exit 标记的列车):
        #      request_id = train_id(批命名空间),C++ eligibility 经
        #      batch_train_ 前缀放行;fire 后经四类 reason 通道送回,
        #      调度器按 train_id 核销 ----
        sentinel_members = {}
        if train_plan.get("sentinel"):
            # 哨兵 watch 固定 (train_id, prefill, 0):decode stage 的 fire
            # 会同时推 REQUEST_COMPLETE(main_online 的 stage→reason 映射)
            # 并触发 ServiceCoordinator 完成计账下溢;prefill stage 只推
            # 一条 PREFILL_DRAIN,Python 侧按 train_id 路由核销。
            for builder in self.builders.values():
                builder.set_context(train_id, "prefill", 0)
            sentinel_members = {
                rank: self._emit_train_marker(
                    rank, "{}_sentinel".format(train_id))
                for rank in group.ranks
            }

        # ---- 共享 end barrier(每列车一个,替代每请求一个) ----
        for builder in self.builders.values():
            builder.set_context(train_id, stage, generation)
        for rank in group.ranks:
            self.builders[rank].all_reduce(
                "{}_end_barrier".format(train_id),
                iterations,
                group.pg_name,
            )
        block_ends = {
            rank: self.builders[rank].previous_id for rank in group.ranks}
        if any(node_id is None for node_id in block_ends.values()):
            raise RuntimeError("train end barrier node IDs were not generated")

        # ---- completion_gates 账本:exit 成员所在列车的 end barrier 即
        #      下一同 session turn 的 interval gate after_node_id(face
        #      旧 decode end barrier 的同款口径) ----
        for member in train_plan.get("exit_members", ()):
            self.completion_gates[member["session_id"]] = (
                instance_index, dict(block_ends))

        self._collect(marker)
        return {
            "drain_members": drain_members,
            "exit_members": exit_members,
            "sentinel_members": sentinel_members,
            "block_ends": block_ends,
        }

    def _emit_train_marker(self, rank: int, name: str) -> int:
        """列车标记节点(每 rank 1 个小 COMP 节点;上下文由调用方设置)。"""
        self.builders[rank].comp(name, 1, 1)
        return self.builders[rank].previous_id

    def emit_decode_batch(self, request_plan: dict) -> dict:
        """发射 request 的 decode 整段(含 prefill_to_decode transfer 3000 /
        decode 整段 / decode_request_end_barrier)。

        拼 batch 改造(2026-08-22)起 strategy 路径不经此方法(decode 移入
        迭代列车,见 emit_iteration_train);保留供 legacy 变体(10.6,
        request-aggregated 旧结构,legacy 语义保留对象)。

        返回 DECODE_COMPLETION watch 成员:{rank: end barrier 前的 decode
        末节点 id}(共享 DECODE_COMPLETION 锚点口径)。
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

    def _emit_admission_actions(self, request_plan: dict) -> None:
        """发射在线请求的准入动作:到达/interval gates、history 迁移与
        readiness 屏障(拼 batch 改造,2026-08-22:recompute 段与当前
        prefill 段的 chunk 主体移入实例迭代列车,见 emit_iteration_train)。

        [根因 #5 裁决,strategy 死锁修复统一(2026-08-19)]:strategy 不做
        任何块末恢复/段内清链——per-rank previous_id 无条件接续当前
        frontier,per-rank 发行序 = 全局发射序,跨实例 P2P 与 collective
        参与序不可能反转成环;同 session 串行化由 interval gate
        (after_node_id 显式编码)保留。"""
        builders = self.builders
        prefill_group = self.group_by_index[request_plan["prefill_instance_index"]]
        prefix = _prefix_of(request_plan)
        request = _request_spec(self.config, request_plan)
        if request_plan["turn_index"] == 0:
            if request.session_arrival_time_ns is None:
                raise RuntimeError("first request lost its session arrival")
            # 在线:timer gate runtime=0,到达时间由 C++ arrival alarm 替代。
            # duration 传真实 session_arrival_time_ns；0 时不发射 gate 节点。
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
            # duration 传真实 inter_request_interval_ns；0 时不发射 interval
            # gate 节点，直接以 after_node_id 建立依赖。
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
        for rank in prefill_group.ranks:
            builders[rank].all_reduce(
                "{}_history_tp_ready_barrier".format(prefix), 1,
                prefill_group.pg_name)
        # prefill 主体(recompute 段 + 当前段 chunk spans)自拼 batch 改造
        # (2026-08-22)起移入 emit_iteration_train 的折叠体与 drain 标记;
        # 此处止于准入动作(到达 gates/历史迁移/屏障)。

    def _emit_decode(self, request_plan: dict) -> dict:
        """发射动态 GraphBatch 的 transfer-3000、decode 和 end-barrier 块。"""
        builders = self.builders
        prefill_group = self.group_by_index[request_plan["prefill_instance_index"]]
        decode_group = self.group_by_index[request_plan["decode_instance_index"]]
        prefix = _prefix_of(request_plan)
        # [frontier 接续裁决,strategy 死锁修复统一(2026-08-19,对齐 sh_1.0/
        # sh_2.0)] strategy **不做任何块末恢复/段内清链**:per-rank
        # previous_id 无条件接续当前 frontier，per-rank 发行序 = 全局发射序——
        # 任意两个发射段在
        # 所有共享 rank 上的相对次序一致,跨请求 P2P(send/recv tag 匹配)
        # 与 collective 参与序不可能反转成环。transfer3000 的 comm_send
        # (prefill rank)链当前 frontier(经 per-rank 全序传递性仍包含本
        # request 的 prefill 块末);comm_recv(decode rank)链当前 frontier
        # (跨 request 边,含上一 request 的 decode end barrier)。(2026-08-15 的
        # own-prefill-end 恢复 / None-restore 裁决自此废止。)
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
            # completion candidate = end barrier 前每 rank 的 decode 末节点
            # (DECODE_COMPLETION watch 成员)。
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


    # ------------------------------------------------- legacy 变体发射(10.6) --

    def emit_prefill_batch_legacy(self, request_plan: dict) -> dict:
        """legacy 变体的动态 GraphBatch prefill 整段。

        与 session_lru 路径的结构差异(legacy 语义,不得抹平):
          - turn-0:global_arrival_timer_gate(同构);
          - turn>0:history = kv_allocation.pieces 的逐 piece 迁移
            (KVAllocator 跨实例分片;无 recompute/control 触发链);
          - prefill:aggregated pass(chunks 折叠为 spans)+
            prefill_chunks_aggregated_end_barrier(bytes=pass_count)。
        返回 PREFILL_DRAIN watch 成员:{rank: 末个真实 prefill 节点}。
        """
        if self.config.trace_granularity != "request_aggregated":
            raise RuntimeError(
                "online emission supports request_aggregated granularity only "
                f"(got {self.config.trace_granularity!r})")
        self._set_context(request_plan, "prefill", 0)
        builders = self.builders
        prefill_group = self.group_by_index[request_plan["prefill_instance_index"]]
        prefix = _prefix_of(request_plan)
        request = _request_spec(self.config, request_plan)
        # strategy 保持物理跨 request 链。
        marker = self._mark()
        if request_plan["turn_index"] == 0:
            if request.session_arrival_time_ns is None:
                raise RuntimeError("first request lost its session arrival")
            timers = {
                rank: builders[rank].timer_gate(
                    "{}_global_arrival_timer_gate".format(prefix),
                    request.session_arrival_time_ns)
                for rank in prefill_group.ranks
            }
            for rank in prefill_group.ranks:
                builders[rank].arm_timer_gate(timers[rank])
        else:
            # history pieces:cross-instance 分片逐 piece 迁移
            # (offline: HistoryPieceGate 列表,kv_allocation.pieces)。
            gates = {rank: None for rank in prefill_group.ranks}
            for piece_number, piece in enumerate(
                    request_plan.get("kv_allocation_pieces", ())):
                source_group = self.group_by_index[piece["instance_index"]]
                if piece["bytes"] == 0 and \
                        source_group.name == prefill_group.name:
                    continue
                _emit_control_trigger(
                    builders=builders,
                    queue_index=request_plan["queue_index"],
                    name="{}_history_piece{}_control".format(
                        prefix, piece_number),
                    source_group=prefill_group,
                    target_group=source_group,
                    timer_gates=gates,
                )
                gates = {rank: None for rank in source_group.ranks}
                _paired_transfer(
                    config=self.config, builders=builders,
                    queue_index=request_plan["queue_index"], category=1000,
                    name="{}_history_piece{}".format(prefix, piece_number),
                    source_group=source_group, target_group=prefill_group,
                    total_bytes=piece["bytes"],
                    timer_gates=gates,
                )
                gates = {rank: None for rank in prefill_group.ranks}
        # prefill aggregated(chunks -> spans)+ end barrier。p_chunk 取
        # legacy 标定常数(plan dict 携带;session_lru 路径不经此方法,
        # config.prefill_chunk_size=512 不是 legacy 的分块粒度)。
        p_chunk = int(request_plan.get("p_chunk") or self.config.prefill_chunk_size)
        spans = []
        processed = 0
        prefill_length = request_plan["prefill_length"]
        while processed < prefill_length:
            chunk_tokens = min(p_chunk, prefill_length - processed)
            spans.append((chunk_tokens,
                          request_plan["history_tokens_before"]
                          + processed + chunk_tokens))
            processed += chunk_tokens
        members = {}
        tp = len(prefill_group.ranks)
        from generate_trace import transformer_pass_aggregated  # noqa: E402
        for relative_rank, rank in enumerate(prefill_group.ranks):
            transformer_pass_aggregated(
                builders[rank],
                phase="{}_prefill_request_aggregated".format(prefix),
                pass_spans=tuple(spans), layers=self.config.layers,
                hidden_size=self.config.hidden_size,
                ffn_size=self.config.ffn_size, tensor_parallel=tp,
                pg_name=prefill_group.pg_name,
                vocab_size=self.config.vocab_size,
                bytes_per_elem=self.config.bytes_per_elem,
                num_heads=self.config.num_heads,
                tensor_parallel_rank=relative_rank,
                mlp_variant=self.config.mlp_variant,
            )
            members[rank] = builders[rank].previous_id
            builders[rank].all_reduce(
                "{}_prefill_chunks_aggregated_end_barrier".format(prefix),
                len(spans), prefill_group.pg_name)
        self._collect(marker)
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
