#!/usr/bin/env python3
"""graph_batch_builder.py -- 在线 GraphBatch 构图器(方案 §4 步骤 1-8 操作 4)。

阶段 1 最关键的对齐点:复用共享发射原语的节点结构。per-request 发射
(generate_face_trace.py 模块级函数)由
助手函数 _emit_control_trigger / _paired_transfer(自 generate_face_trace
import)与 generate_trace 的 transformer_pass_aggregated 组成——本模块
直接 import 它们,用 OnlineTraceBuilder
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
  - timer gate 在 duration_ns == 0 时不发射节点(timer_gate 直接返回
    前驱):到达时间由 C++ 的 arrival alarm(future_alarms)替代,gate
    不再等待——保持时长会双重等待;duration_ns > 0 时照常发射;
  - watch 锚点(拼 batch 改造起由列车标记承载):PREFILL_DRAIN = drain 标记
    节点(列车体后、end barrier 前),DECODE_COMPLETION/REQUEST_COMPLETE =
    exit 标记节点(同位置);C++ watch fire 自动同时推 DECODE_COMPLETION +
    REQUEST_COMPLETE(共享机制不变)。
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
from generate_face_trace import NOC_MIGRATE  # noqa: E402

def first_token_split_enabled() -> bool:
    """WP9 首步批拆分总开关（SH_FIRST_TOKEN_SPLIT，B4 起缺省 "0" 关）。

    缺省翻转（2026-08-26，主规格 §1.6 A 类处置）：B3_FACE 60s 决策
    等价门-2 失败——拆分的物理扰动（首列拆批 → tick 漂移，row 3 起
    +6.5us）在闭环逐轮放大，t=25.947s 处同 tick 处理顺序翻转并级联
    （1184/1454 请求实例选择翻转、1073/1454 decode 候选集翻转、
    train_id 多重集不等），ON/OFF 字节等价在 60s 窗不可达（2s 窗保持
    通过）。调度器 tie-break 属禁改区，无补救路径。证据：
    /tmp/slo_wps/gates/B3_FACE.DONE（gate2 FAIL）与
    /tmp/slo_wps/b3/FACE/（t3_full_off_split vs t3_full_split_on 对拍
    产物）。默认口径退回 proxy（first_token_source=train_interpolated，
    见 metrics_postprocess），本开关显式置 "1" 仍可启用拆分取 exact
    首 token（研究/对拍用）。

    "1" = 拆分开启：debut 列车两段式发射 + first_token 标记；
    其他值（含缺省）= 拆分完全关闭：调度器不拆列车、不发射 first_token
    标记，构图/决策/账本产物与拆分上线前逐字节一致（A/B 对拍的 OFF
    侧）。每次 _emit_train 现场读取（而非构造期缓存），测试与复跑可在
    进程内切换。
    """
    return os.environ.get("SH_FIRST_TOKEN_SPLIT", "0") == "1"


class OnlineTraceBuilder:
    """与共享 TraceBuilder API 同构的在线侧每-rank builder。

    同一接口面:timer_gate / arm_timer_gate / comp / all_reduce / comm_send /
    comm_recv / next_id / previous_id——共享助手函数可直接驱动。
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
        # M1 收集即释放(2026-08-29):这两个 list 只保存自上次 _collect
        # 以来尚未交付的记录；交付后立即 clear，跨批状态只由 next_id、
        # previous_id 与账本保存。禁止按 list 位置回读节点(id 来自
        # next_id 计数器,与位置无关)。
        self.nodes = []   # 本 rank 尚未交付的节点 dict(发射序)
        self.edges = []   # 本 rank 尚未交付的 parent edges
        # 当前 request-stage 反向索引上下文(每次 per-request 发射前设置)。
        self.request_id = ""
        self.stage = ""
        self.generation = 0
        # Optional capacity-release ordering hook installed by
        # GraphBatchBuilder.  Keeping it absent is the legacy emission path.
        self.before_capacity_comp = None

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
        previous_id = self.previous_id
        pending_dependencies = self.pending_extra_dependencies
        # The common serial chain has no extra dependency: avoid allocating a
        # temporary list and deduplication dict.  The multi-dependency fallback
        # deliberately keeps dict.fromkeys() for its stable first-seen order.
        if not pending_dependencies:
            if previous_id is not None:
                self.edges.append({
                    "rank": self.rank,
                    "from": previous_id,
                    "to": self.next_id,
                    "kind": "data",
                })
        elif len(pending_dependencies) == 1:
            if previous_id is not None:
                self.edges.append({
                    "rank": self.rank,
                    "from": previous_id,
                    "to": self.next_id,
                    "kind": "data",
                })
            dependency_id = pending_dependencies[0]
            if previous_id is None or dependency_id != previous_id:
                self.edges.append({
                    "rank": self.rank,
                    "from": dependency_id,
                    "to": self.next_id,
                    "kind": "data",
                })
        else:
            dependency_ids = []
            if previous_id is not None:
                dependency_ids.append(previous_id)
            dependency_ids.extend(pending_dependencies)
            for dependency_id in dict.fromkeys(dependency_ids):
                self.edges.append({
                    "rank": self.rank,
                    "from": dependency_id,
                    "to": self.next_id,
                    "kind": "data",
                })
        pending_dependencies.clear()
        self.previous_id = self.next_id
        self.next_id += 1
        self.nodes.append(node)
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
        return node["id"]

    def comp(self, name: str, num_ops: int, tensor_size: int,
             remote_read_size: int = 0, *,
             die_spec: dict = None, op_role: str = None) -> None:
        """COMP 节点发射(阶段 C 扩展,契约 §2 compute 键集)。

        die_spec 非 None 时附 compute.die_op_spec(dict;die 后端执行该
        节点,runtime_ns 保持缺省 0——die_op_spec 存在 ⇒ runtime_ns 必为
        0);op_role 缺省 "compute"(不写键,schema 缺省即 compute),仅
        marker/哨兵显式写 "control_marker"(契约 §7 裁定 8)。"""
        if (self.before_capacity_comp is not None
                and op_role != "control_marker"):
            self.before_capacity_comp()
        node = self._new_node(name, COMP_NODE)
        node["compute"]["num_ops"] = self._uint64(num_ops)
        node["compute"]["tensor_size"] = self._uint64(tensor_size)
        if op_role is not None:
            if op_role not in ("compute", "control_marker"):
                raise ValueError(
                    "op_role must be 'compute' or 'control_marker', got "
                    "{!r}".format(op_role))
            node["compute"]["op_role"] = op_role
        if die_spec is not None:
            if node["compute"]["runtime_ns"] != 0:
                raise ValueError(
                    "die_op_spec requires runtime_ns == 0 (contract "
                    "section 2)")
            node["compute"]["die_op_spec"] = die_spec
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


class GraphBatchBuilder:
    """在线构图器:持有 per-rank OnlineTraceBuilder(状态跨批次),按决策边界
    发射 prefill 整段 / decode 整段,并维护 completion_gates 账本。"""

    def __init__(self, config, *, source_release_capacity_gates=False):
        self.config = config
        if not isinstance(source_release_capacity_gates, bool):
            raise TypeError("source_release_capacity_gates must be bool")
        self.source_release_capacity_gates = source_release_capacity_gates
        if source_release_capacity_gates:
            # Finite producer: place new ACKs (and LRU pool transfers) above
            # every possible old stage tag, even beyond 1000 requests.
            max_tp = max(len(group.ranks) for group in config.inference_groups)
            release_tag_base = max(10_000_000,
                len(config.request_queue) * 10000 + 4000 + max_tp)
            if release_tag_base > 0xFFFFFFFF:
                raise OverflowError("source release communication tag space was exhausted")
            self._next_source_release_tag = release_tag_base
        # die 后端发射语境(阶段 C;None = legacy)。online_service.py 构造
        # 本构图器时不感知 backend——语境由 OnlineSchedulerBase 从 manifest
        # 的 compute-cost-model/die-model-config 键构造后在每批发射前经
        # set_die_context 注入(online_scheduler_base._begin_graph_batch;
        # manifest 是 backend 的唯一声明源,契约 §7 裁定 2/7)。
        self.die_context = None
        # strategy 模式保持物理跨 request 链。
        self.builders = {
            rank: OnlineTraceBuilder(
                rank, remote_operand_loads=config.remote_operand_loads)
            for rank in range(config.npus_count)
        }
        # Source HBM releases cross sessions and batches.  Keep them separate
        # from any same-session pool ordering ledger.
        self.pending_source_release_tails = {}
        if self.source_release_capacity_gates:
            for rank, builder in self.builders.items():
                builder.before_capacity_comp = (
                    lambda rank=rank:
                    self._consume_source_release_tails(rank))
        self.group_by_index = dict(enumerate(config.inference_groups))
        # session_id -> (decode_instance_index, {rank: end_barrier_id});
        # completion_gates 只保存下一同 session turn 尚未消费的 interval
        # gate 来源；turn>0 admission 或 terminal completion 后即释放。
        self.completion_gates = {}
        self.batch = None  # 当前批次累加器(由 begin_batch 建立)

    def set_die_context(self, die_context) -> None:
        """注入 die 发射语境(幂等;None 恢复 legacy 发射)。"""
        self.die_context = die_context

    # ------------------------------------------------------------- 批次 --

    def begin_batch(self) -> None:
        # 构图器累加器只产 nodes/parent_edges(及其私有 rank 账本);
        # watches/assignments/kv_actions/future_alarms 由变体策略写入
        # 在线骨架的 self._batch,build_graph_batch 组装时两边合并,
        # 发往 C++ 的 wire JSON 恒含四键(基类 _start_batch 模板)。
        self.batch = {
            "nodes": [],
            "parent_edges": [],
            # Private exact ledger: _collect already knows the source rank of
            # every appended node, so downstream GraphBatch metadata need not
            # rescan the complete node payload.
            "_touched_ranks": set(),
        }

    def _collect(self, marker: dict) -> None:
        """把各 builder 自 marker 起新增的节点/边并入本批次。"""
        for rank, builder in self.builders.items():
            node_mark, edge_mark = marker[rank]
            nodes = builder.nodes
            edges = builder.edges
            # 正常路径的 marker 为 0，直接 extend 避免临时 slice；非零
            # marker 仅保留本次新增尾部。所有 _mark() 都在同一发射调用内
            # 被单次 _collect() 消费，故旧前缀已在先前批次交付，可立即
            # clear 释放对节点/边 dict 的最后一层 builder 引用。
            if len(nodes) > node_mark:
                self.batch["_touched_ranks"].add(int(rank))
            self.batch["nodes"].extend(
                nodes if node_mark == 0 else nodes[node_mark:])
            self.batch["parent_edges"].extend(
                edges if edge_mark == 0 else edges[edge_mark:])
            nodes.clear()
            edges.clear()

    def _register_source_release_routes(self, routes) -> None:
        if not self.source_release_capacity_gates:
            return
        for route in routes or ():
            source_rank = route.get("source_rank")
            release_node_id = route.get("source_release_node_id")
            if (isinstance(source_rank, bool)
                    or not isinstance(source_rank, int)
                    or source_rank not in self.builders):
                raise RuntimeError(
                    "source-release route is missing a valid source rank")
            if (isinstance(release_node_id, bool)
                    or not isinstance(release_node_id, int)
                    or release_node_id < 0
                    or release_node_id >= self.builders[source_rank].next_id):
                raise RuntimeError(
                    "source-release route is missing an emitted source tail")
            self.pending_source_release_tails.setdefault(
                source_rank, []).append(release_node_id)

    def _source_release_ack_tag(self) -> int:
        tag = self._next_source_release_tag
        if tag > 0xFFFFFFFF:
            raise OverflowError("source release communication tag space was exhausted")
        self._next_source_release_tag += 1
        return tag

    def _take_source_release_tails(self, rank: int) -> tuple[int, ...]:
        if not self.source_release_capacity_gates:
            return ()
        return tuple(self.pending_source_release_tails.pop(rank, ()))

    def _take_source_release_arms(self, ranks) -> dict[int, tuple[int, ...]]:
        arms = {}
        for rank in ranks:
            tails = self._take_source_release_tails(int(rank))
            if tails:
                arms[int(rank)] = tails
        return arms

    def _consume_source_release_tails(self, rank: int) -> None:
        builder = self.builders[rank]
        for node_id in self._take_source_release_tails(rank):
            # FACE base predates the generic arm_dependency helper; this is
            # the same checked-by-construction per-rank dependency queue that
            # _new_node consumes when it emits the capacity COMP.
            builder.pending_extra_dependencies.append(node_id)

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
          first_token       WP9 首 token 观测(2026-08-26;缺省 None = 拆分
                            开关关闭或无 debut,行为与上线前逐字节一致):
                            "split": False 时仅做不拆车的标记增强——多
                            token debut 成员在列车体后挂 first_token 标记,
                            decode_length=1 的 debut 成员其 exit 标记改名
                            _exit_first_token_(C++ 名字子串锚点);"split":
                            True 时本方法拒绝,阶段 1/2 经 emit_train_
                            first_step / emit_train_remainder 发射

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
        first_token = train_plan.get("first_token")
        if first_token is not None and first_token.get("split"):
            raise RuntimeError(
                "split train plans must be emitted through "
                "emit_train_first_step/emit_train_remainder")
        marker = self._mark()
        self._emit_train_head(train_plan)
        self._emit_train_body(
            train_plan, list(train_plan["pass_spans"]),
            int(train_plan["iterations"]))
        result = self._emit_train_tail_markers(train_plan, first_token)
        self._collect(marker)
        return result

    def emit_train_first_step(self, train_plan: dict) -> dict:
        """WP9 首步批发射(2026-08-26;拆分阶段 1;WP9_CONTRACT §2 face
        混拼列车条款)。

        首步批 = 所有列车成员的第 1 个 span(decode 成员首迭代)+ prefill
        队头的第 1 个 chunk,加上锚定迭代位置在第 1 迭代内的 joiner 迁移
        (face 3000 类 paired transfer,整列发射时本就先于列车体)与
        join/pstart 起始标记(挂点语义与整列发射完全一致)。列车体以
        weight_passes=1 折叠(权重恰读一次),体后挂各多 token debut 成员
        的 first_token 标记节点(1-op COMP,名字含 "first_token" 子串——
        C++ 锚点按名字子串注册 code 8,取每 rank min tick);decode_length
        =1 的 debut 不另挂标记(其余量批 exit 标记改名携带 first_token
        子串,同节点双锚点)。

        本批不含任何请求级 watch/drain/exit/哨兵标记;唯一附加物是尾部
        的批命名空间唤醒标记(每 rank 1 个小节点,request_id = "<train_id>_
        first_step" 前缀 batch_train_,复用 C++ 哨兵 watch 通道):其 fire
        经 PREFILL_DRAIN 通道送回调度器,作为余量批的交付边界——没有它,
        无 watch 的首步批完成后不存在任何决策工作,C++ tick-end 门不会
        再交付,运行尾部(全部其余工作已排空)将永久等待(死锁)。这是对
        "首步批无任何 watch 标记"的必要工程化偏移:唤醒 watch 不挂任何
        请求、不触发核销/记账/决策(调度器按 first_step id 识别后无操作)。

        返回 {"first_token_members": {request_id: {rank: 标记节点 id}},
              "wakeup_members": {rank: 唤醒标记节点 id}}。"""
        if self.config.trace_granularity != "request_aggregated":
            raise RuntimeError(
                "online emission supports request_aggregated granularity only "
                f"(got {self.config.trace_granularity!r})")
        first_token = train_plan.get("first_token")
        if first_token is None or not first_token.get("split"):
            raise RuntimeError(
                "emit_train_first_step requires a split first_token plan")
        marker = self._mark()
        self._emit_train_head(train_plan)
        self._emit_train_body(
            train_plan, list(first_token["first_spans"]), 1)
        first_token_members = self._emit_first_token_markers(
            train_plan, first_token["debut_marker_members"])
        group = self.group_by_index[train_plan["instance_index"]]
        wakeup_id = first_token["wakeup_id"]
        for builder in self.builders.values():
            builder.set_context(wakeup_id, "prefill", 0)
        wakeup_members = {
            rank: self._emit_train_marker(
                rank, "{}_wakeup".format(wakeup_id))
            for rank in group.ranks
        }
        self._collect(marker)
        return {
            "first_token_members": first_token_members,
            "wakeup_members": wakeup_members,
        }

    def emit_train_remainder(self, train_plan: dict) -> dict:
        """WP9 余量批发射(2026-08-26;拆分阶段 2,唤醒交付处调用)。

        余量批 = 剩余迭代(成员第 2 个 span 起)+ prefill 队头剩余 chunk
        (weight_passes = iterations-1,与首步批的 1 次恰合回整列的迭代
        数;激活/KV/AR 逐 span 精确,总量与整列发射一致)。drain/exit/
        哨兵标记与共享 end barrier 全部照常挂本批(挂点语义不变,仍
        "barrier 前末节点");decode_length=1 的 debut 成员 exit 标记
        改名携带 first_token 子串(同节点 code 4/8 双锚点)。返回结构与
        emit_iteration_train 相同。"""
        if self.config.trace_granularity != "request_aggregated":
            raise RuntimeError(
                "online emission supports request_aggregated granularity only "
                f"(got {self.config.trace_granularity!r})")
        first_token = train_plan.get("first_token")
        if first_token is None or not first_token.get("split"):
            raise RuntimeError(
                "emit_train_remainder requires a split first_token plan")
        marker = self._mark()
        self._emit_train_body(
            train_plan, list(first_token["rest_spans"]),
            int(train_plan["iterations"]) - 1)
        result = self._emit_train_tail_markers(train_plan, first_token)
        self._collect(marker)
        return result

    def _emit_train_head(self, train_plan: dict) -> None:
        """列车头:joiner 迁移 + join/pstart 起始标记节点。

        拆分时整段归首步批(face 迁移/起始标记锚定的迭代位置在第 1 迭代
        内,语义不变);发射序与整列发射逐节点一致。"""
        group = self.group_by_index[train_plan["instance_index"]]
        train_id = train_plan["train_id"]

        # ---- joiner 迁移(face 3000 类 paired transfer;上下文
        #      (joiner, decode, 1) = decode_start 指标锚点) ----
        joiners = list(train_plan.get("joiners", ()))
        for joiner in joiners:
            self._set_context(joiner, "decode", 1)
            prefill_group = self.group_by_index[
                joiner["prefill_instance_index"]]
            decode_group = self.group_by_index[
                joiner["decode_instance_index"]]
            joiner_routes = _paired_transfer(
                config=self.config, builders=self.builders,
                queue_index=joiner["queue_index"], category=3000,
                name="{}_prefill_to_decode_kv".format(_prefix_of(joiner)),
                source_group=prefill_group, target_group=decode_group,
                total_bytes=kv_cache_bytes_for_tokens(
                    self.config.model, joiner["prefill_context_tokens"]),
                source_release_ack=self.source_release_capacity_gates,
                source_release_ack_tag_factory=self._source_release_ack_tag,
                source_release_arm_ids=(
                    self._take_source_release_arms(decode_group.ranks)
                    if (self.source_release_capacity_gates
                        and prefill_group.name != decode_group.name) else {}),
            )
            self._register_source_release_routes(joiner_routes)

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

    def _emit_train_body(self, train_plan: dict, pass_spans,
                         weight_passes: int) -> None:
        """折叠列车体(17 类聚合节点;weight_passes = 权重读取次数)。

        拆分时首步批传首步 span 组 + weight_passes=1,余量批传余量组 +
        iterations-1;两批激活/KV/AR 字节按 span 求和与整列一致。

        die 后端(阶段 C):die_context 非 None 时按 rank 写入发射语境
        (phase = 列车 stage,TP = 实例 rank 数,span 记录带绝对位置——
        derive_span_records 按 face span 恒等式还原),列车体经
        transformer_pass_aggregated 的 die 分支附 die_op_spec 并做
        §7.2 region 融合;legacy 时 die_ctx=None,发射逐字节不变。"""
        group = self.group_by_index[train_plan["instance_index"]]
        train_id = train_plan["train_id"]
        stage = train_plan["stage"]
        generation = 1 if stage == "decode" else 0
        for builder in self.builders.values():
            builder.set_context(train_id, stage, generation)
        tensor_parallel = len(group.ranks)
        for relative_rank, rank in enumerate(group.ranks):
            die_ctx = None
            if self.die_context is not None:
                self.die_context.begin_emission(
                    phase=stage,
                    tensor_parallel=tensor_parallel,
                    tensor_parallel_rank=relative_rank,
                    span_namespace=train_id,
                    pass_spans=pass_spans,
                )
                die_ctx = self.die_context
            transformer_pass_aggregated(
                self.builders[rank],
                phase=train_id,
                pass_spans=pass_spans,
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
                weight_passes=weight_passes,
                die_ctx=die_ctx,
            )

    def _emit_first_token_markers(self, train_plan: dict,
                                  debut_members) -> dict:
        """WP9 first_token 标记节点(debut 成员各自挂;每 rank 1 个
        1-op COMP 小节点,名字含 "first_token" 子串,C++ 按 (request_id,
        rank) 取 min tick 收 code 8 事件)。锚定位置 = 该成员链条在所在
        批内列车体后的末节点(共享折叠体 ⇒ 各 debut 标记顺序挂体后)。"""
        train_id = train_plan["train_id"]
        group = self.group_by_index[train_plan["instance_index"]]
        members = {}
        for member in debut_members:
            request_id = member["request_id"]
            self._set_context(member, "decode", 1)
            members[request_id] = {
                rank: self._emit_train_marker(
                    rank, "{}_first_token_{}".format(
                        train_id, sanitize_node_prefix(request_id)))
                for rank in group.ranks
            }
        return members

    def _emit_train_tail_markers(self, train_plan: dict,
                                 first_token) -> dict:
        """列车尾:drain/exit 标记 + 哨兵标记 + 共享 end barrier +
        completion_gates 账本(挂点语义与整列发射一致)。

        WP9:不拆车时的 first_token 标记(iterations==1 场景;拆车时标记
        已在首步批挂过,余量批不重挂);decode_length=1 的 debut 成员
        exit 标记名附加 first_token 子串——同节点双锚点(code 4 watch +
        code 8 名字),first_token_ns == completion_ns 不变量由同一节点
        保证(TP 组内 rank 斜台下 code8-min/code4-max 聚合不对称的残余
        偏差由 collector 侧裁决,发射侧不再绕过)。"""
        instance_index = train_plan["instance_index"]
        group = self.group_by_index[instance_index]
        train_id = train_plan["train_id"]
        stage = train_plan["stage"]
        generation = 1 if stage == "decode" else 0
        iterations = int(train_plan["iterations"])

        # ---- WP9:不拆车时的 first_token 标记(iterations==1;拆车时
        #      标记已在首步批挂过) ----
        if first_token is not None and not first_token.get("split"):
            self._emit_first_token_markers(
                train_plan, first_token["debut_marker_members"])

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
        exit_first_token = set(
            (first_token or {}).get("debut_exit_first_token") or ())
        for member in train_plan.get("exit_members", ()):
            request_id = member["request_id"]
            self._set_context(member, "decode", 1)
            exit_members[request_id] = {
                rank: self._emit_train_marker(
                    rank, (
                        "{}_exit_first_token_{}".format(
                            train_id, sanitize_node_prefix(request_id))
                        if request_id in exit_first_token else
                        "{}_exit_{}".format(
                            train_id, sanitize_node_prefix(request_id))))
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
        # 注(2026-09-26 审查澄清,A2-2,待用户裁决):comm_size 传迭代数
        # iterations 是自逐请求时代延续的既定口径,证据上不属笔误——旧
        # per-request 发射的 prefill_chunks_aggregated_end_barrier 就以
        # 聚合 pass 数(len(spans))作 collective bytes(commit e2516b8),
        # 旧 decode_request_end_barrier 才是 1;列车化统一取 iterations,
        # 与姊妹仓(wscllm / wscllm-LRU / face-LRU)同款。语义上它是列车
        # 聚合规模的 collective 时序计量输入(直接进入 C++ collective
        # 时延模型),不是协议字节数;改成 1 会改变仿真时序基线,未经
        # 全仓族对拍裁决前不得擅动。
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

        return {
            "drain_members": drain_members,
            "exit_members": exit_members,
            "sentinel_members": sentinel_members,
            "block_ends": block_ends,
        }

    def _emit_train_marker(self, rank: int, name: str) -> int:
        """列车标记节点(每 rank 1 个小 COMP 节点;上下文由调用方设置)。

        契约 §7 裁定 8:一切 comp(name,1,1) 微节点(first_token/drain/
        exit/sentinel/join/pstart/wakeup 等 marker)一律显式
        op_role="control_marker"——不靠"数值很小"识别(设计文档 §7.3);
        legacy 后端下该属性惰性(C++ 分发保持旧微作业语义,裁定 3)。"""
        self.builders[rank].comp(name, 1, 1, op_role="control_marker")
        return self.builders[rank].previous_id

    def _set_context(self, request_plan: dict, stage: str,
                     generation: int) -> None:
        request_id = request_plan["request_id"]
        for builder in self.builders.values():
            builder.set_context(request_id, stage, generation)

    def retire_completion_gate(self, session_id: str) -> None:
        """释放 terminal session 不会再被下一 turn 消费的完成门。"""
        self.completion_gates.pop(session_id, None)

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
            # interval gate 是该 session completion gate 的唯一正常消费者；
            # 取用即删，后续只保留已在图边中编码的 after_node_id。
            previous = self.completion_gates.pop(
                request_plan["session_id"], None)
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
                migration_routes = _paired_transfer(
                    config=self.config, builders=builders,
                    queue_index=request_plan["queue_index"], category=1000,
                    name="{}_history_kv".format(prefix),
                    source_group=source_group, target_group=prefill_group,
                    total_bytes=request_plan["history_transfer_bytes"],
                    timer_gates=timers,
                    source_release_ack=self.source_release_capacity_gates,
                    source_release_ack_tag_factory=self._source_release_ack_tag,
                    source_release_arm_ids=(
                        self._take_source_release_arms(prefill_group.ranks)
                        if self.source_release_capacity_gates else {}),
                )
                self._register_source_release_routes(migration_routes)
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
