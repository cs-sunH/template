#!/usr/bin/env python3
"""graph_batch_builder.py -- sh_3.0 在线 GraphBatch 构图器（方案 §4 步骤 1-8 操作 4）。

阶段 1 最关键的对齐点：复用共享发射原语的节点结构。per-request 发射
（generate_face_trace.py 模块级发射助手函数）组成
（_emit_kv_transfer / _emit_tp_readiness_barrier /
_emit_tp_point_to_point_readiness_barrier / transformer_pass_aggregated）——
本模块直接 import 它们，用 OnlineTraceBuilder（与 TraceBuilder 同构的在线侧
builder）驱动，保证节点属性、插入顺序、rank ownership 跨 request 链一致
（节点级审计口径）。

sh_3.0 发射结构（拼 batch 改造，2026-08-22；原两段式（准入+prefill 整段 /
decode 整段）的列车化重构，设计文档《层次 B Continuous Batching 改造》§3.2）：
  - 准入动作（ARRIVAL 边界，emit_admission_batch）= 到达/interval gate →
    history_evictions → history_transfer（含 partial 流水恢复全部节点）→
    prefill_evictions → prefill readiness barrier；prefill 主体不再在此发射；
  - 迭代列车（各决策边界，emit_iteration_train）= joiner 的 decode_evictions
    （触发门 = drain 列车 barrier）→ prefill→decode 迁移 →（有 joiner 时）
    共享 readiness barrier → join/pstart 标记 → 折叠列车体（成员×迭代 span，
    weight_passes=迭代数：权重每迭代只读一次）→ drain/exit 标记节点（挂
    PREFILL_DRAIN / DECODE_COMPLETION watch）→ 每列车一个共享 end barrier；
  - 完成批（REQUEST_COMPLETE 边界，emit_completion_batch）=
    completion_evictions + 下一 turn interval gate 的依赖登记。

在线语义差异（刻意，注释标注）：
  - timer gate 始终发射节点（结构保留）但 runtime_ns=0：到达时间由 C++ 的
    arrival alarm（future_alarms）替代；
  - timer gate 的 duration 语义与共享参考同参（duration=admission_time_ns /
    interval + hbm_wait_ns；0 时不发射 gate 节点，在线同款跳过）；
  - pending_history / deferred_session_locations 账本与共享调度语义同构
    （跨请求的 arrival/interval gate 关联）；
  - partial 流水恢复的 chain_checkpoint/restore_chain 段内分支并行机制
    原样支持（拼 batch 改造起 body 移入列车，suffix 恢复完成门经
    _suffix_body_arms 账本挂到含其首 chunk 的列车体，见
    emit_iteration_train 的 first_chunk_member 处理）；
  - strategy 保持物理链与真实 MEM/HBM 物理时长；
  - watch 锚点与共享指标口径一致（拼 batch 改造起由列车标记承载）：
    PREFILL_DRAIN = drain 标记节点（列车体后、end barrier 前）；DECODE_
    COMPLETION/REQUEST_COMPLETE = exit 标记节点（同位置）。触发门角色
    （joiner decode_evictions 的 node_gates、completion_evictions 触发门、
    下一 turn interval gate 的 after_node_id、块末账本）仍用 post-barrier
    节点（列车 end barrier；五仓一致口径，不随 watch 锚点变化）。
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

# M1 收集即释放的摊销压缩水位（2026-08-23）：_collect 把已发射节点切片
# 进当批后，per-rank 已收集前缀达到该水位即整段删除（节点 id 来自
# next_id 计数器，与 list 位置无关）。8192 保证工作集有界且删除频度
# 足够低——每节点均摊 O(1)，禁止逐批前缀删除（O(n²) 反例）。
_COLLECT_COMPACT_THRESHOLD = 8192


def first_token_split_enabled() -> bool:
    """WP9 首步批拆分总开关（SH_FIRST_TOKEN_SPLIT，B4 起缺省 "0" 关）。

    缺省翻转（2026-08-26，主规格 §1.6 A 类处置）：B3_S3 60s 决策等价
    门-2 失败——拆分的物理扰动（首列拆批 → tick 漂移 → 闭环逐轮放大 →
    决策边界穿越 → 实例选择翻转，tput +13.2%）使 ON/OFF 字节等价在
    60s 窗不可达（2s 门保持）。证据：/tmp/slo_wps/b3/S3/t3_full_off_split
    与 /tmp/slo_wps/gates/B3_S3.FAILED（wp9_gate_60s）。默认口径退回
    proxy（first_token_source=train_interpolated，见 metrics_postprocess），
    本开关显式置 "1" 仍可启用拆分取 exact 首 token（研究/对拍用）。

    "1" = 拆分开启：debut 列车两段式发射 + first_token 标记；
    其他值（含缺省）= 拆分完全关闭：调度器不拆列车、不发射 first_token
    标记，构图/决策/账本产物与拆分上线前逐字节一致（A/B 对拍的 OFF
    侧）。每次 _emit_train 现场读取（而非构造期缓存），测试与复跑可在
    进程内切换。
    """
    return os.environ.get("SH_FIRST_TOKEN_SPLIT", "0") == "1"


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
        # M1 收集即释放（2026-08-23）：本 list 只保留"已发射未收集"的
        # 尾部——_collect 切片进批后按水位摊销压缩前缀（见 _collect），
        # 全量历史节点不再常驻。禁止按 list 位置回读节点（id 来自
        # next_id 计数器，与位置无关）。
        self.nodes = []   # 本 rank 已发射节点 dict（发射序，可被压缩）
        self.edges = []   # 本 rank parent edges（随节点水位一并压缩）
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
    按决策边界发射准入动作批 / 实例迭代列车 / completion 批，并维护
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
        # request_id -> {"seg1": {rank: id}, "seg2": {rank: id}}
        # 列车块末账本（post-barrier 口径；emitted-ranks-only 语义）：seg1 =
        # 覆盖其最后 prefill chunk 的列车 end barrier（joiner decode_evictions
        # 触发门），seg2 = 覆盖其退出迭代的列车 end barrier（completion_
        # evictions 触发门 + 下一 turn interval gate after_node_id）。不再
        # 恢复进 builders——frontier 接续裁决 2026-08-19，见 emit_iteration_train。
        self._block_ends = {}
        # request_id -> {rank: suffix 恢复完成门节点 id}（partial 流水恢复
        # 的 body 依赖账本，拼 batch 改造起首 chunk 主体移入列车：恢复分支
        # 仍在准入批发射（与 suffix 恢复并行的语义改为经此账本挂到含其首
        # chunk 的列车体），见 emit_iteration_train 的 first_chunk_member。
        self._suffix_body_arms = {}
        self.batch = None  # 当前批次累加器（由 begin_batch 建立）
        # WP9 首步批标记（2026-08-26）：本批是否含首步批发射（digests 行
        # first_step=True 标记的来源；OFF 侧恒 False，不进任何产物）。
        self.batch_first_step = False
        # request_id -> 跨阶段连续的 action 序号（离线 writer 的 per-
        # request action_sequence 闭包在列车发射下的等价物：准入/列车/
        # completion 批共享同一计数器，保证 actionNNN 命名与逐字节稳定
        # ——canonical 命名对照前提）。
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
        self.batch_first_step = False

    def _collect(self, marker: dict) -> None:
        for rank, builder in self.builders.items():
            node_mark, edge_mark = marker[rank]
            self.batch["nodes"].extend(builder.nodes[node_mark:])
            self.batch["parent_edges"].extend(builder.edges[edge_mark:])
            # M1 收集即释放（摊销压缩，2026-08-23）：已切片进本批的节点/
            # 边不再驻留 builder——已收集水位 ≥ 8192 且不小于现存总量一半
            # 时才删前缀（每次删除搬运的尾部 ≤ 现存一半，均摊 O(1)/节点）。
            # 安全前提（全仓 grep 证实）：节点 id 来自 next_id 计数器，无
            # 任何按 list 位置回读节点的代码；每个 _mark() 都在同一次发射
            # 调用内被紧随的单次 _collect() 消费（无跨发射延迟消费），水位
            # 即本次切片在当前 list 中的绝对长度，压缩后下一次 _mark 重新
            # 取 len，自洽。
            if (node_mark >= _COLLECT_COMPACT_THRESHOLD
                    and node_mark * 2 >= len(builder.nodes)):
                del builder.nodes[:node_mark]
                del builder.edges[:edge_mark]

    def _mark(self) -> dict:
        return {
            rank: (len(builder.nodes), len(builder.edges))
            for rank, builder in self.builders.items()
        }

    # ------------------------------------------------------------- 发射 --

    def emit_admission_batch(self, request_plan: dict) -> None:
        """发射 request 的准入动作（ARRIVAL 决策的图；拼 batch 改造
        2026-08-22）：到达/interval gate + history_evictions +
        history_transfer（含 partial 流水恢复全部节点）+ prefill_evictions +
        prefill readiness barrier。

        prefill 主体（chunk 序列）与 PREFILL_DRAIN watch 不再在此发射——
        移入实例迭代列车（emit_iteration_train 的折叠体与 drain 标记）；
        本方法无 watch 返回值（调度器在列车发射处注册）。

        strategy 恒走 roofline 物理时钟；[frontier 接续裁决，strategy
        死锁修复统一(2026-08-19)] 的无条件接续语义不变：准入动作链到该
        rank 当前 frontier（实例列车在飞时物理排在列车后）。"""
        if self.config.trace_granularity != "request_aggregated":
            raise RuntimeError(
                "online emission supports request_aggregated granularity "
                "only (production config)")
        self._set_context(request_plan, "prefill", 0)
        marker = self._mark()
        self._emit_admission_actions(request_plan)
        self._collect(marker)

    def emit_iteration_train(self, train_plan: dict) -> dict:
        """发射一趟实例迭代列车（拼 batch 改造核心，2026-08-22；设计
        文档 §3.2"迭代列车聚合发射"；WP9 拆分重构 2026-08-26 起
        头/体/尾分解为 _emit_train_head/_emit_train_body/
        _emit_train_tail_markers，整列路径行为逐字节不变）。

        train_plan（调度器冻结的成员快照）字段：
          train_id            批命名空间 id（"batch_train_i<实例>_<序号>"；
                              共享体节点归属，物理完成与逻辑请求完成分离）
          instance_index      列车所在实例
          stage               "decode"（有 decode 成员，含混合迭代）或
                              "prefill"（纯 prefill 列车）
          joiners             新成员 request_plan 列表（含 decode_evictions/
                              prefill_decode_transfer/prefill_drain_block_ends
                              {rank: drain 列车 barrier 节点 id}）——迁移
                              节点先于列车体，经共享 readiness barrier 栅栏
          pass_spans          成员×迭代展开的 (tokens, kv) 平铺列表
          iterations          迭代数（= weight_passes：权重字节 ×迭代数，
                              与批成员数无关；激活/KV/AR 逐 span 精确）
          first_chunk_member  队列头首 chunk 在本列车时的成员（pstart 标记
                              + partial 恢复 suffix 门挂接；None = 无）
          drain_members       本列车内完成最后 prefill chunk 的请求 plan 列表
          exit_members        本列车内退出 decode 的成员 plan 列表
          first_token         WP9 首 token 观测（2026-08-26；缺省 None =
                              拆分开关关闭或无 debut，行为与上线前逐字节
                              一致）：{"split": True} 拒绝本方法（两段式
                              经 emit_train_first_step/emit_train_remainder
                              发射）；{"split": False} 为不拆车标记增强
                              （首 token 标记/exit 改名在尾标记处挂）。

        每实例 rank 上的结构（链序）：
          [joiner 迁移 ...] → [共享 readiness barrier] → 17 类聚合体节点
          (transformer_pass_aggregated, weight_passes=iterations) →
          [first_token 标记（不拆车增强时）] → [drain 标记 ...] →
          [exit 标记 ...] → 共享 end barrier。

        返回 {"drain_members": {request_id: {rank: 标记节点 id}},
              "exit_members": {request_id: {rank: 标记节点 id}},
              "sentinel_members": {rank: 标记节点 id},
              "block_ends": {rank: end barrier 节点 id}}——标记即
        PREFILL_DRAIN / DECODE_COMPLETION watch 成员（barrier 前末节点
        口径：标记是列车体后、end barrier 前的真实节点）；块末账本
        _block_ends[req]["seg1"]/["seg2"] = 本列车 post-barrier 节点
        （decode_evictions/completion_evictions 触发门与下一 turn
        interval gate after_node_id 的来源，五仓一致口径不随 watch 锚点
        变化）。"""
        if self.config.trace_granularity != "request_aggregated":
            raise RuntimeError(
                "online emission supports request_aggregated granularity "
                "only (production config)")
        first_token = train_plan.get("first_token")
        if first_token is not None and first_token.get("split"):
            raise RuntimeError(
                "split train plans must be emitted through "
                "emit_train_first_step/emit_train_remainder")
        marker = self._mark()
        self._emit_train_head(train_plan)
        self._emit_train_body(
            train_plan, list(train_plan["pass_spans"]),
            int(train_plan["iterations"]),
            first_chunk_member=train_plan.get("first_chunk_member"))
        result = self._emit_train_tail_markers(train_plan, first_token)
        self._collect(marker)
        return result

    def emit_train_first_step(self, train_plan: dict) -> dict:
        """WP9 首步批发射（2026-08-26；拆分阶段 1）。

        首步批 = 所有列车成员的第 1 个 span（decode 成员首迭代）+ prefill
        队头的第 1 个 chunk，加上按迭代位置锚定在首步内的 joiner 迁移/
        readiness barrier/起始标记/partial 恢复 suffix 门（挂点语义与整列
        发射完全一致，发射序逐节点不变）。列车体以 weight_passes=1 折叠
        （权重恰读一次），体后挂各 debut 成员的 first_token 标记节点
        （1-op COMP，名字含 "first_token" 子串——C++ 锚点按名字子串注册
        code 8，取每 rank min tick）。

        本批不含任何请求级 watch/drain/exit/哨兵标记；唯一附加物是尾部的
        批命名空间唤醒标记（每 rank 1 个小节点，request_id =
        "<train_id>_first_step"，前缀 batch_train_，复用 C++ 哨兵 watch
        通道）：其 fire 经 PREFILL_DRAIN 通道送回调度器，作为余量批的
        交付边界——没有它，无 watch 的首步批完成后不存在任何决策工作，
        C++ tick-end 门不会再交付，运行尾部（全部其余工作已排空）将永久
        等待（死锁）。这是对"首步批无任何 watch 标记"的必要工程化偏移
        （与 sh_1.0/sh_2.0 同构）：唤醒 watch 不挂任何请求、不触发核销/
        记账/决策（调度器按 first_step id 识别后无操作）。

        返回 {"first_token_members": {request_id: {rank: 标记节点 id}},
              "wakeup_members": {rank: 唤醒标记节点 id}}。"""
        if self.config.trace_granularity != "request_aggregated":
            raise RuntimeError(
                "online emission supports request_aggregated granularity "
                "only (production config)")
        first_token = train_plan.get("first_token")
        if first_token is None or not first_token.get("split"):
            raise RuntimeError(
                "emit_train_first_step requires a split first_token plan")
        marker = self._mark()
        self._emit_train_head(train_plan)
        self._emit_train_body(
            train_plan, list(first_token["first_spans"]), 1,
            first_chunk_member=train_plan.get("first_chunk_member"))
        first_token_members = self._emit_first_token_markers(
            train_plan, first_token["debut_marker_members"])
        group = self.group_by_index[train_plan["instance_index"]]
        wakeup_id = first_token["wakeup_id"]
        for builder in self.builders.values():
            builder.set_context(wakeup_id, "prefill", 0)
        wakeup_members = {
            rank: self._emit_train_marker(rank, f"{wakeup_id}_wakeup")
            for rank in group.ranks
        }
        self.batch_first_step = True
        self._collect(marker)
        return {
            "first_token_members": first_token_members,
            "wakeup_members": wakeup_members,
        }

    def emit_train_remainder(self, train_plan: dict) -> dict:
        """WP9 余量批发射（2026-08-26；拆分阶段 2，唤醒交付处调用）。

        余量批 = 剩余迭代（成员第 2 个 span 起）+ prefill 队头剩余 chunk
        （weight_passes = iterations-1，与首步批的 1 次恰合回整列的迭代
        数；激活/KV/AR 逐 span 精确，总量与整列发射一致）。drain/exit/
        哨兵标记与共享 end barrier 全部照常挂本批（挂点语义不变，仍
        "barrier 前末节点"）；decode_length=1 的 debut 成员 exit 标记
        改名携带 first_token 子串（同节点 code 4/8 双锚点，保证
        first_token_ns == completion_ns）。返回结构与
        emit_iteration_train 相同。"""
        if self.config.trace_granularity != "request_aggregated":
            raise RuntimeError(
                "online emission supports request_aggregated granularity "
                "only (production config)")
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
        """列车头：joiner 迁移 + 共享 readiness barrier + 起始标记节点。

        拆分时整段归首步批（迁移/栅栏/起始标记锚定的迭代位置在第 1
        迭代内，语义不变）；发射序与整列发射逐节点一致。"""
        instance_index = train_plan["instance_index"]
        group = self.group_by_index[instance_index]
        train_id = train_plan["train_id"]

        # ---- joiner 迁移（触发门 = 该成员 drain 列车的 post-barrier
        #      块末；上下文 (joiner, decode, 1) = decode_start 指标锚点） ----
        joiners = list(train_plan.get("joiners", ()))
        for joiner in joiners:
            self._set_context(joiner, "decode", 1)
            self._action_seq.setdefault(joiner["request_id"], [0])
            prefix = _prefix_of(joiner)
            drain_gates = joiner.get("prefill_drain_block_ends") or {}
            prefill_group = self.group_by_index[
                joiner["prefill_instance_index"]]
            decode_eviction_trigger = TransferTriggerGate(
                control_instance_index=joiner["prefill_instance_index"],
                node_gates=tuple(
                    drain_gates[rank] for rank in prefill_group.ranks),
            )

            def emit_transfer(transfer, stage_name: str, *,
                              gate=None, trigger_gate=None) -> dict:
                action_state = self._action_seq[joiner["request_id"]]
                action_sequence = action_state[0]
                action_name = (
                    f"{prefix}_{stage_name}_"
                    f"action{action_sequence:03d}_"
                    f"{sanitize_node_prefix(transfer.session_id)}_{transfer.kind}"
                )
                record = _emit_kv_transfer(
                    config=self.config,
                    builders=self.builders,
                    group_by_index=self.group_by_index,
                    tag_allocator=self.tag_allocator,
                    transfer=transfer,
                    action_name=action_name,
                    pending_gate=gate,
                    trigger_gate=trigger_gate,
                )
                record["sequence_stage"] = stage_name
                record["action_sequence"] = action_sequence
                action_state[0] = action_sequence + 1
                # 2026-08-23 修订：发射侧标记已移除（决策时点同步是唯一路径，
                # 迟到的旧转移标记会把门回退到过期位置，见 sh_2.0 seq4689 事故）。
                return record

            for transfer in joiner.get("decode_evictions") or ():
                emit_transfer(transfer, "decode_evictions",
                              trigger_gate=decode_eviction_trigger)
            pd_transfer = joiner.get("prefill_decode_transfer")
            if pd_transfer is None:
                raise RuntimeError(
                    "joiner is missing its Prefill-to-Decode KV action")
            emit_transfer(pd_transfer, "prefill_decode_transfer")

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

    def _emit_train_body(self, train_plan: dict, pass_spans,
                         weight_passes: int, *,
                         first_chunk_member=None) -> None:
        """折叠列车体（17 类聚合节点；weight_passes = 权重读取次数）。

        拆分时首步批传首步 span 组 + weight_passes=1（含队列头首 chunk
        与 partial 恢复 suffix 门挂接），余量批传余量组 + iterations-1；
        两批激活/KV/AR 字节按 span 求和与整列一致。"""
        instance_index = train_plan["instance_index"]
        group = self.group_by_index[instance_index]
        train_id = train_plan["train_id"]
        stage = train_plan["stage"]
        generation = 1 if stage == "decode" else 0
        # partial 流水恢复的 suffix 完成门：首 chunk 主体自拼 batch 改造起
        # 移入列车体，恢复分支仍在准入批发射——其完成门经 _suffix_body_arms
        # 账本挂到含其首 chunk 的列车体首节点（拆分时首 chunk 恒在首步批，
        # 账本在首步批发射处弹出；原"首 chunk 前缀层并行计算"降级为整列车
        # 等恢复，物理保守方向，见类 docstring）。
        suffix_arms = None
        if first_chunk_member is not None:
            suffix_arms = self._suffix_body_arms.pop(
                first_chunk_member["request_id"], None)
            if suffix_arms:
                missing = [
                    rank for rank in group.ranks
                    if suffix_arms.get(rank) is None]
                if missing:
                    raise RuntimeError(
                        "suffix restore gate missing on ranks {}".format(
                            missing))
        for builder in self.builders.values():
            builder.set_context(train_id, stage, generation)
        tensor_parallel = len(group.ranks)
        for relative_rank, rank in enumerate(group.ranks):
            if suffix_arms:
                self.builders[rank].arm_dependency(suffix_arms[rank])
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
            )

    def _emit_first_token_markers(self, train_plan: dict,
                                  debut_members) -> dict:
        """WP9 first_token 标记节点（debut 成员各自挂；每 rank 1 个
        1-op COMP 小节点，名字含 "first_token" 子串，C++ 按
        (request_id, rank) 取 min tick 收 code 8 事件）。锚定位置 =
        该成员链条在所在批内列车体后的末节点（共享折叠体 ⇒ 各 debut
        标记顺序挂体后）。"""
        train_id = train_plan["train_id"]
        group = self.group_by_index[train_plan["instance_index"]]
        members = {}
        for member in debut_members:
            request_id = member["request_id"]
            self._set_context(member, "decode", 1)
            members[request_id] = {
                rank: self._emit_train_marker(
                    rank, f"{train_id}_first_token_"
                    f"{sanitize_node_prefix(request_id)}")
                for rank in group.ranks
            }
        return members

    def _emit_train_tail_markers(self, train_plan: dict,
                                 first_token) -> dict:
        """列车尾：first_token 标记（不拆车增强时）+ drain/exit 标记 +
        哨兵标记 + 共享 end barrier + 块末账本（挂点语义与整列发射
        一致）。"""
        instance_index = train_plan["instance_index"]
        group = self.group_by_index[instance_index]
        train_id = train_plan["train_id"]
        stage = train_plan["stage"]
        generation = 1 if stage == "decode" else 0
        iterations = int(train_plan["iterations"])

        # ---- WP9：不拆车时的 first_token 标记（iterations==1 时整列即
        #      首步，无需两段式；拆车时标记已在首步批挂过，余量批不重挂） ----
        if first_token is not None and not first_token.get("split"):
            self._emit_first_token_markers(
                train_plan, first_token["debut_marker_members"])

        # ---- drain / exit 标记（列车体后、end barrier 前；每成员每 rank
        #      1 个小节点，承载该请求的 PREFILL_DRAIN / DECODE_COMPLETION
        #      watch 与指标 end 锚点；物理完成时刻 = 标记完成时刻。
        #      WP9：decode_length=1 的 debut 成员 exit 标记名附加 first_
        #      token 子串——同节点双锚点（code 4 watch + code 8 名字），
        #      first_token_ns == completion_ns 不变量由同一节点保证） ----
        exit_first_token = set(
            (first_token or {}).get("debut_exit_first_token") or ())
        drain_members = {}
        for member in train_plan.get("drain_members", ()):
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
                    rank, (
                        f"{train_id}_exit_first_token_"
                        f"{sanitize_node_prefix(request_id)}"
                        if request_id in exit_first_token else
                        f"{train_id}_exit_"
                        f"{sanitize_node_prefix(request_id)}"))
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
        #      （joiner decode_evictions 触发门），exit 成员写 seg2
        #      （completion_evictions 触发门 + 下一 turn interval gate） ----
        for member in train_plan.get("drain_members", ()):
            self._block_ends.setdefault(member["request_id"], {})[
                "seg1"] = dict(block_ends)
        for member in train_plan.get("exit_members", ()):
            self._block_ends.setdefault(member["request_id"], {})[
                "seg2"] = dict(block_ends)

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

    def emit_completion_batch(self, request_plan: dict) -> dict:
        """completion 批（DECODE_COMPLETION/REQUEST_COMPLETE 边界）：
        completion_evictions + 下一 turn interval gate 的依赖登记。
        返回空成员（无 watch）。"""
        self._set_context(request_plan, "completion", 1)
        marker = self._mark()
        self._emit_completion(request_plan)
        self._collect(marker)
        # M4 核销即删（2026-08-23）：completion 批是本请求图发射的终点
        # （seg2 块末在 _emit_completion 内消费、action 序号此后无读者，
        # 全仓 grep 证实无更晚读者）——逐出块末/action 计数账本条目在请求
        # 完成后即死重，当场弹出（下一 turn 是不同 request_id）。
        self._block_ends.pop(request_plan["request_id"], None)
        self._action_seq.pop(request_plan["request_id"], None)
        return {}

    def _set_context(self, request_plan: dict, stage: str,
                     generation: int) -> None:
        request_id = request_plan["request_id"]
        for builder in self.builders.values():
            builder.set_context(request_id, stage, generation)

    # ------------------------------------------------- per-request 发射主体 --

    def sync_pending_history_after_evictions(self, transfers) -> None:
        """决策时点补偿：KV 变更点返回的逐出转移，立即镜像到 pending 门。
        唯一标记路径（2026-08-23 seq4689 修订：发射侧重复标记在多级逐出
        发射乱序时会把门回退到过期位置，已移除）。partial_hbm_remote /
        remote_memory 的半驻留语义由 _mark_pending_history_store 自身推导。"""
        for transfer in transfers or ():
            if transfer.kind == "remote_store":
                self._mark_pending_history_store(transfer)

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

    def _emit_admission_actions(self, request_plan: dict) -> None:
        """turn-gates / history / prefill 准入动作的在线发射
        （拼 batch 改造，2026-08-22）。
        request_plan 为 dict（strategy 自在线账本；
        字段与 FaceRequestPlan 同名）。

        [frontier 接续裁决,strategy 死锁修复统一(2026-08-19,对齐 sh_1.0/
        sh_2.0)] strategy **不做任何块末恢复/段内清链**:per-rank
        previous_id 无条件接续当前 frontier(= 离线 writer 跨 request
        物理链同构),per-rank 发行序 = 全局发射序——跨请求 P2P 与
        collective 参与序不可能反转成环。

        prefill 主体（chunk 序列 + end barrier）与 PREFILL_DRAIN watch 自
        拼 batch 改造起移入实例迭代列车（emit_iteration_train 的折叠体与
        drain 标记）；此处止于准入动作（到达 gates/历史迁移/逐出/屏障）。"""
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
            # 2026-08-23 修订：发射侧标记已移除（决策时点同步是唯一路径，
            # 迟到的旧转移标记会把门回退到过期位置，见 sh_2.0 seq4689 事故）。
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
            # turn-0 gate duration 先做 µs 下取整：admission_time_ns 是准入
            # 时刻的虚拟 tick（Roofline 任意 ns 粒度，如 153516157242），
            # timer_gate 的离线同构校验要求整 µs（duration_ns % 1000 != 0
            # 即 raise）。该 duration 不进节点（runtime_ns=0、不存储），仅
            # 驱动校验与 0 跳过，下取整对既有通过路径零影响；消除
            # 2026-08-22 30s 窗 delivery seq=1145 确定性崩溃（阻塞 turn-0
            # 在非 µs 对齐 tick 准入时触发）。
            duration = arrival
            duration -= duration % 1000
            # turn-0 gate 命名用短前缀（q{queue:04d}_{request_id}，
            # 无 session/turn 段）——canonical 命名 name 键逐字节一致前提。
            short_prefix = (
                f"q{request_plan['queue_index']:04d}_"
                f"{sanitize_node_prefix(request_plan['request_id'])}")
            timers = tuple(
                builders[rank].timer_gate(
                    f"{short_prefix}_global_arrival_timer_gate", duration)
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
            # 拼 batch 改造（2026-08-22）：首 chunk 主体不再在本批发射，
            # suffix 恢复完成门从"本批首 chunk 前缀层并行计算"改为经
            # _suffix_body_arms 账本挂到含其首 chunk 的列车体首节点
            # （emit_iteration_train 的 first_chunk_member）。链回滚保留
            # ——恢复分支与本 rank 后续发射（其他请求的准入动作）保持
            # 并行，与改造前的跨请求并行口径一致；列车体本身等恢复
            # （保守方向，类 docstring 登记）。
            self._suffix_body_arms[request_plan["request_id"]] = (
                suffix_ready_nodes_by_rank)
        else:
            _emit_tp_readiness_barrier(
                builders=builders, group=prefill_group,
                name=f"{prefix}_prefill_kv_ready_barrier")
        # prefill 主体（chunk spans + end barrier + seg1 块末）自拼 batch
        # 改造（2026-08-22）起移入 emit_iteration_train 的折叠体与 drain
        # 标记；此处止于准入动作（到达 gates/历史迁移/逐出/屏障）。

    def _emit_completion(self, request_plan: dict) -> None:
        """completion_evictions + 下一 turn interval gate 段的在线发射
        （拼 batch 改造，2026-08-22：触发门改经 _block_ends["seg2"] =
        覆盖其退出迭代的列车 post-barrier 块末，与旧 decode 段末口径一致）。

        [frontier 接续裁决,strategy 死锁修复统一(2026-08-19)] strategy
        **不做任何块末恢复/段内清链**:per-rank previous_id 无条件接续当前
        frontier，本段的触发门显式编码列车 barrier 依赖（跨 request 边）。"""
        builders = self.builders
        config = self.config
        prefix = _prefix_of(request_plan)
        decode_instance_index = request_plan["decode_instance_index"]
        decode_group = self.group_by_index[decode_instance_index]
        decode_completion_nodes = tuple(
            self._block_ends.get(request_plan["request_id"], {})
            .get("seg2", {}).get(rank)
            for rank in decode_group.ranks)
        if any(node_id is None for node_id in decode_completion_nodes):
            # 列车发射后退出成员必有 seg2 块末；缺即账本损坏（fail-closed）。
            raise RuntimeError(
                "segment-2 block ends missing on decode ranks")
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
            # 2026-08-23 修订：发射侧标记已移除（决策时点同步是唯一路径，
            # 迟到的旧转移标记会把门回退到过期位置，见 sh_2.0 seq4689 事故）。
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
