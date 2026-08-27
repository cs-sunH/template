#!/usr/bin/env python3
"""graph_batch_builder.py -- sh_1.0 在线 GraphBatch 构图器(方案 §4 步骤 1-8 操作 4)。

阶段 1 最关键的对齐点:复用共享发射原语的节点结构。`generate_face_trace.py`
提供的 _emit_kv_transfer / _emit_transfer_trigger /
_emit_tp_readiness_barrier 与 transformer_pass_aggregated 由本模块直接
import，并用 OnlineTraceBuilder 驱动，保证:

  - 节点属性、插入顺序、rank ownership 跨 request 链结构一致(节点级审计口径);
  - per-rank 节点 id 跨批次全局递增;
  - interval gate 的 after_node_id 指向上一同 session request 的 decode 完成
    barrier 节点 id(pending_history 账本,跨批次解析)。

sh_1.0 发射结构(拼 batch 改造,2026-08-22;原三段式的列车化重构,
设计文档《层次 B Continuous Batching 改造》§3.2):
  - 准入动作(ARRIVAL 边界,emit_admission_batch)= 到达/interval timer
    gates + history_evictions + history_transfer(或 turn-0 arm gate)+
    prefill_evictions + prefill 屏障;prefill 主体不再在此发射;
  - 迭代列车(各决策边界,emit_iteration_train)= joiner 的
    decode_evictions(触发门 = drain 列车 barrier)+ prefill→decode 迁移
    + 共享 readiness barrier + 折叠列车体(成员×迭代 span,weight_passes
    =迭代数:权重每迭代只读一次)+ drain/exit 标记节点(挂 PREFILL_
    DRAIN / DECODE_COMPLETION watch)+ 每列车一个共享 end barrier;
  - 完成段(REQUEST_COMPLETE 边界,emit_completion_batch)= completion_
    evictions(触发门 = 退出列车 barrier)+ 下一同 session turn 的
    interval timer gates(after_node_id=退出列车 barrier,
    duration = interval + hbm_wait_ns)。

在线语义差异(刻意,注释标注;蓝本裁决 3/7/9 的三段推广):
  - timer gate 始终发射节点(结构保留)但 runtime_ns=0:到达/间隔时刻由 C++
    arrival alarm(future_alarms)替代,gate 不再等待;
  - 每段发射后按 (request_id, rank) 记录块末 previous_id(_block_ends 账本,
    emitted-ranks-only 语义),供后续段的触发门与 watch 锚点使用,不回灌
    builders(2026-08-15 的段间恢复裁决已于 2026-08-19 废止,见
    _emit_admission_actions 的 frontier 接续裁决块);
  - strategy 保持物理跨 request 链(无条件接续 frontier);
  - watch 锚点与共享指标口径一致(拼 batch 改造起由列车标记承载):
    PREFILL_DRAIN = drain 标记节点(列车体后、end barrier 前,与
    EVENT_PREFILL_END 锚点同款"barrier 前末节点"口径);
    DECODE_COMPLETION/REQUEST_COMPLETE = exit 标记节点(同位置)。
    触发门角色(joiner decode_evictions / completion_evictions 的
    node_gates、下一 turn interval gate 的 after_node_id、_block_ends
    账本)仍用 post-barrier 节点(列车 end barrier;五仓一致口径,
    不随 watch 锚点变化)。决策边界因此比 post-barrier 口径早一个
    all_reduce——这是五仓统一的预期时间线变化。

共享调度语义在在线侧的账本:pending_history(request_id ->
PendingHistoryGate 等价 dict)/pending_request_by_session/
deferred_remote_sessions。
"""

import os
import sys

# --------------------------------------------------------------------------
# import 路径:本文件位于 workload/llama2_7b_inference/online/,共享配置与
# 发射原语模块在上一级。路径只做 import 用途(红线:generate_face_trace.py /
# face_scheduler.py 只读 import 与注释)。
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
    _emit_tp_readiness_barrier,
    sanitize_node_prefix,
)
from face_scheduler import KVTransfer, KVTransferShard  # noqa: E402

# M1 收集即释放的摊销压缩水位（2026-08-23）：_collect 把已发射节点切片
# 进当批后，per-rank 已收集前缀达到该水位即整段删除（节点 id 来自
# next_id 计数器，与 list 位置无关）。8192 保证工作集有界且删除频度
# 足够低——每节点均摊 O(1)，禁止逐批前缀删除（O(n²) 反例）。
_COLLECT_COMPACT_THRESHOLD = 8192


def first_token_split_enabled() -> bool:
    """WP9 首步批拆分总开关（SH_FIRST_TOKEN_SPLIT，B4 起缺省 "0" 关）。

    缺省翻转（2026-08-27，主规格 §1.6 A 类处置）：B3_S1 60s 决策等价
    门-2 失败——拆分的物理扰动（首步批 → 每拆分列车多一次交付 →
    tick 漂移 → 闭环逐轮放大 → 决策边界穿越 → 实例选择翻转，首分歧
    decision row 329 / sim t=16.27s，其后 4009/4362 决策分叉）使
    ON/OFF 字节等价在 60s 窗不可达（2s 窗保持通过）。证据：
    /tmp/slo_wps/b3/S1/（t3_full_off_split vs t3_full_split_on 对拍）
    与 /tmp/slo_wps/gates/B3_S1.FAILED（wp9_gate_60s）。默认口径退回
    proxy（first_token_source=train_interpolated，见
    metrics_postprocess.py），本开关显式置 "1" 仍可启用拆分取 exact
    首 token（研究/对拍用）。

    "1" = 拆分开启：debut 列车两段式发射 + first_token 标记；
    其他值（含缺省）= 拆分完全关闭：调度器不拆列车、不发射
    first_token 标记，构图/决策/账本产物与拆分上线前逐字节一致（A/B
    对拍的 OFF 侧）。每次 _emit_train 现场读取（而非构造期缓存），
    测试与复跑可在进程内切换。
    """
    return os.environ.get("SH_FIRST_TOKEN_SPLIT", "0") == "1"


class OnlineTraceBuilder:
    """与共享 TraceBuilder API 同构的在线侧每-rank builder。

    同一接口面:timer_gate / arm_timer_gate / comp / all_reduce / comm_send /
    comm_recv / mem_store / mem_load / next_id / previous_id / node_count——
    共享助手函数(_emit_kv_transfer 等)可直接驱动。与 TraceBuilder 的差异:
    节点发射为 GraphBatch nodes[] dict(而非 ChakraNode 字节流),依赖记录为
    parent_edges[] 以边列表携带依赖;
    timer_gate 忽略 duration(runtime_ns=0,alarm 替代等待)。duration==0 时
    跳过节点(返回 after_node_id)。
    """

    def __init__(self, rank: int, *, remote_operand_loads: bool):
        self.rank = rank
        self.remote_operand_loads = remote_operand_loads
        self.next_id = 0
        self.previous_id = None
        self.pending_extra_dependencies = []
        # M1 收集即释放（2026-08-23）：本 list 只保留"已发射未收集"的
        # 尾部——_collect 切片进批后按水位摊销压缩前缀（见 _collect），
        # 全量历史节点不再常驻。禁止按 list 位置回读节点（id 来自
        # next_id 计数器，与位置无关）。
        self.nodes = []   # 本 rank 已发射节点 dict（发射序，可被压缩）
        self.edges = []   # 本 rank parent edges（随节点水位一并压缩）
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

    def _standalone_node(self, name: str, node_type: int) -> dict:
        """独立节点(无父边、不消费/不更新 previous_id)。"""
        node = {
            "rank": self.rank,
            "id": self.next_id,
            "name": name,
            "type": node_type,
            "is_cpu_op": False,
            "is_timer_op": False,
            "inputs_values": "",
            "request_id": self.request_id,
            "stage": self.stage,
            "generation": self.generation,
            "compute": {"num_ops": 0, "tensor_size": 0, "runtime_ns": 0},
            "comm": {"bytes": 0, "src": 0, "dst": 0, "tag": 0},
            "coll": {"comm_type": 0, "bytes": 0, "priority": 0,
                     "pg_name": "", "involved_dim": []},
        }
        self.next_id += 1
        self.nodes.append(node)
        self.node_count += 1
        return node

    def arm_timer_gate(self, timer_node_id) -> None:
        if timer_node_id is not None:
            self.pending_extra_dependencies.append(int(timer_node_id))

    def timer_gate(self, name: str, duration_ns: int, *,
                   after_node_id=None):
        """在线 timer gate:duration==0 跳过节点(返回 after_node_id);否则发射节点
        (runtime_ns=0,is_timer_op,不经 _new_node——不链 previous_id、不消费
        pending_extra_dependencies、不更新 previous_id;仅 after_node_id 依赖)。
        离线语义(duration = 到达/interval,gate 等待)由 C++ arrival alarm
        替代——gate 只保留结构与依赖,保持时长会双重等待。"""
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
                  comm_tag: int, hbm_charge: bool = True) -> None:
        node = self._new_node(name, COMM_SEND_NODE)
        node["comm"]["src"] = int(src)
        node["comm"]["dst"] = int(dst)
        node["comm"]["bytes"] = self._uint64(comm_size)
        node["comm"]["tag"] = int(comm_tag)
        # Local-HBM contention: only carried when opting OUT (C++ default
        # true), keeping the offline ET attr encoding (absent = charged) and
        # the online node JSON identical in meaning.
        if not hbm_charge:
            node["comm"]["hbm_charge"] = False

    def comm_recv(self, name: str, *, src: int, dst: int, comm_size: int,
                  comm_tag: int, hbm_charge: bool = True) -> None:
        # 回程 recv(noc_migrate ack_from / remote_store ack_from_edge /
        # remote_load request_from_rank)与其他节点一样经 _new_node 链式
        # 发射:frontier 接续裁决(2026-08-19,见 emit_prefill_batch 注释块)
        # 保证 per-rank 发行序 = 全局发射序,跨实例 P2P 参与序不可能反转
        # 成环,无需额外断链。历史备注:frontier 统一前曾以"回程 recv
        # 独立发射(不链 previous_id)"规避对向迁移并发死锁(实测 delivery
        # 156 后 EventQueue 排空;差分归因类别②,sh_1.0改造执行实录.md
        # §15.1 登记),该机制连同死开关 standalone_backchannel_recv 已于
        # 2026-08-20 随本注释清理移除。
        node = self._new_node(name, COMM_RECV_NODE)
        node["comm"]["src"] = int(src)
        node["comm"]["dst"] = int(dst)
        node["comm"]["bytes"] = self._uint64(comm_size)
        node["comm"]["tag"] = int(comm_tag)
        if not hbm_charge:
            node["comm"]["hbm_charge"] = False

    def mem_store(self, name: str, tensor_size: int,
                  hbm_access_mode: int = 0) -> None:
        node = self._new_node(name, MEM_STORE_NODE)
        node["compute"]["tensor_size"] = self._uint64(tensor_size)
        # Local-HBM contention: 1 = read job, 2 = write job (absent = no
        # local HBM access) -- same rule as the offline ET attr.
        if hbm_access_mode:
            node["compute"]["hbm_access_mode"] = int(hbm_access_mode)

    def mem_load(self, name: str, tensor_size: int,
                 hbm_access_mode: int = 0) -> None:
        node = self._new_node(name, MEM_LOAD_NODE)
        node["compute"]["tensor_size"] = self._uint64(tensor_size)
        if hbm_access_mode:
            node["compute"]["hbm_access_mode"] = int(hbm_access_mode)

    # ------------------------------------------------------------- 只读属性 --

    @property
    def node_count_total(self) -> int:
        return self.node_count


def _restore_kv_transfer(record: dict) -> KVTransfer:
    """把决策日志/在线决策里的 transfer dict 重建为 KVTransfer(只读重建,
    驱动共享 _emit_kv_transfer 发射)。"""
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
            )
            for shard in record["shards"]
        ),
    )


def _restore_transfers(records) -> tuple:
    return tuple(_restore_kv_transfer(record) for record in (records or ()))


class GraphBatchBuilder:
    """在线构图器:持有 per-rank OnlineTraceBuilder(状态跨批次),按决策边界
    发射段 1/段 2/段 3,并维护 pending_history / _block_ends 账本。

    调用方喂入的 request_plan dict 至少含:queue_index/session_id/turn_index/
    request_id/prefill_instance_index/decode_instance_index/history_location_before
    (location 字符串或 None)/history_transfer(dict|None)/history_evictions/
    prefill_evictions/decode_evictions/completion_evictions(transfer dict 列表)/
    history_tokens_before/prefill_context_tokens(构图 span 用)。
    """

    def __init__(self, config, *, digest_sink=None):
        self.config = config
        # strategy 无条件接续 frontier(物理跨 request 链)。
        self.builders = {
            rank: OnlineTraceBuilder(
                rank,
                remote_operand_loads=config.remote_operand_loads)
            for rank in range(config.npus_count)
        }
        self.group_by_index = dict(enumerate(config.inference_groups))
        self.tag_allocator = TransferTagAllocator()
        self.p_chunk = int(config.prefill_chunk_size)
        # 共享 pending_history 账本的在线等价(request_id -> dict:
        # source_instance_index/timer_gates/location)。
        self.pending_history = {}
        self.pending_request_by_session = {}
        self.deferred_remote_sessions = set()
        # request_id -> {"seg1": {rank: id|None}, "seg2": {rank: id|None}}
        # 段块末账本(emitted-ranks-only:只记实际发射了节点的 rank 的实际
        # 值,其余 None——蓝本裁决 7/9 语义);不作段间恢复,仅供段间触发门
        # (decode_evictions/completion_evictions 的 node_gates、下一 turn
        # interval gate 的 after_node_id)使用——post-barrier 口径(五仓
        # 一致,不随 watch 锚点变化)。PREFILL_DRAIN/DECODE_COMPLETION
        # watch 锚点自 2026-08-20(R2-2)起改用 barrier 前末节点,不经本账本。
        self._block_ends = {}
        # request_id -> 连续 action 计数:每个 request 的全部 stage 共用
        # 同一计数器，保证跨三个在线段的 actionNNN 命名稳定；canonical
        # 命名 key 含 name，分段各自归零会造成同逻辑节点改名。
        self._action_sequence_by_request = {}
        self.batch = None  # 当前批次累加器(由 begin_batch 建立)

    # ------------------------------------------------------------- 批次 --

    def begin_batch(self) -> None:
        self.batch = {
            "nodes": [],
            "parent_edges": [],
        }

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

    def _set_context(self, request_plan: dict, stage: str,
                     generation: int) -> None:
        request_id = request_plan["request_id"]
        for builder in self.builders.values():
            builder.set_context(request_id, stage, generation)

    def _mark_block_end(self, request_id: str, seg: str, ranks,
                        before: dict = None) -> None:
        # emitted-ranks-only(蓝本裁决 7/9 语义):只对本段实际发射了节点的
        # rank 记实际块末 previous_id;其余 rank 记 None——否则会把该 rank
        # 上其他 request 的链尾误当本段块末(stale 跨 request 边,污染
        # _block_ends 账本,进而被后续段误作触发门/watch 锚点)。
        # before = 段发射前的 per-rank node_count 快照(_mark() 的 node 分量)。
        ends = self._block_ends.setdefault(request_id, {})
        ends[seg] = {
            rank: (self.builders[rank].previous_id
                   if (before is not None
                       and self.builders[rank].node_count_total > before[rank]
                       and self.builders[rank].previous_id is not None)
                   else None)
            for rank in ranks
        }

    def _mark_pending_history_remote(self, session_id: str) -> None:
        # remote_store 逐出后会话 history 迁往 remote_memory 的在线登记。
        pending_request_id = self.pending_request_by_session.get(session_id)
        if pending_request_id is None:
            self.deferred_remote_sessions.add(session_id)
            return
        self.pending_history[pending_request_id]["location"] = "remote_memory"

    def sync_pending_history_after_evictions(self, transfers) -> None:
        """决策时点补偿：KV 变更点返回的逐出转移，立即镜像到 pending 门。
        唯一标记路径（2026-08-23 修订：发射侧重复标记已移除——迟到的旧
        转移标记会把门回退到过期位置，见 sh_2.0 seq4689 事故）。
        背景（TOCTOU 修复，2026-08-23）：账本逐出在决策时同步翻转位置，
        补偿若等物理发射，窗口内下一 turn 到达即命中 :528-532 位置一致性
        检查 fail-closed。"""
        for transfer in transfers or ():
            if transfer.kind == "remote_store":
                self._mark_pending_history_remote(transfer.session_id)

    # ------------------------------------------------------------- 发射 --

    def emit_admission_batch(self, request_plan: dict) -> None:
        """段 1 动作发射(ARRIVAL 边界;拼 batch 改造,2026-08-22):
        到达/间隔 timer gates + history_evictions + history_transfer
        (或 turn-0 arm gate)+ prefill_evictions + prefill 屏障。

        prefill 主体(chunk 序列)与 PREFILL_DRAIN watch 不再在此发射——
        移入实例迭代列车(emit_iteration_train 的折叠体与 drain 标记);
        本方法无 watch 返回值(调度器在列车发射处注册)。[frontier 接续
        裁决,strategy 死锁修复统一(2026-08-19)] 的无条件接续语义不变:
        准入动作链到该 rank 当前 frontier(实例列车在飞时物理排在列车后)。"""
        if self.config.trace_granularity != "request_aggregated":
            raise RuntimeError(
                "online emission supports request_aggregated granularity only "
                f"(got {self.config.trace_granularity!r})")
        marker = self._mark()
        self._set_context(request_plan, "prefill", 0)
        if (request_plan["turn_index"] == 0
                and request_plan["request_id"] not in self.pending_history):
            # turn-0 到达 timer gates 在段 1 批次内发射(与准入动作同批提交,
            # 保证 arm_timer_gate 的依赖边可解析)。
            self._emit_arrival_gate(request_plan)
        self._emit_admission_actions(request_plan)
        self._collect(marker)

    def emit_completion_batch(self, request_plan: dict) -> dict:
        """段 3(REQUEST_COMPLETE 边界):completion_evictions + 下一 turn 的
        interval timer gates。返回 REQUEST_COMPLETE watch 成员
        {rank: 段内末节点 id}(本段无 end barrier,成员即段内真实末节点);
        无 completion_evictions 时成员 = 段 2 的 decode 块末(调用方按
        _block_ends["seg2"] 的 post-barrier 口径显式记录;REQUEST_COMPLETE
        无独立 watch 注册,decode watch fire 同时推送两条决策)。"""
        self._set_context(request_plan, "decode", 1)
        marker = self._mark()
        # 注：_seg3_before 在本仓是活属性（_mark_block_end 的 before 快照
        # 入参，段内 touched_ranks 判定用），与 sh_3.0 母本的死属性不同，
        # 保留不删。
        self._seg3_before = {r: marker[r][0] for r in marker}
        members = self._emit_segment3(request_plan)
        self._collect(marker)
        # M4 核销即删（2026-08-23）：completion 批是本请求图发射的终点
        # （action 序号此后无读者——下一 turn 是不同 request_id，全仓
        # grep 证实无更晚读者），action 计数账本条目在请求完成后即死重，
        # 当场弹出。_block_ends[req] 的弹出在调度器 _emit_segment3 的
        # seg2 兜底回读之后执行（sh_1.0 策略差异点：无 completion_
        # evictions 时 watch 成员 = _block_ends["seg2"]，builder 内弹出
        # 会截断该读者）。
        self._action_sequence_by_request.pop(request_plan["request_id"], None)
        return members

    def _emit_admission_actions(self, request_plan: dict) -> None:
        """发射在线请求的准入动作:到达 gates、历史/KV 准备与屏障。

        [frontier 接续裁决,strategy 死锁修复统一(2026-08-19,移植
        sh_2.0 已验证修复)]:strategy 不做任何块末恢复/段内清链——
        per-rank previous_id 无条件接续当前 frontier(= 离线 writer 跨
        request 物理链同构),per-rank 发行序 = 全局发射序,跨实例 P2P
        与 collective 参与序不可能反转成环(sh_1.0 strategy 真实网络下
        的实测死锁机理与修复记录见 sh_1.0改造执行实录.md §15.1);同
        session 串行化由 interval gate(after_node_id 显式编码)保留。"""
        builders = self.builders
        prefill_group = self.group_by_index[
            request_plan["prefill_instance_index"]]
        prefix = _prefix_of(request_plan)

        pending_gate = self.pending_history.pop(request_plan["request_id"], None)
        if pending_gate is None:
            raise RuntimeError(
                f"request {request_plan['request_id']} has no arrival/history gate")
        self.pending_request_by_session.pop(request_plan["session_id"], None)

        def emit_transfer(transfer: KVTransfer, stage: str, *,
                          gate=None, trigger_gate=None) -> None:
            action_sequence = self._action_sequence_by_request
            action_name = (
                f"{prefix}_{stage}_action{action_sequence[request_plan['request_id']]:03d}_"
                f"{sanitize_node_prefix(transfer.session_id)}_{transfer.kind}"
            )
            _emit_kv_transfer(
                config=self.config,
                builders=builders,
                group_by_index=self.group_by_index,
                tag_allocator=self.tag_allocator,
                transfer=transfer,
                action_name=action_name,
                pending_gate=gate,
                trigger_gate=trigger_gate,
            )
            action_sequence[request_plan["request_id"]] += 1
            # 2026-08-23 修订：发射侧标记已移除（决策时点同步是唯一路径，
            # 迟到的旧转移标记会把门回退到过期位置，见 sh_2.0 seq4689 事故）。

        self._action_sequence_by_request.setdefault(
            request_plan["request_id"], 0)
        history_eviction_trigger = TransferTriggerGate(
            control_instance_index=pending_gate["source_instance_index"],
            node_gates=pending_gate["timer_gates"],
        )
        for transfer in _restore_transfers(request_plan["history_evictions"]):
            emit_transfer(transfer, "history_evictions",
                          trigger_gate=history_eviction_trigger)

        history_transfer = request_plan.get("history_transfer")
        if history_transfer is None:
            if request_plan["turn_index"] != 0:
                raise RuntimeError(
                    "later request is missing its history transfer action")
            source_group = self.group_by_index[
                pending_gate["source_instance_index"]]
            if source_group.ranks != prefill_group.ranks:
                raise RuntimeError(
                    "first-request arrival gate is not on its Prefill ranks")
            for relative_index, rank in enumerate(prefill_group.ranks):
                builders[rank].arm_timer_gate(
                    pending_gate["timer_gates"][relative_index])
        else:
            location_before = request_plan.get("history_location_before")
            if location_before is None:
                raise RuntimeError(
                    "history transfer is missing its location snapshot")
            if pending_gate["location"] != location_before:
                raise RuntimeError(
                    f"history gate location {pending_gate['location']!r} does "
                    f"not match planned location {location_before!r}")
            emit_transfer(
                _restore_kv_transfer(history_transfer),
                "history_transfer",
                gate=_gate_dataclass(pending_gate),
            )

        for transfer in _restore_transfers(request_plan["prefill_evictions"]):
            emit_transfer(transfer, "prefill_evictions")

        _emit_tp_readiness_barrier(
            builders=builders,
            group=prefill_group,
            name=f"{prefix}_prefill_kv_ready_barrier",
        )
        # prefill 主体(chunk spans + end barrier + seg1 块末)自拼 batch
        # 改造(2026-08-22)起移入 emit_iteration_train 的折叠体与 drain
        # 标记;此处止于准入动作(到达 gates/历史迁移/逐出/屏障)。

    def emit_iteration_train(self, train_plan: dict) -> dict:
        """发射一趟实例迭代列车(拼 batch 改造核心,2026-08-22;设计
        文档 §3.2"迭代列车聚合发射")。

        train_plan(调度器冻结的成员快照)字段:
          train_id          批命名空间 id("batch_train_i<实例>_<序号>";
                            共享体节点归属,物理完成与逻辑请求完成分离)
          instance_index    列车所在实例
          stage             "decode"(有 decode 成员,含混合迭代)或
                            "prefill"(纯 prefill 列车)
          joiners           新成员 request_plan 列表(含 decode_evictions/
                            prefill_decode_transfer/prefill_drain_block_ends
                            {rank: drain 列车 barrier 节点 id})——迁移
                            节点先于列车体,经共享 readiness barrier 栅栏
          pass_spans        成员×迭代展开的 (tokens, kv) 平铺列表
          iterations        迭代数(= weight_passes:权重字节 ×迭代数,
                            与批成员数无关;激活/KV/AR 逐 span 精确)
          drain_members     本列车内完成最后 prefill chunk 的请求 plan 列表
          exit_members      本列车内退出 decode 的成员 plan 列表
          first_token       WP9 首 token 观测(2026-08-26;缺省 None = 拆分
                            开关关闭,行为与上线前逐字节一致):{"split":
                            False 时仅做不拆车的标记增强——多 token debut
                            成员在列车体后挂 first_token 标记,decode_length
                            =1 的 debut 成员其 exit 标记改名为 _exit_first_
                            token_(C++ 名字子串锚点,first_token==completion
                            不变量由同节点保证);"split": True 时本方法拒绝,
                            阶段 1/2 经 emit_train_first_step /
                            emit_train_remainder 发射}

        每实例 rank 上的结构(链序):
          [joiner 迁移 ...] → [共享 readiness barrier] → 17 类聚合体节点
          (transformer_pass_aggregated, weight_passes=iterations) →
          [first_token 标记(不拆车增强时)] → [drain 标记 ...] →
          [exit 标记 ...] → 共享 end barrier。

        返回 {"drain_members": {request_id: {rank: 标记节点 id}},
              "exit_members": {request_id: {rank: 标记节点 id}},
              "block_ends": {rank: end barrier 节点 id}}——标记即
        PREFILL_DRAIN / DECODE_COMPLETION watch 成员(barrier 前末节点
        口径,R2-2 同款:标记是列车体后、end barrier 前的真实节点);
        块末账本 _block_ends[req]["seg1"]/["seg2"] = 本列车 post-barrier
        节点(decode_evictions/completion_evictions 触发门与下一 turn
        interval gate after_node_id 的来源,五仓一致口径不随 watch 锚点
        变化)。"""
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
        """WP9 首步批发射(2026-08-26;拆分阶段 1)。

        首步批 = 所有列车成员的第 1 个 span(decode 成员首迭代)+ prefill
        队头的第 1 个 chunk,加上按迭代位置锚定在首步内的 joiner 迁移/
        readiness barrier/起始标记(挂点语义与整列发射完全一致)。列车体
        以 weight_passes=1 折叠(权重恰读一次),体后挂各 debut 成员的
        first_token 标记节点(1-op COMP,名字含 "first_token" 子串——
        C++ 锚点按名字子串注册 code 8,取每 rank min tick)。

        本批不含任何请求级 watch/drain/exit/哨兵标记;唯一附加物是尾部的
        批命名空间唤醒标记(每 rank 1 个小节点,request_id = "<train_id>_
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
        self._emit_train_body(train_plan, list(first_token["first_spans"]), 1)
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
        改名携带 first_token 子串(同节点锚点保证 first_token==
        completion)。返回结构与 emit_iteration_train 相同。"""
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
        """列车头:joiner 迁移 + 共享 readiness barrier + 起始标记节点。

        拆分时整段归首步批(迁移/栅栏/起始标记锚定的迭代位置在第 1 迭代
        内,语义不变);发射序与整列发射逐节点一致。"""
        instance_index = train_plan["instance_index"]
        group = self.group_by_index[instance_index]
        train_id = train_plan["train_id"]
        stage = train_plan["stage"]
        generation = 1 if stage == "decode" else 0

        # ---- joiner 迁移(触发门 = 该成员 drain 列车的 post-barrier
        #      块末;上下文 (joiner, decode, 1) = decode_start 指标锚点) ----
        joiners = list(train_plan.get("joiners", ()))
        for joiner in joiners:
            self._set_context(joiner, "decode", 1)
            self._action_sequence_by_request.setdefault(
                joiner["request_id"], 0)
            prefix = _prefix_of(joiner)
            drain_gates = joiner.get("prefill_drain_block_ends") or {}
            prefill_group = self.group_by_index[
                joiner["prefill_instance_index"]]
            decode_eviction_trigger = TransferTriggerGate(
                control_instance_index=joiner["prefill_instance_index"],
                node_gates=tuple(
                    drain_gates[rank] for rank in prefill_group.ranks),
            )

            def emit_transfer(transfer: KVTransfer, stage_name: str, *,
                              gate=None, trigger_gate=None) -> None:
                action_sequence = self._action_sequence_by_request
                action_name = (
                    f"{prefix}_{stage_name}_"
                    f"action{action_sequence[joiner['request_id']]:03d}_"
                    f"{sanitize_node_prefix(transfer.session_id)}_"
                    f"{transfer.kind}"
                )
                _emit_kv_transfer(
                    config=self.config,
                    builders=self.builders,
                    group_by_index=self.group_by_index,
                    tag_allocator=self.tag_allocator,
                    transfer=transfer,
                    action_name=action_name,
                    pending_gate=gate,
                    trigger_gate=trigger_gate,
                )
                action_sequence[joiner["request_id"]] += 1
                # 2026-08-23 修订：发射侧标记已移除（决策时点同步是唯一路径，
                # 迟到的旧转移标记会把门回退到过期位置，见 sh_2.0 seq4689 事故）。

            for transfer in _restore_transfers(joiner.get("decode_evictions")):
                emit_transfer(transfer, "decode_evictions",
                              trigger_gate=decode_eviction_trigger)
            prefill_decode_transfer = joiner.get("prefill_decode_transfer")
            if prefill_decode_transfer is None:
                raise RuntimeError(
                    "joiner is missing its Prefill-to-Decode KV action")
            emit_transfer(
                _restore_kv_transfer(prefill_decode_transfer),
                "prefill_decode_transfer",
            )

        # ---- 共享 readiness barrier(仅在有 joiner 时发射;无 joiner 的
        #      列车成员 KV 已就绪,无需再栅栏) ----
        if joiners:
            _emit_tp_readiness_barrier(
                builders=self.builders,
                group=group,
                name=f"{train_id}_decode_kv_ready_barrier",
            )

        # ---- 起始标记节点(指标锚点;§3.5:decode_start = 请求加入后
        #      第一个迭代所在列车节点;prefill_start = 请求首个 chunk
        #      所在列车的首节点。joiner 迁移零节点(如 local_hit)时这是
        #      唯一锚点;迁移有节点时 min-tick 语义取更早者,不冲突) ----
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
                         weight_passes: int) -> None:
        """折叠列车体(17 类聚合节点;weight_passes = 权重读取次数)。

        拆分时首步批传首步 span 组 + weight_passes=1,余量批传余量组 +
        iterations-1;两批激活/KV/AR 字节按 span 求和与整列一致。"""
        instance_index = train_plan["instance_index"]
        group = self.group_by_index[instance_index]
        train_id = train_plan["train_id"]
        stage = train_plan["stage"]
        generation = 1 if stage == "decode" else 0
        for builder in self.builders.values():
            builder.set_context(train_id, stage, generation)
        tensor_parallel = len(group.ranks)
        for relative_rank, rank in enumerate(group.ranks):
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
                    rank, f"{train_id}_first_token_"
                    f"{sanitize_node_prefix(request_id)}")
                for rank in group.ranks
            }
        return members

    def _emit_train_tail_markers(self, train_plan: dict,
                                 first_token) -> dict:
        """列车尾:first_token 标记(不拆车增强时) + drain/exit 标记 +
        哨兵标记 + 共享 end barrier + 块末账本(挂点语义与整列发射一致)。"""
        instance_index = train_plan["instance_index"]
        group = self.group_by_index[instance_index]
        train_id = train_plan["train_id"]
        stage = train_plan["stage"]
        generation = 1 if stage == "decode" else 0
        iterations = int(train_plan["iterations"])

        # ---- WP9:不拆车时的 first_token 标记(iterations==1 或单 debut
        #      无需拆车的场景;拆车时标记已在首步批挂过,余量批不重挂) ----
        if first_token is not None and not first_token.get("split"):
            self._emit_first_token_markers(
                train_plan, first_token["debut_marker_members"])

        # ---- drain / exit 标记(列车体后、end barrier 前;每成员每 rank
        #      1 个小节点,承载该请求的 PREFILL_DRAIN / DECODE_COMPLETION
        #      watch 与指标 end 锚点;物理完成时刻 = 标记完成时刻。
        #      WP9:decode_length=1 的 debut 成员 exit 标记名附加 first_
        #      token 子串——同节点双锚点(code 4 watch + code 8 名字),
        #      first_token_ns == completion_ns 不变量由同一节点保证) ----
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
                    rank, f"{train_id}_sentinel")
                for rank in group.ranks
            }

        # ---- 共享 end barrier(每列车一个,替代每请求一个) ----
        for builder in self.builders.values():
            builder.set_context(train_id, stage, generation)
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

        # ---- 块末账本(post-barrier 口径,五仓一致):drain 成员写 seg1
        #      (decode_evictions 触发门),exit 成员写 seg2(completion_
        #      evictions 触发门 + 下一 turn interval gate after_node_id) ----
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
        """列车标记节点(每 rank 1 个小 COMP 节点;上下文由调用方设置)。"""
        self.builders[rank].comp(name, 1, 1)
        return self.builders[rank].previous_id

    def _emit_arrival_gate(self, request_plan: dict) -> None:
        """turn-0 request 的到达 timer gates(段 1 批次内发射)。
        duration = admission_time_ns(0 时不发射节点,timer_gate 同款跳过)。"""
        if request_plan["turn_index"] != 0:
            raise RuntimeError("arrival gate is a turn-0-only structure")
        # turn-0 gate duration 先做 µs 下取整：admission_time_ns 是准入
        # 时刻的虚拟 tick（Roofline 任意 ns 粒度，如 153516157242），
        # timer_gate 的离线同构校验要求整 µs（duration_ns % 1000 != 0
        # 即 raise）。该 duration 不进节点（runtime_ns=0、不存储），仅
        # 驱动校验与 0 跳过，下取整对既有通过路径零影响；消除
        # 2026-08-22 30s 窗 delivery seq=1145 确定性崩溃（阻塞 turn-0
        # 在非 µs 对齐 tick 准入时触发）。
        duration = request_plan["admission_time_ns"]
        duration -= duration % 1000
        group = self.group_by_index[request_plan["prefill_instance_index"]]
        prefix = (
            f"q{request_plan['queue_index']:04d}_"
            f"{sanitize_node_prefix(request_plan['request_id'])}"
        )
        timers = tuple(
            self.builders[rank].timer_gate(
                f"{prefix}_global_arrival_timer_gate",
                duration,
            )
            for rank in group.ranks
        )
        self.pending_history[request_plan["request_id"]] = {
            "source_instance_index": request_plan["prefill_instance_index"],
            "timer_gates": timers,
            "location": "new_session",
        }
        self.pending_request_by_session[request_plan["session_id"]] = \
            request_plan["request_id"]

    def _emit_segment3(self, request_plan: dict) -> dict:
        """主循环段 3 的在线发射:completion_evictions
        (触发门 = decode 段末节点)+ 下一 turn 的 interval gates。"""
        builders = self.builders
        decode_group = self.group_by_index[
            request_plan["decode_instance_index"]]
        prefix = _prefix_of(request_plan)
        # [frontier 接续裁决,同段 2(sh_2.0 修复移植)] strategy 无条件接续 frontier。
        decode_completion_nodes = tuple(
            self._block_ends.get(request_plan["request_id"], {})
            .get("seg2", {}).get(rank)
            for rank in decode_group.ranks)
        if any(node_id is None for node_id in decode_completion_nodes):
            raise RuntimeError("segment-2 block ends missing on decode ranks")

        self._action_sequence_by_request.setdefault(
            request_plan["request_id"], 0)

        def emit_transfer(transfer: KVTransfer, stage: str, *,
                          gate=None, trigger_gate=None) -> None:
            action_sequence = self._action_sequence_by_request
            action_name = (
                f"{prefix}_{stage}_action{action_sequence[request_plan['request_id']]:03d}_"
                f"{sanitize_node_prefix(transfer.session_id)}_{transfer.kind}"
            )
            _emit_kv_transfer(
                config=self.config,
                builders=builders,
                group_by_index=self.group_by_index,
                tag_allocator=self.tag_allocator,
                transfer=transfer,
                action_name=action_name,
                pending_gate=gate,
                trigger_gate=trigger_gate,
            )
            action_sequence[request_plan["request_id"]] += 1
            # 2026-08-23 修订：发射侧标记已移除（决策时点同步是唯一路径，
            # 迟到的旧转移标记会把门回退到过期位置，见 sh_2.0 seq4689 事故）。

        completion_eviction_trigger = TransferTriggerGate(
            control_instance_index=request_plan["decode_instance_index"],
            node_gates=decode_completion_nodes,
        )
        touched_ranks = set()
        for transfer in _restore_transfers(request_plan["completion_evictions"]):
            emit_transfer(transfer, "completion_evictions",
                          trigger_gate=completion_eviction_trigger)
            for shard in transfer.shards:
                if shard.source_rank is not None:
                    touched_ranks.add(shard.source_rank)
                if shard.target_rank is not None:
                    touched_ranks.add(shard.target_rank)
                if shard.edge_rank is not None:
                    touched_ranks.add(shard.edge_rank)

        members = {}
        for rank in sorted(touched_ranks):
            node_id = builders[rank].previous_id
            if node_id is None:
                raise RuntimeError(
                    f"completion segment rank {rank} has no emitted node")
            members[rank] = node_id
        self._mark_block_end(request_plan["request_id"], "seg3",
                             sorted(touched_ranks), before=self._seg3_before)

        # 下一同 session turn 的 interval gates(duration =
        # interval + hbm_wait_ns——在线 hbm_wait_ns 由策略按准入时刻账本给出,
        # 缺省 0 之外的值由 request_plan["next_hbm_wait_ns"] 携带)。
        following = request_plan.get("following")
        if following is not None:
            following_request = self.config.request_queue[following["queue_index"]]
            interval = following_request.inter_request_interval_ns
            if interval is None:
                raise RuntimeError("later request lost its inter-request interval")
            hbm_wait_ns = following.get("hbm_wait_ns", 0)
            timer_prefix = (
                f"q{following['queue_index']:04d}_"
                f"{sanitize_node_prefix(following['request_id'])}"
            )
            timers = tuple(
                builders[rank].timer_gate(
                    f"{timer_prefix}_history_rank{rank}_interval_gate",
                    interval + hbm_wait_ns,
                    after_node_id=decode_completion_nodes[relative_index],
                )
                for relative_index, rank in enumerate(decode_group.ranks)
            )
            completion_location = request_plan.get("kv_location_after_completion")
            if request_plan["session_id"] in self.deferred_remote_sessions:
                completion_location = "remote_memory"
                self.deferred_remote_sessions.discard(
                    request_plan["session_id"])
            if completion_location not in {"local_hbm", "remote_memory"}:
                raise RuntimeError("completed request has no valid KV location")
            self.pending_history[following["request_id"]] = {
                "source_instance_index": request_plan["decode_instance_index"],
                "timer_gates": timers,
                "location": completion_location,
            }
            self.pending_request_by_session[request_plan["session_id"]] = \
                following["request_id"]
        else:
            self.deferred_remote_sessions.discard(request_plan["session_id"])
        return members


def _gate_dataclass(pending_gate: dict) -> PendingHistoryGate:
    return PendingHistoryGate(
        source_instance_index=pending_gate["source_instance_index"],
        timer_gates=pending_gate["timer_gates"],
        location=pending_gate["location"],
    )


def _prefix_of(request_plan: dict) -> str:
    return (
        f"q{request_plan['queue_index']:04d}_"
        f"{sanitize_node_prefix(request_plan['session_id'])}_"
        f"turn{request_plan['turn_index']}_"
        f"{sanitize_node_prefix(request_plan['request_id'])}"
    )
