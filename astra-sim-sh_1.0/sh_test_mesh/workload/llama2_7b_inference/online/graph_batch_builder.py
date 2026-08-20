#!/usr/bin/env python3
"""graph_batch_builder.py -- sh_1.0 在线 GraphBatch 构图器(方案 §4 步骤 1-8 操作 4)。

阶段 1 最关键的对齐点:复用离线写出逻辑的节点结构。离线 per-request 十段发射
(write_face_trace 主发射循环, generate_face_trace.py:2033-2413)由模块级助手
函数组成(_emit_kv_transfer / _emit_transfer_trigger / _emit_tp_readiness_barrier
/ transformer_pass_aggregated)——本模块直接 import 它们,用 OnlineTraceBuilder
(与 TraceBuilder 同构的在线侧 builder)驱动,保证:

  - 节点属性、插入顺序、rank ownership 跨 request 链结构一致(节点级审计口径);
  - per-rank 节点 id 跨批次全局递增;
  - interval gate 的 after_node_id 指向上一同 session request 的 decode 完成
    barrier 节点 id(pending_history 账本,跨批次解析)。

sh_1.0 三段式发射(方案 §4 步骤 1-8 操作 4;与蓝本两段式的差异,离线
transfer_order_per_request 十段 manifest 的三次分组):
  - 段 1(ARRIVAL 边界) = 到达/interval timer gates + history_evictions +
    history_transfer(或 turn-0 arm gate)+ prefill_evictions + prefill 屏障 +
    prefill 整段(chunked-aggregated + end barrier);
  - 段 2(PREFILL_DRAIN 边界) = decode_evictions(触发门 = prefill 段末
    per-rank 节点)+ prefill→decode 迁移 + decode 屏障 + decode 整段 +
    per-rank decode end barrier;
  - 段 3(DECODE_COMPLETION/REQUEST_COMPLETE 边界) = completion_evictions
    (触发门 = decode 段末 per-rank 节点)+ 下一同 session turn 的 interval
    timer gates(离线 :2373-2399 语义,after_node_id=decode 完成节点,
    duration = interval + hbm_wait_ns)。

在线语义差异(刻意,注释标注;蓝本裁决 3/7/9 的三段推广):
  - timer gate 始终发射节点(结构保留)但 runtime_ns=0:到达/间隔时刻由 C++
    arrival alarm(future_alarms)替代,gate 不再等待;
  - 每段发射后按 (request_id, rank) 记录块末 previous_id(_block_ends 账本,
    emitted-ranks-only 语义),供后续段的触发门与 watch 锚点使用,不回灌
    builders(2026-08-15 的段间恢复裁决已于 2026-08-19 废止,见
    emit_prefill_batch 的 frontier 接续裁决块);
  - strategy 保持物理跨 request 链(无条件接续 frontier);
  - watch 锚点与离线 metrics 锚点一致(2026-08-20 五仓统一,R2-2):
    PREFILL_DRAIN = 每 rank 末个真实 prefill 计算节点(end barrier 之前,
    与离线 EVENT_PREFILL_END 锚点一致,排除 end barrier);
    DECODE_COMPLETION = end barrier 前每 rank 的 decode 末节点(离线
    EVENT_DECODE_END 口径);REQUEST_COMPLETE = completion_evictions 段
    每 rank 末节点(无 completion_evictions 时 = 调用方显式记录的段 2
    decode 块末)。触发门角色(decode_evictions/completion_evictions 的
    node_gates、下一 turn interval gate 的 after_node_id、_block_ends
    账本)仍用 post-barrier 节点(五仓一致口径,不随 watch 锚点变化)。
    决策边界因此比 post-barrier 口径早一个 all_reduce——这是五仓统一的
    预期时间线变化。

离线账本在线复刻:pending_history(request_id -> PendingHistoryGate 等价
dict)/pending_request_by_session/deferred_remote_sessions
(离线 :1950-1987 与 :2373-2402 的语义)。
"""

import os
import sys

# --------------------------------------------------------------------------
# import 路径:本文件位于 workload/llama2_7b_inference/online/,离线写出模块在
# 上一级。路径只做 import 用途(红线:generate_face_trace.py / face_scheduler.py
# 只读 import 与注释)。
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


class OnlineTraceBuilder:
    """与离线 TraceBuilder 同构的在线侧每-rank builder(generate_trace.py:648)。

    同一接口面:timer_gate / arm_timer_gate / comp / all_reduce / comm_send /
    comm_recv / mem_store / mem_load / next_id / previous_id / node_count——
    离线助手函数(_emit_kv_transfer 等)可直接驱动。与 TraceBuilder 的差异:
    节点发射为 GraphBatch nodes[] dict(而非 ChakraNode 字节流),依赖记录为
    parent_edges[](离线 .et 的 data_deps 内联在节点里,在线按边列表携带);
    timer_gate 忽略 duration(runtime_ns=0,alarm 替代等待)。duration==0 时
    与离线 :711-712 同款跳过节点(返回 after_node_id)。
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
        """在线 timer gate:duration==0 与离线 :711-712 同款跳过;否则发射节点
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
    驱动离线 _emit_kv_transfer 发射)。"""
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
        # 离线 pending_history 账本的在线等价(request_id -> dict:
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
        # request_id -> 连续 action 计数(离线 write_face_trace 的
        # per-request action_sequence:2057 计数跨全部 stage 连续;在线三段
        # 发射共用同一计数器,保证 actionNNN 命名与离线 .et 逐节点一致——
        # canonical 命名 key 含 name,分段各自归零会造成同逻辑节点改名)。
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
        # 离线 mark_pending_history_remote(:1954-1959)的在线复刻。
        pending_request_id = self.pending_request_by_session.get(session_id)
        if pending_request_id is None:
            self.deferred_remote_sessions.add(session_id)
            return
        self.pending_history[pending_request_id]["location"] = "remote_memory"

    # ------------------------------------------------------------- 发射 --

    def emit_prefill_batch(self, request_plan: dict) -> dict:
        """段 1(ARRIVAL 边界):返回 PREFILL_DRAIN watch 成员
        {rank: 末个真实 prefill 节点 id}(end barrier 之前捕获,与离线
        EVENT_PREFILL_END 锚点一致,排除 end barrier;四仓统一口径)。
        decode_evictions 触发门不经返回值,仍用 _block_ends["seg1"] 的
        post-barrier 块末。"""
        self._set_context(request_plan, "prefill", 0)
        prefill_group = self.group_by_index[
            request_plan["prefill_instance_index"]]
        # [frontier 接续裁决,strategy 死锁修复统一(2026-08-19,移植 sh_2.0
        # 已验证修复,主控指令 2026-08-16)] strategy **不做任何块末恢复/段内
        # 清链**:段 1 亦不例外——per-rank previous_id 无条件接续当前
        # frontier(= 离线 writer 跨 request 物理链同构),per-rank 发行序 =
        # 全局发射序,跨实例 P2P 与 collective 参与序不可能反转成环。
        # sh_1.0 的 KV transfer 节点(noc_migrate/remote_store/remote_load
        # 的跨 rank send/recv 对)分布在共享 edge rank 上,收发配对序一旦
        # 反转即成环(recv 等待配对 send,send 链在本 rank 排在 recv 之后)
        # ——strategy 真实网络下实测死锁(delivery 142 后 EventQueue 排空、
        # 每 rank 1-2 个 in-flight recv),2026-08-15 的"段 1 清空 prefill
        # 组 previous_id / 段间 own-block-end 恢复"规避裁决自此废止;
        # 同 session 串行化由 interval gate(after_node_id 显式编码)保留,
        # 段 1 块末账本(post-barrier)仅作 decode_evictions 触发门,
        # 不回灌 builders;PREFILL_DRAIN watch 成员自 R2-2(2026-08-20)起
        # 由返回值携带 barrier 前末节点,不经块末账本。
        # sh_1.0改造执行实录.md §15.1 记录。
        marker = self._mark()
        self._seg1_before = {r: marker[r][0] for r in marker}
        if (request_plan["turn_index"] == 0
                and request_plan["request_id"] not in self.pending_history):
            # turn-0 到达 timer gates 在段 1 批次内发射(与段 1 同批提交,
            # 保证 arm_timer_gate 的依赖边可解析)。
            self._emit_arrival_gate(request_plan)
        members = self._emit_segment1(request_plan)
        self._collect(marker)
        return members

    def emit_decode_batch(self, request_plan: dict) -> dict:
        """段 2(PREFILL_DRAIN 边界):返回 DECODE_COMPLETION watch 成员
        {rank: end barrier 前的 decode 末节点 id}(与离线 EVENT_DECODE_END
        锚点一致,排除 end barrier;四仓统一口径)。completion_evictions
        触发门与下一 turn interval gate 的 after_node_id 仍用
        _block_ends["seg2"] 的 post-barrier 块末。"""
        self._set_context(request_plan, "decode", 1)
        marker = self._mark()
        self._seg2_before = {r: marker[r][0] for r in marker}
        members = self._emit_segment2(request_plan)
        self._collect(marker)
        return members

    def emit_completion_batch(self, request_plan: dict) -> dict:
        """段 3(REQUEST_COMPLETE 边界):completion_evictions + 下一 turn 的
        interval timer gates。返回 REQUEST_COMPLETE watch 成员
        {rank: 段内末节点 id}(本段无 end barrier,成员即段内真实末节点);
        无 completion_evictions 时成员 = 段 2 的 decode 块末(调用方按
        _block_ends["seg2"] 的 post-barrier 口径显式记录;REQUEST_COMPLETE
        无独立 watch 注册,decode watch fire 同时推送两条决策)。"""
        self._set_context(request_plan, "decode", 1)
        marker = self._mark()
        self._seg3_before = {r: marker[r][0] for r in marker}
        members = self._emit_segment3(request_plan)
        self._collect(marker)
        return members

    def _emit_segment1(self, request_plan: dict) -> dict:
        """离线 write_face_trace 主循环的段 1 部分(:2044-2240 的在线复刻)。"""
        builders = self.builders
        prefill_group = self.group_by_index[
            request_plan["prefill_instance_index"]]
        decode_group = self.group_by_index[
            request_plan["decode_instance_index"]]
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
            if transfer.kind == "remote_store":
                self._mark_pending_history_remote(transfer.session_id)

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

        # prefill 整段(request-aggregated;span 与离线 :2103-2163 同款)。
        # Context sidecar:按实际工作量发射。recompute 口径下
        # context - history == request.prefill_length;sidecar_restore 口径
        # 下 = input_tokens_total - min(prefix, 账本)(turn-0 prefix 计入,
        # 复用部分剔除),kv span 仍 = history + processed + chunk。
        prefill_spans: list[tuple[int, int]] = []
        processed = 0
        prefill_work_tokens = (
            int(request_plan["prefill_context_tokens"])
            - int(request_plan["history_tokens_before"])
        )
        if prefill_work_tokens <= 0:
            raise ValueError(
                f"request {request_plan['request_id']} has no Prefill work tokens"
            )
        while processed < prefill_work_tokens:
            chunk_tokens = min(self.p_chunk, prefill_work_tokens - processed)
            prefill_spans.append(
                (chunk_tokens,
                 request_plan["history_tokens_before"] + processed + chunk_tokens))
            processed += chunk_tokens
        tensor_parallel = len(prefill_group.ranks)
        prefill_last_node_by_rank = {}
        for relative_rank, rank in enumerate(prefill_group.ranks):
            transformer_pass_aggregated(
                builders[rank],
                phase=f"{prefix}_prefill_request_aggregated",
                pass_spans=prefill_spans,
                layers=self.config.layers,
                hidden_size=self.config.hidden_size,
                ffn_size=self.config.ffn_size,
                tensor_parallel=tensor_parallel,
                pg_name=prefill_group.pg_name,
                vocab_size=self.config.vocab_size,
                bytes_per_elem=self.config.bytes_per_elem,
                num_heads=self.config.num_heads,
                tensor_parallel_rank=relative_rank,
                mlp_variant=self.config.mlp_variant,
            )
            # PREFILL_DRAIN watch 成员 = 每 rank 末个真实 prefill 节点
            # (与离线 EVENT_PREFILL_END 锚点一致,排除 end barrier——五仓
            # 统一口径,R2-2,2026-08-20;sh_2.0/sh_3.0 同款拆分形态)。
            # 捕获点 = 最后一个 prefill 计算节点之后、end barrier 之前。
            prefill_last_node_by_rank[rank] = builders[rank].previous_id
            builders[rank].all_reduce(
                f"{prefix}_prefill_chunks_aggregated_end_barrier",
                len(prefill_spans),
                prefill_group.pg_name,
            )
        # post-barrier 块末(= end barrier 节点)仅承载触发门角色:段 1
        # 块末账本(_block_ends["seg1"])是段 2 decode_evictions 触发门的
        # node_gates 来源,保持 post-barrier(五仓一致口径,不随 watch
        # 锚点变化)。
        prefill_completion_nodes = tuple(
            builders[rank].previous_id for rank in prefill_group.ranks)
        if any(node_id is None for node_id in prefill_completion_nodes):
            raise RuntimeError("Prefill completion node IDs were not generated")

        # 段 1 块末账本(emitted-ranks-only):prefill 组 rank 记实际值,
        # decode 组 rank 本段无节点 -> None(蓝本裁决 7/9 语义)。
        self._mark_block_end(
            request_plan["request_id"], "seg1",
            sorted(set(prefill_group.ranks) | set(decode_group.ranks)),
            before=self._seg1_before)
        # watch 角色(barrier 前末节点)与触发门角色(post-barrier 块末)
        # 至此拆分:返回值只承载 PREFILL_DRAIN watch 成员。
        return dict(prefill_last_node_by_rank)

    def _emit_segment2(self, request_plan: dict) -> dict:
        """离线主循环的段 2 部分(:2241-2360 的在线复刻)。"""
        builders = self.builders
        prefill_group = self.group_by_index[
            request_plan["prefill_instance_index"]]
        decode_group = self.group_by_index[
            request_plan["decode_instance_index"]]
        prefix = _prefix_of(request_plan)
        request = self.config.request_queue[request_plan["queue_index"]]
        # [frontier 接续裁决,strategy 死锁修复(移植 sh_2.0 已验证修复,
        # 主控指令 2026-08-16)] strategy **不做任何块末恢复/段内清链**:per-rank
        # previous_id 无条件接续当前 frontier(= 离线 writer 跨 request
        # 物理链同构),per-rank 发行序 = 全局发射序,跨实例 P2P 与
        # collective 参与序不可能反转成环(sh_2.0 waits-for 环证据同款
        # 机理;差分归因类别②既有口径)。
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
            if transfer.kind == "remote_store":
                self._mark_pending_history_remote(transfer.session_id)

        self._action_sequence_by_request.setdefault(
            request_plan["request_id"], 0)
        prefill_completion_nodes = tuple(
            self._block_ends.get(request_plan["request_id"], {})
            .get("seg1", {}).get(rank)
            for rank in prefill_group.ranks)
        if any(node_id is None for node_id in prefill_completion_nodes):
            # 段 1 发射后 prefill 组必有块末;缺即账本损坏(fail-closed)。
            raise RuntimeError("segment-1 block ends missing on prefill ranks")
        decode_eviction_trigger = TransferTriggerGate(
            control_instance_index=request_plan["prefill_instance_index"],
            node_gates=prefill_completion_nodes,
        )
        for transfer in _restore_transfers(request_plan["decode_evictions"]):
            emit_transfer(transfer, "decode_evictions",
                          trigger_gate=decode_eviction_trigger)
        prefill_decode_transfer = request_plan.get("prefill_decode_transfer")
        if prefill_decode_transfer is None:
            raise RuntimeError("request is missing its Prefill-to-Decode KV action")
        emit_transfer(
            _restore_kv_transfer(prefill_decode_transfer),
            "prefill_decode_transfer",
        )

        _emit_tp_readiness_barrier(
            builders=builders,
            group=decode_group,
            name=f"{prefix}_decode_kv_ready_barrier",
        )

        # decode 整段(request-aggregated;span 与离线 :2313-2316 同款)。
        tensor_parallel = len(decode_group.ranks)
        decode_spans = tuple(
            (1, request_plan["prefill_context_tokens"] + step + 1)
            for step in range(int(request.decode_length))
        )
        decode_last_node_by_rank = {}
        for relative_rank, rank in enumerate(decode_group.ranks):
            transformer_pass_aggregated(
                builders[rank],
                phase=f"{prefix}_decode_request_aggregated",
                pass_spans=decode_spans,
                layers=self.config.layers,
                hidden_size=self.config.hidden_size,
                ffn_size=self.config.ffn_size,
                tensor_parallel=tensor_parallel,
                pg_name=decode_group.pg_name,
                vocab_size=self.config.vocab_size,
                bytes_per_elem=self.config.bytes_per_elem,
                num_heads=self.config.num_heads,
                tensor_parallel_rank=relative_rank,
                mlp_variant=self.config.mlp_variant,
            )
            # DECODE_COMPLETION watch 成员 = end barrier 前每 rank 的
            # decode 末节点(与离线 EVENT_DECODE_END 锚点一致,排除 end
            # barrier——五仓统一口径,R2-2,2026-08-20;sh_2.0/sh_3.0
            # 同款拆分形态)。捕获点 = 最后一个 decode 计算节点之后、
            # end barrier 之前。
            decode_last_node_by_rank[rank] = builders[rank].previous_id
            builders[rank].all_reduce(
                f"{prefix}_decode_request_end_barrier",
                1,
                decode_group.pg_name,
            )
        # post-barrier 块末(= end barrier 节点)仅承载触发门角色:
        # _block_ends["seg2"] 是段 3 completion_evictions 触发门与下一
        # turn interval gate after_node_id 的来源,保持 post-barrier
        # (五仓一致口径,不随 watch 锚点变化)。
        decode_completion_nodes = tuple(
            builders[rank].previous_id for rank in decode_group.ranks)
        if any(node_id is None for node_id in decode_completion_nodes):
            raise RuntimeError("Decode completion node IDs were not generated")

        self._mark_block_end(request_plan["request_id"], "seg2",
                             sorted(set(decode_group.ranks)),
                             before=self._seg2_before)
        return dict(decode_last_node_by_rank)

    def _emit_arrival_gate(self, request_plan: dict) -> None:
        """turn-0 request 的到达 timer gates(离线 :1961-1987 预发射的在线
        等价:段 1 批次内调用)。duration 参数与离线一致(admission_time_ns;
        0 时离线亦不发射节点,在线同款跳过)。"""
        if request_plan["turn_index"] != 0:
            raise RuntimeError("arrival gate is a turn-0-only structure")
        group = self.group_by_index[request_plan["prefill_instance_index"]]
        prefix = (
            f"q{request_plan['queue_index']:04d}_"
            f"{sanitize_node_prefix(request_plan['request_id'])}"
        )
        timers = tuple(
            self.builders[rank].timer_gate(
                f"{prefix}_global_arrival_timer_gate",
                request_plan["admission_time_ns"],
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
        """离线主循环的段 3 部分(:2362-2402 的在线复刻):completion_evictions
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
            if transfer.kind == "remote_store":
                self._mark_pending_history_remote(transfer.session_id)

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

        # 下一同 session turn 的 interval gates(离线 :2373-2399;duration =
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
