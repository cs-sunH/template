#!/usr/bin/env python3
"""graph_batch_builder.py -- 在线 GraphBatch 构图器(方案 §4 步骤 1-8 操作 4)。

阶段 1 最关键的对齐点:复用共享发射原语的节点结构。per-request 发射
(generate_face_trace.py 模块级函数)由
助手函数组成(_emit_control_trigger / _paired_transfer /
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
    MEM_LOAD_NODE,
    MEM_STORE_NODE,
    transformer_pass_aggregated,
)
from generate_face_trace import (  # noqa: E402
    PendingHistoryGate,
    TransferTagAllocator,
    TransferTriggerGate,
    _emit_control_trigger,
    _emit_kv_transfer,
    _paired_transfer,
    kv_cache_bytes_for_tokens,
    sanitize_node_prefix,
)
from generate_face_trace import NOC_MIGRATE  # noqa: E402
from session_kv_manager import (  # noqa: E402
    KVTransfer,
    KVTransferShard,
    LOCAL_HIT,
    NO_HISTORY,
)

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

    def arm_dependency(self, node_id) -> None:
        """B3(2026-09-06,sh :224-226):同 rank 通用依赖挂载(跨批次边——
        经持久 (rank,id) 解析;跨 rank 依赖边被桥拒绝,跨实例时由发射侧
        1B p2p 承载时序)。"""

        if node_id is not None:
            self.pending_extra_dependencies.append(int(node_id))

    def chain_checkpoint(self):
        """B3(2026-09-06,sh :228-229):并行分支起点快照(previous_id +
        未消费的 pending 依赖)。"""

        return self.previous_id, tuple(self.pending_extra_dependencies)

    def restore_chain(self, checkpoint) -> None:
        """B3(2026-09-06,sh :231-233):恢复链快照——恢复分支悬挂后,主链
        (以及后续列车体)不受其阻塞。"""

        self.previous_id, dependencies = checkpoint
        self.pending_extra_dependencies = list(dependencies)

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

    def mem_store(self, name: str, tensor_size: int,
                  hbm_access_mode: int = 0) -> None:
        """B3(2026-09-06,sh :283-289):MEM_STORE 原语。``hbm_access_mode``
        进 compute 键(在线 snake 键与离线 kebab ET attr 同义;仅真值发射,
        absent = 无本地 HBM 访问)。"""

        node = self._new_node(name, MEM_STORE_NODE)
        node["compute"]["tensor_size"] = self._uint64(tensor_size)
        if hbm_access_mode:
            node["compute"]["hbm_access_mode"] = int(hbm_access_mode)

    def mem_load(self, name: str, tensor_size: int,
                 hbm_access_mode: int = 0) -> None:
        """B3(2026-09-06,sh :291-296):MEM_LOAD 原语(remote_load 链的边缘
        池 FIFO 读,默认零本地 HBM 计费)。"""

        node = self._new_node(name, MEM_LOAD_NODE)
        node["compute"]["tensor_size"] = self._uint64(tensor_size)
        if hbm_access_mode:
            node["compute"]["hbm_access_mode"] = int(hbm_access_mode)

    def local_hbm_kv_restore(self, name: str, tensor_size: int) -> None:
        """B3(2026-09-06,sh :298-302):目标 HBM DMA 写(RESTORE,可与推理
        计算重叠)。节点级可选键 ``is_local_hbm_kv_restore`` 仅真值发射
        (契约 §2:absent == false)——恢复不得退化为普通 mem_load。"""

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
            # B3(sh :320-332):直通流量不建本端 HBM 作业(默认 true=发送端
            # HBM 读,absent = charged;契约 §12.5 计费三险之一)。
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


class _HistorySnapshot:
    """B3(sh :351-358):history_location_before 的轻量重建——face 的
    strategy 路径直接携带 SessionKVSnapshot 对象(字段同名),本类仅供
    dict 形状(决策日志回放)的重建路径使用。"""

    def __init__(self, location: str, instance_index, resident_prefix_layers: int):
        self.location = location
        self.instance_index = instance_index
        self.resident_prefix_layers = resident_prefix_layers


def kv_transfer_from_log(record) -> KVTransfer:
    """B3(sh :361-396):从 KVTransfer dict 重建对象;strategy 路径传入的
    已是 KVTransfer 对象(face_online_scheduler 只读复用),原样返回。"""
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
    """B3(sh :399-422):重建 history_location_before。face 的 plan
    ["history_location_before"] 直接是 SessionKVSnapshot 对象(字段同名,
    strategy 模式,live 路径),原样返回;dict 形状按三字段重建——仅供
    旧决策日志(legacy old-log-only)回放解析,新运行不再产生
    partial_hbm_remote 位置,该推断分支对旧产物只读。"""
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
    """在线构图器:持有 per-rank OnlineTraceBuilder(状态跨批次),按决策边界
    发射 prefill 整段 / decode 整段,并维护 completion_gates 账本。"""

    def __init__(self, config):
        self.config = config
        # strategy 模式保持物理跨 request 链。
        self.builders = {
            rank: OnlineTraceBuilder(
                rank, remote_operand_loads=config.remote_operand_loads)
            for rank in range(config.npus_count)
        }
        self.group_by_index = dict(enumerate(config.inference_groups))
        # session_id -> (decode_instance_index, {rank: end_barrier_id});
        # completion_gates 只保存下一同 session turn 尚未消费的 interval
        # gate 来源；turn>0 admission 或 terminal completion 后即释放。
        self.completion_gates = {}
        # ---- B3(2026-09-06,sh :439-461):两态 KV 发射账本(session 级
        # Tiered-LRU 后 location 值域收敛 {local_hbm, remote_memory},
        # note_request_complete 值域校验同口径)----
        # 跨请求 history location 链(face 适配:face 的 interval gates 在
        # 下一 turn 准入时发射,history 门不跨批驻留,location 链经
        # deferred_session_locations 承载,见 note_request_complete)。
        self.pending_request_by_session = {}
        self.deferred_session_locations = {}
        # request_id -> {"seg1": {rank: node_id|None}, "seg2": {rank: node_id|None}}
        # 列车块末账本(sh :443-449;post-barrier 口径):seg1 = drain 列车
        # barrier(joiner decode_evictions 触发门),seg2 = 退出列车 barrier
        # (completion_gates 的同源口径)。
        self._block_ends = {}
        # tag 分配器(契约 §6:基址 10_000_000,错开 face 现行
        # queue*10000+{1000,1900,3000} 段)。
        self._tag_allocator = TransferTagAllocator()
        # B4(2026-09-13,逐出支链化):session_id -> [(edge_rank,
        # mem_store_node_id, source_ack_node_id)] 在飞 store 支链尾部
        # 登记表(主方案 §3.3)。session 级 Tiered-LRU 下逐出恒为单笔整体
        # store,每会话同至多一条登记(restore 读全区间须等齐);回迁发射
        # 时消费清除,terminal session 回收时清除。
        self.pending_store_tails = {}
        # 每 request 的 action_sequence 账本(节点名含 _action{seq:03d}_,
        # 跨该 request 的全部 transfer 递增——保证节点名跨 request 逐字节
        # 稳定)。
        self._action_sequence = {}
        self.batch = None  # 当前批次累加器(由 begin_batch 建立)

    # ------------------------------------------------------------- 批次 --

    def begin_batch(self) -> None:
        self.batch = {
            "nodes": [],
            "parent_edges": [],
            # Private exact ledger: _collect already knows the source rank of
            # every appended node, so downstream GraphBatch metadata need not
            # rescan the complete node payload.
            "_touched_ranks": set(),
            "watches": [],
            "assignments": [],
            "kv_actions": [],
            "future_alarms": [],
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

    def _mark(self) -> dict:
        return {
            rank: (len(builder.nodes), len(builder.edges))
            for rank, builder in self.builders.items()
        }

    def _emit_side_branch(self, emit_fn) -> None:
        """把一段逐出发射包成旁路分支:全 rank 暂存并清空既有 pending 依赖
        → fork 快照 → 发射(分支内自行接续成链;分支触发门的 arming 必须
        在 emit_fn 内部完成)→ 恢复主链 → 归还暂存依赖。分支不 join——
        被包裹的物理逐出链不再阻塞主链任何节点,HBM 争用由 C++
        LocalHbmBandwidthModel 的 N-way 均分模型在线裁决。

        fork 点可能合法携带属于主链的 armed 依赖(turn-0 准入形态:到达门
        armed 后尚未被首个主链节点消费,而 turn-0 也会为腾容量逐出其他
        会话)——暂存清空使其不进分支,主链恢复后照常由本来的消费者消费,
        图依赖与逐出链在主链上时完全一致。分支内部 arming 而未被任何
        节点消费(触发门泄漏)仍 fail-closed(主方案 §3.2 统一契约,
        2026-09-13 勘误)。"""
        stashed = {}
        for rank, builder in self.builders.items():
            stashed[rank] = builder.pending_extra_dependencies
            if builder.pending_extra_dependencies:
                builder.pending_extra_dependencies = []
        checkpoints = {
            rank: builder.chain_checkpoint()
            for rank, builder in self.builders.items()
        }
        try:
            emit_fn()
            for builder in self.builders.values():
                if builder.pending_extra_dependencies:
                    raise RuntimeError(
                        "side-branch left unconsumed pending deps on "
                        f"rank {builder.rank}")
        finally:
            for rank, builder in self.builders.items():
                builder.restore_chain(checkpoints[rank])
                if stashed[rank]:
                    builder.pending_extra_dependencies.extend(stashed[rank])

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
        """列车头:joiner decode 逐出(触发门 = drain 列车 post-barrier)
        + prefill→decode 迁移(face 3000 类 paired transfer)+ join/pstart
        起始标记节点。

        拆分时整段归首步批(face 迁移/起始标记锚定的迭代位置在第 1 迭代
        内,语义不变);发射序与整列发射逐节点一致。
        B3(2026-09-06,sh :702-761):joiner 的 decode_evictions(remote_store
        链)先于迁移发射,触发门 = 该成员 drain 列车的 post-barrier 块末
        (runtime.drain_block_ends,调度器随 plan 传入)。"""
        group = self.group_by_index[train_plan["instance_index"]]
        train_id = train_plan["train_id"]

        # ---- joiner 段(上下文 (joiner, decode, 1) = decode_start 指标
        #      锚点) ----
        joiners = list(train_plan.get("joiners", ()))
        for joiner in joiners:
            self._set_context(joiner, "decode", 1)
            decode_evictions = [
                kv_transfer_from_log(transfer)
                for transfer in (joiner.get("decode_evictions") or ())
            ]
            drain_gates = joiner.get("prefill_drain_block_ends") or {}
            if decode_evictions:
                prefill_group = self.group_by_index[
                    joiner["prefill_instance_index"]]
                decode_eviction_trigger = TransferTriggerGate(
                    control_instance_index=joiner["prefill_instance_index"],
                    node_gates=tuple(
                        drain_gates.get(rank) for rank in prefill_group.ranks),
                )
                # B4(2026-09-13):decode 逐出旁路分支——drain 列车
                # post-barrier 触发门原样保留(门只对齐逐出的开始时刻),
                # 分支与 decode 列车计算时间重叠,HBM 争用由 N-way 均分
                # 在线裁决;紧随其后的 joiner prefill→decode 迁移
                # (_paired_transfer)保持主链不动(恢复类迁移语义上必须
                # 先于 decode 计算完成)。

                def _emit_decode_evictions():
                    for transfer in decode_evictions:
                        self._register_store_tails(self._emit_plan_transfer(
                            joiner, transfer,
                            "decode_evictions",
                            trigger_gate=decode_eviction_trigger))

                self._emit_side_branch(_emit_decode_evictions)
            _paired_transfer(
                config=self.config, builders=self.builders,
                queue_index=joiner["queue_index"], category=3000,
                name="{}_prefill_to_decode_kv".format(_prefix_of(joiner)),
                source_group=self.group_by_index[
                    joiner["prefill_instance_index"]],
                target_group=self.group_by_index[
                    joiner["decode_instance_index"]],
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

    def _emit_train_body(self, train_plan: dict, pass_spans,
                         weight_passes: int) -> None:
        """折叠列车体(17 类聚合节点;weight_passes = 权重读取次数)。

        拆分时首步批传首步 span 组 + weight_passes=1,余量批传余量组 +
        iterations-1;两批激活/KV/AR 字节按 span 求和与整列一致。
        Session-level Tiered-LRU (2026-09-25):partial 恢复列车的首 chunk
        prefix/suffix 层段两段拆分发射随 PARTIAL 态一并删除——恢复恒为
        单笔全量 remote_load,列车体统一单一聚合发射(WP9 首步/余量两批
        机制本身不动,与本改动正交)。"""
        group = self.group_by_index[train_plan["instance_index"]]
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
                weight_passes=weight_passes,
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
        #      旧 decode end barrier 的同款口径)----
        # B3(sh :948-956):post-barrier 块末账本(五仓一致口径)——drain
        # 成员写 seg1(joiner decode_evictions 触发门),exit 成员写 seg2
        # (completion 段触发门口径;face 的 completion 路径零逐出,seg2 由
        # completion_gates 同源承载,此处仅为账本对齐与 REQUEST_COMPLETE
        # 回收点服务)。
        for member in train_plan.get("drain_members", ()):
            self._block_ends.setdefault(member["request_id"], {})[
                "seg1"] = dict(block_ends)
        for member in train_plan.get("exit_members", ()):
            self._block_ends.setdefault(member["request_id"], {})[
                "seg2"] = dict(block_ends)
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
        """列车标记节点(每 rank 1 个小 COMP 节点;上下文由调用方设置)。"""
        self.builders[rank].comp(name, 1, 1)
        return self.builders[rank].previous_id

    def _set_context(self, request_plan: dict, stage: str,
                     generation: int) -> None:
        request_id = request_plan["request_id"]
        for builder in self.builders.values():
            builder.set_context(request_id, stage, generation)

    def retire_completion_gate(self, session_id: str) -> None:
        """释放 terminal session 不会再被下一 turn 消费的完成门。"""
        self.completion_gates.pop(session_id, None)
        # terminal completion:该 session 不会再有下一 turn 消费 deferred
        # location 链(sh _emit_completion :1321-1325 同款回收)。
        self.deferred_session_locations.pop(session_id, None)
        self.pending_request_by_session.pop(session_id, None)
        # B4:terminal session 不会再有回迁消费在飞 store 尾部登记
        # (与 kv_manager.retire_terminal_session 同点位,不发传输)。
        self.pending_store_tails.pop(session_id, None)

    def note_request_complete(self, session_id: str, following_request_id,
                              completion_location: str) -> None:
        """B3(sh ``_emit_completion`` :1341-1355 的 face 适配):REQUEST_COMPLETE
        边界登记下一 turn 的 pending location 链。

        face 的 interval gates 在下一 turn 准入时发射(不同于 sh 在完成时
        发射),故此处只登记 location(completion 态 + 后续逐出经
        sync_pending_history_after_evictions 覆盖),PendingHistoryGate 在
        下一 turn 准入时以 interval gate 现场构建。完成路径零逐出
        (mark_complete 不逐出),location 即完成快照。"""
        if following_request_id is None:
            self.deferred_session_locations.pop(session_id, None)
            self.pending_request_by_session.pop(session_id, None)
            return
        # Session-level Tiered-LRU:two-state location domain only.
        if completion_location not in {"local_hbm", "remote_memory"}:
            raise RuntimeError("completed request has no valid KV location")
        self.deferred_session_locations[session_id] = completion_location
        self.pending_request_by_session[session_id] = following_request_id

    def retire_request_state(self, request_id: str) -> None:
        """B3(sh ``emit_completion_batch`` M4 核销即删的 face 落点):
        REQUEST_COMPLETE 后逐出块末/action 计数账本条目即死重,当场弹出。"""
        self._block_ends.pop(request_id, None)
        self._action_sequence.pop(request_id, None)

    def sync_pending_history_after_evictions(self, transfers) -> None:
        """B3(sh :1253-1260,历史事故修复版):决策时点补偿——KV 变更点返回
        的逐出转移立即镜像到 pending location 链(face:经
        deferred_session_locations 承载,见 note_request_complete)。
        唯一标记路径(发射侧不重复标记;多级逐出乱序时重复标记会把门
        回退到过期位置)。"""
        for transfer in transfers or ():
            transfer = kv_transfer_from_log(transfer)
            if transfer.kind == "remote_store":
                self._mark_pending_history_store(transfer)

    def _mark_pending_history_store(self, transfer: KVTransfer) -> None:
        # 与共享 history-gate 账本一致(sh :1262-1292;turn-0 deferred 通道
        # 保留:in-flight turn-0 request 的 session key 无 pending 门,
        # 逐出标记推迟到 session 完成时结算)。Session-level Tiered-LRU:
        # remote_store 恒为整体外迁(resident_prefix_layers_after == 0),
        # 部分层域 store fail-closed 拒绝。
        if transfer.resident_prefix_layers_after != 0:
            raise RuntimeError("remote store did not fully offload the session")
        location = "remote_memory"
        session_id = transfer.session_id
        # face:下一 turn 准入时才现场构建 pending 门(PendingHistoryGate),
        # remote_store 的 location 一律落 deferred、准入时作为 location 链
        # 初值(sh turn-0 事故同款语义;pending_history 账本在 face 适配下
        # 零写入恒空,原先对它的双分支查询动作等价,已并为直写)。
        self.deferred_session_locations[session_id] = location

    def emit_eviction_actions(self, request_plan: dict, transfers) -> None:
        """B3:独立逐出动作发射(face 特有路径——admission_blocked 的
        prepare/grow 逐出是真实 mutation(ensure_physical_fit 失败不回滚),
        物理链必须随当前批发射;节点上下文 = 阻塞请求的 (prefill, 0),
        in-flight 资格门天然放行)。无触发门——链到各 rank 当前 frontier
        (strategy 无条件接续语义)。无 watch 纯 mem 批不驱动决策交付:
        本方法只随既有决策边界的批调用,不单独成批。
        B4(2026-09-13):发射循环 fork 到旁路分支——逐出物理链不再阻塞
        主链其后节点;_set_context 与 _mark/_collect 留在分支外(上下文
        stamping 供 C++ in-flight 资格门消费,节点仍随本批收集交付,
        "无 watch 纯 mem 批不驱动决策交付"语义不变)。"""
        self._set_context(request_plan, "prefill", 0)
        marker = self._mark()
        blocked_transfers = tuple(transfers or ())
        if blocked_transfers:

            def _emit_blocked_evictions():
                for transfer in blocked_transfers:
                    self._register_store_tails(self._emit_plan_transfer(
                        request_plan, kv_transfer_from_log(transfer),
                        "blocked_admission_evictions"))

            self._emit_side_branch(_emit_blocked_evictions)
        self._collect(marker)

    # ------------------------------------------------- per-request 发射主体 --

    def _emit_plan_transfer(self, request_plan: dict, transfer: KVTransfer,
                            stage: str, *, gate=None,
                            trigger_gate=None) -> dict:
        """B3(sh :990-1017):发射一个 KV 迁移动作(准入/列车 joiner/阻塞
        逐出段共用;action 序号跨该 request 的全部段共享,保证 canonical
        命名稳定)。发射侧不重复标记 pending 门(决策时点同步是唯一标记
        路径)。"""
        prefix = _prefix_of(request_plan)
        action_name = (
            "{}_{}_action{:03d}_{}_{}".format(
                prefix, stage,
                self._next_action_sequence(request_plan),
                sanitize_node_prefix(transfer.session_id),
                transfer.kind)
        )
        return _emit_kv_transfer(
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

    def _next_action_sequence(self, request_plan: dict) -> int:
        key = request_plan["request_id"]
        value = self._action_sequence.get(key, 0)
        self._action_sequence[key] = value + 1
        return value

    def _register_store_tails(self, record) -> None:
        """B4(主方案 §3.3):逐出 remote_store 支链的尾部登记——同会话下一
        轮回迁(全量 remote_load)发射时查表补
        store→restore 前递依赖。粒度 = 边缘 mem_store 完成(池写落盘);
        source_ack 仅为保守变体备查。"""
        if record.get("kind") != "remote_store":
            return
        entries = self.pending_store_tails.setdefault(
            record["session_id"], [])
        for shard in record["shards"]:
            store_node_id = shard.get("edge_store_node_id")
            if not isinstance(store_node_id, int):
                raise RuntimeError(
                    "remote store shard is missing its edge store tail")
            entries.append((
                int(shard["edge_rank"]), int(store_node_id),
                shard.get("source_ack_node_id")))

    def _arm_pending_store_tails(self, session_id, name_prefix,
                                 restore_shards) -> None:
        """B4(主方案 §3.3):回迁发射统一入口的 store→restore 补边。

        逐出支链悬空后,"同会话逐出池写先于其下一轮池读"的链序传递性
        失效——回迁链发射前按登记表补显式依赖(懒处理:store 早已物理
        完成时依赖边即刻满足,无额外时延):
          - 同缘:arm_dependency(store 的边缘 mem_store id)——同 rank
            显式 data_dep 合法(跨批次经持久 (rank,id) 解析),由回迁链
            在该边缘 rank 上的首个节点(mem_load 或其前 1B request)
            消费,经串行链传递到 mem_load;
          - 跨缘:复用 _emit_transfer_trigger 的 1B p2p 中继模式——store
            边缘 arm 后发 1B(其 data_dep 挂 store 的 mem_store 同 rank
            节点),restore 边缘收 1B,mem_load 经串行链排其后(桥拒绝
            跨 rank 直边,1B p2p 是协议内唯一合法时序载体);同
            (store 边缘 → restore 边缘) 多笔 store 共用一条中继(send
            节点一次消费全部 armed 依赖)。
        Session-level Tiered-LRU 下逐出恒为单笔整体 store,每会话至多
        一条登记——本方法把该会话登记整体消费后清除。"""
        tails = self.pending_store_tails.pop(session_id, None)
        if not tails:
            return
        restore_edges = []
        for shard in restore_shards:
            edge_rank = int(shard.edge_rank)
            if edge_rank not in restore_edges:
                restore_edges.append(edge_rank)
        same_edge_tails = {}
        relay_tails = {}
        for edge_rank, store_node_id, _source_ack in tails:
            if edge_rank in restore_edges:
                same_edge_tails.setdefault(edge_rank, []).append(
                    store_node_id)
            else:
                relay_tails.setdefault(edge_rank, []).append(store_node_id)
        for edge_rank, store_node_ids in same_edge_tails.items():
            for store_node_id in store_node_ids:
                self.builders[edge_rank].arm_dependency(store_node_id)
        for store_edge, store_node_ids in relay_tails.items():
            for restore_edge in restore_edges:
                relay_tag = self._tag_allocator.take()
                for store_node_id in store_node_ids:
                    self.builders[store_edge].arm_dependency(store_node_id)
                self.builders[store_edge].comm_send(
                    "{}_store_relay_edge{}_to_edge{}".format(
                        name_prefix, store_edge, restore_edge),
                    src=store_edge, dst=restore_edge, comm_size=1,
                    comm_tag=relay_tag)
                self.builders[restore_edge].comm_recv(
                    "{}_store_relay_arrival_edge{}_to_edge{}".format(
                        name_prefix, store_edge, restore_edge),
                    src=store_edge, dst=restore_edge, comm_size=1,
                    comm_tag=relay_tag)

    def _emit_admission_actions(self, request_plan: dict) -> None:
        """发射在线请求的准入动作:到达/interval gates、history 逐出
        (remote_store 链)、history 迁移/恢复(NOC_MIGRATE 1000 类 /
        remote_load 全量恢复链)、prefill 增长逐出与 readiness 屏障
        (拼 batch 改造,2026-08-22;B3 2026-09-06 KV 物理链:当前段 chunk
        主体移入实例迭代列车,见 emit_iteration_train;session 级
        Tiered-LRU 2026-09-25:partial 恢复流水整段删除,恢复恒单笔
        全量 remote_load)。

        [根因 #5 裁决,strategy 死锁修复统一(2026-08-19)]:strategy 不做
        任何块末恢复/段内清链——per-rank previous_id 无条件接续当前
        frontier,per-rank 发行序 = 全局发射序,跨实例 P2P 与 collective
        参与序不可能反转成环;同 session 串行化由 interval gate
        (after_node_id 显式编码)保留。"""
        builders = self.builders
        config = self.config
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
            pending_gate = PendingHistoryGate(
                source_instance_index=request_plan["prefill_instance_index"],
                timer_gates=tuple(
                    timers[rank] for rank in prefill_group.ranks),
                location="new_session",
            )
            control_group = prefill_group
            control_timers = dict(timers)
            # turn-0 request 在本段发射后成为本 session 的 pending request
            # (remote_store 折算目标)。
            self.pending_request_by_session[request_plan["session_id"]] = (
                request_plan["request_id"])
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
            # B3:pending location 链(completion -> 后续逐出 -> 本准入),
            # face 经 deferred_session_locations 承载(sh 在完成时建门、
            # 此处 pop;face 的门在准入时现场构建)。
            pending_location = self.deferred_session_locations.pop(
                request_plan["session_id"], None)
            if pending_location is None:
                pending_location = request_plan.get("kv_location_after_completion")
            # Session-level Tiered-LRU:pending location 值域收敛为两态。
            if pending_location not in ("local_hbm", "remote_memory"):
                raise RuntimeError(
                    "later request has no valid pending history location: "
                    "{!r}".format(pending_location))
            pending_gate = PendingHistoryGate(
                source_instance_index=previous_index,
                timer_gates=tuple(
                    timers.get(rank) for rank in previous_group.ranks),
                location=pending_location,
            )
            control_group = previous_group
            control_timers = dict(timers)
            self.pending_request_by_session.pop(
                request_plan["session_id"], None)

        history_before = history_snapshot_from_log(request_plan)
        if history_before is not None and pending_gate.location not in (
                history_before.location, "new_session"):
            # pending location(经 remote_store 折算的 completion 链)必须
            # 与计划一致(sh :1074-1082 同款校验,收敛为统一 mismatch
            # raise——session 级 Tiered-LRU 下无 partial 折算特例)。
            raise RuntimeError(
                "history gate location {!r} does not match planned location "
                "{!r}".format(
                    pending_gate.location, history_before.location))

        # ---- history 逐出(触发门 = 到达/interval gate;sh :1084-1092) ----
        history_eviction_trigger = TransferTriggerGate(
            control_instance_index=pending_gate.source_instance_index,
            node_gates=pending_gate.timer_gates,
        )
        # B4(2026-09-13,逐出支链化):history 逐出链 fork 到旁路分支——
        # 分支首节点依赖 = fork 节点 + 到达/间隔触发门(门语义不变,只对齐
        # 逐出的开始时刻,去掉的是"完成阻塞");主链从同一 fork 点继续。
        # turn-0 准入同样可能携带 history 逐出(为给新请求腾容量逐出其他
        # 会话),fork 点的到达门 pending 属于主链——统一 helper 暂存归还。
        history_evictions = tuple(request_plan.get("history_evictions") or ())
        if history_evictions:

            def _emit_history_evictions():
                for transfer in history_evictions:
                    self._register_store_tails(self._emit_plan_transfer(
                        request_plan, kv_transfer_from_log(transfer),
                        "history_evictions",
                        trigger_gate=history_eviction_trigger))

            self._emit_side_branch(_emit_history_evictions)

        # turn-0 的旧 plan 形状(无 history_action 键)按 NO_HISTORY 归一
        # (生产路径调度器恒显式携带 action 值)。
        history_action = request_plan.get("history_action") or NO_HISTORY
        history_transfers = tuple(
            kv_transfer_from_log(transfer)
            for transfer in request_plan.get("history_transfers") or ()
        )
        history_transfer = history_transfers[0] if history_transfers else None

        if history_action in (NO_HISTORY, LOCAL_HIT, NOC_MIGRATE):
            # face 既有路径原样保留(策略 §1.4:LOCAL 跨实例整份 NoC 迁移 =
            # face 1000 类路径;时序依赖经 control trigger 跨实例 1B 承载)。
            if history_action == NO_HISTORY:
                if request_plan["turn_index"] != 0:
                    raise RuntimeError(
                        "later request is missing its history transfer action")
                if control_group.ranks != prefill_group.ranks:
                    raise RuntimeError(
                        "first-request arrival gate is not on its Prefill ranks")
            elif history_action == NOC_MIGRATE:
                source_index = request_plan["history_source_instance_index"]
                if source_index is None:
                    raise RuntimeError("NoC history action has no source")
                source_group = self.group_by_index[source_index]
                migration_timers = control_timers
                if source_group.name != control_group.name:
                    _emit_control_trigger(
                        builders=builders,
                        queue_index=request_plan["queue_index"],
                        name="{}_history_source_control".format(prefix),
                        source_group=control_group,
                        target_group=source_group,
                        timer_gates=control_timers,
                    )
                    migration_timers = {
                        rank: None for rank in source_group.ranks}
                _paired_transfer(
                    config=config, builders=builders,
                    queue_index=request_plan["queue_index"], category=1000,
                    name="{}_history_kv".format(prefix),
                    source_group=source_group, target_group=prefill_group,
                    total_bytes=request_plan["history_transfer_bytes"],
                    timer_gates=migration_timers,
                )
            else:
                # LOCAL_HIT:时序依赖经 control trigger(同组 arm / 跨组 1B)。
                _emit_control_trigger(
                    builders=builders,
                    queue_index=request_plan["queue_index"],
                    name="{}_interval_control".format(prefix),
                    source_group=control_group,
                    target_group=prefill_group,
                    timer_gates=control_timers,
                )
        else:
            # 恢复路径(remote_load 族):session 级 Tiered-LRU 下唯一恢复
            # 动作 = REMOTE_RESTORE 单笔全量回迁(partial 后缀恢复/两段链
            # 已随 PARTIAL 态删除)。位置一致性由上方统一 mismatch 校验
            # 承载。
            if history_transfer is None or history_before is None:
                raise RuntimeError(
                    "restore history action is missing its transfer or "
                    "location snapshot")

        # B4(2026-09-13):prefill 增长逐出同款旁路分支(本发射点无触发
        # 门)。到达/间隔门的 pending 依赖属于主链(后续恢复链/readiness
        # 屏障的消费源,turn-0 NO_HISTORY 与 turn>0 LOCAL_HIT 同实例路径
        # 此时尚未消费)——统一 helper 暂存归还;分支经 fork 节点直接
        # 起链(无门口径)。
        prefill_evictions = tuple(request_plan.get("prefill_evictions") or ())
        if prefill_evictions:

            def _emit_prefill_evictions():
                for transfer in prefill_evictions:
                    self._register_store_tails(self._emit_plan_transfer(
                        request_plan, kv_transfer_from_log(transfer),
                        "prefill_evictions"))

            self._emit_side_branch(_emit_prefill_evictions)

        # ---- 全量恢复链 + readiness barrier ----
        if history_transfer is not None and history_action not in (
                NO_HISTORY, LOCAL_HIT, NOC_MIGRATE):
            # 全量远端恢复(REMOTE_RESTORE):remote_load 链挂 pending
            # gate(interval/到达 gates 为控制源),随后共享 readiness
            # 屏障(sh :1133-1135)。
            # B4(store→restore 前递依赖):回迁发射前查表补边——同会话
            # 在飞 store 支链的边缘 mem_store 完成门挂到本回迁链(同缘
            # 直挂 / 跨缘 1B 中继),防池读先于池写。
            self._arm_pending_store_tails(
                request_plan["session_id"], prefix,
                history_transfer.shards)
            self._emit_plan_transfer(
                request_plan, history_transfer, "history_transfer",
                gate=pending_gate)
        for rank in prefill_group.ranks:
            builders[rank].all_reduce(
                "{}_history_tp_ready_barrier".format(prefix), 1,
                prefill_group.pg_name)
        # prefill 主体(当前段 chunk spans)自拼 batch 改造(2026-08-22)
        # 起移入 emit_iteration_train 的折叠体与 drain 标记;
        # 此处止于准入动作(到达 gates/历史逐出/迁移/恢复/屏障)。

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
