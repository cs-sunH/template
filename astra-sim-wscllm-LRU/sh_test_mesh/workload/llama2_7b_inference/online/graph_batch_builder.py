#!/usr/bin/env python3
"""graph_batch_builder.py -- 在线 GraphBatch 构图器(方案 §4 步骤 1-8 操作 4)。

阶段 1 最关键的对齐点:复用共享发射原语的节点结构。per-request 发射
(generate_wsc_llm_trace.py 模块级函数)由
助手函数组成(_paired_transfer / _emit_prefill_stage /
transformer_pass_aggregated)——本模块直接 import 它们,用 OnlineTraceBuilder
(与 TraceBuilder 同构的在线侧 builder)驱动,保证:

  - 节点属性、插入顺序、rank ownership 和跨 request 链结构稳定(节点级审计口径);
  - per-rank 节点 id 跨批次全局递增;
  - interval gate 的 after_node_id 指向上一 request 的完成 barrier 节点 id
    (completion_gates 账本,跨批次解析)。

在线语义差异(刻意,注释标注):
  - timer gate 始终发射节点(结构保留)但 runtime_ns=0:到达时间由 C++ 的
    arrival alarm(future_alarms)替代,gate 不再等待——保持时长会双重等待;
  - 每 request 拆两次发射:ARRIVAL 边界发射 prefill 整段(含 gates/history/
    recompute/barrier),PREFILL_DRAIN 边界发射 decode 整段(含 transfer 3000/
    decode/end barrier);
  - watch 锚点遵循共享 metrics 口径:PREFILL_DRAIN = prefill_bounds 每 rank
    末节点(排除 end barrier),DECODE_COMPLETION = end barrier 前每 rank 的
    decode 末节点(离线 doc sec.6.2/6.3/6.4 口径)。

拼 batch 改造(2026-08-22,设计文档《层次 B Continuous Batching 改造》
§3.2/§3.6 wscllm PD 分离豁免):D 侧 decode 发射从"每请求整段"重构为
"实例迭代列车"(emit_iteration_train)——列车只含 decode 成员 span,一个
迭代同时算 B 个成员各 1 token,权重每迭代只读一次(weight_passes=迭代数);
列车内绝不混入 prefill chunk span。P 侧保持 emit_prefill_batch 现有整段
骨架(逐 chunk 不拼:聚合调用 weight_passes 缺省 = len(spans) = chunk 数,
每 chunk 恰读一遍权重,口径不变)。列车发射结构(每 decode 实例 rank):
[joiner transfer 3000] → [join 标记] → 17 类聚合体节点(weight_passes=
迭代数) → [exit 标记] → 共享 end barrier;DECODE_COMPLETION watch 成员 =
exit 标记节点(列车体后、end barrier 前的真实节点);completion_gates
账本 = 本列车 post-barrier 节点(下一 turn interval gate 的 after_node_id
来源,与旧整段发射的 end-barrier 口径一致)。
"""

import os
import sys
from typing import Optional

# --------------------------------------------------------------------------
# import 路径:本文件位于 workload/llama2_7b_inference/online/,共享配置与
# 发射模块在上一级。路径只做 import 用途(红线:generate_wsc_llm_trace.py / wsc_llm_
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
from generate_wsc_llm_trace import (  # noqa: E402
    PendingHistoryGate,
    TransferTagAllocator,
    TransferTriggerGate,
    _emit_kv_transfer,
    _emit_prefill_stage,
    _paired_transfer,
    kv_cache_bytes_for_tokens,
    sanitize_node_prefix,
)
from session_kv_manager import KVTransfer  # noqa: E402


def first_token_split_enabled() -> bool:
    """WP9 首步批拆分总开关(SH_FIRST_TOKEN_SPLIT,B4 起缺省 "0" 关)。

    缺省翻转(2026-08-26,主规格 §1.6 A 类处置):B3_W 60s 决策等价
    门-2 失败——拆分的物理扰动(首列拆批 → tick 漂移,row 3 起
    +11.9us)在闭环逐轮放大,t=2.736s 处同 tick 处理顺序翻转
    (OFF completion(session_6_request_1) vs ON prefill(session_7_
    request_0))并级联(1139/1454 请求实例选择翻转、train_id 多重集
    不等:OFF 22945 vs ON 24275 行),ON/OFF 字节等价在 60s 窗不可达
    (2s 窗保持通过)。调度器 tie-break 属禁改区,无补救路径。证据:
    /tmp/slo_wps/gates/B3_W.DONE(gate2 FAIL)与 /tmp/slo_wps/b3/W/
    (t3_full_off_split vs t3_full_split_on 对拍产物)。默认口径退回
    proxy(first_token_source=train_interpolated,见 metrics_postprocess),
    本开关显式置 "1" 仍可启用拆分取 exact 首 token(研究/对拍用)。

    "1" = 拆分开启:debut 列车两段式发射 + first_token 标记;
    其他值(含缺省) = 拆分完全关闭:调度器不拆列车、不发射 first_token
    标记,构图/决策/账本产物与拆分上线前逐字节一致(A/B 对拍的 OFF
    侧)。每次 _emit_train 现场读取(而非构造期缓存),测试与复跑可在
    进程内切换。
    """
    return os.environ.get("SH_FIRST_TOKEN_SPLIT", "0") == "1"


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
        # M1 收集即释放(2026-08-29):这两个 list 只保存自上次 _collect
        # 以来尚未交付的记录；交付后立即 clear，跨批状态只由 next_id、
        # previous_id 与账本保存。禁止按 list 位置回读节点(id 来自
        # next_id 计数器,与位置无关)。
        self.nodes = []   # 本 rank 尚未交付的节点 dict(发射序)
        self.edges = []   # 本 rank 尚未交付的 parent edges
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
        self.node_count += 1
        return node

    @staticmethod
    def _uint64(value: int) -> int:
        return max(1, int(value))

    def arm_timer_gate(self, timer_node_id) -> None:
        if timer_node_id is not None:
            self.pending_extra_dependencies.append(int(timer_node_id))

    def arm_dependency(self, node_id) -> None:
        """B3(2026-09-06)链操作三件套之一(照抄 sh_2.0 builder :228-233):
        挂一条同 rank 的持久依赖边(旁路支链触发门/store→restore 前递边)。"""
        if node_id is not None:
            self.pending_extra_dependencies.append(int(node_id))

    def chain_checkpoint(self):
        """链检查点:(previous_id, pending_extra_dependencies) 快照——
        逐出旁路支链 fork 自检查点前的主链 frontier,发射后 restore 回
        检查点(分支不 join)。"""
        return self.previous_id, tuple(self.pending_extra_dependencies)

    def restore_chain(self, checkpoint) -> None:
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
        self.node_count += 1
        return node["id"]

    def mem_store(self, name: str, tensor_size: int,
                  hbm_access_mode: int = 0) -> None:
        """B3(2026-09-06)池写原语(照抄 sh_2.0 builder :283-290;在线
        snake 键驻 compute 段)。``hbm_access_mode=1`` = 池端点直连路径的
        POOL_READ 唯一 HBM 计费 + 池端口 FIFO 双异步 join;缺省 0 = 纯
        池写(本地零计费,仅池 FIFO)。"""
        node = self._new_node(name, MEM_STORE_NODE)
        node["compute"]["tensor_size"] = self._uint64(tensor_size)
        if hbm_access_mode:
            node["compute"]["hbm_access_mode"] = int(hbm_access_mode)

    def mem_load(self, name: str, tensor_size: int,
                 hbm_access_mode: int = 0) -> None:
        """B3 池读原语(照抄 sh_2.0 builder :292-296)。"""
        node = self._new_node(name, MEM_LOAD_NODE)
        node["compute"]["tensor_size"] = self._uint64(tensor_size)
        if hbm_access_mode:
            node["compute"]["hbm_access_mode"] = int(hbm_access_mode)

    def local_hbm_kv_restore(self, name: str, tensor_size: int) -> None:
        """B3 目标 HBM DMA 写(RESTORE 唯一数据计费;hbm_dma 单槽,可与
        推理计算重叠)——节点级 ``is_local_hbm_kv_restore`` 路由位(照抄
        sh_2.0 builder :298-302;absent == false)。"""
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
            # B3(2026-09-06):N-way HBM contention 直通流量不建本端 HBM
            # 作业(默认 true = 发送端 HBM 读;absent = charged)。计费
            # 三险之一:漏标即双计(照抄 sh_2.0 builder :320-332)。
            node["comm"]["hbm_charge"] = False

    def comm_recv(self, name: str, *, src: int, dst: int, comm_size: int,
                  comm_tag: int, hbm_charge: bool = True) -> None:
        node = self._new_node(name, COMM_RECV_NODE)
        node["comm"]["src"] = int(src)
        node["comm"]["dst"] = int(dst)
        node["comm"]["bytes"] = self._uint64(comm_size)
        node["comm"]["tag"] = int(comm_tag)
        if not hbm_charge:
            # B3:同上(过路/目标写由 restore 承担的接收端零计费位)。
            node["comm"]["hbm_charge"] = False


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
        # completion_gates 只保存下一同 session turn 尚未消费的 interval
        # gate 来源；turn>0 admission 或 terminal completion 后即释放。
        self.completion_gates = {}
        # ---- B3(2026-09-06)跨请求 history/流水账本(照抄 sh_2.0 builder
        # :439-460,适配本仓 P/D 分离发射骨架)----
        # request_id -> PendingHistoryGate(REQUEST_COMPLETE 登记、下一
        # turn prefill 发射消费;location 由 sync_pending_history_after_
        # evictions 镜像逐出,timer_gates 在消费时由 interval gate 填充)。
        self.pending_history = {}
        self.pending_request_by_session = {}
        # turn-0 deferred 通道(历史事故修复版,sh_2.0 :1276-1291):无门
        # 可更新的会话位置延迟到下一 gate 登记点消费。
        self.deferred_session_locations = {}
        # request_id -> {rank: prefill 段 end barrier 节点 id}(post-
        # barrier 口径;joiner decode_evictions 的触发门来源)。
        self._prefill_segment_ends = {}
        # ---- 逐出旁路支链(2026-09-13)store→restore 前递登记表 ----
        # session_id -> [(edge_rank, mem_store_node_id)]——该会话在飞逐出
        # store 支链的池写尾部(整体逐出每会话恰一条);回迁(remote_load)
        # 发射前查表补 store→restore 依赖边(主方案 §3.3);retire_
        # terminal_session 边界(retire_completion_gate)清除。懒处理:
        # store 早已物理完成时补的边即刻满足,无额外时延。
        self.pending_store_tails = {}
        self._tag_allocator = TransferTagAllocator()
        # 每 request 的 action_sequence 账本(action 名含 _action{seq:03d}_,
        # 跨该 request 的全部 transfer 递增;请求完成核销时弹出)。
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

    # ------------------------------------------------------------- 发射 --

    def emit_prefill_batch(self, request_plan: dict) -> dict:
        """发射 request 的 prefill 整段(ARRIVAL 决策的图)。

        strategy 恒走 roofline 物理时钟。

        返回 PREFILL_DRAIN watch 成员:{rank: 末个真实 prefill 节点 id}
        (遵循共享 EVENT_PREFILL_END 锚点口径，排除 end barrier)。
        """
        if self.config.trace_granularity != "request_aggregated":
            raise RuntimeError(
                "online emission supports request_aggregated granularity only "
                f"(got {self.config.trace_granularity!r})")
        self._set_context(request_plan, "prefill", 0)
        # strategy 模式保持物理跨 request 链(根因 #5 裁决):prefill 组各
        # rank 的 previous_id 不清空,本 request 的 prefill 链首可链上一
        # request 在本 rank 的链尾(共享 rank 上的链串行化)。
        # 保留:within-request 串行化;decode 段 frontier 接续(2026-08-19
        # 统一,见列车发射的 transfer-3000 触发链);同 session interval gate。
        marker = self._mark()
        members = self._emit_prelim(request_plan)
        self._collect(marker)
        return members

    def _set_context(self, request_plan: dict, stage: str,
                     generation: int) -> None:
        request_id = request_plan["request_id"]
        for builder in self.builders.values():
            builder.set_context(request_id, stage, generation)

    def retire_completion_gate(self, session_id: str) -> None:
        """释放 terminal session 不会再被下一 turn 消费的完成门。

        B3(2026-09-06):terminal REQUEST_COMPLETE 同时清理 pending/
        deferred 位置账本(会话 KV 已随 retire_terminal_session 静默
        核销,残留位置记录再无消费者)。
        逐出旁路支链(2026-09-13):terminal 会话不会再有下一轮回迁,
        其在飞 store 支链尾部登记一并清除(调度器在同一边界紧随其后
        调用 kv_manager.retire_terminal_session)。"""
        self.completion_gates.pop(session_id, None)
        self.pending_request_by_session.pop(session_id, None)
        self.deferred_session_locations.pop(session_id, None)
        self.pending_store_tails.pop(session_id, None)

    def register_pending_history(self, *, request_id: str, session_id: str,
                                 source_instance_index: int,
                                 location: Optional[str]) -> None:
        """B3:REQUEST_COMPLETE 边界为下一同 session turn 登记 pending
        history 门(sh 在完成段发射时建门;本仓无完成段,改由调度器在
        REQUEST_COMPLETE 边界调用,timer_gates 留待下一 turn prefill
        发射时由 interval gate 节点 id 填充)。"""
        if location not in ("local_hbm", "remote_memory"):
            raise RuntimeError(
                f"completed session {session_id!r} has no valid KV location "
                f"for its pending turn (got {location!r}; the legacy "
                f"partial_hbm_remote runtime location is unreachable)")
        self.pending_history[request_id] = PendingHistoryGate(
            source_instance_index=source_instance_index,
            timer_gates=(),
            location=location,
        )
        self.pending_request_by_session[session_id] = request_id

    def pop_deferred_session_location(self, session_id: str):
        """B3:取出 turn-0 deferred 通道为该会话暂存的位置(无则 None)。"""
        return self.deferred_session_locations.pop(session_id, None)

    def release_request_state(self, request_id: str) -> None:
        """B3:REQUEST_COMPLETE 边界释放该 request 的发射侧账本(action
        序号计数;prefill 段块末通常已在首列车 join 时消费,此处兜底)。
        sh 在完成段发射处同款核销(sh builder :984-986)。"""
        self._action_sequence.pop(request_id, None)
        self._prefill_segment_ends.pop(request_id, None)

    def sync_pending_history_after_evictions(self, transfers) -> None:
        """B3 决策时点补偿(照抄 sh_2.0 builder :1253-1260,seq4689 修订
        版):KV 变更点返回的逐出转移立即镜像到 pending 门——决策时点
        同步是唯一标记路径,发射侧不重复标记(多级逐出乱序时迟到的旧
        转移标记会把门回退到过期位置)。remote_memory 的全量语义由
        _mark_pending_history_store 自身推导。"""
        for transfer in transfers or ():
            if transfer is not None and transfer.kind == "remote_store":
                self._mark_pending_history_store(transfer)

    def _mark_pending_history_store(self, transfer: KVTransfer) -> None:
        if transfer.resident_prefix_layers_after != 0:
            # 整会话逐出必须把本地驻留清零;半层/部分逐出已删除,
            # 任何非零残留即 fail-closed。
            raise RuntimeError("remote store did not reduce resident KV layers")
        location = "remote_memory"
        session_id = transfer.session_id
        pending_request_id = self.pending_request_by_session.get(session_id)
        if pending_request_id is None:
            self.deferred_session_locations[session_id] = location
            return
        gate = self.pending_history.get(pending_request_id)
        if gate is None:
            # turn-0 deferred 通道(sh_2.0 :1276-1291 历史事故修复):在途
            # turn-0 request 在 prefill 发射时登记 session 键但无 pending_
            # history 门(turn-0 的门内联为 new_session 不落账本);此时他
            # 人逐出(remote_store)命中该会话,旧直接下标会 KeyError。镜像
            # turn>0 无门语义:延迟到会话下一 gate 登记点消费。
            self.deferred_session_locations[session_id] = location
            return
        gate.location = location

    def _next_action_sequence(self, request_plan: dict) -> int:
        key = request_plan["request_id"]
        value = self._action_sequence.get(key, 0)
        self._action_sequence[key] = value + 1
        return value

    def _emit_plan_transfer(self, request_plan: dict, transfer: KVTransfer,
                            stage: str, *, gate=None,
                            trigger_gate=None) -> dict:
        """B3:发射一个 KV 转移动作(prefill 段 history/prefill 逐出与
        history 恢复、列车头 decode 逐出共用;action 序号跨该 request
        的全部段共享,保证 canonical 命名稳定——照抄 sh_2.0 :990-1017)。

        节点命名随 _emit_kv_transfer 拷自 sh(契约 §7);新节点名不含
        first_token/batch_train_ 子串(C++ 名字锚点)。"""
        prefix = _prefix_of(request_plan)
        action_name = (
            "{}_{}_action{:03d}_{}_{}".format(
                prefix, stage,
                self._next_action_sequence(request_plan),
                sanitize_node_prefix(transfer.session_id),
                transfer.kind,
            )
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

    # --------------------------------------- 逐出旁路支链(2026-09-13) --

    def _emit_side_branch(self, emit_fn) -> None:
        """把一段逐出发射包成旁路分支（主方案 §3.2，2026-09-13 统一契约）：
        全 rank 暂存并清空既有 pending 依赖 → fork 快照 → 发射（分支内自行
        接续成链；分支触发门的 arming 必须在 emit_fn 内部完成）→ 恢复主链
        → 归还暂存依赖。分支不 join——被包裹的物理逐出链不再阻塞主链任何
        节点，HBM 争用由 C++ LocalHbmBandwidthModel 的 N-way 均分模型在线
        裁决。

        fork 点可能合法携带属于主链的 armed 依赖（turn-0 准入形态：到达门
        armed 后尚未被首个主链节点消费，而 turn-0 也会为腾容量逐出其他会
        话）——暂存清空使其不进分支，主链恢复后照常由其本来的消费者消费，
        图依赖与逐出链在主链上时完全一致。分支内部 arming 而未被任何节点
        消费（触发门泄漏）仍 fail-closed。"""
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

    def _register_store_tails(self, records) -> None:
        """逐出发射后登记该会话的在飞 store 支链尾部(§3.3):每条
        remote_store shard 记 (edge_rank, 边缘 mem_store 节点)。整体
        逐出每会话恰一条;restore 须等齐(restore 读全层区间)。"""
        for record in records:
            if record.get("kind") != "remote_store":
                continue
            tails = self.pending_store_tails.setdefault(
                record["session_id"], [])
            for shard in record.get("shards", ()):
                if "edge_store_node_id" not in shard:
                    raise RuntimeError(
                        "remote store shard is missing its edge store tail")
                tails.append((
                    shard["edge_rank"],
                    shard["edge_store_node_id"],
                ))

    def _arm_store_tail_dependencies(self, transfer: KVTransfer,
                                     name_prefix: str) -> None:
        """回迁发射统一入口的 store→restore 补边(§3.3,唯一必须新增的
        正确性边):逐出支链悬空后,"同会话逐出池写先于其下一轮池读"的
        传递性保障失效,回迁(remote_load)发射前查 pending_store_tails,
        把 store 的边缘 mem_store 完成挂到回迁池读之前:

          - 同边缘 rank:对该 rank arm 依赖(store mem_store 节点),由
            回迁链在该 rank 的首个节点(1B request recv 或 mem_load)
            消费——传递性保证 mem_load 晚于 mem_store;
          - 跨边缘 rank:1B p2p 中继(_emit_transfer_trigger 同款模式):
            store 边缘在 mem_store 完成后发 1B,回迁边缘收 1B、其后的
            池读链在其后(桥拒绝跨 rank 直边,1B p2p 是协议内唯一载体)。

        粒度 = store 的边缘 mem_store 完成(池写落盘),不必等源端 ack
        全程;懒处理——store 早已完成时补边即刻满足,无需判在飞。"""
        if transfer.kind != "remote_load":
            return
        tails = self.pending_store_tails.get(transfer.session_id)
        if not tails:
            return
        shard_edges = {
            shard.edge_rank for shard in transfer.shards
            if shard.edge_rank is not None}
        # 两遍式:先发跨缘 1B 中继(其 recv 节点在回缘 rank 上链于
        # 回迁链之前),后 arm 同缘依赖——保证同缘 arm 恰好落在回迁链
        # 首节点上,不被中继 recv 抢先消费。
        direct_tails = []
        for edge_rank, store_node_id in tails:
            if edge_rank in shard_edges:
                direct_tails.append((edge_rank, store_node_id))
                continue
            for target_edge in sorted(shard_edges):
                if target_edge == edge_rank:
                    continue
                relay_tag = self._tag_allocator.take()
                self.builders[edge_rank].arm_dependency(store_node_id)
                self.builders[edge_rank].comm_send(
                    f"{name_prefix}_store_tail_relay_e{edge_rank}"
                    f"_to_e{target_edge}",
                    src=edge_rank,
                    dst=target_edge,
                    comm_size=1,
                    comm_tag=relay_tag,
                )
                self.builders[target_edge].comm_recv(
                    f"{name_prefix}_store_tail_relay_from_e{edge_rank}",
                    src=edge_rank,
                    dst=target_edge,
                    comm_size=1,
                    comm_tag=relay_tag,
                )
        for edge_rank, store_node_id in direct_tails:
            self.builders[edge_rank].arm_dependency(store_node_id)

    # ------------------------------------------------- per-request 发射主体 --

    def _emit_prelim(self, request_plan: dict) -> dict:
        """发射动态 GraphBatch 的 turn-gates/history/barrier/current-prefill 块。

        B3(2026-09-06)发射编排(照抄 sh_2.0 _emit_admission :1019-1251
        的动作编排,适配本仓 P/D 分离骨架:prefill 主体仍在 P 实例整段
        发射,不经实例迭代列车)。逐出旁路支链(2026-09-13):下述 2/4
        两段的逐出链 fork 到旁路分支(不 join),与主链计算并行。

          1. 到达/interval gates(turn-0 内联 new_session 门并登记
             pending_request_by_session;turn>0 消费 REQUEST_COMPLETE 登记
             的 pending history 门,timer_gates 由 interval gate 填充)+
             位置对账(gate.location 必须等于准入时点快照——漏接 sync
             即 fail-closed);
          2. history 逐出(prepare_history 的 fit 逐出,触发门 = pending
             门;control rank 与源 rank 不同实例时 1B p2p 触发)——
             逐出旁路支链(2026-09-13):fork 自各 rank frontier、不 join,
             逐出物理传输与主链计算并行;
          3. history 恢复/迁移分流:NO_HISTORY/LOCAL_HIT 仅 arm 门;
             NOC_MIGRATE/REMOTE_RESTORE(全量恢复,唯一远端路径)经
             _emit_kv_transfer(remote_load 发射前查 pending_store_tails
             补 store→restore 前递边,§3.3);
          4. prefill 增长逐出(grow_prefill 的 fit 逐出,逐出旁路支链,
             支链根 = rank frontier;fork 点的主链 armed 依赖由
             _emit_side_branch 暂存清空、恢复后归还原主链消费者);
          5. 就绪屏障(history_tp_ready_barrier:恢复完成前不能消费
             历史 KV——单一全层就绪屏障);
          6. prefill 主体(整段聚合发射,无层段拆分)。

        返回 PREFILL_DRAIN watch 成员:{rank: 末个真实 prefill 节点 id}
        (排除 end barrier);post-barrier 的段末节点记入 _prefill_
        segment_ends(joiner decode 逐出的触发门来源)。"""
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
            pending_gate = PendingHistoryGate(
                source_instance_index=request_plan["prefill_instance_index"],
                timer_gates=tuple(
                    timers[rank] for rank in prefill_group.ranks),
                location="new_session",
            )
            # turn-0 request 在本段发射后成为本 session 的 pending request
            # (remote_store 折算目标;无门的 deferred 通道入口)。
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
            pending_gate = self.pending_history.pop(
                request_plan["request_id"], None)
            if pending_gate is None:
                raise RuntimeError(
                    "request {} has no arrival/history gate".format(
                        request_plan["request_id"]))
            if pending_gate.source_instance_index != previous_index:
                raise RuntimeError(
                    "history gate source instance does not match the "
                    "completion gate instance")
            # duration 传真实 inter_request_interval_ns；0 时不发射 interval
            # gate 节点，直接以 after_node_id 建立依赖。
            timers = {
                rank: builders[rank].timer_gate(
                    "{}_interval_timer_gate".format(prefix),
                    request.inter_request_interval_ns,
                    after_node_id=previous_nodes.get(rank))
                for rank in previous_group.ranks
            }
            pending_gate.timer_gates = tuple(
                timers[rank] for rank in previous_group.ranks)
            self.pending_request_by_session.pop(
                request_plan["session_id"], None)
            # 位置对账(sh reconcile 同款 fail-closed):pending 门位置
            # (完成时点位置 + 期间逐出的 sync 镜像)必须等于准入时点
            # 的会话历史位置快照——任何不一致 = sync 接线缺口。
            planned_location = request_plan.get("history_location_before")
            if (planned_location is not None
                    and pending_gate.location != planned_location):
                raise RuntimeError(
                    "history gate location {!r} does not match planned "
                    "location {!r}; request={}".format(
                        pending_gate.location, planned_location,
                        request_plan["request_id"]))

        # ---- history 逐出(prepare_history fit 逐出;触发门 = pending 门;
        #      逐出旁路支链 2026-09-13:fork 前各 rank 无 pending 依赖
        #      (触发门在 _emit_kv_transfer 内部 arm→发射),分支不 join,
        #      逐出物理链不再阻塞其后的回迁/屏障/prefill 主体)----
        history_eviction_trigger = TransferTriggerGate(
            control_instance_index=pending_gate.source_instance_index,
            node_gates=pending_gate.timer_gates,
        )
        history_evictions = tuple(request_plan.get("history_evictions") or ())
        if history_evictions:
            def _emit_history_eviction_branch():
                self._register_store_tails(
                    self._emit_plan_transfer(
                        request_plan, transfer, "history_evictions",
                        trigger_gate=history_eviction_trigger)
                    for transfer in history_evictions)
            self._emit_side_branch(_emit_history_eviction_branch)

        # ---- history 恢复/迁移分流(REMOTE_RESTORE 全量恢复为唯一远端
        #      路径;恢复完成前不能消费历史 KV,由下方单一全层就绪屏障
        #      保障)----
        history_transfers = tuple(request_plan.get("history_transfers") or ())
        if len(history_transfers) > 1:
            raise RuntimeError(
                "history must carry at most one transfer")
        if history_transfers:
            # 回迁发射前补 store→restore 前递边(§3.3;懒处理,无在飞
            # store 时补边即刻满足、零成本)。
            self._arm_store_tail_dependencies(
                history_transfers[0], prefix)
            self._emit_plan_transfer(
                request_plan, history_transfers[0], "history_transfer",
                gate=pending_gate)
        else:
            # NO_HISTORY / LOCAL_HIT(以及合成 plan 的 None):无数据
            # 传输,到达/interval 门直接由门源实例各 rank 的后续链
            # 消费(turn-0 门源 = prefill 实例,与既有行为一致)。
            gate_group = self.group_by_index[
                pending_gate.source_instance_index]
            for relative_index, rank in enumerate(gate_group.ranks):
                builders[rank].arm_timer_gate(
                    pending_gate.timer_gates[relative_index])

        # ---- prefill 增长逐出(grow_prefill fit 逐出;逐出旁路支链,
        #      2026-09-13 统一契约:无显式触发门,支链根 = fork 时各
        #      rank 主链 frontier——fork 点可能携带主链 armed 依赖(上文
        #      的到达/interval 门 arm),由 _emit_side_branch 暂存清空、
        #      不进分支,恢复后归还原主链消费者;发射后登记 store 尾部)----
        prefill_evictions = tuple(request_plan.get("prefill_evictions") or ())
        if prefill_evictions:
            def _emit_prefill_eviction_branch():
                self._register_store_tails(
                    self._emit_plan_transfer(
                        request_plan, transfer, "prefill_evictions")
                    for transfer in prefill_evictions)

            self._emit_side_branch(_emit_prefill_eviction_branch)

        # ---- 恢复完成门:单一全层就绪屏障(history_tp_ready_barrier,
        #      既有命名与位置)——恢复完成前不能消费历史 KV;整体恢复
        #      恒为单段全层传输,无需 prefix/suffix 双栅栏与首 chunk
        #      层段拆分(该 PARTIAL 真流水已随部分恢复路径删除)。----
        for rank in prefill_group.ranks:
            builders[rank].all_reduce(
                "{}_history_tp_ready_barrier".format(prefix), 1,
                prefill_group.pg_name)

        # ---- prefill 主体(整段聚合发射;trace chunking 为合法物理
        #      机制,无层段拆分)----
        prefill_bounds = _emit_prefill_stage(
            config=self.config, builders=builders, group=prefill_group,
            prefix=prefix, stage="current_prefill",
            tokens=request_plan["prefill_length"],
            initial_context_tokens=request_plan["history_tokens_before"],
        )
        # PREFILL_DRAIN watch 成员 = 每 rank 末个真实 prefill 节点
        # (遵循共享 EVENT_PREFILL_END 锚点口径，排除 end barrier)。
        members = {rank: bounds[1] for rank, bounds in prefill_bounds.items()}
        # post-barrier 段末账本(end barrier 节点):joiner decode 逐出的
        # 触发门来源(列车头消费后弹出)。
        self._prefill_segment_ends[request_plan["request_id"]] = {
            rank: builders[rank].previous_id
            for rank in prefill_group.ranks
        }
        return members

    def emit_iteration_train(self, train_plan: dict) -> dict:
        """发射一趟 D 侧 decode 迭代列车(拼 batch 改造核心,2026-08-22;
        设计文档 §3.2"迭代列车聚合发射"+ §3.6 wscllm PD 分离豁免)。

        wscllm 范围 = 仅 decode 互拼:列车只含 decode 成员 span,绝无
        prefill chunk span(§3.6:P 侧逐 chunk 不拼,保持 emit_prefill_
        batch 现有发射骨架——其聚合调用 weight_passes 缺省 = len(spans)
        = chunk 数,与"chunk 之间不拼、每 chunk 读一遍权重"口径一致)。

        train_plan(调度器冻结的成员快照)字段:
          train_id          批命名空间 id("batch_train_i<实例>_<序号>";
                            共享体节点归属,物理完成与逻辑请求完成分离)
          instance_index    列车所在 decode 实例
          joiners           新成员 request_plan 列表(含 prefill_instance_
                            index/prefill_context_tokens)——transfer 3000
                            迁移节点先于列车体(触发链 = per-rank frontier,
                            与旧整段 decode 发射同款;drain 决策事件已在
                            发射前交付,prefill 主体物理已完成)
          pass_spans        成员×迭代展开的 (tokens, kv) 平铺列表
          iterations        迭代数(= weight_passes:权重字节 ×迭代数,
                            与批成员数无关;激活/KV/AR 逐 span 精确)
          exit_members      本列车内退出 decode 的成员 plan 列表
          first_token       WP9 首 token 观测(2026-08-26;缺省 None = 拆分
                            开关关闭,行为与上线前逐字节一致):{"split":
                            False 时仅做不拆车的标记增强——多 token debut
                            成员在列车体后挂 first_token 标记,decode_length
                            =1 的 debut 成员其 exit 标记改名为 _exit_first_
                            token_(C++ 名字子串锚点,first_token==completion
                            不变量由同节点保证);"split": True 时本方法拒绝,
                            经 emit_train_first_step / emit_train_remainder
                            两段式发射}

        每 decode 实例 rank 上的结构(链序):
          [joiner transfer 3000 ...] → [join 标记 ...] → 17 类聚合体节点
          (transformer_pass_aggregated, weight_passes=iterations) →
          [first_token 标记(不拆车增强时)] → [exit 标记 ...] → 共享
          end barrier。

        返回 {"exit_members": {request_id: {rank: 标记节点 id}},
              "block_ends": {rank: end barrier 节点 id}}——exit 标记即
        DECODE_COMPLETION watch 成员(barrier 前末节点口径:标记是列车
        体后、end barrier 前的真实节点);completion_gates 账本(下一
        turn interval gate 的 after_node_id 来源)= 本列车 post-barrier
        节点,与旧整段发射的 end-barrier 口径一致。"""
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
        """WP9 首步批发射(2026-08-26;拆分阶段 1,WP9_CONTRACT §2)。

        首步批 = 所有列车成员的第 1 个 span(decode 成员首迭代;wscllm
        列车无 prefill chunk,§2 wscllm 特例),加上锚定迭代位置在第 1
        迭代内的列车头(joiner transfer 3000 迁移 + join 标记——挂点
        语义与整列发射完全一致)。列车体以 weight_passes=1 折叠(权重恰
        读一次),体后挂各 debut 成员的 first_token 标记节点(1-op COMP,
        名字含 "first_token" 子串——C++ 锚点按名字子串注册 code 8,取
        每 rank min tick;锚定该成员自己链条首步批内末节点=共享折叠体
        后,多 debut 各自挂)。

        本批不含任何请求级 watch/exit/哨兵标记;唯一附加物是尾部的批
        命名空间唤醒标记(每 rank 1 个小节点,request_id = "<train_id>_
        first_step",batch_train_ 前缀,复用 C++ 哨兵 watch 通道):其
        fire 经 PREFILL_DRAIN 通道送回调度器,作为余量批的交付边界——
        没有它,无 watch 的首步批完成后不存在任何决策工作,C++ tick-end
        门不会再交付,运行尾部(全部其余工作已排空)将永久等待(死锁)。
        这是对"首步批无任何 watch 标记"的必要工程化偏移(与 sh_1.0/
        face 同款):唤醒 watch 不挂任何请求、不触发核销/记账/决策
        (调度器按 first_step id 识别后无操作)。

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
        decode_group = self.group_by_index[train_plan["instance_index"]]
        wakeup_id = first_token["wakeup_id"]
        for builder in self.builders.values():
            builder.set_context(wakeup_id, "prefill", 0)
        wakeup_members = {
            rank: self._emit_train_marker(rank, f"{wakeup_id}_wakeup")
            for rank in decode_group.ranks
        }
        self._collect(marker)
        return {
            "first_token_members": first_token_members,
            "wakeup_members": wakeup_members,
        }

    def emit_train_remainder(self, train_plan: dict) -> dict:
        """WP9 余量批发射(2026-08-26;拆分阶段 2,唤醒交付处调用)。

        余量批 = 剩余迭代(成员第 2 个 span 起;weight_passes =
        iterations-1,与首步批的 1 次恰合回整列的迭代数;激活/KV/AR 逐
        span 精确,总量与整列发射一致)。exit 标记/哨兵标记与共享 end
        barrier 全部照常挂本批(挂点语义不变,仍"barrier 前末节点");
        decode_length=1 的 debut 成员 exit 标记改名携带 first_token 子串
        (同节点 code 4/8 双锚点,first_token_ns == completion_ns 不变量
        由同一节点保证)。返回结构与 emit_iteration_train 相同。"""
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
        """列车头:joiner decode 逐出(触发门 = prefill 段 post-barrier 块末)
        + joiner 迁移(transfer 3000)+ join 标记节点。

        B3(2026-09-06):decode 侧逐出(#2 reserve/#4 move/#5 grow_decode
        累积,decode 决策行落盘时核销)随 joiner 首列车发射,触发门 =
        该成员 prefill 段的 end barrier 节点(post-barrier 口径,sh 的
        drain 列车 barrier 同源——物理上逐出发生在本请求 prefill 完成
        之后、P→D 迁移数据到达之前)。拆分时整段归首步批(逐出/迁移/
        标记锚定的迭代位置在第 1 迭代内,语义不变)。"""
        instance_index = train_plan["instance_index"]
        decode_group = self.group_by_index[instance_index]
        train_id = train_plan["train_id"]

        # ---- joiner decode 逐出(触发门 = prefill 段块末;照抄 sh
        #      :710-734 的 decode_evictions 段,块末来源换 P 侧整段) ----
        joiners = list(train_plan.get("joiners", ()))
        for joiner in joiners:
            decode_evictions = tuple(joiner.get("decode_evictions") or ())
            if not decode_evictions:
                self._prefill_segment_ends.pop(joiner["request_id"], None)
                continue
            block_ends = self._prefill_segment_ends.pop(
                joiner["request_id"], None)
            if block_ends is None:
                raise RuntimeError(
                    "joiner {} has decode evictions but no Prefill segment "
                    "block end".format(joiner["request_id"]))
            prefill_group = self.group_by_index[
                joiner["prefill_instance_index"]]
            decode_eviction_trigger = TransferTriggerGate(
                control_instance_index=joiner["prefill_instance_index"],
                node_gates=tuple(
                    block_ends[rank] for rank in prefill_group.ranks),
            )
            self._set_context(joiner, "decode", 1)
            # 逐出旁路支链(2026-09-13):prefill 段末门、_prefill_
            # segment_ends.pop、(decode, 1) 上下文 stamping、触发门构造
            # 全部原样保留,仅逐出链本体 fork 到旁路分支(批首发射,fork
            # 时各 rank 无 pending 依赖;分支不 join)——逐出物理链与本
            # 列车计算并行,HBM 争用由 C++ N-way 均分模型在线裁决。其
            # 后的 joiner 迁移(transfer 3000)与 join 标记保持主链。
            def _emit_decode_eviction_branch():
                self._register_store_tails(
                    self._emit_plan_transfer(
                        joiner, transfer, "decode_evictions",
                        trigger_gate=decode_eviction_trigger)
                    for transfer in decode_evictions)

            self._emit_side_branch(_emit_decode_eviction_branch)

        # ---- joiner 迁移(transfer 3000;上下文 (joiner, decode, 1) =
        #      decode_start 指标锚点之一;触发链 = per-rank frontier) ----
        for joiner in joiners:
            self._set_context(joiner, "decode", 1)
            prefill_group = self.group_by_index[
                joiner["prefill_instance_index"]]
            _paired_transfer(
                config=self.config, builders=self.builders,
                queue_index=joiner["queue_index"], category=3000,
                name="{}_prefill_to_decode_kv".format(_prefix_of(joiner)),
                source_group=prefill_group, target_group=decode_group,
                total_bytes=kv_cache_bytes_for_tokens(
                    self.config.model, joiner["prefill_context_tokens"]),
            )

        # ---- join 标记节点(decode_start 指标锚点;§3.5:请求加入后第一
        #      个迭代所在列车节点。迁移有节点时 min-tick 语义取更早者,
        #      不冲突;与五仓机制同构保留) ----
        for joiner in joiners:
            self._set_context(joiner, "decode", 1)
            for rank in decode_group.ranks:
                self._emit_train_marker(
                    rank, "{}_join_{}".format(
                        train_id,
                        sanitize_node_prefix(joiner["request_id"])))

    def _emit_train_body(self, train_plan: dict, pass_spans,
                         weight_passes: int) -> None:
        """折叠列车体(17 类聚合节点;weight_passes = 权重读取次数)。

        拆分时首步批传首步 span 组 + weight_passes=1,余量批传余量组 +
        iterations-1;两批激活/KV/AR 字节按 span 求和与整列一致。"""
        instance_index = train_plan["instance_index"]
        decode_group = self.group_by_index[instance_index]
        train_id = train_plan["train_id"]
        for builder in self.builders.values():
            builder.set_context(train_id, "decode", 1)
        tensor_parallel = len(decode_group.ranks)
        for relative_rank, rank in enumerate(decode_group.ranks):
            transformer_pass_aggregated(
                self.builders[rank],
                phase=train_id,
                pass_spans=pass_spans,
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
                weight_passes=weight_passes,
            )

    def _emit_first_token_markers(self, train_plan: dict,
                                  debut_members) -> dict:
        """WP9 first_token 标记节点(debut 成员各自挂;每 rank 1 个
        1-op COMP 小节点,名字含 "first_token" 子串,C++ 按 (request_id,
        rank) 取 min tick 收 code 8 事件)。锚定位置 = 该成员链条在所在
        批内列车体后的末节点(共享折叠体 ⇒ 各 debut 标记顺序挂体后)。"""
        train_id = train_plan["train_id"]
        decode_group = self.group_by_index[train_plan["instance_index"]]
        members = {}
        for member in debut_members:
            request_id = member["request_id"]
            self._set_context(member, "decode", 1)
            members[request_id] = {
                rank: self._emit_train_marker(
                    rank, "{}_first_token_{}".format(
                        train_id, sanitize_node_prefix(request_id)))
                for rank in decode_group.ranks
            }
        return members

    def _emit_train_tail_markers(self, train_plan: dict,
                                 first_token) -> dict:
        """列车尾:first_token 标记(不拆车增强时)+ exit 标记 + 哨兵标记
        + 共享 end barrier + completion gate 账本(挂点语义与整列发射
        一致)。"""
        instance_index = train_plan["instance_index"]
        decode_group = self.group_by_index[instance_index]
        train_id = train_plan["train_id"]
        iterations = int(train_plan["iterations"])

        # ---- WP9:不拆车时的 first_token 标记(iterations==1 等无需拆车
        #      的场景;拆车时标记已在首步批挂过,余量批不重挂) ----
        if first_token is not None and not first_token.get("split"):
            self._emit_first_token_markers(
                train_plan, first_token["debut_marker_members"])

        # ---- exit 标记(列车体后、end barrier 前;每成员每 rank 1 个小
        #      节点,承载该请求的 DECODE_COMPLETION watch 与指标 completion
        #      锚点;物理完成时刻 = 标记完成时刻)。WP9:decode_length=1
        #      的 debut 成员 exit 标记名附加 first_token 子串——同节点
        #      双锚点(code 4 watch + code 8 名字),first_token_ns ==
        #      completion_ns 不变量由同一节点保证 ----
        exit_first_token = set(
            (first_token or {}).get("debut_exit_first_token") or ())
        exit_members = {}
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
                for rank in decode_group.ranks
            }

        # ---- 哨兵标记(T_max 截断且无自然 exit 标记的列车):
        #      request_id = train_id(批命名空间),固定 (prefill, 0) 单事件
        #      通道(decode 通道双事件 + 计账下溢);fire 后经 PREFILL_DRAIN
        #      通道送回,调度器按 train_id 核销 ----
        sentinel_members = {}
        if train_plan.get("sentinel"):
            for builder in self.builders.values():
                builder.set_context(train_id, "prefill", 0)
            sentinel_members = {
                rank: self._emit_train_marker(
                    rank, f"{train_id}_sentinel")
                for rank in decode_group.ranks
            }

        # ---- 共享 end barrier(每列车一个,替代每请求一个) ----
        for builder in self.builders.values():
            builder.set_context(train_id, "decode", 1)
        for rank in decode_group.ranks:
            self.builders[rank].all_reduce(
                "{}_end_barrier".format(train_id),
                iterations,
                decode_group.pg_name,
            )
        block_ends = {
            rank: self.builders[rank].previous_id for rank in decode_group.ranks}
        if any(node_id is None for node_id in block_ends.values()):
            raise RuntimeError("train end barrier node IDs were not generated")

        # ---- completion gate 账本(post-barrier 口径,与旧整段发射一致):
        #      每个退出成员的 session 记录本列车 barrier(下一 turn 的
        #      interval gate after_node_id 指向它)。同 session 请求串行
        #      (turn k+1 在 k 完成后才到达),无覆盖竞争。 ----
        for member in train_plan.get("exit_members", ()):
            self.completion_gates[member["session_id"]] = (
                instance_index, dict(block_ends))

        return {
            "exit_members": exit_members,
            "sentinel_members": sentinel_members,
            "block_ends": block_ends,
        }

    def _emit_train_marker(self, rank: int, name: str) -> int:
        """列车标记节点(每 rank 1 个小 COMP 节点;上下文由调用方设置)。"""
        self.builders[rank].comp(name, 1, 1)
        return self.builders[rank].previous_id


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
