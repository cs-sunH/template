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
    prefill remote-read 前缀读流旁挂分支（2026-09-25 规格书§三：与后缀
    池恢复并行、同一准入 frontier 分叉，gate 1B relay 触发 + 逐组 recv
    完成门入 _prefill_remote_read_arms/_prefill_remote_read_layers）→
    prefill_evictions → prefill readiness barrier（结构性主链节点）；prefill 主体不再在此发射；
  - KV 逐出并行化（2026-09-13）：history/prefill/decode 三类逐出发射循环
    经 _emit_side_branch 包成旁路支链（fork 自各 rank 当前 frontier，触发门
    挂分支首节点，不 join 回主链）——逐出物理传输与主链上的推理计算并行，
    HBM 带宽由 C++ LocalHbmBandwidthModel 的 N-way 均分在线裁决；"同会话
    store 池写先于其 restore 池读"的前递保障改由 pending_store_tails 登记
    + 回迁发射前 _arm_pending_store_edges 补边承担（同缘 arm / 跨缘 1B
    p2p 中继）；prefill_decode_transfer 等恢复类迁移保持主链不动；
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
    _emit_transfer_trigger,
    _history_control,
    reconcile_pending_history_location,
    sanitize_node_prefix,
)

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
        # M1 收集即释放（2026-08-29）：这两个 list 只保存自上次 _collect
        # 以来尚未交付的记录；交付后立即 clear，跨批状态只由 next_id、
        # previous_id 与账本保存。禁止按 list 位置回读节点（id 来自
        # next_id 计数器，与位置无关）。
        self.nodes = []   # 本 rank 尚未交付的节点 dict（发射序）
        self.edges = []   # 本 rank 尚未交付的 parent edges
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
        # C15 逐组恢复门（2026-09-22，设计方案 §5.1）：request_id ->
        # ((group_index, layer_start, layer_end, {rank: 该 rank 该组目标
        # HBM 写完成门节点 id}), ...)——后缀逐组恢复支链（rank 内消费顺
        # 序串行链）的就绪门登记；列车体按层段门控消费（首 chunk 的逐层
        # 段各自等其组门，替换"整列车等整段后缀"的列车级保守门——
        # C12-G7/E17 登记的缺口闭合）。旧单笔口径（restore_group 无标记
        # 的 remote_load）仍走 _suffix_body_arms 整段门（回归锚）。
        self._suffix_restore_arms = {}
        # KV 逐出并行化（2026-09-13）：store→restore 前递依赖登记表——
        # session_id → 该会话全部在飞 remote_store 逐出支链的尾部
        # (edge_rank, edge mem_store 节点 id, 源端 ack_recv 节点 id) 列表。
        # 同会话两段式逐出（suffix + full fallback）逐 shard 分别登记
        # （restore 读全区间，须等齐全部在飞 store）；回迁发射时统一补边
        # 消费（_arm_pending_store_edges），会话终结时清除（_emit_completion
        # 的 terminal 分支）。
        self.pending_store_tails = {}
        # remote-read credit（唯一执行口径）：request_id ->
        # {切片块号 b: {rank: 该 rank 上块 b 的 recv 完成门节点 id}}——
        # 旁挂支链发射的尾块（块 2..M）流式到达语义的 body 依赖账本
        # （_suffix_body_arms 同款手法：支链登记、体块消费点 arm）。列车
        # 尾标记发射时清账（每列车恰一次——整列/余量批同路径）。
        self._credit_arms = {}
        # C13 copy 逐 chunk 交接（2026-09-22，设计文档 §2.3 四步协议）：
        # request_id -> {交接块号 c: {rank: 该 rank 上块 c 的 recv 完成
        # 门节点 id}}——准入批发射的尾块（块 1..M-1）旁挂支链，流式到达
        # 语义与 _credit_arms 同款；含首 chunk 的列车体按消费顺序逐块
        # arm（I2 门并集同源），块 0 在准入主链（readiness barrier 只等
        # 它——不设"先整份搬运后计算"串行段）。列车体发射时消费（joiner
        # 恰一次），残留即 fail-closed。
        self._copy_handoff_arms = {}
        # K6（P1-③，2026-09-23 外部审计）：request_id -> {交接块号 c:
        # (layer_start, layer_end)}——尾块层区间账本（首体块层段门控的
        # 段边界来源；生命周期与 _copy_handoff_arms 同步：发射登记 /
        # 体块消费弹出 / 完成残留 fail-closed）。
        self._copy_handoff_layers = {}
        # C13：request_id -> {交接块号 c: {home_rank: 该 rank 上块 c 的
        # ack_recv 节点 id}}——交接完成事件在源端的物理挂点（source
        # release dependency = noc_migration_ack_recv；FS 账本侧的逐块
        # home 释放即以此为到达事实锚）。completion 批弹出（审计面）。
        self._copy_handoff_release_anchors = {}
        # 规格书§三.5（2026-09-25 prefill remote-read 前缀读流）：
        # request_id -> {组号 g: {exec target rank: 该 rank 上组 g 的
        # home→exec 前缀 recv 完成门节点 id}}——前缀读流逐组旁挂支链
        # （home send 链 / exec recv 链顺序连接）的 recv 完成门登记
        # （_copy_handoff_arms 同款账本形）；含首 chunk 的列车体按层段
        # 门控消费（前缀层段各自等其组 recv 门——前缀读流与后缀池恢复
        # 两腿并行，不设"整段就绪才计算"全局 barrier）。列车体发射时
        # 消费（恰一次），残留即 fail-closed。
        self._prefill_remote_read_arms = {}
        # 规格书§三.5：request_id -> {组号 g: (layer_start, layer_end)}
        # ——前缀读流组层区间账本（首体块层段门控的段边界来源，K6
        # _copy_handoff_layers 同款；生命周期与 _prefill_remote_read_
        # arms 同步：准入发射登记 / 列车体消费弹出 / 完成残留
        # fail-closed）。
        self._prefill_remote_read_layers = {}
        self.batch = None  # 当前批次累加器（由 begin_batch 建立）
        # WP9 首步批标记（2026-08-26）：本批是否含首步批发射（digests 行
        # first_step=True 标记的来源；OFF 侧恒 False，不进任何产物）。
        self.batch_first_step = False
        # request_id -> 跨阶段连续的 action 序号（离线 writer 的 per-
        # request action_sequence 闭包在列车发射下的等价物：准入/列车/
        # completion 批共享同一计数器，保证 actionNNN 命名与逐字节稳定
        # ——canonical 命名对照前提）。
        self._action_seq = {}
        # request_id -> 下一 turn 的 request_id / plan dict / None。每个
        # 当前 request 只会在 completion 发射时读取一次，随后立即释放。
        self.next_plan = {}
        self._plan_resolver = lambda request_id: None

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
        self.batch_first_step = False

    def _collect(self, marker: dict) -> None:
        # R6-2（2026-09-14）：逐秩边审计**代码内默认常开**（SH_EDGE_AUDIT=0
        # 显式关闭）。O(E)/批只读校验：每条 parent 边的 from/to 必须是本
        # rank 自身 id 空间内 from ≤ to 的既有节点（悬挂/前向引用立即带
        # 节点名 fail-closed）——把 D3 类静默错连提前到 Python 侧。跨
        # rank 的 id 串用（id 数值碰撞于两个计数器空间）由 R5 的结构性
        # 重建 + source_ranks 断言承担，本审计不重复覆盖。
        audit_enabled = os.environ.get("SH_EDGE_AUDIT", "1") != "0"
        for rank, builder in self.builders.items():
            node_mark, edge_mark = marker[rank]
            edges = builder.edges
            # 正常路径的 marker 为 0，直接 extend 避免临时 slice；非零
            # marker 仅保留本次新增尾部。所有 _mark() 都在同一发射调用内
            # 被单次 _collect() 消费，故旧前缀已在先前批次交付，可立即
            # clear 释放对节点/边 dict 的最后一层 builder 引用。
            if len(nodes := builder.nodes) > node_mark:
                self.batch["_touched_ranks"].add(int(rank))
            self.batch["nodes"].extend(
                nodes if node_mark == 0 else nodes[node_mark:])
            self.batch["parent_edges"].extend(
                edges if edge_mark == 0 else edges[edge_mark:])
            if audit_enabled and edge_mark < len(edges):
                next_id = builder.next_id
                for edge in edges[edge_mark:]:
                    source, target = edge["from"], edge["to"]
                    if not (
                        isinstance(source, int) and 0 <= source < next_id
                        and isinstance(target, int) and 0 <= target < next_id
                        and source <= target
                    ):
                        raise RuntimeError(
                            "edge audit failure on rank {}: {} (ids must "
                            "reference existing earlier nodes, from <= to; "
                            "next_id={})".format(rank, edge, next_id))
            nodes.clear()
            edges.clear()

    def _mark(self) -> dict:
        return {
            rank: (len(builder.nodes), len(builder.edges))
            for rank, builder in self.builders.items()
        }

    # ------------------------------------------------------------- 发射 --

    def emit_admission_batch(self, request_plan: dict) -> dict:
        """发射 request 的准入动作（ARRIVAL 决策的图；拼 batch 改造
        2026-08-22）：到达/interval gate + history_evictions +
        history_transfer（含 partial 流水恢复全部节点）+ prefill_evictions +
        prefill readiness barrier。

        prefill 主体（chunk 序列）与 PREFILL_DRAIN watch 不再在此发射——
        移入实例迭代列车（emit_iteration_train 的折叠体与 drain 标记）。
        独立逐出支链返回自己的尾节点 watch，不能借 prefill drain 提前核销。

        strategy 恒走 roofline 物理时钟；[frontier 接续裁决，strategy
        死锁修复统一(2026-08-19)] 的无条件接续语义不变：准入动作链到该
        rank 当前 frontier（实例列车在飞时物理排在列车后）。"""
        if self.config.trace_granularity != "request_aggregated":
            raise RuntimeError(
                "online emission supports request_aggregated granularity "
                "only (production config)")
        self._set_context(request_plan, "prefill", 0)
        marker = self._mark()
        eviction_watches = self._emit_admission_actions(request_plan)
        self._collect(marker)
        return {"eviction_watches": eviction_watches}

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
        eviction_watches = self._emit_train_head(train_plan)
        credit_blocks = (train_plan.get("remote_credit") or {}).get(
            "body_blocks")
        copy_blocks, copy_armed = self._copy_handoff_body_blocks(
            train_plan, list(train_plan["pass_spans"]))
        if credit_blocks is not None and copy_blocks is not None:
            raise RuntimeError(
                "train carries both remote-credit body blocks and copy "
                "handoff arms -- a request is either remote-read or copy; "
                "mixed block splitting is unsupported (fail-closed)")
        self._emit_train_body(
            train_plan, list(train_plan["pass_spans"]),
            int(train_plan["iterations"]),
            first_chunk_member=train_plan.get("first_chunk_member"),
            credit_blocks=(credit_blocks if credit_blocks is not None
                           else copy_blocks))
        if copy_armed:
            # C13：交接尾块门账本随体发射消费（joiner 恰一次）；残留尾块
            # 未被任何体块 arm = 切分/发射 bug，fail-closed。K6：层区间
            # 账本同步弹出（段边界已进入首体块层段发射）。
            consumed = self._copy_handoff_arms.pop(copy_armed[0], None)
            self._copy_handoff_layers.pop(copy_armed[0], None)
            if consumed and any(
                    block_index not in {
                        gate[1] for block in (copy_blocks or ())
                        for gate in block["gates"]}
                    for block_index in consumed):
                raise RuntimeError(
                    "copy handoff arms were not fully consumed by the "
                    "train body (un-gated tail chunks would let compute "
                    "outrun the migration stream)")
        # 规格书§三.8：前缀读流组门账本的列车尾检查（首 chunk 成员的
        # 账本必须已被体消费；未消费 fail-closed）。
        self._assert_prefill_read_arms_consumed(train_plan)
        result = self._emit_train_tail_markers(train_plan, first_token)
        result["eviction_watches"] = eviction_watches
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
        if self._train_has_pending_copy_arms(train_plan):
            # C13：首步拆分列车不支持 copy 交接体门控（SH_FIRST_TOKEN_SPLIT
            # 缺省关——B3 裁决后 OFF 为唯一生产姿态；整列发射路径才是
            # 四步协议的体块化载体）。
            raise RuntimeError(
                "WP9 first-step split trains with pending copy handoff "
                "arms are unsupported (whole-train body gating required)")
        marker = self._mark()
        eviction_watches = self._emit_train_head(train_plan)
        self._emit_train_body(
            train_plan, list(first_token["first_spans"]), 1,
            first_chunk_member=train_plan.get("first_chunk_member"),
            credit_blocks=first_token.get("first_body_blocks"))
        # 规格书§三.8：首步批消费前缀读流组门（首 chunk 恒在首步批）——
        # 尾检查钉"首步已消费、余量批不重复消费"。
        self._assert_prefill_read_arms_consumed(train_plan)
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
            "eviction_watches": eviction_watches,
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
            int(train_plan["iterations"]) - 1,
            credit_blocks=first_token.get("rest_body_blocks"))
        # 规格书§三.8：余量批不重复消费前缀读流组门（体不传
        # first_chunk_member）——尾检查在首步批已消费时恒通过；残留 =
        # 首步批未消费，fail-closed。
        self._assert_prefill_read_arms_consumed(train_plan)
        result = self._emit_train_tail_markers(train_plan, first_token)
        self._collect(marker)
        return result

    def _emit_train_head(self, train_plan: dict) -> list[dict]:
        """列车头：joiner 迁移 + 共享 readiness barrier + 起始标记节点。

        拆分时整段归首步批（迁移/栅栏/起始标记锚定的迭代位置在第 1
        迭代内，语义不变）；发射序与整列发射逐节点一致。"""
        instance_index = train_plan["instance_index"]
        group = self.group_by_index[instance_index]
        train_id = train_plan["train_id"]

        # ---- joiner 迁移（触发门 = 该成员 drain 列车的 post-barrier
        #      块末；上下文 (joiner, decode, 1) = decode_start 指标锚点） ----
        joiners = list(train_plan.get("joiners", ()))
        eviction_watches = []
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

            # KV 逐出并行化（2026-09-13）：joiner decode 逐出搬进旁路分支
            # ——drain 列车 barrier 触发门保留在分支首节点（逐出开始时刻
            # 不变），分支不 join（逐出物理完成不再阻塞 readiness barrier
            # 与列车体）；紧随其后的 prefill_decode_transfer（恢复类迁移，
            # decode_start 指标锚点）保持主链不动。
            decode_transfers = joiner.get("decode_evictions") or ()
            if decode_transfers:
                watch_id = joiner.get("decode_eviction_watch_id")
                if watch_id is None:
                    watch_id = (
                        "batch_train_evict_{}_decode_{}".format(
                            joiner["request_id"], train_id))
                def emit_decode_evictions() -> None:
                    for transfer in decode_transfers:
                        record = emit_transfer(
                            transfer, "decode_evictions",
                            trigger_gate=decode_eviction_trigger)
                        self._register_store_tails(transfer, record)
                members = self._emit_side_branch(
                    emit_decode_evictions,
                    watch_context=(watch_id, "prefill", 0))
                if members:
                    eviction_watches.append({
                        "request_id": watch_id,
                        "owner_request_id": joiner["request_id"],
                        "branch": "decode_joiner",
                        "members": members})
            # remote-read credit 首列车（唯一执行口径）：切片
            # 块 2..M 旁挂支链（D2 发射序 = checkpoint → 支链尾块 →
            # restore → 块 1 上主链——restore_chain 会把主链回滚到
            # checkpoint（:246-248 回滚语义），块 1 先发射会被回滚抹掉，
            # 次序不可反）。尾块 recv 完成门入 _credit_arms 账本，由
            # 对应体块首节点 arm 消费（块 b ← 门 b，I2）。
            credit_tail = joiner.get("remote_credit_tail") or ()
            if credit_tail:
                def emit_joiner_credit_tail(
                        joiner=joiner, credit_tail=credit_tail) -> None:
                    self._emit_credit_stream_tail(
                        joiner, credit_tail, stage="remote_credit")
                self._emit_side_branch(emit_joiner_credit_tail)
            pd_transfer = joiner.get("prefill_decode_transfer")
            if pd_transfer is None:
                raise RuntimeError(
                    "joiner is missing its Prefill-to-Decode KV action")
            emit_transfer(pd_transfer, "prefill_decode_transfer")

        # ---- remote-read credit 续列车（T2+ 续坐成员，D7 新发射槽位）：
        #      本列车切片块 1 上主链（发射序先于列车体；无 barrier——
        #      per-rank 链序保证先行，§4.1 v3：与 T1 的形态区别仅缺
        #      barrier），块 2..M 旁挂 + arm 门，与 T1 同构。上下文
        #      (member, decode, 1) 与 joiner 槽位同构。 ----
        for member in ((train_plan.get("remote_credit") or {}).get(
                "continuations") or ()):
            member_plan = member["plan"]
            self._set_context(member_plan, "decode", 1)
            tail = member.get("tail") or ()
            if tail:
                def emit_cont_credit_tail(
                        member_plan=member_plan, tail=tail) -> None:
                    self._emit_credit_stream_tail(
                        member_plan, tail, stage="remote_credit")
                self._emit_side_branch(emit_cont_credit_tail)
            self._emit_credit_head_transfer(member_plan, member["block1"])

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
        return eviction_watches

    def _emit_train_body(self, train_plan: dict, pass_spans,
                         weight_passes: int, *,
                         first_chunk_member=None,
                         credit_blocks=None) -> None:
        """折叠列车体（17 类聚合节点；weight_passes = 权重读取次数）。

        拆分时首步批传首步 span 组 + weight_passes=1（含队列头首 chunk
        与 partial 恢复 suffix 门挂接），余量批传余量组 + iterations-1；
        两批激活/KV/AR 字节按 span 求和与整列一致。

        remote-read credit 体块化（唯一执行口径，
        credit_blocks 非 None 时）：体按块切分发射——每体块一次
        transformer_pass_aggregated（块内 spans 子集 + 块迭代数的
        weight_passes；跨块求和与整列一致），首节点额外依赖 = 覆盖
        该块迭代区间的全部 remote-read 成员切片块完成门之并集（I2，
        经 _credit_arms 账本逐 rank arm；块 b 不得依赖后续块）。suffix
        恢复门挂首块（首 chunk 恒在首个体块）。credit_blocks=None =
        本列车无 remote-credit 参与，单段发射路径与 v1 逐字节一致。

        多块集体节点命名（A6' 修复，2026-09-22）：C++ commit 预检按
        (pg_name, 节点名) 计集体参与者——同名集体在单 rank 出现 N 次
        即 N participants（期望恰 1），auto-K 多块（M>1）下 8 个体块
        同用 train_id 作 phase 会使 ``{train_id}_all_layers_*_all_
        reduce`` 同名 8 份、预检拒批（C4b 首次后端受控执行暴露）。
        规则与 C13 copy 体块 ``_cb{k}``` 后缀同款：无显式 phase 的
        credit 体块（remote-read 路径，SH 规划侧不带 phase 键）在
        M>1 时逐块挂 ``{train_id}_rcb{b}`` 唯一后缀；M=1 不加后缀
        ——K≥S 单块 = v1 等价锚逐字节不变（I3a）。copy 体块自带
        phase（_copy_handoff_body_blocks）不受影响。"""
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
        restore_arms = None
        prefix_arms = None
        prefix_layer_map = None
        if first_chunk_member is not None:
            suffix_arms = self._suffix_body_arms.pop(
                first_chunk_member["request_id"], None)
            restore_arms = self._suffix_restore_arms.pop(
                first_chunk_member["request_id"], None)
            # 规格书§三.5/§三.8（2026-09-25 prefill remote-read）：首体
            # 恰一次消费前缀读流组门账本（首 token 拆分时首步批即首体
            # ——余量批不传 first_chunk_member，不重复消费）。
            prefix_arms = self._prefill_remote_read_arms.pop(
                first_chunk_member["request_id"], None)
            prefix_layer_map = self._prefill_remote_read_layers.pop(
                first_chunk_member["request_id"], None)
            if suffix_arms and restore_arms:
                raise RuntimeError(
                    "train head carries both legacy whole-suffix arms and "
                    "per-group restore arms (plan shape bug)")
            if suffix_arms and prefix_arms is not None:
                raise RuntimeError(
                    "train head carries both legacy whole-suffix arms and "
                    "prefill remote-read prefix arms (plan shape bug)")
            if prefix_arms is not None or prefix_layer_map is not None:
                # 双账本成对登记/成对消费（准入发射同时写两账；单边在场 =
                # 生命周期破损，fail-closed）。
                if prefix_arms is None or prefix_layer_map is None:
                    raise RuntimeError(
                        "prefill remote-read arm/layer ledgers must be "
                        "registered and consumed in pairs (arms present: "
                        f"{prefix_arms is not None}, layers present: "
                        f"{prefix_layer_map is not None})")
                if set(prefix_arms) != set(prefix_layer_map):
                    raise RuntimeError(
                        "prefill remote-read arm/layer ledgers disagree "
                        f"on group indexes (arms {sorted(prefix_arms)} vs "
                        f"layers {sorted(prefix_layer_map)})")
            if suffix_arms:
                missing = [
                    rank for rank in group.ranks
                    if suffix_arms.get(rank) is None]
                if missing:
                    raise RuntimeError(
                        "suffix restore gate missing on ranks {}".format(
                            missing))
            if restore_arms:
                # 分组覆盖校验：层区间自驻留前缀顶向上连续铺到 L（不连
                # 续 = 发射/切分 bug，fail-closed）；组序 = 消费顺序。
                expected_start = restore_arms[0][1]
                for _g, layer_start, layer_end, _gates in restore_arms:
                    if layer_start != expected_start or layer_end <= layer_start:
                        raise RuntimeError(
                            "restore group arms are not contiguous in "
                            f"consumption order (got [{layer_start}, "
                            f"{layer_end}), expected start {expected_start})")
                    expected_start = layer_end
                if expected_start != self.config.layers:
                    raise RuntimeError(
                        "restore group arms do not tile the suffix up to the "
                        f"model layer count ({expected_start} != "
                        f"{self.config.layers})")
        for builder in self.builders.values():
            builder.set_context(train_id, stage, generation)
        tensor_parallel = len(group.ranks)

        def emit_pass(spans, passes, rank, relative_rank, phase=None,
                      layer_start=0, layer_end=None, include_output=True,
                      extra_gates_by_rank=None) -> None:
            if extra_gates_by_rank is not None:
                gate = extra_gates_by_rank.get(rank)
                if gate is not None:
                    self.builders[rank].arm_dependency(gate)
            transformer_pass_aggregated(
                self.builders[rank],
                phase=phase if phase is not None else train_id,
                pass_spans=spans,
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
                weight_passes=passes,
                layer_start=layer_start,
                layer_end=layer_end,
                include_output=include_output,
            )

        def emit_layer_segmented(spans, passes, phase=None):
            """C15+K6+§三.6：含首 chunk 计算体的逐层段发射（设计方案
            §5.1 逐层恢复与 prefill 重叠；2026-09-23 扩展至 copy 交接尾
            块；2026-09-25 扩展至 prefill remote-read 前缀读流组）。

            层段 = [0, 首段起点) 的热前缀段（无门——copy 块 0 在准入主链
            /restore 前的零 KV 层区间）＋各 copy 交接尾块层段（逐 rank 等
            该尾块 recv 完成门——K6/P1-③：C13(4)"未到达块按就绪事件等
            待"的层粒度兑现；原体块粒度门只挂尾块 c，全层聚合 pass 计算
            尾块 c+1..M-1 层不等其到达 = 时序乐观，生产 32 层 4-chunk 必
            现）＋各前缀读流组层段（remote-read：逐 rank 等该组 home→exec
            前缀 recv 完成门，规格书§三.6）＋各恢复组层段（逐 rank 等该
            组目标 HBM 写完成门）。remote-read 混合形态铺满序 =
            [0, p) 前缀读流组 → [p, L) 后缀恢复组（§三.6：两段并集恒连
            续覆盖 [0, L)，gap/overlap 即 fail-closed）。段集连续铺满
            [0, L]（不连续 = 发射/切分 bug，fail-closed）；每段一次
            transformer_pass_aggregated（同一 spans 集、该段层区间、
            weight_passes=passes）——跨段求和与整段单次发射逐字节一致
            （激活/KV/AR 按 Σ 段层跨度 × spans、权重按 passes × Σ 层跨
            度、final norm/logits 仅末段一次），物理上首 chunk 的计算
            按层序分段推进：段 i 只等其数据源——**不设**"全部冷层就绪
            才开始 prefill"串行段。
            """
            segments = []
            covered = 0
            copy_gate_request_id = (
                first_chunk_member or {}).get("request_id")
            copy_layers = (
                self._copy_handoff_layers.get(copy_gate_request_id)
                if copy_gate_request_id is not None else None)
            if copy_layers and prefix_layer_map is not None:
                # copy 交接与 remote-read 前缀读流分属互斥动作（copy /
                # remote-read）——两账本并存 = 计划形态破损，fail-closed。
                raise RuntimeError(
                    "copy handoff layers and prefill remote-read prefix "
                    "arms cannot coexist on the same request (mutually "
                    "exclusive joint actions)")
            if copy_layers:
                copy_arms = self._copy_handoff_arms.get(
                    copy_gate_request_id, {})
                for chunk_index in sorted(copy_layers):
                    layer_start, layer_end = copy_layers[chunk_index]
                    if layer_start != covered:
                        if layer_start > covered:
                            if covered > 0:
                                # L5（2026-09-23 复核审计）：账本内部缺
                                # 口 = 不变式破坏（_emit_copy_handoff_
                                # tail 登记面恒连续）——静默插无门段会
                                # 让缺口层不等到达即计算（时序乐观），与
                                # 下方 restore 侧同情形 raise 对称。
                                raise RuntimeError(
                                    "copy handoff tail chunk {} layers "
                                    "[{}, {}) leave an internal gap after "
                                    "covered {} (ledger must be "
                                    "contiguous)".format(
                                        chunk_index, layer_start,
                                        layer_end, covered))
                            # 热前缀段（covered==0：copy 块 0 走准入主
                            # 链，无旁挂门——合法无门段）。
                            segments.append((covered, layer_start, None))
                            covered = layer_start
                        else:
                            raise RuntimeError(
                                "copy handoff tail chunk {} layers "
                                "[{}, {}) overlap the covered prefix "
                                "{}".format(
                                    chunk_index, layer_start, layer_end,
                                    covered))
                    arm_map = copy_arms.get(chunk_index)
                    if arm_map is None:
                        raise RuntimeError(
                            "copy handoff arm ledger is missing chunk {} "
                            "of request {} (tail emission must precede "
                            "the train body)".format(
                                chunk_index, copy_gate_request_id))
                    segments.append((layer_start, layer_end, arm_map))
                    covered = layer_end
            if prefix_layer_map is not None:
                # §三.6（2026-09-25 prefill remote-read）：前缀读流组层段
                # （[0, p)）——逐组逐 rank 等该组 home→exec recv 完成门；
                # 组序 = 层序 = 消费顺序，自 0 连续铺满（gap/overlap 即
                # fail-closed；组 0 起点 > 0 = 前缀层无门先算，时序乐观
                # 同罪）。消费顺序与后缀恢复组衔接处由下方 restore 走查
                # 的连续性检查钉住（[0, p) 前缀 → [p, L) 后缀）。
                for group_index in sorted(prefix_layer_map):
                    layer_start, layer_end = prefix_layer_map[group_index]
                    if layer_start != covered or layer_end <= layer_start:
                        raise RuntimeError(
                            "prefill remote-read prefix groups are not "
                            "contiguous in consumption order (group {} "
                            "got [{}, {}) covered {})".format(
                                group_index, layer_start, layer_end,
                                covered))
                    arm_map = prefix_arms.get(group_index)
                    if arm_map is None:
                        raise RuntimeError(
                            "prefill remote-read arm ledger is missing "
                            "group {} of request {} (stream emission must "
                            "precede the train body)".format(
                                group_index, copy_gate_request_id))
                    segments.append((layer_start, layer_end, arm_map))
                    covered = layer_end
            if restore_arms:
                first_group_start = restore_arms[0][1]
                if first_group_start > covered:
                    if covered > 0:
                        # L5：copy 尾段与恢复组之间缺口 = 账本并集不变
                        # 式破坏（同上 fail-closed；covered==0 的纯
                        # restore 热前缀仍是合法无门段）。
                        raise RuntimeError(
                            "restore group layers [{}, {}) leave an "
                            "internal gap after covered {} (copy/restore "
                            "ledgers must tile contiguously)".format(
                                first_group_start,
                                restore_arms[0][2], covered))
                    segments.append((covered, first_group_start, None))
                    covered = first_group_start
                for _group, layer_start, layer_end, gates in restore_arms:
                    if layer_start != covered or layer_end <= layer_start:
                        raise RuntimeError(
                            "layer segments are not contiguous in "
                            "consumption order (got [{}, {}) covered "
                            "{})".format(
                                layer_start, layer_end, covered))
                    segments.append((layer_start, layer_end, gates))
                    covered = layer_end
            if not segments:
                raise RuntimeError(
                    "layer-segmented emission invoked with no segments")
            if covered < self.config.layers:
                # 尾部零 KV 层区间（PARTIAL 基零后缀字节形态——无恢复组
                # 覆盖）：无门段补齐到 L（跨段字节守恒需 [0, L) 全覆盖）。
                segments.append((covered, self.config.layers, None))
                covered = self.config.layers
            if covered != self.config.layers:
                raise RuntimeError(
                    "layer segments do not tile the model layer count "
                    "({} != {})".format(covered, self.config.layers))
            last_index = len(segments) - 1
            for position, (layer_start, layer_end, gates) in enumerate(
                    segments):
                for relative_rank, rank in enumerate(group.ranks):
                    emit_pass(
                        spans, passes, rank, relative_rank, phase=phase,
                        layer_start=layer_start, layer_end=layer_end,
                        include_output=(position == last_index),
                        extra_gates_by_rank=gates)

        if credit_blocks is None:
            if restore_arms is None and prefix_layer_map is None:
                for relative_rank, rank in enumerate(group.ranks):
                    if suffix_arms:
                        self.builders[rank].arm_dependency(suffix_arms[rank])
                    emit_pass(pass_spans, weight_passes, rank, relative_rank)
                return
            emit_layer_segmented(pass_spans, weight_passes)
            return
        # K6（P1-③）：copy 交接尾块层区间在场 ⇒ 首体块走层段化发射
        # （段边界与逐 rank 门见 emit_layer_segmented）；remote-credit
        # 体块（无 layers 账本）路径不变。
        copy_gate_request_id = (first_chunk_member or {}).get("request_id")
        copy_tail_layers = (
            self._copy_handoff_layers.get(copy_gate_request_id)
            if copy_gate_request_id is not None else None)
        multi_credit_blocks = len(credit_blocks) > 1
        for position, block in enumerate(credit_blocks):
            if not block["spans"]:
                raise RuntimeError(
                    "credit body block [{},{}] has no spans".format(
                        block["start_iter"], block["end_iter"]))
            # A6'：无显式 phase 的 credit 体块在多块形态下逐块挂唯一
            # phase 后缀（同名集体节点单 rank 8 participants 预检拒批
            # 的修复面；单块不加——v1 等价锚逐字节不变）。copy 体块
            # 自带 phase（_cb{k}），原样透传。
            block_phase = block.get("phase")
            if block_phase is None and multi_credit_blocks:
                block_phase = "{}_rcb{}".format(train_id, position + 1)
            segment_first_block = (
                position == 0
                and (restore_arms is not None or copy_tail_layers
                     or prefix_layer_map is not None))
            for rank in group.ranks:
                if position == 0 and suffix_arms:
                    self.builders[rank].arm_dependency(suffix_arms[rank])
                for gate_request_id, block_index in block["gates"]:
                    if segment_first_block and copy_tail_layers:
                        # K6：首体块的 copy 尾块门由层段发射逐段精确
                        # arm（块粒度 arm 与段门并存 = 同门双重依赖，
                        # 冗余且掩盖段边界语义）。
                        continue
                    if block_index < 2 and not (
                            self._copy_handoff_arms.get(
                                gate_request_id, {}).get(block_index)):
                        # 切片块 1 无旁挂门：主链先行性承担（T1 经
                        # barrier、T2+ 经 per-rank 链序，§4.1）。C13 copy
                        # 交接尾块从 1 起编号（块 0 在准入主链）——copy 门
                        # 的块 1 有旁挂支链，须正常 arm。
                        continue
                    arm_map = (
                        self._credit_arms.get(
                            gate_request_id, {}).get(block_index)
                        or self._copy_handoff_arms.get(
                            gate_request_id, {}).get(block_index))
                    if arm_map is None:
                        # 尾块发射恒先于体（_emit_train_head 先于
                        # _emit_train_body；copy 尾块在准入批发射）——门
                        # 账本整块缺失只可能是 bug（尾块未发射/未登记/
                        # 块号错位），fail-closed（2026-09-17 交付后复核
                        # 硬化；C13 copy 同款）。
                        raise RuntimeError(
                            "credit arm ledger is missing block {} of "
                            "request {} (tail emission must precede the "
                            "train body)".format(
                                block_index, gate_request_id))
                    gate_node = arm_map.get(rank)
                    if gate_node is not None:
                        # 该 rank 有 shard 才有 recv 门；零字节 rank 与
                        # v1 同口径跳过（无数据即无到达可等）。
                        self.builders[rank].arm_dependency(gate_node)
            if segment_first_block:
                # C15+K6+§三.6 组合形态：首块（覆盖迭代 1 = 首 chunk）按
                # 层段切分发射——恢复组门/copy 尾块门/前缀读流组门逐段挂
                # （见 emit_layer_segmented）；块 ≥ 1 原样（其迭代的层消
                # 费已被首块各段门传递覆盖：块 1+ 的 spans 计算晚于首块
                # 全部层段完成 ⇒ 晚于全部尾块/恢复组/前缀组——既有
                # per-block 门成为冗余无害保险）。
                emit_layer_segmented(
                    block["spans"], block["weight_passes"],
                    phase=block_phase)
                continue
            for relative_rank, rank in enumerate(group.ranks):
                emit_pass(
                    block["spans"], block["weight_passes"],
                    rank, relative_rank,
                    phase=block_phase)

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

        # remote-read credit：清本列车的 arm 账本——尾标记每列车恰发射
        # 一次（整列发射/余量批同路径），此刻全部体块消费点均已发射；
        # 下一列车该请求的切片在列车头重新登记（块号从 2 起重用，不残留
        # 陈旧门）。O10④：本列车发射过尾块（remote_credit_tail 非空）
        # 的请求在 pop 点必须有账本——尾块发射登记与尾标记结清同列车
        # 成对出现，缺失 = 发射/记账链破损，fail-closed；单块切片无尾块
        # （账本合法缺席，pop 空放）。
        credit_plan = train_plan.get("remote_credit") or {}
        expected_arm_owners = set()
        for joiner in train_plan.get("joiners") or ():
            if joiner.get("remote_credit_tail"):
                expected_arm_owners.add(joiner["request_id"])
        for member in credit_plan.get("continuations") or ():
            if member.get("tail"):
                expected_arm_owners.add(member["plan"]["request_id"])
        for request_id in credit_plan.get("request_ids") or ():
            leftover_arms = self._credit_arms.pop(request_id, None)
            if request_id in expected_arm_owners and leftover_arms is None:
                raise RuntimeError(
                    "credit arm ledger for {} is missing at train "
                    "tail-marker settlement although this train emitted "
                    "its credit tail blocks (emission/bookkeeping chain "
                    "broken)".format(request_id))

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

    # ---------------------------------- remote-read credit 发射（v2）--

    def _emit_credit_stream_tail(self, member_plan, blocks, *, stage) -> None:
        """remote-read credit 尾块（切片块 2..M）旁挂支链的物理发射
        （D2 credit 发射拓扑，经 _emit_side_branch 包裹——分支首节点
        parent = fork frontier，分支不 join，与 decode_evictions 支链
        同款）：

        P4（PARTIAL 跨实例 copy 流水化 2026-09-25，与 _emit_copy_
        handoff_tail 同款机制一致性）：块逐笔并行支链——每块对涉及
        rank 做 chain_checkpoint/restore_chain 段内分支：

        - 块间无边：全部块的首节点 parent = fork frontier，home send /
          exec recv 逐笔互并发（旧「home send 直连链 / exec recv 顺序
          链 / ack 链尾随」的整段串行依赖移除——两腿共用同一 NoC/HBM
          模型，图形态不再不对称）；
        - 块内父边为新形态：ack_recv_b ← 本块 send_b、ack_send_b ← 本
          块 recv_b；跨 rank send/recv 无图边，因果由 tag 配对在运行时
          承载（CommonNetworkApi 迟到 sim_recv 立即补回调）；
        - 块 b 的逐 rank recv 完成门记入 _credit_arms[rid][b]，由覆盖
          该迭代区间的体块首节点 arm 消费（I2 门并集）。

        blocks = ((块号 b, KVTransfer), ...)——块号为切片内绝对序
        （体块门规格按同一块号引用）。M=1 时无尾块（调度器不置
        remote_credit_tail），切片块 1 走 v1 原路径——I3a 逐字节锚。"""
        request_id = member_plan["request_id"]
        # O10④（2026-09-23 终轮审计）：前序列车尾标记必须已结清本
        # 账本（每列车尾标记发射点统一 pop——:1255-1261）；残留即
        # 前一列车的 pop 名单漏本请求 = 记账破损被 setdefault 静默
        # 吞并（旧门账与新门账混装），fail-closed。检查须在块循环
        # **之前**（单次调用内块 2..M 连续登记本账本，同调用内的
        # 后续块非残留）。
        if request_id in self._credit_arms:
            raise RuntimeError(
                "credit arm ledger for {} still holds blocks {} from "
                "a previous train (tail-marker settlement missed this "
                "request -- bookkeeping bug)".format(
                    request_id, sorted(self._credit_arms[request_id])))
        prefix = _prefix_of(member_plan)
        action_state = self._action_seq.setdefault(request_id, [0])
        named_blocks = []
        for block_index, transfer in blocks:
            action_sequence = action_state[0]
            action_name = (
                f"{prefix}_{stage}_action{action_sequence:03d}_"
                f"{sanitize_node_prefix(transfer.session_id)}_{transfer.kind}"
            )
            action_state[0] = action_sequence + 1
            named_blocks.append((block_index, transfer, action_name))
        for block_index, transfer, action_name in named_blocks:
            involved_ranks = []
            for shard in transfer.shards:
                for rank in (shard.source_rank, shard.target_rank):
                    if rank not in involved_ranks:
                        involved_ranks.append(rank)
            checkpoints = {
                rank: self.builders[rank].chain_checkpoint()
                for rank in involved_ranks}
            per_rank = {}
            for shard_index, shard in enumerate(transfer.shards):
                data_tag = self.tag_allocator.take()
                self.builders[shard.source_rank].comm_send(
                    f"{action_name}_shard{shard_index}"
                    f"_credit{block_index}_send",
                    src=shard.source_rank, dst=shard.target_rank,
                    comm_size=shard.bytes, comm_tag=data_tag)
                self.builders[shard.target_rank].comm_recv(
                    f"{action_name}_shard{shard_index}"
                    f"_credit{block_index}_recv",
                    src=shard.source_rank, dst=shard.target_rank,
                    comm_size=shard.bytes, comm_tag=data_tag)
                recv_node = self.builders[shard.target_rank].previous_id
                if recv_node is None:
                    raise RuntimeError(
                        "credit stream recv did not generate a node")
                per_rank[shard.target_rank] = recv_node
                ack_tag = self.tag_allocator.take()
                self.builders[shard.target_rank].comm_send(
                    f"{action_name}_shard{shard_index}"
                    f"_credit{block_index}_ack_to_rank{shard.source_rank}",
                    src=shard.target_rank, dst=shard.source_rank,
                    comm_size=1, comm_tag=ack_tag)
                self.builders[shard.source_rank].comm_recv(
                    f"{action_name}_shard{shard_index}"
                    f"_credit{block_index}_ack_from_rank{shard.target_rank}",
                    src=shard.target_rank, dst=shard.source_rank,
                    comm_size=1, comm_tag=ack_tag)
            self._credit_arms.setdefault(request_id, {})[block_index] = (
                per_rank)
            for rank in involved_ranks:
                self.builders[rank].restore_chain(checkpoints[rank])

    def _emit_copy_handoff_tail(self, member_plan, chunks, *,
                                pending_gate=None) -> None:
        """C13 copy 交接尾块（块 1..M-1）旁挂支链的物理发射（四步协议
        的图侧 (2)：源 HBM 读取 + 传输；目标 HBM 写完成 = exec 侧 recv
        节点——逐块就绪门控/交接完成事件挂点；ack_recv = 源端释放依赖
        挂点）。经 _emit_side_branch 包裹调用——分支首节点 parent =
        fork frontier，分支不 join。

        P1（PARTIAL 跨实例 copy 流水化 2026-09-25）：尾块逐笔并行支链
        ——每块对涉及 rank 做 chain_checkpoint/restore_chain 段内分支：

        - 块间无边：全部块的首节点 parent = fork frontier（头块后的主
          链），home send / exec recv 逐笔互并发（首块仍是唯一启动栅
          栅栏——块 0 在准入主链，本方法只接块 ≥ 1）；
        - 块内父边为新形态：ack_recv_c ← 本块 send_c（source builder
          链序）、ack_send_c ← 本块 recv_c（target builder 链序）；跨
          rank send/recv 无图边，因果由 tag 配对在运行时承载（
          CommonNetworkApi：流完成后迟到的 sim_recv 立即补回调——任意
          发射序安全）；旧「home send 直连链 / exec recv 顺序链 / ack
          链尾随整条数据链」的整段串行依赖移除，不保留；
        - 块 c 的逐 rank recv 完成门记入 _copy_handoff_arms[rid][c]，
          由含首 chunk 的列车体按消费顺序逐块 arm 消费（I2 门并集同源；
          块 c 的消费区间覆盖由 _copy_handoff_body_blocks 比例映射）；
        - 逐块 ack_recv 节点记入 _copy_handoff_release_anchors[rid][c]
          （源端立即释放的物理到达事实锚，账本侧在 prefill drain 边界
          结算，见 FS _settle_copy_handoffs）。

        触发门：gate 实例 == 源实例时每块逐 source rank 重 arm timer
        gate（与头块 noc_migrate 主链同策略；每块 send 都带门依赖，等
        价旧链式「全部尾块等 gate」）；跨实例 gate（上轮执行实例）时
        无门控发射（NEW-1 同款）。
        """
        request_id = member_plan["request_id"]
        prefix = _prefix_of(member_plan)
        action_state = self._action_seq.setdefault(request_id, [0])
        stage = "history_handoff"
        named_chunks = []
        for transfer in chunks:
            chunk_index = transfer.handoff_chunk
            if transfer.kind != "noc_migrate":
                raise RuntimeError(
                    "copy handoff tail chunk must be a NoC migration "
                    f"(got {transfer.kind!r})")
            action_sequence = action_state[0]
            action_name = (
                f"{prefix}_{stage}_action{action_sequence:03d}_"
                f"{sanitize_node_prefix(transfer.session_id)}_{transfer.kind}"
            )
            action_state[0] = action_sequence + 1
            named_chunks.append((chunk_index, transfer, action_name))
        named_chunks.sort(key=lambda item: item[0])
        for chunk_index, transfer, action_name in named_chunks:
            involved_ranks = []
            for shard in transfer.shards:
                for rank in (shard.source_rank, shard.target_rank):
                    if rank not in involved_ranks:
                        involved_ranks.append(rank)
            checkpoints = {
                rank: self.builders[rank].chain_checkpoint()
                for rank in involved_ranks}
            # 触发门 arming（分支内完成——_emit_side_branch 的
            # stash-and-clear 契约）：仅 gate 实例 == 源实例时（头块同
            # 策略）；_history_control 的 control_rank==source_rank 校验
            # 保留。
            if (
                pending_gate is not None
                and transfer.source_instance_index is not None
                and pending_gate.source_instance_index
                == transfer.source_instance_index
            ):
                source_group = self.group_by_index[
                    transfer.source_instance_index]
                for relative_index, source_rank in enumerate(
                        source_group.ranks):
                    control_rank, timer_gate = _history_control(
                        group_by_index=self.group_by_index,
                        pending_gate=pending_gate,
                        relative_index=relative_index,
                    )
                    if control_rank != source_rank:
                        raise RuntimeError(
                            "copy handoff tail control rank is not its "
                            "source rank")
                    self.builders[source_rank].arm_timer_gate(timer_gate)
            per_rank = {}
            release_per_rank = {}
            for shard_index, shard in enumerate(transfer.shards):
                data_tag = self.tag_allocator.take()
                self.builders[shard.source_rank].comm_send(
                    f"{action_name}_shard{shard_index}"
                    f"_handoff{chunk_index}_send",
                    src=shard.source_rank, dst=shard.target_rank,
                    comm_size=shard.bytes, comm_tag=data_tag)
                self.builders[shard.target_rank].comm_recv(
                    f"{action_name}_shard{shard_index}"
                    f"_handoff{chunk_index}_recv",
                    src=shard.source_rank, dst=shard.target_rank,
                    comm_size=shard.bytes, comm_tag=data_tag)
                recv_node = self.builders[shard.target_rank].previous_id
                if recv_node is None:
                    raise RuntimeError(
                        "copy handoff recv did not generate a node")
                per_rank[shard.target_rank] = recv_node
                # 块内 ack 新形态：ack_send 尾随本块 recv（target 链序）、
                # ack_recv 尾随本块 send（source 链序）。
                ack_tag = self.tag_allocator.take()
                self.builders[shard.target_rank].comm_send(
                    f"{action_name}_shard{shard_index}"
                    f"_handoff{chunk_index}_ack_to_rank{shard.source_rank}",
                    src=shard.target_rank, dst=shard.source_rank,
                    comm_size=1, comm_tag=ack_tag)
                self.builders[shard.source_rank].comm_recv(
                    f"{action_name}_shard{shard_index}"
                    f"_handoff{chunk_index}_ack_from_rank{shard.target_rank}",
                    src=shard.target_rank, dst=shard.source_rank,
                    comm_size=1, comm_tag=ack_tag)
                ack_node = self.builders[shard.source_rank].previous_id
                if ack_node is not None:
                    release_per_rank[shard.source_rank] = ack_node
            self._copy_handoff_arms.setdefault(
                request_id, {})[chunk_index] = per_rank
            # K6（P1-③）：尾块层区间随发射登记（首体块层段门控的段
            # 边界——块 c 的 recv 门只应 gate 其层区间 [start, end) 的
            # 计算段，全层聚合 pass 在 c 之前的层段不等 c 即可计算）。
            self._copy_handoff_layers.setdefault(
                request_id, {})[chunk_index] = (
                    transfer.layer_start, transfer.layer_end)
            self._copy_handoff_release_anchors.setdefault(
                request_id, {})[chunk_index] = release_per_rank
            for rank in involved_ranks:
                self.builders[rank].restore_chain(checkpoints[rank])

    def _train_has_pending_copy_arms(self, train_plan) -> bool:
        """本列车成员/队列头是否有待消费的 copy 交接尾块门（C13）。"""
        if not self._copy_handoff_arms:
            return False
        request_ids = {
            request_id for request_id, _ in train_plan.get("members") or ()}
        request_ids.add(train_plan.get("head_request_id"))
        request_ids.discard(None)
        return any(request_id in self._copy_handoff_arms
                   for request_id in request_ids)

    def _assert_prefill_read_arms_consumed(self, train_plan) -> None:
        """规格书§三.8（2026-09-25 prefill remote-read）：本列车含首
        chunk 成员时，前缀读流组门账本必须已被列车体消费——整列发射 =
        本列车体消费；首 token 拆分 = 首步批消费（余量批不重复消费，
        其尾检查在账本已空时恒通过）。残留 = 未消费/消费链破损，两端
        实例（body 弹出点 + 列车尾标记点）fail-closed。"""
        first_chunk_member = train_plan.get("first_chunk_member")
        if first_chunk_member is None:
            return
        request_id = first_chunk_member["request_id"]
        leftover_arms = self._prefill_remote_read_arms.get(request_id)
        leftover_layers = self._prefill_remote_read_layers.get(request_id)
        if leftover_arms is not None or leftover_layers is not None:
            raise RuntimeError(
                "prefill remote-read arms for request {} were not "
                "consumed by the train body (groups {} remain) -- the "
                "prefix stream was never gated into a first-chunk "
                "body".format(
                    request_id,
                    sorted(leftover_arms
                           if leftover_arms is not None
                           else leftover_layers)))

    def _copy_handoff_body_blocks(self, train_plan, pass_spans):
        """C13：含首 chunk 列车体的逐块就绪门控切分（消费顺序映射）。

        体块数 B = min(尾块数 + 1, 迭代数)（首体块无门——readiness
        barrier 已等交接块 0；迭代数不足时尾块并入最后体块，仍保证
        "列车体完成 ⇒ 全部尾块到达"⇒ prefill drain 结算因果成立）。尾块
        c → 体块 min(c, B)：1:1 流水（体块 c 等尾块 c，读流与计算重
        叠，不设"先整份搬运后计算"串行段）。

        span 划分（布局契约：SH _plan_train 平铺序 = [队列头 chunk
        spans（每迭代恰一条，共 iterations 条）][成员连续段]——GB 侧
        train_plan 不携带 members/prefill_chunk_tokens，头部 span 恒为
        前 iterations 条）：头部 span 按迭代区间精确入块（迭代 i ↔ 第
        i 条）；成员 span 按块迭代数比例确定性地分配到尾随块（每 span
        恰入一块、块序保持；权重字节按块迭代数——跨块求和与整列一
        致）。返回 (blocks, (request_id,)) 或 (None, ())。
        """
        first_chunk_member = train_plan.get("first_chunk_member")
        if first_chunk_member is None:
            return None, ()
        request_id = first_chunk_member["request_id"]
        arms = self._copy_handoff_arms.get(request_id)
        if not arms:
            return None, ()
        tail_indices = sorted(arms)
        iterations = int(train_plan["iterations"])
        if iterations <= 0:
            raise RuntimeError(
                "copy handoff train has no iterations to gate")
        if len(pass_spans) < iterations:
            raise RuntimeError(
                "copy handoff train span layout does not match the "
                "frozen plan (fewer spans than head iterations)")
        head_spans = pass_spans[:iterations]
        member_spans = pass_spans[iterations:]
        block_count = min(len(tail_indices) + 1, iterations)
        per_block = -(-iterations // block_count)
        # C15 修复（C13 先在缺陷，跨车道披露——非 C15 引入）：per_block
        # 上取整可使尾块数超出覆盖迭代所需块数，空尾块（end_iter <
        # start_iter）在 stress 10s 窗实测 raise（crash 形 (it, tails) ∈
        # {(4,2),(5,3),(6,3),(9,3),(6,4),(7,4),…}）。收紧 block_count 至
        # 全覆盖所需最小块数；多出尾块经 min(tail_index, block_count)
        # 并入末块（与"迭代数不足时尾块并入最后体块"的既有披露语义一
        # 致，"体块完成 ⇒ 全部尾块到达"不变量保持）。
        block_count = min(block_count, -(-iterations // per_block))
        # 成员 span 的比例分配（确定性，块序保持；最后块吃余量）。
        member_allocation: list[int] = []
        assigned_members = 0
        remaining_members = len(member_spans)
        remaining_iters = iterations
        for block_position in range(1, block_count + 1):
            count = min(
                per_block, iterations - (block_position - 1) * per_block)
            if block_position == block_count:
                take = remaining_members
            else:
                take = remaining_members * count // max(1, remaining_iters)
            member_allocation.append(take)
            assigned_members += take
            remaining_members -= take
            remaining_iters -= count
        if assigned_members != len(member_spans):
            raise RuntimeError(
                "copy handoff member-span allocation lost spans "
                f"({assigned_members} != {len(member_spans)})")
        member_cursor = 0
        blocks = []
        for block_position in range(1, block_count + 1):
            start_iter = (block_position - 1) * per_block + 1
            end_iter = min(block_position * per_block, iterations)
            if end_iter < start_iter:
                raise RuntimeError(
                    "copy handoff body block covers no iterations")
            take = member_allocation[block_position - 1]
            spans = (
                list(head_spans[start_iter - 1:end_iter])
                + member_spans[member_cursor:member_cursor + take])
            member_cursor += take
            if not spans:
                raise RuntimeError(
                    "copy handoff body block covers no spans")
            gates = [
                (request_id, tail_index)
                for tail_index in tail_indices
                if min(tail_index, block_count) == block_position]
            blocks.append({
                "start_iter": start_iter,
                "end_iter": end_iter,
                "weight_passes": end_iter - start_iter + 1,
                "spans": spans,
                "gates": gates,
                # 块唯一 phase 标签：同列车多体块的聚合节点/集体节点不
                # 可重名（C++ commit 预检按名字对集体做跨 rank 签名一
                # 致性校验——同块内逐 rank 字节一致，跨块字节不同，重
                # 名即误配；remote-credit 多块在现行 auto-K 下恒单块，
                # 本后缀是首个真实多块路径的必要区分）。
                "phase": f"{train_plan['train_id']}_cb{block_position}",
            })
        return blocks, (request_id,)

    def _emit_credit_head_transfer(self, member_plan, transfer) -> dict:
        """remote-read credit 续列车切片块 1 的主链发射（exec 实例
        per-rank 链序保证其先于列车体；无 barrier）。发射走 v1 同一
        _emit_kv_transfer noc_migrate 路径，命名经 _action_seq 闭包
        （与 joiner pd_transfer 同款手法）。"""
        request_id = member_plan["request_id"]
        prefix = _prefix_of(member_plan)
        action_state = self._action_seq.setdefault(request_id, [0])
        action_sequence = action_state[0]
        action_name = (
            f"{prefix}_remote_credit_action{action_sequence:03d}_"
            f"{sanitize_node_prefix(transfer.session_id)}_{transfer.kind}"
        )
        record = _emit_kv_transfer(
            config=self.config,
            builders=self.builders,
            group_by_index=self.group_by_index,
            tag_allocator=self.tag_allocator,
            transfer=transfer,
            action_name=action_name,
        )
        record["sequence_stage"] = "remote_credit"
        record["action_sequence"] = action_sequence
        action_state[0] = action_sequence + 1
        return record

    def emit_completion_batch(self, request_plan: dict) -> dict:
        """completion 批（DECODE_COMPLETION/REQUEST_COMPLETE 边界）：
        completion_evictions + 下一 turn interval gate 的依赖登记。
        R11（2026-09-14）：merge 流存在时返回 merge 尾标记节点表
        （{"merge_done_members": {rank: id}, "has_merge": bool}）——调度器
        据此注册 merge-done watch（到达重锚 merge_done）并把下一轮
        interval gate 前递依赖挂到标记上；无 merge 流返回 has_merge=False
        （行为与改造前一致）。"""
        self._set_context(request_plan, "completion", 1)
        marker = self._mark()
        merge_info = self._emit_completion(request_plan)
        self._collect(marker)
        # M4 核销即删（2026-08-23）：completion 批是本请求图发射的终点
        # （seg2 块末在 _emit_completion 内消费、action 序号此后无读者，
        # 全仓 grep 证实无更晚读者）——逐出块末/action 计数账本条目在请求
        # 完成后即死重，当场弹出（下一 turn 是不同 request_id）。
        self._block_ends.pop(request_plan["request_id"], None)
        self._action_seq.pop(request_plan["request_id"], None)
        # C13：交接门账本残留 = 请求完成而尾块从未被体块消费（发射/切分
        # bug）——fail-closed；释放挂点账本为审计面，完成后弹出。
        leftover_arms = self._copy_handoff_arms.pop(
            request_plan["request_id"], None)
        if leftover_arms:
            raise RuntimeError(
                "request {} completed with unconsumed copy handoff arms "
                "{} -- the migration stream was never gated into a train "
                "body".format(
                    request_plan["request_id"], sorted(leftover_arms)))
        self._copy_handoff_layers.pop(request_plan["request_id"], None)
        self._copy_handoff_release_anchors.pop(
            request_plan["request_id"], None)
        # C15：恢复组门账本残留 = 请求完成而恢复组从未被体块消费（发射/
        # 切分 bug）——fail-closed（C13 copy 同款纪律）。
        leftover_restore_arms = self._suffix_restore_arms.pop(
            request_plan["request_id"], None)
        if leftover_restore_arms:
            raise RuntimeError(
                "request {} completed with unconsumed restore group arms "
                "{} -- the suffix restore stream was never gated into a "
                "train body".format(
                    request_plan["request_id"],
                    [arm[0] for arm in leftover_restore_arms]))
        # O10④：credit arm 账本残留 = 请求完成而其切片尾块门账本未被
        # 列车尾标记结清（发射/切分 bug）——fail-closed（C13 copy /
        # C15 restore 同款纪律；单块切片无尾块，账本合法缺席）。
        leftover_credit_arms = self._credit_arms.pop(
            request_plan["request_id"], None)
        if leftover_credit_arms:
            raise RuntimeError(
                "request {} completed with unconsumed credit arms "
                "{} -- the credit tail stream was never gated into a "
                "train body".format(
                    request_plan["request_id"],
                    sorted(leftover_credit_arms)))
        # 规格书§三.5/§三.8（2026-09-25 prefill remote-read）：前缀读流
        # 组门/层区间账本残留 = 请求完成而前缀读流从未被首 chunk 体消费
        # （发射/切分 bug）——fail-closed（C13 copy / C15 restore 同款
        # 纪律；非前缀读流请求无账本，pop 空放）。
        leftover_prefix_arms = self._prefill_remote_read_arms.pop(
            request_plan["request_id"], None)
        leftover_prefix_layers = self._prefill_remote_read_layers.pop(
            request_plan["request_id"], None)
        if leftover_prefix_arms or leftover_prefix_layers is not None:
            raise RuntimeError(
                "request {} completed with unconsumed prefill remote-read "
                "arms {} -- the prefix read stream was never gated into a "
                "train body".format(
                    request_plan["request_id"],
                    sorted(
                        leftover_prefix_arms
                        if leftover_prefix_arms is not None
                        else leftover_prefix_layers)))
        return merge_info

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

    # ------------------------------------ KV 逐出旁路支链（2026-09-13）--

    def _emit_side_branch(
            self, emit_fn, *, watch_context=None) -> dict[int, int]:
        """把一段逐出发射包成旁路分支（主方案 §3.2）：全 rank 暂存并清空
        既有 pending 依赖 → fork 快照 → 发射（分支内自行接续成链；触发门
        的 arming 必须在 emit_fn 内部完成）→ 恢复主链 → 归还暂存依赖。
        分支不 join——被包裹的物理逐出链不再阻塞主链任何节点，HBM 争用由
        C++ LocalHbmBandwidthModel 的 N-way 均分模型在线裁决。

        fork 前非空 pending 的处理（对任务书原案的偏差，REPORT
        DEVIATIONS 登记）：S3 源码现实中 prefill_evictions 发射点之前存在
        合法的主链 armed 依赖（turn-0 arrival gate 的 arm_timer_gate 与
        local_hit history 的 arm），按原案字面 fail-closed 会在正常流量
        崩溃。改为 stash-and-clear 的语义等价实现：分支首节点 parent 恰为
        fork frontier（不带任何 armed 门），主链恢复后原依赖照常由其本来
        的消费者（readiness barrier 等）消费——与逐出链在主链上时完全一致
        的图依赖。分支内部 arming 而未被任何节点消费（触发门泄漏）仍
        fail-closed。
        """
        stashed = {}
        node_counts = {}
        for rank, builder in self.builders.items():
            stashed[rank] = builder.pending_extra_dependencies
            node_counts[rank] = len(builder.nodes)
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
            members = {}
            touched_ranks = [
                rank for rank, builder in self.builders.items()
                if len(builder.nodes) > node_counts[rank]
            ]
            if watch_context is not None and touched_ranks:
                request_id, stage, generation = watch_context
                for rank in touched_ranks:
                    builder = self.builders[rank]
                    builder.set_context(request_id, stage, generation)
                    builder.comp(
                        "eviction_done_rank{}".format(rank), 1, 1)
                    members[rank] = builder.previous_id
            return members
        finally:
            for rank, builder in self.builders.items():
                builder.restore_chain(checkpoints[rank])
                if stashed[rank]:
                    builder.pending_extra_dependencies.extend(stashed[rank])

    def _register_store_tails(self, transfer, record) -> None:
        """逐出支链的边缘 mem_store 尾部登记（主方案 §3.3 登记侧；§4.2.4
        层区间化 2026-09-17）。

        仅 remote_store 逐出登记；record 为 _emit_kv_transfer 的返回值
        （2026-09-13 起携带 edge_store_node_id / source_ack_recv_node_id）。
        条目 = (edge_rank, edge_store_node_id, source_ack_recv_node_id,
        layer_start, layer_end)——层区间取 transfer.layer_start/layer_end
        （逐出池写恒为后缀形 [k, L)）；消费侧按区间交集选择性消费
        （_arm_pending_store_edges）。
        """
        if transfer.kind != "remote_store":
            return
        for shard_record in record["shards"]:
            tails = self.pending_store_tails.setdefault(
                transfer.session_id, [])
            tails.append((
                shard_record["edge_rank"],
                shard_record["edge_store_node_id"],
                shard_record["source_ack_recv_node_id"],
                transfer.layer_start,
                transfer.layer_end,
            ))

    def _match_store_entries(self, session_id, layer_start, layer_end):
        """P2-R2 助手 1/3（PARTIAL 跨实例 copy 流水化 2026-09-25）：纯
        查询——按 §4.2.4 层区间交集判据（max(ls1,ls2) < min(le1,le2)）
        返回 (matched, retained)；不发射节点、不改账本。无匹配 raise
        （fail-closed 文案与 _arm_pending_store_edges 历史一致：本会话
        此前无池写登记却要回迁读池，账目不一致）。"""
        entries = self.pending_store_tails.get(session_id) or []
        matched = []
        retained = []
        for entry in entries:
            entry_layer_start, entry_layer_end = entry[3], entry[4]
            if (max(entry_layer_start, layer_start)
                    < min(entry_layer_end, layer_end)):
                matched.append(entry)
            else:
                retained.append(entry)
        if not matched:
            registered = (
                ", ".join(
                    "[{}, {})".format(entry[3], entry[4])
                    for entry in entries)
                or "none")
            raise RuntimeError(
                "store->restore 前递保障失效: session {!r} restore 区间 "
                "[{}, {}) 无交集池写登记条目（现有条目区间: {}）".format(
                    session_id,
                    layer_start, layer_end,
                    registered))
        return matched, retained

    def _emit_store_relays_once(self, matched, restore_edges, name_prefix,
                                relay_cache=None):
        """P2-R2 助手 2/3：跨缘 1B p2p 中继（桥拒绝跨 rank 直边，1B
        p2p 是协议内唯一合法载体），每个 (store_edge, restore_edge) 对
        **恰发射一次**——send 侧 arm 该对全部 store node ids、restore
        侧收 1B（recv 节点 = 恢复链的链序前递锚）。

        relay_cache（可选）按对去重：同批同对第二次调用返回既有 relay
        recv 节点而不重发（评审 #2 的 O(轮数²) 中继膨胀回归钉——恢复组
        支链的中继提升到组循环之前，组数无关）。返回
        {restore_edge: relay_recv_node_id}（本调用实际发射/命中的对）。
        发射次序契约：先中继、后同缘 arm（此时 restore 边缘 rank 尚无
        armed 依赖，中继 recv 不吞并同缘 arm）。"""
        relay_arms = {}
        for edge_rank, store_node_id, _ack_node_id, _ls, _le in matched:
            for restore_edge in restore_edges:
                if edge_rank == restore_edge:
                    continue
                relay_arms.setdefault(
                    (edge_rank, restore_edge),
                    []).append(store_node_id)
        relay_recv_by_edge = {}
        for (store_edge, restore_edge), store_ids in relay_arms.items():
            cached = (
                relay_cache.get((store_edge, restore_edge))
                if relay_cache is not None else None)
            if cached is not None:
                relay_recv_by_edge[restore_edge] = cached
                continue
            tag = self.tag_allocator.take()
            store_builder = self.builders[store_edge]
            for node_id in store_ids:
                store_builder.arm_dependency(node_id)
            store_builder.comm_send(
                f"{name_prefix}_store_sidelink_s{store_edge}"
                f"_r{restore_edge}",
                src=store_edge,
                dst=restore_edge,
                comm_size=1,
                comm_tag=tag,
            )
            self.builders[restore_edge].comm_recv(
                f"{name_prefix}_store_sidelink_s{store_edge}"
                f"_r{restore_edge}",
                src=store_edge,
                dst=restore_edge,
                comm_size=1,
                comm_tag=tag,
            )
            relay_recv_node = self.builders[restore_edge].previous_id
            if relay_cache is not None:
                relay_cache[(store_edge, restore_edge)] = relay_recv_node
            relay_recv_by_edge[restore_edge] = relay_recv_node
        return relay_recv_by_edge

    def _consume_store_entries(self, session_id, matched) -> None:
        """P2-R2 助手 3/3：恰移除 matched 条目（对象身份相等），其余
        保留在 pending_store_tails 供后续消费者；清空即弹会话键。消费
        窗口契约：恢复组支链在**同批发射内**调用（store 发射→restore
        发射一轮内），pending_store_tails 的在案窗口不放大——sh30
        merge_tail_gated 判据（本实例边缘 rank 有在案尾部 ⇒ 列车采样
        排除）的输入面与旧 consume 口径同阶。"""
        if not matched:
            return
        matched_ids = {id(entry) for entry in matched}
        remaining = [
            entry
            for entry in (self.pending_store_tails.get(session_id) or [])
            if id(entry) not in matched_ids]
        if remaining:
            self.pending_store_tails[session_id] = remaining
        else:
            self.pending_store_tails.pop(session_id, None)

    def _restore_group_involved_ranks(self, group_transfer, prefill_group):
        """P2-R2：单组恢复腿涉及的 rank 集（组区域 checkpoint/restore
        的范围）= Prefill ranks ∪ shard edge ranks ∪ shard target
        ranks——组间并行要求每组对全部涉及 rank 独立 checkpoint。"""
        ranks = set(prefill_group.ranks)
        for shard in group_transfer.shards:
            if shard.edge_rank is not None:
                ranks.add(shard.edge_rank)
            if shard.target_rank is not None:
                ranks.add(shard.target_rank)
        return sorted(ranks)

    def _arm_pending_store_edges(self, transfer, name_prefix: str) -> None:
        """回迁发射前的 store→restore 前递依赖补偿（主方案 §3.3；§4.2.4
        硬化 2026-09-17；2026-09-25 起机制实现于三个助手
        _match_store_entries/_emit_store_relays_once/_consume_store_
        entries——公开签名与行为对本方法消费者逐字节不变）。

        逐出支链化后"同会话 store 池写先于其 restore 池读"的主链传递性
        保障失效；本方法在回迁（remote_load）发射前查 pending_store_tails
        补边，粒度取 store 的边缘 mem_store 完成（池写落盘）：

        - 同边缘 rank：直接 arm（同 rank 显式 data_dep 跨批次合法，桥经
          持久 (rank,id) 解析）；
        - 跨边缘 rank：1B p2p 中继（助手 2；restore 边缘收 1B，回迁链
          链其后）。R1' 去钉扎（2026-09-14）后 REMOTE 全量回迁与
          **PARTIAL 跨实例 copy 的后缀池恢复**均可达跨缘（后缀 store 在
          home 边缘、restore 在执行实例边缘；1B 中继路径两用）。

        §4.2.4 层区间交集选择性消费：restore 传输区间
        [transfer.layer_start, transfer.layer_end)；匹配条目 = 区间有
        交集（max(ls1,ls2) < min(le1,le2)）的全部条目；只移除匹配条目，
        未匹配保留在 pending_store_tails 供后续消费者（新增消费者 =
        PARTIAL remote-read 后缀恢复 [p, L)）。行为保持注记：现有全部
        逐出池写层区间恒为后缀形 [k,L)，现有消费者区间（REMOTE 全量
        [0,L) / copy 后缀 [p,L)）与之恒有交集 ⇒ 交集选择性消费与旧
        pop-all 在现行模式下行为等价，硬化属防御性改造，零时间线扰动。

        fail-closed（原 fail-open 静默返回已退役）：会话在
        pending_store_tails 无任何条目与 restore 区间交集（"有条目但
        都不交集"与"无条目"两态同罪——本会话此前无池写登记却要回迁
        读池，账目不一致）→ raise（store→restore 前递保障失效）。

        store 早已物理完成时补边即刻满足（懒处理，无需判在飞）。消费即
        清（匹配条目）：该回迁的池读已排序于全部匹配 store 之后；未匹配
        条目的层区间池数据不受本次读影响。发射次序：先跨缘中继（此时
        restore 边缘 rank 尚无 armed 依赖，中继 recv 不吞并同缘 arm），
        后同缘 arm（由回迁链在该 rank 的首节点消费）。"""
        matched, _retained = self._match_store_entries(
            transfer.session_id, transfer.layer_start, transfer.layer_end)
        self._consume_store_entries(transfer.session_id, matched)
        restore_edges = []
        for shard in transfer.shards:
            if (shard.edge_rank is not None
                    and shard.edge_rank not in restore_edges):
                restore_edges.append(shard.edge_rank)
        self._emit_store_relays_once(matched, restore_edges, name_prefix)
        for edge_rank, store_node_id, _ack_node_id, _ls, _le in matched:
            if edge_rank in restore_edges:
                self.builders[edge_rank].arm_dependency(store_node_id)

    def _rebuild_interval_gate_on_target(
        self, pending_gate, request_plan: dict,
    ):
        """R5：把跨实例的 interval gate 结构性重建到本轮 prefill 实例。

        每相对位 rank 对：源 rank arm 原 gate 后发 1B p2p（源 gate 触发
        后启动）→ 目标 rank 收 1B → 目标 rank 发射重建 timer 节点
        （runtime_ns=0，仅依赖该 recv）。返回源实例 = prefill 实例的新
        PendingHistoryGate。R16-5（2026-09-15）按代码重枚举四个真实消费
        点（交叉断言过：history_evictions 触发门 :1362-1369（发射器
        remote_store 分支消费 trigger）/ no-transfer arm :1421-1423 /
        partial 恢复 arm :1484-1486+:1506-1507 / 通用 remote_load arm
        :1450-1451→发射器 remote_load 分支 arm+1B 中继）——旧 docstring
        的"noc 源端触发"是死参数分支（发射器 noc_migrate 分支从不消费
        trigger_gate，实参已删）、"local_hit arm"是幻影（通用循环对
        local_hit 先 continue，发射器 local_hit+gate 路径自本构建器
        不可达；pd_transfer :715 不传 gate）。孤儿 timer 节点
        O(跨实例轮次 × TP)、runtime_ns=0，可忽略。"""
        source_group = self.group_by_index[
            pending_gate.source_instance_index]
        prefill_group_ranks = self.group_by_index[
            request_plan["prefill_instance_index"]].ranks
        if len(source_group.ranks) != len(prefill_group_ranks):
            raise RuntimeError(
                "cross-instance gate rebuild requires equal TP sizes")
        prefix = _prefix_of(request_plan)
        following_request = self.config.request_queue[
            request_plan["queue_index"]]
        interval = following_request.inter_request_interval_ns
        if interval is None:
            raise RuntimeError("later request lost its inter-request interval")
        # K5（kimi 复审，2026-09-14）：interval + hbm_wait_ns 为任意 ns
        # 粒度，而 timer_gate 的离线同构校验要求整 µs（% 1000 != 0 即
        # raise）——跨实例轮换 + 非零准入等待（恰为 R5/R2 目标工况）会
        # 确定性崩溃。对齐 turn-0 先例（本文件 arrival gate）做 µs 下
        # 取整：duration 不进节点（runtime_ns=0、不存储），仅驱动校验
        # 与 0 跳过，下取整对既有通过路径零影响。
        duration = interval + request_plan.get("hbm_wait_ns", 0)
        duration -= duration % 1000
        rebuilt_gates = []
        for relative_index, (source_rank, target_rank) in enumerate(
                zip(source_group.ranks, prefill_group_ranks)):
            source_gate = pending_gate.timer_gates[relative_index]
            if source_gate is None:
                raise RuntimeError(
                    "cross-instance gate rebuild found a missing source "
                    "gate node (rank {})".format(source_rank))
            relay_tag = self.tag_allocator.take()
            self.builders[source_rank].set_context(
                request_plan["request_id"], "prefill", 0)
            self.builders[source_rank].arm_dependency(source_gate)
            self.builders[source_rank].comm_send(
                f"{prefix}_gate_relay_s{source_rank}_r{target_rank}",
                src=source_rank,
                dst=target_rank,
                comm_size=1,
                comm_tag=relay_tag,
            )
            self.builders[target_rank].set_context(
                request_plan["request_id"], "prefill", 0)
            self.builders[target_rank].comm_recv(
                f"{prefix}_gate_relay_s{source_rank}_r{target_rank}",
                src=source_rank,
                dst=target_rank,
                comm_size=1,
                comm_tag=relay_tag,
            )
            recv_node_id = self.builders[target_rank].previous_id
            rebuilt_gates.append(self.builders[target_rank].timer_gate(
                f"{prefix}_interval_gate_rebuilt_rank{target_rank}",
                duration,
                after_node_id=recv_node_id,
            ))
        return PendingHistoryGate(
            source_instance_index=request_plan["prefill_instance_index"],
            timer_gates=tuple(rebuilt_gates),
            location=pending_gate.location,
        )

    def emit_eviction_side_branch(
            self, transfers, tick: int, *, watch_id: str | None = None):
        """R2/D7（2026-09-14）：准入事务失败路径上**已提交**逐出的图侧
        发射（旁路支链、无触发门——与 prefill_evictions 的发射形态同构；
        失败请求不入队，逐出是合法容量释放，其池写必须进图，否则 C++
        水位盲区 + pending store 不登记）。"""
        transfers = tuple(transfers or ())
        if not transfers:
            return
        marker = self._mark()
        first = transfers[0]
        context_plan = {
            "request_id": first.trigger_request_id,
            "session_id": first.session_id,
        }
        self._set_context(context_plan, "prefill", 0)
        action_state = self._action_seq.setdefault(
            first.trigger_request_id, [0])
        if watch_id is None:
            watch_id = "batch_train_evict_{}_failed_q{:03d}".format(
                first.trigger_request_id, action_state[0])

        def emit_failed_admission_evictions() -> None:
            for transfer in transfers:
                action_name = (
                    f"evict_{sanitize_node_prefix(transfer.trigger_request_id)}"
                    f"_action{action_state[0]:03d}_"
                    f"{sanitize_node_prefix(transfer.session_id)}_{transfer.kind}"
                )
                record = _emit_kv_transfer(
                    config=self.config,
                    builders=self.builders,
                    group_by_index=self.group_by_index,
                    tag_allocator=self.tag_allocator,
                    transfer=transfer,
                    action_name=action_name,
                )
                record["sequence_stage"] = "failed_admission_evictions"
                record["action_sequence"] = action_state[0]
                action_state[0] += 1
                self._register_store_tails(transfer, record)

        members = self._emit_side_branch(
            emit_failed_admission_evictions,
            watch_context=(watch_id, "prefill", 0))
        self._collect(marker)
        if not members:
            return None
        return {
            "request_id": watch_id,
            "owner_request_id": first.trigger_request_id,
            "branch": "failed_admission",
            "members": members,
        }

    def _emit_admission_actions(self, request_plan: dict) -> list[dict]:
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
        eviction_watches = []

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
            # R5（D3，2026-09-14）：跨实例轮换的 interval gate 重建。上轮
            # gate 节点在源执行实例的 rank id 空间；本轮 prefill 实例不同
            # 时直接 arm 会形成跨实例悬空依赖（id 数值在目标空间碰撞 =
            # 静默错连；不碰撞 = C++ 拒绝跨 rank 边，冒烟尺度起即不稳）。
            # 重建 = 目标实例 ranks 上的结构性 timer 节点（duration 沿用
            # 原值、runtime_ns=0——时序因果由 C++ arrival calendar 承载，
            # 重建只迁移结构依赖），逐相对位经 1B p2p 中继依赖源 gate。
            # 同实例路径逐字节不变；保留 source_ranks == prefill_ranks
            # 断言作 backstop。
            if (pending_gate.source_instance_index
                    != request_plan["prefill_instance_index"]):
                pending_gate = self._rebuild_interval_gate_on_target(
                    pending_gate, request_plan)
            else:
                source_group = self.group_by_index[
                    pending_gate.source_instance_index]
                if source_group.ranks != prefill_group.ranks:
                    raise RuntimeError(
                        "interval gate ranks do not match the Prefill "
                        "instance ranks (backstop assertion)")
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

        # ---- history_evictions（KV 逐出并行化 2026-09-13：旁路支链化；
        #      到达/间隔 timer gate 触发门在分支内构建并挂分支首节点——
        #      逐出开始时刻不变，物理完成不再阻塞其后主链的一切计算）----
        if request_plan["history_evictions"]:
            watch_id = request_plan.get(
                "history_eviction_watch_id",
                "batch_train_evict_{}_admission_history".format(
                    request_plan["request_id"]))
            def emit_history_evictions() -> None:
                history_eviction_trigger = TransferTriggerGate(
                    control_instance_index=pending_gate.source_instance_index,
                    node_gates=pending_gate.timer_gates,
                )
                for transfer in request_plan["history_evictions"]:
                    record = emit_transfer(
                        transfer, "history_evictions",
                        trigger_gate=history_eviction_trigger)
                    self._register_store_tails(transfer, record)
            members = self._emit_side_branch(
                emit_history_evictions,
                watch_context=(watch_id, "prefill", 0))
            if members:
                eviction_watches.append({
                    "request_id": watch_id,
                    "owner_request_id": request_plan["request_id"],
                    "branch": "admission_history",
                    "members": members})

        partial_history_restore = (
            request_plan["history_location_before"] is not None
            and request_plan["history_location_before"].location
            == "partial_hbm_remote"
        )
        suffix_ready_nodes_by_rank = {}

        # ---- history_transfers（joint：stay 单笔 / copy 前缀 NoC+后缀池
        #      恢复两笔 / recompute 与 remote-read 零搬运）----
        joint_action = request_plan.get("joint_action", "stay")
        if "history_transfers" in request_plan:
            history_transfers = tuple(
                transfer
                for transfer in (request_plan.get("history_transfers") or ())
                if transfer is not None
            )
        else:
            # 旧 plan 形态（单数字段；fixtures/verify 路径）回退。
            legacy_transfer = _as_plan(request_plan).history_transfer
            history_transfers = (
                (legacy_transfer,) if legacy_transfer is not None else ())
        single_history_transfer = (
            history_transfers[0]
            if len(history_transfers) == 1 else None)
        # C13 copy 逐 chunk 交接：交接块 1..M-1（handoff_chunk ≥ 1）旁挂
        # 支链流水发射（fork 点 = 头块发射后的主链 frontier），块 0 与
        # 其余腿（池恢复后缀等）走下方既有主链循环——readiness barrier
        # 只等主链节点，即只等交接块 0（计算不等整份搬运完成，不设
        # "先整份搬运后计算"串行段；尾块由列车体逐块就绪门控消费）。
        copy_handoff_tail = tuple(
            transfer for transfer in history_transfers
            if (getattr(transfer, "handoff_chunk", None) or 0) >= 1)
        # C15 后缀逐组恢复腿（restore_group 标记；消费顺序 = 层自低向
        # 高）：从主链循环摘出——readiness barrier 不再等整段后缀；逐组
        # 旁挂/分支发射（rank 内串行链）+ 逐组就绪门登记（列车体按层段
        # 门控消费，_suffix_restore_arms）。旧单笔口径（无标记）不变。
        restore_group_transfers = tuple(sorted(
            (
                transfer for transfer in history_transfers
                if transfer.kind == "remote_load"
                and getattr(transfer, "restore_group", None) is not None
            ),
            key=lambda transfer: transfer.restore_group))

        def _is_restore_group_tail(transfer) -> bool:
            return (
                transfer.kind == "remote_load"
                and getattr(transfer, "restore_group", None) is not None)

        # 规格书§三.1（2026-09-25 prefill remote-read 前缀读流）：独立于
        # history_transfers 的前缀读流腿（调度器 plan_dict 新字段）——
        # home 驻留前缀 [0, p) 的逐组 noc_migrate 瞬时读流（stream_only，
        # 不物化入容量账本）。PARTIAL 基与后缀池恢复两腿并行、共享同一
        # 准入 frontier；LOCAL 基 = 全层 [0, L) 读流（history_transfers
        # 恒空）。旧 plan 形态（fixtures/verify 路径）无该键 → 空元组。
        prefill_read_transfers = tuple(
            transfer
            for transfer in (
                request_plan.get("prefill_remote_read_transfers") or ())
            if transfer is not None)

        if copy_handoff_tail or restore_group_transfers:
            main_history_transfers = tuple(
                transfer for transfer in history_transfers
                if (getattr(transfer, "handoff_chunk", None) or 0) < 1
                and not _is_restore_group_tail(transfer))
        else:
            main_history_transfers = history_transfers
        # partial 流水恢复分支仅适用于 stay（同实例单笔后缀恢复）；
        # 跨实例 copy 的前缀+后缀走通用发射：C13 交接尾块（handoff_
        # chunk>=1）与 C15 后缀逐组恢复腿被上方主链选择摘出主链、旁挂
        # 流水发射，规格书§三 前缀读流组则独立旁挂分支——readiness
        # barrier 只等主链（即交接块 0/首块），计算不等整份搬运或整段
        # 后缀恢复。C15：逐组恢复腿不走旧单笔 partial 分支（分组发射
        # 替换整段门；单组亦然——restore_group 有标记即走分组路径）。
        partial_history_restore = (
            partial_history_restore
            and single_history_transfer is not None
            and getattr(single_history_transfer, "restore_group", None)
            is None
            and request_plan["history_location_before"].instance_index
            == request_plan["prefill_instance_index"])

        if not history_transfers:
            if joint_action == "remote-read":
                # remote-read（LOCAL 基——history_transfers 恒空）：本批
                # 无历史搬运主链节点（旧口径此处 pass = interval/arrival
                # gate 从未被消费 + 前缀层计算无数据门，2026-09-25 规格
                # 书§三.2 废除）。前缀读流由下方独立旁挂分支发射：gate
                # 的触发消费、home→exec 层组读流均在分支内完成（前缀
                # 计算的层段门由列车体消费 recv 完成门承担）；gate 结构
                # 性重建（跨实例轮换 → 本轮 prefill 实例）已在 gate 段
                # 经 R5 _rebuild_interval_gate_on_target 完成。decode 相
                # 的 home 前缀 credit 读流仍经 joiner 动作发射（drain
                # 边界），与本批无交集。
                if not prefill_read_transfers:
                    raise RuntimeError(
                        "remote-read admission without its prefill prefix "
                        "read stream (plan_prefill_remote_read_transfers "
                        f"must supply it; request "
                        f"{request_plan['request_id']!r})")
            elif joint_action == "recompute":
                # 重算历史在执行期物化——本批无历史搬运节点。
                pass
            elif request_plan["turn_index"] != 0:
                raise RuntimeError(
                    "later request is missing its history transfer action")
            else:
                source_group = group_by_index[
                    pending_gate.source_instance_index]
                if source_group.ranks != prefill_group.ranks:
                    raise RuntimeError(
                        "first-request arrival gate is not on its Prefill ranks")
                for relative_index, rank in enumerate(prefill_group.ranks):
                    builders[rank].arm_timer_gate(
                        pending_gate.timer_gates[relative_index])
        elif not partial_history_restore:
            # gate location 已在弹出时归一化到权威快照（见上方裁决注释）。
            for history_transfer in main_history_transfers:
                if history_transfer.kind == "local_hit":
                    # 零节点本地命中（stay 驻留复用）——无发射节点。
                    continue
                if history_transfer.kind == "remote_load":
                    # KV 逐出并行化（2026-09-13）：REMOTE 全量回迁入口的
                    # store→restore 前递补边（跨实例回迁可达跨缘，1B 中继
                    # 仅此路径需要）。
                    self._arm_pending_store_edges(
                        history_transfer, f"{prefix}_history_transfer")
                if (
                    history_transfer.kind == "noc_migrate"
                    and history_transfer.source_instance_index is not None
                    and pending_gate.source_instance_index
                    != history_transfer.source_instance_index
                ):
                    # joint（§2.2）：上一轮在异地执行时，到达/间隔 gate 在
                    # 上一执行实例的 rank 上，而 copy 的源端在 home。在线
                    # 模式下因果由"图批于准入时刻喂入"承载（NEW-1：发射器
                    # noc_migrate 分支从不消费 trigger_gate，R16-5 已删原
                    # 死参数实参——本分支为无门控发射，链式顺序兜底；若
                    # 未来接通离线/回放模式，接通 noc_migrate 的 trigger
                    # 消费是前置条件，见 PROVENANCE 登记）。
                    emit_transfer(history_transfer, "history_transfer")
                else:
                    emit_transfer(history_transfer,
                                  "history_transfer", gate=pending_gate)
            if copy_handoff_tail:
                # C13：交接尾块支链（fork 顺序 = 头块主链发射之后；分支
                # 不 join——readiness barrier 不等尾块，列车体逐块 arm 消
                # 费）。home 侧 send 链全速泵出，exec 侧 recv 链流式到达
                # （与 _emit_credit_stream_tail 同款拓扑）；ack 链尾随整
                # 条数据链（source release dependency 挂点）。
                def emit_copy_handoff_branch(
                        tail=copy_handoff_tail, gate=pending_gate):
                    self._emit_copy_handoff_tail(
                        request_plan, tail, pending_gate=gate)
                self._emit_side_branch(emit_copy_handoff_branch)

        # ---- prefill remote-read 前缀读流旁挂分支（规格书§三.1-§三.3，
        #      2026-09-25）：与后缀池恢复分支并行——同一准入 frontier
        #      分叉（本分支与 restore 分支均在 readiness barrier 前的
        #      主链 frontier 上 fork，分支互不 join、不加 history 主链
        #      节点），前缀层计算与后缀层恢复传输重叠（不设"全部传输
        #      完成才计算"全局 barrier；数据就绪由列车体层段的 per-rank
        #      recv 完成门保证）。触发链（§三.3）：gate 已在 gate 段结构
        #      性重建到本轮 prefill（exec）实例（R5 同语义），经
        #      TransferTriggerGate(control=exec) 走与 _emit_transfer_
        #      trigger() 相同的 1B relay——exec target timer → exec
        #      trigger send → home trigger recv → home KV send → exec
        #      KV recv；后续层组在 home send 链与 exec recv 链上顺序
        #      连接，不逐组重复消费 interval gate。
        if prefill_read_transfers:
            if joint_action != "remote-read":
                raise RuntimeError(
                    "prefill remote-read prefix stream requires "
                    f"joint_action 'remote-read' (got {joint_action!r}; "
                    f"request {request_plan['request_id']!r})")
            self._emit_prefill_remote_read_branch(
                request_plan, prefill_read_transfers, pending_gate)
        elif joint_action == "remote-read":
            # PLAN 形态破损 fail-closed：remote-read 必有前缀读流（LOCAL
            # 基全层 / PARTIAL 基前缀 [0, p)；plan_prefill_remote_read_
            # transfers 的适用性合同保证非空）。
            raise RuntimeError(
                "remote-read admission without its prefill prefix read "
                "stream (plan_prefill_remote_read_transfers must supply "
                f"it; request {request_plan['request_id']!r})")

        # ---- prefill_evictions（KV 逐出并行化 2026-09-13：旁路支链化，
        #      无触发门；fork 时可能存在主链 armed 依赖（turn-0 arrival
        #      gate 的 arm_timer_gate——R16-5 订正：local_hit 在通用循环
        #      先 continue、不 arm，原"local_hit 的 arm"表述失实），由
        #      _emit_side_branch 的 stash-and-clear 保真留给主链屏障消费）
        #      ----
        # R17-1a'（2026-09-17 死通道钉死）：joint 下本发射块条件恒假
        # ——prefill_evictions 结构性恒空（准入 R1' 预约覆盖全动作足迹、
        # drain expand gap≡0；2026-09-16 三方裁决），读但永不触发（净
        # 效果=死块）。保留不删（R16 §8 红线：删死只删实参/保守），本
        # 注释消除下一个"伪消费者"式误读；非 joint 调用面若未来启用该
        # 字段，此处语义照旧。
        if request_plan["prefill_evictions"]:
            watch_id = request_plan.get(
                "prefill_eviction_watch_id",
                "batch_train_evict_{}_admission_prefill".format(
                    request_plan["request_id"]))
            def emit_prefill_evictions() -> None:
                for transfer in request_plan["prefill_evictions"]:
                    record = emit_transfer(transfer, "prefill_evictions")
                    self._register_store_tails(transfer, record)
            members = self._emit_side_branch(
                emit_prefill_evictions,
                watch_context=(watch_id, "prefill", 0))
            if members:
                eviction_watches.append({
                    "request_id": watch_id,
                    "owner_request_id": request_plan["request_id"],
                    "branch": "admission_prefill",
                    "members": members})

        # ---- readiness barrier / partial 流水恢复 / C15 逐组恢复 ----
        restore_groups_partial = bool(
            restore_group_transfers
            and request_plan["history_location_before"] is not None
            and request_plan["history_location_before"].location
            == "partial_hbm_remote"
            and request_plan["history_location_before"].instance_index
            == request_plan["prefill_instance_index"])
        if restore_groups_partial:
            # C15（设计方案 §5.1）：后缀逐组恢复——同实例 partial 快速
            # 路径的同款拓扑（驻留前缀屏障 + checkpoint 分支）。P2-R2
            # （PARTIAL 跨实例 copy 流水化 2026-09-25）：恢复组并行化
            # ——组间无边、互并发（旧的"同 rank 下一组首节点链在上一组
            # 之后 = 物理串行"整段串行依赖移除），逐组逐 rank 目标 HBM
            # 写完成门登记 _suffix_restore_arms，由含首 chunk 的列车体
            # 按层段门控消费——**不设**"整段后缀恢复完才开始 prefill"
            # 的串行门（无 suffix p2p readiness barrier——逐 rank 门控
            # 替换整段栅栏，C12-G7/E17 登记的列车级保守门闭合）。
            history_before = request_plan["history_location_before"]
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
            # store→restore 前递（P2-R2 形态，①-④）：
            #   ① fail-closed 面与现状逐字节一致：先对组 0 区间做无匹配
            #      raise（_match_store_entries 历史文案；后缀形条目下组 0
            #      有匹配 ⇒ 各组至少与该条目相交）；
            #   ② 组循环前对并集区间 [min_start, max_end) 发射跨缘中继，
            #      每跨缘对恰一次（中继 recv 落在各 restore 缘 rank 的分
            #      支链上——组区域 checkpoint 捕获其后 previous_id ⇒ 每组
            #      首节点经链序排在全部并集匹配 store 完成之后）；
            #   ③ 每组：checkpoint → 同缘 arm（**只 arm 与本组层区间交
            #      集**的条目——交集判据 max(ls1,ls2)<min(le1,le2)，后缀
            #      条目与 end≤k 的组无交集）→ emit_transfer → 组门登记
            #      → restore_chain；组间无边，组 i 首节点依赖 = 同缘交集
            #      store（直接 arm）∪ 本缘共享中继（链序），不弱于旧
            #      "组 i 链在组 i-1 后"的传递性保障；
            #   ④ 组循环后同批消费并集匹配条目（span 外条目保留）——消
            #      费窗口回到 store 发射→restore 发射一轮内，
            #      pending_store_tails 在案窗口不放大（sh30
            #      merge_tail_gated 判据输入不变）。
            _session_id = restore_group_transfers[0].session_id
            self._match_store_entries(
                _session_id,
                restore_group_transfers[0].layer_start,
                restore_group_transfers[0].layer_end)
            _union_start = min(
                group.layer_start for group in restore_group_transfers)
            _union_end = max(
                group.layer_end for group in restore_group_transfers)
            group_matched, _group_retained = self._match_store_entries(
                _session_id, _union_start, _union_end)
            _group_restore_edges = []
            for group in restore_group_transfers:
                for shard in group.shards:
                    if (shard.edge_rank is not None
                            and shard.edge_rank not in _group_restore_edges):
                        _group_restore_edges.append(shard.edge_rank)
            self._emit_store_relays_once(
                group_matched, _group_restore_edges,
                f"{prefix}_history_transfer")
            group_arms = []
            for group_transfer in restore_group_transfers:
                involved_ranks = self._restore_group_involved_ranks(
                    group_transfer, prefill_group)
                group_checkpoints = {
                    rank: builders[rank].chain_checkpoint()
                    for rank in involved_ranks
                }
                for entry_edge, entry_store, _ack, entry_ls, entry_le in (
                        group_matched):
                    if (entry_edge in _group_restore_edges
                            and max(entry_ls, group_transfer.layer_start)
                            < min(entry_le, group_transfer.layer_end)):
                        builders[entry_edge].arm_dependency(entry_store)
                group_record = emit_transfer(
                    group_transfer, "history_transfer", gate=branch_gate)
                group_arms.append(self._restore_group_arm(
                    group_transfer, group_record, prefill_group))
                for rank in involved_ranks:
                    builders[rank].restore_chain(group_checkpoints[rank])
            self._consume_store_entries(_session_id, group_matched)
            for rank in prefill_group.ranks:
                builders[rank].restore_chain(checkpoints[rank])
            # 链回滚保留（恢复分支与本 rank 后续发射并行，与旧口径一致）。
            self._suffix_restore_arms[request_plan["request_id"]] = (
                tuple(group_arms))
        elif partial_history_restore:
            history_transfer = single_history_transfer
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
            # KV 逐出并行化（2026-09-13）：PARTIAL 后缀恢复支链入口的
            # store→restore 前递补边——arm 必须在 checkpoint 之后、回迁
            # 发射之前（分支区域内部），由回迁链在该 rank 的首节点消费
            # （PARTIAL 钉扎同实例，实测恒同缘直接 arm）。
            self._arm_pending_store_edges(
                history_transfer, f"{prefix}_history_transfer")
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
            if restore_group_transfers:
                # C15：跨实例后缀逐组恢复（copy/remote-read 工作副本的
                # 池恢复后缀）——主链迁移腿发射后旁挂支链逐组流水发射
                # （fork 点 = 主链迁移后的 frontier；分支不 join——
                # readiness barrier 只等主链，计算不等整段后缀恢复）。
                # P2-R2（2026-09-25）：组间并行化（组间无边、互并发）+
                # 每组交集直接 arm + 每跨缘对恰一次共享中继 + 并集条目
                # 同批消费（①-④ 同 stay 实例分支注释；消费窗口有界，
                # merge_tail_gated 判据输入不放大）。
                def emit_restore_group_branch(
                        groups=restore_group_transfers, gate=pending_gate):
                    session_id = groups[0].session_id
                    self._match_store_entries(
                        session_id, groups[0].layer_start,
                        groups[0].layer_end)
                    union_start = min(g.layer_start for g in groups)
                    union_end = max(g.layer_end for g in groups)
                    matched, _retained = self._match_store_entries(
                        session_id, union_start, union_end)
                    restore_edges = []
                    for group in groups:
                        for shard in group.shards:
                            if (shard.edge_rank is not None
                                    and shard.edge_rank
                                    not in restore_edges):
                                restore_edges.append(shard.edge_rank)
                    self._emit_store_relays_once(
                        matched, restore_edges,
                        f"{prefix}_history_transfer")
                    group_arms = []
                    for group_transfer in groups:
                        involved_ranks = self._restore_group_involved_ranks(
                            group_transfer, prefill_group)
                        group_checkpoints = {
                            rank: self.builders[rank].chain_checkpoint()
                            for rank in involved_ranks
                        }
                        for (entry_edge, entry_store, _ack, entry_ls,
                             entry_le) in matched:
                            if (entry_edge in restore_edges
                                    and max(entry_ls,
                                            group_transfer.layer_start)
                                    < min(entry_le,
                                          group_transfer.layer_end)):
                                self.builders[entry_edge].arm_dependency(
                                    entry_store)
                        group_record = emit_transfer(
                            group_transfer, "history_transfer", gate=gate)
                        group_arms.append(self._restore_group_arm(
                            group_transfer, group_record, prefill_group))
                        for rank in involved_ranks:
                            self.builders[rank].restore_chain(
                                group_checkpoints[rank])
                    self._consume_store_entries(session_id, matched)
                    self._suffix_restore_arms[
                        request_plan["request_id"]] = tuple(group_arms)
                self._emit_side_branch(emit_restore_group_branch)
            _emit_tp_readiness_barrier(
                builders=builders, group=prefill_group,
                name=f"{prefix}_prefill_kv_ready_barrier")
        # prefill 主体（chunk spans + end barrier + seg1 块末）自拼 batch
        # 改造（2026-08-22）起移入 emit_iteration_train 的折叠体与 drain
        # 标记；此处止于准入动作（到达 gates/历史迁移/逐出/屏障）。
        return eviction_watches

    def _restore_group_arm(self, group_transfer, group_record, prefill_group):
        """C15：单组恢复腿的逐 rank 就绪门（目标 HBM 写完成节点）。

        逐 rank 门控（非 TP 栅栏）：rank r 的列车体层段只等 rank r 的
        该组写完成——与预测器的逐 rank 恢复链口径一致；零字节 rank 无
        shard 即无门（无数据即无到达可等）。组序/层区间取 transfer 值。
        """
        gates_by_rank = {}
        for shard_record in group_record["shards"]:
            target_rank = shard_record.get("target_rank")
            completion_node = shard_record.get("target_hbm_completion_node_id")
            if not isinstance(target_rank, int) or not isinstance(
                    completion_node, int):
                raise RuntimeError(
                    "restore group is missing a target HBM gate")
            gates_by_rank[target_rank] = completion_node
        if not set(gates_by_rank) <= set(prefill_group.ranks):
            raise RuntimeError(
                "restore group gates leaked outside the Prefill instance")
        return (
            int(getattr(group_transfer, "restore_group")),
            int(group_transfer.layer_start),
            int(group_transfer.layer_end),
            gates_by_rank,
        )

    def _emit_prefill_remote_read_branch(
            self, request_plan: dict, groups, pending_gate) -> None:
        """prefill remote-read 前缀读流旁挂分支（规格书§三.1-§三.4，
        2026-09-25）的发射与账本登记。经 _emit_side_branch 包裹——分支
        首节点 parent = 准入 frontier，分支不 join：readiness barrier
        只作结构性主链节点（§三.7），前缀数据就绪由列车体层段消费本
        分支登记的 recv 完成门保证（emit_layer_segmented）。

        形态：
        - 组切分复用 KV 管理器 plan_layer_groups（RESTORE_GROUP_LAYERS
          同款，§三.4）——组区间由 plan 传入，本方法复检 [0, p) 自 0
          连续铺满（缺口/重叠即 raise，与列车体铺满检查同款风格）；
        - 触发链恰消费一次 interval/arrival gate（§三.3）：gate 已在
          gate 段结构性重建到本轮 prefill（exec）实例（R5 同语义），
          构造 TransferTriggerGate(control=exec, node_gates=重建 timer
          gates)，每个有数据的相对位 rank 对走与 _emit_transfer_trigger
          () 相同的 1B relay——exec target timer → exec trigger send →
          home trigger recv → home KV send → exec KV recv；
        - 后续组不重复消费 interval gate（§三.3）：组 g+1 的 home send
          / exec recv 经 builder 链序顺序链接组 g 的 send/recv 链（发射
          序 = 链序）；
        - 每组 exec recv 完成门登记 _prefill_remote_read_arms[rid][组号]
          [target_rank]、组层区间登记 _prefill_remote_read_layers[rid]
          [组号]（§三.5）——列车体按层段门控消费，残留 fail-closed。
        """
        request_id = request_plan["request_id"]
        exec_instance = request_plan["prefill_instance_index"]
        home_instance = groups[0].source_instance_index
        if request_id in self._prefill_remote_read_arms:
            raise RuntimeError(
                "prefill remote-read arm ledger for {} already holds "
                "groups {} (admission emission must register exactly "
                "once)".format(
                    request_id,
                    sorted(self._prefill_remote_read_arms[request_id])))
        history_before = request_plan.get("history_location_before")
        if history_before is None:
            raise RuntimeError(
                "prefill remote-read requires a history location "
                f"snapshot (request {request_id!r})")
        if history_before.instance_index != home_instance:
            raise RuntimeError(
                "prefill remote-read prefix stream home instance "
                f"{home_instance} does not match the history location "
                f"instance {history_before.instance_index} (request "
                f"{request_id!r})")
        if history_before.location == "local_hbm":
            expected_prefix_end = self.config.layers
        elif history_before.location == "partial_hbm_remote":
            expected_prefix_end = history_before.resident_prefix_layers
        else:
            raise RuntimeError(
                "prefill remote-read requires a LOCAL/PARTIAL base "
                "history location (got "
                f"{history_before.location!r}; request {request_id!r})")
        exec_group = self.group_by_index[exec_instance]
        home_group = self.group_by_index[home_instance]
        if len(exec_group.ranks) != len(home_group.ranks):
            raise RuntimeError(
                "prefill remote-read requires equal TP sizes")
        expected_start = 0
        for group_index, transfer in enumerate(groups):
            if transfer.kind != "noc_migrate":
                raise RuntimeError(
                    "prefill remote-read prefix group must be a "
                    f"noc_migrate (group {group_index} got "
                    f"{transfer.kind!r})")
            if (transfer.source_instance_index != home_instance
                    or transfer.target_instance_index != exec_instance):
                raise RuntimeError(
                    "prefill remote-read prefix group endpoints do not "
                    f"match home/exec instances (group {group_index}: "
                    f"{transfer.source_instance_index} -> "
                    f"{transfer.target_instance_index}, expected "
                    f"{home_instance} -> {exec_instance})")
            if (transfer.session_id != request_plan["session_id"]
                    or transfer.trigger_request_id != request_id):
                raise RuntimeError(
                    "prefill remote-read prefix group belongs to another "
                    f"request/session (group {group_index}: "
                    f"{transfer.trigger_request_id!r}/"
                    f"{transfer.session_id!r})")
            if not transfer.stream_only:
                raise RuntimeError(
                    "prefill remote-read prefix group must be a transient "
                    f"stream (stream_only=True; group {group_index})")
            if (transfer.layer_start != expected_start
                    or transfer.layer_end <= transfer.layer_start):
                raise RuntimeError(
                    "prefill remote-read prefix groups are not contiguous "
                    f"from layer 0 (group {group_index} "
                    f"[{transfer.layer_start}, {transfer.layer_end}), "
                    f"expected start {expected_start})")
            expected_start = transfer.layer_end
            for shard in transfer.shards:
                if (home_group.ranks.index(shard.source_rank)
                        != exec_group.ranks.index(shard.target_rank)):
                    raise RuntimeError(
                        "prefill remote-read shard does not preserve "
                        "relative TP rank (home rank "
                        f"{shard.source_rank} -> exec rank "
                        f"{shard.target_rank})")
        if groups[-1].layer_end != expected_prefix_end:
            raise RuntimeError(
                "prefill remote-read prefix groups do not tile the "
                f"resident prefix (last end {groups[-1].layer_end} != "
                f"{expected_prefix_end}; request {request_id!r})")
        if pending_gate.source_instance_index != exec_instance:
            raise RuntimeError(
                "prefill remote-read trigger gate is not on the prefill "
                f"instance (gate source {pending_gate.source_instance_index},"
                f" prefill {exec_instance}; the interval/arrival gate "
                "rebuild must precede the prefix read stream)")
        prefix = _prefix_of(request_plan)
        action_state = self._action_seq.setdefault(request_id, [0])
        # §三.3：跨实例触发门构造——control = 本轮 prefill（exec）实例，
        # node_gates = 重建后的 exec 实例 timer gates（同实例退化时即原
        # arrival gate；exec != home 恒成立，1B relay 恒走跨实例路径）。
        trigger_gate = TransferTriggerGate(
            control_instance_index=exec_instance,
            node_gates=pending_gate.timer_gates,
        )

        def emit_prefill_read_stream() -> None:
            # ① 触发链（恰一次消费 interval/arrival gate）：每个出现数
            #    据 shard 的相对位 rank 对，exec timer gate arm 后经
            #    _emit_transfer_trigger 同款 1B relay 触发 home（home
            #    trigger recv 落在 home builder 链上 ⇒ 组 0 send 链序
            #    后随；exec trigger send 落在 exec builder 链上 ⇒ 组 0
            #    recv 链序后随）。零字节相对位无数据可触发，不 arm。
            data_relative_indexes = set()
            for transfer in groups:
                for shard in transfer.shards:
                    data_relative_indexes.add(
                        home_group.ranks.index(shard.source_rank))
            for relative_index in sorted(data_relative_indexes):
                _emit_transfer_trigger(
                    builders=self.builders,
                    group_by_index=self.group_by_index,
                    tag_allocator=self.tag_allocator,
                    trigger_gate=trigger_gate,
                    relative_index=relative_index,
                    source_rank=home_group.ranks[relative_index],
                    action_name=f"{prefix}_prefill_read",
                    shard_index=relative_index,
                )
            # ② 组序列（层自低向高 = 消费顺序）：每组逐 shard home KV
            #    send → exec KV recv（完成门）→ exec ack send → home ack
            #    recv；组间靠 home/exec builder 链序顺序连接（home send
            #    链 / exec recv 链），不逐组重复消费 interval gate。
            arms_by_group = {}
            layers_by_group = {}
            for group_index, transfer in enumerate(groups):
                action_sequence = action_state[0]
                action_name = (
                    f"{prefix}_prefill_read_action{action_sequence:03d}_"
                    f"{sanitize_node_prefix(transfer.session_id)}_"
                    f"{transfer.kind}")
                action_state[0] = action_sequence + 1
                per_rank = {}
                for shard_index, shard in enumerate(transfer.shards):
                    data_tag = self.tag_allocator.take()
                    self.builders[shard.source_rank].comm_send(
                        f"{action_name}_shard{shard_index}_send",
                        src=shard.source_rank, dst=shard.target_rank,
                        comm_size=shard.bytes, comm_tag=data_tag)
                    self.builders[shard.target_rank].comm_recv(
                        f"{action_name}_shard{shard_index}_recv",
                        src=shard.source_rank, dst=shard.target_rank,
                        comm_size=shard.bytes, comm_tag=data_tag)
                    recv_node = self.builders[shard.target_rank].previous_id
                    if recv_node is None:
                        raise RuntimeError(
                            "prefill remote-read recv did not generate "
                            "a node")
                    per_rank[shard.target_rank] = recv_node
                    ack_tag = self.tag_allocator.take()
                    self.builders[shard.target_rank].comm_send(
                        f"{action_name}_shard{shard_index}"
                        f"_ack_to_rank{shard.source_rank}",
                        src=shard.target_rank, dst=shard.source_rank,
                        comm_size=1, comm_tag=ack_tag)
                    self.builders[shard.source_rank].comm_recv(
                        f"{action_name}_shard{shard_index}"
                        f"_ack_from_rank{shard.target_rank}",
                        src=shard.target_rank, dst=shard.source_rank,
                        comm_size=1, comm_tag=ack_tag)
                arms_by_group[group_index] = per_rank
                layers_by_group[group_index] = (
                    transfer.layer_start, transfer.layer_end)
            self._prefill_remote_read_arms[request_id] = arms_by_group
            self._prefill_remote_read_layers[request_id] = layers_by_group

        self._emit_side_branch(emit_prefill_read_stream)

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

        # joint merge 回传流（§2.2/§3.2）：compute_done 后的新增量归并
        # origin_home 的真实传输（noc 回传 + 池写回），触发门 = 退出迭代
        # 的 seg2 块末；store 尾部登记入 pending_store_tails——下一轮对本
        # 会话的池读/回迁经前递补边等待写入完成（合并成本进入下一轮
        # 数据等待；所有动作/八组合一致处理）。
        merge_transfers = list(request_plan.get("merge_transfers") or ())
        for transfer in merge_transfers:
            record = emit_transfer(transfer, "merge_transfers")
            self._register_store_tails(transfer, record)
        # R11(ii)：merge 尾标记（每 decode rank 1 个小 COMP 节点，链在
        # 本 rank 的 merge 传输链之后）——下一轮 interval gate 的前递
        # 依赖（消除"下一轮物理消费尚未落地的 KV"）与 merge-done watch
        # 的成员锚点。
        merge_done_members = {}
        if merge_transfers:
            # 标记节点用 watch 命名空间上下文（batch_train_merge_<rid> /
            # prefill / 0——C++ 提交预检要求 watch 成员节点的
            # request/stage/generation 与注册一致，哨兵标记同款）。
            watch_context_id = (
                "batch_train_merge_" + request_plan["request_id"])
            for rank in decode_group.ranks:
                self.builders[rank].set_context(
                    watch_context_id, "prefill", 0)
                self.builders[rank].comp(
                    f"{prefix}_merge_done_rank{rank}", 1, 1)
                merge_done_members[rank] = self.builders[rank].previous_id

        for transfer in request_plan["completion_evictions"]:
            emit_transfer(transfer, "completion_evictions")

        request_id = request_plan["request_id"]
        try:
            # completion 是 next_plan 唯一消费者；即使值为 terminal None
            # 也必须移除，避免预建的全量 request 链永久驻留。
            following = self.next_plan.pop(request_id)
        except KeyError as exc:
            raise RuntimeError(
                "completion request has no next-plan entry: {!r}".format(
                    request_id)) from exc
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
            # R11(ii)：下一轮 interval gate 前递依赖 merge 尾标记（本会话
            # 增量落地之前下一轮不得物理消费）；无 merge 流时保持 seg2
            # 块末（与改造前一致）。
            gate_anchor_by_rank = (
                merge_done_members if merge_done_members else
                {rank: decode_completion_nodes[relative_index]
                 for relative_index, rank in enumerate(decode_group.ranks)}
            )
            timers = tuple(
                builders[rank].timer_gate(
                    f"q{following['queue_index']:04d}_"
                    f"{sanitize_node_prefix(following['request_id'])}_"
                    f"history_rank{rank}_interval_gate",
                    # §9-低6：同 K5 族——interval + hbm_wait 任意 ns 粒度
                    # 须 µs 下取整（timer_gate 离线同构校验；现行输入恰在
                    # µs 网格上惰性通过，非 µs 对齐输入会确定性崩溃）。
                    (interval + following.get("hbm_wait_ns", 0)) // 1000 * 1000,
                    after_node_id=gate_anchor_by_rank[rank],
                )
                for rank in decode_group.ranks
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
            # KV 逐出并行化（2026-09-13）：终态会话清 store 尾部登记
            # （不再有回迁读者；等价于 kv_manager.retire_terminal_session
            # 的登记侧镜像，改动面约束下落点在构图器）。完成路径零逐出
            # 维持，不发任何传输。残余：终结完成之后才被逐出的"永不再
            # 来"会话条目驻留至 run 结束（O(会话数) 上界，无正确性影响）。
            self.pending_store_tails.pop(request_plan["session_id"], None)
        return {
            "merge_done_members": merge_done_members,
            "has_merge": bool(merge_done_members),
        }

    # ------------------------------------------------------------- 属性 --

    def set_next_plan(self, next_plan: dict) -> None:
        """request_id -> 下一 turn 的 request_id / plan dict / None。

        completion 发射是每项唯一消费点：对应 key 将在其 interval gate
        发射后 pop；terminal None 同样消费，以保持账本随在途窗口有界。
        """
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
