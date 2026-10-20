#!/usr/bin/env python3
"""wsc_llm_online_scheduler.py -- 关感知策略调度器(strategy 模式,步骤 1-9)。

自离线事件循环蓝本迁移(该离线蓝本已随 2026-09-24 只在线化从
wsc_llm_scheduler.py 删除,该文件现仅保留拓扑/静态路由/排队语义),
保持决策顺序逐段对应(迁移处用 `# offline: wsc_llm_scheduler.py` 注释
标注;离线蓝本删减后原行号锚点已失效——指向旧蓝本的行内 `# :NNNN`
历史锚点已于 2026-09-30 全部清除,一律以阶段/函数名对应)。
离线事件循环与在线边界的一一对应:

  离线事件循环                                    在线边界
  ----------------                                ----------------
  预置 arrival 堆                                 ingress ARRIVAL 事件喂入同一
                                                  arrival heap(字段保持
                                                  (time_ns, priority, sequence,
                                                  kind, payload) 形状,同 tick
                                                  排序规则逐字节同源)
  completion 批(completion 分支)                  PREFILL_DRAIN / DECODE_COMPLETION
                                                  / REQUEST_COMPLETE 事件处理:
    prefill 完成                                    _on_prefill_drain:
      qp.popleft / waiting_decode_admissions          busy=False + qp.pop +
      append / dirty                                  等待 decode 准入登记
    decode 完成                                     _on_decode_complete:
      mark_complete + note_capacity_change            mark_complete + snapshots
      + 下一次 arrival 排程                           (下一次 arrival 排程在
                                                       REQUEST_COMPLETE 边界做,
                                                       与离线同 tick 同顺序)
  arrival 批                                       _on_arrival(经 arrival heap):
    select_prefill_instance + route_for_prefill      快照 -> 选择 -> 静态路由
    + qp.append                                      -> qp.append(顺序一致)
  start_ready_iterations                           _admit_pass(同 tick 末尾):
    try_admit_waiting_decodes                         decode 准入(容量 epoch/dirty
                                                      门控,原样保留)
    逐实例 serve                                      per-instance 发射:
      prefill: try_admit_prefill(qp[0]) + 发射         try_admit_prefill + 发射
      decode:  发射 active_decode 整批                 prefill 整段 / decode 整段
    **计时部分删除**:离线用 LUT 估计时长 push
    iteration_complete 事件;在线由真实完成事件
    (C++ 物理时钟)推进,排队/配对逻辑原样保留于
    账本/准入逻辑,不体现在图结构上(构图粒度与
    离线一致:prefill 整段 + decode 整段)。

决策顺序(line-by-line):每 tick 先处理 completion 批,再处理 arrival 批,最后
跑准入/发射 pass——与离线同 tick 批序(priority 0 completion < 1 arrival)
一致。

关感知口径(方案 §4.1 第 6 条):策略输入全部来自 Python 账本——prefill 排队
深度(len(qp))、KV 容量(kv_manager 账本 + capacity_epoch)、静态路由
(build_static_pd_mapping);不新增任何 C++ 状态读取。C++ 物理(真实完成
tick、网络竞争)在阶段 1 即真实生效,只负责推进完成边界。

与离线蓝图的刻意差异(real-online 语义,合同⑦ Tier B real-online 验收):
  - 计时/迭代粒度:离线 LUT 时钟 + 逐 chunk 迭代 -> 在线真实完成事件 +
    request-aggregated 构图(prefill 整段 + decode 整段,与历史规划的工作量
    粒度一致,方案 §4 步骤 1-9 操作 1);
  - 实例 busy 语义:离线 prefill 实例 busy 覆盖逐 chunk 迭代,decode 实例
    busy 覆盖整批迭代;在线 busy 覆盖"一个 prefill/decode 整段在飞";
  - decode 段发射:离线整批 active_decode 一次迭代;在线一次发射
    active_decode 队首整段,完成后再发射下一个(per-rank 物理链天然
    串行化同实例 decode 段；strategy 模式保持动态跨 request 链，
    不适用 replay 的 LUT 时钟裁决);
  - 完成顺序:真实完成 tick 决定(网络竞争、物理链),不要求与离线决策
    序列 exact(合同⑦:real-online 只验不变量与差异可解释性)。

拼 batch 改造(2026-08-22,设计文档《层次 B Continuous Batching 改造》
§3.2 + §3.6 wscllm PD 分离豁免):D 侧 decode 发射从"每请求整段"重构为
"实例迭代列车"——decode 互拼(一个迭代同时算 B 个成员各 1 token,权重每
迭代只读一次,weight_passes=迭代数),批成员只在列车边界变化;列车内绝不
混入 prefill chunk span(§3.6:P 侧逐 chunk 不拼,保持 _emit_prefill 现有
整段骨架——聚合调用 weight_passes 缺省 = len(spans) = chunk 数,口径
不变)。每 D 实例状态机:active_decode(批成员表)/in_flight_train(唯一
在飞列车,冻结成员快照——成员/参与度在列车边界冻结,快照即
plan dict;busy 门 = 一个列车在飞)/
finalized_trains(幂等核销账本:同列车 exit 标记 watch 跨 tick fire、
事件拆交付,首个信号核销恰一次,后续信号清 pending 集合)。列车终点 =
全部 decode 工作耗尽(无 prefill 工作 ⇒ 迭代 = max 剩余 token;退出
成员在列车内挂 exit 标记,退出不是列车边界)。决策边界仍是四类
reason(ARRIVAL/PREFILL_DRAIN/DECODE_COMPLETION/REQUEST_COMPLETE),
D 侧由列车 exit 标记节点的 watch 驱动;P 侧 PREFILL_DRAIN 不变。
"""

import heapq
import os
import sys
from collections import deque

# --------------------------------------------------------------------------
# import 路径:本文件位于 workload/llama2_7b_inference/online/,共享配置、
# 发射与调度模块在上一级。路径只做 import 用途(红线:generate_wsc_llm_trace.py / wsc_llm_
# scheduler.py / session_kv_manager.py 只读 import 与注释)。
# --------------------------------------------------------------------------
_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _path in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from online.graph_batch_builder import first_token_split_enabled  # noqa: E402
from online.online_scheduler_base import (  # noqa: E402
    STAGE_DECODE,
    STAGE_PREFILL,
    STAGE_REQUEST,
    OnlineSchedulerBase,
)
from session_kv_manager import (  # noqa: E402
    LOCAL_HBM,
    SessionKVCacheManager,
    kv_cache_shard_bytes_for_tokens,
)
from wsc_llm_scheduler import (  # noqa: E402
    DECODE_ROLE,
    PREFILL_ROLE,
    InstanceGraph,
    PrefillQueueSnapshot,
    WscLlmInstanceSpec,
    build_instances,
    build_static_pd_mapping,
    select_prefill_instance,
)


class _OnlineInstanceState:
    """在线实例账本(离线 _InstanceRuntime,wsc_llm_scheduler.py 的
    在线子集):qp = prefill FCFS 队列(deque),active_decode = 已准入 decode
    列表。busy 语义按 phase_role 分裂(拼 batch 改造,§3.6):
      - PREFILL_ROLE:busy = 一个 prefill 整段在飞(保持原骨架,P 侧
        逐 chunk 不拼);
      - DECODE_ROLE:busy 门 = 一个迭代列车在飞(in_flight_train);
        train_seq/finalized_trains 为列车账本。
    """

    __slots__ = ("index", "phase_role", "qp", "active_decode",
                 "active_decode_lookup", "busy",
                 "in_flight_train", "finalized_trains",
                 "train_seq", "first_step_remainder")

    def __init__(self, *, index: int, phase_role: str) -> None:
        self.index = index
        self.phase_role = phase_role
        # 阶段 4 §7.3:deque 替换 list.pop(0)(O(N) 首出 -> O(1)popleft)。
        self.qp = deque()  # 顺序即 FCFS(append 尾入,popleft 首出)
        self.active_decode = []
        # §7.3:完成事件直接定位(active_decode 成员判定/移除 O(N) -> O(1);
        # 与列表保持同步,append/remove 双侧更新)。
        self.active_decode_lookup = set()
        self.busy = False
        # ---- 拼 batch 列车账本(2026-08-22,D 实例;P 实例不使用) ----
        self.in_flight_train = None      # 唯一在飞列车(冻结成员快照)
        self.finalized_trains = []       # 已核销列车(待收后续跨交付信号)
        self.train_seq = 0               # 列车序号(命名/审计用)
        # WP9 首步批拆分(2026-08-26):两段式发射的余量批挂起(首步唤醒
        # 到达后的首个决策边界发射;None = 无待发余量批)。
        self.first_step_remainder = None


class _OnlineRequestRuntime:
    """在线请求运行账本(离线 _RequestRuntime 的在线子集)。

    输入事实(request-neutral,来自 manifest,policy-independent):
      history_tokens_before / prefill_context_tokens / final_context_tokens /
      prefill_length / decode_length / queue_index / session_id / turn_index。
    运行期事实(在线决策产出的字段,语义与离线蓝图同名同义)。
    """

    __slots__ = (
        "request_id", "session_id", "turn_index", "queue_index",
        "prefill_length", "decode_length",
        "history_tokens_before", "prefill_context_tokens", "final_context_tokens",
        "estimated_arrival_ns",
        "prefill_instance_index", "prefill_assignment_key",
        "decode_instance_index", "static_route",
        "admitted_prefill", "prefill_attempt_epoch", "decode_capacity_reserved",
        "history_cache_state_before",
        "history_action", "history_source_instance_index",
        "history_location_before", "history_resident_prefix_layers",
        "history_transfer_bytes", "history_transfers",
        "history_recompute_tokens",
        "admission_evictions", "decode_target_evictions",
        "decode_target_evictions_prefill_snapshot",
        "history_evictions", "prefill_evictions",
        "decode_evictions_pending",
        "reservation_credit",
        "decode_queue_depth_before_enqueue",
        "waiting_decode_admission", "prefill_decode_transfer",
        "completion_evictions", "kv_state_after_completion",
        "kv_instance_after_completion",
        "decode_tokens_consumed", "decode_train_joined",
    )

    def __init__(self, record: dict) -> None:
        self.request_id = record["request_id"]
        self.session_id = record["session_id"]
        self.turn_index = record["turn_index"]
        self.queue_index = record["queue_index"]
        self.prefill_length = record["prefill_length"]
        self.decode_length = record["decode_length"]
        self.history_tokens_before = record["history_tokens_before"]
        self.prefill_context_tokens = record["prefill_context_tokens"]
        self.final_context_tokens = record["final_context_tokens"]
        self.estimated_arrival_ns = None
        self.prefill_instance_index = None
        self.prefill_assignment_key = None
        self.decode_instance_index = None
        self.static_route = None
        self.admitted_prefill = False
        self.prefill_attempt_epoch = None
        self.decode_capacity_reserved = False
        self.history_cache_state_before = None
        self.history_action = None
        self.history_source_instance_index = None
        # B2(位置二态):准入时点的会话历史位置快照(契约 §3 的
        # history_location_before / history_resident_prefix_layers 序列化源)。
        self.history_location_before = None
        self.history_resident_prefix_layers = None
        self.history_transfer_bytes = None
        # B2(位置二态):history 恢复/迁移的 KVTransfer 留存(holder 模式镜像
        # decode_target_evictions——准入时赋值、prefill 决策序列化、M4
        # 核销置空;LOCAL_HIT/NO_HISTORY 恒空元组,REMOTE_RESTORE 全量
        # 恢复为单段)。
        self.history_transfers = ()
        # RECOMPUTE 删除(B2):恒 0,字段保留供冻结 schema 序列化。
        self.history_recompute_tokens = 0
        self.admission_evictions = ()
        self.decode_target_evictions = ()
        # 结构性防双计守卫的发射端单点记录(2026-10-01 审查修正):
        # prefill 决策行在 _emit_prefill 序列化 decode_target_evictions
        # 时点的快照(纯内部账本,不对外发射);_decode_decision_dict 据
        # 此 fail-closed 断言 decode 行快照 ⊇ prefill 行快照(两行同键
        # 双列为冻结 schema 既定形状,防双计纪律由本断言钉死结构前提)。
        self.decode_target_evictions_prefill_snapshot = ()
        # B3 分段发射(2026-09-06):admission_evictions 的两个组成段分账
        # (history = prepare_history 的 fit 逐出;prefill = grow_prefill
        # 的增长逐出)——发射触发门与决策日志契约 §3 字段(history_
        # evictions/prefill_evictions)按段取用;union 账本保留(2026-09-30
        # 审查澄清:消费面仅决策日志序列化 + watermark 旧映射口径,无纪
        # 元接线——纪元与发射门直接读下面两个分段账本)。
        self.history_evictions = ()
        self.prefill_evictions = ()
        # C-2/M-25 修复(2026-09-30):decode 侧逐出的"已决策未发射"子账本
        # (#2 预占逐出经 _flush_decode_evictions 在决策时点就近发射后清空;
        # #4/#5 move/grow 逐出保留至 joiner 列车头发射)。决策日志账本
        # decode_target_evictions 保留全集不变。
        self.decode_evictions_pending = ()
        # 准入预占净额修正(2026-09-06):全量预约失败改净额重试成功时记录
        # 旧驻留 credit(tuple,供迁移后 extend 回补差值);None = 全量路径
        # 或已回补/已核销。
        self.reservation_credit = None
        self.decode_queue_depth_before_enqueue = None
        self.waiting_decode_admission = False
        self.prefill_decode_transfer = None
        self.completion_evictions = ()
        self.kv_state_after_completion = None
        self.kv_instance_after_completion = None
        # ---- 拼 batch 列车推进字段(2026-08-22;列车核销时闭式推进,
        # 余额与逐 token 精确值逐点一致) ----
        self.decode_tokens_consumed = 0  # 已物理完成 decode token 数
        # 是否已加入过某趟列车(首个列车随发射其 transfer 3000 迁移;
        # 后续列车不再重复迁移)。
        self.decode_train_joined = False


def _kv_event_dict(event) -> dict:
    """Serialize one KVCacheEvent for the online kv_actions stream
    (与 _eviction_record_dict / _kv_transfer_dict 同构的在线序列化)。"""
    return {
        "event_index": event.event_index,
        "planner_time_ns": event.planner_time_ns,
        "phase": event.phase,
        "event_type": event.event_type,
        "reason": event.reason,
        "trigger_request_id": event.trigger_request_id,
        "session_id": event.session_id,
        "source_instance_index": event.source_instance_index,
        "target_instance_index": event.target_instance_index,
        "context_tokens": event.context_tokens,
        "total_bytes": event.total_bytes,
        "shard_bytes": list(event.shard_bytes),
        "last_completion_ns": event.last_completion_ns,
        "instance_remaining_before_bytes": list(
            event.instance_remaining_before_bytes),
        "instance_remaining_after_bytes": list(
            event.instance_remaining_after_bytes),
        "insufficient_ranks": list(event.insufficient_ranks),
    }


def _kv_transfer_rows(transfers) -> list:
    """B2(2026-09):KVTransfer 逐条序列化(决策日志行结构,契约 §3
    _transfer_entry_rows 同款:kind/reason/session_id/total_bytes/
    source_instance_index/target_instance_index/layer_start/layer_end)。

    逐出(remote_store)与恢复(remote_load/noc_migrate)共用同一行结构;
    history_transfers 列表承载恢复/迁移的逐对象记录(一条 prefill
    决策记录,不拆两条)。纯输出字段,进 ON/OFF 对拍剥离清单。"""
    rows = []
    for transfer in transfers or ():
        if transfer is None:
            continue
        rows.append({
            "kind": transfer.kind,
            "reason": transfer.reason,
            "session_id": transfer.session_id,
            "total_bytes": transfer.total_bytes,
            "source_instance_index": transfer.source_instance_index,
            "target_instance_index": transfer.target_instance_index,
            "layer_start": transfer.layer_start,
            "layer_end": transfer.layer_end,
        })
    return rows


def _kv_transfer_shard_rows(transfer) -> list:
    """B2:单个 KVTransfer 的 shard 级序列化(含构造期已定的 XY 路由)。"""
    rows = []
    for shard in transfer.shards:
        rows.append({
            "source_rank": shard.source_rank,
            "target_rank": shard.target_rank,
            "edge_rank": shard.edge_rank,
            "bytes": shard.bytes,
            "noc_path": list(shard.noc_path),
            "noc_hops": max(0, len(shard.noc_path) - 1),
            "layer_start": shard.layer_start,
            "layer_end": shard.layer_end,
        })
    return rows


def _kv_transfer_dict(transfer):
    """B2:KVTransfer 对象序列化(None 透传;shards 逐项 noc_path)。"""
    if transfer is None:
        return None
    return {
        "kind": transfer.kind,
        "phase": transfer.phase,
        "reason": transfer.reason,
        "session_id": transfer.session_id,
        "trigger_request_id": transfer.trigger_request_id,
        "source_instance_index": transfer.source_instance_index,
        "target_instance_index": transfer.target_instance_index,
        "total_bytes": transfer.total_bytes,
        "model_layers": transfer.model_layers,
        "layer_start": transfer.layer_start,
        "layer_end": transfer.layer_end,
        "resident_prefix_layers_before": transfer.resident_prefix_layers_before,
        "resident_prefix_layers_after": transfer.resident_prefix_layers_after,
        "shards": _kv_transfer_shard_rows(transfer),
    }


def _eviction_source_instances(evictions) -> tuple:
    """B2:逐出 KVTransfer 的受害实例索引元组(纪元唤醒接线用;remote_store
    的 source_instance_index 即被逐会话原驻留实例)。"""
    return tuple(
        transfer.source_instance_index
        for transfer in evictions or ()
        if transfer is not None
    )


class WscLlmOnlineScheduler(OnlineSchedulerBase):
    """strategy 变体:真实策略(关感知)在在线骨架中运行。

    离线事件循环蓝本已随 2026-09-24 只在线化删除,本类是在线策略的
    唯一实现。kv_cache_policy 值域:"session_lru_tiered"(session 级
    二态冷热管理:整体 LRU 逐出 + 远端池全量恢复;行为上新 KV 管理器
    唯一,无档位分支)。拓扑 / 静态路由 / KV 账本在 __init__ 一次性构建,
    运行期策略输入全部来自这些 Python 账本(关感知)。
    """

    def __init__(self, *, manifest, config, graph, digest_sink=None,
                 decision_log_sink=None, train_ledger_sink=None,
                 profile_sink=None, mode: str = "strategy",
                 sensing: bool = False,
                 defensive_reply_cache: bool = False,
                 sensing_query_sink=None, online_stats_sink=None,
                 ledger_sink=None, workspace_admission: str = "off"):
        super().__init__(
            manifest=manifest,
            config=config,
            digest_sink=digest_sink,
            sensing=sensing,
            decision_log_sink=decision_log_sink,
            profile_sink=profile_sink,
            defensive_reply_cache=defensive_reply_cache,
            sensing_query_sink=sensing_query_sink,
            online_stats_sink=online_stats_sink,
            ledger_sink=ledger_sink,
            workspace_admission=workspace_admission,
        )
        if mode != "strategy":
            raise ValueError("WscLlmOnlineScheduler requires mode == 'strategy'")
        if config.kv_cache_policy != "session_lru_tiered":
            raise ValueError(
                "strategy scheduler supports kv_cache_policy "
                "'session_lru_tiered', got {!r}".format(
                    config.kv_cache_policy))
        self.graph = graph  # GraphBatchBuilder(与 replay 路径共用)
        # C-2/M-25 fail-closed(2026-09-30):注入"已决策未发射 remote_store"
        # 探针——回迁补边空表时构图器回查调度器,命中即结构性时序违例
        # (见 graph_batch_builder._arm_store_tail_dependencies)。
        self.graph.set_pending_remote_store_probe(
            self._has_pending_decode_store_evictions)

        # 蓝图（离线蓝本，按阶段/函数名对应）:拓扑 + 静态 PD 路由(离线同参;alpha 调参旋钮与
        # alpha/adjusted_transfer_cost 只写字段已按深挖文档只写不读族清除,
        # 2026-09-24——该参数从不影响路由选择)。
        # offline: wsc_llm_scheduler.py
        specs = tuple(
            WscLlmInstanceSpec(
                name=group.name,
                pg_name=group.pg_name,
                ranks=group.ranks,
                phase_role=group.phase_role,
            )
            for group in config.inference_groups
        )
        self.topology = build_instances(config.hardware, specs)
        self._workspace_rank_to_instance = {
            rank: instance.index
            for instance in self.topology.instances
            for rank in instance.ranks
        }
        self.static_mapping = build_static_pd_mapping(self.topology)
        # Hop-Bytes 覆盖调查(2026-08-26,W B2wp9py 任务 4):prefill 决策
        # 的 history NOC_MIGRATE 迁移此前无路由字段(基线 60s 覆盖 70.4%,
        # 缺口 29.6% = history_transfer_bytes)。迁移路由在线可知
        # (history_source_instance_index -> prefill_instance_index 的
        # 实例图最短路,与 decode static_route.hop_count 同为实例间粒
        # 度),prefill 决策附加 noc_hops 输出字段(只读派生,不触 PD
        # 静态映射语义;A/B 对拍剥离清单条目)。
        self._instance_graph = InstanceGraph(self.topology)

        # 蓝图（离线蓝本，按阶段/函数名对应）:KV 账本。
        # offline: wsc_llm_scheduler.py
        self.workspace_envelope_certificate = None
        prospective_workspace = None
        if (workspace_admission == "required" and
                self.die_model_context is not None and
                self.die_model_context.profile == "coarse_die_v2"):
            from online.manifest_workspace_envelope import derive_manifest_workspace_envelope
            self.workspace_envelope_certificate = derive_manifest_workspace_envelope(
                manifest, config)
            prospective_workspace = self.workspace_envelope_certificate.bytes_by_rank
        self.kv_manager = SessionKVCacheManager(
            self.topology,
            config.model,
            prospective_workspace_bytes_by_rank=prospective_workspace,
        )

        # 蓝图（离线蓝本，按阶段/函数名对应）:实例账本。
        # offline: wsc_llm_scheduler.py
        self.instances = [
            _OnlineInstanceState(index=instance.index,
                                  phase_role=instance.phase_role)
            for instance in self.topology.instances
        ]

        # 蓝图（离线蓝本，按阶段/函数名对应）:future arrival min-heap(在线由 ingress ARRIVAL
        # 事件喂入)。阶段 4 §7.3:键含 queue_index,同 tick 到期项按冻结
        # 队列序稳定弹出(与 C++ 序列化的 arrivals 冻结队列序一致);
        # 消费规则 tick <= current_tick(见 _drain_arrival_heap)。
        # offline: wsc_llm_scheduler.py
        self.arrival_heap = []
        self._sequence = 0

        # 蓝图（离线蓝本，按阶段/函数名对应）:等待 decode 准入登记(按实例)。
        # §7.3:deque 替换 list.pop(0)。
        # offline: wsc_llm_scheduler.py
        self.waiting_decode_admissions = {
            instance.index: deque() for instance in self.topology.instances
        }

        # 阶段 4 §7.3:按 rank 的 ready frontier——非忙且有排队工作的实例
        # 集合(发射时清除,完成/到达时设置;结束审计必须为空)。_admit_pass
        # 只访问该集合(sorted 保持实例 index 序 = 离线循环序,决策确定性
        # 不受影响),不做全量实例扫描。
        self._ready_frontier = set()
        # 蓝图（离线蓝本，按阶段/函数名对应）:容量 epoch / 准入门控。
        # offline: wsc_llm_scheduler.py
        self.capacity_epoch = [0 for _ in self.topology.instances]
        # Last failed admission epoch lives on the request runtime itself.
        # A second request_id->epoch dictionary would retain completed
        # requests for the entire run.
        self.decode_admission_epoch = [-1 for _ in self.topology.instances]
        self.decode_admission_dirty = set()

        # 请求运行账本(queue_index 序;manifest 事实 policy-independent)。
        # offline: wsc_llm_scheduler.py(runtimes 由
        # _validate_and_expand_requests 构造;在线输入事实直接来自 manifest)。
        self.runtimes = [
            _OnlineRequestRuntime(record)
            for record in sorted(
                manifest["requests"], key=lambda item: item["queue_index"])
        ]
        self.runtime_by_request_id = {
            runtime.request_id: runtime for runtime in self.runtimes
        }
        # batch_train_ 保留前缀冲突拦截(G6-5):基类构造已按
        # BATCH_TRAIN_PREFIX 对 manifest request_id 做构造期 fail-closed
        # (见基类 __init__ 同名校验),本变体不再重复实现。
        # 阶段 4 §7.3:request_id -> runtimes 下标(O(1) 定位,替换
        # runtime_by_request_id_index 的 O(N) 全量扫描)。
        self._runtime_index = {
            runtime.request_id: index
            for index, runtime in enumerate(self.runtimes)
        }
        # 蓝图（离线蓝本，按阶段/函数名对应） 的 next_request(同 session 下一 turn 的 runtime 对象;
        # 没有则 None)。§7.3:(session_id, turn_index) 字典索引替换 O(N²)
        # 顺序扫描;turn+1 精确语义与扫描版逐项一致(值 = runtime 对象,
        # 与原扫描版返回对象一致)。
        by_turn = {}
        for runtime in self.runtimes:
            by_turn[(runtime.session_id, runtime.turn_index)] = runtime
        self.next_request = [None] * len(self.runtimes)
        for index, runtime in enumerate(self.runtimes):
            self.next_request[index] = by_turn.get(
                (runtime.session_id, runtime.turn_index + 1))

        self.completed_requests = 0
        # kv 事件水位:kv_actions 流(phase 2 对照 kv_cache_events.csv 基线)。
        self._kv_events_emitted = 0
        # 拼 batch 列车台账(§7.3 不变量断言输入):每次列车/整段发射一行,
        # 由 online_service 落 bridge 目录 train_ledger.jsonl(审计产物)。
        # D 侧行 = 迭代列车(member_iterations>0);P 侧行 = 整段 prefill
        # (prefill_chunks>0;§3.6 P 侧逐 chunk 不拼,退化列车);两类字段
        # 互斥(无混合列车)。
        # M3 流式落盘(2026-08-23,批次B 移植自 sh_3.0 母本):提供
        # train_ledger_sink 时行即写即弃;缺省 None(嵌入/测试)行即弃,
        # 无驻留缓冲。
        self.train_ledger_sink = train_ledger_sink
        # T_max 列车长度上限(§3.2.8/§7.4 治理旋钮 + A2 逐迭代 oracle):
        # SH_TRAIN_MAX_ITER 正整数 = 每列车至多 N 个迭代;0 = 不设限。
        # 交付默认 = 8(sh_1.0 母本 2026-08-22 §7.4 A2 对拍裁决:无上限
        # TTFT -67.3%,16 仍 -19.1%,8 全指标 ≤1.3%——"固定 T_max 为使
        # 位移 ≤5% 的最大值";原则 1 优先于节点数)。
        train_max_iter_env = os.environ.get("SH_TRAIN_MAX_ITER", "8")
        # low 修复(2026-09-26):空串此前经 `or 0` 静默解释为 0=不设限——
        # 即上文 A2 裁决否决的无上限形态;对空串与其余非法词法同通道
        # 构造期 fail-fast(全空白串 int() 本就会抛,此处一并覆盖)。
        if not train_max_iter_env.strip():
            raise ValueError(
                "SH_TRAIN_MAX_ITER must be a non-negative integer "
                "(0 = unlimited), got {!r}".format(train_max_iter_env))
        self._train_max_iter = int(train_max_iter_env)
        # low 修复(2026-09-24):负值此前无构造期校验,进入 _plan_train 后
        # `iterations > 负数` 恒真、iterations 被截为负值,要到发射期才以
        # 难懂错误暴露——改 fail-fast。非整数字面值在 int() 已抛
        # ValueError;0 = 不设限(值域见上注释)。
        if self._train_max_iter < 0:
            raise ValueError(
                "SH_TRAIN_MAX_ITER must be a non-negative integer "
                "(0 = unlimited), got {!r}".format(
                    os.environ.get("SH_TRAIN_MAX_ITER")))
        # train_id -> instance_index(哨兵事件路由)。
        self._train_instance_index = {}
        # WP9 首 token 首步批拆分(2026-08-26):batch_train_<id>_first_step
        # 唤醒 id -> 实例索引。首步批(无请求级 watch)完成时其唤醒 watch
        # fire 经 PREFILL_DRAIN 通道送回,本表区分"自己发射的首步唤醒"与
        # 真哨兵信号;唤醒只作余量批的交付边界,不进任何决策/核销路径
        # (no-op 交付,WP9_CONTRACT §3)。
        self._pending_first_steps = {}

    def _apply_delivery_extension(self, delta: dict) -> None:
        if self.workspace_admission != "required":
            return
        revision_by_rank, bytes_by_rank = self._workspace_state_maps(
            delta["workspace_state"])
        changed_ranks = self.kv_manager.set_external_workspace_view(
            revision_by_rank, bytes_by_rank)
        if changed_ranks:
            self._note_capacity_change(*(
                self._workspace_rank_to_instance[rank]
                for rank in changed_ranks))

    def _attach_response_extension(self, delta: dict, batch: dict) -> None:
        if self.workspace_admission == "required":
            batch["workspace_capacity_snapshot"] = (
                self.kv_manager.workspace_capacity_snapshot())
            if self.workspace_envelope_certificate is not None:
                for row in batch["workspace_capacity_snapshot"]["ranks"]:
                    row["prospective_workspace_bytes"] = (
                        self.workspace_envelope_certificate.bytes_by_rank[row["rank"]])

    # ------------------------------------------------------------- 策略 --

    def run_variant_policy(self, delta) -> None:
        """决策顺序逐行对应离线事件循环(已删蓝本,按阶段名对应):completion 批先于
        arrival 批,最后 start_ready_iterations。

        拼 batch 改造(§3.2 边界原子提交顺序):completion 批先按信号核销
        已完成列车(_finalize_completed_trains:核验列车 → 冻结成员闭式
        推进 token → 退出成员移出 active_decode → 清 in_flight_train),
        再逐请求处理 decode 完成账本与 REQUEST_COMPLETE 排程,最后
        准入/发射 pass 冻结并发射各空闲 D 实例的下一列车。DECODE_
        COMPLETION 与 REQUEST_COMPLETE 同 tick 同请求交付(C++ 同 fire
        推两条),顺序保持 decode 先于 request。
        """
        tick = delta["tick"]

        # ---- completion 批(离线 priority 0;同 tick 先于 arrival)----
        # offline: wsc_llm_scheduler.py
        drained = []
        decode_done = []
        request_done = []
        sentinel_trains = []
        for group in delta["completed_groups"]:
            stage = group["stage"]
            request_id = group["request_id"]
            if request_id.startswith("batch_train_"):
                sentinel_trains.append(request_id)
                continue
            if stage == STAGE_PREFILL:
                drained.append(request_id)
            elif stage == STAGE_DECODE:
                decode_done.append(request_id)
            elif stage == STAGE_REQUEST:
                request_done.append(request_id)
            else:
                raise ValueError(
                    "unknown completion stage {!r} for request {!r}".format(
                        stage, request_id))
        # 拼 batch 列车账本核销(先于逐请求完成处理:退出成员的
        # active_decode 移除与 token 推进在此完成,恰一次)。
        # WP9 首 token 拆分(2026-08-26):batch_train_*_first_step 唤醒
        # 信号是"自己发射的首步批"的交付回声——无操作(不决策/不记账/
        # 不写 decision_log 新条目),从哨兵核销通道剥离;余量批在本交付
        # 的准入/发射 pass 前发射(余量节点经依赖边排在首步节点之后,
        # 早发不改物理序)。首步批的 commit ack 走基类协议记账
        # (ack_count/幂等门),变体侧零动作。
        wakeup_instances = []
        routed_sentinels = []
        for train_id in sentinel_trains:
            instance_index = self._consume_first_step_wakeup(train_id)
            if instance_index is None:
                routed_sentinels.append(train_id)
            else:
                wakeup_instances.append(instance_index)
        if wakeup_instances:
            # WP9:本交付是首步唤醒交付(ON-only,OFF 无此交付边界)——
            # digest 行带 first_step 标记供 A/B 对拍剥离(剥离清单条目)。
            self._batch["first_step"] = True
        self._finalize_completed_trains(decode_done, routed_sentinels, tick)
        for request_id in drained:
            self._on_prefill_drain(request_id, tick)
        for request_id in decode_done:
            self._on_decode_complete(request_id, tick)
        for request_id in request_done:
            self._on_request_complete(request_id, tick)

        # ---- arrival 批(离线 priority 1)----
        # offline: wsc_llm_scheduler.py
        for arrival in delta["arrivals"]:
            self._push_arrival(arrival, tick)
        self._drain_arrival_heap(tick)

        # ---- 准入/发射 pass(离线 start_ready_iterations,计时部分删除)----
        # offline: wsc_llm_scheduler.py
        # WP9 拆分列车的余量批先于本 pass 发射(首步唤醒到达后的首个
        # 决策边界;发射位置与 sh_1.0 母本一致 = 策略尾段)。
        for instance_index in wakeup_instances:
            self._emit_train_remainder(self.instances[instance_index], tick)
        self._admit_pass(tick)

        # ---- kv 动作流:本批次 kv_manager 新产出的账本事件 ----
        # B1(2026-08-28):改走 events_since 增量读取(消费方游标本就存在),
        # 消除每批 events property 的全量 tuple() 拷贝;已消费前缀按水位
        # 整段压缩(游标同步回退),事件载荷/event_index 不变。
        new_events = self.kv_manager.events_since(self._kv_events_emitted)
        if new_events:
            self._batch["kv_actions"].extend(
                _kv_event_dict(event)
                for event in new_events)
            self._kv_events_emitted += len(new_events)
            removed = self.kv_manager.compact_events(
                self._kv_events_emitted)
            if removed:
                self._kv_events_emitted -= removed

    # ------------------------------------------------------ 列车账本 --

    def _finalize_completed_trains(self, decode_done, sentinel_trains,
                                   tick: int) -> None:
        """核销本交付中 exit 标记 watch 已 fire 的列车(§3.2 原子提交的
        前半;wscllm §3.6:仅 D 侧列车,P 侧整段发射无共享节点,经
        PREFILL_DRAIN 逐请求处理,无需列车核销)。

        exit 标记节点是列车体后的最末真实节点,任一标记 watch fire 即
        列车物理主体完成。同一列车不同成员的标记 watch 可能跨 tick
        fire、事件拆到不同交付——首个信号执行核销(账本推进恰一次),
        后续信号只清已核销列车的 pending 集合(幂等);信号不属于任何
        在飞/已核销列车即陈旧完成错配 fail-closed。推进量全部闭式
        (每成员 participation 次 token,无逐 token 循环)。"""
        signaled = {}
        for request_id in decode_done:
            # DECODE_COMPLETION 与 REQUEST_COMPLETE 同 tick 同请求交付且
            # decode 先于 request(run_variant_policy 头注),runtime 仅在
            # REQUEST_COMPLETE 边界弹出——此处查无 runtime 即协议违例,
            # fail-closed 报样式化错误(与下方 stale completion mismatch
            # 同款),不让裸 KeyError 抢先掩盖诊断信息。
            runtime = self.runtime_by_request_id.get(request_id)
            if runtime is None:
                raise RuntimeError(
                    "decode completion signal for request {} matches no "
                    "live runtime (stale completion mismatch)".format(
                        request_id))
            signaled.setdefault(
                runtime.decode_instance_index, set()).add(request_id)
        for train_id in sentinel_trains:
            instance_index = self._train_instance_index.pop(train_id, None)
            if instance_index is None:
                raise RuntimeError(
                    "sentinel signal for unknown train {}".format(train_id))
            signaled.setdefault(instance_index, set()).add(train_id)
        for instance_index, signals in signaled.items():
            state = self.instances[instance_index]
            pending_signals = set(signals)
            consumed = set()
            train = state.in_flight_train
            if train is not None:
                inflight_hits = pending_signals & train["signal_set"]
                if inflight_hits:
                    # 该列车首个信号到达:核销(账本推进恰一次),残余
                    # 信号(exit 标记跨 tick fire)登记待收。
                    for runtime, participation in train["members"]:
                        runtime.decode_tokens_consumed += participation
                    for request_id in train["exit_set"]:
                        runtime = self.runtime_by_request_id.get(request_id)
                        if runtime is None:
                            raise RuntimeError(
                                "exiting member {} lost its runtime before "
                                "train finalization (stale completion "
                                "mismatch)".format(request_id))
                        if runtime not in state.active_decode_lookup:
                            raise RuntimeError(
                                "exiting member {} is not in the decode "
                                "batch".format(request_id))
                        state.active_decode_lookup.discard(runtime)
                        state.active_decode.remove(runtime)
                    state.in_flight_train = None
                    # M4 核销即删(2026-08-23,批次B 移植自 sh_3.0 母本):
                    # 列车核销后其 train_id→实例索引条目即死重(哨兵条目
                    # 已在信号路由处弹出,此 pop 对其为幂等 no-op;本仓
                    # grep 证实核销后无读者)。
                    self._train_instance_index.pop(train["train_id"], None)
                    state.finalized_trains.append({
                        "train_id": train["train_id"],
                        "pending": train["exit_set"] - inflight_hits,
                    })
                    consumed |= inflight_hits
                    pending_signals -= inflight_hits
                    # busy 门解除:仍有成员则就绪下一列车(同 tick 的
                    # _admit_pass 发射),否则出 frontier。
                    if state.active_decode:
                        self._ready_frontier.add(state.index)
                    else:
                        self._ready_frontier.discard(state.index)
            # 已核销列车的后续信号对账(幂等清 pending,清空即出列表)。
            for record in state.finalized_trains:
                hits = pending_signals & record["pending"]
                record["pending"] -= hits
                consumed |= hits
            state.finalized_trains = [
                record for record in state.finalized_trains
                if record["pending"]]
            unresolved = pending_signals - consumed
            if unresolved:
                raise RuntimeError(
                    "completion signals {} for instance {} match no "
                    "in-flight or recently finalized train (stale "
                    "completion mismatch)".format(
                        sorted(unresolved), instance_index))

    def _consume_first_step_wakeup(self, train_id: str):
        """识别并吞掉自己发射的首步批唤醒信号(WP9,2026-08-26)。

        首步批的唤醒 watch 用批命名空间 id("<train_id>_first_step",
        batch_train_ 前缀),fire 后与真哨兵同通道送达。返回实例索引 =
        这是首步唤醒(无操作,仅从哨兵核销列表剥离);None = 非首步 id
        (真哨兵或未知 batch_train_ id,交回哨兵核销逻辑处理)。"""
        return self._pending_first_steps.pop(train_id, None)

    def _first_token_split_spans(self, plan):
        """WP9 首/余量 span 组切分(机械操作,2026-08-26)。

        首步 = 各 decode 成员的第 1 个 span(wscllm 列车无 prefill
        chunk,§2 wscllm 特例);余量 = 各成员剩余 span。聚合节点对
        span 求和与顺序无关,两组的激活/KV/AR 字节总量与整列一致;
        权重经 weight_passes(1 + iterations-1)合计不变。"""
        spans = list(plan["pass_spans"])
        member_parts = plan["members"]
        if sum(p for _, p in member_parts) != len(spans):
            raise RuntimeError(
                "train span layout does not match the frozen plan")
        first_spans = []
        rest_spans = []
        offset = 0
        for _, participation in member_parts:
            first_spans.append(spans[offset])
            rest_spans.extend(spans[offset + 1:offset + participation])
            offset += participation
        if offset != len(spans):
            raise RuntimeError(
                "train span layout does not match the frozen plan")
        return first_spans, rest_spans

    def _first_token_plan(self, plan, joiners):
        """WP9 首 token 观测计划(None = 开关关闭/无 debut,行为与拆分
        上线前逐字节一致)。

        debut 成员 = 本交付加入列车且 decode_tokens_consumed == 0 的
        decode 成员(计划期可知,WP9_CONTRACT §2)。多 token debut 挂
        独立 first_token 标记;decode_length=1 的 debut 其 exit 标记名
        附加 first_token 子串(同节点 code 4/8 双锚点,保证 first_token_
        ns == completion_ns)。iterations >= 2 时物理拆两批(首步批 +
        余量批),否则仅做不拆车的标记增强。"""
        if not first_token_split_enabled():
            return None
        debut = [
            runtime for runtime in joiners
            if runtime.decode_tokens_consumed == 0]
        if not debut:
            return None
        debut_marker_members = [
            {"request_id": runtime.request_id}
            for runtime in debut
            if runtime.decode_length != 1]
        debut_exit_first_token = [
            runtime.request_id for runtime in debut
            if runtime.decode_length == 1]
        if plan["iterations"] >= 2:
            first_spans, rest_spans = self._first_token_split_spans(plan)
            return {
                "split": True,
                "first_spans": first_spans,
                "rest_spans": rest_spans,
                "debut_marker_members": debut_marker_members,
                "debut_exit_first_token": debut_exit_first_token,
                "wakeup_id": "{}_first_step".format(plan["train_id"]),
            }
        return {
            "split": False,
            "debut_marker_members": debut_marker_members,
            "debut_exit_first_token": debut_exit_first_token,
        }

    def _plan_train(self, state):
        """冻结 D 实例的下一列车成员快照(§3.2 构造规则;§3.6:列车只含
        decode 成员 span,绝无 prefill chunk span——P 侧逐 chunk 不拼,
        不进入 D 实例列车)。

        列车终点 = 全部 decode 工作耗尽(无 prefill 工作 ⇒ 迭代 = max
        剩余 token;默认不设 T_max,§3.2.8)。成员退出不是列车边界
        (先验):退出成员在列车内挂 exit 标记。返回 None = 实例无工作。"""
        members = []
        for runtime in state.active_decode:
            remaining = (
                runtime.decode_length - runtime.decode_tokens_consumed)
            if remaining <= 0:
                raise RuntimeError(
                    "decode member {} has no remaining tokens".format(
                        runtime.request_id))
            members.append((runtime, runtime.prefill_context_tokens,
                            runtime.decode_tokens_consumed, remaining))
        if not members:
            return None
        iterations = max(remaining for _, _, _, remaining in members)
        capped = False
        if (self._train_max_iter and iterations > self._train_max_iter):
            iterations = self._train_max_iter
            capped = True
        # span 展开(成员×迭代;KV 逐迭代 +1,退出截断,无 padding)。
        # 聚合对 span 求和与顺序无关,故按成员连续段平铺(总 span 数与
        # 旧 request-aggregated 同级,非新增热路径)。
        pass_spans = []
        member_parts = []
        exit_members = []
        for runtime, context, consumed, remaining in members:
            participation = min(remaining, iterations)
            pass_spans.extend(
                (1, context + consumed + step)
                for step in range(1, participation + 1))
            member_parts.append((runtime, participation))
            if participation >= remaining:
                exit_members.append(runtime)
        sentinel = bool(capped and not exit_members)  # wscllm 无 drain 标记
        signal_set = {runtime.request_id for runtime in exit_members}
        if sentinel:
            signal_set.add("batch_train_i{}_{}".format(
                state.index, state.train_seq + 1))
        state.train_seq += 1
        train_id = "batch_train_i{}_{}".format(state.index, state.train_seq)
        return {
            "train_id": train_id,
            "iterations": iterations,
            "members": member_parts,
            "exit_members": exit_members,
            "exit_set": {runtime.request_id
                         for runtime in exit_members},
            "signal_set": signal_set,
            "sentinel": sentinel,
            "pass_spans": pass_spans,
        }

    def _decode_decision_dict(self, runtime, route) -> dict:
        """decode 决策行构造(问题 2A 修复 2026-09-05:从 _emit_train
        joiner 循环的内联构造抽出;decode_target_evictions 序列化——
        decode 侧逐出(#2 预占/#4 move/#5 grow_decode 累积)随 decode 决策
        行落盘;发射后由调用点置空核销,镜像 completion 对
        completion_evictions 的先序列化后置空语义)。

        B2(位置二态,2026-09):逐出对象从 EvictionRecord 换 KVTransfer
        (remote_store,契约 §3 行结构);prefill_decode_transfer 为 kind
        载体 KVTransfer,shard 级 noc_path/noc_hops 构造期已定。"""
        # 结构性防双计守卫(2026-10-01 审查修正):prefill 决策行已在
        # _emit_prefill 序列化过 decode_target_evictions 的当时快照
        # (#2 预占逐出),decode 行按冻结契约对同一账本再次双列序列化
        # (decode_target_evictions 与 decode_evictions 同内容);两行
        # 跨行求和即双计是已知纪律(消费侧按单键取用),本断言把纪律的
        # 结构前提钉死——decode_target_evictions 在 prefill 行与 decode
        # 行之间只增不减,decode 行快照必须逐对象包含 prefill 行快照;
        # 任何"提前置空/重赋值"都会让单键消费方静默丢逐出,fail-closed。
        prefill_snapshot = getattr(
            runtime, "decode_target_evictions_prefill_snapshot", ())
        if prefill_snapshot:
            current = runtime.decode_target_evictions
            for transfer in prefill_snapshot:
                if not any(transfer is held for held in current):
                    raise RuntimeError(
                        "decode decision row would drop eviction "
                        f"({transfer.kind}, session={transfer.session_id}) "
                        "already serialized in the prefill decision row's "
                        "decode_target_evictions; decode_target_evictions "
                        "must stay append-only between the two rows")
        return {
            "decode_instance_index": runtime.decode_instance_index,
            "static_route": {
                "prefill_instance_index": route.prefill_instance_index,
                "decode_instance_index": route.decode_instance_index,
                "path": list(route.path),
                "hop_count": route.hop_count,
                "shared_edges": [list(edge)
                                 for edge in route.shared_edges],
            },
            "decode_queue_depth_before_enqueue":
                runtime.decode_queue_depth_before_enqueue,
            # B2:decode 侧逐出(#2 reserve/#4 move/#5 grow_decode 累积)
            # 为 remote_store KVTransfer,按契约 §3 行结构逐条序列化。
            "decode_target_evictions": _kv_transfer_rows(
                runtime.decode_target_evictions),
            # B3 契约 §3 字段名逐字对齐(2026-09-06):decode_evictions 与
            # decode_target_evictions 同内容(watermark 旧映射按单键取用,
            # 防双计;S2 升级口径消费 decode_evictions)。
            "decode_evictions": _kv_transfer_rows(
                runtime.decode_target_evictions),
            # （2026-10-01 注记）*_eviction_count 为跨仓契约对齐字段：
            # 本仓写后不读（slo_tools 消费逐出行本体），sh_2.0 变体的
            # count-only 映射才读同名字段。
            "decode_eviction_count": len(runtime.decode_target_evictions),
            # B2:prefill→decode 迁移为 kind 载体 KVTransfer(含层域
            # 元数据与 shard 级 XY 路由)。
            "prefill_decode_transfer": _kv_transfer_dict(
                runtime.prefill_decode_transfer),
        }

    def _emit_train(self, state, tick: int) -> None:
        """把冻结的列车计划交给构图器发射,注册 exit 标记 watch,并挂起
        in_flight_train(busy 门 = 一个列车在飞;§3.6 仅 D 实例)。

        joiner = 尚未加入过任何列车的 active_decode 成员(其 transfer
        3000 迁移随本列车发射,物理先于列车体;drain 决策事件已在发射
        前交付,prefill 主体物理已完成)。

        WP9(2026-08-26):列车含 debut 成员且拆分开启、迭代数 >= 2 时
        两段式发射——本交付只发首步批(joiner 迁移/join 标记/首迭代体/
        first_token 标记/唤醒标记),exit/哨兵 watch、completion gate
        账本与正常 train_ledger 行移至余量批(_emit_train_remainder,
        首步唤醒到达后的首个决策边界);成员选择/成员排序/KV 动作/
        watch 挂载语义全部不变(PD 静态映射语义零触碰)。"""
        plan = self._plan_train(state)
        if plan is None:
            return
        joiners = [runtime for runtime in state.active_decode
                   if not runtime.decode_train_joined]
        joiner_ids = [runtime.request_id for runtime in joiners]
        # B3 发射(2026-09-06):joiner 计划携带 decode 侧逐出的待发射子集
        # (#2 预占逐出已在决策时点经 _flush_decode_evictions 就近发射,
        # 不再随列车头重发;此处承载 #4/#5 move/grow 逐出——运行时经预约
        # 不变量恒零逐出)——列车头以 prefill 段块末为触发门发射。
        joiner_plans = []
        for runtime in joiners:
            joiner_plan = self._plan_dict(runtime)
            joiner_plan["decode_evictions"] = runtime.decode_evictions_pending
            joiner_plans.append(joiner_plan)
        train_plan = {
            "train_id": plan["train_id"],
            "instance_index": state.index,
            "joiners": joiner_plans,
            "pass_spans": plan["pass_spans"],
            "iterations": plan["iterations"],
            "sentinel": plan["sentinel"],
            "exit_members": [
                {"request_id": runtime.request_id,
                 "session_id": runtime.session_id}
                for runtime in plan["exit_members"]],
        }
        # joiner 账本:首个列车加入(note_emitted/issue/decode 决策日志;
        # 与旧整段发射的 _emit_decode 同款内容,时点移至加入列车)。
        # WP9(2026-08-26):账本先于列车发射执行(拆分路径在发射处分支
        # 提前返回,joiner 账本仍须在首步批发射交付内完成;账本不读
        # 发射结果,两路径内容/顺序逐字节一致)。
        for runtime in joiners:
            runtime.decode_train_joined = True
            self._note_emitted(runtime.request_id, STAGE_DECODE)
            self._ledger_issue(runtime.request_id, tick, STAGE_DECODE,
                               state.index)
            route = runtime.static_route
            self.log_decision(
                {"kind": "decode", "request_id": runtime.request_id,
                 "priority": 0},
                tick,
                decision=self._decode_decision_dict(runtime, route),
            )
            # 问题 2A 修复(2026-09-05):decode 决策核销——上行的临时快照
            # 即全集(时间线已证:发射点必在 decode_target_evictions 全部
            # append(#2 reserve/#4 move/#5 grow_decode)之后、之后无
            # append;joiner 必在 append 后才进 active_decode),发射后置空
            # 镜像 completion 对 completion_evictions 的先序列化后置空
            # 语义,防未来工具读 decode 行时与 prefill 行 #2/#3 双计。
            # M4 的既有置空(见 _on_decode_complete 的 M4 核销块)保留为幂等兜底。待发射子账本
            # (C-2/M-25)随列车头发射同步核销。
            runtime.decode_target_evictions = ()
            runtime.decode_evictions_pending = ()
        first_token = self._first_token_plan(plan, joiners)
        if first_token is not None:
            train_plan["first_token"] = first_token
            if first_token["split"]:
                # WP9:joiner id 快照随 train_plan 走(余量批的台账行
                # 需要与首步行相同的 joiners 记录;joiner plan 只在首步
                # 批消费)。
                train_plan["joiner_ids_of_record"] = list(joiner_ids)
                self._emit_train_first_step(state, plan, train_plan, tick)
                return
        result = self.graph.emit_iteration_train(train_plan)
        # exit 标记 watch(每退出成员一个;C++ 同 fire 推 DECODE_
        # COMPLETION + REQUEST_COMPLETE 两条 completed_groups)。
        for runtime in plan["exit_members"]:
            self._batch["watches"].append({
                "request_id": runtime.request_id,
                "stage": STAGE_DECODE,
                "generation": 1,
                "members": result["exit_members"][runtime.request_id],
                "statuses": ["Success", "Skipped"],
            })
        if plan["sentinel"]:
            # T_max 截断且无 exit 标记:哨兵 watch(request_id = train_id,
            # 批命名空间;固定 prefill stage 单事件通道——decode 通道会
            # 双事件 + ServiceCoordinator 计账下溢)。
            self._batch["watches"].append({
                "request_id": plan["train_id"],
                "stage": STAGE_PREFILL,
                "generation": 0,
                "members": result["sentinel_members"],
                "statuses": ["Success", "Skipped"],
            })
        self._train_instance_index[plan["train_id"]] = state.index
        state.in_flight_train = plan
        self._ready_frontier.discard(state.index)  # §7.3:发射即忙
        self._emit_train_ledger_row(state, plan, joiner_ids, tick)

    def _emit_train_ledger_row(self, state, plan, joiner_ids, tick: int,
                               first_step: bool = False) -> None:
        """train_ledger 行(M3 流式落盘)。WP9(2026-08-26):拆分列车的
        首步批先写 first_step=True 行(ON/OFF 对拍剥离标记),余量批再
        写正常行(与整列发射的行同构,仅 tick 为余量发射交付)。"""
        ledger_row = {
            "train_id": plan["train_id"],
            "instance_index": state.index,
            "tick": tick,
            "iterations": plan["iterations"],
            "member_count": len(plan["members"]),
            "member_iterations": sum(
                participation for _, participation in plan["members"]),
            "joiners": list(joiner_ids),
            "drains": [],
            "exits": [runtime.request_id
                      for runtime in plan["exit_members"]],
            "sentinel": plan["sentinel"],
            "prefill_chunks": 0,  # §3.6:D 侧列车绝无 prefill chunk span
            "pass_spans": len(plan["pass_spans"]),
        }
        if first_step:
            ledger_row["first_step"] = True
        # M3 流式落盘:提供 train_ledger_sink 时行即写即弃;缺省行即弃。
        if self.train_ledger_sink is not None:
            self.train_ledger_sink(ledger_row)

    def _emit_train_first_step(self, state, plan, train_plan,
                               tick: int) -> None:
        """WP9 首步批发射(两段式前半,2026-08-26):构图 + 唤醒 watch
        注册 + busy 门挂起 + first_step 台账行。exit/哨兵 watch、
        completion gate 账本与正常台账行全部移至余量批。joiner 账本
        (decode 决策日志)已在 _emit_train 发射分支前完成(与 OFF 同
        交付同内容)。"""
        first_token = train_plan["first_token"]
        result = self.graph.emit_train_first_step(train_plan)
        self._batch["watches"].append({
            # 批命名空间唤醒 watch(哨兵同款单事件通道):首步批完成即
            # 交付余量批;不挂任何请求,fire 事件在 run_variant_policy
            # 的 _consume_first_step_wakeup 处无操作剥离(no-op 交付,
            # WP9_CONTRACT §3)。
            "request_id": first_token["wakeup_id"],
            "stage": STAGE_PREFILL,
            "generation": 0,
            "members": result["wakeup_members"],
            "statuses": ["Success", "Skipped"],
        })
        self._train_instance_index[plan["train_id"]] = state.index
        state.in_flight_train = plan
        state.first_step_remainder = train_plan
        self._pending_first_steps[first_token["wakeup_id"]] = state.index
        self._ready_frontier.discard(state.index)  # §7.3:发射即忙
        self._emit_train_ledger_row(
            state, plan, train_plan["joiner_ids_of_record"],
            tick, first_step=True)

    def _emit_train_remainder(self, state, tick: int) -> None:
        """WP9 余量批发射(两段式后半,2026-08-26;首步唤醒到达后的
        首个决策边界调用):余量体 + exit/哨兵标记 + end barrier,随后
        注册 exit/哨兵 watch 与正常台账行——与整列发射的后半完全同构
        (挂点语义不变:标记挂列车体后、end barrier 前)。"""
        train_plan = state.first_step_remainder
        if train_plan is None:
            raise RuntimeError(
                "remainder emission requested without a pending first step")
        state.first_step_remainder = None
        plan = state.in_flight_train
        result = self.graph.emit_train_remainder(train_plan)
        # exit 标记 watch(每退出成员一个;C++ 同 fire 推 DECODE_
        # COMPLETION + REQUEST_COMPLETE 两条 completed_groups)。
        for runtime in plan["exit_members"]:
            self._batch["watches"].append({
                "request_id": runtime.request_id,
                "stage": STAGE_DECODE,
                "generation": 1,
                "members": result["exit_members"][runtime.request_id],
                "statuses": ["Success", "Skipped"],
            })
        if plan["sentinel"]:
            # T_max 截断且无 exit 标记:哨兵 watch(request_id = train_id,
            # 批命名空间;固定 prefill stage 单事件通道——decode 通道会
            # 双事件 + ServiceCoordinator 计账下溢)。
            self._batch["watches"].append({
                "request_id": plan["train_id"],
                "stage": STAGE_PREFILL,
                "generation": 0,
                "members": result["sentinel_members"],
                "statuses": ["Success", "Skipped"],
            })
        self._emit_train_ledger_row(
            state, plan, train_plan["joiner_ids_of_record"], tick)

    # ------------------------------------------------------------- 边界 --

    def _on_arrival(self, runtime, tick: int) -> None:
        """离线 arrival 批单条(已删蓝本,按阶段名对应):选择 prefill 实例 + 静态路由 +
        qp 入队。快照在 append 之前取(ordering_key 反映选择时刻的排队深度)。

        offline: wsc_llm_scheduler.py
        """
        runtime.estimated_arrival_ns = tick
        snapshots = self._prefill_snapshots()
        selected = select_prefill_instance(snapshots)
        selected_snapshot = next(
            snapshot for snapshot in snapshots
            if snapshot.instance_index == selected)
        route = self.static_mapping.route_for_prefill(selected)
        runtime.prefill_instance_index = selected
        runtime.prefill_assignment_key = selected_snapshot.ordering_key
        runtime.static_route = route
        runtime.decode_instance_index = route.decode_instance_index
        self.instances[selected].qp.append(runtime)
        self._note_instance_ready(selected)  # §7.3 ready frontier
        # 阶段 3 感知账本:进入 admitted 层(prefill_qp 排队账本成员;
        # contract ⑥)。查询/审计数据,不进策略判据。
        self._ledger_admit(runtime.request_id, tick,
                           {"type": "prefill_qp", "instance_index": selected})

    def _on_prefill_drain(self, request_id: str, tick: int) -> None:
        """离线 prefill 完成分支(已删蓝本,按阶段/函数名对应):实例空闲 + qp 出队 + 等待
        decode 准入登记 + dirty。

        offline: wsc_llm_scheduler.py
        """
        runtime = self.runtime_by_request_id[request_id]
        state = self.instances[runtime.prefill_instance_index]
        if not state.qp or state.qp[0] is not runtime:
            raise RuntimeError("Prefill FCFS queue order was corrupted")
        state.busy = False
        state.qp.popleft()  # (§7.3:deque O(1) 首出)
        if state.qp:
            self._ready_frontier.add(state.index)  # §7.3:队列仍有等待
        else:
            self._ready_frontier.discard(state.index)
        runtime.waiting_decode_admission = True
        if runtime.decode_instance_index is None:
            raise RuntimeError("WSC Prefill completion lost Decode target")
        self.waiting_decode_admissions[runtime.decode_instance_index].append(
            runtime)
        self.decode_admission_dirty.add(runtime.decode_instance_index)
        # 阶段 3 感知账本:admitted 层排队类型更新(waiting_decode)。
        self._ledger_admit(
            runtime.request_id, tick,
            {"type": "waiting_decode",
             "instance_index": runtime.decode_instance_index})

    def _on_decode_complete(self, request_id: str, tick: int) -> None:
        """离线 decode 完成分支(已删蓝本,按阶段/函数名对应):mark_complete + 快照。
        下一次 arrival 排程在 REQUEST_COMPLETE 边界(同 tick,见
        _on_request_complete)。

        拼 batch 改造(§3.2):active_decode 出队与 busy 复位移至
        _finalize_completed_trains(退出迭代在列车内先验已知,物理完成
        时刻 = exit 标记节点完成时刻;本方法在核销之后执行,成员已移除)。

        offline: wsc_llm_scheduler.py
        """
        runtime = self.runtime_by_request_id[request_id]
        state = self.instances[runtime.decode_instance_index]
        self.completed_requests += 1
        runtime.completion_evictions = self.kv_manager.mark_complete(
            runtime.session_id,
            tick,
            runtime.request_id,
        )
        # B3 发射账本镜像:完成路径零逐出(契约 §9),同步调用保持接线
        # 均匀性(非空即 fail 的口径由 mark_complete 语义保证)。
        self.graph.sync_pending_history_after_evictions(
            runtime.completion_evictions)
        self._note_capacity_change(
            state.index,
            *_eviction_source_instances(runtime.completion_evictions),
        )
        snapshot = self.kv_manager.session_snapshot(runtime.session_id)
        if snapshot is None:
            raise RuntimeError("completed session disappeared from KV manager")
        runtime.kv_state_after_completion = snapshot.location  # (B2 位置二态)
        runtime.kv_instance_after_completion = snapshot.instance_index
        self.log_decision(
            {"kind": "completion", "request_id": runtime.request_id,
             "priority": 0},
            tick,
            decision={
                "terminal_kv_release_at_completion": False,
                "kv_state_after_completion": runtime.kv_state_after_completion,
                "kv_instance_after_completion":
                    runtime.kv_instance_after_completion,
                # B2:完成路径零逐出(恒空列表),序列化保持字段在场。
                "completion_evictions": _kv_transfer_rows(
                    runtime.completion_evictions),
            },
        )
        # M4 核销即删(2026-08-23,批次B 移植自 sh_3.0 母本;字段清单为本仓
        # runtime 形态):请求完成后其 KV 逐出元组/迁移对象/history 决策与
        # 完成快照等胖字段再无读者(逐出/迁移/快照已随 prefill/decode/
        # completion 决策行落盘,history_* 已随 _plan_dict 消费进图;
        # ledger_reconcile 读的是决策日志文件非 runtime 字段;下一 turn 是
        # 独立 runtime;runtimes 列表运行全程存活,不置空会随完成请求数
        # 线性常驻)——置空即删。
        runtime.admission_evictions = ()
        runtime.decode_target_evictions = ()
        runtime.decode_evictions_pending = ()
        runtime.completion_evictions = ()
        runtime.history_evictions = ()   # B3 发射分账同步核销
        runtime.prefill_evictions = ()
        runtime.prefill_decode_transfer = None
        runtime.history_action = None
        runtime.history_source_instance_index = None
        runtime.history_location_before = None      # B2 位置快照同步核销
        runtime.history_resident_prefix_layers = None
        runtime.history_transfer_bytes = None
        runtime.history_transfers = ()             # B2 holder 同步核销
        runtime.history_recompute_tokens = 0
        runtime.history_cache_state_before = None
        runtime.reservation_credit = None    # 净额修正(2026-09-06)同步核销
        runtime.kv_state_after_completion = None
        runtime.kv_instance_after_completion = None

    def _on_request_complete(self, request_id: str, tick: int) -> None:
        """离线 decode 完成分支的下一次 arrival 排程(已删蓝本,按阶段/函数名对应):
        now + 该 turn 的 inter_request_interval_ns 注册未来 alarm
        (向 ingress,不再 push 到事件堆)。

        offline: wsc_llm_scheduler.py
        """
        runtime = self.runtime_by_request_id[request_id]
        # 原 run-end 的 decode 终值检查前移到释放边界；之后 runtime 不再
        # 常驻，仍以同一严格条件 fail-closed。
        if (runtime.decode_tokens_consumed != runtime.decode_length
                or not runtime.decode_train_joined):
            raise RuntimeError(
                "completed request {!r} has an incomplete decode ledger"
                .format(request_id))
        # §7.3:_runtime_index O(1) 定位(替换 O(N) 全量扫描)。
        following = self.next_request[self._runtime_index[runtime.request_id]]
        if following is not None:
            interval = self._interval_ns(following)
            # 蓝图 push_event(now_ns + interval, 1, "arrival", following) ->
            # 在线等价:future alarm(阶段 1 的 alarm 语义,见 C++ Phase 4)。
            # interval==0 合法:此处记账的到达 tick 即提交 tick;EventQueue
            # 严格未来规则会把物理到达钳到 tick+1(契约内准点边界钳制,
            # 与 RequestIngress/GraphBatchCommitter 双侧同口径,量级 1ns)。
            self._batch["future_alarms"].append({
                "arrival_world_ns": tick + interval,
                "envelope": {
                    "request_id": following.request_id,
                    "session_id": following.session_id,
                    "turn_index": following.turn_index,
                    "prefill_length": following.prefill_length,
                    "decode_length": following.decode_length,
                    "inter_request_interval_ns": interval,
                },
            })
            # B3 分段发射(2026-09-06):为下一同 session turn 登记 pending
            # history 门(sh 在完成段发射时建门;本仓无完成段,REQUEST_
            # COMPLETE 是等价登记点)。位置 = turn-0 deferred 通道优先,
            # 否则管理器即时快照(DECODE_COMPLETION 与 REQUEST_COMPLETE
            # 同交付同 tick,之间无 mutation;此后到下一 turn 准入之间
            # 的逐出经 sync 镜像)。
            location = self.graph.pop_deferred_session_location(
                runtime.session_id)
            if location is None:
                snapshot = self.kv_manager.session_snapshot(
                    runtime.session_id)
                location = None if snapshot is None else snapshot.location
            self.graph.register_pending_history(
                request_id=following.request_id,
                session_id=runtime.session_id,
                source_instance_index=runtime.decode_instance_index,
                location=location,
            )
        else:
            # terminal turn 的 completion gate 不会再被下一次 admission
            # 消费；REQUEST_COMPLETE 是所有图边已发射后的安全回收点。
            self.graph.retire_completion_gate(runtime.session_id)
            released_instance_index = self.kv_manager.retire_terminal_session(
                runtime.session_id,
                tick,
                runtime.request_id,
            )
            # Retirement may free local HBM after the completion snapshot and
            # graph batch have been consumed.  Publish that capacity change
            # without consulting the now-deleted session state.
            self._note_capacity_change(released_instance_index)
        # B3:发射侧 per-request 账本核销(action 序号计数/prefill 段块末
        # 兜底弹出;下一 turn 是独立 request_id,本条目此后无读者)。
        self.graph.release_request_state(runtime.request_id)
        # REQUEST_COMPLETE 边界后该 runtime 及其索引再无读者。释放 map、
        # list slot 和已消费的 next link，避免完成请求仍被预构建链持有。
        self.runtime_by_request_id.pop(request_id, None)
        index = self._runtime_index.pop(request_id, None)
        if index is None:
            raise RuntimeError(
                "completed request lost runtime index {!r}".format(request_id))
        self.next_request[index] = None
        self.runtimes[index] = None

    # ------------------------------------------------------- arrival heap --

    def _push_arrival(self, arrival: dict, tick: int) -> None:
        """离线蓝本（按阶段名对应）push_event(time_ns, priority=1, kind="arrival")。
        字段形状与离线一致;阶段 4 §7.3 键含 queue_index —— 同 tick 到期项
        按冻结队列序稳定弹出,与 C++ 序列化的 arrivals 冻结队列序一致。
        """
        runtime = self.runtime_by_request_id[arrival["request_id"]]
        heapq.heappush(
            self.arrival_heap,
            (tick, 1, runtime.queue_index, self._sequence, "arrival",
             runtime))
        self._sequence += 1

    def _drain_arrival_heap(self, tick: int) -> None:
        """离线蓝本同 tick 批 按 (priority, sequence) 排序。
        阶段 4 §7.3:只消费 tick <= current_tick 的到期项(在线单 tick 单次
        交付下堆内全部为当前批 push 的项,规则与离线循环同构;未来 alarm
        的到期事件由 C++ 在到期 tick 交付,不会滞留堆中)。
        """
        while self.arrival_heap and self.arrival_heap[0][0] <= tick:
            _, _, _, _, kind, payload = heapq.heappop(self.arrival_heap)
            if kind != "arrival":
                raise RuntimeError("arrival heap contains {!r}".format(kind))
            self._profile_scan()  # §7.3:堆弹出条目(到期事件)
            self._on_arrival(payload, tick)

    # ------------------------------------------------------------- 准入 --

    def _admit_pass(self, tick: int) -> None:
        """离线 start_ready_iterations(已删蓝本,按阶段/函数名对应)的排队/准入部分;计时部分
        (LUT 估计 + push iteration_complete)删除,由真实完成事件推进。
        先 try_admit_waiting_decodes(已删蓝本,按阶段/函数名对应),再逐实例 serve(已删蓝本,按阶段/函数名对应)。

        阶段 4 §7.3:逐实例 serve 改为只访问 ready frontier(非忙且有排队
        工作的实例;sorted 保持实例 index 序 = 离线蓝本（按阶段名对应）的循环序,决策
        逐字节不变)。frontier 在到达/完成(设)与发射(清)时增量维护,
        单批访问条目数 = 到期/受影响条目数,与总 request 数无关。

        offline: wsc_llm_scheduler.py
        """
        self._try_admit_waiting_decodes(tick)
        for instance_index in sorted(self._ready_frontier):  # §7.3 frontier
            self._profile_scan()  # §7.3:frontier 访问条目(就绪实例)
            state = self.instances[instance_index]
            # busy 门(设计行为,非内部错误):frontier 的增量维护允许
            # 忙实例合法滞留(发射点 discard 前的到达/完成可再入),此处
            # 跳过即防重复发射(test_train_machinery::
            # test_busy_gate_skips_in_flight_instance 冻结该契约:跳过、
            # 不发射、不抛)。
            if self._instance_busy(state):
                continue
            if state.phase_role == PREFILL_ROLE:
                if state.active_decode:
                    raise RuntimeError(
                        "Prefill-only instance contains Decode work")
                if not state.qp:
                    continue
                runtime = state.qp[0]
                if not self._try_admit_prefill(runtime, tick):
                    continue
                self._emit_prefill(runtime, tick)  # 聚合:prefill 整段
                state.busy = True
                self._ready_frontier.discard(state.index)  # §7.3:发射即忙
            else:  # DECODE_ROLE（离线蓝本 serve 的 decode 分支，按阶段名对应）
                if state.qp:
                    raise RuntimeError(
                        "Decode-only instance contains Prefill work")
                if not state.active_decode:
                    continue
                # 拼 batch 改造(§3.2):decode 发射 = 冻结并发射整批
                # active_decode 的迭代列车(decode 互拼,权重每迭代只读
                # 一次);busy 门 = 一个列车在飞(在 _emit_train 挂起,
                # 列车 exit 标记 watch fire 时核销解除)。
                self._emit_train(state, tick)

    def _try_admit_prefill(self, runtime, now_ns: int) -> bool:
        """离线 try_admit_prefill(已删蓝本,按阶段/函数名对应),逐行对应;无逐 chunk 计时/
        剩余块账本(聚合粒度)。

        offline: wsc_llm_scheduler.py
        """
        if runtime.admitted_prefill:
            return True
        if (runtime.prefill_instance_index is None
                or runtime.static_route is None):
            raise RuntimeError("WSC Prefill admission lost its static mapping")
        prefill_instance = runtime.prefill_instance_index
        decode_instance = runtime.static_route.decode_instance_index
        admission_epoch = (
            self.capacity_epoch[prefill_instance],
            self.capacity_epoch[decode_instance],
        )
        if runtime.prefill_attempt_epoch == admission_epoch:
            return False
        runtime.prefill_attempt_epoch = admission_epoch
        final_shards = kv_cache_shard_bytes_for_tokens(
            self.config.model, runtime.final_context_tokens,
            self.topology.instances[0].size)
        reservation = self.kv_manager.reserve_request_capacity(
            runtime.request_id,
            runtime.session_id,
            decode_instance,
            final_shards,
            now_ns,
            phase="prefill_admission",
            reason="static_decode_final_kv_reservation",
        )
        runtime.decode_target_evictions = tuple(
            (*runtime.decode_target_evictions, *reservation.evictions))
        # C-2/M-25(2026-09-30):预占逐出同登"待发射"子账本——准入成功随
        # prefill 批发射(段块末触发门),阻塞则在本边界立即发射(无门)。
        runtime.decode_evictions_pending = tuple(
            (*runtime.decode_evictions_pending, *reservation.evictions))
        # B3 发射账本镜像:逐出立即同步 pending history 门(决策时点
        # 同步是唯一标记路径,见 builder.sync_pending_history_after_
        # evictions;漏接会在下一 turn 的位置对账处 fail-closed)。
        self.graph.sync_pending_history_after_evictions(reservation.evictions)
        if not reservation.admitted:
            # 准入预占净额修正(2026-09-06):全量预约把"本会话尚未迁走的
            # 旧 KV"与"终态全量"重复计入 decode 实例(super-linear 记账,
            # 超长会话在队列尾部永久 deep-gap)。仅当旧 KV 确实完整驻留
            # (LOCAL)于 decode 目标(它稍后被本流程的 prepare_history
            # 迁走)时,改按净额 max(0, 终态−旧本地驻留)重试;否则维持
            # 现状全量语义。credit 取本地账面 local_shard_bytes(LOCAL
            # 单态下即全量向量)。
            snap = self.kv_manager.session_snapshot(runtime.session_id)
            if (snap is not None
                    and snap.location == LOCAL_HBM
                    and snap.instance_index == decode_instance):
                credit = tuple(snap.local_shard_bytes)
                net_shards = tuple(
                    max(0, final - have)
                    for final, have in zip(final_shards, credit))
                full_attempt_evictions = reservation.evictions
                reservation = self.kv_manager.reserve_request_capacity(
                    runtime.request_id,
                    runtime.session_id,
                    decode_instance,
                    net_shards,
                    now_ns,
                    phase="prefill_admission",
                    reason="static_decode_final_kv_reservation_net_credit",
                )
                runtime.decode_target_evictions = tuple(
                    (*runtime.decode_target_evictions, *reservation.evictions))
                runtime.decode_evictions_pending = tuple(
                    (*runtime.decode_evictions_pending,
                     *reservation.evictions))
                # B3:净额重试的逐出同样镜像(两次尝试的逐出都是真实
                # mutation)。
                self.graph.sync_pending_history_after_evictions(
                    reservation.evictions)
                if not reservation.admitted:
                    # 净额仍失败:两次尝试的逐出都是真实 mutation,epoch
                    # 唤醒必须合并覆盖(丢第一次会楔死等待重试的准入)。
                    # 逐出为 remote_store KVTransfer,受害实例取
                    # source_instance_index(整会话逐出同款)。
                    if full_attempt_evictions or reservation.evictions:
                        self._note_capacity_change(
                            decode_instance,
                            *_eviction_source_instances(
                                full_attempt_evictions),
                            *_eviction_source_instances(
                                reservation.evictions),
                        )
                    # C-2/M-25:阻塞 attempt 的预占逐出立即随当前批发射
                    # (无触发门——本请求 prefill 段块末尚不存在),决策→
                    # 发射窗口不得跨 capacity_epoch 重试存续。
                    # B2 注记(2026-09-30 审查):现管理器下本臂结构性
                    # 不可达——ensure_physical_fit 的边际门保证
                    # blocked ⟹ evictions==()(全量/净额失败时循环未入),
                    # 上方 if 恒空;防御性保留(若未来管理器恢复部分逐出
                    # 语义,此臂即正确行为)。
                    self._flush_decode_evictions(runtime, gated=False)
                    return False
                runtime.reservation_credit = credit
            else:
                if reservation.evictions:
                    self._note_capacity_change(
                        decode_instance,
                        *_eviction_source_instances(reservation.evictions),
                    )
                # B2 注记(2026-09-30 审查):同上,现管理器下
                # blocked ⟹ reservation.evictions==(),防御性保留。
                self._flush_decode_evictions(runtime, gated=False)
                return False
        runtime.decode_capacity_reserved = True
        before_snapshot = self.kv_manager.session_snapshot(
            runtime.session_id)
        runtime.history_cache_state_before = (  # (B2 位置二态 location)
            "ABSENT" if before_snapshot is None
            else before_snapshot.location)
        # B2 三态:准入时点的历史位置快照(契约 §3 history_location_before
        # / history_resident_prefix_layers 序列化源)。
        runtime.history_location_before = (
            None if before_snapshot is None else before_snapshot.location)
        runtime.history_resident_prefix_layers = (
            None if before_snapshot is None
            else before_snapshot.resident_prefix_layers)
        decision = self.kv_manager.prepare_history(
            runtime.session_id,
            runtime.prefill_instance_index,
            runtime.history_tokens_before,
            now_ns,
            runtime.request_id,
            required_context_tokens=runtime.prefill_context_tokens,
        )
        runtime.admission_evictions = tuple(
            (*runtime.admission_evictions, *decision.evictions))
        # B3 发射分账:prepare_history 的 fit 逐出 = history_evictions 段
        # (触发门 = 到达/interval 门);同步镜像 pending 门。
        runtime.history_evictions = tuple(
            (*runtime.history_evictions, *decision.evictions))
        self.graph.sync_pending_history_after_evictions(decision.evictions)
        if decision.admission_blocked:
            self.kv_manager.release_request_capacity(runtime.request_id, now_ns)
            runtime.decode_capacity_reserved = False
            # 净额修正(2026-09-06):release 是 pop 语义,按登记值(净额)
            # 对称释放自动正确;credit 随之失效,防跨 attempt 残留。
            runtime.reservation_credit = None
            self._note_capacity_change(
                prefill_instance,
                decode_instance,
                decision.source_instance_index,
                *_eviction_source_instances(reservation.evictions),
                *_eviction_source_instances(decision.evictions),
            )
            # C-2/M-25:已放行的预占逐出立即随当前批发射(无触发门),
            # 不跨 capacity_epoch 重试存续。
            # B2 注记(2026-09-30;2026-10-01 R3 修正):本臂(prepare_
            # history 阻塞、reserve 已成功携带逐出)是**生产可达的活路
            # 径**——reserve-blocked 的两个 flush 臂才是"恒空转"的防御
            # 性保留(边际门保证彼处 evictions==());此处 pending 可非空,
            # 必须发射。
            self._flush_decode_evictions(runtime, gated=False)
            return False
        # 净额修正回补(2026-09-06):prepare_history 的迁移/恢复已把旧 KV
        # 的本地部分从 decode 目标删掉(resident→prefill 侧),立即把净额
        # 预约回补到全量——复刻原设计"从准入占位到 P→D move 完成"的
        # 防抢占语义(move 时刻 decode 侧 existing=0,需要全量空间);
        # resident→reserved 1:1 换位,任何瞬间不超订。
        if runtime.reservation_credit is not None:
            if decision.source_instance_index != decode_instance:
                raise RuntimeError(
                    "net-credit reservation expected the history migration "
                    "to vacate the static Decode instance (source="
                    f"{decision.source_instance_index!r}, decode="
                    f"{decode_instance!r})")
            self.kv_manager.extend_request_capacity(
                runtime.request_id,
                runtime.reservation_credit,
                now_ns,
                reason="history_vacated_decode_target",
            )
            runtime.reservation_credit = None
        runtime.history_action = decision.action
        runtime.history_source_instance_index = decision.source_instance_index
        runtime.history_transfer_bytes = decision.transfer_bytes
        # 留存恢复/迁移 KVTransfer 对象(REMOTE_RESTORE 全量恢复为单段;
        # local_hit/ABSENT 恒空元组),供 prefill 决策 history_transfers
        # 序列化。
        runtime.history_transfers = decision.transfers
        runtime.history_recompute_tokens = 0  # RECOMPUTE 删除(B2)
        growth = self.kv_manager.grow_prefill(
            runtime.session_id,
            runtime.prefill_context_tokens,
            now_ns,
            runtime.request_id,
        )
        if not growth.admitted:
            raise RuntimeError("Prefill growth lost its successful capacity preflight")
        runtime.admission_evictions = tuple(
            (*runtime.admission_evictions, *growth.evictions))
        # B3 发射分账:grow_prefill 的增长逐出 = prefill_evictions 段
        # (无触发门,链在 rank frontier)。
        runtime.prefill_evictions = tuple(
            (*runtime.prefill_evictions, *growth.evictions))
        self.graph.sync_pending_history_after_evictions(growth.evictions)
        runtime.admitted_prefill = True
        self._note_capacity_change(
            prefill_instance,
            decode_instance,
            decision.source_instance_index,
            *_eviction_source_instances(reservation.evictions),
            *_eviction_source_instances(decision.evictions),
            *_eviction_source_instances(growth.evictions),
        )
        return True

    def _try_admit_waiting_decodes(self, now_ns: int) -> None:
        """离线 try_admit_waiting_decodes(已删蓝本,按阶段/函数名对应),逐行对应。

        offline: wsc_llm_scheduler.py
        """
        ready_targets = tuple(
            instance_index
            for instance_index, queue in sorted(
                self.waiting_decode_admissions.items())
            if queue
            and (
                instance_index in self.decode_admission_dirty
                or self.decode_admission_epoch[instance_index]
                != self.capacity_epoch[instance_index]
            )
        )
        for target_instance in ready_targets:
            self.decode_admission_epoch[target_instance] = (
                self.capacity_epoch[target_instance])
            self.decode_admission_dirty.discard(target_instance)
            queue = self.waiting_decode_admissions[target_instance]
            pending_count = len(queue)
            for _ in range(pending_count):
                runtime = queue.popleft()  # (§7.3:deque O(1) 首出)
                if not runtime.waiting_decode_admission:
                    continue
                if runtime.decode_instance_index != target_instance:
                    raise RuntimeError("WSC Decode target queue was corrupted")
                if not runtime.decode_capacity_reserved:
                    raise RuntimeError(
                        "WSC Decode target was not reserved before Prefill")
                # 本边界 = Prefill→Decode 交接(下一行 move_prefill_to_
                # decode 即 decode 侧锚),预约释放行的审计 anchor 同边界
                # 标 decode_start(默认 prefill_start 是 Prefill 准入边界,
                # 与本边界不符);只影响 journal 审计标注列。
                self.kv_manager.release_request_capacity(
                    runtime.request_id, now_ns, anchor_kind="decode_start")
                runtime.decode_capacity_reserved = False
                move = self.kv_manager.move_prefill_to_decode(
                    runtime.session_id,
                    target_instance,
                    now_ns,
                    runtime.request_id,
                    final_context_tokens=runtime.final_context_tokens,
                )
                runtime.decode_target_evictions = tuple(
                    (*runtime.decode_target_evictions, *move.evictions))
                # C-2/M-25(2026-09-30):move 逐出保留至 joiner 列车头
                # 发射(运行时经预约不变量恒零逐出,原路径不变)。
                runtime.decode_evictions_pending = tuple(
                    (*runtime.decode_evictions_pending, *move.evictions))
                # B3 发射账本镜像:move/grow_decode 逐出(decode 侧)。
                self.graph.sync_pending_history_after_evictions(
                    move.evictions)
                if move.admission_blocked:
                    # 防御死管道(2026-09-30 审查登记,结构性不可达:
                    # ensure_physical_fit 的净额预约不变量保证本边界
                    # blocked 恒零逐出/恒放行——释放的静态预约额物理上
                    # 恰够 move 所需),保留作 fail-closed 兜底。
                    raise RuntimeError(
                        "released static Decode reservation was not "
                        "physically usable")
                runtime.prefill_decode_transfer = move.transfer
                growth = self.kv_manager.grow_decode(
                    runtime.session_id,
                    runtime.final_context_tokens,
                    now_ns,
                    runtime.request_id,
                )
                if not growth.admitted:
                    # 防御死管道(同 move.admission_blocked 分支:预约不变
                    # 量保证本边界恒放行),保留作 fail-closed 兜底。
                    raise RuntimeError(
                        "Decode growth lost its successful target admission")
                runtime.decode_target_evictions = tuple(
                    (*runtime.decode_target_evictions, *growth.evictions))
                runtime.decode_evictions_pending = tuple(
                    (*runtime.decode_evictions_pending, *growth.evictions))
                self.graph.sync_pending_history_after_evictions(
                    growth.evictions)
                decode_state = self.instances[target_instance]
                if decode_state.phase_role != DECODE_ROLE:
                    raise RuntimeError(
                        "static route selected a non-Decode instance")
                runtime.decode_queue_depth_before_enqueue = (
                    len(decode_state.active_decode))
                decode_state.active_decode.append(runtime)
                decode_state.active_decode_lookup.add(runtime)  # §7.3 双侧同步
                if not self._instance_busy(decode_state):  # §7.3
                    self._ready_frontier.add(target_instance)
                runtime.waiting_decode_admission = False
                # 阶段 3 感知账本:admitted 层排队类型更新(active_decode)。
                self._ledger_admit(
                    runtime.request_id, now_ns,
                    {"type": "active_decode", "instance_index": target_instance})
                self._note_capacity_change(
                    target_instance,
                    move.source_instance_index,
                    *_eviction_source_instances(move.evictions),
                    *_eviction_source_instances(growth.evictions),
                )

    # ------------------------------------------------------------- 发射 --

    def _emit_prefill(self, runtime, tick: int) -> None:
        """聚合粒度发射 prefill 整段(与离线 ET 按 request 整段一致;
        离线逐 chunk 迭代在在线不存在)。watch 注册 PREFILL_DRAIN。

        §3.6 P 侧逐 chunk 不拼:整段聚合的 weight_passes 缺省 = len(spans)
        = chunk 数(每 chunk 恰读一遍权重,chunk 之间不互拼),发射骨架
        与改造前逐字节一致;拼 batch 改造仅追加 train_ledger 行(退化
        纯 prefill 列车:prefill_chunks>0 与 member_count>0 互斥)。

        offline: wsc_llm_scheduler.py(发射对象为整段而非 chunk)
        """
        plan = self._plan_dict(runtime)
        members = self.graph.emit_prefill_batch(plan)
        # C-2/M-25(2026-09-30):预占逐出(#2 reserve)随本 prefill 批发射
        # (触发门 = 刚登记的 prefill 段块末,与列车头旧行为同门)——决策
        # 到发射的窗口收敛为零,受害会话的任何后续回迁/退休都只可能发生在
        # 登记之后(旧"随 joiner 首列车发射"路径的倒序发射/泄漏两面一并
        # 关闭;见 graph_batch_builder.emit_decode_eviction_actions)。
        self._flush_decode_evictions(runtime, gated=True)
        # 阶段 3 感知账本:本批次发射记录(prefill 阶段;commit ack 到达后
        # 转移入 committed 层)。
        self._note_emitted(runtime.request_id, STAGE_PREFILL)
        # 阶段 7 §10.1:issued 层登记(已发射未完成;completion 核销时由
        # 基类 _settle_completions 移除)。
        self._ledger_issue(runtime.request_id, tick, STAGE_PREFILL,
                           runtime.prefill_instance_index)
        self._batch["watches"].append({
            "request_id": runtime.request_id,
            "stage": STAGE_PREFILL,
            "generation": 0,
            "members": members,
            "statuses": ["Success", "Skipped"],
        })
        self._batch["assignments"].append({
            "request_id": runtime.request_id,
            "prefill_instance_index": runtime.prefill_instance_index,
            "decode_instance_index": runtime.decode_instance_index,
        })
        prefill_decision = {
                "prefill_instance_index": runtime.prefill_instance_index,
                "prefill_assignment_key": list(runtime.prefill_assignment_key),
                "estimated_arrival_ns": runtime.estimated_arrival_ns,
                # RECOMPUTE 删除(B2):有效 prefill = 本 turn 自身长度
                # (历史经远端/NoC 恢复,不再重算)。
                "effective_prefill_tokens": runtime.prefill_length,
                "history_action": runtime.history_action,
                "history_cache_state_before":
                    runtime.history_cache_state_before,
                "history_source_instance_index":
                    runtime.history_source_instance_index,
                "history_transfer_bytes": runtime.history_transfer_bytes,
                "history_recompute_tokens": 0,
                # ---- B2 三态决策日志契约(契约 §3;纯输出字段) ----
                # 准入时点会话历史位置(二态 local_hbm|remote_memory)与
                # 驻留层数(model_layers|0)。
                "history_location_before": runtime.history_location_before,
                "history_location_before_instance_index": (
                    None if runtime.history_location_before is None
                    else runtime.history_source_instance_index),
                "history_resident_prefix_layers":
                    runtime.history_resident_prefix_layers,
                # 恢复/迁移对象:REMOTE_RESTORE 全量恢复(单段)在同一条
                # prefill 记录内承载。
                "history_transfers": _kv_transfer_rows(
                    runtime.history_transfers),
                "transfer_hop_bytes": [
                    {"kind": transfer.kind,
                     "session_id": transfer.session_id,
                     "total_bytes": transfer.total_bytes,
                     "hops": max(
                         (len(shard.noc_path) - 1)
                         for shard in transfer.shards
                     ) if transfer.shards else 0}
                    for transfer in runtime.history_transfers
                ],
                # 逐出对象为 remote_store KVTransfer(契约 §3 行结构)。
                "admission_evictions": _kv_transfer_rows(
                    runtime.admission_evictions),
                "decode_target_evictions": _kv_transfer_rows(
                    runtime.decode_target_evictions),
                # ---- B3 契约 §3 字段名逐字对齐(2026-09-06):sh_2.0 的
                # history_evictions/prefill_evictions 分段字段 + 计数字段;
                # 与 admission_evictions(union,watermark 旧映射消费)并存
                # 为超集,防双计由消费侧按单键取用保证。----
                "history_eviction_count": len(runtime.history_evictions),
                "history_evictions": _kv_transfer_rows(
                    runtime.history_evictions),
                "prefill_evictions": _kv_transfer_rows(
                    runtime.prefill_evictions),
        }
        if (runtime.history_transfer_bytes
                and runtime.history_source_instance_index is not None):
            # Hop-Bytes 覆盖(见 __init__ 注释):NOC_MIGRATE 迁移的实例间
            # hop 数(实例图最短路,与 decode static_route.hop_count 同粒
            # 度);只加输出字段,剥离清单条目。B3(2026-09-06):REMOTE
            # 恢复的 source_instance_index 为 None(远端池无实例语义),
            # 跳过实例间 hop 诊断(逐 shard 路由在 transfer_hop_bytes)。
            prefill_decision["noc_hops"] = (
                self._instance_graph.shortest_distance(
                    runtime.history_source_instance_index,
                    runtime.prefill_instance_index))
        # 发射端单点记录(2026-10-01 审查修正):prefill 决策行在此序列化
        # decode_target_evictions 当时快照;_decode_decision_dict 的
        # fail-closed 断言以本快照为基准钉死"两行之间只增不减"的
        # 防双计结构前提(纯内部账本,不进入任何对外发射字节)。
        runtime.decode_target_evictions_prefill_snapshot = (
            runtime.decode_target_evictions)
        self.log_decision(
            {"kind": "prefill", "request_id": runtime.request_id,
             "priority": 0},
            tick,
            decision=prefill_decision,
        )
        # 拼 batch 列车台账:P 侧行(退化纯 prefill 列车 = 整段 chunk 序列,
        # 每 chunk 一次前向;iterations = chunk 数与权重口径一致)。
        state = self.instances[runtime.prefill_instance_index]
        state.train_seq += 1
        # RECOMPUTE 删除(B2):remaining_chunks 公式的 ceil(history/p_chunk)
        # 项随重算段一并删除,迭代数 = 本 turn prefill 的 chunk 数。
        # prefill_length 恒为正(loader parse_int 强制 >0,plan 物化沿用),
        # 不做 max(...,1) 钳位。M-26 修复(2026-09-30):ceil 表达式恢复
        # 母本 -(-len // chunk)(负 floor 而非 ceil 会让 P 侧台账三字段
        # 恒负/零,违反自身 prefill_chunks>0 契约)。
        prefill_chunks = -(
            -runtime.prefill_length // int(self.config.prefill_chunk_size))
        ledger_row = {
            "train_id": "prefill_train_i{}_{}".format(
                state.index, state.train_seq),
            "instance_index": state.index,
            "tick": tick,
            "iterations": prefill_chunks,
            "member_count": 0,  # §3.6:P 侧绝不与 decode 成员同列车
            "member_iterations": 0,
            "joiners": [],
            "drains": [runtime.request_id],
            "exits": [],
            "prefill_chunks": prefill_chunks,
            "pass_spans": prefill_chunks,
        }
        # M3 流式落盘:提供 train_ledger_sink 时行即写即弃;缺省行即弃
        #(P 侧退化列车行,字段与改前一致)。
        if self.train_ledger_sink is not None:
            self.train_ledger_sink(ledger_row)

    # ------------------------------------------------------------- 助手 --

    def _plan_dict(self, runtime) -> dict:
        """graph_batch_builder 消费的 plan 字段(request 事实 + 在线决策)。
        与 replay 路径同构;history_action 等 KV 决策字段在准入时已落账本。

        B3 三态发射(2026-09-06):补 builder 消费的发射面字段——
        history_transfers(恢复/迁移 KVTransfer 对象,REMOTE_RESTORE 单段
        承载)、history_evictions/prefill_evictions(分账逐出段,
        触发门口径不同)、history_location_before/history_resident_
        prefix_layers(pending 门位置对账;二态取值 local_hbm|remote_memory
        与 model_layers|0)。
        """
        return {
            "request_id": runtime.request_id,
            "session_id": runtime.session_id,
            "turn_index": runtime.turn_index,
            "queue_index": runtime.queue_index,
            "prefill_instance_index": runtime.prefill_instance_index,
            "decode_instance_index": runtime.decode_instance_index,
            "history_action": runtime.history_action,
            "history_source_instance_index":
                runtime.history_source_instance_index,
            "history_transfer_bytes": runtime.history_transfer_bytes,
            "history_recompute_tokens": runtime.history_recompute_tokens,
            "history_tokens_before": runtime.history_tokens_before,
            "prefill_context_tokens": runtime.prefill_context_tokens,
            "prefill_length": runtime.prefill_length,
            "decode_length": runtime.decode_length,
            "history_location_before": runtime.history_location_before,
            "history_resident_prefix_layers":
                runtime.history_resident_prefix_layers,
            "history_transfers": runtime.history_transfers,
            "history_evictions": runtime.history_evictions,
            "prefill_evictions": runtime.prefill_evictions,
        }

    def _flush_decode_evictions(self, runtime, *, gated: bool) -> None:
        """C-2/M-25(2026-09-30):把该 runtime 已决策未发射的 decode 侧
        逐出(#2 预占)交给构图器发射并核销待发射子账本(决策日志账本
        decode_target_evictions 保留全集,decode 决策行序列化口径不变)。

        gated=True(准入成功,随 prefill 批):触发门 = prefill 段块末,
        与旧行为"随 joiner 首列车发射"同门;gated=False(阻塞 attempt,
        本边界立即发射):无触发门,链到各 rank 当前 frontier——阻塞路径
        的 prefill 段块末尚不存在,物理逐出不得等待一个可能跨多个
        capacity_epoch 重试才到来的 prefill(face 仓 emit_eviction_
        actions 阻塞逐出同款)。"""
        if not runtime.decode_evictions_pending:
            return
        self.graph.emit_decode_eviction_actions(
            self._plan_dict(runtime), runtime.decode_evictions_pending,
            gated_on_prefill_segment_end=gated)
        runtime.decode_evictions_pending = ()

    def _has_pending_decode_store_evictions(self, session_id: str) -> bool:
        """C-2/M-25 fail-closed 探针:该会话是否存在已决策未发射的
        decode 侧 remote_store。现役路径下 #2 预占逐出在准入边界就近
        发射、#4/#5 move/grow 经预约不变量恒零逐出,本探针运行时恒
        False;命中即"决策时点就近发射"不变量被破坏(fail-closed,
        见 graph_batch_builder._arm_store_tail_dependencies)。"""
        return any(
            any(transfer.session_id == session_id
                for transfer in runtime.decode_evictions_pending)
            for runtime in self.runtime_by_request_id.values()
        )

    def _prefill_snapshots(self):
        """离线 prefill_snapshots(已删蓝本,按阶段/函数名对应):仅 prefill 角色实例,
        在 instances 序(实例 index 序)上构建。"""
        return tuple(
            PrefillQueueSnapshot(instance_index=state.index,
                                 request_count=len(state.qp))
            for state in self.instances
            if state.phase_role == PREFILL_ROLE
        )

    def _note_capacity_change(self, *instance_indexes) -> None:
        """离线 note_capacity_change(已删蓝本,按阶段/函数名对应)。"""
        for instance_index in set(instance_indexes):
            if instance_index is not None:
                self.capacity_epoch[instance_index] += 1

    def _note_instance_ready(self, instance_index: int) -> None:
        """§7.3 ready frontier:实例非忙且有排队工作 -> 就绪集(发射时由
        _admit_pass 清除;完成/到达时设置)。"""
        if not self._instance_busy(self.instances[instance_index]):
            self._ready_frontier.add(instance_index)

    @staticmethod
    def _instance_busy(state) -> bool:
        """拼 batch 改造后的 busy 门(按 phase_role 分裂,§3.6):P 实例 =
        一个 prefill 整段在飞(busy);D 实例 = 一个迭代列车在飞
        (in_flight_train 非 None)。"""
        if state.phase_role == PREFILL_ROLE:
            return state.busy
        return state.in_flight_train is not None

    def _sensing_ready_view(self) -> dict:
        """阶段 7 §10.1 ready 层边界视图:ready frontier 实例队列中等待
        服务的 request(prefill qp / active_decode;frontier = 非忙且有排队
        工作的实例,§7.3)。访问条目 = 就绪实例数(有界,非全量扫描)。
        查询/审计输入,不进策略判据。"""
        detail = []
        for instance_index in sorted(self._ready_frontier):
            state = self.instances[instance_index]
            if state.phase_role == PREFILL_ROLE:
                stage, queue = STAGE_PREFILL, state.qp
            else:
                stage, queue = STAGE_DECODE, state.active_decode
            for runtime in queue:
                detail.append({
                    "request_id": runtime.request_id,
                    "stage": stage,
                    "instance_index": instance_index,
                    "admitted_tick": self.ledger_admitted.get(
                        runtime.request_id, {}).get("admitted_tick"),
                })
        return {"ready_count": len(detail), "detail": detail}

    def _interval_ns(self, runtime) -> int:
        spec = self.config.request_queue[runtime.queue_index]
        interval = spec.inter_request_interval_ns
        # B4(2026-09-30):turn>0 行缺 interval 时旧实现把 None 原样返回,
        # 首个触点是下游 tick + None 的裸 TypeError(无诊断);物化链
        # 校验字段在场使生产不可达,此处对齐"缺失输入结构化 fail-closed"
        # 纪律显式报错(与下游 _emit_prelim 的既有报错同源)。
        if interval is None:
            raise ValueError(
                f"turn>0 request {runtime.request_id!r} lacks "
                f"inter_request_interval_ns (queue_index="
                f"{runtime.queue_index})——下一轮到达间隔缺失")
        return interval

    # --------------------------------------------------------------- 收尾 --

    def verify_run_end(self) -> None:
        """基类协议校验之上,叠加阶段 4 §7.3 结束审计(事件索引队列全部
        为空:arrival heap / ready frontier)。拼 batch 改造:增加列车账本
        清空断言(在飞列车/已核销待收信号/未加入列车的 decode 成员)。"""
        super().verify_run_end()
        if self.completed_requests != self.expected_request_count:
            raise RuntimeError(
                "strategy run ended with {}/{} requests complete".format(
                    self.completed_requests, self.expected_request_count))
        if any(state.busy or state.qp or state.active_decode
               or state.in_flight_train is not None or state.finalized_trains
               or state.first_step_remainder is not None
               for state in self.instances):
            raise RuntimeError("strategy run ended with non-idle instance state")
        if self._pending_first_steps:
            raise RuntimeError(
                "strategy run ended with unconsumed first-step wakeups: "
                "{!r}".format(sorted(self._pending_first_steps)))
        if self.runtime_by_request_id or self._runtime_index or \
                any(runtime is not None for runtime in self.runtimes):
            raise RuntimeError("strategy run ended with unreleased runtimes")
        if any(queue for queue in self.waiting_decode_admissions.values()):
            raise RuntimeError(
                "strategy run ended with pending decode admissions")
        # 阶段 4 §7.3 结束审计:heap/ready set 全空(与 C++ 侧 mailbox/
        # watch registry 审计同批)。
        if self.arrival_heap:
            raise RuntimeError(
                "run ended with {} unconsumed arrival events".format(
                    len(self.arrival_heap)))
        if self._ready_frontier:
            raise RuntimeError(
                "run ended with non-empty ready frontier: {!r}".format(
                    sorted(self._ready_frontier)))
        if self.graph.completion_gates:
            raise RuntimeError(
                "strategy run ended with unretired completion gates: {!r}"
                .format(sorted(self.graph.completion_gates)))
        # B3 发射账本结束审计:跨请求 history 门/延迟位置/prefill 段块末
        # 全部清空(每个会话终态 retire、每个请求完成核销后不应有残留)。
        if (self.graph.pending_history
                or self.graph.deferred_session_locations
                or self.graph.pending_request_by_session):
            raise RuntimeError(
                "strategy run ended with pending history ledgers: "
                "gates={!r} deferred={!r} sessions={!r}".format(
                    sorted(self.graph.pending_history),
                    sorted(self.graph.deferred_session_locations),
                    sorted(self.graph.pending_request_by_session)))
        # 私有账本访问防御(2026-10-01 审查修正):本审计深入构图器私有
        # 字段;字段漂移(改名/移位)时的 AttributeError 会脱离"结束审计"
        # 语境报错,这里显式带文件与字段名 fail-closed,正常路径零行为差。
        try:
            prefill_segment_ends = self.graph._prefill_segment_ends
            action_sequence = self.graph._action_sequence
        except AttributeError as error:
            raise RuntimeError(
                "run-end audit cannot read the graph builder's private "
                "emission ledgers (online/graph_batch_builder.py fields "
                "_prefill_segment_ends / _action_sequence have drifted); "
                "update this audit point") from error
        if prefill_segment_ends or action_sequence:
            raise RuntimeError(
                "strategy run ended with per-request emission ledgers: "
                "segments={!r} actions={!r}".format(
                    sorted(prefill_segment_ends),
                    sorted(action_sequence)))
        # 同族账本补审计(2026-09-26):在飞 store 支链尾部(§3.3,restore/
        # REQUEST_COMPLETE 清理)与哨兵列车路由表(_emit_train 登记、
        # 核销/信号路由弹出)运行末必空——静态推理全部会话终态 retire,
        # 与 _pending_first_steps 同批次 fail-closed,防未来路径演化
        # 静默漏检。
        if self.graph.pending_store_tails:
            raise RuntimeError(
                "strategy run ended with pending store tails: {!r}".format(
                    sorted(self.graph.pending_store_tails)))
        if self._train_instance_index:
            raise RuntimeError(
                "strategy run ended with un-routed train sentinels: {!r}"
                .format(sorted(self._train_instance_index)))
        self.kv_manager.assert_final_state()
        # P1 权威 HBM delta journal run 末 checksum 门(fail-closed,doc
        # §6-P1):流式重放 kv_delta_journal.jsonl 与 manager 终态逐 rank
        # 对账并断言守恒(resident=0/reserved=0/physical=weight),产物
        # kv_delta_journal_checksum.json;journal 未装配时为 no-op(旁路态
        # 与改前行为一致)。挂在 assert_final_state 之后同一校验链。
        self.kv_manager.verify_journal_checksum()
        # journal-off 旁路态防御(2026-09-29):上方守恒门(resident=0)
        # 仅在 journal 装配时生效;旁路态下改前对"终态仍有非活跃会话占留
        # 本地 HBM"零拦截(正常静态推理全部会话终态 retire,残留等价于
        # journal-on 的 resident=0 违约信号)。故此处补 stderr 结构化告警
        # 留痕——刻意不 fail-closed,不改旁路态合法语义,零产物数值影响。
        # 私有字段访问防御(2026-10-01 审查修正):_metrics_recorder 为
        # None 是旁路态正常路径(不 raise);字段本体漂移(改名/移位)时
        # 带文件与字段名 fail-closed,避免静默按"journal-off"降级审计。
        if not hasattr(self.kv_manager, "_metrics_recorder"):
            raise RuntimeError(
                "run-end audit cannot read the KV manager's metrics "
                "recorder (session_kv_manager.py field _metrics_recorder "
                "has drifted); update this audit point")
        metrics_recorder = self.kv_manager._metrics_recorder
        if metrics_recorder is not None and not hasattr(
                metrics_recorder, "journal_enabled"):
            raise RuntimeError(
                "run-end audit cannot read the metrics recorder's "
                "journal-mode flag (metrics_integration.py "
                "MemoryActionRecorder field journal_enabled has drifted); "
                "update this audit point")
        if not (
                metrics_recorder is not None
                and metrics_recorder.journal_enabled):
            leftover_sessions = [
                session_id
                for session_id in self.kv_manager.session_ids
                if self._inactive_session_resident(session_id)
            ]
            if leftover_sessions:
                print(
                    "wsc_llm_online_scheduler: journal-off bypass run ended "
                    "with inactive sessions still holding local HBM KV "
                    "(journal-on conservation gate would reject these): "
                    "{}".format(leftover_sessions),
                    file=sys.stderr,
                )

    def _inactive_session_resident(self, session_id: str) -> bool:
        """journal-off 旁路态终态审计:非活跃会话是否仍占留本地 HBM。"""

        snapshot = self.kv_manager.session_snapshot(session_id)
        return (
            snapshot is not None
            and not snapshot.active
            and snapshot.location == LOCAL_HBM
            and any(snapshot.local_shard_bytes)
        )
