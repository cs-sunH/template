#!/usr/bin/env python3
"""sh30_online_scheduler.py -- sh_3.0 关感知策略调度器（strategy 模式，步骤 1-9）。

以已移除的离线 plan_face_requests（2026-08-21 离线 planner 清除批删去）为蓝本迁移，保持决策顺序
逐行对应（每处迁移用 `# offline: face_scheduler.py:XXXX` 注释标注）。离线事件
循环与在线边界的一一对应：

  离线事件循环                                   在线边界
  ----------------                               ----------------
  预置 turn-0 arrival 堆（:3574-3579）            ingress ARRIVAL 事件喂入同一
                                                 arrival heap（字段保持
                                                 (time_ns, priority, sequence,
                                                 kind, payload) 形状）
  iteration_complete 批（:3907-4006）             PREFILL_DRAIN /
    prefill 末 chunk（:3931-3984）                  _on_prefill_drain（逐行迁移）
    decode 完成（:3986-3999）                       _complete_requests（同 tick）
    下一 turn arrival（:4000-4006）                 _complete_requests（同 tick）
  completion_order 批（:4008-4042）               _complete_requests
    mark_complete / 快照                            同名调用逐行迁移
  arrival 批（:4044-4058）                        _on_arrival（记 arrival →
                                                 pending_admissions → retry；
                                                 truncate_history 已随 sidecar
                                                 机制移除，recompute 单口径下
                                                 恒 no-op）
  admit_waiting_requests（:4060-4061）            _admit_pass 同 tick 末尾复查
  start_ready_iterations（:3833-3890）            _plan_and_emit_trains 的列车
                                                 发射：**直接 Roofline 计时部分
                                                 删除**（:3853-3858 估算与
                                                 :3885-3890 推事件），排队/配对
                                                 语义（busy 判定、FCFS qp、
                                                 active_decode 成员）原样保留
                                                 于账本逻辑。

拼 batch 改造（2026-08-22，设计文档《层次 B Continuous Batching 改造》
§3.2"迭代列车聚合发射"；照母本 sh_1.0 定型版机制移植）：层次 B 从
"请求级整段串行"重构为"实例迭代级列车"——decode 互拼、decode 与 prefill
chunk 混拼、chunk 之间不拼、批成员只在列车边界变化。每实例状态机
（§3.2）：qp（FCFS prefill 队列）/active_decode（批成员表）/
pending_decode_ready（KV 就绪待加入）/in_flight_train（唯一在飞列车 +
train_id/membership_digest）；列车终点 = 下一个不可预测事件（队列头
prefill drain / 全部工作耗尽），默认不设 T_max。边界原子提交顺序：核验
digest → 推进冻结成员 token → 退出成员移除 → 推进 prefill chunk → 处理
drain/完成 → 合入 arrival → KV 就绪成员入批 → 冻结下一列车成员 → 发射。
决策边界仍是四类 reason（ARRIVAL/PREFILL_DRAIN/DECODE_COMPLETION/
REQUEST_COMPLETE），由列车 drain/exit 标记节点的 watch 驱动。

sh_3.0 特性保留（红线，策略公式/阈值/KV 语义/映射规则一律不动）：
  - sticky 亲和与批齐发-等完成骨架的列车化重构：原"批齐发-等完成"
    （_start_ready_iterations 的发射对象 = qp[0] prefill 整段 + 全部
    active_decode decode 整段，busy 等待）适配为列车发射（一趟列车 =
    队列头 chunk × 迭代 + 全部 active_decode 成员，等待语义由
    in_flight_train 替代 busy）；三段式准入的 sticky 判据
    （LOCAL/PARTIAL 驻留实例 sticky / REMOTE 边缘负载均衡 / 首请求
    非边缘）与亲和规则本身一行不动；
  - decode 固定 prefill 同实例（红线 #4：selected = state_index，
    decode_candidates = ()）；
  - task_load_snapshot 三分量：打分公式与阈值不动，仅物理折算随列车化
    重订（active 段"段在飞=全量剩余"假设作废 → decode_tokens_consumed
    闭式迭代级剩余量；在飞 chunk 负载 = 冻结列车账本）。

关感知口径（方案 §4.1 第 6 条 + 合同⑥）：策略输入全部来自 Python 账本——
task_load_snapshot 三分量（Roofline 估计服务时间；queued 分量逐行复用
:3612-3638；running/active 分量的进度输入按列车闭式账本折算）、HBM 可行
掩码、edge_free 掩码（:466-483）、KV 快照（KVCacheManager 只读复用，
红线 #6-#9）。不新增任何 C++ 状态读取。

与离线蓝图的刻意差异（real-online 语义，合同⑦）：
  - 计时/迭代粒度：离线直接 Roofline 时钟 + 逐 chunk 迭代 -> 在线真实完成
    事件 + 迭代列车构图（成员×迭代 span，weight_passes=迭代数）；
  - 完成顺序：真实完成 tick 决定，不要求与离线决策序列 exact（合同⑦
    real-online 只验不变量与差异可解释性）。
"""

import hashlib
import heapq
import json
import math
import os
import sys
import time
from collections import deque
from dataclasses import replace as _dataclass_replace

# --------------------------------------------------------------------------
# import 路径：本文件位于 workload/llama2_7b_inference/online/，离线规划模块
# 在上一级。路径只做 import 用途（红线：face_scheduler.py /
# generate_face_trace.py 只读 import 与注释）。
# --------------------------------------------------------------------------
_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _path in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from face_scheduler import (  # noqa: E402
    FaceInstanceSpec,
    KVCacheManager,
    KVCapacityError,
    KVPhysicalInfeasibleError,
    edge_free_instance_mask,
    edge_instance_mask,
    build_instances,
    estimate_decode_remaining_task_load_ns,
    estimate_prefill_task_load_ns,
    kv_cache_shard_bytes_for_layer_range,
    kv_cache_shard_bytes_for_tokens,
    select_prefill_instance,
)
from joint.joint_config import (  # noqa: E402
    parse_joint_config,
    remote_credit_block_size,
)
from joint.hbm_port_flow_registry import HbmPortFlowRegistry  # noqa: E402
from joint.joint_cost_model import (  # noqa: E402
    ACTION_ORDER,
    ACTION_RECOMPUTE,
    CausalHorizonEstimator,
    InstanceLoadView,
    JointCostModel,
    JointHardwareRates,
    LinkFlowRegistry,
    RequestView,
    ServiceFactors,
    SessionKVView,
)
from joint.joint_scheduler import select_instance_and_action  # noqa: E402
from joint.link_quota import (  # noqa: E402
    DELTA_ADM_INITIAL_NS,
    FLOW_ONESHOT,
    FLOW_REALTIME,
    MERGE_DIRECTION_FORWARD,
    MERGE_DIRECTION_REVERSE,
    QUOTA_AIMD,
    LinkQuotaError,
    LinkQuotaTracker,
    QuotaVerdict,
    WAIT_CAPACITY,
    WAIT_QUOTA_LINK,
    WAIT_QUOTA_PORT,
    extend_retry_key,
    quota_deferred_requeue,
)
from online.online_scheduler_base import (  # noqa: E402
    STAGE_DECODE,
    STAGE_PREFILL,
    STAGE_REQUEST,
    OnlineSchedulerBase,
)
from online.graph_batch_builder import (  # noqa: E402
    first_token_split_enabled,
)


_TASK_LOAD_CACHE_CAPACITY = 4096

#: C5（F8 冻结，2026-09-22）：joint_admission 决策日志的
#: ActionCostBreakdown 序列化字段序——全 11 字段与
#: joint_cost_model.ActionCostBreakdown 声明逐一同名（值由 C2 物化填充，
#: 本卡只冻结字段与序列化——值可为 None，字段存在性与可解析性为本卡
#: 判据）。schema 先于 WP3 冻结（F8），配额类字段（port_snapshot 族）
#: 本卡记 "NA"、由 C2/C11 接通后替换，不缺席。
_JOINT_BREAKDOWN_LOG_FIELDS = (
    "target_wait_ns", "history_prep_ns", "eviction_wait_ns", "compute_ns",
    "remote_read_ns", "merge_ns", "contention_divisor", "hops", "notes",
    "remote_read_first_credit_ns", "remote_read_stream_ns",
)


class _PoolPortRegistry:
    """R15-3：片外池边缘端口的在途流登记表（键 = edge rank 端口）。

    机制与链路登记（LinkFlowRegistry）同族：remote_store/remote_load
    发射时登记占用端口，完成事件（drain/completion/merge-watch 交付）
    注销；除数 = 实例各端口最大在途数 + 候选自身 1。供 J 计价的池路径
    与 E 内核 r_j（P1）共用同一份额视图。
    """

    def __init__(self) -> None:
        self._counts: dict[int, int] = {}
        self._owners: dict[str, list[int]] = {}

    def register(self, edge_rank: int, *, owner: str) -> None:
        self._counts[edge_rank] = self._counts.get(edge_rank, 0) + 1
        self._owners.setdefault(owner, []).append(edge_rank)

    def release_owner(self, owner: str) -> int:
        edges = self._owners.pop(owner, None)
        if not edges:
            return 0
        for edge_rank in edges:
            count = self._counts.get(edge_rank, 0) - 1
            if count > 0:
                self._counts[edge_rank] = count
            else:
                self._counts.pop(edge_rank, None)
        return len(edges)

    def count(self, edge_rank: int) -> int:
        return self._counts.get(edge_rank, 0)

    def leaked_owners(self) -> dict:
        """在册归属视图（漏释放审计：结算边界后仍非空 = 泄漏证据）。

        O10③（2026-09-23 终轮审计）：与 LinkFlowRegistry/HbmPortFlowRegistry
        的 leaked_owners 同款语义与返回形态——``{owner: tuple(在途
        edge_rank)}`` 有序 dict，空 = 干净，只计在途。本注册表注销仅有
        release_owner 一通道（无逐端口释放 API），``_owners`` 残留即在
        途证据（生产完成事件全走 release_owner）。"""
        return {
            owner: tuple(edge_ranks)
            for owner, edge_ranks in sorted(self._owners.items())
            if edge_ranks}


def _bounded_task_load_memo(cache: dict, key, compute, *, capacity: int) -> int:
    """确定性 FIFO 的精确 Roofline memo。

    缓存只保存同参纯估算器的整数结果；淘汰最早插入项只会触发同函数
    重算，不会改变任务负载值或任何映射 / KV 决策输入。
    """
    if capacity <= 0:
        raise ValueError("task-load memo capacity must be positive")
    try:
        return cache[key]
    except KeyError:
        pass
    value = compute()
    if len(cache) >= capacity:
        del cache[next(iter(cache))]
    cache[key] = value
    return value


# --------------------------------------------------------------------------
class _OnlineInstanceState:
    """在线实例账本（离线 _InstanceRuntime 的在线子集 + 拼 batch 列车
    状态机，§3.2）：qp = 已准入 prefill FCFS 队列（deque[runtime]），
    active_decode = decode 批成员表，pending_decode_ready = KV 就绪待加入
    下一列车的成员，in_flight_train = 唯一在飞列车（冻结成员快照）；
    busy 门语义 = "一个列车在飞"（in_flight_train 替代原 busy 的等待语义，
    2026-08-22 列车化重构），last_arrival_ns 供 ordering_key。"""

    __slots__ = ("index", "qp", "active_decode", "active_decode_lookup",
                 "pending_decode_ready", "in_flight_train", "finalized_trains",
                 "iteration_count", "train_seq", "last_arrival_ns",
                 "ledger_epoch", "snapshot_epoch", "snapshot_cache",
                 "snapshot_horizon_version",
                 "first_step_remainder", "last_train_finalize_tick")

    def __init__(self, *, index: int):
        self.index = index
        self.qp = deque()
        self.active_decode = []
        self.active_decode_lookup = set()
        # ---- 拼 batch 列车账本（2026-08-22） ----
        self.pending_decode_ready = []   # KV 就绪待加入下一列车的成员
        self.in_flight_train = None      # 唯一在飞列车（冻结成员快照）
        self.finalized_trains = []       # 已核销列车（待收后续跨交付信号）
        self.iteration_count = 0         # 已完成迭代数（列车核销时闭式推进）
        self.train_seq = 0               # 列车序号（命名/审计用）
        self.last_arrival_ns = None
        # ---- 改法S2：实例账本纪元 + 快照纪元缓存（2026-08-23）----
        # ledger_epoch：本实例快照输入（qp/active_decode 成员、成员进度
        # 字段 prefill_tokens_completed/decode_tokens_consumed、
        # in_flight_train、last_arrival_ns）的每次变更 +1；快照缓存
        # key = 实例 + 纪元（对齐改法D _kv_ledger_epoch/
        # _admit_attempt_epoch 的姿态，见 _bump_instance_epoch）。
        self.ledger_epoch = 0
        self.snapshot_epoch = -1        # 缓存快照所属纪元（-1 = 未缓存）
        self.snapshot_cache = None      # 上次快照（冻结 dataclass，可复用）
        # M3（kimi 复审）：缓存快照所属的在线时域估计器版本——快照的
        # decode 分量经 CausalHorizonEstimator.estimate 是隐藏输入，估计
        # 器更新（observe_completed）不 bump 实例纪元；漏查会让缓存跨
        # 估计器版本陈旧复用，SH_SNAPSHOT_VERIFY 影子断言假阳性。
        self.snapshot_horizon_version = -1
        # ---- WP9 首步批拆分（2026-08-26）：两段式发射的余量批挂起 ----
        self.first_step_remainder = None  # 待发射余量批 train_plan（None=无）
        # R14（2026-09-14）：上一列车核销 tick——ServiceFactors 纯样本的
        # 服务段起点（列车 k 的纯服务时长 = 核销_k − max(发射_k, 核销_{k-1})，
        # 排除列车前排队的污染）。
        self.last_train_finalize_tick = None


class _OnlineRequestRuntime:
    """在线请求运行账本（离线 _RequestRuntime 的在线子集 + plan dict 字段）。

    输入事实（request-neutral，来自 manifest）：history_tokens_before /
    prefill_context_tokens / final_context_tokens / prefill_length /
    decode_length / queue_index / session_id / turn_index。运行期事实由在线
    决策产出（语义与离线 _RequestRuntime 同名同义）。

    拼 batch 列车推进字段（2026-08-22）：remaining_chunks /
    prefill_tokens_completed / decode_tokens_consumed /
    current_decode_token / drain_block_ends——决策边界上闭式推进，
    余额与逐 token 精确值逐点一致（供 task_load_snapshot 折算与列车规划）。
    """

    __slots__ = (
        "request_id", "session_id", "turn_index", "queue_index",
        "prefill_length", "decode_length",
        "history_tokens_before", "prefill_tokens_to_process",
        "prefill_context_tokens", "final_context_tokens",
        "remaining_chunks", "prefill_tokens_completed",
        "decode_tokens_consumed", "current_decode_token",
        "prompt_tokens_processed", "drain_block_ends",
        "queued_chunk_load_ns",
        "prefill_complete_ns", "decode_start_ns",
        "estimated_arrival_ns", "admission_time_ns", "hbm_wait_ns",
        "prefill_instance_index", "prefill_assignment_key",
        "prefill_instance_loads", "prefill_hbm_feasible_instances",
        "prefill_affinity_reason",
        "decode_instance_index",
        "history_source_instance_index", "history_transfer_bytes",
        "history_tokens_discarded", "history_location_before",
        "history_transfer", "history_evictions", "prefill_evictions",
        "prefill_decode_transfer", "decode_evictions", "completion_evictions",
        "kv_location_after_completion", "kv_instance_after_completion",
        "admitted", "completed", "completion_ns",
        # ---- joint 三机制字段（设计方案 §2/§6） ----
        "joint_action", "origin_home_instance", "history_transfers",
        "merge_transfers", "merge_outcome", "joint_prefill_work",
        "joint_span_base_context",
        "joint_input_tokens", "joint_cost_ns", "joint_working_copy",
        # ---- 规格书§二（2026-09-25 prefill remote-read 分阶段）：前缀
        # 读流独立账本（与 history_transfers 严格分离——后缀池恢复留在
        # history_transfers，前缀 [0,p) 读流若混入会被误当历史工作副本
        # 迁移，错误触发 readiness barrier、history_transfer_bytes 与
        # merge 账本）----
        "prefill_remote_read_transfers", "prefill_remote_read_plan",
        "prefill_remote_read_bytes",
        # ---- R14（2026-09-14）：decode 增长停滞/唤醒 ----
        "decode_stalled", "stall_wake_key", "stall_reason", "stall_gap_records",
        # ---- remote-read credit 执行口径（唯一机制）----
        "remote_read_credit_plan", "remote_read_credit_trains",
        "remote_read_slice_summaries",
        # ---- C8（WP2-preadmit）：同 tick 读流承诺预登记账本 ----
        "remote_read_preplan",
        # ---- C11（WP3c）：quota_deferred 首等待时标（驻留时长样本） ----
        "quota_deferred_since_ns",
        # ---- 逐出尾 watch（2026-09-24 修复）：decode joiner drain 登记 ----
        "decode_eviction_watch_id",
    )

    def __init__(self, record: dict, p_chunk: int = 512) -> None:
        self.request_id = record["request_id"]
        self.session_id = record["session_id"]
        self.turn_index = record["turn_index"]
        self.queue_index = record["queue_index"]
        self.prefill_length = record["prefill_length"]
        self.decode_length = record["decode_length"]
        self.history_tokens_before = record["history_tokens_before"]
        self.prefill_tokens_to_process = (
            record["prefill_context_tokens"] - record["history_tokens_before"])
        self.prefill_context_tokens = record["prefill_context_tokens"]
        self.final_context_tokens = record["final_context_tokens"]
        # joint：物理 prefill 工作量与 span 上下文基（recompute 折入历史
        # 时两者在准入点改写；缺省 = 原 recompute 单口径）。
        self.joint_action = "stay"
        self.joint_prefill_work = self.prefill_tokens_to_process
        self.joint_span_base_context = self.history_tokens_before
        self.joint_input_tokens = self.prefill_length
        self.joint_cost_ns = None
        # R14：decode 增长停滞态（False = 正常；停滞会话暂缓后续列车
        # 参与，容量释放（纪元 bump）后唤醒重试）。
        self.decode_stalled = False
        self.stall_wake_key = None
        self.stall_reason = None
        # remote-read credit（唯一执行口径）：drain 边界建立的
        # 持久读计划（每步读量/路径一次冻结；逐列车切片在列车规划期从其
        # 派生）、已切片列车计数 j（R15 owner 键 rid#decode#{j} 的序号）
        # 与逐列车切片摘要（completion 决策行披露）。非 remote-read
        # 动作的请求三者恒为初始值。
        self.remote_read_credit_plan = None
        self.remote_read_credit_trains = 0
        self.remote_read_slice_summaries = []
        # C8（WP2-preadmit，§4.1 同 tick 承诺可见性）：准入相 remote-read
        # 读流承诺预登记（owner = rid#readplan）的运行账本——est/真值/核销
        # 余量与登记模板；对账链 = drain 冻结（_reconcile_readplan_at_drain）
        # → 逐列车核销（_consume_readplan_units）→ 完成清残（_settle_
        # readplan_residual）。非 remote-read 恒 None。
        self.remote_read_preplan = None
        # C11：quota_deferred 首次等待时标（None = 未经历配额等待）；
        # 最终准入时结算驻留时长样本入指标。
        self.quota_deferred_since_ns = None
        # K6：停滞携带的容量缺口记录（KVCapacityError.deep_gap_records）；
        # 停滞是可恢复路径不落账，仅在死锁守卫确认终态时提交台账。
        self.stall_gap_records = ()
        self.origin_home_instance = None
        self.history_transfers = ()
        self.merge_transfers = ()
        # 规格书§二.1（2026-09-25）：prefill remote-read 前缀读流独立
        # 账本——准入时由 plan_prefill_remote_read_transfers 规划（纯
        # 规划零副作用；stream_only 瞬时流，不物化 exec 容量账本），
        # owner = rid#prefill_read、prefill drain 释放。非 remote-read
        # （或退化无读流）恒为初始值 ()/None/0。
        self.prefill_remote_read_transfers = ()
        self.prefill_remote_read_plan = None
        self.prefill_remote_read_bytes = 0
        # 合并方向 v2（2026-09-17）：merge_back 后 face_scheduler 设置的
        # last_merge_outcome 快照（direction/winner/loser/transferred_bytes/
        # home_flipped 等；stay 亦照实落账）。None = 尚无 merge 事务（F6
        # 销账：真 KVCacheManager __init__ 恒设该属性，完成路径直接读取）。
        self.merge_outcome = None
        # ---- 拼 batch 列车推进字段（闭式账本） ----
        self.remaining_chunks = math.ceil(
            self.prefill_tokens_to_process / p_chunk)
        self.prefill_tokens_completed = 0
        self.decode_tokens_consumed = 0   # 已物理完成 decode token 数
        self.current_decode_token = record["prefill_context_tokens"]
        self.prompt_tokens_processed = 0
        # 改法S2-B：剩余 prefill chunk 的 Roofline 负载聚合账本（整数
        # int；准入时一次置全量、列车核销时按已完 chunk 精确扣减、drain
        # 出队清零——仅 qp 会员期内有效，见 _compute_task_load_snapshot）。
        self.queued_chunk_load_ns = 0
        self.drain_block_ends = None      # drain 列车 barrier（joiner 触发门）
        self.prefill_complete_ns = None
        self.decode_start_ns = None
        self.estimated_arrival_ns = None
        self.admission_time_ns = None
        self.hbm_wait_ns = 0
        self.prefill_instance_index = None
        self.prefill_assignment_key = None
        self.prefill_instance_loads = ()
        self.prefill_hbm_feasible_instances = ()
        self.prefill_affinity_reason = None
        self.decode_instance_index = None
        self.history_source_instance_index = None
        self.history_transfer_bytes = 0
        # recompute 单口径下恒 0（截断通道已随 sidecar 移除）；字段保留
        # 以稳定 decision_log/manifest 输出 schema。
        self.history_tokens_discarded = 0
        self.history_location_before = None
        self.history_transfer = None
        self.history_evictions = ()
        # R17-1a'（2026-09-17 死通道钉死）：结构性恒空——准入 R1' 预约
        # （joint_reservation_context_tokens，face_scheduler.py 单源）覆盖
        # prefill 全动作足迹（stay/copy/recompute=history+input 整份、
        # remote-read=input），drain 时 delta ≤ 自有预约 ⟹ gap≡0 ⟹
        # _ensure_capacity 提前空返；2026-09-16 三方裁决（kimi 通道归因
        # + 本方代码级证实 + R17-7 探针 A2 运行级复现）。字段保留仅为
        # prefill 行 schema 稳定（见 _emit_admission 的同名空字段注释）。
        self.prefill_evictions = ()
        self.prefill_decode_transfer = None
        self.decode_evictions = ()
        # decode joiner 逐出支链的尾 watch id（_on_prefill_drain :1697
        # 登记、joiner plan 构图 :1236 消费；decode_evictions 为空时恒
        # None——现行 joint 模式三来源结构性恒空，见钉测文件的可达性刻画）。
        self.decode_eviction_watch_id = None
        self.completion_evictions = ()
        self.kv_location_after_completion = None
        self.kv_instance_after_completion = None
        self.admitted = False
        self.completion_ns = None
        self.completed = False

    def plan_dict(self) -> dict:
        """graph_batch_builder 消费的 plan 字段（KV 决策字段在准入/完成时
        已落账本）。"""
        return {
            "request_id": self.request_id,
            "session_id": self.session_id,
            "turn_index": self.turn_index,
            "queue_index": self.queue_index,
            "prefill_length": self.prefill_length,
            "decode_length": self.decode_length,
            "prefill_instance_index": self.prefill_instance_index,
            "decode_instance_index": (
                self.decode_instance_index or self.prefill_instance_index),
            "prefill_affinity_reason": self.prefill_affinity_reason,
            "history_tokens_before": self.history_tokens_before,
            "prefill_context_tokens": self.prefill_context_tokens,
            "final_context_tokens": self.final_context_tokens,
            "admission_time_ns": self.admission_time_ns,
            "hbm_wait_ns": self.hbm_wait_ns,
            "history_source_instance_index": self.history_source_instance_index,
            "history_transfer_bytes": self.history_transfer_bytes,
            "history_tokens_discarded": self.history_tokens_discarded,
            "history_location_before": self.history_location_before,
            "history_transfer": self.history_transfer,
            "history_evictions": self.history_evictions,
            "prefill_evictions": self.prefill_evictions,
            "decode_evictions": self.decode_evictions,
            "prefill_decode_transfer": self.prefill_decode_transfer,
            "completion_evictions": self.completion_evictions,
            "kv_location_after_completion": self.kv_location_after_completion,
            "kv_instance_after_completion": self.kv_instance_after_completion,
            # joint（§7.1：决策日志标签与开关状态一致）
            "joint_action": self.joint_action,
            "origin_home_instance": self.origin_home_instance,
            "joint_cost_ns": self.joint_cost_ns,
            "history_transfers": self.history_transfers,
            "merge_transfers": self.merge_transfers,
            # 规格书§二.3（2026-09-25）：prefill remote-read 前缀读流独立
            # 字段（图侧按同 frontier 分叉铺前缀读流/后缀池恢复两条腿；
            # 非前缀读流请求为 ()/0——键恒在，读者前向兼容）。
            "prefill_remote_read_bytes": self.prefill_remote_read_bytes,
            "prefill_remote_read_layers": (
                self.prefill_remote_read_plan["read_prefix_layers"]
                if self.prefill_remote_read_plan is not None else 0),
            "prefill_remote_read_transfers":
                self.prefill_remote_read_transfers,
        }


class Sh30OnlineScheduler(OnlineSchedulerBase):
    """strategy 变体：sh_3.0 三段式准入 + decode 同实例 + 三态 KV（关感知）
    + 迭代列车拼 batch（2026-08-22）。

    蓝本：已移除的离线 plan_face_requests。拓扑 / edge_free
    掩码 / KV 账本（与离线同一函数、同参数）在 __init__ 一次性构建，运行期
    策略输入全部来自这些 Python 账本。
    """

    # F6（2026-09-22 销账）：上方的 C11 类级软缺省块已删——全部 16 个
    # 属性在真 __init__ 无条件赋值，类级缺省只服务 __new__ 测试替身
    # （漏设静默绕过，如 HBM 端口登记丢失无销账）。替身现须自行补齐
    # __init__ 初值（漏设 = AttributeError，fail-loud）。

    def __init__(self, *, manifest, config, graph, digest_sink=None,
                 decision_log_sink=None, train_ledger_sink=None,
                 profile_sink=None, mode: str = "strategy",
                 sensing: bool = False,
                 defensive_reply_cache: bool = False,
                 sensing_query_sink=None, online_stats_sink=None,
                 ledger_sink=None):
        super().__init__(
            manifest=manifest,
            config=config,
            replay=None,
            digest_sink=digest_sink,
            mode=mode,
            sensing=sensing,
            decision_log_sink=decision_log_sink,
            profile_sink=profile_sink,
            defensive_reply_cache=defensive_reply_cache,
            sensing_query_sink=sensing_query_sink,
            online_stats_sink=online_stats_sink,
            ledger_sink=ledger_sink,
        )
        if mode != "strategy":
            raise ValueError("Sh30OnlineScheduler requires mode == 'strategy'")
        self.graph = graph
        p_chunk = int(config.prefill_chunk_size)
        if p_chunk <= 0:
            raise ValueError("online path requires an explicit positive p_chunk")
        self.p_chunk = p_chunk

        # ---- joint 三机制开关（设计方案 §7.1；env 一次读取 fail-closed，
        #      取代基底的 SH30_ABLATION——八组合消融统一经 JOINT_* 配置）。 ----
        if os.environ.get("SH30_ABLATION") is not None:
            # 开关已退役：任何显式值（含 "none"）即启动失败——陈旧脚本
            # 不得静默假装旧消融生效或旧缺省档（§8.2-3 扫描零命中口径）。
            raise ValueError(
                "SH30_ABLATION is retired in astra-sim-joint (any explicit "
                "value fails closed, including 'none'); use "
                "JOINT_ABLATION_COMBO / JOINT_CATEGORY_MODE / "
                "JOINT_SCHEDULER_MODE / JOINT_LAYER_POLICY")
        self.joint_config = parse_joint_config()
        self._joint_mode = self.joint_config.scheduler_mode
        # no_lb/no_affinity 消融旗标已随 SH30_ABLATION 退役删除：顺序参照
        # 由 scheduler_mode 的 load-first / affinity-first 承担（同一套
        # 候选枚举与计费）。

        # offline: face_scheduler.py 的 build_instances 同参。
        specs = tuple(
            FaceInstanceSpec(group.name, group.pg_name, group.ranks)
            for group in config.inference_groups
        )
        self.topology = build_instances(
            config.hardware, specs, require_equal_size=True)
        # joint：T（category_mode）与 E（layer_policy）注入 KV 账本；
        # 池速率来自硬件配置 remote-memory（E 的 r_j 估计与 J 的池路径
        # 计价共用；1 GB/s == 1 B/ns）。红线 2（R6-4，2026-09-14）：
        # 配置缺 remote-memory 段即拒绝启动——静默 1.0 GB/s 回退会把池
        # 路径计价错数量级（与 N10 自述失实同族教训），warning 不足以免
        # 静默；带宽/时延齐全性进启动校验。
        remote_memory = getattr(config, "remote_memory", None)
        if remote_memory is None:
            raise ValueError(
                "joint requires the remote-memory configuration section "
                "(bandwidth/latency); a missing section fails closed instead "
                "of silently falling back to a 1.0 GB/s pool rate")
        pool_bandwidth_gbps = float(remote_memory.remote_mem_bw_gbps)
        pool_latency_ns = int(remote_memory.remote_mem_latency_ns)
        # ---- R15 在线反馈通道（2026-09-14 接线，N10 修复）----
        # 链路流登记表（F-B 逐 shard 全路径）+ 池端口份额表：发射时点
        # 登记、完成事件（drain/completion/merge-watch）注销，决策时刻
        # 快照因果可见；池端口份额同时注入 E 内核（P1）与 J 计价。
        self._joint_flows = LinkFlowRegistry()
        self._pool_ports = _PoolPortRegistry()
        # C8（WP2，C6 桥遥测的 SH 半）：link_telemetry[] 解析状态——最新
        # 决策 epoch 窗口的逐链路实测有效速率 {link_id: B/ns}（served_
        # bytes/active_ns；_joint_cost_model 构造时按冻结接口喂 JCM，
        # JCM 侧消费 = 并行卡 C7）与 collective_coverage 翻转条件证据
        # （run 全程键在场 ∧ 窗口链无重叠回退 ∧ 至少一个样本）。
        self._link_telemetry_rates = {}
        # M1（2026-09-23 验收审计）：C++ 时间加权活跃流数（窗口
        # flow_active_ns 差分/active_ns 差分，含 collective）——
        # divisor_effective 与 AIMD 的物理一致分母（合计吞吐无法反推
        # 流数：满载链路合计≈容量与流数无关、下游瓶颈流会被误读为
        # 争用）。与 _link_telemetry_rates 同生命周期（同落账/同失效）。
        self._link_telemetry_flow_counts = {}
        self._link_telemetry_epoch_count = 0
        self._link_telemetry_sample_count = 0
        self._telemetry_absent_seen = False
        self._telemetry_window_broken = False
        self._telemetry_window_end_ns = 0
        self._telemetry_last_tick_ns = 0
        # A10'(a)/A12'（2026-09-22，§4.3 补遗）：served_bytes==0 ∧
        # active_ns>0 = 有活动无载荷字节的有效速率零窗口（uint64 差分
        # 计量无"整数截断伪影"，A12' 定性订正）——样本丢弃计数（决策
        # 日志遥测块披露位 _telemetry_coverage_decision；真零速率持续
        # 场景由 A9' 缺席失效兜底）。
        self._telemetry_zero_rate_dropped = 0
        # ---- C11（WP3c，2026-09-22）：链路 ∪ HBM 端口动作级准入配额 ----
        # off（缺省，F7）= tracker 不构造、全部配额调用点短路——既有臂
        # 决策序列与决策日志逐字节零漂移；static/aimd 构造 tracker（簿记
        # + 门）。aimd 的遥测喂入（observe_telemetry）经 _ingest_link_
        # telemetry（C8 交付段）驱动；δ_adm = DELTA_ADM_INITIAL_NS = 0
        # （A3' 终冻，C10 决议维持 0——challenger_flips 谓词与既有 argmin
        # 合取为恒等，见 _quota_candidate_filter 的 δ_adm=0 合取确认）。
        # 端口粒度裁定（C11）：port_id = instance_index——TP 全 rank 同账
        # 下逐 rank 端口 u_port ≡ 实例级流计数（每 TP 并行流对实例各
        # rank 端口各占 1 条），与 C5 port_snapshot 逐实例 schema 同构。
        # tracker 构造在 _joint_rates 之后（硬件速率派生 rho_eff/Q_init）。
        self._quota_tracker = None
        # 借还配对登记（守恒审计：收尾必须为空——全部流经生命周期释放
        # 点 settle；owner → rid 反查 + run 末泄漏 fail-closed）。
        self._quota_enrolled: dict[str, str] = {}
        # merge 义务预留账目（rid -> {state, source, target}；state ∈
        # reserved/adjudicated——service_done 裁决、merge_done 释放）。
        self._quota_merge_reserves: dict[str, dict] = {}
        self._quota_merge_reserves_created = 0
        # decode 相增量 enrollment 的 owner 序号（同请求多次 grow 事件
        # 各自独立 oneshot 流，#decode#{seq} 后缀防 owner 撞车）。
        self._quota_decode_owner_seq: dict[str, int] = {}
        # 弃赛守卫埋点（C11 步骤 5，为 C20 预置——本卡只埋点不判读）：
        # 四动作分列选中计数（recompute 仅 elected 口径、forced 按
        # no_history/quota_deferred/evicted_permanent 三成因单列——
        # 与 _recompute_selection_tier 枚举同步，L1/P2-1 补齐）+
        # quota 等待分列计数 + deferred 驻留时长样本（首 defer →
        # 最终准入）。
        self._joint_action_selection_counts = {
            "stay": 0, "copy": 0, "remote-read": 0,
            "recompute_elected": 0,
            "recompute_forced_no_history": 0,
            "recompute_forced_quota_deferred": 0,
            "recompute_forced_evicted_permanent": 0,
        }
        self._quota_deferred_wait_counts = {
            WAIT_CAPACITY: 0, WAIT_QUOTA_LINK: 0, WAIT_QUOTA_PORT: 0}
        self._quota_deferred_dwell_ns: list[int] = []
        # 决策时延埋点（C11 步骤 4：全候选 × 4 动作 × 逐资源检查的
        # Python 侧增量记录——纯决策段墙时 + 配额判据层墙时分开计）。
        self._admission_decision_wall_ns_total = 0
        self._admission_decision_wall_ns_max = 0
        self._admission_decision_count = 0
        self._quota_verdict_wall_ns_total = 0
        self._quota_verdict_candidate_checks = 0
        self._quota_admit_events = 0
        self._quota_release_events = 0
        # O12（2026-09-23 终轮审计）：oneshot 入册披露事件计数
        # （kind=quota_oneshot_overflow——decode 相深占用降級 + O2 准入
        # 逐出支链降级两发射点；修前深占用记账失明：事件落决策日志但
        # 不进 run 级配额指标汇总）。收尾行 quota_events 披露。
        self._quota_oneshot_overflow_events = 0
        # AIMD 动作披露聚合（observe_telemetry 返回值的紧凑计数；逐
        # epoch 全量行不落日志——体积控制，收尾行汇总）。
        self._quota_aimd_action_counts: dict[str, int] = {}
        # A5'/B3（C7 移交 rider）：LinkId → (src, dst) 端点键换算表——
        # C++ MultiDimTopology::connect_dimension 确定性枚举（逐维升序
        # src、connect(src, src+stride, bidirectional) 顺次正/反两条；
        # dims = network.yml npus-count = [mesh_cols, mesh_rows]，C++ 平
        # 层 rank = col + mesh_cols*row ≡ Python rank 空间）。首样本惰性
        # 构建；换算后 JCM divisor_effective 的 max 合并在生产路径真实
        # 生效（整型键仅剩未知 id 的退化披露路径）。
        self._telemetry_link_id_map: dict[int, tuple[int, int]] | None = None
        # C2（WP1b，2026-09-22）：实例 HBM 端口流注册表（F4 u_port 除数
        # 的执行器侧来源）——noc_migrate 在途流登记端点端口（源 = home
        # 读腿恒有；目标 = exec 写腿，copy 留存写 / merge 落点写 /
        # remote-read credit 到达写——A4' 补价后计价与执行同腿型）；
        # 活跃 decode 消费流经因果负载视图 provider 派生（闲置为 0，
        # 不假设恒 1），不经登记通道（杜绝漏更新）。池路径
        # （remote_load/remote_store）不进本表（F3：池端口口径独占）。
        self._hbm_ports = HbmPortFlowRegistry()
        self._hbm_ports.attach_active_decode_provider(
            self._hbm_active_decode_streams)
        self._rank_to_instance = {
            rank: instance.index
            for instance in self.topology.instances
            for rank in instance.ranks}
        self.kv_manager = KVCacheManager(
            self.topology,
            config.model,
            category_mode=self.joint_config.category_mode,
            layer_policy=self.joint_config.layer_policy,
            pool_bandwidth_gbps=pool_bandwidth_gbps,
            pool_latency_ns=pool_latency_ns,
            pool_divisor_fn=self._pool_port_divisor,
        )
        self._instance_edge_ports = {
            instance.index: tuple(sorted({
                self.kv_manager.nearest_edge(rank)
                for rank in instance.ranks}))
            for instance in self.topology.instances
        }
        # joint J 组件（§6/§13.2）：硬件速率、链路流登记表、在线因子、
        # 因果时域估计器（无 oracle：decode_length/final_context_tokens
        # 不进任何决策输入）。pool_port_gbps 直取配置值（红线 2 已在上方
        # fail-closed 校验存在性，无 1.0 静默回退）。
        self._joint_rates = JointHardwareRates.from_gbps(
            noc_link_gbps=config.hardware.d2d_bandwidth_gbps,
            pool_port_gbps=pool_bandwidth_gbps,
            local_hbm_gbps=config.hardware.local_hbm_bandwidth_gbps,
            d2d_latency_ns=int(config.hardware.d2d_latency_ns),
            pool_latency_ns=int(pool_latency_ns or 0),
        )
        self._joint_factors = ServiceFactors()
        # C11（WP3c）：配额 tracker（off = None——全部配额调用点短路，
        # 既有臂决策序列零漂移；static/aimd 构造。端口粒度 = 实例级，
        # 见 __init__ 上方 C11 块注释）。δ_adm = 0（A3' 终冻）。
        if self.joint_config.quota_enabled:
            self._quota_tracker = LinkQuotaTracker(
                mode=self.joint_config.quota_mode,
                noc_link_bytes_per_ns=(
                    self._joint_rates.noc_link_bytes_per_ns),
                local_hbm_bytes_per_ns=(
                    self._joint_rates.local_hbm_bytes_per_ns),
                delta_adm_ns=DELTA_ADM_INITIAL_NS,
            )
        self.edge_free_mask = edge_free_instance_mask(
            self.topology, self.kv_manager.edge_ranks)
        self.edge_mask = edge_instance_mask(
            self.topology, self.kv_manager.edge_ranks)
        self.instances = [
            _OnlineInstanceState(index=i)
            for i in range(len(self.topology.instances))
        ]
        # N12（R15-4，2026-09-14）：decode 负载标定在线化——active_decode
        # 剩余估计与 horizon 冷启动全部由 CausalHorizonEstimator 因果供给
        # （session 已完成均值 → run 已完成均值 → 冷启动 1 token 物理常数
        # + "乐观可能"披露）。全 trace decode 均值常数不再进入任何决策
        # 输入（总纲 §13.1/§13.2 明文；本条推翻 2026-08-15"禁止在线增量
        # mean"裁决在 joint 语境的适用——其语境是 sh 基底工程常数与在线/
        # 离线对拍纪律，joint 中该常数已成研究方法决策输入）。离线蓝本
        # 保留 config.source_average_decode_length 仅作对拍锚（不消费）。
        self._joint_horizon = CausalHorizonEstimator(
            cold_start_default_tokens=1)
        self.hardware = config.hardware
        self.model = config.model
        self._prefill_task_cache = {}
        self._decode_task_load_cache = {}
        # 高基数 context/token 输入的纯 Roofline memo 固定有界；逐出后按
        # 原估算器精确重算，不参与或改变任何策略语义。
        self._task_load_cache_capacity = _TASK_LOAD_CACHE_CAPACITY

        self.runtimes = [
            _OnlineRequestRuntime(record, p_chunk)
            for record in sorted(
                manifest["requests"], key=lambda item: item["queue_index"])
        ]
        self.runtime_by_request_id = {
            runtime.request_id: runtime for runtime in self.runtimes}
        self._runtime_index = {
            runtime.request_id: index
            for index, runtime in enumerate(self.runtimes)}
        by_turn = {}
        for runtime in self.runtimes:
            by_turn[(runtime.session_id, runtime.turn_index)] = runtime
        self.next_request = {
            runtime.request_id:
                by_turn.get((runtime.session_id, runtime.turn_index + 1))
            for runtime in self.runtimes
        }
        # graph builder 的 interval gate 发射需要下一 turn 的 plan 字段
        # （queue_index/hbm_wait_ns/request_id）；strategy 模式 hbm_wait_ns
        # 在准入时落账本，经 resolver 取实时值。
        self.graph.set_next_plan({
            runtime.request_id:
                (self.next_request[runtime.request_id].request_id
                 if self.next_request[runtime.request_id] else None)
            for runtime in self.runtimes
        })
        self.graph.set_plan_resolver(
            lambda request_id: self.runtime_by_request_id[
                request_id].plan_dict())

        self.arrival_heap = []
        self._sequence = 0
        # 自查 A：已到达请求数（runtimes 槽位完成后置 None，无法从槽位
        # 区分"未到达"与"已完成"——死锁守卫判"未来到达存在"用本计数）。
        self._arrived_request_count = 0
        # offline: face_scheduler.py 的 pending_admissions FIFO。
        self.pending_admissions = deque()
        # 改法D：KV 账本纪元重试门（face capacity_epoch 的调度器全局版——
        # sh 系列 try_admit 的全部 False 判据是"全账本 HBM 可行性纯函数
        # ∧ 静态掩码"，实例账本（qp/busy/active_decode）只影响选哪个、
        # 不影响能不能）。_kv_ledger_epoch 在 9 个 KV 变更点后各 bump 一次；
        # _admit_attempt_epoch 记录各 pending 条目上次失败时的纪元，
        # 纪元未变则该条目本批跳过重试（重试必返同样的 False）。
        self._kv_ledger_epoch = 0
        self._admit_attempt_epoch = {}
        # R2（2026-09-14）：准入失败折叠计数（D5/P4——同一请求同一失败
        # 分类：首条全表 + 后续紧凑计数行，防"延迟 × 纪元重试"乘积下
        # 失败日志体积逼近成功日志）。
        self._admission_failure_state: dict[str, dict] = {}
        # C3b（2026-09-22 到达合同迁移）：merge 尾 watch id → merge 流
        # 注销义务（R15）+ merge_done 事件侧披露行所需最小账目。下一轮
        # 到达 alarm 已在响应完成点（service_done，REQUEST_COMPLETE 交付
        # tick）随 _complete_requests 排定，不再由 merge watch 重排；本表
        # 非空仍表示图侧存在待交付的 merge watch（死锁守卫逃逸项 (a) 的
        # "必有图交付"事件源语义保留）。**名称沿用 _pending_merge_alarms**
        # （R11 历史名——当时承载到达 alarm 重排；语义收窄后为免跨车道
        # 测试文件联动改名而保留，条目内不再有 interval/envelope）。
        self._pending_merge_alarms: dict[str, dict] = {}
        # KV eviction branches are independent of request drain/completion.
        # Each batch_train_evict_* watch owns only the flows and quota entry
        # for its physical branch; scheduled=False entries are decode-joiner
        # branches registered at drain and awaiting their first train batch.
        self._pending_eviction_watches: dict[str, dict] = {}
        self._eviction_watch_seq: dict[tuple[str, str], int] = {}
        # R14：停滞会话登记（instance_index -> set[runtime]），唤醒 pass
        # O(停滞数) 扫描（正常路径零成本）。
        self._stalled_by_instance: dict[int, set] = {}
        # 影子验证开关（SH_ADMIT_GATE_VERIFY=1）：门跳过的条目仍完整评估
        # 并断言必返 False——验证跑零收益、全检查；不设或非"1"则正常运行。
        self._admit_gate_verify = (
            os.environ.get("SH_ADMIT_GATE_VERIFY") == "1")
        # 改法S2 影子验证开关（SH_SNAPSHOT_VERIFY=1）：快照纪元缓存命中
        # 与 qp 聚合账本两条路径都仍按原始逐 chunk while 现算完整重算并
        # 逐字段断言相等——验证跑零收益、全检查；不设或非"1"则正常运行
        # （生产走缓存）。开关姿态与命名对齐 SH_ADMIT_GATE_VERIFY（改法D）。
        self._snapshot_verify = (
            os.environ.get("SH_SNAPSHOT_VERIFY") == "1")
        self.completed_requests = 0
        # §7.3 ready frontier：非忙（无在飞列车）且有排队工作的实例集合。
        self._ready_frontier = set()
        # 拼 batch 列车台账（§7.3 不变量断言输入）：每次列车发射一行，
        # 由 online_service 落 bridge 目录 train_ledger.jsonl（审计产物）。
        # M3 流式落盘（2026-08-23）：提供 train_ledger_sink 时行即写即
        # 弃，不驻留本列表；缺省 None = 兼容旧路径（行仍缓冲）。
        self.train_ledger_sink = train_ledger_sink
        self.train_ledger_rows = []
        # T_max 列车长度上限(§3.2.8/§7.4 治理旋钮 + A2 逐迭代 oracle):
        # 交付默认 = 8(2026-08-22 §7.4 A2 对拍裁决,sh_1.0 母本统一);
        # 0 = 不设限(oracle/灵敏度复跑用)。
        self._train_max_iter = int(
            os.environ.get("SH_TRAIN_MAX_ITER", "8") or 0)
        self._train_instance_index = {}
        # WP9 首 token 首步批拆分（2026-08-26）：batch_train_<id>_first_step
        # 唤醒 id -> 实例索引。首步批（无请求级 watch）完成时其唤醒 watch
        # fire 经 PREFILL_DRAIN 通道送回，本表区分"自己发射的首步唤醒"与
        # 真哨兵信号；唤醒只作余量批的交付边界，不进任何决策/核销路径。
        self._pending_first_steps = {}

    # ------------------------------------------------------------- 策略 --

    def run_variant_policy(self, delta) -> None:
        """决策顺序逐行对应离线事件循环：completion 批（priority 0，含
        completion_order 处理）先于 arrival 批（priority 1），最后
        admit + 列车发射（:4060-4062）。

        拼 batch 列车账本（§3.2 边界原子提交顺序）：先核销已完成列车
        （核验 train_id + membership_digest → 冻结成员推进 token → 退出
        成员移出 active_decode → 推进 prefill chunk），再处理 drain/
        完成/到达，最后冻结并发射各空闲实例的下一列车。"""
        tick = delta["tick"]
        # C8 步骤 3：桥请求顶层 link_telemetry[] 解析（C6 交付的逐链路
        # 窗口差分数组；键缺省 = --link-telemetry 关闭——本 run 遥测不完
        # 备，collective_coverage 不翻）。每已应用交付恰一次（幂等重放在
        # 基类 sequence 门早退，不重复采样）。
        self._ingest_link_telemetry(delta)
        # ---- merge 尾 watch 交付（C3b，2026-09-22 到达合同迁移）。 ----
        # 09-21 新合同（实验思路 §三.4 / 设计方案 §2.3 / 设计文档 §2.4）：
        # 下一轮到达 = 上一轮向外完成响应的时刻（service_done，本仓事件侧
        # = REQUEST_COMPLETE 交付 tick）+ interval——thinktime 自响应完成
        # 起算，alarm 在 _complete_requests 排定、与本回调无关。merge watch
        # 交付只承担 ①merge 在途流注销（R15）与 ②merge_done 事件侧披露行
        # （F11 分报配套）。到达后的数据依赖门控由图侧承载（interval gate
        # 前递锚 merge 尾标记——保守 merge_done 整体口径，块级门控留后续）。
        # watch id 走 batch_train_merge_ 前缀（基类 _settle_completions 对
        # batch_train_ 命名空间跳过请求核销；本变体在哨兵路由前拦截）。
        merge_done_ids = []
        eviction_done_ids = []
        # ---- completion 批（offline: face_scheduler.py）----
        drained = []
        completed_now = []
        sentinel_trains = []
        for group in delta["completed_groups"]:
            stage = group["stage"]
            request_id = group["request_id"]
            if request_id.startswith("batch_train_evict_"):
                eviction_done_ids.append((request_id, stage))
                continue
            if request_id.startswith("batch_train_merge_"):
                merge_done_ids.append(request_id)
                continue
            if request_id.startswith("batch_train_"):
                sentinel_trains.append(request_id)
                continue
            if stage == STAGE_PREFILL:
                drained.append(request_id)
            elif stage == STAGE_DECODE:
                completed_now.append(request_id)
            elif stage == STAGE_REQUEST:
                # REQUEST_COMPLETE 与 DECODE_COMPLETION 同 tick 交付；完成
                # 处理在 _complete_requests（下）一并处理。
                continue
            else:
                raise ValueError(
                    "unknown completion stage {!r} for request {!r}".format(
                        stage, request_id))
        # WP9 首 token 拆分（2026-08-26）：batch_train_*_first_step 唤醒
        # 信号是"自己发射的首步批"的交付回声——无操作（不决策/不记账/
        # 不写 decision_log），从哨兵核销通道剥离；余量批在
        # _plan_and_emit_trains 的 busy 分支发射（本交付或任一后续交付，
        # 余量节点经依赖边排在首步节点之后，早发不改物理序）。首步批的
        # commit ack 走基类协议记账（ack_count/幂等门），变体侧零动作。
        sentinel_trains = [
            train_id for train_id in sentinel_trains
            if not self._consume_first_step_wakeup(train_id)
        ]
        for watch_id in merge_done_ids:
            self._on_merge_done(watch_id, tick)
        for watch_id, stage in eviction_done_ids:
            self._on_eviction_watch(watch_id, stage, tick)
        self._finalize_completed_trains(
            drained, completed_now, sentinel_trains, tick)
        for request_id in drained:
            self._on_prefill_drain(request_id, tick)
        if completed_now:
            self._complete_requests(completed_now, tick)
        # ---- arrival 批（offline: face_scheduler.py）----
        for arrival in delta["arrivals"]:
            self._push_arrival(arrival, tick)
        self._drain_arrival_heap(tick)
        # ---- 准入/列车发射 pass（offline: face_scheduler.py）----
        self._admit_pass(tick)

    # ------------------------------------------------------ 列车账本 --

    def _finalize_completed_trains(self, drained, completed_now,
                                   sentinel_trains, tick: int) -> None:
        """核销本交付中标记 watch 已 fire 的列车（§3.2 原子提交的前半；
        照母本 sh_1.0 定型版移植，账本对象换为本仓 runtime）。

        drain/exit 标记节点是列车体后的最末真实节点，任一标记 watch fire
        即列车物理主体完成。同一列车不同成员的标记 watch 可能跨 tick
        fire、事件拆到不同交付——首个信号执行核销（账本推进恰一次），
        后续信号只清已核销列车的 pending 集合（幂等）；信号不属于任何
        在飞/已核销列车即陈旧完成错配 fail-closed。推进量全部闭式
        （每成员 participation 次 token，无逐 token 循环）。"""
        signaled = {}
        for request_id in drained:
            runtime = self.runtime_by_request_id[request_id]
            signaled.setdefault(
                runtime.prefill_instance_index, set()).add(request_id)
        for request_id in completed_now:
            runtime = self.runtime_by_request_id[request_id]
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
                    # 该列车首个信号到达：核销（账本推进恰一次），残余
                    # 信号（drain/exit 标记跨 tick fire）登记待收。
                    iterations = train["iterations"]
                    # R15-2（2026-09-14）：ServiceFactors 纯样本采集——
                    # 纯服务段 = 核销 tick − max(发射 tick, 上次核销 tick)
                    # （排除列车前排队污染）；基数 = 本列车 roofline 闭式
                    # 账本（chunk 负载 + 逐 decode span 单步负载）。混合
                    # 列车（chunk + decode / 含 joiner 迁移）样本不可因果
                    # 分离，跳过（纯度约束：排队/传输不得入分子）。
                    self._observe_service_factors(state, train, tick)
                    for request_id, participation in train["members"]:
                        runtime = self.runtime_by_request_id[request_id]
                        runtime.decode_tokens_consumed += participation
                        runtime.current_decode_token += participation
                        # joint（§3.1）：decode KV 按实际消费进展因果增长
                        # （写入前经 T+E 真实释放准备空间；R14：容量类
                        # 失败转停滞/唤醒，不再 fail-closed）。
                        self._joint_grow_decode(runtime, instance_index)
                    # remote-read credit（§4.3.2 v3，R-6）：Tj 核销边界
                    # 注销本列车切片的 R15 在途流（owner = rid#decode#{j}
                    # 逐切片一键；先例 = 准入相流 drain 边界注销——其
                    # 消费栅栏（体块 arm 门/barrier/链序）已物理通过）。
                    # 不在此注销会让后续列车规划的链路除数躺着 j-1 条
                    # 已完成陈旧流，J 决策被系统性污染且不报错。
                    for request_id, slice_index in (
                            train.get("remote_credit_slices") or {}).items():
                        self._finalize_remote_credit_train_flows(
                            self.runtime_by_request_id[request_id],
                            slice_index)
                    for request_id in train["exit_set"]:
                        runtime = self.runtime_by_request_id[request_id]
                        if runtime not in state.active_decode_lookup:
                            raise RuntimeError(
                                "exiting member {} is not in the decode "
                                "batch".format(request_id))
                        state.active_decode_lookup.discard(runtime)
                        state.active_decode.remove(runtime)
                    # 改法S2-B：核销 chunk 的负载逐键求和与核销推进合流——
                    # span 键 = history + 累计（含本 chunk），与 _plan_train
                    # 冻结 span 及 _reference_task_load_snapshot 的 queued
                    # while 现算对同一 chunk 序列取同一 memo 值（整数和差
                    # 逐位等价），队列头成员聚合账本同步扣减。列车 chunk
                    # 全部属队列头（_plan_train 只取 qp_head）。
                    head_runtime = None
                    processed_before = 0
                    finalized_chunk_load_ns = 0
                    instance_size = self.topology.instance(
                        instance_index).size
                    for (request_id,
                         chunk_tokens) in train["prefill_chunk_tokens"]:
                        runtime = self.runtime_by_request_id[request_id]
                        if head_runtime is None:
                            head_runtime = runtime
                            processed_before = runtime.prefill_tokens_completed
                        finalized_chunk_load_ns += (
                            self._prefill_chunk_task_load_ns(
                                instance_size=instance_size,
                                chunk_tokens=chunk_tokens,
                                context_tokens=(
                                    runtime.joint_span_base_context
                                    + processed_before + chunk_tokens)))
                        processed_before += chunk_tokens
                        runtime.prefill_tokens_completed += chunk_tokens
                        runtime.remaining_chunks -= 1
                    if head_runtime is not None:
                        head_runtime.queued_chunk_load_ns -= (
                            finalized_chunk_load_ns)
                    state.iteration_count += iterations
                    state.in_flight_train = None
                    state.last_train_finalize_tick = tick
                    # M4 核销即删（2026-08-23）：列车核销后其 train_id→
                    # 实例索引条目即死重（哨兵条目已在信号路由处弹出，
                    # 此 pop 对其为幂等 no-op；全仓 grep 证实核销后无读者）。
                    self._train_instance_index.pop(train["train_id"], None)
                    state.finalized_trains.append({
                        "train_id": train["train_id"],
                        "pending": (train["signal_set"] - inflight_hits),
                    })
                    consumed |= inflight_hits
                    pending_signals -= inflight_hits
                    self._refresh_frontier(state)
                    # 改法S2-A：列车核销改写本实例全部快照输入类别（成员
                    # decode 进度/active_decode 退出成员/队列头 chunk 进度/
                    # in_flight_train 清空）——实例纪元 +1，快照缓存失效。
                    self._bump_instance_epoch(state)
            # 已核销列车的后续信号对账（幂等清 pending，清空即出列表）。
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

    def _plan_train(self, state):
        """冻结实例的下一列车成员快照（§3.2 构造规则；照母本移植）。

        列车终点 = 下一个不可预测事件之前的最后一个完整迭代：队列头
        prefill 的 drain 迭代（剩余 chunk 数，先验）或全部 decode 工作
        耗尽（无 prefill 工作时 = max 剩余 token；默认不设 T_max，§3.2.8）。
        成员退出不是列车边界（先验）：退出成员在列车内挂 exit 标记。
        每迭代至多 1 个 prefill chunk（FCFS 队列头）；chunk 之间不互拼。
        返回 None = 实例无工作。"""
        qp_head = None
        for runtime in state.qp:
            # drain 事件跨交付未达的头部（remaining_chunks 已在列车核销
            # 时清零，drain 决策事件尚在途中）不提供 chunk 工作；其后续
            # 请求的 chunk 物理上已可开始（头部 prefill 主体已完成）。
            if runtime.remaining_chunks > 0:
                qp_head = runtime
                break
        members = []
        for runtime in state.active_decode:
            if runtime.decode_stalled:
                # R14（2026-09-14）：停滞成员暂缓列车参与（KV 增长已
                # 停滞；让行共存成员，避免队头阻塞）；唤醒后在下一列车
                # 边界回归。停滞成员必有剩余 token（consumed 只经列车
                # 推进），不存在"停滞且已完成"的悬挂态。
                continue
            remaining = (
                runtime.decode_length - runtime.decode_tokens_consumed)
            if remaining <= 0:
                raise RuntimeError(
                    "decode member {} has no remaining tokens".format(
                        runtime.request_id))
            members.append((runtime.request_id,
                            runtime.prefill_context_tokens,
                            runtime.decode_tokens_consumed, remaining))
        natural_iterations = None
        if qp_head is None:
            if not members:
                return None
            natural_iterations = max(
                remaining for _, _, _, remaining in members)
        else:
            natural_iterations = qp_head.remaining_chunks
            if natural_iterations <= 0:
                raise RuntimeError(
                    "prefill queue head {} has no remaining chunks".format(
                        qp_head.request_id))
        iterations = natural_iterations
        capped = False
        if (self._train_max_iter and iterations > self._train_max_iter):
            iterations = self._train_max_iter
            capped = True
        # span 展开（成员×迭代；KV 逐迭代 +1，退出截断，无 padding）。
        # 聚合对 span 求和与顺序无关，故按 [chunk 序列]+[成员连续段]
        # 平铺（总 span 数与旧 request-aggregated 同级，非新增热路径）。
        pass_spans: list[tuple[int, int]] = []
        chunk_records = []
        chunk_spans = []
        if qp_head is not None:
            work = qp_head.prefill_tokens_to_process
            completed = qp_head.prefill_tokens_completed
            # R13：span 上下文基 = 动作口径（跨实例 recompute 从 0 物化
            # 工作副本；其余动作 = 真实历史基数）。
            history = qp_head.joint_span_base_context
            for _ in range(iterations):
                chunk_tokens = min(self.p_chunk, work - completed)
                if chunk_tokens <= 0:
                    raise RuntimeError(
                        "prefill queue head {} ran out of work inside the "
                        "planned train".format(qp_head.request_id))
                span = (chunk_tokens, history + completed + chunk_tokens)
                pass_spans.append(span)
                chunk_spans.append(span)
                chunk_records.append((qp_head.request_id, chunk_tokens))
                completed += chunk_tokens
        member_parts = []
        exit_members = []
        for request_id, context, consumed, remaining in members:
            participation = min(remaining, iterations)
            pass_spans.extend(
                (1, context + consumed + step)
                for step in range(1, participation + 1))
            member_parts.append((request_id, participation))
            if participation >= remaining:
                exit_members.append(request_id)
        # 队列头在列车内完成其全部剩余 chunk（列车长度 = 头部剩余 chunk
        # 数）⇒ 列车终于头部 drain 迭代（drain 是先验已知的列车边界）。
        # T_max 截断时头部未必 drain —— 重算(先验边界)。
        drain_members = [qp_head.request_id] if (
            qp_head is not None and not capped) else []
        head_first_chunk = (
            qp_head is not None
            and qp_head.prefill_tokens_completed == 0)
        state.train_seq += 1
        train_id = "batch_train_i{}_{}".format(state.index, state.train_seq)
        sentinel = bool(capped and not drain_members and not exit_members)
        signal_set = set(exit_members) | set(drain_members)
        if sentinel:
            signal_set.add(train_id)
        snapshot = json.dumps(
            {
                "train_id": train_id,
                "iterations": iterations,
                "members": member_parts,
                "exits": exit_members,
                "drains": drain_members,
                "chunks": chunk_records,
                "capped": capped,
            },
            sort_keys=True,
        )
        return {
            "train_id": train_id,
            "iterations": iterations,
            "members": member_parts,
            "exit_members": exit_members,
            "exit_set": set(exit_members),
            "signal_set": signal_set,
            "sentinel": sentinel,
            "drain_members": drain_members,
            "drain_set": set(drain_members),
            "prefill_chunk_tokens": chunk_records,
            "prefill_chunk_spans": chunk_spans,
            "head_request_id": qp_head.request_id if qp_head else None,
            "head_first_chunk": head_first_chunk,
            "pass_spans": pass_spans,
            "membership_digest": hashlib.sha256(
                snapshot.encode()).hexdigest(),
        }

    def _plan_and_emit_trains(self, tick: int) -> None:
        """为每个空闲且有工作的实例冻结并发射下一列车（§3.2 原子提交
        的后半：KV 就绪成员（pending_decode_ready，迁移随加入列车发射，
        物理先于列车体）进入 active_decode → 冻结成员 → 发射）。

        原"批齐发-等完成"骨架（_start_ready_iterations 发射部分）的列车
        化形态：一趟列车 = 队列头 chunk × 迭代 + 全部 active_decode 成员
        （与旧骨架 qp[0] prefill 整段 + 全部 decode 整段的批齐发口径一
        致），等待语义由 in_flight_train 替代 busy。busy 门 = 一个列车
        在飞（§3.2）：在飞实例跳过，不重复发射；WP9 拆分列车的余量批
        在 frontier 循环前的独立 pass 发射（busy 实例不在 _ready_
        frontier 内——_refresh_frontier 对在飞实例恒摘除，余量挂起期间
        亦然；首步唤醒到达后的首个决策边界即冲批）。"""
        # WP9 余量冲批 pass：两段式发射的后半（余量体 + drain/exit/哨兵
        # 标记 + end barrier）。发射后 in_flight_train 语义恢复整列口径
        # （标记 watch 全部在本批注册）；in_flight_train 挂起状态与快照
        # 输入不变，不 bump 实例纪元、不 refresh frontier（busy 不变）。
        for state in self.instances:
            if (state.in_flight_train is not None
                    and state.first_step_remainder is not None):
                self._emit_train_remainder(state, tick)
        for instance_index in sorted(self._ready_frontier):
            state = self.instances[instance_index]
            if state.in_flight_train is not None:
                continue  # busy 门：一个列车在飞
            if not (state.qp or state.active_decode or
                    state.pending_decode_ready):
                continue
            joiners = []
            if state.pending_decode_ready:
                joiners = list(state.pending_decode_ready)
                state.pending_decode_ready.clear()
                state.active_decode.extend(joiners)
                for runtime in joiners:
                    state.active_decode_lookup.add(runtime)
                    self._note_emitted(runtime.request_id, STAGE_DECODE)
                    self._ledger_issue(
                        runtime.request_id, tick, STAGE_DECODE, state.index)
                # 改法S2-A：decode 成员入批（active_decode 变更）——实例
                # 纪元 +1，快照缓存失效。
                self._bump_instance_epoch(state)
            plan = self._plan_train(state)
            if plan is None:
                continue
            self._emit_train(state, plan, joiners, tick)

    def _emit_train(self, state, plan, joiner_runtimes, tick: int) -> None:
        """把冻结的列车计划交给构图器发射，注册 drain/exit 标记 watch，
        并挂起 in_flight_train（busy 门 = 一个列车在飞）。

        WP9（2026-08-26）：列车含 debut 成员且拆分开启、迭代数 >= 2 时
        两段式发射——本交付只发首步批（迁移/起始标记/首迭代体/first_
        token 标记/唤醒标记），drain/exit/哨兵 watch、块末账本与正常
        train_ledger 行移至余量批（_emit_train_remainder）；成员选择/
        排序/KV 动作/挂点语义全部不变。"""
        joiner_plans = []
        for runtime in joiner_runtimes:
            joiner_plan = runtime.plan_dict()
            joiner_plan["prefill_drain_block_ends"] = dict(
                runtime.drain_block_ends or {})
            watch_id = getattr(runtime, "decode_eviction_watch_id", None)
            if watch_id is not None:
                joiner_plan["decode_eviction_watch_id"] = watch_id
            joiner_plans.append(joiner_plan)
        # remote-read credit 逐列车切片（列车规划期；R15 登记钩子 = 切片
        # 创建点，§4.3.2）。None = 本列车无 remote-credit 成员（credit
        # 关闭或成员不涉）——发射与 v1 逐字节一致。
        remote_credit = self._plan_remote_credit_slices(
            plan, joiner_runtimes, joiner_plans)
        # R15-2：因子样本纯度标记（joiner 迁移/partial 后缀恢复门会向
        # 列车 span 混入传输段，含任一者的列车不进 ServiceFactors 样本）。
        plan["emit_tick"] = tick
        plan["had_joiners"] = bool(joiner_runtimes)
        head_runtime = (
            self.runtime_by_request_id.get(plan.get("head_request_id"))
            if plan.get("head_request_id") else None)
        plan["suffix_gated"] = bool(
            plan.get("head_first_chunk")
            and head_runtime is not None
            and head_runtime.history_location_before is not None
            and head_runtime.history_location_before.location
            == "partial_hbm_remote")
        # M5（kimi 复审）：首 chunk 门控 NoC 传输（copy 前缀/REMOTE 池
        # 恢复）同样向列车 span 混入传输段——与 partial 后缀恢复门同款
        # 纯度排除（local_hit 无传输不计；remote-read 无历史传输、其
        # 读流门控 decode 列车，在 decode 分支按成员动作排除）。
        plan["history_transfer_gated"] = bool(
            plan.get("head_first_chunk")
            and head_runtime is not None
            and any(
                transfer.kind != "local_hit" and transfer.total_bytes > 0
                for transfer in (head_runtime.history_transfers or ())))
        # N5（2026-09-23 复核审计5）：上一轮 merge 尾门（R11(ii)——下一
        # 轮 interval gate 锚定上一轮 merge_done 标记节点，graph 层
        # batch_builder :2532-2539）——merge watch 未交付时本列车 emit
        # 后物理计算被 hold 至 merge 完成 + interval，span 混入门控等待
        # 且该样本会初始化 γ/更新双因子通道——纯度排除（与 joiner/恢
        # 复门/传输门同哲学：不可因果分离不收样）。
        # O3（2026-09-23 终轮审计，判据扩域到"本实例"粒度）：门控等待
        # 污染源两类——① 未交付 merge watch：R11(ii) interval gate 按
        # **实例 frontier** 锚定（非按 session），同实例任意 session 的
        # 未交付 merge watch 都 hold 本实例下一列车（原"同 session"匹配
        # 漏跨 session 污染）；② frontier 未交付完成批尾段：完成批池写
        # 尾（completion_evictions / merge 传输的 remote_store）在
        # graph.pending_store_tails 在案（GB :411 登记 / :1735-1806
        # 消费），本实例边缘 rank 有在案尾部 ⇒ 后续 restore/计算的
        # emit→启动段同样等尾段交付。实例归属：edge_rank ∈ 本实例
        # ranks（topology 直查，不依赖 _rank_to_instance 替身覆盖）。
        instance_ranks = self.topology.instance(state.index).ranks
        plan["merge_tail_gated"] = bool(
            any(
                alarm.get("instance_index") == state.index
                for alarm in self._pending_merge_alarms.values())
            or any(
                edge_rank in instance_ranks
                for tails in self.graph.pending_store_tails.values()
                for (edge_rank, *_rest) in tails))
        stage = "decode" if (plan["members"] or joiner_runtimes) else "prefill"
        prefill_start_member = None
        first_chunk_member = None
        if plan.get("head_first_chunk"):
            # pstart 标记（prefill_start 锚点）与 partial 恢复 suffix 门
            # 挂接（first_chunk_member）同条件：队列头首 chunk 在本列车。
            member = {"request_id": plan["head_request_id"]}
            prefill_start_member = member
            first_chunk_member = member
        train_plan = {
            "train_id": plan["train_id"],
            "instance_index": state.index,
            "stage": stage,
            "joiners": joiner_plans,
            "pass_spans": plan["pass_spans"],
            "iterations": plan["iterations"],
            "sentinel": plan["sentinel"],
            "prefill_start_member": prefill_start_member,
            "first_chunk_member": first_chunk_member,
            "drain_members": [{"request_id": request_id}
                              for request_id in plan["drain_members"]],
            "exit_members": [{"request_id": request_id}
                             for request_id in plan["exit_members"]],
        }
        if remote_credit is not None:
            train_plan["remote_credit"] = remote_credit
        first_token = self._first_token_plan(plan, joiner_runtimes)
        if first_token is not None:
            train_plan["first_token"] = first_token
            if remote_credit is not None:
                # WP9 拆分与 credit 体块化正交：首步批/余量批各自的体块
                # 规格（首步 = 迭代 1；余量沿 K 对齐切块），门按区间重叠
                # 取并集（I2 泛化式）。
                first_token["first_body_blocks"] = remote_credit[
                    "split_first_blocks"]
                first_token["rest_body_blocks"] = remote_credit[
                    "split_rest_blocks"]
            if first_token["split"]:
                # WP9:joiner id 快照随 train_plan 走（余量批的台账行
                # 需要与首步行相同的 joiners 记录；joiner_plans 只在
                # 首步批消费）。
                train_plan["joiner_ids_of_record"] = [
                    runtime.request_id for runtime in joiner_runtimes]
                self._emit_train_first_step(state, plan, train_plan, tick)
                return
        result = self.graph.emit_iteration_train(train_plan)
        self._register_train_watches(plan, result)
        self._train_instance_index[plan["train_id"]] = state.index
        state.in_flight_train = plan
        # 改法S2-A：在飞列车挂起（快照 running 分量输入变更）——实例
        # 纪元 +1，快照缓存失效。
        self._bump_instance_epoch(state)
        self._refresh_frontier(state)
        self._emit_train_ledger_row(
            state, plan,
            [runtime.request_id for runtime in joiner_runtimes], tick)

    def _register_train_watches(self, plan, result) -> None:
        """drain/exit/哨兵标记 watch 注册 + drain 块末账本（整列发射与
        WP9 余量批共用；登记内容与挂点语义逐字段不变）。"""
        self._register_train_eviction_watches(result)
        for request_id, members in result["drain_members"].items():
            self._batch["watches"].append({
                "request_id": request_id,
                "stage": STAGE_PREFILL,
                "generation": 0,
                "members": members,
                "statuses": ["Success", "Skipped"],
            })
            runtime = self.runtime_by_request_id[request_id]
            runtime.drain_block_ends = dict(result["block_ends"])
        for request_id, members in result["exit_members"].items():
            self._batch["watches"].append({
                "request_id": request_id,
                "stage": STAGE_DECODE,
                "generation": 1,
                "members": members,
                "statuses": ["Success", "Skipped"],
            })
        if plan["sentinel"]:
            # T_max 截断且无自然标记:哨兵 watch(train_id 批命名空间,
            # 固定 prefill 单事件通道——decode 通道会双事件 + 完成计账
            # 下溢)。
            self._batch["watches"].append({
                "request_id": plan["train_id"],
                "stage": STAGE_PREFILL,
                "generation": 0,
                "members": result["sentinel_members"],
                "statuses": ["Success", "Skipped"],
            })

    def _emit_train_ledger_row(self, state, plan, joiner_ids, tick: int,
                               first_step: bool = False) -> None:
        """train_ledger 行（M3 流式落盘）。WP9：拆分列车的首步批先写
        first_step=True 行（ON/OFF 对拍剥离标记），余量批再写正常行
        （与整列发射的行同构）。"""
        ledger_row = {
            "train_id": plan["train_id"],
            "instance_index": state.index,
            "tick": tick,
            "iterations": plan["iterations"],
            "member_count": len(plan["members"]),
            "member_iterations": sum(
                participation for _, participation in plan["members"]),
            "joiners": list(joiner_ids),
            "drains": list(plan["drain_members"]),
            "exits": list(plan["exit_members"]),
            "sentinel": plan["sentinel"],
            "prefill_chunks": len(plan["prefill_chunk_tokens"]),
            "pass_spans": len(plan["pass_spans"]),
        }
        if first_step:
            ledger_row["first_step"] = True
        # M3 流式落盘：提供 train_ledger_sink 时行即写即弃；缺省缓冲。
        if self.train_ledger_sink is not None:
            self.train_ledger_sink(ledger_row)
        else:
            self.train_ledger_rows.append(ledger_row)

    def _first_token_split_spans(self, plan):
        """WP9 首/余量 span 组切分（机械操作，2026-08-26）。

        首步 = [prefill 队头第 1 个 chunk] + [各 decode 成员第 1 个
        span]；余量 = [剩余 chunk] + [各成员剩余 span]。聚合节点对 span
        求和与顺序无关，两组的激活/KV/AR 字节总量与整列一致；权重经
        weight_passes（1 + iterations-1）合计不变。"""
        spans = list(plan["pass_spans"])
        chunk_count = len(plan["prefill_chunk_tokens"])
        member_parts = plan["members"]
        if chunk_count + sum(p for _, p in member_parts) != len(spans):
            raise RuntimeError(
                "train span layout does not match the frozen plan")
        first_spans = spans[:1] if chunk_count else []
        rest_spans = list(spans[1:chunk_count]) if chunk_count else []
        offset = chunk_count
        for _, participation in member_parts:
            first_spans.append(spans[offset])
            rest_spans.extend(spans[offset + 1:offset + participation])
            offset += participation
        return first_spans, rest_spans

    def _first_token_plan(self, plan, joiner_runtimes):
        """WP9 首 token 观测计划（None = 开关关闭/无 debut，行为与拆分
        上线前逐字节一致）。

        debut 成员 = 本交付加入列车的 decode 成员（decode_tokens_consumed
        == 0，计划期可知）。多 token debut 挂独立 first_token 标记；
        decode_length=1 的 debut 其 exit 标记名附加 first_token 子串
        （同节点 code 4/8 双锚点，保证 first_token_ns == completion_ns）。
        iterations >= 2 时物理拆两批（首步批 + 余量批），否则仅做不拆车
        的标记增强。"""
        if not first_token_split_enabled():
            return None
        debut = [
            runtime for runtime in joiner_runtimes
            if runtime.decode_tokens_consumed == 0
        ]
        if not debut:
            return None
        debut_marker_members = [
            {"request_id": runtime.request_id}
            for runtime in debut
            if runtime.decode_length != 1
        ]
        debut_exit_first_token = [
            runtime.request_id for runtime in debut
            if runtime.decode_length == 1
        ]
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

    def _emit_train_first_step(self, state, plan, train_plan,
                               tick: int) -> None:
        """WP9 首步批发射（两段式前半，2026-08-26）：构图 + 唤醒 watch
        注册 + busy 门挂起 + first_step 台账行。drain/exit/哨兵 watch、
        drain_block_ends 与正常台账行全部移至余量批。"""
        first_token = train_plan["first_token"]
        result = self.graph.emit_train_first_step(train_plan)
        self._register_train_eviction_watches(result)
        self._batch["watches"].append({
            # 批命名空间唤醒 watch（哨兵同款单事件通道）：首步批完成即
            # 交付余量批；不挂任何请求，fire 事件在 run_variant_policy
            # 的 _consume_first_step_wakeup 处无操作剥离。
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
        # 改法S2-A：在飞列车挂起（快照 running 分量输入变更）——实例
        # 纪元 +1，快照缓存失效。
        self._bump_instance_epoch(state)
        self._refresh_frontier(state)
        self._emit_train_ledger_row(
            state, plan, train_plan["joiner_ids_of_record"],
            tick, first_step=True)

    def _emit_train_remainder(self, state, tick: int) -> None:
        """WP9 余量批发射（两段式后半，2026-08-26）：余量体 + drain/
        exit/哨兵标记 + end barrier，随后注册 drain/exit/哨兵 watch、
        drain_block_ends 账本与正常台账行——与整列发射的后半完全同构。
        train_id→实例索引、纪元/frontier 在首步批发射时已处理且本步
        输入不变（in_flight_train 保持挂起），不重复 bump/refresh。"""
        train_plan = state.first_step_remainder
        state.first_step_remainder = None
        plan = state.in_flight_train
        result = self.graph.emit_train_remainder(train_plan)
        self._register_train_watches(plan, result)
        self._emit_train_ledger_row(
            state, plan, train_plan["joiner_ids_of_record"], tick)

    def _consume_first_step_wakeup(self, train_id: str) -> bool:
        """识别并吞掉自己发射的首步批唤醒信号（WP9，2026-08-26）。

        首步批的唤醒 watch 用批命名空间 id（"<train_id>_first_step"，
        batch_train_ 前缀），fire 后与真哨兵同通道送达。返回 True = 这是
        首步唤醒（无操作，仅从哨兵核销列表剥离）；False = 非首步 id
        （真哨兵或未知 batch_train_ id，交回哨兵核销逻辑处理）。"""
        if train_id not in self._pending_first_steps:
            return False
        self._pending_first_steps.pop(train_id)
        return True

    def _refresh_frontier(self, state) -> None:
        """§7.3 ready frontier 增量维护（原 busy 判定的列车化等价）：
        实例非忙（无在飞列车）且有排队工作（qp / active_decode /
        pending_decode_ready 任一非空）即入 frontier，否则出。"""
        if (state.in_flight_train is None
                and (state.qp or state.active_decode
                     or state.pending_decode_ready)):
            self._ready_frontier.add(state.index)
        else:
            self._ready_frontier.discard(state.index)

    # ------------------------------------------------------------- 边界 --

    def _push_arrival(self, arrival: dict, tick: int) -> None:
        """offline: face_scheduler.py push_event(time_ns, 1,
        "arrival", index) 的在线等价（ingress ARRIVAL 事件喂入）。"""
        runtime = self.runtime_by_request_id[arrival["request_id"]]
        heapq.heappush(
            self.arrival_heap,
            (tick, 1, runtime.queue_index, self._sequence, "arrival",
             runtime))
        self._sequence += 1

    def _drain_arrival_heap(self, tick: int) -> None:
        """offline: face_scheduler.py 同 tick 批 + (priority,
        sequence) 排序（:3903）。"""
        due = []
        while self.arrival_heap and self.arrival_heap[0][0] <= tick:
            due.append(heapq.heappop(self.arrival_heap))
        due.sort(key=lambda item: (item[1], item[2]))
        for _, _, _, _, kind, runtime in due:
            if kind != "arrival":
                raise RuntimeError("arrival heap contains {!r}".format(kind))
            self._on_arrival(runtime, tick)

    def _on_arrival(self, runtime, tick: int) -> None:
        """offline: face_scheduler.py arrival 批单条。"""
        if runtime.estimated_arrival_ns is not None:
            raise RuntimeError(
                "request arrival was delivered more than once")
        runtime.estimated_arrival_ns = tick
        self._arrived_request_count += 1
        self.pending_admissions.append(runtime)
        self._retry = True

    def _on_prefill_drain(self, request_id: str, tick: int) -> None:
        """offline: face_scheduler.py prefill 末 chunk 完成分支
        （decode 固定同实例 :3944-3949），逐行迁移；拼 batch 改造：decode
        段发射移至加入列车（_plan_and_emit_trains → emit_iteration_train），
        成员先进 pending_decode_ready（§3.2 KV 就绪栅栏），DECODE_
        COMPLETION watch 由列车 exit 标记承载。"""
        runtime = self.runtime_by_request_id[request_id]
        state = self.instances[runtime.prefill_instance_index]
        # R15-1：准入主链在途流在 drain 栅栏通过后释放；独立逐出支链
        # 仍可能在飞，owner=#evict 只由其尾 watch 释放。
        self._release_transfer_flows(request_id)
        # 规格书§二.4（2026-09-25）：prefill remote-read 前缀读流在
        # prefill drain 释放——decode credit 读流（#readplan /
        # rid#decode#{j}）自本边界之后才建立，同一条前缀读流不得同时
        # 挂 prefill/decode 两个身份（release_owner 幂等空放，非前缀
        # 读流请求零副作用）。
        self._release_transfer_flows(request_id + "#prefill_read")
        # C11：准入主链配额流 settle；#evict 支链配额由尾 watch 释放，
        # remote-read 读流与 merge 预留生命周期到完成边界。
        if self._quota_tracker is not None:
            self._quota_release_admission_phase(request_id, tick)
        # drain 记实际 prefill 工作量（recompute 单口径 ==
        # request.prefill_length；离线 chunk 累计的整段等价）。
        runtime.prompt_tokens_processed = runtime.prefill_tokens_to_process
        runtime.remaining_chunks = 0
        # offline: :3989-3991 FCFS qp popleft（离线 iteration 串行化保证
        # drain 序 = FCFS 序）。在线列车构图下同实例列车头 chunk 的 drain
        # 物理完成可先于队列中更早请求的 drain 决策事件到达（信号拆交付
        # 时头部跳过规则允许后续请求的 chunk 先行）——账本适配 = 从 qp
        # 移除该已完成成员（母本同款；排队深度口径 remaining_chunks 求和
        # 不变，策略输入语义等价）。
        if runtime not in state.qp:
            raise RuntimeError("draining request is not in its prefill queue")
        state.qp.remove(runtime)
        # 改法S2-B：drain 出队核销聚合账本——drain watch 挂在覆盖末 chunk
        # 的列车 drain 标记上，该列车核销（本交付或更早交付，恒先于本
        # drain 处理）已把 prefill_tokens_completed 推进到全量，账本余量
        # 必为 0（现算口径的剩余贡献亦恰为 0）；残值非 0 = 增量账本漂移，
        # fail-closed。
        if runtime.queued_chunk_load_ns != 0:
            raise RuntimeError(
                "queued chunk load ledger drift for request {} at drain: "
                "{} ns".format(request_id, runtime.queued_chunk_load_ns))
        runtime.queued_chunk_load_ns = 0
        # 改法S2-A：qp 成员变更——实例纪元 +1，快照缓存失效。
        self._bump_instance_epoch(state)
        runtime.prefill_complete_ns = tick
        self._refresh_frontier(state)
        # offline: face_scheduler.py（expand_prefill）
        # joint：工作副本上下文按动作推进（stay/copy/recompute 至
        # prefill_context；remote-read 仅新增输入的 KV 落执行端）。
        runtime.prefill_evictions = self.kv_manager.expand_prefill(
            session_id=runtime.session_id,
            instance_index=state.index,
            context_tokens=self._joint_working_context(runtime),
            trigger_request_id=runtime.request_id,
            reservation_request_id=runtime.request_id,
        )
        self._bump_kv_ledger_epoch()  # 改法D：KV 变更点 1/9（expand_prefill）
        self.graph.sync_pending_history_after_evictions(
            runtime.prefill_evictions)
        # joint：本轮 prefill/decode 保持同一执行 instance（设计方案
        # §1.1）；decode 不做二次选点（基底 no_affinity 消融随 SH30_
        # ABLATION 一并退役）。
        selected = state.index
        decode_state = state
        runtime.decode_instance_index = selected
        reservation_move_evictions = (
            self.kv_manager.move_request_capacity_reservation(
                request_id=runtime.request_id,
                target_instance_index=selected,
            )
        )
        self._bump_kv_ledger_epoch()  # 改法D：KV 变更点 2/9（move_request_capacity_reservation）
        self.graph.sync_pending_history_after_evictions(
            reservation_move_evictions)
        (runtime.prefill_decode_transfer,
         decode_move_evictions) = self.kv_manager.move_prefill_to_decode(
            session_id=runtime.session_id,
            target_instance_index=selected,
            trigger_request_id=runtime.request_id,
            reservation_request_id=runtime.request_id,
        )
        # joint remote-read（v2 执行口径，2026-09-17 用户裁定唯一机制）：
        # drain 边界建立持久读计划（每步读量 = 均匀终态上下文、逐 shard
        # 路由一次冻结——字节口径与 v1 批量流逐位同源，I1 守恒基数）；
        # 逐列车切片在列车规划期从计划派生（彼时 participation/S_j 才
        # 冻结，§4.3.2），切片旁挂支链 + 体块 arm 门——读与 decode 计算
        # 重叠。旧 v1"批量读流 + readiness barrier 硬栅栏"串行口径已删除
        # （不作为开关可选项保留；K >= S_j 单块切片的块 1 仍走原
        # pd_transfer 发射路径 = 逐字节等价锚）。
        if runtime.joint_action == "remote-read":
            # 规格书§四.3（2026-09-25）：decode credit 计划建立前，断言
            # 后缀 restore journal 已在 prefill_drain 边界（上方
            # expand_prefill → _settle_restore_groups）全部 consumed 并
            # 关账——decode 后缀层直接复用 exec HBM 已恢复的历史后缀
            # [p,L)，不得再从池恢复；journal 仍开 = 恢复链破损，
            # fail-closed 不带病建立 decode 读计划。
            drain_session = self.kv_manager._sessions[runtime.session_id]
            if drain_session.restore_journal is not None:
                raise RuntimeError(
                    "suffix restore journal for session {} is still open "
                    "at prefill drain (request {}): decode remote-read "
                    "credit plan requires the suffix restore to be fully "
                    "consumed first".format(
                        runtime.session_id, runtime.request_id))
            runtime.remote_read_credit_plan = (
                self._joint_remote_read_credit_plan(runtime, selected))
            # C8 步骤 2（对账核销之一）：真计划冻结处——核销准入相的
            # #readplan est 账本、以真值建立注册表登记（#prefill_read
            # 已在本边界上方释放）；预估/实际差值进决策日志。
            self._reconcile_readplan_at_drain(runtime, tick)
        self._bump_kv_ledger_epoch()  # 改法D：KV 变更点 3/9（move_prefill_to_decode）
        self.graph.sync_pending_history_after_evictions(
            decode_move_evictions)
        # joint（§3.1）：decode 不按真实最终长度一次扩容——drain 边界只
        # 覆盖已知 prefill 上下文；后续逐列车按实际消费进展因果增长
        # （_finalize_completed_trains 内的 _joint_grow_decode 钩子）。
        decode_growth_evictions = ()
        runtime.decode_evictions = (
            reservation_move_evictions + decode_move_evictions
            + decode_growth_evictions)
        if runtime.decode_evictions and any(
                getattr(transfer, "shards", ())
                for transfer in runtime.decode_evictions):
            watch_id = self._next_eviction_watch_id(
                runtime.request_id, "decode_joiner")
            flow_owner = watch_id + "#flow"
            self._register_transfer_flows(
                runtime.decode_evictions, owner=flow_owner)
            quota_owner = self._quota_enroll_eviction_branch(
                watch_id + "#quota", runtime.request_id,
                runtime.decode_evictions, tick)
            self._pending_eviction_watches[watch_id] = {
                "request_id": runtime.request_id,
                "flow_owners": (flow_owner,),
                "quota_owners": (
                    (quota_owner,) if quota_owner is not None else ()),
                "scheduled": False,
            }
            runtime.decode_eviction_watch_id = watch_id
        self.kv_manager.release_request_capacity_reservation(runtime.request_id)
        self._bump_kv_ledger_epoch()  # 改法D：KV 变更点 5/9（release_request_capacity_reservation）
        # R15-1：decode 主链迁移/读流在 service_done 注销；逐出支链在各自
        # 的 eviction_done 尾 watch 注销（不能与请求的 decode owner 合并）。
        decode_phase_transfers = (
            runtime.prefill_evictions
            + ((runtime.prefill_decode_transfer,)
               if runtime.prefill_decode_transfer is not None else ()))
        self._register_transfer_flows(
            decode_phase_transfers,
            owner=request_id + "#decode")
        # C11：decode 相 oneshot 配额入册（同一传输集，同生命周期——
        # 完成边界 settle）。
        if self._quota_tracker is not None:
            self._quota_enroll_decode_phase(
                request_id, decode_phase_transfers, tick)
        # 拼 batch 改造（§3.2 KV 就绪栅栏）：drain 决策（decode 实例选择
        # ＝固定同实例/KV 迁移规划）在此完成，成员进入 pending_decode_
        # ready，待加入 decode 实例的下一列车（迁移随加入列车发射，物理
        # 先于列车体；restore/迁移列车中途完成的也只能等下列车边界）。
        # TaskA no_affinity：decode_state 可能≠prefill 实例（none 档恒等）。
        decode_state.pending_decode_ready.append(runtime)
        self._refresh_frontier(decode_state)
        # online: start_ready_iterations 的 decode 起始记账（:2927-2928）。
        runtime.decode_start_ns = tick
        # ledger（阶段 3 感知账本，查询/审计输入不进判据）。
        self._ledger_admit(
            runtime.request_id, tick,
            {"type": "active_decode", "instance_index": selected})
        # decode 段发射移至加入列车；drain 边界的 decode 决策记录在此。
        self._emit_join_decision(runtime, tick)

    def _complete_requests(self, completed_now, tick: int):
        """offline: :3986-4042 decode 完成分支 + completion_order 批
        （下一 turn arrival 排程 :4000-4006 + mark_complete :4016-4020 +
        快照 :4032-4042）。

        拼 batch 改造：active_decode 移除已移至 _finalize_completed_trains
        （退出迭代在列车内先验已知，物理完成时刻 = exit 标记节点完成时刻）。

        C3b（2026-09-22 到达合同迁移，F11 到达锚独立条款）：下一轮到达
        alarm 锚 **service_done + interval**——service_done = 本轮向外
        完成响应的时刻（本仓事件侧 = REQUEST_COMPLETE 交付 tick，thinktime
        自该时刻起算；有/无 merge 流同锚点，全部方法与对照统一适用，
        FACE/WSC 适配版同口径）。到达 ≠ 就绪：到达后的数据准备仍受真实
        块就绪与合并依赖门控——图侧 interval gate 前递锚 merge 尾标记
        （R11(ii) 结构保留，语义自"到达锚点"改为"到达后门控"；local_hit
        路径经同 rank 串行链接续覆盖、池在途块经 store 尾前递边覆盖），
        **保守口径 = merge_done 整体门控，块级门控留后续（不冒称已块级）**。
        N2 承接（2026-09-14 修复关切在新合同下的承接）：thinktime < merge
        时长时到达早于 merge_done，merge 成本以"下一轮数据等待"形式保留
        在闭环内——既不逃出会话 E2E（到达不被 merge 推迟），也不消失
        （门控由 merge watch 交付事件驱动解除）。预测终点（F11 = merge_done）
        是独立条款，本方法不动终点口径。
        """
        # offline: :4008-4012 completion_order 排序
        completion_order = sorted(
            completed_now,
            key=lambda request_id: (
                self.runtime_by_request_id[request_id].session_id,
                request_id,
                self.runtime_by_request_id[request_id].queue_index,
            ),
        )
        # offline: :3986-3999 decode 完成分支。下一轮 alarm 的排程与
        # merge 流存在与否解耦（C3b：有/无 merge 同锚 service_done +
        # interval；merge watch 仅承担流注销与 merge_done 披露）。
        completion_facts = []
        for request_id in completion_order:
            runtime = self.runtime_by_request_id[request_id]
            runtime.completed = True
            runtime.completion_ns = tick  # 基类侧字段由 log_decision 行携带
            self.completed_requests += 1
            following = self.next_request[request_id]
            interval = None
            if following is not None:
                interval = self.config.request_queue[
                    following.queue_index].inter_request_interval_ns
                if interval is None:
                    raise RuntimeError(
                        "validated later request lost its interval")
            completion_facts.append((request_id, following, interval))
        # joint（§2.2/§3.2，merge v2 少并多 2026-09-17）：compute_done
        # 后、mark_complete 前执行 merge 事务——两侧保留量比大小、小侧
        # 整份搬大侧（零池写、全层 LOCAL@胜者、home 迁移；copy/recompute
        # 零字节翻转；REMOTE 无主就地保留；胜者侧空间经统一 T+E 准备，
        # 双侧深缺口才 fail-closed）。完成时已观测的 decode 长度/输入
        # 长度入在线估计器（实际结算，非决策 oracle）。
        # C14（2026-09-22）remote 完成检查清单审计落字：完成检查五件
        # = 源端保护解除（败者侧释放/home 标签迁移）、暂存归还（F14
        # 无留存型口径 ⇒ 无操作，journal staging_return_bytes=0 披露位）、
        # 增量结算（少并多按两侧实际保留量——账本真值裁决）、合并目标
        # 容量（merge_back 内 _ensure_capacity 取完成时刻真实剩余，不
        # 沿用发射时预测空闲）、失败分支闭合（版本键防重复回调、双侧
        # 深缺口 fail-closed 不丢权威副本）；逐请求结算事实入 FS 侧
        # kv_delta_journal（C16 消费面），不进 decision-log schema。
        for request_id, following, interval in completion_facts:
            runtime = self.runtime_by_request_id[request_id]
            new_tokens = (
                runtime.joint_input_tokens + runtime.decode_length)
            # K2/K3 配套（kimi 复审）：完成行显式披露"执行端工作副本"
            # 真值（merge 前的 working_kind）——水印重放不再从 origin_home
            # 启发式重建（REMOTE 基 + 执行端恰等于 home 的组合会误判）。
            session_state = self.kv_manager._sessions.get(
                runtime.session_id)
            runtime.joint_working_copy = bool(
                session_state is not None
                and session_state.working_kind is not None)
            runtime.merge_transfers = self.kv_manager.merge_back(
                session_id=runtime.session_id,
                trigger_request_id=runtime.request_id,
                new_tokens=new_tokens,
            )
            # 合并方向 v2（2026-09-17）：完成行披露 merge_back 的方向
            # 语义——face_scheduler 每次 merge_back 后设置
            # last_merge_outcome（键 session_id/direction("stay"|
            # "forward"|"reverse"|"in_place")/zero_byte_flip/
            # winner_instance/loser_instance/transferred_bytes/
            # home_flipped；stay 亦照实落账）。真 KVCacheManager 的
            # __init__ 恒设该属性（face_scheduler :1747）——直接访问，
            # 替身 kv_manager 漏设 = AttributeError（F6 销账）。
            runtime.merge_outcome = self.kv_manager.last_merge_outcome
            self._bump_kv_ledger_epoch()
            self.graph.sync_pending_history_after_evictions(
                runtime.merge_transfers)
            self._joint_horizon.observe_completed(
                runtime.session_id, runtime.decode_length)
            self.kv_manager.observe_completed_input(
                runtime.session_id, runtime.joint_input_tokens)
            # R14 完成清理：停滞态会话完成时退出停滞登记（其 exit 物理
            # 完成不依赖 KV 增长追平）。
            if runtime.decode_stalled:
                stalled = self._stalled_by_instance.get(
                    runtime.decode_instance_index)
                if stalled is not None:
                    stalled.discard(runtime)
                    if not stalled:
                        self._stalled_by_instance.pop(
                            runtime.decode_instance_index, None)
                runtime.decode_stalled = False
            # R15-1：decode 相在途流注销（完成边界）。
            self._release_transfer_flows(request_id + "#decode")
            # C11 步骤 7（service_done 时刻方向裁决）：释放 merge 预留
            # 败者侧（胜者侧到 _on_merge_done；零传输分支整体释放）。
            if self._quota_tracker is not None:
                self._quota_on_service_done(request_id, runtime, tick)
                self._quota_release_decode_phase(request_id, tick)
            # C8 步骤 2（对账核销之三）：#readplan 残余承诺清零 + 真值
            # 闭式与逐列车实际登记块数之差披露（列车碎片化残差属正常
            # 非零差，完成边界收口；泄漏审计在 verify_run_end 兜底）。
            self._settle_readplan_residual(runtime, tick)
            # C11：remote-read 读流 settle（流寿命样本随双时标进入
            # AIMD EWMA）。
            if self._quota_tracker is not None:
                self._quota_release_readplan_stream(request_id, tick)
            if following is None:
                continue
            # C3b（2026-09-22 到达合同）：下一轮到达 = service_done +
            # interval（有/无 merge 流同锚点——thinktime 自响应完成起算，
            # 实验思路 §三.4 a(s,k+1) = f(s,k) + z(s,k)）。到达后的数据
            # 依赖门控不在本 alarm 侧承载（图侧 interval gate / 同 rank
            # 串行链 / store 尾边，见 docstring——到达 ≠ 就绪）。
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
        # offline: :4016-4020 先全部 mark_complete
        # typed eviction：在线 runtime 是 manifest 派生账本（无 FaceRequest
        # 字段），完成请求自身的 next_trigger_type 经
        # config.request_queue[queue_index]（FaceTraceConfig 装载的
        # RequestSpec，与 manifest 同序）取回传给 mark_complete。
        for request_id in completion_order:
            runtime = self.runtime_by_request_id[request_id]
            self.kv_manager.mark_complete(
                runtime.session_id,
                tick,
                next_request_type=(
                    self.config.request_queue[
                        runtime.queue_index].next_trigger_type
                ),
            )
            self._bump_kv_ledger_epoch()  # 改法D：KV 变更点 6/9（mark_complete）
        # offline: :4021-4031 完成路径段（D-clear 2026-09-05）：完成边界
        # 零逐出，仅保留结构守卫与恒空 completion_evictions 的下游同步。
        for request_id in completion_order:
            runtime = self.runtime_by_request_id[request_id]
            if runtime.decode_instance_index is None:
                raise RuntimeError("completed request has no Decode instance")
            # D-clear (2026-09-05): 主动驱逐已物理清除——完成路径零逐出。
            runtime.completion_evictions = ()
            self.graph.sync_pending_history_after_evictions(
                runtime.completion_evictions)
        # offline: :4032-4042 完成快照 + completion 批（合同①）：
        # completion_evictions + 下一 turn interval gate 依赖登记。
        facts_by_id = {
            request_id: (following, interval)
            for request_id, following, interval in completion_facts}
        for request_id in completion_order:
            runtime = self.runtime_by_request_id[request_id]
            snapshot = self.kv_manager.session_snapshot(runtime.session_id)
            runtime.kv_location_after_completion = snapshot.location
            runtime.kv_instance_after_completion = snapshot.instance_index
            # 合并方向 v2（2026-09-17）：merge 事务的单行方向披露（读取
            # merge_back 后快照的 runtime.merge_outcome；face_scheduler
            # 尚未落地 last_merge_outcome 时三键落 None）。
            merge_outcome = runtime.merge_outcome
            completion_result = self.graph.emit_completion_batch(
                runtime.plan_dict())
            following, _interval = facts_by_id[request_id]
            if runtime.merge_transfers:
                # merge 尾标记 watch（R11(i) 结构保留；C3b 语义收窄为
                # 流注销 + merge_done 披露）——完成回调经 PREFILL_DRAIN
                # 通道送达，batch_train_merge_ 前缀在 run_variant_policy
                # 拦截为 _on_merge_done：注销 merge 流 + 落 merge_done
                # 事件侧披露行，**不再排下一轮 alarm**（到达已在响应完成
                # 点排定）。图侧 watch 成员同时是下一轮 interval gate 的
                # 前递依赖锚（到达后数据门控，保守 merge_done 整体口径）。
                # K2（P1-②，2026-09-23 外部审计）：注册条件去掉
                # "following is not None"——终轮同样有物理 merge 传输，
                # watch 不注册则胜者侧预留（quota 模式）滞留到
                # verify_run_end fail-closed、merge 流不进 R15 登记；
                # following 三字段终轮落 None（披露行条件化）。
                members = completion_result.get("merge_done_members")
                if not members:
                    raise RuntimeError(
                        "merge transfers were emitted without merge-done "
                        "marker nodes (request {})".format(request_id))
                watch_id = "batch_train_merge_" + request_id
                self._batch["watches"].append({
                    "request_id": watch_id,
                    "stage": STAGE_PREFILL,
                    "generation": 0,
                    "members": members,
                    "statuses": ["Success", "Skipped"],
                })
                self._pending_merge_alarms[watch_id] = {
                    "request_id": request_id,
                    # O3（2026-09-23 终轮审计）：merge 尾门纯度判定的实例
                    # 归属（R11(ii) interval gate 按实例 frontier 锚定——
                    # merge_done 标记落在完成批发射实例 = 本请求的 decode
                    # 实例；_emit_train 按 instance_index 匹配扩域门）。
                    "instance_index": runtime.decode_instance_index,
                    # merge_done 披露行所需最小账目（C12 G5 缺口路由承接：
                    # 物理 merge_done 此前无独立事件侧披露行，锚迁移后
                    # "下一轮到达时刻"代理消失）。终轮（无 following）
                    # session 锚退 runtime 自身、arrival 三字段 None。
                    "session_id": (
                        following.session_id if following is not None
                        else runtime.session_id),
                    "following_request_id": (
                        following.request_id if following is not None
                        else None),
                    "next_arrival_world_ns": (
                        tick + _interval if following is not None else None),
                }
                self._register_transfer_flows(
                    runtime.merge_transfers, owner=request_id + "#merge")
            # 阶段 3 感知账本：completed-unreconciled 核销（基类
            # _settle_completions 已在策略前完成；此处为决策日志）。
            self.log_decision(
                {"kind": "completion", "request_id": runtime.request_id,
                 "priority": 0},
                tick,
                decision={
                    "kv_location_after_completion":
                        runtime.kv_location_after_completion,
                    "kv_instance_after_completion":
                        runtime.kv_instance_after_completion,
                    "completion_evictions": [
                        _transfer_summary(transfer)
                        for transfer in runtime.completion_evictions],
                    # joint（§7：merge 事务审计）——compute_done 后归并回
                    # origin_home 的真实传输摘要（stay 本地提交为空）。
                    "merge_transfers": [
                        _transfer_summary(transfer)
                        for transfer in runtime.merge_transfers],
                    "joint_action": runtime.joint_action,
                    "origin_home_instance": runtime.origin_home_instance,
                    # 合并方向 v2（2026-09-17）：merge_back 方向语义三键
                    # （stay 亦照实落 "stay"/None/0；无 merge 事务时
                    # None——F6 销账：直接读 runtime.merge_outcome，真
                    # KVCacheManager __init__ 恒设该属性）。读者
                    # 忽略未知键，旧日志零行为差。
                    "merge_direction": (
                        merge_outcome["direction"]
                        if merge_outcome is not None else None),
                    "home_flipped_to": (
                        merge_outcome["winner_instance"]
                        if (merge_outcome is not None
                            and merge_outcome["home_flipped"])
                        else None),
                    "merge_transferred_bytes": (
                        merge_outcome["transferred_bytes"]
                        if merge_outcome is not None else None),
                    # K2/K3 配套：工作副本真值（merge 前 working_kind）。
                    # F6 销账：本方法首循环（completion_facts）对每个
                    # runtime 无条件先行赋值，读点恒晚于写点——直接访问
                    # （__slots__ 漏赋值读即 AttributeError，fail-loud）。
                    "joint_working_copy": bool(
                        runtime.joint_working_copy),
                    # remote-read credit（§4.3.5）：逐列车切片摘要（每请求
                    # 恰一处聚合落盘；非 remote-read 恒空列表——schema 对
                    # credit 开关前向兼容）。Σ(total_bytes) ≡ 计划总量
                    # （I1）可在日志侧直接审计。
                    "remote_read_slices": list(
                        runtime.remote_read_slice_summaries),
                },
            )
            following = self.next_request[request_id]
            if following is None:
                # This is the terminal-only ownership boundary: graph and log
                # consumers above have read the completion state, while an
                # intermediate turn must retain its KV for the future alarm.
                self.kv_manager.retire_terminal_session(
                    runtime.session_id,
                    tick,
                    request_id,
                )
                self._bump_kv_ledger_epoch()
            # M4 核销即删（2026-08-23）：请求完成后其 KV 转移对象/准入
            # 负载快照等胖字段再无读者（逐出已随 completion 批发射进图、
            # 审计已随决策行落盘；下一 turn 是独立 runtime；runtimes 列表
            # 运行全程存活，不置空会随完成请求数线性常驻）——置空即删。
            runtime.history_evictions = ()
            runtime.prefill_evictions = ()
            runtime.decode_evictions = ()
            runtime.completion_evictions = ()
            runtime.history_transfer = None
            runtime.prefill_decode_transfer = None
            runtime.history_location_before = None
            runtime.drain_block_ends = None
            runtime.prefill_instance_loads = ()
            # REQUEST_COMPLETE 边界后该 runtime 及其索引再无读者。释放
            # map、slot 和已消费的 next link，避免已完成 session turn 被
            # predecessor 的链条或预构建列表永久钉住。
            self.runtime_by_request_id.pop(request_id, None)
            index = self._runtime_index.pop(request_id, None)
            if index is None:
                raise RuntimeError(
                    "completed request lost runtime index {!r}".format(
                        request_id))
            self.runtimes[index] = None
            self.next_request.pop(request_id, None)

    # ------------------------------------------------------------- 准入 --

    def _bump_kv_ledger_epoch(self) -> None:
        """改法D：KV 账本纪元 +1。仅调度器的 9 个 KV 变更点后调用
        （见各调用处注释）；可行性读取与 _check_invariants 只读、不 bump。"""
        self._kv_ledger_epoch += 1

    def _bump_instance_epoch(self, state) -> None:
        """改法S2-A：实例账本纪元 +1（_task_load_snapshot 快照缓存失效）。
        仅在本实例快照输入的每个变更点后调用——qp/active_decode 成员变化
        （准入入队/drain 出队/decode 成员入批/退出成员移除）、成员进度字段
        推进与 in_flight_train 挂起/核销（列车核销）、last_arrival_ns
        （准入）；_task_load_snapshot 读取不 bump（对齐改法D
        _bump_kv_ledger_epoch 的姿态，bump 多只会损失命中、不会错复用）。"""
        state.ledger_epoch += 1

    def _admit_pass(self, tick: int) -> None:
        """offline: face_scheduler.py（admit_waiting_requests +
        start_ready_iterations 的排队/发射部分；直接 Roofline 计时删除）。
        拼 batch 改造：发射部分 = _plan_and_emit_trains（列车化）。

        R14：每交付先跑唤醒 pass（停滞会话的纪元键变化即重试，无停滞
        O(1)）+ 死锁守卫（全停滞显式 fail-closed，N8）。"""
        self._retry = False
        self._wake_stalled_decodes(tick)
        self._check_decode_deadlock()
        self._admit_waiting_requests(tick)
        self._plan_and_emit_trains(tick)

    def _admit_waiting_requests(self, now_ns: int) -> None:
        """offline: face_scheduler.py admit_waiting_requests，
        逐行对应（blocked FIFO 重排语义保留）+ 改法D 纪元重试门。

        N6+F3 重试键修复（2026-09-14，M2 复审修订）：joint 下 False 的
        语义是"J **选中**的 (instance, action) 容量不可行"，而 J 的选择
        依赖 task loads——负载迁移 bump 实例纪元但不 bump KV 纪元，纯
        KV 纪元门会把本可换选的请求跳过（时机损失 + SH_ADMIT_GATE_
        VERIFY 影子断言被合法击穿）。键改为 KV 纪元 ⊕ **失败候选集**
        （选中实例 ∪ applicable 候选实例，冻结于失败时刻）的实例纪元
        ——候选集内任一实例负载迁移都可能翻转 argmin 到可行候选；严禁
        粗化为全实例纪元之或（F3：实例纪元是列车事件粒度的高频信号，
        全挂会让重试门恒翻转、每次重试跑全候选成本模型、深尾部墙钟爆
        炸；冻结候选集只含本次判定相关的实例）。KV 纪元已覆盖全部容量
        释放（容量 False 翻 True 的必要条件），实例纪元只补负载迁移带
        来的换选时机。"""
        blocked = deque()
        while self.pending_admissions:
            runtime = self.pending_admissions.popleft()
            rid = runtime.request_id
            last_key = self._admit_attempt_epoch.get(rid)
            skip = (
                last_key is not None
                and not self._admit_gate_verify
                and last_key == self._current_retry_key(last_key))
            if skip:
                # 键未变 → 容量未释放且选中实例负载未迁移，重试必返
                # 同样的 False，跳过（FIFO 位置不变）。
                blocked.append(runtime)
                continue
            if (self._admit_gate_verify and last_key is not None
                    and last_key == self._current_retry_key(last_key)):
                # 影子断言：门判跳过 ≡ 重试必返 False（失败路径已在
                # _try_admit_request 内留 _last_admit_failure_key）。
                if self._try_admit_request(runtime, now_ns):
                    raise RuntimeError(
                        "admit gate equivalence violated: request {} was "
                        "admitted on a skipped retry (key {})".format(
                            rid, last_key))
                self._record_admit_failure_key(rid)
                blocked.append(runtime)
                continue
            if not self._try_admit_request(runtime, now_ns):
                self._record_admit_failure_key(rid)
                blocked.append(runtime)
            else:
                self._admit_attempt_epoch.pop(rid, None)  # 成功即清除（有界）
                self._admission_failure_state.pop(rid, None)
        self.pending_admissions.extend(blocked)

    def _record_admit_failure_key(self, rid: str) -> None:
        """失败返回后记录重试键（_try_admit_request 留下的
        _last_admit_failure_key；缺省回退 = (当前 KV 纪元, ((0, 实例 0
        纪元),))——仅打桩/异常路径，语义为"多开一次门"，无正确性影响）。"""
        self._admit_attempt_epoch[rid] = getattr(
            # F6 判定：保留（合法条件缺省，非替身软门）——该属性仅在
            # _try_admit_request 失败分支赋值（:4180/:4242/:4344），首
            # 请求成功前结构性缺席；__slots__ 无、__init__ 不设。
            self, "_last_admit_failure_key", None) or (
                self._kv_ledger_epoch,
                ((0, self.instances[0].ledger_epoch),))

    def _current_retry_key(self, last_key):
        """重试键求值：KV 纪元 ⊕ last_key 记录的失败候选集实例纪元。

        last_key = (kv_epoch_at_failure, ((instance, epoch_at_failure),
        ...))——候选集冻结于失败时刻（选中实例 ∪ applicable 候选，M2）；
        当前键 = (当前 KV 纪元, 同一实例集的纪元现值)。与上次失败键
        整体相等 = 无新信息（容量未释放且候选集内无负载迁移、argmin
        不会翻转）。

        C11（C9 对接合同）：配额代数分量追加在末位（extend_retry_key
        ——流 settle/预留释放/方向裁决/配额调整均 bump 该代数 ⇒ 配额
        释放后 deferred 请求必获再评估）。容量路径的失败键同步扩展
        （_compose_admit_failure_key 单一裁决点），两端对称比较。"""
        key = (self._kv_ledger_epoch,
               tuple((index, self.instances[index].ledger_epoch)
                     for index, _ in last_key[1]))
        return self._extend_retry_key_with_quota(key)

    def _extend_retry_key_with_quota(self, key: tuple) -> tuple:
        """配额代数并入（tracker 缺席 = 纯容量键，off 模式零漂移）。"""
        if self._quota_tracker is None:
            return key
        return extend_retry_key(key, self._quota_tracker.quota_retry_key())

    def _compose_admit_failure_key(self, candidate_instances) -> tuple:
        """失败键合成（容量路径与配额路径共用：KV 纪元 ⊕ 候选集实例
        纪元 ⊕ 配额代数）。"""
        return self._extend_retry_key_with_quota((
            self._kv_ledger_epoch,
            tuple((index, self.instances[index].ledger_epoch)
                  for index in sorted(candidate_instances))))

    def _select_prefill_instance(self, snapshots, candidate_mask) -> int:
        """负载均衡选点（SH30_ABLATION 消融门已随该开关退役删除；
        本方法保留为基底调用点兼容的直转）。"""
        return select_prefill_instance(snapshots, candidate_mask)

    # ------------------------------------------------- joint 视图与选择 --

    def _joint_working_context(self, runtime) -> int:
        """工作副本/基础会话的当前应覆盖 token 数（因果：只含已到达输入
        与已实际完成的 decode）。remote-read 的工作副本仅承载新增量；
        跨实例 recompute 自 0 物化（工作上下文 = 已物化 token 数）；
        其余动作（含 R13 的 recompute@home——缺失后缀是**层**物化，不
        增加 token 上下文）覆盖完整上下文。"""
        if runtime.joint_action == "remote-read":
            return runtime.joint_input_tokens + runtime.decode_tokens_consumed
        if (runtime.joint_action == "recompute"
                and runtime.joint_span_base_context == 0):
            return (runtime.joint_prefill_work
                    + runtime.decode_tokens_consumed)
        return (runtime.history_tokens_before
                + runtime.joint_input_tokens
                + runtime.decode_tokens_consumed)

    def _joint_grow_decode(self, runtime, instance_index: int) -> None:
        """逐列车 decode 因果增长（§3.1：写入前准备资源，不提前按真实
        最终长度预约）。列车核销推进 decode_tokens_consumed 后调用；
        增量经 _expand_local_session → _ensure_capacity（T+E）真实逐出。

        R14（M1(a) 裁定，2026-09-14）：容量类失败不再 fail-closed——
        会话进入 stalled 态（暂缓后续列车参与、等待如实计入 E2E，
        与基线背压语义同构、八组合同一合同可比）；唤醒复用 R2 纪元键
        （KV 纪元 ⊕ 实例纪元，F3 粒度），容量释放（逐出/完成）必有
        新信息、无自旋。审计按 stall/wake 事件计（非每 tick 轮询）。

        自查 D（2026-09-15，D7 盲区）：增长逐出（成功/停滞两路）此前
        只做 sync 簿记、从不进图——池写物理流成 C++ 水位盲区 + pending
        store 不登记，且 R15 在途流漏登记（除数低估并发池写）。基底
        的一次性增长经 runtime.decode_evictions 随 joiner 列车进图，
        joint 因果化后该路径消失。修法 = 旁路支链发射（与 R2 准入失败
        路径同构：无触发门、合法容量释放）+ 流登记挂 rid#decode
        owner（完成边界统一注销）。"""
        target_context = self._joint_working_context(runtime)
        session = self.kv_manager._sessions.get(runtime.session_id)
        if session is None or session.context_tokens == target_context:
            return
        try:
            evictions = self.kv_manager.expand_decode(
                session_id=runtime.session_id,
                instance_index=instance_index,
                final_context_tokens=target_context,
                trigger_request_id=runtime.request_id,
            )
        except KVCapacityError as exc:
            if exc.evictions:
                # D7 清账：raise 前已提交的逐出入图（旁路支链）+ 流
                # 登记 + 纪元 bump。
                self._bump_kv_ledger_epoch()
                self.graph.sync_pending_history_after_evictions(exc.evictions)
                self._emit_eviction_only_nodes(
                    exc.evictions, self._require_batch_tick(),
                    trigger_request_id=runtime.request_id)
            self._enter_decode_stall(
                runtime, instance_index, str(exc),
                gap_records=exc.deep_gap_records)
            return
        # 纪元 bump 无条件：expand 即使零逐出也已推进上下文账本。
        self._bump_kv_ledger_epoch()
        if evictions:
            self.graph.sync_pending_history_after_evictions(evictions)
            # 自查 D：增长的已提交逐出进图（旁路支链）+ 流登记。
            self._emit_eviction_only_nodes(
                evictions, self._require_batch_tick(),
                trigger_request_id=runtime.request_id)

    def _wake_key(self, instance_index: int) -> tuple:
        """R2/R14 共用重试键：KV 纪元 ⊕ 指定实例纪元（F3 粒度红线：
        只挂失败相关实例，严禁粗化为全实例纪元之或——实例纪元是列车
        事件粒度的高频信号，全挂会让重试门恒翻转、深尾部墙钟爆炸）。"""
        return (self._kv_ledger_epoch,
                self.instances[instance_index].ledger_epoch)

    def _enter_decode_stall(
        self, runtime, instance_index: int, reason: str,
        *, gap_records: tuple = (),
    ) -> None:
        if runtime.decode_stalled:
            runtime.stall_wake_key = self._wake_key(instance_index)
            runtime.stall_gap_records = tuple(gap_records)
            return
        runtime.decode_stalled = True
        runtime.stall_reason = reason
        runtime.stall_wake_key = self._wake_key(instance_index)
        runtime.stall_gap_records = tuple(gap_records)
        self._stalled_by_instance.setdefault(instance_index, set()).add(
            runtime)
        self.log_decision(
            {"kind": "joint_decode_stall", "request_id": runtime.request_id,
             "priority": 0},
            self._batch["tick"] if self._batch else 0,
            decision={
                "instance_index": instance_index,
                "session_id": runtime.session_id,
                "target_context_tokens": self._joint_working_context(runtime),
                "reason": reason,
            },
        )
        # 停滞成员本列车已不参与后续列车规划——当前在飞列车核销后由
        # _wake_stalled_decodes 在纪元变化时唤醒；全停滞死锁由
        # _check_decode_deadlock 守卫显式 fail-closed（N8：宁要有诊断
        # 信息的 abort，不要静默永久挂起）。

    def _wake_stalled_decodes(self, tick: int) -> None:
        """R14 唤醒 pass（每交付调用；无停滞时 O(1)）。

        唤醒必有新信息：仅重试唤醒键（KV 纪元 ⊕ 实例纪元）已变的停滞
        会话；键不变 = 无容量释放且无负载迁移，重试必返同样失败（无
        自旋）。审计行按 wake 事件计。"""
        if not self._stalled_by_instance:
            return
        for instance_index in sorted(self._stalled_by_instance):
            stalled = self._stalled_by_instance.get(instance_index)
            if not stalled:
                continue
            state = self.instances[instance_index]
            for runtime in sorted(stalled, key=lambda item: item.request_id):
                key = self._wake_key(instance_index)
                if runtime.stall_wake_key == key:
                    continue
                runtime.stall_wake_key = key
                target_context = self._joint_working_context(runtime)
                session = self.kv_manager._sessions.get(runtime.session_id)
                if session is None or session.context_tokens == target_context:
                    self._exit_decode_stall(runtime, instance_index, tick)
                    continue
                try:
                    evictions = self.kv_manager.expand_decode(
                        session_id=runtime.session_id,
                        instance_index=instance_index,
                        final_context_tokens=target_context,
                        trigger_request_id=runtime.request_id,
                    )
                except KVCapacityError as exc:
                    runtime.stall_reason = str(exc)
                    runtime.stall_gap_records = exc.deep_gap_records
                    if exc.evictions:
                        # D7 清账（自查 D 同款）：唤醒重试的已提交逐出
                        # 进图 + 流登记 + 纪元 bump。
                        self._bump_kv_ledger_epoch()
                        self.graph.sync_pending_history_after_evictions(
                            exc.evictions)
                        self._emit_eviction_only_nodes(
                            exc.evictions, tick,
                            trigger_request_id=runtime.request_id)
                    continue
                if evictions:
                    self._bump_kv_ledger_epoch()
                    self.graph.sync_pending_history_after_evictions(evictions)
                    # 自查 D 同款：唤醒增长的逐出进图 + 流登记。
                    self._emit_eviction_only_nodes(
                        evictions, tick,
                        trigger_request_id=runtime.request_id)
                self._exit_decode_stall(runtime, instance_index, tick)
                # 唤醒后本实例有新工作——刷新 frontier 让列车规划接手。
                self._refresh_frontier(state)

    def _exit_decode_stall(
        self, runtime, instance_index: int, tick: int,
    ) -> None:
        runtime.decode_stalled = False
        runtime.stall_wake_key = None
        stalled = self._stalled_by_instance.get(instance_index)
        if stalled is not None:
            stalled.discard(runtime)
            if not stalled:
                self._stalled_by_instance.pop(instance_index, None)
        self.log_decision(
            {"kind": "joint_decode_wake", "request_id": runtime.request_id,
             "priority": 0},
            tick,
            decision={
                "instance_index": instance_index,
                "session_id": runtime.session_id,
                "stall_reason": runtime.stall_reason,
            },
        )
        runtime.stall_reason = None

    def _check_decode_deadlock(self) -> None:
        """N8 死锁守卫：活跃全停滞 ∧ 无 qp 工作 ∧ 无在飞列车 → 显式
        fail-closed 上报（deep_gap 同类现场：实例/rank/停滞会话清单）。

        触发条件 = 系统不再有任何可推进事件源：停滞会话等容量、容量
        只能由逐出（inactive，已在失败尝试中耗尽）或活跃完成（停滞中
        不可达）释放——纪元永不再 bump，等待是永久的。宁要有诊断信息
        的 abort，不要静默挂起。

        K1（kimi 复审，2026-09-14）：逃逸条件必须含 pending_decode_
        ready——drain 到站的请求先落该队列、待业务规划阶段（_plan_and_
        emit_trains）才加入 decode 列车；其 decode 空间已在 drain 边界
        事务内落账（预约/迁移，非加入时分配），加入不会因容量失败。
        "最后一个未停滞请求刚 drain、其余全停滞"的窗口里它是唯一的
        推进事件源（列车发射 → in_flight_train），漏查会把合法 run 误
        判为死锁。

        自查 A（2026-09-15，K1 补全）：三类**跨 tick 事件源**同样必须
        逃逸——(a) _pending_merge_alarms 非空（merge 尾 watch 是真实图
        节点，C++ 必有交付 → _on_merge_done 注销 merge 流；C3b 后下一轮
        alarm 已随 _complete_requests 在响应完成点排定，"未来到达释放
        容量"的推链由 (c) 承载，本项保留"图侧必有交付"的事件源语义；
        _complete_requests 与 _admit_pass 同 tick 先后执行，完成批刚注册
        watch 的窗口里守卫必见非空）；(b) arrival_heap 非空（已见到达
        待 drain）；(c) manifest 尚有未到达请求（_arrived_request_count
        < len(runtimes) 即 C++ 侧已排 alarm——未来到达 → 新准入 → 逐出
        释放；runtimes 槽位完成后置 None，不得按槽位判）。守卫只在 EOF
        终态（无任何未来事件源）的全停滞才触发。"""
        if not any(state.active_decode for state in self.instances):
            return
        for state in self.instances:
            if state.in_flight_train is not None:
                return
            if state.pending_decode_ready:
                return
            for runtime in state.qp:
                if runtime.remaining_chunks > 0:
                    return
        if self._pending_merge_alarms:
            return
        if self.arrival_heap:
            return
        # 未来到达存在（runtimes 槽位完成后置 None，用到达计数判）：
        # _arrived_request_count < len(runtimes) 即 manifest 尚有请求
        # 未到达（C++ 侧已排 alarm → 事件源在途）。
        if self._arrived_request_count < len(self.runtimes):
            return
        for state in self.instances:
            for runtime in state.active_decode:
                if not runtime.decode_stalled:
                    return
        stalled_detail = {
            state.index: sorted(
                runtime.request_id
                for runtime in state.active_decode
                if runtime.decode_stalled)
            for state in self.instances if state.active_decode
        }
        # K6：停滞是可恢复路径、此前不落 deep_gap 台账；守卫判死即确认
        # 不可恢复——此刻提交各停滞会话**最近一次**缺口记录（每次停滞
        # 覆盖，无复利），台账恢复"落账 = run 终止"语义。
        terminal_records = []
        for state in self.instances:
            for runtime in state.active_decode:
                if runtime.decode_stalled and runtime.stall_gap_records:
                    self.kv_manager.commit_deep_gap_records(
                        runtime.stall_gap_records)
                    terminal_records.extend(runtime.stall_gap_records)
        raise RuntimeError(
            "joint decode-growth deadlock: every active decode session is "
            "stalled with no queued prefill work and no in-flight train -- "
            "no event source can ever release capacity (design sec.3.3 "
            "invariant 7). stalled={}; deep_gap_records={!r}".format(
                stalled_detail, terminal_records))

    def _on_merge_done(self, watch_id: str, tick: int) -> None:
        """merge 尾 watch 交付（C3b，2026-09-22 到达合同迁移后语义收窄）：

        ① 注销该请求的 merge 在途流（R15，职责保留）；
        ② 落 merge_done 事件侧披露行（决策日志 kind="merge_done"）——与
        completion 行的 service_done tick（REQUEST_COMPLETE 交付）分别
        可观测（F11 分报配套；C12 G5 缺口路由承接：物理 merge_done 此前
        无独立事件侧披露行，锚迁移后"下一轮到达时刻"代理消失）。
        ③ C14（2026-09-22）结算闭合审计：watch 交付 ⇔ FS 侧结算事实在
        案（kv_delta_journal 行；结算时刻逐请求披露通路——**不进 C5
        冻结的 joint_admission schema**，决策时刻记录）。
        下一轮到达 alarm 已在响应完成点（service_done + interval）排定，
        本回调**不再重排 alarm**；到达后数据门控的解除即本 watch 交付
        （事件驱动，保守 merge_done 整体口径——图侧 interval gate 前递
        锚的成员完成）。

        remote 结算生命周期锚点（C14 步骤 1 审计落字）：源端基础历史
        在远读有效期内保持源端有效并受结构保护（prepare 后 primary
        instance 指向执行端 ⇒ home 侧逐出候选集结构性不含本会话，且
        active 位守卫直接逐出入口；保护解除 = merge_back 败者侧释放 +
        mark_complete 同 tick，物理消费序由同 rank 链的顺序发射保持）。
        """
        pending = self._pending_merge_alarms.pop(watch_id, None)
        if pending is None:
            raise RuntimeError(
                "merge-done watch {} delivered without a pending entry".format(
                    watch_id))
        request_id = pending["request_id"]
        # C14 闭合门：有 merge 流 ⇒ merge_back 已结算 ⇒ journal 必有行。
        # 缺行 = watch 通道与结算账本脱钩（账本破损类），fail-closed。
        # F6 销账：真 KVCacheManager 恒定义 kv_delta_find（face_scheduler
        # :5277）——直接调用，替身 kv_manager 漏接口 = AttributeError。
        if self.kv_manager.kv_delta_find(request_id) is None:
            raise RuntimeError(
                "merge-done watch {} delivered but no settlement row exists "
                "for request {} (kv_delta_journal closure failure)".format(
                    watch_id, request_id))
        # C11 步骤 7（联合断言）：结算闭合门（上方 kv_delta 行在案）之后
        # 释放 merge 预留胜者侧——释放时序与 C14 的结算事实源联合锁定
        #（watch 交付 ⇔ journal 行 ⇒ 胜者侧释放必晚于结算落账）。未裁决
        # 条目 = watch 通道与预留账目脱钩，fail-closed（见
        # _quota_release_merge_reserve）。
        if self._quota_tracker is not None:
            self._quota_release_merge_reserve(request_id)
        self._release_transfer_flows(request_id + "#merge")
        self.log_decision(
            {"kind": "merge_done", "request_id": request_id,
             "priority": 0},
            tick,
            decision={
                "merge_done_ns": tick,
                "session_id": pending["session_id"],
                "next_turn_request_id": pending["following_request_id"],
                "next_arrival_world_ns": pending["next_arrival_world_ns"],
                # N2 可观测：thinktime < merge 时长时为 True——到达早于
                # merge_done，等待窗口 [arrival, merge_done] 落在下一轮
                # E2E 内（merge 成本以"下一轮数据等待"形式承接 2026-09-14
                # N2 关切）。K2：终轮无下一轮 ⇒ None（不参与比较）。
                "next_turn_arrived_before_merge_done": (
                    (tick > pending["next_arrival_world_ns"])
                    if pending["next_arrival_world_ns"] is not None
                    else None),
            },
        )

    def _next_eviction_watch_id(self, request_id: str, phase: str) -> str:
        key = (request_id, phase)
        sequence = self._eviction_watch_seq.get(key, 0)
        self._eviction_watch_seq[key] = sequence + 1
        return "batch_train_evict_{}_{}_{:04d}".format(
            request_id, phase, sequence)

    def _register_eviction_watch(
            self, watch: dict, *, flow_owners=(), quota_owners=()) -> None:
        """把真实旁支尾标记挂入本 batch，并保留其唯一释放义务。

        Decode joiner 可在 drain 时先登记流、稍后等列车构图；此时同一
        watch id 已有 scheduled=False 条目，构图阶段只把它转成 scheduled。
        空 shard 分支没有成员，GraphBatchBuilder 不会返回 watch，因而绝
        不会在此造一个永远等不到的 handle。
        """
        watch_id = watch.get("request_id")
        request_id = watch.get("owner_request_id")
        members = watch.get("members")
        if (not isinstance(watch_id, str)
                or not watch_id.startswith("batch_train_evict_")
                or not isinstance(request_id, str)
                or not isinstance(members, dict) or not members):
            raise RuntimeError("malformed or empty eviction watch")
        if self._batch is None:
            raise RuntimeError("eviction watch registered outside a batch")
        pending = self._pending_eviction_watches.get(watch_id)
        if pending is None:
            pending = {
                "request_id": request_id,
                "flow_owners": tuple(flow_owners),
                "quota_owners": tuple(quota_owners),
                "scheduled": False,
            }
            self._pending_eviction_watches[watch_id] = pending
        else:
            if pending["request_id"] != request_id:
                raise RuntimeError(
                    "eviction watch {} changed owner from {} to {}".format(
                        watch_id, pending["request_id"], request_id))
            if pending["scheduled"]:
                raise RuntimeError(
                    "eviction watch {} was scheduled twice".format(watch_id))
            if flow_owners or quota_owners:
                raise RuntimeError(
                    "pre-registered eviction watch {} received duplicate "
                    "owners".format(watch_id))
        pending["scheduled"] = True
        self._batch["watches"].append({
            "request_id": watch_id,
            "stage": STAGE_PREFILL,
            "generation": 0,
            "members": members,
            "statuses": ["Success", "Skipped"],
        })

    def _on_eviction_watch(self, watch_id: str, stage: str, tick: int) -> None:
        """仅在逐出尾标记真实完成后注销该旁支自己的流与配额。"""
        pending = self._pending_eviction_watches.get(watch_id)
        if pending is None:
            raise RuntimeError(
                "unknown or duplicate eviction watch {}".format(watch_id))
        if not pending["scheduled"]:
            raise RuntimeError(
                "eviction watch {} fired before it was scheduled".format(
                    watch_id))
        if stage != STAGE_PREFILL:
            raise RuntimeError(
                "eviction watch {} fired on unexpected stage {!r}".format(
                    watch_id, stage))
        del self._pending_eviction_watches[watch_id]
        for owner in pending["flow_owners"]:
            self._release_transfer_flows(owner)
        for owner in pending["quota_owners"]:
            self._quota_release_owner(owner, now_ns=tick)

    def _register_train_eviction_watches(self, result: dict) -> None:
        for watch in result.get("eviction_watches", ()):
            watch_id = watch["request_id"]
            pending = self._pending_eviction_watches.get(watch_id)
            if pending is None:
                raise RuntimeError(
                    "decode eviction watch {} has no drain-time owner "
                    "registration".format(watch_id))
            self._register_eviction_watch(watch)

    def _quota_enroll_eviction_branch(
            self, owner: str, request_id: str, transfers, now_ns: int):
        """Optional oneshot quota entry owned by one physical branch watch."""
        if self._quota_tracker is None:
            return None
        edges, ports = self._quota_transfer_footprint(transfers)
        if not (edges or ports):
            return None
        if self._quota_admit_flow(
                owner, request_id, flow_class=FLOW_ONESHOT, links=edges,
                **({"port_id": ports[0]} if ports else {}),
                now_ns=now_ns):
            return owner
        self._quota_oneshot_overflow_events += 1
        self.log_decision(
            {"kind": "quota_oneshot_overflow", "request_id": request_id,
             "priority": 0},
            now_ns,
            decision={"owner": owner,
                      "note": "eviction branch oneshot enrollment "
                              "deferred by quota gate; occupancy-only "
                              "disclosure (D4 accounting continues on "
                              "release path)"},
        )
        return None

    # ------------------------------------------------------ R15 流登记 --

    def _register_transfer_flows(
            self, transfers, *, owner: str,
            serial_credit_stream: bool = False) -> None:
        """R15-1/F-B：发射时点登记在途流——逐 shard 完整链路序列（TP
        并行全路径，非代表对）+ 池端口占用（remote_store/remote_load 的
        edge 端口）+ HBM 端点端口（C2/F4：noc_migrate 的源端口 home 读
        腿恒有、目标端口 exec 写腿对 copy/merge/remote-read 恒有——A4'
        补价后计价与执行同腿型；中间跳 rank 不触碰端点 HBM）。注销由
        完成事件驱动（drain/completion/逐切片/merge-watch），决策时刻
        快照因果可见。

        ``serial_credit_stream`` 仅用于 remote-read 的 ``#readplan`` 与
        ``#decode#j`` owner：一个 owner 可携带多个未来切片或当前列车的
        多个真实 credit block，但图侧对每 rank 的 recv block 建顺序链，
        同请求同 shard 路径同刻最多一条 credit stream。注册表只计该
        owner 中每个物理 shard 路径一次；完整 block 列表仍保留给图发射、
        字节统计和计价。不同 TP shard 即使共享链路仍分别登记。"""
        seen_credit_shards = set()
        for transfer in transfers or ():
            if transfer.kind == "local_hit":
                continue
            for shard in transfer.shards:
                if serial_credit_stream:
                    # credit blocks / readplan units 是同一物理 shard 路径
                    # 上的串行工作量，不是同刻并发副本。键按 shard 身份
                    # 区分 TP 流，同时忽略 block 字节数以折叠顺序块。
                    shard_key = (
                        transfer.kind, shard.source_rank,
                        shard.target_rank, shard.edge_rank,
                        tuple(shard.noc_path))
                    if shard_key in seen_credit_shards:
                        continue
                    seen_credit_shards.add(shard_key)
                self._joint_flows.register_path(
                    shard.noc_path, owner=owner)
                if shard.edge_rank is not None:
                    self._pool_ports.register(shard.edge_rank, owner=owner)
                if (transfer.kind == "noc_migrate"
                        and len(shard.noc_path) >= 2):
                    # C2/F4：池路径（remote_load/remote_store）的端点
                    # HBM 经池端口模型计价（F3 边界），不进 HBM 端口表。
                    self._hbm_ports.register(shard.noc_path[0], owner=owner)
                    self._hbm_ports.register(shard.noc_path[-1], owner=owner)

    def _release_transfer_flows(self, owner: str) -> None:
        self._joint_flows.release_owner(owner)
        self._pool_ports.release_owner(owner)
        # C2：HBM 端点端口随同注销（同 owner 同生命周期；幂等空放）。
        # F6 销账：__init__ 恒设 _hbm_ports——直接访问（替身漏设 =
        # AttributeError，端口登记/注销静默丢失已不可能）。
        self._hbm_ports.release_owner(owner)

    # --------------------------- C11（WP3c）链路 ∪ HBM 端口动作级准入 ----

    def _quota_r_hat_kv_bytes_per_ns(
            self, representative_context_tokens: int | None = None) -> float:
        """r̂_KV（§4.2 平价门分母来源）：负载视图派生的每流 KV 消费速率。

        口径（因果、不假设恒满带宽）：上下文 = 当前活跃 decode 的平均
        当前上下文（无活跃 decode 时取调用方代表值，缺省 1——冷启动物理
        常数同 CausalHorizonEstimator 姿态）；每步 KV 读字节 = 逐 rank
        shard 字节（context+1 token 增量，均匀 TP 切分的均值）；步时长
        = roofline 单步负载（与 _decode_step_load_ns 同式）。二者均决策
        时点可得，无 oracle。

        L4（P2-4，2026-09-23 复核审计）两口径窗披露（不改数值）：
        ① AIMD observe（:2932）/port_snapshot（:4097）两调用点无
        session 语境不传代表值——active_decode 瞬空而 remote-read 读流
        在册的窗口（准入后、drain 入批前）回退 ctx=1（r̂ 最小值）⇒
        AIMD 判据 rate ≥ 1.2·r̂ 恒 comfort；O6②（终轮审计）已闭环该
        残压：回退窗 observe_telemetry 传 allow_expansion=False（扩张
        冻结、舒适 streak 不计入、解冻后重新累计——解冻瀑布 structurally
        杜绝；代价 disclosure 侧恒 comfort 的窗口不再驱动 Q 增长，K1
        缩窗 + 本冻结双保险）；② decode_tokens_
        consumed 仅列车核销点推进 ⇒ "当前上下文"实为最近列车结算点
        上下文，长列车在飞期间系统性低估（平价门偏宽/AIMD 收缩阈值
        偏低，乐观方向）——既有账粒结构。
        """
        contexts = []
        for state in self.instances:
            # K1（P1-①，2026-09-23 外部审计）：active_decode 成员是
            # _OnlineRequestRuntime 对象（入批 :1165-1169 / 退出 :941-942
            # 同证）——原实现把成员当 request_id 字符串查
            # runtime_by_request_id.get() 恒 None，contexts 恒空、恒退
            # 代表值；直接迭代对象取当前上下文。
            for member in state.active_decode:
                contexts.append(
                    member.prefill_context_tokens
                    + member.decode_tokens_consumed)
        if contexts:
            context = max(
                1, int(round(sum(contexts) / len(contexts))))
        else:
            context = max(1, int(representative_context_tokens or 1))
        step_shards = kv_cache_shard_bytes_for_tokens(
            self.model, context + 1, self.kv_manager.tp_degree)
        bytes_per_step = sum(step_shards) / len(step_shards)
        step_ns = self._decode_task_load_ns_cached(
            instance_size=self.topology.instances[0].size,
            current_context_tokens=context,
            generated_tokens=0,
            average_decode_length=1.0,
            running_step_fraction_remaining=1.0)
        return bytes_per_step / max(1.0, float(step_ns))

    def _quota_route_edges(self, source_instance: int,
                           target_instance: int) -> tuple:
        """候选跨实例流的有向链路多重集（逐 shard 完整 XY 路径的边序）
        ——与 C8 预登记 / JCM register_path 同足迹（同一路由源）。"""
        from face_scheduler import deterministic_xy_route
        edges: list[tuple[int, int]] = []
        source_group = self.topology.instance(source_instance)
        target_group = self.topology.instance(target_instance)
        for source_rank, target_rank in zip(
                source_group.ranks, target_group.ranks):
            path = deterministic_xy_route(
                self.hardware, source_rank, target_rank)
            for hop_a, hop_b in zip(path, path[1:]):
                edges.append((hop_a, hop_b))
        return tuple(edges)

    def _quota_transfer_footprint(self, transfers) -> tuple:
        """transfers → (链路多重集, 端口集)——与 _register_transfer_flows
        同判据（local_hit 跳过；noc_migrate 双端点入端口，池路径端点不
        入——F3 边界）。端口粒度 = 端点 rank 所属实例（TP 全 rank 同账，
        每 TP 并行流对实例各占 1 条，见 __init__ C11 块注释）。"""
        edges: list[tuple[int, int]] = []
        ports: list[int] = []
        for transfer in transfers or ():
            if transfer.kind == "local_hit":
                continue
            for shard in transfer.shards:
                for hop_a, hop_b in zip(shard.noc_path, shard.noc_path[1:]):
                    edges.append((hop_a, hop_b))
                if (transfer.kind == "noc_migrate"
                        and len(shard.noc_path) >= 2):
                    for endpoint_rank in (shard.noc_path[0],
                                          shard.noc_path[-1]):
                        port = self._rank_to_instance.get(endpoint_rank)
                        if port is not None and port not in ports:
                            ports.append(port)
        return tuple(edges), tuple(ports)

    def _quota_candidate_verdict(self, candidate, session_view,
                                 r_hat_kv: float) -> QuotaVerdict:
        """动作级准入裁决（C11 步骤 1：构建代价模型时逐候选执行配额
        判据；实例永不掩码——判据不过只把该**动作**标记不适用）。

        判据集合（link_quota 冻结语义的只读镜像，经模块公共谓词求值，
        不做试探性借还——配额代数不受候选评估扰动）：

        * stay/recompute：无跨实例主流程（本地动作）——恒适用（准入相
          逐出支链按 oneshot 事后入册收紧后续判据，D4）；
        * copy（oneshot）：链路门（占用+预留+新增 <= Q）；
        * remote-read（realtime）：链路门（读流需求与 merge 预留需求
          同事务 per-link 取 max——N1 时序复用）+ 双端点端口平价门
          （B_HBM/(u_port+1) >= r̂_KV，u_port = 活跃 decode 消费流 +
          在册流）+ 双候选胜者侧 merge bulk 名额余量（reserve_merge 的
          端口判据前瞻，封顶 N_bulk = Q_init）。

        δ_adm 合取确认（A3' 终冻 = 0）：重选沿用既有全候选 argmin
        （(cost_ns, order_key) 字典序），delta_adm = 0 时
        challenger_flips 谓词与该比较恒等（margin >= 0 同真域）——
        组合语义 ≡ argmin，不借配额接入夹带选择规则变更（不改值）。
        """
        tracker = self._quota_tracker
        action = candidate.action
        target = candidate.instance_index
        if action in ("stay", "recompute"):
            return QuotaVerdict(admitted=True)
        source = (
            session_view.resident_instance
            if session_view.resident_instance is not None
            else session_view.home_instance)
        if source is None or source == target:
            # 无跨实例流（与计价适用性同判据的防御分支；正常路径
            # copy/remote-read 的适用候选必有异地源）。
            return QuotaVerdict(admitted=True)
        demand: dict[tuple[int, int], int] = {}
        for edge in self._quota_route_edges(source, target):
            demand[edge] = demand.get(edge, 0) + 1
        if action == "remote-read":
            # N1（2026-09-23 复核审计1）：读流与 merge bulk 写是同一
            # 事务的时序先后阶段（读流 settle → 计算 → service_done
            # 裁决 → merge 写），同事务同链峰值 = **max**（读阶段、写
            # 阶段各自需求）而非 sum——sum 叠加使空链路 Q=2 时 TP=2
            # 共链 need=3 恒拒、ρ<1（Q=1）结构性排除远读（违反 C9
            # "ρ<1 域空化由定价涌现"）。跨事务叠加不变（他事务占用/
            # 预留在 remaining 分母）。merge 预留需求 = 逐方向路径
            # 链路集合各 1 槽（dedup）；与入册半 reserve_merge 的同
            # 事务借槽语义同构，杜绝"判据过而入册拒"。
            merge_demand: dict[tuple[int, int], int] = {}
            for edge in set(self._quota_route_edges(source, target)):
                merge_demand[edge] = merge_demand.get(edge, 0) + 1
            for edge in set(self._quota_route_edges(target, source)):
                merge_demand[edge] = merge_demand.get(edge, 0) + 1
            for edge, merge_need in merge_demand.items():
                demand[edge] = max(demand.get(edge, 0), merge_need)
        for link, need in demand.items():
            remaining = tracker.link_remaining(link)
            if remaining < need:
                return QuotaVerdict(
                    admitted=False,
                    flow_class=(
                        FLOW_REALTIME if action == "remote-read"
                        else FLOW_ONESHOT),
                    resource_kind="link",
                    resource_id=link,
                    remaining=remaining,
                    wait_reason=WAIT_QUOTA_LINK,
                    inapplicable_reason=(
                        "quota_link: link={} remaining={} of Q={}, "
                        "need={} (action={} {}->{}; occupancy={}, "
                        "reserved={})".format(
                            link, remaining, tracker.link_quota(link), need,
                            action, source, target,
                            tracker.link_occupancy(link),
                            tracker.link_reserved(link))),
                )
        if action == "remote-read":
            for port in (target, source):
                u_port_base = len(self.instances[port].active_decode)
                if not tracker.port_parity_admits(
                        port, r_hat_kv, u_port_base=u_port_base):
                    return QuotaVerdict(
                        admitted=False,
                        flow_class=FLOW_REALTIME,
                        resource_kind="port",
                        resource_id=port,
                        remaining=0,
                        wait_reason=WAIT_QUOTA_PORT,
                        inapplicable_reason=(
                            "quota_port: parity gate at instance {} rejects "
                            "realtime stream (u_port_base={}, enrolled={}, "
                            "r_hat_kv={:.6g} B/ns; busy port admits "
                            "nothing)".format(
                                port, u_port_base,
                                tracker.port_enrolled(port), r_hat_kv)),
                    )
            for port in (target, source):
                if tracker.bulk_remaining(port) < 1:
                    return QuotaVerdict(
                        admitted=False,
                        flow_class=FLOW_ONESHOT,
                        resource_kind="port",
                        resource_id=port,
                        remaining=0,
                        wait_reason=WAIT_QUOTA_PORT,
                        inapplicable_reason=(
                            "quota_port: merge bulk-write slots at instance "
                            "{} exhausted (used={} of N_bulk={}; remote-read "
                            "merge obligation reserve)".format(
                                port, tracker.bulk_used(port),
                                tracker.n_bulk)),
                    )
        return QuotaVerdict(admitted=True)

    def _quota_filter_candidates(self, record, session_view, rid: str,
                                 now_ns: int):
        """逐候选配额判据 + 配额可行集内重选（argmin 同序）。

        返回 (chosen, candidates, deferred)：deferred 非 None = 配额导致
        全动作不可行（C9 冻结语义：回 pending_admissions——quota_deferred_
        requeue 模块级实现，**永不 raise**；重试键扩展配额代数）。"""
        tracker = self._quota_tracker
        r_hat_kv = self._quota_r_hat_kv_bytes_per_ns(
            session_view.history_tokens)
        verdict_wall_start = time.perf_counter_ns()
        verdicts_by_action: dict[str, QuotaVerdict] = {}
        filtered: list = []
        for candidate in record.candidates:
            if not (candidate.applicable
                    and candidate.cost_ns is not None):
                filtered.append(candidate)
                continue
            verdict = self._quota_candidate_verdict(
                candidate, session_view, r_hat_kv)
            self._quota_verdict_candidate_checks += 1
            if not verdict.admitted:
                # M2（2026-09-23 验收审计）：配额拒保留 cost_ns——
                # applicable=False 已挡 feasible/argmin（:2690 同判据），
                # 决策行 candidates 保留配额前预测成本 ⇒ 离线双域
                # （D_econ/配额域之差 = 配额作用量）可从日志重建
                # （修前清 None ⇒ quota-on run 的经济域被配额自身裁
                # 掉、日志无法重建配额前域）。
                filtered.append(_dataclass_replace(
                    candidate, applicable=False,
                    inapplicable_reason=verdict.inapplicable_reason))
                verdicts_by_action.setdefault(candidate.action, verdict)
            else:
                filtered.append(candidate)
        self._quota_verdict_wall_ns_total += (
            time.perf_counter_ns() - verdict_wall_start)
        feasible = [
            candidate for candidate in filtered
            if candidate.applicable and candidate.cost_ns is not None]
        if not feasible:
            deferred = quota_deferred_requeue(
                request_id=rid,
                verdicts_by_action=verdicts_by_action,
                quota_retry_key=tracker.quota_retry_key(),
                base_retry_key=(self._kv_ledger_epoch, ()),
            )
            return None, tuple(filtered), (deferred, r_hat_kv)
        chosen = min(
            feasible,
            key=lambda candidate: (
                candidate.cost_ns, candidate.order_key()))
        return chosen, tuple(filtered), None

    def _quota_admit_flow(self, owner: str, rid: str, **kwargs) -> bool:
        """admit_flow 包装（借还配对登记 + 计数；重复 owner = 调用方
        bug，fail-closed 交模块守卫；判据失败返回 False 零登记——由
        调用方回滚已入册流，杜绝半册）。"""
        verdict = self._quota_tracker.admit_flow(owner=owner, **kwargs)
        if not verdict.admitted:
            return False
        self._quota_enrolled[owner] = rid
        self._quota_admit_events += 1
        return True

    def _quota_release_owner(self, owner: str, now_ns: int | None) -> None:
        """配对释放（幂等空放——未知 owner 跳过；已知必成对）。"""
        if owner not in self._quota_enrolled:
            return
        self._quota_tracker.release_flow(owner, now_ns=now_ns)
        del self._quota_enrolled[owner]
        self._quota_release_events += 1

    def _quota_enroll_admission(self, runtime, session_view, action: str,
                                target: int, now_ns: int) -> bool:
        """事务成功后的配额入册（C11 步骤 1 的入册半）：

        * copy：oneshot 主流程（owner=rid 双端口 = 写腿 exec + 读腿
          home 源侧 #src；drain 释放）；
        * remote-read：realtime 读流（owner=rid#readplan——与 C8 预登记
          同 owner 约定、同生命周期；完成边界释放）+ merge 义务预留
          （reserve_merge 双向各 1 槽 + 双候选胜者侧 bulk 名额；量级
          分层 = 双向各 1 槽原语，copy/recompute 轮零预留）；
        * 逐出/池恢复支链（全部动作，D4 通用治理）：oneshot 入册
          （owner=rid#evict；逐出尾 watch 交付后释放）——**恒置于主流程之后**
          （O2 序不变量，见下）。

        O2（2026-09-23 终轮审计，事故链写实）：判据时刻本请求的逐出
        足迹**不可知**——history_evictions 由准入事务产出（受害选择依赖
        执行期 KV 账本，不做试探性事务即不可前瞻）、事务成功才落账
        （:4449），而判据在事务前执行（:2705）⇒ 判据 demand 只含主流程
        足迹。探针实证事故链：判据 PASS → 支链先入册占掉末槽 → 主流程
        #readplan 入册 FAIL → LinkQuotaError 返回 False → 调用方对已物化
        预约 fail-closed RuntimeError 整 run abort（A11' 无 un-prepare
        逆路径的必然终点）。修法 = 入册侧分派：主流程恒先于支链——
        单线程内判据与入册之间 tracker 零变更、主流程需求判据已前瞻
        ⇒ 主流程入册失败只剩真异常（保持 raise → 回滚已入册 → False，
        调用方 fail-closed）；支链失败 = 判据不可前瞻的固有残差，按
        quota_oneshot_overflow 披露**降级**（与 decode 相 oneshot 溢出
        同姿态：执行侧流已由 kv_manager/graph 落账，回滚不可行，不
        abort run）。释放路径 _quota_release_owner 幂等空放，未入册
        半边（#evict）在 drain/完成边界无副作用；成功入册的支链由独立
        eviction_done watch 负责结算。
        """
        tracker = self._quota_tracker
        rid = runtime.request_id
        source = (
            session_view.resident_instance
            if session_view.resident_instance is not None
            else session_view.home_instance)
        enrolled: list[str] = []
        try:
            cross_instance = source is not None and source != target
            if action == "copy" and cross_instance:
                edges = self._quota_route_edges(source, target)
                if not self._quota_admit_flow(
                        rid, rid, flow_class=FLOW_ONESHOT, links=edges,
                        port_id=target, now_ns=now_ns):
                    raise LinkQuotaError(
                        "copy primary enrollment failed (verdict flip "
                        "inside the admission transaction)")
                enrolled.append(rid)
                if not self._quota_admit_flow(
                        rid + "#src", rid, flow_class=FLOW_ONESHOT,
                        links=(), port_id=source, now_ns=now_ns):
                    raise LinkQuotaError(
                        "copy source-side enrollment failed (verdict flip "
                        "inside the admission transaction)")
                enrolled.append(rid + "#src")
            if action == "remote-read" and cross_instance:
                r_hat_kv = self._quota_r_hat_kv_bytes_per_ns(
                    session_view.history_tokens)
                edges = self._quota_route_edges(source, target)
                reverse_edges = self._quota_route_edges(target, source)
                if not self._quota_admit_flow(
                        rid + "#readplan", rid, flow_class=FLOW_REALTIME,
                        links=edges, port_id=target,
                        r_hat_kv_bytes_per_ns=r_hat_kv,
                        u_port_base=len(
                            self.instances[target].active_decode),
                        now_ns=now_ns):
                    raise LinkQuotaError(
                        "remote-read stream enrollment failed (verdict "
                        "flip inside the admission transaction)")
                enrolled.append(rid + "#readplan")
                if not self._quota_admit_flow(
                        rid + "#readplan#src", rid, flow_class=FLOW_REALTIME,
                        links=(), port_id=source,
                        r_hat_kv_bytes_per_ns=r_hat_kv,
                        u_port_base=len(
                            self.instances[source].active_decode),
                        now_ns=now_ns):
                    raise LinkQuotaError(
                        "remote-read source-side enrollment failed "
                        "(verdict flip inside the admission transaction)")
                enrolled.append(rid + "#readplan#src")
                # 预留链路集 = 逐方向 dedup（C9"各 1 槽"= 路径链路
                # 集合各 1 槽，非逐 shard 多槽——与判据半同构）。
                verdict = tracker.reserve_merge(
                    rid,
                    links_forward=set(edges),
                    links_reverse=set(reverse_edges),
                    port_forward=target, port_reverse=source)
                if not verdict.admitted:
                    raise LinkQuotaError(verdict.inapplicable_reason)
                self._quota_merge_reserves[rid] = {
                    "state": "reserved",
                    "source": source, "target": target}
                self._quota_merge_reserves_created += 1
        except LinkQuotaError:
            for owner in enrolled:
                self._quota_release_owner(owner, now_ns=now_ns)
            return False
        # O2 序不变量：逐出支链入册恒在主流程之后（判据时刻本请求逐出
        # 足迹不可知——demand 未前瞻，支链若在主流程前入册可在末槽上
        # 把主流程撞成 fail-closed abort；主先支后使该链结构性不可达）。
        # 支链失败 = 不可前瞻的固有残差 ⇒ 按披露降级，不 abort run（与
        # _quota_enroll_decode_phase 同姿态）；主流程失败在上面的
        # try 内 raise（真异常——单线程内判据与入册间 tracker 零变更，
        # 主流程需求判据已前瞻）。
        if runtime.history_evictions:
            edges, ports = self._quota_transfer_footprint(
                runtime.history_evictions)
            if edges or ports:
                if not self._quota_admit_flow(
                        rid + "#evict", rid, flow_class=FLOW_ONESHOT,
                        links=edges,
                        **({"port_id": ports[0]} if ports else {}),
                        now_ns=now_ns):
                    # O12：事件计入 run 级配额指标（quota_events.
                    # oneshot_overflows）。
                    self._quota_oneshot_overflow_events += 1
                    self.log_decision(
                        {"kind": "quota_oneshot_overflow",
                         "request_id": rid, "priority": 0},
                        now_ns,
                        decision={"owner": rid + "#evict",
                                  "note": "admission eviction-branch "
                                          "oneshot enrollment deferred by "
                                          "quota gate; this request's own "
                                          "eviction footprint is not "
                                          "foreseeable at verdict time "
                                          "(O2) -- occupancy-only "
                                          "disclosure, D4 accounting "
                                          "continues on release path"},
                    )
        return True

    def _quota_enroll_decode_phase(self, request_id: str, transfers,
                                   now_ns: int) -> None:
        """decode 相在途流的 oneshot 入册（drain 边界登记 + 增长事件
        追加——每次事件独立 owner #decode#{seq}；完成边界释放）。"""
        edges, ports = self._quota_transfer_footprint(transfers)
        if not (edges or ports):
            return
        seq = self._quota_decode_owner_seq.get(request_id, 0)
        self._quota_decode_owner_seq[request_id] = seq + 1
        owner = "{}#decode#q{}".format(request_id, seq)
        if not self._quota_admit_flow(
                owner, request_id, flow_class=FLOW_ONESHOT, links=edges,
                **({"port_id": ports[0]} if ports else {}),
                now_ns=now_ns):
            # 防御：候选层未前瞻的 decode 相事件（逐出支链）——oneshot
            # 无端口门、链路门在深占用时可能拒；按披露计数不阻断执行
            # （执行侧流已由 kv_manager/graph 落账，回滚不可行）。
            # O12：事件计入 run 级配额指标（quota_events.oneshot_overflows）。
            self._quota_oneshot_overflow_events += 1
            self.log_decision(
                {"kind": "quota_oneshot_overflow", "request_id": request_id,
                 "priority": 0},
                now_ns,
                decision={"owner": owner,
                          "note": "decode-phase oneshot enrollment "
                          "deferred by link gate; occupancy-only "
                          "disclosure (D4 accounting continues on "
                          "release path)"},
            )

    def _quota_release_admission_phase(self, request_id: str,
                                       tick: int) -> None:
        """drain 边界：准入相流（copy 主流程 + 准入逐出）settle。"""
        for owner in (request_id, request_id + "#src"):
            self._quota_release_owner(owner, now_ns=tick)

    def _quota_release_readplan_stream(self, request_id: str,
                                       tick: int) -> None:
        """完成边界：remote-read 读流（decode 相全程）settle——流寿命
        样本（admit/release 双时标）随释放进入 AIMD EWMA。"""
        for owner in (request_id + "#readplan",
                      request_id + "#readplan#src"):
            self._quota_release_owner(owner, now_ns=tick)
        self._quota_decode_owner_seq.pop(request_id, None)

    def _quota_release_decode_phase(self, request_id: str,
                                    tick: int) -> None:
        """完成边界：decode 相增量流（#decode#q{seq}）settle。"""
        prefix = request_id + "#decode#q"
        for owner in [key for key in self._quota_enrolled
                      if key.startswith(prefix)]:
            self._quota_release_owner(owner, now_ns=tick)

    def _quota_on_service_done(self, request_id: str, runtime,
                               tick: int) -> None:
        """C11 步骤 7（#merge-reserve 释放时机，C14 移交）：service_done
        （REQUEST_COMPLETE 交付 tick）时刻方向裁决——释放败者侧；胜者侧
        保持到 _on_merge_done（与 C14 kv_delta_journal 结算闭合门联合
        断言：watch 交付 ⇔ journal 行在案 ⇒ 胜者侧释放必在结算事实之后）。

        零传输分支（stay/in_place/零字节翻转 ⇒ merge_transfers 为空 ⇒ 无
        merge watch）在 service_done 即整体释放（release_merge 未裁决
        双侧撤销语义），不等待不存在的 merge_done 事件。"""
        reserve = self._quota_merge_reserves.get(request_id)
        if reserve is None:
            return
        outcome = runtime.merge_outcome
        direction = outcome.get("direction") if outcome else None
        if direction in ("forward", "reverse") and runtime.merge_transfers:
            winner_instance = outcome.get("winner_instance")
            winner = (
                MERGE_DIRECTION_FORWARD
                if winner_instance == reserve["target"]
                else MERGE_DIRECTION_REVERSE)
            self._quota_tracker.adjudicate_merge_direction(
                request_id, winner)
            reserve["state"] = "adjudicated"
            reserve["winner"] = winner
            return
        self._quota_tracker.release_merge(request_id)
        del self._quota_merge_reserves[request_id]

    def _quota_release_merge_reserve(self, request_id: str) -> None:
        """merge_done（_on_merge_done）释放胜者侧（C9 冻结：胜者侧到
        _on_merge_done）。联合断言：到达此处须已裁决（有 merge 传输 ⇒
        service_done 已裁决）；未裁决条目 = watch 通道与预留账目脱钩，
        fail-closed。胜者侧预留槽即实际合并流的占用（预留→实占转换：
        预留自准入持续持有至 merge_done，1 槽连续覆盖实际传输，无双重
        计数——C11 裁定披露）。"""
        reserve = self._quota_merge_reserves.pop(request_id, None)
        if reserve is None:
            return
        if reserve.get("state") != "adjudicated":
            raise RuntimeError(
                "merge-done watch delivered for {} but its quota "
                "merge reserve was never adjudicated at service_done "
                "(reserve/watch channel desynchronized)".format(request_id))
        self._quota_tracker.release_merge(request_id)

    def _quota_ingest_telemetry(self, now_ns: int) -> None:
        """C11 步骤 8（AIMD 闭环）：C8 `_ingest_link_telemetry` 产出的
        有效速率喂 link_quota 的 observe_telemetry（仅 aimd 模式）。

        键换算契约（A5'/B3 rider 后）：`_link_telemetry_rates` 已按端点
        形 (src_rank, dst_rank) 键控——与配额链路键空间（同 JCM 注册表
        的 rank 对）一致。每流速率换算 = 链路实测速率 ÷ 该链路分母
        （无在册流的链路不出现在字典——C9/C10 冻结契约"无测量即无
        信号"）。M1（2026-09-23 验收审计⑥）：分母 = max(配额在册流
        数, C++ 时间加权活跃流数)——实测速率含 collective 等 non-KV
        流量，只除在册流会把 collective 的份额归因给 KV 流（KV 流+
        collective 均分 B 时输入 B ≥ 1.2·r̂ 恒 comfort——真实份额 B/2
        应收缩：反向信号，F6 披露"保守向"用词据此订正）；流数缺席
        （旧二进制）退在册流数。r̂_KV = 负载视图现值（因果）。

        O6①（2026-09-23 终轮审计）：**空窗也推进遥测序号**——原
        `per_flow` 空即早退使 observe_telemetry 不被调用、_telemetry_seq
        停摆：断供/全闲 epoch 不消耗序号 ⇒ 缺测后首个样本的 dt 含整段
        断供时长，K7 连续采样纪律（contiguous 判定）被结构性架空（断供
        期 dt 误计 quiet/进度，单样本触发扩张的缺测变体）。改为恒调用
        observe_telemetry（per_flow 空 = 空字典采样：零链路动作，仅
        序号/遥测钟簿记推进——断供期 dt 自此不计入）。

        O6②（2026-09-23 终轮审计）：r̂ 回退窗冻结扩张——active_decode
        瞬空时 _quota_r_hat_kv_bytes_per_ns 退 ctx=1 代表值（r̂ 最小值），
        该代表信号不满足扩张的资格（非真实舒适证据）⇒ allow_expansion
        = False（additive-increase 冻结：舒适判定照常披露，streak 进度
        不计入、解冻后重新累计 T_expand——防解冻瀑布）。"""
        tracker = self._quota_tracker
        if tracker is None or tracker.mode != QUOTA_AIMD:
            return
        per_flow: dict = {}
        for link, rate in self._link_telemetry_rates.items():
            occupancy = tracker.link_occupancy(link)
            if occupancy <= 0:
                continue
            denominator = max(
                occupancy,
                self._link_telemetry_flow_counts.get(link, 0.0))
            per_flow[link] = rate / denominator
        # O6②：代表值回退判定 = active_decode 全实例瞬空（与
        # _quota_r_hat_kv_bytes_per_ns 内部回退分支同条件同源）。
        has_active_decode = any(
            state.active_decode for state in self.instances)
        disclosure = tracker.observe_telemetry(
            now_ns, per_flow,
            r_hat_kv_bytes_per_ns=self._quota_r_hat_kv_bytes_per_ns(),
            allow_expansion=has_active_decode)
        for action in {entry["action"]
                       for entry in disclosure["links"].values()}:
            self._quota_aimd_action_counts[action] = (
                self._quota_aimd_action_counts.get(action, 0) + 1)

    def _telemetry_endpoint_link_key(self, link_id: int):
        """A5'/B3（C7 移交 rider）：整型 LinkId → (src, dst) 端点键。

        换算配方 = C++ MultiDimTopology::connect_dimension 确定性枚举
        （逐维升序 src、connect(src, src+stride, bidirectional=true)
        顺次 append 正/反向两条 LinkId；strides = 维大小前缀积）。dims
        = network.yml npus-count = [mesh_cols, mesh_rows]
        （config_resolver.py 同源），C++ 平层 rank = col + mesh_cols
        ×row ≡ Python rank 空间——端点键与 LinkFlowRegistry/quota 链路
        键空间一致，JCM divisor_effective 的 max 合并自此在生产路径
        真实生效。C7 对拍已验证 186/186 吻合 spool link_count 真值。
        未知 LinkId（生产路径）= 注入逻辑损坏（拓扑维序与 C++ 枚举不
        一致），fail-closed。F6 销账：hasattr 软门已删——__init__ 恒设
        _telemetry_link_id_map（None，首样本惰性构建）；__new__ 测试替身
        走本方法必须显式补该属性：要么补 None（替身同时须装配 hardware
        网格，映射真实构建），要么预置整型恒等映射 {i: i}（等价旧退化
        路径、语义自担），要么让 KeyError→ValueError 的 fail-closed 路径
        在测试中显式断言——不允许"属性缺席静默透传"."""
        if self._telemetry_link_id_map is None:
            dims = (self.hardware.mesh_cols, self.hardware.mesh_rows)
            total = 1
            strides: list[int] = []
            for size in dims:
                strides.append(total)
                total *= size
            mapping: dict[int, tuple[int, int]] = {}
            next_id = 0
            for dim, dim_size in enumerate(dims):
                stride = strides[dim]
                for src in range(total):
                    if src // stride % dim_size + 1 >= dim_size:
                        continue
                    dst = src + stride
                    mapping[next_id] = (src, dst)
                    next_id += 1
                    mapping[next_id] = (dst, src)
                    next_id += 1
            self._telemetry_link_id_map = mapping
        try:
            return self._telemetry_link_id_map[link_id]
        except KeyError:
            raise ValueError(
                "link_telemetry sample link_id {} is outside the derived "
                "endpoint map (dims=({}, {}), links={}) -- topology dim "
                "order mismatch between C++ MultiDimTopology and the "
                "Python hardware mesh".format(
                    link_id, self.hardware.mesh_cols,
                    self.hardware.mesh_rows,
                    len(self._telemetry_link_id_map))) from None

    # --------------------------- C8（WP2-preadmit）同 tick 读流承诺预登记 --

    def _readplan_flow_units(self, steps: int) -> tuple[int, int]:
        """C8：readplan 对账覆盖窗口闭式——列车数 × 每列车切片块数。

        设计文档 §4.1"在飞解码后续读流按合法预测时域表示，不能把瞬时
        切片等同于整轮承诺"：est/truth/consumed/remaining 仍按时域内
        block 总足迹对账；这份账本量不再直接作为瞬时并发流数登记。
        注册表在 idle 间隔只登记一条 future 代表流，列车发射时由当前
        owner 接管，Tj 后若仍有 decode 工作再承诺下一条。steps 恒因果
        （准入 = CausalHorizonEstimator 现值；drain 对账 = 冻结计划真值
        步数）。列车步数上界取 T_max（_train_max_iter；0 = 不设限 → 单列
        车上界即 steps）；每列车块数 = ceil(T_eff / K)，K 经
        remote_credit_block_size 与决策计价同源同刻（§4.3.1 单一裁决点）。"""
        capped = steps if self._train_max_iter <= 0 else min(
            self._train_max_iter, steps)
        if capped <= 0:
            capped = 1
        trains = -(-steps // capped)
        block_size = remote_credit_block_size(
            self.joint_config.remote_credit_iters, capped)
        return trains, -(-capped // block_size)

    def _readplan_unit_transfers(self, shard_specs, *, units,
                                 session_id, request_id,
                                 home_instance, exec_instance,
                                 read_prefix, steps_per_unit):
        """预登记单位流：units 个同构 KVTransfer（每单位 = 每 shard 一条
        完整 XY 路径流；经 _register_transfer_flows 单一登记通道 = 链路
        流 + noc_migrate 双端点 HBM 端口，与逐列车切片登记同腿型，无第
        二套登记语义）。字节仅作披露摘要（登记只消费路径）。"""
        from face_scheduler import KVTransfer, KVTransferShard
        shards = tuple(
            KVTransferShard(
                source_rank=source_rank,
                target_rank=target_rank,
                edge_rank=None,
                bytes=per_step_bytes * steps_per_unit,
                noc_path=noc_path,
                layer_start=0,
                layer_end=read_prefix,
            )
            for (source_rank, target_rank, per_step_bytes, noc_path)
            in shard_specs)
        transfer = KVTransfer(
            kind="noc_migrate",
            phase="decode",
            reason="remote_read_readplan",
            session_id=session_id,
            trigger_request_id=request_id,
            source_instance_index=home_instance,
            target_instance_index=exec_instance,
            total_bytes=sum(shard.bytes for shard in shards),
            shards=shards,
            model_layers=self.model.layers,
            layer_start=0,
            layer_end=read_prefix,
            resident_prefix_layers_before=read_prefix,
            resident_prefix_layers_after=read_prefix,
        )
        return tuple(transfer for _ in range(units))

    def _preregister_readplan_flows(self, runtime, session_view,
                                    exec_instance: int,
                                    estimated_decode: int) -> None:
        """C8 步骤 1（§4.1 同 tick 承诺可见性）：准入事务成功且选中
        remote-read 后，建立读流承诺的 est 账本（owner 语义
        ``rid#readplan``，注册表半边延迟到 drain 对账以真值建立）。
        2026-09-25（规格书§二.5）起准入相**零注册表登记**：prefill 期间
        注册表上的远读流 = ``rid#prefill_read`` 前缀读流实流本身（同
        shard 路径集，同 tick 后续决策照样可见本笔已提交足迹）——同一条
        前缀读流不得同时被登记为 prefill 流与 decode 流（prefill/decode
        两阶段不得算成同时并发的两条远读流）；#readplan 注册表半边自
        drain（_reconcile_readplan_at_drain）起承接 decode credit。

        路径 = 冻结于 drain 的那套逐 shard XY 路由预估（与
        _joint_remote_read_credit_plan / route_paths_fn 同源
        deterministic_xy_route）；账本覆盖窗口 = 估计 decode 时域内列车
        数 × 每列车切片块数（时域来自 CausalHorizonEstimator，因果；
        est/truth/consumed/remaining 披露不确定性）。decode 期间注册表
        只登记一条每 shard 的代表流；当前列车发射后由实际 owner 接管，
        避免串行 blocks/train 被误算成同时在途流。

        与 drain 退化分支同口径：home 缺失 / home==exec / read_prefix
        <=0（REMOTE 基不适用 remote-read）不建账本——届时真计划亦为
        None，无承诺可登记。对账核销链见 remote_read_preplan 注释。"""
        home = session_view.home_instance
        if home is None or home == exec_instance:
            return
        session = self.kv_manager._sessions.get(runtime.session_id)
        if session is None:
            return
        if session.working_kind is not None:
            base_tokens = session.base_history_tokens
            read_prefix = session.base_resident_prefix_layers
        else:
            base_tokens = session.context_tokens
            read_prefix = self.model.layers
        if read_prefix <= 0:
            return
        from face_scheduler import deterministic_xy_route
        estimated_steps = max(1, int(estimated_decode))
        context_per_step = (
            base_tokens + runtime.joint_input_tokens + estimated_steps)
        per_step_shards = kv_cache_shard_bytes_for_layer_range(
            self.model, context_per_step, self.kv_manager.tp_degree,
            layer_start=0, layer_end=read_prefix)
        source_group = self.topology.instance(home)
        target_group = self.topology.instance(exec_instance)
        shard_specs = []
        for source_rank, target_rank, per_step_bytes in zip(
                source_group.ranks, target_group.ranks, per_step_shards):
            if per_step_bytes * estimated_steps <= 0:
                continue  # 零字节 rank 与真计划同口径跳过
            shard_specs.append((
                source_rank, target_rank, per_step_bytes,
                deterministic_xy_route(
                    self.hardware, source_rank, target_rank)))
        if not shard_specs:
            return
        trains, blocks = self._readplan_flow_units(estimated_steps)
        units = trains * blocks
        unit_transfers = self._readplan_unit_transfers(
            shard_specs, units=units,
            session_id=runtime.session_id,
            request_id=runtime.request_id,
            home_instance=home, exec_instance=exec_instance,
            read_prefix=read_prefix,
            steps_per_unit=max(1, -(-estimated_steps // units)))
        runtime.remote_read_preplan = {
            "home_instance": home,
            "exec_instance": exec_instance,
            "read_prefix_layers": read_prefix,
            "est_steps": estimated_steps,
            "est_total_bytes": (
                sum(spec[2] for spec in shard_specs) * estimated_steps),
            "est_flow_units": units,
            "truth_flow_units": None,    # drain 对账时落真值
            "consumed_units": 0,
            "remaining_units": units,
            "unit_transfers": unit_transfers,
            # 账本 units 表示整个因果预测时域；注册表在 decode 期间只
            # 表示一条物理可并发流（准入相不登记——prefill 期由
            # rid#prefill_read 实流占位）。当前列车接管时暂挂，Tj 后
            # 仍有 decode 工作再恢复。
            "registry_active": False,
        }

    def _reconcile_readplan_at_drain(self, runtime, tick: int) -> None:
        """C8 步骤 2（对账核销之一）：drain 边界真计划冻结处——核销
        准入相 #readplan est 账本、以真值建立注册表登记（2026-09-25
        规格书§二.5 起准入相零登记，本函数 = #readplan 注册表半边的
        建立点；release_owner 幂等空放兼容旧语义）；预估/实际差值进
        决策日志。

        真值登记后 #readplan 的账本仍承载全时域 block 单位；注册表只
        放一条每 shard 的未来代表流。列车发射时按本列车真实 block 数
        核销账本、暂挂 #readplan，再由 rid#decode#{j} owner 登记当前流；
        Tj 完成后如请求仍有 decode 工作则恢复一条未来代表流。两个 owner
        不会同时代表同请求同路径的串行流（§4.1 自注册流量去重纪律）。"""
        preplan = runtime.remote_read_preplan
        if preplan is None:
            return
        owner = runtime.request_id + "#readplan"
        self._release_transfer_flows(owner)
        preplan["registry_active"] = False
        plan = runtime.remote_read_credit_plan
        if plan is None:
            # drain 侧退化（准入与 drain 同判据，正常不可达；会话状态在
            # 准入后被改写时防御）：无承诺可重登记，余量清零、差值披露。
            preplan["truth_flow_units"] = 0
            preplan["remaining_units"] = 0
            preplan["unit_transfers"] = ()
            self.log_decision(
                {"kind": "readplan_reconcile", "request_id":
                 runtime.request_id, "priority": 0},
                tick,
                decision={
                    "est_steps": preplan["est_steps"],
                    "actual_steps": 0,
                    "est_total_bytes": preplan["est_total_bytes"],
                    "actual_total_bytes": 0,
                    "est_flow_units": preplan["est_flow_units"],
                    "truth_flow_units": 0,
                    "delta_bytes": -preplan["est_total_bytes"],
                    "est_uncertainty": "causal_horizon_estimate",
                    "note": "degenerate plan at drain: estimate written "
                            "off with no read stream",
                })
            return
        trains, blocks = self._readplan_flow_units(plan["steps"])
        truth_units = trains * blocks
        unit_transfers = self._readplan_unit_transfers(
            plan["shard_specs"], units=truth_units,
            session_id=runtime.session_id,
            request_id=runtime.request_id,
            home_instance=plan["home_instance"],
            exec_instance=plan["exec_instance"],
            read_prefix=plan["read_prefix_layers"],
            steps_per_unit=max(1, -(-plan["steps"] // truth_units)))
        self._register_transfer_flows(
            unit_transfers, owner=owner, serial_credit_stream=True)
        preplan["truth_flow_units"] = truth_units
        preplan["remaining_units"] = truth_units
        preplan["unit_transfers"] = unit_transfers
        preplan["registry_active"] = True
        self.log_decision(
            {"kind": "readplan_reconcile", "request_id":
             runtime.request_id, "priority": 0},
            tick,
            decision={
                "est_steps": preplan["est_steps"],
                "actual_steps": plan["steps"],
                "est_total_bytes": preplan["est_total_bytes"],
                "actual_total_bytes": plan["total_bytes"],
                "est_flow_units": preplan["est_flow_units"],
                "truth_flow_units": truth_units,
                "delta_bytes": (
                    plan["total_bytes"] - preplan["est_total_bytes"]),
                "est_uncertainty": "causal_horizon_estimate",
            })

    def _consume_readplan_units(self, runtime, consumed_units: int) -> None:
        """C8 步骤 2（对账核销之二）：逐列车实际登记处——本列车切片
        块数自 #readplan 承诺余量核销（实际登记 rid#decode#{j} 接管
        当前列车份额）。整份预承诺在当前列车在飞期间暂挂，避免把当前
        owner 与同请求未来串行 train 的代表流叠加；Tj 后若请求仍有 decode
        工作，由 _finalize_remote_credit_train_flows 恢复一条未来代表流。

        注册表时间口径不改本账本量纲：consumed/remaining 仍以实际 credit
        block 数对账，碎片化差额在完成边界披露。"""
        preplan = runtime.remote_read_preplan
        if preplan is None or consumed_units <= 0:
            return
        preplan["consumed_units"] += consumed_units
        remaining = preplan["remaining_units"] - consumed_units
        if remaining < 0:
            # 实际块数超过真值闭式（列车碎片化：Σ ceil(S_j/K_j) 可大于
            # 满列车闭式）——截 0，负差在完成边界披露（variance_units）。
            remaining = 0
        preplan["remaining_units"] = remaining
        owner = runtime.request_id + "#readplan"
        self._release_transfer_flows(owner)
        preplan["registry_active"] = False

    def _restore_readplan_flow_reservation(self, runtime) -> None:
        """在 Tj 释放当前 credit owner 后，为未完成请求恢复一条未来流。

        ``remaining_units`` 是闭式账本的剩余 block 估计，可能因列车碎片
        化先到 0；只要请求仍有 decode 工作，物理上就仍有未来 credit
        train。因此恢复条件取请求生命周期，并用原 readplan shard 模板
        仅登记一条同路径代表流。"""
        preplan = runtime.remote_read_preplan
        if (preplan is None or preplan["registry_active"]
                or not preplan["unit_transfers"]):
            return
        self._register_transfer_flows(
            preplan["unit_transfers"],
            owner=runtime.request_id + "#readplan",
            serial_credit_stream=True)
        preplan["registry_active"] = True

    def _finalize_remote_credit_train_flows(
            self, runtime, slice_index: int) -> None:
        """Tj 核销边界：当前 train 代表流完成后交还给未来承诺。

        一个 request 的同路径 current ``#decode#j`` 与 future
        ``#readplan`` 不能同时占注册表。若 decode 尚未结束，先释放实际
        owner，再恢复一条 #readplan 代表流供后续同 tick 决策读取。"""
        self._release_transfer_flows(
            "{}#decode#{}".format(runtime.request_id, slice_index))
        if runtime.decode_tokens_consumed < runtime.decode_length:
            self._restore_readplan_flow_reservation(runtime)

    def _settle_readplan_residual(self, runtime, tick: int) -> None:
        """C8 步骤 2（对账核销之三）：完成边界清 #readplan 残余承诺；
        真值闭式与逐列车实际登记块数之差（列车碎片化残差）落决策日志。
        泄漏兜底审计 = verify_run_end（_assert_no_readplan_leaks）。"""
        preplan = runtime.remote_read_preplan
        if preplan is None:
            return
        runtime.remote_read_preplan = None
        residual = preplan["remaining_units"]
        self._release_transfer_flows(runtime.request_id + "#readplan")
        truth_units = preplan["truth_flow_units"]
        if truth_units is None:
            truth_units = preplan["est_flow_units"]
        if residual or preplan["consumed_units"] != truth_units:
            self.log_decision(
                {"kind": "readplan_settle", "request_id":
                 runtime.request_id, "priority": 0},
                tick,
                decision={
                    "residual_units_released": residual,
                    "truth_flow_units": truth_units,
                    "consumed_units": preplan["consumed_units"],
                    "variance_units": (
                        truth_units - preplan["consumed_units"]),
                    "note": "train fragmentation variance between the "
                            "closed-form truth footprint and per-train "
                            "actual registrations",
                })

    def _contention_coverage_value(self) -> str:
        """C8（E13 唯一读者）：joint_admission 行 contention_coverage 的
        值构造——注册表层（链路流 + 池端口）之上，遥测完备（collective_
        coverage 翻转 = run 全程键在场 ∧ 窗口链无重叠回退 ∧ 有样本；
        F6：遥测盖 NoC 含 collective、池端口不覆盖）时追加
        +link_telemetry 段；无遥测 run 值域与 C8 前逐字节同
        （cold_start / link_flows+pool_ports 两态保持）。"""
        value = ("link_flows+pool_ports" if (
            self._joint_flows.has_registrations
            or self._pool_ports._counts) else "cold_start")
        if self._joint_flows.collective_coverage:
            value += "+link_telemetry"
        return value

    def _assert_no_readplan_leaks(self) -> None:
        """C8 收尾审计：#readplan est 账本必须全部经 drain 对账 → 逐列车
        核销 → 完成清残余归零；#prefill_read 前缀读流必须已在 prefill
        drain 释放（规格书§二.5）。收尾仍非空 = 对应生命周期破损泄漏，
        fail-closed。复用 C2 的 HbmPortFlowRegistry.leaked_owners 审计
        通道（两类 owner 均经 noc_migrate 端点登记，HBM 端口表与链路
        流表同通道成对注册/注销）。"""
        leaks = {
            owner: ports
            for owner, ports in self._hbm_ports.leaked_owners().items()
            if owner.endswith("#readplan")
            or owner.endswith("#prefill_read")}
        if leaks:
            raise RuntimeError(
                "remote-read stream owner leaked past settlement "
                "(#readplan drain reconciliation/consumption/settlement "
                "chain or #prefill_read prefill-drain release broken): "
                "{}".format(sorted(leaks)[:5]))

    def _assert_quota_tracker_ledgers_clean(self) -> None:
        """O10①：run 尾 tracker snapshot() 空账直审——occ/res/enroll/
        bulk/merges/merge_borrowed 全零（quota=off 由调用方条件化跳过：
        tracker 恒 None）。merges（_merges 预留表）经 res/bulk 间接归零
        ——reserve_merge 恒持双向链路预留 + 双端口 bulk 名额，预留存活
        ⇒ res/bulk 非零必被捉；merge_borrowed（N1 借槽账）随 settle/
        撤销消解，残留即借还失配。"""
        snapshot = self._quota_tracker.snapshot()
        leftovers = []
        for link_repr, entry in sorted(snapshot["links"].items()):
            if entry["occupancy"]:
                leftovers.append("link {} occupancy={}".format(
                    link_repr, entry["occupancy"]))
            if entry["reserved"]:
                leftovers.append("link {} reserved={}".format(
                    link_repr, entry["reserved"]))
        for port_repr, entry in sorted(snapshot["ports"].items()):
            if entry["enrolled_total"]:
                leftovers.append("port {} enrolled={}".format(
                    port_repr, entry["enrolled_total"]))
            if entry["bulk_used"]:
                leftovers.append("port {} bulk_used={}".format(
                    port_repr, entry["bulk_used"]))
        if snapshot["merge_borrowed"]:
            leftovers.append("merge_borrowed={}".format(
                snapshot["merge_borrowed"]))
        if leftovers:
            raise RuntimeError(
                "quota tracker ledgers non-empty at run end (borrow/"
                "release chain broken): {}".format(leftovers[:5]))

    def _assert_no_flow_registry_leaks(self) -> None:
        """O10③：R15 在途流注册表全 owner 清账断言——链路
        （LinkFlowRegistry）/池端口（_PoolPortRegistry）/HBM 端口
        （HbmPortFlowRegistry）的 leaked_owners 全空；非空 = 完成事件
        链破损（发射登记未配对注销），fail-closed。"""
        leaks = {}
        for name, registry in (
                ("link", self._joint_flows),
                ("pool_port", self._pool_ports),
                ("hbm_port", self._hbm_ports)):
            for owner, live in registry.leaked_owners().items():
                leaks["{}:{}".format(name, owner)] = live
        if leaks:
            raise RuntimeError(
                "transfer flow registries leaked owners at run end "
                "(completion chain broken): {}".format(sorted(leaks)[:5]))

    def _ingest_link_telemetry(self, delta) -> None:
        """C8 步骤 3（SH 半）：解析桥请求顶层 ``link_telemetry[]``
        （C6 交付，--link-telemetry 开启时每交付 epoch 一条逐链路窗口
        差分数组），按冻结接口把实测有效速率（served_bytes/active_ns，
        B/ns——与 noc_link_bytes_per_ns 同量纲）刷新到
        ``_link_telemetry_rates``，供 ``_joint_cost_model`` 构造喂入
        （``{link_id: 实测有效速率}``，JCM 侧消费 = 并行卡 C7）。

        F1 补遗三件（2026-09-22，§4.3 A8'/A9'/A10'(a)）：
        - A8'：逐窗口样本同时喂 ``_joint_factors.observe_transfer_
          from_link_window``（transfer_factor EWMA 接线；actual/base =
          active_ns/(served/名义速率) ⇒ 样本比 = 名义/实测有效速率，
          ≥1 拥胀方向）；
        - A9'：rates 条目随每个 tick 包做在场核对，本包缺席即删除
          （空闲链路按 C6 契约省略 = 空闲信号，退回注册表 divisor）；
        - A10'(a)/A12'：served_bytes==0 ∧ active_ns>0 的有效速率零窗口
          样本（有活动无载荷字节——可出自 C++ 整字节进位 carry <1 字节的
          合法短窗；原"uint64 差分计量无整数截断伪影"表述与进位事实冲突，
          O7② 订正）丢弃 + 披露计数（``_telemetry_zero_rate_dropped``，
          入决策日志遥测块 _telemetry_coverage_decision），并同窗失效
          缓存速率，不写入 rates（流数条目按 N3 保留）；
        - O7②：served_bytes==0 ∧ active_ns==0 的双零样本**带 active_flows
          字段**（C++ 显式全闲置窗报告）缓存速率与流数**同步** pop——
          在场键豁免 A9' 剪除，修前速率驻留、流数被本窗字段重落地
          （幽灵流），只清一边污染除数口径；字段缺席的双零样本（旧二
          进制）无流数通道，维持 A9' 在场作保语义。

        遥测完备性（collective_coverage 翻转条件，C7 卡冻结语义 =
        "该 run 全程遥测开启且窗口无空洞"的可观测判定）：
        - 键缺席（旗标关）→ 本 run 遥测不完备（sticky）；
        - 窗口链重叠回退（window_start_ns < 上一窗 window_end_ns =
          双采样/重放类破损）→ sticky 破损。相邻采样窗之间的间隙归因
          于不可见的全闲置 epoch（C++ 侧 prev_tick 差分结构性无空洞，
          闲置链路按 C6 契约省略），不判破损；
        - 至少一个样本在场（observer 关闭时空数组恒空——不伪造覆盖）。
        任一条件不满足即 False（披露真值，可回翻）。

        **窗口时间加权均值 ≠ 决策瞬时流数**（O7④/A19'(g)④，
        2026-09-23 落字，措辞同 joint_cost_model.py TelemetryLinkFlowView
        docstring / joint/test_joint_telemetry_divisor.py 文件头）：本入口
        落账的 active_flows（时间加权活跃流数）与 served/active_ns 速率
        均为 C++ 周期采样窗口的窗口量，非决策时刻瞬时值；流进出频繁时
        均值系统性低于瞬时峰值——方向 = 低估旧流并发（乐观向），窗口
        盲区 ≤1 epoch。改瞬时口径 = 遥测合同侵入（A18'(b) 已否决），
        此处仅落字披露。"""
        if "link_telemetry" not in delta:
            # A12'：遥测面整体缺席（旗标关/桥断供）时缓存实测速率一并
            # 失效——与 A9' 单链路缺席失效同语义，防陈旧速率在遥测
            # 停发后永久驻留（旗标为 per-run 常量时空字典上为空操作）。
            self._link_telemetry_rates.clear()
            self._link_telemetry_flow_counts.clear()
            self._telemetry_absent_seen = True
            self._joint_flows.collective_coverage = False
            # O6①（2026-09-23 终轮审计）：断供期也推进遥测序号——rates
            # 已清 ⇒ _quota_ingest_telemetry 内 per_flow 恒空 ⇒
            # observe_telemetry 空字典采样（零链路动作、仅序号/遥测钟
            # 簿记），断供期 dt 不计入缺测后首个样本（K7 连续采样纪律
            # 的断供侧补全；修前缺席 epoch 不消耗序号，缺测后首样本
            # dt 含整段断供时长，AIMD 误计 quiet/进度）。
            self._quota_ingest_telemetry(int(delta["tick"]))
            return
        samples = delta["link_telemetry"]
        if not isinstance(samples, list):
            raise ValueError("delta link_telemetry must be a list")
        self._link_telemetry_epoch_count += 1
        tick_ns = int(delta["tick"])
        self._telemetry_last_tick_ns = tick_ns
        window_start = None
        window_end = None
        # A9'（2026-09-22，§4.3 补遗）：本包在场键集——tick 包缺席的既有
        # rates 条目即失效删除（C++ 契约"空闲链路省略"即空闲信号，退回
        # 注册表 divisor；差分计量下有载荷即有增量——慢而非零的链路
        # 每窗在场不会被剪，双零链路重载后缺席一拍重现即重建键）。
        present_keys = set()
        # A8'（2026-09-22，§4.3 补遗）：transfer_factor EWMA 接线——逐窗
        # 口样本喂 ServiceFactors.observe_transfer_from_link_window（样本
        # 比 = 名义速率/实测有效速率，≥1 拥胀方向；同 tick 多链路经
        # _record 时刻汇总纪律合并）。F6 销账：__init__ 恒设 _joint_factors
        # （ServiceFactors()）——直接访问，替身漏设 = AttributeError。
        # O7①：active_flows 字段同 epoch 在场/缺席计数（混合 = producer
        # 异常，循环内 fail-closed；见循环内注释）。
        flows_field_present = 0
        flows_field_absent = 0
        for sample in samples:
            try:
                link_id = int(sample["link_id"])
                served_bytes = int(sample["served_bytes"])
                active_ns = int(sample["active_ns"])
                start = int(sample["window_start_ns"])
                end = int(sample["window_end_ns"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "malformed link_telemetry sample: {}".format(exc))
            if served_bytes < 0 or active_ns < 0 or start > end:
                raise ValueError(
                    "malformed link_telemetry sample bounds: "
                    "{!r}".format(sample))
            if served_bytes > 0 and active_ns == 0:
                # A14'（H5，2026-09-22 第三轮复审）：uint64 差分契约
                # （有载荷字节必有活跃时间）下的不可能形态——静默跳过
                # 会让该样本以在场键豁免 A9' 剪除、陈旧速率钉驻，是本
                # 入口唯一病态样本静默通道；与未知 link_id 的 fail-
                # closed 教义对齐封死。双零样本（served=0∧active=0）
                # 维持良性在场语义不动。
                raise ValueError(
                    "malformed link_telemetry sample: served_bytes>0 "
                    "with active_ns==0 violates the uint64 differential "
                    "contract: {!r}".format(sample))
            # A5'/B3 rider（C7 移交，2026-09-22 裁定落 C11）：LinkId
            # 整型键换算为端点形 (src, dst)——JCM divisor_effective 的
            # max 合并自此在生产路径真实生效（换算前整型键与
            # LinkFlowRegistry 键空间不相交、合并结构性不发生）。
            # 未知 id = 拓扑维序与 C++ 枚举不一致 = 注入逻辑损坏，
            # fail-closed（不静默退回整型键）。A9' 后换算对本包全部样本
            # 执行（在场核对需要；零活跃样本同样占位在场集）。
            endpoint_key = self._telemetry_endpoint_link_key(link_id)
            present_keys.add(endpoint_key)
            # M1：active_flows 可选字段（旧二进制缺席 = 不落账，JCM/
            # AIMD 侧退既有口径）。活跃时间非零时至少有一条流；双零
            # 闲置窗的流数必须为零。病态值 fail-closed，不钳到 1。
            raw_flows = sample.get("active_flows")
            # O7①（2026-09-23 终轮审计）：同 epoch 混合缺席 fail-closed——
            # active_flows 的写死可选性只随二进制版本整体缺席/整体在场；
            # 同一 epoch 内部分链路有、部分无 = 不可能形态 ⇒ producer
            # 异常（部分样本被吞/被注入），静默容忍会把混合版本流量混进
            # 除数口径。只有全缺席才按旧二进制容忍（退在册流数）。
            if raw_flows is None:
                flows_field_absent += 1
            else:
                flows_field_present += 1
            if flows_field_absent and flows_field_present:
                raise ValueError(
                    "mixed active_flows presence within one telemetry "
                    "epoch: {} samples with the field, {} without "
                    "(impossible from a single binary version -- "
                    "producer anomaly; fail-closed)".format(
                        flows_field_present, flows_field_absent))
            if raw_flows is not None:
                try:
                    flows_value = float(raw_flows)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "malformed link_telemetry sample "
                        "active_flows: {}".format(exc))
                if (not math.isfinite(flows_value)
                        or flows_value < 0
                        or (active_ns > 0 and flows_value < 1)
                        or (active_ns == 0 and flows_value != 0)):
                    raise ValueError(
                        "malformed link_telemetry sample active_flows: "
                        "{!r}".format(sample))
                self._link_telemetry_flow_counts[endpoint_key] = (
                    flows_value)
            if active_ns > 0 and served_bytes > 0:
                # A12'：因子喂入先于 rates 落账——喂入侧 AttributeError
                # （替身漏设 _joint_factors 的 fail-loud）时速率条目不
                # 落账，不留半笔状态。
                self._joint_factors.observe_transfer_from_link_window(
                    served_bytes=served_bytes, active_ns=active_ns,
                    nominal_rate_bytes_per_ns=(
                        self._joint_rates.noc_link_bytes_per_ns),
                    tick_ns=tick_ns)
                self._link_telemetry_rates[endpoint_key] = (
                    served_bytes / active_ns)
            elif active_ns > 0:
                # A10'(a)/A12'：served==0 ∧ active>0 = 有活动无载荷字节
                # 的有效速率零窗口。无法构成有效速率样本（0 字节），丢
                # 弃 + 披露计数，并同窗失效该链路缓存实测速率：A9' 在场
                # 核对不剪除在场键，不失效则陈旧高速率驻留、NoC 计价除
                # 数欠计拥塞（下一 served>0 窗口即重建）。样本可出自
                # C++ 整字节进位（carry <1 字节的合法短窗，N3 口径——
                # 原"uint64 差分计量无整数截断伪影"的表述与进位事实相冲
                # 突，O7② 据此订正）。N3：流数条目**保留**——流数是该
                # 窗口 collective 等未登记流在场的唯一证据；JCM 支持仅
                # 流数、舍弃速率的合并分支（__post_init__ 假速率退化）。
                # A9' 剪除循环同步遍历两字典并集（纯流数条目不漏剪陈旧
                # 驻留）。
                self._link_telemetry_rates.pop(endpoint_key, None)
                self._telemetry_zero_rate_dropped += 1
            elif served_bytes == 0 and raw_flows is not None:
                # O7②（2026-09-23 终轮审计）：双零样本带 active_flows 字段
                # （served=0 ∧ active=0）= C++ 显式全闲置窗报告——其缓存
                # 速率与流数须**同步** pop：修前速率不 pop（在场键豁免
                # A9' 剪除 ⇒ 陈旧速率钉驻）、流数被本窗字段重落地
                # （max(1.0, 0) = 1 的幽灵流）⇒ 只清一边、除数口径污染。
                # 字段缺席的双零样本（旧二进制）无流数通道、"同步"无对
                # 象，维持 A9' "包内样本为链路在场作保"语义（A9Telemetry
                # RateExpiryTest 钉死）。前置不可能形态（served>0 ∧
                # active=0）已在上面 fail-closed，本分支 = 双零的字段在场
                # 形态。
                self._link_telemetry_rates.pop(endpoint_key, None)
                self._link_telemetry_flow_counts.pop(endpoint_key, None)
            self._link_telemetry_sample_count += 1
            window_start = (
                start if window_start is None else min(window_start, start))
            window_end = (
                end if window_end is None else max(window_end, end))
        if window_start is not None:
            if window_start < self._telemetry_window_end_ns:
                self._telemetry_window_broken = True
            self._telemetry_window_end_ns = window_end
        # A9'：在场核对——本包缺席的既有条目失效删除（键缺席 = C++ 侧
        # 按契约省略 = 空闲信号；divisor_effective 退回注册表除数，消除
        # "忙转闲后陈旧实测速率永久驻留、长 run 单调抬升 NoC 计价除数"）。
        # N3：遍历 rates ∪ flow_counts 键并集——仅流数条目（零速率窗
        # 保留分支的产物）同样受在场核对约束，不漏剪陈旧驻留。
        for stale_key in tuple(set(self._link_telemetry_rates)
                               | set(self._link_telemetry_flow_counts)):
            if stale_key not in present_keys:
                self._link_telemetry_rates.pop(stale_key, None)
                self._link_telemetry_flow_counts.pop(stale_key, None)
        self._joint_flows.collective_coverage = (
            not self._telemetry_absent_seen
            and not self._telemetry_window_broken
            and self._link_telemetry_sample_count > 0)
        # C11 步骤 8（AIMD 闭环）：本 epoch 有效速率喂 link_quota 的
        # observe_telemetry（仅 aimd 模式；每流换算与"无在册流不出现在
        # 字典"契约见 _quota_ingest_telemetry）。收缩/扩张经
        # set_link_quota 落地（tracker 内部），调整 bump 配额代数 ⇒
        # deferred 重试门重开。
        self._quota_ingest_telemetry(tick_ns)

    def _hbm_active_decode_streams(self, port_rank: int) -> int:
        """C2/F4：该端口所属实例的活跃 decode KV 消费流数（因果负载
        视图派生——active_decode 现值；闲置为 0，不假设恒 1）。TP 全
        rank 同账（每活跃会话逐 rank 各占一条消费流）。"""
        instance_index = self._rank_to_instance.get(port_rank)
        if instance_index is None:
            return 0
        return len(self.instances[instance_index].active_decode)

    def _pool_port_divisor(self, instance_index: int) -> int:
        """实例各池端口最大在途数 + 自身 1（R15-3；E 内核 r_j 与 J 池
        路径计价共用）。"""
        edges = self._instance_edge_ports.get(instance_index, ())
        if not edges:
            return 1
        return max(1, max(self._pool_ports.count(edge) for edge in edges) + 1)

    def _observe_service_factors(self, state, train, tick: int) -> None:
        """R15-2：列车核销时采集 ServiceFactors 纯样本（P3 α 公式）。

        样本纯净性：只收**纯列车**（无 joiner 迁移、无 chunk×decode 混合
        ——两段不可因果分离）的服务段实测 = 核销 tick − max(发射 tick,
        上次核销 tick)（排除列车前排队的污染，排队在 target_wait 段计
        价）；基数 = 同内容 roofline 闭式账本。transfer 因子样本源 =
        链路遥测窗口（A8' 接线：_ingest_link_telemetry 逐窗口喂
        observe_transfer_from_link_window——列车核销通道无纯传输段可
        因果分离，故本函数不收 transfer 样本；A12' 名实订正：
        transfer_factor EWMA 为披露/观测通道，传输争用的计价修正实效
        由链路/端口除数通道唯一承担——divisor_effective 的
        max(registered, B/measured) 合并；因子不进 estimate_action，
        避免与除数通道双计同一遥测信息）。"""
        emit_tick = train.get("emit_tick")
        if emit_tick is None:
            return
        start = max(emit_tick, state.last_train_finalize_tick or emit_tick)
        span_ns = tick - start
        if span_ns <= 0:
            return
        chunk_records = train.get("prefill_chunk_tokens") or ()
        member_parts = train.get("members") or []
        instance_size = self.topology.instance(state.index).size
        # 纯 decode 列车（无 chunk、无 joiner——迁移会混入传输段；M5：
        # remote-read 成员的逐列车门控读流同样混入传输段，一并排除；
        # O3：merge 尾门/完成批尾段门控等待同样混进 emit→核销 span——
        # 与 prefill 分支同款纯度排除（_emit_train 已按本实例粒度
        # 打好 merge_tail_gated 标记））。
        if (not chunk_records and member_parts
                and not train.get("had_joiners")
                and not train.get("merge_tail_gated")):
            base_ns = 0
            for request_id, participation in member_parts:
                runtime = self.runtime_by_request_id.get(request_id)
                if runtime is None:
                    return
                if runtime.joint_action == "remote-read":
                    return
                context0 = runtime.prefill_context_tokens
                consumed = runtime.decode_tokens_consumed
                for step in range(1, participation + 1):
                    base_ns += self._decode_step_load_ns(
                        instance_size, context0 + consumed + step)
            if base_ns > 0:
                self._joint_factors.observe_decode(
                    actual_ns=span_ns, base_ns=base_ns,
                    service_ns=span_ns, tick_ns=tick)
            return
        # 纯 prefill 列车（无 decode 成员、无 joiner；含 partial 后缀
        # 恢复门或 copy/REMOTE 首 chunk 门控 NoC 传输的列车混入传输段，
        # 跳过——纯度约束；N5+O3 增 merge 尾门排除：本实例有未交付
        # merge watch（任意 session）或 frontier 完成批尾段在案时，列车
        # emit→计算启动被 R11(ii) 门/尾段交付 hold，span 混入门控等待）。
        if (chunk_records and not member_parts
                and not train.get("had_joiners")
                and not train.get("suffix_gated")
                and not train.get("history_transfer_gated")
                and not train.get("merge_tail_gated")):
            head_runtime = self.runtime_by_request_id.get(
                chunk_records[0][0])
            if head_runtime is None:
                return
            base_ns = 0
            processed = head_runtime.prefill_tokens_completed
            for _, chunk_tokens in chunk_records:
                base_ns += self._prefill_chunk_task_load_ns(
                    instance_size=instance_size,
                    chunk_tokens=chunk_tokens,
                    context_tokens=(
                        head_runtime.joint_span_base_context
                        + processed + chunk_tokens))
                processed += chunk_tokens
            if base_ns > 0:
                self._joint_factors.observe_prefill(
                    actual_ns=span_ns, base_ns=base_ns,
                    service_ns=span_ns, tick_ns=tick)
                # M4（2026-09-23 验收审计）：同一样本桥接 C15 递推预测
                # 器的 γ_prefill（face KVCacheManager.service_factors
                # ——与 JCM 的 _joint_factors 是两套：JCM 侧为披露通
                # 道、face 侧为递推修正入参，此前零生产喂入点、恒冷启
                # 动 1.0）。纯度约束同源：本分支已排 joiner/恢复门/
                # 传输门 ⇒ span 为可分离纯计算段；observed_ratio =
                # 实际/roofline 闭式（§5.4 契约方向）。η_pool 不桥接
                # （无可分离纯池传输样本源——含争用样本会使 η 退化为
                # 争用指标、与递推份额仲裁双计，A12' 镜像论证，落字
                # A17'(d)）。
                self.kv_manager.service_factors.observe_valid_service(
                    "prefill",
                    observed_ratio=span_ns / base_ns,
                    completion_ns=tick,
                    service_duration_ns=span_ns)

    def _decode_step_load_ns(self, instance_size: int, context: int) -> int:
        """单 decode 步的 roofline 负载（average=1/generated=0 的单步口径）。"""
        return self._decode_task_load_ns_cached(
            instance_size=instance_size,
            current_context_tokens=context,
            generated_tokens=0,
            average_decode_length=1.0,
            running_step_fraction_remaining=1.0)

    def _joint_session_view(self, session_id: str) -> SessionKVView:
        """session KV 的决策时点只读视图（跨实例执行期间描述**基础历史**：
        home 侧驻留/池 backing 状态；工作副本由动作计费另行覆盖）。"""
        if not self.kv_manager.has_session(session_id):
            zero = (0,) * self.topology.instances[0].size
            return SessionKVView(
                session_id=session_id, home_instance=None,
                resident_instance=None, location="none",
                history_tokens=0, resident_prefix_layers=0,
                history_bytes_by_tp_rank=zero,
                missing_bytes_by_tp_rank=zero)
        snapshot = self.kv_manager.session_snapshot(session_id)
        session = self.kv_manager._sessions[session_id]
        home = session.home_instance
        if session.working_kind is not None:
            # O14（2026-09-23 终轮审计，镜像 face_scheduler.py :4976-4984
            # merge_back 同款改法）：空串 base_location = 元数据腐坏
            # fail-closed（None 是池基唯一合法未设形态）；None 显式落
            # REMOTE_MEMORY——`or` 借道把空串误判为池基，腐坏被吞。
            base_location = session.base_location
            if base_location == "":
                raise RuntimeError(
                    "session {} has empty-string base_location in the "
                    "joint session view -- session metadata corrupt "
                    "(None is the only legal unset form for a pool-only "
                    "base)".format(session_id))
            location = (
                base_location if base_location is not None
                else self.kv_manager.REMOTE_MEMORY)
            base_tokens = session.base_history_tokens
            base_prefix = session.base_resident_prefix_layers
            resident = home
        else:
            location = snapshot.location
            base_tokens = snapshot.context_tokens
            base_prefix = snapshot.resident_prefix_layers
            resident = snapshot.instance_index
        from face_scheduler import kv_cache_shard_bytes_for_layer_range
        local_shards = (
            kv_cache_shard_bytes_for_layer_range(
                self.model, base_tokens, self.kv_manager.tp_degree,
                layer_start=0, layer_end=base_prefix)
            if location in ("local_hbm", "partial_hbm_remote") and base_prefix
            else (0,) * self.kv_manager.tp_degree)
        full_shards = (
            kv_cache_shard_bytes_for_tokens(
                self.model, base_tokens, self.kv_manager.tp_degree)
            if base_tokens else (0,) * self.kv_manager.tp_degree)
        missing = tuple(
            total - local
            for total, local in zip(full_shards, local_shards))
        return SessionKVView(
            session_id=session_id, home_instance=home,
            resident_instance=resident, location=location,
            history_tokens=base_tokens, resident_prefix_layers=base_prefix,
            history_bytes_by_tp_rank=tuple(local_shards),
            missing_bytes_by_tp_rank=missing)

    def _joint_cost_model(self, now_ns: int, *,
                          decode_context_tokens: int,
                          decode_average_length: int) -> JointCostModel:
        """决策时点构造代价模型（只读快照；无资源副作用）。

        R3'/R15（2026-09-14）：负载视图补 reclaimable（驱逐等待定价区分
        逐出可解/深缺口）；route_paths_fn 给全部 TP 并行 shard 路径
        （F-B 并集除数）；pool_divisor_fn 给池端口仲裁份额（P1 同源）。

        C3（WP1c，2026-09-22）：decode 计价上下文依赖——decode_ns_per_
        token 的求值口径从 context=1/长度 1 旧口径换为真实上下文
        （decode_context_tokens = history + input，准入时已知、对全部
        候选同值，非 oracle——设计文档 §3.2"decode 计算：使用已知上下文
        和合法预测时域"）与 CausalHorizonEstimator 现值（调用方传入；
        冷启动 1 如实保留，不隐藏）。attention 二次项由
        estimate_decode_remaining_task_load_ns 的 d_token 逐步
        （face_scheduler.py）自然进入；剩余总量按 horizon 折回
        per-token 均值口径——JCM 消费端 estimated_decode_tokens ×
        decode_ns_per_token 形态不动（同值 horizon 乘回 ≡ 逐步总量）。
        上下文与 horizon 均为请求级标量 → 构造处一次求值（memo 化），
        无逐候选钩子，O(1) 保持。"""
        loads = {}
        for state in self.instances:
            snapshot = self._task_load_snapshot(state, now_ns)
            remaining = self.kv_manager._effective_remaining_by_tp_rank(
                state.index)
            reclaimable = self.kv_manager._instance_reclaimable_capacity_by_tp_rank(
                state.index)
            loads[state.index] = InstanceLoadView(
                instance_index=state.index,
                queued_task_load_ns=snapshot.queued_prefill_task_load_ns,
                running_task_load_ns=snapshot.running_prefill_task_load_ns,
                active_decode_task_load_ns=(
                    snapshot.active_decode_task_load_ns),
                hbm_remaining_bytes_by_tp_rank=tuple(remaining),
                reclaimable_bytes_by_tp_rank=tuple(reclaimable),
            )

        def route(source_instance: int, target_instance: int):
            # 实例间路由 = 代表 rank 对的确定性 XY 路线（真实拓扑派生）；
            # 返回 (path_ranks, hops)。
            source_group = self.topology.instance(source_instance)
            target_group = self.topology.instance(target_instance)
            source_rank = source_group.ranks[0]
            target_rank = target_group.ranks[0]
            from face_scheduler import deterministic_xy_route
            path = deterministic_xy_route(
                self.hardware, source_rank, target_rank)
            return path, max(0, len(path) - 1)

        def route_paths(source_instance: int, target_instance: int):
            # F-B：候选自身 TP 并行流的全部 shard 路径（相对位配对）。
            from face_scheduler import deterministic_xy_route
            source_group = self.topology.instance(source_instance)
            target_group = self.topology.instance(target_instance)
            paths = []
            for source_rank, target_rank in zip(
                    source_group.ranks, target_group.ranks):
                paths.append(deterministic_xy_route(
                    self.hardware, source_rank, target_rank))
            return paths

        prefill_ns_per_token = float(
            estimate_prefill_task_load_ns(
                self.hardware, self.model,
                instance_size=self.topology.instances[0].size,
                chunk_tokens=1, context_tokens=1))
        # C3：decode per-token 率按真实上下文 × horizon 逐步求值再折回
        # 均值（d_token = context + offset 的二次项随之进入）；horizon 下
        # 掉到 0 的合法在线态（session 均值可为 0）按 1 求率——该率随即
        # 被 JCM 端 estimated_decode_tokens=0 乘回 0，口径惰性不生效。
        decode_horizon = float(max(1, decode_average_length))
        decode_ns_per_token = (
            self._decode_task_load_ns_cached(
                instance_size=self.topology.instances[0].size,
                current_context_tokens=max(1, decode_context_tokens),
                generated_tokens=0,
                average_decode_length=decode_horizon,
                running_step_fraction_remaining=1.0)
            / decode_horizon)
        # C8（WP2，冻结交接接口）：桥遥测实测有效速率 {link_id: B/ns}
        # （C6 link_telemetry[] → _ingest_link_telemetry）按冻结契约喂
        # JCM 构造——JCM 侧消费（divisor_effective/transfer_factor）=
        # 并行卡 C7。F6 判定：此处 getattr 保留（非替身软门）——探测的
        # 是 JointCostModel 类的 dataclass 字段（joint_cost_model.py，
        # 他卡交付面），跨模块 schema 可选语义；self 侧 rates 已改直接
        # 访问（C7 已交付该字段，分支恒走通）。
        telemetry_kwargs = {}
        if "link_telemetry_rates" in getattr(
                JointCostModel, "__dataclass_fields__", {}):
            telemetry_kwargs["link_telemetry_rates"] = dict(
                self._link_telemetry_rates)
        # M1：时间加权活跃流数（含 collective）——在场时 JCM 遥测除数
        # 取流数口径（物理除数，无"下游瓶颈"歧义；合计吞吐满载退 1/
        # 瓶颈误判争用的旧口径只在字段缺席时兜底——旧二进制兼容）。
        if self._link_telemetry_flow_counts and (
                "link_telemetry_flow_counts" in getattr(
                    JointCostModel, "__dataclass_fields__", {})):
            telemetry_kwargs["link_telemetry_flow_counts"] = dict(
                self._link_telemetry_flow_counts)
        # N4（2026-09-23 复核审计4）：prefill 整段负载注入生产同形实现
        # （p_chunk 切分 + 累计 context 逐 chunk roofline 求和——与
        # _plan_train/:963 核销逐键同值同 memo；线性外推丢二次形状，
        # 对照 decode 的 C3 真实上下文折算，prefill 此前无同款）。
        if "prefill_task_load_ns_fn" in getattr(
                JointCostModel, "__dataclass_fields__", {}):
            telemetry_kwargs["prefill_task_load_ns_fn"] = (
                self._joint_prefill_total_load_ns)
        return JointCostModel(
            rates=self._joint_rates,
            loads=loads,
            flow_registry=self._joint_flows,
            service_factors=self._joint_factors,
            prefill_ns_per_token=prefill_ns_per_token,
            decode_ns_per_token=decode_ns_per_token,
            model_layers=self.model.layers,
            instance_tp_size=self.topology.instances[0].size,
            route_fn=route,
            route_paths_fn=route_paths,
            pool_divisor_fn=self._pool_port_divisor,
            # C2（WP1b）：实例 HBM 端口流注册表（u_port 除数供给，F4）。
            # F6 销账：__init__ 恒设 _hbm_ports——直接传入（替身漏设 =
            # AttributeError，不再静默退回端点腿独占带宽口径）。
            hbm_port_registry=self._hbm_ports,
            # §4.3.1：计价与执行同公式同 K 源（credit 交错流唯一机制，
            # 2026-09-17 用户裁定——无执行口径开关，无窗口矛盾）。
            remote_credit_iters=self.joint_config.remote_credit_iters,
            # 读流层区间化（2026-09-17）：PARTIAL 基 remote-read 适用面
            # 开关（joint_config.remote_read_partial "on"/"off"）同源传入
            # 计价模型——计价与执行共用同一开关位，杜绝"计价按混合形态、
            # 执行按 LOCAL-only"的窗口矛盾。
            remote_read_partial=(
                self.joint_config.remote_read_partial_enabled),
            **telemetry_kwargs,
        )

    def _joint_remote_read_credit_plan(self, runtime, exec_instance: int):
        """remote-read credit（唯一执行口径）的持久读
        计划：drain 边界一次冻结每步读量（均匀终态上下文——与 v1 单笔
        字节口径逐位同源，字节语义零变更）与逐 shard 路由；总量 =
        S × f(终态) 为 I1 守恒基数。逐列车切片（S_j 步 × 块大小 K）在
        列车规划期从本计划派生（_joint_remote_read_slice）。

        读流层区间化（2026-09-17）：读流只覆盖前缀层 [0, p)——工作副本
        形态（working_kind 非 None）取 base_resident_prefix_layers，
        LOCAL 基 p==L 为不变锚（逐字节同旧全层口径）；PARTIAL 基混合
        形态 p<L：后缀 [p, L) 不走读流，在准入相经 kv_manager 的
        _remote_load_transfer 池恢复物化（热 KV，2026-09-17 用户裁定）。
        I1 守恒基数随之 = S × f(终态) × [0, p) 层区间（per_step_shards
        按层区间派生）。非工作会话（REMOTE 基）read_prefix<=0 时防御
        返回 None（remote-read 适用性本就排除 REMOTE 基，正常不可达）。"""
        from face_scheduler import deterministic_xy_route
        home = runtime.origin_home_instance
        if home is None or home == exec_instance:
            return None  # 无异地历史（退化：无读流，与 v1 同口径）
        session = self.kv_manager._sessions[runtime.session_id]
        if session.working_kind is not None:
            # LOCAL 基 p==L（不变锚）；PARTIAL 混合基 p<L（后缀 [p,L)
            # 准入相池恢复物化，不经读流）。
            base_tokens = session.base_history_tokens
            read_prefix = session.base_resident_prefix_layers
        else:
            base_tokens = session.context_tokens
            read_prefix = self.model.layers
        if read_prefix <= 0:
            # 防御：无前缀层可读（REMOTE 基不适用 remote-read，正常不可达）。
            return None
        context_per_step = (
            base_tokens + runtime.joint_input_tokens + runtime.decode_length)
        steps = max(1, runtime.decode_length)
        tp = self.kv_manager.tp_degree
        per_step_shards = kv_cache_shard_bytes_for_layer_range(
            self.model, context_per_step, tp,
            layer_start=0, layer_end=read_prefix)
        source_group = self.topology.instance(home)
        target_group = self.topology.instance(exec_instance)
        shard_specs = []
        for source_rank, target_rank, per_step_bytes in zip(
                source_group.ranks, target_group.ranks, per_step_shards):
            if per_step_bytes * steps <= 0:
                continue  # 零字节 rank 与 v1 同口径跳过
            shard_specs.append((
                source_rank, target_rank, per_step_bytes,
                deterministic_xy_route(
                    self.hardware, source_rank, target_rank)))
        return {
            "home_instance": home,
            "exec_instance": exec_instance,
            "steps": steps,
            "context_per_step": context_per_step,
            "read_prefix_layers": read_prefix,
            "total_bytes": sum(spec[2] for spec in shard_specs) * steps,
            "shard_specs": tuple(shard_specs),
        }

    def _joint_remote_read_slice(self, runtime, steps: int, k: int):
        """从持久 credit 计划切出本列车切片（块大小 k，列车统一值）：
        块 b 步数 = min(k, steps-(b-1)·k)、字节 = 步数 × 每步读量（整数
        乘法 ⇒ 逐 shard Σ块 ≡ 计划总量，I1 字面成立）。返回块 1..M 的
        KVTransfer 列表（kind/reason/字段与 v1 单笔同形——M=1 时与 v1
        逐字节同对象，块 1 复用 v1 发射路径，I3a tag 序锚）。

        层区间（2026-09-17）：KVTransferShard/KVTransfer 的 layer_end 与
        resident_prefix_layers_before/after 取计划的 read_prefix_layers
        （区间惯例 = layer_start/layer_end；LOCAL 基 p==L 时与旧全层
        硬编码逐字节相同——回归锚）。

        stream_only=True（2026-09-25 补前序 KV 管理器卡登记的偏差项，
        规格书§一.7）：decode credit 读流与 prefill 前缀读流同为瞬时
        流——只产 send/recv/HBM 写服务节点与完成门，不物化入任何持久
        账本（不 _add_local_shards / 不改 shard_bytes / 不进 merge 工
        作副本账）。字节口径零变更（total_bytes 仍逐 shard 精确）。"""
        from face_scheduler import KVTransfer, KVTransferShard
        plan = runtime.remote_read_credit_plan
        read_prefix = plan["read_prefix_layers"]
        block_count = -(-steps // k)
        blocks = []
        for b in range(block_count):
            block_steps = min(k, steps - b * k)
            shards = tuple(
                KVTransferShard(
                    source_rank=source_rank,
                    target_rank=target_rank,
                    edge_rank=None,
                    bytes=per_step_bytes * block_steps,
                    noc_path=noc_path,
                    layer_start=0,
                    layer_end=read_prefix,
                )
                for (source_rank, target_rank, per_step_bytes, noc_path)
                in plan["shard_specs"])
            blocks.append(KVTransfer(
                kind="noc_migrate",
                phase="decode",
                reason="remote_read_stream",
                session_id=runtime.session_id,
                trigger_request_id=runtime.request_id,
                source_instance_index=plan["home_instance"],
                target_instance_index=plan["exec_instance"],
                total_bytes=sum(shard.bytes for shard in shards),
                shards=shards,
                model_layers=self.model.layers,
                layer_start=0,
                layer_end=read_prefix,
                resident_prefix_layers_before=read_prefix,
                resident_prefix_layers_after=read_prefix,
                stream_only=True,
            ))
        return blocks

    def _plan_remote_credit_slices(self, plan, joiner_runtimes,
                                   joiner_plans):
        """列车规划期生成 remote-read credit 切片（§4.3.2/§4.3.3）。

        R15 登记钩子 = 切片创建点（本函数；drain 边界 S_j 未知），
        owner 键 = ``rid#decode#{j}`` 逐切片一键（j = 该请求第 j 个
        切片列车），注销 = Tj 核销边界（_finalize_completed_trains，
        先例 :1305-1307"消费栅栏已物理通过"）——禁止全部切片挂同一
        owner 到完成才注销（Tj 期间登记表会躺 j-1 条已物理完成的陈旧
        流，链路除数最多虚高 n 倍，R-6）。R14 停滞成员不进 members
        （_plan_train :830-835 先行跳过）⇒ 本列车无切片、顺延，I2 联动。

        返回构图器消费规格（None = 本列车无 remote-credit 成员，体发射
        单段路径不变）。块大小取列车统一值 K_train = max(成员 K_j)
        ——体块区间按列车迭代切（I2 span 规则），全部成员切片块界对齐
        同一 K；K_train ≥ K_j 保证各成员块数 ≤ 自适应上限（D1）。"""
        joiner_by_id = {
            runtime.request_id: (runtime, joiner_plan)
            for runtime, joiner_plan in zip(joiner_runtimes, joiner_plans)}
        credit_members = []
        for request_id, participation in plan["members"]:
            runtime = self.runtime_by_request_id[request_id]
            if (runtime.joint_action == "remote-read"
                    and runtime.remote_read_credit_plan is not None):
                credit_members.append((runtime, participation))
        if not credit_members:
            return None
        iters_config = self.joint_config.remote_credit_iters
        k_train = max(
            remote_credit_block_size(iters_config, participation)
            for _, participation in credit_members)
        slice_indices = {}
        continuations = []
        for runtime, participation in credit_members:
            runtime.remote_read_credit_trains += 1
            j = runtime.remote_read_credit_trains
            blocks = self._joint_remote_read_slice(
                runtime, participation, k_train)
            # C8 步骤 2（对账核销之二）：本列车切片的实际登记接管当前
            # 列车份额。先核销并暂挂 #readplan，再登记当前 credit owner；
            # 图侧块链串行，因此该 owner 仅登记一条每 shard 代表流。
            self._consume_readplan_units(runtime, len(blocks))
            self._register_transfer_flows(
                blocks, owner="{}#decode#{}".format(
                    runtime.request_id, j), serial_credit_stream=True)
            slice_indices[runtime.request_id] = j
            runtime.remote_read_slice_summaries.append({
                "train_id": plan["train_id"],
                "slice_index": j,
                "k": k_train,
                "steps": participation,
                "block_steps": [
                    min(k_train, participation - b * k_train)
                    for b in range(len(blocks))],
                "total_bytes": sum(
                    transfer.total_bytes for transfer in blocks),
            })
            joiner_entry = joiner_by_id.get(runtime.request_id)
            tail_blocks = tuple(
                (block_index, blocks[block_index - 1])
                for block_index in range(2, len(blocks) + 1))
            if joiner_entry is not None:
                # 首列车 T1（joiner）：块 1 = pd_transfer 主链槽位（barrier
                # 前，barrier 语义自动降级为等 barrier 之前的节点）；块
                # 2..M 旁挂支链 + 体块 arm 门（构图器 D2 发射序）。
                _, joiner_plan = joiner_entry
                joiner_plan["prefill_decode_transfer"] = blocks[0]
                joiner_plan["remote_credit_tail"] = tail_blocks
            else:
                # 续列车 T2+（续坐成员，D7 新槽位）：块 1 上主链（发射序
                # 先于列车体；无 barrier，per-rank 链序保证先行），块
                # 2..M 旁挂 + arm 门——与 T1 同构，仅少 barrier。
                continuations.append({
                    "plan": runtime.plan_dict(),
                    "block1": blocks[0],
                    "tail": tail_blocks,
                })
        plan["remote_credit_slices"] = slice_indices
        assignments = self._train_span_iterations(plan)
        body_blocks = self._remote_credit_body_blocks(
            plan, assignments, k_train, credit_members,
            start_iter=1, end_iter=plan["iterations"])
        split_first_blocks = self._remote_credit_body_blocks(
            plan, assignments, k_train, credit_members,
            start_iter=1, end_iter=1)
        # 拆分（WP9）与体块化正交：首步批 = 迭代 1（refine 体块 1），
        # 余量批块界沿 K 对齐（[2..K], [K+1..2K], ...）——体块的切片门
        # 按"区间重叠"取并集（I2 泛化式），两批覆盖区间与整列一致。
        split_rest_blocks = []
        rest_start = 2
        while rest_start <= plan["iterations"]:
            rest_end = min(
                ((rest_start - 1) // k_train + 1) * k_train,
                plan["iterations"])
            split_rest_blocks.append(self._remote_credit_body_blocks(
                plan, assignments, k_train, credit_members,
                start_iter=rest_start, end_iter=rest_end)[0])
            rest_start = rest_end + 1
        return {
            "k": k_train,
            "continuations": continuations,
            "request_ids": sorted(slice_indices),
            "body_blocks": body_blocks,
            "split_first_blocks": split_first_blocks,
            "split_rest_blocks": split_rest_blocks,
        }

    @staticmethod
    def _train_span_iterations(plan):
        """span → 列车迭代映射（(迭代, span 下标) 序列；与 pass_spans 的
        平铺布局一一对应：chunk i ↔ 迭代 i+1，成员 m 第 s 步 ↔ 迭代 s）。"""
        assignments = []
        chunk_count = len(plan["prefill_chunk_tokens"])
        for i in range(chunk_count):
            assignments.append((i + 1, i))
        offset = chunk_count
        for _, participation in plan["members"]:
            for step in range(1, participation + 1):
                assignments.append((step, offset + step - 1))
            offset += participation
        if offset != len(plan["pass_spans"]):
            raise RuntimeError(
                "train span layout does not match the frozen plan")
        return assignments

    def _remote_credit_body_blocks(self, plan, assignments, k,
                                   credit_members, *, start_iter,
                                   end_iter):
        """[start_iter, end_iter] 区间的体块切分（I2 span 划分规则：块 b
        覆盖迭代 (b-1)·k+1..min(b·k, end)；chunk/member span 随迭代归属
        入块）。每体块的 gates = 覆盖该区间的全部 remote-read 成员切片块
        (rid, 块号) 列表——并集语义（同列车可并存多个 remote-read 成员，
        各自 participation 不同；构图器经 _credit_arms 账本解析为逐 rank
        recv 完成门）。"""
        blocks = []
        b = 0
        start = start_iter
        while start <= end_iter:
            b += 1
            end = min(b * k, end_iter)
            gates = []
            for runtime, participation in credit_members:
                slice_block_count = -(-participation // k)
                for slice_b in range(1, slice_block_count + 1):
                    s0 = (slice_b - 1) * k + 1
                    e0 = min(slice_b * k, participation)
                    if s0 <= end and start <= e0:
                        gates.append((runtime.request_id, slice_b))
            blocks.append({
                "start_iter": start,
                "end_iter": end,
                "weight_passes": end - start + 1,
                "spans": [
                    plan["pass_spans"][span_index]
                    for iteration, span_index in assignments
                    if start <= iteration <= end],
                "gates": gates,
            })
            start = end + 1
        return blocks

    # --------------------------------------- C5 决策日志 schema（F8 冻结） --

    def _joint_decision_route_hops(self, cost_model, session_view,
                                   instance_index: int) -> int:
        """C5 字段 4：逐候选 hop 披露——与 JCM ``_route`` 同语义
        （source = resident ?? home；source 缺失或同实例恒 0 跳）。
        route_fn 是确定性 XY 路由函数，重复求值零副作用，且与
        estimate_action 计价内 breakdown.hops 同值（单测冗余断言位）。
        applicable 与 inapplicable 候选均落（每决策 × 每候选可重建）。"""
        source = (
            session_view.resident_instance
            if session_view.resident_instance is not None
            else session_view.home_instance)
        if source is None or source == instance_index:
            return 0
        _path, hops = cost_model.route_fn(source, instance_index)
        return int(hops)

    def _joint_breakdown_log_dict(self, breakdown):
        """C5 字段 1：逐候选 ActionCostBreakdown 全 11 字段序列化
        （``_JOINT_BREAKDOWN_LOG_FIELDS`` 冻结字段序）。仅 applicable
        候选调用（体量控制）；值可为 None（C2 物化前），字段存在性与
        可解析性为本卡判据；notes 元组落为 list（JSON 可序列化）。"""
        if breakdown is None:
            return None
        payload = {}
        for name in _JOINT_BREAKDOWN_LOG_FIELDS:
            value = getattr(breakdown, name, None)
            payload[name] = list(value) if isinstance(value, tuple) else value
        return payload

    def _account_decision_wall(self, start_ns: int) -> None:
        """C11 步骤 4：纯决策段墙时累计（总量/次数/最大值）。"""
        elapsed = time.perf_counter_ns() - start_ns
        self._admission_decision_wall_ns_total += elapsed
        self._admission_decision_wall_ns_max = max(
            self._admission_decision_wall_ns_max, elapsed)
        self._admission_decision_count += 1

    def _note_action_selection(self, action: str, candidates,
                               session_view) -> None:
        """C11 步骤 5（弃赛守卫埋点）：四动作分列选中计数——recompute
        仅 elected 口径、forced（no_history/quota_deferred/
        evicted_permanent 三成因，_recompute_selection_tier 单一裁决点
        同源）单列，随 C5 schema 的 selected_action/recompute_selection
        字段同源（tier 经 _recompute_selection_tier 单一裁决点）。
        F6 销账：__init__ 恒设计数表——直接访问（替身漏设 =
        AttributeError）；L1（P2-1，2026-09-23 复核审计）：计数键
        初始化与 forced_reason 枚举同步扩 recompute_forced_
        quota_deferred（动态 format 键曾潜伏 KeyError）。"""
        counts = self._joint_action_selection_counts
        if action == "recompute":
            tier = self._recompute_selection_tier(candidates, session_view)
            if tier["tier"] == "elected":
                counts["recompute_elected"] += 1
            else:
                counts["recompute_forced_{}".format(
                    tier["forced_reason"])] += 1
        else:
            counts[action] += 1

    @staticmethod
    def _recompute_selection_tier(candidates, session_view) -> dict:
        """recompute 的 elected/forced 分位（C5 schema 口径，决策行与
        选中计数埋点同源单一裁决点）。

        forced ⇔ 适用动作集 = {recompute} 单元素；原因 ∈ {no_history
        （首轮无历史 KV）, evicted_permanent（历史被永久驱逐、无有效
        后备副本——三态模型下 REMOTE 可池恢复，枚举先行冻结）,
        quota_deferred（K7/P2-4，2026-09-23 外部审计补入：配额判据
        压塌成 {recompute}——非 recompute 候选存在 quota_ 前缀不可行
        理由；规格变更登记 A15'/PROVENANCE §38，C5 枚举扩一值）}。"""
        applicable_actions = [
            action for action in ACTION_ORDER
            if any(candidate.action == action and candidate.applicable
                   for candidate in candidates)]
        if len(applicable_actions) > 1:
            return {"tier": "elected", "forced_reason": None}
        quota_collapsed = any(
            candidate.action != ACTION_RECOMPUTE
            and not candidate.applicable
            and (candidate.inapplicable_reason or "").startswith("quota_")
            for candidate in candidates)
        # 优先序：no_history（物理成因——历史不存在，配额无关）>
        # quota_deferred（策略成因——历史在而配额挡路）>
        # evicted_permanent。
        return {
            "tier": "forced",
            "forced_reason": (
                "no_history" if session_view.history_tokens == 0 else (
                    "quota_deferred" if quota_collapsed
                    else "evicted_permanent")),
        }

    def _joint_decision_audit_fields(self, record, cost_model,
                                     session_view, flow_snapshot) -> dict:
        """C5（F8 冻结，2026-09-22）：joint_admission 决策行的审计扩展
        字段——可重建"每决策 × 每候选 × 每字段"的完整决策时刻审计流。

        - ``selected_action``：选中动作 ∈ {stay, recompute, remote-read,
          copy}（v2.8 用户裁定；与既有 joint_action 同值，作显式冗余
          断言位）；
        - ``applicable_actions``：适用动作集（ACTION_ORDER 序、可由
          candidates 导出，显式冗余断言位）；
        - ``recompute_selection``：选中 recompute 时的 elected/forced
          分位（设计文档 §5.2 计数口径）。forced ⇔ 适用动作集 = {recompute}
          单元素（无任何其余适用动作，必经重算）；forced 原因 ∈
          {no_history（首轮，无任何历史 KV）, evicted_permanent（历史
          已被永久驱逐、无有效后备副本）, quota_deferred（配额判据压塌
          ——非 recompute 候选带 quota_ 前缀不可行理由；K7/P2-4 枚举
          扩展，A15' 规格变更登记）}。elected ⇔ 至少一个其余适用动作
          存在且联合比较仍胜出（真选中）。计数语义：recompute 选中
          数仅含 elected 口径，forced 单列计数披露、不并入、不静默丢弃
          出分母——两口径由 tier 字段分列，下游计数不得混报。非
          recompute 决策记 None；
        - ``load_view``：逐决策负载视图（InstanceLoadView 构造点数据：
          instance_index + 五个负载量字段，决策时刻快照）；
        - ``flow_snapshot``：流表快照摘要（每链路登记流数，LinkFlowRegistry
          .snapshot()；调用方在本请求准入相登记前捕获 = 计价所见环境）；
        - ``port_snapshot``：端口快照——逐实例 u_port 分解（活跃 decode
          消费流/在册传输流/合计）、bulk 名额已用与上限、平价门余量。
          配额类字段 WP3 前记 "NA"（F8：不缺席；C2 注册表/C11 接通后
          替换为实测值）。

        披露边界（冻结）：本 schema 只含决策时刻记录；结算时刻的逐请求
        合并披露（合并方向/零字节结算/home 迁移）走 FS 侧
        kv_delta_journal（C14 步骤 5 接口声明），不进本行。"""
        chosen = record.chosen
        applicable_actions = [
            action for action in ACTION_ORDER
            if any(candidate.action == action and candidate.applicable
                   for candidate in record.candidates)]
        if chosen.action != ACTION_RECOMPUTE:
            recompute_selection = None
        else:
            # C11：tier 计算经 _recompute_selection_tier 单一裁决点
            # （决策行 schema 与选中计数埋点同源）。
            recompute_selection = self._recompute_selection_tier(
                record.candidates, session_view)
        load_view = [
            {
                "instance_index": view.instance_index,
                "queued_task_load_ns": view.queued_task_load_ns,
                "running_task_load_ns": view.running_task_load_ns,
                "active_decode_task_load_ns": view.active_decode_task_load_ns,
                "hbm_remaining_bytes_by_tp_rank": list(
                    view.hbm_remaining_bytes_by_tp_rank),
                "reclaimable_bytes_by_tp_rank": list(
                    view.reclaimable_bytes_by_tp_rank),
            }
            for _, view in sorted(cost_model.loads.items())
        ]
        # C11 步骤 6（port_snapshot 接通）：配额 on = 实测值替换 NA——
        # u_port 分解读 C2 注册表（hbm_port_flow_registry.snapshot()，
        # 逐实例取其 rank 端口的最忙值——TP 全 rank 同账下各 rank 同
        # 计数，多传输 rank 子集差异取 max = 约束端口口径）；bulk 名额
        # 已用/上限与平价门余量读配额 tracker（实例级端口）。配额 off
        # （缺省）保持 C5 冻结的 "NA" 占位（F8：不缺席）——off 臂决策
        # 日志与 C11 前逐字节同。
        if self._quota_tracker is not None:
            tracker = self._quota_tracker
            r_hat_kv = self._quota_r_hat_kv_bytes_per_ns()
            rank_snap = self._hbm_ports.snapshot()
            port_snapshot = {
                "instances": [
                    {
                        "instance_index": index,
                        "u_port_active_decode_streams": len(
                            self.instances[index].active_decode),
                        "u_port_registered_transfer_flows": max(
                            (rank_snap[rank][
                                "u_port_registered_transfer_flows"]
                             for rank in self.topology.instance(index).ranks
                             if rank in rank_snap), default=0),
                        "u_port_total": (
                            len(self.instances[index].active_decode)
                            + max((rank_snap[rank][
                                    "u_port_registered_transfer_flows"]
                                   for rank
                                   in self.topology.instance(index).ranks
                                   if rank in rank_snap), default=0)),
                        "bulk_slots_used": tracker.bulk_used(index),
                        "bulk_slots_cap": tracker.n_bulk,
                        "parity_gate_headroom": (
                            tracker.port_parity_headroom(
                                index, r_hat_kv,
                                u_port_base=len(
                                    self.instances[index].active_decode))),
                    }
                    for index in sorted(cost_model.loads)
                ],
            }
        else:
            port_snapshot = {
                "instances": [
                    {
                        "instance_index": index,
                        "u_port_active_decode_streams": "NA",
                        "u_port_registered_transfer_flows": "NA",
                        "u_port_total": "NA",
                        "bulk_slots_used": "NA",
                        "bulk_slots_cap": "NA",
                        "parity_gate_headroom": "NA",
                    }
                    for index in sorted(cost_model.loads)
                ],
            }
        return {
            "selected_action": chosen.action,
            "applicable_actions": applicable_actions,
            "recompute_selection": recompute_selection,
            "load_view": load_view,
            "flow_snapshot": flow_snapshot,
            "port_snapshot": port_snapshot,
        }

    def _try_admit_request(self, runtime, now_ns: int) -> bool:
        """joint 准入（设计方案 §1/§2/§7.1）：全 instance 候选（无容量/
        边缘/距离掩码——HBM 只影响动作计价中的驱逐等待，不做准入过滤），
        联合/顺序选择 instance × stay/recompute/copy/remote-read；选择后
        物化（预约已知输入 + prepare_prefill 动作 + 入队）。决策输入无
        oracle：runtime.final_context_tokens / decode_length 不进视图。

        R2（2026-09-14）准入事务化：预约+prepare 为一个事务——全部
        runtime 字段改写移到事务成功之后（D7：失败请求零残留，预约账
        本/双纪元/runtime 字段不被污染）；容量类失败（KVCapacityError，
        N3' 类型化）按三态分类（P5-a：对任一 (instance, action) 组合判
        物理可行性——存在任一组合物理可行 → 暂时不可行 return False
        接通纪元门；全部组合不可行 → 结构性 fail-closed 带逐 rank 缺
        口）；失败也落盘（D5）+ 同分类折叠计数（防日志体积乘积爆炸）。
        合同类异常（重复预约/context 收缩等）不被延迟路径吞掉（负例
        单测钉死）。"""
        if runtime.estimated_arrival_ns is None:
            raise RuntimeError("request cannot be admitted before arrival")
        # C11 步骤 4（决策时延埋点）：纯决策段（视图构造 → 候选计价 →
        # 配额判据 → 选择）墙时——事务/登记段不计（那是执行提交，非
        # 决策增量）；30s 尺度实测归档、不取性能结论。
        decision_wall_start = time.perf_counter_ns()

        session_view = self._joint_session_view(runtime.session_id)
        estimated_decode, horizon_source = self._joint_horizon.estimate(
            runtime.session_id)
        input_kv_shards = kv_cache_shard_bytes_for_tokens(
            self.model, runtime.joint_input_tokens,
            self.kv_manager.tp_degree)
        request_view = RequestView(
            request_id=runtime.request_id,
            session_id=runtime.session_id,
            input_tokens=runtime.joint_input_tokens,
            history_tokens_before=session_view.history_tokens,
            estimated_decode_tokens=int(estimated_decode),
            horizon_source=horizon_source,
            input_kv_bytes_by_tp_rank=tuple(input_kv_shards),
        )
        # C3（WP1c）：decode 计价上下文 = history + input（准入时已知，
        # 与 reserve 的 final_context_tokens 同式）+ estimator 现值 horizon
        # （与 RequestView.estimated_decode_tokens 同源同刻——JCM 端乘回
        # 同值 horizon，per-token 均值 × 步数 ≡ 逐步剩余总量）。
        cost_model = self._joint_cost_model(
            now_ns,
            decode_context_tokens=(
                session_view.history_tokens + runtime.joint_input_tokens),
            decode_average_length=estimated_decode)
        record = select_instance_and_action(
            mode=self._joint_mode,
            cost_model=cost_model,
            session=session_view,
            request=request_view,
            remote_enabled=self.joint_config.remote_enabled,
        )
        # ---- C11 步骤 1：逐候选配额判据（动作级，实例永不掩码）+ 配额
        # 可行集内重选（argmin 同序；δ_adm=0 与既有比较合取恒等，见
        # _quota_candidate_verdict 注释）。全不适用 ⇒ quota_deferred 回队
        # （C9 语义：重试键扩展配额代数，复用本类 epoch 键重试门）。
        # ----
        if self._quota_tracker is not None:
            (quota_chosen, quota_candidates,
             quota_deferred) = self._quota_filter_candidates(
                record, session_view, runtime.request_id, now_ns)
            if quota_deferred is not None:
                deferred_record, _r_hat = quota_deferred
                self._account_decision_wall(decision_wall_start)
                self._log_admission_failure(
                    runtime, now_ns, record,
                    reason="; ".join(
                        "{}: {}".format(action, reason)
                        for action, reason
                        in sorted(deferred_record.inapplicable_reasons
                                  .items())),
                    failure_class=(
                        "quota_deferred_" + deferred_record.wait_reason
                        if deferred_record.wait_reason != WAIT_CAPACITY
                        else "quota_deferred"),
                )
                self._quota_deferred_wait_counts[
                    deferred_record.wait_reason] += 1
                if runtime.quota_deferred_since_ns is None:
                    runtime.quota_deferred_since_ns = now_ns
                candidate_instances = {
                    candidate.instance_index
                    for candidate in record.candidates
                    if candidate.applicable}
                candidate_instances.add(record.chosen.instance_index)
                # 失败键 = KV 纪元 ⊕ 失败候选集实例纪元 ⊕ 配额代数
                # （quota_deferred_requeue 的 retry_key 同构——C9 冻结
                # 语义：配额代数分量经 _extend_retry_key_with_quota 并入，
                # 流 settle/预留释放 bump 代数即重开重试门）。
                self._last_admit_failure_key = (
                    self._compose_admit_failure_key(candidate_instances))
                return False
            record = _dataclass_replace(
                record, chosen=quota_chosen, candidates=quota_candidates)
        chosen = record.chosen
        selected = chosen.instance_index
        self._account_decision_wall(decision_wall_start)
        self._note_action_selection(chosen.action, record.candidates,
                                    session_view)

        # ---- 规格书§二.2（2026-09-25 prefill remote-read 分阶段）：前缀
        # 读流规划。基形态会话（prepare_prefill 之前）+ 纯规划零副作用
        # ——置于 reserve 之前，fail-closed raise 不留预约残留。LOCAL 基
        # = 全层 [0,L) 读流（history_transfers 恒空）；PARTIAL 基 = 前缀
        # [0,p) 读流（后缀 [p,L) 池恢复仍由 prepare_prefill 产
        # history_transfers，两腿共享同一准入 frontier 并行分叉）。
        prefill_read_transfers = ()
        if chosen.action == "remote-read":
            base_session = self.kv_manager._sessions.get(runtime.session_id)
            if base_session is not None:
                # 与 _preregister_readplan_flows 同守卫：无基础会话时真
                # 计划亦为 None，无读流可规划（退化防御，正常不可达）。
                prefill_read_transfers = (
                    self.kv_manager.plan_prefill_remote_read_transfers(
                        base_session, selected, session_view.history_tokens,
                        runtime.request_id))

        # ---- 事务段：reserve + prepare（失败零残留，D7）。 ----
        try:
            admission_evictions = self.kv_manager.reserve_request_capacity(
                request_id=runtime.request_id,
                session_id=runtime.session_id,
                instance_index=selected,
                final_context_tokens=(
                    session_view.history_tokens
                    + runtime.joint_input_tokens),
                action=chosen.action,
            )
            (runtime.history_location_before,
             history_transfers,
             prepare_evictions) = self.kv_manager.prepare_prefill(
                session_id=runtime.session_id,
                target_instance_index=selected,
                history_tokens=session_view.history_tokens,
                trigger_request_id=runtime.request_id,
                reservation_request_id=runtime.request_id,
                action=chosen.action,
            )
        except KVCapacityError as exc:
            # D7 清账：raise 前已提交的逐出保留（合法释放）但必须补
            # 纪元 bump + 图侧同步；预约未登记（reserve 失败）或已
            # 释放（prepare 失败时事务回滚由 kv_manager 内部保证——
            # reserve 成功而 prepare 失败的容量类回退在下面统一处理）。
            if exc.evictions:
                self._bump_kv_ledger_epoch()
                self.graph.sync_pending_history_after_evictions(exc.evictions)
                self._emit_eviction_only_nodes(
                    exc.evictions, now_ns,
                    trigger_request_id=runtime.request_id)
            self._release_orphan_reservation(runtime.request_id)
            self._log_admission_failure(
                runtime, now_ns, record, str(exc), "capacity_deferred")
            # N6+F3+M2（kimi 复审）：失败键 = KV 纪元 ⊕ **失败候选集**
            # 实例（选中实例 ∪ 本次候选表中 applicable 实例，冻结于失败
            # 时刻）的纪元。规格 R2.5 原文"上次判定不可行的选中/失败候
            # 选集内实例"——只挂选中实例时，任一未选候选实例的负载迁移
            # （argmin 翻转到可行候选）不重开重试门：生产模式损失换选
            # 时机，SH_ADMIT_GATE_VERIFY 影子断言被合法击穿（假阳性
            # abort）。F3 红线仍守：键只含失败候选集（冻结集合、非全
            # 实例之或）。_admit_waiting_requests 在 False 返回后读取。
            candidate_instances = {
                candidate.instance_index
                for candidate in record.candidates
                if candidate.applicable}
            candidate_instances.add(selected)
            # C11：失败键经 _compose_admit_failure_key 合成（KV 纪元 ⊕
            # 候选集实例纪元 ⊕ 配额代数——off 模式退化为既有纯容量键，
            # 与 _current_retry_key 的对称求值保持门语义零漂移）。
            self._last_admit_failure_key = (
                self._compose_admit_failure_key(candidate_instances))
            return False
        # 事务成功：统一补纪元 + 图侧同步（预约/prepare 两处逐出）。
        self._bump_kv_ledger_epoch()
        self.graph.sync_pending_history_after_evictions(
            admission_evictions + prepare_evictions)

        # ---- 事务成功段：runtime 字段落账本（D7 重排）+ R13 重算口径。 ----
        snapshots = tuple(
            self._task_load_snapshot(state, now_ns)
            for state in self.instances)
        runtime.prefill_instance_index = selected
        runtime.joint_action = chosen.action
        runtime.joint_cost_ns = chosen.cost_ns
        runtime.prefill_assignment_key = snapshots[selected].ordering_key
        runtime.prefill_instance_loads = snapshots
        runtime.prefill_affinity_reason = "joint_{}".format(chosen.action)
        runtime.admission_time_ns = now_ns
        runtime.hbm_wait_ns = now_ns - runtime.estimated_arrival_ns
        runtime.origin_home_instance = session_view.home_instance
        runtime.history_transfers = history_transfers
        runtime.history_transfer = (
            history_transfers[0] if history_transfers else None)
        runtime.history_evictions = admission_evictions + prepare_evictions
        if runtime.turn_index > 0:
            runtime.history_source_instance_index = (
                session_view.resident_instance)
            runtime.history_transfer_bytes = sum(
                transfer.total_bytes
                for transfer in history_transfers
                if transfer.kind != "local_hit")
        # 规格书§二.1/§二.2（2026-09-25）：前缀读流入 runtime 账本——
        # 与 history_transfers 严格分离（后缀池恢复留 history_transfers；
        # 前缀读流不进 history_transfer_bytes / readiness barrier / merge
        # 账本），owner = rid#prefill_read、prefill drain 释放。计划摘要
        # 锚同 remote_read_credit_plan 形态（home/exec/前缀层界/字节/逐
        # 组层段），组区间并集恒 = [0, p)。无条件覆盖赋值——quota_link
        # 失败回队重试改选非 remote-read 动作时，不残留上一失败事务的
        # 前缀读流账本。
        runtime.prefill_remote_read_transfers = prefill_read_transfers
        runtime.prefill_remote_read_bytes = sum(
            transfer.total_bytes
            for transfer in prefill_read_transfers)
        runtime.prefill_remote_read_plan = None
        if prefill_read_transfers:
            runtime.prefill_remote_read_plan = {
                "home_instance": (
                    prefill_read_transfers[0].source_instance_index),
                "exec_instance": selected,
                "read_prefix_layers": prefill_read_transfers[-1].layer_end,
                "total_bytes": runtime.prefill_remote_read_bytes,
                "groups": tuple(
                    (transfer.layer_start, transfer.layer_end,
                     transfer.total_bytes)
                    for transfer in prefill_read_transfers),
            }

        # R13（N7(b)，2026-09-14）：recompute 只重算**缺失区间**——
        # 驻留目标（home，LOCAL/PARTIAL 前缀复用）仅池后缀层折算 token
        # （ceil(H×(L−prefix)/L)，LOCAL 为 0），span 基 = H；异地/REMOTE
        # 基础整份重算，span 基 = 0。prefill_tokens_to_process 同步改写
        # ——_plan_train 的 chunk 工作量来源（此前漏改，recompute 带
        # H>0 会在列车规划撞 "ran out of work" 死端）。
        if chosen.action == "recompute":
            resident_here = (
                session_view.resident_instance == selected
                and session_view.location in (
                    "local_hbm", "partial_hbm_remote"))
            if resident_here:
                history_tokens = session_view.history_tokens
                prefix = session_view.resident_prefix_layers
                layers = self.model.layers
                missing_equiv = (
                    (history_tokens * (layers - prefix) + layers - 1)
                    // layers if prefix < layers else 0)
                runtime.joint_span_base_context = history_tokens
            else:
                missing_equiv = session_view.history_tokens
                runtime.joint_span_base_context = 0
            runtime.joint_prefill_work = (
                missing_equiv + runtime.joint_input_tokens)
            runtime.prefill_tokens_to_process = runtime.joint_prefill_work
            runtime.remaining_chunks = math.ceil(
                runtime.joint_prefill_work / self.p_chunk)

        # R15-1：准入主链与逐出支链分属不同 owner。主链历史迁移在 drain
        # 边界注销；#evict 由独立逐出尾 watch 注销。
        # C5：决策时刻流表快照须在本次登记**之前**捕获——与 estimate_
        # action 计价所见的链路争用环境一致（登记的是本决策自身的在途
        # 足迹，属决策结果而非决策输入）。
        decision_flow_snapshot = self._joint_flows.snapshot()
        self._register_transfer_flows(
            tuple(runtime.history_transfers), owner=runtime.request_id)
        self._register_transfer_flows(
            runtime.history_evictions,
            owner=runtime.request_id + "#evict")
        # 规格书§二.4（2026-09-25）：准入时登记 prefill remote-read 前缀
        # 读流（owner = rid#prefill_read；prefill drain 释放）。NoC 路径
        # ＋ home HBM 读端口 + exec HBM 写端口由 _register_transfer_flows
        # 按 noc_migrate 腿型成对登记。同 tick 后续决策经同一路径集看到
        # 本笔已提交读流足迹（C8 承诺可见性语义由真实在途流承担）。
        if runtime.prefill_remote_read_transfers:
            self._register_transfer_flows(
                runtime.prefill_remote_read_transfers,
                owner=runtime.request_id + "#prefill_read")
        # C8（WP2-preadmit，§4.1 同 tick 承诺可见性）：选中 remote-read
        # 即建 est 承诺账本（事务成功路径）——同 tick 串行贪婪的第二笔
        # 决策（_admit_waiting_requests 循环内后续 _try_admit_request →
        # _joint_cost_model 读同一张注册表）立即可见第一笔已提交的读流
        # 足迹。est 时域 = CausalHorizonEstimator 现值（与本次计价同刻
        # 同源），不确定性经 drain 对账披露。2026-09-25 起准入相注册表
        # 零登记（#readplan 半边延迟到 drain 对账以真值建立）：prefill
        # 期间注册表上的远读流 = rid#prefill_read 实流本身——同一条前缀
        # 读流不得同时被登记为 prefill 流与 decode 流（规格书§二.5，除数
        # 不双计两阶段）。
        if chosen.action == "remote-read":
            self._preregister_readplan_flows(
                runtime, session_view, selected, estimated_decode)
        # C11（入册半）：选中动作的配额流生命周期入册（oneshot 支链 +
        # copy 主流程 / remote-read 读流 + merge 义务预留）。入册序与失
        # 败分派（O2，2026-09-23 终轮审计）：主流程恒先于逐出支链——判据
        # 时刻本请求逐出足迹不可知（history_evictions 由准入事务产出、
        # 下方事务成功才落账；判据在事务前），探针实证事故链：判据 PASS
        # → 支链占末槽 → 主流程 FAIL → 已物化预约 fail-closed RuntimeError
        # 整 run abort。主先支后使该链结构性不可达；支链失败已在入册内
        # 按 quota_oneshot_overflow 披露降级（返回 True 不走本分支）。
        # 本防御分支 = 主流程入册失败（真异常，单线程内判据与入册间
        # tracker 零变更）——回滚已入册流并返回 False（不留半册）。
        if (self._quota_tracker is not None
                and not self._quota_enroll_admission(
                    runtime, session_view, chosen.action, selected, now_ns)):
            # A10'(b)/A11'（2026-09-22，§4.3 补遗）：防御分支回滚——准入
            # 事务成功段已登记的在途流承诺（owner=rid 历史迁移/逐出支链
            # ＋ owner=rid#readplan 读流预登记）与 preplan 账目**先行**注销
            # （可逆半边恒回滚，幂等空放：非 remote-read 路径 #readplan
            # 半边天然为空），消除"重试改选动作成功前注册表躺着幽灵承诺
            # （除数虚高）"窗口；预约处置随后按物化状态分派——未物化
            # 释放+照常 quota_deferred 回队，已物化 fail-closed（无
            # un-prepare 逆路径，假回队会在重试的重复预约上崩溃，详见
            # _release_reserved_admission_or_fail）。
            self._rollback_admission_registrations(runtime)
            self._release_reserved_admission_or_fail(runtime.request_id)
            self._log_admission_failure(
                runtime, now_ns, record,
                reason="quota enrollment failed after transaction "
                "(main-flow enrollment failure: verdict admitted but "
                "enrollment rejected -- between verdict and enrollment "
                "the tracker changed outside this request's own "
                "eviction footprint; O2 deferred only the eviction "
                "branch, main flow stays fail-closed)",
                # A14'（H6，2026-09-22 第三轮复审）：failure_class 带
                # quota_link 后缀——决策日志 wait_reason 与下方
                # _quota_deferred_wait_counts 计数键同源（原裸
                # "quota_deferred" 被映射表判 capacity，与指标行自相矛
                # 盾；防御分支不掌握翻转侧别，沿指标侧既有归类取
                # quota_link）。
                failure_class="quota_deferred_quota_link")
            self._quota_deferred_wait_counts[
                WAIT_QUOTA_LINK] += 1
            if runtime.quota_deferred_since_ns is None:
                runtime.quota_deferred_since_ns = now_ns
            # A14'（H6）：重试门键对齐容量/配额两路径口径——applicable
            # 候选 ∪ selected（此处 record 已是配额过滤后的候选表；原
            # {selected, chosen.instance_index} 中 chosen.instance_index
            # 即 selected，恒单元素——翻转后 argmin 转向可行候选时重试
            # 门不随该候选负载迁移重开）。
            candidate_instances = {
                candidate.instance_index
                for candidate in record.candidates
                if candidate.applicable}
            candidate_instances.add(selected)
            self._last_admit_failure_key = (
                self._compose_admit_failure_key(candidate_instances))
            return False
        # C11 步骤 5：deferred 驻留时长样本结算（首等待 → 最终准入）。
        if runtime.quota_deferred_since_ns is not None:
            self._quota_deferred_dwell_ns.append(max(
                0, now_ns - runtime.quota_deferred_since_ns))
            runtime.quota_deferred_since_ns = None

        # joint 决策日志（§7：origin_home/execution/action/成本分解；
        # R15 披露：链路/端口争用覆盖与在线因子状态；C5（F8 冻结）：
        # 审计扩展字段——selected_action/适用动作集/recompute 分位/
        # 负载视图/流表快照/port_snapshot + 逐候选 hop 与 breakdown
        # 全 11 字段，可重建"每决策 × 每候选 × 每字段"）。
        self.log_decision(
            {"kind": "joint_admission", "request_id": runtime.request_id,
             "priority": 0},
            now_ns,
            decision={
                "joint_mode": record.mode,
                "joint_action": chosen.action,
                "joint_instance_index": selected,
                "joint_cost_ns": chosen.cost_ns,
                "origin_home_instance": runtime.origin_home_instance,
                # 规格书§二.3（2026-09-25）：prefill remote-read 前缀读流
                # 独立披露（bytes/layers/逐流 shard 摘要；非前缀读流请求
                # 为 0/0/[]——键恒在，读者前向兼容）。
                "prefill_remote_read_bytes":
                    runtime.prefill_remote_read_bytes,
                "prefill_remote_read_layers": (
                    runtime.prefill_remote_read_plan["read_prefix_layers"]
                    if runtime.prefill_remote_read_plan is not None
                    else 0),
                "prefill_remote_read_transfers": [
                    _transfer_summary(transfer)
                    for transfer in runtime.prefill_remote_read_transfers],
                "horizon_source": horizon_source,
                "estimated_decode_tokens": int(estimated_decode),
                "category_mode": self.joint_config.category_mode,
                "layer_policy": self.joint_config.layer_policy,
                "remote_enabled": self.joint_config.remote_enabled,
                # C11：配额模式随行披露（run 级事实，与 category_mode
                # 等开关键对称；off 臂值恒 "off"）。
                "quota_mode": self.joint_config.quota_mode,
                "instance_rule_note": record.instance_rule_note,
                # C8（E13 唯一读者）：争用覆盖披露（值构造见
                # _contention_coverage_value——注册表层 + 遥测完备段）。
                "contention_coverage": self._contention_coverage_value(),
                "service_factors": self._joint_factors.as_dict(),
                # ---- C5（F8 冻结）决策时刻审计 schema 扩展 ----
                **self._joint_decision_audit_fields(
                    record, cost_model, session_view,
                    decision_flow_snapshot),
                "candidates": [
                    {
                        "instance_index": candidate.instance_index,
                        "action": candidate.action,
                        "applicable": candidate.applicable,
                        "cost_ns": candidate.cost_ns,
                        "inapplicable_reason":
                            candidate.inapplicable_reason,
                        # C5 字段 4：逐候选 hop——applicable 与
                        # inapplicable 候选均落（route_fn 确定性可算）；
                        # applicable 候选与 breakdown.hops 同值（冗余
                        # 断言位，单测钉死）。
                        "hops": self._joint_decision_route_hops(
                            cost_model, session_view,
                            candidate.instance_index),
                        # C5 字段 1：逐候选 ActionCostBreakdown 全 11
                        # 字段——仅 applicable 候选（体量控制），
                        # inapplicable 候选记 None。
                        "breakdown": self._joint_breakdown_log_dict(
                            candidate.breakdown),
                        # M6（kimi 复审）：notes 随候选落决策日志——
                        # R3'.3 的 deep_gap_unresolved 等注记须可检索
                        # （§9 验收"可检索"口径），此前只进内存不落盘。
                        # notes 载于 breakdown（不适用候选无 breakdown）。
                        "notes": list(candidate.breakdown.notes)
                        if candidate.breakdown is not None else [],
                    }
                    for candidate in record.candidates
                ],
            },
        )
        self.instances[selected].qp.append(runtime)
        self.instances[selected].last_arrival_ns = now_ns
        runtime.queued_chunk_load_ns = self._queued_chunk_load_full_ns(
            instance_size=self.topology.instance(selected).size,
            runtime=runtime)
        self._bump_instance_epoch(self.instances[selected])
        runtime.admitted = True
        self._refresh_frontier(self.instances[selected])
        self._ledger_admit(
            runtime.request_id, now_ns,
            {"type": "prefill_qp", "instance_index": selected})
        self._emit_admission(runtime, now_ns)
        return True

    def _release_orphan_reservation(self, request_id: str) -> None:
        """R2 清账：事务失败路径上若预约已登记（reserve 成功、prepare
        容量失败），释放之——失败请求零残留（D7）。

        守卫语义（kimi 复审注记）：部分物化预约（extra shards 非空）
        不是容量失败回退的合法形态——prepare 容量路径要么全未物化
        （_ensure_capacity 首次 raise 于任何 _add_local_shards 之前），
        要么已物化即事务成功；当前实现该守卫不可达，作为防御性
        fail-closed 保留（若未来 prepare 改为增量物化，此处是第一道
        报警线，不得静默释放）。"""
        reservation = self.kv_manager._reservations.get(request_id)
        if reservation is None:
            return
        extra = self.kv_manager._reservation_extra_shards(reservation)
        if any(extra):
            # 预约已部分物化说明 prepare 已推进到改写会话——该状态不
            # 属于容量失败回退的合法形态，交由 fail-closed 上报。
            raise RuntimeError(
                "admission rollback found a partially materialized "
                "reservation for request {}".format(request_id))
        self.kv_manager.release_request_capacity_reservation(request_id)
        self._bump_kv_ledger_epoch()

    def _release_reserved_admission_or_fail(self, request_id: str) -> None:
        """A11'（2026-09-22，§4.3 补遗）：配额入册失败防御分支的预约
        处置——按物化状态分派，与 ``_release_orphan_reservation``
        （容量失败路径，未物化守卫）分立：两语境的合法形态集合不同，
        不得共用守卫。

        未物化（extra 全零）：释放预约 + 纪元 bump，调用方照常按
        quota_deferred 回队（重试走全量重登记，零残留）。已物化
        （prepare 已推进会话账目——正常输入>0 的准入恒此态）：
        fail-closed——``release_request_capacity_reservation`` 对部分
        物化预约恒 raise（无 un-prepare 逆路径），保留预约回队则重试
        必崩在 ``reserve_request_capacity`` 的重复预约 ValueError 上；
        静默保留或假回队都把判据翻转的腐坏转嫁到远离成因的下游。
        """
        reservation = self.kv_manager._reservations.get(request_id)
        if reservation is None:
            return
        extra = self.kv_manager._reservation_extra_shards(reservation)
        if any(extra):
            raise RuntimeError(
                "quota enrollment failed after a materialized admission "
                "transaction for request {}: reservation holds {} extra "
                "shard bytes with no un-prepare path (verdict flip between "
                "evaluation and enrollment indicates admission corruption; "
                "registrations were rolled back above -- fail-closed, do "
                "not requeue)".format(request_id, sum(extra)))
        self.kv_manager.release_request_capacity_reservation(request_id)
        self._bump_kv_ledger_epoch()

    def _rollback_admission_registrations(self, runtime) -> None:
        """A10'(b)（2026-09-22，§4.3 补遗）：配额入册失败防御分支的承诺
        回滚——``_release_transfer_flows(rid)``（历史迁移）＋
        ``_release_transfer_flows(rid#evict)``（准入逐出支链）＋
        ``_release_transfer_flows(rid#prefill_read)``（prefill 前缀读流，
        规格书§二.5）＋ ``_release_transfer_flows(rid#readplan)``（C8
        est 账本对齐的注册表半边，现准入相零登记、幂等空放）＋ 清
        ``runtime.remote_read_preplan``。调用点 =
        _quota_enroll_admission 返回 False 的主流程失败防御路径（O2：
        逐出支链失败不入此路径——已按披露降级；主流程需求判据已前瞻、
        单线程内判据与入册间 tracker 零变更，到达此处 = 真异常的安全
        网）；_release_transfer_flows 幂等空放，未登记半边无副作用。
        回队重试的再准入走全量重登记，本回滚保证其间注册表零幽灵承诺
        （除数不被死承诺虚抬）。"""
        self._release_transfer_flows(runtime.request_id)
        self._release_transfer_flows(runtime.request_id + "#evict")
        self._release_transfer_flows(runtime.request_id + "#prefill_read")
        self._release_transfer_flows(runtime.request_id + "#readplan")
        runtime.remote_read_preplan = None

    def _require_batch_tick(self) -> int:
        """R17-1b tick 守护（kimi N4，2026-09-17）：容量逐出日志行的
        tick 必须取自在场批次——_batch 由基类生命周期重建
        （online_scheduler_base.py 初始化置 None / _start_batch 先于
        策略处理重建），策略处理路径恒在场；原 `if self._batch else 0`
        的 0 回退是虚构防御，落日志后会破坏全序单调（hbm_watermark
        重放侧 tick 回退即 fail-closed）。改为断言式：不在场 = 生命
        周期破损，当场 fail-closed 早爆。"""
        assert self._batch is not None, (
            "eviction-side log emitted outside an active batch "
            "(scheduler lifecycle violation)")
        return self._batch["tick"]

    def _emit_eviction_only_nodes(
        self, evictions, tick: int, *, trigger_request_id: str):
        """D7/R2：旁路逐出的图侧发射 + 决策日志披露（R17-1b 咽喉点）。

        图侧：旁路支链、无触发门——与 prefill_evictions 同构；逐出是
        合法的容量释放，其池写必须进图（否则 C++ 水位盲区 + pending
        store 不登记）。
        R17-1b（2026-09-17）：同一咽喉点补决策日志行（kind=kv_eviction，
        逐出精确 tick）——此前通道 2（decode 增长逐出，成功/停滞两路）
        与通道 3（准入失败已提交逐出；R17-7 探针实证当批零触发、属
        潜伏位点）物理进图但决策日志零落点，hbm_watermark 重放对
        "victim 部分层逐出→池化→全层池恢复"链路系统性失明（第三次
        错误，session_000411 受害链）。五个现存调用点一次全覆盖；
        未来任何失败路径走旁路发射即自动带日志——"每个执行点都有
        日志落点"的制度化（机制级强化项见方案 §8-C3）。日志行在图
        发射成功之后落（披露忠实于已发生的物理事实）。行不入
        joint_audit_kinds、不经 seen_kinds 去重（同请求可多行）；
        trigger_request_id 供重放侧 request_id 校验复用。"""
        if not evictions:
            return None
        watch_id = self._next_eviction_watch_id(
            trigger_request_id, "side")
        watch = self.graph.emit_eviction_side_branch(
            [transfer for transfer in evictions], tick, watch_id=watch_id)
        self.log_decision(
            {"kind": "kv_eviction", "request_id": trigger_request_id,
             "priority": 0},
            tick,
            decision={"evictions": [
                _transfer_summary(transfer) for transfer in evictions]},
        )
        if watch is None:
            return None
        flow_owner = watch_id + "#flow"
        self._register_transfer_flows(evictions, owner=flow_owner)
        quota_owner = self._quota_enroll_eviction_branch(
            watch_id + "#quota", trigger_request_id, evictions, tick)
        self._register_eviction_watch(
            watch,
            flow_owners=(flow_owner,),
            quota_owners=((quota_owner,) if quota_owner is not None else ()),
        )
        return watch

    def _classify_physical_feasibility(
        self, runtime, session_view,
    ) -> tuple[bool, list[str]]:
        """P5-a 三态分类：对全部 (instance, action) 组合判物理可行性
        （request_hbm_eventually_feasible_instances，动作感知口径）。

        M1（kimi 复审，2026-09-14）：按**适用动作集合**过滤（规格 R2.4
        "逐实例按其适用动作集合的 footprint 逐一判"）——remote-read 仅
        在其适用面非空时参与判定，否则其 input-only 足迹几乎恒可行，把
        结构性不可行掩盖为暂时不可行，丢失 KVPhysicalInfeasibleError 的
        逐 rank 缺口诊断。

        O12（2026-09-23 终轮审计）：remote-read 适用面与 N1(a) 解除
        （2026-09-17《部分层逐出kv管理改造分析方案》需求①）同步——
        PARTIAL 基（partial_hbm_remote）remote-read 是合法适用动作
        （准入相后缀 [p,L) 池恢复物化 + decode 相前缀 [0,p) credit
        读流，适用性判据与 JCM action_applicability 逐条件同构：
        joint_cost_model.py :1410-1416）。修前"仅 LOCAL 基参与判定"
        是 N1(a) 解除前的旧口径残留——PARTIAL 基上 remote-read 为
        唯一可行动作时，分类器漏判该组合 ⇒ structural_infeasible
        诊断误报（本应可行的请求被报结构性不可行）。REMOTE 基仍拒
        （无主 session，裁定③走池恢复/就地重算）；消融开关
        JOINT_REMOTE_READ_PARTIAL=off 时 PARTIAL 基照旧不进判定。
        返回 (structural_infeasible, per_instance_detail)。"""
        final_context = (
            session_view.history_tokens + runtime.joint_input_tokens)
        detail = []
        any_feasible = False
        # O12：与 JCM remote_ok 逐条件同构（location ∈ {local_hbm,
        # partial_hbm_remote} ∧ PARTIAL 受 remote_read_partial 消融 ∧
        # 有主 resident——resident != target 的逐实例过滤天然内蕴于
        # 下方逐实例可行性循环）。
        remote_allowed = (
            self.joint_config.remote_enabled
            and session_view.location in (
                "local_hbm", "partial_hbm_remote")
            and (session_view.location == "local_hbm"
                 or self.joint_config.remote_read_partial_enabled)
            and session_view.resident_instance is not None)
        for action in ("stay", "recompute", "copy", "remote-read"):
            if action == "remote-read" and not remote_allowed:
                continue
            feasible = self.kv_manager.request_hbm_eventually_feasible_instances(
                session_id=runtime.session_id,
                final_context_tokens=final_context,
                action=action,
            )
            for instance_index, ok in enumerate(feasible):
                if ok:
                    any_feasible = True
                    detail.append(
                        "instance {} x {} physically feasible".format(
                            instance_index, action))
        return (not any_feasible), detail

    def _log_admission_failure(
        self, runtime, now_ns, record, reason: str, failure_class: str,
    ) -> None:
        """R2.6（D5/P4）：失败也落盘——首条失败分类带全候选表；同请求
        同分类的后续失败折叠为紧凑计数行（防深尾部"延迟 × 纪元重试"
        乘积下失败日志体积逼近成功日志）。

        C11：wait 行携带 wait_reason ∈ {capacity, quota_link,
        quota_port} 分列（C9 冻结枚举）——capacity = SH 既有容量等待，
        quota_link/quota_port = 配额判据分列；等待计入从请求到达到完成
        的延迟（deferred 驻留时长另入 run 末指标行）。"""
        wait_reason = (
            WAIT_QUOTA_LINK if failure_class == "quota_deferred_quota_link"
            else WAIT_QUOTA_PORT
            if failure_class == "quota_deferred_quota_port"
            else WAIT_CAPACITY)
        state = self._admission_failure_state.get(runtime.request_id)
        if state is not None and state["class"] == failure_class:
            state["count"] += 1
            self.log_decision(
                {"kind": "joint_admission_wait",
                 "request_id": runtime.request_id, "priority": 0},
                now_ns,
                decision={
                    "failure_class": failure_class,
                    "wait_reason": wait_reason,
                    "attempt_count": state["count"],
                    "reason": reason,
                },
            )
            return
        self._admission_failure_state[runtime.request_id] = {
            "class": failure_class, "count": 1}
        session_view = self._joint_session_view(runtime.session_id)
        structural, detail = self._classify_physical_feasibility(
            runtime, session_view)
        self.log_decision(
            {"kind": "joint_admission_failed",
             "request_id": runtime.request_id, "priority": 0},
            now_ns,
            decision={
                "failure_class": failure_class,
                "wait_reason": wait_reason,
                "reason": reason,
                "structural_infeasible": structural,
                "feasibility_detail": detail[:16],
                "chosen_instance": record.chosen.instance_index,
                "chosen_action": record.chosen.action,
                "candidates": [
                    {
                        "instance_index": candidate.instance_index,
                        "action": candidate.action,
                        "applicable": candidate.applicable,
                        "cost_ns": candidate.cost_ns,
                        "inapplicable_reason":
                            candidate.inapplicable_reason,
                        # M6（kimi 复审）：notes 随候选落决策日志——
                        # R3'.3 的 deep_gap_unresolved 等注记须可检索
                        # （§9 验收"可检索"口径），此前只进内存不落盘。
                        # notes 载于 breakdown（不适用候选无 breakdown）。
                        "notes": list(candidate.breakdown.notes)
                        if candidate.breakdown is not None else [],
                    }
                    for candidate in record.candidates
                ],
            },
        )
        if structural:
            # P5-a：全部 (instance, action) 组合物理不可行——结构性
            # fail-closed（设计方案 §3.3-7：显式报告带逐 rank 缺口）。
            final_shards = kv_cache_shard_bytes_for_tokens(
                self.model,
                session_view.history_tokens
                + runtime.joint_input_tokens,
                self.kv_manager.tp_degree)
            gaps = []
            for instance in self.topology.instances:
                for rank, shard_bytes in zip(instance.ranks, final_shards):
                    capacity = (
                        self.kv_manager._rank_states[rank].capacity_bytes
                        - self.kv_manager._rank_states[rank].model_weight_bytes)
                    if shard_bytes > capacity:
                        gaps.append(
                            "rank {}: needs {} > free {}".format(
                                rank, shard_bytes, capacity))
            raise KVPhysicalInfeasibleError(
                "request {} final KV fits no (instance, action) "
                "combination; per-rank gaps: {}".format(
                    runtime.request_id,
                    "; ".join(gaps) if gaps else "aggregate capacity"))
    def _emit_admission(self, runtime, tick: int) -> None:
        """准入动作发射 + 决策/账本记录（拼 batch 改造，2026-08-22：
        PREFILL_DRAIN watch 不再在此注册——移至覆盖其最后 chunk 的列车
        drain 标记；原 _emit_prefill 的整段发射与 watch 部分删除）。"""
        result = self.graph.emit_admission_batch(runtime.plan_dict())
        for watch in result.get("eviction_watches", ()):
            owner_request_id = watch["owner_request_id"]
            if owner_request_id != runtime.request_id:
                raise RuntimeError(
                    "admission eviction watch changed request owner")
            flow_owner = None
            quota_owner = None
            if watch.get("branch") == "admission_history":
                flow_owner = runtime.request_id + "#evict"
                if (self._quota_tracker is not None
                        and flow_owner in self._quota_enrolled):
                    quota_owner = flow_owner
            elif watch.get("branch") == "admission_prefill":
                flow_owner = runtime.request_id + "#evict#prefill"
                if runtime.prefill_evictions:
                    self._register_transfer_flows(
                        runtime.prefill_evictions, owner=flow_owner)
                    quota_owner = self._quota_enroll_eviction_branch(
                        flow_owner + "#quota", runtime.request_id,
                        runtime.prefill_evictions, tick)
                    flow_owner = flow_owner
            else:
                raise RuntimeError(
                    "unexpected admission eviction watch branch {!r}".format(
                        watch.get("branch")))
            self._register_eviction_watch(
                watch,
                flow_owners=((flow_owner,) if flow_owner is not None else ()),
                quota_owners=((quota_owner,)
                              if quota_owner is not None else ()),
            )
        self._note_emitted(runtime.request_id, STAGE_PREFILL)
        self._ledger_issue(runtime.request_id, tick, STAGE_PREFILL,
                           runtime.prefill_instance_index)
        self._batch["assignments"].append({
            "request_id": runtime.request_id,
            "prefill_instance_index": runtime.prefill_instance_index,
            # prefill 边界 decode 尚未选定（decode 固定 prefill 同实例，
            # 红线 #4：PREFILL_DRAIN 时 selected = state_index）；assignment
            # 结构校验要求非负，填 prefill 实例（与最终 decode 实例恒等）。
            "decode_instance_index": (
                runtime.decode_instance_index
                if runtime.decode_instance_index is not None
                else runtime.prefill_instance_index),
            "prefill_affinity_reason": runtime.prefill_affinity_reason,
        })
        self.log_decision(
            {"kind": "prefill", "request_id": runtime.request_id,
             "priority": 0},
            tick,
            decision={
                "prefill_instance_index": runtime.prefill_instance_index,
                "prefill_assignment_key": list(
                    runtime.prefill_assignment_key),
                "prefill_affinity_reason": runtime.prefill_affinity_reason,
                "estimated_arrival_ns": runtime.estimated_arrival_ns,
                "admission_time_ns": runtime.admission_time_ns,
                "hbm_wait_ns": runtime.hbm_wait_ns,
                "history_tokens_before": runtime.history_tokens_before,
                "prefill_context_tokens": runtime.prefill_context_tokens,
                "history_source_instance_index":
                    runtime.history_source_instance_index,
                "history_transfer_bytes": runtime.history_transfer_bytes,
                "history_tokens_discarded":
                    runtime.history_tokens_discarded,
                # 问题 4a(S3 路由序列化,2026-09-05):对齐 S1
                # (sh10_online_scheduler.py:1140-1142)补齐 holder 的
                # shard 级序列化——_transfer_summary 只读导出 shards[].
                # noc_hops/noc_path(创建时由 deterministic_xy_route
                # 算好,零新算);既有聚合 history_transfer_bytes 保留。
                # A/B 剥离清单字段;hopbytes.py collect_sh30 读取端已
                # 前向兼容(旧 log 无新键 → 数值不变)。
                "history_transfer": (
                    _transfer_summary(runtime.history_transfer)
                    if runtime.history_transfer else None),
                # R12（2026-09-14）：水印重放的 joint 专属字段——动作/全
                # 部历史迁移/重算口径（多动作多区间传输无法由单数字段
                # reconstruct）。
                "joint_action": runtime.joint_action,
                "joint_span_base_context": runtime.joint_span_base_context,
                "joint_prefill_work": runtime.joint_prefill_work,
                # 规格书§二.3（2026-09-25）：prefill remote-read 前缀读流
                # 独立披露（与 history_transfers 分列——前缀读流不进
                # history_transfer_bytes/readiness barrier/merge 账本）。
                "prefill_remote_read_bytes":
                    runtime.prefill_remote_read_bytes,
                "prefill_remote_read_layers": (
                    runtime.prefill_remote_read_plan["read_prefix_layers"]
                    if runtime.prefill_remote_read_plan is not None
                    else 0),
                "prefill_remote_read_transfers": [
                    _transfer_summary(transfer)
                    for transfer in runtime.prefill_remote_read_transfers],
                "history_transfers": [
                    _transfer_summary(transfer)
                    for transfer in runtime.history_transfers],
                "history_evictions": [
                    _transfer_summary(transfer)
                    for transfer in runtime.history_evictions],
                # R17-1a'：结构性恒空死通道（准入 R1' 预约覆盖全动作足迹，
                # drain expand gap≡0；2026-09-16 三方裁决 + R17-7 探针
                # A2）。保留空字段仅为 schema 稳定；不在此复制"字段存在≠
                # 字段被填"反模式——容量逐出的披露走 kind=kv_eviction
                # 咽喉点行（_emit_eviction_only_nodes）。
                "prefill_evictions": [
                    _transfer_summary(transfer)
                    for transfer in runtime.prefill_evictions],
            },
        )

    def _emit_join_decision(self, runtime, tick: int) -> None:
        """drain 边界的 decode 决策记录（拼 batch 改造：decode 段发射
        移至加入列车，即 _plan_and_emit_trains → emit_iteration_train；
        DECODE_COMPLETION watch 由列车 exit 标记承载）。"""
        self._batch["assignments"].append({
            "request_id": runtime.request_id,
            "prefill_instance_index": runtime.prefill_instance_index,
            "decode_instance_index": runtime.decode_instance_index,
            "prefill_affinity_reason": runtime.prefill_affinity_reason,
        })
        self.log_decision(
            {"kind": "decode", "request_id": runtime.request_id,
             "priority": 0},
            tick,
            decision={
                "decode_instance_index": runtime.decode_instance_index,
                # 问题 4a(S3 路由序列化,2026-09-05):对齐 S1
                # (sh10_online_scheduler.py:1187-1188)补齐 prefill→decode
                # 迁移与 decode 侧逐出的 shard 级序列化(_transfer_summary
                # 同上,零新算)。核销置空沿用既有 completion M4 块
                # (:1362-1367),不新增提前置空——每请求恰一条 decode 行,
                # completion 行已存在,无双计。A/B 剥离清单字段。
                "prefill_decode_transfer": (
                    _transfer_summary(runtime.prefill_decode_transfer)
                    if runtime.prefill_decode_transfer else None),
                # remote-read credit（唯一执行口径）：drain 边
                # 界的持久读计划摘要（非 remote-read 恒 None——行 schema 对
                # credit 开关前向兼容）；逐列车切片摘要在 completion 行
                # 披露（每请求恰一处聚合，无逐列车行膨胀）。
                "remote_read_credit_plan": (
                    {
                        "home_instance": plan["home_instance"],
                        "steps": plan["steps"],
                        "context_per_step": plan["context_per_step"],
                        "total_bytes": plan["total_bytes"],
                    }
                    if (plan := runtime.remote_read_credit_plan) is not None
                    else None),
                "joint_action": runtime.joint_action,
                "decode_evictions": [
                    _transfer_summary(transfer)
                    for transfer in runtime.decode_evictions],
            },
        )

    # ------------------------------------------------------------- 负载 --

    def _prefill_chunk_task_load_ns(self, *, instance_size, chunk_tokens,
                                    context_tokens) -> int:
        """offline: face_scheduler.py（缓存同款）。"""
        key = (instance_size, chunk_tokens, context_tokens)
        return _bounded_task_load_memo(
            self._prefill_task_cache, key,
            lambda: estimate_prefill_task_load_ns(
                self.hardware, self.model,
                instance_size=instance_size, chunk_tokens=chunk_tokens,
                context_tokens=context_tokens),
            # F6 销账：__init__ 恒设 _task_load_cache_capacity——直接
            # 访问（替身漏设 = AttributeError）。
            capacity=self._task_load_cache_capacity,
        )

    def _joint_prefill_total_load_ns(self, input_tokens: int,
                                     history_tokens: int) -> int:
        """N4：prefill 整段负载（生产同形）——p_chunk 切分 + 累计
        context 逐 chunk roofline 求和，与 _plan_train span 冻结式
        (:1080-1088) 及核销逐键求和 (:963-977) 同值同 memo（
        _prefill_chunk_task_load_cache 命中）。JCM prefill_task_load_
        ns_fn 注入点；recompute 调用方传**语义值** history（O1/O10-e：
        JCM 内部已按 _resident_here 同判据二分——驻留目标传
        session.history_tokens、异地/REMOTE 传 0，调用方不再保证
        history=0 的副本 0 基语义）。
        """
        instance_size = self.topology.instances[0].size
        total = 0
        completed = 0
        while completed < input_tokens:
            chunk_tokens = min(self.p_chunk, input_tokens - completed)
            total += self._prefill_chunk_task_load_ns(
                instance_size=instance_size, chunk_tokens=chunk_tokens,
                context_tokens=history_tokens + completed + chunk_tokens)
            completed += chunk_tokens
        return total

    def _decode_task_load_ns_cached(self, *, instance_size,
                                    current_context_tokens, generated_tokens,
                                    average_decode_length,
                                    running_step_fraction_remaining):
        """改法A：estimate_decode_remaining_task_load_ns 的全参 key memo
        （与 _prefill_task_cache 同款）。hardware/model 为运行期不可变量，
        经绑定不入 key。"""
        key = (instance_size, current_context_tokens, generated_tokens,
               average_decode_length, running_step_fraction_remaining)
        return _bounded_task_load_memo(
            self._decode_task_load_cache, key,
            lambda: estimate_decode_remaining_task_load_ns(
                self.hardware, self.model,
                instance_size=instance_size,
                current_context_tokens=current_context_tokens,
                generated_tokens=generated_tokens,
                average_decode_length=average_decode_length,
                running_step_fraction_remaining=running_step_fraction_remaining),
            # F6 销账：__init__ 恒设 _task_load_cache_capacity——直接
            # 访问（替身漏设 = AttributeError）。
            capacity=self._task_load_cache_capacity,
        )

    def _task_load_snapshot(self, state, now_ns: int):
        """offline: face_scheduler.py task_load_snapshot 的在线
        子集。三分量口径（合同⑥），每个请求恰计一次（打分公式与阈值
        不动，红线；拼 batch 改造 2026-08-22 仅重订物理折算）：
          - queued_prefill：逐 chunk Roofline 求和（函数逐行复用
            :3612-3638；自 prefill_tokens_completed 闭式推进的剩余 chunk
            折算——列车核销时推进，决策边界上与逐 chunk 精确值逐点一致）；
          - running_prefill：在飞列车冻结的队列头 chunk 负载（列车账本
            prefill_chunk_spans；原"在飞段全量剩余 × fraction=1.0"的
            不可分假设作废，改为列车级冻结量）；
          - active_decode：estimate_decode_remaining_task_load_ns 逐请求
            （N12 在线化后 average_decode_length = CausalHorizonEstimator
            因果估计（session→run→冷启动 1 token），非全 trace 常数；
            generated_tokens =
            decode_tokens_consumed 闭式迭代级剩余量、current_context_
            tokens = prefill_ctx + consumed——原 generated=0/fraction=1.0
            的"active 段不可分"假设作废；当前在飞迭代仍整计一次
            fraction=1.0，列车粒度下属保守方向）。

        改法S2（2026-08-23，快照纪元缓存 + qp 聚合账本）：
          - S2-A 快照纪元缓存：实例账本（qp/active_decode 成员、成员
            进度字段、in_flight_train、last_arrival_ns）纪元未变则复用
            上次快照（InstanceTaskLoadSnapshot 为冻结 dataclass，复用
            引用零风险；now_ns 非取值输入——离线蓝本签名保留，不入缓存
            条件）。调用方（_try_admit_request 每 admit 尝试对全部实例
            求快照）在同一批内的重复求值自此走 O(1) 缓存命中。
          - S2-B qp 聚合账本：queued 分量的"逐请求逐 chunk while 现算"
            改为成员聚合账本之和（准入置全量/列车核销按冻结 span 精确
            扣减/drain 清零），实际计算体见 _compute_task_load_snapshot。
          - 影子断言（SH_SNAPSHOT_VERIFY=1）：缓存路径与聚合路径都仍按
            _reference_task_load_snapshot 的原始逐 chunk 现算完整重算并
            逐字段断言相等（零收益、全检查）；生产（不设或非"1"）走缓存。
        """
        if (state.snapshot_cache is not None
                and state.snapshot_epoch == state.ledger_epoch
                and state.snapshot_horizon_version
                == self._joint_horizon.version
                and not self._snapshot_verify):
            return state.snapshot_cache
        snapshot = self._compute_task_load_snapshot(state)
        if self._snapshot_verify:
            reference = self._reference_task_load_snapshot(state)
            if snapshot != reference:
                raise RuntimeError(
                    "task-load snapshot aggregate ledger diverged from the "
                    "reference while-loop computation for instance {} "
                    "(epoch {})".format(state.index, state.ledger_epoch))
            if (state.snapshot_cache is not None
                    and state.snapshot_epoch == state.ledger_epoch
                    and state.snapshot_horizon_version
                    == self._joint_horizon.version):
                if state.snapshot_cache != snapshot:
                    raise RuntimeError(
                        "task-load snapshot epoch cache diverged for "
                        "instance {} (epoch {})".format(
                            state.index, state.ledger_epoch))
                return state.snapshot_cache
        state.snapshot_cache = snapshot
        state.snapshot_epoch = state.ledger_epoch
        state.snapshot_horizon_version = self._joint_horizon.version
        return snapshot

    def _compute_task_load_snapshot(self, state):
        """_task_load_snapshot 的实际计算体（改法S2 后形态；仅缓存未命中
        或影子验证时执行）。三分量口径见 _task_load_snapshot docstring。

        S2-B 整数精确性证明（红线：折算公式一个数都不能变）：三分量
        全部是 Python int 的加減——estimate_prefill_task_load_ns 与
        estimate_decode_remaining_task_load_ns 的返回值都是 math.ceil
        产生的 int（face_scheduler.py:604/:688），int 加減精确、满足
        结合/交换律且无舍入，故聚合账本（入队置全量 + 核销扣减）与
        逐项 while 现算在数学上逐位相等，不存在浮点求和序漂移：
          - queued_prefill = Σ_qp 成员聚合账本 − 在飞列车冻结 chunk 负载
            （成员账本 = 自当前 prefill_tokens_completed 起的剩余 chunk
            逐键求和，由 _queued_chunk_load_full_ns 与列车核销增量维护）；
          - running_prefill / active_decode 与原始计算体相同（span 有界
            ≤ T_max、成员有界，memo 查找非热点，S2 未动）。"""
        instance_size = self.topology.instance(state.index).size
        # queued_prefill（offline: :3612-3638 的增量等价：成员聚合账本
        # 之和）+ 在飞列车 chunk 负载分离（原"queued 全量再减 running"
        # 的恰计一次口径不变）。
        queued_load_ns = 0
        for runtime in state.qp:
            queued_load_ns += runtime.queued_chunk_load_ns
        # running_prefill：在飞列车的冻结 chunk 负载（含在 qp 头部的
        # 剩余量中，恰计一次——自 queued 扣除）。
        running_prefill_load_ns = 0
        train = state.in_flight_train
        if train is not None:
            for chunk_tokens, context_tokens in train["prefill_chunk_spans"]:
                running_prefill_load_ns += self._prefill_chunk_task_load_ns(
                    instance_size=instance_size,
                    chunk_tokens=chunk_tokens,
                    context_tokens=context_tokens)
            queued_load_ns -= running_prefill_load_ns
        # active_decode（offline: :3665-3684；迭代级闭式剩余量折算）。
        # N12（R15-4，2026-09-14）：decode 剩余标定在线化——逐成员用
        # CausalHorizonEstimator（session 均值 → run 均值 → 冷启动 1），
        # 全 trace decode 均值常数不再进入负载视图（决策输入零未来
        # 信息，总纲 §13.1/§13.2；J-off load-first 臂同口径）。
        active_decode_load_ns = 0
        for runtime in state.active_decode:
            estimated_decode, _source = self._joint_horizon.estimate(
                runtime.session_id)
            active_decode_load_ns += self._decode_task_load_ns_cached(
                instance_size=instance_size,
                current_context_tokens=(
                    runtime.prefill_context_tokens
                    + runtime.decode_tokens_consumed),
                generated_tokens=runtime.decode_tokens_consumed,
                average_decode_length=float(estimated_decode),
                running_step_fraction_remaining=1.0,
            )
        from face_scheduler import InstanceTaskLoadSnapshot
        return InstanceTaskLoadSnapshot(
            instance_index=state.index,
            running_prefill_task_load_ns=running_prefill_load_ns,
            queued_prefill_task_load_ns=queued_load_ns,
            active_decode_task_load_ns=active_decode_load_ns,
            last_arrival_ns=state.last_arrival_ns,
        )

    def _reference_task_load_snapshot(self, state):
        """改法S2 影子参照（SH_SNAPSHOT_VERIFY=1 专用）：S2 之前的原始
        _task_load_snapshot 计算体逐行保留（qp 逐请求逐 chunk while 现算，
        含改法A 的 memo 读法）——生产路径零调用。"""
        instance_size = self.topology.instance(state.index).size
        # queued_prefill（offline: :3612-3638 逐行复用；剩余 chunk 自
        # prefill_tokens_completed 折算）+ 在飞列车 chunk 负载分离。
        queued_load_ns = 0
        for runtime in state.qp:
            remaining_tokens = (
                runtime.prefill_tokens_to_process
                - runtime.prefill_tokens_completed)
            processed = runtime.prefill_tokens_completed
            while remaining_tokens > 0:
                chunk_tokens = min(self.p_chunk, remaining_tokens)
                context_tokens = (
                    runtime.joint_span_base_context
                    + processed + chunk_tokens)
                queued_load_ns += self._prefill_chunk_task_load_ns(
                    instance_size=instance_size, chunk_tokens=chunk_tokens,
                    context_tokens=context_tokens)
                processed += chunk_tokens
                remaining_tokens -= chunk_tokens
        # running_prefill：在飞列车的冻结 chunk 负载（含在 qp 头部的
        # 剩余量中，恰计一次——自 queued 扣除）。
        running_prefill_load_ns = 0
        train = state.in_flight_train
        if train is not None:
            for chunk_tokens, context_tokens in train["prefill_chunk_spans"]:
                running_prefill_load_ns += self._prefill_chunk_task_load_ns(
                    instance_size=instance_size,
                    chunk_tokens=chunk_tokens,
                    context_tokens=context_tokens)
            queued_load_ns -= running_prefill_load_ns
        # active_decode（offline: :3665-3684；迭代级闭式剩余量折算）。
        # N12（R15-4，2026-09-14）：decode 剩余标定在线化——逐成员用
        # CausalHorizonEstimator（session 均值 → run 均值 → 冷启动 1），
        # 全 trace decode 均值常数不再进入负载视图（决策输入零未来
        # 信息，总纲 §13.1/§13.2；J-off load-first 臂同口径）。
        active_decode_load_ns = 0
        for runtime in state.active_decode:
            estimated_decode, _source = self._joint_horizon.estimate(
                runtime.session_id)
            active_decode_load_ns += self._decode_task_load_ns_cached(
                instance_size=instance_size,
                current_context_tokens=(
                    runtime.prefill_context_tokens
                    + runtime.decode_tokens_consumed),
                generated_tokens=runtime.decode_tokens_consumed,
                average_decode_length=float(estimated_decode),
                running_step_fraction_remaining=1.0,
            )
        from face_scheduler import InstanceTaskLoadSnapshot
        return InstanceTaskLoadSnapshot(
            instance_index=state.index,
            running_prefill_task_load_ns=running_prefill_load_ns,
            queued_prefill_task_load_ns=queued_load_ns,
            active_decode_task_load_ns=active_decode_load_ns,
            last_arrival_ns=state.last_arrival_ns,
        )

    def _queued_chunk_load_full_ns(self, *, instance_size, runtime) -> int:
        """改法S2-B：成员聚合账本的一次性全量计算（准入时点调用）——与
        _reference_task_load_snapshot 的 queued while 现算对同一请求自其
        当前 prefill_tokens_completed 起逐 chunk 同键同值。此后仅由列车
        核销按已完 chunk 的冻结负载精确扣减（_finalize_completed_trains）、
        drain 出队清零（余量必 0，fail-closed 断言）。"""
        remaining_tokens = (
            runtime.prefill_tokens_to_process
            - runtime.prefill_tokens_completed)
        processed = runtime.prefill_tokens_completed
        total_ns = 0
        while remaining_tokens > 0:
            chunk_tokens = min(self.p_chunk, remaining_tokens)
            context_tokens = (
                runtime.joint_span_base_context
                + processed + chunk_tokens)
            total_ns += self._prefill_chunk_task_load_ns(
                instance_size=instance_size, chunk_tokens=chunk_tokens,
                context_tokens=context_tokens)
            processed += chunk_tokens
            remaining_tokens -= chunk_tokens
        return total_ns

    def _sensing_ready_view(self) -> dict:
        """§7.3/§10.1 ready 层边界视图（查询/审计输入，不进判据）。"""
        detail = []
        for instance_index in sorted(self._ready_frontier):
            state = self.instances[instance_index]
            for runtime in state.qp:
                detail.append({
                    "request_id": runtime.request_id,
                    "stage": STAGE_PREFILL,
                    "instance_index": instance_index,
                    "admitted_tick": self.ledger_admitted.get(
                        runtime.request_id, {}).get("admitted_tick"),
                })
            for runtime in state.active_decode:
                detail.append({
                    "request_id": runtime.request_id,
                    "stage": STAGE_DECODE,
                    "instance_index": instance_index,
                    "admitted_tick": self.ledger_admitted.get(
                        runtime.request_id, {}).get("admitted_tick"),
                })
        return {"ready_count": len(detail), "detail": detail}

    # --------------------------------------------------------------- 收尾 --

    def _telemetry_coverage_decision(self) -> dict:
        """决策日志遥测块（kind=link_telemetry_coverage 行的 decision
        载荷；verify_run_end 落行，本 helper 为单一构造点）。

        A10'(a)/A12'（2026-09-22，§4.3 补遗）：新增
        ``telemetry_zero_rate_samples_dropped``——served_bytes==0 ∧
        active_ns>0 有效速率零窗口样本（有活动无载荷字节，可出自 C++
        整字节进位短窗）的丢弃计数（ingest 处丢弃、不写入 rates，计数
        在此披露；真零速率持续场景由 A9' 缺席失效兜底）。

        O7③（2026-09-23 终轮审计）：遥测降级三键披露
        （``telemetry_degradation`` 块）——修前 JCM disclosure 的降级
        可辨认键（flow_count_links / legacy_rate_divisor_links）只存
        在于逐决策重建的 JCM 实例内，run 级覆盖度裁决零生产落点；此处
        按 SH 账本同口径镜像三键（全部派生自现存状态，零新增账本）：
        旧二进制回退（速率在场 ∧ 流数缺席的链路数）/ 仅流数条目
        （流数在场 ∧ 速率缺席，N3 零速率窗 carry 的 collective 在场
        证据可见性）/ 零速率窗丢弃计数（降级形态发生计数）。
        """
        rate_keys = set(self._link_telemetry_rates)
        flow_keys = set(self._link_telemetry_flow_counts)
        return {
            "telemetry_epoch_count": self._link_telemetry_epoch_count,
            "telemetry_sample_count": (
                self._link_telemetry_sample_count),
            "telemetry_key_absent_seen": self._telemetry_absent_seen,
            "telemetry_window_broken": self._telemetry_window_broken,
            "telemetry_rate_entries": len(self._link_telemetry_rates),
            # A10'(a)：零速率窗口样本丢弃披露（0 = 本 run 无此形态或
            # 遥测未开）。
            "telemetry_zero_rate_samples_dropped": (
                self._telemetry_zero_rate_dropped),
            "collective_coverage": (
                self._joint_flows.collective_coverage),
            # O7③：遥测降级三键（见 docstring）。
            "telemetry_degradation": {
                "legacy_rate_divisor_entries": len(
                    rate_keys - flow_keys),
                "flow_count_only_entries": len(flow_keys - rate_keys),
                "zero_rate_samples_dropped": (
                    self._telemetry_zero_rate_dropped),
            },
        }

    def verify_run_end(self) -> None:
        """基类协议校验之上，叠加离线 :4064-4076 的收尾断言 + §7.3 结束
        审计（arrival heap / ready frontier / 列车账本全空）。"""
        super().verify_run_end()
        # C8：遥测完备性收尾披露行（kind=link_telemetry_coverage）——
        # collective_coverage 翻转条件的全量证据。joint_mechanism_
        # manifest.json 侧车落字在 online_service/joint_config（本卡文件
        # 清单外），决策日志收尾行为本卡完备性档案，侧车接线归 C11 车道。
        self.log_decision(
            {"kind": "link_telemetry_coverage", "request_id": "",
             "priority": 0},
            self._telemetry_last_tick_ns,
            decision=self._telemetry_coverage_decision())
        # C11 步骤 4/5（指标埋点收尾行）：决策时延 + 分档动作分布 +
        # quota 等待分列/驻留时长 + AIMD 动作计数——为 C20 预置，本卡
        # 只埋点不判读；remote-read 选中率按负载档由决策行 load_view
        # 侧导出（不在此预聚合档位）。F6 销账：__init__ 恒设埋点账本
        # ——直接访问（替身漏设 = AttributeError，不再静默跳行）。
        dwell = sorted(self._quota_deferred_dwell_ns)
        dwell_summary = {"count": len(dwell)}
        if dwell:
            dwell_summary.update({
                "min_ns": dwell[0],
                "p50_ns": dwell[len(dwell) // 2],
                "p90_ns": dwell[min(
                    len(dwell) - 1, (len(dwell) * 9) // 10)],
                "max_ns": dwell[-1],
            })
        self.log_decision(
            {"kind": "joint_decision_metrics", "request_id": "",
             "priority": 0},
            self._telemetry_last_tick_ns,
            decision={
                "quota_mode": self.joint_config.quota_mode,
                "action_selection_counts": dict(
                    self._joint_action_selection_counts),
                "quota_deferred_wait_counts": dict(
                    self._quota_deferred_wait_counts),
                "quota_deferred_dwell": dwell_summary,
                "admission_decision_wall_ns": {
                    "count": self._admission_decision_count,
                    "total_ns": self._admission_decision_wall_ns_total,
                    "avg_ns": (
                        self._admission_decision_wall_ns_total
                        // max(1, self._admission_decision_count)),
                    "max_ns": self._admission_decision_wall_ns_max,
                },
                "quota_verdict_layer": {
                    "wall_ns_total": self._quota_verdict_wall_ns_total,
                    "candidate_checks": (
                        self._quota_verdict_candidate_checks),
                },
                "quota_events": {
                    "admits": self._quota_admit_events,
                    "releases": self._quota_release_events,
                    "merge_reserves_created": (
                        self._quota_merge_reserves_created),
                    # O12：oneshot 入册披露事件（decode 相深占用降级 +
                    # O2 准入逐出支链降级）计入 run 级配额指标——修前
                    # 深占用记账失明（事件落决策日志但不进汇总）。
                    "oneshot_overflows": (
                        self._quota_oneshot_overflow_events),
                },
                "aimd_action_counts": dict(
                    self._quota_aimd_action_counts or {}),
            },
        )
        # C11 守恒审计（验收：配额借还进 ledger，全 PASS）：收尾仍在册
        # 的流/预留 = 借还失配（漏释放/双释放由模块 fail-closed 守卫、
        # 此处兜底生命周期断链），fail-closed。__init__ 恒设 _quota_tracker
        # （off = None / static|aimd = tracker——生产条件语义，F6 只删
        # 属性存在性软门，is not None 判据保留）。
        if self._quota_tracker is not None:
            if self._quota_enrolled:
                raise RuntimeError(
                    "quota flows leaked past settlement (lifecycle chain "
                    "broken): {}".format(sorted(self._quota_enrolled)[:5]))
            if self._quota_merge_reserves:
                raise RuntimeError(
                    "quota merge reserves leaked past merge-done "
                    "settlement: {}".format(
                        sorted(self._quota_merge_reserves)[:5]))
            # O10①（2026-09-23 终轮审计）：tracker 内部账本空账直审——
            # SH 侧 _quota_enrolled/_quota_merge_reserves 之上补影子翼
            # （双翼守恒：module 内部账本破损不经过 SH 侧登记时在此被
            # 捉）。occ/res/enroll/bulk/merge_borrowed 全零；merges 经
            # res/bulk 间接归零（reserve_merge 恒持双向链路预留 + 双端
            # 口 bulk，预留存活 ⇒ res/bulk 非零必被捉）。
            self._assert_quota_tracker_ledgers_clean()
        # C8/规格书§二.5 收尾审计：remote-read 读流 owner 泄漏（#readplan
        # 预登记 est 账本未对账核销 / #prefill_read 未在 prefill drain
        # 释放，任一环破损即在此报警——失败注入单测的断言位）。置于全
        # owner 收口之前，使泄漏诊断携带生命周期语义（通用审计兜底）。
        self._assert_no_readplan_leaks()
        # O10③：R15 在途流注册表（链路/池端口/HBM 端口）全 owner 清账
        # 断言——remote-read 读流半边由上方 _assert_no_readplan_leaks
        # 先行审计，此处覆盖其余全部 owner（R15 盲区的收口）。
        self._assert_no_flow_registry_leaks()
        if self.pending_admissions:
            pending_ids = [
                runtime.request_id for runtime in self.pending_admissions]
            raise RuntimeError(
                "strategy run ended with blocked HBM admissions: "
                "{}".format(pending_ids[:5]))
        if self.completed_requests != self.expected_request_count:
            raise RuntimeError(
                "strategy run ended with {}/{} requests complete".format(
                    self.completed_requests, self.expected_request_count))
        if self.runtime_by_request_id or self._runtime_index or \
                any(runtime is not None for runtime in self.runtimes):
            raise RuntimeError("strategy run ended with unreleased runtimes")
        if any(state.qp or state.active_decode
               or state.pending_decode_ready
               or state.in_flight_train is not None
               or state.finalized_trains
               for state in self.instances):
            raise RuntimeError(
                "strategy run ended with non-idle instance state")
        # WP9：全部首步拆分必须已交付余量批、唤醒信号已核销。
        if self._pending_first_steps:
            raise RuntimeError(
                "online run ended with undelivered first-step wakeups: "
                "{}".format(sorted(self._pending_first_steps)[:5]))
        if any(state.first_step_remainder is not None
               for state in self.instances):
            raise RuntimeError(
                "online run ended with an undelivered train remainder")
        if self.arrival_heap:
            raise RuntimeError(
                "run ended with {} unconsumed arrival events".format(
                    len(self.arrival_heap)))
        if self._ready_frontier:
            raise RuntimeError(
                "run ended with non-empty ready frontier: {!r}".format(
                    sorted(self._ready_frontier)))
        if self._pending_merge_alarms:
            # merge 尾 watch 必须在 EOF 前交付（标记是真实图节点，C++
            # 处理后必有交付）；残留即 watch 通道破损。
            raise RuntimeError(
                "run ended with undelivered merge-done watches: {}".format(
                    sorted(self._pending_merge_alarms)[:5]))
        if self._pending_eviction_watches:
            pending = sorted(
                (watch_id, entry["scheduled"])
                for watch_id, entry
                in self._pending_eviction_watches.items())
            raise RuntimeError(
                "run ended with pending/scheduled eviction handles: {}"
                .format(pending[:5]))
        if self._stalled_by_instance:
            # R14：停滞会话残留（EOF 边界——唤醒 pass 与死锁守卫的兜底
            # 报错；到这里的停滞 = 守卫条件外的事件源缺失，显式上报）。
            stalled_ids = sorted(
                runtime.request_id
                for stalled in self._stalled_by_instance.values()
                for runtime in stalled)
            raise RuntimeError(
                "run ended with stalled decode sessions: {}".format(
                    stalled_ids[:5]))
        if self._admit_attempt_epoch:
            raise RuntimeError(
                "strategy run ended with stale admit attempt epochs: "
                "{}".format(sorted(self._admit_attempt_epoch)[:5]))
        self.kv_manager.assert_final_state()


def _transfer_summary(transfer):
    """KVTransfer 的日志摘要（online_decision_log 行内嵌）。

    WP9 hopbytes 调查（2026-08-26）：在线 KV 迁移计划由 face_scheduler
    的 KVCacheManager 产出，每 shard 携带 deterministic_xy_route 的
    noc_path——路由信息在决策时点完全可知。摘要按 sh_1.0 同构补落
    shard 级路由字段（bytes/noc_hops/noc_path；noc_hops = len
    (noc_path)-1，slo_tools/hopbytes.py 优先读显式 noc_hops），只加
    字段不改路由；该字段进 ON/OFF 与 B0 基线对拍剥离清单。

    R17-1d（2026-09-17）：补序列化 KVTransfer 本就携带、此前被丢弃的
    resident_prefix_layers_before/after（逐出区间连锁不变量免重建——
    每条逐出自证前后驻留前缀）与 source_instance_index（victim 归位
    校验）。前缀字段语义按行 kind 分读（kimi C1）：eviction 条目 =
    primary 前缀迁移；merge_transfers 的 home_merge_base_degrade =
    base 前缀迁移；history_transfers = primary 前缀建立。读者忽略
    未知键，旧日志零行为差。"""
    if transfer is None:
        return None
    return {
        "kind": transfer.kind,
        "phase": transfer.phase,
        "reason": transfer.reason,
        "session_id": transfer.session_id,
        "total_bytes": transfer.total_bytes,
        "layer_start": transfer.layer_start,
        "layer_end": transfer.layer_end,
        "resident_prefix_layers_before":
            transfer.resident_prefix_layers_before,
        "resident_prefix_layers_after":
            transfer.resident_prefix_layers_after,
        "source_instance_index": transfer.source_instance_index,
        "shards": [
            {
                "source_rank": shard.source_rank,
                "target_rank": shard.target_rank,
                "edge_rank": shard.edge_rank,
                "bytes": shard.bytes,
                "noc_hops": len(shard.noc_path) - 1,
                "noc_path": list(shard.noc_path),
            }
            for shard in transfer.shards
        ],
    }
