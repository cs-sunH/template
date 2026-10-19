#!/usr/bin/env python3
"""sh30_online_scheduler.py -- sh_3.0 关感知策略调度器（strategy 模式，步骤 1-9）。

以已移除的离线 plan_face_requests（2026-08-21 离线 planner 清除批删去）为蓝本迁移，保持决策顺序
逐行对应（每处迁移用 `# offline:` 符号锚 注释标注）。离线事件
循环与在线边界的一一对应：

  离线事件循环                                   在线边界
  ----------------                               ----------------
  预置 turn-0 arrival 堆（:3574-3579）            ingress ARRIVAL 事件喂入同一
                                                 arrival heap（字段保持
                                                 (time_ns, priority, sequence,
                                                 kind, payload, runtime)
                                                 6 元组形状）
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
                                                 推事件），排队/配对
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
  - decode 固定 prefill 同实例（原红线 #4：selected = state_index）——
    **D18（2026-09-26，v2.0 §2/§16）废除**：P→D 边界二次决策落地，
    decode 实例/动作在真实边界快照上恰好一次全候选遍历重选（原地
    继续 x0 恒为候选且平局优先，式 1c）；状态机
    P_COMPUTE_DONE -> D_DECISION_PENDING -> D_COMMITTED，
    已知终止直接结算（_joint_boundary_terminal）；
  - **D19（2026-09-26，v2.0 §5.3/§6.5/§13.1）D 候选专用边界视图**：
    estimate_D_remaining 字节口径改由 joint/boundary_layout.py 的
    BoundaryKVLayout 逐目标区间推导（copy 只补目标缺少部分），
    等待接口拆分原地接续（不重
    排队——等待基 = 在服流 running+active）与新接纳（不免费插队——
    目标全部在账负载）；边界决策记录披露 stage_work/继承义务表/
    boundary_wait_kind（_joint_boundary_stage_view）。
  - task_load_snapshot 三分量：打分公式与阈值不动，仅物理折算随列车化
    重订（active 段"段在飞=全量剩余"假设作废 → decode_tokens_consumed
    闭式迭代级剩余量；在飞 chunk 负载 = 冻结列车账本）。

关感知口径（方案 §4.1 第 6 条 + 合同⑥）：策略输入全部来自 Python 账本——
task_load_snapshot 三分量（Roofline 估计服务时间；queued 分量逐行复用；running/active 分量的进度输入按列车闭式账本折算）、HBM 可行
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
from collections import deque, OrderedDict
from dataclasses import asdict as _dataclass_asdict
from dataclasses import replace as _dataclass_replace
from typing import Optional, Sequence

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
    KVStageVersionConflict,
    KVPhysicalInfeasibleError,
    build_instances,
    estimate_decode_remaining_task_load_ns,
    estimate_iteration_workload,
    estimate_prefill_workload,
    estimate_prefill_task_load_ns,
    kv_cache_shard_bytes_for_layer_range,
    kv_cache_shard_bytes_for_tokens,
)
from joint.joint_config import (  # noqa: E402
    ALL_ACTIONS,
    BOUNDARY_POLICY_OFF,
    BOUNDARY_POLICY_P_CONTINUATION,
    SCHEDULER_MODE_JOINT,
    parse_joint_config,
    parse_train_max_iter,
    MODEL_SEMANTICS_VERSION,
)
from joint.hbm_port_flow_registry import HbmPortFlowRegistry  # noqa: E402
from joint.boundary_layout import (  # noqa: E402
    BoundaryKvInterval,
    BoundaryFirstTokenState,
    BoundaryRuntimeStateEnvelope,
    boundary_wait_kind,
)
from joint.joint_cost_model import (  # noqa: E402
    ACTION_ORDER,
    ACTION_STAY,
    CausalHorizonEstimator,
    InstanceLoadView,
    JointCostModel,
    JointHardwareRates,
    LinkFlowRegistry,
    PDBoundaryView,
    RequestView,
    ServiceFactors,
    SessionKVView,
    fold_repeat_service_ns,
    synthesise_summary_from_reclaimable,
)
from joint.joint_scheduler import (  # noqa: E402
    select_decode_target_and_action,
    select_instance_and_action,
)
from joint.joint_config import (  # noqa: E402
    SCHEDULER_MODE_AFFINITY,
    SCHEDULER_MODE_LOAD_FIRST,
)
from joint.predictor_cost_accounting import (  # noqa: E402
    DECISION_STAGE_D,
    DECISION_STAGE_P,
    PredictorCostAccounting,
)
from joint.link_quota import (  # noqa: E402
    DELTA_ADM_INITIAL_NS,
    FLOW_ONESHOT,
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
from online.graph_batch_builder import first_token_split_enabled  # noqa: E402


_TASK_LOAD_CACHE_CAPACITY = 4096

# A2(2026-10-17, joint提速改造): 估值 memo 容量上限——LRU 逐出仅防御长跑
# 内存缓涨（实测每轮 36 唯一键 ≪ 4096，日常零逐出，命中语义不变）。
_ESTIMATE_MEMO_CAP = 4096

# ---- D18 单请求两阶段状态机（v2.0 §16.4，2026-09-26）→ W5 统一结算
# 状态机（§7.4，2026-09-29）：旧 MERGE_PENDING → MERGE_COMMITTED →
# DONE 抽象替换为 SETTLE_PENDING → SETTLE_COMMITTED → SETTLE_DONE；
# SERVICE_DONE 单独记录并触发闭环响应语义（响应完成锚定下一轮到达
# alarm，不隐含"必须先物理合并才能下一轮到达"，与 settle 状态分立）。
#
# P_DECISION -> P_COMMITTED -> P_RUNNING
#     -> P_COMPUTE_DONE
#         -> [已知终止] SETTLE_PENDING（直接结算，不选 D）
#         -> [还需 D] D_DECISION_PENDING
#             -> D_PLAN_SELECTED
#             -> D_COMMITTED -> D_RUNNING -> SETTLE_PENDING
#     -> SETTLE_COMMITTED（结算事务提交：consolidate 的物理 merge =
#        settle 子协议，账本已闭；物理载荷自此启动）
#     -> SETTLE_DONE（全部 settle 义务闭合：merge 载荷 ack 送达 /
#        零载荷即时刻）
#
# SERVICE_DONE（响应完成）= 单独记录的事实：completion 行 tick +
# ``service_done_ns`` 键 + 下一轮到达 alarm 锚（父请求 code-4 完成
# tick + interval,与指标层 resolve_arrival 同源同式——M16,2026-09-29;
# 同 tick 交付时即交付 tick + interval,future_alarms 语义不变）；
# 早于/晚于 SETTLE_DONE 均合法（到达后数据
# 消费受 merge_done 整体口径门控——图侧 interval gate 前递锚）。
#
# ``P_COMMITTED`` 冻结已提交 P 操作，不冻结未提交 D；``D_COMMITTED``
# 的界限 = 第一笔不可撤销的新阶段操作（预约换址/KV 迁移发射），此后
# 本轮 D 不再选点（无第三次决策）。强制 P/D 同址已废除：D 决策在真实
# P→D 边界以真实快照做恰好一次全候选遍历（式 1b/1c）。
JOINT_STAGE_P_DECISION = "P_DECISION"
JOINT_STAGE_P_COMMITTED = "P_COMMITTED"
JOINT_STAGE_P_RUNNING = "P_RUNNING"
JOINT_STAGE_P_COMPUTE_DONE = "P_COMPUTE_DONE"
JOINT_STAGE_SETTLE_PENDING = "SETTLE_PENDING"
JOINT_STAGE_D_DECISION_PENDING = "D_DECISION_PENDING"
JOINT_STAGE_D_PLAN_SELECTED = "D_PLAN_SELECTED"
JOINT_STAGE_D_COMMITTED = "D_COMMITTED"
JOINT_STAGE_D_RUNNING = "D_RUNNING"
JOINT_STAGE_SETTLE_COMMITTED = "SETTLE_COMMITTED"
JOINT_STAGE_SETTLE_DONE = "SETTLE_DONE"
#: W5（§7.4）：SERVICE_DONE 不是阶段机成员——单独记录的响应完成事实
#: （settlement 记录的 service_done_ns 位 + completion 行 service_done_ns
#: 披露键 + 闭环后继 alarm 锚定）。
JOINT_SERVICE_DONE = "SERVICE_DONE"
#: 结算策略（W9 起双值均为合法执行路径——A12 范围变化：merge-only 不
#: 能覆盖 distributed）：consolidate = merge 子协议，同次候选比较固定
#: 它；distributed = 目录提交型子协议（``KVCacheManager.settle_
#: distributed``：零物理归并/版本键防重/旧义务闭合/目录事务提交，
#: §7.4）。分派由 ``_joint_settlement_policy`` 单点裁决，未知策略
#: fail-closed 拒绝；W3-W5 时期的双侧显式拒绝（定价侧
#: ``_reject_distributed``/执行侧同门）已随 W9 退役，引用该旧拒绝
#: 机制的说明一并作废。
SETTLEMENT_POLICY_CONSOLIDATE = "consolidate"
SETTLEMENT_POLICY_DISTRIBUTED = "distributed"


class _VersionDrift(RuntimeError):
    """W5（§7.2 第 1 步，调度器侧）：决策→提交间 runtime 纪元（KV 账本
    纪元）漂移——与 face 侧 KVStageVersionConflict 同语义（回退继续方案，
    不沿用决策时点快照）。"""

#: C5（F8 冻结，2026-09-22）+ D17 扩展（2026-09-27，v2.0 §18/D17）+
#: §5.1/§5.6/§5.5 交付面扩展（2026-09-30 修复）：joint_admission /
#: D 边界决策日志的 ActionCostBreakdown 序列化字段序——全 31 字段与
#: joint_cost_model.ActionCostBreakdown 声明逐一同名（值由 C2 物化填
#: 充，本卡只冻结字段与序列化——值可为 None，字段存在性与可解析性为
#: 本卡判据）。D17 扩展 = 公式版本（model_semantics_version）、决策阶
#: 段/纪元、两阶段实例/动作配对、继续标记、剩余前向、预测里程碑
#: （predicted_* 前缀 = 预测字段，与真实时间严格分列）、区间布局/继
#: 承义务/分阶段空间/暂存峰值/近似状态。2026-09-30 扩展三键 = 复杂度
#: 披露（projection_stats：预测事件/资源/工作/闭包计数 + 场景与耗时，
#: §5.6）、候选反作用逐背景工作对照（background_delays，§5.1 "单独
#: 报告背景延迟"）、容量摘要 θ 预测结构化披露（capacity_theta_
#: prediction，§5.5 "记录摘要预测与实际 E 的差异"的预测半面——实际
#: 半面在 kv_eviction 行对账块）。决策日志可重建每候选的公式版本、
#: 输入来源、字节/空间与关键等待；候选未被实际执行时其真值为 NA
#: （决策行逐候选 measured 槽位恒 None，受控同状态执行另行填充，不
#: 用预测值冒充实测）。
_JOINT_BREAKDOWN_LOG_FIELDS = (
    "target_wait_ns", "history_prep_ns", "eviction_wait_ns", "compute_ns",
    "merge_ns", "contention_divisor", "hops", "notes",
    # D17 schema 扩展（joint_cost_model.ActionCostBreakdown 同名）。
    "model_semantics_version", "decision_stage", "decision_epoch",
    "prefill_instance", "decode_instance", "prefill_action",
    "decode_action", "is_continuation", "estimated_remaining_decode_steps",
    "predicted_next_token_done", "predicted_service_done",
    "predicted_merge_done", "interval_layout", "inherited_obligation_ids",
    "phase_space_bytes", "approximations",
    # §5.1/§5.6/§5.5 交付面扩展（2026-09-30）。
    "projection_stats", "background_delays", "capacity_theta_prediction",
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
                 "snapshot_horizon_version", "snapshot_length_version",
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
        # N14（2026-09-30，§8.2/A08）：active_decode 标定切条件剩余估计
        # 器（ConditionalRemainingLengthEstimator）——同为快照隐藏输入，
        # 版本一并入缓存键（observe_completed/observe_censored 更新不
        # bump 实例纪元）。
        self.snapshot_length_version = -1
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
        "drain_block_ends",
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
        # ---- R14（2026-09-14）：decode 增长停滞/唤醒 ----
        "decode_stalled", "stall_wake_key", "stall_reason", "stall_gap_records",
        # ---- C11（WP3c）：quota_deferred 首等待时标（驻留时长样本） ----
        "quota_deferred_since_ns",
        # ---- 逐出尾 watch（2026-09-24 修复）：decode joiner drain 登记 ----
        "decode_eviction_watch_id",
        # ---- D18（v2.0 §16.4）两阶段状态机与边界决策账本 ----
        "joint_stage", "joint_decode_action", "joint_decode_is_continuation",
        "boundary_decision",
        # ---- C4（v2.0 §19.1/D10/D19）：D 边界完整动作面的运行时元数据
        # ——首 token 状态（issued/consumed/rng_advances——首个 D 前向
        # 消费恰一次，二次消费 fail-closed）。----
        "joint_boundary_first_token",
        # ---- D20（v2.0 §14.5/§16.4）跨阶段预约事务与引用交接 ----
        # joint_boundary_epoch：P→D 边界阶段 epoch（每进入一次未提交
        # 边界 +1；同 epoch 重复通知/在途 ack 幂等，回退按 §16.4 记
        # aborted）；completion_instance_index = 完成归属（边界提交时
        # 锁定，完成信号按其归属实例核销）；joint_stage_reference_id =
        # 边界窗口内本请求基础 KV 的阶段引用句柄（P 末尾到 D 提交不进
        # 冷 victim 集合；D 目标预留/来源引用建立后释放）。
        "joint_boundary_epoch", "completion_instance_index",
        "joint_stage_reference_id",
        # ---- W5（W6 接线 §1）：到达时刻冻结的已揭示类型键（完成样本
        # category 的同源键——P/D 同刻同实例同一规则；None = 首请求/
        # 未揭示，回退链走 run/prior 层） ----
        "joint_revealed_category",
        # ---- W5（§7.2 所有来源引用）：D 提交点经目录 acquire_entry_
        # reference 获取的来源引用句柄清单（[(entry_id, exec_pin), ...]；
        # None = 无目录/无来源）。结算闭合时释放（守恒：泄漏审计在
        # verify_run_end 的目录审计）。
        "joint_source_reference_ids",
        # ---- §5.5（2026-09-30 修复）：容量摘要 θ 预测 vs 实际逐出对账
        # 簿记（随请求 runtime 生命周期——完成即回收，无调度器级账本）。
        # capacity_theta_prediction = 选中候选 breakdown 的 θ 结构化预测
        # （dict；None = 未暂存）；capacity_actual_eviction_bytes = 该请
        # 求触发的真实容量逐出累计字节（kv_eviction 行构造时累加）。----
        "capacity_theta_prediction", "capacity_actual_eviction_bytes",
    )

    def __init__(self, record: dict, p_chunk: int = 512) -> None:
        self.request_id = record["request_id"]
        self.session_id = record["session_id"]
        self.turn_index = record["turn_index"]
        self.queue_index = record["queue_index"]
        self.prefill_length = record["prefill_length"]
        self.decode_length = record["decode_length"]
        # R4-P2（2026-09-29 face 移植）：prefill_length<=0 的请求无 chunk
        # 工作（remaining_chunks 恒 0），列车计划器的 qp 头门（
        # natural_iterations <= 0 → RuntimeError）要等到该请求成为队头
        # 才崩，此前零报错滞留 qp（官方路径不可达，manifest 信任边界外
        # 构造期拒绝）。
        if self.prefill_length <= 0:
            raise ValueError(
                "request {!r} has non-positive prefill_length {!r}: a "
                "zero-work request would stall in the prefill queue"
                .format(self.request_id, self.prefill_length))
        self.history_tokens_before = record["history_tokens_before"]
        self.prefill_tokens_to_process = (
            record["prefill_context_tokens"] - record["history_tokens_before"])
        self.prefill_context_tokens = record["prefill_context_tokens"]
        self.final_context_tokens = record["final_context_tokens"]
        # joint：物理 prefill 工作量与 span 上下文基（stay/copy 共用
        # 缺省口径：工作量 = 本轮输入增量、span 基 = 真实历史基数）。
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
        # C11：quota_deferred 首次等待时标（None = 未经历配额等待）；
        # 最终准入时结算驻留时长样本入指标。
        self.quota_deferred_since_ns = None
        # K6：停滞携带的容量缺口记录（KVCapacityError.deep_gap_records）；
        # 停滞是可恢复路径不落账，仅在死锁守卫确认终态时提交台账。
        self.stall_gap_records = ()
        # D18（v2.0 §16.4）：两阶段状态机。初始态 = P_DECISION（发射点
        # 选择前）；准入事务成功推进 P_COMMITTED，准入批发射推进
        # P_RUNNING；drain 边界 P_COMPUTE_DONE →（终止：MERGE_PENDING /
        # 否则 D_DECISION_PENDING → D_PLAN_SELECTED → D_COMMITTED）→
        # 加入列车 D_RUNNING → 退出 SETTLE_PENDING → merge 结算
        # SETTLE_COMMITTED → SETTLE_DONE。joint_decode_action / _
        # is_continuation = 边界二次决策产物（P 阶段动作在 joint_action
        # 不动——完成归属与工作副本口径分账）；boundary_decision =
        # 边界全遍历的记录摘要（决策行披露）。
        self.joint_stage = JOINT_STAGE_P_DECISION
        self.joint_decode_action = None
        self.joint_decode_is_continuation = False
        self.boundary_decision = None
        # W5（W6 接线）：已揭示类型键（到达时刻由上一轮完成落账的
        # session.next_request_type 填充；None = 未揭示）。
        self.joint_revealed_category = None
        # W5（§7.2）：来源引用句柄（D 提交点 acquire；结算闭合释放）。
        self.joint_source_reference_ids = None
        # C4（v2.0 §19.1/D10/D19）：D 边界完整动作面元数据——
        # 边界首 token 状态 dict（P 末 chunk 采样恰一次
        # 后 issued 待首个 D 前向消费——consume_once 二次消费 fail-closed）。
        self.joint_boundary_first_token = None
        # D20（v2.0 §14.5/§16.4）：阶段 epoch / 完成归属 / 阶段引用
        # 句柄（语义见 __slots__ 注；epoch 自边界首次进入起 +1）。
        self.joint_boundary_epoch = 0
        self.completion_instance_index = None
        self.joint_stage_reference_id = None
        self.origin_home_instance = None
        self.history_transfers = ()
        self.merge_transfers = ()
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
        # （history+input 整份，face_scheduler.py 单源）覆盖
        # prefill 全动作足迹，drain 时 delta ≤ 自有预约 ⟹ gap≡0 ⟹
        # _ensure_capacity 提前空返；2026-09-16 三方裁决（kimi 通道归因
        # + 本方代码级证实 + R17-7 探针 A2 运行级复现）。字段保留仅为
        # prefill 行 schema 稳定（见 _emit_admission 的同名空字段注释）。
        self.prefill_evictions = ()
        self.prefill_decode_transfer = None
        self.decode_evictions = ()
        # decode joiner 逐出支链的尾 watch id（_on_prefill_drain 的
        # D20 边界段登记、_emit_train 的 joiner plan 构图消费；
        # decode_evictions 为空时恒
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
        已落账本）。

        decode_instance_index 回退判据必须显式 ``is not None``：实例 0
        是合法 decode 实例（J 模式边界二次决策可跨实例选中），按真值
        回退会把 decode@0 请求的完成批锚到 prefill 实例（_emit_completion
        以本字段取 decode_group 查 _block_ends 块末——实例不符即
        fail-closed 误报"块末缺失"）。与 prefill 准入 assignment 的占位
        判据（D18 注释处）同形。"""
        return {
            "request_id": self.request_id,
            "session_id": self.session_id,
            "turn_index": self.turn_index,
            "queue_index": self.queue_index,
            "prefill_length": self.prefill_length,
            "decode_length": self.decode_length,
            "prefill_instance_index": self.prefill_instance_index,
            "decode_instance_index": (
                self.decode_instance_index
                if self.decode_instance_index is not None
                else self.prefill_instance_index),
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
            # D18（v2.0 §16.4）：两阶段状态机与边界决策产物（plan 消费面
            # 与决策日志同源；P 阶段未到边界时 decode 侧恒缺省值）。
            "joint_stage": self.joint_stage,
            "joint_decode_action": self.joint_decode_action,
            "joint_decode_is_continuation": self.joint_decode_is_continuation,
            # C4：D 边界完整动作面元数据（首 token
            # 消费状态——构图器/重放侧与决策日志同源；非该形态恒缺省值，
            # 键恒在前向兼容）。
            "joint_boundary_first_token": dict(
                self.joint_boundary_first_token)
            if self.joint_boundary_first_token is not None else None,
            # D20（v2.0 §14.5/§16.4）：请求运行态的完成归属（边界提交
            # 时锁定；P 阶段未到边界时 None——构图/重放侧按缺省键前向
            # 兼容）与阶段 epoch。
            "completion_instance_index": self.completion_instance_index,
            "joint_boundary_epoch": self.joint_boundary_epoch,
            "origin_home_instance": self.origin_home_instance,
            "joint_cost_ns": self.joint_cost_ns,
            "history_transfers": self.history_transfers,
            "merge_transfers": self.merge_transfers,
        }


def _capacity_summary_audit_entry(view) -> dict:
    """InstanceLoadView → 决策行容量驱逐摘要冻结块（D17 可重算）。

    冻结口径 = ``_space_ready_ns_estimate``/``consume_gap_theta`` 实际消
    费的完整计价输入：逐桶键轴（category/suffix_class/shard_ratio/
    writeback_ports）＋逐桶逐 rank 向量（R/clean/dirty/原子数/最大原
    子）。视图未注入增量摘要（直接构造的测试夹具/离线视图）时按与计
    价侧**同一退路**（synthesise_summary_from_reclaimable 单桶合成锚）
    冻结——日志侧不得出现与计价侧不同的第二套桶。外部仅凭本块 +
    load_view 的 hbm_remaining（缺口向量来源之一）即可重算桶选择与逐
    rank θ 份额。

    ``summary_id`` = 冻结桶内容 canonical JSON 的 sha256 前 16 hex：
    同内容同 ID（跨决策行去重/对账键），内容变即变——不是行序号。
    体积纪律：O(实例 × 桶 × rank)，D11 增量摘要有界，不随已完成
    session 数增长；逐决策一行、不写侧车文件（行内 JSON 自足）。
    """
    # getattr 容错：鸭子类型视图（测试 SimpleNamespace）缺省无 D11 字段
    # ——按"未注入"处理走合成锚（与 _space_ready_ns_estimate 同款约定；
    # 生产 InstanceLoadView 恒有字段）。
    summary = getattr(view, "capacity_summary", None)
    if summary is None:
        summary = synthesise_summary_from_reclaimable(
            view.instance_index, view.reclaimable_bytes_by_tp_rank)
    buckets = [
        {
            "category": bucket.key.category,
            "suffix_class": bucket.key.suffix_class,
            "shard_ratio": list(bucket.key.shard_ratio),
            "writeback_ports": list(bucket.key.writeback_ports),
            "reclaimable_by_rank": list(bucket.reclaimable_by_rank),
            "clean_by_rank": list(bucket.clean_by_rank),
            "dirty_by_rank": list(bucket.dirty_by_rank),
            "atom_count_by_rank": list(bucket.atom_count_by_rank),
            "max_atom_by_rank": list(bucket.max_atom_by_rank),
        }
        for bucket in summary.buckets
    ]
    digest = hashlib.sha256(json.dumps(
        buckets, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()[:16]
    return {
        "instance_index": view.instance_index,
        "summary_id": "capsum-{}".format(digest),
        # §5.5（2026-09-30 修复）：摘要版本与登记误差来源随审计块落盘
        # （"摘要预测 vs 实际 E"对账按版本判定同代性；合成锚/鸭子视图
        # 缺省 None = 无版本事实，不虚构代数）。
        "summary_version": getattr(summary, "snapshot_version", None),
        "maintenance_event_count": getattr(
            summary, "maintenance_event_count", None),
        "error_sources": list(getattr(summary, "error_sources", ()) or ()),
        "buckets": buckets,
    }


# 同状态对照 probe（run_scripts/joint_same_state_probe.py，式(24) 选择
# 损失验收）的验证专用强制钩子 env 对——契约由该工具写死，调度器侧
# 按同值实现，不得偏离。
FORCE_BOUNDARY_REQUEST_ENV = "JOINT_FORCE_BOUNDARY_REQUEST"
FORCE_BOUNDARY_ACTION_ENV = "JOINT_FORCE_BOUNDARY_ACTION"
# D05/probe 配套（修复轮二任务 7，2026-09-27）：P 入口单点强制钩子——
# 复用同一 REQUEST env（请求命中面），ACTION 换 P 入口专用
# JOINT_FORCE_ENTRY_ACTION（值域同四动作；只作用于 P 准入选择点，不触
# D 边界）。与 JOINT_FORCE_BOUNDARY_ACTION 互斥武装（同时设置 = 仪器
# 二义，fail-closed）。
FORCE_ENTRY_ACTION_ENV = "JOINT_FORCE_ENTRY_ACTION"
# W8（方案 §11.3/§12.1：instance×action 精确强制）：TARGET env 对——值
# 形如 "<instance_index>:<action>"，与共享 REQUEST env（单一请求命中面）
# 成对武装。旧 per-action 变量（上方两 ACTION env）保留向后兼容但为弃用
# 方向（deprecated: use *_TARGET）；同族 TARGET 与 ACTION 同设 = 仪器二义
# fail-closed（不形成双路径歧义）。
FORCE_BOUNDARY_TARGET_ENV = "JOINT_FORCE_BOUNDARY_TARGET"
FORCE_ENTRY_TARGET_ENV = "JOINT_FORCE_ENTRY_TARGET"
# 一次映射对照臂（2026-09-28，验收纪律第 2 类"一次映射 vs 两阶段闭环
# 对照"）：run 级策略 env——定义/解析在 joint.joint_config（取值域
# BOUNDARY_POLICIES），本模块消费（D 边界选点两形态单一裁决）。
BOUNDARY_POLICY_ENV = "JOINT_BOUNDARY_POLICY"


def _parse_force_boundary_env(lookup) -> tuple[Optional[str], Optional[str]]:
    """强制钩子 env 解析（fail-closed：成对设置 + 动作名 ∈ ALL_ACTIONS）。

    这是测量仪器不是双路径开关（工作区"直接替换规范"明示豁免）：两 env
    成对设置才武装；只设其一 / 空 request_id / 动作名非法即启动失败
    ——半配置的仪器比没有更糟。缺省（均未设）= off，行为零变化。

    修复轮二：REQUEST env 与 P 入口钩子（JOINT_FORCE_ENTRY_ACTION）
    共享——"REQUEST 在场 + 仅 ENTRY ACTION 在场"是 P 入口钩子的合法
    武装形态，边界钩子保持 off（两仪器互斥，不得同时武装）。
    W8：REQUEST 在场 + 仅 *_TARGET 在场（instance×action 精确强制）
    同为合法武装形态——本解析器（per-action 边界钩子）返回 off。
    """
    request_id = lookup.get(FORCE_BOUNDARY_REQUEST_ENV)
    action = lookup.get(FORCE_BOUNDARY_ACTION_ENV)
    if (request_id is None) != (action is None):
        if (action is None and request_id is not None
                and lookup.get(FORCE_ENTRY_ACTION_ENV) is not None):
            # REQUEST 由 P 入口钩子武装：边界钩子 off。
            return None, None
        if (action is None and request_id is not None
                and (lookup.get(FORCE_BOUNDARY_TARGET_ENV) is not None
                     or lookup.get(FORCE_ENTRY_TARGET_ENV) is not None)):
            # REQUEST 由 W8 TARGET 钩子武装：per-action 边界钩子 off。
            return None, None
        raise RuntimeError(
            f"{FORCE_BOUNDARY_REQUEST_ENV}/{FORCE_BOUNDARY_ACTION_ENV} must "
            "be set together to arm the same-state probe force hook; got "
            f"request={request_id!r}, action={action!r}")
    if action is not None:
        if not request_id:
            raise RuntimeError(
                f"{FORCE_BOUNDARY_REQUEST_ENV} must be a non-empty request "
                f"id when the probe force hook is armed, got {request_id!r}")
        if action not in ALL_ACTIONS:
            raise RuntimeError(
                f"{FORCE_BOUNDARY_ACTION_ENV} must be one of {ALL_ACTIONS}, "
                f"got {action!r}")
    return request_id, action


def _parse_force_entry_env(lookup) -> tuple[Optional[str], Optional[str]]:
    """P 入口单点强制钩子 env 解析（fail-closed；修复轮二任务 7）。
    仪器合同（仿 ``_parse_force_boundary_env``）：
    - ``JOINT_FORCE_BOUNDARY_REQUEST`` + ``JOINT_FORCE_ENTRY_ACTION``
      成对武装（REQUEST env 复用既有钩子的请求命中面）；
    - 只设 ENTRY ACTION 不设 REQUEST / 空 REQUEST / 动作名 ∉ ALL_ACTIONS
      ⇒ 启动失败（半配置的仪器比没有更糟）；
    - 与 ``JOINT_FORCE_BOUNDARY_ACTION`` 同时设置 = 两仪器二义 ⇒ 启动
      失败；
    - 缺省（未设）= off，行为零变化（REQUEST 单独在场属边界钩子的
      配置面，本解析器返回 off）。
    """
    action = lookup.get(FORCE_ENTRY_ACTION_ENV)
    if action is None:
        return None, None
    request_id = lookup.get(FORCE_BOUNDARY_REQUEST_ENV)
    if lookup.get(FORCE_BOUNDARY_ACTION_ENV) is not None:
        raise RuntimeError(
            f"{FORCE_ENTRY_ACTION_ENV} and {FORCE_BOUNDARY_ACTION_ENV} "
            "are mutually exclusive probe instruments (arm exactly one "
            "action hook per run)")
    if not request_id:
        raise RuntimeError(
            f"{FORCE_BOUNDARY_REQUEST_ENV} must be a non-empty request id "
            f"when {FORCE_ENTRY_ACTION_ENV} arms the P-entry force hook, "
            f"got {request_id!r}")
    if action not in ALL_ACTIONS:
        raise RuntimeError(
            f"{FORCE_ENTRY_ACTION_ENV} must be one of {ALL_ACTIONS}, "
            f"got {action!r}")
    return request_id, action


def _parse_force_target_env(lookup, target_env: str,
                            sibling_action_env: str):
    """W8 instance×action 精确强制钩子 env 解析（fail-closed）。

    仪器合同（方案 §11.3"不能只按 action 汇总最便宜 instance"）：
    - 值形如 ``"<instance_index>:<action>"``（instance 为非负整数，
      action ∈ ALL_ACTIONS），语法/值域错误即启动失败；
    - 复用 ``JOINT_FORCE_BOUNDARY_REQUEST`` 作为唯一请求命中面（TARGET
      值不重复携带 request——单一请求选择器），缺 REQUEST / 空 REQUEST
      ⇒ 启动失败；
    - 与**同族** per-action 变量（``<sibling_action_env>``）同设 = 仪器
      二义（同边界两套强制身份），fail-closed——旧变量保留向后兼容但
      为弃用方向，二者不得并存武装；
    - 与**对族**动作钩子（entry vs boundary 的另一族 ACTION/TARGET）
      同设 = 每 run 恰一个动作钩子先例的沿用，fail-closed。

    返回 (request_id, instance_index, action)；未设返回 None。
    """
    value = lookup.get(target_env)
    if value is None:
        return None
    other_family_envs = (
        (FORCE_ENTRY_ACTION_ENV, FORCE_ENTRY_TARGET_ENV)
        if target_env == FORCE_BOUNDARY_TARGET_ENV
        else (FORCE_BOUNDARY_ACTION_ENV, FORCE_BOUNDARY_TARGET_ENV))
    if lookup.get(sibling_action_env) is not None:
        raise RuntimeError(
            f"{target_env} and {sibling_action_env} are mutually exclusive "
            "probe instruments (per-action hook is deprecated; arm exactly "
            "one force identity per boundary)")
    for other_env in other_family_envs:
        if lookup.get(other_env) is not None:
            raise RuntimeError(
                f"{target_env} and {other_env} are mutually exclusive probe "
                "instruments (arm exactly one action hook per run: entry or "
                "boundary)")
    request_id = lookup.get(FORCE_BOUNDARY_REQUEST_ENV)
    if not request_id:
        raise RuntimeError(
            f"{FORCE_BOUNDARY_REQUEST_ENV} must be a non-empty request id "
            f"when {target_env} arms the instance-action force hook, got "
            f"{request_id!r}")
    instance_text, separator, action = value.partition(":")
    if not separator or not instance_text.strip().isdigit():
        raise RuntimeError(
            f"{target_env} must be '<instance_index>:<action>' with a "
            f"non-negative integer instance index, got {value!r}")
    if action not in ALL_ACTIONS:
        raise RuntimeError(
            f"{target_env} action must be one of {ALL_ACTIONS}, got "
            f"{action!r} (value {value!r})")
    return request_id, int(instance_text.strip()), action


def _check_boundary_policy_instrument_conflict(
        boundary_policy: str, force_boundary_action: Optional[str],
        force_entry_action: Optional[str],
        force_boundary_target: Optional[tuple] = None,
        force_entry_target: Optional[tuple] = None) -> None:
    """一次映射对照臂 × probe 强制钩子互斥武装裁决（fail-closed 单一裁决
    点；2026-09-28 验收纪律第 2 类）。

    ``JOINT_BOUNDARY_POLICY=p-continuation``（一次映射对照臂）下 D 边界
    不做二次全候选重估——候选面为空，边界强制钩子（JOINT_FORCE_BOUNDARY_
    ACTION/JOINT_FORCE_BOUNDARY_TARGET）没有可强制的候选对象；P 入口钩子
    （JOINT_FORCE_ENTRY_ACTION/JOINT_FORCE_ENTRY_TARGET）同 run 混用则让
    一次映射臂的对照读数混入 P 面强制（同状态前提失真）。两类仪器同设 =
    仪器二义，启动失败（仿两 ACTION 钩子互斥先例）。off（缺省）与钩子
    任意组合照常合法。
    """
    if boundary_policy != BOUNDARY_POLICY_P_CONTINUATION:
        return
    armed = [name for name, value in (
        (FORCE_BOUNDARY_ACTION_ENV, force_boundary_action),
        (FORCE_ENTRY_ACTION_ENV, force_entry_action),
        (FORCE_BOUNDARY_TARGET_ENV,
         "armed" if force_boundary_target is not None else None),
        (FORCE_ENTRY_TARGET_ENV,
         "armed" if force_entry_target is not None else None))
        if value is not None]
    if armed:
        raise RuntimeError(
            f"JOINT_BOUNDARY_POLICY={BOUNDARY_POLICY_P_CONTINUATION} and "
            f"probe force hooks ({', '.join(armed)}) are mutually "
            "exclusive instruments (arm exactly one boundary-policy "
            "semantics per run): p-continuation skips the D-boundary "
            "candidate traversal, leaving the boundary force hook with "
            "no candidate surface to force")


class _JointDiePlanProvider:
    """§9.6 两层接口的调度器侧装配（joint 二期，2026-10-03）。

    名义 service 查询与投影 compute leg 用**同一批 DiePlan**：全部时间/
    字节/服务量数值来自 extern/die_model 共享库（die_estimate_batch 与
    A3 执行协议，经 online.die_model_oracle 的 ctypes 客户端）；本类只
    构造 DieOpSpec（契约 §2 schema，类别映射与 region 融合走
    online.die_model_adapter 的同源入口——不写第二套公式）并施加两条
    预测侧语义：

    * §9.6-1 除数语义：``hbm_share_divisor=s`` 时仅内存服务率变为
      ``B/s``（service_units ÷ (B/s)），不整段 ×s、NoC/PE/vec 不降速
      （``nominal_service_ns`` 同式）；
    * §9.6-4 service factor：只缩放预测 service_units（描述符的内存
      阶段经 ScaledExecution 包装；物理 actual_bytes 不动）。

    估算 memo：key = (形态, tokens…) ——跨候选/跨决策复用（同形状重复
    查询是决策遍历的常态）；计划编译缓存在库内 PlanCache（同 spec 复
    用），执行句柄每腿每候选独立（§8.5：执行句柄不入缓存）。

    口径披露（构造一次、随首个描述符查询落 notes）：

    * ``die_decode_aggregate_plan``——decode 地平线的名义查询与投影腿
      用同一聚合计划（spans = 逐步 (1, z_k)、weight_passes = n̂、
      layer_count = 全模）。GEMM/向量工作量与逐 step 发射严格守恒
      （M 合并的线性性）；tile 口径按合并 M 映射（逐 step M=1 的运行
      时发射更不利），launch 次数 = layer_count×weight_passes 与逐
      step 一致。
    * 实例级名义量 = TP 各 rank 并行执行的最慢 rank（逐 rank 估值取
      max——头分片不均匀时重 rank 决定完成）。
    """

    def __init__(self, *, die_context, tp_degree: int, layers: int,
                 chunk_tokens: int, hbm_bytes_per_ns: float,
                 factors_getter, mlp_variant: str) -> None:
        from online.die_model_adapter import (
            require_legacy_projection_context,
        )
        require_legacy_projection_context(die_context)
        # factors_getter() -> (prefill_factor, decode_factor)：决策时点读
        # 当前值（service factor 只缩放 service_units 的 §9.6-4 语义按
        # 查询时刻生效；供给器本体跨决策持久——估算 memo 与库内
        # PlanCache 复用，"执行句柄不入缓存"纪律不受影响）。
        from online.die_model_adapter import DieModelContext
        # 预测私有语境：绝不触碰发射态（die_context 的 begin_emission
        # 状态属于构图器；这里克隆数据面）。
        self._ctx = DieModelContext(
            profile=die_context.profile,
            die_config_path=die_context.die_config_path,
            config_hash=die_context.config_hash,
            dtype=die_context.dtype,
            model_dims=dict(die_context.model_dims),
            num_q_heads=die_context._num_q_heads,
            num_kv_heads=die_context._num_kv_heads,
            ffn_size=die_context._ffn_size,
            vocab_size=die_context._vocab_size,
            task_schema=getattr(
                die_context, "task_schema", "die_op_spec_v1"),
            execution_profile=getattr(
                die_context, "execution_profile", None),
        )
        self._die_config_path = die_context.die_config_path
        # Keep the resolved hardware semantics in the compact native input.
        # The ctypes oracle path may use a handle/factory, but neither belongs
        # in the immutable projection wire.  Read once per provider and pass
        # the full parsed object into each descriptor.
        try:
            self._hardware_config = json.loads(
                self._die_config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                "cannot read resolved die hardware config {}: {}".format(
                    self._die_config_path, exc)) from exc
        if not isinstance(self._hardware_config, dict):
            raise ValueError("resolved die hardware config must be a JSON object")
        self._tp = int(tp_degree)
        self._layers = int(layers)
        self._chunk_tokens = int(chunk_tokens)
        self._hbm_bytes_per_ns = float(hbm_bytes_per_ns)
        self._factors_getter = factors_getter
        if mlp_variant == "swiglu":
            self._categories = (
                "attention_input_rmsnorm", "attention_qkv_projection",
                "attention_qk_matmul", "attention_scale_mask",
                "attention_softmax", "attention_av_matmul",
                "attention_output_projection", "attention_residual_add",
                "mlp_input_rmsnorm", "mlp_gate_up_projection", "mlp_swiglu",
                "mlp_down_projection", "mlp_residual_add")
        else:
            self._categories = (
                "attention_input_layernorm", "attention_qkv_projection",
                "attention_qk_matmul", "attention_scale_mask",
                "attention_softmax", "attention_av_matmul",
                "attention_output_projection", "attention_residual_add",
                "mlp_input_layernorm", "mlp_up_projection", "mlp_gelu",
                "mlp_down_projection", "mlp_residual_add")
        self._oracle = None
        # A2(2026-10-17, joint提速改造): OrderedDict 承载 + _ESTIMATE_MEMO_CAP
        # 挪尾淘汰（原为无界 dict，长跑内存缓涨；命中语义不变：键相同才命中）。
        self._estimate_memo = OrderedDict()
        # 墙时/计数埋点(2026-10-05):memo 命中/未命中计数——
        # _aggregate_estimate 逐次入账,run 末经 estimate_timing_snapshot
        # 汇入决策日志/汇总行。
        self._estimate_memo_hits = 0
        self._estimate_memo_misses = 0
        self._op_groups = None

    # ------------------------------------------------------------- 基础 --

    def _library(self):
        if self._oracle is None:
            from online.die_model_oracle import DieModelOracle
            self._oracle = DieModelOracle(
                config_json_path=str(self._die_config_path))
        return self._oracle

    @property
    def _prefill_factor(self) -> float:
        return float(self._factors_getter()[0])

    @property
    def _decode_factor(self) -> float:
        return float(self._factors_getter()[1])

    def _group_specs(self, *, phase, spans, layer_count, weight_passes,
                     tensor_parallel_rank, kv_identity):
        """一次（发射）规景的逐融合组 spec 序列（与执行侧 13 类→融合组
        同源：compile_regions 的结构面（external_consumers=False、
        boundary_after=通信边界类别）；num_ops/tensor_size 只进聚合披露，
        融合判定不读它们）。"""
        from online.die_model_adapter import build_die_spec
        if self._op_groups is None:
            records = []
            for category in self._categories:
                records.append({
                    "name": f"predict_{category}",
                    "category": category,
                    "op_kind": self._ctx.op_kind_of(category),
                    "num_ops": 0, "tensor_size": 0,
                    "layer_count": 1,
                    "weight_passes": 1,
                    "external_consumers": False,
                    "boundary_after": category in (
                        "attention_output_projection",
                        "mlp_down_projection"),
                    "identity": {
                        "tp": self._tp, "rank": tensor_parallel_rank,
                        "layer": layer_count, "span_set": tuple(
                            (int(s["q_tokens"]), int(s["kv_tokens"]))
                            for s in spans)},
                })
            self._op_groups = self._ctx.compile_regions(records)
        self._ctx.phase = phase
        self._ctx.tensor_parallel = self._tp
        self._ctx.tensor_parallel_rank = int(tensor_parallel_rank)
        return tuple(
            build_die_spec(
                group["op_kind"], self._ctx, spans,
                layer_count=layer_count,
                weight_passes=weight_passes,
                fusion_policy="auto" if group["fused"] else "none",
                kv_identity=kv_identity)
            for group in self._op_groups)

    # ------------------------------------------------------------- 形状 --

    def _prefill_spans(self, input_tokens: int, history_tokens: int):
        spans = []
        done = 0
        while done < input_tokens:
            take = min(self._chunk_tokens, int(input_tokens) - done)
            spans.append({
                "request_id": "predict",
                "q_tokens": take,
                "kv_tokens": int(history_tokens) + done + take,
                "q_start_pos": int(history_tokens) + done,
                "kv_start_pos": 0,
            })
            done += take
        return spans

    def _decode_step_span(self, context_tokens: int):
        return {
            "request_id": "predict",
            "q_tokens": 1,
            "kv_tokens": int(context_tokens),
            "q_start_pos": int(context_tokens) - 1,
            "kv_start_pos": 0,
        }

    def _decode_horizon_spans(self, base_context_tokens: int, steps: int):
        return [
            self._decode_step_span(int(base_context_tokens) + offset + 1)
            for offset in range(max(1, int(steps)))]

    # ------------------------------------------------------------- 估值 --

    def _aggregate_estimate(self, specs, key):
        cached = self._estimate_memo.get(key)
        if cached is None:
            self._estimate_memo_misses += 1
            results = self._library().estimate(list(specs))
            from joint.die_plan_projection import (
                result_actual_bytes, result_non_memory_ns,
                result_service_units,
            )
            for index, result in enumerate(results):
                if result is None:
                    raise RuntimeError(
                        "die estimate rejected prediction spec {} of {}: "
                        "{}".format(
                            index, key, self._library().last_errors[index]))
            cached = (
                max(result_non_memory_ns(result) for result in results),
                sum(result_service_units(result) for result in results),
                sum(result_actual_bytes(result) for result in results))
            self._estimate_memo[key] = cached
            if len(self._estimate_memo) > _ESTIMATE_MEMO_CAP:
                self._estimate_memo.popitem(last=False)
        else:
            self._estimate_memo.move_to_end(key)
            self._estimate_memo_hits += 1
        return cached

    def estimate_timing_snapshot(self) -> dict:
        """墙时/计数埋点的只读副本:memo 命中/未命中 + 底层 ctypes
        oracle 的批量调用计数/总 spec 数/累计墙时(oracle 未惰性创建
        时 library=None 如实披露)。"""
        oracle = self._oracle
        library_stats = (oracle.estimate_timing_snapshot()
                         if oracle is not None else None)
        return {
            "memo_hits": self._estimate_memo_hits,
            "memo_misses": self._estimate_memo_misses,
            "library": library_stats,
        }

    def _instance_estimate(self, *, phase, spans, layer_count, weight_passes,
                           kv_identity):
        """实例级（= 最慢 TP rank）非内存 ns / 服务量 / 物理字节。

        **rank 签名去重**（性能裁定，2026-10-03）：TP 分片extent 相同的
        rank 工作量逐位相同（spec 的 tensor_parallel_rank 只经 shard_
        extent 进数值；库内无 rank 序号语义）——按 (q_heads_local,
        kv_heads_local, ffn_local, vocab_local) 签名去重后每签名估一次
        （32 头/6 rank = 2 个签名而非 6 次编译），取签名间 max。整除
        分片退化为单签名。
        """
        from generate_trace import shard_extent
        num_q_heads = int(self._ctx._num_q_heads)
        num_kv_heads = int(self._ctx._num_kv_heads)
        best = None
        seen = set()
        for rank in range(self._tp):
            signature = (
                shard_extent(num_q_heads, self._tp, rank),
                shard_extent(num_kv_heads, self._tp, rank),
                shard_extent(int(self._ctx._ffn_size), self._tp, rank),
                shard_extent(int(self._ctx._vocab_size), self._tp, rank),
            )
            if signature in seen:
                continue
            seen.add(signature)
            specs = self._group_specs(
                phase=phase, spans=spans, layer_count=layer_count,
                weight_passes=weight_passes,
                tensor_parallel_rank=rank, kv_identity=kv_identity)
            non_mem, units, actual = self._aggregate_estimate(
                specs, (phase, layer_count, weight_passes, kv_identity,
                        tuple((s["q_tokens"], s["kv_tokens"],
                               s["q_start_pos"]) for s in spans), signature))
            per_rank_nominal = (
                non_mem + units / self._hbm_bytes_per_ns)
            if best is None or per_rank_nominal > best[0]:
                best = (per_rank_nominal, non_mem, units, actual)
        return best[1], best[2], best[3]

    def _descriptor(self, *, phase, spans, layer_count, weight_passes,
                    tensor_parallel_rank, kv_identity, service_factor,
                    route, label):
        from joint.die_plan_projection import (
            DiePlanDescriptor, ScaledDieExecution,
        )
        specs = self._group_specs(
            phase=phase, spans=spans, layer_count=layer_count,
            weight_passes=weight_passes,
            tensor_parallel_rank=tensor_parallel_rank,
            kv_identity=kv_identity)
        library = self._library()

        def _factory(_descriptor, _specs=tuple(specs),
                     _factor=float(service_factor), _lib=library):
            from joint.die_plan_projection import SequencedDieExecution
            return ScaledDieExecution(
                SequencedDieExecution(
                    _lib.create_execution(spec) for spec in _specs),
                _factor)

        return DiePlanDescriptor(
            specs=specs, route=tuple(route), label=label,
            execution_factory=_factory,
            hardware_config=self._hardware_config,
            service_factor=float(service_factor))

    # ------------------------------------------------ 名义查询（§9.6-1） --

    def prefill_nominal_ns(self, input_tokens, history_tokens,
                           hbm_share_divisor: int = 1) -> int:
        """prefill 名义 service（§9.6-1：内存服务率 B/s；factor 只缩放
        service_units；非内存资源时间不 ×s、不 ×factor）。"""
        non_mem, units, _actual = self._instance_estimate(
            phase="prefill", spans=self._prefill_spans(
                int(input_tokens), int(history_tokens)),
            layer_count=self._layers, weight_passes=1, kv_identity="local")
        divisor = max(1, int(hbm_share_divisor))
        service_ns = non_mem + (
            units * self._prefill_factor
            / (self._hbm_bytes_per_ns / divisor))
        return int(math.ceil(service_ns))

    def decode_step_nominal_ns(self, context_tokens,
                               hbm_share_divisor: int = 1) -> int:
        non_mem, units, _actual = self._instance_estimate(
            phase="decode",
            spans=[self._decode_step_span(int(context_tokens))],
            layer_count=self._layers, weight_passes=1, kv_identity="local")
        service_ns = (
            non_mem + units * self._decode_factor
            / (self._hbm_bytes_per_ns / max(1, int(hbm_share_divisor))))
        return int(math.ceil(service_ns))

    def decode_horizon_nominal_ns(self, base_context_tokens, steps,
                                  hbm_share_divisor: int = 1) -> float:
        non_mem, units, _actual = self._instance_estimate(
            phase="decode",
            spans=self._decode_horizon_spans(
                int(base_context_tokens), int(steps)),
            layer_count=self._layers, weight_passes=max(1, int(steps)),
            kv_identity="local")
        return (non_mem + units / (
            self._hbm_bytes_per_ns / max(1, int(hbm_share_divisor))))

    # ------------------------------------------- profile 双量（§9.6-3） --

    def prefill_non_memory_ns(self, input_tokens, history_tokens) -> int:
        non_mem, _units, _actual = self._instance_estimate(
            phase="prefill", spans=self._prefill_spans(
                int(input_tokens), int(history_tokens)),
            layer_count=self._layers, weight_passes=1, kv_identity="local")
        return non_mem

    def prefill_service_units(self, input_tokens, history_tokens) -> float:
        _non_mem, units, _actual = self._instance_estimate(
            phase="prefill", spans=self._prefill_spans(
                int(input_tokens), int(history_tokens)),
            layer_count=self._layers, weight_passes=1, kv_identity="local")
        return units

    def decode_non_memory_ns(self, context_tokens) -> int:
        non_mem, _units, _actual = self._instance_estimate(
            phase="decode",
            spans=[self._decode_step_span(int(context_tokens))],
            layer_count=self._layers, weight_passes=1, kv_identity="local")
        return non_mem

    def decode_service_units(self, context_tokens) -> float:
        _non_mem, units, _actual = self._instance_estimate(
            phase="decode",
            spans=[self._decode_step_span(int(context_tokens))],
            layer_count=self._layers, weight_passes=1, kv_identity="local")
        return units

    def decode_horizon_profile(self, base_context_tokens, steps):
        """decode 地平线的 (非内存 ns, service_units)（聚合计划口径）。"""
        non_mem, units, _actual = self._instance_estimate(
            phase="decode",
            spans=self._decode_horizon_spans(
                int(base_context_tokens), int(steps)),
            layer_count=self._layers, weight_passes=max(1, int(steps)),
            kv_identity="local")
        return non_mem, units

    # ------------------------------------------- 投影描述符（§9.6-2） --

    def prefill_layer_descriptor(self, *, input_tokens, history_tokens,
                                 tensor_parallel_rank, service_factor,
                                 kv_identity, route):
        """prefill 逐层段的投影描述符（§9.6 第 2 条）。

        **返回 None = 聚合规景**：tile 级计划的协议动作量（~265 动作/
        22-chunk 层段注意力、随 chunk 数线性增长）× 32 层 × 6 rank × 36
        候选 ≈ 1e5-3e5 协议事件/候选前推——超出投影事件预算（262,144）
        且 ctypes 逐动作往返不可行（与 decode 地平线同族，见
        ``decode_horizon_descriptor``）。逐层段工作改走**逐层 DieOpSpec
        估值**（``prefill_layer_profile``——每层 layer_count=1 的同源
        估算，非 pure-time 比例分摊）；协议推进面保留给有界计划（测试
        替身钉内核语义，§10.2-②）。
        """
        return None

    def prefill_layer_profile(self, input_tokens, history_tokens):
        """逐层段 (非内存 ns, service_units)（最慢 TP rank；§9.6-3）。

        每层 layer_count=1 的 DieOpSpec 估值（与执行侧逐层段发射同形——
        每层一份宏组序列、chunk spans 同一切分），memo 复用。返回长度
        = layers 的两个列表（逐层量，不做 pure-time 比例分摊）。
        """
        # 整 pass（layer_count=全模）估值 ÷ 层数——变换器层结构恒等
        # （逐层 GEMM/向量/KV 装载与 launch 均随层线性），避免逐层形状
        # 的第二次库编译（~45s/形状实测）；纯算术分配，不引入新公式。
        non_mem, units, _actual = self._instance_estimate(
            phase="prefill",
            spans=self._prefill_spans(int(input_tokens), int(history_tokens)),
            layer_count=self._layers, weight_passes=1, kv_identity="local")
        per_layer_pure = int(non_mem / self._layers)
        remainder = int(non_mem) - per_layer_pure * self._layers
        per_layer_units = units / float(self._layers)
        pure_by_layer = [per_layer_pure] * self._layers
        pure_by_layer[0] += remainder
        return pure_by_layer, [per_layer_units] * self._layers

    def decode_horizon_descriptor(self, *, decode_steps, base_context_tokens,
                                  tensor_parallel_rank, service_factor,
                                  kv_identity, route):
        """decode 地平线的投影描述符（§9.6 第 2 条）。

        **返回 None = 聚合规景**：逐 step span 的精确计划在 tile 级展开
        为 ~10^5 协议动作/链（359,680 memreqs @ n̂=8 实测），每候选每
        rank 驱动不可行；K 段折叠会改变物理（跨 step 的片上 KV 复用——
        units 比逐 step 低 31%，实测 0.69）。§8.5 的 repeat 压缩需要
        区间内竞争集合不变的证明，候选投影无法给出。因此 decode 腿以
        聚合 (非内存 ns, service_units) 走 compute 腿（同源估值、双量
        分列、不伪造字节），协议推进面保留给有界的 prefill 逐层段（生产
        实际驱动）。测试替身仍可构造 horizon 描述符
        钉内核语义（§10.2-②）。
        """
        return None


class Sh30OnlineScheduler(OnlineSchedulerBase):
    """strategy 变体：sh_3.0 三段式准入 + P→D 边界二次决策（D18 前为
    decode 同实例）+ 三态 KV（关感知）+ 迭代列车拼 batch（2026-08-22）。

    蓝本：已移除的离线 plan_face_requests。拓扑 / edge_free
    掩码 / KV 账本（与离线同一函数、同参数）在 __init__ 一次性构建，运行期
    策略输入全部来自这些 Python 账本。
    """

    # F6（2026-09-22 销账）：上方的 C11 类级软缺省块已删——全部 16 个
    # 属性在真 __init__ 无条件赋值，类级缺省只服务 __new__ 测试替身
    # （漏设静默绕过，如 HBM 端口登记丢失无销账）。替身现须自行补齐
    # __init__ 初值（漏设 = AttributeError，fail-loud）。

    # 同状态对照 probe 强制钩子的类级缺省（off）——只服务 __new__ 测试
    # 替身（真 __init__ 无条件覆写；F6 注记的既定模式）。
    _force_boundary_request: Optional[str] = None
    _force_boundary_action: Optional[str] = None
    # P 入口单点强制钩子的类级缺省（off；修复轮二任务 7）——同款模式。
    _force_entry_request: Optional[str] = None
    _force_entry_action: Optional[str] = None
    # W8 instance×action 精确强制钩子的类级缺省（off）——(request_id,
    # instance_index, action) 三元组；同款 __new__ 替身模式。
    _force_boundary_target: Optional[tuple] = None
    _force_entry_target: Optional[tuple] = None

    def __init__(self, *, manifest, config, graph, digest_sink=None,
                 decision_log_sink=None, train_ledger_sink=None,
                 profile_sink=None, mode: str = "strategy",
                 sensing: bool = False,
                 defensive_reply_cache: bool = False,
                 sensing_query_sink=None, online_stats_sink=None,
                 ledger_sink=None, work_ledger_sink=None):
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
            # W5（2026-09-29，W1 INTEGRATION_NOTES §1.1 整合义务）：显式
            # 转发 work_ledger_sink——基类构造 joint_work_ledger（因果运行
            # 账本 + 侧车 sink）；online_service.py 的构造后注入过渡行已删
            # （本参数为唯一注入通道，缺省 None = 无侧车的嵌入/测试路径）。
            work_ledger_sink=work_ledger_sink,
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
        # W3（2026-09-29，方案 §9.1/W6 INTEGRATION_NOTES §5）：版本化
        # 预测配置一次加载（JOINT_PREDICTION_CONFIG 唯一 env 入口）+
        # cross_validate 组合校验（J-off 时预测轴记 inactive，不输出
        # "历史已改善 J"式生效声明；矛盾组合 fail-closed）。JCM 构造经
        # ``_joint_cost_model`` 注入；manifest 披露面 =
        # ``joint_prediction_manifest`` 属性（online_service 侧车接线归
        # W5/主 Agent——本文件冻结清单外的 online_service 不可改）。
        from joint.prediction_config import (
            cross_validate as _cross_validate_prediction,
            load_prediction_config as _load_prediction_config,
        )
        self._joint_prediction_config = _load_prediction_config()
        self._joint_prediction_cross_validation = (
            _cross_validate_prediction(
                self.joint_config, self._joint_prediction_config))
        # W5（2026-09-29，W6 INTEGRATION_NOTES §5 装配）：W6 统计/估计器
        # 三件套按 shadow 纪律接线——构造参数即 ``pred_config.params``
        # （零平行常数纪律；不手拼常数）：
        # * ``_joint_length_history``：条件剩余长度估计器——**消费切换
        #   后为 P 准入/D 边界的正式 n̂ 来源**（P/D 同刻同实例同一规则
        #   与版本，方案 §8.2；``_joint_horizon`` 保留为诊断/先验）；
        # * ``_joint_arrival_stats``：互斥类别到达强度统计（shadow——
        #   只统计不进选择规则）；
        # * ``_joint_service_samples``：服务样本统一估计器（§8.1 单账本
        #   ——2026-09-30 修复起为纯列车服务样本**唯一更新点**；J 计价
        #   ``_joint_factors`` 与 face γ/k_hide 均在决策时点读其只读
        #   发布值，不再并联更新）；
        # * ``_joint_epoch_log``：§8.5 epoch/分布变化登记（注入预测引擎
        #   回退链 + 样本硬件版本分叉挂点；manifest 披露——本 run 无
        #   epoch 触发时显式登记空 epochs，不缺省缺席）。
        from joint.history_estimator import (
            ArrivalStatistics as _ArrivalStatistics,
            ConditionalRemainingLengthEstimator as _ConditionalEstimator,
            EpochLog as _EpochLog,
            ServiceSampleRecorder as _ServiceSampleRecorder,
        )
        _pred_params = self._joint_prediction_config.params
        self._joint_length_history = _ConditionalEstimator(_pred_params)
        self._joint_arrival_stats = _ArrivalStatistics(_pred_params)
        self._joint_epoch_log = _EpochLog(_pred_params)
        self._joint_service_samples = _ServiceSampleRecorder(
            _pred_params, epoch_log=self._joint_epoch_log)
        # W5（§7.4 统一结算状态机）：逐请求结算账本（request_id ->
        # {stage, service_done_ns, settle_committed_ns, settle_done_ns,
        # merge_payload_bytes, merge_done_ns, settlement_policy}）——
        # runtime 在 _complete_requests 退休而 merge 载荷可在途，SETTLE_
        # DONE 由本账本追到（_on_merge_done / 零载荷即时刻）；闭合即弹出
        # （有界；泄漏审计 verify_run_end）。
        self._joint_settlements = {}
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
        # ---- ACC（joint修改文档 验收纪律第 3 类，v2.0 §19.3）：预测
        # 器自身开销统计账本。恒开、无开关（§19.3 是验收必需项）——
        # P/D 决策墙时逐次样本（P50/P95/P99 汇总在 run 末快照）、D11
        # 容量摘要查询费、后端事件量在决策/发射路径直接记账；宿主墙
        # 时锚 = 本构造点（快照占比分母），峰值 RSS 与决策日志大小在
        # 快照时读取/由 online_service 回填。宿主 Python 墙时 ≠ 片上
        # 决策时延，快照只披露测量口径不外推部署结论。
        self.predictor_cost = PredictorCostAccounting()
        self._predictor_cost_host_anchor_ns = time.monotonic_ns()
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
        # §4.4/A03（2026-09-30 修复）：max(在册, 遥测) 合并冲突披露的
        # 时间窗（最近采样窗 (start, end)；None = 本 run 尚无遥测样本）。
        self._telemetry_last_window_bounds_ns = None
        # A10'(a)/A12'（2026-09-22，§4.3 补遗）：served_bytes==0 ∧
        # active_ns>0 = 有活动无载荷字节的有效速率零窗口（uint64 差分
        # 计量无"整数截断伪影"，A12' 定性订正）——样本丢弃计数（决策
        # 日志遥测块披露位 _telemetry_coverage_decision；真零速率持续
        # 场景由 A9' 缺席失效兜底）。
        self._telemetry_zero_rate_dropped = 0
        # §5.4（ARRIVAL 批内缓存，零语义性能修改方案 2026-10-16）：遥测
        # 状态版本——_ingest_link_telemetry 两个出口（断供清缓存分支与
        # 正常末尾）各 +1，供 P 准入缓存键感知 _link_telemetry_rates/
        # flow_counts/window_bounds 的任何变化。缓存关闭时无人读取，
        # 不影响任何决策路径。
        self._link_telemetry_version = 0
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
        # 二动作分列选中计数 +
        # quota 等待分列计数 + deferred 驻留时长样本（首 defer →
        # 最终准入）。
        self._joint_action_selection_counts = {
            "stay": 0, "copy": 0,
        }
        self._quota_deferred_wait_counts = {
            WAIT_CAPACITY: 0, WAIT_QUOTA_LINK: 0, WAIT_QUOTA_PORT: 0}
        self._quota_deferred_dwell_ns: list[int] = []
        # 决策时延埋点（C11 步骤 4：全候选 × 4 动作 × 逐资源检查的
        # Python 侧增量记录——纯决策段墙时 + 配额判据层墙时分开计）。
        self._admission_decision_wall_ns_total = 0
        self._admission_decision_wall_ns_max = 0
        self._admission_decision_count = 0
        # 墙时分段（2026-10-05 埋点）：纯决策段按"视图+代价模型构建"
        # （premodel）与"候选枚举遍历"（enum）两段分列——P 准入
        # （_try_admit_request）与 D 边界（_on_prefill_drain）各一对
        # 累计器；随 joint_decision_metrics 收尾行披露。
        self._admission_premodel_wall_ns_total = 0
        self._admission_enum_wall_ns_total = 0
        self._boundary_premodel_wall_ns_total = 0
        self._boundary_enum_wall_ns_total = 0
        self._boundary_decision_count = 0
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
        # cfg2（2026-09-30，§5.6/§9.2/§10.1 计算量组）：逐决策预测前推
        # 计算量的 run 级累计（P/D 两入口共享一张累加表；JCM 实例按决
        # 策构造，账面在其上的命中/事件计数经 _joint_accumulate_
        # projection_cost 汇入）+ 版本失效/容量回退的额外遍历计数（§5.6
        # "版本失效、失败重评等额外遍历须累计报告"——D 边界提交点回退
        # 继续方案的发生次数）。收尾行 joint_decision_metrics 披露。
        self._joint_projection_cost_totals: dict[str, int] = {
            "decisions_priced": 0,
            "world_context_builds": 0,
            "world_context_cache_hits": 0,
            "projections": 0,
            "projection_events_total": 0,
            "projection_events_max": 0,
            "closure_resources_max": 0,
            "closure_works_max": 0,
            "background_drain_truncated": 0,
            "projection_budget_exceeded": 0,
            # 墙时分段（2026-10-05 埋点）累计位 + 按阶段投影计数。
            "build_wall_ns_total": 0,
            "closure_wall_ns_total": 0,
            "candidate_world_wall_ns_total": 0,
            "reference_world_wall_ns_total": 0,
            "reference_world_runs": 0,
            "projections_by_stage": {"P": 0, "D": 0},
        }
        # Bounded prior snapshot used to report per-decision DIE estimate
        # counter deltas alongside cumulative values.
        self._joint_last_decision_die_estimate_stats = None
        self._joint_projection_world_scale_last: dict | None = None
        self._joint_boundary_fallback_count = 0
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
        # 读腿恒有；目标 = exec 写腿，copy 留存写 / merge 落点写
        # ——A4' 补价后计价与执行同腿型）；
        # D04（2026-09-26）起 pool 路径（remote_load/remote_store）端
        # 点同样入表：恢复 = 目标 HBM 写端点、写回 = 源 HBM 读端点
        # （edge 侧是池端口非 rank HBM）——F3 旧豁免废除，候选池腿
        # （式 7）端点除数自此看见在途池流；活跃 decode 消费流经因果
        # 负载视图 provider 派生（闲置为 0，不假设恒 1），不经登记通
        # 道（杜绝漏更新）。
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
            # §8.1 单账本（2026-09-30 修复）：face γ/η 消费方（k_hide
            # 递推/adaptive 披露）经 PublishedServiceFactorGroup 读统一
            # 服务样本估计器的只读发布值——同一样本不再二次更新 face
            # 侧时间衰减组（旧 M4 桥接退役）。
            service_factor_source=self._joint_service_samples,
        )
        # ACC（验收纪律第 3 类，v2.0 §19.3）补全：容量摘要**事件侧**增量
        # 维护（_refresh_capacity_summary）墙时送入与查询侧（快照计时）
        # 同一统计通道（predictor_cost.note_summary_maintenance）——维护
        # 次数/累计耗时在 manager 分列，run 末经 predictor_cost_snapshot
        # 并入 summary_maintenance 块（10% 预算口径读同一键）。notifier
        # 在 KVCacheManager 侧缺省 None（零行为差），此处装配点接线。
        self.kv_manager.capacity_summary_maintenance_notifier = (
            self.predictor_cost.note_summary_maintenance)
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
        # §8.1 单账本（2026-09-30 修复）：J 计价因子注入统一估计器后为
        # 只读发布视图——prefill/decode 因子在 flush（JCM 构造点）刷新
        # 自 ServiceSampleRecorder 发布值；transfer 遥测通道仍由本对象
        # 自身承担（链路窗口样本不与 T/E 共享）。
        self._joint_factors = ServiceFactors(
            unified_source=self._joint_service_samples)
        # W5（2026-09-29，A3 快照接线）：决策级硬件摘要——DecisionSnapshot
        # versions["hardware"] 的稳定摘要（速率 + 拓扑规模；变配置重启 run
        # 才会变）。当前交付的 delta 引用（快照构造输入）在
        # run_variant_policy 入口捕获。
        self._joint_current_delta = None
        self._joint_hardware_digest = "rates={}/{}/{}/lat={}/{};inst={}x{}".format(
            self._joint_rates.noc_link_bytes_per_ns,
            self._joint_rates.pool_port_bytes_per_ns,
            self._joint_rates.local_hbm_bytes_per_ns,
            self._joint_rates.d2d_latency_ns,
            self._joint_rates.pool_latency_ns,
            len(self.topology.instances),
            self.topology.instances[0].size if self.topology.instances else 0,
        )
        # W5：背景世界源状态（"ledger" = 账本权威投影；"absent_legacy_peer"
        # = 旧对端/fixture 缺 work_ledger 字段——显式降级 LoadsWorldSource，
        # 不静默空背景；run 末经 manifest 回填披露）。
        self._joint_world_source_status = "absent_legacy_peer"
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
        self.instances = [
            _OnlineInstanceState(index=i)
            for i in range(len(self.topology.instances))
        ]
        # N12（R15-4，2026-09-14）：decode 负载标定在线化——active_decode
        # 剩余估计与 horizon 冷启动全部由 CausalHorizonEstimator 因果供给
        # （session 已完成均值 → run 已完成均值 → 冷启动 32 decode 前向
        # 步——v2.0 式(4) 冻结先验，D14；source="cold_start_default" 显式
        # 披露）。全 trace decode 均值常数不再进入任何决策
        # 输入（总纲 §13.1/§13.2 明文；本条推翻 2026-08-15"禁止在线增量
        # mean"裁决在 joint 语境的适用——其语境是 sh 基底工程常数与在线/
        # 离线对拍纪律，joint 中该常数已成研究方法决策输入）。离线蓝本
        # 保留 config.source_average_decode_length 仅作对拍锚（不消费）。
        self._joint_horizon = CausalHorizonEstimator(
            cold_start_default_forward_steps=32)
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
        # §5.4（ARRIVAL 批内缓存，零语义性能修改方案 2026-10-16）两级
        # 缓存开关（缺省关 = 行为与改动前完全一致，语义安全底线）：
        # SH_JOINT_ARRIVAL_CACHE=1 生产启用——P 准入（_try_admit_request）
        # 的代价模型重建（L1，有界 2）与候选选择记录（L2，有界）批内
        # 复用；select_instance_and_action/estimate_P_with_continuation
        # 本体零改动，边界决策（带 obligation_resolver）不入缓存。
        # SH_JOINT_ARRIVAL_CACHE_VERIFY=1 影子双路重算逐字段断言（缓存
        # 命中也完整重算 uncached 路径并对每候选 cost/breakdown 断言相
        # 等，剔除 projection_stats.wall_ns 宿主墙时位）——零收益全检
        # 查，姿态与命名对齐 SH_ADMIT_GATE_VERIFY（改法D）。
        self._joint_arrival_cache_enabled = (
            os.environ.get("SH_JOINT_ARRIVAL_CACHE") == "1")
        self._joint_arrival_cache_verify = (
            os.environ.get("SH_JOINT_ARRIVAL_CACHE_VERIFY") == "1")
        # 两级缓存本体 + 命中/未命中计数（run 末 joint_decision_metrics
        # 行披露；两开关均关时不落披露键，行形态与改动前逐字节一致）。
        self._joint_arrival_model_cache: dict = {}
        self._joint_arrival_selection_cache: dict = {}
        self._joint_arrival_cache_stats = {
            "l1_hits": 0, "l1_misses": 0,
            "l2_hits": 0, "l2_misses": 0,
            "verify_checks": 0, "verify_mismatches": 0,
        }
        # 同状态对照 probe（run_scripts/joint_same_state_probe.py，式(24)
        # 选择损失验收）的验证专用强制钩子：JOINT_FORCE_BOUNDARY_REQUEST/
        # ACTION 成对设置时武装——只把指定请求边界决策（P 准入与 D 边界
        # 两入口）的最终选择强制为指定动作。这是测量仪器不是双路径开关
        # （工作区"直接替换规范"明示豁免）：不改状态构造、不改计价（候选
        # 照常全量计算与披露），强制动作出适用面 fail-closed（仪器不得
        # 制造非法状态）；缺省 off，行为零变化。
        (self._force_boundary_request,
         self._force_boundary_action) = _parse_force_boundary_env(os.environ)
        # P 入口单点强制钩子（修复轮二任务 7）：REQUEST env 复用 + P 入
        # 口专用 ACTION env；与边界钩子互斥武装（二义即启动失败）。
        # 缺省 off，行为零变化。
        (self._force_entry_request,
         self._force_entry_action) = _parse_force_entry_env(os.environ)
        # W8 instance×action 精确强制钩子（测量仪器，方案 §11.3）：指定
        # (request, instance, action) 三元组——适用面核验在裁决点做
        # （指定实例不在候选面 = 显式失败，不静默重选实例）。缺省 off。
        self._force_boundary_target = _parse_force_target_env(
            os.environ, FORCE_BOUNDARY_TARGET_ENV, FORCE_BOUNDARY_ACTION_ENV)
        self._force_entry_target = _parse_force_target_env(
            os.environ, FORCE_ENTRY_TARGET_ENV, FORCE_ENTRY_ACTION_ENV)
        # 一次映射对照臂（JOINT_BOUNDARY_POLICY，joint_config 解析）与
        # probe 强制钩子互斥武装裁决（仪器二义 fail-closed——上）。
        _check_boundary_policy_instrument_conflict(
            self.joint_config.boundary_policy,
            self._force_boundary_action, self._force_entry_action,
            self._force_boundary_target, self._force_entry_target)
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
        # D16 有界性(v2.0 §17.1):显式 ≤0 fail-closed 拒绝(0=无界诊断
        # 配置会预建整轮节点)——缺省/未设置按协议推导合法值,只拒显式
        # ≤0(joint_config.parse_train_max_iter 单一裁决点)。
        self._train_max_iter = parse_train_max_iter(
            os.environ.get("SH_TRAIN_MAX_ITER"))
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
        # W5（2026-09-29，A3）：当前交付引用捕获——决策级 DecisionSnapshot
        # （build_decision_snapshot）的 delta 输入（arrivals 白名单投影/
        # delivery_sequence/tick 对齐本交付；本体内全部决策均发生在本
        # 方程内，快照与决策同刻）。引用共享（不拷贝；只读消费）。
        self._joint_current_delta = delta
        # W5（W6 接线 §1）：到达统计的模拟时间推进（幂等；空桶按模拟
        # 时间衰减，不依赖下一请求到达——方案 §8.4）+ 非到达事件披露
        # （当前队列/当前请求剩余 D/已提交通信；只落披露计数，绝不进 λ）。
        self._joint_arrival_stats_or_default().advance_to(tick)
        self._joint_note_excluded_events(tick)
        # W7：新增需求预测的成熟驱动（到达窗桶预测在桶末端过后用类别累
        # 计差分结算误差，恰一次；shadow——不影响选择；none 档零行为）。
        self._joint_forecast_mature(tick)
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
            # M16（face-LRU 审计同构修复,2026-09-29）：alarm 基准 = 父请求
            # 物理 code-4 完成 tick（delta.deferred_from_tick 携带本 epoch
            # 的形成 tick = watch fire = code-4 完成,C++ 端不变量"epoch
            # 事件全部形成于 deferred_from_tick";普通同 tick 交付该值为
            # 0——交付 tick 即形成 tick）,与指标层 resolve_arrival
            # （MetricCollector: parent completion + interval）同源同式。
            # deferral 交付（Step 1-11 T→T+1）下交付 tick 晚于物理完成
            # tick,原"交付 tick + interval"恒晚于指标声明到达 1 tick,
            # queue_ns/e2e_ns 系统性高估;alarm 计算时刻已过基准点（负/
            # 零间隔）由 ingress 既有"立即触发"钳位路径处理。
            self._complete_requests(
                completed_now, tick,
                alarm_base_tick=delta["deferred_from_tick"] or tick)
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
            # D20（§14.5/§16.4）：完成信号按边界提交点锁定的完成归属
            # 实例核销；未经边界的完成信号 = 状态机破损，fail-closed。
            attribution = runtime.completion_instance_index
            if attribution is None:
                raise RuntimeError(
                    "completion signal for request {} without a boundary "
                    "completion attribution (stage {})".format(
                        request_id, runtime.joint_stage))
            signaled.setdefault(attribution, set()).add(request_id)
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
                        # W5（W6 接线 §2 删失）：列车核销推进的已暴露
                        # decode 进度（真实前向步消费）——observe_censored
                        # 只登记"存活至 d、尚未结束"，不入完成样本
                        # （exposed_progress = decode_tokens_consumed 执行
                        # 侧进度字段；隐藏真值不经此通道）。
                        self._joint_length_history_or_default().observe_censored(
                            session_id=runtime.session_id,
                            exposed_progress_steps=(
                                runtime.decode_tokens_consumed),
                            category=runtime.joint_revealed_category,
                            tick_ns=tick)
                        # joint（§3.1）：decode KV 按实际消费进展因果增长
                        # （写入前经 T+E 真实释放准备空间；R14：容量类
                        # 失败转停滞/唤醒，不再 fail-closed）。
                        self._joint_grow_decode(runtime, instance_index)
                    for request_id in train["exit_set"]:
                        runtime = self.runtime_by_request_id[request_id]
                        if runtime not in state.active_decode_lookup:
                            raise RuntimeError(
                                "exiting member {} is not in the decode "
                                "batch".format(request_id))
                        state.active_decode_lookup.discard(runtime)
                        state.active_decode.remove(runtime)
                        # W5（§7.4）：退出迭代 = 待结算（D_RUNNING ->
                        # SETTLE_PENDING；consolidate 的物理 merge =
                        # settle 子协议）。
                        runtime.joint_stage = JOINT_STAGE_SETTLE_PENDING
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

    def _consume_boundary_first_token(self, runtime) -> None:
        """C4（§19.1 首 token 契约）：首个 D 前向消费 pending 首 token
        恰一次。调用点 = 列车规划期成员定容（原子冻结后即发射——首 D
        前向排定时刻）；经 ``BoundaryFirstTokenState.consume_once`` 落
        账，二次消费 fail-closed。无边界状态（直构/遗留测试面）为无操
        作——消费事实以边界状态 dict 在场为界。"""
        state = getattr(runtime, "joint_boundary_first_token", None)
        if state is None or state.get("consumed"):
            return
        consumed = BoundaryFirstTokenState(**state).consume_once()
        runtime.joint_boundary_first_token = {
            "issued": consumed.issued,
            "consumed": consumed.consumed,
            "rng_advances": consumed.rng_advances,
        }

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
                # D20 构图守卫：仍有 chunk 工作的队列头不得处于 P 后
                # 阶段（D_*/MERGE_*/DONE）——越过 P_COMPUTE_DONE 的请求
                # 已出队，其未定/已定的 D 永不作为 prefill 工作入列车。
                if runtime.joint_stage not in (
                        JOINT_STAGE_P_DECISION, JOINT_STAGE_P_COMMITTED,
                        JOINT_STAGE_P_RUNNING):
                    raise RuntimeError(
                        "train composer: prefill queue head {} carries "
                        "chunk work in post-P stage {}".format(
                            runtime.request_id, runtime.joint_stage))
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
            # C4（§19.1）：首个 D 前向消费 pending 首 token 恰一次——
            # 成员进入列车规划 = 首 D 前向排定（原子冻结后即发射），
            # 此刻经 BoundaryFirstTokenState.consume_once 落账（二次消
            # 费 fail-closed；token 序列解释与图侧 span 一致）。
            self._consume_boundary_first_token(runtime)
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
            # R13：span 上下文基 = 真实历史基数（stay/copy 同口径）。
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
                # D20（v2.0 §14.5/§16.4）构图器边界守卫（先验全体成员，
                # 拒绝时不做任何账面变更）：只有完成阶段事务提交
                # （D_COMMITTED）的成员才可入列车——尚未决定位置/动作
                # 的边界请求（D_DECISION_PENDING/D_PLAN_SELECTED 及任何
                # 更早阶段）永远不进 P/decode 列车；选定成员的迁移腿 +
                # 共享 readiness barrier 即目标实例上的第一步门（列车
                # 体在其后）。
                for joiner in joiners:
                    if joiner.joint_stage != JOINT_STAGE_D_COMMITTED:
                        raise RuntimeError(
                            "train composer: request {} cannot join "
                            "instance {} in stage {} (pre-commit boundary "
                            "requests must never be composed into a "
                            "train)".format(
                                joiner.request_id, state.index,
                                joiner.joint_stage))
                state.pending_decode_ready.clear()
                state.active_decode.extend(joiners)
                for runtime in joiners:
                    state.active_decode_lookup.add(runtime)
                    # D18：加入列车 = D_COMMITTED -> D_RUNNING（此后本
                    # 轮 D 不再选点——无第三次位置决策）。
                    runtime.joint_stage = JOINT_STAGE_D_RUNNING
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
        # 纯度排除（local_hit 无传输不计）。
        plan["history_transfer_gated"] = bool(
            plan.get("head_first_chunk")
            and head_runtime is not None
            and any(
                transfer.kind != "local_hit" and transfer.total_bytes > 0
                for transfer in (head_runtime.history_transfers or ())))
        # N5（2026-09-23 复核审计5）：上一轮 merge 尾门（R11(ii)——下一
        # 轮 interval gate 锚定上一轮 merge_done 标记节点，graph 层
        # batch_builder）——merge watch 未交付时本列车 emit
        # 后物理计算被 hold 至 merge 完成 + interval，span 混入门控等待
        # 且该样本会初始化 γ/更新双因子通道——纯度排除（与 joiner/恢
        # 复门/传输门同哲学：不可因果分离不收样）。
        # O3（2026-09-23 终轮审计，判据扩域到"本实例"粒度）：门控等待
        # 污染源两类——① 未交付 merge watch：R11(ii) interval gate 按
        # **实例 frontier** 锚定（非按 session），同实例任意 session 的
        # 未交付 merge watch 都 hold 本实例下一列车（原"同 session"匹配
        # 漏跨 session 污染）；② frontier 未交付完成批尾段：完成批池写
        # 尾（completion_evictions / merge 传输的 remote_store）在
        # graph.pending_store_tails 在案（GB 登记 /
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
        first_token = self._first_token_plan(plan, joiner_runtimes)
        if first_token is not None:
            train_plan["first_token"] = first_token
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
        # (self._retry was removed with the 2026-10-01 dead-code audit: the
        # offline blueprint's arrival-forces-recheck signal had no reader in
        # the online path -- every delivery re-runs _admit_pass anyway.)
        # W5（W6 接线 §1）：真实到达进 ArrivalStatistics（每真实到达恰
        # 一次——上方重复到达 fail-closed 保证；turn_index==0 ⇒ 新
        # session 首请求、>0 ⇒ 闭环后继返回；revealed_type 只用已揭示
        # 来源 = 上一轮完成时 mark_complete 落账的 session.next_request_
        # type，不读 manifest/next_request 对象的隐藏真值）。
        self._joint_arrival_stats_or_default().observe_arrival(
            now_ns=tick,
            turn_index=int(runtime.turn_index),
            revealed_type=self._joint_revealed_category(
                runtime.session_id),
        )
        # 到达时刻的已揭示类别随 runtime 冻结（完成样本的 category 键
        # 同源——P/D 同刻同实例同一规则）。
        runtime.joint_revealed_category = self._joint_revealed_category(
            runtime.session_id)

    def _joint_revealed_category(self, session_id: str):
        """已揭示类型键（W6 category/revealed_type 白名单通道）。

        唯一来源 = kv 会话的 ``next_request_type``（上一轮完成时
        ``mark_complete(next_request_type=...)`` 落账——完成响应揭示的
        类别键；与 T 机制 typed 逐出序同源）。None（首请求/未揭示）=
        无类别键（回退链 run/prior 层）。"""
        session = self.kv_manager._sessions.get(session_id)
        if session is None:
            return None
        value = getattr(session, "next_request_type", None)
        return value if value in ("human", "tool") else None

    # ---- W5（W6 接线）：__new__ 测试替身的惰性兜底（真 __init__ 恒构造
    # 三件套；替身经此按 PredictionParams 缺省档构造并驻留——缺省档与
    # 生产 JOINT_PREDICTION_CONFIG 缺省 preset 同源，零平行常数）。
    def _joint_length_history_or_default(self):
        estimator = getattr(self, "_joint_length_history", None)
        if estimator is None:
            from joint.history_estimator import (
                ConditionalRemainingLengthEstimator)
            estimator = ConditionalRemainingLengthEstimator(
                self._joint_prediction_config_or_load().params)
            self._joint_length_history = estimator
        return estimator

    def _joint_arrival_stats_or_default(self):
        stats = getattr(self, "_joint_arrival_stats", None)
        if stats is None:
            from joint.history_estimator import ArrivalStatistics
            stats = ArrivalStatistics(
                self._joint_prediction_config_or_load().params)
            self._joint_arrival_stats = stats
        return stats

    def _joint_service_samples_or_default(self):
        recorder = getattr(self, "_joint_service_samples", None)
        if recorder is None:
            from joint.history_estimator import ServiceSampleRecorder
            recorder = ServiceSampleRecorder(
                self._joint_prediction_config_or_load().params,
                epoch_log=self._joint_epoch_log_or_default())
            self._joint_service_samples = recorder
        return recorder

    def _joint_epoch_log_or_default(self):
        """§8.5 epoch/分布变化登记（惰性；真 __init__ 恒构造，替身经此
        按缺省档兜底——与 W6 三件套同纪律）。"""
        log = getattr(self, "_joint_epoch_log", None)
        if log is None:
            from joint.history_estimator import EpochLog
            log = EpochLog(self._joint_prediction_config_or_load().params)
            self._joint_epoch_log = log
        return log

    def _joint_note_excluded_events(self, tick: int) -> None:
        """W6 §1 非到达事件披露（每决策批一次；shadow——只统计不进选择
        规则）：当前队列占用 / 当前请求剩余 D / 已提交通信。"""
        queued = sum(len(state.qp) for state in self.instances)
        active = sum(
            len(state.active_decode) for state in self.instances)
        if queued:
            self._joint_arrival_stats_or_default().note_excluded_event(
                kind="current_queue", now_ns=tick)
        if active:
            self._joint_arrival_stats_or_default().note_excluded_event(
                kind="current_request_remaining_decode", now_ns=tick)
        if self._batch is not None and self._batch.get("watches"):
            self._joint_arrival_stats_or_default().note_excluded_event(
                kind="submitted_communications", now_ns=tick)

    # ---- W7（新增需求预测）：引擎/事实/闭环下界/成熟驱动（最小面）----

    def _joint_forecast_engine_or_default(self):
        """背景预测引擎（惰性：仅 future_demand_mode != none 时构造；
        组件 = W6 到达统计 + 预测配置 + 共享 epoch 登记账本，全部
        PredictionParams 单源）。§8.5（2026-09-30 修复）：epoch_log 由
        调度器装配注入——distribution_shift 回退分支自此生产可达，不再
        只能靠测试手工注入复现。"""
        engine = getattr(self, "_joint_forecast_engine", None)
        if engine is None:
            from joint.background_forecast import (
                BackgroundForecastEngine)
            engine = BackgroundForecastEngine(
                arrival_stats=self._joint_arrival_stats_or_default(),
                prediction_config=(
                    self._joint_prediction_config_or_load()),
                epoch_log=self._joint_epoch_log_or_default(),
                length_estimator=self._joint_length_history_or_default())
            self._joint_forecast_engine = engine
        return engine

    def _joint_forecast_mature(self, tick: int) -> None:
        """W7 成熟驱动挂点（每决策批一次；none 档零行为——不构造引擎）。"""
        if (self._joint_prediction_config_or_load().future_demand_mode
                == "none"):
            return
        engine = getattr(self, "_joint_forecast_engine", None)
        if engine is not None:
            engine.mature(tick)

    def _joint_completion_maturity_or_default(self):
        """cfg1（§10.3）：P/D 完成时间预测的在线成熟账本（惰性；实际
        执行候选在决策行发射点登记、settle_done 闭合点配对实测——
        forecast_validation.CompletionMaturityLedger 单一机制面）。"""
        ledger = getattr(self, "_joint_completion_maturity", None)
        if ledger is None:
            from joint.forecast_validation import CompletionMaturityLedger
            ledger = CompletionMaturityLedger()
            self._joint_completion_maturity = ledger
        return ledger

    def _joint_forecast_status_decision(self) -> Optional[dict]:
        """cfg1（§10.3/W7 收尾披露）：run 末 forecast_status 行载荷。

        * ``forecast_engine``：背景预测引擎披露（disclosure()——maturity
          账/开销账/回退链/排除事件；此前 run 结束即丢的全仓唯一生产
          消费出口）；none 档未构造（惰性）= None 如实披露。
        * ``completion_time_forecasts``：P/D 完成预测成熟账（本 run 的
          直接真值配对结果——绝对/带方向/分位误差与归一化分母）。
        无引擎且无配对样本时返回 None（调用方不落行——off/旧形态逐字
        节不变；有任一 Forecast 机制即恒落行，验收必需项）。"""
        decision: dict = {}
        engine = getattr(self, "_joint_forecast_engine", None)
        decision["forecast_engine"] = (
            engine.disclosure() if engine is not None else None)
        ledger = getattr(self, "_joint_completion_maturity", None)
        if ledger is not None and ledger.has_entries():
            decision["completion_time_forecasts"] = ledger.as_dict()
        if engine is None and (
                ledger is None or not ledger.has_entries()):
            return None
        decision["future_demand_mode"] = (
            self._joint_prediction_config_or_load().future_demand_mode)
        return decision

    def _joint_forecast_floor_offset(self, now_ns: int) -> int:
        """闭环返回约束下界（§8.3：处理中 session 的下一轮不早于其
        service_done 注入——共享等待假设，不预知真实返回 tick）。

        冻结规则：now + min(在途请求 n̂ × 名义单步服务)。名义单步取
        context=1 标量率（最快可能步）⇒ 该下界恒 ≤ 真实 service_done
        （保守方向：只会把注入推晚到不早于该下界，不会早于真实完成）。
        无在途请求 = 0（已完成的 session 返回随时可能发生）。
        """
        runtimes = [
            runtime for runtime in self.runtime_by_request_id.values()
            if runtime is not None]
        if not runtimes:
            return 0
        step_ns = self._decode_step_load_ns(
            instance_size=self.topology.instances[0].size, context=1)
        estimates = []
        for runtime in runtimes:
            estimate = self._joint_conditional_estimate(runtime)
            estimates.append(
                int(estimate.remaining_forward_steps) * max(0, int(step_ns)))
        return min(estimates) if estimates else 0

    def _joint_conditional_estimate(self, runtime):
        """W5 消费切换（W6 §2）：条件剩余长度的正式查询入口。

        P 准入与 D 边界共用本入口（P/D 同刻同实例同一规则与版本，方案
        §8.2）；``_joint_horizon``（CausalHorizonEstimator）保留为诊断/
        先验来源。exposed_progress = ``decode_tokens_consumed``（执行侧
        已暴露进度；隐藏真值 decode_length 不经此通道——接口无该形参）。
        """
        # getattr 防御 __new__ 壳替身（SimpleNamespace runtime 无消费
        # 进度字段 = 0 前向步口径；真 runtime 恒有该 slot）。
        return self._joint_length_history_or_default().estimate_conditional(
            session_id=runtime.session_id,
            exposed_progress_steps=int(
                getattr(runtime, "decode_tokens_consumed", 0) or 0),
            category=getattr(
                runtime, "joint_revealed_category", None),
        )

    def _on_prefill_drain(self, request_id: str, tick: int) -> None:
        """offline: face_scheduler.py prefill 末 chunk 完成分支
        （decode 固定同实例），逐行迁移；拼 batch 改造：decode
        段发射移至加入列车（_plan_and_emit_trains → emit_iteration_train），
        成员先进 pending_decode_ready（§3.2 KV 就绪栅栏），DECODE_
        COMPLETION watch 由列车 exit 标记承载。

        D20（v2.0 §14.5/§16.4）：本边界是 P→D 阶段提交的入口——重复
        P 完成通知幂等忽略（阶段 epoch 门）；P 末尾到 D 提交期间本请求
        基础 KV 持阶段引用保护（不进冷 victim 集合）；预约换址/resize
        经 ``commit_stage_reservation_transaction`` 幂等 epoch 事务恰一
        次提交；完成归属在提交点锁定。"""
        runtime = self.runtime_by_request_id[request_id]
        if runtime.prefill_complete_ns is not None:
            # D20 幂等门：同一 P_COMPUTE_DONE 的重复通知（交付层重复/
            # 在途 ack 双交付）不再重入边界——状态机已越过 P_COMPUTE_DONE
            # （首次进入时已写 prefill_complete_ns），重跑会二次执行 D
            # 决策与预约事务。落披露行可观测，无任何账本副作用。
            self.log_decision(
                {"kind": "boundary_duplicate_ignored",
                 "request_id": request_id, "priority": 0},
                tick,
                decision={
                    "joint_stage": runtime.joint_stage,
                    "joint_boundary_epoch": runtime.joint_boundary_epoch,
                    "rule": "duplicate P_COMPUTE_DONE notification is "
                            "idempotent (v2.0 sec.16.4 stage epoch)",
                },
            )
            return
        state = self.instances[runtime.prefill_instance_index]
        # R15-1：准入主链在途流在 drain 栅栏通过后释放；独立逐出支链
        # 仍可能在飞，owner=#evict 只由其尾 watch 释放。
        self._release_transfer_flows(request_id)
        # C11：准入主链配额流 settle；#evict 支链配额由尾 watch 释放，
        # merge 预留生命周期到完成边界。
        if self._quota_tracker is not None:
            self._quota_release_admission_phase(request_id, tick)
        # (runtime.prompt_tokens_processed -- the drain-side actual-work
        # ledger -- was removed with the 2026-10-01 dead-code audit: it was
        # write-only, never reaching plan_dict/decision rows/consumers.)
        runtime.remaining_chunks = 0
        # offline blueprint FCFS qp popleft（离线 iteration 串行化保证
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
        # joint：工作副本上下文按动作推进（stay/copy 至
        # prefill_context）。
        runtime.prefill_evictions = self.kv_manager.expand_prefill(
            session_id=runtime.session_id,
            instance_index=state.index,
            context_tokens=self._joint_working_context(runtime),
            trigger_request_id=runtime.request_id,
            reservation_request_id=runtime.request_id,
            at_ns=tick,  # M4：restore/adaptive 披露行携带决策时刻
        )
        self._bump_kv_ledger_epoch()  # 改法D：KV 变更点 1/9（expand_prefill）
        self.graph.sync_pending_history_after_evictions(
            runtime.prefill_evictions)
        # ==== D18：P→D 真实边界二次决策（v2.0 §2.3/§16，2026-09-26）====
        # 强制 P/D 同址废除。P_COMPUTE_DONE（末 chunk 采样本轮首 token、
        # prefill 工作副本就绪、首笔 D 未提交）：已知终止 → 直接结算
        # （不做无意义的 D 选点）；否则 D_DECISION_PENDING——以真实边界
        # 快照做恰好一次全候选遍历（式 1b），原地继续恒为候选且平局
        # 优先（式 1c），D 提交（第一笔不可撤销操作）后锁定，无第三次
        # 选点。P 已完成的时间/计算量/传输全部沉没；切换目标的迁移腿是
        # t_B 后的新工作，不回填为 P 期间重叠。
        runtime.joint_stage = JOINT_STAGE_P_COMPUTE_DONE
        # D20（§14.5/§16.4）：边界阶段 epoch 前进 + 基础 KV 阶段引用
        # 保护——P 末尾到 D 提交期间本请求基础 KV 不进冷 victim 集合
        # （同 tick 其他请求的容量事务、以及取消/在途 ack 竞合下的任何
        # 逐出都不得选中它；"暂时没有活跃 D"不解除引用）。
        runtime.joint_boundary_epoch += 1
        # 边界块整体异常安全（M2/D18 fail-closed 纪律）：自阶段引用
        # 获取起，任何异常先幂等释放该引用再重抛——fail-closed 语义
        # 不变，但 P→D 窗口的基础 KV 保护不得随异常泄漏（引用与请求
        # 生命周期同终，请求不悬挂在持引用的 D_DECISION_PENDING）；
        # D 提交点后引用已正常闭合，此处幂等空放。
        try:
            runtime.joint_stage_reference_id = (
                self.kv_manager.acquire_stage_reference(
                    runtime.session_id,
                    request_id=runtime.request_id,
                    stage_epoch=runtime.joint_boundary_epoch,
                    reason="pd_boundary_base_kv"))
            if self._joint_boundary_terminal(runtime):
                self._settle_terminal_at_boundary(runtime, state, tick)
                return
            runtime.joint_stage = JOINT_STAGE_D_DECISION_PENDING
            # 墙时分段（2026-10-05 埋点）premodel 起点：边界视图构造 →
            # 阶段工作视图 → 边界代价模型构建，至候选枚举起点。
            boundary_premodel_start = time.perf_counter_ns()
            boundary = self._joint_boundary_view(runtime, state)
            # D19：边界阶段工作视图 + 继承义务表（执行侧材料化；落边界
            # 决策记录披露——决策行 schema 前向兼容）。
            stage_work, inherited_obligations = (
                self._joint_boundary_stage_view(runtime, state, boundary))
            # ---- D 边界选点（两形态，JOINT_BOUNDARY_POLICY 单一裁决；
            # 定义源 joint.joint_config.BOUNDARY_POLICIES）----
            # * off（缺省）= 两阶段闭环（D18 主语义）：以真实边界快照做
            #   恰好一次全候选遍历（式 1b），原地继续平局优先（式 1c）。
            # * p-continuation（测量仪器，一次映射对照臂；生产 run 不设）
            #   = 不做二次全候选重估，直接按 P 准入
            #   continuation 披露的默认延续形态提交——decode_instance =
            #   prefill_instance、decode_action = stay@i_P。
            #   一次映射下 D 提交后的完成归属/构图器/merge 链路与两阶段
            #   的延续分支完全同路（同一 commit 路径——延续 = 原地继续，
            #   本来就没有迁移腿）。掩码交互：P 准入已按掩码过滤（P 动作
            #   在掩码内）；延续动作 stay 已由 joint_config __post_init__
            #   钉在掩码内（p-continuation 要求掩码放行 stay），故无需
            #   边界重查。已知终止在上方已直结结算，不进本分支。
            boundary_policy = self.joint_config.boundary_policy
            forced_probe_note = None
            causal_prestate = None
            if boundary_policy == BOUNDARY_POLICY_P_CONTINUATION:
                # ACC 照常记样本：candidates=0 = 无候选遍历的如实披露
                # （控制开销对照读数 = 两 run D 阶段账本差，probe 汇总
                # 消费）。无 D 预测代价——不虚构候选代价（决策行
                # cost_ns 真值 NA）。
                boundary_decision_wall_start = time.perf_counter_ns()
                chosen_action = ACTION_STAY
                self.predictor_cost.note_decision(
                    DECISION_STAGE_D,
                    time.perf_counter_ns() - boundary_decision_wall_start,
                    candidates=0)
                chosen = None
                chosen_instance = state.index
                chosen_cost_ns = None
                boundary_candidates: tuple = ()
            else:
                boundary_cost_model = self._joint_boundary_cost_model(
                    runtime, boundary, tick,
                    obligations=inherited_obligations)
                # ACC（验收纪律第 3 类，v2.0 §19.3）：边界二次决策自身的
                # 宿主墙时样本（D 阶段分列；与准入 P 样本同一账本、run 末
                # 汇总 P50/P95/P99）。量纲只如实披露，不外推片上决策时延。
                # 墙时分段（2026-10-05 埋点）：候选枚举遍历段
                # （select_decode_target_and_action 全候选计价）。
                boundary_decision_wall_start = time.perf_counter_ns()
                record = select_decode_target_and_action(
                    mode=self._joint_mode,
                    cost_model=boundary_cost_model,
                    boundary=boundary,
                    # D17 schema：边界阶段 epoch 与继承义务 ID 逐候选透传
                    # 进 breakdown 披露（纯披露通道——不计价、不改选择）。
                    decision_epoch=runtime.joint_boundary_epoch,
                    inherited_obligation_ids=tuple(
                        obligation.obligation_id for obligation in
                        inherited_obligations),
                )
                _boundary_enum_elapsed = (
                    time.perf_counter_ns() - boundary_decision_wall_start)
                self._note_wall_split(
                    premodel_ns=(boundary_decision_wall_start
                                 - boundary_premodel_start),
                    enum_ns=_boundary_enum_elapsed, boundary=True)
                self.predictor_cost.note_decision(
                    DECISION_STAGE_D,
                    time.perf_counter_ns() - boundary_decision_wall_start,
                    candidates=len(record.candidates))
                # cfg2（§5.6/§10.1 计算量组）：D 边界全候选遍历的前推
                # 计算量汇入 run 级累计（逐候选面在决策行 breakdown 内）。
                # stage="D"（2026-10-05 埋点）：投影计数按阶段分列。
                self._joint_accumulate_projection_cost(
                    boundary_cost_model, stage="D")
                # 同状态 probe 强制钩子（D 入口，测量仪器）：选择器产出后
                # 强制最终选择——candidates 不动（式 1b 计价输入与披露面
                # 不变），强制动作出适用面即 fail-closed（不得制造非法
                # 状态）。与 p-continuation 互斥武装（__init__ 裁决）。
                self._joint_last_probe_causal_prestate = None
                record, forced_probe_note = self._apply_forced_boundary_choice(
                    record, runtime.request_id, DECISION_STAGE_D,
                    now_ns=tick, capture_causal_prestate=True)
                causal_prestate = getattr(
                    self, "_joint_last_probe_causal_prestate", None)
                chosen = record.chosen
                chosen_instance = chosen.instance_index
                chosen_action = chosen.action
                chosen_cost_ns = chosen.cost_ns
                boundary_candidates = record.candidates
            runtime.joint_stage = JOINT_STAGE_D_PLAN_SELECTED
            selected = chosen_instance
            boundary_fallback = False
            stage_transaction = None
            reservation_move_evictions = ()
            decode_move_evictions = ()
            # ---- D20 跨阶段预约事务（§14.5：同一事务内创建新目标承诺、
            # 转用/换址承诺、释放未被物理工作占用的旧承诺；§16.4：阶段
            # epoch 保证一次提交）→ 工作副本迁移。同 epoch 重复提交原子
            # 重放回执；改变方案的候选提交失败且无不可逆工作时，保留继续
            # 方案的合法状态（abort 回执记录 epoch 结局，§16.4）——已提交
            # 逐出是真实容量事件照常入账（含失败前半边的换址逐出），继续
            # 态自身不受影响。
            stage_history_tokens = int(boundary.history_tokens)
            stage_input_tokens = 0
            # W5（§7.2 事务顺序第 1 步 / C5 控制延迟复查）：决策时点版本
            # 面（runtime = kv_ledger_epoch（调度器侧复查，下方先查）；
            # layout = 目录事务版本；reservation = 阶段纪元表）——提交点
            # 经 expected_versions 复查，漂移（同 tick 竞争先提交）=
            # KVStageVersionConflict → 回退继续方案（§7.2 刷新重评语义
            # 在本仓形态 = 回退到合法接续）。
            boundary_versions = self._joint_boundary_versions()
            try:
                if (boundary_versions is not None
                        and boundary_versions[0]
                        != getattr(self, "_kv_ledger_epoch", None)):
                    # runtime 维度（KV 账本纪元）在调度器侧复查：决策→
                    # 提交之间 KV 变更点已触发 = 快照陈旧（§7.2 第 1 步）。
                    raise _VersionDrift(
                        "runtime epoch drifted between decision and commit "
                        "({} -> {})".format(
                            boundary_versions[0],
                            getattr(self, "_kv_ledger_epoch", None)))
                stage_transaction = (
                    self.kv_manager.commit_stage_reservation_transaction(
                        request_id=runtime.request_id,
                        target_instance_index=selected,
                        stage_action=chosen_action,
                        stage_epoch=runtime.joint_boundary_epoch,
                        history_tokens=stage_history_tokens,
                        input_tokens=stage_input_tokens,
                        at_ns=tick,  # M4：adaptive 决策行携带决策时刻
                        **({"expected_versions": boundary_versions[1]}
                           if boundary_versions is not None and
                           boundary_versions[1] is not None else {}),
                    )
                )
                reservation_move_evictions = stage_transaction.evictions
                (runtime.prefill_decode_transfer,
                 decode_move_evictions,
                 boundary_transition) = (
                    self.kv_manager.commit_boundary_decode_transition(
                        session_id=runtime.session_id,
                        p_instance_index=boundary.p_instance_index,
                        target_instance_index=selected,
                        action=chosen_action,
                        history_tokens=int(boundary.history_tokens),
                        trigger_request_id=runtime.request_id,
                        reservation_request_id=runtime.request_id,
                        at_ns=tick,
                    )
                )
            except (KVCapacityError, KVStageVersionConflict,
                    _VersionDrift) as exc:
                if selected == state.index:
                    # 继续态自身的失败非回退语义（容量/版本同款），
                    # fail-closed 上抛。
                    raise
                # W5（§7.2 第 1 步）：KVStageVersionConflict/_VersionDrift
                # = 决策→提交间版本漂移（同 tick 竞争先提交）——与容量
                # 失败同款回退结构（abort 半边 + 回退继续方案），rule
                # 字串分列披露失败类别。
                # 回退换址半边（迁移腿在加入列车才发射，此刻零字节移动）：
                # D20 abort 事务把本 epoch 结局记为 aborted（足迹恢复提交
                # 前快照），同 epoch 重复通知重放该结局不再重做半边提交；
                # 失败候选已提交的逐出照常清账（bump + 图同步 + 支链发射）。
                # 换址半边已提交的逐出（reservation_move_evictions）保持
                # 在账——照常走下方统一同步/入图路径，不因回退丢失。
                stage_transaction = (
                    self.kv_manager.abort_stage_reservation_transaction(
                        request_id=runtime.request_id,
                        source_instance_index=state.index,
                        stage_epoch=runtime.joint_boundary_epoch,
                    )
                ) or stage_transaction
                fallback_evictions = tuple(getattr(exc, "evictions", ()) or ())
                if fallback_evictions:
                    self._bump_kv_ledger_epoch()
                    self.graph.sync_pending_history_after_evictions(
                        fallback_evictions)
                    self._emit_eviction_only_nodes(
                        fallback_evictions, tick,
                        trigger_request_id=runtime.request_id)
                boundary_fallback = True
                # cfg2（§5.6"版本失效、失败重评等额外遍历须累计报告"）：
                # 提交点版本失效/容量回退 = 该边界的一次额外遍历（回退
                # 继续方案），run 级累计披露。
                self._joint_boundary_fallback_count += 1
                selected = state.index
                (runtime.prefill_decode_transfer,
                 decode_move_evictions,
                 boundary_transition) = (
                    self.kv_manager.commit_boundary_decode_transition(
                        session_id=runtime.session_id,
                        p_instance_index=boundary.p_instance_index,
                        target_instance_index=selected,
                        action=ACTION_STAY,
                        history_tokens=int(boundary.history_tokens),
                        trigger_request_id=runtime.request_id,
                        reservation_request_id=runtime.request_id,
                        at_ns=tick,
                    )
                )
                fallback_rule = (
                    "Eq.1c fallback: continuation stays legal "
                    "when the switch commit fails without "
                    "irreversible work"
                    if isinstance(exc, KVCapacityError)
                    else "sec.7.2 step-1 version conflict: decision-time "
                         "layout/reservation versions drifted before the "
                         "commit point (same-tick race); continuation "
                         "stays legal")
                self.log_decision(
                    {"kind": "boundary_commit_fallback",
                     "request_id": runtime.request_id, "priority": 0},
                    tick,
                    decision={
                        "failed_instance_index": chosen_instance,
                        "failed_action": chosen_action,
                        "fallback_instance_index": selected,
                        "stage_epoch": runtime.joint_boundary_epoch,
                        "reason": str(exc),
                        "failure_class": (
                            "kv_capacity_error"
                            if isinstance(exc, KVCapacityError)
                            else "stage_version_conflict"),
                        "rule": fallback_rule,
                    },
                )
            if boundary_fallback:
                # 实际提交态 = 原地继续（回退即继续方案的合法状态）。
                runtime.joint_decode_action = ACTION_STAY
                runtime.joint_decode_is_continuation = True
            else:
                runtime.joint_decode_action = chosen_action
                # 一次映射对照臂：提交即默认延续（is_continuation 恒
                # True）；两阶段沿用选择器的继续标记（式 1c）。
                runtime.joint_decode_is_continuation = (
                    True if boundary_policy == BOUNDARY_POLICY_P_CONTINUATION
                    else bool(chosen.is_continuation))
                # §5.5（2026-09-30 修复）：D 提交成功点的 θ 预测暂存
                # （覆盖 P 暂存——D 阶段容量压力以最新选中候选为准；
                # 回退分支不暂存，失败方案的预测不参与对账）。
                self._stash_capacity_theta_prediction(runtime, chosen)
            # C4（D10/D19）：选中动作的运行时元数据——
            # 边界首 token
            # 状态（issued 待首个 D 前向消费恰一次）。
            runtime.joint_boundary_first_token = {
                "issued": True,
                "consumed": False,
                "rng_advances": 1,
            }
            # D_COMMITTED 的界限 = 第一笔不可撤销的新阶段操作（预约换址）
            # ——此后本轮 D 不再选点（无第三次位置决策，§16.4）。
            runtime.joint_stage = JOINT_STAGE_D_COMMITTED
            # W5（§7.2 第 2 步）：所有来源引用经目录 acquire_entry_
            # reference（最终提交目标——含回退后的继续目标；执行 pin
            # = 源保护，settle 闭合释放）。
            runtime.joint_source_reference_ids = (
                self._joint_acquire_source_references(runtime, selected))
            runtime.boundary_decision = {
                "chosen_instance_index": selected,
                "chosen_action": runtime.joint_decode_action,
                "is_continuation": runtime.joint_decode_is_continuation,
                "cost_ns": chosen_cost_ns,
                "decode_steps": int(boundary.decode_steps),
                "fallback": boundary_fallback,
                # D17 披露：本边界候选计算采用的公式/语义版本（与准入行、
                # run manifest 同键同源；旧实验不自动代表新模型）。
                "model_semantics_version": MODEL_SEMANTICS_VERSION,
                # D20 披露：阶段 epoch 与跨阶段预约事务回执（幂等键/换址
                # 半边/resize 释放分量；重复通知与在途 ack 的审计面）。
                "stage_epoch": runtime.joint_boundary_epoch,
                # W8（方案 §11.3 状态一致硬证书）additive 披露：决策时点
                # 版本面（_joint_boundary_versions 单一事实源 = 提交点复
                # 查的同一快照）——probe 证书的目录/预约维度事实源；读
                # 者忽略未知键，off/旧工具零行为差。
                "state_versions": self._joint_state_versions_disclosure(
                    boundary_versions),
                "stage_transaction": (
                    {
                        "stage_epoch": stage_transaction.stage_epoch,
                        "stage_action": stage_transaction.stage_action,
                        "source_instance_index":
                            stage_transaction.source_instance_index,
                        "target_instance_index":
                            stage_transaction.target_instance_index,
                        "reserved_shard_bytes": list(
                            stage_transaction.reserved_shard_bytes),
                        "released_shard_bytes": list(
                            stage_transaction.released_shard_bytes),
                        "eviction_count": len(stage_transaction.evictions),
                        "irreversible": stage_transaction.irreversible,
                        "aborted": stage_transaction.aborted,
                        # W5（§7.2）：attempt 维度 + A22 不可撤销物理子义
                        # 务分记（retained_obligations 的身份/字节面）。
                        "attempt": stage_transaction.attempt,
                        "retained_obligation_count": len(
                            stage_transaction.retained_obligations),
                    }
                    if stage_transaction is not None else None),
                # D19 披露：等待接口类别（选中目标的接纳类别）、边界阶段
                # 工作视图与继承义务表（asdict 序列化，决策行透传）。
                "boundary_wait_kind": boundary_wait_kind(
                    selected, boundary.p_instance_index),
                "boundary_transition": boundary_transition,
                # W5（§7.1/C5 控制延迟）：目标系统共同控制延迟与提交点
                # 复查披露。params.control_latency_ns 非零时预测侧全 D
                # 候选已承担 #ctrl 定长腿（W3）；执行侧提交点复查 =
                # 版本（§7.2 第 1 步，expected_versions）+ 容量
                # （_ensure_capacity）+ 源保护（来源引用 acquire）三重。
                # 真实时延推进（提交事件延迟到 tick+L、期间背景推进）需
                # C++ 侧定时事件——Python 决策侧无自治 tick 源（决策由
                # C++ 交付驱动，空转等待会静止），登记为缺口不硬造
                # （EVIDENCE 遗留；0 = assumed-zero 已披露不变）。
                # cfg3（2026-10-02）：正值在配置面一律 fail-closed；只有
                # assumed-zero 合法，因此这里的 0 是实际执行口径，不是
                # 依靠 deferred 标记逃过兑现义务。
                "control_latency_ns": int(
                    self._joint_prediction_config_or_load()
                    .params.control_latency_ns),
                "control_commit_recheck": (
                    "versions+capacity+source_references"),
                # 同状态 probe 强制钩子披露（仅武装且命中本请求时落键；
                # off 时键缺席 = 行与既有钉测逐位一致）。
                **({"forced_by_probe": forced_probe_note}
                   if forced_probe_note is not None else {}),
                **({"causal_prestate": causal_prestate}
                   if causal_prestate is not None else {}),
                # 一次映射对照臂专用披露键（独立键，与 forced_reason 词表
                # 无冲突；off 时键缺席 = 行与既有钉测逐位一致）。
                **({"boundary_policy": boundary_policy}
                   if boundary_policy != BOUNDARY_POLICY_OFF else {}),
                "stage_work": _dataclass_asdict(stage_work),
                "inherited_obligations": [
                    _dataclass_asdict(item) for item in inherited_obligations],
                # D17 披露：逐候选全 breakdown（公式版本/输入来源/字节空
                # 间/关键等待——predicted_* 与真实时间分列）+ 真值槽位
                # （未实际执行的候选恒 None，受控同状态执行另行填充）。
                # 一次映射对照臂 = 无候选遍历，恒空列表（如实披露）。
                "candidates": [
                    {
                        "instance_index": candidate.instance_index,
                        "action": candidate.action,
                        "applicable": candidate.applicable,
                        "cost_ns": candidate.cost_ns,
                        "is_continuation": candidate.is_continuation,
                        "inapplicable_reason": candidate.inapplicable_reason,
                        "breakdown": self._joint_breakdown_log_dict(
                            candidate.breakdown),
                        "measured": None,
                    }
                    for candidate in boundary_candidates
                ],
            }
            # cfg1（§10.3 完成预测成熟通道）：实际执行候选的 D 预测登记
            # ——非回退 = 选中候选本体；回退 = 继续方案（按执行身份在
            # 候选表取回其 breakdown）。一次映射对照臂无 D 计价（chosen
            # 为 None）⇒ 不登记不虚构（该臂无预测真值可配对）。
            executed_breakdown = None
            executed_action = chosen_action
            if boundary_fallback:
                executed_action = runtime.joint_decode_action
                for candidate in boundary_candidates:
                    if (candidate.instance_index == selected
                            and candidate.action == executed_action
                            and candidate.is_continuation
                            and candidate.breakdown is not None):
                        executed_breakdown = candidate.breakdown
                        break
            elif chosen is not None:
                executed_breakdown = chosen.breakdown
            if executed_breakdown is not None:
                self._joint_completion_maturity_or_default().note_decision(
                    request_id=runtime.request_id,
                    decision_stage="D",
                    decision_tick_ns=int(tick),
                    action=str(executed_action),
                    predicted_next_token_done_ns=(
                        executed_breakdown.predicted_next_token_done),
                    predicted_service_done_ns=(
                        executed_breakdown.predicted_service_done),
                    predicted_settle_done_ns=(
                        executed_breakdown.predicted_merge_done),
                )
            decode_state = self.instances[selected]
            runtime.decode_instance_index = selected
            # D20（§14.5/§16.4）：完成归属在边界提交点锁定（D 提交后本轮
            # 不再选点；完成信号按归属实例核销——终局不再随任何后续状态
            # 变化改写）。
            runtime.completion_instance_index = selected
            self._bump_kv_ledger_epoch()  # 改法D：KV 变更点 2/9（阶段预约事务 commit_stage_reservation_transaction）
            self.graph.sync_pending_history_after_evictions(
                reservation_move_evictions)
            self._bump_kv_ledger_epoch()  # 改法D：KV 变更点 3/9（move_prefill_to_decode）
            self.graph.sync_pending_history_after_evictions(
                decode_move_evictions)
            # joint（§3.1）：decode 不按真实最终长度一次扩容——drain 边界只
            # 覆盖已知 prefill 上下文；后续逐列车按实际消费进展因果增长
            # （_finalize_completed_trains 内的 _joint_grow_decode 钩子）。
            # (2026-10-01 R2 清理) decode_growth_evictions 恒空局部量
            # 删除（R14 因果化后该来源结构性不存在）。
            runtime.decode_evictions = (
                reservation_move_evictions + decode_move_evictions)
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
            # D20（§14.5）：边界窗口闭合——D 目标预留与工作副本均已建立，
            # P 阶段引用至此结束
            # （先于预约注销：release 守卫钉"引用与预约不得双活到结算"）。
            # 之后本请求回到常规 active 会话语义（完成前本就不是 victim）。
            if runtime.joint_stage_reference_id is not None:
                self.kv_manager.release_stage_reference(
                    runtime.session_id,
                    reference_id=runtime.joint_stage_reference_id)
                runtime.joint_stage_reference_id = None
            # 预约注销 = 准入容量语义终结。
            self.kv_manager.release_request_capacity_reservation(
                runtime.request_id)
            self._bump_kv_ledger_epoch()  # 改法D：KV 变更点 5/9（release_request_capacity_reservation）
            # R15-1：decode 主链迁移/读流在 service_done 注销；逐出支链在各自
            # 的 eviction_done 尾 watch 注销（不能与请求的 decode owner 合并）。
            # C4：多腿 copy-in（元组形态）逐腿平铺入册。
            pd_transfer_entries = (
                runtime.prefill_decode_transfer
                if isinstance(runtime.prefill_decode_transfer, tuple)
                else ((runtime.prefill_decode_transfer,)
                      if runtime.prefill_decode_transfer is not None
                      else ()))
            decode_phase_transfers = (
                runtime.prefill_evictions + pd_transfer_entries)
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
            # online: start_ready_iterations 的 decode 起始记账。
            runtime.decode_start_ns = tick
            # ledger（阶段 3 感知账本，查询/审计输入不进判据）。
            self._ledger_admit(
                runtime.request_id, tick,
                {"type": "active_decode", "instance_index": selected})
            # decode 段发射移至加入列车；drain 边界的 decode 决策记录在此。
            self._emit_join_decision(runtime, tick)
        except Exception:
            # ---- D20 修复轮二（2026-09-27，通用异常路径预约回滚补齐）----
            # try 块全程的"预约已提交"中间态与逐态处置：
            #   (a) ``commit_stage_reservation_transaction`` 已提交、物理
            #       迁移未落地（异常发生在 commit_boundary_decode_
            #       transition 内部/之前）——abort 窗口仍开：显式调用
            #       ``abort_stage_reservation_transaction``（幂等 epoch
            #       事务回退半边：预约换址回源实例 + 足迹恢复提交前快照
            #       + 本 epoch 结局记 aborted；已提交 T+E 逐出照常在账
            #       ——真实容量事件不回滚）。同 epoch 重复回退原子重放
            #       aborted 回执，不双重恢复；容量失败专门回退路径
            #       （KVCapacityError → abort + stay 重提交）已
            #       aborted 的事务在此被 ``aborted`` 位排除——不双跑。
            #   (b) D_COMMITTED 之后（D_COMMITTED 起物理工作已提交、字节搬运
            #       已随 transition 落地）——abort 合同窗口已闭
            #       （face_scheduler：仅在字节搬运尚未发射的提交窗口内
            #       合法），此处不再伪造一次与物理状态矛盾的换址回退；
            #       预约若尚未走到常规注销点（drain 边界的直接注
            #       销），留账随 fail-closed run 终止一并
            #       消亡（run 尾审计兜底），不悬挂 epoch。
            # 回滚面自身故障不得掩盖原始异常（与 tick=None 同纪律）。
            try:
                stage_txn = self.kv_manager.last_stage_transaction(
                    runtime.request_id)
                if (stage_txn is not None and not stage_txn.aborted
                        and stage_txn.stage_epoch
                        == runtime.joint_boundary_epoch
                        and runtime.joint_stage != JOINT_STAGE_D_COMMITTED
                        and runtime.request_id
                        in self.kv_manager._reservations):
                    self.kv_manager.abort_stage_reservation_transaction(
                        request_id=runtime.request_id,
                        source_instance_index=(
                            int(stage_txn.source_instance_index)),
                        stage_epoch=runtime.joint_boundary_epoch)
            except Exception:
                pass  # run 尾 verify 兜底；原始异常优先上抛。
            # 就绪队列成员资格同样不悬挂（异常点晚于 pending_decode_
            # ready.append 时移除——本请求的 D 边界未闭合，不得以半破
            # 状态滞留 frontier；幂等：不在队则空放）。
            if runtime.decode_instance_index is not None:
                decode_state = self.instances[runtime.decode_instance_index]
                if runtime in decode_state.pending_decode_ready:
                    decode_state.pending_decode_ready.remove(runtime)
                    self._refresh_frontier(decode_state)
            if runtime.joint_stage_reference_id is not None:
                self.kv_manager.release_stage_reference(
                    runtime.session_id,
                    reference_id=runtime.joint_stage_reference_id)
                runtime.joint_stage_reference_id = None
            raise

    def _complete_requests(self, completed_now, tick: int, *,
                           completion_block_end_stage: str = "seg2",
                           alarm_base_tick: Optional[int] = None):
        """offline 蓝图（已删的 plan_face_requests，行号锚不可核）: 3986-4042 decode 完成分支 + completion_order 批
        （下一 turn arrival 排程 + mark_complete +
        快照）。

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

        M16（face-LRU 审计同构修复，2026-09-29）：到达 alarm 与
        next_arrival_world_ns 披露代理的锚用 **alarm_base_tick**（调用
        点传 delta.deferred_from_tick，0/None 回退交付 tick；None = 直连
        测试的旧口径回退，交付 tick 即锚）——deferral 交付（Step 1-11
        T→T+1）下交付 tick 晚于物理完成 tick 1 tick，锚改与指标层
        resolve_arrival（parent code-4 completion + interval）同源同式。
        本参数**只**承载到达锚：service_done_ns/completion 行 tick 等结算
        与披露账本仍用交付 tick（C3b 合同字面不变）。
        """
        alarm_base = tick if alarm_base_tick is None else alarm_base_tick
        # offline blueprint completion_order 排序
        completion_order = sorted(
            completed_now,
            key=lambda request_id: (
                self.runtime_by_request_id[request_id].session_id,
                request_id,
                self.runtime_by_request_id[request_id].queue_index,
            ),
        )
        # offline blueprint decode 完成分支。下一轮 alarm 的排程与
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
        # 整份搬大侧（零池写、全层 LOCAL@胜者、home 迁移；copy
        # 零字节翻转；REMOTE 无主就地保留；胜者侧空间经统一 T+E 准备，
        # 双侧深缺口才 fail-closed）。完成时已观测的 decode 长度/输入
        # 长度入在线估计器（实际结算，非决策 oracle）。
        # C14（2026-09-22）完成检查清单审计落字：完成检查 = 源端保护
        # 解除（败者侧释放/home 标签迁移）、暂存归还（F14 无留存型口径
        # ⇒ 无操作，journal staging_return_bytes=0 披露位）、
        # 增量结算（账本真值裁决）、合并目标容量、失败分支闭合（版本键
        # 防重复回调）；逐请求结算事实入 FS 侧 kv_delta_journal（C16
        # 消费面），不进 decision-log schema。
        for request_id, following, interval in completion_facts:
            runtime = self.runtime_by_request_id[request_id]
            # 零字节翻转族（copy）final = 工作副本 context
            # 并集，new_tokens 仅入披露行，维持 input+decode 口径。
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
            # W5（§7.2 来源引用）：settle 子协议提交前释放来源引用——
            # D 执行期（D 提交 → service_done）的源保护结束；merge_back
            # 的账本结算使来源内容被胜者权威副本取代（mirror 同步的
            # retire_entry 以引用=0 为前提），在途物理流的保护自此由
            # R15 流登记 + merge watch 承担（_on_merge_done 闭合）。
            # W9（§7.4）：distributed = 目录提交型结算——同一释放点
            # （执行 pin 转可管理驻留引用，不永久 pin），结算走
            # settle_distributed（无物理归并；来源内容保留为跨轮来源）。
            self._joint_release_source_references(request_id)
            settlement_policy = self._joint_settlement_policy()
            if settlement_policy == SETTLEMENT_POLICY_DISTRIBUTED:
                runtime.merge_transfers = ()
                self.kv_manager.settle_distributed(
                    session_id=runtime.session_id,
                    trigger_request_id=runtime.request_id,
                    new_tokens=new_tokens,
                    at_ns=tick,
                )
                # W9 结算披露（consolidate 的 last_merge_outcome 消费面
                # 复用同键形态——真 KVCacheManager __init__ 恒设该属性；
                # 替身漏设 = AttributeError，F6 同款销账）。
                runtime.merge_outcome = dict(
                    self.kv_manager.last_distributed_settle_outcome.as_dict(),
                    direction="distributed",
                    winner_instance=None,
                    loser_instance=None,
                    transferred_bytes=0,
                    home_flipped=False,
                )
            else:
                runtime.merge_transfers = self.kv_manager.merge_back(
                    session_id=runtime.session_id,
                    trigger_request_id=runtime.request_id,
                    new_tokens=new_tokens,
                    at_ns=tick,  # M4：restore complete 兜底结算带时刻
                )
                # 合并方向 v2（2026-09-17）：完成行披露 merge_back 的方向
                # 语义——face_scheduler 每次 merge_back 后设置
                # last_merge_outcome（键 session_id/direction("stay"|
                # "forward"|"reverse"|"in_place")/zero_byte_flip/
                # winner_instance/loser_instance/transferred_bytes/
                # home_flipped；stay 亦照实落账）。真 KVCacheManager 的
                # __init__ 恒设该属性（face_scheduler）——直接访问，
                # 替身 kv_manager 漏设 = AttributeError（F6 销账）。
                runtime.merge_outcome = self.kv_manager.last_merge_outcome
            # W5（§7.4）：结算事务提交（SETTLE_PENDING ->
            # SETTLE_COMMITTED；merge_back = consolidate 的 settle 子协
            # 议，账本已闭，物理载荷自此启动；distributed = 目录提交型
            # 子协议——零载荷，闭合走同刻 settle_done）。结算账本登记
            # （runtime 退休后由本账本追到 SETTLE_DONE）。
            runtime.joint_stage = JOINT_STAGE_SETTLE_COMMITTED
            self._joint_settlement_ledger()[request_id] = {
                "stage": JOINT_STAGE_SETTLE_COMMITTED,
                "settlement_policy": settlement_policy,
                # W9：会话锚（早到下一轮的结算闭合门按会话查在册结算）。
                "session_id": runtime.session_id,
                # SERVICE_DONE（§7.4 单独记录）：响应完成 = 本 completion
                # 交付 tick；下一轮到达 alarm 已在此刻 + interval 排定
                # （future_alarms 语义不变）。
                "service_done_ns": tick,
                "settle_committed_ns": tick,
                "merge_payload_bytes": sum(
                    int(transfer.total_bytes)
                    for transfer in runtime.merge_transfers),
                "merge_done_ns": None,
                "settle_done_ns": None,
                # W5（§7.2/§7.4）：来源引用句柄与 t_B 义务表随 settlement
                # 转移（runtime 退休后由结算账本闭合）。
                "source_reference_ids": getattr(
                    runtime, "joint_source_reference_ids", None),
                "inherited_obligation_ids": tuple(
                    obligation.obligation_id for obligation in (
                        self._joint_pending_obligations(runtime))),
                # W9：distributed 目录提交块（settle_done 行披露）。
                "directory_commit": (
                    self.kv_manager.last_distributed_settle_outcome.as_dict()
                    if settlement_policy == SETTLEMENT_POLICY_DISTRIBUTED
                    and getattr(self.kv_manager,
                                "last_distributed_settle_outcome", None)
                    is not None else None),
            }
            self._bump_kv_ledger_epoch()
            self.graph.sync_pending_history_after_evictions(
                runtime.merge_transfers)
            # D14 语义审计（修复轮二，2026-09-27，v2.0 §5.2）：runtime
            # .decode_length = 本轮 decode 前向步数，**不含** debut 首
            # token 步——首 token 由 P 末前向采样产出（证据：_plan_train
            # 首 decode span = (1, ctx+1)，prompt 在 drain 时已全量入 KV
            # （kv=ctx），+1 即 P 末采样 token 作为首 decode 前向的输入；
            # decode_length==0 的请求经 _settle_terminal_at_boundary 在
            # P 完成即整轮完成——P 末 token 即唯一输出 token）。故
            # §5.2"输出 D 个 token = D−1 次额外 decode 前向"中的
            # D = decode_length + 1，前向步 = decode_length = D−1——
            # observe_completed 直收 decode_length 已是 §5.2 前向步单位，
            # 无需 −1 换算（round-1 docstring 预期的"届时换算"由本审计
            # 取消；估计器单位 = 前向步，全部消费者一致，见
            # joint/test_d14_forward_step_semantics.py）。
            self._joint_horizon.observe_completed(
                runtime.session_id, runtime.decode_length)
            # W5（W6 接线 §2 完成）：条件剩余估计器并联收完成样本（同
            # 刻同样本——completed_forward_steps = decode_length 前向步
            # 口径与上方 horizon 估计器一致；category = 到达时刻冻结的
            # 已揭示类型键）。
            self._joint_length_history_or_default().observe_completed(
                session_id=runtime.session_id,
                completed_forward_steps=runtime.decode_length,
                category=runtime.joint_revealed_category,
                tick_ns=tick)
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
            if following is None:
                continue
            # C3b（2026-09-22 到达合同）：下一轮到达 = service_done +
            # interval（有/无 merge 流同锚点——thinktime 自响应完成起算，
            # 实验思路 §三.4 a(s,k+1) = f(s,k) + z(s,k)）。到达后的数据
            # 依赖门控不在本 alarm 侧承载（图侧 interval gate / 同 rank
            # 串行链 / store 尾边，见 docstring——到达 ≠ 就绪）。
            # M16（2026-09-29）：锚 = alarm_base（父请求 code-4 完成
            # tick，与指标层 resolve_arrival 同源同式；见 docstring）。
            self._batch["future_alarms"].append({
                "arrival_world_ns": alarm_base + interval,
                "envelope": {
                    "request_id": following.request_id,
                    "session_id": following.session_id,
                    "turn_index": following.turn_index,
                    "prefill_length": following.prefill_length,
                    "decode_length": following.decode_length,
                    "inter_request_interval_ns": interval,
                },
            })
        # offline blueprint 先全部 mark_complete
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
        # offline blueprint 完成路径段（D-clear 2026-09-05）：完成边界
        # 零逐出，仅保留结构守卫与恒空 completion_evictions 的下游同步。
        for request_id in completion_order:
            runtime = self.runtime_by_request_id[request_id]
            if runtime.decode_instance_index is None:
                raise RuntimeError("completed request has no Decode instance")
            # D-clear (2026-09-05): 主动驱逐已物理清除——完成路径零逐出。
            runtime.completion_evictions = ()
            self.graph.sync_pending_history_after_evictions(
                runtime.completion_evictions)
        # offline blueprint 完成快照 + completion 批（合同①）：
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
                runtime.plan_dict(),
                block_end_stage=completion_block_end_stage)
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
                        alarm_base + _interval if following is not None
                        else None),
                }
                self._register_transfer_flows(
                    runtime.merge_transfers, owner=request_id + "#merge")
            else:
                # W5（§7.4）：零载荷结算（无物理 merge 传输）——
                # SETTLE_COMMITTED -> SETTLE_DONE 同刻闭合；settle_done
                # 事件行在此落（service/settle 同 tick、merge_done=None）。
                self._joint_close_settlement(request_id, tick)
            # 阶段 3 感知账本：completed-unreconciled 核销（基类
            # _settle_completions 已在策略前完成；此处为决策日志）。
            self.log_decision(
                {"kind": "completion", "request_id": runtime.request_id,
                 "priority": 0},
                tick,
                decision={
                    # W5（§7.4 SERVICE_DONE 单独记录）：响应完成时刻显式
                    # 键（= 本行 tick；下一轮到达 alarm 锚）。additive——
                    # 旧读者忽略未知键。
                    "service_done_ns": tick,
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
            # W5（§7.4）：runtime 生命周期终态——零载荷结算即
            # SETTLE_DONE（同刻闭合）；merge 载荷在途时保持
            # SETTLE_COMMITTED（runtime 退休后由 _joint_settlements
            # 账本追到 SETTLE_DONE，见 _on_merge_done）。
            runtime.joint_stage = (
                JOINT_STAGE_SETTLE_COMMITTED
                if self._joint_settlement_ledger().get(request_id, {}).get(
                    "stage") == JOINT_STAGE_SETTLE_COMMITTED
                else JOINT_STAGE_SETTLE_DONE)

    # ------------------------------------------------------------- 准入 --

    def _bump_kv_ledger_epoch(self) -> None:
        """改法D：KV 账本纪元 +1。仅调度器的 9 个 KV 变更点后调用
        （见各调用处注释）；可行性读取与 _check_invariants 只读、不 bump。
        getattr 缺省 0：__new__ 直构的测试替身不经 __init__（2026-10-01
        R2——settlement 闭合点新增 bump 后替身路径首次到达此处）。"""
        self._kv_ledger_epoch = getattr(self, "_kv_ledger_epoch", 0) + 1

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
            # _try_admit_request 的失败分支赋值（quota_deferred/容量回
            # 退/预检回退三分支的 ``_last_admit_failure_key`` 写入点），
            # 首请求成功前结构性缺席；__slots__ 无、__init__ 不设。
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

    # ------------------------------------------------- joint 视图与选择 --

    def _joint_working_context(self, runtime) -> int:
        """工作副本/基础会话的当前应覆盖 token 数（因果：只含已到达输入
        与已实际完成的 decode）。stay/copy 均覆盖完整上下文。

        D18：P 提交后 decode 阶段以**边界二次决策的动作族**为准
        （``joint_decode_action`` 优先于 P 阶段 ``joint_action``）。"""
        return (runtime.history_tokens_before
                + runtime.joint_input_tokens
                + runtime.decode_tokens_consumed)

    @staticmethod
    def _joint_state_versions_disclosure(boundary_versions):
        """W8（方案 §11.3 状态一致硬证书）：决策时点版本面的日志披露形态。

        输入 = ``_joint_boundary_versions()`` 的返回（或 None）。输出恒为
        JSON 可序列化 dict（``{"kv_ledger_epoch", "directory_version",
        "stage_epochs"}``；降级维 None = 不参与比较——嵌入测试壳无纪元
        账本）。只做形状转换，不改任何状态。"""
        if boundary_versions is None:
            return {"kv_ledger_epoch": None, "directory_version": None,
                    "stage_epochs": None}
        kv_epoch, manager_versions = boundary_versions
        manager_versions = manager_versions or {}
        stage_epochs = manager_versions.get("stage_epochs")
        return {
            "kv_ledger_epoch": kv_epoch,
            "directory_version": manager_versions.get("directory_version"),
            "stage_epochs": ([list(pair) for pair in stage_epochs]
                             if stage_epochs is not None else None),
        }

    def _joint_boundary_versions(self):
        """W5（§7.2 第 1 步）：决策时点版本面（提交点复查的基准）。

        返回 ``(kv_ledger_epoch, manager_versions)``；manager_versions =
        ``{"directory_version": int, "stage_epochs": tuple}``（真
        KVCacheManager 的目录镜像/阶段纪元表；替身 kv_manager 无目录
        → 该维 None = 不参与复查）。返回 None = 完全降级（嵌入测试壳
        无纪元账本）。
        """
        kv_epoch = getattr(self, "_kv_ledger_epoch", None)
        manager = getattr(self.kv_manager, "_kv_directory", None)
        stage_epochs = getattr(self.kv_manager, "_stage_epochs", None)
        if kv_epoch is None and manager is None and stage_epochs is None:
            return None
        manager_versions = None
        if manager is not None or stage_epochs is not None:
            manager_versions = {
                "directory_version": (
                    int(manager.version) if manager is not None else None),
                "stage_epochs": (
                    tuple(sorted(stage_epochs.items()))
                    if stage_epochs is not None else None),
            }
        return (kv_epoch, manager_versions)

    def _joint_boundary_terminal(self, runtime) -> bool:
        """P→D 边界"已知终止"判据（v2.0 §2.3/§16.4 的 observed_terminal
        在本协议下的形态）。

        本仓执行协议为确定性回放：停止条件由 workload 脚本定义（无
        EOS 事件侧）。decode 脚本为空（``decode_length == 0``）的请求，
        其 prefill 末 chunk 采样的首 token 即满足停止条件——P 完成即
        整轮完成，直接进入结算，不做无意义的 D 选点（v2.0 §2.3"若已知
        无需任何额外 D 前向……直接进入结算"）。oracle 纪律不变：
        ``decode_length`` 不进任何代价视图/候选计价——本判据只门控状态
        机的"是否需要 D 阶段"分支，不参与任何选点。"""
        return runtime.decode_length <= 0

    def _settle_terminal_at_boundary(self, runtime, state, tick: int) -> None:
        """已知终止：``P_COMPUTE_DONE -> SETTLE_PENDING`` 直接结算（不选
        D）。完成归属锚 = P 实例（工作副本已在其上，无迁移、无 D 前向、
        不加入任何列车）；预约与在途流按 drain 纪律释放后走标准完成
        事务（merge 结算/到达 alarm/完成批）。

        W5（§7.4）：已知终止的结算同样经 settle 入口——W9 起按
        settlement_policy 分派（consolidate=merge 子协议；
        distributed=目录提交型子协议），两策略都在 _complete_requests
        的统一结算点闭合。"""
        self._joint_settlement_policy()  # fail-closed：未知策略拒绝结算
        runtime.joint_stage = JOINT_STAGE_SETTLE_PENDING
        selected = state.index
        runtime.decode_instance_index = selected
        # D20：完成归属在边界结算点锁定 = P 实例（无 D 选点、无迁移）。
        runtime.completion_instance_index = selected
        runtime.joint_decode_action = runtime.joint_action
        runtime.joint_decode_is_continuation = True
        runtime.boundary_decision = {
            "chosen_instance_index": selected,
            "chosen_action": None,
            "is_continuation": True,
            "cost_ns": 0,
            "decode_steps": 0,
            "fallback": False,
            "terminal": True,
            "candidates": (),
            "stage_epoch": runtime.joint_boundary_epoch,
            "stage_transaction": None,
        }
        # 预约随 prefill 终结注销（与常规 drain 边界同一纪律）；无
        # decode 相传输可登记（零迁移零增长）。
        # D20：边界窗口闭合——已知终止无 D 阶段，P 阶段引用随结算释放
        # （先于预约注销，release 守卫钉"引用与预约不得双活"）。
        if runtime.joint_stage_reference_id is not None:
            self.kv_manager.release_stage_reference(
                runtime.session_id,
                reference_id=runtime.joint_stage_reference_id)
            runtime.joint_stage_reference_id = None
        self.kv_manager.release_request_capacity_reservation(
            runtime.request_id)
        self._bump_kv_ledger_epoch()  # 改法D：KV 变更点（release@terminal）
        self.log_decision(
            {"kind": "boundary_terminal", "request_id": runtime.request_id,
             "priority": 0},
            tick,
            decision={
                "instance_index": selected,
                "rule": "observed terminal at P_COMPUTE_DONE: skip D "
                        "selection, settle directly (v2.0 sec.16.4)",
            },
        )
        runtime.joint_stage = JOINT_STAGE_SETTLE_COMMITTED
        # 已知终止请求无退出列车：completion 触发门锚 drain 列车 seg1
        # 块末（P 完成即本轮全部计算完成——语义精确，非近似）。
        self._complete_requests(
            [runtime.request_id], tick, completion_block_end_stage="seg1")
        runtime.joint_stage = (
            JOINT_STAGE_SETTLE_COMMITTED
            if self._joint_settlement_ledger().get(runtime.request_id, {}).get(
                "stage") == JOINT_STAGE_SETTLE_COMMITTED
            else JOINT_STAGE_SETTLE_DONE)

    def _joint_boundary_view(self, runtime, state):
        """P→D 边界只读快照 S_B（式 1b；只读已到达事实——真实 KV 位置
        与就绪状态、当前负载经成本模型、已发出 token 与采样状态；无
        后端未来完成事件）。

        D19 轮二：返回 BoundaryRuntimeStateEnvelope——PDBoundaryView
        字段面全透传（计价/选择器零改动）+ 三项运行时状态与层覆盖的
        可选属性面（见构造点注）。"""
        p_index = state.index
        session = self.kv_manager._sessions[runtime.session_id]
        snapshot = self.kv_manager.session_snapshot(runtime.session_id)
        working = tuple(int(value) for value in snapshot.shard_bytes)
        tp = self.kv_manager.tp_degree
        kv_token_bytes_by_tp_rank = tuple(
            int(value) for value in kv_cache_shard_bytes_for_tokens(
                self.model, 1, tp))
        kv_token_bytes = kv_token_bytes_by_tp_rank[0]
        history_tokens = (
            runtime.history_tokens_before + runtime.joint_input_tokens)
        # 完整型（三动作唯一布局族）：H_B 全量在 i_P，无异地基础区间。
        base_instance = session.instance_index
        # W5 消费切换（W6 §2）：D 边界正式 n̂_B 切到条件剩余估计器
        # （estimate_conditional——与 P 准入同一实例同一规则同一版本，
        # 方案 §8.2；下限 1 由估计器保证 = 式(4a) max(1, n̂)）。旧
        # CausalHorizonEstimator 保留为诊断/先验来源（不进计价；其
        # estimate_boundary 并行查询曾声称"仅供对拍/披露"，但结果从未
        # 落任何披露行——2026-10-01 审查删除该死计算）。
        _conditional = self._joint_conditional_estimate(runtime)
        n_hat_b = int(_conditional.remaining_forward_steps)
        _horizon_source = "conditional:" + _conditional.source
        # 完整型工作副本 = 全上下文单矩形 [0, history) × 全层 @ i_P。
        working_intervals = (BoundaryKvInterval(
            replica_id="boundary_work_context",
            role="context",
            holder_instance=p_index,
            authority=True,
            token_start=0,
            token_end=int(history_tokens),
            layer_start=0,
            layer_end=self.model.layers,
            bytes_by_tp_rank=working),)
        view = PDBoundaryView(
            session_id=runtime.session_id,
            home_instance=base_instance,
            p_instance_index=p_index,
            p_action=runtime.joint_action,
            history_tokens=history_tokens,
            working_bytes_by_tp_rank=working,
            decode_steps=int(n_hat_b),
            kv_token_bytes=int(kv_token_bytes),
            kv_token_bytes_by_tp_rank=kv_token_bytes_by_tp_rank,
            # D12（式(21)）：边界逻辑 home ω_B——merge 目标确定性平局
            # 首位；与基础实例（base_instance，D-BI）分立，
            # 不等于数据所在地。
            logic_home_instance=session.home_instance,
        )
        # D19 轮二（2026-09-27 执行面填充接通）：三项运行时状态 + 真实
        # 层覆盖随计价视图进场——envelope 透传全部 PDBoundaryView 字段
        # （frozen dataclass 字段面不动，共享文件零冲突；计价/选择器
        # 零改动），``BoundaryKVLayout.from_boundary`` 经可选属性拾取：
        # * ``terminated`` 源 = _joint_boundary_terminal（完成/终止判定
        #   ——已知终止不进 D 选点，正常 D 路径恒 False；字段真实接通，
        #   非 schema 占位）；
        # * ``first_token`` 源 = StageWork 的 BoundaryFirstTokenState
        #   （P 末 chunk 已采样恰一次、待首个 D 前向消费）；
        # * ``obligations`` 源 = 继承义务表（t_B 在册已提交义务）；
        # * 层覆盖：working = 全模型层数（stay/copy 唯一完整型布局族）。
        # 首 token/义务表复用 _joint_boundary_stage_view 的同一材料化
        # （纯读、零账本副作用——drain 主链 的既有调用重取同值，
        # 两处结果恒等）。
        stage_work, inherited_obligations = (
            self._joint_boundary_stage_view(runtime, state, view))
        return BoundaryRuntimeStateEnvelope(
            view,
            terminated=self._joint_boundary_terminal(runtime),
            first_token=stage_work.first_token,
            obligations=inherited_obligations,
            working_layers=self.model.layers,
            base_layers=self.model.layers,
        )

    def _joint_boundary_stage_view(self, runtime, state, boundary):
        """D19（v2.0 §13.1/§5.3，2026-09-26）：边界阶段工作视图与继承
        义务表——D 候选专用因果视图的执行侧材料化。

        * ``StageWork`` = D 边界 stay/copy 工作模板（§13.1 第二
          行：零新 U-token prefill、只做必要状态准备、n̂_B 次剩余前向
          ——长度入口归一为剩余前向）；首输出 pending token 状态：P 末
          chunk 已采样恰一次（RNG 推进一次、本轮首 token 已发出待首个
          D 前向消费恰一次）。
        * 继承义务表：t_B 仍在册的已提交义务逐条登记（存在性校验，
          不推测登记；receiver = 已承诺接收端）。未完义务的资源占用经
          实例负载视图照常进计价（不因二次决策清零）——本表是审计/接
          口面，非额外计价输入。

        返回 (stage_work, obligations)。"""
        from joint.boundary_layout import (
            BoundaryFirstTokenState,
            InheritedObligation,
            boundary_stage_work,
        )
        session = self.kv_manager._sessions[runtime.session_id]
        first_token = BoundaryFirstTokenState(
            issued=True, consumed=False, rng_advances=1)
        stage_work = boundary_stage_work(
            decode_steps=max(0, int(boundary.decode_steps)),
            first_token=first_token)
        obligations = self._joint_pending_obligations(
            runtime, state=state)
        return stage_work, tuple(obligations)

    def _joint_pending_obligations(self, runtime, state=None):
        """W5（§7.2 执行侧）：t_B 仍在册的已提交义务逐条登记（存在性
        校验，不推测登记；receiver = 已承诺接收端）——边界视图与结算
        闭合门（_joint_outstanding_obligations）共用同一枚举（活账本
        单一事实源）。"""
        from joint.boundary_layout import InheritedObligation
        session = self.kv_manager._sessions.get(runtime.session_id)
        obligations = []
        if session is None:
            return tuple(obligations)
        # C13 copy 逐 chunk 交接账本（None = 本轮非 copy 跨实例或已闭
        # 合；轮内权威 = home 侧残量事实源——drain 边界通常已闭合）。
        if getattr(session, "copy_handoff", None) is not None:
            obligations.append(InheritedObligation(
                obligation_id=f"{runtime.request_id}#copy_handoff",
                kind="copy_handoff",
                receiver_instance=session.home_instance,
                summary="C13 chunk handoff journal still open at t_B"))
        # C15 后缀逐组恢复区间账本（drain 边界应已 consumed 关账）。
        if getattr(session, "restore_journal", None) is not None:
            obligations.append(InheritedObligation(
                obligation_id=f"{runtime.request_id}#restore_journal",
                kind="restore_journal",
                receiver_instance=session.instance_index,
                summary="C15 suffix restore journal still open at t_B"))
        # 仍在册的请求容量预约（本边界的提交事务随後换址/注销）。
        if runtime.request_id in getattr(
                self.kv_manager, "_reservations", {}):
            obligations.append(InheritedObligation(
                obligation_id=f"{runtime.request_id}#reservation",
                kind="capacity_reservation",
                receiver_instance=(
                    state.index if state is not None else None),
                summary="request capacity reservation in ledger at t_B"))
        return tuple(obligations)

    def _joint_outstanding_obligations(self, request_id: str):
        """W5（§7.2 inherited_obligation_ids 执行侧）：settle 闭合门——
        活账本枚举仍开放的义务（runtime 退休后枚举为空 = 全部核销）。"""
        runtimes = getattr(self, "runtime_by_request_id", None) or {}
        runtime = runtimes.get(request_id)
        if runtime is None:
            return ()
        return self._joint_pending_obligations(runtime)

    def _joint_boundary_cost_model(self, runtime, boundary, tick: int,
                                   obligations=()):
        """边界决策时点代价模型（与准入同一构造器；D08 起构造不携带
        decode 上下文/时域——JCM 边界入口现场取 H_B = boundary.
        history_tokens × n̂_B = boundary.decode_steps（式 4a 下界），
        τ 一次前向服务模板由构造统一注入；负载/流/端口快照取 t_B 时
        刻——未完已提交义务照常占资源）。

        W5（§7.2 C6）：``obligation_resolver`` 接入 JCM——t_B 义务表
        （InheritedObligation）适配为 ObligationView 参竞争/门控 settle
        （W3 预留的正式消费通道）。remaining_bytes 恒空元组：义务的物
        理占用经实例负载视图照常进计价（单一事实源——resolver 只承担
        **settle 门控**语义，不重复字节竞争）；W1 账本的 reservation_
        id/obligation_link 接通后可升级逐义务真实剩余量（遗留登记）。"""
        return self._joint_cost_model(
            tick,
            obligation_resolver=self._joint_obligation_resolver(
                obligations))

    def _joint_obligation_resolver(self, obligations):
        """t_B 义务表 → ObligationView 解析器（W3 接口；空表 = None
        ——JCM 缺省降级披露，不虚构义务腿）。"""
        if not obligations:
            return None
        from joint.action_work_graph import ObligationView
        table = {}
        for obligation in obligations:
            table[obligation.obligation_id] = ObligationView(
                obligation_id=obligation.obligation_id,
                kind=obligation.kind,
                target_instance=obligation.receiver_instance,
                remaining_bytes_by_tp_rank=(),
                gate_settle=True,
                done=False)
        return table.get

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
                # C4：增长容量核按"排除自有预约"口径（预约在册期间
                # 不排除会双重需求；无预约时 exclude 为空操作）。
                reservation_request_id=runtime.request_id,
                at_ns=self._require_batch_tick(),  # M4：决策行带时刻
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
            # R17-1b 同纪律（修复轮三 D2）：唯一生产调用点
            # _joint_grow_decode 位于 run_variant_policy 批生命周期内，
            # 且其在 KVCapacityError 抛出前已调用 _require_batch_tick()
            # ——`if self._batch else 0` 的 0 回退生产不可达，仅剩
            # __new__ 壳替身直调（测试夹具已补 _batch）；沿咽喉点口径
            # 断言式取 tick，不在场 = 生命周期破损当场早爆。
            self._require_batch_tick(),
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
                        at_ns=tick,  # M4：决策行带时刻
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

    def _joint_settlement_ledger(self) -> dict:
        """结算账本的惰性兜底（__new__ 测试替身不经 __init__——真
        __init__ 恒构造；替身首次使用时建空账本）。"""
        ledger = getattr(self, "_joint_settlements", None)
        if ledger is None:
            ledger = {}
            self._joint_settlements = ledger
        return ledger

    def _joint_acquire_source_references(self, runtime, target_instance):
        """W5（§7.2 事务顺序第 2 步）：候选提交时**所有来源引用**经目录
        ``acquire_entry_reference``（W4 执行面接线）——最终提交目标
        （含回退后的继续目标）的全部片上来源条目持执行 pin 引用，直至
        结算闭合（_joint_close_settlement 释放；来源侧在此期间不可被
        逐出——源保护）。

        无目录（替身 kv_manager）/无来源 → None（不虚构引用）。"""
        directory = getattr(self.kv_manager, "kv_directory", None)
        if directory is None:
            return None
        handles = []
        for entry in directory.query_sources(
                runtime.session_id, int(target_instance)):
            directory.acquire_entry_reference(
                entry.entry_id, exec_pin=True)
            handles.append((entry.entry_id, True))
        return tuple(handles) or None

    def _joint_release_source_references(self, request_id: str) -> None:
        """结算闭合释放来源引用（幂等；配对守恒由目录 fail-closed 保证）。"""
        runtime = (getattr(self, "runtime_by_request_id", None)
                   or {}).get(request_id)
        handles = (
            getattr(runtime, "joint_source_reference_ids", None)
            if runtime is not None else None)
        if handles is None:
            # 结算记录兜底（runtime 已退休——引用清单随 settlement 转移）。
            record = self._joint_settlement_ledger().get(request_id) or {}
            handles = record.get("source_reference_ids")
        if not handles:
            return
        directory = getattr(self.kv_manager, "kv_directory", None)
        if directory is None:
            return
        for entry_id, exec_pin in handles:
            directory.release_entry_reference(
                entry_id, exec_pin=bool(exec_pin))
        if runtime is not None:
            runtime.joint_source_reference_ids = None

    def _joint_settlement_policy(self) -> str:
        """W9（§7.4）：结算策略单一裁决点（consolidate 缺省；distributed
        = 目录提交型结算子协议——执行侧 `_complete_requests` 分派；一次
        run 固定，预测配置冻结）。"""
        policy = getattr(
            self._joint_prediction_config_or_load(),
            "settlement_policy", SETTLEMENT_POLICY_CONSOLIDATE)
        if policy not in (SETTLEMENT_POLICY_CONSOLIDATE,
                          SETTLEMENT_POLICY_DISTRIBUTED):
            raise RuntimeError(
                f"unknown settlement_policy {policy!r} on the execution "
                "side (must be consolidate or distributed)")
        return policy

    def _joint_session_settlement_open(self, session_id: str) -> bool:
        """W9：会话是否有未闭合结算（结算账本在册行按会话锚查）。"""
        for record in self._joint_settlement_ledger().values():
            if record.get("session_id") == session_id:
                return True
        return False

    def _joint_distributed_source_surface(self, session_id: str):
        """W9（§7.4 披露）：下一轮 P 构图的目录全量来源面（区间数/位置数
        分别统计 + 跨轮增长断言）。无目录（替身面）返回 None。"""
        directory = getattr(self.kv_manager, "kv_directory", None)
        if directory is None:
            return None
        from joint.distributed_settlement import (
            assert_round_growth_within_budget,
            enumerate_source_surface,
        )
        meter = getattr(
            self.kv_manager, "distributed_directory_meter", None)
        previous = 0
        if meter is not None:
            rounds = meter.session_rounds.get(session_id)
            if rounds:
                previous = rounds[-1]["source_count"]
        surface = enumerate_source_surface(directory, session_id)
        growth = assert_round_growth_within_budget(previous, surface)
        if meter is not None:
            meter.record_op("enumerate_source_surface")
        disclosure = surface.as_dict()
        disclosure["previous_source_count"] = previous
        disclosure["growth"] = growth
        return disclosure

    def _joint_close_settlement(self, request_id: str, tick: int) -> None:
        """W5（§7.4）：SETTLE_COMMITTED -> SETTLE_DONE 闭合 + settle_done
        事件侧披露行。

        三类结算事件的落点（§10.2"记录实际 service_done、settle_done"）：
        * service 事件 = completion 行（tick + service_done_ns 键）；
        * 物理 merge 事件 = completion 行 merge_transfers/merge_direction
          + kind="merge_done" 行（载荷/ack）；
        * settle 事件 = 本方法落的 kind="settle_done" 行（settle 统一语
          义；echo service_done_ns / merge_done_ns——分布式可无 merge 事件
          而仍有 settle，W9 后同键复用）。

        结算义务门（§7.2 inherited_obligation_ids 执行侧）：t_B 在册义
        务经完成边界核销（预约注销/journal 关账在 _complete_requests
        前序段完成）；本方法只披露在
        案事实，未闭合义务在既有 fail-closed 点拦截（kv_delta 闭合门）。
        """
        record = self._joint_settlement_ledger().get(request_id)
        if record is None:
            # 已闭合（重复回调幂等——_on_merge_done 与零载荷路径同刻竞
            # 争时第二次进入直接空放）。
            return
        # W5（§7.2 所有来源引用）：结算闭合释放（源保护解除——此后来源
        # 条目可被合法逐出；配对守恒 fail-closed 在目录侧）。
        self._joint_release_source_references(request_id)
        # W5（§7.2 inherited_obligation_ids 执行侧）：t_B 义务表闭合门
        # ——settle 不得早于未完义务（门控 release/settle；已完义务零腿
        # 不阻）。runtime 在册时以活账本核验，退休后以 settlement 转移
        # 清单对照完成核销事实（completion 边界的 fail-closed 点已在
        # 前序段落拦截未闭合项——此处防御性复验）。
        outstanding = self._joint_outstanding_obligations(request_id)
        if outstanding:
            raise RuntimeError(
                "settlement for request {} cannot close with outstanding "
                "inherited obligations {!r} (sec.7.2 settle gate)".format(
                    request_id, outstanding))
        record["stage"] = JOINT_STAGE_SETTLE_DONE
        record["settle_done_ns"] = tick
        # (2026-10-01 审查修复) 结算闭合 bump KV 纪元：settlement_open_
        # dependency_gate 延迟的请求按纪元键重试——不 bump 时闭合不改
        # 任何键维度，被门请求可能永久 skip 到 run 尾审计 fail-closed
        # （当前生产时序下完成交付同刻闭合、门不可达；此为分布式结算
        # 出现载荷型/延迟闭合形态时的正确性前提）。
        self._bump_kv_ledger_epoch()
        if record.get("merge_done_ns") is None and record.get(
                "merge_payload_bytes"):
            # merge 载荷在途的闭合由 _on_merge_done 承担（watch 交付带
            # merge_done_ns）；防御：载荷非零而 merge_done 缺席 = 通道
            # 错接，fail-closed。
            raise RuntimeError(
                "settlement closed for request {} with merge payload {} "
                "bytes but no merge_done tick".format(
                    request_id, record.get("merge_payload_bytes")))
        self.log_decision(
            {"kind": "settle_done", "request_id": request_id,
             "priority": 0},
            tick,
            decision={
                "settle_done_ns": tick,
                "service_done_ns": record.get("service_done_ns"),
                "settle_committed_ns": record.get("settle_committed_ns"),
                "merge_done_ns": record.get("merge_done_ns"),
                "merge_payload_bytes": record.get("merge_payload_bytes"),
                "settlement_policy": record.get("settlement_policy"),
                "settle_state": JOINT_STAGE_SETTLE_DONE,
                # W9（§7.4/§10）：distributed 目录提交块（条目/区间/来源
                # 数、目录版本增量、提交墙时；consolidate 键缺席）。
                **({"directory_commit": record["directory_commit"]}
                   if record.get("directory_commit") is not None else {}),
            },
        )
        # cfg1（§10.3 完成预测成熟）：settle_done 即 P/D 完成预测的成熟
        # 配对点——实测 service/settle（结算账本同源事实）与本请求登记
        # 的执行候选 predicted_* 配平（恰一次、幂等；service_done 缺席
        # 单列不丢）。账面经 run 末 kind=forecast_status 行披露。
        self._joint_completion_maturity_or_default().mature(
            request_id=request_id,
            service_done_ns=record.get("service_done_ns"),
            settle_done_ns=tick)
        # 结算账本闭合即弹出（有界内存；泄漏审计在 verify_run_end）。
        self._joint_settlement_ledger().pop(request_id, None)

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

        结算生命周期锚点（C14 步骤 1 审计落字）：源端基础历史
        在跨实例执行有效期内保持源端有效并受结构保护（prepare 后 primary
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
        #）——直接调用，替身 kv_manager 漏接口 = AttributeError。
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
        # W5（§7.4）：merge 载荷 ack 送达 = settle 义务闭合前置——
        # settlement 记录必须在案（SETTLE_COMMITTED 段登记；缺记录 =
        # watch 通道与结算账本脱钩，fail-closed）。闭合（SETTLE_DONE +
        # settle_done 事件行）在物理 merge 事件行之后落——事件序 =
        # 物理载荷 ack → settle 闭合。
        settlement = self._joint_settlement_ledger().get(request_id)
        if settlement is None:
            raise RuntimeError(
                "merge-done watch {} delivered without a pending settlement "
                "record for request {} (settlement ledger closure failure)"
                .format(watch_id, request_id))
        settlement["merge_done_ns"] = tick
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
        # W5（§7.4）：物理 merge 事件行已落——SETTLE_COMMITTED ->
        # SETTLE_DONE 闭合（settle_done 事件行 echo 本行 merge_done_ns）。
        self._joint_close_settlement(request_id, tick)

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
        """仅在逐出尾标记真实完成后注销该旁支自己的流与配额。

        M4（2026-09-26）：尾标记完成处补决策日志行（kind=kv_eviction_
        done，完成精确 tick）——与 _emit_eviction_only_nodes 的
        kind=kv_eviction 行按 decision.watch_id 配对，差值 = 实际池写
        回时长（驱逐/恢复时间线测量的执行侧真值）。行不入
        joint_audit_kinds 的账本通道（工具侧跳过词表随本行扩容）、
        不经 seen_kinds 去重（同请求可多行）；行在尾标记真实完成
        之后落（披露忠实于已发生的物理事实），释放流/配额之前。"""
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
        self.log_decision(
            {"kind": "kv_eviction_done", "request_id": pending["request_id"],
             "priority": 0},
            tick,
            decision={"watch_id": watch_id},
        )
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

    def _register_transfer_flows(self, transfers, *, owner: str) -> None:
        """R15-1/F-B：发射时点登记在途流——逐 shard 完整链路序列（TP
        并行全路径，非代表对）+ 池端口占用（remote_store/remote_load 的
        edge 端口）+ HBM 端点端口（C2/F4 + D04：noc_migrate 的源端口
        home 读腿恒有、目标端口 exec 写腿对 copy/merge 恒有——A4' 补价
        后计价与执行同腿型；D04 起 remote_load 登记 noc_path 末端目标
        HBM 写端点、remote_store 登记 noc_path 首端源 HBM 读端点——
        F3 旧豁免废除；中间跳 rank 不触碰端点 HBM）。
        修复轮三（2026-09-27）：remote_load/remote_store 的端点登记
        **无条件**（不再要求 len(noc_path) >= 2）——零距离池事务
        （edge == 端点 rank、deterministic_xy_route 同端点返回单 rank
        路径）生产可达，执行侧漏登会与计价侧无条件收端点腿（
        _pool_transfer_ns_shards）失衡；单 rank 路径 path[0] ==
        path[-1]，每 shard 恰登一个端点。noc_migrate 仍限多跳
        （len >= 2 门内双端点）：零距离 noc_migrate 无生产构造点，且
        注册表 register 纯累加无去重，移出门会对同一 rank 双登记抬升
        除数。
        注销由完成事件驱动（drain/completion/merge-watch 经
        _release_transfer_flows 按 owner 统一注销，含 D04 池端点），
        决策时刻快照因果可见。"""
        for transfer in transfers or ():
            if transfer.kind == "local_hit":
                continue
            for shard in transfer.shards:
                self._joint_flows.register_path(
                    shard.noc_path, owner=owner)
                if shard.edge_rank is not None:
                    self._pool_ports.register(shard.edge_rank, owner=owner)
                # 真实 HBM 端点登记（C2/F4 + D04；修复轮三 2026-09-27
                # 执行侧零距离补全）：remote_load/remote_store 端点
                # **无条件**入表——零距离池事务（edge == 端点 rank、
                # noc_path 单 rank）生产可达（边界 rank 自身即最近池
                # 边、deterministic_xy_route 同端点返回单 rank 路径），
                # 旧 len(noc_path) >= 2 门漏登端点 HBM，与计价侧
                # _pool_transfer_ns_shards 无条件收端点腿失衡（候选
                # u_port 低估）；单 rank 路径 path[0] == path[-1]，每
                # shard 恰登一个端点、语义一致。F3 旧边界（池路径端
                # 点留在 HBM 端口表之外）已由 D04 废除：恢复
                # （remote_load）登记目标 HBM 写端点、写回
                # （remote_store）登记源 HBM 读端点（edge 侧是池端
                # 口、非 rank HBM——式 7 池路径资源集：池端口+NoC+
                # 端点 HBM）。noc_migrate 双端点（源 home 读腿 + 目标
                # exec 写腿恒有）仍限多跳 len >= 2：零距离
                # noc_migrate 无生产构造点，且 HbmPortFlowRegistry
                # register 纯累加无去重，移出门会对同一 rank 双登记
                # 抬升除数。注销由完成事件 release_owner 统一承担
                # （_release_transfer_flows 同 owner 同生命周期）。
                if transfer.kind == "noc_migrate":
                    if len(shard.noc_path) >= 2:
                        self._hbm_ports.register(
                            shard.noc_path[0], owner)
                        self._hbm_ports.register(
                            shard.noc_path[-1], owner)
                elif transfer.kind == "remote_load":
                    self._hbm_ports.register(
                        shard.noc_path[-1], owner)
                elif transfer.kind == "remote_store":
                    self._hbm_ports.register(
                        shard.noc_path[0], owner)

    def _release_transfer_flows(self, owner: str) -> None:
        self._joint_flows.release_owner(owner)
        self._pool_ports.release_owner(owner)
        # C2 + D04：HBM 端点端口随同注销（noc_migrate 双端点与 D04 池
        # 路径端点同 owner 同生命周期；幂等空放）。F6 销账：__init__
        # 恒设 _hbm_ports——直接访问（替身漏设 = AttributeError，端口
        # 登记/注销静默丢失已不可能）。
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
        ① AIMD observe（observe_telemetry）/port_snapshot
        两调用点无
        session 语境不传代表值——active_decode 瞬空的窗口（准入后、
        drain 入批前）回退 ctx=1（r̂ 最小值）⇒
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
            # _OnlineRequestRuntime 对象（入批 / 退出
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
        """transfers → (链路多重集, 端口集)——配额准入镜像（local_hit
        跳过；noc_migrate 双端点入端口）。

        D04（2026-09-26）登记边界分化注记：流登记侧（
        _register_transfer_flows）自 D04 起对 remote_load/remote_store
        也登记真实 HBM 端点（计价保真）；本配额镜像**保持**
        noc_migrate 端点判据不变——端口域此处是实例级准入资源（parity
        门 / bulk 槽，C11 WP3c 冻结语义），池流端点入镜像会改变准入判
        决（超出 D04 计价边界）；分化为有意行为，两处 docstring 同步
        落字防漂。端口粒度 = 端点 rank 所属实例（TP 全 rank 同账，
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

        * stay：无跨实例主流程（本地动作）——恒适用（准入相
          逐出支链按 oneshot 事后入册收紧后续判据，D4）；
        * copy（oneshot）：链路门（占用+预留+新增 <= Q）。

        δ_adm 合取确认（A3' 终冻 = 0）：joint 模式重选 = 既有全候选
        argmin（(cost_ns, order_key) 字典序）——joint 臂组合语义 ≡
        argmin、不借配额接入夹带选择规则变更（原 challenger_flips
        谓词已随 2026-10-14 远读退役删除，D9）。
        （2026-10-01 审查修复后补记：load-first/affinity-first 顺序模
        式经 _quota_reselect_feasible 在可行集上**重跑模式选点规则**，
        非纯 argmin——配额过滤仍只做动作级准入，不改值。）
        """
        tracker = self._quota_tracker
        action = candidate.action
        target = candidate.instance_index
        if action == "stay":
            return QuotaVerdict(admitted=True)
        source = (
            session_view.resident_instance
            if session_view.resident_instance is not None
            else session_view.home_instance)
        if source is None or source == target:
            # 无跨实例流（与计价适用性同判据的防御分支；正常路径
            # copy 的适用候选必有异地源）。
            return QuotaVerdict(admitted=True)
        demand: dict[tuple[int, int], int] = {}
        for edge in self._quota_route_edges(source, target):
            demand[edge] = demand.get(edge, 0) + 1
        for link, need in demand.items():
            remaining = tracker.link_remaining(link)
            if remaining < need:
                return QuotaVerdict(
                    admitted=False,
                    flow_class=FLOW_ONESHOT,
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
        return QuotaVerdict(admitted=True)

    def _quota_filter_candidates(self, record, session_view, rid: str,
                                 now_ns: int, cost_model=None):
        """逐候选配额判据 + 配额可行集内重选（按 scheduler_mode 分派）。

        返回 (chosen, candidates, deferred)：deferred 非 None = 配额导致
        全动作不可行（C9 冻结语义：回 pending_admissions——quota_deferred_
        requeue 模块级实现，**永不 raise**；重试键扩展配额代数）。

        (2026-10-01 审查修复) 重选规则按模式分派：joint = (cost,
        order_key) 全局 argmin（原行为）；load-first/affinity-first =
        可行集上重跑顺序选点规则（README §1 J 行 off 语义：先按在线负载
        选实例、再在该位置选动作；affinity 先 home 后回退）——旧实现恒
        argmin，quota × J-off 合法臂的选点面漂移成 joint 规则。face_static
        恒 quota-off（F7 耦合例外），不经本方法。"""
        _ = now_ns
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
        chosen = self._quota_reselect_feasible(
            feasible, session_view, cost_model)
        return chosen, tuple(filtered), None

    def _quota_reselect_feasible(self, feasible, session_view, cost_model):
        """配额可行集上的模式一致重选（镜像 joint_scheduler 的选点规则）。

        joint：全候选 (cost_ns, order_key) argmin。顺序模式：先定实例
        （load-first = 可行实例上 min(total_task_load_ns, index)；
        affinity-first = 可复用 home 优先、缺省回退 load-first），再在该
        实例的可行动作里取 (cost_ns, order_key) 最优——与
        select_instance_and_action 的顺序分支同判据（同 tick 同快照，
        loads 未变）。"""
        # getattr 缺省 joint：__new__ 直构的测试替身不经 __init__——
        # 缺省臂 = 原 argmin 行为（替身所钉的口径）。
        if (getattr(self, "_joint_mode", SCHEDULER_MODE_JOINT)
                == SCHEDULER_MODE_JOINT
                or cost_model is None):
            return min(
                feasible,
                key=lambda candidate: (
                    candidate.cost_ns, candidate.order_key()))
        feasible_instances = sorted({
            candidate.instance_index for candidate in feasible})
        home_instance = getattr(session_view, "home_instance", None)
        location = getattr(session_view, "location", None)
        if (self._joint_mode == SCHEDULER_MODE_AFFINITY
                and home_instance is not None
                and home_instance in feasible_instances
                and location in ("local_hbm", "partial_hbm_remote",
                                 "remote_memory")):
            chosen_instance = home_instance
        else:
            chosen_instance = min(
                feasible_instances,
                key=lambda index: (
                    cost_model.loads[index].total_task_load_ns, index))
        at_instance = [
            candidate for candidate in feasible
            if candidate.instance_index == chosen_instance]
        if not at_instance:
            # 可行实例集非空 ⇒ 每实例至少一个可行候选；防御性回退全局
            # argmin（不 raise——本方法是 C9 "永不 raise" 语义的一部分）。
            return min(
                feasible,
                key=lambda candidate: (
                    candidate.cost_ns, candidate.order_key()))
        return min(
            at_instance,
            key=lambda candidate: (
                candidate.cost_ns, candidate.order_key()))

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
        * 逐出/池恢复支链（全部动作，D4 通用治理）：oneshot 入册
          （owner=rid#evict；逐出尾 watch 交付后释放）——**恒置于主流程之后**
          （O2 序不变量，见下）。

        O2（2026-09-23 终轮审计，事故链写实）：判据时刻本请求的逐出
        足迹**不可知**——history_evictions 由准入事务产出（受害选择依赖
        执行期 KV 账本，不做试探性事务即不可前瞻）、事务成功才落账
        （_try_admit_request 事务成功段），而判据在事务前执行
        （_try_admit_request 内 _quota_filter_candidates 调用
       ）⇒ 判据 demand 只含主流程
        足迹。探针实证事故链：判据 PASS → 支链先入册占掉末槽 → 主流程
        # 主流程入册 FAIL → LinkQuotaError 返回 False → 调用方对已物化
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

    # ------------------------------------------------ R15/C11 收尾审计 --

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
            # §5.4（ARRIVAL 批内缓存）：断供清缓存也是遥测状态变化——
            # bump（两个出口之一；缓存关闭时无人读取）。
            self._link_telemetry_version += 1
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
            # §4.4/A03（2026-09-30 修复）：max(在册, 遥测) 合并的冲突
            # 披露需要时间窗——最近采样窗界随摄入落账（构造 JCM 视图时
            # 注入 TelemetryLinkFlowView.telemetry_window_ns）。
            self._telemetry_last_window_bounds_ns = (
                int(window_start), int(window_end))
        # A9'：在场核对——本包缺席的既有条目失效删除（键缺席 = C++ 侧
        # 按契约省略 = 空闲信号；divisor_effective 退回注册表在册流近
        # 似，消除"忙转闲后陈旧实测速率/流数永久驻留、长 run 单调抬升
        # NoC 计价除数"——D13：速率不再产生除数，陈旧流数同受此约束）。
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
        # §5.4（ARRIVAL 批内缓存）：遥测摄入末尾 +1（方案键设计：遥测
        # 版本计数器；rates/flow_counts/window 本包任何变化都由此感知
        # ——含零变化包的保守 bump，只损失命中不错复用）。
        self._link_telemetry_version += 1

    def _hbm_active_decode_streams(self, port_rank: int) -> int:
        """C2/F4：该端口所属实例的活跃 decode KV 消费流数（因果负载
        视图派生——active_decode 现值；闲置为 0，不假设恒 1）。TP 全
        rank 同账（每活跃会话逐 rank 各占一条消费流）。"""
        instance_index = self._rank_to_instance.get(port_rank)
        if instance_index is None:
            return 0
        return len(self.instances[instance_index].active_decode)

    def _pool_port_divisor(self, instance_index: int) -> int:
        """实例各池端口最大在途数 + 自身 1（R15-3；E 内核 r_j 通道——
        KVCacheManager.pool_divisor_fn 注入）。

        D04（2026-09-26）起 J 池路径计价不再消费本实例级聚合除数：
        JCM 逐 shard 按流查询实际 edge（``pool_edge_flows_fn =
        _pool_ports.count`` 注入，在册在途 + 候选自身经该端口 shard
        数，见 joint_cost_model._pool_edge_divisor）——实例级最大除
        数替代所有端口的旧口径仅在 E 内核（P1 r_j）保留。"""
        edges = self._instance_edge_ports.get(instance_index, ())
        if not edges:
            return 1
        return max(1, max(self._pool_ports.count(edge) for edge in edges) + 1)

    @staticmethod
    def _hbm_owner_belongs_to_request(owner: str, request_ids) -> bool:
        """端口流归属键是否属于给定请求集合（纯字符串前缀判定）。

        owner 键型全集（_register_transfer_flows 登记点）：``rid``、
        ``rid#decode``、``rid#decode#q{seq}``、``rid#merge``、
        ``rid#evict``、``rid#evict#prefill`` 与
        逐出尾 watch 的 ``batch_train_evict_{rid}_{phase}_{seq}#flow``。
        rid 自身可含 "_"，不得反向解析字符串——逐 rid 做三种前缀匹配。
        """
        for request_id in request_ids:
            if (owner == request_id
                    or owner.startswith(request_id + "#")
                    or owner.startswith(
                        "batch_train_evict_" + request_id + "_")):
                return True
        return False

    def _foreign_hbm_flows_on_instance(self, instance_index: int,
                                       own_request_ids) -> dict:
        """本实例 rank 端口上**他请求**在册 HBM 传输流（owner -> 命中端
        口；只读查询）。

        只查既有在册流登记（HbmPortFlowRegistry 在册归属视图），不新建
        平行账本；本列车成员自身的流（含其逐出支链/读计划/merge 尾
        watch 归属键）不判为争用。在册状态由发射/完成事件因果驱动，
        观测窗口内曾在册但已完成注销的流不可回溯——采样终点在册即拒收
        （能确证的争用下界，宁可少收不可污染）。
        """
        ranks = set(self.topology.instance(instance_index).ranks)
        foreign: dict[str, list[int]] = {}
        for owner, ports in self._hbm_ports.leaked_owners().items():
            if self._hbm_owner_belongs_to_request(owner, own_request_ids):
                continue
            hit = sorted(port for port in ports if port in ranks)
            if hit:
                foreign[owner] = hit
        return foreign

    def _reject_contended_sample(self, state, tick: int, start: int,
                                 name: str, own_request_ids,
                                 foreign_flows: dict) -> None:
        """争用样本拒收（§9.2"分不开就不更新该样本"）+ 观测窗口端口
        占用摘要披露（决策行 kind=joint_sample_contended）。"""
        self._joint_factors.reject_contended(name)
        rank_snapshot = self._hbm_ports.snapshot()
        ranks = self.topology.instance(state.index).ranks
        self.log_decision(
            {"kind": "joint_sample_contended",
             "request_id": (
                 own_request_ids[0] if own_request_ids else ""),
             "priority": 0},
            tick,
            decision={
                "instance_index": state.index,
                "factor": name,
                "window_start_ns": start,
                "window_end_ns": tick,
                "own_request_ids": list(own_request_ids),
                # 他请求在册流（争用证据）与同刻端口占用分解（活跃
                # decode 消费流/在册传输流/合计）分列披露。
                "foreign_hbm_flows": dict(foreign_flows),
                "port_occupancy": {
                    str(rank): rank_snapshot[rank]
                    for rank in ranks if rank in rank_snapshot},
            },
        )

    def _observe_unified_service_sample(
            self, category: str, tick: int, baseline_work_ns: int,
            busy_service_ns: int, *, foreign_participants: int = 0,
            work_id: Optional[str] = None,
            instance_index: Optional[int] = None,
            rank_ids: Optional[Sequence[int]] = None) -> None:
        """§8.1 单账本采样（W6 接线 §3；2026-09-30 修复起为纯列车服务
        样本的唯一估计器更新点——J 计价 ``_joint_factors`` 与 face γ/
        k_hide 在决策时点读本统一估计器的只读发布值，不再并联更新任一
        侧）。纯度门在 recorder 内：foreign>0 ⇒ contended 计数+残差，
        不进 η。样本使用 canonical train/instance/rank identity；重复完成
        回声按该身份幂等，同一事实只更新 recorder 一次。一个 TP 列车是
        共享事实，因此 rank 集合编码进单一 work_id，而不是按 rank 复制
        样本。"""
        from joint.history_estimator import ServiceWorkSample
        canonical_id = str(work_id or f"{category}#{tick}")
        if instance_index is not None:
            ranks = tuple(sorted({int(rank) for rank in (rank_ids or ())}))
            rank_suffix = ",".join(str(rank) for rank in ranks) or "unknown"
            canonical_id = (
                f"{canonical_id}#i{int(instance_index)}#r{rank_suffix}"
                f"#{category}")
        self._joint_service_samples_or_default().observe(ServiceWorkSample(
            work_id=canonical_id,
            category=category,
            observation_tick_ns=tick,
            baseline_work_ns=int(baseline_work_ns),
            busy_service_ns=int(busy_service_ns),
            queue_wait_ns=0,
            data_wait_ns=0,
            capacity_wait_ns=0,
            foreign_participants=int(foreign_participants),
            hardware_version=MODEL_SEMANTICS_VERSION,
        ))

    def _observe_service_factors(self, state, train, tick: int) -> None:
        """列车核销时采集纯服务样本（§8.1 单账本：唯一更新点 = W6
        ServiceSampleRecorder——D14 固定 1/16 EWMA 平滑效率 η、因子 =
        1/η_new 规则经 PredictionParams 单源生效，v2.0 §9.2；J 计价
        ServiceFactors 与 face γ/k_hide 在决策时点读其只读发布值，
        2026-09-30 修复起不再并联更新任一侧）。

        样本纯净性：只收**纯列车**（无 joiner 迁移、无 chunk×decode 混合
        ——两段不可因果分离）的服务段实测 = 核销 tick − max(发射 tick,
        上次核销 tick)（排除列车前排队的污染，排队在 target_wait 段计
        价）；基数 = 同内容 roofline 闭式账本。跨请求纯度门：观测窗口
        内本实例 rank 端口存在他请求在册 HBM 传输流（_foreign_hbm_
        flows_on_instance 查在册登记）即拒收并计 contended——他流争用
        已由端口份额除数通道计价，混入样本会把同一争用重复写进 η。
        transfer 因子样本源 = 链路遥测窗口（A8' 接线：_ingest_link_
        telemetry 逐窗口喂 observe_transfer_from_link_window——列车核
        销通道无纯传输段可因果分离，故本函数不收 transfer 样本；A12'
        名实订正：transfer_factor EWMA 为披露/观测通道，传输争用的计价
        修正实效由链路/端口除数通道唯一承担——divisor_effective 的
        max(registered, 遥测流数) 合并（D13：流数口径，capacity/速率
        旧口径废除）；因子不进 estimate_P_with_continuation，
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
        sample_ranks = self.topology.instance(state.index).ranks
        # 纯 decode 列车（无 chunk、无 joiner——迁移会混入传输段；
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
                context0 = runtime.prefill_context_tokens
                consumed = runtime.decode_tokens_consumed
                for step in range(1, participation + 1):
                    base_ns += self._decode_step_load_ns(
                        instance_size, context0 + consumed + step)
            if base_ns > 0:
                own_request_ids = tuple(
                    request_id for request_id, _ in member_parts)
                foreign_flows = self._foreign_hbm_flows_on_instance(
                    state.index, own_request_ids)
                if foreign_flows:
                    self._reject_contended_sample(
                        state, tick, start, "decode_factor",
                        own_request_ids, foreign_flows)
                    # W5（W6 接线 §3）：争用样本入统一 recorder（唯一
                    # 更新点；contended 计数 + 残差，不进 η）。
                    self._observe_unified_service_sample(
                        "decode", tick, base_ns, span_ns,
                        foreign_participants=len(foreign_flows),
                        work_id=train.get("train_id"),
                        instance_index=state.index,
                        rank_ids=sample_ranks)
                    return
                self._observe_unified_service_sample(
                    "decode", tick, base_ns, span_ns,
                    work_id=train.get("train_id"),
                    instance_index=state.index,
                    rank_ids=sample_ranks)
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
                own_request_ids = tuple(
                    request_id for request_id, _ in chunk_records)
                foreign_flows = self._foreign_hbm_flows_on_instance(
                    state.index, own_request_ids)
                if foreign_flows:
                    self._reject_contended_sample(
                        state, tick, start, "prefill_factor",
                        own_request_ids, foreign_flows)
                    # W5（W6 接线 §3）：争用样本入统一 recorder（唯一
                    # 更新点；contended 残差不进 η）。
                    self._observe_unified_service_sample(
                        "prefill", tick, base_ns, span_ns,
                        foreign_participants=len(foreign_flows),
                        work_id=train.get("train_id"),
                        instance_index=state.index,
                        rank_ids=sample_ranks)
                    return
                # §8.1 单账本（2026-09-30 修复）：纯 prefill 样本只入
                # 统一 recorder 一次；γ_prefill 由 face 侧发布视图
                # （PublishedServiceFactorGroup）在 k_hide 决策时点读
                # 同一发布值——旧 M4 时间衰减桥接（同一样本二次更新
                # face ServiceFactorGroup）退役。η_pool 不桥接（无可
                # 分离纯池传输样本源——含争用样本会使 η 退化为争用指
                # 标、与递推份额仲裁双计，A12' 镜像论证，落字 A17'(d)）。
                self._observe_unified_service_sample(
                    "prefill", tick, base_ns, span_ns,
                    work_id=train.get("train_id"),
                    instance_index=state.index,
                    rank_ids=sample_ranks)

    def _decode_step_load_ns(self, instance_size: int, context: int,
                             hbm_share_divisor: int = 1) -> int:
        """单 decode 步的 roofline 负载（average=1/generated=0 的单步口
        径）。D08：``hbm_share_divisor`` = 式(6) 候选阶段 HBM 份额分母
        （JCM decode 上下文模板 τ(z) 的第二入参经此落地）；缺省 1 =
        独占带宽。"""
        return self._decode_task_load_ns_cached(
            instance_size=instance_size,
            current_context_tokens=context,
            generated_tokens=0,
            average_decode_length=1.0,
            running_step_fraction_remaining=1.0,
            hbm_share_divisor=hbm_share_divisor)

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
        # W9（§7.4 下一轮 P 构图目录化）：distributed_layout 会话的内容
        # 跨多位置——history/覆盖按目录权威并集取值（不能重新合成单
        # home：home 锚保留粘性值作计价锚点，来源面全量枚举另行披露
        # distributed_source_surface）。consolidate 缺省不进此分支。
        if getattr(session, "distributed_layout", False):
            from face_scheduler import kv_cache_shard_bytes_for_layer_range
            directory = getattr(self.kv_manager, "kv_directory", None)
            if directory is not None:
                union_tokens = 0
                anchor_prefix = 0
                anchor_instance = home
                anchor_tokens = 0
                for entry in directory.entries_for_session(session_id):
                    if not entry.is_valid():
                        continue
                    key = entry.content_key
                    union_tokens = max(union_tokens, key.token_end)
                    if (entry.is_pool_backed()
                            or key.token_start > 0):
                        continue
                    # token_start==0 的条目 = base 锚候选（[0, ·) 前缀持有
                    # 者；片上优先，取最大覆盖）。
                    coverage = (
                        key.token_end * key.layer_end)
                    if (entry.holder is not None
                            and coverage > anchor_tokens * anchor_prefix):
                        anchor_tokens = key.token_end
                        anchor_prefix = key.layer_end
                        anchor_instance = int(entry.holder)
                full = (
                    kv_cache_shard_bytes_for_tokens(
                        self.model, union_tokens, self.kv_manager.tp_degree)
                    if union_tokens else (
                        0,) * self.kv_manager.tp_degree)
                anchor_local = (
                    kv_cache_shard_bytes_for_layer_range(
                        self.model, anchor_tokens,
                        self.kv_manager.tp_degree,
                        layer_start=0, layer_end=anchor_prefix)
                    if anchor_tokens and anchor_prefix else (
                        0,) * self.kv_manager.tp_degree)
                # D-BI 同款裁决：视图 home = 物理基础锚所在（计价端点/
                # 适用性按基础所在，不按逻辑 home——逻辑 home 粘性在
                # session.home_instance 保持，不暗中重置）。
                #
                # 适用性口径（W9 分布式布局）：
                # * 锚完整覆盖并集（全层+全 token）⇒ local_hbm@锚——
                #   stay/copy/remote 按锚裁决；
                # * 锚层残缺且残缺段**池 backing** ⇒ partial_hbm_remote@
                #   锚——stay 走池恢复（legacy 语义保留）；
                # * 锚层残缺且残缺段在**其他实例**（跨轮留存）⇒
                #   resident_instance=None（stay/remote 不适用——残缺
                #   非池可恢复；copy 适用，缺失字节按锚差计）。
                anchor_complete = (
                    anchor_tokens >= union_tokens
                    and anchor_prefix >= self.model.layers)
                pool_backed = any(
                    entry.is_pool_backed()
                    for entry in directory.entries_for_session(session_id)
                    if entry.is_valid())
                if anchor_complete:
                    location = "local_hbm"
                    resident = anchor_instance
                elif pool_backed:
                    location = "partial_hbm_remote"
                    resident = anchor_instance
                else:
                    # 跨轮分裂布局（无单一完整持有者、残缺非池 backing）：
                    # stay/remote 不适用（resident=None——无完整本地/单源
                    # 基础）；copy 适用——home 锚 = 最大覆盖段
                    # 持有者（copy 计价的源路由锚 + 缺失字节按真并集差，
                    # 多源执行腿由目录驱动；单源路由近似披露于 W9 证据）。
                    location = "local_hbm"
                    resident = None
                return SessionKVView(
                    session_id=session_id,
                    home_instance=anchor_instance,
                    resident_instance=resident,
                    location=location,
                    history_tokens=union_tokens,
                    resident_prefix_layers=anchor_prefix,
                    history_bytes_by_tp_rank=tuple(anchor_local),
                    missing_bytes_by_tp_rank=tuple(
                        total - local for total, local in zip(
                            full, anchor_local)),
                )
        if session.working_kind is not None:
            # O14（2026-09-23 终轮审计，镜像 face_scheduler.py
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

    def _joint_prediction_config_or_load(self):
        """预测配置的惰性读取（__init__ 已加载；直构/部分构造的测试
        替身面经此兜底——不静默缺省轴）。"""
        config = getattr(self, "_joint_prediction_config", None)
        if config is None:
            from joint.prediction_config import (
                load_prediction_config as _load_prediction_config,
            )
            config = _load_prediction_config()
            self._joint_prediction_config = config
        return config

    @property
    def joint_prediction_manifest(self) -> dict:
        """W3：预测配置的 manifest 披露节（prediction_config 七轴 +
        参数 preset + cross_validate 组合披露；接线进
        joint_mechanism_manifest.json 归 W5/主 Agent——online_service
        属本包冻结清单）。§8.5（2026-09-30 修复）：epoch 登记账本状态
        随行披露——epochs 空 = 本运行无 epoch 触发源的显式登记（配置/
        策略/硬件均在装配期一次读取 fail-closed、run 内不变；唯一运行
        期可分叉源 = 服务样本硬件版本，已接 mark_epoch_change 挂点），
        非缺省缺席。"""
        manifest = dict(
            self._joint_prediction_config_or_load().manifest_dict())
        manifest["cross_validation"] = getattr(
            self, "_joint_prediction_cross_validation", None)
        manifest["epoch_log"] = self._joint_epoch_log_or_default().as_dict()
        return manifest

    def _joint_decision_snapshot(self, now_ns: int):
        """W5（A3）：决策级因果快照（``joint.runtime_snapshot``）。

        versions 面（§4.1 DecisionSnapshot.versions，全七键）：

        * runtime = 实例纪元逐实例表（``_OnlineInstanceState.ledger_epoch``
          ——任务负载快照失效键的既有先例）；
        * layout = KV 账本纪元（``_kv_ledger_epoch``，改法D 九变更点）；
        * reservation = 阶段事务纪元表（kv_manager ``_stage_epochs``）；
        * history = 估计器版本（horizon + W6 条件剩余估计器并联）；
        * hardware = 构造期速率/拓扑摘要（run 内不变）；
        * model = MODEL_SEMANTICS_VERSION；ledger = 账本版本/水位。

        §4.1（2026-09-30 修复）：生产路径首交付 full、其后恒 delta。
        在线计价只取 ``WorkLedger.pricing_snapshot()`` 的当前资源前沿和
        pipeline ledger 的当前 active_resources；完整 work_index/
        resource_state 只由显式离线快照调用物化，不在每次交付中复制；
        capacity_view 由 kv_manager 容量账/摘要版本投影（轻量读面——
        完整 θ 桶已在决策行 capacity_summary_view 落盘，此处不重复
        O(桶数) 拷贝、不双计摘要查询样本）。
        """
        from joint.runtime_snapshot import build_decision_snapshot
        delta = getattr(self, "_joint_current_delta", None)
        if delta is None:
            # 嵌入/测试路径（未经 on_decision_batch 直调决策）：以当前
            # 账本/纪元构造最小 delta 面——delivery_sequence/tick 取
            # 决策参数，arrivals 空（无交付可投影）。
            delta = {
                "delivery_sequence": max(
                    0, int(getattr(self, "last_applied_sequence", -1))),
                "tick": int(now_ns),
                "arrivals": [],
            }
        ledger = getattr(self, "joint_work_ledger", None)
        pipeline_ledger = getattr(self, "joint_pipeline_ledger", None)
        pipeline_state = (
            pipeline_ledger.pricing_snapshot()
            if pipeline_ledger is not None and pipeline_ledger.initialized
            else None)
        pipeline_version = (
            pipeline_state.get("state_version")
            if pipeline_state is not None else None)
        manager = getattr(self, "kv_manager", None)
        directory = getattr(manager, "_kv_directory", None)
        directory_version = getattr(directory, "version", None)
        capacity_version = getattr(
            manager, "capacity_summary_version", None)
        ledger_version = int(getattr(ledger, "version", 0))
        # ledger_version already changes once for each accepted delivery and
        # invalidates this cache. Copying/sorting every resource-key epoch
        # duplicated that invalidation at O(history).
        resource_versions = ()
        runtime_versions = tuple(
            (state.index, state.ledger_epoch) for state in self.instances)
        layout_version = int(getattr(self, "_kv_ledger_epoch", 0))
        reservation_versions = tuple(sorted(
            (getattr(manager, "_stage_epochs", {}) or {}).items()))
        history_version = (
            self._joint_horizon.version,
            self._length_estimator_version())
        hardware_version = getattr(
            self, "_joint_hardware_digest", "hardware_digest_absent")
        delivery_version = (
            int(delta.get("delivery_sequence", 0)),
            int(delta.get("tick", now_ns)),
        )
        cache_key = (
            ledger_version, resource_versions, directory_version,
            capacity_version, int(now_ns), MODEL_SEMANTICS_VERSION,
            runtime_versions, layout_version, reservation_versions,
            history_version, hardware_version, delivery_version,
            pipeline_version,
        )
        cached_key = getattr(self, "_joint_snapshot_cache_key", None)
        cached_snapshot = getattr(self, "_joint_snapshot_cache", None)
        if cache_key == cached_key and cached_snapshot is not None:
            return cached_snapshot

        ledger_state = None
        ledger_projection = None
        ledger_ready = (ledger is not None and (
            ledger.full_load_count > 0 or ledger.delta_load_count > 0))
        if ledger_ready:
            ledger_state = {}
            ledger_projection = ledger.pricing_snapshot()
        snapshot = build_decision_snapshot(
            delta,
            versions={
                "runtime": runtime_versions,
                "layout": layout_version,
                "reservation": reservation_versions,
                "history": history_version,
                "hardware": hardware_version,
                "model": MODEL_SEMANTICS_VERSION,
            },
            ledger_versions={
                "version": ledger_version,
                "event_watermark": int(getattr(
                    ledger, "event_watermark", 0)),
            },
            request_queues=self._joint_frozen_queue_view(),
            ledger_state=ledger_state,
            capacity_view=self._joint_capacity_view(),
            now_ns=int(now_ns),
            ledger_projection=ledger_projection,
            pipeline_ledger_state=pipeline_state,
            resource_version_vector=resource_versions,
            directory_version=directory_version,
            capacity_version=capacity_version,
            pricing_only=ledger_ready,
        )
        self._joint_snapshot_cache_key = cache_key
        self._joint_snapshot_cache = snapshot
        return snapshot

    @staticmethod
    def _joint_probe_json_value(value):
        """Probe 前态的无哈希、确定性 JSON 投影。

        强制 probe 只在命中时调用；投影保留事实值和嵌套结构，不把完整
        前态压成 digest。未知的 dataclass/映射替身按公开字段递归展开，
        使 __new__ 测试壳和生产对象共享同一落盘形状。
        """
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {
                str(key): Sh30OnlineScheduler._joint_probe_json_value(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        if isinstance(value, (tuple, list, set, frozenset)):
            return [
                Sh30OnlineScheduler._joint_probe_json_value(item)
                for item in value
            ]
        if hasattr(value, "__dataclass_fields__"):
            return Sh30OnlineScheduler._joint_probe_json_value(
                _dataclass_asdict(value))
        if hasattr(value, "as_dict") and callable(value.as_dict):
            return Sh30OnlineScheduler._joint_probe_json_value(value.as_dict())
        return str(value)

    def _joint_probe_directory_view(self) -> dict:
        """目录事实投影（条目、区间、引用/pin 和在途 copy 全量保留）。"""
        directory = getattr(self.kv_manager, "_kv_directory", None)
        if directory is None:
            return {"present": False, "version": None, "entries": [],
                    "inflight_copies": []}
        entries = []
        for entry in sorted(
                directory._entries.values(),
                key=lambda item: str(item.entry_id)):
            key = entry.content_key
            entries.append({
                "entry_id": entry.entry_id,
                "session_id": key.session_id,
                "model_version": key.model_version,
                "layer_start": int(key.layer_start),
                "layer_end": int(key.layer_end),
                "token_start": int(key.token_start),
                "token_end": int(key.token_end),
                "tp_rank": key.tp_rank,
                "holder": int(entry.holder),
                "role": entry.role,
                "bytes_by_tp_rank": list(entry.bytes_by_tp_rank),
                "padding_by_tp_rank": list(entry.padding_by_tp_rank),
                "generation": int(entry.generation),
                "state": entry.state,
                "pending_acks": list(entry.pending_acks),
                "release_condition": entry.release_condition,
                "read_refs": int(directory.read_refs(entry.entry_id)),
                "exec_pins": int(getattr(directory, "_exec_pins", {}).get(
                    entry.entry_id, 0)),
            })
        copies = []
        for copy_id, item in sorted(
                getattr(directory, "_inflight_copies", {}).items(),
                key=lambda pair: str(pair[0])):
            copies.append({
                "copy_id": copy_id,
                "session_id": item.session_id,
                "source_holder": int(item.source_holder),
                "target_instance": int(item.target_instance),
                "layer_start": int(item.layer_start),
                "layer_end": int(item.layer_end),
                "token_start": int(item.token_start),
                "token_end": int(item.token_end),
                "bytes_by_tp_rank": list(item.bytes_by_tp_rank),
                "ack_id": item.ack_id,
            })
        return {
            "present": True,
            "schema_version": getattr(directory, "schema_version", None),
            "model_version": getattr(directory, "model_version", None),
            "tp_degree": int(getattr(directory, "tp_degree", 0)),
            "version": int(directory.version),
            "entries": entries,
            "inflight_copies": copies,
        }

    def _joint_probe_capacity_summary(self, summary) -> dict:
        """容量摘要的逐桶事实投影（不引入摘要哈希）。"""
        if summary is None:
            return {"present": False}
        instances = {}
        for instance_index, instance in sorted(
                getattr(summary, "instances", {}).items(),
                key=lambda pair: int(pair[0])):
            buckets = []
            for bucket in instance.buckets:
                key = bucket.key
                buckets.append({
                    "category": key.category,
                    "suffix_class": key.suffix_class,
                    "shard_ratio": list(key.shard_ratio),
                    "writeback_ports": list(key.writeback_ports),
                    "reclaimable_by_rank": list(bucket.reclaimable_by_rank),
                    "clean_by_rank": list(bucket.clean_by_rank),
                    "dirty_by_rank": list(bucket.dirty_by_rank),
                    "atom_count_by_rank": list(bucket.atom_count_by_rank),
                    "max_atom_by_rank": list(bucket.max_atom_by_rank),
                })
            instances[str(instance_index)] = {
                "instance_index": int(instance.instance_index),
                "snapshot_version": int(instance.snapshot_version),
                "maintenance_event_count": int(
                    instance.maintenance_event_count),
                "error_sources": list(instance.error_sources),
                "buckets": buckets,
            }
        return {
            "present": True,
            "snapshot_version": int(summary.snapshot_version),
            "error_sources": list(summary.error_sources),
            "instances": instances,
        }

    def _joint_probe_capacity_view(self) -> dict:
        """容量账与冻结摘要的直接事实面。"""
        manager = getattr(self, "kv_manager", None)
        if manager is None:
            return {"present": False}
        ledgers = {}
        for state in getattr(self, "instances", ()):
            index = int(state.index)
            ledger_fn = getattr(manager, "capacity_ledger_by_tp_rank", None)
            ledgers[str(index)] = (
                [list(row) for row in ledger_fn(index)]
                if callable(ledger_fn) else None)
        summary_fn = getattr(manager, "capacity_summary_snapshot", None)
        summary = summary_fn() if callable(summary_fn) else None
        maintenance_fn = getattr(
            manager, "capacity_summary_maintenance_stats", None)
        return {
            "present": True,
            "ledgers_by_instance": ledgers,
            "summary_version": getattr(
                manager, "capacity_summary_version", None),
            "summary_error_sources": list(getattr(
                manager, "capacity_summary_error_sources", ()) or ()),
            "summary": self._joint_probe_capacity_summary(summary),
            "maintenance": (
                maintenance_fn() if callable(maintenance_fn) else None),
            "stage_epochs": [
                [str(key), value]
                for key, value in sorted(
                    (getattr(manager, "_stage_epochs", {}) or {}).items(),
                    key=lambda pair: str(pair[0]))],
        }

    def _joint_probe_work_view(self) -> dict:
        """队列及已暴露剩余工作事实（不读取未来 decode 终点）。"""
        runtimes = []
        for request_id, runtime in sorted(
                getattr(self, "runtime_by_request_id", {}).items(),
                key=lambda pair: str(pair[0])):
            runtimes.append({
                "request_id": runtime.request_id,
                "session_id": runtime.session_id,
                "turn_index": int(runtime.turn_index),
                "queue_index": int(runtime.queue_index),
                "stage": getattr(runtime, "joint_stage", None),
                "admitted": bool(runtime.admitted),
                "completed": bool(runtime.completed),
                "estimated_arrival_ns": runtime.estimated_arrival_ns,
                "admission_time_ns": runtime.admission_time_ns,
                "prefill_complete_ns": runtime.prefill_complete_ns,
                "decode_start_ns": runtime.decode_start_ns,
                "prefill_instance_index": runtime.prefill_instance_index,
                "decode_instance_index": runtime.decode_instance_index,
                "remaining_chunks": int(runtime.remaining_chunks),
                "prefill_tokens_to_process": int(
                    runtime.prefill_tokens_to_process),
                "prefill_tokens_completed": int(
                    runtime.prefill_tokens_completed),
                "decode_tokens_consumed": int(runtime.decode_tokens_consumed),
                "current_decode_token": int(runtime.current_decode_token),
                "queued_chunk_load_ns": int(runtime.queued_chunk_load_ns),
                "joint_action": runtime.joint_action,
                "joint_decode_action": runtime.joint_decode_action,
                "joint_input_tokens": int(runtime.joint_input_tokens),
                "joint_prefill_work": int(runtime.joint_prefill_work),
                "source_reference_ids": list(
                    runtime.joint_source_reference_ids or ()),
            })
        instances = []
        for state in getattr(self, "instances", ()):
            train = state.in_flight_train
            train_view = None
            if train is not None:
                train_view = {
                    "train_id": train.get("train_id"),
                    "request_id": train.get("request_id"),
                    "stage": train.get("stage"),
                    "members": [list(member) for member in train.get(
                        "members", ())],
                    "iteration_count": train.get("iteration_count"),
                }
            instances.append({
                "index": int(state.index),
                "qp": [item.request_id for item in state.qp],
                "active_decode": [
                    item.request_id for item in state.active_decode],
                "pending_decode_ready": [
                    item.request_id for item in state.pending_decode_ready],
                "in_flight_train": train_view,
                "first_step_remainder": state.first_step_remainder is not None,
                "iteration_count": int(state.iteration_count),
                "train_seq": int(state.train_seq),
                "last_arrival_ns": state.last_arrival_ns,
                "ledger_epoch": int(state.ledger_epoch),
            })
        return {
            "instances": instances,
            "pending_admissions": [
                item.request_id for item in getattr(
                    self, "pending_admissions", ())],
            "arrival_heap": [
                [item[0], item[1], item[2], item[3], item[4],
                 getattr(item[5], "request_id", None)]
                for item in sorted(getattr(self, "arrival_heap", ()))],
            "in_flight": {
                str(key): sorted(value)
                for key, value in sorted(
                    getattr(self, "in_flight", {}).items(),
                    key=lambda pair: str(pair[0]))},
            "runtimes": runtimes,
            "arrived_request_count": int(getattr(
                self, "_arrived_request_count", 0)),
            "completed_requests": int(getattr(self, "completed_requests", 0)),
        }

    def _joint_probe_arbitration_view(self) -> dict:
        """仲裁登记、端口、遥测和配额的当前值。"""
        flows = getattr(self, "_joint_flows", None)
        pool = getattr(self, "_pool_ports", None)
        hbm = getattr(self, "_hbm_ports", None)
        quota = getattr(self, "_quota_tracker", None)
        return {
            "noc_flows": flows.snapshot() if flows is not None else None,
            "noc_owners": flows.leaked_owners() if flows is not None else None,
            "pool_ports": ({
                "counts": {
                    str(key): int(value)
                    for key, value in sorted(pool._counts.items())},
                "owners": {
                    str(key): list(value)
                    for key, value in sorted(pool._owners.items())},
            } if pool is not None else None),
            "hbm_ports": hbm.snapshot() if hbm is not None else None,
            "hbm_owners": hbm.leaked_owners() if hbm is not None else None,
            "telemetry": {
                "rates": {
                    str(key): value for key, value in sorted(
                        getattr(self, "_link_telemetry_rates", {}).items(),
                        key=lambda pair: str(pair[0]))},
                "flow_counts": {
                    str(key): value for key, value in sorted(
                        getattr(self, "_link_telemetry_flow_counts", {}).items(),
                        key=lambda pair: str(pair[0]))},
                "epoch_count": int(getattr(
                    self, "_link_telemetry_epoch_count", 0)),
                "sample_count": int(getattr(
                    self, "_link_telemetry_sample_count", 0)),
                "last_tick_ns": int(getattr(
                    self, "_telemetry_last_tick_ns", 0)),
                "window_broken": bool(getattr(
                    self, "_telemetry_window_broken", False)),
                "last_window_bounds_ns": getattr(
                    self, "_telemetry_last_window_bounds_ns", None),
            },
            "quota": quota.snapshot() if quota is not None else None,
        }

    def _joint_probe_history_view(self) -> dict:
        """长度、到达、统一服务效率三类统计的版本和值摘要。"""
        length = self._joint_length_history_or_default()
        arrivals = self._joint_arrival_stats_or_default()
        recorder = self._joint_service_samples_or_default()
        factors = self._joint_factors
        pending = {}
        for key, values in sorted(
                getattr(recorder, "_pending", {}).items(),
                key=lambda pair: repr(pair[0])):
            pending[str(key)] = list(values)
        eta = {
            str(category): 1.0 / float(value)
            for category, value in sorted(
                getattr(recorder, "_eta", {}).items(),
                key=lambda pair: str(pair[0]))}
        return {
            "length": self._joint_probe_json_value(length.as_dict()),
            "arrivals": self._joint_probe_json_value(arrivals.snapshot()),
            "efficiency": self._joint_probe_json_value(factors.as_dict()),
            "service_samples": {
                "version": int(getattr(recorder, "version", 0)),
                "efficiency_factors": eta,
                "updates": dict(sorted(getattr(recorder, "updates", {}).items())),
                "rejected": dict(sorted(getattr(recorder, "rejected", {}).items())),
                "anomalous": dict(sorted(getattr(recorder, "anomalous", {}).items())),
                "contended": dict(sorted(getattr(recorder, "contended", {}).items())),
                "residuals": self._joint_probe_json_value(
                    getattr(recorder, "residuals", {})),
                "hardware_versions": sorted(
                    getattr(recorder, "_hardware_versions", {})),
                "pending": pending,
            },
            "horizon": {
                "version": int(getattr(
                    getattr(self, "_joint_horizon", None), "version", -1)),
            },
            "epoch_log": self._joint_probe_json_value(
                self._joint_epoch_log_or_default().as_dict()),
        }

    def _joint_causal_prestate(self, now_ns: int, *, stage: str,
                               request_id: str) -> dict:
        """强制 P/D 前的完整因果事实（probe 专用，直接逐字段比较）。

        捕获必须发生在 selector 返回且任何强制选择/执行副作用之前；所有
        版本、队列、目录、容量、仲裁和统计值均原样投影，不新增摘要哈希。
        """
        snapshot = self._joint_decision_snapshot(int(now_ns))
        prestate = {
            "schema_version": "joint-causal-prestate-v1",
            "stage": stage,
            "request_id": request_id,
            "decision_snapshot": snapshot.to_dict(),
            "work_remaining": self._joint_probe_work_view(),
            "directory": self._joint_probe_directory_view(),
            "capacity": self._joint_probe_capacity_view(),
            "arbitration": self._joint_probe_arbitration_view(),
            "history": self._joint_probe_history_view(),
            "versions": {
                "kv_ledger_epoch": getattr(self, "_kv_ledger_epoch", None),
                "directory_version": getattr(
                    getattr(self.kv_manager, "_kv_directory", None),
                    "version", None),
                "stage_epochs": [
                    [str(key), value]
                    for key, value in sorted(
                        (getattr(self.kv_manager, "_stage_epochs", {}) or {}).items(),
                        key=lambda pair: str(pair[0]))],
                "length": int(getattr(
                    self._joint_length_history_or_default(), "version", -1)),
                "arrivals": int(getattr(
                    self._joint_arrival_stats_or_default(), "version", -1)),
                "efficiency": int(getattr(
                    self._joint_service_samples_or_default(), "version", -1)),
            },
        }
        return self._joint_probe_json_value(prestate)

    def _joint_capacity_view(self):
        """§4.1 CapacityView 生产面（kv_manager 只读投影；无生产者时
        None——快照层显式登记"容量面缺席"，不静默 None）。"""
        kv_manager = getattr(self, "kv_manager", None)
        if kv_manager is None:
            return None
        try:
            instances_view = {}
            for state in self.instances:
                instances_view[state.index] = {
                    # 式(19) 分量分明容量账：(C, W, O, Z) 逐 rank。
                    "capacity_ledger_by_rank": [
                        list(entry) for entry
                        in kv_manager.capacity_ledger_by_tp_rank(
                            state.index)],
                    # 摘要引用面：版本/误差来源/总量（完整 θ 桶已在决策
                    # 行 capacity_summary_view，此处不重复拷贝）。
                    "summary_version": getattr(
                        kv_manager, "capacity_summary_version", None),
                    "summary_error_sources": list(getattr(
                        kv_manager, "capacity_summary_error_sources",
                        ()) or ()),
                }
            return {"instances": instances_view}
        except (AttributeError, TypeError):
            # __new__/SimpleNamespace 替身：无 kv_manager 容量接口——
            # 返回 None 走快照层缺席登记（不虚构零容量）。
            return None

    def _joint_frozen_queue_view(self) -> dict:
        """冻结队列视图（快照 request_queues 白名单投影的输入）。

        只含因果可见量：逐实例排队/活跃计数 + 已暴露 decode 步
        （``decode_tokens_consumed``——白名单投影自动剔除 decode_length
        等隐藏未来字段）。"""
        queues: dict = {"instances": {}}
        for state in self.instances:
            queues["instances"][state.index] = {
                "qp": len(getattr(state, "qp", ()) or ()),
                "waiting_decode_admissions": len(
                    getattr(state, "waiting_decode_admissions", ()) or ()),
                "active_decode": len(
                    getattr(state, "active_decode", ()) or ()),
            }
        exposed = {}
        for runtime in getattr(self, "runtime_by_request_id", {}).values():
            if runtime is not None:
                exposed[runtime.request_id] = (
                    runtime.decode_tokens_consumed)
        queues["exposed_decode_steps"] = exposed
        return queues

    def _joint_world_source(self, now_ns: int):
        """W5（A3）：决策级背景世界源——账本在场走 LedgerWorldSource
        （权威剩余量投影 + 逐键注册表降级），缺席（旧对端/fixture）显式
        降级 LoadsWorldSource 并置 ``_joint_world_source_status``（run 末
        manifest 披露，不静默空背景）。

        W7（future_demand 消费钩子，最小面）：future_demand_mode !=
        none 且 J-on 时包裹 ForecastAugmentedWorldSource（recent/
        reverting 背景模板注入；所有候选共享同一情景；none/J-off 不包
        裹——逐字节零漂移）。"""
        from joint.action_work_graph import LedgerWorldSource
        # __new__ 测试替身不经基类 __init__（无 joint_work_ledger）——与
        # 旧对端同款缺席路径（getattr，非替身软门：真 __init__ 恒设该属性，
        # 官方在线主恒走 ledger 支）。
        ledger = getattr(self, "joint_work_ledger", None)
        absent_seen = getattr(self, "work_ledger_absent_seen", False)
        source = None
        if (ledger is not None
                and not absent_seen
                and (ledger.full_load_count > 0
                     or ledger.delta_load_count > 0)):
            self._joint_world_source_status = "ledger"
            source = LedgerWorldSource(
                ledger=ledger,
                snapshot=self._joint_decision_snapshot(now_ns),
                link_id_map_fn=self._ledger_link_id_endpoint,
            )
        else:
            self._joint_world_source_status = "absent_legacy_peer"
        # W7：future_demand != none 且 J-on 时包裹预测世界源（降级路径也
        # 包裹——内源 LoadsWorldSource，status 后缀披露）。
        if (self._joint_prediction_config_or_load().future_demand_mode
                != "none"
                and self._joint_mode == "joint"):
            from joint.action_work_graph import LoadsWorldSource
            from joint.background_forecast import (
                ForecastAugmentedWorldSource)
            inner = source if source is not None else LoadsWorldSource()
            if source is None:
                self._joint_world_source_status = (
                    "absent_legacy_peer+forecast")
            return ForecastAugmentedWorldSource(
                inner,
                engine=self._joint_forecast_engine_or_default(),
                now_ns=now_ns,
                session_return_floor_offset_ns=(
                    self._joint_forecast_floor_offset(now_ns)))
        return source

    def _ledger_link_id_endpoint(self, link_id: int):
        """账本 NoC 流路由的 C++ 链路 id → (src, dst) 端点换算（A5' 表
        首样本惰性构建；表未建 = None——该流记入 approximation 披露，
        不虚构链路键）。"""
        table = getattr(self, "_telemetry_link_id_map", None)
        if not table:
            return None
        return table.get(int(link_id))

    def _length_estimator_version(self) -> int:
        """W6 条件剩余估计器版本（快照失效键 history 的并联分量）。"""
        estimator = getattr(self, "_joint_length_history", None)
        return int(estimator.version) if estimator is not None else -1

    def _joint_cost_model(
            self, now_ns: int,
            obligation_resolver=None) -> JointCostModel:
        """决策时点构造代价模型（只读快照；无资源副作用）。

        R3'/R15（2026-09-14）：负载视图补 reclaimable（驱逐等待定价区分
        逐出可解/深缺口）；route_paths_fn 给全部 TP 并行 shard 路径
        （F-B 并集除数）；D04（2026-09-26）起池路径仲裁份额按流查询
        实际 edge——`nearest_edge_fn`/`rank_route_fn`/`instance_ranks_
        fn`/`pool_edge_flows_fn` 四注入（式 7 逐 shard；旧实例级
        pool_divisor_fn 通道退役，E 内核 r_j 保留于 KVCacheManager）。

        C3（WP1c，2026-09-22）→ D08（2026-09-27，v2.0 §13.2 式(16)）
        改宗：decode 计价上下文化从"构造期传 decode_context_tokens ×
        horizon 折回 per-token 均值"退役为**一次前向服务模板注入**——
        ``decode_task_load_ns_fn(z, hbm_share_divisor)``（z = 该前向 KV
        可见上下文；式(6) 阶段份额分母由 JCM 按候选阶段传入）。P 默认
        延续（H_z = H+U × n̂_D）与 D 边界（H_B × n̂_B）的上下文/时域由
        JCM 两入口现场取（式 18b 的 H_z 口径），构造期不再传参；剩余
        步总量 = K=min(8,n̂) 区段中点折叠（fold_repeat_service_ns，
        式 16），不逐 n̂ 循环（**W3/v4-projection 起 decode 正式路径
        已退役折叠——事件前推逐步精确 Στ(z_k)；fold 仅存 N4 prefill
        整 chunk 折叠消费面**）。旧机制退役原因：逐步求和循环 O(n̂) 挂在
        构造处，且 per-token 均值无法携带候选阶段 HBM 份额（D06 预登
        记的 decode 份额接线随本卡落地）。``decode_ns_per_token`` 标量
        位 = context=1 单步名义率（模板缺省的夹具/离线回退锚，模板在
        场时不消费——prefill_ns_per_token 的 N4 名义率同族）。"""
        loads = {}
        # D11（2026-09-27，v2.0 §14）：容量驱逐摘要冻结快照一次取全实
        # 例（KVCacheManager 增量维护；查询 O(桶数×rank 数)，不随已完
        # 成 session 累计数增长）——候选经 load view 只查询；逐出真实
        # T/E/victim 决策保留在 _ensure_capacity 原位。
        # ACC（验收纪律第 3 类，v2.0 §19.3）：摘要查询墙时即"摘要维护
        # 费"查询侧样本（P/D 两入口共用本构造器，逐次入账）；事件侧
        # （_refresh_capacity_summary）维护样本经 manager notifier 进同
        # 一通道（装配点 = __init__ kv_manager 构造后）。
        _summary_maintenance_start = time.perf_counter_ns()
        capacity_summary = self.kv_manager.capacity_summary_snapshot()
        self.predictor_cost.note_summary_maintenance(
            time.perf_counter_ns() - _summary_maintenance_start)
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
                capacity_summary=capacity_summary.instance(state.index),
                capacity_ledger_by_tp_rank=(
                    self.kv_manager.capacity_ledger_by_tp_rank(state.index)),
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

        def rank_route(source_rank: int, target_rank: int):
            # D04：rank 级确定性 XY 路径（池腿 edge↔rank 端点段同源）。
            from face_scheduler import deterministic_xy_route
            return deterministic_xy_route(
                self.hardware, source_rank, target_rank)

        prefill_ns_per_token = float(
            estimate_prefill_task_load_ns(
                self.hardware, self.model,
                instance_size=self.topology.instances[0].size,
                chunk_tokens=1, context_tokens=1))
        # D08（2026-09-27，v2.0 §13.2 式(16)）：decode 一次前向服务模板
        # 与标量名义率（见本方法 docstring 的 C3→D08 改宗段）。标量位
        # = context=1 单步 roofline（模板缺省的回退锚，模板在场时不
        # 消费）。
        decode_ns_per_token = float(self._decode_step_load_ns(
            instance_size=self.topology.instances[0].size, context=1))

        def decode_task_load_ns(context_tokens: int, *,
                                hbm_share_divisor: int = 1) -> int:
            # D08 固定成本模板：一次 decode 前向的基础服务（d_batch=1、
            # d_token=z；z 随步增长——长/短上下文 decode 服务差异经此
            # 进入候选计价；hbm_share_divisor = 候选阶段 HBM 份额分母
            # ，式 6）。单步求值走既有 memo（_decode_task_load_cache）。
            return self._decode_step_load_ns(
                instance_size=self.topology.instances[0].size,
                context=int(context_tokens),
                hbm_share_divisor=hbm_share_divisor)

        # Formal projection profiles: expose the FACE FLOP floor and complete
        # model/activation/KV HBM work separately.  The profile callbacks do
        # not turn an already roofline-maxed service time into a fake compute
        # floor or derive bytes as service×bandwidth; candidate-stage HBM
        # sharing remains applied by the task-load callback and event graph.
        profile_instance_size = self.topology.instances[0].size

        def prefill_work_profile(input_tokens: int,
                                 history_tokens: int) -> tuple[int, int]:
            profile = estimate_prefill_workload(
                self.hardware, self.model,
                instance_size=profile_instance_size,
                input_tokens=int(input_tokens),
                history_tokens=int(history_tokens),
                p_chunk=self.p_chunk)
            return profile.pure_compute_ns, profile.hbm_bytes_per_rank

        def decode_work_profile(context_tokens: int) -> tuple[int, int]:
            profile = estimate_iteration_workload(
                self.hardware, self.model,
                instance_size=profile_instance_size,
                p_chunk=0, d_batch=1, d_token=int(context_tokens),
                p_context_tokens=0)
            return profile.pure_compute_ns, profile.hbm_bytes_per_rank

        def prefill_pure_compute_ns(input_tokens: int,
                                    history_tokens: int) -> int:
            return prefill_work_profile(input_tokens, history_tokens)[0]

        def prefill_hbm_bytes(input_tokens: int, history_tokens: int) -> int:
            return prefill_work_profile(input_tokens, history_tokens)[1]

        def prefill_latency_profile(
                input_tokens: int, history_tokens: int) -> tuple[int, int]:
            profile = estimate_prefill_workload(
                self.hardware, self.model,
                instance_size=profile_instance_size,
                input_tokens=int(input_tokens),
                history_tokens=int(history_tokens),
                p_chunk=self.p_chunk)
            return profile.hbm_latency_ns, profile.controller_latency_ns

        def decode_pure_compute_ns(context_tokens: int) -> int:
            return decode_work_profile(context_tokens)[0]

        def decode_hbm_bytes(context_tokens: int) -> int:
            return decode_work_profile(context_tokens)[1]

        def decode_latency_profile(context_tokens: int) -> tuple[int, int]:
            profile = estimate_iteration_workload(
                self.hardware, self.model,
                instance_size=profile_instance_size,
                p_chunk=0, d_batch=1, d_token=int(context_tokens),
                p_context_tokens=0)
            return profile.hbm_latency_ns, profile.controller_latency_ns

        # C8（WP2，冻结交接接口）：桥遥测实测有效速率 {link_id: B/ns}
        # （C6 link_telemetry[] → _ingest_link_telemetry）按冻结契约喂
        # JCM 构造——JCM 侧消费（divisor_effective/transfer_factor）=
        # 并行卡 C7。F6 判定：此处 getattr 保留（非替身软门）——探测的
        # 是 JointCostModel 类的 dataclass 字段（joint_cost_model.py，
        # 他卡交付面），跨模块 schema 可选语义；self 侧 rates 已改直接
        # 访问（C7 已交付该字段，分支恒走通）。
        telemetry_kwargs = {}
        # §9.6（joint 二期，2026-10-03）：die 后端（manifest compute-
        # cost-model == die → self.die_model_context 在场）的预测侧换
        # 同源 DiePlan——名义 service 查询与投影 compute leg 描述符都
        # 出自 _JointDiePlanProvider（估算/推进全部经共享库 ctypes；
        # hbm_share_divisor 语义 = 内存服务率 B/s，service factor 只缩
        # 放 service_units）。legacy roofline 注释面（任务负载模板/
        # 双量分解）在该后端不注入。
        die_provider = None
        if getattr(self, "die_model_context", None) is not None:
            # 阶段2 coarse 默认（2026-10-18）：coarse_die_v2 下不构造 legacy
            # §9.6 供给器——die_plan_projection 冻结语义（阶段1）裁定
            # coarse 计划走 native shared Runtime 的冻结 profile，legacy
            # execution factory / service factor / hbm_share_divisor 语义
            # 不适用（_JointDiePlanProvider 构造期的
            # require_legacy_projection_context 即该裁定的 fail-closed 钉）。
            # 供给器缺席 = JCM/工作图的既有合法形态：名义聚合量走
            # FaceHardware 的 coarse typed 标量（coarse_die_face_peak_
            # scalars：int4 峰值/1e12 + 两行 HBM 峰值/1e9），投影腿走
            # native projection 的 coarse wire；真实执行延迟由 C++ 侧
            # CoarseDieRuntime（coarse_die_task_v2 ET 载荷）承担。
            from online.die_model_adapter import (
                COARSE_DIE_PROFILE,
                COARSE_DIE_TASK_SCHEMA,
            )
            coarse_backend = (
                getattr(self.die_model_context, "task_schema", None)
                == COARSE_DIE_TASK_SCHEMA
                or getattr(self.die_model_context, "profile", None)
                == COARSE_DIE_PROFILE)
            if not coarse_backend:
                # 供给器跨决策持久（估算 memo + 库内 PlanCache 复用；oracle
                # 进程内单例）；service factors 经 getter 决策时点读取。
                die_provider = getattr(self, "_joint_die_plan_provider", None)
                if die_provider is None:
                    die_provider = _JointDiePlanProvider(
                        die_context=self.die_model_context,
                        tp_degree=self.kv_manager.tp_degree,
                        layers=self.model.layers,
                        chunk_tokens=int(self.p_chunk),
                        hbm_bytes_per_ns=(
                            float(self.hardware.local_hbm_bandwidth_gbps)),
                        factors_getter=lambda: (
                            self._joint_factors.prefill_factor,
                            self._joint_factors.decode_factor),
                        mlp_variant=str(getattr(
                            self.model, "mlp_variant", "gelu")),
                    )
                    self._joint_die_plan_provider = die_provider
        if "link_telemetry_rates" in getattr(
                JointCostModel, "__dataclass_fields__", {}):
            telemetry_kwargs["link_telemetry_rates"] = dict(
                self._link_telemetry_rates)
        # M1 + D13：时间加权活跃流数（含 collective）——JCM 遥测除数
        # 取流数口径（物理除数，无"下游瓶颈"歧义）；字段缺席（旧二进
        # 制速率-only 形态）= 流数显式未知——除数退在册流近似，旧
        # capacity/速率满载退 1 的冒充口径已废除（v2.0 §9.2 不做清单）。
        if self._link_telemetry_flow_counts and (
                "link_telemetry_flow_counts" in getattr(
                    JointCostModel, "__dataclass_fields__", {})):
            telemetry_kwargs["link_telemetry_flow_counts"] = dict(
                self._link_telemetry_flow_counts)
        # §4.4/A03（2026-09-30 修复）：max(在册, 遥测) 合并的时间窗随
        # 视图登记（TelemetryLinkFlowView.disclosure 的 telemetry_window_
        # ns / 冲突记录面）——字段在位守卫同上（跨模块 schema 可选语
        # 义）；__new__ 测试替身不经 __init__（getattr，非替身软门——
        # 真 __init__ 恒设该属性）。
        _window_bounds = getattr(
            self, "_telemetry_last_window_bounds_ns", None)
        if ("link_telemetry_window_ns" in getattr(
                JointCostModel, "__dataclass_fields__", {})
                and _window_bounds is not None):
            telemetry_kwargs["link_telemetry_window_ns"] = tuple(
                _window_bounds)
        # N4（2026-09-23 复核审计4）：prefill 整段负载注入生产同形实现
        # （首 chunk 逐值 + 整 chunk K=min(8,m) 区段折叠 + 尾 chunk 单独
        # ——D08 有界折叠改宗，见 _joint_prefill_total_load_ns；线性外
        # 推丢二次形状，prefill 此前无同款）。
        if (die_provider is None and "prefill_task_load_ns_fn" in getattr(
                JointCostModel, "__dataclass_fields__", {})):
            telemetry_kwargs["prefill_task_load_ns_fn"] = (
                self._joint_prefill_total_load_ns)
        if die_provider is not None:
            if "prefill_task_load_ns_fn" in getattr(
                    JointCostModel, "__dataclass_fields__", {}):
                telemetry_kwargs["prefill_task_load_ns_fn"] = (
                    die_provider.prefill_nominal_ns)
        # D08（2026-09-27，v2.0 §13.2 式(16)）：decode 一次前向服务模板
        # 注入（上下文 z + 式(6) 阶段份额分母入参；W3/v4-projection 起
        # JCM 端消费 = 事件前推逐步精确求值——K=min(8,n̂) 折叠仅存 N4
        # prefill 整 chunk 面）。与 prefill_task_load_ns_fn 同款字段在位守卫
        # （跨模块 schema 可选语义，C8/F6 同族）。
        # §9.6（joint 二期）：die 后端的 roofline 工作模板/双量分解注释
        # 不注入（预测侧同源 DiePlan 经 die_plan_provider；die 分支不读
        # 这些槽位——注入即构成第二套公式面）。
        if die_provider is None:
            if "decode_task_load_ns_fn" in getattr(
                    JointCostModel, "__dataclass_fields__", {}):
                telemetry_kwargs["decode_task_load_ns_fn"] = (
                    decode_task_load_ns)
            if "prefill_pure_compute_ns_fn" in getattr(
                    JointCostModel, "__dataclass_fields__", {}):
                telemetry_kwargs["prefill_pure_compute_ns_fn"] = (
                    prefill_pure_compute_ns)
            if "prefill_hbm_bytes_fn" in getattr(
                    JointCostModel, "__dataclass_fields__", {}):
                telemetry_kwargs["prefill_hbm_bytes_fn"] = prefill_hbm_bytes
            if "decode_pure_compute_ns_fn" in getattr(
                    JointCostModel, "__dataclass_fields__", {}):
                telemetry_kwargs["decode_pure_compute_ns_fn"] = (
                    decode_pure_compute_ns)
            if "decode_hbm_bytes_fn" in getattr(
                    JointCostModel, "__dataclass_fields__", {}):
                telemetry_kwargs["decode_hbm_bytes_fn"] = decode_hbm_bytes
        else:
            if "decode_task_load_ns_fn" in getattr(
                    JointCostModel, "__dataclass_fields__", {}):
                telemetry_kwargs["decode_task_load_ns_fn"] = (
                    die_provider.decode_step_nominal_ns)
            if "die_plan_provider" in getattr(
                    JointCostModel, "__dataclass_fields__", {}):
                telemetry_kwargs["die_plan_provider"] = die_provider
        if "prefill_latency_profile_fn" in getattr(
                JointCostModel, "__dataclass_fields__", {}):
            telemetry_kwargs["prefill_latency_profile_fn"] = (
                prefill_latency_profile)
        if "decode_latency_profile_fn" in getattr(
                JointCostModel, "__dataclass_fields__", {}):
            telemetry_kwargs["decode_latency_profile_fn"] = (
                decode_latency_profile)
        # W5（A3）：背景世界源——work_ledger 在场（官方对端）注入
        # LedgerWorldSource（权威剩余量投影；逐键注册表降级不静默空背景）；
        # 缺席（旧对端/直构 fixture）= None → JCM 缺省 LoadsWorldSource
        # （显式降级标记 self._joint_world_source_status，run 末 manifest
        # 披露）。
        world_source = self._joint_world_source(now_ns)
        world_source_kwargs = (
            {"world_source": world_source}
            if world_source is not None else {})
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
            # D04（2026-09-26，v2.0 C02/式 7）：池路径逐 shard 计价的
            # 拓扑/端口供给——真实 edge（KVCacheManager.nearest_edge
            # 同源）、rank 级确定性 XY 路径、实例逐 TP rank 空间、池边
            # 缘端口在册在途流数（_PoolPortRegistry.count；不含候选自
            # 身，自身份额由计价原语显式加）。旧 pool_divisor_fn 实例
            # 级最大除数通道退役（E 内核 r_j 通道保留于 KVCacheManager）。
            nearest_edge_fn=self.kv_manager.nearest_edge,
            rank_route_fn=rank_route,
            instance_ranks_fn=lambda index: tuple(
                self.topology.instance(index).ranks),
            pool_edge_flows_fn=self._pool_ports.count,
            # C2（WP1b）：实例 HBM 端口流注册表（u_port 除数供给，F4）。
            # F6 销账：__init__ 恒设 _hbm_ports——直接传入（替身漏设 =
            # AttributeError，不再静默退回端点腿独占带宽口径）。D04 起
            # remote_load/remote_store 端点同表（_register_transfer_
            # flows 登记边界同步放宽）。
            hbm_port_registry=self._hbm_ports,
            # M2（2026-09-26 消融矩阵前置）：动作掩码 JOINT_ALLOWED_
            # ACTIONS 同源注入——计价侧 applicable_actions 尾部过滤与
            # 执行侧 _classify_physical_feasibility 第二过滤点共用同一
            # 开关位（杜绝"计价已剔除、可行性判定仍判"的窗口矛盾）。
            allowed_actions=self.joint_config.allowed_action_set,
            # W3：正式计价切换的预测配置注入（v4-projection 世代；
            # known_work/frozen_diagnostic、settlement_policy、
            # control_latency_ns、前推预算均经此面进入 JCM）。
            prediction_config=(
                self._joint_prediction_config_or_load()),
            **world_source_kwargs,
            # W5（C6）：义务解析器（None = 无 ledger 降级披露——边界
            # 入口注入 t_B 义务表；P 准入恒 None）。
            **({"obligation_resolver": obligation_resolver}
               if obligation_resolver is not None else {}),
            **telemetry_kwargs,
        )

    # ---------------------- §5.4 ARRIVAL 批内缓存（两级，缺省关） ------

    _ARRIVAL_SELECTION_CACHE_CAPACITY = 256

    def _joint_arrival_model_key(self, now_ns: int) -> tuple:
        """§5.4（零语义性能修改方案 2026-10-16）：P 准入代价模型缓存键
        （MODEL_KEY）。命中的充分条件是"键相同 ⟹ 新建 JointCostModel 与
        缓存实例在计价面上逐位同值"。组成（方案逐项 + 保守补强项）：

        * 方案列出：KV 账本纪元（改法D 九变更点）、逐实例
          (index, ledger_epoch)（任务负载快照失效键的既有先例，now_ns
          非取值输入）、horizon.version、length_history.version（两者是
          快照的隐藏输入，M3/N14）、遥测版本（_ingest_link_telemetry 两
          出口 bump）、流表状态版本（LinkFlowRegistry 原语 bump；SH 侧
          _pool_ports/_hbm_ports 与本表成对登记注销，故同时覆盖端口表）、
          服务因子版本（ServiceFactors 取值真实变化处 bump）、work_
          ledger.version、预测源 scenario_key（future_demand != none 且
          J-on 时的背景预测情景键；其余世界源 scenario_key 恒 ""）、
          id(prediction_config)（_joint_prediction_config_or_load 惰性
          单例）；
        * 保守补强（多 bump 只损失命中不错复用，向 _joint_decision_
          snapshot 缓存键看齐）：work_ledger.event_watermark、kv 目录
          version、容量摘要 capacity_summary_version、阶段事务纪元表
          （_stage_epochs）、pipeline ledger state_version——这五者是
          世界源/负载视图的增量维护版本面，独立于上列纪元推进时宁可
          失效；
        * future_demand != none 且 J-on 时并入 int(now_ns)：
          ForecastAugmentedWorldSource.build_plan 按 now_ns 前推，跨 tick
          复用不保真（缺省 none 档不入，保住方案的跨 tick L2 命中面）。

        quota 代数不入键（只影响重试门，配额判据照常逐决策重跑——
        L2 只缓存 select_instance_and_action 的 SelectionRecord）。
        """
        ledger = getattr(self, "joint_work_ledger", None)
        manager = getattr(self, "kv_manager", None)
        pipeline_ledger = getattr(self, "joint_pipeline_ledger", None)
        prediction = self._joint_prediction_config_or_load()
        scenario_key = ""
        forecast_active = (
            prediction.future_demand_mode != "none"
            and self._joint_mode == SCHEDULER_MODE_JOINT)
        if forecast_active:
            scenario_key = (
                self._joint_forecast_engine_or_default().scenario_key())
        return (
            int(getattr(self, "_kv_ledger_epoch", 0)),
            tuple((state.index, state.ledger_epoch)
                  for state in self.instances),
            int(self._joint_horizon.version),
            int(self._length_estimator_version()),
            int(getattr(self, "_link_telemetry_version", 0)),
            int(getattr(self._joint_flows, "state_version", 0)),
            int(getattr(self._joint_factors, "service_version", 0)),
            (int(getattr(ledger, "version", -1)),
             int(getattr(ledger, "event_watermark", -1)))
            if ledger is not None else (-1, -1),
            # 保守补强组（世界源/负载视图的增量维护版本面）。
            getattr(getattr(manager, "_kv_directory", None),
                    "version", None),
            getattr(manager, "capacity_summary_version", None),
            tuple(sorted(
                (getattr(manager, "_stage_epochs", {}) or {}).items())),
            (int(getattr(pipeline_ledger, "state_version", -1))
             if (pipeline_ledger is not None
                 and getattr(pipeline_ledger, "initialized", False))
             else None),
            scenario_key,
            id(prediction),
            int(now_ns) if forecast_active else None,
        )

    def _joint_arrival_cost_model(self, now_ns: int):
        """§5.4 L1：P 准入代价模型的批内复用（{MODEL_KEY → JCM}，有界
        2——当前键 + 上一键，FIFO 淘汰，不随 run 增长）。

        键不变 ⟹ 新建模型计价面逐位同值（组成见 _joint_arrival_model_
        key）；键变化 ⟹ 必然 miss 重建，不会错复用。构造期 flush 的幂等
        论证（方案②）：服务样本仅两通道——列车核销（统一估计器喂样点
        与实例纪元/KV 纪元 bump 同刻）与遥测 ingest（telemetry_version
        bump 点）——键不变 ⟹ 无新样本 ⟹ __post_init__ 的 flush() 重跑是
        no-op，跳过构造（= 跳过 flush）不改变任何因子读数；VERIFY 影子
        双路重算在每次使用处对该命题做逐字段断言兜底。

        miss 建新模型后按**构造后**的键入库：构造期 flush 若推进了
        service_version（本就是 miss），按旧键入库只会制造永不命中的
        死槽（版本单调），按新键入库保证刚建好的模型下一决策可复用。
        """
        cache = getattr(self, "_joint_arrival_model_cache", None)
        stats = getattr(self, "_joint_arrival_cache_stats", None)
        if cache is None or stats is None:
            # __new__ 壳替身不经 __init__（F6 同款兼容）：退直构路径。
            return self._joint_cost_model(now_ns)
        key = self._joint_arrival_model_key(now_ns)
        model = cache.get(key)
        if model is not None:
            stats["l1_hits"] += 1
            # 重置逐决策计算量账（cfg2）：复用模型的账面是自构造起的
            # 累计值，直接汇总会双计；归 None 走 _projection_accounting_
            # or_default 惰性零账重建，本决策定价增量照常累计——与新建
            # 模型的"每决策零账起步"同口径。
            model._projection_cost_accounting = None
            return model
        stats["l1_misses"] += 1
        model = self._joint_cost_model(now_ns)
        store_key = self._joint_arrival_model_key(now_ns)
        cache[store_key] = model
        while len(cache) > 2:
            cache.pop(next(iter(cache)))
        return model

    def _joint_arrival_selection(self, now_ns: int, cost_model,
                                 session_view, request_view):
        """§5.4 L2：P 准入选择记录的批内复用并顺带执行 VERIFY 影子断言。

        L2 键 = (MODEL_KEY, request_view, session_view,
        tuple(sorted(prep.items())), mode)：RequestView/SessionKVView 是
        frozen dataclass 直接可哈希；prep_frontier（D03 准备腿 frontier）
        入候选键（方案要求）；mode 入键（joint/load-first/affinity/
        face_static 的选择规则不同）。quota 代数不入键——配额判据
        （_quota_filter_candidates）、强制钩子（_apply_forced_entry_
        choice）与事务段在调用方照常逐决策执行，本缓存只替代
        select_instance_and_action 的全候选计价遍历。边界决策
        （select_decode_target_and_action，带 obligation_resolver）不
        入 L2（方案⑤红线）。

        SelectionRecord 的消费面核实为纯读（配额过滤/强制钩子经
        _dataclass_replace 换新对象，diagnostics 无写入点），缓存返回
        共享引用安全；候选的 projection_context 已在首次选择的 finally
        释放（close 幂等），命中路径不再经选择入口故无二次释放义务。

        VERIFY（SH_JOINT_ARRIVAL_CACHE_VERIFY=1）：命中或未命中都完整
        重算 uncached 路径（新建模型 + 全候选遍历）并逐字段断言——
        零收益全检查，对"键不变 ⟹ 计价逐位同值"的命题做运行期兜底；
        断言剔除 projection_stats["wall_ns"]（宿主墙时，仪器位）。
        """
        cache = getattr(self, "_joint_arrival_selection_cache", None)
        stats = getattr(self, "_joint_arrival_cache_stats", None)
        prep_frontier = self._prep_frontier_by_instance()
        if cache is None or stats is None:
            record = select_instance_and_action(
                mode=self._joint_mode,
                cost_model=cost_model,
                session=session_view,
                request=request_view,
                prep_frontier_by_instance=prep_frontier,
            )
            return record
        key = (
            self._joint_arrival_model_key(now_ns),
            request_view,
            session_view,
            tuple(sorted(prep_frontier.items())),
            self._joint_mode,
        )
        record = cache.get(key)
        if record is not None:
            stats["l2_hits"] += 1
        else:
            stats["l2_misses"] += 1
            record = select_instance_and_action(
                mode=self._joint_mode,
                cost_model=cost_model,
                session=session_view,
                request=request_view,
                prep_frontier_by_instance=prep_frontier,
            )
            cache[key] = record
            while len(cache) > self._ARRIVAL_SELECTION_CACHE_CAPACITY:
                cache.pop(next(iter(cache)))
        if getattr(self, "_joint_arrival_cache_verify", False):
            # 影子双路重算：新建模型（含 __post_init__ flush）+ 全候选
            # 遍历，与缓存面逐字段断言。select 只读、投影上下文在
            # finally 释放，双跑无泄漏；验证样本/计数面差 = 仪器噪声。
            shadow_model = self._joint_cost_model(now_ns)
            shadow_record = select_instance_and_action(
                mode=self._joint_mode,
                cost_model=shadow_model,
                session=session_view,
                request=request_view,
                prep_frontier_by_instance=prep_frontier,
            )
            stats["verify_checks"] += 1
            if not self._arrival_records_equal(record, shadow_record):
                stats["verify_mismatches"] += 1
                raise RuntimeError(
                    "SH_JOINT_ARRIVAL_CACHE_VERIFY divergence: cached "
                    "arrival selection (l1/l2) differs from the freshly "
                    "recomputed path for request {} (model key state "
                    "evolved without a key component bump -- unsound "
                    "reuse; aborting)".format(request_view.request_id))
        return record

    @staticmethod
    def _arrival_records_equal(cached, shadow) -> bool:
        """§5.4 VERIFY 断言谓词：两 SelectionRecord 逐字段相等。

        覆盖面：mode / instance_rule_note / tie_break_note / diagnostics；
        chosen 身份（instance_index, action）；逐候选 instance_index、
        action、applicable、inapplicable_reason、cost_ns 与完整
        ActionCostBreakdown（frozen dataclass 全字段——ActionCandidate
        的 projection_* 三字段 compare=False 天然排除）。breakdown.
        projection_stats 剔除 "wall_ns" 键（本次前推宿主墙时，逐次必
        异的仪器位）后比较。
        """
        for field_name in ("mode", "instance_rule_note",
                           "tie_break_note", "diagnostics"):
            if (getattr(cached, field_name)
                    != getattr(shadow, field_name)):
                return False
        if (cached.chosen.instance_index != shadow.chosen.instance_index
                or cached.chosen.action != shadow.chosen.action
                or cached.chosen.cost_ns != shadow.chosen.cost_ns):
            return False
        if len(cached.candidates) != len(shadow.candidates):
            return False
        for cached_item, shadow_item in zip(
                cached.candidates, shadow.candidates):
            if (cached_item.instance_index != shadow_item.instance_index
                    or cached_item.action != shadow_item.action
                    or cached_item.applicable != shadow_item.applicable
                    or (cached_item.inapplicable_reason
                        != shadow_item.inapplicable_reason)
                    or cached_item.cost_ns != shadow_item.cost_ns):
                return False
            cached_breakdown = cached_item.breakdown
            shadow_breakdown = shadow_item.breakdown
            if (cached_breakdown is None) != (shadow_breakdown is None):
                return False
            if cached_breakdown is None:
                continue
            # 仪器位剔除：projection_stats["wall_ns"] = 本次前推宿主墙时，
            # 逐次必异、与计价语义无关（§1.1 *_wall_ns 豁免同族）——两侧
            # 一律改写剔除后再比整体，其余全字段（含 projection_stats
            # 的计数/scenario 面）逐位相等。
            def _strip_wall_ns(breakdown):
                return _dataclass_replace(
                    breakdown,
                    projection_stats={
                        key: value
                        for key, value in breakdown.projection_stats.items()
                        if key != "wall_ns"})
            if _strip_wall_ns(cached_breakdown) != _strip_wall_ns(
                    shadow_breakdown):
                return False
        return True

    # --------------------------------------- C5 决策日志 schema（F8 冻结） --

    def _joint_decision_route_hops(self, cost_model, session_view,
                                   instance_index: int) -> int:
        """C5 字段 4：逐候选 hop 披露——与 JCM ``_route`` 同语义
        （source = resident ?? home；source 缺失或同实例恒 0 跳）。
        route_fn 是确定性 XY 路由函数，重复求值零副作用，且与
        estimate_P_with_continuation 计价内 breakdown.hops 同值（单测冗余断言位）。
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
        """C5 字段 1 + D17 扩展 + §5.1/§5.6/§5.5 交付面：逐候选
        ActionCostBreakdown 全 31 字段序列化（``_JOINT_BREAKDOWN_LOG_
        FIELDS`` 冻结字段序）。仅 applicable 候选调用（体量控制）；值
        可为 None（C2 物化前），字段存在性与可解析性为本卡判据；元组
        （含嵌套——区间布局/分阶段空间/背景延迟对照）与映射（projection_
        stats/capacity_theta_prediction）递归转 list/dict 保留（JSON 可
        序列化）。"""
        if breakdown is None:
            return None

        def _jsonify(value):
            if isinstance(value, tuple):
                return [_jsonify(item) for item in value]
            if isinstance(value, dict):
                return {key: _jsonify(item)
                        for key, item in value.items()}
            return value

        payload = {}
        for name in _JOINT_BREAKDOWN_LOG_FIELDS:
            value = getattr(breakdown, name, None)
            payload[name] = _jsonify(value)
        return payload

    def _account_decision_wall(self, start_ns: int, *, stage=DECISION_STAGE_P,
                               candidates: int = 0) -> None:
        """C11 步骤 4：纯决策段墙时累计（总量/次数/最大值）。

        ACC（验收纪律第 3 类，v2.0 §19.3）：同一差分同步喂入预测器开
        销账本（逐次样本 → run 末 P50/P95/P99）；C11 聚合面口径不变。
        """

        elapsed = time.perf_counter_ns() - start_ns
        self._admission_decision_wall_ns_total += elapsed
        self._admission_decision_wall_ns_max = max(
            self._admission_decision_wall_ns_max, elapsed)
        self._admission_decision_count += 1
        self.predictor_cost.note_decision(
            stage, elapsed, candidates=candidates)

    def _note_wall_split(self, *, premodel_ns: int, enum_ns: int,
                         boundary: bool = False) -> None:
        """墙时分段累计（2026-10-05 埋点）：premodel（视图+代价模型构
        建）/ enum（候选枚举遍历）两段入账；boundary=True 入 D 边界账
        并对遍历计数。getattr 零值补建——__new__ 壳替身直调决策路径
        的 F6 同款兼容（生产经 __init__ 恒设账本）。"""
        if boundary:
            self._boundary_premodel_wall_ns_total = (getattr(
                self, "_boundary_premodel_wall_ns_total", 0) + premodel_ns)
            self._boundary_enum_wall_ns_total = (getattr(
                self, "_boundary_enum_wall_ns_total", 0) + enum_ns)
            self._boundary_decision_count = (
                getattr(self, "_boundary_decision_count", 0) + 1)
        else:
            self._admission_premodel_wall_ns_total = (getattr(
                self, "_admission_premodel_wall_ns_total", 0) + premodel_ns)
            self._admission_enum_wall_ns_total = (getattr(
                self, "_admission_enum_wall_ns_total", 0) + enum_ns)

    def _joint_die_estimate_stats(self):
        """die 估值计数/墙时埋点的读取点（2026-10-05）：provider 惰性
        创建（die 后端未启用/从未触发 = None 如实披露）；__new__ 壳
        替身无 provider 属性同路径兜底。"""
        provider = getattr(self, "_joint_die_plan_provider", None)
        if provider is None:
            return None
        snapshot = getattr(provider, "estimate_timing_snapshot", None)
        return snapshot() if callable(snapshot) else None

    def _joint_decision_cost_diagnostics(self, cost_model=None) -> dict:
        """Bounded cost-accounting snapshots attached to each P/D decision.

        Cumulative counters remain useful if the run aborts before its normal
        final metrics row.  The provider delta is against only the previous
        decision snapshot, so memory stays constant and completed decisions
        carry both per-decision increments and cumulative totals.
        """
        die_current = self._joint_die_estimate_stats()

        def counter_delta(current, previous):
            if not isinstance(current, dict):
                return current
            result = {}
            for key, value in current.items():
                old_value = (previous.get(key)
                             if isinstance(previous, dict) else None)
                if isinstance(value, dict):
                    result[key] = counter_delta(value, old_value)
                elif (isinstance(value, int)
                      and not isinstance(value, bool)):
                    old_count = (old_value if isinstance(old_value, int)
                                 and not isinstance(old_value, bool) else 0)
                    result[key] = value - old_count
                elif value is None:
                    result[key] = None
            return result

        die_delta = None
        if isinstance(die_current, dict):
            previous = getattr(
                self, "_joint_last_decision_die_estimate_stats", None)
            die_delta = counter_delta(die_current, previous)
            # estimate_timing_snapshot returns fresh, bounded counter dicts.
            self._joint_last_decision_die_estimate_stats = die_current

        projection_totals = self._joint_projection_cost_summary()
        by_stage = projection_totals.get("projections_by_stage")
        if isinstance(by_stage, dict):
            projection_totals["projections_by_stage"] = dict(by_stage)

        decision_projection = None
        accessor = getattr(cost_model, "projection_cost_accounting", None)
        if callable(accessor):
            decision_projection = dict(accessor())
            for key, value in tuple(decision_projection.items()):
                if isinstance(value, dict):
                    decision_projection[key] = dict(value)

        return {
            "die_estimate": die_current,
            "die_estimate_delta": die_delta,
            "projection_cost_accounting": projection_totals,
            "decision_projection_cost_accounting": decision_projection,
        }

    def decision_wall_timing_totals(self) -> dict:
        """墙时分段累计的只读副本（2026-10-05 埋点；run 末 online_service
        汇总行消费）：P 准入 / D 边界两入口的 premodel（视图+代价模型
        构建）与 enum（候选枚举遍历）段累计。与 joint_decision_metrics
        收尾行同源同刻。"""
        return {
            "admission_premodel_ns_total": getattr(
                self, "_admission_premodel_wall_ns_total", 0),
            "admission_enum_ns_total": getattr(
                self, "_admission_enum_wall_ns_total", 0),
            "boundary_decision_count": getattr(
                self, "_boundary_decision_count", 0),
            "boundary_premodel_ns_total": getattr(
                self, "_boundary_premodel_wall_ns_total", 0),
            "boundary_enum_ns_total": getattr(
                self, "_boundary_enum_wall_ns_total", 0),
        }

    def _joint_accumulate_projection_cost(self, cost_model, *,
                                          stage: str = "") -> None:
        """cfg2（§5.6/§9.2/§10.1 计算量组）：一次全候选遍历的前推计算
        量汇入 run 级累计（P 准入与 D 边界两入口共用；JCM 按决策构造，
        其账面 = 本遍历的世界快照命中/重建 + 逐候选前推事件/闭包/超限
        计数）。__new__ 壳替身/SimpleNamespace 夹具无计价面时不添账
        （F6 同款替身兼容，快照形态回落——生产 JCM 恒有该接口）。

        stage（2026-10-05 埋点）= "P"/"D" 时把本次遍历的投影计数按
        阶段分列入 projections_by_stage；墙时分段键（候选图构建 +
        前推粗相三段 + 背景参考运行计数）随原合并循环一同累计。"""
        accessor = getattr(cost_model, "projection_cost_accounting", None)
        if not callable(accessor):
            return
        accounting = accessor()
        totals = getattr(self, "_joint_projection_cost_totals", None)
        if totals is None:
            # 部分装配的 __new__ 壳替身直调计价路径（test_joint_decision_
            # schema 夹具族）：零值账惰性补建，如实记账不 AttributeError。
            totals = {
                "decisions_priced": 0,
                "world_context_builds": 0,
                "world_context_cache_hits": 0,
                "projections": 0,
                "projection_events_total": 0,
                "projection_events_max": 0,
                "closure_resources_max": 0,
                "closure_works_max": 0,
                "background_drain_truncated": 0,
                "projection_budget_exceeded": 0,
                # 墙时分段（2026-10-05 埋点）累计位 + 按阶段投影计数。
                "build_wall_ns_total": 0,
                "closure_wall_ns_total": 0,
                "candidate_world_wall_ns_total": 0,
                "reference_world_wall_ns_total": 0,
                "reference_world_runs": 0,
                "projections_by_stage": {"P": 0, "D": 0},
            }
            self._joint_projection_cost_totals = totals
        totals["decisions_priced"] += 1
        for key in ("world_context_builds", "world_context_cache_hits",
                    "projections", "projection_events_total",
                    "background_drain_truncated",
                    "projection_budget_exceeded",
                    "build_wall_ns_total", "closure_wall_ns_total",
                    "candidate_world_wall_ns_total",
                    "reference_world_wall_ns_total",
                    "reference_world_runs"):
            totals[key] += int(accounting.get(key, 0))
        if stage in ("P", "D"):
            totals["projections_by_stage"][stage] += int(
                accounting.get("projections", 0))
        for key in ("projection_events_max", "closure_resources_max",
                    "closure_works_max"):
            totals[key] = max(totals[key], int(accounting.get(key, 0)))
        scale = accounting.get("world_context_scale")
        if scale is not None:
            self._joint_projection_world_scale_last = dict(scale)

    def _joint_projection_cost_summary(self) -> dict:
        """cfg2 计算量组的 run 级收尾聚合（joint_decision_metrics 行的
        projection_cost_accounting 键）：累计计数 + 快照/索引规模终值 +
        预算超限触发率（§9.2"报告触发率"；幸存 run 的 fatal 超限恒 0——
        超限 fail-closed 即 run 终止，触发率>0 只在异常路径计数可见）。
        getattr 兜底零账形态 = __new__ 壳替身（verify_run_end 直调的测
        试夹具不挂计价账——零值如实，不造样本）。"""
        totals = dict(getattr(self, "_joint_projection_cost_totals", None)
                      or {
                          "decisions_priced": 0,
                          "world_context_builds": 0,
                          "world_context_cache_hits": 0,
                          "projections": 0,
                          "projection_events_total": 0,
                          "projection_events_max": 0,
                          "closure_resources_max": 0,
                          "closure_works_max": 0,
                          "background_drain_truncated": 0,
                          "projection_budget_exceeded": 0,
                          "build_wall_ns_total": 0,
                          "closure_wall_ns_total": 0,
                          "candidate_world_wall_ns_total": 0,
                          "reference_world_wall_ns_total": 0,
                          "reference_world_runs": 0,
                          "projections_by_stage": {"P": 0, "D": 0},
                      })
        scale_last = getattr(self, "_joint_projection_world_scale_last", None)
        totals["world_context_scale_last"] = (
            dict(scale_last) if scale_last is not None else None)
        totals["projection_budget_exceeded_rate"] = (
            totals["projection_budget_exceeded"]
            / max(1, totals["projections"]))
        totals["boundary_fallback_extra_traversals"] = getattr(
            self, "_joint_boundary_fallback_count", 0)
        return totals

    def _note_action_selection(self, action: str, candidates,
                               session_view) -> None:
        """C11 步骤 5（弃赛守卫埋点）：二动作分列选中计数。
        F6 销账：__init__ 恒设计数表——直接访问（替身漏设 =
        AttributeError）。"""
        counts = self._joint_action_selection_counts
        counts[action] += 1

    def _joint_decision_audit_fields(self, record, cost_model,
                                     session_view, flow_snapshot) -> dict:
        """C5（F8 冻结，2026-09-22）：joint_admission 决策行的审计扩展
        字段——可重建"每决策 × 每候选 × 每字段"的完整决策时刻审计流。

        - ``selected_action``：选中动作 ∈ {stay, copy}
          （与既有 joint_action 同值，作显式冗余
          断言位）；
        - ``applicable_actions``：适用动作集（ACTION_ORDER 序、可由
          candidates 导出，显式冗余断言位）；
        - ``load_view``：逐决策负载视图（InstanceLoadView 构造点数据：
          instance_index + 五个负载量字段，决策时刻快照）；
        - ``capacity_summary_view``：逐实例容量驱逐摘要冻结块（D17 可
          重算：稳定 summary_id + 逐桶逐 rank R/clean/dirty/原子数/最
          大原子/shard_ratio/写回端口——load_view 聚合 reclaimable 标
          量之外的完整 θ 计价输入，行内 JSON、O(实例×桶×rank) 有界）；
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
        # D17 可重算（load_view 聚合标量之外的完整计价输入）：逐实例冻
        # 结容量驱逐摘要桶（_capacity_summary_audit_entry——稳定
        # summary_id + 逐桶逐 rank R/clean/dirty/原子/ratio/写回端口，
        # 行内 JSON；仅凭日志即可重算 θ 份额与桶选择）。
        capacity_summary_view = [
            _capacity_summary_audit_entry(view)
            for _, view in sorted(cost_model.loads.items())
        ]
        # C11 步骤 6（port_snapshot 接通）：配额 on = 实测值替换 NA——
        # u_port 分解读 C2 注册表（hbm_port_flow_registry.snapshot()，
        # 逐实例取其 rank 端口的最忙值——TP 全 rank 同账下各 rank 同
        # 计数，多传输 rank 子集差异取 max = 约束端口口径）；bulk 名额
        # 已用/上限读配额 tracker（实例级端口）。配额 off（缺省）保持
        # C5 冻结的 "NA" 占位（F8：不缺席）——off 臂决策日志与 C11 前
        # 逐字节同。（parity_gate_headroom 披露键已随 2026-10-14 远读
        # 退役删除——端口平价门对象已删，D7/D9。）
        if self._quota_tracker is not None:
            tracker = self._quota_tracker
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
                    }
                    for index in sorted(cost_model.loads)
                ],
            }
        # §4.4/A03（2026-09-30 修复）：候选 NoC 竞争除数基的不确定性标
        # 记 + 注册表/遥测冲突记录（覆盖/时间窗/来源）随决策行落盘——
        # max 合成除数是实际计价输入，其不确定性必须可核账。鸭子替身
        # （合成 record/直构 SimpleNamespace）无该接口时按关闭态同值披
        # 露（与无遥测生产模型一致，键恒在场）。
        _noc_disclosure_fn = getattr(
            cost_model, "link_telemetry_disclosure", None)
        return {
            "selected_action": chosen.action,
            "applicable_actions": applicable_actions,
            "load_view": load_view,
            "capacity_summary_view": capacity_summary_view,
            "flow_snapshot": flow_snapshot,
            "port_snapshot": port_snapshot,
            "noc_telemetry_divisor": (
                _noc_disclosure_fn() if callable(_noc_disclosure_fn)
                else {"telemetry": False}),
            **self._joint_decision_cost_diagnostics(cost_model),
        }

    def _apply_forced_boundary_choice(
            self, record, request_id: str, stage: str, *, now_ns=None,
            capture_causal_prestate: bool = False):
        """同状态 probe boundary 臂的单一裁决点（仅 D 边界接线）。

        probe 合同（轮三 Y 修复）：boundary 臂只强制 D 边界、不碰 P 入
        口——P 面强制一律走 ``_apply_forced_entry_choice``（entry 臂）。
        曾有 P 准入侧接线（修复轮三 R8 删除）：boundary 强制会在到达 D
        边界前先被 P 面适用性核验 fail-closed（D 面可适用的动作在
        turn-0 P 面恒不适用）。

        测量仪器语义（不是双路径开关——工作区"直接替换规范"明示豁免）：
        env 只指定 (request_id, action) 或 W8 起的 (request_id,
        instance_index, action)，本方法在**选择器产出 record 之后**把指
        定请求的最终选择强制为该候选——record.candidates 原样保留
        （式 1b 计价输入与披露面不变），只是最终选择被强制。

        两种武装形态（W8，方案 §11.3"不能只按 action 汇总最便宜
        instance"）：
        - per-action（JOINT_FORCE_BOUNDARY_ACTION，弃用方向）：多实例可
          用时按 D 边界键式(1c) (cost, boundary_order_key) 取最优实例；
        - instance×action（JOINT_FORCE_BOUNDARY_TARGET）：**精确匹配**
          (instance_index, action)——指定实例不在候选面 / 该组合不适用
          ⇒ 显式 RuntimeError（含同动作各实例的不适用理由），**不静默
          重选其他实例**。

        强制动作必须在本时刻适用面（掩码 + 物理适用性）内，否则
        fail-closed——测量仪器不得制造非法状态。钩子 off（env 未设或
        请求不匹配）恒原样返回，行为零变化。

        返回 (record, forced_note)：forced_note 非 None 时决策行落
        ``forced_by_probe`` 披露键（完整强制身份：stage/hook/action/
        instance_index/is_continuation；off 时键缺席）。
        """
        if (self._force_boundary_request is None
                or self._force_boundary_request != request_id):
            target = self._force_boundary_target
            if target is None or target[0] != request_id:
                return record, None
            _, forced_instance, action = target
            hook = "boundary-target"
        else:
            action = self._force_boundary_action
            forced_instance = None
            hook = "boundary"
        if capture_causal_prestate:
            if now_ns is None:
                raise RuntimeError(
                    "probe causal prestate requires the D decision tick")
            self._joint_last_probe_causal_prestate = (
                self._joint_causal_prestate(
                    int(now_ns), stage=stage, request_id=request_id))
        if forced_instance is None:
            applicable = [
                candidate for candidate in record.candidates
                if candidate.action == action and candidate.applicable]
            if not applicable:
                reasons = sorted({
                    candidate.inapplicable_reason
                    for candidate in record.candidates
                    if candidate.action == action
                    and candidate.inapplicable_reason is not None})
                if not reasons:
                    reasons = ["action not in candidate surface"]
                raise RuntimeError(
                    f"{FORCE_BOUNDARY_ACTION_ENV}={action!r} is outside the "
                    f"applicable surface for request {request_id!r} at its "
                    f"{stage}-boundary (inapplicable reasons: {reasons}); the "
                    "same-state probe is a measurement instrument and must "
                    "not manufacture illegal states")
            if stage == DECISION_STAGE_D:
                best = min(applicable, key=lambda candidate: (
                    candidate.cost_ns, candidate.boundary_order_key()))
            else:
                best = min(applicable, key=lambda candidate: (
                    candidate.cost_ns, candidate.order_key()))
        else:
            # W8 instance×action 精确强制：唯一候选 = (instance, action)
            # 完全匹配且适用；不适用/缺位 = 显式失败（不静默重选实例）。
            matching = [
                candidate for candidate in record.candidates
                if candidate.action == action
                and candidate.instance_index == forced_instance]
            applicable = [
                candidate for candidate in matching if candidate.applicable]
            if not applicable:
                reasons = sorted({
                    candidate.inapplicable_reason
                    for candidate in matching
                    if candidate.inapplicable_reason is not None})
                if not reasons:
                    reasons = [
                        "instance/action pair not in candidate surface"]
                same_action_instances = sorted(
                    candidate.instance_index for candidate in record.candidates
                    if candidate.action == action)
                raise RuntimeError(
                    f"{FORCE_BOUNDARY_TARGET_ENV}={forced_instance}:{action}"
                    f" is outside the applicable surface for request "
                    f"{request_id!r} at its {stage}-boundary (inapplicable "
                    f"reasons: {reasons}; candidates with action {action!r} "
                    f"exist at instances {same_action_instances}); the "
                    "same-state probe is a measurement instrument and must "
                    "not manufacture illegal states or silently re-select "
                    "a different instance")
            best = applicable[0]
        forced_note = {
            "stage": stage,
            "hook": hook,
            "action": action,
            "instance_index": best.instance_index,
            "is_continuation": bool(best.is_continuation),
        }
        return _dataclass_replace(record, chosen=best), forced_note

    def _apply_forced_entry_choice(
            self, record, request_id: str, *, now_ns=None,
            capture_causal_prestate: bool = False):
        """P 入口单点强制钩子的裁决点（修复轮二任务 7，2026-09-27）。

        语义与 ``_apply_forced_boundary_choice`` 同款（测量仪器不是双
        路径开关）：env 指定 (request_id, action) 或 W8 起 (request_id,
        instance_index, action)，本方法在选择器产出 record 之后把指定请
        求的 **P 准入最终选择**强制为该候选——record.candidates 原样保
        留（计价照常算全部候选、决策行照常披露逐候选适用性与代价、决策
        墙照常记账），只是最终选择绕过 argmin 被强制；per-action 形态多
        实例可用时按 P 选择器同序 (cost, order_key) 取最优实例，TARGET
        形态精确匹配 (instance, action)（不适用/缺位 = 显式失败，不静默
        重选实例）。强制候选必须在（掩码 + 物理适用性 ∩ 配额可行）面内，
        否则 fail-closed——仪器不得制造非法状态。只作用于 P 入口（D 边
        界不受本钩子影响）。off（env 未设或请求不匹配）恒原样返回，行
        为零变化。

        返回 (record, forced_note)：forced_note 非 None 时决策行落
        ``forced_by_probe`` 披露键（stage="P"、hook="entry"/"entry-
        target"、完整强制身份；off 时键缺席）。
        """
        if (self._force_entry_request is None
                or self._force_entry_request != request_id):
            target = self._force_entry_target
            if target is None or target[0] != request_id:
                return record, None
            _, forced_instance, action = target
            hook = "entry-target"
        else:
            action = self._force_entry_action
            forced_instance = None
            hook = "entry"
        if capture_causal_prestate:
            if now_ns is None:
                raise RuntimeError(
                    "probe causal prestate requires the P decision tick")
            self._joint_last_probe_causal_prestate = (
                self._joint_causal_prestate(
                    int(now_ns), stage=DECISION_STAGE_P,
                    request_id=request_id))
        if forced_instance is None:
            applicable = [
                candidate for candidate in record.candidates
                if candidate.action == action and candidate.applicable]
            if not applicable:
                reasons = sorted({
                    candidate.inapplicable_reason
                    for candidate in record.candidates
                    if candidate.action == action
                    and candidate.inapplicable_reason is not None})
                if not reasons:
                    reasons = ["action not in candidate surface"]
                raise RuntimeError(
                    f"{FORCE_ENTRY_ACTION_ENV}={action!r} is outside the "
                    f"applicable surface for request {request_id!r} at its "
                    "P-entry admission (inapplicable reasons: "
                    f"{reasons}); the same-state probe is a measurement "
                    "instrument and must not manufacture illegal states")
            best = min(applicable, key=lambda candidate: (
                candidate.cost_ns, candidate.order_key()))
        else:
            # W8 instance×action 精确强制：唯一候选 = (instance, action)
            # 完全匹配且适用；不适用/缺位 = 显式失败（不静默重选实例）。
            matching = [
                candidate for candidate in record.candidates
                if candidate.action == action
                and candidate.instance_index == forced_instance]
            applicable = [
                candidate for candidate in matching if candidate.applicable]
            if not applicable:
                reasons = sorted({
                    candidate.inapplicable_reason
                    for candidate in matching
                    if candidate.inapplicable_reason is not None})
                if not reasons:
                    reasons = [
                        "instance/action pair not in candidate surface"]
                same_action_instances = sorted(
                    candidate.instance_index for candidate in record.candidates
                    if candidate.action == action)
                raise RuntimeError(
                    f"{FORCE_ENTRY_TARGET_ENV}={forced_instance}:{action}"
                    f" is outside the applicable surface for request "
                    f"{request_id!r} at its P-entry admission (inapplicable "
                    f"reasons: {reasons}; candidates with action {action!r} "
                    f"exist at instances {same_action_instances}); the "
                    "same-state probe is a measurement instrument and must "
                    "not manufacture illegal states or silently re-select "
                    "a different instance")
            best = applicable[0]
        forced_note = {
            "stage": DECISION_STAGE_P,
            "hook": hook,
            "action": action,
            "instance_index": best.instance_index,
            "is_continuation": bool(best.is_continuation),
        }
        return _dataclass_replace(record, chosen=best), forced_note

    def _prep_frontier_by_instance(self) -> dict:
        """D03 修复轮二（2026-09-27）：逐实例"准备腿最早发射时刻"（相
        对决策 tick ns；空闲 = 0）。

        构图事实（graph_batch_builder "frontier 接续裁决"）：准入动作
        链到该 rank 当前 frontier——目标实例列车在飞时，P 准备腿物理排
        在该列车之后（"决策时刻发射"只在空闲时成立）。本量 = 在飞列车
        的冻结剩余工作估计：

        - chunk 分量：在飞列车 prefill_chunk_spans 的 roofline 负载
          （与 _task_load_snapshot 的 running 分量同式同源）；
        - decode 分量：在飞列车 members 的 participation 步（本列车
          冻结快照的**真实步数**，非 horizon 估计）逐成员按一次前向
          服务模板折算（与 _task_load_snapshot 的 decode 负载同式）。

        队列工作（qp 等待 chunk / 未上车 decode 成员）**不计入**——准
        入链物理先于下一列车，准备与排队的重叠是真实执行顺序（D03 原
        口径保留）。无在飞列车 = 0（决策时刻发射）。不虚构精度：这是
        roofline 服务估计，非事件侧真值——notes 经
        ``prep_issue_frontier=P_prep_at_target_frontier(+Nns)`` 披露。

        近似边界登记（§5.1a/§5.3，2026-09-30）：本标量 frontier 只承载
        "可发射时间"一项的 roofline 近似（在飞列车剩余工作），**槽位/
        迭代状态、真实 release 与逐 rank channel/credit 门未以事件门进
        入生产前推图**（快照侧 ``resource_state["slot_gate"]`` 携带五类
        在途计数聚合快照、目录携带逐 rank 字节真值，但无消费方将其转为
        图内 slot/credit 准入门——GATE_KIND_SLOT 接口已立
        （resource_projection），生产图暂以本标量近似替代；A 清单无豁
        免条目，登记缺口非能力声明）。
        """
        frontier = {}
        for state in self.instances:
            train = state.in_flight_train
            if train is None:
                frontier[state.index] = 0
                continue
            instance_size = self.topology.instance(state.index).size
            load_ns = sum(
                self._prefill_chunk_task_load_ns(
                    instance_size=instance_size,
                    chunk_tokens=chunk_tokens,
                    context_tokens=context_tokens)
                for chunk_tokens, context_tokens in train[
                    "prefill_chunk_spans"])
            for request_id, participation in train["members"]:
                runtime = self.runtime_by_request_id.get(request_id)
                if runtime is None or participation <= 0:
                    continue
                load_ns += self._decode_task_load_ns_cached(
                    instance_size=instance_size,
                    current_context_tokens=(
                        runtime.prefill_context_tokens
                        + runtime.decode_tokens_consumed),
                    generated_tokens=runtime.decode_tokens_consumed,
                    # participation = 本列车冻结的真实步数（exact，非
                    # horizon 均值）：expected = avg − generated ⇒ 传
                    # generated + participation 恰得 participation 步。
                    average_decode_length=(
                        runtime.decode_tokens_consumed + participation),
                    running_step_fraction_remaining=1.0)
            frontier[state.index] = int(load_ns)
        return frontier

    def _try_admit_request(self, runtime, now_ns: int) -> bool:
        """joint 准入（设计方案 §1/§2/§7.1）：全 instance 候选（无容量/
        边缘/距离掩码——HBM 只影响动作计价中的驱逐等待，不做准入过滤），
        联合/顺序选择 instance × stay/copy；选择后
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
        # W9（§7.4 早到下一轮依赖限制）：distributed 模式下，会话上一轮
        # 结算未闭合（结算账本在册）时不消费——真实到达/排队照常（alarm
        # 锚 service_done+interval 不变），但数据与目录依赖未就绪，准入
        # 延迟留队（settle_done 后放行）。consolidate 缺省不进此门。
        if (self._joint_settlement_policy() == SETTLEMENT_POLICY_DISTRIBUTED
                and self._joint_session_settlement_open(runtime.session_id)):
            # (2026-10-01 审查修复) 本 False 路径显式清失败键：不清时
            # _record_admit_failure_key 会拿到**上一个无关请求**失败时
            # 留下的粘性键（docstring 的"缺省回退仅打桩/异常路径"此前
            # 被本生产路径击穿）；None → 记录新鲜缺省键，配
            # _joint_close_settlement 的纪元 bump（结算闭合 = 重开门）。
            self._last_admit_failure_key = None
            self.log_decision(
                {"kind": "joint_admission", "request_id": runtime.request_id,
                 "priority": 0},
                now_ns,
                decision={
                    "admitted": False,
                    "reason": "settlement_open_dependency_gate",
                    "settlement_policy": SETTLEMENT_POLICY_DISTRIBUTED,
                },
            )
            return False
        # C11 步骤 4（决策时延埋点）：纯决策段（视图构造 → 候选计价 →
        # 配额判据 → 选择）墙时——事务/登记段不计（那是执行提交，非
        # 决策增量）；30s 尺度实测归档、不取性能结论。
        decision_wall_start = time.perf_counter_ns()

        session_view = self._joint_session_view(runtime.session_id)
        # W9（§7.4 披露）：distributed 会话的来源面枚举（区间数/位置数
        # 分别统计；增长断言 fail-closed——超 +2 无拆分即拒）。
        _distributed_session = self.kv_manager._sessions.get(
            runtime.session_id)
        surface_disclosure = None
        if (self._joint_settlement_policy() == SETTLEMENT_POLICY_DISTRIBUTED
                and getattr(_distributed_session, "distributed_layout",
                            False)):
            surface_disclosure = self._joint_distributed_source_surface(
                runtime.session_id)
        # W5 消费切换（W6 §2）：P 准入正式 n̂_D 切到条件剩余估计器
        # （estimate_conditional——d = decode_tokens_consumed，准入时刻
        # = 0；与 D 边界同一估计器实例/规则/版本）。旧 horizon.estimate
        # 保留为诊断/先验（不进计价；其并联查询曾声称"仅供对拍披露"，
        # 但结果从未落任何披露行——2026-10-01 审查删除该死计算）。
        _conditional = self._joint_conditional_estimate(runtime)
        estimated_decode = int(_conditional.remaining_forward_steps)
        horizon_source = "conditional:" + _conditional.source
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
        # C3（WP1c）→ D08（2026-09-27，v2.0 §13.2 式 16）：decode 计价
        # 上下文/时域不再构造期传参——JCM P 入口现场取 H_z =
        # session_view.history_tokens + joint_input_tokens（准入时已知，
        # 非 oracle）× n̂_D = estimated_decode（CausalHorizonEstimator
        # 现值，与 RequestView.estimated_decode_tokens 同源同刻）；τ 一
        # 次前向服务模板由构造注入，剩余步 K=min(8,n̂) 区段中点折叠。
        # §5.4（ARRIVAL 批内缓存）：SH_JOINT_ARRIVAL_CACHE=1 时 P 准入的
        # 代价模型（L1）与选择记录（L2）走批内缓存（缺省关 = 直构路径，
        # 与改动前逐字节一致；getattr 兼容 __new__ 壳替身，F6 姿态）。
        _arrival_cache = getattr(
            self, "_joint_arrival_cache_enabled", False)
        cost_model = (self._joint_arrival_cost_model(now_ns)
                      if _arrival_cache
                      else self._joint_cost_model(now_ns))
        # 墙时分段（2026-10-05 埋点）：候选枚举遍历段
        # （select_instance_and_action 全候选计价）；premodel 段 =
        # 决策起点（会话视图/条件估计/RequestView/代价模型构建）至
        # 枚举起点。配额判据/强制钩子等枚举后段仍计入既有
        # _account_decision_wall 总账（不重复入分段）。
        _admission_enum_t0 = time.perf_counter_ns()
        record = (self._joint_arrival_selection(
                      now_ns, cost_model, session_view, request_view)
                  if _arrival_cache else
                  select_instance_and_action(
                      mode=self._joint_mode,
                      cost_model=cost_model,
                      session=session_view,
                      request=request_view,
                      # D03 修复轮二：逐候选目标实例的准备腿发射 frontier（在飞
                      # 列车剩余估计；空闲 = 0 = 旧口径）——构图事实对齐。
                      prep_frontier_by_instance=(
                          self._prep_frontier_by_instance()),
                  ))
        _admission_enum_elapsed = (
            time.perf_counter_ns() - _admission_enum_t0)
        self._note_wall_split(
            premodel_ns=_admission_enum_t0 - decision_wall_start,
            enum_ns=_admission_enum_elapsed)
        # cfg2（§5.6/§10.1 计算量组）：P 准入全候选遍历的前推计算量
        # 汇入 run 级累计（失败路径前的计价遍历照常入账——计算量是
        # 计费事实，不是披露成功面）。
        # stage="P"（2026-10-05 埋点）：投影计数按阶段分列。
        self._joint_accumulate_projection_cost(cost_model, stage="P")
        # ---- C11 步骤 1：逐候选配额判据（动作级，实例永不掩码）+ 配额
        # 可行集内重选（按 scheduler_mode 分派：joint=argmin 同序、顺序
        # 模式=可行集上重跑选点规则，见 _quota_reselect_feasible；δ_adm=0
        # 与既有比较合取恒等）。全不适用 ⇒ quota_deferred 回队
        # （C9 语义：重试键扩展配额代数，复用本类 epoch 键重试门）。
        # ----
        if self._quota_tracker is not None:
            (quota_chosen, quota_candidates,
             quota_deferred) = self._quota_filter_candidates(
                record, session_view, runtime.request_id, now_ns,
                cost_model=cost_model)
            if quota_deferred is not None:
                deferred_record, _r_hat = quota_deferred
                self._account_decision_wall(
                    decision_wall_start,
                    candidates=len(record.candidates))
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
                    cost_model=cost_model,
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
        # P 入口单点强制钩子（修复轮二任务 7；修复轮三 R8：BOUNDARY 钩子
        # 的 P 侧接线已删除——boundary 臂只强制 D 边界、不碰 P 入口，
        # P 面强制一律走 entry 臂，否则 boundary 强制会在到达 D 边界前
        # 先被 P 面适用性核验 fail-closed）。在配额可行集重选之后强制
        # 最终选择——candidates 不动（计价输入与披露面不变），强制动作
        # 出（掩码+物理适用性∩配额可行）面即 fail-closed。与边界钩子
        # 互斥武装（解析期已 fail-closed），至多一个命中——同样的披露
        # 键 forced_by_probe（stage="P"，hook="entry" 标识仪器）。
        self._joint_last_probe_causal_prestate = None
        record, forced_probe_note = self._apply_forced_entry_choice(
            record, runtime.request_id, now_ns=now_ns,
            capture_causal_prestate=True)
        causal_prestate = getattr(
            self, "_joint_last_probe_causal_prestate", None)
        chosen = record.chosen
        selected = chosen.instance_index
        self._account_decision_wall(
            decision_wall_start, candidates=len(record.candidates))
        self._note_action_selection(chosen.action, record.candidates,
                                    session_view)

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
                at_ns=now_ns,  # M4：adaptive 决策行携带决策时刻
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
                at_ns=now_ns,  # M4：restore issue 事件带发射时刻
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
                runtime, now_ns, record, str(exc), "capacity_deferred",
                cost_model=cost_model)
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
        # D18：P 提交事务成功 = P_DECISION -> P_COMMITTED（冻结已提交
        # P 操作；不冻结未提交 D——D 位置在真实 P→D 边界二次决策）。
        runtime.joint_stage = JOINT_STAGE_P_COMMITTED
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
        # C11（入册半）：选中动作的配额流生命周期入册（oneshot 支链 +
        # copy 主流程）。入册序与失
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
            # 事务成功段已登记的在途流承诺（owner=rid 历史迁移/逐出支链）
            # **先行**注销
            # （可逆半边恒回滚，幂等空放），消除"重试改选动作成功前注册表躺着幽灵承诺
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
                failure_class="quota_deferred_quota_link",
                cost_model=cost_model)
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
        # 审计扩展字段——selected_action/适用动作集/
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
                "horizon_source": horizon_source,
                "estimated_decode_tokens": int(estimated_decode),
                # ---- D17（v2.0 §18/D17，2026-09-27）决策行 schema：
                # 公式版本 + 决策阶段/纪元 + 两阶段配对 + 预测里程碑由
                # 逐候选 breakdown 承载（predicted_* 前缀 = 预测字段，
                # 与真实时间严格分列；非执行候选真值 NA——candidates[]
                # measured 槽位恒 None，受控同状态执行另行填充）。固定
                # 配置 manifest 随 run manifest.json 落盘（online_service
                # 侧车，joint_config.manifest_dict 的 model_semantics_
                # version 键）——本行以版本号键接榫。
                "model_semantics_version": MODEL_SEMANTICS_VERSION,
                "decision_stage": "P",
                "decision_epoch": 0,
                "prefill_instance": selected,
                # D18 默认延续 π0：初始决策只提交 P，decode 侧按同实例
                # 默认延续披露（本地接续）。
                "decode_instance": selected,
                "prefill_action": chosen.action,
                "decode_action": ACTION_STAY,
                "is_continuation": False,
                "estimated_remaining_decode_steps": int(estimated_decode),
                "category_mode": self.joint_config.category_mode,
                "layer_policy": self.joint_config.layer_policy,
                # C11：配额模式随行披露（run 级事实，与 category_mode
                # 等开关键对称；off 臂值恒 "off"）。
                "quota_mode": self.joint_config.quota_mode,
                # M2（2026-09-26 消融矩阵前置）：动作掩码随行披露（run
                # 级事实，与 quota_mode 等开关键对称；"all" = 无剔除）。
                "allowed_actions": self.joint_config.allowed_actions,
                "instance_rule_note": record.instance_rule_note,
                # C8（E13 唯一读者）：争用覆盖披露（值构造见
                # _contention_coverage_value——注册表层 + 遥测完备段）。
                "contention_coverage": self._contention_coverage_value(),
                "service_factors": self._joint_factors.as_dict(),
                # ---- C5（F8 冻结）决策时刻审计 schema 扩展 ----
                **self._joint_decision_audit_fields(
                    record, cost_model, session_view,
                    decision_flow_snapshot),
                # 同状态 probe 强制钩子披露（仅武装且命中本请求时落键；
                # off 时键缺席 = 行与既有钉测逐位一致）。
                **({"forced_by_probe": forced_probe_note}
                   if forced_probe_note is not None else {}),
                **({"causal_prestate": causal_prestate}
                   if causal_prestate is not None else {}),
                # W9（§7.4 披露）：distributed 会话的目录全量来源面
                # （source_count=位置数 / interval_count=区间数分别统计 +
                # 跨轮增长；consolidate/非 distributed 会话键缺席）。
                **({"distributed_source_surface": surface_disclosure}
                   if surface_disclosure is not None else {}),
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
                        # R3'.3/D11 的 capacity_wait_uncertain 等注记须可检索
                        # （§9 验收"可检索"口径），此前只进内存不落盘。
                        # notes 载于 breakdown（不适用候选无 breakdown）。
                        "notes": list(candidate.breakdown.notes)
                        if candidate.breakdown is not None else [],
                        # D17（§19.2）：候选真值槽位——本候选未实际执行
                        # 时恒 None（NA）。真值只来自受控同状态执行，不
                        # 用预测值冒充实测。
                        "measured": None,
                    }
                    for candidate in record.candidates
                ],
            },
        )
        # §5.5（2026-09-30 修复）：选中候选 θ 预测暂存（kv_eviction 行
        # 对账的决策半面；提交已定 = 预测与本请求容量执行绑定）。
        self._stash_capacity_theta_prediction(runtime, chosen)
        # cfg1（§10.3 完成预测成熟通道）：实际执行候选（且仅它）登记 P
        # 预测里程碑——settle_done 闭合点按 (request, "P") 配对实测（恰
        # 一次）；未执行候选不登记（保持 missing，measured 恒 None）。
        if chosen.breakdown is not None:
            self._joint_completion_maturity_or_default().note_decision(
                request_id=runtime.request_id,
                decision_stage="P",
                decision_tick_ns=int(now_ns),
                action=str(chosen.action),
                predicted_next_token_done_ns=(
                    chosen.breakdown.predicted_next_token_done),
                predicted_service_done_ns=(
                    chosen.breakdown.predicted_service_done),
                predicted_settle_done_ns=(
                    chosen.breakdown.predicted_merge_done),
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
        ``_release_transfer_flows(rid#evict)``（准入逐出支链）。
        调用点 =
        _quota_enroll_admission 返回 False 的主流程失败防御路径（O2：
        逐出支链失败不入此路径——已按披露降级；主流程需求判据已前瞻、
        单线程内判据与入册间 tracker 零变更，到达此处 = 真异常的安全
        网）；_release_transfer_flows 幂等空放，未登记半边无副作用。
        回队重试的再准入走全量重登记，本回滚保证其间注册表零幽灵承诺
        （除数不被死承诺虚抬）。"""
        self._release_transfer_flows(runtime.request_id)
        self._release_transfer_flows(runtime.request_id + "#evict")

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

    # ------------------------------------------------ §5.5 容量 θ 对账 --

    def _stash_capacity_theta_prediction(self, runtime, chosen) -> None:
        """§5.5（2026-09-30 修复）：选中候选提交点暂存 θ 结构化预测。

        ``chosen.breakdown.capacity_theta_prediction``（决策时点
        ``_space_ready_ns_estimate`` 的逐桶 θ 消费输出）挂到请求 runtime
        ——kv_eviction 行（同 trigger_request_id 的真实容量逐出）构造时
        对账"摘要预测 vs 实际 E"。预测缺席（零缺口/空摘要候选）不暂存
        ——kv_eviction 行对账块键缺席，行形态与无预测时一致。
        """
        breakdown = getattr(chosen, "breakdown", None)
        prediction = getattr(
            breakdown, "capacity_theta_prediction", None) \
            if breakdown is not None else None
        if prediction:
            runtime.capacity_theta_prediction = dict(prediction)
            runtime.capacity_actual_eviction_bytes = 0

    def _capacity_theta_reconciliation(
            self, request_id: str, evicted_bytes_total: int):
        """§5.5（2026-09-30 修复）：kv_eviction 行的 θ 预测 vs 实际对账块。

        * 预测半面 = 决策时点暂存的 ``released_by_rank``（θ 份额预计释
          放向量）+ 摘要版本（§5.5 "包含版本"）；
        * 实际半面 = 本行逐出字节并入同请求累计（跨多条 kv_eviction 行
          的执行侧真值）；
        * ``delta_bytes`` = 累计实际 − 预测总量。语义边界：θ 预测是"为
          满足该决策空间需求预计释放多少"的连续份额近似；实际逐出可能
          含同请求其他相的容量逐出——差值是**观测差**（供事后核账与摘
          要校准），不是单因果误差判定；下游工具按 tick/watch_id 细分。

        无暂存预测（本请求从未携带 θ 预测）返回 None——行不添键。
        """
        runtime = self.runtime_by_request_id.get(request_id)
        prediction = getattr(runtime, "capacity_theta_prediction", None)
        if not prediction:
            return None
        actual_total = (
            getattr(runtime, "capacity_actual_eviction_bytes", 0)
            + int(evicted_bytes_total))
        runtime.capacity_actual_eviction_bytes = actual_total
        predicted_by_rank = list(
            prediction.get("released_by_rank", ()) or ())
        predicted_total = sum(int(value or 0) for value in predicted_by_rank)
        return {
            "predicted_released_bytes_by_rank": predicted_by_rank,
            "predicted_total_released_bytes": predicted_total,
            "actual_evicted_bytes_this_row": int(evicted_bytes_total),
            "actual_evicted_bytes_cumulative": actual_total,
            "delta_bytes": actual_total - predicted_total,
            "summary_version": prediction.get("summary_version"),
        }

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
        trigger_request_id 供重放侧 request_id 校验复用。
        M4（2026-09-26）：decision 加 watch_id——与尾标记完成处的
        kind=kv_eviction_done 行按同键配对（差值 = 实际写回时长）。"""
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
            decision={
                "watch_id": watch_id,
                "evictions": [
                    _transfer_summary(transfer)
                    for transfer in evictions],
                # §5.5（2026-09-30 修复）：容量摘要 θ 预测 vs 实际逐出
                # 的对账块（预测 released 向量 + 实际累计 + 差值；无
                # 暂存预测的请求键缺席，行形态不变）。
                **({"capacity_theta_reconciliation": block}
                   if (block := self._capacity_theta_reconciliation(
                       trigger_request_id,
                       sum(int(transfer.total_bytes)
                           for transfer in evictions))) is not None
                   else {}),
            },
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
        "逐实例按其适用动作集合的 footprint 逐一判"）。
        M2（2026-09-26 消融矩阵前置）：动作掩码 JOINT_ALLOWED_ACTIONS
        同步剔除（第二过滤点，与 JCM applicable_actions 尾部过滤同源
        同开关位）——掩码动作不进物理可行性判定集；掩码把唯一物理可行
        动作剔除时按 structural_infeasible 上报（掩码是候选集上限不是
        强制：不可行动作不能 rescue 可行性；零适用/零可行维持
        fail-closed，非容量拒绝）。
        返回 (structural_infeasible, per_instance_detail)。"""
        final_context = (
            session_view.history_tokens + runtime.joint_input_tokens)
        detail = []
        any_feasible = False
        # M2：掩码放行集（joint_config 单一裁决点派生；"all" = 全集，
        # 无剔除零漂移）。
        allowed_actions = self.joint_config.allowed_action_set
        for action in ACTION_ORDER:
            if action not in allowed_actions:
                # M2：掩码剔除的动作不进判定集（与 JCM applicable_
                # actions 尾部过滤同源；易漏的第二过滤点）。
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
        *, cost_model=None,
    ) -> None:
        """R2.6（D5/P4）：失败也落盘——首条失败分类带全候选表；同请求
        同分类的后续失败折叠为紧凑计数行（防深尾部"延迟 × 纪元重试"
        乘积下失败日志体积逼近成功日志）。

        C11：wait 行携带 wait_reason ∈ {capacity, quota_link,
        quota_port} 分列（C9 冻结枚举）——capacity = SH 既有容量等待，
        quota_link/quota_port = 配额判据分列；等待计入从请求到达到完成
        的延迟（deferred 驻留时长另入 run 末指标行）。"""
        cost_diagnostics = self._joint_decision_cost_diagnostics(cost_model)
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
                    **cost_diagnostics,
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
                **cost_diagnostics,
                "candidates": [
                    {
                        "instance_index": candidate.instance_index,
                        "action": candidate.action,
                        "applicable": candidate.applicable,
                        "cost_ns": candidate.cost_ns,
                        "inapplicable_reason":
                            candidate.inapplicable_reason,
                        # M6（kimi 复审）：notes 随候选落决策日志——
                        # R3'.3/D11 的 capacity_wait_uncertain 等注记须可检索
                        # （§9 验收"可检索"口径），此前只进内存不落盘。
                        # notes 载于 breakdown（不适用候选无 breakdown）。
                        "notes": list(candidate.breakdown.notes)
                        if candidate.breakdown is not None else [],
                        # D17（§19.2）：候选真值槽位——本候选未实际执行
                        # 时恒 None（NA）。真值只来自受控同状态执行，不
                        # 用预测值冒充实测。
                        "measured": None,
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
        # D18：准入批发射 = P 执行开始（P_COMMITTED -> P_RUNNING）。
        runtime.joint_stage = JOINT_STAGE_P_RUNNING
        self._batch["assignments"].append({
            "request_id": runtime.request_id,
            "prefill_instance_index": runtime.prefill_instance_index,
            # prefill 边界 decode 尚未选定（D18 前 decode 固定 prefill
            # 同实例；D18 起在 P→D 边界二次决策，此处仍为决策前锚）；
            # assignment 结构校验要求非负，填 prefill 实例占位（最终
            # decode 实例以 drain 边界决策行为准）。
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
                # (sh10_online_scheduler.py)补齐 holder 的
                # shard 级序列化——_transfer_summary 只读导出 shards[].
                # noc_hops/noc_path(创建时由 deterministic_xy_route
                # 算好,零新算);既有聚合 history_transfer_bytes 保留。
                # A/B 剥离清单字段;hopbytes.py collect_sh30 读取端已
                # 前向兼容(旧 log 无新键 → 数值不变)。
                "history_transfer": (
                    _transfer_summary(runtime.history_transfer)
                    if runtime.history_transfer else None),
                # R12（2026-09-14）：水印重放的 joint 专属字段——动作/全
                # 部历史迁移口径（多动作多区间传输无法由单数字段
                # reconstruct）。
                "joint_action": runtime.joint_action,
                "joint_span_base_context": runtime.joint_span_base_context,
                "joint_prefill_work": runtime.joint_prefill_work,
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
        DECODE_COMPLETION watch 由列车 exit 标记承载）。

        D18：本行同时落两阶段边界决策产物（decision_stage /
        joint_decode_action / is_continuation / boundary_decision 摘要
        ——含 n̂_B、继续标记与全部边界候选的适用性与代价）。

        D11/D17 修复轮二（2026-09-27）：D 边界决策行补容量审计字段
        ``capacity_summary_view``（与 P 行 joint_admission 同构造
        ``_capacity_summary_audit_entry``——冻结桶结构逐位同构）与
        ``capacity_ledger_by_tp_rank``（逐实例 C/W/O/Z 台账 + 派生
        F = C−W−O−Z；P 行未携带台账，此处为 D 行新增键）——此前仅 P
        行携带容量摘要，D 阶段容量判断仅凭日志不可重算。D 行在边界
        提交后发射：两键取 emit 时点 kv_manager 快照（逐实例）。"""
        self._batch["assignments"].append({
            "request_id": runtime.request_id,
            "prefill_instance_index": runtime.prefill_instance_index,
            "decode_instance_index": runtime.decode_instance_index,
            "prefill_affinity_reason": runtime.prefill_affinity_reason,
        })
        boundary = runtime.boundary_decision
        # ---- D11/D17 修复轮二：D 边界时刻容量审计块（与 P 行
        # _joint_decision_audit_fields 的 capacity_summary_view 同一
        # 构造器/同一冻结桶 schema；台账键为 D 行新增——逐实例
        # (C, W, O, Z) 分量 + F 派生，消费方与 _space_ready_ns_estimate
        # 的式 19 交叉核账同口径）。----
        from types import SimpleNamespace as _DRowNamespace
        _capacity_snapshot = self.kv_manager.capacity_summary_snapshot()
        capacity_summary_view = []
        capacity_ledger_by_tp_rank = []
        for state in self.instances:
            capacity_summary_view.append(_capacity_summary_audit_entry(
                _DRowNamespace(
                    instance_index=state.index,
                    capacity_summary=_capacity_snapshot.instance(
                        state.index),
                    reclaimable_bytes_by_tp_rank=(
                        self.kv_manager
                        ._instance_reclaimable_capacity_by_tp_rank(
                            state.index)))))
            ledger = self.kv_manager.capacity_ledger_by_tp_rank(state.index)
            capacity_ledger_by_tp_rank.append({
                "instance_index": state.index,
                "ledger_by_tp_rank": [
                    [int(c), int(w), int(o), int(z)]
                    for c, w, o, z in ledger],
                "free_by_tp_rank": [
                    int(c) - int(w) - int(o) - int(z)
                    for c, w, o, z in ledger],
            })
        # 同状态 probe 强制钩子披露键（仅武装时在位；off 时行形态与
        # 既有钉测逐位一致——键缺席）。
        boundary_keys = (
            "chosen_instance_index", "chosen_action",
            "is_continuation", "cost_ns", "decode_steps",
            "fallback",
            # D19 披露：等待接口类别（原地接续/新接纳）、边界阶段
            # 工作视图与继承义务表。
            "boundary_wait_kind", "stage_work",
            "inherited_obligations",
            # D20 披露：阶段 epoch 与跨阶段预约事务回执。
            "stage_epoch", "stage_transaction",
            # W8（方案 §11.3 状态一致硬证书）additive：决策时点版本面
            # （kv_ledger_epoch/directory_version/stage_epochs——probe
            # 证书目录/预约维度事实源）。
            "state_versions",
            # D17 披露：边界候选逐项 breakdown（公式版本/输入来源/
            # 字节空间/关键等待可重建）与真值槽位。
            "candidates",
            # W5（§7.1/C5）披露：目标系统共同控制延迟与执行侧提交点
            # 复查面。配置面已一律拒绝正值，0 明确表示 assumed-zero。
            "control_latency_ns", "control_commit_recheck")
        if boundary is not None and "forced_by_probe" in boundary:
            boundary_keys = boundary_keys + ("forced_by_probe",)
        if boundary is not None and "causal_prestate" in boundary:
            boundary_keys = boundary_keys + ("causal_prestate",)
        # 一次映射对照臂专用披露键（仅生效时在位；off 时行形态与既有
        # 钉测逐位一致——键缺席）。
        if boundary is not None and "boundary_policy" in boundary:
            boundary_keys = boundary_keys + ("boundary_policy",)
        self.log_decision(
            {"kind": "decode", "request_id": runtime.request_id,
             "priority": 0},
            tick,
            decision={
                # 修复轮三 F7：decision_stage 词表四方对齐为 "P"/"D"
                # （P 行、候选面 notes、ACC、probe 合成行同此）；状态
                # 机细节（joint_stage = D_COMMITTED 等）已由
                # boundary_transaction/stage_epoch 披露，不进本词表位。
                "decision_stage": "D",
                # D17（v2.0 §18/D17，2026-09-27）：公式版本 + 决策纪元 +
                # 两阶段配对/剩余前向（预测里程碑由逐候选 breakdown 的
                # predicted_* 字段承载——与真实时间严格分列；候选真值
                # 槽位 measured 恒 None，受控同状态执行另行填充）。
                "model_semantics_version": MODEL_SEMANTICS_VERSION,
                "decision_epoch": runtime.joint_boundary_epoch,
                "prefill_action": runtime.joint_action,
                "estimated_remaining_decode_steps": (
                    int(boundary["decode_steps"])
                    if boundary is not None else None),
                "joint_decode_action": runtime.joint_decode_action,
                "is_continuation": runtime.joint_decode_is_continuation,
                "boundary_decision": (
                    {key: boundary[key] for key in boundary_keys}
                    if boundary is not None else None),
                **self._joint_decision_cost_diagnostics(),
                "decode_instance_index": runtime.decode_instance_index,
                # 问题 4a(S3 路由序列化,2026-09-05):对齐 S1
                # (sh10_online_scheduler.py)补齐 prefill→decode
                # 迁移与 decode 侧逐出的 shard 级序列化(_transfer_summary
                # 同上,零新算)。核销置空沿用既有 completion M4 块
                # （completion M4 置空块）,不新增提前置空——每请求恰一条 decode 行,
                # completion 行已存在,无双计。A/B 剥离清单字段。
                "prefill_decode_transfer": (
                    _transfer_summary(runtime.prefill_decode_transfer)
                    if runtime.prefill_decode_transfer else None),
                "joint_action": runtime.joint_action,
                "decode_evictions": [
                    _transfer_summary(transfer)
                    for transfer in runtime.decode_evictions],
                # ---- D11/D17 修复轮二：D 边界容量审计字段（构造见本
                # 方法头部注——与 P 行同构的冻结桶结构 + D 行新增的
                # C/W/O/Z 台账；仅凭本行即可重算 D 阶段容量判断）。----
                "capacity_summary_view": capacity_summary_view,
                "capacity_ledger_by_tp_rank": capacity_ledger_by_tp_rank,
            },
        )

    # ------------------------------------------------------------- 负载 --

    def _prefill_chunk_task_load_ns(self, *, instance_size, chunk_tokens,
                                    context_tokens,
                                    hbm_share_divisor=1) -> int:
        """offline: face_scheduler.py（缓存同款）。hbm_share_divisor
        （D06，v2.0 式(6)）入 key——不同阶段份额的求值不串缓存。"""
        key = (instance_size, chunk_tokens, context_tokens,
               hbm_share_divisor)
        return _bounded_task_load_memo(
            self._prefill_task_cache, key,
            lambda: estimate_prefill_task_load_ns(
                self.hardware, self.model,
                instance_size=instance_size, chunk_tokens=chunk_tokens,
                context_tokens=context_tokens,
                hbm_share_divisor=hbm_share_divisor),
            # F6 销账：__init__ 恒设 _task_load_cache_capacity——直接
            # 访问（替身漏设 = AttributeError）。
            capacity=self._task_load_cache_capacity,
        )

    def _joint_prefill_total_load_ns(self, input_tokens: int,
                                     history_tokens: int,
                                     hbm_share_divisor: int = 1) -> int:
        """N4：prefill 整段负载（生产同形）→ D08（2026-09-27，v2.0
        §13.1/§13.2 式(16)）有界折叠改宗：长 chunk 序列同 decode 规则
        ——首 chunk 逐值（第一个真实完整模型前向，恢复启动与稳态分开
        ，§13.1）、尾 chunk 单独计（不等长不折）、其余 m = a−1 个整
        chunk 用 K=min(8,m) 区段中点折叠（fold_repeat_service_ns，
        context 随 chunk 累计 z_k = H + k·q_c）——注入函数每候选调用
        的 τ 求值次数受 K=8 界，不随 chunk 数增长（原逐 chunk 长循环
        废除）。m ≤ 8 时逐 chunk 求值，与逐 chunk 精确和逐位一致。
        **执行侧列车 span 冻结/核销仍逐 chunk 精确**（真实工作台账，
        不入本有界规则——式(16) 只约束候选估值，两账分离）。

        D06（2026-09-27，v2.0 §9.1 式(6)）：``hbm_share_divisor`` =
        候选阶段 HBM 端口份额分母 m_e + k_e（JCM 侧由
        CandidateStageHbmShare.stage_denominator 供给）——逐 chunk
        roofline 的 HBM 带宽项按共享份额计价；缺省 1 = 独占带宽（
        台账/排队负载等非候选计价路径不传，口径不变）。
        """
        instance_size = self.topology.instances[0].size
        input_tokens = int(input_tokens)
        if input_tokens <= 0:
            return 0
        chunk_tokens_full = min(self.p_chunk, input_tokens)
        if chunk_tokens_full <= 0:
            raise ValueError("prefill chunk size must be positive")
        full_chunks, tail_tokens = divmod(input_tokens, chunk_tokens_full)
        total = 0.0
        if full_chunks >= 1:
            # 首 chunk 逐值（第一个真实完整模型前向，§13.1）。
            total += self._prefill_chunk_task_load_ns(
                instance_size=instance_size,
                chunk_tokens=chunk_tokens_full,
                context_tokens=history_tokens + chunk_tokens_full,
                hbm_share_divisor=hbm_share_divisor)
            remaining_chunks = full_chunks - 1
            if remaining_chunks > 0:
                # 其余整 chunk：K=min(8,m) 区段中点折叠（式 16）——
                # 第 k 个折叠项 = 原 (k+1) 号整 chunk，context =
                # H + (k+1)·q_c ⇒ z_k = (H + 2q_c) + (k−1)·q_c。
                total += fold_repeat_service_ns(
                    steps=remaining_chunks,
                    first_context_tokens=(
                        history_tokens + 2 * chunk_tokens_full),
                    context_stride_tokens=chunk_tokens_full,
                    service_fn=lambda context: (
                        self._prefill_chunk_task_load_ns(
                            instance_size=instance_size,
                            chunk_tokens=chunk_tokens_full,
                            context_tokens=int(context),
                            hbm_share_divisor=hbm_share_divisor)))
        if tail_tokens > 0:
            # 尾 chunk 单独计（不等长不折；context = H + U 终态）。
            total += self._prefill_chunk_task_load_ns(
                instance_size=instance_size, chunk_tokens=tail_tokens,
                context_tokens=history_tokens + input_tokens,
                hbm_share_divisor=hbm_share_divisor)
        return int(total)

    def _decode_task_load_ns_cached(self, *, instance_size,
                                    current_context_tokens, generated_tokens,
                                    average_decode_length,
                                    running_step_fraction_remaining,
                                    hbm_share_divisor: int = 1):
        """改法A：estimate_decode_remaining_task_load_ns 的全参 key memo
        （与 _prefill_task_cache 同款）。hardware/model 为运行期不可变量，
        经绑定不入 key。D08：``hbm_share_divisor``（式(6) 候选阶段 HBM
        份额分母）入 key——D08 模板的份额化求值不与独占口径串缓存；
        缺省 1 = 独占带宽（台账/代表率等非候选计价路径口径不变）。"""
        key = (instance_size, current_context_tokens, generated_tokens,
               average_decode_length, running_step_fraction_remaining,
               hbm_share_divisor)
        return _bounded_task_load_memo(
            self._decode_task_load_cache, key,
            lambda: estimate_decode_remaining_task_load_ns(
                self.hardware, self.model,
                instance_size=instance_size,
                current_context_tokens=current_context_tokens,
                generated_tokens=generated_tokens,
                average_decode_length=average_decode_length,
                running_step_fraction_remaining=running_step_fraction_remaining,
                hbm_share_divisor=hbm_share_divisor),
            # F6 销账：__init__ 恒设 _task_load_cache_capacity——直接
            # 访问（替身漏设 = AttributeError）。
            capacity=self._task_load_cache_capacity,
        )

    def _task_load_snapshot(self, state, now_ns: int):
        """offline: face_scheduler.py task_load_snapshot 的在线
        子集。三分量口径（合同⑥），每个请求恰计一次（打分公式与阈值
        不动，红线；拼 batch 改造 2026-08-22 仅重订物理折算）：
          - queued_prefill：逐 chunk Roofline 求和（函数逐行复用
            （见 _joint_prefill_total_load_ns）；自 prefill_tokens_completed 闭式推进的剩余 chunk
            折算——列车核销时推进，决策边界上与逐 chunk 精确值逐点一致）；
          - running_prefill：在飞列车冻结的队列头 chunk 负载（列车账本
            prefill_chunk_spans；原"在飞段全量剩余 × fraction=1.0"的
            不可分假设作废，改为列车级冻结量）；
          - active_decode：estimate_decode_remaining_task_load_ns 逐请求
            （N12 在线化 + N14 条件化：average_decode_length = generated
            + E[N−d|N>d]（条件剩余估计器，P 准入/D 边界同款入口，下限
            1）⇒ 折算式恰好消费条件剩余步——已消费超过会话均值的活跃
            请求保留正剩余负载，不再"均值减进度截零"估为零步，§8.2/A08；
            CausalHorizonEstimator 保留诊断/先验；generated_tokens =
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
                and state.snapshot_length_version
                == self._joint_length_history_or_default().version
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
                    == self._joint_horizon.version
                    and state.snapshot_length_version
                    == self._joint_length_history_or_default().version):
                if state.snapshot_cache != snapshot:
                    raise RuntimeError(
                        "task-load snapshot epoch cache diverged for "
                        "instance {} (epoch {})".format(
                            state.index, state.ledger_epoch))
                return state.snapshot_cache
        state.snapshot_cache = snapshot
        state.snapshot_epoch = state.ledger_epoch
        state.snapshot_horizon_version = self._joint_horizon.version
        state.snapshot_length_version = (
            self._joint_length_history_or_default().version)
        return snapshot

    def _compute_task_load_snapshot(self, state):
        """_task_load_snapshot 的实际计算体（改法S2 后形态；仅缓存未命中
        或影子验证时执行）。三分量口径见 _task_load_snapshot docstring。

        S2-B 整数精确性证明（红线：折算公式一个数都不能变）：三分量
        全部是 Python int 的加減——estimate_prefill_task_load_ns 与
        estimate_decode_remaining_task_load_ns 的返回值都是 math.ceil
        产生的 int（face_scheduler.py 的 ceil 同判据），int 加減精确、满足
        结合/交换律且无舍入，故聚合账本（入队置全量 + 核销扣减）与
        逐项 while 现算在数学上逐位相等，不存在浮点求和序漂移：
          - queued_prefill = Σ_qp 成员聚合账本 − 在飞列车冻结 chunk 负载
            （成员账本 = 自当前 prefill_tokens_completed 起的剩余 chunk
            逐键求和，由 _queued_chunk_load_full_ns 与列车核销增量维护）；
          - running_prefill / active_decode 与原始计算体相同（span 有界
            ≤ T_max、成员有界，memo 查找非热点，S2 未动）。"""
        instance_size = self.topology.instance(state.index).size
        # queued_prefill（offline blueprint 的增量等价：成员聚合账本
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
        # active_decode（offline blueprint；迭代级闭式剩余量折算）。
        # N12（R15-4，2026-09-14）：decode 剩余标定在线化（全 trace decode
        # 均值常数不再进入负载视图——决策输入零未来信息，总纲 §13.1/
        # §13.2；J-off load-first 臂同口径）。
        # N14（2026-09-30，§8.2/A08/W6）：标定切**条件剩余估计器**（P 准
        # 入/D 边界同款入口 ``_joint_conditional_estimate``——每成员
        # E[N−d|N>d]、下限 1；``average_decode_length`` 传 generated+
        # 条件剩余使折算式恰好消费条件剩余步，不再"均值减进度截零"——
        # 已消费超过会话均值的活跃请求保留正剩余负载，不再估为零步）。
        # CausalHorizonEstimator 保留诊断/先验（horizon version 仍入缓存
        # 键——冷启动先验仍经 horizon 通道）。P/D/负载视图同刻同规则
        # （W6"所有候选长度规则一致"）。
        active_decode_load_ns = 0
        for runtime in state.active_decode:
            conditional = self._joint_conditional_estimate(runtime)
            generated = int(runtime.decode_tokens_consumed)
            active_decode_load_ns += self._decode_task_load_ns_cached(
                instance_size=instance_size,
                current_context_tokens=(
                    runtime.prefill_context_tokens + generated),
                generated_tokens=generated,
                average_decode_length=float(
                    generated
                    + int(conditional.remaining_forward_steps)),
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
        # queued_prefill（offline blueprint 逐行复用；剩余 chunk 自
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
        # active_decode（offline blueprint；迭代级闭式剩余量折算）。
        # N14（2026-09-30，§8.2/A08/W6）：与生产路径同款——标定切条件
        # 剩余估计器（影子断言要求与本路径逐字段一致）。
        active_decode_load_ns = 0
        for runtime in state.active_decode:
            conditional = self._joint_conditional_estimate(runtime)
            generated = int(runtime.decode_tokens_consumed)
            active_decode_load_ns += self._decode_task_load_ns_cached(
                instance_size=instance_size,
                current_context_tokens=(
                    runtime.prefill_context_tokens + generated),
                generated_tokens=generated,
                average_decode_length=float(
                    generated
                    + int(conditional.remaining_forward_steps)),
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

        O7③（2026-09-23 终轮审计）→ D13（2026-09-27）：遥测降级三键
        披露（``telemetry_degradation`` 块）——JCM disclosure 的覆盖
        可辨认键（flow_count_links / unknown_flow_count_links，D13 起
        速率-only 链路 = 流数显式未知，除数退在册流近似，旧
        legacy_rate_divisor_links 键随 capacity/速率除数口径一并废
        除）只存在于逐决策重建的 JCM 实例内，run 级覆盖度裁决零生产
        落点；此处按 SH 账本同口径镜像三键（全部派生自现存状态，零新
        增账本）：流数未知条目（速率在场 ∧ 流数缺席的链路数——旧二
        进制缺 active_flows 字段的覆盖缺口，可见非静默）/ 仅流数条目
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
            # O7③ + D13：遥测覆盖三键（见 docstring；D13 起速率-only
            # 条目 = 流数显式未知，不再有 capacity/速率除数回退）。
            "telemetry_degradation": {
                "unknown_flow_count_entries": len(
                    rate_keys - flow_keys),
                "flow_count_only_entries": len(flow_keys - rate_keys),
                "zero_rate_samples_dropped": (
                    self._telemetry_zero_rate_dropped),
            },
        }

    # ------------------------------------------------------------ ACC --

    def _note_backend_batch(self, batch: dict) -> None:
        """ACC（验收纪律第 3 类，v2.0 §19.3）：图批事件量入预测器开销
        账本——节点/边/watch/未来 alarm 四计数与 digests 行同源同刻。"""

        self.predictor_cost.note_backend_batch(
            nodes=len(batch["nodes"]),
            parent_edges=len(batch["parent_edges"]),
            watches=len(batch["watches"]),
            future_alarms=len(batch["future_alarms"]))

    def predictor_cost_snapshot(self, *, host_wall_ns=None,
                                decision_log_bytes=None) -> dict:
        """ACC run 末快照（§19.3：决策墙时 P50/P95/P99、总占比、峰值
        RSS、摘要维护费、日志大小、后端事件量）。

        host_wall_ns 缺省 = 构造锚（__init__）到现在的宿主墙时差；
        decision_log_bytes 由调用方（online_service）在决策日志落定后
        以文件字节回填，缺省 None = 未读（不猜）。宿主 Python 墙时不
        自动等于真实片上决策时延（§19.3 末段）——快照只供验收证据。"""

        if host_wall_ns is None:
            host_wall_ns = (time.monotonic_ns()
                            - self._predictor_cost_host_anchor_ns)
        snapshot = self.predictor_cost.snapshot(
            host_wall_ns=host_wall_ns,
            decision_log_bytes=decision_log_bytes)
        # ACC 补全（v2.0 §19.3）：summary_maintenance 块并入**事件侧**
        # 增量维护分列（维护次数/累计耗时，KVCacheManager 分账）——查询
        # 侧样本（count/total_ns/max_ns）与维护侧计数同块分列，10% 预算
        # 口径一次读齐。__new__ 测试替身的 SimpleNamespace kv_manager 无
        # 该接口时不添键（快照形态回落账本原生面）。
        maintenance_stats = getattr(
            self.kv_manager, "capacity_summary_maintenance_stats", None)
        if callable(maintenance_stats):
            snapshot["summary_maintenance"]["event_maintenance"] = (
                maintenance_stats())
        return snapshot

    def verify_run_end(self) -> None:
        """基类协议校验之上，叠加离线 的收尾断言 + §7.3 结束
        审计（arrival heap / ready frontier / 列车账本全空）。"""
        super().verify_run_end()
        # W5（§7.4 G-B 门）：统一结算闭合审计——run 结束时全部请求的
        # settlement 必须已到 SETTLE_DONE（账本闭合即弹出；残留 = merge
        # watch 丢失/settle 义务未闭合，fail-closed）。
        pending_settlements = sorted(self._joint_settlement_ledger())
        if pending_settlements:
            raise RuntimeError(
                "settlement ledger not empty at run end (requests without "
                "SETTLE_DONE): {}".format(pending_settlements))
        # C8：遥测完备性收尾披露行（kind=link_telemetry_coverage）——
        # collective_coverage 翻转条件的全量证据。joint_mechanism_
        # manifest.json 侧车落字在 online_service/joint_config（本卡文件
        # 清单外），决策日志收尾行为本卡完备性档案，侧车接线归 C11 车道。
        self.log_decision(
            {"kind": "link_telemetry_coverage", "request_id": "",
             "priority": 0},
            self._telemetry_last_tick_ns,
            decision=self._telemetry_coverage_decision())
        # W9（§10 开销账）：目录空间/操作/增长 run 级披露行——条目峰值/
        # 逐会话逐轮来源面（区间数/位置数/增长）/目录操作计数与累计墙时
        # （distributed 目录提交型结算的管理开销；consolidate 下计量账
        # 恒零值形态，行照落——开销面可对照）。
        _dir_meter = getattr(
            self.kv_manager, "distributed_directory_meter", None)
        if _dir_meter is not None:
            self.log_decision(
                {"kind": "distributed_directory_stats", "request_id": "",
                 "priority": 0},
                self._telemetry_last_tick_ns,
                decision=_dir_meter.snapshot())
        # cfg1（2026-09-30，§10.3 误差成熟与归因 / §10.1 输出组 / W7 开
        # 销账收尾）：forecast_status 收尾行——背景预测引擎披露
        # （disclosure()：到达桶成熟账/开销账/回退链/排除事件，此前
        # engine 无任何生产消费出口、run 结束即丢）+ P/D 完成预测成熟
        # 账（本 run 实际执行候选的直接真值配对结果；§8.5 恰一次成熟、
        # 未完成/超时单列）。无引擎且无配对样本不落行（off/短夹具形态
        # 逐字节不变）；分组细化报告由 slo_tools/domain_metrics 离线
        # 展开（completion_error_report）。
        _forecast_status = self._joint_forecast_status_decision()
        if _forecast_status is not None:
            self.log_decision(
                {"kind": "forecast_status", "request_id": "",
                 "priority": 0},
                self._telemetry_last_tick_ns,
                decision=_forecast_status)
        # C11 步骤 4/5（指标埋点收尾行）：决策时延 + 分档动作分布 +
        # quota 等待分列/驻留时长 + AIMD 动作计数——为 C20 预置，本卡
        # 只埋点不判读；动作选中率按负载档由决策行 load_view
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
        # 墙时分段累计(getattr 零值兜底,__new__ 壳替身安全)。
        _wall_splits = self.decision_wall_timing_totals()
        # §5.4（ARRIVAL 批内缓存）：命中/未命中/影子断言计数披露——仅
        # 在缓存或 VERIFY 任一开启时落键（两开关均关时本行与改动前逐
        # 字节一致，缺省关 = 语义安全底线）；getattr 兜底壳替身。
        _arrival_disclosure = None
        if (getattr(self, "_joint_arrival_cache_enabled", False)
                or getattr(self, "_joint_arrival_cache_verify", False)):
            _arrival_disclosure = dict(getattr(
                self, "_joint_arrival_cache_stats", {}) or {})
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
                # §5.4：命中/未命中计数（新增披露字段）——仅缓存或
                # VERIFY 开启时落键，两开关均关时本行与改动前逐字节
                # 一致（缺省关 = 语义安全底线）。
                **({"arrival_cache": _arrival_disclosure}
                   if _arrival_disclosure is not None else {}),
                "admission_decision_wall_ns": {
                    "count": self._admission_decision_count,
                    "total_ns": self._admission_decision_wall_ns_total,
                    "avg_ns": (
                        self._admission_decision_wall_ns_total
                        // max(1, self._admission_decision_count)),
                    "max_ns": self._admission_decision_wall_ns_max,
                    # 墙时分段（2026-10-05 埋点）：视图+代价模型构建 /
                    # 候选枚举遍历两段累计（与 total_ns 的差 = 配额判据
                    # /强制钩子等枚举后段，不入分段）。
                    "premodel_ns_total": _wall_splits[
                        "admission_premodel_ns_total"],
                    "enum_ns_total": _wall_splits["admission_enum_ns_total"],
                },
                # 墙时分段（2026-10-05 埋点）：D 边界二次决策两段累计
                # （计数 = 全候选遍历次数；p-continuation 臂无遍历不入账）。
                "boundary_decision_wall_ns": {
                    "count": _wall_splits["boundary_decision_count"],
                    "premodel_ns_total": _wall_splits[
                        "boundary_premodel_ns_total"],
                    "enum_ns_total": _wall_splits["boundary_enum_ns_total"],
                },
                # 计数埋点（2026-10-05）：die memo 命中/未命中 + ctypes
                # 批量估值调用/规格数/墙时（die 后端未启用 = None）。
                "die_estimate": self._joint_die_estimate_stats(),
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
                # cfg2（2026-09-30，§5.6/§9.2/§10.1 计算量组）：run 级
                # 预测前推计算量聚合——世界快照命中/重建与快照/索引规
                # 模、逐候选前推事件总量/峰值、闭包规模峰值、背景排空
                # 截断、预算超限触发率与边界回退额外遍历（逐候选面在
                # 决策行 breakdown.projection_stats 分列）。
                "projection_cost_accounting":
                    self._joint_projection_cost_summary(),
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
        # O10③：R15 在途流注册表（链路/池端口/HBM 端口）全 owner 清账
        # 断言（R15 盲区的收口）。
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
    未知键，旧日志零行为差。

    C4 多腿形态（增量 copy 两腿元组 (transfer, base_leg)，W2 2026-09-27
    发射面修复）：逐腿披露，腿序 = 发射序（graph_batch_builder 对同
    元组按同序逐腿发射）。顶层保持首腿（主迁移腿）的单腿契约键
    （kind/source_instance_index/total_bytes 等——slo_tools/hbm_
    watermark 的 P→D 迁移重放按首腿 noc_migrate 校验跟踪副本字节，
    不得改为腿汇总）；shards = 全部腿物理 shard 按发射序平铺（hopbytes
    逐 shard 采集不漏腿）；``multi_leg``/``leg_count``/``legs[]`` 为
    多腿专属披露（单腿形态不落，与既有钉测逐位一致）。"""
    if transfer is None:
        return None
    if isinstance(transfer, tuple):
        # 空元组按调用点 None 语义处理（构造侧恒非空，防御位）。
        if not transfer:
            return None
        legs = [_transfer_summary(leg) for leg in transfer]
        summary = dict(legs[0])
        summary["multi_leg"] = True
        summary["leg_count"] = len(legs)
        summary["legs"] = legs
        summary["shards"] = [
            shard for leg in legs for shard in leg["shards"]]
        return summary
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
