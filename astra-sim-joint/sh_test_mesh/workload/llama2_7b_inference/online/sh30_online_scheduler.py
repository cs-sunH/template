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
from collections import deque

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
    kv_cache_shard_bytes_for_tokens,
    select_prefill_instance,
)
from joint.joint_config import parse_joint_config  # noqa: E402
from joint.joint_cost_model import (  # noqa: E402
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
        "merge_transfers", "joint_prefill_work", "joint_span_base_context",
        "joint_input_tokens", "joint_cost_ns", "joint_working_copy",
        # ---- R14（2026-09-14）：decode 增长停滞/唤醒 ----
        "decode_stalled", "stall_wake_key", "stall_reason", "stall_gap_records",
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
        # K6：停滞携带的容量缺口记录（KVCapacityError.deep_gap_records）；
        # 停滞是可恢复路径不落账，仅在死锁守卫确认终态时提交台账。
        self.stall_gap_records = ()
        self.origin_home_instance = None
        self.history_transfers = ()
        self.merge_transfers = ()
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
        }


class Sh30OnlineScheduler(OnlineSchedulerBase):
    """strategy 变体：sh_3.0 三段式准入 + decode 同实例 + 三态 KV（关感知）
    + 迭代列车拼 batch（2026-08-22）。

    蓝本：已移除的离线 plan_face_requests。拓扑 / edge_free
    掩码 / KV 账本（与离线同一函数、同参数）在 __init__ 一次性构建，运行期
    策略输入全部来自这些 Python 账本。
    """

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
        # R11（2026-09-14）：merge 尾 watch id → 待排下一轮 alarm（到达
        # 重锚 merge_done + interval；service_done 合同落地）。
        self._pending_merge_alarms: dict[str, dict] = {}
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
        # ---- R11（2026-09-14）：merge 尾 watch 交付 → 排下一轮 alarm。 ----
        # service_done 合同（设计方案 §2.3）：下一轮到达 = merge_done +
        # interval（此前锚 compute_done，thinktime < merge 时长时 merge
        # 成本逃出会话 E2E、J 异地动作边际收益被系统性高估——N2 修复）。
        # watch id 走 batch_train_merge_ 前缀（基类 _settle_completions 对
        # batch_train_ 命名空间跳过请求核销；本变体在哨兵路由前拦截）。
        merge_done_ids = []
        # ---- completion 批（offline: face_scheduler.py）----
        drained = []
        completed_now = []
        sentinel_trains = []
        for group in delta["completed_groups"]:
            stage = group["stage"]
            request_id = group["request_id"]
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
        # 纯度排除（local_hit 无传输不计；remote-read 无历史传输、其
        # 读流门控 decode 列车，在 decode 分支按成员动作排除）。
        plan["history_transfer_gated"] = bool(
            plan.get("head_first_chunk")
            and head_runtime is not None
            and any(
                transfer.kind != "local_hit" and transfer.total_bytes > 0
                for transfer in (head_runtime.history_transfers or ())))
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
        # R15-1：准入相在途流注销（历史迁移/逐出支链在 drain 边界视作
        # 完成——其消费栅栏（readiness barrier → 列车体）已物理通过）。
        self._release_transfer_flows(request_id)
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
        # joint remote-read（v1 执行口径，README 披露）：joiner 的 KV 动作
        # = 合成远读流（home→exec 的真实 NoC 字节流；总量 = 剩余 decode
        # 步 × 每步全上下文 KV 读，终态上下文保守近似）——经列车头
        # readiness barrier 门控列车体（读流未完成不开始 decode）。逐迭代
        # credit 交错流为后续项；本口径保守（读不与计算重叠）。
        if runtime.joint_action == "remote-read":
            runtime.prefill_decode_transfer = self._joint_remote_read_stream(
                runtime, selected)
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
        self.kv_manager.release_request_capacity_reservation(runtime.request_id)
        self._bump_kv_ledger_epoch()  # 改法D：KV 变更点 5/9（release_request_capacity_reservation）
        # R15-1：decode 相在途流登记（drain 边界逐出/迁移 + remote 读流；
        # 完成边界注销）。
        self._register_transfer_flows(
            runtime.prefill_evictions + runtime.decode_evictions
            + ((runtime.prefill_decode_transfer,)
               if runtime.prefill_decode_transfer is not None else ()),
            owner=request_id + "#decode")
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

        R11（N2，2026-09-14）：service_done 合同落地——下一轮到达 alarm
        重锚 **merge_done** + interval（merge 尾标记节点 watch 完成回调
        进调度器再排 alarm；无 merge 流的 stay/recompute@home 本地提交
        保持 compute_done + interval 不变）。同时下一轮 interval gate
        前递依赖 merge 尾标记（图依赖侧在构图器 _emit_completion），
        消除"下一轮物理消费尚未落地的 KV"。E2E 真实拉长为预期行为
        （合并成本进主 E2E，§2.3）。
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
        # offline: :3986-3999 decode 完成分支。下一轮 alarm 的排程判定
        # 推迟：merge 流存在与否在 merge 事务（下一循环）后才可知。
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
        # joint（§2.2/§2.3/§3.2）：compute_done 后、mark_complete 前执行
        # merge 事务——新增量归并回 origin_home（home 侧空间经 T+E 真实
        # 准备 + R4 分层自降级；目标工作副本释放）。完成时已观测的
        # decode 长度/输入长度入在线估计器（实际结算，非决策 oracle）。
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
            if following is None:
                continue
            if not runtime.merge_transfers:
                # stay / recompute@home 本地提交：无 merge 流，到达 =
                # compute_done + interval（与基底逐字节同口径）。
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
            completion_result = self.graph.emit_completion_batch(
                runtime.plan_dict())
            following, _interval = facts_by_id[request_id]
            if (following is not None and runtime.merge_transfers):
                # R11(i)：merge 尾标记 watch——完成回调经 PREFILL_DRAIN
                # 通道送达（batch_train_merge_ 前缀在 run_variant_policy
                # 拦截为 _on_merge_done：排 alarm + 注销 merge 流）。
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
                    "interval": _interval,
                    "envelope": {
                        "request_id": following.request_id,
                        "session_id": following.session_id,
                        "turn_index": following.turn_index,
                        "prefill_length": following.prefill_length,
                        "decode_length": following.decode_length,
                        "inter_request_interval_ns": _interval,
                    },
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
                    # K2/K3 配套：工作副本真值（merge 前 working_kind）。
                    "joint_working_copy": bool(
                        getattr(runtime, "joint_working_copy", False)),
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
            self, "_last_admit_failure_key", None) or (
                self._kv_ledger_epoch,
                ((0, self.instances[0].ledger_epoch),))

    def _current_retry_key(self, last_key):
        """重试键求值：KV 纪元 ⊕ last_key 记录的失败候选集实例纪元。

        last_key = (kv_epoch_at_failure, ((instance, epoch_at_failure),
        ...))——候选集冻结于失败时刻（选中实例 ∪ applicable 候选，M2）；
        当前键 = (当前 KV 纪元, 同一实例集的纪元现值)。与上次失败键
        整体相等 = 无新信息（容量未释放且候选集内无负载迁移、argmin
        不会翻转）。"""
        return (self._kv_ledger_epoch,
                tuple((index, self.instances[index].ledger_epoch)
                      for index, _ in last_key[1]))

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
                self._register_transfer_flows(
                    exc.evictions, owner=runtime.request_id + "#decode")
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
            self._register_transfer_flows(
                evictions, owner=runtime.request_id + "#decode")

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
                        self._register_transfer_flows(
                            exc.evictions,
                            owner=runtime.request_id + "#decode")
                    continue
                if evictions:
                    self._bump_kv_ledger_epoch()
                    self.graph.sync_pending_history_after_evictions(evictions)
                    # 自查 D 同款：唤醒增长的逐出进图 + 流登记。
                    self._emit_eviction_only_nodes(
                        evictions, tick,
                        trigger_request_id=runtime.request_id)
                    self._register_transfer_flows(
                        evictions, owner=runtime.request_id + "#decode")
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
        节点，C++ 必有交付 → _on_merge_done → 下一轮 alarm → 到达 →
        准入逐出可释放容量；_complete_requests 与 _admit_pass 同 tick
        先后执行，完成批刚注册 watch 的窗口里守卫必见非空）；(b)
        arrival_heap 非空（已见到达待 drain）；(c) manifest 尚有未到达
        请求（_arrived_request_count < len(runtimes) 即 C++ 侧已排
        alarm——未来到达 → 新准入 → 逐出释放；runtimes 槽位完成后置
        None，不得按槽位判）。守卫只在 EOF 终态（无任何未来事件源）
        的全停滞才触发。"""
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
        """R11：merge 尾 watch 交付 → 下一轮 alarm 重锚 merge_done + 间隔
        （service_done 合同）；同时注销该请求的 merge 在途流（R15）。"""
        pending = self._pending_merge_alarms.pop(watch_id, None)
        if pending is None:
            raise RuntimeError(
                "merge-done watch {} delivered without a pending alarm".format(
                    watch_id))
        request_id = pending["request_id"]
        self._batch["future_alarms"].append({
            "arrival_world_ns": tick + pending["interval"],
            "envelope": pending["envelope"],
        })
        self._release_transfer_flows(request_id + "#merge")

    # ------------------------------------------------------ R15 流登记 --

    def _register_transfer_flows(self, transfers, *, owner: str) -> None:
        """R15-1/F-B：发射时点登记在途流——逐 shard 完整链路序列（TP
        并行全路径，非代表对）+ 池端口占用（remote_store/remote_load 的
        edge 端口）。注销由完成事件驱动（drain/completion/merge-watch），
        决策时刻快照因果可见。"""
        for transfer in transfers or ():
            if transfer.kind == "local_hit":
                continue
            for shard in transfer.shards:
                self._joint_flows.register_path(
                    shard.noc_path, owner=owner)
                if shard.edge_rank is not None:
                    self._pool_ports.register(shard.edge_rank, owner=owner)

    def _release_transfer_flows(self, owner: str) -> None:
        self._joint_flows.release_owner(owner)
        self._pool_ports.release_owner(owner)

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
        价）；基数 = 同内容 roofline 闭式账本。transfer 因子保留接口位
        （节点级传输完成遥测未交付，updates=0 披露；传输争用的在线修正
        由链路/端口除数通道承担）。"""
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
        # remote-read 成员的逐列车门控读流同样混入传输段，一并排除）。
        if (not chunk_records and member_parts
                and not train.get("had_joiners")):
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
        # 跳过——纯度约束）。
        if (chunk_records and not member_parts
                and not train.get("had_joiners")
                and not train.get("suffix_gated")
                and not train.get("history_transfer_gated")):
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
            location = session.base_location or self.kv_manager.REMOTE_MEMORY
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

    def _joint_cost_model(self, now_ns: int) -> JointCostModel:
        """决策时点构造代价模型（只读快照；无资源副作用）。

        R3'/R15（2026-09-14）：负载视图补 reclaimable（驱逐等待定价区分
        逐出可解/深缺口）；route_paths_fn 给全部 TP 并行 shard 路径
        （F-B 并集除数）；pool_divisor_fn 给池端口仲裁份额（P1 同源）。"""
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
        decode_ns_per_token = float(
            estimate_decode_remaining_task_load_ns(
                self.hardware, self.model,
                instance_size=self.topology.instances[0].size,
                current_context_tokens=1, generated_tokens=0,
                average_decode_length=1.0))
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
        )

    def _joint_remote_read_stream(self, runtime, exec_instance: int):
        """remote-read v1 合成读流：home→exec 的 noc 传输（真实字节/链路，
        列车 readiness 门控；每步读 = 全上下文 KV，步数 = 剩余 decode，
        上下文取终态保守近似——后端工作真值生成物理量，不进决策输入）。"""
        from face_scheduler import KVTransfer, KVTransferShard
        from face_scheduler import deterministic_xy_route
        home = runtime.origin_home_instance
        if home is None or home == exec_instance:
            return None  # 无异地历史（退化：无读流）
        session = self.kv_manager._sessions[runtime.session_id]
        base_tokens = (
            session.base_history_tokens if session.working_kind is not None
            else session.context_tokens)
        context_per_step = (
            base_tokens + runtime.joint_input_tokens + runtime.decode_length)
        steps = max(1, runtime.decode_length)
        tp = self.kv_manager.tp_degree
        per_step_shards = kv_cache_shard_bytes_for_tokens(
            self.model, context_per_step, tp)
        shards = []
        source_group = self.topology.instance(home)
        target_group = self.topology.instance(exec_instance)
        for source_rank, target_rank, per_step_bytes in zip(
                source_group.ranks, target_group.ranks, per_step_shards):
            total = per_step_bytes * steps
            if total <= 0:
                continue
            shards.append(KVTransferShard(
                source_rank=source_rank,
                target_rank=target_rank,
                edge_rank=None,
                bytes=total,
                noc_path=deterministic_xy_route(
                    self.hardware, source_rank, target_rank),
                layer_start=0,
                layer_end=self.model.layers,
            ))
        return KVTransfer(
            kind="noc_migrate",
            phase="decode",
            reason="remote_read_stream",
            session_id=runtime.session_id,
            trigger_request_id=runtime.request_id,
            source_instance_index=home,
            target_instance_index=exec_instance,
            total_bytes=sum(shard.bytes for shard in shards),
            shards=tuple(shards),
            model_layers=self.model.layers,
            layer_start=0,
            layer_end=self.model.layers,
            resident_prefix_layers_before=self.model.layers,
            resident_prefix_layers_after=self.model.layers,
        )

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
        cost_model = self._joint_cost_model(now_ns)
        record = select_instance_and_action(
            mode=self._joint_mode,
            cost_model=cost_model,
            session=session_view,
            request=request_view,
            remote_enabled=self.joint_config.remote_enabled,
        )
        chosen = record.chosen
        selected = chosen.instance_index

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
            self._last_admit_failure_key = (
                self._kv_ledger_epoch,
                tuple((index, self.instances[index].ledger_epoch)
                      for index in sorted(candidate_instances)))
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

        # R15-1：准入相在途流登记（历史迁移 + 逐出支链；drain 边界注销）。
        self._register_transfer_flows(
            runtime.history_evictions + tuple(runtime.history_transfers),
            owner=runtime.request_id)

        # joint 决策日志（§7：origin_home/execution/action/成本分解；
        # R15 披露：链路/端口争用覆盖与在线因子状态）。
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
                "category_mode": self.joint_config.category_mode,
                "layer_policy": self.joint_config.layer_policy,
                "remote_enabled": self.joint_config.remote_enabled,
                "instance_rule_note": record.instance_rule_note,
                "contention_coverage": (
                    "link_flows+pool_ports" if (
                        self._joint_flows.has_registrations
                        or self._pool_ports._counts) else "cold_start"),
                "service_factors": self._joint_factors.as_dict(),
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
        self, evictions, tick: int, *, trigger_request_id: str) -> None:
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
            return
        self.graph.emit_eviction_side_branch(
            [transfer for transfer in evictions], tick)
        self.log_decision(
            {"kind": "kv_eviction", "request_id": trigger_request_id,
             "priority": 0},
            tick,
            decision={"evictions": [
                _transfer_summary(transfer) for transfer in evictions]},
        )

    def _classify_physical_feasibility(
        self, runtime, session_view,
    ) -> tuple[bool, list[str]]:
        """P5-a 三态分类：对全部 (instance, action) 组合判物理可行性
        （request_hbm_eventually_feasible_instances，动作感知口径）。

        M1（kimi 复审，2026-09-14）：按**适用动作集合**过滤（规格 R2.4
        "逐实例按其适用动作集合的 footprint 逐一判"）——remote-read 仅
        在 remote on 且基础驻留 LOCAL 时参与判定（N1(a)：PARTIAL 后缀
        在 home 不可直读；remote-off 为正交消融）。否则其 input-only
        足迹几乎恒可行，把结构性不可行掩盖为暂时不可行，丢失
        KVPhysicalInfeasibleError 的逐 rank 缺口诊断。
        返回 (structural_infeasible, per_instance_detail)。"""
        final_context = (
            session_view.history_tokens + runtime.joint_input_tokens)
        detail = []
        any_feasible = False
        remote_allowed = (
            self.joint_config.remote_enabled
            and session_view.location == "local_hbm")
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
        乘积下失败日志体积逼近成功日志）。"""
        state = self._admission_failure_state.get(runtime.request_id)
        if state is not None and state["class"] == failure_class:
            state["count"] += 1
            self.log_decision(
                {"kind": "joint_admission_wait",
                 "request_id": runtime.request_id, "priority": 0},
                now_ns,
                decision={
                    "failure_class": failure_class,
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
        self.graph.emit_admission_batch(runtime.plan_dict())
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
            capacity=getattr(
                self, "_task_load_cache_capacity", _TASK_LOAD_CACHE_CAPACITY),
        )

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
            capacity=getattr(
                self, "_task_load_cache_capacity", _TASK_LOAD_CACHE_CAPACITY),
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

    def verify_run_end(self) -> None:
        """基类协议校验之上，叠加离线 :4064-4076 的收尾断言 + §7.3 结束
        审计（arrival heap / ready frontier / 列车账本全空）。"""
        super().verify_run_end()
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
            # R11：merge 尾 watch 必须在 EOF 前交付（标记是真实图节点，
            # C++ 处理后必有交付）；残留即 watch 通道破损。
            raise RuntimeError(
                "run ended with undelivered merge-done watches: {}".format(
                    sorted(self._pending_merge_alarms)[:5]))
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
