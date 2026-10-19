#!/usr/bin/env python3
"""test_forecast_future_isolation.py -- §8.3 预测器与隐藏真值隔离的端到端
调度器/引擎级验证（调度器喂入通道 → W6 统计 → W7 build_plan 全链）。

《joint仓库当下+历史修改方案》§8.3 末句钉子："后端用真实 interval 安排
future alarm 属执行真值，预测器只用已揭示类型与因果等待估计；通过改动
隐藏 interval/长度而保持已揭示类型不变的测试验证隔离"。

既有覆盖为何不满足本条（缺口登记，2026-09-30 修复批补齐）：

  * W6 接口层结构钉（``joint/test_history_estimator.py``
    ConditionalRemainderTest.test_hidden_truth_parameters_structurally_
    rejected）：只钉"隐藏真值形参在估计器接口无处可传"，非行为面扰动；
  * W1 快照因果性（``joint/test_runtime_snapshot.py``
    TestCausalityV1.test_hidden_suffix_change_leaves_snapshot_bytes_
    identical）：改隐藏 decode_length/interval_ns 断言输出不变，但被测
    对象是 DecisionSnapshot（方案 §4.1），非 §8.3 预测器；
  * E 层因果性（``joint/test_joint_mechanisms.py`` /
    ``online/test_joint_layer_restore.py`` 的 test_future_info_isolation）：
    KVCacheManager._estimate_next_input 层——后者为 no-write-channel
    结构钉（本地 future_rows 从不喂入系统），均未触及
    BackgroundForecastEngine/ArrivalStatistics 预测路径。

本文件的端到端设计（§8.3 字面）：同一 session 三请求（r0→r1→r2），两条
调度臂在**相同**的可观测事件时刻线上各跑一遍：

  * 臂 A = baseline manifest；臂 B = 只改写**后继请求**（r1/r2）的
    manifest 隐藏真值——``prefill_length``/``decode_length``（runtime
    记录）与 ``inter_request_interval_ns``（RequestSpec，r1 的到达
    alarm 由它排定）。
  * 已揭示通道双臂逐位不变：queue[0].next_trigger_type = "human"（r0
    完成时经 ``mark_complete`` 落账 ``session.next_request_type``——
    sh30 ``_joint_revealed_category`` 的唯一来源）⇒ r1 到达的
    revealed_type 恒 "human"；r1 的到达**交付 tick** 由测试直接固定同
    刻（真实 interval 只驱动 future_alarms envelope 的
    ``arrival_world_ns``——执行侧排程值，C++ 交付前预测器不可见）。

    隔离命题 = "隐藏字段不进预测通道"，不是"不同到达时刻线给出相同
    预测"（后者不成立也不应断言——交付时刻本身是已揭示可观测面）。
  * 断言三面逐字段不变：
    1. ``ArrivalStatistics.snapshot()``（λ/计数/开放桶/版本）全等；
    2. ``ConditionalRemainingLengthEstimator.estimate_conditional`` 在
       (d × category) 网格上 frozen-dataclass 全等（含 disclosure）；
    3. ``BackgroundForecastEngine.build_plan``（recent 与 reverting 两
       档）注入集全等——ForecastPlan frozen-dataclass 递归相等（record/
       work 腿/barriers/resources/offset/placeholders/scenario_key/
       accounting），并显式逐字段复核 record.as_dict 与腿元组。
  * 扰动有效性反证（等式有牙齿）：双臂的 future_alarms envelope 三字
    段（prefill_length/decode_length/inter_request_interval_ns）与
    alarm 时刻**确已相异**——隐藏真值真实流经执行通道，而预测侧零响应。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 online/test_forecast_future_isolation.py （或 pytest）
"""
import os
import sys
import unittest
from collections import deque
from types import SimpleNamespace

_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _path in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from joint.background_forecast import (  # noqa: E402
    BackgroundForecastEngine,
    ForecastFacts,
)
from joint.history_estimator import (  # noqa: E402
    ArrivalStatistics,
    ConditionalRemainingLengthEstimator,
)
from joint.prediction_config import (  # noqa: E402
    FUTURE_DEMAND_RECENT,
    FUTURE_DEMAND_REVERTING,
    HISTORY_USE_VALIDATED,
    JointPredictionConfig,
    PredictionParams,
)
from online.sh30_online_scheduler import (  # noqa: E402
    Sh30OnlineScheduler,
    _OnlineRequestRuntime,
)

SESSION = "session_iso_0"
R0 = f"{SESSION}_request_0"   # turn 0（本轮完成者——已揭示面）
R1 = f"{SESSION}_request_1"   # turn 1（r0 的后继——隐藏真值承载者）
R2 = f"{SESSION}_request_2"   # turn 2（后继的后继——同样改写）

# 受控时刻线（fixture 尺度；预测参数 Δ=1000ns 见 _pred_params）：
SERVICE_DONE_NS = 2_000       # r0 响应完成（service_done，到达锚基点）
THINKTIME_NS = 1_000          # 臂 A：r1 的 inter_request_interval_ns
THINKTIME_B_NS = 7_000        # 臂 B：同字段改写值（×7）
THINKTIME_C_NS = 11_000       # 臂 B：r2 后继 interval 改写值（r1 完成时才读）
ARRIVAL_NS = SERVICE_DONE_NS + THINKTIME_NS   # 双臂共同的 r1 交付 tick
NOW_NS = 4_000                # build_plan 决策时刻（r1 到达桶已闭合）


def _pred_params() -> PredictionParams:
    """预测参数 fixture（test_future_demand_forecast._params 同族尺度：
    桶/半衰期/时域/距离桶/成熟门槛/预算——保证 build_plan 产出非空注入
    且全确定性）。"""
    return PredictionParams(
        arrival_bucket_delta_ns=1000,
        arrival_half_life_fast_ns=2000,
        arrival_half_life_slow_ns=200_000,
        revert_tau_ns=4000,
        forecast_horizon_ns=10_000,
        forecast_distance_buckets_ns=(1000, 4000),
        history_min_mature_samples=2,
        history_sample_validity_ns=10 ** 9,
        history_drift_quarantine_ns=5000,
        forecast_template_budget=64,
        cold_start_forward_steps=4,
        forecast_return_restore_tokens=2,
    )


def _facts() -> ForecastFacts:
    """模板事实 fixture（test_future_demand_forecast._facts 同形）。"""
    return ForecastFacts(
        model_layers=2,
        instance_tp_size=2,
        kv_token_bytes_per_rank=100,
        prefill_compute_ns=500.0,
        prefill_hbm_bytes_per_rank=500_000,
        decode_step_compute_ns=10.0,
        decode_step_hbm_bytes_per_rank=10_000,
        local_hbm_bytes_per_ns=1000.0,
        noc_link_bytes_per_ns=100.0,
        pool_port_bytes_per_ns=50.0,
        instance_ranks={0: (0, 1), 1: (2, 3)},
        pool_edge_by_rank={0: 4, 1: 4, 2: 5, 3: 5},
        pool_latency_ns=5,
    )


def _record(request_id, turn_index, queue_index, *, prefill_length=512,
            decode_length=8):
    """manifest 请求记录（隐藏真值字段 = prefill_length/decode_length）。"""
    return {
        "request_id": request_id,
        "session_id": SESSION,
        "turn_index": turn_index,
        "queue_index": queue_index,
        "prefill_length": prefill_length,
        "decode_length": decode_length,
        "history_tokens_before": 0 if turn_index == 0 else 512,
        "prefill_context_tokens": 512,
        "final_context_tokens": 520,
    }


class _RecordingFlows:
    """R15 在途流登记表的记录替身（test_joint_arrival_contract 同款）。"""

    def __init__(self):
        self.released = []

    def release_owner(self, owner):
        self.released.append(owner)


class _KvManagerIso:
    """最小 KV 账本替身：``_sessions`` 携带会话对象，``mark_complete``
    落账 ``next_request_type``（已揭示类型通道的真实落点；隐藏的
    interval/长度字段不在本账本任何入口上）。"""

    def __init__(self):
        self._sessions = {
            SESSION: SimpleNamespace(working_kind=None,
                                     next_request_type=None),
        }
        self.calls = []
        self.last_merge_outcome = None

    def merge_back(self, **kwargs):
        self.calls.append(("merge_back", kwargs))
        return ()

    def kv_delta_find(self, trigger_request_id):
        return {"trigger_request_id": trigger_request_id}

    def observe_completed_input(self, *args, **kwargs):
        pass

    def mark_complete(self, session_id, completion_ns,
                      next_request_type=None):
        self.calls.append(("mark_complete", session_id, completion_ns,
                           next_request_type))
        self._sessions[session_id].next_request_type = next_request_type

    def session_snapshot(self, session_id):
        return SimpleNamespace(location="local_hbm", instance_index=0)

    def retire_terminal_session(self, *args, **kwargs):
        pass


class _GraphIso:
    """emit_completion_batch 记录替身（无 merge 流路径）。"""

    def __init__(self):
        self.synced = []
        self.completion_plans = []

    def sync_pending_history_after_evictions(self, transfers):
        self.synced.append(transfers)

    def emit_completion_batch(self, plan, *, block_end_stage="seg2"):
        self.completion_block_end_stages = getattr(
            self, "completion_block_end_stages", []) + [block_end_stage]
        self.completion_plans.append(plan)
        return {"merge_done_members": {}, "has_merge": False}


def _bare_scheduler_iso(mutate_future: bool):
    """裸调度器替身（Sh30OnlineScheduler.__new__ + 最小字段集，
    test_joint_arrival_contract._bare_scheduler 同法扩到三请求）。

    ``mutate_future``：只改写后继请求（r1/r2）的 manifest 隐藏真值
    （记录内 prefill_length/decode_length + RequestSpec 的
    inter_request_interval_ns）；已揭示通道（queue[0].next_trigger_
    type）与 r0 记录逐位不变。"""
    queue = [
        # 已揭示通道：r0 完成时揭示后继类型 "human"（mark_complete 落账
        # ——双臂同值，§8.3"保持已揭示类型不变"）。
        SimpleNamespace(inter_request_interval_ns=None,
                        next_trigger_type="human"),
        SimpleNamespace(inter_request_interval_ns=THINKTIME_NS,
                        next_trigger_type="human"),
        SimpleNamespace(inter_request_interval_ns=THINKTIME_NS,
                        next_trigger_type=None),
    ]
    records = [
        _record(R0, 0, 0),   # 本轮完成者：双臂逐位相同（其完成长度是
                             # §8.2 允许的真实结束数据，非隐藏未来）
        _record(R1, 1, 1),
        _record(R2, 2, 2),
    ]
    if mutate_future:
        records[1] = _record(R1, 1, 1, prefill_length=1024,
                             decode_length=64)
        records[2] = _record(R2, 2, 2, prefill_length=2048,
                             decode_length=128)
        queue[1].inter_request_interval_ns = THINKTIME_B_NS
        queue[2].inter_request_interval_ns = THINKTIME_C_NS

    scheduler = Sh30OnlineScheduler.__new__(Sh30OnlineScheduler)
    scheduler.config = SimpleNamespace(request_queue=queue)
    scheduler.completed_requests = 0
    scheduler.kv_manager = _KvManagerIso()
    scheduler.graph = _GraphIso()
    scheduler._joint_horizon = SimpleNamespace(
        observe_completed=lambda *args, **kwargs: None)
    scheduler._kv_ledger_epoch = 0
    scheduler._stalled_by_instance = {}
    scheduler._joint_flows = _RecordingFlows()
    scheduler._pool_ports = _RecordingFlows()
    scheduler._hbm_ports = _RecordingFlows()
    scheduler._quota_tracker = None
    scheduler._batch = {"future_alarms": [], "watches": []}
    scheduler._pending_merge_alarms = {}
    scheduler.online_log_count = 0
    scheduler.decision_log_sink = None
    scheduler.online_log_rows = []
    runtimes = [_OnlineRequestRuntime(record, 512) for record in records]
    for runtime in runtimes:
        runtime.prefill_instance_index = 0
        runtime.decode_instance_index = 0
    scheduler.runtimes = runtimes
    scheduler.runtime_by_request_id = {
        runtime.request_id: runtime for runtime in runtimes}
    scheduler._runtime_index = {
        runtime.request_id: index
        for index, runtime in enumerate(runtimes)}
    scheduler.next_request = {R0: runtimes[1], R1: runtimes[2], R2: None}
    scheduler.arrival_heap = []
    scheduler._sequence = 0
    scheduler._arrived_request_count = 0
    scheduler.pending_admissions = deque()
    scheduler._retry = False
    # 预测组件预置（fixture 尺度参数；调度器惰性兜底让位于预置实例——
    # 见 _joint_arrival_stats_or_default / _joint_length_history_or_
    # default 的 getattr 短路）。双臂同参构造 + 同历史喂入（本窗口前已
    # 完成请求的样本，§8.2 允许的真实结束数据）。
    params = _pred_params()
    scheduler._joint_arrival_stats = ArrivalStatistics(params)
    estimator = ConditionalRemainingLengthEstimator(params)
    for value in (4, 6, 8, 12) * 4:
        estimator.observe_completed(
            session_id=SESSION, completed_forward_steps=value,
            category="human", tick_ns=0)
    scheduler._joint_length_history = estimator
    return scheduler


def _run_arm(mutate_future: bool):
    """跑一条臂的可观测事件时刻线（双臂同刻，只有隐藏字段不同）：

    r0 到达(turn 0) → r0 完成(service_done=2000，排 r1 到达 alarm) →
    r1 交付到达(3000，turn 1，revealed "human") → 统计推进到 NOW_NS。
    """
    scheduler = _bare_scheduler_iso(mutate_future)
    scheduler._push_arrival({"request_id": R0}, 0)
    scheduler._drain_arrival_heap(0)
    scheduler._complete_requests([R0], SERVICE_DONE_NS)
    scheduler._push_arrival({"request_id": R1}, ARRIVAL_NS)
    scheduler._drain_arrival_heap(ARRIVAL_NS)
    scheduler._joint_arrival_stats.advance_to(NOW_NS)
    return scheduler


def _validation_record(params, **overrides):
    from joint.joint_config import MODEL_SEMANTICS_VERSION
    record = {
        "samples": 64, "mean_abs_normalized_error": 0.2,
        "selection_loss_samples": 64, "mean_selection_loss_ns": 1,
        "baseline_mean_selection_loss_ns": 2,
        "validation_source": "controlled-validation-fixture",
        "validation_scope": "independent_prefix",
        "model_semantics_version": MODEL_SEMANTICS_VERSION,
        "params": params.params_manifest(),
    }
    record.update(overrides)
    return record


def _plan_for(scheduler, mode):
    """在该臂已喂入的统计/估计器上构造引擎并出一次预测计划（新引擎：
    decision_seq 同起点 ⇒ 双臂 work_id 可逐字段比较；确定性构造无
    RNG）。双臂同走显式研究臂形态（validated＋全档验证登记——§8.7
    shadow 档零注入不进物化路径；本测试的双臂等价性断言在注入面）。"""
    from joint.forecast_validation import (
        DISTANCE_BAND_FAR, DISTANCE_BAND_MID, DISTANCE_BAND_NEAR,
        DistanceAdmissionLedger)
    params = _pred_params()
    ledger = DistanceAdmissionLedger(params, validation_registry={
        (mode, band): _validation_record(params)
        for band in (DISTANCE_BAND_NEAR, DISTANCE_BAND_MID,
                     DISTANCE_BAND_FAR)})
    engine = BackgroundForecastEngine(
        arrival_stats=scheduler._joint_arrival_stats,
        prediction_config=JointPredictionConfig(
            future_demand_mode=mode, history_use=HISTORY_USE_VALIDATED,
            params=params),
        length_estimator=scheduler._joint_length_history,
        distance_ledger=ledger)
    return engine.build_plan(now_ns=NOW_NS, facts=_facts())


class FutureMutationEffectivenessTest(unittest.TestCase):
    """扰动有效性反证：隐藏真值确已流经执行通道（等式断言才有牙齿）。"""

    def test_future_fields_differ_on_execution_side(self):
        arm_a = _run_arm(False)
        arm_b = _run_arm(True)
        alarms_a = arm_a._batch["future_alarms"]
        alarms_b = arm_b._batch["future_alarms"]
        self.assertEqual(len(alarms_a), 1)
        self.assertEqual(len(alarms_b), 1)
        envelope_a = alarms_a[0]["envelope"]
        envelope_b = alarms_b[0]["envelope"]
        self.assertEqual(envelope_a["request_id"], R1)
        self.assertEqual(envelope_b["request_id"], R1)
        # 三隐藏字段在 future_alarms envelope 上逐字段相异。
        for key in ("prefill_length", "decode_length",
                    "inter_request_interval_ns"):
            self.assertNotEqual(envelope_a[key], envelope_b[key], msg=key)
        # 真实 interval 驱动的 alarm 时刻（执行侧排程值）相异。
        self.assertNotEqual(alarms_a[0]["arrival_world_ns"],
                            alarms_b[0]["arrival_world_ns"])
        # runtime 记录内的隐藏长度确已不同（manifest 改写真实生效）。
        for key in ("prefill_length", "decode_length"):
            self.assertNotEqual(
                getattr(arm_a.runtime_by_request_id[R1], key),
                getattr(arm_b.runtime_by_request_id[R1], key), msg=key)
            self.assertNotEqual(
                getattr(arm_a.runtime_by_request_id[R2], key),
                getattr(arm_b.runtime_by_request_id[R2], key), msg=key)

    def test_revealed_channel_held_fixed(self):
        """§8.3 前提钉：已揭示类型序列与已交付到达时刻线双臂逐位不变。"""
        arm_a = _run_arm(False)
        arm_b = _run_arm(True)
        for arm in (arm_a, arm_b):
            # mark_complete 落账的已揭示类型（revealed_type 唯一来源）。
            self.assertEqual(
                arm.kv_manager._sessions[SESSION].next_request_type,
                "human")
            # r1 到达冻结的已揭示类别键。
            self.assertEqual(
                arm.runtime_by_request_id[R1].joint_revealed_category,
                "human")
            # 交付 tick = 测试固定的同刻（真实 interval 只改 alarm 排程
            # 值，不进本时刻线）。
            self.assertEqual(
                arm.runtime_by_request_id[R1].estimated_arrival_ns,
                ARRIVAL_NS)
        # r0（当前请求）的记录双臂逐位相同——完成样本通道输入不变
        # （REQUEST_COMPLETE 边界后完成 runtime 自 map 退休弹出，比较
        # 取构造面：manifest 改写只落在后继记录上）。
        pre_a = _bare_scheduler_iso(False)
        pre_b = _bare_scheduler_iso(True)
        for key in ("prefill_length", "decode_length",
                    "history_tokens_before", "prefill_context_tokens",
                    "final_context_tokens"):
            self.assertEqual(
                getattr(pre_a.runtime_by_request_id[R0], key),
                getattr(pre_b.runtime_by_request_id[R0], key), msg=key)


class ForecastIsolationTest(unittest.TestCase):
    """§8.3 主断言：改隐藏 interval/长度、保持已揭示类型不变 ⇒
    预测器侧（统计/估计/计划）逐字段不变。"""

    def test_arrival_statistics_snapshot_identical(self):
        arm_a, arm_b = _run_arm(False), _run_arm(True)
        self.assertEqual(
            arm_a._joint_arrival_stats.snapshot(),
            arm_b._joint_arrival_stats.snapshot(),
            msg="ArrivalStatistics 全披露（λ/计数/开放桶/版本）逐字段相等")
        # 非平凡前提：session_return|human 类别确有到达入账（全等不是
        # 空对空）。
        snapshot = arm_a._joint_arrival_stats.snapshot()
        self.assertEqual(
            snapshot["categories"]["session_return|human"]
            ["total_arrivals"], 1)
        self.assertEqual(
            snapshot["categories"]["new_session_first|"]
            ["total_arrivals"], 1)

    def test_estimate_conditional_grid_identical(self):
        arm_a, arm_b = _run_arm(False), _run_arm(True)
        # d 网格覆盖回退链各级：d<12 命中 session 级、d=12 超观测最大
        # 逐级降级、d=50 正尾先验；category ∈ {None, "human"} 覆盖无/
        # 有已揭示类型键两路。
        for d in (0, 1, 3, 5, 8, 11, 12, 50):
            for category in (None, "human"):
                estimate_a = arm_a._joint_length_history.estimate_conditional(
                    session_id=SESSION, exposed_progress_steps=d,
                    category=category)
                estimate_b = arm_b._joint_length_history.estimate_conditional(
                    session_id=SESSION, exposed_progress_steps=d,
                    category=category)
                self.assertEqual(
                    estimate_a, estimate_b,
                    msg=f"d={d} category={category!r}（frozen dataclass "
                    "逐字段：值/来源/样本量/删失/disclosure）")
        # run 级披露账全等（隐藏长度未从任何侧门进样本）。
        self.assertEqual(
            arm_a._joint_length_history.as_dict(),
            arm_b._joint_length_history.as_dict())
        # 非平凡前提：网格命中真统计层（非常数先验）——估计走 session
        # 条件级而非 cold_start_prior。
        probe = arm_a._joint_length_history.estimate_conditional(
            session_id=SESSION, exposed_progress_steps=3, category="human")
        self.assertEqual(probe.source, "session_conditional")

    def test_build_plan_injections_identical_both_modes(self):
        arm_a, arm_b = _run_arm(False), _run_arm(True)
        for mode in (FUTURE_DEMAND_RECENT, FUTURE_DEMAND_REVERTING):
            plan_a = _plan_for(arm_a, mode)
            plan_b = _plan_for(arm_b, mode)
            # 注入集非空（等式有牙齿），且走正常注入路径（无回退噪音）。
            self.assertTrue(plan_a.injections, msg=mode)
            self.assertFalse(plan_a.fallbacks, msg=mode)
            # ForecastPlan frozen-dataclass 递归全等：record/work 腿/
            # barriers/resources/offset/scenario_key/accounting。
            self.assertEqual(plan_a, plan_b, msg=mode)
            # 显式逐字段复核（不黑盒依赖 dataclass __eq__）。
            self.assertEqual(
                [injection.record.as_dict()
                 for injection in plan_a.injections],
                [injection.record.as_dict()
                 for injection in plan_b.injections], msg=mode)
            self.assertEqual(
                [injection.arrival_offset_ns
                 for injection in plan_a.injections],
                [injection.arrival_offset_ns
                 for injection in plan_b.injections], msg=mode)
            self.assertEqual(
                tuple(injection.work.legs
                      for injection in plan_a.injections),
                tuple(injection.work.legs
                      for injection in plan_b.injections), msg=mode)
            self.assertEqual(
                tuple(injection.barriers
                      for injection in plan_a.injections),
                tuple(injection.barriers
                      for injection in plan_b.injections), msg=mode)
            self.assertEqual(
                tuple(injection.resources
                      for injection in plan_a.injections),
                tuple(injection.resources
                      for injection in plan_b.injections), msg=mode)
            self.assertEqual(plan_a.placeholders_ns,
                             plan_b.placeholders_ns, msg=mode)
            self.assertEqual(plan_a.scenario_key, plan_b.scenario_key,
                             msg=mode)
            self.assertEqual(plan_a.accounting, plan_b.accounting,
                             msg=mode)


if __name__ == "__main__":
    unittest.main(verbosity=2)
