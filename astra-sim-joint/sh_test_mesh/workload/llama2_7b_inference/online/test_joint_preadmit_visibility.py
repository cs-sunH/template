#!/usr/bin/env python3
"""test_joint_preadmit_visibility.py -- C8（WP2-preadmit，2026-09-22）零后端单测。

覆盖（卡 C8 测试清单，设计文档 §4.1 同 tick 承诺可见性 / 遥测覆盖）：
     1. 同 tick 两 remote-read 准入互见：第一笔 #readplan 预登记（准入事务
     成功路径的登记通道）后，第二笔决策的 divisor_multi 已含第一笔一条
     物理并发代表流（注册表级精确断言 + JCM estimate_action 计价抬升）；
     HBM 端点端口同通道成对登记（C2 腿型）；
  2. 对账核销全生命周期：drain 真计划冻结处核销估计、以真值重登记
     （est/actual 差值进决策日志 readplan_reconcile 行）；逐列车实际登记
     按块数核销（当前列车 rid#decode#{j} 接管、无同流量双计）；完成边界
     清残余（碎片化残差披露 readplan_settle 行）；收尾注册表全空；
  3. 失败注入（预登记泄漏）：drain 未核销 → 收尾审计报警（复用 C2 的
     HbmPortFlowRegistry.leaked_owners 审计通道）；
  4. 合成遥测字典 → _joint_cost_model 构造（step 3 零后端覆盖）：速率
     数值 {link_id: served_bytes/active_ns}、窗口链完备/重叠回退/键缺席
     三分支、collective_coverage 翻转条件、冻结接口 kwarg
     link_telemetry_rates（C7 落地前软门构造不炸；C7 落地后字典逐位
     落 model——本测试自动切到强断言分支，G2 联集成即生效）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 online/test_joint_preadmit_visibility.py   （或 pytest 同路径）
"""
import os
import sys
import unittest
from types import SimpleNamespace

_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _p in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from face_scheduler import (  # noqa: E402
    FaceHardware,
    FaceInstanceSpec,
    FaceModel,
    build_instances,
)
from joint.hbm_port_flow_registry import HbmPortFlowRegistry  # noqa: E402
from joint.joint_config import JointMechanismConfig  # noqa: E402
from joint.joint_cost_model import (  # noqa: E402
    ACTION_REMOTE,
    InstanceLoadView,
    JointCostModel,
    JointHardwareRates,
    LinkFlowRegistry,
    RequestView,
    ServiceFactors,
    SessionKVView,
)
from online.sh30_online_scheduler import (  # noqa: E402
    _OnlineInstanceState,
    _OnlineRequestRuntime,
    _PoolPortRegistry,
    _TASK_LOAD_CACHE_CAPACITY,  # F6 销账：替身对齐 __init__ 初值
    Sh30OnlineScheduler,
)


# ============================================================ 夹具构造 ==

def _hardware():
    return FaceHardware(
        mesh_rows=2,
        mesh_cols=4,
        local_hbm_capacity_bytes=1_000_000_000,
        local_hbm_bandwidth_gbps=100.0,
        d2d_bandwidth_gbps=200.0,
        peak_perf_tflops=1.0,
        d2d_latency_ns=0,
        local_hbm_latency_ns=0,
    )


def _model():
    return FaceModel(
        layers=4,
        hidden_size=64,
        ffn_size=128,
        num_heads=4,
        vocab_size=32,
        bytes_per_elem=2,
        mlp_variant="swiglu",
    )


def _topology():
    # 2×4 mesh 四实例（tp=2）：实例 0 = ranks (0,1)、实例 1 = ranks
    # (2,3)（XY 路由 1→0 的两条 shard 路径 (2,1,0)/(3,2,1) 共享链路
    # (2,1)）；实例 2/3 = (4,5)/(6,7) 仅为覆盖全部 NPU（build_instances
    # 要求恰一次全覆盖）。
    return build_instances(
        _hardware(),
        (FaceInstanceSpec("g0", "pg0", (0, 1)),
         FaceInstanceSpec("g1", "pg1", (2, 3)),
         FaceInstanceSpec("g2", "pg2", (4, 5)),
         FaceInstanceSpec("g3", "pg3", (6, 7))),
        require_equal_size=True)


def _make_runtime(request_id, *, session_id="s1"):
    return _OnlineRequestRuntime({
        "request_id": request_id,
        "session_id": session_id,
        "turn_index": 0,
        "queue_index": 0,
        "prefill_length": 50,
        "decode_length": 16,
        "history_tokens_before": 100,
        "prefill_context_tokens": 150,
        "final_context_tokens": 166,
    })


def _session_view(home=1, resident=1):
    return SessionKVView(
        session_id="s1", home_instance=home, resident_instance=resident,
        location="local_hbm", history_tokens=100,
        resident_prefix_layers=4,
        history_bytes_by_tp_rank=(400, 400),
        missing_bytes_by_tp_rank=(0, 0))


def _scheduler(*, train_max_iter=8):
    """__new__ 范式：只装配被测路径（预登记/对账核销/遥测）用到的属性。"""
    scheduler = Sh30OnlineScheduler.__new__(Sh30OnlineScheduler)
    scheduler._joint_flows = LinkFlowRegistry()
    scheduler._pool_ports = _PoolPortRegistry()
    scheduler._hbm_ports = HbmPortFlowRegistry()
    scheduler.hardware = _hardware()
    scheduler.model = _model()
    scheduler.topology = _topology()
    scheduler._train_max_iter = train_max_iter
    scheduler.joint_config = JointMechanismConfig(
        category_mode="typed", scheduler_mode="joint",
        layer_policy="adaptive", remote_actions="on")
    scheduler.kv_manager = SimpleNamespace(
        tp_degree=2,
        _sessions={
            "s1": SimpleNamespace(
                working_kind=None, context_tokens=100,
                base_history_tokens=100,
                base_resident_prefix_layers=4),
        },
        _effective_remaining_by_tp_rank=lambda index: (10**9, 10**9),
        _instance_reclaimable_capacity_by_tp_rank=lambda index: (0, 0),
    )
    # log_decision 基类契约（__new__ 替身手工置初值）。
    scheduler.online_log_count = 0
    scheduler.decision_log_sink = None
    scheduler.online_log_rows = []
    # C8 遥测状态（__init__ 同款初值）。
    scheduler._link_telemetry_rates = {}
    # M1：C8 遥测状态 __init__ 同款（时间加权活跃流数）。
    scheduler._link_telemetry_flow_counts = {}
    scheduler.p_chunk = 512  # N4：_joint_prefill_total_load_ns 同形切分（F6 替身补设）
    scheduler._link_telemetry_epoch_count = 0
    scheduler._link_telemetry_sample_count = 0
    scheduler._telemetry_absent_seen = False
    scheduler._telemetry_window_broken = False
    scheduler._telemetry_window_end_ns = 0
    scheduler._telemetry_last_tick_ns = 0
    # 对齐 __init__ 初值（F6 销账：hasattr 软门已删，_telemetry_
    # endpoint_link_key 直达属性——替身漏设 = AttributeError）。
    scheduler._telemetry_link_id_map = None
    # A8'/A10'(a)（F1，2026-09-22）：ingest 接线所需速率/因子对象与
    # 伪影丢弃计数（_ingest_link_telemetry 逐窗口喂
    # observe_transfer_from_link_window；缺省夹具 B_link = 200 B/ns）。
    scheduler._joint_rates = JointHardwareRates.from_gbps(
        noc_link_gbps=200.0, pool_port_gbps=5.0,
        local_hbm_gbps=100.0, d2d_latency_ns=0, pool_latency_ns=100)
    scheduler._joint_factors = ServiceFactors()
    scheduler._telemetry_zero_rate_dropped = 0
    # 对齐 __init__ 初值（F6 销账：类级软缺省已删——收尾审计/注销路径
    # 直达 _quota_tracker，替身漏设 = AttributeError）。
    scheduler._quota_tracker = None  # off 档（JointMechanismConfig 缺省）
    return scheduler


def _decision_rows(scheduler, kind):
    return [row for row in scheduler.online_log_rows
            if row["kind"] == kind]


# 第二笔决策的逐 shard 路径（home 实例 1 → exec 实例 0，与第一笔同路）。
_REQ2_PATHS = ((2, 1, 0), (3, 2, 1))


# ================================================ 1. 同 tick 互见（§4.1）==


class SameTickReadplanVisibilityTest(unittest.TestCase):
    """同 tick 串行贪婪第二笔决策已见第一笔 #readplan 承诺。"""

    def _preregister(self, scheduler, *, estimated_decode=8):
        runtime = _make_runtime("r1")
        scheduler._preregister_readplan_flows(
            runtime, _session_view(), 0, estimated_decode)
        return runtime

    def test_second_decision_divisor_multi_includes_first_readplan(self):
        # 卡 C8 判据：第二笔 divisor_multi 已含第一笔 #readplan 足迹。
        # E=80、T_max=8 时账本仍披露 10 trains × 8 blocks = 80 units；
        # 但列车逐次发射、每 rank block recv 顺序链，所以该请求每 shard
        # 路径只贡献一条同时在途流。候选的两 shard 在链路 (2,1) 汇合，
        # 精确除数 = 自身两流 + 该请求两流 = 4（旧错误值 162）。
        scheduler = _scheduler()
        quiet = scheduler._joint_flows.divisor_multi(
            _REQ2_PATHS, include_self=True)
        self.assertEqual(quiet, 2)
        runtime = self._preregister(scheduler, estimated_decode=80)
        units = runtime.remote_read_preplan["est_flow_units"]
        self.assertEqual(units, 80)  # E=80、T_max=8：10 列车 × 8 块
        contended = scheduler._joint_flows.divisor_multi(
            _REQ2_PATHS, include_self=True)
        self.assertEqual(contended, 4)
        self.assertEqual(scheduler._joint_flows.snapshot()["2->1"], 2)
        # C2 腿型：noc_migrate 双端点 HBM 端口同通道成对登记。
        self.assertEqual(scheduler._hbm_ports.divisor(0), 1)
        self.assertEqual(scheduler._hbm_ports.divisor(1), 1)
        self.assertIn("r1#readplan", scheduler._hbm_ports.leaked_owners())

    def test_remote_read_price_lifts_after_first_commitment(self):
        # 决策级证据：JCM estimate_action 的 remote-read 计价在第一笔
        # 承诺登记后抬升（contention_divisor 与 cost_ns 同向）。NoC 腿
        # 主导配置（慢链路 / 快端点）——除数直接进入流送段墙钟。
        scheduler = _scheduler()

        def build_model():
            return JointCostModel(
                rates=JointHardwareRates.from_gbps(
                    noc_link_gbps=10.0, pool_port_gbps=5.0,
                    local_hbm_gbps=1000.0, d2d_latency_ns=0,
                    pool_latency_ns=100),
                loads={0: InstanceLoadView(
                    instance_index=0, queued_task_load_ns=0,
                    running_task_load_ns=0, active_decode_task_load_ns=0,
                    hbm_remaining_bytes_by_tp_rank=(10**9, 10**9)),
                       1: InstanceLoadView(
                    instance_index=1, queued_task_load_ns=0,
                    running_task_load_ns=0, active_decode_task_load_ns=0,
                    hbm_remaining_bytes_by_tp_rank=(10**9, 10**9))},
                flow_registry=scheduler._joint_flows,
                service_factors=ServiceFactors(),
                prefill_ns_per_token=1.0, decode_ns_per_token=2.0,
                model_layers=4, instance_tp_size=2,
                route_fn=lambda s, t: ((2, 1, 0), 2),
                route_paths_fn=lambda s, t: _REQ2_PATHS)

        def remote_candidate():
            return build_model().estimate_action(
                session=_session_view(), request=RequestView(
                    request_id="r2", session_id="s2", input_tokens=50,
                    history_tokens_before=100, estimated_decode_tokens=3,
                    horizon_source="session_online_mean",
                    input_kv_bytes_by_tp_rank=(5000, 5000)),
                instance_index=0, action=ACTION_REMOTE, remote_enabled=True)

        quiet = remote_candidate()
        self._preregister(scheduler, estimated_decode=8)
        contended = remote_candidate()
        self.assertGreater(
            contended.breakdown.contention_divisor,
            quiet.breakdown.contention_divisor)
        self.assertGreater(contended.cost_ns, quiet.cost_ns)

    def test_coverage_units_closed_form(self):
        # 覆盖窗口闭式 = 列车数 × 每列车块数（auto K：块数 ≤ 8）。
        scheduler = _scheduler()
        self.assertEqual(scheduler._readplan_flow_units(1), (1, 1))
        self.assertEqual(scheduler._readplan_flow_units(8), (1, 8))
        self.assertEqual(scheduler._readplan_flow_units(32), (4, 8))
        self.assertEqual(scheduler._readplan_flow_units(100), (13, 8))
        # T_max = 0（不设限）：单列车闭式（K=ceil(100/8)=13 → 8 块）。
        unlimited = _scheduler(train_max_iter=0)
        self.assertEqual(unlimited._readplan_flow_units(100), (1, 8))

    def test_degenerate_cases_skip_preregistration(self):
        # 与 drain 退化分支同口径：home 缺失 / home==exec / read_prefix
        # <=0 不预登记（真计划亦 None，无承诺可登记）。
        scheduler = _scheduler()
        runtime = _make_runtime("r1")
        scheduler._preregister_readplan_flows(
            runtime,
            SessionKVView(
                session_id="s1", home_instance=None,
                resident_instance=None, location="none",
                history_tokens=0, resident_prefix_layers=0,
                history_bytes_by_tp_rank=(0, 0),
                missing_bytes_by_tp_rank=(0, 0)),
            0, 8)
        scheduler._preregister_readplan_flows(
            runtime, _session_view(home=0), 0, 8)  # home == exec
        scheduler.kv_manager._sessions["s1"].base_resident_prefix_layers = 0
        scheduler._preregister_readplan_flows(
            runtime,
            _session_view(home=1, resident=1), 1, 8)  # read_prefix 0
        self.assertIsNone(runtime.remote_read_preplan)
        self.assertEqual(scheduler._joint_flows.snapshot(), {})
        self.assertEqual(scheduler._hbm_ports.snapshot(), {})


# ==================================================== 2. 对账核销链 ==


class ReadplanLifecycleTest(unittest.TestCase):
    """drain 对账 → 逐列车核销 → 完成清残余的全生命周期。"""

    def _truth_plan(self, steps):
        return {
            "home_instance": 1,
            "exec_instance": 0,
            "steps": steps,
            "context_per_step": 100 + 50 + steps,
            "read_prefix_layers": 4,
            "total_bytes": 2 * 512 * steps,
            "shard_specs": (
                (2, 0, 512, (2, 1, 0)),
                (3, 1, 512, (3, 2, 1)),
            ),
        }

    def _admitted_remote_read(self, scheduler, *, estimated_decode=8):
        runtime = _make_runtime("r1")
        runtime.joint_action = "remote-read"
        scheduler._preregister_readplan_flows(
            runtime, _session_view(), 0, estimated_decode)
        return runtime

    def test_eighty_step_plan_tracks_units_but_only_one_live_stream(self):
        # E=80、T_max=8、auto K=1：10 列车 × 8 credit blocks = 80
        # units 必须完整保留。每次 train 在飞只由当前 #decode#j owner
        # 代表一条 TP 流；Tj 后请求未完成时恢复一条 #readplan，最后一列
        # 完成后两类账都清空。这同时覆盖同请求当前/未来不双计与 run 尾。
        scheduler = _scheduler(train_max_iter=8)
        runtime = self._admitted_remote_read(scheduler, estimated_decode=80)
        runtime.decode_length = 80
        runtime.remote_read_credit_plan = self._truth_plan(80)
        scheduler._reconcile_readplan_at_drain(runtime, 500)
        preplan = runtime.remote_read_preplan
        self.assertEqual(preplan["est_flow_units"], 80)
        self.assertEqual(preplan["truth_flow_units"], 80)

        for slice_index in range(1, 11):
            blocks = scheduler._joint_remote_read_slice(runtime, 8, 1)
            self.assertEqual(len(blocks), 8)
            scheduler._consume_readplan_units(runtime, len(blocks))
            owner = "r1#decode#{}".format(slice_index)
            scheduler._register_transfer_flows(
                blocks, owner=owner, serial_credit_stream=True)
            self.assertEqual(
                scheduler._joint_flows.snapshot()["2->1"], 2)
            self.assertEqual(scheduler._hbm_ports.divisor(0), 1)
            self.assertEqual(scheduler._hbm_ports.divisor(1), 1)
            self.assertFalse(preplan["registry_active"])
            self.assertNotIn("r1#readplan",
                             scheduler._hbm_ports.leaked_owners())
            self.assertIn(owner, scheduler._hbm_ports.leaked_owners())

            runtime.decode_tokens_consumed = slice_index * 8
            scheduler._finalize_remote_credit_train_flows(
                runtime, slice_index)
            if slice_index < 10:
                self.assertEqual(
                    scheduler._joint_flows.snapshot()["2->1"], 2)
                self.assertTrue(preplan["registry_active"])
                self.assertIn("r1#readplan",
                              scheduler._hbm_ports.leaked_owners())
            else:
                self.assertEqual(scheduler._joint_flows.snapshot(), {})
                self.assertFalse(preplan["registry_active"])
                self.assertEqual(scheduler._hbm_ports.leaked_owners(), {})

        self.assertEqual(preplan["consumed_units"], 80)
        self.assertEqual(preplan["remaining_units"], 0)
        scheduler._settle_readplan_residual(runtime, 900)
        scheduler._assert_no_readplan_leaks()
        self.assertEqual(scheduler._joint_flows.snapshot(), {})
        self.assertEqual(scheduler._hbm_ports.leaked_owners(), {})
        self.assertEqual(_decision_rows(scheduler, "readplan_settle"), [])

    def test_drain_reconciliation_rewrites_with_truth_and_logs(self):
        # est E=8 → 8 账本单位；真值 steps=16 → 16 单位。物理流注册表
        # 仍只放每 shard 一条代表流（共享链路 2 条），不随 units 放大；
        # est/actual 差值落 readplan_reconcile 行。
        scheduler = _scheduler()
        runtime = self._admitted_remote_read(scheduler, estimated_decode=8)
        runtime.remote_read_credit_plan = self._truth_plan(16)
        scheduler._reconcile_readplan_at_drain(runtime, 500)
        preplan = runtime.remote_read_preplan
        self.assertEqual(preplan["truth_flow_units"], 16)
        self.assertEqual(preplan["remaining_units"], 16)
        self.assertEqual(scheduler._joint_flows.snapshot()["2->1"], 2)
        self.assertEqual(scheduler._hbm_ports.divisor(0), 1)
        self.assertEqual(scheduler._hbm_ports.divisor(1), 1)
        rows = _decision_rows(scheduler, "readplan_reconcile")
        self.assertEqual(len(rows), 1)
        decision = rows[0]["decision"]
        self.assertEqual(decision["est_steps"], 8)
        self.assertEqual(decision["actual_steps"], 16)
        self.assertEqual(decision["est_flow_units"], 8)
        self.assertEqual(decision["truth_flow_units"], 16)
        self.assertEqual(
            decision["delta_bytes"],
            runtime.remote_read_credit_plan["total_bytes"]
            - preplan["est_total_bytes"])

    def test_per_train_consumption_then_exact_settlement(self):
        # 逐列车按块数核销：当前列车由 rid#decode#{j} 实际登记接管，
        # 在飞期间不叠加 future #readplan；Tj 释放当前 owner 后为下一列
        # 车恢复一条代表流。恰耗尽时完成边界无残差且无 owner 泄漏。
        scheduler = _scheduler()
        runtime = self._admitted_remote_read(scheduler, estimated_decode=8)
        runtime.remote_read_credit_plan = self._truth_plan(16)
        scheduler._reconcile_readplan_at_drain(runtime, 500)
        first_blocks = scheduler._joint_remote_read_slice(runtime, 8, 1)
        scheduler._consume_readplan_units(runtime, len(first_blocks))
        scheduler._register_transfer_flows(
            first_blocks, owner="r1#decode#1", serial_credit_stream=True)
        self.assertEqual(runtime.remote_read_preplan["remaining_units"], 8)
        self.assertEqual(scheduler._joint_flows.snapshot()["2->1"], 2)
        self.assertFalse(runtime.remote_read_preplan["registry_active"])
        self.assertNotIn("r1#readplan", scheduler._hbm_ports.leaked_owners())
        self.assertIn("r1#decode#1", scheduler._hbm_ports.leaked_owners())

        runtime.decode_tokens_consumed = 8
        scheduler._finalize_remote_credit_train_flows(runtime, 1)
        self.assertEqual(scheduler._joint_flows.snapshot()["2->1"], 2)
        self.assertTrue(runtime.remote_read_preplan["registry_active"])
        self.assertIn("r1#readplan", scheduler._hbm_ports.leaked_owners())

        second_blocks = scheduler._joint_remote_read_slice(runtime, 8, 1)
        scheduler._consume_readplan_units(runtime, len(second_blocks))
        scheduler._register_transfer_flows(
            second_blocks, owner="r1#decode#2", serial_credit_stream=True)
        self.assertEqual(runtime.remote_read_preplan["remaining_units"], 0)
        self.assertEqual(scheduler._joint_flows.snapshot()["2->1"], 2)
        self.assertFalse(runtime.remote_read_preplan["registry_active"])
        runtime.decode_tokens_consumed = 16
        scheduler._finalize_remote_credit_train_flows(runtime, 2)
        self.assertEqual(scheduler._joint_flows.snapshot(), {})
        scheduler._settle_readplan_residual(runtime, 900)
        self.assertIsNone(runtime.remote_read_preplan)
        self.assertEqual(_decision_rows(scheduler, "readplan_settle"), [])
        self.assertEqual(scheduler._hbm_ports.leaked_owners(), {})

    def test_fragmentation_settlement_discloses_variance(self):
        # 列车碎片化：真值闭式 16 单位、实际列车块数和 10（5+5）——完成
        # 边界清残余 6 并披露 variance_units。
        scheduler = _scheduler()
        runtime = self._admitted_remote_read(scheduler, estimated_decode=8)
        runtime.remote_read_credit_plan = self._truth_plan(16)
        scheduler._reconcile_readplan_at_drain(runtime, 500)
        for slice_index in (1, 2):
            blocks = scheduler._joint_remote_read_slice(runtime, 5, 1)
            scheduler._consume_readplan_units(runtime, len(blocks))
            scheduler._register_transfer_flows(
                blocks, owner="r1#decode#{}".format(slice_index),
                serial_credit_stream=True)
            runtime.decode_tokens_consumed = slice_index * 5
            if slice_index == 1:
                scheduler._finalize_remote_credit_train_flows(
                    runtime, slice_index)
            else:
                # This fixture deliberately models fewer registered blocks
                # than the closed-form truth; mark request completion so the
                # Tj hook does not restore a future stream.
                runtime.decode_tokens_consumed = runtime.decode_length
                scheduler._finalize_remote_credit_train_flows(
                    runtime, slice_index)
        self.assertEqual(
            runtime.remote_read_preplan["remaining_units"], 6)
        scheduler._settle_readplan_residual(runtime, 900)
        self.assertEqual(scheduler._joint_flows.snapshot(), {})
        rows = _decision_rows(scheduler, "readplan_settle")
        self.assertEqual(len(rows), 1)
        decision = rows[0]["decision"]
        self.assertEqual(decision["residual_units_released"], 6)
        self.assertEqual(decision["truth_flow_units"], 16)
        self.assertEqual(decision["consumed_units"], 10)
        self.assertEqual(decision["variance_units"], 6)

    def test_settle_without_drain_releases_estimate_basis(self):
        # 防御路径：drain 未发生（无真值）——完成边界仍清残余，披露以
        # 估计单位为基（truth_flow_units 回退 est_flow_units）。
        scheduler = _scheduler()
        runtime = self._admitted_remote_read(scheduler, estimated_decode=8)
        scheduler._settle_readplan_residual(runtime, 900)
        self.assertIsNone(runtime.remote_read_preplan)
        self.assertEqual(scheduler._joint_flows.snapshot(), {})
        rows = _decision_rows(scheduler, "readplan_settle")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["decision"]["truth_flow_units"], 8)
        self.assertEqual(rows[0]["decision"]["residual_units_released"], 8)

    def test_degenerate_truth_plan_writes_off_estimate(self):
        # drain 侧计划退化（None）：估计核销、余量清零、差值披露。
        scheduler = _scheduler()
        runtime = self._admitted_remote_read(scheduler, estimated_decode=8)
        runtime.remote_read_credit_plan = None
        scheduler._reconcile_readplan_at_drain(runtime, 500)
        preplan = runtime.remote_read_preplan
        self.assertEqual(preplan["truth_flow_units"], 0)
        self.assertEqual(preplan["remaining_units"], 0)
        self.assertEqual(scheduler._joint_flows.snapshot(), {})
        decision = _decision_rows(
            scheduler, "readplan_reconcile")[0]["decision"]
        self.assertEqual(decision["truth_flow_units"], 0)


# ==================================================== 3. 泄漏审计 ==


class ReadplanLeakAuditTest(unittest.TestCase):
    """失败注入：预登记泄漏（drain 未核销）→ 收尾审计报警。"""

    def test_unreconciled_preregistration_raises_at_run_end(self):
        scheduler = _scheduler()
        runtime = _make_runtime("r1")
        scheduler._preregister_readplan_flows(
            runtime, _session_view(), 0, 8)
        self.assertTrue(scheduler._joint_flows.has_registrations)
        with self.assertRaises(RuntimeError) as caught:
            scheduler._assert_no_readplan_leaks()
        self.assertIn("readplan", str(caught.exception))
        self.assertIn("r1#readplan", str(caught.exception))

    def test_clean_lifecycle_passes_audit(self):
        scheduler = _scheduler()
        runtime = _make_runtime("r1")
        scheduler._preregister_readplan_flows(
            runtime, _session_view(), 0, 8)
        runtime.remote_read_credit_plan = {
            "home_instance": 1, "exec_instance": 0, "steps": 8,
            "context_per_step": 158, "read_prefix_layers": 4,
            "total_bytes": 2 * 512 * 8,
            "shard_specs": ((2, 0, 512, (2, 1, 0)),
                            (3, 1, 512, (3, 2, 1)))}
        scheduler._reconcile_readplan_at_drain(runtime, 500)
        scheduler._consume_readplan_units(runtime, 8)
        scheduler._settle_readplan_residual(runtime, 900)
        scheduler._assert_no_readplan_leaks()  # 不抛 = 通过
        self.assertEqual(scheduler._hbm_ports.leaked_owners(), {})


# ============================================ 4. 遥测解析与构造（step 3）==


def _sample(link_id, served, active, start, end):
    return {"link_id": link_id, "served_bytes": served,
            "active_ns": active, "window_start_ns": start,
            "window_end_ns": end}


class LinkTelemetryIngestTest(unittest.TestCase):
    """link_telemetry[] 解析：速率、窗口链、collective_coverage 翻转。"""

    def test_rates_window_and_coverage_flip(self):
        scheduler = _scheduler()
        # F6 销账：hasattr 透传软门已删——本测试主题是窗链/覆盖语义，
        # 整型键形态改由替身显式预置恒等映射（端点换算语义在
        # quota_integration/fix1_pricing 的 endpoint 键测试单独钉死）。
        scheduler._telemetry_link_id_map = {5: 5, 7: 7}
        scheduler._ingest_link_telemetry({
            "tick": 100, "link_telemetry": [
                _sample(5, 1000, 500, 0, 100),
                _sample(7, 500, 500, 0, 100)]})
        self.assertEqual(
            scheduler._link_telemetry_rates, {5: 2.0, 7: 1.0})
        self.assertTrue(scheduler._joint_flows.collective_coverage)
        # 相邻窗链（start == 上一窗 end）→ 覆盖保持；速率刷新为最新窗。
        scheduler._ingest_link_telemetry({
            "tick": 250, "link_telemetry": [
                _sample(5, 2000, 500, 100, 250)]})
        self.assertEqual(scheduler._link_telemetry_rates[5], 4.0)
        self.assertTrue(scheduler._joint_flows.collective_coverage)

    def test_absent_key_sticks_incomplete(self):
        # 旗标关（键缺席）→ 本 run 遥测不完备（sticky，后续好窗不翻）。
        scheduler = _scheduler()
        scheduler._ingest_link_telemetry({
            "tick": 100, "link_telemetry": [_sample(5, 1000, 500, 0, 100)]})
        self.assertTrue(scheduler._joint_flows.collective_coverage)
        scheduler._ingest_link_telemetry({"tick": 200})
        self.assertFalse(scheduler._joint_flows.collective_coverage)
        scheduler._ingest_link_telemetry({
            "tick": 300, "link_telemetry": [
                _sample(5, 1000, 500, 200, 300)]})
        self.assertFalse(scheduler._joint_flows.collective_coverage)

    def test_window_overlap_marks_broken(self):
        # 窗口链重叠回退（start < 上一窗 end = 双采样/重放类破损）。
        scheduler = _scheduler()
        scheduler._ingest_link_telemetry({
            "tick": 100, "link_telemetry": [
                _sample(5, 1000, 500, 0, 100)]})
        scheduler._ingest_link_telemetry({
            "tick": 150, "link_telemetry": [
                _sample(5, 1000, 500, 50, 150)]})  # start 50 < end 100
        self.assertFalse(scheduler._joint_flows.collective_coverage)

    def test_gap_between_sampled_windows_is_not_broken(self):
        # 相邻采样窗之间的间隙 = 不可见的全闲置 epoch（C++ prev_tick 差分
        # 结构性无空洞、闲置链路按 C6 契约省略）——不判破损。
        scheduler = _scheduler()
        scheduler._ingest_link_telemetry({
            "tick": 100, "link_telemetry": [
                _sample(5, 1000, 500, 0, 100)]})
        scheduler._ingest_link_telemetry({"tick": 200, "link_telemetry": []})
        scheduler._ingest_link_telemetry({
            "tick": 300, "link_telemetry": [
                _sample(5, 1000, 500, 200, 300)]})
        self.assertTrue(scheduler._joint_flows.collective_coverage)

    def test_zero_active_ns_sample_skips_rate_counts_evidence(self):
        scheduler = _scheduler()
        scheduler._ingest_link_telemetry({
            "tick": 100, "link_telemetry": [
                _sample(5, 0, 0, 0, 100)]})
        self.assertEqual(scheduler._link_telemetry_rates, {})
        self.assertEqual(scheduler._link_telemetry_sample_count, 1)
        self.assertTrue(scheduler._joint_flows.collective_coverage)

    def test_malformed_telemetry_fails_closed(self):
        scheduler = _scheduler()
        with self.assertRaises(ValueError):
            scheduler._ingest_link_telemetry({
                "tick": 100, "link_telemetry": {"5": 1.0}})
        with self.assertRaises(ValueError):
            scheduler._ingest_link_telemetry({
                "tick": 100, "link_telemetry": [
                    {"link_id": 5, "served_bytes": -1, "active_ns": 10,
                     "window_start_ns": 0, "window_end_ns": 100}]})
        with self.assertRaises(ValueError):
            scheduler._ingest_link_telemetry({
                "tick": 100, "link_telemetry": [{"link_id": 5}]})


class CostModelTelemetryConstructionTest(unittest.TestCase):
    """合成遥测字典 → _joint_cost_model 构造（冻结接口断言）。"""

    def _full_scheduler(self):
        scheduler = _scheduler()
        scheduler.instances = [_OnlineInstanceState(index=0),
                               _OnlineInstanceState(index=1)]
        scheduler._task_load_snapshot = lambda state, now_ns: SimpleNamespace(
            queued_prefill_task_load_ns=0,
            running_prefill_task_load_ns=0,
            active_decode_task_load_ns=0)
        scheduler._joint_rates = JointHardwareRates.from_gbps(
            noc_link_gbps=200.0, pool_port_gbps=5.0,
            local_hbm_gbps=100.0, d2d_latency_ns=0, pool_latency_ns=100)
        scheduler._joint_factors = ServiceFactors()
        scheduler._instance_edge_ports = {}
        scheduler._decode_task_load_cache = {}
        # 对齐 __init__ 初值（F6 销账：软门已删，_joint_cost_model 的
        # memo 包装直达读——替身漏设 = AttributeError）。
        scheduler._task_load_cache_capacity = _TASK_LOAD_CACHE_CAPACITY
        return scheduler

    def test_frozen_interface_dict_reaches_cost_model(self):
        # 冻结契约：{link_id: 实测有效速率} 字典传入 _joint_cost_model
        # 构造。C7（JCM 半）已交付——字段在场、字典逐位可见（F4 起硬
        # 断言，软门分支已删）。F6 销账：整型键形态改由替身显式预置
        # 恒等映射（hasattr 透传软门已删，语义自担）。
        scheduler = self._full_scheduler()
        scheduler._telemetry_link_id_map = {5: 5, 9: 9}
        scheduler._ingest_link_telemetry({
            "tick": 100, "link_telemetry": [
                _sample(5, 1000, 500, 0, 100),
                _sample(9, 250, 500, 0, 100)]})
        model = scheduler._joint_cost_model(
            0, decode_context_tokens=150, decode_average_length=5)
        self.assertIs(model.flow_registry, scheduler._joint_flows)
        # F4（复审修复）：原软门自切换（字段在场则强断言、否则
        # assertNotIn 自证缺席）恒绿零效——C7 已交付 link_telemetry_rates
        # 字段（joint_cost_model.py dataclass），改硬断言：字段必须在
        # 场且字典逐位可见。
        self.assertIn(
            "link_telemetry_rates", JointCostModel.__dataclass_fields__)
        self.assertEqual(
            model.link_telemetry_rates, {5: 2.0, 9: 0.5})

    def test_contention_coverage_value_extension(self):
        # E13 唯一读者的值域：无遥测两态与 C8 前逐字节同；遥测完备时
        # 追加 +link_telemetry 段。
        scheduler = self._full_scheduler()
        self.assertEqual(
            scheduler._contention_coverage_value(), "cold_start")
        scheduler._ingest_link_telemetry({
            "tick": 100, "link_telemetry": [
                _sample(5, 1000, 500, 0, 100)]})
        scheduler._register_transfer_flows(
            scheduler._readplan_unit_transfers(
                ((2, 0, 512, (2, 1, 0)), (3, 1, 512, (3, 2, 1))),
                units=1, session_id="s1", request_id="r1",
                home_instance=1, exec_instance=0, read_prefix=4,
                steps_per_unit=1),
            owner="r1#readplan")
        self.assertEqual(scheduler._contention_coverage_value(),
                         "link_flows+pool_ports+link_telemetry")
        scheduler._ingest_link_telemetry({"tick": 200})
        self.assertEqual(scheduler._contention_coverage_value(),
                         "link_flows+pool_ports")


if __name__ == "__main__":
    unittest.main()
