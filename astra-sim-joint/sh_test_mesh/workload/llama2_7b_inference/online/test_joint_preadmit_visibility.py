#!/usr/bin/env python3
"""test_joint_preadmit_visibility.py -- C8（WP2-preadmit，2026-09-22）零后端单测。

覆盖（卡 C8 测试清单，设计文档 §4.1 遥测覆盖；原 #readplan 承诺
账本/对账核销/泄漏审计三组用例已随 2026-10-14 远读退役删除——
readplan 注册表、_preregister_readplan_flows/_reconcile_readplan_at_
drain/_consume_readplan_units/_settle_readplan_residual/_assert_no_
readplan_leaks 整族退役）：
  1. 合成遥测字典 → _ingest_link_telemetry：速率数值
     {link_id: served_bytes/active_ns}、窗口链完备/重叠回退/键缺席
     三分支、collective_coverage 翻转条件；
  2. 合成遥测字典 → _joint_cost_model 构造（step 3 零后端覆盖）：
     冻结接口 kwarg link_telemetry_rates 字典逐位落 model。

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
from joint.capacity_summary import CapacitySummary  # noqa: E402
from joint.joint_cost_model import (  # noqa: E402
    JointCostModel,
    JointHardwareRates,
    LinkFlowRegistry,
    ServiceFactors,
)
from online.sh30_online_scheduler import (  # noqa: E402
    _OnlineInstanceState,
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


def _scheduler(*, train_max_iter=8):
    """__new__ 范式：只装配被测路径（遥测解析/构造）用到的属性。"""
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
        layer_policy="adaptive")
    scheduler.kv_manager = SimpleNamespace(
        tp_degree=2,
        # D04：JCM 构造注入 nearest_edge_fn——替身把池端口钉在每 rank
        # 本地（pool 腿退化口径：无 NoC 段；本文件用例不依赖池腿）。
        nearest_edge=lambda rank: rank,
        _sessions={
            "s1": SimpleNamespace(
                working_kind=None, context_tokens=100,
                base_history_tokens=100,
                base_resident_prefix_layers=4),
        },
        _effective_remaining_by_tp_rank=lambda index: (10**9, 10**9),
        _instance_reclaimable_capacity_by_tp_rank=lambda index: (0, 0),
        # D11：容量驱逐摘要/容量账替身（零桶 = 快照时刻无合法 victim；
        # 逐 rank 台账 (C, W, O, Z) 与 remaining=1e9 同源核对守恒）。
        capacity_summary_snapshot=lambda: CapacitySummary(instances={}),
        capacity_ledger_by_tp_rank=lambda index: (
            (2 * 10**9, 0, 0, 10**9), (2 * 10**9, 0, 0, 10**9)),
    )
    # log_decision 基类契约（__new__ 替身手工置初值）。
    scheduler.online_log_count = 0
    scheduler.decision_log_sink = None
    scheduler.online_log_rows = []
    # C8 遥测状态（__init__ 同款初值）。
    scheduler._link_telemetry_rates = {}
    # M1：C8 遥测状态 __init__ 同款（时间加权活跃流数）。
    scheduler._link_telemetry_flow_counts = {}
    # ACC（验收纪律第 3 类，v2.0 §19.3）：预测器开销账本 __init__ 同款
    # 初值（F6 销账口径：决策/摘要记账直访——替身漏设 = AttributeError）。
    from joint.predictor_cost_accounting import PredictorCostAccounting
    scheduler.predictor_cost = PredictorCostAccounting()
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


def _copy_transfer(request_id="r1"):
    """跨实例 copy 交接流（noc_migrate 双 shard：home 1 → exec 0，
    与退役前 readplan 单元传输同路的通用腿型——用于流注册表可见性
    断言）。"""
    from face_scheduler import KVTransfer, KVTransferShard
    return KVTransfer(
        kind="noc_migrate", phase="prefill",
        reason="copy_handoff", session_id="s1",
        trigger_request_id=request_id, source_instance_index=1,
        target_instance_index=0, total_bytes=2 * 512 * 4,
        shards=tuple(
            KVTransferShard(
                source_rank=source_rank, target_rank=target_rank,
                edge_rank=None, bytes=512 * 4, noc_path=noc_path,
                layer_start=0, layer_end=4)
            for source_rank, target_rank, noc_path in (
                (2, 0, (2, 1, 0)), (3, 1, (3, 2, 1)))),
        model_layers=4, layer_start=0, layer_end=4,
        resident_prefix_layers_before=4,
        resident_prefix_layers_after=4)


def _decision_rows(scheduler, kind):
    return [row for row in scheduler.online_log_rows
            if row["kind"] == kind]


# ============================================ 1. 遥测解析与构造（step 3）==


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
        scheduler = _scheduler()
        scheduler._ingest_link_telemetry({
            "tick": 100, "link_telemetry": [
                _sample(5, 1000, 500, 0, 100)]})
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
        model = scheduler._joint_cost_model(0)
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
        # 追加 +link_telemetry 段（流注册面用跨实例 copy 交接流承载）。
        scheduler = self._full_scheduler()
        self.assertEqual(
            scheduler._contention_coverage_value(), "cold_start")
        scheduler._ingest_link_telemetry({
            "tick": 100, "link_telemetry": [
                _sample(5, 1000, 500, 0, 100)]})
        scheduler._register_transfer_flows(
            (_copy_transfer(),), owner="r1#copy")
        self.assertEqual(scheduler._contention_coverage_value(),
                         "link_flows+pool_ports+link_telemetry")
        scheduler._ingest_link_telemetry({"tick": 200})
        self.assertEqual(scheduler._contention_coverage_value(),
                         "link_flows+pool_ports")


if __name__ == "__main__":
    unittest.main()
