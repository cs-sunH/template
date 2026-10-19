#!/usr/bin/env python3
"""test_c7a_summary_maintenance_accounting.py -- C7a/ACC 钉测：容量摘要
**事件侧**增量维护（``KVCacheManager._refresh_capacity_summary``）耗时
入同一统计通道。

背景（ACC 补全，v2.0 §19.3）：summary_maintenance 计时此前只含查询侧
（``capacity_summary_snapshot``，调度器 P/D 入口计时，0.8-0.9ms 量级）
——真实事件增量维护的耗时没入账，10% 预算口径读数失真。修正面：

* KVCacheManager：逐次维护墙时入分列账（维护次数/累计耗时）+
  可选 notifier 回调（缺省 None = 零行为差）；
* 调度器装配点：notifier 接 ``predictor_cost.note_summary_maintenance``
  （与查询侧同一统计通道）；
* run 末 ``predictor_cost_snapshot``：summary_maintenance 块并入
  ``event_maintenance``（count/total_ns 分列）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 -m pytest online/test_c7a_summary_maintenance_accounting.py -q
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
    KVCacheManager,
    build_instances,
)
from joint.predictor_cost_accounting import (  # noqa: E402
    PredictorCostAccounting,
)


def _tiny_hardware():
    return FaceHardware(
        mesh_rows=2,
        mesh_cols=2,
        local_hbm_capacity_bytes=1_000_000_000,
        local_hbm_bandwidth_gbps=100.0,
        d2d_bandwidth_gbps=200.0,
        peak_perf_tflops=1.0,
        d2d_latency_ns=0,
        local_hbm_latency_ns=0,
    )


def _tiny_model():
    return FaceModel(
        layers=4, hidden_size=16, ffn_size=32, num_heads=4,
        vocab_size=32, bytes_per_elem=2, mlp_variant="swiglu")


def _topology():
    return build_instances(
        _tiny_hardware(),
        (FaceInstanceSpec("ins0", "1", (0, 1)),
         FaceInstanceSpec("ins1", "2", (2, 3))),
    )


def _trigger_capacity_event(kv):
    """触发一次真实容量事件（prepare + expand → _refresh_capacity_
    summary 至少一次）。"""
    kv.prepare_prefill(
        session_id="s", target_instance_index=0,
        history_tokens=0, trigger_request_id="r0")
    return kv.expand_prefill(
        session_id="s", instance_index=0,
        context_tokens=64, trigger_request_id="r0")


class EventSideMaintenanceLedgerTest(unittest.TestCase):
    """KVCacheManager 事件侧维护账（分列计数/累计 + notifier 通道）。"""

    def test_capacity_event_records_maintenance_count_and_ns(self):
        kv = KVCacheManager(_topology(), _tiny_model())
        # 构造期 _initialize_incremental_invariants 即有一次真实维护
        # （全量建账也是维护事件）——账本只认事件数，不区分构造/运行。
        before = kv.capacity_summary_maintenance_stats()
        self.assertGreaterEqual(before["count"], 1)
        self.assertGreaterEqual(before["total_ns"], 0)
        _trigger_capacity_event(kv)
        after = kv.capacity_summary_maintenance_stats()
        # 钉测要求：触发一次容量事件后维护计数 ≥1 且有耗时记录。
        self.assertGreaterEqual(after["count"], before["count"] + 1)
        self.assertGreaterEqual(after["total_ns"], before["total_ns"])

    def test_notifier_default_off_and_same_channel_delivery(self):
        kv = KVCacheManager(_topology(), _tiny_model())
        # 缺省 None = 零行为差（独立可用，不强制接线）。
        self.assertIsNone(kv.capacity_summary_maintenance_notifier)
        accounting = PredictorCostAccounting()
        # 接线后（构造期维护已发生，不计入通道）：通道只收接线后的
        # 事件侧样本，账本分列数 = 接线前计数 + 通道计数（同一通道、
        # 无双计）。
        pre_wired = kv.capacity_summary_maintenance_stats()
        kv.capacity_summary_maintenance_notifier = (
            accounting.note_summary_maintenance)
        _trigger_capacity_event(kv)
        snapshot = accounting.snapshot(host_wall_ns=1)
        self.assertGreaterEqual(
            snapshot["summary_maintenance"]["count"], 1)
        stats = kv.capacity_summary_maintenance_stats()
        self.assertEqual(
            stats["count"],
            pre_wired["count"]
            + snapshot["summary_maintenance"]["count"])
        self.assertEqual(
            snapshot["summary_maintenance"]["total_ns"],
            stats["total_ns"] - pre_wired["total_ns"])


class RunEndSnapshotMergeTest(unittest.TestCase):
    """调度器 run 末快照并入事件侧分列（event_maintenance 键）。"""

    def _shell(self, kv_manager):
        from online.sh30_online_scheduler import Sh30OnlineScheduler
        scheduler = Sh30OnlineScheduler.__new__(Sh30OnlineScheduler)
        scheduler.predictor_cost = PredictorCostAccounting()
        scheduler.kv_manager = kv_manager
        scheduler._predictor_cost_host_anchor_ns = 0
        return scheduler

    def test_snapshot_carries_event_maintenance_breakdown(self):
        kv = KVCacheManager(_topology(), _tiny_model())
        accounting = PredictorCostAccounting()
        kv.capacity_summary_maintenance_notifier = (
            accounting.note_summary_maintenance)
        _trigger_capacity_event(kv)
        scheduler = self._shell(kv)
        # 替身 kv_manager 是真 manager：notifier 直连替身账本（装配点
        # 语义同生产 __init__）。
        scheduler.predictor_cost = accounting
        snapshot = scheduler.predictor_cost_snapshot(host_wall_ns=1)
        block = snapshot["summary_maintenance"]
        self.assertIn("event_maintenance", block)
        stats = kv.capacity_summary_maintenance_stats()
        self.assertEqual(block["event_maintenance"]["count"], stats["count"])
        self.assertEqual(
            block["event_maintenance"]["total_ns"], stats["total_ns"])
        self.assertGreaterEqual(block["event_maintenance"]["count"], 1)

    def test_snapshot_without_real_manager_omits_key(self):
        # __new__ 测试替身的 SimpleNamespace kv_manager（无该接口）：
        # 快照回落账本原生面，不添键（形态不炸）。
        from online.sh30_online_scheduler import Sh30OnlineScheduler
        scheduler = Sh30OnlineScheduler.__new__(Sh30OnlineScheduler)
        scheduler.predictor_cost = PredictorCostAccounting()
        scheduler.kv_manager = SimpleNamespace()
        scheduler._predictor_cost_host_anchor_ns = 0
        snapshot = scheduler.predictor_cost_snapshot(host_wall_ns=1)
        self.assertNotIn("event_maintenance",
                         snapshot["summary_maintenance"])


class ProductionWiringTest(unittest.TestCase):
    """装配点接线：真 __init__ 把 notifier 接进同一统计通道。"""

    def test_init_wires_notifier_into_accounting_channel(self):
        import inspect
        from online.sh30_online_scheduler import Sh30OnlineScheduler
        source = inspect.getsource(Sh30OnlineScheduler.__init__)
        self.assertIn("capacity_summary_maintenance_notifier", source)
        self.assertIn("note_summary_maintenance", source)


if __name__ == "__main__":
    unittest.main()
