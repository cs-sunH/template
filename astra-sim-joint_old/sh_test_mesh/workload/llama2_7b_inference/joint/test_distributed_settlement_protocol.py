#!/usr/bin/env python3
"""test_distributed_settlement_protocol.py -- W9 目录提交型结算协议（纯逻辑）。

G-DIST/V9 的目录侧判据（不依赖仿真/KVCacheManager）：

* 来源面枚举：位置数/区间数分别统计（>3 来源结构性支持）；
* 单轮增长预算断言（无拆分 ≤ +2，超限 fail-closed）；
* 单元格级唯一内容可定位审计（跨持有者部分重叠区间的真并集）；
* 会话终局清理审计（悬挂引用/未完 ack/在途 copy fail-closed）；
* 开销计量（DirectoryOverheadMeter 操作计数/逐会话逐轮时间线）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 joint/test_distributed_settlement_protocol.py
"""
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WL = os.path.dirname(_HERE)
for _p in (_HERE, _WL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from joint.distributed_settlement import (  # noqa: E402
    DEFAULT_ROUND_GROWTH_BUDGET,
    DirectoryOverheadMeter,
    DistributedSettleError,
    DistributedSettleOutcome,
    assert_full_content_locatable,
    assert_no_dangling_session_state,
    assert_round_growth_within_budget,
    cell_coverage_complete,
    enumerate_source_surface,
)
from joint.kv_location_directory import (  # noqa: E402
    ENTRY_STATE_EVICTED_TO_POOL,
    ENTRY_STATE_READY,
    ENTRY_STATE_WRITING,
    POOL_HOLDER,
    ROLE_BASE,
    ROLE_BOUNDARY_SOURCE,
    ROLE_POOL_BACKED,
    ROLE_RESIDENT,
    KvLocationDirectory,
)

_TP = 2


def _directory():
    return KvLocationDirectory(tp_degree=_TP, model_version="test-v1")


def _register(directory, *, holder, role, layer, tokens, bytes_per_rank,
              state=ENTRY_STATE_READY):
    start, end = tokens
    return directory.register_entry(
        session_id="s1", holder=holder,
        role=(ROLE_POOL_BACKED
              if holder == POOL_HOLDER and state == ENTRY_STATE_EVICTED_TO_POOL
              else role),
        layer_start=0, layer_end=layer,
        token_start=start, token_end=end,
        bytes_by_tp_rank=(bytes_per_rank,) * _TP,
        state=state,
    )


class SourceSurfaceEnumerationTest(unittest.TestCase):
    """§7.4：位置数与区间数分别统计；>3 来源无上限。"""

    def test_positions_and_intervals_counted_separately(self):
        directory = _directory()
        # 4 个片上来源位置、5 个区间（位置 0 持两段）+ 池段另列。
        _register(directory, holder=0, role=ROLE_BASE, layer=4,
                  tokens=(0, 10), bytes_per_rank=100)
        _register(directory, holder=1, role=ROLE_BOUNDARY_SOURCE, layer=4,
                  tokens=(10, 14), bytes_per_rank=40)
        _register(directory, holder=2, role=ROLE_RESIDENT, layer=4,
                  tokens=(10, 14), bytes_per_rank=40)
        _register(directory, holder=2, role=ROLE_RESIDENT, layer=4,
                  tokens=(14, 16), bytes_per_rank=20)
        _register(directory, holder=3, role=ROLE_RESIDENT, layer=2,
                  tokens=(0, 16), bytes_per_rank=80)
        _register(directory, holder=POOL_HOLDER, role=ROLE_RESIDENT,
                  layer=4, tokens=(0, 16), bytes_per_rank=160,
                  state=ENTRY_STATE_EVICTED_TO_POOL)
        surface = enumerate_source_surface(directory, "s1")
        self.assertEqual(surface.source_count, 4)      # 位置数 >3
        self.assertEqual(surface.interval_count, 5)    # 区间数（片上）
        self.assertEqual(len(surface.pooled), 1)       # 池段另列
        disclosure = surface.as_dict()
        self.assertEqual(disclosure["source_count"], 4)
        self.assertEqual(disclosure["interval_count"], 5)

    def test_exclude_holder_filters_target_position(self):
        directory = _directory()
        _register(directory, holder=0, role=ROLE_BASE, layer=4,
                  tokens=(0, 10), bytes_per_rank=100)
        _register(directory, holder=1, role=ROLE_RESIDENT, layer=4,
                  tokens=(10, 14), bytes_per_rank=40)
        surface = enumerate_source_surface(directory, "s1", exclude_holder=1)
        self.assertEqual(surface.source_count, 1)
        self.assertEqual(surface.interval_count, 1)


class RoundGrowthBudgetTest(unittest.TestCase):
    """§7.4：无拆分时单轮位置增量 ≤ +2（P/D 实例）；超限 fail-closed。"""

    def _surface(self, source_count):
        directory = _directory()
        for holder in range(source_count):
            _register(directory, holder=holder, role=ROLE_RESIDENT,
                      layer=4, tokens=(0, 10), bytes_per_rank=100)
        return enumerate_source_surface(directory, "s1")

    def test_growth_within_budget_passes(self):
        growth = assert_round_growth_within_budget(
            3, self._surface(5))
        self.assertEqual(growth, 2)
        self.assertEqual(DEFAULT_ROUND_GROWTH_BUDGET, 2)

    def test_growth_beyond_budget_fails_closed(self):
        with self.assertRaisesRegex(
                DistributedSettleError, "grew by 3"):
            assert_round_growth_within_budget(3, self._surface(6))

    def test_explicit_splits_allow_more(self):
        growth = assert_round_growth_within_budget(
            3, self._surface(6), explicit_splits=1)
        self.assertEqual(growth, 3)


class CellCoverageTest(unittest.TestCase):
    """G-DIST"全部有效区间可定位"：跨持有者部分重叠区间的真并集审计。"""

    def test_overlapping_replicas_cover_union_once(self):
        directory = _directory()
        # [0,12)@1 与 [0,10)@0 部分重叠——单元格并集 = [0,12) 全覆盖。
        _register(directory, holder=1, role=ROLE_RESIDENT, layer=4,
                  tokens=(0, 12), bytes_per_rank=120)
        _register(directory, holder=0, role=ROLE_BASE, layer=4,
                  tokens=(0, 10), bytes_per_rank=100)
        complete, area = cell_coverage_complete(
            directory, "s1", history_tokens=12, model_layers=4)
        self.assertTrue(complete)
        self.assertEqual(area, 12 * 4)
        self.assertEqual(
            assert_full_content_locatable(
                directory, "s1", history_tokens=12, model_layers=4),
            12 * 4)

    def test_missing_cell_fails_closed(self):
        directory = _directory()
        _register(directory, holder=0, role=ROLE_BASE, layer=4,
                  tokens=(0, 10), bytes_per_rank=100)
        _register(directory, holder=1, role=ROLE_RESIDENT, layer=4,
                  tokens=(10, 14), bytes_per_rank=40)
        # 期望 [0,16)：[14,16) 无覆盖。
        with self.assertRaisesRegex(
                DistributedSettleError, "not fully locatable"):
            assert_full_content_locatable(
                directory, "s1", history_tokens=16, model_layers=4)

    def test_pool_entry_counts_as_locatable(self):
        directory = _directory()
        _register(directory, holder=0, role=ROLE_BASE, layer=4,
                  tokens=(0, 10), bytes_per_rank=100)
        _register(directory, holder=POOL_HOLDER, role=ROLE_RESIDENT,
                  layer=4, tokens=(10, 16), bytes_per_rank=60,
                  state=ENTRY_STATE_EVICTED_TO_POOL)
        # 唯一内容在池等合法位置 ⇒ 仍可定位（§7.4 驱逐不清 owner）。
        assert_full_content_locatable(
            directory, "s1", history_tokens=16, model_layers=4)

    def test_holder_restriction(self):
        directory = _directory()
        _register(directory, holder=0, role=ROLE_BASE, layer=4,
                  tokens=(0, 10), bytes_per_rank=100)
        _register(directory, holder=1, role=ROLE_RESIDENT, layer=4,
                  tokens=(10, 14), bytes_per_rank=40)
        complete, area = cell_coverage_complete(
            directory, "s1", history_tokens=14, model_layers=4, holder=0)
        self.assertFalse(complete)
        self.assertEqual(area, 10 * 4)


class TerminalCleanupAuditTest(unittest.TestCase):
    """§7.4 会话终局：悬挂引用/未完 ack/在途 copy fail-closed。"""

    def test_clean_state_passes(self):
        directory = _directory()
        _register(directory, holder=0, role=ROLE_BASE, layer=4,
                  tokens=(0, 10), bytes_per_rank=100)
        assert_no_dangling_session_state(directory, "s1")

    def test_live_reference_fails_closed(self):
        directory = _directory()
        entry = _register(directory, holder=0, role=ROLE_BASE, layer=4,
                          tokens=(0, 10), bytes_per_rank=100)
        directory.acquire_entry_reference(entry.entry_id)
        with self.assertRaisesRegex(DistributedSettleError, "read references"):
            assert_no_dangling_session_state(directory, "s1")
        directory.release_entry_reference(entry.entry_id)

    def test_execution_pin_fails_closed(self):
        directory = _directory()
        entry = _register(directory, holder=0, role=ROLE_BASE, layer=4,
                          tokens=(0, 10), bytes_per_rank=100)
        directory.acquire_entry_reference(entry.entry_id, exec_pin=True)
        with self.assertRaisesRegex(DistributedSettleError, "execution pin"):
            assert_no_dangling_session_state(directory, "s1")
        directory.release_entry_reference(entry.entry_id, exec_pin=True)

    def test_pending_ack_fails_closed(self):
        directory = _directory()
        entry = _register(directory, holder=0, role=ROLE_BASE, layer=4,
                          tokens=(0, 10), bytes_per_rank=100,
                          state=ENTRY_STATE_WRITING)
        directory.add_ack_obligation(entry.entry_id, "copy-chunk-0")
        with self.assertRaisesRegex(DistributedSettleError, "pending ack"):
            assert_no_dangling_session_state(directory, "s1")

    def test_inflight_copy_fails_closed(self):
        directory = _directory()
        directory.begin_inflight_copy(
            session_id="s1", source_holder=0, target_instance=1,
            layer_start=0, layer_end=4, token_start=0, token_end=10,
            bytes_by_tp_rank=(100,) * _TP, ack_id="ack-1")
        with self.assertRaisesRegex(DistributedSettleError, "inflight"):
            assert_no_dangling_session_state(directory, "s1")


class OverheadMeterTest(unittest.TestCase):
    """§10：目录操作计数 + 逐会话逐轮时间线（计量不改行为）。"""

    def test_op_counts_and_round_timeline(self):
        meter = DirectoryOverheadMeter()
        meter.record_op("query_sources")
        meter.record_op("query_sources", elapsed_ns=5)
        meter.record_op("evict_to_pool", elapsed_ns=7)
        directory = _directory()
        _register(directory, holder=0, role=ROLE_BASE, layer=4,
                  tokens=(0, 10), bytes_per_rank=100)
        outcome = DistributedSettleOutcome(
            session_id="s1", trigger_request_id="r1",
            source_count=2, interval_count=3, retained_keys=((0, 0, 4, 0, 10),),
            directory_version=directory.version)
        meter.record_directory_commit(directory, "s1", outcome)
        meter.record_directory_commit(directory, "s1", outcome)  # 第二轮
        snapshot = meter.snapshot()
        self.assertEqual(snapshot["op_counts"]["query_sources"], 2)
        self.assertEqual(snapshot["op_total_ns"]["evict_to_pool"], 7)
        self.assertEqual(snapshot["peak_entry_count"], 1)
        rounds = snapshot["session_rounds"]["s1"]
        self.assertEqual([r["round"] for r in rounds], [1, 2])
        # 第二轮增长 = source_count 差（无拆分预算口径的计量输入）。
        self.assertEqual(rounds[1]["growth"], 0)

    def test_timed_op_context(self):
        meter = DirectoryOverheadMeter()
        with meter.timed_op("enumerate_source_surface"):
            pass
        self.assertEqual(meter.snapshot()["op_counts"]
                         ["enumerate_source_surface"], 1)


if __name__ == "__main__":
    unittest.main()
