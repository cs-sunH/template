#!/usr/bin/env python3
"""test_c7a_transfer_summary_multi_leg.py -- C7a/W2 钉测：_transfer_summary
对多腿元组形态（增量 copy@第三目标/i_P 的 ``primary=(transfer, base_leg)``
两腿）的发射面修复。

背景（W2 只读法证）：commit_boundary_decode_transition 的增量 copy 路径
返回多腿元组（腿序 = 发射序，graph_batch_builder 逐腿发射），而
``_transfer_summary(runtime.prefill_decode_transfer)`` 按单腿读
``.kind`` → AttributeError（'tuple' object has no attribute 'kind'），
decode 决策行的迁移披露崩溃。修法 = 多腿形态逐腿披露：

* 顶层保持**首腿（主迁移腿）**的单腿契约键（kind/source_instance_
  index/total_bytes——slo_tools/hbm_watermark 的 P→D 迁移重放按首腿
  noc_migrate 校验跟踪副本字节，不得改为腿汇总）；
* ``shards`` = 全部腿物理 shard 按发射序平铺（hopbytes 逐 shard 采集
  不漏腿）；
* ``multi_leg``/``leg_count``/``legs[]`` 多腿专属披露（单腿形态不落，
  与既有钉测逐位一致）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 -m pytest online/test_c7a_transfer_summary_multi_leg.py -q
"""
import os
import sys
import unittest

_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _p in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from face_scheduler import (  # noqa: E402
    KVTransfer,
    KVTransferShard,
)
from online.sh30_online_scheduler import _transfer_summary  # noqa: E402


def _shard(source_rank, target_rank, nbytes):
    return KVTransferShard(
        source_rank=source_rank, target_rank=target_rank,
        edge_rank=None, bytes=nbytes, noc_path=(source_rank, target_rank),
        layer_start=0, layer_end=4)


def _leg(kind, source_instance, target_instance, nbytes, shards):
    return KVTransfer(
        kind=kind, phase="prefill_decode", reason="boundary_test",
        session_id="s", trigger_request_id="r0",
        source_instance_index=source_instance,
        target_instance_index=target_instance,
        total_bytes=nbytes, shards=tuple(shards), model_layers=4,
        layer_start=0, layer_end=4,
        resident_prefix_layers_before=4,
        resident_prefix_layers_after=4)


def _two_leg_tuple():
    """增量 copy@第三目标的两腿形态：腿 1 = U 迁移（i_P→j），腿 2 =
    H 复制（home→j）——腿序 = 发射序（与 face_scheduler 构造序一致）。"""
    u_leg = _leg("noc_migrate", 0, 1, 400, [_shard(0, 2, 200),
                                            _shard(1, 3, 200)])
    h_leg = _leg("noc_migrate", 2, 1, 800, [_shard(4, 2, 400),
                                            _shard(5, 3, 400)])
    return u_leg, h_leg


class MultiLegSummaryUnitTest(unittest.TestCase):
    """_transfer_summary 多腿分支（纯函数面）。"""

    def test_tuple_does_not_crash_and_discloses_legs_in_order(self):
        # W2 隐患回归：修前在 transfer.kind 处 AttributeError。
        u_leg, h_leg = _two_leg_tuple()
        summary = _transfer_summary((u_leg, h_leg))
        self.assertIsNotNone(summary)
        self.assertTrue(summary["multi_leg"])
        self.assertEqual(summary["leg_count"], 2)
        # 腿序 = 发射序（U 迁移在前、H 复制在后）。
        self.assertEqual(
            summary["legs"],
            [_transfer_summary(u_leg), _transfer_summary(h_leg)])
        # 顶层 = 首腿（主迁移腿）契约键。
        self.assertEqual(summary["kind"], "noc_migrate")
        self.assertEqual(summary["source_instance_index"], 0)
        self.assertEqual(summary["total_bytes"], u_leg.total_bytes)
        # shards 平铺全部腿的物理 shard（hopbytes 不漏腿）。
        self.assertEqual(
            summary["shards"],
            _transfer_summary(u_leg)["shards"]
            + _transfer_summary(h_leg)["shards"])
        self.assertEqual(len(summary["shards"]), 4)

    def test_single_leg_shape_unchanged(self):
        # 单腿形态不落多腿键（与既有钉测逐位一致——读者零感知）。
        u_leg, _h_leg = _two_leg_tuple()
        single = _transfer_summary(u_leg)
        for key in ("multi_leg", "leg_count", "legs"):
            self.assertNotIn(key, single)
        self.assertEqual(single["kind"], "noc_migrate")

    def test_none_and_empty_passthrough(self):
        self.assertIsNone(_transfer_summary(None))
        self.assertIsNone(_transfer_summary(()))


class DecodeRowMultiLegEmissionTest(unittest.TestCase):
    """decode 决策行的迁移披露（真两腿元组 → 真行，不崩且字段齐全）。"""

    def test_emit_join_decision_with_two_leg_primary(self):
        from online.test_c4_boundary_full_surface import (
            HISTORY_TOKENS,
            HOME_INSTANCE,
            INPUT_TOKENS,
            P_INSTANCE,
            THIRD_INSTANCE,
            _increment_kv,
        )
        from online.test_decision_route_serialization import (
            _decision_shell,
            _make_runtime,
        )
        kv, model, topology = _increment_kv()
        # 真边界事务产出两腿元组（C4 增量 copy@第三目标）。
        primary, _evictions, outcome = kv.commit_boundary_decode_transition(
            session_id="s", p_instance_index=P_INSTANCE,
            target_instance_index=THIRD_INSTANCE, action="copy",
            history_tokens=HISTORY_TOKENS + INPUT_TOKENS,
            trigger_request_id="r0")
        self.assertEqual(outcome["kind"], "increment_copy_third_h_plus_u")
        self.assertIsInstance(primary, tuple)
        scheduler = _decision_shell(kv, topology=topology, model=model)
        runtime = _make_runtime("r0")
        runtime.prefill_instance_index = P_INSTANCE
        runtime.decode_instance_index = THIRD_INSTANCE
        runtime.joint_action = "remote-read"
        runtime.joint_decode_action = "copy"
        runtime.prefill_decode_transfer = primary
        # 修前：本调用 AttributeError（tuple 无 .kind）。
        scheduler._emit_join_decision(runtime, 1000)
        rows = [row for row in scheduler.online_log_rows
                if row["kind"] == "decode"]
        self.assertEqual(len(rows), 1)
        decision = rows[0]["decision"]
        summary = decision["prefill_decode_transfer"]
        self.assertTrue(summary["multi_leg"])
        self.assertEqual(summary["leg_count"], 2)
        # 首腿契约字段齐全（W1 迁移重放口径：noc_migrate + 源端 + 字节）。
        self.assertEqual(summary["kind"], "noc_migrate")
        self.assertEqual(summary["source_instance_index"], P_INSTANCE)
        self.assertEqual(
            summary["total_bytes"], primary[0].total_bytes)
        # 逐腿完整字段（发射序）。
        self.assertEqual(
            summary["legs"][0], _transfer_summary(primary[0]))
        self.assertEqual(
            summary["legs"][1], _transfer_summary(primary[1]))
        self.assertEqual(
            summary["legs"][1]["source_instance_index"], HOME_INSTANCE)
        # 全腿 shard 平铺且字段齐全（bytes/noc_hops/noc_path 逐 shard）。
        flat_shards = summary["shards"]
        self.assertEqual(
            len(flat_shards),
            len(primary[0].shards) + len(primary[1].shards))
        for shard in flat_shards:
            self.assertIn("bytes", shard)
            self.assertIn("noc_hops", shard)
            self.assertIn("noc_path", shard)
            self.assertEqual(shard["noc_hops"],
                             len(shard["noc_path"]) - 1)


if __name__ == "__main__":
    unittest.main()
