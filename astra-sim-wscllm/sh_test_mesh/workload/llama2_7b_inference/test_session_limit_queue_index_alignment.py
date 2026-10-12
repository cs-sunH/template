#!/usr/bin/env python3
"""test_session_limit_queue_index_alignment.py -- P2-21 钉死测试。

契约（2026-09-28 修复）：request_queue_session_limit>0（过滤态）下，
queue_index 必须与 C++ 同源——WindowedTraceReader 对它读到的未过滤全量
CSV 按"数据行序（0 起）"编号 queue_index（process_indexed_row:
``queue_index = data_rows_ - 1``）；session_limit 过滤只发生在 Python 侧，
按过滤后序 enumerate 会与 C++ 错位。修复后：

  * config 携带 request_queue_indices（每过滤行的源 CSV 行序）与逆映射
    queue_position_by_index（源行序 -> 过滤位次，供
    config.request_queue[pos] 定位）；
  * plan_materializer 两个 manifest 的 requests[].queue_index 与
    after_request 的 parent_queue_index 取源 CSV 行序（含跳号）；
  * graph_batch_builder/_interval_ns 经逆映射定位，不再把 queue_index
    当过滤位次使用。

对拍口径：构造 limit>0 的小输入（首会话行散布在中间行号），manifest
queue_index 必须等于各行的源 CSV 数据行序（= C++ 会编的号），而不是
0..K-1 连续过滤序。
"""

import sys
import unittest
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
SH_TEST_DIR = MODULE_DIR.parents[1]
for _p in (str(MODULE_DIR), str(SH_TEST_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from generate_wsc_llm_trace import (  # noqa: E402
    queue_index_mappings,
    select_first_session_requests,
)
from generate_trace import RequestSpec  # noqa: E402
from plan_materializer import (  # noqa: E402
    _derive_manifest_requests,
    _derive_metrics_requests,
)


def _spec(rid, sid, turn, arrival=None, interval=None):
    return RequestSpec(
        session_id=sid, turn_index=turn, request_id=rid,
        prefill_length=100, decode_length=10,
        session_arrival_time_ns=arrival,
        inter_request_interval_ns=interval)


# 源 CSV 行序：A0(0) B0(1) A1(2) C0(3) A2(4) B1(5) —— 会话行交错。
SOURCE = [
    _spec("A0", "A", 0, arrival=1000),
    _spec("B0", "B", 0, arrival=2000),
    _spec("A1", "A", 1, interval=500),
    _spec("C0", "C", 0, arrival=3000),
    _spec("A2", "A", 2, interval=500),
    _spec("B1", "B", 1, interval=500),
]


def _stub_config(specs, indices):
    from types import SimpleNamespace
    return SimpleNamespace(
        request_queue=list(specs),
        request_queue_indices=tuple(indices),
        queue_position_by_index={
            index: position
            for position, index in enumerate(indices)},
        inference_groups=[SimpleNamespace(ranks=(0, 1, 2))],
    )


class SessionLimitQueueIndexTests(unittest.TestCase):
    """过滤态 queue_index 两端同源（C++ 数据行序）钉死。"""

    def test_mappings_source_ordinals_with_gaps(self):
        selected, _ = select_first_session_requests(SOURCE, 1)  # 会话 A
        self.assertEqual([s.request_id for s in selected],
                         ["A0", "A1", "A2"])
        indices, position_by_index = queue_index_mappings(SOURCE, selected)
        # 源 CSV 行序（= C++ queue_index），非 0..K-1 过滤序。
        self.assertEqual(indices, (0, 2, 4))
        self.assertEqual(position_by_index, {0: 0, 2: 1, 4: 2})

    def test_mappings_identity_when_unfiltered(self):
        selected, _ = select_first_session_requests(SOURCE, 0)
        indices, position_by_index = queue_index_mappings(SOURCE, selected)
        self.assertEqual(indices, tuple(range(len(SOURCE))))
        self.assertEqual(
            position_by_index,
            {index: index for index in range(len(SOURCE))})

    def test_manifests_carry_source_ordinals(self):
        selected, _ = select_first_session_requests(SOURCE, 1)
        indices, _ = queue_index_mappings(SOURCE, selected)
        config = _stub_config(selected, indices)
        sidecar = {spec.request_id: {"request_id": spec.request_id,
                                     "request_type": "human"}
                   for spec in selected}

        requests = _derive_manifest_requests(config, sidecar)
        self.assertEqual([r["queue_index"] for r in requests], [0, 2, 4])

        metrics = _derive_metrics_requests(config, sidecar)
        self.assertEqual([r["queue_index"] for r in metrics], [0, 2, 4])
        # turn>0 的 after_request 父序号也必须是源行序（A1 的父 = A0 的 0，
        # A2 的父 = A1 的 2），C++ MetricCollector 按同一序号解析。
        by_id = {r["request_id"]: r for r in metrics}
        self.assertEqual(by_id["A1"]["arrival"]["parent_queue_index"], 0)
        self.assertEqual(by_id["A2"]["arrival"]["parent_queue_index"], 2)

    def test_position_lookup_round_trip(self):
        """graph_batch_builder/_interval_ns 的定位口径：源行序经逆映射回
        过滤位次后取到同一 spec（queue_index 不再被当过滤位次使用）。"""
        selected, _ = select_first_session_requests(SOURCE, 1)
        indices, position_by_index = queue_index_mappings(SOURCE, selected)
        for position, spec in enumerate(selected):
            lookup = selected[position_by_index[indices[position]]]
            self.assertIs(lookup, spec)


if __name__ == "__main__":
    unittest.main()
