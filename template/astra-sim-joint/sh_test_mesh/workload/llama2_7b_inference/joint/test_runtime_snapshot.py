"""test_runtime_snapshot.py -- W1 因果一致性快照验收单测。

覆盖（方案 §4.1 DecisionSnapshot + W1 验收"快照不含禁止字段"）：

* V1 因果性：改隐藏后缀长度（decode_length / inter_request_interval_ns /
  CSV 后继行 / final_context_tokens）→ 快照逐字节不变；projection_log
  在场记录过滤项（记字段名与条数，不记值）；
* 白名单投影：arrivals 只保留因果可见字段；请求队列冻结视图的禁止
  字段被防御性剔除；快照全域（递归扫描）不含 FORBIDDEN_FIELDS；
* work_index：按 resource_leg 分组的只读索引（每链路 active flow、每
  HBM rank 活跃 job、每 pool port job、每 rank 在途节点）；
* resource_state 四分列：仲裁状态 / 吞吐观测 / slot gate / 物理容量；
  遥测缺席 = None（不补零）；
* existing_unknown_work：exposed_progress + 删失标记 + estimate 来源，
  不含 decode_length；
* 版本失效校验接口：cache_key 相等/失效判定；
* 旧对端兼容：work_ledger 字段缺席 → 空骨架 + projection_log 显式记
  录（不冒装观测）。
"""

import copy
import json
import os
import sys
import unittest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_THIS_DIR)
for _p in (_WORKLOAD_DIR, os.path.dirname(_WORKLOAD_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from joint.runtime_snapshot import (  # noqa: E402
    ARRIVAL_WHITELIST,
    FORBIDDEN_FIELDS,
    RESOURCE_STATE_COLUMNS,
    VERSION_KEYS,
    build_decision_snapshot,
    build_existing_unknown_work,
    build_work_index,
    project_arrivals,
)
from joint.schema_registry import SNAPSHOT_SCHEMA_VERSION  # noqa: E402


def _delta(seq=0, tick=10, decode_length=32, interval_ns=5000,
           telemetry=None, work_ledger=None):
    delta = {
        "schema_version": 1,
        "delivery_sequence": seq,
        "delivery_epoch": seq,
        "tick": tick,
        "deferred_from_tick": 0,
        "reasons": ["ARRIVAL"],
        "arrivals": [{
            "request_id": "req-1",
            "session_id": "sess-1",
            "turn_index": 0,
            "prefill_length": 128,
            "decode_length": decode_length,
            "inter_request_interval_ns": interval_ns,
            "arrival_world_ns": 0,
            "ingress_seq": 1,
            "queue_index": 0,
        }],
        "completed_groups": [],
        "completed_nodes": [],
        "affected_ranks": [],
        "snapshot_handle": {"epoch": seq, "tick": tick, "kind": ""},
        "ledger_summary": {"injected_unfinished": []},
    }
    if telemetry is not None:
        delta["link_telemetry"] = telemetry
    if work_ledger is not None:
        delta["work_ledger"] = work_ledger
    return delta


def _state():
    return {
        "nodes": [{"rank": 0, "nodes": [{
            "node_id": 1, "request_id": "req-1", "stage": "prefill",
            "generation": 0, "kind": "compute", "issued": True,
            "finished": False, "unresolved_parents": 0, "num_ops": 100,
            "bytes": 0, "runtime_ns": 50, "tensor_size": 0, "parents": [],
        }]}],
        "noc": {
            "flows": [{
                "flow_id": 3, "total_bytes": 1000, "remaining_bytes": 400,
                "current_rate_Bpns": 1.5, "last_rate_update_time": 8,
                "route": [0, 1], "state": "Active",
            }],
            "links": [{"link_id": 0, "capacity_Bpns": 2.0,
                       "active_flow_count": 1}],
        },
        "hbm": [{"rank": 0, "as_of_tick": 10, "jobs": [{
            "kind": "COMPUTE", "node_id": 1, "remaining_ops": 60,
            "remaining_bytes": 200, "remaining_latency_ns": 0,
            "request_id": "req-1", "stage": "prefill", "generation": 0,
        }]}],
        "pool": [{"port_index": 0, "as_of_ns": 10, "bandwidth_Bpns": 6.0,
                  "latency_waiting": 1,
                  "streaming": 0, "completion_waiting": 0, "jobs": [{
                      "issue_seq": 1, "sys_id": 0, "node_id": 9,
                      "bytes": 500, "remaining_bytes": 500,
                      "state": "LatencyWaiting", "latency_ready_ns": 20,
                      "request_id": "req-1", "stage": "prefill",
                      "generation": 0,
                  }]}],
        "slots": [{"rank": 0, "cpu": 0, "gpu_comp": 1, "gpu_comm": 0,
                   "hbm_dma": 0, "remote_mem": 1}],
    }


def _queues(exposed=2, decode_length=32):
    return {
        "qp": [{"request_id": "req-1", "prefill_length": 128,
                "decode_length": decode_length}],
        "exposed_decode_steps": {"req-1": exposed},
    }


def _contains_forbidden(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in FORBIDDEN_FIELDS:
                return key
            found = _contains_forbidden(item)
            if found:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _contains_forbidden(item)
            if found:
                return found
    return None


class TestProjection(unittest.TestCase):
    def test_arrival_whitelist(self):
        projected, log = project_arrivals(_delta()["arrivals"])
        self.assertEqual(
            set(projected[0]), {
                "request_id", "session_id", "turn_index", "prefill_length",
                "arrival_world_ns", "ingress_seq", "queue_index"})
        self.assertIn("arrivals[].decode_length filtered (1 entries)", log)
        self.assertIn(
            "arrivals[].inter_request_interval_ns filtered (1 entries)", log)

    def test_projection_log_carries_no_values(self):
        _, log = project_arrivals(_delta(decode_length=31337)["arrivals"])
        for entry in log:
            self.assertNotIn("31337", entry)


class TestCausalityV1(unittest.TestCase):
    """V1 验证锚：改隐藏后缀长度不改快照（逐字节）。"""

    def _snapshots(self):
        common = dict(
            versions={"runtime": 4, "layout": 2},
            request_queues=_queues(exposed=2),
            observation_lag_ns=0,
        )
        snap_a = build_decision_snapshot(
            _delta(decode_length=32, interval_ns=5000), **common)
        snap_b = build_decision_snapshot(
            _delta(decode_length=999, interval_ns=123456), **common)
        return snap_a, snap_b

    def test_hidden_suffix_change_leaves_snapshot_bytes_identical(self):
        snap_a, snap_b = self._snapshots()
        self.assertEqual(snap_a.to_json(), snap_b.to_json())

    def test_visible_change_alters_snapshot(self):
        snap_a, _ = self._snapshots()
        snap_c = build_decision_snapshot(
            _delta(decode_length=32, interval_ns=5000, tick=11),
            versions={"runtime": 4, "layout": 2},
            request_queues=_queues(exposed=3),
            observation_lag_ns=0)
        self.assertNotEqual(snap_a.to_json(), snap_c.to_json())

    def test_no_forbidden_field_anywhere(self):
        snap, _ = self._snapshots()
        leaked = _contains_forbidden(json.loads(snap.to_json()))
        self.assertIsNone(leaked, "forbidden field leaked: %s" % leaked)

    def test_exposed_progress_is_causal(self):
        """已消费 decode 步数（exposed_progress）因果可见：不同值 → 快照
        不同；且它来自观测而非 decode_length。"""
        snap_more = build_decision_snapshot(
            _delta(), versions={"runtime": 1}, request_queues=_queues(exposed=5))
        snap_less = build_decision_snapshot(
            _delta(), versions={"runtime": 1}, request_queues=_queues(exposed=1))
        self.assertNotEqual(snap_more.to_json(), snap_less.to_json())


class TestWorkIndex(unittest.TestCase):
    def test_grouping_by_resource_leg(self):
        index = build_work_index(_state())
        self.assertEqual(index["by_link"][0], [3])       # flow 3 on link 0
        self.assertEqual(len(index["by_hbm_rank"][0]), 1)
        self.assertEqual(index["by_hbm_rank"][0][0]["node_id"], 1)
        self.assertEqual(len(index["by_pool_port"][0]), 1)
        self.assertEqual(index["by_rank_nodes"][0], [1])

    def test_node_view_and_leg_view_keyed_separately(self):
        """同一物理腿的两张摘要分键入索引：节点视图按 rank、HBM 腿按
        rank 的 job 集——不互相覆盖、不重复合并计费。"""
        index = build_work_index(_state())
        self.assertIn(0, index["by_rank_nodes"])
        self.assertIn(0, index["by_hbm_rank"])
        self.assertEqual(index["by_rank_nodes"][0], [1])
        self.assertEqual(index["by_hbm_rank"][0][0]["remaining_ops"], 60)

    def test_propagating_tail_is_not_an_active_link_member(self):
        state = {"noc": {"flows": [{
            "flow_id": 9, "state": "PropagatingTail", "route": [4, 5],
        }, {
            "flow_id": 10, "state": "Active", "route": [5],
        }]}}
        index = build_work_index(state)
        self.assertNotIn(4, index["by_link"])
        self.assertEqual(index["by_link"][5], [10])


class TestResourceState(unittest.TestCase):
    def test_four_columns_present(self):
        state = build_resource_state_helper()
        self.assertEqual(
            tuple(sorted(state)), tuple(sorted(RESOURCE_STATE_COLUMNS)))

    def test_arbitration_vs_capacity_vs_slot(self):
        state = build_resource_state_helper()
        arb = state["arbitration"]["noc_links"][0]
        cap = state["physical_capacity"]["noc_link_Bpns"][0]
        gate = state["slot_gate"][0]
        self.assertEqual(arb["active_flow_count"], 1)   # 仲裁：active set
        self.assertEqual(cap, 2.0)                      # 容量：带宽标量
        self.assertEqual(
            state["physical_capacity"]["pool_Bpns"][0], 6.0)
        self.assertEqual(gate["gpu_comp"], 1)           # slot gate：在途计数
        self.assertEqual(arb["active_flow_count"], 1)
        self.assertNotEqual(arb, cap)                   # 语义分列不混用

    def test_throughput_observation_none_when_telemetry_absent(self):
        state = build_resource_state_helper()
        self.assertIsNone(state["throughput_observation"])

    def test_throughput_observation_passthrough_when_present(self):
        state = build_resource_state_helper(telemetry=[
            {"link_id": 0, "served_bytes": 10}])
        self.assertEqual(state["throughput_observation"][0]["served_bytes"], 10)


def build_resource_state_helper(telemetry=None):
    from joint.runtime_snapshot import build_resource_state
    return build_resource_state(
        _delta(telemetry=telemetry), _state())


class TestUnknownWork(unittest.TestCase):
    def test_estimate_marked_and_censored(self):
        unknown = build_existing_unknown_work(_queues(exposed=7), tuple(
            project_arrivals(_delta()["arrivals"])[0]))
        self.assertEqual(
            unknown["req-1"],
            {"exposed_progress": 7, "censoring": True, "source": "estimate"})
        leaked = _contains_forbidden(unknown)
        self.assertIsNone(leaked)


class TestVersions(unittest.TestCase):
    def test_cache_key_covers_all_versions(self):
        snap = build_decision_snapshot(
            _delta(), versions={"runtime": 1, "layout": 2, "model": "v3"},
            request_queues={})
        keys = [name for name, _ in snap.cache_key()]
        self.assertEqual(keys, list(VERSION_KEYS))
        self.assertIn(("model", "v3"), snap.cache_key())

    def test_version_invalidation(self):
        snap_a = build_decision_snapshot(
            _delta(), versions={"runtime": 1}, request_queues={})
        snap_same = build_decision_snapshot(
            _delta(), versions={"runtime": 1}, request_queues={})
        snap_bumped = build_decision_snapshot(
            _delta(), versions={"runtime": 2}, request_queues={})
        self.assertTrue(snap_a.is_valid_against(snap_same))
        self.assertFalse(snap_a.is_valid_against(snap_bumped))

    def test_unknown_version_key_rejected(self):
        with self.assertRaises(ValueError):
            build_decision_snapshot(
                _delta(), versions={"surprise": 1}, request_queues={})

    def test_schema_version_pinned(self):
        snap = build_decision_snapshot(_delta(), request_queues={})
        self.assertEqual(snap.schema_version, SNAPSHOT_SCHEMA_VERSION)


class TestLegacyPeer(unittest.TestCase):
    def test_absent_work_ledger_leaves_empty_skeleton_disclosed(self):
        snap = build_decision_snapshot(
            _delta(), versions={"runtime": 1}, request_queues={})
        self.assertEqual(snap.work_index["by_link"], {})
        self.assertIn(
            "work_ledger field absent (legacy peer)", " ".join(
                snap.projection_log))

    def test_full_payload_drives_index(self):
        envelope = {
            "schema_version": 1,
            "snapshot_kind": "full",
            "delivery_sequence": 0,
            "tick": 10,
            "event_watermark": 0,
            "state": _state(),
        }
        snap = build_decision_snapshot(
            _delta(work_ledger=envelope), versions={"runtime": 1},
            request_queues={})
        self.assertEqual(snap.work_index["by_link"][0], [3])
        self.assertEqual(snap.event_watermark, 0)

    def test_pricing_only_does_not_rebuild_full_frame_indexes(self):
        envelope = {
            "schema_version": 1,
            "snapshot_kind": "full",
            "delivery_sequence": 0,
            "tick": 10,
            "event_watermark": 0,
            "state": _state(),
        }
        snap = build_decision_snapshot(
            _delta(work_ledger=envelope), request_queues={},
            ledger_projection=object(), pricing_only=True)
        self.assertEqual(snap.work_index["by_link"], {})
        self.assertIn(
            "hot pricing snapshot omits full work_index/resource_state",
            " ".join(snap.projection_log))


if __name__ == "__main__":
    unittest.main()
