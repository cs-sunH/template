#!/usr/bin/env python3
"""test_decision_bridge_die_tables.py -- 修复项 2.2 hoist 纯函数单测。

覆盖:
  1. 含 die_op_spec 的 response 被提升为 die_spec_tables + 引用式 spec,
     表内容(按 ref 回填)与原 spec 逐字段一致;
  2. 同内容 spans/tp/context 去重为同一表条目(同 id 引用), 不同内容分列;
  3. 无 die 节点 => 返回 None(legacy/fixture 零改动路径);
  4. 形状门 fail-closed: spec 键集偏离契约 section 2 内联 13 键全集即
     BridgeError;
  5. 纯函数性: 提升后输入 response 不被改写(调度器 batch/digest 语义源)。

运行: cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 online/test_decision_bridge_die_tables.py
"""

import copy
import os
import sys
import unittest

_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _p in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from online.decision_bridge import (  # noqa: E402
    BridgeError,
    _hoist_die_spec_tables,
)


def _span(request_id, q_tokens, kv_tokens, q_start_pos, kv_start_pos=0):
    return {
        "request_id": request_id,
        "q_tokens": q_tokens,
        "kv_tokens": kv_tokens,
        "q_start_pos": q_start_pos,
        "kv_start_pos": kv_start_pos,
    }


def _spec(op_kind, spans, *, tp_rank=0, layer_count=32, weight_passes=3):
    return {
        "schema_version": 1,
        "op_kind": op_kind,
        "phase": "decode",
        "dtype": "bf16",
        "model_dims": {
            "hidden": 4096, "ffn": 11008, "vocab": 32000,
            "mlp_variant": "swiglu", "norm_type": "rmsnorm",
            "position_encoding": "rope", "num_q_heads": 32,
            "num_kv_heads": 32, "head_dim": 128,
        },
        "tp_slice": {
            "tensor_parallel": 2, "tensor_parallel_rank": tp_rank,
            "q_heads_local": 16, "kv_heads_local": 16,
            "ffn_local": 5504, "vocab_local": 16000,
        },
        "spans": spans,
        "layer_count": layer_count,
        "weight_passes": weight_passes,
        "kv_access": {"identity": "local"},
        "fusion_policy": "auto",
        "provenance": "transformer_pass_aggregated",
        "config_hash": "deadbeef" * 8,
    }


def _compute_node(name, spec):
    return {
        "rank": 0, "id": 1, "name": name, "type": 4,
        "is_cpu_op": False, "is_timer_op": False, "inputs_values": "",
        "request_id": "train", "stage": "decode", "generation": 1,
        "compute": {
            "num_ops": 0, "tensor_size": 0, "runtime_ns": 0,
            "die_op_spec": spec,
        },
        "comm": {}, "coll": {},
    }


class HoistDieSpecTablesTest(unittest.TestCase):
    def test_hoist_tables_and_refs_round_trip(self):
        spans = [_span("train#s0", 1, 41, 40), _span("train#s1", 1, 42, 41)]
        response = {
            "schema_version": 1, "batch_id": 5,
            "source_delivery_sequence": 5,
            "nodes": [
                _compute_node("a", _spec("attention_region", spans)),
                _compute_node("b", _spec("ffn_region", spans)),
            ],
            "parent_edges": [], "watches": [], "assignments": [],
            "kv_actions": [], "future_alarms": [], "touched_ranks": [0],
        }
        wire = _hoist_die_spec_tables(response)
        self.assertIsNotNone(wire)
        tables = wire["die_spec_tables"]
        self.assertEqual(len(tables["contexts"]), 1)
        self.assertEqual(len(tables["tp_slices"]), 1)
        self.assertEqual(len(tables["spans"]), 1)
        self.assertEqual(tables["spans"][0], spans)
        # 两份同内容 spec 共享全部三个表 id（op_kind 等逐组字段保留在 spec 内）。
        refs = [node["compute"]["die_op_spec"] for node in wire["nodes"]]
        self.assertEqual(
            (refs[0]["context_ref"], refs[0]["tp_slice_ref"],
             refs[0]["spans_ref"]),
            (refs[1]["context_ref"], refs[1]["tp_slice_ref"],
             refs[1]["spans_ref"]))
        self.assertEqual(refs[0]["op_kind"], "attention_region")
        self.assertEqual(refs[1]["op_kind"], "ffn_region")
        self.assertEqual(
            sorted(refs[0]),
            ["context_ref", "fusion_policy", "layer_count", "op_kind",
             "spans_ref", "tp_slice_ref", "weight_passes"])
        # 表内容按 ref 回填后与原 spec 逐字段一致。
        spec0 = response["nodes"][0]["compute"]["die_op_spec"]
        context = tables["contexts"][refs[0]["context_ref"]]
        for key in ("schema_version", "phase", "dtype", "model_dims",
                    "kv_access", "provenance", "config_hash"):
            self.assertEqual(context[key], spec0[key])
        self.assertEqual(
            tables["tp_slices"][refs[0]["tp_slice_ref"]], spec0["tp_slice"])
        self.assertEqual(
            tables["spans"][refs[0]["spans_ref"]], spec0["spans"])

    def test_distinct_spans_get_distinct_entries(self):
        spans_a = [_span("t1#s0", 1, 41, 40)]
        spans_b = [_span("t2#s0", 128, 256, 128)]
        response = {
            "nodes": [
                _compute_node("a", _spec("norm", spans_a)),
                _compute_node("b", _spec("norm", spans_b)),
                _compute_node(
                    "c", _spec("norm", spans_a, tp_rank=1)),
            ],
        }
        wire = _hoist_die_spec_tables(response)
        tables = wire["die_spec_tables"]
        self.assertEqual(len(tables["spans"]), 2)
        self.assertEqual(len(tables["tp_slices"]), 2)  # tp_rank 0/1
        self.assertEqual(len(tables["contexts"]), 1)
        specs = [node["compute"]["die_op_spec"] for node in wire["nodes"]]
        self.assertEqual(specs[0]["spans_ref"], specs[2]["spans_ref"])
        self.assertNotEqual(specs[0]["spans_ref"], specs[1]["spans_ref"])
        self.assertNotEqual(specs[0]["tp_slice_ref"], specs[2]["tp_slice_ref"])

    def test_no_die_nodes_returns_none(self):
        response = {"nodes": [{"compute": {"num_ops": 1}}], "watches": []}
        self.assertIsNone(_hoist_die_spec_tables(response))
        self.assertIsNone(_hoist_die_spec_tables({"nodes": []}))

    def test_malformed_spec_fails_closed(self):
        spec = _spec("norm", [_span("s#0", 1, 8, 7)])
        del spec["spans"]
        with self.assertRaises(BridgeError):
            _hoist_die_spec_tables(
                {"nodes": [_compute_node("a", spec)]})
        spec = _spec("norm", [_span("s#0", 1, 8, 7)])
        spec["surprise"] = 1
        with self.assertRaises(BridgeError):
            _hoist_die_spec_tables(
                {"nodes": [_compute_node("a", spec)]})
        with self.assertRaises(BridgeError):
            _hoist_die_spec_tables(
                {"nodes": [_compute_node("a", {"op_kind": "norm"})]})

    def test_hoist_is_pure(self):
        spans = [_span("train#s0", 1, 41, 40)]
        response = {
            "nodes": [_compute_node("a", _spec("logits", spans))],
        }
        before = copy.deepcopy(response)
        wire = _hoist_die_spec_tables(response)
        self.assertIsNotNone(wire)
        self.assertEqual(response, before)  # 输入未被改写
        # 幂等性来源: 再次 hoist 原始 dict 得到相同 wire 形态。
        wire2 = _hoist_die_spec_tables(response)
        self.assertEqual(wire, wire2)


if __name__ == "__main__":
    unittest.main()
