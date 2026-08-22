#!/usr/bin/env python3
"""test_graph_batch_builder.py -- sh_1.0 watch 锚点统一(barrier 前末节点)
钉子测试。

背景(《5仓库本该一致却不同排查报告-第二轮.md》R2-2,
2026-08-20):PREFILL_DRAIN/
DECODE_COMPLETION watch 成员原取 end-barrier 之后的节点(barrier 本身),
face/sh_2.0/sh_3.0/wscllm 四仓均取 barrier 之前每 rank 真实末节点(与
离线 EVENT_PREFILL_END/EVENT_DECODE_END 锚点一致,排除 end barrier)。
本测试经 emit_prefill_batch/emit_decode_batch 完整发射(多 chunk prefill、
多 step decode)后钉住拆分语义:
  (a) watch 成员 == 每 rank 最后一个 prefill/decode 计算节点 id
      (aggregated 段的 *_all_passes_logits_projection COMP 节点,
      即 barrier 前紧邻节点);
  (b) watch 成员 != end-barrier 节点 id(改前必败——改前成员即 barrier);
  (c) 触发门口径回归:_block_ends["seg1"]/["seg2"](decode_evictions/
      completion_evictions 触发门与下一 turn interval gate after_node_id
      的来源)仍为 post-barrier 节点(== 各自 end-barrier 节点 id)。

运行:cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 online/test_graph_batch_builder.py   （或 pytest 同路径）
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

from generate_trace import COMP_NODE  # noqa: E402
from online.graph_batch_builder import GraphBatchBuilder  # noqa: E402

SESSION = "session_nail_0"
REQUEST_ID = f"{SESSION}_request_0"
PREFILL_TOKENS = 300   # chunk 128 -> 3 个 chunk(多 chunk 灯具)
CHUNKS = 3
DECODE_LENGTH = 4      # 多 step decode 灯具
PREFILL_RANKS = (0, 1)
DECODE_RANKS = (2, 3)


def _last_barrier(builder, rank: int, suffix: str) -> tuple:
    """返回 (barrier 节点, barrier 前紧邻节点)——该 rank 节点序列里
    最后一个名字以 suffix 结尾的 barrier 及其前一节点。"""
    nodes = builder.builders[rank].nodes
    index = max(
        i for i, node in enumerate(nodes)
        if node["name"].endswith(suffix)
    )
    assert index >= 1, "barrier is the first node on the rank?"
    return nodes[index], nodes[index - 1]


class WatchAnchorNailTest(unittest.TestCase):
    """R2-2:watch 成员(barrier 前)与触发门(post-barrier)的拆分钉子。"""

    def setUp(self) -> None:
        self.config = SimpleNamespace(
            npus_count=4,
            remote_operand_loads=False,
            trace_granularity="request_aggregated",
            inference_groups=[
                SimpleNamespace(ranks=PREFILL_RANKS, pg_name="tp_prefill"),
                SimpleNamespace(ranks=DECODE_RANKS, pg_name="tp_decode"),
            ],
            prefill_chunk_size=128,
            layers=2,
            hidden_size=64,
            ffn_size=128,
            vocab_size=256,
            bytes_per_elem=2,
            num_heads=8,
            mlp_variant="gelu",
            request_queue=[
                SimpleNamespace(
                    decode_length=DECODE_LENGTH,
                    inter_request_interval_ns=0,
                )
            ],
        )
        self.builder = GraphBatchBuilder(self.config)
        self.builder.begin_batch()
        self.prefill_plan = {
            "request_id": REQUEST_ID,
            "session_id": SESSION,
            "turn_index": 0,
            "queue_index": 0,
            "prefill_instance_index": 0,
            "decode_instance_index": 0,   # 准入时刻占位(调度器同款)
            "admission_time_ns": 0,
            "history_location_before": None,
            "history_transfer": None,
            "history_evictions": [],
            "prefill_evictions": [],
            "history_tokens_before": 0,
            "prefill_context_tokens": PREFILL_TOKENS,
        }
        self.decode_plan = {
            "request_id": REQUEST_ID,
            "session_id": SESSION,
            "turn_index": 0,
            "queue_index": 0,
            "prefill_instance_index": 0,
            "decode_instance_index": 1,
            "prefill_context_tokens": PREFILL_TOKENS,
            "decode_evictions": [],
            "prefill_decode_transfer": {
                "kind": "local_hit",
                "phase": "prefill_to_decode",
                "reason": "nail_test_same_tp_group",
                "session_id": SESSION,
                "trigger_request_id": REQUEST_ID,
                "source_instance_index": 0,
                "target_instance_index": 1,
                "total_bytes": 0,
                "shards": [],
            },
        }

    def test_prefill_drain_watch_members_are_pre_barrier_nodes(self):
        """(a)+(b):PREFILL_DRAIN 成员 = 每 rank 末个真实 prefill 计算节点
        (barrier 前紧邻的 *_all_passes_logits_projection),非 end barrier
        节点(改前成员 == barrier,本用例必败)。"""
        members = self.builder.emit_prefill_batch(self.prefill_plan)
        self.assertEqual(sorted(members), sorted(PREFILL_RANKS))
        for rank in PREFILL_RANKS:
            barrier, last_compute = _last_barrier(
                self.builder, rank,
                "_prefill_chunks_aggregated_end_barrier")
            # 灯具核验:确为多 chunk prefill(end barrier 的 collective
            # payload = len(prefill_spans) = 300/128 -> 3 chunks)。
            self.assertEqual(barrier["coll"]["bytes"], CHUNKS)
            # (b) watch 成员 != end-barrier 节点(改前口径的病灶身份)。
            self.assertNotEqual(
                members[rank], barrier["id"],
                f"PREFILL_DRAIN member on rank {rank} is the end barrier")
            # (a) watch 成员 == barrier 前紧邻的真实计算节点。
            self.assertEqual(members[rank], last_compute["id"])
            self.assertEqual(last_compute["type"], COMP_NODE)
            self.assertTrue(
                last_compute["name"].endswith("_all_passes_logits_projection"),
                last_compute["name"])

    def test_prefill_trigger_gate_ledger_stays_post_barrier(self):
        """(c):_block_ends["seg1"](decode_evictions 触发门 node_gates 来源)
        保持 post-barrier 节点(== end-barrier 节点 id)——触发门角色不随
        watch 锚点统一而变化;decode 组 rank 本段无节点 -> None。"""
        self.builder.emit_prefill_batch(self.prefill_plan)
        seg1 = self.builder._block_ends[REQUEST_ID]["seg1"]
        for rank in PREFILL_RANKS:
            barrier, _ = _last_barrier(
                self.builder, rank,
                "_prefill_chunks_aggregated_end_barrier")
            self.assertEqual(seg1[rank], barrier["id"])
        for rank in DECODE_RANKS:
            # 段 1 的 decode 实例 = 准入占位(prefill 实例,调度器同款),
            # decode 组 rank 不入 seg1 账本(emitted-ranks-only)。
            self.assertIsNone(seg1.get(rank))

    def test_decode_completion_watch_members_are_pre_barrier_nodes(self):
        """decode 段同款 (a)+(b):DECODE_COMPLETION 成员 = end barrier 前
        每 rank 的 decode 末计算节点,非 *_decode_request_end_barrier 节点
        (改前成员 == barrier,本用例必败);(c) seg2 触发门账本仍
        post-barrier。"""
        self.builder.emit_prefill_batch(self.prefill_plan)
        members = self.builder.emit_decode_batch(self.decode_plan)
        self.assertEqual(sorted(members), sorted(DECODE_RANKS))
        for rank in DECODE_RANKS:
            barrier, last_compute = _last_barrier(
                self.builder, rank, "_decode_request_end_barrier")
            self.assertNotEqual(
                members[rank], barrier["id"],
                f"DECODE_COMPLETION member on rank {rank} is the end barrier")
            self.assertEqual(members[rank], last_compute["id"])
            self.assertEqual(last_compute["type"], COMP_NODE)
            self.assertTrue(
                last_compute["name"].endswith("_all_passes_logits_projection"),
                last_compute["name"])
        seg2 = self.builder._block_ends[REQUEST_ID]["seg2"]
        for rank in DECODE_RANKS:
            barrier, _ = _last_barrier(
                self.builder, rank, "_decode_request_end_barrier")
            self.assertEqual(seg2[rank], barrier["id"])


if __name__ == "__main__":
    unittest.main()
