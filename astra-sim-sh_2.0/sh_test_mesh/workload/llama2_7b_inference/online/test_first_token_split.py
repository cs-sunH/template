#!/usr/bin/env python3
"""test_first_token_split.py -- WP9 首 token 首步批拆分钉子测试(sh_2.0,
2026-08-26;母本 sh_1.0 test_first_token_split.py 同构 + S2 特有钉子)。

钉住拆分语义(SH_FIRST_TOKEN_SPLIT，B4 起缺省关；显式 "1" 开——
60s 决策等价门-2 失败后的主规格 §1.6 A 类处置，默认 proxy 口径，
证据 /tmp/slo_wps/b3/S2/ 与 /tmp/slo_wps/gates/B3_S2.FAILED):
  (a) debut 判定:joiner(decode_tokens_consumed==0)即 debut;开关关 /
      无 joiner → None(行为与拆分上线前逐字节一致);
  (b) 批结构:首步批 = joiner 迁移 + 起始标记 + 首迭代体(weight_passes=1)
      + first_token 标记 + 唤醒标记(批命名空间),无 drain/exit/哨兵/
      end barrier;余量批 = 余量体(weight_passes=iterations-1)+ 全部
      标记 + end barrier;两段权重/ops 合计 == 整列;
  (b') partial 恢复列车:前缀组(=首 chunk + 各成员第 1 span)恰为首步
      组,余量批只含纯聚合段;
  (c) T_max 截断:8 迭代列车拆 1+7;
  (d) decode_length=1 边界:debut 无独立标记,exit 标记改名含
      first_token 子串;iterations==1 不拆车(标记增强仍生效);
  (e) OFF 等价:SH_FIRST_TOKEN_SPLIT=0 时发射与无 first_token 计划的
      整列发射逐节点逐边一致;
  (f) 唤醒 no-op:调度器按 first_step id 识别并吞掉自己的唤醒信号;
  (g) S2 诊断字段落盘:admission 决策含 history_location_before 三态
      序列化;传输对象 noc 路由摘要(_transfer_hop_rows)。

运行:cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 online/test_first_token_split.py   （或 pytest 同路径）
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

from face_scheduler import KVTransfer, KVTransferShard  # noqa: E402
from online.graph_batch_builder import (  # noqa: E402
    GraphBatchBuilder,
    first_token_split_enabled,
)
from online.sh20_online_scheduler import (  # noqa: E402
    Sh20OnlineScheduler,
    _OnlineInstanceState,
    _transfer_hop_rows,
)

SESSION = "session_fts_0"
REQUEST_A = f"{SESSION}_request_0"
REQUEST_B = f"{SESSION}_request_1"
DECODE_RANKS = (2, 3)


def _make_config():
    return SimpleNamespace(
        npus_count=4,
        remote_operand_loads=False,
        trace_granularity="request_aggregated",
        inference_groups=[
            SimpleNamespace(ranks=(0, 1), pg_name="tp_prefill"),
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
        request_queue=[],
    )


def _joiner_plan(request_id, context_tokens, decode_length=4):
    return {
        "request_id": request_id,
        "session_id": SESSION,
        "turn_index": 0,
        "queue_index": 1,
        "decode_length": decode_length,
        "prefill_instance_index": 0,
        "decode_instance_index": 1,
        "prefill_context_tokens": context_tokens,
        "decode_evictions": [],
        "prefill_decode_transfer": {
            "kind": "local_hit",
            "phase": "prefill_to_decode",
            "reason": "train_test_same_tp_group",
            "session_id": SESSION,
            "trigger_request_id": request_id,
            "source_instance_index": 0,
            "target_instance_index": 1,
            "total_bytes": 0,
            "shards": [],
            "model_layers": 2,
            "layer_start": 0,
            "layer_end": 2,
            "resident_prefix_layers_before": 2,
            "resident_prefix_layers_after": 2,
        },
        "prefill_drain_block_ends": {rank: 0 for rank in (0, 1)},
    }


def _train_plan(train_id, spans, iterations, joiners=(), drains=(),
                exits=(), stage="decode", prefill_start=None,
                instance_index=1, first_token=None,
                partial_count=None):
    plan = {
        "train_id": train_id,
        "instance_index": instance_index,
        "stage": stage,
        "joiners": list(joiners),
        "pass_spans": list(spans),
        "iterations": iterations,
        "prefill_start_member": prefill_start,
        "drain_members": [{"request_id": rid} for rid in drains],
        "exit_members": [{"request_id": rid} for rid in exits],
    }
    if partial_count is not None:
        plan["partial_first_chunk_count"] = partial_count
    if first_token is not None:
        plan["first_token"] = first_token
    return plan


def _split_first_token(train_id, first_spans, rest_spans, iterations,
                       debut_markers=(), debut_exit=()):
    return {
        "split": True,
        "first_spans": list(first_spans),
        "rest_spans": list(rest_spans),
        "debut_marker_members": list(debut_markers),
        "debut_exit_first_token": list(debut_exit),
        "wakeup_id": f"{train_id}_first_step",
    }


def _nodes(builder, rank):
    return [node for node in builder.batch["nodes"]
            if node["rank"] == rank]


def _train_bytes(builder, train_id, ranks=DECODE_RANKS):
    total_ops = 0
    total_bytes = 0
    for rank in ranks:
        for node in _nodes(builder, rank):
            if node["request_id"] == train_id:
                total_ops += node["compute"]["num_ops"]
                total_bytes += node["compute"]["tensor_size"]
    return total_ops, total_bytes


def _bare_scheduler():
    s = Sh20OnlineScheduler.__new__(Sh20OnlineScheduler)
    s._train_instance_index = {}
    s._pending_first_steps = {}
    s.instances = [_OnlineInstanceState(i) for i in range(2)]
    s.runtimes = []
    return s


def _runtime(request_id, *, decode=4, consumed=0, ctx=100,
             location="local_hbm"):
    runtime = SimpleNamespace(
        request=SimpleNamespace(
            request_id=request_id, session_id=SESSION, turn_index=1,
            queue_index=1, prefill_length=128, decode_length=decode,
            inter_request_interval_ns=None),
        queue_index=1,
        admission_time_ns=1000,
        prefill_instance_index=0,
        decode_instance_index=1,
        prefill_affinity_reason="resident_local_hbm",
        history_transfer_bytes=0,
        history_source_instance_index=1,
        kv_location_after_completion=None,
        history_tokens_before=64,
        prefill_context_tokens=ctx,
        final_context_tokens=ctx + decode,
        remaining_chunks=0,
        prefill_tokens_to_process=128,
        decode_tokens_consumed=consumed,
        history_evictions=(),
        prefill_evictions=(),
        decode_evictions=(),
        completion_evictions=(),
        history_prefix_transfer=None,
        history_transfer=None,
        prefill_decode_transfer=None,
        history_location_before=(
            None if location is None else SimpleNamespace(
                location=location, instance_index=1,
                resident_prefix_layers=2)),
    )
    return runtime


def _plan_for_scheduler(iterations=8):
    return {
        "train_id": "batch_train_i1_9",
        "iterations": iterations,
        "members": [(0, 8)],
        "drain_members": [],
        "exit_members": [],
        "sentinel": False,
        "partial": False,
        "partial_first_chunk_count": None,
        "prefill_chunk_tokens": [(0, 128)] * iterations,
        "pass_spans": (
            [(128, 128 + 128 * (i + 1)) for i in range(iterations)]
            + [(1, 201 + i) for i in range(8)]),
    }


class SplitEnvironmentTest(unittest.TestCase):
    """开关姿态(B4 起缺省关;显式 "1" 开)。"""

    def test_default_off_and_explicit_on(self):
        try:
            os.environ.pop("SH_FIRST_TOKEN_SPLIT", None)
            # 2026-08-27 B4 缺省翻转:B3_S2 60s 门-2(决策等价)失败 →
            # 默认 proxy 口径,拆分仅显式 "1" 启用(研究/对拍)。
            self.assertFalse(first_token_split_enabled())
            os.environ["SH_FIRST_TOKEN_SPLIT"] = "0"
            self.assertFalse(first_token_split_enabled())
            os.environ["SH_FIRST_TOKEN_SPLIT"] = "1"
            self.assertTrue(first_token_split_enabled())
        finally:
            os.environ.pop("SH_FIRST_TOKEN_SPLIT", None)


class DebutPlanTest(unittest.TestCase):
    """(a)+(c)+(d)+(f) 调度器侧。

    B4 起拆分缺省关——本类钉拆分语义,setUp 显式置 "1"。"""

    def setUp(self):
        os.environ["SH_FIRST_TOKEN_SPLIT"] = "1"
        self.s = _bare_scheduler()
        self.s.runtimes = [_runtime(REQUEST_B)]

    def tearDown(self):
        os.environ.pop("SH_FIRST_TOKEN_SPLIT", None)

    def test_debut_is_fresh_joiner_only(self):
        token = self.s._first_token_plan(_plan_for_scheduler(), [0])
        self.assertIsNotNone(token)
        self.assertEqual(
            [m["request_id"] for m in token["debut_marker_members"]],
            [REQUEST_B])
        self.assertTrue(token["split"])

    def test_no_debut_returns_none(self):
        self.s.runtimes[0].decode_tokens_consumed = 3
        self.assertIsNone(self.s._first_token_plan(_plan_for_scheduler(), [0]))

    def test_disabled_returns_none(self):
        try:
            os.environ["SH_FIRST_TOKEN_SPLIT"] = "0"
            self.assertIsNone(
                self.s._first_token_plan(_plan_for_scheduler(), [0]))
        finally:
            os.environ.pop("SH_FIRST_TOKEN_SPLIT", None)

    def test_tmax_1_plus_7_split(self):
        """(c):8 迭代拆 1+7;两组合计与整列平铺多重集一致。"""
        plan = _plan_for_scheduler(iterations=8)
        token = self.s._first_token_plan(plan, [0])
        first, rest = token["first_spans"], token["rest_spans"]
        self.assertEqual(len(first), 2)          # 1 chunk + 1 member span
        self.assertEqual(len(rest), 14)
        self.assertEqual(sorted(first + rest), sorted(plan["pass_spans"]))
        self.assertTrue(token["split"])

    def test_partial_train_split_at_prefix_group(self):
        """(b'):partial 列车 layout = [首 chunk] + [成员第 1 span] +
        [其余];切点 = partial_first_chunk_count(=1+成员数)。"""
        plan = _plan_for_scheduler()
        plan["partial"] = True
        plan["partial_first_chunk_count"] = 2
        plan["pass_spans"] = [(128, 128), (1, 201), (128, 384), (1, 202)]
        token = self.s._first_token_plan(plan, [0])
        self.assertEqual(token["first_spans"], [(128, 128), (1, 201)])
        self.assertEqual(token["rest_spans"], [(128, 384), (1, 202)])

    def test_decode_length_1_debut_uses_exit_marker(self):
        self.s.runtimes[0].request.decode_length = 1
        token = self.s._first_token_plan(_plan_for_scheduler(), [0])
        self.assertEqual(token["debut_marker_members"], [])
        self.assertEqual(token["debut_exit_first_token"], [REQUEST_B])

    def test_single_iteration_enhances_without_split(self):
        token = self.s._first_token_plan(_plan_for_scheduler(iterations=1), [0])
        self.assertFalse(token["split"])
        self.assertEqual(
            [m["request_id"] for m in token["debut_marker_members"]],
            [REQUEST_B])

    def test_wakeup_consume_noop(self):
        """(f):唤醒 id 识别即吞;真哨兵/未知 id 不受影响。"""
        self.s._pending_first_steps["batch_train_i1_9_first_step"] = 1
        self.assertTrue(
            self.s._consume_first_step_wakeup("batch_train_i1_9_first_step"))
        self.assertEqual(self.s._pending_first_steps, {})
        self.s._train_instance_index["batch_train_i1_9"] = 1
        self.assertFalse(
            self.s._consume_first_step_wakeup("batch_train_i1_9"))


class SplitEmissionTest(unittest.TestCase):
    """(b)+(b')+(d)+(e) 构图器侧。"""

    def setUp(self):
        self.config = _make_config()

    def _new_builder(self):
        builder = GraphBatchBuilder(self.config)
        builder.begin_batch()
        return builder

    def _two_member_plan(self, first_token=None):
        joiners = [_joiner_plan(REQUEST_B, 40)]
        spans = [(1, 101), (1, 102), (1, 103), (1, 41), (1, 42)]
        return _train_plan(
            "batch_train_i1_1", spans, iterations=2, joiners=joiners,
            exits=(REQUEST_A,), prefill_start={"request_id": REQUEST_A},
            first_token=first_token)

    def test_split_batch_structure(self):
        """(b):首步批无标记/barrier;余量批挂全部标记与 end barrier;
        两段权重/ops 合计 == 整列。"""
        whole = self._new_builder()
        whole.emit_iteration_train(self._two_member_plan())
        whole_ops, whole_bytes = _train_bytes(whole, "batch_train_i1_1")

        split_first = _split_first_token(
            "batch_train_i1_1", [(1, 101), (1, 41)],
            [(1, 102), (1, 103), (1, 42)], 2,
            debut_markers=[{"request_id": REQUEST_B}])
        builder = self._new_builder()
        first = builder.emit_train_first_step(
            self._two_member_plan(split_first))
        first_nodes = _nodes(builder, DECODE_RANKS[0])
        first_names = [n["name"] for n in first_nodes]
        self.assertTrue(
            any("first_token" in n and REQUEST_B in n for n in first_names))
        self.assertTrue(
            any(n == "batch_train_i1_1_first_step_wakeup"
                for n in first_names))
        for banned in ("_drain_", "_exit_", "_sentinel", "_end_barrier"):
            self.assertFalse(
                any(banned in n for n in first_names),
                f"first-step batch must not carry {banned}: {first_names}")
        ft = next(n for n in first_nodes if "first_token" in n["name"])
        self.assertEqual(ft["request_id"], REQUEST_B)
        self.assertEqual(ft["stage"], "decode")
        self.assertEqual(sorted(first["wakeup_members"]), sorted(DECODE_RANKS))
        first_ops, first_bytes = _train_bytes(builder, "batch_train_i1_1")

        builder.begin_batch()
        result = builder.emit_train_remainder(
            self._two_member_plan(split_first))
        rest_nodes = _nodes(builder, DECODE_RANKS[0])
        rest_names = [n["name"] for n in rest_nodes]
        self.assertTrue(
            any(n.endswith(f"exit_{REQUEST_A}") for n in rest_names))
        self.assertTrue(
            any(n == "batch_train_i1_1_end_barrier" for n in rest_names))
        self.assertFalse(any("first_token" in n for n in rest_names))
        rest_ops, rest_bytes = _train_bytes(builder, "batch_train_i1_1")
        self.assertEqual(first_ops + rest_ops, whole_ops)
        self.assertEqual(first_bytes + rest_bytes, whole_bytes)

    def test_decode_length_1_exit_marker_carries_first_token(self):
        """(d):decode_length=1 debut 的 exit 标记名含 first_token 子串。"""
        joiners = [_joiner_plan(REQUEST_B, 40, decode_length=1)]
        token = _split_first_token(
            "batch_train_i1_2", [(1, 101), (1, 41)], [(1, 102)], 2,
            debut_exit=[REQUEST_B])
        plan = _train_plan(
            "batch_train_i1_2", [(1, 101), (1, 102), (1, 41)],
            iterations=2, joiners=joiners,
            exits=(REQUEST_A, REQUEST_B), first_token=token)
        builder = self._new_builder()
        builder.emit_train_first_step(plan)
        builder.begin_batch()
        result = builder.emit_train_remainder(plan)
        nodes = _nodes(builder, DECODE_RANKS[0])
        exit_b = next(n for n in nodes
                      if "exit" in n["name"] and n["request_id"] == REQUEST_B)
        self.assertIn("first_token", exit_b["name"])
        self.assertEqual(
            result["exit_members"][REQUEST_B][DECODE_RANKS[0]], exit_b["id"])
        exit_a = next(n for n in nodes
                      if "exit" in n["name"] and n["request_id"] == REQUEST_A)
        self.assertNotIn("first_token", exit_a["name"])

    def test_unsplit_enhancement_markers_after_body(self):
        """(d):不拆车时 first_token 标记挂列车体后、drain 标记前。"""
        token = {
            "split": False,
            "debut_marker_members": [{"request_id": REQUEST_B}],
            "debut_exit_first_token": [],
        }
        plan = _train_plan(
            "batch_train_i1_3", [(1, 101), (1, 41)], iterations=1,
            joiners=[_joiner_plan(REQUEST_B, 40)],
            drains=(REQUEST_A,), prefill_start={"request_id": REQUEST_A},
            first_token=token)
        builder = self._new_builder()
        builder.emit_iteration_train(plan)
        nodes = _nodes(builder, DECODE_RANKS[0])
        ft = next(n for n in nodes if "first_token" in n["name"])
        drain = next(n for n in nodes if n["name"].endswith(
            f"drain_{REQUEST_A}"))
        barrier = next(n for n in nodes
                       if n["name"] == "batch_train_i1_3_end_barrier")
        self.assertLess(ft["id"], drain["id"])
        self.assertLess(drain["id"], barrier["id"])

    def test_off_equivalence_byte_identical(self):
        """(e):OFF 发射与无 first_token 键的整列发射逐节点逐边一致。"""
        plan_plain = self._two_member_plan()
        plan_off = self._two_member_plan()
        builder_plain = self._new_builder()
        builder_plain.emit_iteration_train(plan_plain)
        builder_off = self._new_builder()
        try:
            os.environ["SH_FIRST_TOKEN_SPLIT"] = "0"
            self.assertIsNone(self.s_off_plan_check())
        finally:
            os.environ.pop("SH_FIRST_TOKEN_SPLIT", None)
        builder_off.emit_iteration_train(plan_off)
        self.assertEqual(
            builder_plain.batch["nodes"], builder_off.batch["nodes"])
        self.assertEqual(
            builder_plain.batch["parent_edges"],
            builder_off.batch["parent_edges"])

    @staticmethod
    def s_off_plan_check():
        s = _bare_scheduler()
        s.runtimes = [_runtime(REQUEST_B)]
        return s._first_token_plan(_plan_for_scheduler(), [0])


class S2DiagnosticsFieldTest(unittest.TestCase):
    """(g) S2 新增诊断字段存在性与取值。"""

    def _admit_decision_row(self, location):
        s = _bare_scheduler()
        s.runtimes = [_runtime(REQUEST_B, location=location)]
        s.graph = SimpleNamespace(
            emit_admission_batch=lambda plan: None)
        s._batch = {
            "delivery_sequence": 0, "kv_actions": [],
            "assignments": [],
        }
        s.online_log_rows = []
        s.online_log_count = 0
        s.decision_log_sink = None
        s._emitted_by_delivery = {0: {"tick": 0, "requests": []}}
        s.ledger_issued = {}
        s._emit_admission(s.runtimes[0], 0)
        self.assertEqual(len(s.online_log_rows), 1)
        return s.online_log_rows[0]["decision"]

    def test_admission_serializes_history_location_three_states(self):
        for location in ("local_hbm", "partial_hbm_remote",
                         "remote_memory"):
            decision = self._admit_decision_row(location)
            self.assertEqual(
                decision["history_location_before"], location)
            self.assertEqual(
                decision["history_location_before_instance_index"], 1)
            self.assertEqual(
                decision["history_resident_prefix_layers"], 2)

    def test_admission_serializes_none_location(self):
        decision = self._admit_decision_row(None)
        self.assertIsNone(decision["history_location_before"])
        self.assertIsNone(
            decision["history_location_before_instance_index"])

    def test_transfer_hop_rows(self):
        """noc 路由摘要:hop_bytes = Σ shard bytes × (len(noc_path)-1);
        local_hit(无 shards)零搬运。"""
        def shard(nbytes, path):
            return KVTransferShard(
                source_rank=2, target_rank=3, edge_rank=None,
                bytes=nbytes, noc_path=tuple(path),
                layer_start=0, layer_end=2)

        def transfer(kind, shards, total):
            return KVTransfer(
                kind=kind, phase="history", reason="r",
                session_id=SESSION, trigger_request_id=REQUEST_B,
                source_instance_index=0, target_instance_index=1,
                total_bytes=total, shards=tuple(shards),
                model_layers=2, layer_start=0, layer_end=2,
                resident_prefix_layers_before=2,
                resident_prefix_layers_after=2)

        rows = _transfer_hop_rows([
            transfer("noc_migrate",
                     [shard(100, (2, 5, 3)), shard(50, (2, 3))], 150),
            transfer("local_hit", [], 150),
        ])
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["noc_hop_bytes"], 100 * 2 + 50 * 1)
        self.assertEqual(rows[0]["shard_count"], 2)
        self.assertEqual(rows[0]["total_bytes"], 150)
        self.assertEqual(rows[1]["noc_hop_bytes"], 0)
        self.assertEqual(rows[1]["shard_count"], 0)


if __name__ == "__main__":
    unittest.main()
