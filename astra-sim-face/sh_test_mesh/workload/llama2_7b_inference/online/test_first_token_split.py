#!/usr/bin/env python3
"""test_first_token_split.py -- WP9 首 token 首步批拆分钉子测试(face,
2026-08-26;sh_1.0 母本同构,face 混拼列车/迁移原语适配)。

钉住拆分语义(SH_FIRST_TOKEN_SPLIT,B4 起缺省关;显式 "1" 开):
  (a) debut 判定:joiner(decode_tokens_consumed==0)即 debut;开关关 /
      无 joiner → None(行为与拆分上线前逐字节一致);
  (b) 批结构:首步批 = joiner 迁移 + join/pstart 起始标记 + 首迭代体
      (weight_passes=1)+ first_token 标记(名含 "first_token",携 debut
      request_id,stage=decode/generation=1)+ 唤醒标记(批命名空间),
      无 drain/exit/哨兵/end barrier;余量批 = 余量体(weight_passes=
      iterations-1)+ 全部标记 + end barrier(挂点语义不变,
      completion_gates 随余量批 end barrier 写入);两段权重/ops 字节
      合计 == 整列发射;
  (c) T_max 截断:8 迭代列车拆 1+7(首步 = 首 chunk + 各成员第 1 span,
      余量 = 7 chunk + 成员余量),两组合计与整列平铺一致;
  (d) decode_length=1 边界:debut 无独立标记,exit 标记改名含
      first_token 子串(同节点 code4/8 双锚点);iterations==1 不拆车
      (标记增强仍生效,标记挂列车体后、drain 标记前);
  (e) OFF 等价:SH_FIRST_TOKEN_SPLIT=0 时调度器不产 first_token 计划,
      构图器发射与无该键的整列发射逐节点逐边一致;
  (f) 唤醒 no-op:调度器按 first_step id 识别并吞掉自己的唤醒信号,
      真哨兵不受影响。

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

from online.graph_batch_builder import (  # noqa: E402
    GraphBatchBuilder,
    first_token_split_enabled,
)
from online.face_online_scheduler import (  # noqa: E402
    FaceOnlineScheduler,
    _OnlineInstanceState,
    _OnlineRequestRuntime,
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
            SimpleNamespace(name="instance_0", ranks=(0, 1),
                            pg_name="tp_prefill"),
            SimpleNamespace(name="instance_1", ranks=DECODE_RANKS,
                            pg_name="tp_decode"),
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
        model=SimpleNamespace(
            layers=2, hidden_size=64, num_heads=8, bytes_per_elem=2),
        hardware=SimpleNamespace(mesh_rows=1, mesh_cols=4),
    )


def _joiner_plan(request_id, context_tokens, decode_length=4):
    return {
        "request_id": request_id,
        "session_id": SESSION,
        "turn_index": 0,
        "queue_index": 0,
        "prefill_instance_index": 0,
        "decode_instance_index": 1,
        "prefill_context_tokens": context_tokens,
    }


def _train_plan(train_id, spans, iterations, joiners=(), drains=(),
                exits=(), stage="decode", prefill_start=None,
                instance_index=1, sentinel=False, first_token=None):
    plan = {
        "train_id": train_id,
        "instance_index": instance_index,
        "stage": stage,
        "joiners": list(joiners),
        "pass_spans": list(spans),
        "iterations": iterations,
        "prefill_start_member": prefill_start,
        "sentinel": sentinel,
        "drain_members": [{"request_id": rid, "session_id": SESSION}
                          for rid in drains],
        "exit_members": [{"request_id": rid, "session_id": SESSION}
                         for rid in exits],
    }
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
    """列车归属节点(train_id 命名空间)的 compute ops/字节合计。"""
    total_ops = 0
    total_bytes = 0
    for rank in ranks:
        for node in _nodes(builder, rank):
            if node["request_id"] == train_id:
                total_ops += node["compute"]["num_ops"]
                total_bytes += node["compute"]["tensor_size"]
    return total_ops, total_bytes


def _runtime(request_id, *, ctx=100, decode=4, consumed=0):
    record = {
        "request_id": request_id,
        "session_id": SESSION,
        "turn_index": 0,
        "queue_index": 0,
        "prefill_length": 128,
        "decode_length": decode,
        "history_tokens_before": 0,
        "prefill_context_tokens": ctx,
        "final_context_tokens": ctx + decode,
    }
    runtime = _OnlineRequestRuntime(record)
    runtime.decode_tokens_consumed = consumed
    return runtime


def _bare_scheduler():
    scheduler = FaceOnlineScheduler.__new__(FaceOnlineScheduler)
    scheduler.instances = [_OnlineInstanceState(index=index)
                           for index in range(2)]
    scheduler.runtime_by_request_id = {}
    scheduler._train_instance_index = {}
    scheduler._pending_first_steps = {}
    return scheduler


class SplitEnvironmentTest(unittest.TestCase):
    """开关姿态(B4 起缺省关;显式 "1" 开)。"""

    def test_default_off_and_explicit_on(self):
        try:
            os.environ.pop("SH_FIRST_TOKEN_SPLIT", None)
            # 2026-08-26 B4 缺省翻转:60s 门-2(决策等价)失败 → 默认
            # proxy 口径,拆分仅显式 "1" 启用(研究/对拍)。
            self.assertFalse(first_token_split_enabled())
            os.environ["SH_FIRST_TOKEN_SPLIT"] = "0"
            self.assertFalse(first_token_split_enabled())
            os.environ["SH_FIRST_TOKEN_SPLIT"] = "1"
            self.assertTrue(first_token_split_enabled())
        finally:
            os.environ.pop("SH_FIRST_TOKEN_SPLIT", None)


class DebutPlanTest(unittest.TestCase):
    """(a)+(c)+(d) 调度器侧:debut 判定与 span 切分。

    B4 起拆分缺省关——本类钉拆分语义,setUp 显式置 "1"。"""

    def setUp(self):
        os.environ["SH_FIRST_TOKEN_SPLIT"] = "1"
        self.s = _bare_scheduler()

    def tearDown(self):
        os.environ.pop("SH_FIRST_TOKEN_SPLIT", None)

    def _plan(self, iterations=8, spans=None, members=None, chunks=None):
        members = members if members is not None else [(REQUEST_A, 8)]
        chunks = chunks if chunks is not None else [
            (REQUEST_A, 128)] * iterations
        return {
            "train_id": "batch_train_i1_9",
            "iterations": iterations,
            "members": members,
            "drain_members": [],
            "exit_members": [],
            "sentinel": False,
            "prefill_chunk_tokens": chunks,
            "pass_spans": spans if spans is not None else (
                [(128, 128 + 128 * (i + 1)) for i in range(iterations)]
                + [(1, 201 + i) for i in range(8)]),
        }

    def test_debut_is_fresh_joiner_only(self):
        fresh = _runtime(REQUEST_B, consumed=0)
        veteran = _runtime(REQUEST_A, consumed=3)
        self.s.runtime_by_request_id = {
            REQUEST_B: fresh, REQUEST_A: veteran}
        token = self.s._first_token_plan(
            self._plan(), [veteran, fresh])
        self.assertIsNotNone(token)
        self.assertEqual(
            [m["request_id"] for m in token["debut_marker_members"]],
            [REQUEST_B])
        self.assertTrue(token["split"])

    def test_no_debut_returns_none(self):
        veteran = _runtime(REQUEST_A, consumed=3)
        self.s.runtime_by_request_id = {REQUEST_A: veteran}
        self.assertIsNone(
            self.s._first_token_plan(self._plan(), [veteran]))

    def test_disabled_returns_none(self):
        fresh = _runtime(REQUEST_B, consumed=0)
        self.s.runtime_by_request_id = {REQUEST_B: fresh}
        try:
            os.environ["SH_FIRST_TOKEN_SPLIT"] = "0"
            self.assertIsNone(
                self.s._first_token_plan(self._plan(), [fresh]))
        finally:
            os.environ.pop("SH_FIRST_TOKEN_SPLIT", None)

    def test_tmax_1_plus_7_split(self):
        """(c):8 迭代(T_max=8 截断同构)拆 1+7——首步 = 首 chunk + 各
        成员第 1 span;余量 = 7 chunk + 成员剩余 span;总量守恒。"""
        fresh = _runtime(REQUEST_B, consumed=0)
        self.s.runtime_by_request_id = {REQUEST_B: fresh}
        plan = self._plan(iterations=8)
        token = self.s._first_token_plan(plan, [fresh])
        first, rest = token["first_spans"], token["rest_spans"]
        self.assertEqual(len(first), 2)          # 1 chunk + 1 member span
        self.assertEqual(first[0], plan["pass_spans"][0])
        self.assertEqual(first[1], plan["pass_spans"][8])
        self.assertEqual(len(rest), 14)          # 7 chunks + 7 member spans
        # 聚合对 span 求和与顺序无关:两组合计与整列平铺多重集一致。
        self.assertEqual(
            sorted(first + rest), sorted(plan["pass_spans"]))
        self.assertEqual(len(first) + len(rest), len(plan["pass_spans"]))
        self.assertTrue(token["split"])

    def test_decode_length_1_debut_uses_exit_marker(self):
        """(d):decode_length=1 debut → 无独立标记,进 exit 改名表。"""
        solo = _runtime(REQUEST_B, decode=1)
        self.s.runtime_by_request_id = {REQUEST_B: solo}
        token = self.s._first_token_plan(self._plan(), [solo])
        self.assertEqual(token["debut_marker_members"], [])
        self.assertEqual(token["debut_exit_first_token"], [REQUEST_B])

    def test_single_iteration_enhances_without_split(self):
        """(d):iterations==1(如 T_max=1 oracle 档)不拆车,标记增强生效。"""
        fresh = _runtime(REQUEST_B, decode=4)
        self.s.runtime_by_request_id = {REQUEST_B: fresh}
        plan = self._plan(iterations=1)
        token = self.s._first_token_plan(plan, [fresh])
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
        self.assertIn("batch_train_i1_9", self.s._train_instance_index)


class SplitEmissionTest(unittest.TestCase):
    """(b)+(e) 构图器侧:批结构与 OFF 等价。"""

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
        """(b):首步批无标记/barrier,含 first_token+唤醒标记;余量批挂
        全部标记与 end barrier,completion_gates 随余量批写入;两段
        权重/ops 合计 == 整列。"""
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
        # 首步批:join 标记 + 体 + first_token 标记 + 唤醒标记。
        self.assertTrue(
            any(n.endswith(f"join_{REQUEST_B}") for n in first_names))
        self.assertTrue(
            any("first_token" in n and REQUEST_B in n for n in first_names))
        self.assertTrue(
            any(n == "batch_train_i1_1_first_step_wakeup"
                for n in first_names))
        self.assertTrue(
            any(n["request_id"] == "batch_train_i1_1_first_step"
                for n in first_nodes))
        # 首步批禁物:drain/exit/哨兵/end barrier 与任何 watch 标记。
        for banned in ("_drain_", "_exit_", "_sentinel", "_end_barrier"):
            self.assertFalse(
                any(banned in n for n in first_names),
                f"first-step batch must not carry {banned}: {first_names}")
        # first_token 标记携带 debut 成员上下文(decode, 1)。
        ft = next(n for n in first_nodes if "first_token" in n["name"])
        self.assertEqual(ft["request_id"], REQUEST_B)
        self.assertEqual(ft["stage"], "decode")
        self.assertEqual(ft["compute"]["num_ops"], 1)
        self.assertEqual(sorted(first["wakeup_members"]),
                         sorted(DECODE_RANKS))
        self.assertEqual(
            sorted(first["first_token_members"][REQUEST_B]),
            sorted(DECODE_RANKS))
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
        self.assertFalse(
            any("first_token" in n for n in rest_names))
        self.assertEqual(
            result["exit_members"][REQUEST_A][DECODE_RANKS[0]],
            next(n["id"] for n in rest_nodes
                 if n["name"].endswith(f"exit_{REQUEST_A}")))
        # completion_gates = exit 成员所在(余量批)end barrier。
        self.assertEqual(
            builder.completion_gates[SESSION][1][DECODE_RANKS[0]],
            result["block_ends"][DECODE_RANKS[0]])
        rest_ops, rest_bytes = _train_bytes(builder, "batch_train_i1_1")
        # (b) 权重/ops 守恒:1 + (2-1) 恰合回整列的 2 次权重读取。
        self.assertEqual(first_ops + rest_ops, whole_ops)
        self.assertEqual(first_bytes + rest_bytes, whole_bytes)

    def test_decode_length_1_exit_marker_carries_first_token(self):
        """(d):decode_length=1 debut 的 exit 标记名含 first_token 子串,
        且仍是 watch 成员节点(decode, generation 1)。"""
        joiners = [_joiner_plan(REQUEST_B, 40, decode_length=1)]
        spans = [(1, 101), (1, 102), (1, 41)]
        token = _split_first_token(
            "batch_train_i1_2", [(1, 101), (1, 41)], [(1, 102)], 2,
            debut_exit=[REQUEST_B])
        plan = _train_plan(
            "batch_train_i1_2", spans, iterations=2, joiners=joiners,
            exits=(REQUEST_A, REQUEST_B),
            first_token=token)
        builder = self._new_builder()
        builder.emit_train_first_step(plan)
        builder.begin_batch()
        result = builder.emit_train_remainder(plan)
        nodes = _nodes(builder, DECODE_RANKS[0])
        exit_b = next(n for n in nodes
                      if "exit" in n["name"] and n["request_id"] == REQUEST_B)
        self.assertIn("first_token", exit_b["name"])
        self.assertEqual(
            result["exit_members"][REQUEST_B][DECODE_RANKS[0]],
            exit_b["id"])
        exit_a = next(n for n in nodes
                      if "exit" in n["name"] and n["request_id"] == REQUEST_A)
        self.assertNotIn("first_token", exit_a["name"])

    def test_unsplit_enhancement_markers_after_body(self):
        """(d):不拆车(iterations==1)时 first_token 标记挂列车体后、
        drain 标记前。"""
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
        self.assertEqual(ft["request_id"], REQUEST_B)

    def test_off_equivalence_byte_identical(self):
        """(e):SH_FIRST_TOKEN_SPLIT=0 时调度器不产 first_token 计划,
        构图器发射与无该键的整列发射逐节点逐边一致。"""
        plan_on = self._two_member_plan()
        builder_off = self._new_builder()
        try:
            os.environ["SH_FIRST_TOKEN_SPLIT"] = "0"
            s = _bare_scheduler()
            fresh = _runtime(REQUEST_B)
            s.runtime_by_request_id = {REQUEST_B: fresh}
            self.assertIsNone(
                s._first_token_plan({"train_id": "t", "iterations": 4,
                                     "members": [], "prefill_chunk_tokens":
                                     []}, [fresh]))
            plan_off = self._two_member_plan()
            self.assertNotIn("first_token", plan_off)
        finally:
            os.environ.pop("SH_FIRST_TOKEN_SPLIT", None)
        builder_plain = self._new_builder()
        builder_plain.emit_iteration_train(plan_on)
        builder_off.emit_iteration_train(plan_off)
        self.assertEqual(
            builder_plain.batch["nodes"], builder_off.batch["nodes"])
        self.assertEqual(
            builder_plain.batch["parent_edges"],
            builder_off.batch["parent_edges"])


if __name__ == "__main__":
    unittest.main()
