#!/usr/bin/env python3
"""test_d10_silent_rebuild_execution.py -- D10 静默重建落执行图的构图器
消费面钉测（v2.0 §6.2，2026-09-27；joint 修复轮二任务 1）。

源：《joint修改文档》D10"构图器同步；不重复输出/采样/推进 RNG；首
token 只消费一次"。runtime/plan_dict/决策日志三方已携带
joint_suppress_sampling/joint_suppress_output（/）；本文件钉构图器（graph_batch_builder）对 rebuild span
（D 边界 recompute 的 (H_B, H_B) 静默重建 span）的输出侧抑制：

  (a) 重建 span 的算子时间计入列车（注意力/MLP/KV 分量照常聚合——
      静默重建的计算量真实存在）；
  (b) 无新增 first_token/debut 观测标记（debut first_token 标记不发、
      decode_length=1 的 exit 双锚点改名不发）；
  (c) final norm/logits 聚合的输出位不含重建 span（与"无重建 span 的
      同列车"逐字节相等；抑制关闭时差额 = H_B token 的闭式输出位）；
  (d) suppress 标志关闭（缺省 False/None）时行为与旧路径逐字节一致
      （对照断言：first_token 标记恢复、输出位含 H_B）；
  (e) 账本 fail-closed：声明抑制而 span 未进体 = 计划/发射布局失配。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 online/test_d10_silent_rebuild_execution.py
      （或 pytest 同路径）
"""
import json
import os
import sys
import unittest
from types import SimpleNamespace

_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _p in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from face_scheduler import KVTransfer  # noqa: E402
from generate_trace import (  # noqa: E402
    matmul_ops,
    tensor_bytes,
)
from online.graph_batch_builder import GraphBatchBuilder  # noqa: E402

SESSION = "session_d10_rebuild_0"
REBUILD_RID = f"{SESSION}_request_0"
PLAIN_RID = f"{SESSION}_request_1"
DECODE_RANKS = (2, 3)
H_B = 48                      # D 边界 recompute 的重建基数（历史 token）
CONTEXT = 200                 # 重建成员的 prefill 上下文终点
NORM_OPS_FACTOR = 8           # gelu 变体的 norm ops 系数（与灯具一致）


def _make_config():
    return SimpleNamespace(
        npus_count=4,
        remote_operand_loads=False,
        trace_granularity="request_aggregated",
        inference_groups=[
            SimpleNamespace(ranks=(0, 1), pg_name="tp_prefill"),
            SimpleNamespace(ranks=DECODE_RANKS, pg_name="tp_decode"),
        ],
        layers=2,
        hidden_size=64,
        ffn_size=128,
        vocab_size=256,
        bytes_per_elem=2,
        num_heads=8,
        mlp_variant="gelu",
        request_queue=[
            SimpleNamespace(
                session_arrival_time_ns=0,
                inter_request_interval_ns=None,
            ),
        ],
    )


def _local_hit(request_id):
    return KVTransfer(
        kind="local_hit",
        phase="prefill_to_decode",
        reason="d10_silent_rebuild_test",
        session_id=SESSION,
        trigger_request_id=request_id,
        source_instance_index=0,
        target_instance_index=1,
        total_bytes=0,
        shards=(),
        model_layers=2,
        layer_start=0,
        layer_end=2,
        resident_prefix_layers_before=2,
        resident_prefix_layers_after=2,
    )


def _joiner_plan(request_id, *, rebuild_tokens=0, suppress=False):
    """D 边界 recompute joiner 的 plan_dict 子集（sh30 runtime.plan_dict
    同源键；suppress=True = D 边界静默重建态，flags/stage/tokens 三键
    与  的置位口径一致）。"""
    plan = {
        "request_id": request_id,
        "session_id": SESSION,
        "turn_index": 0,
        "queue_index": 0,
        "prefill_instance_index": 0,
        "decode_instance_index": 1,
        "prefill_context_tokens": CONTEXT,
        "decode_evictions": [],
        "prefill_decode_transfer": _local_hit(request_id),
        "prefill_drain_block_ends": {rank: 0 for rank in (0, 1)},
        # D10 键恒在（前向兼容缺省值 = 非重算口径）。
        "joint_rebuild_context_tokens": 0,
        "joint_prefill_stage": None,
        "joint_suppress_sampling": False,
        "joint_suppress_output": False,
        "joint_decode_rebuild_tokens": 0,
        "joint_decode_rebuild_pending": False,
    }
    if rebuild_tokens > 0:
        plan.update({
            "joint_decode_rebuild_tokens": int(rebuild_tokens),
            "joint_decode_rebuild_pending": True,
            "joint_prefill_stage": (
                "context_rebuild" if suppress else "prefill_replay"),
            "joint_suppress_sampling": bool(suppress),
            "joint_suppress_output": bool(suppress),
        })
    return plan


def _rebuild_train_plan(train_id, *, suppress=True, with_rebuild=True,
                        first_token=None):
    """重建成员（debut，decode_length=3）列车：span 布局照 sh30
    _plan_train —— 重建 span (H_B, H_B) 前置于成员 decode span。"""
    decode_spans = [(1, CONTEXT + step) for step in (1, 2, 3)]
    spans = ([(H_B, H_B)] if with_rebuild else []) + decode_spans
    joiner = _joiner_plan(
        REBUILD_RID, rebuild_tokens=(H_B if with_rebuild else 0),
        suppress=suppress and with_rebuild)
    plan = {
        "train_id": train_id,
        "instance_index": 1,
        "stage": "decode",
        "joiners": [joiner],
        "pass_spans": spans,
        "iterations": len(decode_spans),
        "prefill_start_member": None,
        "first_chunk_member": None,
        "drain_members": [],
        "exit_members": [{"request_id": REBUILD_RID}],
    }
    if first_token is not None:
        plan["first_token"] = first_token
    return plan


def _rank_nodes(builder, rank):
    return [node for node in builder.batch["nodes"]
            if node["rank"] == rank]


def _body_nodes(builder, train_id, rank):
    """列车体节点（17 类聚合体：层循环类带 _all_layers_ 前缀、final
    norm/logits 带 _all_passes_ 前缀——输出位断言需两者）。"""
    return [node for node in _rank_nodes(builder, rank)
            if node["request_id"] == train_id
            and ("_all_layers_" in node["name"]
                 or "_all_passes_" in node["name"])]


def _category_ops(builder, train_id, rank, category):
    node = next(
        node for node in _body_nodes(builder, train_id, rank)
        if node["name"].endswith(category))
    return node["compute"]["num_ops"]


def _payload(builder):
    return json.dumps(
        {"nodes": builder.batch["nodes"],
         "parent_edges": builder.batch["parent_edges"]},
        separators=(",", ":"), sort_keys=True,
    )


class D10SilentRebuildExecutionTest(unittest.TestCase):
    """D10：构图器消费 joint_suppress_output/joint_suppress_sampling。"""

    def setUp(self) -> None:
        self.builder = GraphBatchBuilder(_make_config())
        self.builder.begin_batch()

    # ------------------------------------------------------------------
    # (a)+(c)：算子时间照常、输出位抑制
    # ------------------------------------------------------------------

    def test_rebuild_compute_counted_and_output_positions_suppressed(self):
        """(a)+(c)：重建 span 的注意力/MLP 算子计入列车体；final norm/
        logits 输出位与"无重建 span 的同列车"逐字节相等。"""
        suppressed = GraphBatchBuilder(_make_config())
        suppressed.begin_batch()
        suppressed.emit_iteration_train(
            _rebuild_train_plan("batch_train_i1_sup", suppress=True))
        reference = GraphBatchBuilder(_make_config())
        reference.begin_batch()
        reference.emit_iteration_train(
            _rebuild_train_plan("batch_train_i1_ref", with_rebuild=False))
        for rank in DECODE_RANKS:
            # (a) 注意力/MLP 算子：重建 H_B token 的前向计算真实存在
            # ——抑制列车的注意力算子严格大于无重建参考。
            attention_delta = (
                _category_ops(suppressed, "batch_train_i1_sup", rank,
                              "attention_qkv_projection")
                - _category_ops(reference, "batch_train_i1_ref", rank,
                                "attention_qkv_projection"))
            self.assertGreater(
                attention_delta, 0,
                "silent-rebuild span compute must stay aggregated")
            self.assertEqual(
                attention_delta,
                2 * matmul_ops(H_B, 3 * 32, 64),
                "rebuild span compute delta must equal the H_B-token "
                "closed form (layers=2, heads_per_rank=1, "
                "attention_hidden/rank=32)")
            # (c) 输出位：final norm/logits 与无重建参考逐字节相等。
            self.assertEqual(
                _category_ops(suppressed, "batch_train_i1_sup", rank,
                              "final_layernorm"),
                _category_ops(reference, "batch_train_i1_ref", rank,
                              "final_layernorm"),
                "final norm output positions must exclude the silent "
                "rebuild span")
            self.assertEqual(
                _category_ops(suppressed, "batch_train_i1_sup", rank,
                              "logits_projection"),
                _category_ops(reference, "batch_train_i1_ref", rank,
                              "logits_projection"),
                "logits output positions must exclude the silent rebuild "
                "span")

    def test_output_position_delta_closed_form_when_suppression_off(self):
        """(c)+(d) 对照锚：抑制关闭（P 重放口径 suppress=False）时输出
        位含 H_B —— 与抑制列车的差额 = H_B token 的闭式输出位（norm
        8·H·d + logits 2·H·V_rank）。"""
        suppressed = GraphBatchBuilder(_make_config())
        suppressed.begin_batch()
        suppressed.emit_iteration_train(
            _rebuild_train_plan("batch_train_i1_s", suppress=True))
        unsuppressed = GraphBatchBuilder(_make_config())
        unsuppressed.begin_batch()
        unsuppressed.emit_iteration_train(
            _rebuild_train_plan("batch_train_i1_u", suppress=False))
        for rank in DECODE_RANKS:
            norm_delta = (
                _category_ops(unsuppressed, "batch_train_i1_u", rank,
                              "final_layernorm")
                - _category_ops(suppressed, "batch_train_i1_s", rank,
                                "final_layernorm"))
            logits_delta = (
                _category_ops(unsuppressed, "batch_train_i1_u", rank,
                              "logits_projection")
                - _category_ops(suppressed, "batch_train_i1_s", rank,
                                "logits_projection"))
            self.assertEqual(
                norm_delta, NORM_OPS_FACTOR * H_B * 64,
                "unsuppressed rebuild span contributes its H_B-token "
                "final-norm output positions (closed form)")
            self.assertEqual(
                logits_delta, matmul_ops(H_B, 128, 64),
                "unsuppressed rebuild span contributes its H_B-token "
                "logits output positions (vocab/rank=128)")
            # 非输出位（注意力）不受抑制开关影响：两种口径逐字节相等。
            self.assertEqual(
                _category_ops(suppressed, "batch_train_i1_s", rank,
                              "attention_qkv_projection"),
                _category_ops(unsuppressed, "batch_train_i1_u", rank,
                              "attention_qkv_projection"))

    # ------------------------------------------------------------------
    # (b)：无新增 first_token/debut 观测
    # ------------------------------------------------------------------

    def test_no_first_token_markers_for_suppressed_member(self):
        """(b)：抑制成员不产生 debut first_token 标记；decode_length=1
        的 exit 双锚点改名同样不发（重建 span 不新增首 token 观测——
        首 token 归 P 末，恰一次）。"""
        first_token = {
            "split": False,
            "debut_marker_members": [{"request_id": REBUILD_RID}],
            "debut_exit_first_token": [REBUILD_RID],
        }
        result = self.builder.emit_iteration_train(
            _rebuild_train_plan(
                "batch_train_i1_ft", suppress=True,
                first_token=first_token))
        names = [node["name"] for node in self.builder.batch["nodes"]]
        self.assertFalse(
            any("first_token" in name for name in names),
            f"suppressed rebuild member must not add first_token "
            f"observation markers: {names}")
        # 抑制成员的 exit 标记名不带 first_token 双锚点（code 4 完成
        # watch 照常——完成非输出侧副作用）。
        for rank in DECODE_RANKS:
            marker = self.builder.batch["nodes"][
                result["exit_members"][REBUILD_RID][rank]]
            self.assertEqual(
                marker["name"], f"batch_train_i1_ft_exit_{REBUILD_RID}")
        self.assertEqual(
            sorted(result["exit_members"]), [REBUILD_RID])

    def test_suppression_off_restores_first_token_markers(self):
        """(d)：抑制关闭（旧路径）——debut first_token 标记照常发射，
        decode_length=1 的 exit 改名照常（对照断言）。"""
        first_token = {
            "split": False,
            "debut_marker_members": [{"request_id": REBUILD_RID}],
            "debut_exit_first_token": [REBUILD_RID],
        }
        result = self.builder.emit_iteration_train(
            _rebuild_train_plan(
                "batch_train_i1_ftu", suppress=False,
                first_token=first_token))
        names = [node["name"] for node in self.builder.batch["nodes"]]
        self.assertTrue(
            any("first_token" in name for name in names),
            f"unsuppressed debut member must keep its first_token "
            f"observation marker (legacy path): {names}")
        for rank in DECODE_RANKS:
            marker = self.builder.batch["nodes"][
                result["exit_members"][REBUILD_RID][rank]]
            self.assertEqual(
                marker["name"],
                f"batch_train_i1_ftu_exit_first_token_{REBUILD_RID}")

    # ------------------------------------------------------------------
    # (d)：关闭时与旧路径逐字节一致
    # ------------------------------------------------------------------

    def test_suppress_off_byte_identical_to_plan_without_d10_keys(self):
        """(d)：抑制关闭时（含 P 重放 rebuild_tokens>0 + suppress=False
        与完全无 D10 键两种姿态），发射产物与不含 D10 键的旧 plan 逐
        字节一致。"""
        legacy = GraphBatchBuilder(_make_config())
        legacy.begin_batch()
        legacy_plan = _rebuild_train_plan(
            "batch_train_i1_x", suppress=False, first_token={
                "split": False,
                "debut_marker_members": [{"request_id": REBUILD_RID}],
                "debut_exit_first_token": [],
            })
        # 旧路径 = plan 不携带任何 D10 键（forward-compatible 缺省）。
        joiner = dict(legacy_plan["joiners"][0])
        joiner.pop("joint_prefill_stage")
        joiner.pop("joint_suppress_sampling")
        joiner.pop("joint_suppress_output")
        joiner.pop("joint_decode_rebuild_tokens")
        joiner.pop("joint_decode_rebuild_pending")
        joiner.pop("joint_rebuild_context_tokens")
        legacy_plan["joiners"] = [joiner]
        legacy.emit_iteration_train(legacy_plan)
        modern = GraphBatchBuilder(_make_config())
        modern.begin_batch()
        modern.emit_iteration_train(
            _rebuild_train_plan(
                "batch_train_i1_x", suppress=False, first_token={
                    "split": False,
                    "debut_marker_members": [{"request_id": REBUILD_RID}],
                    "debut_exit_first_token": [],
                }))
        self.assertEqual(_payload(modern), _payload(legacy))

    def test_plain_train_without_rebuild_unaffected(self):
        """(d)：无重建成员的普通列车（无 D10 置位）发射不受账本影响
        （回归锚：与旧路径同形）。"""
        plan = _rebuild_train_plan(
            "batch_train_i1_plain", with_rebuild=False,
            first_token={
                "split": False,
                "debut_marker_members": [{"request_id": REBUILD_RID}],
                "debut_exit_first_token": [],
            })
        result = self.builder.emit_iteration_train(plan)
        names = [node["name"] for node in self.builder.batch["nodes"]]
        self.assertTrue(any("first_token" in name for name in names))
        self.assertEqual(self.builder._rebuild_output_debt, {})
        self.assertIn(REBUILD_RID, result["exit_members"])

    # ------------------------------------------------------------------
    # (e)：fail-closed
    # ------------------------------------------------------------------

    def test_ledger_fail_closed_when_rebuild_span_missing(self):
        """(e)：声明抑制（flags/stage/tokens 置位）而 (H_B, H_B) span
        未进体 = 计划/发射布局失配，fail-closed。"""
        plan = _rebuild_train_plan(
            "batch_train_i1_bad", suppress=True, with_rebuild=False)
        plan["joiners"][0]["joint_prefill_stage"] = "context_rebuild"
        plan["joiners"][0]["joint_suppress_sampling"] = True
        plan["joiners"][0]["joint_suppress_output"] = True
        plan["joiners"][0]["joint_decode_rebuild_tokens"] = H_B
        with self.assertRaisesRegex(RuntimeError, "silent-rebuild"):
            self.builder.emit_iteration_train(plan)

    def test_first_step_split_rejects_rebuild_joiner(self):
        """(e)：首步拆分列车禁静默重建 span（重建段属整列体的前置工作
        量——防御 fail-closed，与调度侧降级不拆车一致）。"""
        first_token = {
            "split": True,
            "first_spans": [(H_B, H_B), (1, CONTEXT + 1)],
            "rest_spans": [(1, CONTEXT + 2), (1, CONTEXT + 3)],
            "debut_marker_members": [{"request_id": REBUILD_RID}],
            "debut_exit_first_token": [],
            "wakeup_id": "batch_train_i1_split_first_step",
        }
        plan = _rebuild_train_plan(
            "batch_train_i1_split", suppress=True, first_token=first_token)
        with self.assertRaisesRegex(RuntimeError, "silent-rebuild"):
            self.builder.emit_train_first_step(plan)

    def test_output_bytes_conservation_across_suppression(self):
        """(c) 补充：抑制只动输出位——重建 span 的非输出字节（以
        attention_qkv 的 tensor_size 为代表）在两种口径下相等；输出
        字节差额 = H_B token 的闭式输出字节数（gelu: 3·H·d + H·V_rank
        的 norm 输入/输出与 logits 输出）。"""
        suppressed = GraphBatchBuilder(_make_config())
        suppressed.begin_batch()
        suppressed.emit_iteration_train(
            _rebuild_train_plan("batch_train_i1_bs", suppress=True))
        unsuppressed = GraphBatchBuilder(_make_config())
        unsuppressed.begin_batch()
        unsuppressed.emit_iteration_train(
            _rebuild_train_plan("batch_train_i1_bu", suppress=False))

        def category_bytes(builder, train_id, rank, category):
            node = next(
                node for node in _body_nodes(builder, train_id, rank)
                if node["name"].endswith(category))
            return node["compute"]["tensor_size"]

        rank = DECODE_RANKS[0]
        for category in ("attention_qkv_projection", "mlp_down_projection"):
            self.assertEqual(
                category_bytes(suppressed, "batch_train_i1_bs", rank,
                               category),
                category_bytes(unsuppressed, "batch_train_i1_bu", rank,
                               category),
                f"{category} bytes must not depend on the suppression "
                "switch")
        output_delta = (
            category_bytes(unsuppressed, "batch_train_i1_bu", rank,
                           "logits_projection")
            - category_bytes(suppressed, "batch_train_i1_bs", rank,
                             "logits_projection"))
        # logits_projection tensor_size 差额 = H_B 激活输入 + H_B 输出
        # 字节（权重分量两口径同含 weight_passes 份，抵消）。
        self.assertEqual(
            output_delta,
            tensor_bytes(H_B * 64, 2) + tensor_bytes(H_B * 128, 2))


class D10DegenerateShapeCollisionTest(unittest.TestCase):
    """D10 钉测轮（2026-09-28）：H_B == 首 prefill chunk（q_c）且零额外
    历史的退化同形碰撞。

    场景：混编列车队头 prefill 请求首 chunk = (q_c, q_c)（零额外历史 ⇒
    chunk span 上下文终点 = chunk token 数），同列车重建成员的静默重建
    span = (H_B, H_B) 且 H_B == q_c —— 两个 span **逐字节同形**。此形态
    下输出位总额无法区分"抑制了 chunk 还是抑制了重建 span"（同形 ⇒ 总
    量恒等），唯一的正确性锚 = 抑制**下标**合同：倒序贪心匹配必须消费
    成员区（重建 span），不得错消费队头首 chunk。若匹配退化为队头正序，
    重建 span 永不入账 ⇒ 列车尾 ``_close_rebuild_output_suppression``
    必 raise——本类两腿（构造器直测 + 端到端发射）对这一失效面双钉。
    """

    HB_EQ_QC = 32               # H_B == q_c 的退化取值
    MEMBER_CTX = 200

    # ------------------------------------------------------------------
    # sh30 _plan_train 布局不变量直测（重建 span 前置）
    # ------------------------------------------------------------------

    def _bare_scheduler(self):
        """最小真实调度器脚手架（test_train_machinery._bare_scheduler
        同款口径；只驱动 _plan_train，不跑仿真）。"""
        from online.sh30_online_scheduler import (
            Sh30OnlineScheduler,
            _OnlineInstanceState,
            _TASK_LOAD_CACHE_CAPACITY,
        )
        scheduler = Sh30OnlineScheduler.__new__(Sh30OnlineScheduler)
        scheduler.p_chunk = self.HB_EQ_QC
        scheduler.instances = [
            _OnlineInstanceState(index=i) for i in range(2)]
        scheduler.runtime_by_request_id = {}
        scheduler._train_max_iter = 0   # 0 = 不设限（自然边界）
        scheduler._train_instance_index = {}
        scheduler._ready_frontier = set()
        scheduler.topology = SimpleNamespace(
            instance=lambda i: SimpleNamespace(size=2))
        scheduler._prefill_task_cache = {}
        scheduler._decode_task_load_cache = {}
        scheduler._task_load_cache_capacity = _TASK_LOAD_CACHE_CAPACITY
        scheduler._quota_tracker = None
        scheduler.kv_manager = SimpleNamespace(_sessions={})
        return scheduler

    def _qp_head_runtime(self, scheduler):
        """队头 prefill：零额外历史（history_tokens_before=0）、单 chunk
        q_c = H_B —— 首 chunk span = (q_c, 0 + 0 + q_c) = (H_B, H_B)。"""
        from online.sh30_online_scheduler import (
            JOINT_STAGE_P_RUNNING,
            _OnlineRequestRuntime,
        )
        head = _OnlineRequestRuntime(
            {
                "request_id": "r_head_d10",
                "session_id": "session_d10_degenerate",
                "turn_index": 0,
                "queue_index": 0,
                "prefill_length": self.HB_EQ_QC,
                "decode_length": 4,
                "history_tokens_before": 0,
                "prefill_context_tokens": self.HB_EQ_QC,
                "final_context_tokens": self.HB_EQ_QC + 4,
            },
            self.HB_EQ_QC,
        )
        head.prefill_tokens_to_process = self.HB_EQ_QC
        head.remaining_chunks = 1
        head.joint_stage = JOINT_STAGE_P_RUNNING
        return head

    def _rebuild_member_runtime(self, scheduler):
        """D 边界 recompute 重建成员：joint_decode_rebuild_tokens = H_B
        （== q_c 的退化取值）、rebuild_pending 置位。"""
        from online.sh30_online_scheduler import (
            JOINT_STAGE_D_RUNNING,
            _OnlineRequestRuntime,
        )
        member = _OnlineRequestRuntime(
            {
                "request_id": REBUILD_RID,
                "session_id": "session_d10_degenerate",
                "turn_index": 0,
                "queue_index": 1,
                "prefill_length": self.MEMBER_CTX,
                "decode_length": 3,
                "history_tokens_before": 0,
                "prefill_context_tokens": self.MEMBER_CTX,
                "final_context_tokens": self.MEMBER_CTX + 3,
            },
            self.HB_EQ_QC,
        )
        member.joint_stage = JOINT_STAGE_D_RUNNING
        member.decode_instance_index = 0
        member.completion_instance_index = 0
        member.joint_decode_rebuild_tokens = self.HB_EQ_QC
        member.joint_decode_rebuild_pending = True
        scheduler.runtime_by_request_id[REBUILD_RID] = member
        return member

    def _planned_degenerate_train(self):
        """退化同形列车规划：真实 sh30 `_plan_train` 产出——队头单
        chunk (q_c, q_c) + 重建成员 (H_B, H_B)（H_B == q_c）。"""
        scheduler = self._bare_scheduler()
        head = self._qp_head_runtime(scheduler)
        member = self._rebuild_member_runtime(scheduler)
        state = scheduler.instances[0]
        state.qp.append(head)
        state.active_decode.append(member)
        plan = scheduler._plan_train(state)
        # 重建 span 只入首个含该成员的列车（规划即清位）。
        self.assertFalse(
            member.joint_decode_rebuild_pending,
            "the rebuild-pending flag must be cleared by planning (the "
            "rebuild span enters only the first train holding the member)")
        return plan

    def test_plan_train_rebuild_span_precedes_member_decode_spans(self):
        """_plan_train 不变量直测：chunk span 严格先于成员抑制 span，
        重建 span 前置于成员 decode span；同形碰撞下两者仍各居其位。"""
        plan = self._planned_degenerate_train()
        # 队头首 chunk（零额外历史）=(q_c, q_c)——与重建 span 同形。
        self.assertEqual(
            plan["prefill_chunk_spans"], [(self.HB_EQ_QC, self.HB_EQ_QC)])
        # 布局不变量：pass_spans = [chunk (H_B,H_B)] + [重建 (H_B,H_B)]
        # + [成员 decode span...]——chunk 严格先于成员抑制 span、重建
        # span 是成员首 span 且先于其 decode span。
        spans = plan["pass_spans"]
        self.assertEqual(
            spans[0], (self.HB_EQ_QC, self.HB_EQ_QC),
            "first chunk span must lead the train body")
        self.assertEqual(
            spans[1], (self.HB_EQ_QC, self.HB_EQ_QC),
            "the silent-rebuild span (H_B, H_B) with H_B == q_c must "
            "directly follow the chunk span (same-shape collision "
            "layout)")
        self.assertEqual(
            spans[2], (1, self.MEMBER_CTX + 1),
            "the member's decode spans must follow the rebuild span")
        self.assertLess(
            spans.index((self.HB_EQ_QC, self.HB_EQ_QC)),
            min(index for index, span in enumerate(spans) if span[0] == 1),
            "chunk + rebuild spans must strictly precede member decode "
            "spans")
        self.assertEqual(plan["rebuild_members"], [REBUILD_RID])

    # ------------------------------------------------------------------
    # 构图器消费腿：倒序匹配不错消费 + 列车尾 _close 无残留
    # ------------------------------------------------------------------

    def test_consume_reverse_match_takes_member_span_not_head_chunk(self):
        """同形碰撞下 _consume 倒序匹配：恰消费成员区重建 span（下标
        1），不错消费队头首 chunk（下标 0）；_close 无残留 raise。"""
        plan = self._planned_degenerate_train()
        builder = GraphBatchBuilder(_make_config())
        joiner = _joiner_plan(
            REBUILD_RID, rebuild_tokens=self.HB_EQ_QC, suppress=True)
        builder._open_rebuild_output_suppression({"joiners": [joiner]})
        self.assertEqual(
            builder._rebuild_output_debt,
            {(self.HB_EQ_QC, self.HB_EQ_QC): 1})
        indices = builder._consume_rebuild_output_suppression(plan["pass_spans"])
        self.assertEqual(
            indices, (1,),
            "reverse greedy matching must consume the member-zone "
            "rebuild span (index 1), never the shape-identical head "
            "chunk (index 0)")
        self.assertNotIn(
            0, indices,
            "the head first prefill chunk must not be mis-consumed as "
            "the suppression span (its output positions are real)")
        builder._close_rebuild_output_suppression()  # 无残留 ⇒ 不 raise
        self.assertEqual(builder._rebuild_output_debt, {})

    def test_degenerate_train_emits_end_to_end_without_close_raise(self):
        """端到端（真实 emit_iteration_train）：同形退化列车整列发射，
        列车尾 _close 无残留 raise——若匹配退化为队头正序（错消费首
        chunk），重建 span 将永不入账、close 必 fail-closed，本测试即
        该失效面的反证钉。"""
        plan = self._planned_degenerate_train()
        builder = GraphBatchBuilder(_make_config())
        builder.begin_batch()
        train_plan = {
            "train_id": "batch_train_i0_d10deg",
            "instance_index": 0,
            "stage": "decode",
            "joiners": [_joiner_plan(
                REBUILD_RID, rebuild_tokens=self.HB_EQ_QC, suppress=True)],
            "pass_spans": plan["pass_spans"],
            "iterations": plan["iterations"],
            "prefill_start_member": None,
            "first_chunk_member": None,
            "drain_members": [{"request_id": "r_head_d10"}],
            "exit_members": [],
        }
        builder.emit_iteration_train(train_plan)
        # 整列发射完成即含尾核销：残留账本必须已清（close 在列车尾被
        # 生产路径调用，无残留 raise）。
        self.assertEqual(
            builder._rebuild_output_debt, {},
            "train tail close must fully settle the same-shape rebuild "
            "debt (a forward-first matcher would leave the member span "
            "in debt and fail closed)")
        # 抑制成员不新增 first_token 观测（D10 (b) 在退化布局下仍成立）。
        names = [node["name"] for node in builder.batch["nodes"]]
        self.assertFalse(any("first_token" in name for name in names))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
