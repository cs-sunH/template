#!/usr/bin/env python3
"""test_relevant_kv_invariants.py -- relevant_distributed 发射层不变量测试
(总文档 §8.2 golden 三件套 + 扩展;执行文档 §5.2,2026-09-02 B2)。

钉住 relevant_kv_emission 三函数(1000/3100/3300)与 local_kv_bytes 覆盖
参数的核心不变量:
  (a) 3100 守恒:Σ3100 + P-piece 字节 = kv_cache_bytes_for_tokens(
      prefill_context)(函数内 fail-closed,此处正/反双向核);
  (b) 3300 字节 = Σ_m p_m × R_{m,s} 精确式(逐 rank 整头 shard × 迭代数);
  (c) pieces ⊆ path 实例集合(assert_pieces_on_path + 发射侧未知实例守卫);
  (d) 读边源集合 = pieces 非 D 实例集合(remote_read_sources);
  (e) tag 唯一性(同请求多源扩展位)与 < 100 上界 fail-closed;
  (f) snake_case hbm_charge 键写对、值极性正确、comm 键集合 ⊆ C++
      check_key_set 白名单(kebab 错拼在白名单外 = 被拒);
  (g) local_kv_bytes 覆盖后 k/v 各计覆盖值、缺省 None 与现状逐字节一致。

运行:cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 test_relevant_kv_invariants.py   （或 pytest 同路径）
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from types import SimpleNamespace

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))
_ONLINE_DIR = os.path.join(_MODULE_DIR, "online")
if str(_ONLINE_DIR) not in sys.path:
    sys.path.insert(0, str(_ONLINE_DIR))

from online.graph_batch_builder import (  # noqa: E402
    GraphBatchBuilder,
    OnlineTraceBuilder,
    _apply_hbm_charge,
)
from generate_trace import transformer_pass_aggregated  # noqa: E402
from generate_wsc_llm_trace import kv_cache_bytes_for_tokens  # noqa: E402
from relevant_kv_emission import (  # noqa: E402
    _stage_tag,
    assert_pieces_on_path,
    emit_history_pulls,
    emit_piece_scatter,
    emit_remote_reads,
    local_kv_override_value,
    remote_read_sources,
)

# 6 实例 × TP2 = 12 rank(mesh 6x2):0=P、1=D、2..5=中间 die——足够构造
# 多源(1 P + 3 die)读边与多源 1000 拉回。模型:512 B/token/实例,
# 逐 rank 整头 shard [256, 256](heads [4,4])。
PREFILL_INSTANCE = 0
DECODE_INSTANCE = 1
TP = 2
BYTES_PER_TOKEN = 512          # 2*layers*tokens*hidden*bytes_per_elem
BYTES_PER_RANK_TOKEN = 256     # 整头 shard:heads [4,4] × head_dim 8


def _make_config():
    groups = [
        SimpleNamespace(name=f"g{index}", ranks=(2 * index, 2 * index + 1),
                        pg_name=f"tp{index}")
        for index in range(6)
    ]
    return SimpleNamespace(
        npus_count=12,
        remote_operand_loads=False,
        trace_granularity="request_aggregated",
        inference_groups=groups,
        prefill_chunk_size=128,
        layers=2,
        hidden_size=64,
        ffn_size=128,
        vocab_size=256,
        bytes_per_elem=2,
        num_heads=8,
        mlp_variant="gelu",
        request_queue=[
            SimpleNamespace(session_arrival_time_ns=1000,
                            inter_request_interval_ns=None),
            SimpleNamespace(session_arrival_time_ns=None,
                            inter_request_interval_ns=1000),
        ],
        model=SimpleNamespace(layers=2, hidden_size=64, num_heads=8,
                              bytes_per_elem=2),
        hardware=SimpleNamespace(mesh_cols=2),
    )


def _piece(owner, start, end, tier="scatter_remote"):
    """KVPiece duck-typing(执行文档 §4.1 字段名;tier 仅元数据)。"""
    return SimpleNamespace(instance_index=owner, token_start=start,
                           token_end=end, tier=tier,
                           distance_to_decode=0, path=())


def _prefill_plan(queue_index=0, context=200):
    return {
        "request_id": f"s{queue_index}_r0",
        "session_id": f"s{queue_index}",
        "turn_index": 0,
        "queue_index": queue_index,
        "prefill_instance_index": PREFILL_INSTANCE,
        "decode_instance_index": DECODE_INSTANCE,
        "history_action": None,
        "history_source_instance_index": None,
        "history_transfer_bytes": 0,
        "history_recompute_tokens": 0,
        "history_tokens_before": 0,
        "prefill_length": context,
        "prefill_context_tokens": context,
    }


def _train_plan(train_id, spans, iterations, joiners=(), exits=()):
    return {
        "train_id": train_id,
        "instance_index": DECODE_INSTANCE,
        "joiners": list(joiners),
        "pass_spans": list(spans),
        "iterations": iterations,
        "exit_members": [{"request_id": rid, "session_id": f"s_{rid}"}
                         for rid in exits],
    }


def _emit_scatter_placement(graph, *, queue_index=0, context=200):
    """标准场景:R 在 P(ctx=200)prefill;pieces = D[0,80) + P[80,140) +
    die2[140,200) + D decode 段 [200,220)。返回 (plan, members, scatter)。"""
    plan = _prefill_plan(queue_index, context)
    members = graph.emit_prefill_batch(plan)
    pieces = [
        _piece(DECODE_INSTANCE, 0, 80),
        _piece(PREFILL_INSTANCE, 80, 140, "prefill_stay"),
        _piece(2, 140, 200),
        _piece(DECODE_INSTANCE, 200, 220, "decode_local"),
    ]
    scatter = emit_piece_scatter(
        graph, queue_index=queue_index, prefix=f"q{queue_index:04d}_pf",
        request_id=plan["request_id"],
        prefill_instance_index=PREFILL_INSTANCE, pieces=pieces,
        prefill_context_tokens=context, prefill_end_members=members)
    return plan, members, scatter


def _comm_nodes(graph, name_part):
    return [node for node in graph.batch["nodes"]
            if name_part in node["name"] and "comm" in node
            and node["type"] in (5, 6)]


class ScatterConservationTest(unittest.TestCase):
    """(a) 3100 守恒(总文档 §2.1/裁决 #9,做 golden)。"""

    def setUp(self) -> None:
        self.graph = GraphBatchBuilder(_make_config())
        self.graph.begin_batch()

    def test_scatter_plus_stay_equals_prefill_kv(self):
        _, _, scatter = _emit_scatter_placement(self.graph, context=200)
        # Σ3100(140 token 散布)+ P-piece(60 token)= kv(200)。
        self.assertEqual(scatter["scatter_bytes"], 140 * BYTES_PER_TOKEN)
        self.assertEqual(scatter["prefill_stay_bytes"], 60 * BYTES_PER_TOKEN)
        self.assertEqual(
            scatter["scatter_bytes"] + scatter["prefill_stay_bytes"],
            kv_cache_bytes_for_tokens(
                self.graph.config.model, 200))
        # 节点侧逐边字节与返回账一致(D 80 token、die 60 token;每边
        # send/recv 各携带同字节,按 send 端去重计一次)。
        node_bytes = sum(node["comm"]["bytes"]
                         for node in _comm_nodes(self.graph, "_kv_scatter_")
                         if "_send_" in node["name"])
        self.assertEqual(node_bytes, scatter["scatter_bytes"])
        self.assertEqual(scatter["scatter_tokens_by_owner"],
                         {DECODE_INSTANCE: 80, 2: 60})

    def test_all_stay_placement_emits_no_edges(self):
        plan = _prefill_plan(context=100)
        members = self.graph.emit_prefill_batch(plan)
        scatter = emit_piece_scatter(
            self.graph, queue_index=0, prefix="q0000_pf",
            request_id=plan["request_id"],
            prefill_instance_index=PREFILL_INSTANCE,
            pieces=[_piece(PREFILL_INSTANCE, 0, 100, "prefill_stay"),
                    _piece(DECODE_INSTANCE, 100, 110, "decode_local")],
            prefill_context_tokens=100, prefill_end_members=members)
        self.assertEqual(scatter["scatter_bytes"], 0)
        self.assertEqual(scatter["prefill_stay_bytes"], 100 * BYTES_PER_TOKEN)
        self.assertEqual(_comm_nodes(self.graph, "_kv_scatter_"), [])

    def test_coverage_gap_fails_closed(self):
        plan = _prefill_plan(context=200)
        members = self.graph.emit_prefill_batch(plan)
        # pieces 漏掉 [80, 140):Σ3100 + P-piece != kv(200) → fail-closed。
        with self.assertRaises(RuntimeError):
            emit_piece_scatter(
                self.graph, queue_index=0, prefix="q0000_pf",
                request_id=plan["request_id"],
                prefill_instance_index=PREFILL_INSTANCE,
                pieces=[_piece(DECODE_INSTANCE, 0, 80),
                        _piece(PREFILL_INSTANCE, 140, 200,
                               "prefill_stay")],
                prefill_context_tokens=200, prefill_end_members=members)

    def test_missing_prefill_end_anchor_fails_closed(self):
        plan = _prefill_plan(context=80)
        members = self.graph.emit_prefill_batch(plan)
        with self.assertRaises(RuntimeError):
            emit_piece_scatter(
                self.graph, queue_index=0, prefix="q0000_pf",
                request_id=plan["request_id"],
                prefill_instance_index=PREFILL_INSTANCE,
                pieces=[_piece(DECODE_INSTANCE, 0, 80)],
                prefill_context_tokens=80,
                prefill_end_members={0: members[0]})  # 缺 rank 1 锚


class RemoteReadBytesTest(unittest.TestCase):
    """(b) 3300 字节 = Σ_m p_m × R_{m,s} 精确式(总文档 §2.1/裁决 #7)。"""

    def setUp(self) -> None:
        self.graph = GraphBatchBuilder(_make_config())
        self.graph.begin_batch()
        self.plan, self.members, self.scatter = _emit_scatter_placement(
            self.graph)

    def _emit_train_with_reads(self, participation, sources_spec):
        plan = _train_plan(
            "batch_train_i1_1", [(1, 201)] * participation, participation,
            joiners=[_prefill_plan()], exits=(self.plan["request_id"],))
        result = self.graph.emit_iteration_train(plan)
        member = {
            "request_id": self.plan["request_id"],
            "queue_index": 0,
            "participation": participation,
            "prefill_instance_index": PREFILL_INSTANCE,
            "sources": [
                {"source_instance_index": index, "piece_tokens": tokens}
                for index, tokens in sources_spec
            ],
            "prefill_end_members": self.members,
            "exit_anchors": result["exit_members"][self.plan["request_id"]],
        }
        reads = emit_remote_reads(
            self.graph, train_id="batch_train_i1_1",
            decode_instance_index=DECODE_INSTANCE, member_reads=[member],
            scatter_recv_ids=self.scatter["recv_ids"])
        return result, reads

    def test_read_bytes_equal_participation_times_remote_pieces(self):
        # R_{m,s}:P-piece 60 token + die2-piece 60 token;p_m = 3。
        _, reads = self._emit_train_with_reads(
            3, [(PREFILL_INSTANCE, 60), (2, 60)])
        by_source = {}
        for route in reads["routes"]:
            by_source.setdefault(route["source_instance_index"],
                                 []).append(route["bytes"])
        # 逐 rank:p_m × R shard = 3 × 256 B/token × 60 token。
        expected = 3 * BYTES_PER_RANK_TOKEN * 60
        self.assertEqual(sorted(by_source[PREFILL_INSTANCE]),
                         [expected, expected])
        self.assertEqual(sorted(by_source[2]), [expected, expected])
        # Σ3300(全列车)= Σ_m p_m × R_m:单成员两源,总量 = 3×512×60×2。
        total = sum(route["bytes"] for route in reads["routes"])
        self.assertEqual(total, 3 * BYTES_PER_TOKEN * 60 * 2)

    def test_zero_remote_source_emits_no_edge(self):
        _, reads = self._emit_train_with_reads(2, [(PREFILL_INSTANCE, 0)])
        self.assertEqual(reads["routes"], [])
        self.assertEqual(_comm_nodes(self.graph, "_kv_remote_read_"), [])

    def test_decode_instance_source_fails_closed(self):
        with self.assertRaises(RuntimeError):
            self._emit_train_with_reads(
                2, [(DECODE_INSTANCE, 40)])   # D 不可为读边源(双计)


class PlacementInvariantHelperTest(unittest.TestCase):
    """(c)+(d) pieces ⊆ path 实例集合;读边源集合 = pieces 非 D 实例集合。"""

    def test_pieces_must_stay_on_path(self):
        pieces = [_piece(0, 0, 80), _piece(2, 80, 140),
                  _piece(1, 140, 220, "decode_local")]
        assert_pieces_on_path(pieces, (0, 1, 2))          # 通过
        with self.assertRaises(RuntimeError):
            assert_pieces_on_path(pieces, (0, 1))         # die2 不在 path

    def test_unknown_owner_instance_fails_closed(self):
        graph = GraphBatchBuilder(_make_config())
        graph.begin_batch()
        plan = _prefill_plan(context=80)
        members = graph.emit_prefill_batch(plan)
        with self.assertRaises(RuntimeError):
            emit_piece_scatter(
                graph, queue_index=0, prefix="q0000_pf",
                request_id=plan["request_id"],
                prefill_instance_index=PREFILL_INSTANCE,
                pieces=[_piece(9, 0, 80)],                # 实例 9 不存在
                prefill_context_tokens=80, prefill_end_members=members)

    def test_read_sources_are_non_decode_pieces(self):
        pieces = [
            _piece(DECODE_INSTANCE, 0, 80),               # D prefill 段
            _piece(PREFILL_INSTANCE, 80, 140, "prefill_stay"),
            _piece(2, 140, 200),
            _piece(DECODE_INSTANCE, 200, 220, "decode_local"),
        ]
        self.assertEqual(
            remote_read_sources(DECODE_INSTANCE, pieces),
            [(PREFILL_INSTANCE, 60), (2, 60)])

    def test_scatter_source_without_recv_anchor_fails_closed(self):
        graph = GraphBatchBuilder(_make_config())
        graph.begin_batch()
        _plan, members, _scatter = _emit_scatter_placement(graph)
        train = _train_plan("batch_train_i1_1", [(1, 201)], 1,
                            joiners=[_prefill_plan()])
        result = graph.emit_iteration_train(train)
        member = {
            "request_id": _plan["request_id"], "queue_index": 0,
            "participation": 1,
            "prefill_instance_index": PREFILL_INSTANCE,
            "sources": [{"source_instance_index": 3, "piece_tokens": 40}],
            "prefill_end_members": members,
            "exit_anchors": result["exit_members"].get(
                _plan["request_id"]),
        }
        # die3 从未散布(无 3100 recv 锚)→ fail-closed。
        with self.assertRaises(RuntimeError):
            emit_remote_reads(
                graph, train_id="batch_train_i1_1",
                decode_instance_index=DECODE_INSTANCE,
                member_reads=[member], scatter_recv_ids={},
                join_anchors=result["block_ends"])


class TagUniquenessTest(unittest.TestCase):
    """(e) tag 规则(总文档 §2.3):同请求多源扩展位唯一、整体 < 100。"""

    def setUp(self) -> None:
        self.graph = GraphBatchBuilder(_make_config())
        self.graph.begin_batch()
        self.plan, self.members, self.scatter = _emit_scatter_placement(
            self.graph)

    def test_multi_source_tags_unique_and_below_100(self):
        # 4 个散布 owner(D + die2/3/4)→ 3300 有 4 个源可用(P + 3 die
        # 不适用:本测试用 P + die2/3/4,源序号 0..3,扩展位 ordinal*2+
        # rank ∈ 0..7)。先铺一个 4 owner 的散布场景拿全部 3100 recv 锚。
        context = 280
        plan = _prefill_plan(context=context)
        members = self.graph.emit_prefill_batch(plan)
        pieces = [
            _piece(DECODE_INSTANCE, 0, 80),
            _piece(PREFILL_INSTANCE, 80, 140, "prefill_stay"),
            _piece(2, 140, 200),
            _piece(3, 200, 240),
            _piece(4, 240, 280),
        ]
        scatter = emit_piece_scatter(
            self.graph, queue_index=0, prefix="q0000_pf",
            request_id=plan["request_id"],
            prefill_instance_index=PREFILL_INSTANCE, pieces=pieces,
            prefill_context_tokens=context, prefill_end_members=members)
        train = _train_plan("batch_train_i1_1", [(1, 281), (1, 282)], 2,
                            joiners=[_prefill_plan(context=context)],
                            exits=(plan["request_id"],))
        result = self.graph.emit_iteration_train(train)
        member = {
            "request_id": plan["request_id"], "queue_index": 0,
            "participation": 2,
            "prefill_instance_index": PREFILL_INSTANCE,
            "sources": [{"source_instance_index": index,
                         "piece_tokens": 20}
                        for index in (PREFILL_INSTANCE, 2, 3, 4)],
            "prefill_end_members": members,
            "exit_anchors": result["exit_members"][plan["request_id"]],
        }
        emit_remote_reads(
            self.graph, train_id="batch_train_i1_1",
            decode_instance_index=DECODE_INSTANCE, member_reads=[member],
            scatter_recv_ids=scatter["recv_ids"])
        tags = [node["comm"]["tag"]
                for node in _comm_nodes(self.graph, "_kv_remote_read_")
                if "_send_" in node["name"]]
        self.assertEqual(len(tags), 8)               # 4 源 × TP2
        self.assertEqual(len(set(tags)), 8)          # 互不撞号
        for tag in tags:
            self.assertLess(tag % 10000 - 3300, 100)
            self.assertGreaterEqual(tag % 10000 - 3300, 0)

    def test_history_pull_multi_source_tags_unique(self):
        pull = emit_history_pulls(
            self.graph, queue_index=0, prefix="q0000_turn1",
            request_id="s0_r1",
            old_pieces=[_piece(DECODE_INSTANCE, 0, 80),
                        _piece(2, 80, 120),
                        _piece(3, 120, 160),
                        _piece(PREFILL_INSTANCE, 160, 200,
                               "prefill_stay")],
            new_prefill_instance_index=PREFILL_INSTANCE,
            history_tokens_before=200)
        self.assertEqual([source["source_ordinal"] for source
                          in pull["sources"]], [0, 1, 2])
        tags = [node["comm"]["tag"]
                for node in _comm_nodes(self.graph, "_kv_history_pull_")
                if "_send_" in node["name"]]
        self.assertEqual(len(tags), 6)               # 3 源 × TP2
        self.assertEqual(len(set(tags)), 6)

    def test_extension_overflow_fails_closed(self):
        # 源序号 50 × TP2 + rank1 = 101 ≥ 100 → fail-closed(上界 ≤ 53)。
        with self.assertRaises(ValueError):
            _stage_tag(0, 3300, 1, source_ordinal=50, tp_degree=TP)
        # 正常上界:8 源 × TP6 + rank5 = 53 < 100。
        self.assertEqual(
            _stage_tag(0, 3300, 5, source_ordinal=8, tp_degree=6),
            3300 + 53)


class HbmChargeKeyTest(unittest.TestCase):
    """(f) snake_case hbm_charge 键与极性(总文档 §2.3/裁决 #11/#35)。"""

    def setUp(self) -> None:
        self.graph = GraphBatchBuilder(_make_config())
        self.graph.begin_batch()
        self.plan, self.members, self.scatter = _emit_scatter_placement(
            self.graph)

    def test_charge_polarity_and_snake_case_key(self):
        plan = _train_plan("batch_train_i1_1", [(1, 201), (1, 202)], 2,
                           joiners=[_prefill_plan()],
                           exits=(self.plan["request_id"],))
        result = self.graph.emit_iteration_train(plan)
        emit_remote_reads(
            self.graph, train_id="batch_train_i1_1",
            decode_instance_index=DECODE_INSTANCE,
            member_reads=[{
                "request_id": self.plan["request_id"], "queue_index": 0,
                "participation": 2,
                "prefill_instance_index": PREFILL_INSTANCE,
                "sources": [{"source_instance_index": PREFILL_INSTANCE,
                             "piece_tokens": 60},
                            {"source_instance_index": 2,
                             "piece_tokens": 60}],
                "prefill_end_members": self.members,
                "exit_anchors": result["exit_members"][
                    self.plan["request_id"]],
            }],
            scatter_recv_ids=self.scatter["recv_ids"])
        scatter_nodes = _comm_nodes(self.graph, "_kv_scatter_")
        self.assertTrue(scatter_nodes)
        for node in scatter_nodes:
            self.assertIs(node["comm"]["hbm_charge"], True)
        read_sends = [node for node in _comm_nodes(
            self.graph, "_kv_remote_read_") if "_send_" in node["name"]]
        read_recvs = [node for node in _comm_nodes(
            self.graph, "_kv_remote_read_") if "_recv_" in node["name"]]
        self.assertTrue(read_sends and read_recvs)
        for node in read_sends:
            self.assertIs(node["comm"]["hbm_charge"], True)
        for node in read_recvs:
            self.assertIs(node["comm"]["hbm_charge"], False)  # 首个显式 false
        pull = emit_history_pulls(
            self.graph, queue_index=1, prefix="q0001_turn1",
            request_id="s1_r1",
            old_pieces=[_piece(DECODE_INSTANCE, 0, 80)],
            new_prefill_instance_index=PREFILL_INSTANCE,
            history_tokens_before=80)
        self.assertTrue(pull["routes"])
        for node in _comm_nodes(self.graph, "_kv_history_pull_"):
            self.assertIs(node["comm"]["hbm_charge"], True)
        # 所有 comm 节点的键集合 ⊆ C++ check_key_set 白名单
        # (ParsedGraphBatch.cc:243-244;kebab 错拼键不在白名单内 = 拒绝)。
        allowlist = {"bytes", "src", "dst", "tag", "hbm_charge"}
        for node in self.graph.batch["nodes"]:
            if node["type"] in (5, 6):
                self.assertLessEqual(set(node["comm"]), allowlist)
                self.assertNotIn("hbm-charge", node["comm"])

    def test_apply_hbm_charge_rejects_non_bool(self):
        node = {"comm": {}}
        _apply_hbm_charge(node, None, "where")     # None = 不写键
        self.assertNotIn("hbm_charge", node["comm"])
        _apply_hbm_charge(node, False, "where")
        self.assertIs(node["comm"]["hbm_charge"], False)
        with self.assertRaises(ValueError):
            _apply_hbm_charge({"comm": {}}, "true", "where")
        with self.assertRaises(ValueError):
            _apply_hbm_charge({"comm": {}}, 1, "where")


class LocalKvOverrideTest(unittest.TestCase):
    """(g) local_kv_bytes 覆盖(总文档 §3.3,防双计)。

    契约(M2 + local_kv_override_value 钉死):覆盖值 = 该 rank 该 span
    的**单层单份 K(=V)本地字节**;transformer_pass_aggregated 的 k/v_cache
    是逐层张量、聚合 ×layers,故体内贡献 = 值 ×2(K/V 各一次)×layers
    == D 本地 piece 的完整 KV shard 字节(缺省路径同理 == 全量 shard)。
    """

    MODEL = SimpleNamespace(layers=2, hidden_size=64, num_heads=8,
                            bytes_per_elem=2)

    @staticmethod
    def _run_pass(local_kv_bytes):
        builder = OnlineTraceBuilder(0, remote_operand_loads=False)
        transformer_pass_aggregated(
            builder,
            phase="aggregated",
            pass_spans=[(1, 101), (1, 41)],
            layers=2, hidden_size=64, ffn_size=128, tensor_parallel=TP,
            pg_name="tp0", vocab_size=256, bytes_per_elem=2,
            num_heads=8, tensor_parallel_rank=0, mlp_variant="gelu",
            local_kv_bytes=local_kv_bytes,
        )
        return builder

    def test_default_none_and_all_none_are_byte_identical(self):
        baseline = self._run_pass(None)
        explicit = self._run_pass([None, None])
        self.assertEqual(
            json.dumps([dict(node) for node in baseline.nodes],
                       sort_keys=True),
            json.dumps([dict(node) for node in explicit.nodes],
                       sort_keys=True))

    def test_override_counts_value_once_per_k_and_v(self):
        # rank0 全量 shard(101 token)= 101×256 = 25856 B;本地 60 token
        # 的覆盖值 = shard(60)/(2×layers) = 15360/4 = 3840(单层单份 K)。
        baseline = self._run_pass(None)
        overridden = self._run_pass(
            [local_kv_override_value(self.MODEL, 60, TP, 0), None])
        base_total = sum(node["compute"]["tensor_size"]
                         for node in baseline.nodes)
        over_total = sum(node["compute"]["tensor_size"]
                         for node in overridden.nodes)
        # 覆盖后该 span 的 KV 分量恰 = 本地 60 token 的 per-rank shard。
        self.assertEqual(base_total - over_total, 101 * 256 - 60 * 256)
        # ops 不受 KV 覆盖影响(纯归因参数)。
        self.assertEqual(
            sum(node["compute"]["num_ops"] for node in baseline.nodes),
            sum(node["compute"]["num_ops"] for node in overridden.nodes))

    def test_train_body_selects_per_rank_values(self):
        config = _make_config()
        graph = GraphBatchBuilder(config)
        graph.begin_batch()
        plain = _train_plan("batch_train_i1_1", [(1, 101), (1, 41)], 2)
        graph.emit_iteration_train(plain)

        override_graph = GraphBatchBuilder(config)
        override_graph.begin_batch()
        plan = _train_plan("batch_train_i1_2", [(1, 101), (1, 41)], 2)
        # span0:该成员 D 本地 60 token(逐 rank 覆盖值);span1:None =
        # 全量口径。
        plan["local_kv_bytes"] = [
            [local_kv_override_value(self.MODEL, 60, TP, relative_rank)
             for relative_rank in range(TP)],
            None,
        ]
        override_graph.emit_iteration_train(plan)

        def body_tensor(graph, train_id, rank):
            return sum(
                node["compute"]["tensor_size"]
                for node in graph.batch["nodes"]
                if node["request_id"] == train_id and node["rank"] == rank)

        for rank in (2, 3):  # D 实例 ranks(相对 rank 0/1 字节相同)
            self.assertEqual(
                body_tensor(override_graph, "batch_train_i1_2", rank),
                body_tensor(graph, "batch_train_i1_1", rank)
                - (101 * 256 - 60 * 256))

    def test_train_local_kv_misalignment_fails_closed(self):
        graph = GraphBatchBuilder(_make_config())
        graph.begin_batch()
        plan = _train_plan("batch_train_i1_1", [(1, 101), (1, 41)], 2)
        plan["local_kv_bytes"] = [[1024, 1024]]      # 长度与 pass_spans 不齐
        with self.assertRaises(RuntimeError):
            graph.emit_iteration_train(plan)


if __name__ == "__main__":
    unittest.main()
