#!/usr/bin/env python3
"""test_wsc_llm_relevant_online_scheduler.py -- relevant_distributed 第三变体
混合骨架在线调度器单元测试(WscLlmRelevantOnlineScheduler;总文档 §8.4,
执行文档 §5.3,2026-09-02 B3)。

合成 delta 流直接驱动 on_decision_batch(不经 C++/bridge,test_wsc_llm_
legacy_online_scheduler.py 同款基建),覆盖:

  - 构造器 fail-closed:错 kv_cache_policy / mode / --sensing 拒绝;
  - 背压→释放→重准入:try_place None → FCFS 队头阻塞(空批重查仍阻塞)
    → 终轮完成释放 → 重查通过 + "背压持续时长"观测字段;
  - 多轮:释放旧 placement → 1000 族多源拉回(源=P 零边 LOCAL_HIT)
    → 连续编址重放置(新 KVPlacement)→ 再释放;canonical hit_state
    映射断言(全源=P → full;否则 partial;永不 miss);
  - 列车发射序 per-rank 全局序不反转(R4 调度器侧):同 tick 完成批先于
    准入批——drain 的 3100 send 与下一准入 prefill 的 per-rank 节点 id
    严格递增(B2 已钉图侧 frontier 前提,此处钉调度器侧交付序);
  - run-end 审计:placements/heap/frontier/gates 全空 + journal 守恒
    (resident=0/reserved=0/physical=weight,纯重放门);
  - ideal_masked 分支:不发 3300、不裁 KV 分量(与 physical 构成严格
    A/B:physical 裁体 + 有读边;ideal 全量体 + 无读边;3100 不受开关
    影响)。

请求规模动态推导(legacy 测试同款纪律):checked-in 配置的 request_
queue_csv 为占位路径,测试用手写合成队列 + 从 trace_config_legacy.csv
派生(kv_cache_policy 覆写为 relevant_distributed);token 数从 probe
调度器的 B1 分配器逐 rank 账本实测推导,避免硬编码越界。

运行:cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 online/test_wsc_llm_relevant_online_scheduler.py
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _p in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from generate_wsc_llm_trace import (  # noqa: E402
    kv_cache_bytes_for_tokens,
    load_wsc_llm_trace_config,
)
from metrics_integration import MemoryActionRecorder  # noqa: E402
from metrics_schema import MemoryMetricsObserver  # noqa: E402
from online.graph_batch_builder import GraphBatchBuilder  # noqa: E402
from online.wsc_llm_relevant_online_scheduler import (  # noqa: E402
    WscLlmRelevantOnlineScheduler,
    _canonical_hit_state,
    verify_relevant_journal_conservation,
)
from session_kv_manager import kv_cache_shard_bytes_for_tokens  # noqa: E402
from wsc_llm_scheduler import PREFILL_ROLE  # noqa: E402

# --------------------------------------------------------------------------
# request-neutral 合成 fixture(legacy 测试同款):trace_config_legacy.csv
# 派生 + 手写合成队列 + kv_cache_policy 覆写。__file__ 锚定(相对路径随
# pytest 调用根失效)。
# --------------------------------------------------------------------------

_FIXTURE_DIR = tempfile.TemporaryDirectory(prefix="wscllm_relevant_fixture_")
_FIXTURE_HEADER = (
    "session_id,turn_index,request_id,prefill_length,decode_length,"
    "session_arrival_time_ns,inter_request_interval_ns,description"
)
# 8 个 turn-0 槽位 + 2 个 turn>0 槽位(_pick_slots 与 legacy 测试同构)。
_FIXTURE_ROWS = (
    ("fixture_s0", "0", "fixture_s0_r0", "100", "5", "0", "", "synthetic fixture"),
    ("fixture_s1", "0", "fixture_s1_r0", "200", "6", "0", "", "synthetic fixture"),
    ("fixture_s2", "0", "fixture_s2_r0", "300", "7", "0", "", "synthetic fixture"),
    ("fixture_s3", "0", "fixture_s3_r0", "400", "8", "0", "", "synthetic fixture"),
    ("fixture_s4", "0", "fixture_s4_r0", "500", "9", "0", "", "synthetic fixture"),
    ("fixture_s5", "0", "fixture_s5_r0", "600", "10", "0", "", "synthetic fixture"),
    ("fixture_s6", "0", "fixture_s6_r0", "700", "11", "0", "", "synthetic fixture"),
    ("fixture_s7", "0", "fixture_s7_r0", "800", "12", "0", "", "synthetic fixture"),
    ("fixture_s0", "1", "fixture_s0_r1", "50", "4", "", "1000", "synthetic fixture"),
    ("fixture_s1", "1", "fixture_s1_r1", "60", "5", "", "2000", "synthetic fixture"),
)


def _load_fixture_config(kv_remote_read: str = "physical"):
    """合成 fixture config:legacy checked-in 配置 + 合成队列 + policy 覆写
    (kv_cache_policy=relevant_distributed;kv_remote_read 按 A/B 场景)。"""

    config_source = Path(_WORKLOAD_DIR) / "trace_config_legacy.csv"
    queue_path = Path(_FIXTURE_DIR.name) / "synthetic_request_queue.csv"
    if not queue_path.exists():
        queue_path.write_text(
            _FIXTURE_HEADER + "\n"
            + "\n".join(",".join(row) for row in _FIXTURE_ROWS) + "\n",
            encoding="utf-8")
    config_path = (
        Path(_FIXTURE_DIR.name)
        / f"trace_config_relevant_{kv_remote_read}.csv")
    if not config_path.exists():
        lines = []
        for raw_line in config_source.read_text(encoding="utf-8").splitlines():
            if raw_line.startswith("config,request_queue_csv,"):
                lines.append(
                    "config,request_queue_csv," + str(queue_path)
                    + ",,,,,request-neutral synthetic fixture queue (unit test)")
            elif raw_line.startswith("config,kv_cache_policy,"):
                lines.append(
                    "config,kv_cache_policy,relevant_distributed,,,,,"
                    "relevant_distributed third variant (unit test fixture)")
            else:
                lines.append(raw_line)
        lines.append(
            f"config,kv_remote_read,{kv_remote_read},,,,,"
            "A/B switch for the relevant fixture (unit test)")
        config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return load_wsc_llm_trace_config(config_path)


def _request_record(queue_index, session_id, turn_index, request_id,
                    history_tokens_before, prefill_context_tokens,
                    final_context_tokens, prefill_length=100,
                    decode_length=5):
    return {
        "queue_index": queue_index,
        "session_id": session_id,
        "turn_index": turn_index,
        "request_id": request_id,
        "prefill_length": prefill_length,
        "decode_length": decode_length,
        "history_tokens_before": history_tokens_before,
        "prefill_context_tokens": prefill_context_tokens,
        "final_context_tokens": final_context_tokens,
    }


def _delta(seq, tick, arrivals=(), completed=(), reasons=None):
    """合成 StateDelta v1(schema 校验按基类)。"""
    return {
        "schema_version": 1,
        "delivery_sequence": seq,
        "delivery_epoch": seq,
        "tick": tick,
        "deferred_from_tick": 0,
        "reasons": list(reasons or ["TICK_END"]),
        "arrivals": list(arrivals),
        "completed_groups": list(completed),
        "completed_nodes": [],
        "retry_items": [],
        "affected_ranks": [],
        "snapshot_handle": {"epoch": seq, "tick": tick, "kind": ""},
        "ledger_summary": {"injected_unfinished": []},
    }


def _arrival(queue_index, request_id, session_id, turn_index,
             prefill_length=100, decode_length=5,
             arrival_world_ns=0, interval_ns=0):
    return {
        "arrival_world_ns": arrival_world_ns,
        "decode_length": decode_length,
        "ingress_seq": queue_index,
        "inter_request_interval_ns": interval_ns,
        "prefill_length": prefill_length,
        "queue_index": queue_index,
        "request_id": request_id,
        "session_id": session_id,
        "turn_index": turn_index,
    }


def _complete(request_id, stage):
    return {"request_id": request_id, "stage": stage}


def _pick_slots(config, turn0_count, later_count):
    """合成 fixture config 的 queue_index 槽位池(legacy 测试同款)。"""
    turn0 = [i for i, spec in enumerate(config.request_queue)
             if spec.session_arrival_time_ns is not None]
    later = [i for i, spec in enumerate(config.request_queue)
             if spec.inter_request_interval_ns is not None]
    return turn0[:turn0_count], later[:later_count]


def _build_scheduler(records, config=None, journal_path=None):
    if config is None:
        config = _load_fixture_config()
    manifest = {
        "requests": records,
        "selected_request_count": len(records),
        "selected_session_count": len(
            {record["session_id"] for record in records}),
    }
    journal_recorder = None
    if journal_path is not None:
        journal_recorder = MemoryActionRecorder(
            MemoryMetricsObserver(4), journal_path=journal_path)
    scheduler = WscLlmRelevantOnlineScheduler(
        manifest=manifest, config=config, graph=GraphBatchBuilder(config),
        decision_log_sink=None, train_ledger_sink=None,
        kv_journal_recorder=journal_recorder)
    return config, scheduler


def _ack(scheduler, seq):
    scheduler.on_commit_ack({
        "schema_version": 1,
        "batch_id": seq,
        "delivery_sequence": seq,
    })


def _apply(scheduler, seq, tick, arrivals=(), completed=(), reasons=None):
    batch = scheduler.on_decision_batch(
        _delta(seq, tick, arrivals=arrivals, completed=completed,
               reasons=reasons))
    _ack(scheduler, seq)
    return batch


def _probe(config):
    """probe 调度器:拓扑/静态映射/B1 分配器只读实例(容量推导用)。

    manifest 携带 1 条不驱动的合成记录(expected_request_count 必须 > 0:
    基类在途尾部上限登记对 0 拒绝)。"""
    manifest = {
        "requests": [_request_record(0, "probe_s", 0, "probe_r0", 0, 1000,
                                     1005)],
        "selected_request_count": 1,
        "selected_session_count": 1,
    }
    return WscLlmRelevantOnlineScheduler(
        manifest=manifest, config=config, graph=GraphBatchBuilder(config))


def _empty_instance_tokens(scheduler, instance_index):
    """实例空账本可整装 token 数(逐 rank min;B1 分配器口径)。"""
    config = scheduler.config
    tp = scheduler.topology.instances[0].size
    per_token = kv_cache_shard_bytes_for_tokens(config.model, 1, tp)
    snapshots = scheduler.allocator.rank_ledger_snapshots(instance_index)
    return min(
        (snap.capacity_bytes - snap.model_weight_bytes) // per_token[rank]
        for rank, snap in enumerate(snapshots))


def _decision_rows(scheduler, kind):
    return [row for row in scheduler.online_log_rows
            if row["kind"] == kind]


def _nodes(batch, name_part=None, request_id=None):
    return [node for node in batch["nodes"]
            if (name_part is None or name_part in node["name"])
            and (request_id is None or node["request_id"] == request_id)]


class ConstructorFailClosedTest(unittest.TestCase):
    """构造器 fail-closed(总文档 §2.4 清单)。"""

    def test_wrong_policy_mode_and_sensing_rejected(self):
        config = _load_fixture_config()
        manifest = {"requests": [], "selected_request_count": 0,
                    "selected_session_count": 0}
        graph = GraphBatchBuilder(config)
        # 错 policy:legacy(构造器显式拒绝,分发层 else 之外的第二道门)。
        legacy_config = dataclasses.replace(
            config, kv_cache_policy="legacy")
        with self.assertRaises(ValueError):
            WscLlmRelevantOnlineScheduler(
                manifest=manifest, config=legacy_config, graph=graph)
        # 错 policy:session_lru_recompute。
        lru_config = dataclasses.replace(
            config, kv_cache_policy="session_lru_recompute")
        with self.assertRaises(ValueError):
            WscLlmRelevantOnlineScheduler(
                manifest=manifest, config=lru_config, graph=graph)
        # 非 strategy 模式。
        with self.assertRaises(ValueError):
            WscLlmRelevantOnlineScheduler(
                manifest=manifest, config=config, graph=graph, mode="replay")
        # --sensing 显式拒绝(legacy :225-230 同款)。
        with self.assertRaises(ValueError):
            WscLlmRelevantOnlineScheduler(
                manifest=manifest, config=config, graph=graph, sensing=True)


class CanonicalHitStateTest(unittest.TestCase):
    """canonical hit_state 映射(总文档 §3.4):全源=P → full;否则
    partial;永不 miss(无 miss 枚举路径)。"""

    @staticmethod
    def _source(index, local_hit):
        from types import SimpleNamespace
        return SimpleNamespace(source_instance_index=index,
                               local_hit=local_hit)

    def test_all_local_sources_map_full(self):
        sources = [self._source(1, True), self._source(1, True)]
        self.assertEqual(_canonical_hit_state(sources, 1), "full")

    def test_any_remote_source_maps_partial(self):
        sources = [self._source(0, False), self._source(1, True)]
        self.assertEqual(_canonical_hit_state(sources, 1), "partial")
        self.assertEqual(_canonical_hit_state([], 1), "full")


class PhysicalRunTest(unittest.TestCase):
    """双请求全流程(physical):kv_placement/kv_scatter/kv_remote_reads
    决策行 + 3100/3300 发射 + 无 3000 + KV 事件流 + run-end 审计 +
    journal 守恒门。

    容量构造(P/D 空账本等容,D′={P,D} 全 1 跳):filler 先准入把 D0
    贪心吸到只剩 SPILL_MARGIN,被测请求 fx_s 的 prefill 段溢出到 P own
    piece——这是 3300 读边(默认拓扑唯一远程源 = P)与 LOCAL_HIT 语义
    的前提。"""

    SPILL_MARGIN = 60_000    # filler 留给被测请求的 D0 空额
    S_CTX = 100_000          # 被测请求 prefill 段(> SPILL_MARGIN ⇒ 溢出)

    def setUp(self):
        self._journal_dir = tempfile.TemporaryDirectory(
            prefix="wscllm_relevant_journal_")
        self.journal_path = str(
            Path(self._journal_dir.name) / "kv_delta_journal.jsonl")
        self.config = _load_fixture_config()
        assert self.config.kv_cache_policy == "relevant_distributed"
        probe = _probe(self.config)
        prefill_instance = min(
            probe.topology.indices_for_role(PREFILL_ROLE))
        # 第二个 prefill 实例(与 P1 同归 D0:route_for_prefill 派生)。
        route = probe.static_mapping.route_for_prefill(prefill_instance)
        self.decode_instance = route.decode_instance_index
        second_p = next(
            index for index in probe.topology.indices_for_role(PREFILL_ROLE)
            if index != prefill_instance
            and probe.static_mapping.route_for_prefill(
                index).decode_instance_index == self.decode_instance)
        self.p1, self.p2 = prefill_instance, second_p
        d_empty = _empty_instance_tokens(probe, self.decode_instance)
        self.fill_ctx = d_empty - 5 - self.SPILL_MARGIN
        turn0_slots, _ = _pick_slots(self.config, 2, 0)
        self.turn0_slots = turn0_slots
        # 槽位序 = 同批到达的冻结处理序:fill 先(占 P1),fx_s 后(→ P2)。
        records = [
            _request_record(turn0_slots[0], "fx_fill", 0, "fx_fill_r0", 0,
                            self.fill_ctx, self.fill_ctx + 5),
            _request_record(turn0_slots[1], "fx_s", 0, "fx_s_r0", 0,
                            self.S_CTX, self.S_CTX + 5),
        ]
        _, self.scheduler = _build_scheduler(
            records, self.config, journal_path=self.journal_path)

    def tearDown(self):
        self._journal_dir.cleanup()

    def test_full_lifecycle_decision_rows_and_run_end(self):
        scheduler = self.scheduler
        by_id = scheduler.runtime_by_request_id
        fill = by_id["fx_fill_r0"]
        subject = by_id["fx_s_r0"]

        # seq0:fill + fx_s 同批到达(heap 冻结序:fill 先 → P1;fx_s → P2,
        # 同归 D0)→ 同 pass 双双准入(先 P1 后 P2,sorted frontier 序)。
        batch0 = _apply(scheduler, 0, 1000, arrivals=[
            _arrival(self.turn0_slots[0], "fx_fill_r0", "fx_fill", 0),
            _arrival(self.turn0_slots[1], "fx_s_r0", "fx_s", 0)])
        self.assertEqual(fill.prefill_instance_index, self.p1)
        self.assertEqual(subject.prefill_instance_index, self.p2)
        self.assertEqual(subject.decode_instance_index, self.decode_instance)
        placement = subject.kv_placement
        self.assertIsNotNone(placement)
        # 位置表:decode 段钉 D + prefill 段贪心(D 优先)+ P own piece
        # (tier=prefill_stay,溢出);pieces 全在 D′(裁决 #32)。
        tiers = {(piece.instance_index, piece.tier)
                 for piece in placement.pieces}
        self.assertIn((self.decode_instance, "decode_local"), tiers)
        self.assertIn((self.decode_instance, "scatter_remote"), tiers)
        self.assertIn((self.p2, "prefill_stay"), tiers)
        self.assertTrue(all(
            piece.instance_index in placement.static_route
            for piece in placement.pieces))
        # run 头登记 d2d_to_hbm_bandwidth_ratio(裁决 #22)。
        header = scheduler.online_log_rows[0]
        self.assertEqual(header["kind"], "run_header")
        self.assertAlmostEqual(
            header["decision"]["d2d_to_hbm_bandwidth_ratio"],
            self.config.hardware.d2d_to_hbm_bandwidth_ratio)
        # kv_placement 决策行 ×2(裁决 #8/#23)。
        placement_rows = _decision_rows(scheduler, "kv_placement")
        self.assertEqual({row["request_id"] for row in placement_rows},
                         {"fx_fill_r0", "fx_s_r0"})
        self.assertTrue(all(
            row["decision"]["backpressure_duration_ns"] is None
            for row in placement_rows))

        # seq1:双 drain → 3100 散布 + staging 释放 + train1(双成员)。
        batch1 = _apply(scheduler, 1, 2000, completed=[
            _complete("fx_fill_r0", "prefill"),
            _complete("fx_s_r0", "prefill")])
        subject_scatter_sends = [
            n for n in _nodes(batch1, "_kv_scatter_")
            if "_send_" in n["name"] and n["request_id"] == "fx_s_r0"]
        self.assertTrue(subject_scatter_sends)
        expected_scatter_tokens = sum(
            piece.token_end - piece.token_start
            for piece in placement.pieces
            if piece.instance_index == self.decode_instance
            and piece.token_end <= self.S_CTX)
        self.assertEqual(
            sum(n["comm"]["bytes"] for n in subject_scatter_sends),
            expected_scatter_tokens
            * kv_cache_bytes_for_tokens(self.config.model, 1))
        scatter_rows = _decision_rows(scheduler, "kv_scatter")
        self.assertEqual({row["request_id"] for row in scatter_rows},
                         {"fx_fill_r0", "fx_s_r0"})
        # staging scratch 已释放(仅 own piece 的 resident 留在 P)。
        for instance in (self.p1, self.p2):
            ledger = scheduler.allocator.rank_ledger_snapshots(instance)
            self.assertTrue(all(snap.staging_scratch_bytes == 0
                                for snap in ledger))
        # train1:双成员同列车,双双退出(decode 各 5)。
        train_row = scheduler.train_ledger_rows[-1]
        train_id = train_row["train_id"]
        self.assertEqual(train_row["member_count"], 2)
        self.assertEqual(sorted(train_row["exits"]),
                         ["fx_fill_r0", "fx_s_r0"])
        # 无 transfer-3000(裁决 #9:joiners=[])。
        self.assertEqual(
            [n for n in batch1["nodes"]
             if "_prefill_to_decode_kv" in n["name"]], [])
        # 3300:fx_s 的远程源 = P2 own piece;字节 = p_m × R 精确式。
        read_sends = [n for n in _nodes(batch1, "_kv_remote_read_")
                      if "_send_" in n["name"]
                      and n["request_id"] == "fx_s_r0"]
        read_recvs = [n for n in _nodes(batch1, "_kv_remote_read_")
                      if "_recv_" in n["name"]]
        remote_tokens = placement.piece_tokens(self.p2)
        self.assertGreater(remote_tokens, 0)
        self.assertEqual(sum(n["comm"]["bytes"] for n in read_sends),
                         5 * remote_tokens
                         * kv_cache_bytes_for_tokens(self.config.model, 1))
        for node in read_sends:
            self.assertIs(node["comm"]["hbm_charge"], True)
        for node in read_recvs:
            self.assertIs(node["comm"]["hbm_charge"], False)  # 首个显式 false
        remote_rows = _decision_rows(scheduler, "kv_remote_reads")
        self.assertEqual(len(remote_rows), 1)
        self.assertEqual(remote_rows[0]["request_id"], train_id)
        self.assertTrue(all(
            "noc_hops" in route
            for route in remote_rows[0]["decision"]["routes"]))

        # seq2:双 decode + request 完成 → 终轮释放 + run-end 审计。
        _apply(scheduler, 2, 3000, completed=[
            _complete("fx_fill_r0", "decode"), _complete("fx_fill_r0", ""),
            _complete("fx_s_r0", "decode"), _complete("fx_s_r0", "")])
        scheduler.verify_run_end()
        # KV 事件流(16 元事件数组,session_lru 同构):placement/release
        # 已产出(history_pull 在多轮用例)。
        payload = scheduler.kv_event_payload_relevant()
        self.assertEqual(payload["policy"], "relevant_distributed")
        kinds = {row[3] for row in payload["events"]}
        self.assertEqual(kinds, {"placement", "release"})
        self.assertTrue(all(len(row) == 16 for row in payload["events"]))
        # journal:权重预载 + placement/staging/release 对账行;守恒门
        #(resident=0/reserved=0/physical=weight,纯重放)复跑通过。
        with open(self.journal_path, "r", encoding="utf-8") as source:
            journal_lines = [json.loads(line)
                            for line in source if line.strip()]
        self.assertTrue(journal_lines)
        causes = {row["cause"] for row in journal_lines}
        self.assertIn("model_weight_preload", causes)
        self.assertIn("relevant_placement", causes)
        self.assertIn("staging_scratch", causes)
        summary = verify_relevant_journal_conservation(
            scheduler.kv_journal_recorder)
        self.assertGreater(summary["line_count"], 0)


class BackpressureReleaseReadmitTest(unittest.TestCase):
    """背压→释放→重准入(总文档 §8.4/§7 背压两档的调度器侧闭环)。

    构造:M 先准入占住 D0;F1..F5 占住其余 5 个 P 实例(使 A/B 与 M/A
    同队 P1);A 的 own piece 占住 P1;B 队头阻塞。时序:
      b0:M 到达+准入+发射(P1 忙,M 在 qp)→ b1:F/A 同批到达(F 冻结
      序先处理占 P4..P8;A 与 M 同队 P1)→ b2:M drain(train1)+ A 准入
      → b3:M 完成(terminal 释放 D0)+ B 到达(排入 A 之后)→ b4:
      A drain(staging 释放后 P1 仍不足)→ B 队头 try_place None →
      FCFS 阻塞 → b5:空批重查仍阻塞 → b6:A 终轮完成释放 → B 重查
      通过(背压持续时长 = b6−b4 = 2 个决策批)。"""

    def setUp(self):
        self.config = _load_fixture_config()
        self.probe = _probe(self.config)
        self.prefill_instance = min(
            self.probe.topology.indices_for_role(PREFILL_ROLE))
        self.route = self.probe.static_mapping.route_for_prefill(
            self.prefill_instance)
        self.decode_instance = self.route.decode_instance_index
        p_empty = _empty_instance_tokens(self.probe, self.prefill_instance)
        d_empty = _empty_instance_tokens(self.probe, self.decode_instance)
        self.p_empty, self.d_empty = p_empty, d_empty
        self.margin = 50_000
        slack = max(0, d_empty - p_empty)
        # F1..F5 在 P4..P8 上,其中路由到 D0 的会在 D0 吸收 1_000 token。
        other_ps = [index for index
                    in self.probe.topology.indices_for_role(PREFILL_ROLE)
                    if index != self.prefill_instance]
        f_d0_tokens = 1_000 * sum(
            1 for index in other_ps
            if self.probe.static_mapping.route_for_prefill(
                index).decode_instance_index == self.decode_instance)
        # M:占住 D0(M_ctx 之后 D0 余 margin+slack);A:A_ctx = p_empty−5K,
        # own piece = A_ctx − D0 吸收量;drain 后 P1 余
        # ≈ 5K+margin+slack−5+f_d0,B 必须大于该值且 ≤ p_empty(②空域可
        # 过、当前档不可过)。
        m_ctx = min(p_empty, d_empty) - self.margin
        a_ctx = p_empty - 5_000
        self.m_ctx, self.a_ctx = m_ctx, a_ctx
        self.b_ctx = (5_000 + self.margin + slack - 5
                      + f_d0_tokens) + 10_000
        self.assertLess(self.b_ctx, p_empty)          # ② 空域档必须可过
        self.assertGreater(m_ctx, self.b_ctx)
        slots, _ = _pick_slots(self.config, 8, 0)
        self.slots = slots
        # 槽位:M=slots[1]、F=slots[2..6]、A=slots[7]、B=slots[0](同批
        # 到达按 queue_index 冻结序处理:F 先于 A;B 独批到达)。
        records = [
            _request_record(slots[1], "fx_M", 0, "fx_M_r0", 0, m_ctx,
                            m_ctx + 5),
        ]
        for i in range(5):
            records.append(_request_record(
                slots[2 + i], f"fx_F{i}", 0, f"fx_F{i}_r0", 0, 1_000,
                1_000 + 2, decode_length=2))
        records.append(_request_record(
            slots[7], "fx_A", 0, "fx_A_r0", 0, a_ctx, a_ctx + 5))
        records.append(_request_record(
            slots[0], "fx_B", 0, "fx_B_r0", 0, self.b_ctx, self.b_ctx + 5))
        _, self.scheduler = _build_scheduler(records, self.config)

    def test_head_of_line_blocking_until_terminal_release(self):
        scheduler = self.scheduler
        by_id = scheduler.runtime_by_request_id
        pi, di = self.prefill_instance, self.decode_instance

        # b0 t1000:M 到达 + 准入 + prefill 发射(D0 吸收 M_ctx;P1 忙)。
        _apply(scheduler, 0, 1000, arrivals=[
            _arrival(self.slots[1], "fx_M_r0", "fx_M", 0)])
        self.assertIsNotNone(by_id["fx_M_r0"].kv_placement)
        # b1 t2000:F1..F5 + A 同批到达(冻结队列序:F 先,占 P4..P8;
        # A 与 M 同队 P1——全部 count 1 打平,min index = P1)。
        arrivals = [
            _arrival(self.slots[2 + i], f"fx_F{i}_r0", f"fx_F{i}", 0,
                     decode_length=2) for i in range(5)
        ]
        arrivals.append(_arrival(self.slots[7], "fx_A_r0", "fx_A", 0))
        _apply(scheduler, 1, 2000, arrivals=arrivals)
        a = by_id["fx_A_r0"]
        self.assertEqual(a.prefill_instance_index, pi)
        self.assertIsNone(a.kv_placement)   # P1 busy(M 在飞),A 仅排队
        # b2 t3000:M drain(train1:M 单成员自然退出)→ 同 pass A 准入。
        _apply(scheduler, 2, 3000, completed=[_complete("fx_M_r0", "prefill")])
        self.assertIsNotNone(a.kv_placement)
        self.assertEqual(a.decode_instance_index, di)
        # b3 t4000:M 完成(terminal 释放 D0)+ B 到达(P1 qp=[A] 与其余
        # P 的 [F] 打平 → min index = P1,排入 A 之后)。
        _apply(scheduler, 3, 4000, completed=[
            _complete("fx_M_r0", "decode"), _complete("fx_M_r0", "")],
            arrivals=[_arrival(self.slots[0], "fx_B_r0", "fx_B", 0)])
        self.assertNotIn("fx_M", scheduler.session_placements)
        b = by_id["fx_B_r0"]
        self.assertEqual(b.prefill_instance_index, pi)
        self.assertIsNone(b.kv_placement)
        # b4 t5000:A drain → B 队头 try_place None → FCFS 队头阻塞
        #(②当前档:P1 own piece 占住;整队停,实例保持 frontier)。
        batch4 = _apply(scheduler, 4, 5000, completed=[
            _complete("fx_A_r0", "prefill")])
        self.assertIsNone(b.kv_placement)            # FCFS 队头阻塞
        self.assertEqual([n for n in batch4["nodes"]
                          if n["request_id"] == "fx_B_r0"], [])
        self.assertIsNotNone(b.backpressure_since_ns)  # 观测:片段开始
        self.assertEqual(scheduler.backpressure_episode_count, 1)
        # b5 t6000:空批(后续调度事件重查容量)——仍阻塞,零发射。
        batch5 = _apply(scheduler, 5, 6000)
        self.assertIsNone(b.kv_placement)
        self.assertEqual(batch5["watches"], [])
        self.assertEqual(batch5["assignments"], [])
        # b6 t7000:A decode+request 完成(终轮释放 P1 own piece 与 D0
        # piece)→ B 重查通过(准入 + 发射 + 背压持续时长观测字段)。
        _apply(scheduler, 6, 7000, completed=[
            _complete("fx_A_r0", "decode"), _complete("fx_A_r0", "")])
        self.assertIsNotNone(b.kv_placement)         # 阻塞解除
        placement_rows = _decision_rows(scheduler, "kv_placement")
        b_row = next(row for row in placement_rows
                     if row["request_id"] == "fx_B_r0")
        self.assertEqual(b_row["decision"]["backpressure_duration_ns"], 2000)
        # F1..F5 完成(drain → train → 完成)。
        _apply(scheduler, 7, 8000, completed=[
            _complete(f"fx_F{i}_r0", "prefill") for i in range(5)])
        _apply(scheduler, 8, 9000, completed=[
            _complete(f"fx_F{i}_r0", "decode") for i in range(5)
        ] + [_complete(f"fx_F{i}_r0", "") for i in range(5)])
        # B 完成(drain → train → decode+request)。
        _apply(scheduler, 9, 10_000, completed=[
            _complete("fx_B_r0", "prefill")])
        _apply(scheduler, 10, 11_000, completed=[
            _complete("fx_B_r0", "decode"), _complete("fx_B_r0", "")])
        scheduler.verify_run_end()
        self.assertEqual(scheduler.completed_requests, 8)
        self.assertEqual(scheduler.session_placements, {})
        self.assertEqual(scheduler.backpressure_total_ns, 2000)


class MultiTurnHistoryPullTest(unittest.TestCase):
    """多轮:拉回(1000 多源,源=P 零边 LOCAL_HIT)→ 重放置(新
    KVPlacement)→ 再释放;canonical hit_state 映射断言。

    容量构造:filler 占 D0 使 fx_s turn0 的 prefill 溢出到 P2 own piece
    (turn1 拉回的 LOCAL_HIT 部分);turn1 到达时 F 占住 P1 使 fx_s_r1
    仍选 P2(LOCAL_HIT 命中同 P)。"""

    SPILL_MARGIN = 60_000
    S_CTX = 100_000
    NEW_TOKENS = 10_000

    def setUp(self):
        self.config = _load_fixture_config()
        probe = _probe(self.config)
        p1 = min(probe.topology.indices_for_role(PREFILL_ROLE))
        route = probe.static_mapping.route_for_prefill(p1)
        self.decode_instance = route.decode_instance_index
        self.p1 = p1
        self.p2 = next(
            index for index in probe.topology.indices_for_role(PREFILL_ROLE)
            if index != p1
            and probe.static_mapping.route_for_prefill(
                index).decode_instance_index == self.decode_instance)
        d_empty = _empty_instance_tokens(probe, self.decode_instance)
        self.fill_ctx = d_empty - 5 - self.SPILL_MARGIN
        turn0_slots, later_slots = _pick_slots(self.config, 3, 1)
        self.slots = turn0_slots
        self.later_slot = later_slots[0]
        self.ctx0 = self.S_CTX
        self.final0 = self.S_CTX + 5
        records = [
            # 槽位序 = 同批到达冻结序:fill(P1)先,fx_s_r0(P2)后。
            _request_record(turn0_slots[0], "fx_fill", 0, "fx_fill_r0", 0,
                            self.fill_ctx, self.fill_ctx + 5),
            _request_record(turn0_slots[1], "fx_s", 0, "fx_s_r0", 0,
                            self.ctx0, self.final0),
            # F:turn1 到达时占住 P1(更小 queue_index → 先处理)。
            _request_record(turn0_slots[2], "fx_F", 0, "fx_F_r0", 0, 1_000,
                            1_002, decode_length=2),
            _request_record(
                later_slots[0], "fx_s", 1, "fx_s_r1", self.final0,
                self.final0 + self.NEW_TOKENS,
                self.final0 + self.NEW_TOKENS + 5),
        ]
        _, self.scheduler = _build_scheduler(records, self.config)

    def test_release_pull_replace_release_cycle(self):
        scheduler = self.scheduler
        by_id = scheduler.runtime_by_request_id
        interval = self.config.request_queue[
            self.later_slot].inter_request_interval_ns
        model = self.config.model

        # b0:fill + fx_s_r0 到达+准入+发射。
        _apply(scheduler, 0, 1000, arrivals=[
            _arrival(self.slots[0], "fx_fill_r0", "fx_fill", 0),
            _arrival(self.slots[1], "fx_s_r0", "fx_s", 0)])
        placement0 = by_id["fx_s_r0"].kv_placement
        remote0 = placement0.piece_tokens(self.p2)
        d0_tokens0 = placement0.piece_tokens(self.decode_instance)
        self.assertGreater(remote0, 0)   # P own piece ⇒ 拉回有 LOCAL_HIT 部分
        # b1:双 drain → train1(双成员,双双退出)。
        _apply(scheduler, 1, 2000, completed=[
            _complete("fx_fill_r0", "prefill"),
            _complete("fx_s_r0", "prefill")])
        # b2:双完成:fill terminal 释放;fx_s_r0 非 terminal 驻留。
        batch2 = _apply(scheduler, 2, 3000, completed=[
            _complete("fx_fill_r0", "decode"), _complete("fx_fill_r0", ""),
            _complete("fx_s_r0", "decode"), _complete("fx_s_r0", "")])
        self.assertNotIn("fx_fill", scheduler.session_placements)
        self.assertIn("fx_s", scheduler.session_placements)
        self.assertEqual(
            batch2["future_alarms"][0]["envelope"]["request_id"], "fx_s_r1")
        self.assertEqual(
            batch2["future_alarms"][0]["arrival_world_ns"], 3000 + interval)

        # b3:F + fx_s_r1 同批到达(F 先 → P1;fx_s_r1 → P2,同 P 命中
        # LOCAL_HIT):释放旧 placement + 多源拉回 + 重准入 + 发射。
        arrival_tick = 3000 + interval
        batch3 = _apply(scheduler, 3, arrival_tick, arrivals=[
            _arrival(self.slots[2], "fx_F_r0", "fx_F", 0, decode_length=2),
            _arrival(self.later_slot, "fx_s_r1", "fx_s", 1,
                     interval_ns=interval)])
        r1 = by_id["fx_s_r1"]
        self.assertEqual(r1.prefill_instance_index, self.p2)
        # 旧 placement 已释放、新 placement 已冻结(重放置)。
        self.assertIsNot(scheduler.session_placements["fx_s"], placement0)
        placement1 = r1.kv_placement
        self.assertEqual(placement1.turn_index, 1)
        self.assertEqual(placement1.total_tokens,
                         self.final0 + self.NEW_TOKENS + 5)
        # canonical hit_state:源含旧 D ⇒ partial;旧 P2 piece ⇒ LOCAL_HIT
        # 零边(总字节 = kv(history),远程字节 = 旧 D 持有部分)。
        self.assertEqual(r1.history_canonical_hit_state, "partial")
        self.assertEqual(r1.history_source_instance_indexes,
                         [self.decode_instance])
        self.assertEqual(
            r1.history_transfer_bytes,
            kv_cache_bytes_for_tokens(model, self.final0))
        self.assertEqual(
            r1.history_remote_transfer_bytes,
            kv_cache_bytes_for_tokens(model, d0_tokens0))
        self.assertEqual(r1.history_local_hit_tokens, remote0)
        # 1000 拉回边:仅旧 D 为源;字节 = 旧 D 持有 token;两端
        # hbm_charge=true;recv 物理先于 turn1 prefill 体(per-rank 序)。
        pull_sends = [n for n in _nodes(batch3, "_kv_history_pull_")
                      if "_send_" in n["name"]]
        pull_recvs = [n for n in _nodes(batch3, "_kv_history_pull_")
                      if "_recv_" in n["name"]]
        self.assertTrue(pull_sends)
        self.assertEqual(sum(n["comm"]["bytes"] for n in pull_sends),
                         d0_tokens0 * kv_cache_bytes_for_tokens(model, 1))
        for node in pull_sends + pull_recvs:
            self.assertIs(node["comm"]["hbm_charge"], True)
        for rank in scheduler.topology.instance(self.p2).ranks:
            recv_ids = [n["id"] for n in pull_recvs if n["rank"] == rank]
            prefill_ids = [
                n["id"] for n in _nodes(batch3, request_id="fx_s_r1")
                if n["rank"] == rank
                and "_kv_history_pull_" not in n["name"]]
            if recv_ids and prefill_ids:
                self.assertLess(max(recv_ids), min(prefill_ids))
        # KV 事件流三类齐全(placement/release/history_pull)。
        payload = scheduler.kv_event_payload_relevant()
        kinds = {row[3] for row in payload["events"]}
        self.assertEqual(kinds, {"placement", "release", "history_pull"})
        # 决策行:prefill(turn1) 证据字段。
        r1_row = next(row for row in _decision_rows(scheduler, "prefill")
                      if row["request_id"] == "fx_s_r1")
        self.assertEqual(
            r1_row["decision"]["history_canonical_hit_state"], "partial")
        self.assertEqual(r1_row["decision"]["history_local_hit_tokens"],
                         remote0)
        self.assertTrue(r1_row["decision"]["history_pull_routes"])
        # b4:F + fx_s_r1 drain → train2;b5:完成 → 再释放 + run-end 审计。
        _apply(scheduler, 4, arrival_tick + 1000, completed=[
            _complete("fx_F_r0", "prefill"),
            _complete("fx_s_r1", "prefill")])
        _apply(scheduler, 5, arrival_tick + 2000, completed=[
            _complete("fx_F_r0", "decode"), _complete("fx_F_r0", ""),
            _complete("fx_s_r1", "decode"), _complete("fx_s_r1", "")])
        scheduler.verify_run_end()
        self.assertEqual(scheduler.session_placements, {})
        # 再释放后账本归零(resident=staging=0)。
        for snap in scheduler.allocator.rank_ledger_snapshots():
            self.assertEqual(snap.resident_kv_bytes, 0)
            self.assertEqual(snap.staging_scratch_bytes, 0)


class R4SchedulerBatchOrderTest(unittest.TestCase):
    """R4 调度器侧:同 tick 完成批先于准入批(总文档 §9 R4)。

    同一 delta 内 A 的 PREFILL_DRAIN 完成与 B 的到达:drain 的 3100 send
    (完成批处理段发射)与 B 的 prefill 节点(准入/发射 pass 段)在同一
    批内,per-rank 节点 id 严格递增(3100 send 先于 B 的链首)——与图侧
    frontier 统一裁决共同保证共享 rank 的发行序 = 全局发射序,无反转。"""

    def test_drain_scatter_precedes_admission_in_rank_order(self):
        config = _load_fixture_config()
        turn0_slots, _ = _pick_slots(config, 2, 0)
        records = [
            _request_record(turn0_slots[0], "fx_a", 0, "fx_a_r0", 0,
                            200_000, 200_005),
            _request_record(turn0_slots[1], "fx_b", 0, "fx_b_r0", 0,
                            100_000, 100_005),
        ]
        _, scheduler = _build_scheduler(records, config)
        prefill_instance = min(
            scheduler.topology.indices_for_role(PREFILL_ROLE))
        p_ranks = scheduler.topology.instance(prefill_instance).ranks
        _apply(scheduler, 0, 1000, arrivals=[
            _arrival(turn0_slots[0], "fx_a_r0", "fx_a", 0)])
        a = scheduler.runtime_by_request_id["fx_a_r0"]
        # 同 tick:A drain 完成批 + B 到达(完成批先处理 → 3100 先发射;
        # B 的准入/发射在 pass 段)。
        batch = _apply(scheduler, 1, 2000,
                       completed=[_complete("fx_a_r0", "prefill")],
                       arrivals=[_arrival(turn0_slots[1], "fx_b_r0",
                                          "fx_b", 0)])
        b = scheduler.runtime_by_request_id["fx_b_r0"]
        self.assertEqual(b.prefill_instance_index, prefill_instance)
        self.assertIsNotNone(b.kv_placement)
        scatter_sends = [n for n in _nodes(batch, "_kv_scatter_")
                         if "_send_" in n["name"]]
        self.assertTrue(scatter_sends)
        b_nodes = _nodes(batch, request_id="fx_b_r0")
        self.assertTrue(b_nodes)
        for rank in p_ranks:
            scatter_ids = [n["id"] for n in scatter_sends
                           if n["rank"] == rank]
            b_ids = [n["id"] for n in b_nodes if n["rank"] == rank]
            if scatter_ids and b_ids:
                self.assertLess(max(scatter_ids), min(b_ids))
        # 3100 send 的显式锚 = A 的 PREFILL_DRAIN watch members(R4
        # fallback 语义,不依赖 frontier 等价)。
        for rank in p_ranks:
            for node in (n for n in scatter_sends if n["rank"] == rank):
                parents = sorted(
                    edge["from"] for edge in batch["parent_edges"]
                    if edge["rank"] == rank and edge["to"] == node["id"])
                self.assertEqual(parents,
                                 [a.prefill_end_members[rank]])
        # 收尾:A 列车完成(terminal)→ B drain → B 列车完成 → verify。
        _apply(scheduler, 2, 3000, completed=[
            _complete("fx_a_r0", "decode"), _complete("fx_a_r0", "")])
        _apply(scheduler, 3, 4000, completed=[
            _complete("fx_b_r0", "prefill")])
        _apply(scheduler, 4, 5000, completed=[
            _complete("fx_b_r0", "decode"), _complete("fx_b_r0", "")])
        scheduler.verify_run_end()


class IdealMaskedAbTest(unittest.TestCase):
    """ideal_masked 分支(裁决 #11):不发 3300、不裁 KV 分量(现状字节
    口径);3100 散布不受开关影响;与 physical 构成严格 A/B。

    容量构造同 PhysicalRunTest(filler 使 fx_s 溢出到 P own piece,从而
    physical 侧存在 3300 读边与 KV 裁剪差)。"""

    SPILL_MARGIN = 60_000
    S_CTX = 100_000

    @staticmethod
    def _run(config):
        probe = _probe(config)
        p1 = min(probe.topology.indices_for_role(PREFILL_ROLE))
        decode_instance = probe.static_mapping.route_for_prefill(
            p1).decode_instance_index
        p2 = next(
            index for index in probe.topology.indices_for_role(PREFILL_ROLE)
            if index != p1
            and probe.static_mapping.route_for_prefill(
                index).decode_instance_index == decode_instance)
        d_empty = _empty_instance_tokens(probe, decode_instance)
        fill_ctx = d_empty - 5 - IdealMaskedAbTest.SPILL_MARGIN
        turn0_slots, _ = _pick_slots(config, 2, 0)
        records = [
            _request_record(turn0_slots[0], "fx_fill", 0, "fx_fill_r0", 0,
                            fill_ctx, fill_ctx + 5),
            _request_record(turn0_slots[1], "fx_s", 0, "fx_s_r0", 0,
                            IdealMaskedAbTest.S_CTX,
                            IdealMaskedAbTest.S_CTX + 5),
        ]
        _, scheduler = _build_scheduler(records, config)
        _apply(scheduler, 0, 1000, arrivals=[
            _arrival(turn0_slots[0], "fx_fill_r0", "fx_fill", 0),
            _arrival(turn0_slots[1], "fx_s_r0", "fx_s", 0)])
        batch1 = _apply(scheduler, 1, 2000, completed=[
            _complete("fx_fill_r0", "prefill"),
            _complete("fx_s_r0", "prefill")])
        d_ranks = scheduler.topology.instance(decode_instance).ranks
        body_bytes = sum(
            node["compute"]["tensor_size"]
            for node in batch1["nodes"]
            if node["request_id"].startswith("batch_train_")
            and node["rank"] in d_ranks)
        return scheduler, batch1, body_bytes, p2

    def test_no_3300_no_kv_trim_but_scatter_kept(self):
        physical_config = _load_fixture_config("physical")
        ideal_config = _load_fixture_config("ideal_masked")
        self.assertEqual(ideal_config.kv_remote_read, "ideal_masked")
        _, _, physical_body, _ = self._run(physical_config)
        (ideal_scheduler, ideal_batch, ideal_body,
         p2) = self._run(ideal_config)
        # ideal:无 3300 节点、无 kv_remote_reads 决策行。
        self.assertEqual(
            [n for n in ideal_batch["nodes"]
             if "_kv_remote_read_" in n["name"]], [])
        self.assertEqual(_decision_rows(ideal_scheduler, "kv_remote_reads"),
                         [])
        # ideal:3100 散布置换不受 A/B 开关影响(仍发射)。
        self.assertTrue([n for n in ideal_batch["nodes"]
                         if "_kv_scatter_" in n["name"]])
        # 字节口径:ideal 不裁 KV 分量(体字节 > physical 的裁剪体);
        # 守恒差 = 远程 piece 的 KV 分量(P own piece token 数 × 迭代数
        # × 全实例每 token KV 字节;filler 全本地,两跑无差)。
        self.assertGreater(ideal_body, physical_body)
        ideal_placement_row = next(
            row for row in _decision_rows(ideal_scheduler, "kv_placement")
            if row["request_id"] == "fx_s_r0")
        p_pieces = [piece for piece
                    in ideal_placement_row["decision"]["pieces"]
                    if piece["tier"] == "prefill_stay"]
        self.assertTrue(p_pieces)
        remote_prefill_tokens = sum(
            piece["token_end"] - piece["token_start"]
            for piece in p_pieces)
        expected_delta = (
            remote_prefill_tokens * 5   # decode_length = participation 总和
            * kv_cache_bytes_for_tokens(ideal_config.model, 1))
        self.assertEqual(ideal_body - physical_body, expected_delta)
        # ideal 全流程也可 run-end 通过(守恒门不依赖 3300)。
        _apply(ideal_scheduler, 2, 3000, completed=[
            _complete("fx_fill_r0", "decode"), _complete("fx_fill_r0", ""),
            _complete("fx_s_r0", "decode"), _complete("fx_s_r0", "")])
        ideal_scheduler.verify_run_end()


class SplitSwitchInheritedTest(unittest.TestCase):
    """SH_FIRST_TOKEN_SPLIT 缺省关继承:显式 "1" 下本变体也不拆车(总文档
    附录"新变体不拆车,行为与 OFF 侧一致")——无 first_step/first_token
    节点,run-end 通过。"""

    def test_explicit_on_still_does_not_split(self):
        saved = os.environ.get("SH_FIRST_TOKEN_SPLIT")
        os.environ["SH_FIRST_TOKEN_SPLIT"] = "1"
        try:
            config = _load_fixture_config()
            turn0_slots, _ = _pick_slots(config, 1, 0)
            records = [_request_record(
                turn0_slots[0], "fx_s", 0, "fx_s_r0", 0, 100_000, 100_005)]
            _, scheduler = _build_scheduler(records, config)
            _apply(scheduler, 0, 1000, arrivals=[
                _arrival(turn0_slots[0], "fx_s_r0", "fx_s", 0)])
            batch1 = _apply(scheduler, 1, 2000, completed=[
                _complete("fx_s_r0", "prefill")])
            for node in batch1["nodes"]:
                self.assertNotIn("first_step", node["name"])
                self.assertNotIn("first_token", node["name"])
            _apply(scheduler, 2, 3000, completed=[
                _complete("fx_s_r0", "decode"),
                _complete("fx_s_r0", "")])
            scheduler.verify_run_end()
        finally:
            if saved is None:
                os.environ.pop("SH_FIRST_TOKEN_SPLIT", None)
            else:
                os.environ["SH_FIRST_TOKEN_SPLIT"] = saved


class JournalConservationGateTest(unittest.TestCase):
    """纯重放守恒门的 fail-closed 方向:未释放的 resident 在重放后触发
    守恒断言(resident != 0)。"""

    def test_unreleased_resident_fails_conservation(self):
        with tempfile.TemporaryDirectory(
                prefix="wscllm_relevant_bad_journal_") as tmp:
            path = str(Path(tmp) / "kv_delta_journal.jsonl")
            recorder = MemoryActionRecorder(
                MemoryMetricsObserver(4), journal_path=path)
            # 手工构造:权重预载 + 一笔未释放的 resident delta(绕过
            # allocator,直接驱动 recorder——守恒门只读 journal)。
            rank = 0
            recorder.initialize_rank(rank, 1 << 40)
            recorder.record(
                planner_time_ns=0, anchor_kind="tick_zero",
                request_id=None, session_id=None, rank=rank,
                allocation_key=f"weight:{rank}", weight_delta_bytes=1 << 30,
                cause="model_weight_preload")
            recorder.record(
                planner_time_ns=1, anchor_kind="prefill_start",
                request_id="r0", session_id="s0", rank=rank,
                allocation_key="relevant:r0:instance0",
                resident_kv_delta_bytes=1 << 20, cause="relevant_placement")
            recorder.close_journal()
            with self.assertRaises(RuntimeError):
                verify_relevant_journal_conservation(recorder)


if __name__ == "__main__":
    unittest.main()
