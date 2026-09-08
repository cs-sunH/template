#!/usr/bin/env python3
"""test_decode_eviction_serialization.py -- wscllm decode 决策逐出序列化
与 shard 级路由序列化钉子(2026-09-05 问题 2A/4b 修复回测)。

缺陷(问题 2A):decode 侧逐出(#2 reserve_request_capacity 预占 /
#4 move_prefill_to_decode / #5 grow_decode 累积进 runtime.
decode_target_evictions)此前从未随 decode 决策行落盘——发射时点
(_emit_train joiner 循环)恰为全集(全部 append 之后、之后无
append),修复 = 发射时快照序列化 + 发射后置空核销(镜像
completion 对 completion_evictions 的先序列化后置空语义)。

缺陷(问题 4b):prefill_decode_transfer 的 shards 与 history 迁移的
transfer_shards 均有 rank 级路由可算(_xy_route 仓内现成),但序列化
时被丢弃——hopbytes 只能按实例级聚合口径回退。修复 = 序列化侧补
noc_path/noc_hops(_kv_transfer_dict 可选 hardware 参数 + prefill 决策
history_transfer_shards 字段),hardware=None 行为与旧版逐字节一致。

覆盖:
  1. _decode_decision_dict 直接单测:(E,) -> 全 9 字段;() -> [];
     原 4 字段原样保留;
  2. _decode_decision_dict 传 hardware:shards 逐项补 noc_path/
     noc_hops,与 _xy_route 手算一致;
  3. 真 _emit_train 驱动 + 真 log_decision 捕获:decode 行含新字段、
     发射后 runtime.decode_target_evictions 置空;
  4. _try_admit_prefill 留存 decision.transfer_shards(NOC_MIGRATE);
  5. _history_transfer_shard_dicts 手算钉子;
  6. _kv_transfer_dict(hardware=None) 与旧版逐字节一致(旧 shard 形状
     恰 4 键,无 noc 字段)。

Run: python3 online/test_decode_eviction_serialization.py
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

from generate_wsc_llm_trace import (  # noqa: E402
    _kv_transfer_dict,
    _xy_route,
)
from online.wsc_llm_online_scheduler import (  # noqa: E402
    _OnlineInstanceState,
    _OnlineRequestRuntime,
    _history_transfer_shard_dicts,
    WscLlmOnlineScheduler,
)
from session_kv_manager import (  # noqa: E402
    CapacityResult,
    EvictionRecord,
    HistoryDecision,
    KVTransfer,
    KVTransferShard,
)
from wsc_llm_scheduler import (  # noqa: E402  (只读 import)
    DECODE_ROLE,
    WscLlmHardware,
    WscLlmModel,
)


def _eviction(victim_instance_index, trigger_request_id, time_ns=1_000):
    return EvictionRecord(
        time_ns=time_ns,
        phase="prefill_admission",
        reason="static_decode_final_kv_reservation",
        trigger_request_id=trigger_request_id,
        victim_session_id=f"victim_session_{victim_instance_index}",
        victim_instance_index=victim_instance_index,
        victim_last_completion_ns=0,
        context_tokens=8,
        shard_bytes=(8,),
    )


def _record(request_id, *, prefill=512, decode=4, ctx=None):
    context = prefill if ctx is None else ctx
    return {
        "request_id": request_id,
        "session_id": f"session_{request_id}",
        "turn_index": 0,
        "queue_index": 0,
        "prefill_length": prefill,
        "decode_length": decode,
        "history_tokens_before": 0,
        "prefill_context_tokens": context,
        "final_context_tokens": context + decode,
    }


def _runtime(request_id, *, instance=0):
    rt = _OnlineRequestRuntime(_record(request_id))
    rt.prefill_instance_index = (instance + 1) % 2
    rt.decode_instance_index = instance
    rt.decode_queue_depth_before_enqueue = 0
    rt.static_route = SimpleNamespace(
        prefill_instance_index=(instance + 1) % 2,
        decode_instance_index=instance,
        path=[(instance + 1) % 2, instance],
        hop_count=1,
        shared_edges=(((instance + 1) % 2, instance),),
    )
    return rt


def _hardware():
    # 2x4 mesh:ranks 0..7,行优先(divmod(rank, mesh_cols))。
    return WscLlmHardware(
        mesh_rows=2, mesh_cols=4,
        local_hbm_capacity_bytes=1 << 40,
        local_hbm_bandwidth_gbps=1.0,
        d2d_bandwidth_gbps=1.0,
        peak_perf_tflops=1.0,
        d2d_latency_ns=1,
        local_hbm_latency_ns=1)


def _transfer(trigger_request_id="session_0_request_0"):
    # rank 0 -> 5(XY:先列 0->1,后行 0->1:path=[0,1,5],hops=2)与
    # rank 2 -> 2(同 rank:path=[2],hops=0)。
    return KVTransfer(
        action="NOC_MIGRATE", phase="prefill_to_decode",
        reason="static_pd_mapping", session_id="session_0",
        trigger_request_id=trigger_request_id,
        source_instance_index=1, target_instance_index=0,
        history_tokens=64, total_bytes=3000,
        shards=(
            KVTransferShard(0, 0, 5, 2000),
            KVTransferShard(1, 2, 2, 1000),
        ))


def _scheduler(hardware):
    """裸实例(参照 test_admission_eviction_accumulation.py 的
    _bare_scheduler 构造):__new__ 跳过 __init__,补 _emit_train /
    _decode_decision_dict / _try_admit_prefill 触达的字段。"""
    scheduler = WscLlmOnlineScheduler.__new__(WscLlmOnlineScheduler)
    scheduler.instances = [
        _OnlineInstanceState(index=0, phase_role=DECODE_ROLE),
        _OnlineInstanceState(index=1, phase_role=DECODE_ROLE),
    ]
    scheduler.runtime_by_request_id = {}
    scheduler.capacity_epoch = [0, 0]
    scheduler.config = SimpleNamespace(
        model=WscLlmModel(
            layers=32, hidden_size=4096, ffn_size=11008, num_heads=32,
            vocab_size=32000, bytes_per_elem=2))
    scheduler.topology = SimpleNamespace(
        hardware=hardware,
        instances=[SimpleNamespace(size=1), SimpleNamespace(size=1)])
    # ---- _emit_train 驱动补充装配 ----
    scheduler._train_max_iter = 0
    scheduler._train_instance_index = {}
    scheduler._ready_frontier = set()
    scheduler._emitted_by_delivery = {1: {"requests": []}}
    scheduler._batch = {"delivery_sequence": 1, "watches": []}
    scheduler.ledger_issued = {}
    scheduler.train_ledger_sink = None
    scheduler.train_ledger_rows = []
    scheduler.online_log_rows = []
    scheduler.online_log_count = 0
    scheduler.decision_log_sink = None
    scheduler.graph = SimpleNamespace(
        emit_iteration_train=lambda plan: {
            "exit_members": {
                member["request_id"]: [member["request_id"]]
                for member in plan["exit_members"]},
            "sentinel_members": []})
    return scheduler


class DecodeDecisionDictTest(unittest.TestCase):
    """_decode_decision_dict 直接单测(问题 2A 核心)。"""

    def setUp(self):
        self.scheduler = _scheduler(_hardware())

    def test_serializes_decode_target_evictions(self):
        e = _eviction(0, "session_0_request_0", time_ns=1_500)
        rt = _runtime("session_0_request_0")
        rt.decode_target_evictions = (e,)
        decision = self.scheduler._decode_decision_dict(
            rt, rt.static_route)
        self.assertEqual(decision["decode_target_evictions"],
                         [json.loads(json.dumps({
                             "time_ns": e.time_ns, "phase": e.phase,
                             "reason": e.reason,
                             "trigger_request_id": e.trigger_request_id,
                             "victim_session_id": e.victim_session_id,
                             "victim_instance_index":
                                 e.victim_instance_index,
                             "victim_last_completion_ns":
                                 e.victim_last_completion_ns,
                             "context_tokens": e.context_tokens,
                             "shard_bytes": list(e.shard_bytes),
                         }))])

    def test_empty_decode_target_evictions_serializes_empty_list(self):
        rt = _runtime("session_0_request_0")
        decision = self.scheduler._decode_decision_dict(
            rt, rt.static_route)
        self.assertEqual(decision["decode_target_evictions"], [])

    def test_legacy_fields_preserved(self):
        """问题 2A 修复只增不改:原 4 字段逐样保留(键集钉子)。"""
        rt = _runtime("session_0_request_0")
        rt.prefill_decode_transfer = _transfer()
        decision = self.scheduler._decode_decision_dict(
            rt, rt.static_route)
        self.assertEqual(decision["decode_instance_index"], 0)
        self.assertEqual(
            decision["static_route"],
            {"prefill_instance_index": 1, "decode_instance_index": 0,
             "path": [1, 0], "hop_count": 1, "shared_edges": [[1, 0]]})
        self.assertEqual(
            decision["decode_queue_depth_before_enqueue"], 0)
        self.assertEqual(
            sorted(decision),
            ["decode_instance_index", "decode_queue_depth_before_enqueue",
             "decode_target_evictions", "prefill_decode_transfer",
             "static_route"])

    def test_shards_carry_rank_level_routes(self):
        """问题 4b:hardware 传入时 shards 逐项补 noc_path/noc_hops,
        与 _xy_route 手算一致(2x4 mesh:0->5 得 [0,1,5]/2;2->2 得 [2]/0)。"""
        rt = _runtime("session_0_request_0")
        rt.prefill_decode_transfer = _transfer()
        decision = self.scheduler._decode_decision_dict(
            rt, rt.static_route)
        shards = decision["prefill_decode_transfer"]["shards"]
        self.assertEqual(shards[0]["noc_path"], [0, 1, 5])
        self.assertEqual(shards[0]["noc_hops"], 2)
        self.assertEqual(shards[1]["noc_path"], [2])
        self.assertEqual(shards[1]["noc_hops"], 0)
        self.assertEqual(
            shards[0]["noc_path"],
            _xy_route(self.scheduler.topology.hardware, 0, 5))


class EmitTrainSerializationTest(unittest.TestCase):
    """真 _emit_train 驱动 + 真 log_decision 捕获(加分项):发射时快照
    序列化 + 发射后置空核销。"""

    def test_emit_train_snapshots_then_clears(self):
        scheduler = _scheduler(_hardware())
        rt = _runtime("session_0_request_0")
        e1 = _eviction(1, "session_0_request_0", time_ns=1_000)
        e2 = _eviction(0, "session_0_request_0", time_ns=2_000)
        rt.decode_target_evictions = (e1, e2)
        rt.prefill_decode_transfer = _transfer()
        state = scheduler.instances[0]
        state.active_decode.append(rt)
        state.active_decode_lookup.add(rt)
        scheduler.runtime_by_request_id[rt.request_id] = rt

        scheduler._emit_train(state, tick=3_000)

        rows = [row for row in scheduler.online_log_rows
                if row["kind"] == "decode"]
        self.assertEqual(len(rows), 1)
        decision = rows[0]["decision"]
        self.assertEqual(rows[0]["request_id"], "session_0_request_0")
        self.assertEqual(rows[0]["tick"], 3_000)
        # 快照 = 发射时点全集(两条逐出,含跨 #2/#4/#5 累积)。
        self.assertEqual(
            [entry["time_ns"]
             for entry in decision["decode_target_evictions"]],
            [1_000, 2_000])
        self.assertEqual(
            [entry["victim_instance_index"]
             for entry in decision["decode_target_evictions"]],
            [1, 0])
        # shard 级路由随 decode 行落盘(问题 4b)。
        self.assertEqual(
            decision["prefill_decode_transfer"]["shards"][0]["noc_hops"], 2)
        # 发射后置空核销(镜像 completion 语义;防 prefill 行 #2/#3 双计)。
        self.assertEqual(rt.decode_target_evictions, ())
        # joiner 账本照常(decode_train_joined 标记)。
        self.assertTrue(rt.decode_train_joined)


class HistoryTransferShardsTest(unittest.TestCase):
    """问题 4b prefill 侧:holder 留存 + shard 级路由序列化。"""

    def test_try_admit_prefill_retains_transfer_shards(self):
        """_try_admit_prefill 取 decision.transfer_shards 赋值
        (此前只取聚合 bytes、shards 被丢弃)。"""
        shards = (KVTransferShard(0, 0, 5, 2000),
                  KVTransferShard(1, 2, 2, 1000))
        kv_manager = SimpleNamespace(
            reserve_request_capacity=lambda *a, **k: CapacityResult(
                (), True, ()),
            session_snapshot=lambda session_id: None,
            hbm_snapshots=lambda instance_index: None,
            prepare_history=lambda *a, **k: HistoryDecision(
                action="NOC_MIGRATE", source_instance_index=1,
                target_instance_index=0, history_tokens=64,
                transfer_shards=shards, recompute_tokens=0,
                evictions=(), admission_blocked=False),
            grow_prefill=lambda *a, **k: CapacityResult((), True, ()),
        )
        scheduler = _scheduler(_hardware())
        scheduler.kv_manager = kv_manager
        rt = _runtime("session_0_request_0")
        self.assertTrue(scheduler._try_admit_prefill(rt, 1_000))
        self.assertEqual(rt.history_transfer_shards, shards)
        self.assertEqual(rt.history_transfer_bytes, 3000)
        # local_hit / ABSENT 路径 transfer_shards=() 自然为空元组。
        fresh = _OnlineRequestRuntime(_record("session_0_request_1"))
        self.assertEqual(fresh.history_transfer_shards, ())

    def test_history_transfer_shard_dicts_hand_computed(self):
        hw = _hardware()
        rows = _history_transfer_shard_dicts(
            hw, (KVTransferShard(0, 0, 5, 2000),
                 KVTransferShard(1, 2, 2, 1000)))
        self.assertEqual(rows, [
            {"relative_tp_rank": 0, "source_rank": 0, "target_rank": 5,
             "bytes": 2000, "noc_path": [0, 1, 5], "noc_hops": 2},
            {"relative_tp_rank": 1, "source_rank": 2, "target_rank": 2,
             "bytes": 1000, "noc_path": [2], "noc_hops": 0},
        ])

    def test_empty_shards_serialize_empty_list(self):
        self.assertEqual(_history_transfer_shard_dicts(_hardware(), ()), [])


class KvTransferDictCompatTest(unittest.TestCase):
    """问题 4b:_kv_transfer_dict 的 hardware=None 向后兼容钉子。"""

    def test_hardware_none_matches_legacy_shape(self):
        """hardware 缺省输出与旧版逐字节一致:shard 恰 4 键、无 noc 字段
        (旧 decision log 回退路径的前提)。"""
        legacy = _kv_transfer_dict(_transfer())
        self.assertEqual(legacy["action"], "NOC_MIGRATE")
        self.assertEqual(legacy["total_bytes"], 3000)
        self.assertEqual(
            legacy["shards"],
            [{"relative_tp_rank": 0, "source_rank": 0, "target_rank": 5,
              "bytes": 2000},
             {"relative_tp_rank": 1, "source_rank": 2, "target_rank": 2,
              "bytes": 1000}])
        for shard in legacy["shards"]:
            self.assertNotIn("noc_path", shard)
            self.assertNotIn("noc_hops", shard)

    def test_none_transfer_serializes_none(self):
        self.assertIsNone(_kv_transfer_dict(None))
        self.assertIsNone(_kv_transfer_dict(None, hardware=_hardware()))

    def test_hardware_appends_route_fields(self):
        routed = _kv_transfer_dict(_transfer(), hardware=_hardware())
        self.assertEqual(routed["shards"][0]["noc_path"], [0, 1, 5])
        self.assertEqual(routed["shards"][0]["noc_hops"], 2)
        self.assertEqual(routed["shards"][1]["noc_path"], [2])
        self.assertEqual(routed["shards"][1]["noc_hops"], 0)


if __name__ == "__main__":
    unittest.main()
