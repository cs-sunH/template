#!/usr/bin/env python3
"""test_decode_eviction_serialization.py -- wscllm decode/prefill 决策的
逐出与传输序列化钉子(B2 三态改写,2026-09)。

B2 接口变化(契约 §3/§4):逐出对象从 EvictionRecord 换 kind 载体的
KVTransfer(remote_store),行结构 = _transfer_entry_rows 同款
(kind/reason/session_id/total_bytes/source_instance_index/
target_instance_index/layer_start/layer_end);prefill_decode_transfer 与
history 恢复/迁移同为 KVTransfer(shard 级 noc_path 构造期已定)。

覆盖:
  1. _decode_decision_dict:decode_target_evictions 按契约 §3 行结构
     逐条序列化;prefill_decode_transfer 携带层域元数据与 shard 路由;
  2. 真 _emit_train 驱动 + 真 log_decision 捕获:decode 行含新字段、
     发射后 runtime.decode_target_evictions 置空;
  3. _try_admit_prefill 留存 decision.transfers(holder 模式;
     REMOTE_RESTORE 全量恢复 = 单段 remote_load);
  4. _kv_transfer_rows / _eviction_source_instances 手算钉子;
  5. _kv_transfer_dict(None) 透传 None。

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

from online.wsc_llm_online_scheduler import (  # noqa: E402
    _eviction_source_instances,
    _kv_transfer_dict,
    _kv_transfer_rows,
    _OnlineInstanceState,
    _OnlineRequestRuntime,
    WscLlmOnlineScheduler,
)
from session_kv_manager import (  # noqa: E402
    CapacityResult,
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
    """remote_store 逐出桩(source_instance_index = 受害实例;整体逐出
    形态:层域 [0, L)、本地驻留清零)。"""
    return KVTransfer(
        kind="remote_store",
        phase="prefill_admission",
        reason="static_decode_final_kv_reservation_session",
        session_id=f"victim_session_{victim_instance_index}",
        trigger_request_id=trigger_request_id,
        source_instance_index=victim_instance_index,
        target_instance_index=None,
        total_bytes=0,
        shards=(),
        model_layers=32,
        layer_start=0,
        layer_end=32,
        resident_prefix_layers_before=32,
        resident_prefix_layers_after=0,
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
    # rank 2 -> 2(同 rank:path=[2],hops=0);层域 [0, 32)。
    return KVTransfer(
        kind="noc_migrate", phase="prefill_decode",
        reason="prefill_decode_instance_migrate", session_id="session_0",
        trigger_request_id=trigger_request_id,
        source_instance_index=1, target_instance_index=0,
        total_bytes=3000,
        shards=(
            KVTransferShard(0, 5, None, 2000, (0, 1, 5), 0, 32),
            KVTransferShard(2, 2, None, 1000, (2,), 0, 32),
        ),
        model_layers=32, layer_start=0, layer_end=32,
        resident_prefix_layers_before=32,
        resident_prefix_layers_after=32)


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
    scheduler.online_log_rows = []
    scheduler.online_log_count = 0
    scheduler.decision_log_sink = None
    # B3(2026-09-06):发射账本镜像 no-op(sync_pending_history_after_
    # evictions;逐出序列化断言不受影响)。
    scheduler.graph = SimpleNamespace(
        sync_pending_history_after_evictions=lambda transfers: None,
        emit_iteration_train=lambda plan: {
            "exit_members": {
                member["request_id"]: [member["request_id"]]
                for member in plan["exit_members"]},
            "sentinel_members": []})
    return scheduler


class DecodeDecisionDictTest(unittest.TestCase):
    """_decode_decision_dict 直接单测(问题 2A 核心,B2 三态形状)。"""

    def setUp(self):
        self.scheduler = _scheduler(_hardware())

    def test_serializes_decode_target_evictions(self):
        e = _eviction(0, "session_0_request_0", time_ns=1_500)
        rt = _runtime("session_0_request_0")
        rt.decode_target_evictions = (e,)
        decision = self.scheduler._decode_decision_dict(
            rt, rt.static_route)
        # 契约 §3 行结构逐字段。
        self.assertEqual(decision["decode_target_evictions"], [{
            "kind": "remote_store",
            "reason": "static_decode_final_kv_reservation_session",
            "session_id": "victim_session_0",
            "total_bytes": 0,
            "source_instance_index": 0,
            "target_instance_index": None,
            "layer_start": 0,
            "layer_end": 32,
        }])

    def test_empty_decode_target_evictions_serializes_empty_list(self):
        rt = _runtime("session_0_request_0")
        decision = self.scheduler._decode_decision_dict(
            rt, rt.static_route)
        self.assertEqual(decision["decode_target_evictions"], [])

    def test_prefill_decode_transfer_carries_layer_domain(self):
        """B2:prefill_decode_transfer 为 kind 载体 KVTransfer,含层域
        元数据与 shard 级 XY 路由(noc_path/noc_hops 构造期已定)。"""
        rt = _runtime("session_0_request_0")
        rt.prefill_decode_transfer = _transfer()
        decision = self.scheduler._decode_decision_dict(
            rt, rt.static_route)
        self.assertEqual(decision["decode_instance_index"], 0)
        self.assertEqual(
            decision["static_route"],
            {"prefill_instance_index": 1, "decode_instance_index": 0,
             "path": [1, 0], "hop_count": 1, "shared_edges": [[1, 0]]})
        transfer = decision["prefill_decode_transfer"]
        self.assertEqual(transfer["kind"], "noc_migrate")
        self.assertEqual(transfer["total_bytes"], 3000)
        self.assertEqual(transfer["layer_start"], 0)
        self.assertEqual(transfer["layer_end"], 32)
        self.assertEqual(transfer["model_layers"], 32)
        self.assertEqual(transfer["shards"][0]["noc_path"], [0, 1, 5])
        self.assertEqual(transfer["shards"][0]["noc_hops"], 2)
        self.assertEqual(transfer["shards"][1]["noc_path"], [2])
        self.assertEqual(transfer["shards"][1]["noc_hops"], 0)


class EmitTrainSerializationTest(unittest.TestCase):
    """真 _emit_train 驱动 + 真 log_decision 捕获:发射时快照序列化 +
    发射后置空核销。"""

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
        # 快照 = 发射时点全集(两条逐出,含跨 #2/#4/#5 累积),行内携带
        # 受害实例与层域。
        self.assertEqual(
            [entry["source_instance_index"]
             for entry in decision["decode_target_evictions"]],
            [1, 0])
        self.assertTrue(all(
            entry["kind"] == "remote_store"
            for entry in decision["decode_target_evictions"]))
        # shard 级路由随 decode 行落盘。
        self.assertEqual(
            decision["prefill_decode_transfer"]["shards"][0]["noc_hops"], 2)
        # 发射后置空核销(镜像 completion 语义;防 prefill 行 #2/#3 双计)。
        self.assertEqual(rt.decode_target_evictions, ())
        # joiner 账本照常(decode_train_joined 标记)。
        self.assertTrue(rt.decode_train_joined)


class HistoryTransfersTest(unittest.TestCase):
    """B2 prefill 侧:holder 留存 + 契约行结构序列化。"""

    def test_try_admit_prefill_retains_history_transfers(self):
        """_try_admit_prefill 取 decision.transfers 赋值(REMOTE_RESTORE
        全量恢复 = 单段 remote_load)。"""
        restore = KVTransfer(
            kind="remote_load", phase="history",
            reason="history_remote_restore", session_id="session_0",
            trigger_request_id="session_0_request_0",
            source_instance_index=None, target_instance_index=0,
            total_bytes=3000,
            shards=(KVTransferShard(1, 0, 1, 3000, (1, 0), 0, 32),),
            model_layers=32, layer_start=0, layer_end=32,
            resident_prefix_layers_before=0,
            resident_prefix_layers_after=32)
        kv_manager = SimpleNamespace(
            reserve_request_capacity=lambda *a, **k: CapacityResult(
                (), True, ()),
            session_snapshot=lambda session_id: None,
            hbm_snapshots=lambda instance_index: None,
            prepare_history=lambda *a, **k: HistoryDecision(
                action="REMOTE_RESTORE", source_instance_index=None,
                target_instance_index=0, history_tokens=64,
                transfers=(restore,),
                evictions=(), admission_blocked=False),
            grow_prefill=lambda *a, **k: CapacityResult((), True, ()),
        )
        scheduler = _scheduler(_hardware())
        scheduler.kv_manager = kv_manager
        rt = _runtime("session_0_request_0")
        self.assertTrue(scheduler._try_admit_prefill(rt, 1_000))
        self.assertEqual(rt.history_transfers, (restore,))
        self.assertEqual(rt.history_transfer_bytes, 3000)
        self.assertEqual(rt.history_location_before, None)
        # local_hit / ABSENT 路径 transfers=() 自然为空元组。
        fresh = _OnlineRequestRuntime(_record("session_0_request_1"))
        self.assertEqual(fresh.history_transfers, ())

    def test_kv_transfer_rows_hand_computed(self):
        rows = _kv_transfer_rows((_transfer(), None))
        self.assertEqual(rows, [{
            "kind": "noc_migrate",
            "reason": "prefill_decode_instance_migrate",
            "session_id": "session_0",
            "total_bytes": 3000,
            "source_instance_index": 1,
            "target_instance_index": 0,
            "layer_start": 0,
            "layer_end": 32,
        }])

    def test_empty_transfers_serialize_empty_list(self):
        self.assertEqual(_kv_transfer_rows(()), [])
        self.assertEqual(_kv_transfer_rows(None), [])

    def test_eviction_source_instances_epoch_wiring(self):
        """纪元唤醒接线:逐出 KVTransfer 的受害实例 = source_instance_
        index(合并覆盖多源唤醒的输入)。"""
        e1 = _eviction(2, "r")
        e2 = _eviction(3, "r")
        self.assertEqual(
            _eviction_source_instances((e1, e2)), (2, 3))
        self.assertEqual(_eviction_source_instances(()), ())


class KvTransferDictCompatTest(unittest.TestCase):
    """_kv_transfer_dict 的 None 透传与 shard 行形状。"""

    def test_none_transfer_serializes_none(self):
        self.assertIsNone(_kv_transfer_dict(None))

    def test_shard_rows_shape(self):
        routed = _kv_transfer_dict(_transfer())
        self.assertEqual(
            routed["shards"],
            [{"source_rank": 0, "target_rank": 5, "edge_rank": None,
              "bytes": 2000, "noc_path": [0, 1, 5], "noc_hops": 2,
              "layer_start": 0, "layer_end": 32},
             {"source_rank": 2, "target_rank": 2, "edge_rank": None,
              "bytes": 1000, "noc_path": [2], "noc_hops": 0,
              "layer_start": 0, "layer_end": 32}])
        # JSON 可序列化(决策日志通道)。
        json.dumps(routed)


if __name__ == "__main__":
    unittest.main()
