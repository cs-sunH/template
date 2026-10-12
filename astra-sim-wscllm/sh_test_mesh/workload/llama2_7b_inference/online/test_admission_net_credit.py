#!/usr/bin/env python3
"""test_admission_net_credit.py -- wscllm 准入预占双重计账修复
(条件净额重试 + 迁移后回补,2026-09-06)回测。

缺陷:`wsc_llm_online_scheduler.py` `_try_admit_prefill` 在静态路由的
decode 实例上预占本会话**终态全量** KV,而容量检查
(`ensure_physical_fit`)不扣减本会话自己已驻留在该实例上的旧 KV,且
protected_sessions 禁止逐出它——旧 KV 要等同一准入流程稍后的
`prepare_history` NOC 迁移才搬去 prefill 实例。于是"旧 + 终态"被重复
计入,超长会话(旧+新 > 预算,新单独 ≤ 预算)在队列尾部永久 deep-gap、
无事件可唤醒重试(C++ 侧 lost-wakeup fail-closed)。

修复语义:
1. 全量预约失败且旧 KV 确实 RESIDENT 于 decode 目标时,按净额
   max(0, 终态 − 旧驻留) 同入口重试(reason 后缀 _net_credit);
2. `prepare_history` 迁移把旧 KV 从 decode 目标删除后,立即
   `extend_request_capacity` 把净额预约回补到全量(防抢占语义:
   从准入占位到 P→D move 完成;resident→reserved 1:1 换位,
   任何瞬间不超订);
3. 旧 KV 在别的实例 / EVICTED / ABSENT → 维持现状(单次全量预约);
4. 净额也失败 → `_note_capacity_change` 覆盖两次尝试的逐出
   (第一次的逐出是真实 mutation,不能丢)。

Run: python3 online/test_admission_net_credit.py
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

from online.wsc_llm_online_scheduler import (  # noqa: E402
    _OnlineInstanceState,
    _OnlineRequestRuntime,
    WscLlmOnlineScheduler,
)
from session_kv_manager import (  # noqa: E402
    NOC_MIGRATE,
    RESIDENT,
    SessionKVCacheManager,
    kv_cache_shard_bytes_for_tokens,
)
from wsc_llm_scheduler import (  # noqa: E402
    DECODE_ROLE,
    WscLlmHardware,
    WscLlmInstanceSpec,
    WscLlmModel,
    build_instances,
)


def _eviction(victim_instance_index, trigger_request_id, time_ns=1_000):
    from session_kv_manager import EvictionRecord

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


# ---------------------------------------------------------------------------
# 桩测:kv_manager 按脚本回放,钉 _try_admit_prefill 的分支语义
# ---------------------------------------------------------------------------

_STUB_MODEL = WscLlmModel(
    layers=32, hidden_size=4096, ffn_size=11008, num_heads=32,
    vocab_size=32000, bytes_per_elem=2)
_FINAL_SHARDS = kv_cache_shard_bytes_for_tokens(
    _STUB_MODEL, 516, 1)  # tp=1 桩拓扑 → 单 rank 分片


class _StubKVManager:
    """桩 kv_manager:按脚本依次回放 reservation / snapshot / history 决策,
    记录调用序供断言(镜像 test_admission_eviction_accumulation.py 桩风格)。
    snapshots 按调用序弹出(session_snapshot 在净额分支与
    history_cache_state_before 各消费一次)。"""

    def __init__(self, reservations, history_decisions, snapshots=()):
        self._reservations = list(reservations)
        self._history = list(history_decisions)
        self._snapshots = list(snapshots)
        self.calls = []

    def reserve_request_capacity(self, *args, **kwargs):
        self.calls.append(("reserve_request_capacity", args, kwargs))
        return self._reservations.pop(0)

    def release_request_capacity(self, *args, **kwargs):
        self.calls.append(("release_request_capacity", args, kwargs))

    def extend_request_capacity(self, *args, **kwargs):
        self.calls.append(("extend_request_capacity", args, kwargs))

    def session_snapshot(self, session_id):
        self.calls.append(("session_snapshot", (session_id,), {}))
        return self._snapshots.pop(0) if self._snapshots else None

    def hbm_snapshots(self, instance_index):
        return None

    def prepare_history(self, *args, **kwargs):
        self.calls.append(("prepare_history", args, kwargs))
        return self._history.pop(0)

    def grow_prefill(self, *args, **kwargs):
        self.calls.append(("grow_prefill", args, kwargs))
        from session_kv_manager import CapacityResult

        return CapacityResult((), True, ())


def _runtime():
    record = {
        "request_id": "session_0_request_1",
        "session_id": "session_0",
        "turn_index": 1,
        "queue_index": 1,
        "prefill_length": 30,
        "decode_length": 5,
        "history_tokens_before": 512,
        "prefill_context_tokens": 512,
        "final_context_tokens": 516,
    }
    runtime = _OnlineRequestRuntime(record)
    runtime.prefill_instance_index = 1
    runtime.static_route = SimpleNamespace(decode_instance_index=0)
    return runtime


def _scheduler(kv_manager):
    # __new__ 跳过 __init__,只补 _try_admit_prefill 触达的字段
    # (test_train_machinery.py:71 同法)。
    scheduler = WscLlmOnlineScheduler.__new__(WscLlmOnlineScheduler)
    scheduler.instances = [
        _OnlineInstanceState(index=0, phase_role=DECODE_ROLE),
        _OnlineInstanceState(index=1, phase_role=DECODE_ROLE),
    ]
    scheduler.runtime_by_request_id = {}
    scheduler.capacity_epoch = [0, 0, 0, 0]
    scheduler.kv_manager = kv_manager
    scheduler.config = SimpleNamespace(model=_STUB_MODEL)
    scheduler.topology = SimpleNamespace(instances=[SimpleNamespace(size=1)])
    return scheduler


def _reserve_calls(kv_manager):
    return [call for call in kv_manager.calls
            if call[0] == "reserve_request_capacity"]


def _extend_calls(kv_manager):
    return [call for call in kv_manager.calls
            if call[0] == "extend_request_capacity"]


class NetCreditStubTests(unittest.TestCase):
    def test_full_fail_net_retry_success_backfills(self):
        """全量失败 + 旧 KV RESIDENT 于 decode → 净额重试成功 →
        prepare_history 迁移后以 credit 回补到全量,准入放行。"""
        from session_kv_manager import CapacityResult, HistoryDecision

        e1 = _eviction(0, "session_0_request_1", time_ns=1_000)
        e2 = _eviction(0, "session_0_request_1", time_ns=1_000)
        credit = (200_000_000,)
        kv_manager = _StubKVManager(
            reservations=[
                CapacityResult((e1,), False, (0,)),   # 全量:deep-gap 失败
                CapacityResult((e2,), True, ()),      # 净额:成功
            ],
            history_decisions=[
                HistoryDecision(
                    action=NOC_MIGRATE, source_instance_index=0,
                    target_instance_index=1, history_tokens=512,
                    transfer_shards=(), recompute_tokens=0,
                    evictions=(), admission_blocked=False),
            ],
            snapshots=[
                SimpleNamespace(state=RESIDENT, instance_index=0,
                                shard_bytes=credit),
                None,  # history_cache_state_before 快照(桩口径可为 None)
            ],
        )
        scheduler = _scheduler(kv_manager)
        runtime = _runtime()

        self.assertTrue(scheduler._try_admit_prefill(runtime, 1_000))

        # 两次预约:第一次全量原参,第二次 shards==max(0, final-credit)、
        # reason 换净额后缀、phase 不变。
        reserves = _reserve_calls(kv_manager)
        self.assertEqual(len(reserves), 2)
        self.assertEqual(reserves[0][1][3], _FINAL_SHARDS)
        self.assertEqual(
            reserves[0][2].get("reason"),
            "static_decode_final_kv_reservation")
        self.assertEqual(
            reserves[1][1][3],
            (max(0, _FINAL_SHARDS[0] - credit[0]),))
        self.assertEqual(
            reserves[1][2].get("reason"),
            "static_decode_final_kv_reservation_net_credit")
        self.assertEqual(
            reserves[1][2].get("phase"), "prefill_admission")

        # prepare_history 被调,extend 以 credit 回补,credit 核销。
        self.assertIn(
            ("prepare_history",
             ("session_0", 1, 512, 1_000, "session_0_request_1"),
             {"required_context_tokens": 512}),
            kv_manager.calls)
        self.assertEqual(len(_extend_calls(kv_manager)), 1)
        extend_args, extend_kwargs = (_extend_calls(kv_manager)[0][1],
                                      _extend_calls(kv_manager)[0][2])
        self.assertEqual(extend_args[0], "session_0_request_1")
        self.assertEqual(extend_args[1], credit)
        self.assertEqual(extend_args[2], 1_000)
        self.assertEqual(
            extend_kwargs.get("reason"), "history_vacated_decode_target")
        self.assertIsNone(runtime.reservation_credit)
        self.assertTrue(runtime.admitted_prefill)
        # 两次尝试的逐出都累积(decode_target_evictions 钉累积语义)。
        self.assertEqual(runtime.decode_target_evictions, (e1, e2))

    def test_net_retry_clamps_per_rank_credit(self):
        """净额按 rank 钳位:credit 某分量 > 终态时该 rank 净额为 0
        (max(0, ·) 防御,不产生负分片)。"""
        from session_kv_manager import CapacityResult, HistoryDecision

        credit = (300_000_000,)  # > final(单 rank)→ 净额钳位为 0
        kv_manager = _StubKVManager(
            reservations=[
                CapacityResult((), False, (0,)),
                CapacityResult((), True, ()),
            ],
            history_decisions=[
                HistoryDecision(
                    action=NOC_MIGRATE, source_instance_index=0,
                    target_instance_index=1, history_tokens=512,
                    transfer_shards=(), recompute_tokens=0,
                    evictions=(), admission_blocked=False),
            ],
            snapshots=[
                SimpleNamespace(state=RESIDENT, instance_index=0,
                                shard_bytes=credit),
                None,
            ],
        )
        scheduler = _scheduler(kv_manager)
        self.assertTrue(scheduler._try_admit_prefill(_runtime(), 1_000))
        reserves = _reserve_calls(kv_manager)
        self.assertEqual(reserves[1][1][3], (0,))
        self.assertEqual(_extend_calls(kv_manager)[0][1][1], credit)

    def test_net_retry_failure_notes_both_attempts(self):
        """净额也失败 → return False,且 `_note_capacity_change` 合并覆盖
        两次尝试的逐出(第一次的逐出是真实 mutation,不能丢)。"""
        from session_kv_manager import CapacityResult

        e1 = _eviction(2, "session_0_request_1", time_ns=1_000)
        e2 = _eviction(3, "session_0_request_1", time_ns=1_000)
        kv_manager = _StubKVManager(
            reservations=[
                CapacityResult((e1,), False, (0,)),
                CapacityResult((e2,), False, (0,)),
            ],
            history_decisions=[],
            snapshots=[
                SimpleNamespace(state=RESIDENT, instance_index=0,
                                shard_bytes=(100_000_000,)),
            ],
        )
        scheduler = _scheduler(kv_manager)
        runtime = _runtime()

        self.assertFalse(scheduler._try_admit_prefill(runtime, 1_000))
        # 两次尝试的逐出 victim 实例(2 与 3)epoch 都被唤醒。
        self.assertEqual(scheduler.capacity_epoch[2], 1)
        self.assertEqual(scheduler.capacity_epoch[3], 1)
        self.assertEqual(runtime.decode_target_evictions, (e1, e2))
        self.assertFalse(runtime.admitted_prefill)
        self.assertIsNone(runtime.reservation_credit)

    def test_snapshot_off_decode_target_keeps_single_full_reserve(self):
        """旧 KV 不在 decode 目标(别的实例)/ EVICTED / ABSENT →
        现状行为:仅一次全量预约,失败即 False,不回补。"""
        from session_kv_manager import CapacityResult

        for snapshot in (
            SimpleNamespace(state=RESIDENT, instance_index=1,
                            shard_bytes=(200_000_000,)),  # 别的实例
            SimpleNamespace(state="EVICTED", instance_index=None,
                            shard_bytes=(0,)),            # 已逐出
            None,                                          # ABSENT
        ):
            kv_manager = _StubKVManager(
                reservations=[CapacityResult((), False, (0,))],
                history_decisions=[],
                snapshots=[snapshot],
            )
            scheduler = _scheduler(kv_manager)
            runtime = _runtime()
            self.assertFalse(
                scheduler._try_admit_prefill(runtime, 1_000),
                msg=f"snapshot={snapshot!r}")
            self.assertEqual(len(_reserve_calls(kv_manager)), 1)
            self.assertEqual(
                _reserve_calls(kv_manager)[0][2].get("reason"),
                "static_decode_final_kv_reservation")  # 全量口径不变
            self.assertEqual(len(_extend_calls(kv_manager)), 0)
            self.assertIsNone(runtime.reservation_credit)

    def test_net_success_then_history_blocked_releases_and_clears(self):
        """净额成功但 prepare_history blocked → release(pop 语义按登记
        净额对称释放)且 credit 清空,防跨 attempt 残留。"""
        from session_kv_manager import CapacityResult, HistoryDecision

        kv_manager = _StubKVManager(
            reservations=[
                CapacityResult((), False, (0,)),
                CapacityResult((), True, ()),
            ],
            history_decisions=[
                HistoryDecision(
                    action=NOC_MIGRATE, source_instance_index=0,
                    target_instance_index=1, history_tokens=512,
                    transfer_shards=(), recompute_tokens=0,
                    evictions=(), admission_blocked=True),
            ],
            snapshots=[
                SimpleNamespace(state=RESIDENT, instance_index=0,
                                shard_bytes=(200_000_000,)),
                None,
            ],
        )
        scheduler = _scheduler(kv_manager)
        runtime = _runtime()

        self.assertFalse(scheduler._try_admit_prefill(runtime, 1_000))
        self.assertIn(
            ("release_request_capacity",
             ("session_0_request_1", 1_000), {}), kv_manager.calls)
        self.assertEqual(len(_extend_calls(kv_manager)), 0)
        self.assertIsNone(runtime.reservation_credit)
        self.assertFalse(runtime.decode_capacity_reserved)

    def test_credit_with_mismatched_migration_source_fails_closed(self):
        """credit>0 蕴涵迁移源==decode;失步(如 LOCAL_HIT 源=prefill)
        即内部错误,随仓 fail-closed 风格 raise。"""
        from session_kv_manager import CapacityResult, HistoryDecision

        kv_manager = _StubKVManager(
            reservations=[
                CapacityResult((), False, (0,)),
                CapacityResult((), True, ()),
            ],
            history_decisions=[
                HistoryDecision(
                    action="local_hit", source_instance_index=1,
                    target_instance_index=1, history_tokens=512,
                    transfer_shards=(), recompute_tokens=0,
                    evictions=(), admission_blocked=False),
            ],
            snapshots=[
                SimpleNamespace(state=RESIDENT, instance_index=0,
                                shard_bytes=(200_000_000,)),
                None,
            ],
        )
        scheduler = _scheduler(kv_manager)
        with self.assertRaises(RuntimeError):
            scheduler._try_admit_prefill(_runtime(), 1_000)


# ---------------------------------------------------------------------------
# 真 manager 不变量测:小拓扑 + 小容量,全流程不变量干净
# ---------------------------------------------------------------------------

# 拓扑:mesh 2×4,tp4 双实例(instance0=prefill ranks 0-3,
# instance1=decode ranks 4-7);模型 heads=2 → 每 rank 头数 (1,1,0,0),
# rank2/3 为零头 rank(KV 恒 0——净额在该 rank 天然钳位为 0 的边界)。
_C1_TOKENS = 70   # 旧驻留:shards (560,560,0,0)
_C2_TOKENS = 105  # 终态:shards (840,840,0,0);旧+终态 ≈ 预算的 150%


def _real_manager():
    hardware = WscLlmHardware(
        mesh_rows=2,
        mesh_cols=4,
        local_hbm_capacity_bytes=1_000,
        local_hbm_bandwidth_gbps=1.0,
        d2d_bandwidth_gbps=2.0,
        peak_perf_tflops=1.0,
        d2d_latency_ns=0,
        local_hbm_latency_ns=0,
    )
    topology = build_instances(
        hardware,
        (
            WscLlmInstanceSpec("prefill", "1", (0, 1, 2, 3), "prefill"),
            WscLlmInstanceSpec("decode", "2", (4, 5, 6, 7), "decode"),
        ),
    )
    model = WscLlmModel(1, 4, 4, 2, 4, 2, "gelu")
    return SessionKVCacheManager(topology, model, strict_invariants=True)


def _seed_turn_one(manager):
    """turn-1 生命周期:会话完成后旧 KV RESIDENT/inactive 留在 decode。"""
    manager.prepare_history(
        "session_0", 0, 0, 10, "session_0_request_0",
        required_context_tokens=_C1_TOKENS)
    manager.grow_prefill("session_0", _C1_TOKENS, 11, "session_0_request_0")
    manager.move_prefill_to_decode(
        "session_0", 1, 12, "session_0_request_0",
        final_context_tokens=_C1_TOKENS)
    manager.mark_complete("session_0", 13, "session_0_request_0")
    snapshot = manager.session_snapshot("session_0")
    assert snapshot is not None and snapshot.state == RESIDENT
    assert snapshot.instance_index == 1
    return snapshot


class NetCreditRealManagerTests(unittest.TestCase):
    def test_full_reserve_deep_gaps_net_reserve_fits_and_backfills(self):
        """旧 KV RESIDENT 于 decode、终态使"旧+新"≈预算 150%:
        全量 reserve deep-gap 失败 → 净额成功(不变量干净)→ 迁移删旧
        驻留 → extend 回补到全量 → release 后 reserved 归零。"""
        manager = _real_manager()
        _seed_turn_one(manager)
        decode_ranks = manager.topology.instance(1).ranks
        final_shards = kv_cache_shard_bytes_for_tokens(
            manager.model, _C2_TOKENS, manager.tp_degree)
        self.assertEqual(final_shards, (840, 840, 0, 0))
        deep_gaps_before = manager.deep_gap_events

        # ① 全量预约:旧(560)+终态(840) 双重计入,rank4/5 超容量,
        #    候选仅本会话(protected)→ deep-gap 优雅推迟。
        full = manager.reserve_request_capacity(
            "session_0_request_1", "session_0", 1, final_shards, 20,
            phase="prefill_admission",
            reason="static_decode_final_kv_reservation")
        self.assertFalse(full.admitted)
        self.assertEqual(manager.deep_gap_events, deep_gaps_before + 1)

        # ② 净额预约:resident(旧)+reserved(净额)= 终态,不变量干净。
        snap = manager.session_snapshot("session_0")
        credit = tuple(snap.shard_bytes)
        self.assertEqual(credit, (560, 560, 0, 0))
        net_shards = tuple(
            max(0, final - have)
            for final, have in zip(final_shards, credit))
        self.assertEqual(net_shards, (280, 280, 0, 0))  # 零头 rank 净额 0
        net = manager.reserve_request_capacity(
            "session_0_request_1", "session_0", 1, net_shards, 20,
            phase="prefill_admission",
            reason="static_decode_final_kv_reservation_net_credit")
        self.assertTrue(net.admitted)
        manager._check_invariants()
        for rank_snapshot in manager.hbm_snapshots(1):
            self.assertLessEqual(
                rank_snapshot.used_bytes, rank_snapshot.capacity_bytes)

        # ③ prepare_history 迁移:旧驻留从 decode 删除(源==decode)。
        decision = manager.prepare_history(
            "session_0", 0, _C1_TOKENS, 20, "session_0_request_1",
            required_context_tokens=100)
        self.assertEqual(decision.action, NOC_MIGRATE)
        self.assertEqual(decision.source_instance_index, 1)
        decode_resident = {
            snapshot.rank: snapshot.resident_kv_bytes
            for snapshot in manager.hbm_snapshots(1)
        }
        self.assertEqual(
            [decode_resident[rank] for rank in decode_ranks],
            [0, 0, 0, 0])

        # ④ 回补:reservation.shard_bytes == 终态全量;不变量不 raise;
        #    各 rank used ≤ capacity(resident→reserved 1:1 换位)。
        extended = manager.extend_request_capacity(
            "session_0_request_1", credit, 20,
            reason="history_vacated_decode_target")
        self.assertEqual(extended.shard_bytes, final_shards)
        manager._check_invariants_after_mutation(
            reservation_ids=("session_0_request_1",))
        manager._check_invariants()
        for rank_snapshot in manager.hbm_snapshots(1):
            self.assertLessEqual(
                rank_snapshot.used_bytes, rank_snapshot.capacity_bytes)

        # ⑤ release 按登记值(=全量)对称释放,reserved 归零。
        released = manager.release_request_capacity(
            "session_0_request_1", 21)
        self.assertEqual(released.shard_bytes, final_shards)
        for rank_snapshot in manager.hbm_snapshots(1):
            self.assertEqual(rank_snapshot.reserved_request_bytes, 0)
        self.assertEqual(manager._reservations, {})

    def test_scheduler_net_credit_flow_on_real_manager(self):
        """端到端:真 manager 上跑 `_try_admit_prefill` —— 全量失败自动
        净额重试成功、迁移后回补;准入完成时 decode 侧预约 == 终态全量
        (防抢占语义);release+move 证明空间确实被占住。"""
        manager = _real_manager()
        _seed_turn_one(manager)
        scheduler = WscLlmOnlineScheduler.__new__(WscLlmOnlineScheduler)
        scheduler.instances = [
            _OnlineInstanceState(index=0, phase_role="prefill"),
            _OnlineInstanceState(index=1, phase_role=DECODE_ROLE),
        ]
        scheduler.runtime_by_request_id = {}
        scheduler.capacity_epoch = [0, 0]
        scheduler.kv_manager = manager
        scheduler.config = SimpleNamespace(model=manager.model)
        scheduler.topology = manager.topology

        record = {
            "request_id": "session_0_request_1",
            "session_id": "session_0",
            "turn_index": 1,
            "queue_index": 1,
            "prefill_length": 30,
            "decode_length": 5,
            "history_tokens_before": _C1_TOKENS,
            "prefill_context_tokens": 100,
            "final_context_tokens": _C2_TOKENS,
        }
        runtime = _OnlineRequestRuntime(record)
        runtime.prefill_instance_index = 0
        runtime.static_route = SimpleNamespace(decode_instance_index=1)

        self.assertTrue(scheduler._try_admit_prefill(runtime, 20))
        self.assertEqual(runtime.history_action, NOC_MIGRATE)
        self.assertIsNone(runtime.reservation_credit)
        final_shards = kv_cache_shard_bytes_for_tokens(
            manager.model, _C2_TOKENS, manager.tp_degree)
        # 回补后 decode 各 rank 预约 == 终态全量(560 净额 + 560 credit)。
        reserved_by_rank = {
            snapshot.rank: snapshot.reserved_request_bytes
            for snapshot in manager.hbm_snapshots(1)
        }
        self.assertEqual(
            [reserved_by_rank[rank]
             for rank in manager.topology.instance(1).ranks],
            list(final_shards))
        manager._check_invariants()

        # P→D 边界:release 后 move 需要 decode 侧全量空间——预约确实
        # 占住了它(move 无需逐出),随后生命周期收尾不变量干净。
        manager.release_request_capacity("session_0_request_1", 21)
        move = manager.move_prefill_to_decode(
            "session_0", 1, 22, "session_0_request_1",
            final_context_tokens=_C2_TOKENS)
        self.assertFalse(move.admission_blocked)
        self.assertEqual(move.evictions, ())
        manager.mark_complete("session_0", 23, "session_0_request_1")
        manager._check_invariants()


if __name__ == "__main__":
    unittest.main()
