"""Focused parity checks for incremental SessionKVCacheManager invariants."""

from __future__ import annotations

import sys
import unittest
from unittest import mock
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from session_kv_manager import (  # noqa: E402
    PARTIAL_HBM_REMOTE,
    SessionKVCacheManager,
    SessionKVState,
    kv_cache_shard_bytes_for_tokens,
)
from wsc_llm_scheduler import (  # noqa: E402
    DECODE_ROLE,
    PREFILL_ROLE,
    WscLlmHardware,
    WscLlmInstanceSpec,
    WscLlmModel,
    build_instances,
)


def _manager(strict_invariants: bool) -> SessionKVCacheManager:
    hardware = WscLlmHardware(
        mesh_rows=2,
        mesh_cols=2,
        local_hbm_capacity_bytes=10_000,
        local_hbm_bandwidth_gbps=1.0,
        d2d_bandwidth_gbps=2.0,
        peak_perf_tflops=1.0,
        d2d_latency_ns=0,
        local_hbm_latency_ns=0,
    )
    topology = build_instances(
        hardware,
        (
            WscLlmInstanceSpec("prefill", "1", (0, 1), PREFILL_ROLE),
            WscLlmInstanceSpec("decode", "2", (2, 3), DECODE_ROLE),
        ),
    )
    model = WscLlmModel(1, 4, 4, 2, 4, 1, "gelu")
    return SessionKVCacheManager(
        topology,
        model,
        strict_invariants=strict_invariants,
    )


def _tiered_manager(strict_invariants: bool) -> SessionKVCacheManager:
    """二态增量不变量:2 层模型 + 小容量,驱动整体逐出与 REMOTE 全量
    恢复路径的增量账本刷新。

    手算(gelu, heads=2, tp=2 → 每 rank 1 头):权重 128 B/rank;
    KV = 8 B/token/rank(20 token = 160,全层)。容量 400 → 每 rank
    空闲 272:一个 20-token 会话驻留后余 112 < 160,下一个同实例会话
    必然逼出前一会话的整体逐出(逐笔重查,过量释放合法)。"""
    hardware = WscLlmHardware(
        mesh_rows=2,
        mesh_cols=2,
        local_hbm_capacity_bytes=400,
        local_hbm_bandwidth_gbps=1.0,
        d2d_bandwidth_gbps=2.0,
        peak_perf_tflops=1.0,
        d2d_latency_ns=0,
        local_hbm_latency_ns=0,
    )
    topology = build_instances(
        hardware,
        (
            WscLlmInstanceSpec("prefill", "1", (0, 1), PREFILL_ROLE),
            WscLlmInstanceSpec("decode", "2", (2, 3), DECODE_ROLE),
        ),
    )
    model = WscLlmModel(2, 4, 4, 2, 4, 1, "gelu")
    return SessionKVCacheManager(
        topology,
        model,
        strict_invariants=strict_invariants,
    )


class IncrementalInvariantTests(unittest.TestCase):
    def test_incremental_and_strict_mutations_stay_equivalent(self) -> None:
        fast = _manager(False)
        strict = _manager(True)
        required = kv_cache_shard_bytes_for_tokens(fast.model, 4, fast.tp_degree)

        def invoke(name: str, *args: object, **kwargs: object) -> None:
            with mock.patch.object(
                fast,
                "_check_invariants",
                side_effect=AssertionError("normal mutation used the full verifier"),
            ):
                fast_result = getattr(fast, name)(*args, **kwargs)
            self.assertEqual(
                fast_result,
                getattr(strict, name)(*args, **kwargs),
            )
            fast._check_invariants()
            strict._check_invariants()
            self.assertEqual(fast.hbm_snapshots(), strict.hbm_snapshots())
            self.assertEqual(fast.events, strict.events)

        invoke(
            "prepare_history",
            "session",
            0,
            0,
            0,
            "request",
            required_context_tokens=4,
        )
        invoke(
            "reserve_request_capacity",
            "request",
            "session",
            0,
            required,
            0,
        )
        invoke("grow_prefill", "session", 4, 1, "request")
        # P1 ①:release 携带真实 mutation 时刻(与序列时序保持非递减)。
        invoke("release_request_capacity", "request", 1)
        invoke(
            "move_prefill_to_decode",
            "session",
            1,
            2,
            "request",
            final_context_tokens=4,
        )
        invoke("grow_decode", "session", 5, 3, "request")
        invoke("mark_complete", "session", 4, "request")
        invoke("retire_terminal_session", "session", 4, "request")
        fast.assert_final_state()
        strict.assert_final_state()

    def test_incremental_and_full_checks_reject_corrupt_accounting(self) -> None:
        for strict_invariants in (False, True):
            manager = _manager(strict_invariants)
            manager.prepare_history(
                "session",
                0,
                0,
                0,
                "request",
                required_context_tokens=1,
            )
            manager.grow_prefill("session", 1, 1, "request")
            rank = manager.topology.instance(0).ranks[0]
            manager._rank_states[rank].resident_kv_bytes += 1
            with self.assertRaisesRegex(RuntimeError, "session/rank KV accounting mismatch"):
                manager._check_invariants_after_mutation(session_ids=("session",))
            with self.assertRaisesRegex(RuntimeError, "session/rank KV accounting mismatch"):
                manager._check_invariants()

    def test_two_state_mutations_stay_equivalent_and_audit(self) -> None:
        """二态:整体逐出 → REMOTE 跨实例全量回迁 → REMOTE 同实例全量
        回迁 → retire 远端核销;增量/全量两条检查路径等价,且对 remote
        账本的腐蚀双双 fail-closed。PARTIAL 运行态不可达(构造即 raise)。"""
        def drive(manager: SessionKVCacheManager) -> None:
            # a 在实例 0(prefill)完成(160/rank,余 112)。
            manager.prepare_history(
                "a", 0, 0, 1, "a0", required_context_tokens=20)
            manager.grow_prefill("a", 20, 2, "a0")
            manager.mark_complete("a", 3, "a0")
            # b 入场:112 < 160 → 整体逐出 a(腾 160 → 272 ≥ 160 满足即停)。
            decision = manager.prepare_history(
                "b", 0, 0, 4, "b0", required_context_tokens=20)
            assert not decision.admission_blocked, decision
            assert any(t.kind == "remote_store" for t in decision.evictions)
            assert not any(
                t.resident_prefix_layers_after for t in decision.evictions)
            manager.grow_prefill("b", 20, 5, "b0")
            manager.mark_complete("b", 6, "b0")
            # a 下一 turn 在实例 1(decode):REMOTE 跨实例全量回迁
            #(实例 1 各 rank 空闲 272 ≥ 160,无逐出)。
            restore = manager.prepare_history(
                "a", 1, 20, 8, "a1", required_context_tokens=20)
            assert restore.action == "REMOTE_RESTORE", restore.action
            manager.grow_prefill("a", 20, 9, "a1")
            manager.mark_complete("a", 10, "a1")
            # c 入场(实例 0,余 112):整体逐出 b(满足即停)→ b REMOTE。
            decision = manager.prepare_history(
                "c", 0, 0, 12, "c0", required_context_tokens=20)
            assert not decision.admission_blocked
            assert any(t.kind == "remote_store" for t in decision.evictions)
            manager.grow_prefill("c", 20, 13, "c0")
            manager.mark_complete("c", 14, "c0")
            # b 下一 turn(实例 0,余 112):REMOTE 同实例全量回迁,逐出 c
            #(整体,满足即停)。
            remote_restore = manager.prepare_history(
                "b", 0, 20, 16, "b1", required_context_tokens=20)
            assert remote_restore.action == "REMOTE_RESTORE", (
                remote_restore.action)
            manager.grow_prefill("b", 20, 17, "b1")
            manager.mark_complete("b", 18, "b1")
            for session_id, tick, request_id in (
                ("a", 20, "a1"), ("b", 21, "b1"), ("c", 22, "c0"),
            ):
                manager.retire_terminal_session(session_id, tick, request_id)

        fast = _tiered_manager(False)
        strict = _tiered_manager(True)
        drive(fast)
        drive(strict)
        fast._check_invariants()
        strict._check_invariants()
        self.assertEqual(fast.hbm_snapshots(), strict.hbm_snapshots())
        self.assertEqual(fast.remote_bytes_by_rank(),
                         strict.remote_bytes_by_rank())
        self.assertEqual(fast.events, strict.events)
        self.assertEqual(
            set(fast.remote_bytes_by_rank().values()), {0},
            "retire 静默核销后远端账面必须归零")

        # remote 账本腐蚀:全量检查器按会话账本重算并对账,必须 fail-closed。
        manager = _tiered_manager(False)
        drive(manager)
        rank = manager.topology.instance(0).ranks[0]
        manager._remote_bytes_by_rank[rank] += 1
        with self.assertRaisesRegex(RuntimeError, "remote-pool"):
            manager._check_invariants()

    def test_partial_runtime_state_is_unreachable(self) -> None:
        """PARTIAL 不可达的运行时证明:手工构造 0<prefix<L 的会话状态
        (legacy PARTIAL_HBM_REMOTE 位置,或 LOCAL 位置携带部分前缀),
        _check_invariants 与增量检查都必须 raise。"""
        manager = _tiered_manager(False)
        manager.prepare_history(
            "a", 0, 0, 1, "a0", required_context_tokens=20)
        manager.grow_prefill("a", 20, 2, "a0")
        ghost = manager._sessions["a"]

        # legacy PARTIAL_HBM_REMOTE 位置(0 < prefix < L = 2)。
        partial = SessionKVState(
            session_id="ghost",
            location=PARTIAL_HBM_REMOTE,
            instance_index=0,
            logical_context_tokens=20,
            shard_bytes=ghost.shard_bytes,
            resident_prefix_layers=1,
        )
        manager._sessions["ghost"] = partial
        with self.assertRaisesRegex(RuntimeError, "invalid KV location"):
            manager._check_invariants()
        with self.assertRaisesRegex(RuntimeError, "invalid KV location"):
            manager._check_incremental_session_invariants(partial)
        del manager._sessions["ghost"]

        # LOCAL 位置携带部分前缀(0 < prefix < L)同样不可达。
        ghost.resident_prefix_layers = 1
        with self.assertRaisesRegex(RuntimeError, "missing layers"):
            manager._check_invariants()
        with self.assertRaisesRegex(RuntimeError, "missing layers"):
            manager._check_incremental_session_invariants(ghost)
        ghost.resident_prefix_layers = manager.model_layers
        manager._check_invariants()


if __name__ == "__main__":
    unittest.main()
