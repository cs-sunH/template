"""Focused parity checks for incremental SessionKVCacheManager invariants.

Session-level Tiered-LRU (2026-09-25 重写):逐出恒为完整 session 整体
外迁(_evict_session 是唯一逐出路径,半层 _evict_suffix 与 PARTIAL 手工
翻转 fixtures 已随两阶段流程删除);跨实例恢复断言 = REMOTE_RESTORE 全量
回迁;新增规格6 manager 级不可达证明——手工把状态置为 PARTIAL 或非两值
驻留层数 -> _check_invariants fail-closed raise。
"""

from __future__ import annotations

import sys
import unittest
from unittest import mock
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from face_scheduler import FaceHardware, FaceInstanceSpec, FaceModel, build_instances  # noqa: E402
from session_kv_manager import (  # noqa: E402
    LOCAL_HBM,
    PARTIAL_HBM_REMOTE,
    REMOTE_MEMORY,
    SessionKVCacheManager,
    kv_cache_shard_bytes_for_tokens,
)


def _manager(strict_invariants: bool) -> SessionKVCacheManager:
    hardware = FaceHardware(2, 2, 10_000, 1.0, 2.0, 1.0, 0, 0)
    topology = build_instances(
        hardware,
        (
            FaceInstanceSpec("ins0", "1", (0, 1)),
            FaceInstanceSpec("ins1", "2", (2, 3)),
        ),
    )
    model = FaceModel(2, 4, 4, 2, 4, 1, "gelu")
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

        def verify() -> None:
            fast._check_invariants()
            strict._check_invariants()
            self.assertEqual(fast.hbm_snapshots(), strict.hbm_snapshots())
            self.assertEqual(
                fast.session_snapshots(), strict.session_snapshots()
            )

        def invoke(name: str, *args: object, **kwargs: object):
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
            verify()
            return fast_result

        # 无历史 -> LOCAL;净额预占(face 保留 API)。
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
        invoke("release_request_capacity", "request")
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

        # 唯一逐出路径:完整 session 整体外迁 LOCAL -> REMOTE(增量 vs
        # 全量审计逐字段一致;每 victim 恰一笔全量 remote_store)。
        with mock.patch.object(
            fast,
            "_check_invariants",
            side_effect=AssertionError("normal mutation used the full verifier"),
        ):
            fast_evict = fast._evict_session(
                fast._sessions["session"],
                now_ns=4,
                phase="test",
                reason="parity",
                trigger_request_id="request",
            )
        self.assertEqual(
            fast_evict,
            strict._evict_session(
                strict._sessions["session"],
                now_ns=4,
                phase="test",
                reason="parity",
                trigger_request_id="request",
            ),
        )
        self.assertEqual(fast_evict.transfer.kind, "remote_store")
        self.assertEqual(fast_evict.transfer.reason, "parity_full")
        self.assertEqual(
            (fast_evict.transfer.layer_start, fast_evict.transfer.layer_end),
            (0, fast.model.layers),
        )
        verify()

        # 唯一恢复路径:REMOTE 全量回迁(单笔 remote_load [0, L))+ 终局核销。
        restored = invoke(
            "prepare_history",
            "session",
            0,
            5,
            5,
            "final",
        )
        self.assertEqual(restored.action, "REMOTE_RESTORE")
        self.assertEqual(len(restored.transfers), 1)
        transfer = restored.transfers[0]
        self.assertEqual(transfer.kind, "remote_load")
        self.assertEqual(
            (transfer.layer_start, transfer.layer_end), (0, fast.model.layers))
        invoke("mark_complete", "session", 6, "final")
        invoke("retire_terminal_session", "session", 6, "final")
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

    def test_two_state_invariants_reject_invalid_locations(self) -> None:
        """两态白名单/全量层域断言/REMOTE 无实例无驻留层(增量检查器)。"""

        for strict_invariants in (False, True):
            manager = _manager(strict_invariants)
            manager.prepare_history(
                "session",
                0,
                0,
                0,
                "request",
                required_context_tokens=4,
            )
            manager.grow_prefill("session", 4, 1, "request")
            manager.mark_complete("session", 4, "request")
            state = manager._sessions["session"]
            self.assertEqual(state.location, LOCAL_HBM)

            # LOCAL 层数不完整 -> 拒绝。
            state.resident_prefix_layers = manager.model.layers - 1
            with self.assertRaisesRegex(RuntimeError, "fully local"):
                manager._check_invariants_after_mutation(
                    session_ids=("session",))
            state.resident_prefix_layers = manager.model.layers

            # 非法 location -> 拒绝。
            state.location = "somewhere_else"
            with self.assertRaisesRegex(RuntimeError, "unknown KV location"):
                manager._check_invariants_after_mutation(
                    session_ids=("session",))

            # REMOTE 保留实例/驻留层 -> 会话级断言先拒绝(真实的 REMOTE
            # 形态由 _evict_session 路径落账,见首测;手工翻转不搬 rank
            # 字节,走到 rank 对账才会失配——会话级检查先于对账,故此处
            # 断言取会话级消息)。
            state.location = REMOTE_MEMORY
            with self.assertRaisesRegex(RuntimeError, "remote KV session"):
                manager._check_invariants_after_mutation(
                    session_ids=("session",))
            state.resident_prefix_layers = 0
            with self.assertRaisesRegex(RuntimeError, "remote KV session"):
                manager._check_invariants_after_mutation(
                    session_ids=("session",))

            # 恢复与 rank 账本一致的 LOCAL 终态后收尾(真实 REMOTE 形态
            # 由 _evict_session 路径覆盖,见首测)。
            state.location = LOCAL_HBM
            state.resident_prefix_layers = manager.model.layers
            manager._check_invariants_after_mutation(session_ids=("session",))
            manager.retire_terminal_session("session", 4, "request")
            manager.assert_final_state()

    def test_partial_state_is_unreachable(self) -> None:
        """规格6:manager 级不可达证明——手工把 state 置为 legacy 的
        PARTIAL_HBM_REMOTE 或非两值驻留层数,全量审计 fail-closed raise
        (运行态白名单已无 PARTIAL:新代码零产生点)。"""

        manager = _manager(False)
        manager.prepare_history(
            "session", 0, 0, 0, "request", required_context_tokens=4
        )
        manager.grow_prefill("session", 4, 1, "request")
        manager.mark_complete("session", 4, "request")
        state = manager._sessions["session"]

        # legacy PARTIAL 串(旧日志解析专用常量)不在两态白名单内。
        state.location = PARTIAL_HBM_REMOTE
        with self.assertRaisesRegex(RuntimeError, "unknown KV location"):
            manager._check_invariants_after_mutation(
                session_ids=("session",))
        with self.assertRaisesRegex(RuntimeError, "unknown KV location"):
            manager._check_invariants()
        state.location = LOCAL_HBM

        # 非两值驻留层数(半层形态)同样 fail-closed:LOCAL 必须持全层。
        state.resident_prefix_layers = manager.model.layers - 1
        with self.assertRaisesRegex(RuntimeError, "fully local"):
            manager._check_invariants()
        state.resident_prefix_layers = 0
        with self.assertRaisesRegex(RuntimeError, "fully local"):
            manager._check_invariants()
        state.resident_prefix_layers = manager.model.layers

        # REMOTE 会话携带驻留层/实例 -> 拒绝(两态语义的对偶半边)。
        state.location = REMOTE_MEMORY
        state.instance_index = 0
        state.resident_prefix_layers = manager.model.layers
        with self.assertRaisesRegex(RuntimeError, "remote KV session"):
            manager._check_invariants()


if __name__ == "__main__":
    unittest.main()
