"""Focused parity checks for incremental SessionKVCacheManager invariants.

B2 (2026-09-06): 内部检查器扩三态(策略 §8)——模型换 2 层以覆盖半层
逐出/恢复路径;parity 序列补 _evict_suffix/_evict_session 与 PARTIAL/
REMOTE 恢复分支,对齐 sh_2.0 test_sh20_kv_incremental_invariants 的
覆盖形状(face 事件流/失败语义保留)。
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
    # B2: 2 层模型 -> partial_resident_prefix_layers = 1,半层逐出可触发。
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

        # B2:阶段1 半层逐出 LOCAL -> PARTIAL(增量 vs 全量审计逐字段一致)。
        with mock.patch.object(
            fast,
            "_check_invariants",
            side_effect=AssertionError("normal mutation used the full verifier"),
        ):
            fast_suffix = fast._evict_suffix(
                fast._sessions["session"],
                now_ns=4,
                phase="test",
                reason="parity",
                trigger_request_id="request",
            )
        self.assertEqual(
            fast_suffix,
            strict._evict_suffix(
                strict._sessions["session"],
                now_ns=4,
                phase="test",
                reason="parity",
                trigger_request_id="request",
            ),
        )
        verify()

        # PARTIAL 跨实例恢复 = 两段链(前缀 NoC 迁移 + 后缀远端回迁)。
        # 会话经 move_prefill_to_decode 驻留实例 1,目标换实例 0。
        partial = invoke(
            "prepare_history",
            "session",
            0,
            5,
            5,
            "next",
            required_context_tokens=5,
        )
        self.assertEqual(partial.action, "PARTIAL_REMOTE_MIGRATE")
        self.assertEqual(len(partial.transfers), 2)
        invoke("mark_complete", "session", 5, "next")

        # B2:阶段2 整体外迁(PARTIAL -> REMOTE)。
        with mock.patch.object(
            fast,
            "_check_invariants",
            side_effect=AssertionError("normal mutation used the full verifier"),
        ):
            fast_evict = fast._evict_session(
                fast._sessions["session"],
                now_ns=5,
                phase="test",
                reason="parity",
                trigger_request_id="next",
            )
        self.assertEqual(
            fast_evict,
            strict._evict_session(
                strict._sessions["session"],
                now_ns=5,
                phase="test",
                reason="parity",
                trigger_request_id="next",
            ),
        )
        verify()

        # REMOTE 全量回迁 + 终态核销。
        restored = invoke(
            "prepare_history",
            "session",
            0,
            5,
            6,
            "final",
        )
        self.assertEqual(restored.action, "REMOTE_RESTORE")
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

    def test_three_state_invariants_reject_invalid_locations(self) -> None:
        """B2:三态白名单/层域断言/REMOTE 无实例无驻留层(增量检查器)。"""

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

            # PARTIAL 层域非法(prefix == L 不是 PARTIAL) -> 拒绝。
            state.location = PARTIAL_HBM_REMOTE
            state.instance_index = 0
            state.resident_prefix_layers = manager.model.layers
            with self.assertRaisesRegex(RuntimeError, "invalid prefix length"):
                manager._check_invariants_after_mutation(
                    session_ids=("session",))

            # 恢复与 rank 账本一致的 LOCAL 终态后收尾(PARTIAL 真实形态
            # 需伴随半层字节离账,由 _evict_suffix 路径覆盖,见首测)。
            state.location = LOCAL_HBM
            state.resident_prefix_layers = manager.model.layers
            manager._check_invariants_after_mutation(session_ids=("session",))
            manager.retire_terminal_session("session", 4, "request")
            manager.assert_final_state()


if __name__ == "__main__":
    unittest.main()
