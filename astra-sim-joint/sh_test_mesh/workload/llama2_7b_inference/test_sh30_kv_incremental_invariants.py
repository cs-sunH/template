"""Focused parity checks for SH3 incremental KV invariants."""

from __future__ import annotations

import sys
import unittest
from unittest import mock
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from face_scheduler import (  # noqa: E402
    FaceHardware,
    FaceInstanceSpec,
    FaceModel,
    KVCacheManager,
    build_instances,
)


def _manager(strict_invariants: bool) -> KVCacheManager:
    hardware = FaceHardware(2, 2, 10_000, 1.0, 2.0, 1.0, 0, 0)
    topology = build_instances(
        hardware,
        (
            FaceInstanceSpec("ins0", "1", (0, 1)),
            FaceInstanceSpec("ins1", "2", (2, 3)),
        ),
    )
    model = FaceModel(2, 4, 4, 2, 4, 1, "gelu")
    return KVCacheManager(
        topology,
        model,
        strict_invariants=strict_invariants,
    )


class IncrementalInvariantTests(unittest.TestCase):
    def test_incremental_and_strict_mutations_stay_equivalent(self) -> None:
        """F4-c 改写（kimi 复审，2026-09-14）：本用例原驱动跨实例
        ``move_request_capacity_reservation(0→1)``——joint 仓 M3 钉死
        decode 于 prefill 实例（红线 #4：跨实例预约移动 fail-closed），
        该驱动面已从合同中移除。语义意图（增量/严格不变量在**全部合法
        变更面**上等价）保留：跨实例轮换改走 copy 动作（尾部
        prepare@copy + merge_back），同实例生命周期覆盖
        move/move_prefill_to_decode 的同实例分支。"""
        fast = _manager(False)
        strict = _manager(True)

        def verify() -> None:
            fast._check_invariants()
            strict._check_invariants()
            self.assertEqual(fast.hbm_snapshots(), strict.hbm_snapshots())
            self.assertEqual(fast.session_snapshots(), strict.session_snapshots())

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
            verify()

        invoke(
            "prepare_prefill",
            session_id="session",
            target_instance_index=0,
            history_tokens=0,
            trigger_request_id="request",
        )
        invoke(
            "reserve_request_capacity",
            request_id="request",
            session_id="session",
            instance_index=0,
            final_context_tokens=4,
        )
        invoke(
            "expand_prefill",
            session_id="session",
            instance_index=0,
            context_tokens=4,
            trigger_request_id="request",
            reservation_request_id="request",
        )
        invoke(
            "move_request_capacity_reservation",
            request_id="request",
            target_instance_index=0,  # M3 钉死：同实例 no-op（跨实例 raise）
        )
        invoke(
            "move_prefill_to_decode",
            session_id="session",
            target_instance_index=0,
            trigger_request_id="request",
            reservation_request_id="request",
        )
        invoke("release_request_capacity_reservation", "request")
        invoke(
            "expand_decode",
            session_id="session",
            instance_index=0,
            final_context_tokens=5,
            trigger_request_id="request",
        )
        invoke("mark_complete", "session", 4, "human")
        with mock.patch.object(
            fast,
            "_check_invariants",
            side_effect=AssertionError("normal mutation used the full verifier"),
        ):
            fast_suffix = fast._evict_suffix(
                fast._sessions["session"],
                phase="test",
                reason="parity",
                trigger_request_id="request",
            )
        self.assertEqual(
            fast_suffix,
            strict._evict_suffix(
                strict._sessions["session"],
                phase="test",
                reason="parity",
                trigger_request_id="request",
            ),
        )
        verify()
        invoke(
            "prepare_prefill",
            session_id="session",
            target_instance_index=0,
            history_tokens=5,
            trigger_request_id="next",
        )
        invoke("mark_complete", "session", 5, "human")
        with mock.patch.object(
            fast,
            "_check_invariants",
            side_effect=AssertionError("normal mutation used the full verifier"),
        ):
            fast_evict = fast._evict_session(
                fast._sessions["session"],
                phase="test",
                reason="parity",
                trigger_request_id="next",
            )
        self.assertEqual(
            fast_evict,
            strict._evict_session(
                strict._sessions["session"],
                phase="test",
                reason="parity",
                trigger_request_id="next",
            ),
        )
        verify()
        invoke(
            "prepare_prefill",
            session_id="session",
            target_instance_index=0,
            history_tokens=5,
            trigger_request_id="final",
            action="copy",
        )
        # joint（§2.3）：跨实例工作副本必须先 merge_back（增量归并回
        # home）再 mark_complete——service_done 位于 merge_done 之后。
        invoke(
            "merge_back",
            session_id="session",
            trigger_request_id="final",
            new_tokens=0,
        )
        invoke("mark_complete", "session", 6, "human")
        invoke("retire_terminal_session", "session", 6, "final")
        fast.assert_final_state()
        strict.assert_final_state()

    def test_incremental_and_full_checks_reject_corrupt_accounting(self) -> None:
        for strict_invariants in (False, True):
            manager = _manager(strict_invariants)
            manager.prepare_prefill(
                session_id="session",
                target_instance_index=0,
                history_tokens=0,
                trigger_request_id="request",
            )
            manager.expand_prefill(
                session_id="session",
                instance_index=0,
                context_tokens=1,
                trigger_request_id="request",
            )
            rank = manager.topology.instance(0).ranks[0]
            manager._rank_states[rank].kv_cache_bytes += 1
            with self.assertRaisesRegex(RuntimeError, "KV accounting mismatch"):
                manager._check_invariants_after_mutation(session_ids=("session",))
            with self.assertRaisesRegex(RuntimeError, "KV accounting mismatch"):
                manager._check_invariants()


if __name__ == "__main__":
    unittest.main()
