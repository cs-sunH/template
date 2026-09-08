"""Focused parity checks for incremental SessionKVCacheManager invariants."""

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
    model = FaceModel(1, 4, 4, 2, 4, 1, "gelu")
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
            # The normal path has already used the local verifier.  The full
            # audit here proves its cached totals still match strict mode.
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


if __name__ == "__main__":
    unittest.main()
