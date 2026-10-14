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
        """F4-c 改写（kimi 复审，2026-09-14；D18 修订 2026-09-26；D20
        修订 2026-09-26）：本用例原驱动跨实例 ``move_request_capacity_
        reservation(0→1)``——joint 仓 M3 曾钉死 decode 于 prefill 实例
        （红线 #4：跨实例预约移动 fail-closed）。D18 废除强制同址并将
        跨实例移动重设计为真实预约事务，D20 再升级为幂等 epoch 的跨阶
        段预约事务（face_scheduler.commit_stage_reservation_transaction，
        事务语义由 joint/test_d20_stage_transaction.py 与 test_joint_
        fixes.py 钉测覆盖）；本用例沿同实例生命周期覆盖
        commit_stage_reservation_transaction / move_prefill_to_decode
        的原地分支，跨实例轮换仍走 merge v2 腿（2026-09-17：PARTIAL
        混合 remote-read 反向翻转 + REMOTE 基 copy 就地转正）。语义意
        图（增量/严格不变量在**全部合法变更面**上等价）不变。"""
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
            # D20（2026-09-26）：预约移动经幂等 epoch 的跨阶段预约事务
            # （commit_stage_reservation_transaction，v2.0 §14.5/§16.4）；
            # 同实例原地继续 = 纯 resize 事务（stay 足迹 = 准入足迹）。
            "commit_stage_reservation_transaction",
            request_id="request",
            target_instance_index=0,
            stage_action="stay",
            stage_epoch=1,
            history_tokens=0,
            input_tokens=4,
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
        # ---- merge v2（少并多，2026-09-17）：混合形态 + 翻转腿。 ----
        # 再截半层 → PARTIAL p=1；remote-read 混合形态（后缀池恢复热 KV）
        # + 增量 expand；merge v2 反向（H < S+I）：基础前缀 NoC 搬 exec、
        # home 侧释放、home 迁移到 exec——全程增量/严格审计等价。
        with mock.patch.object(
            fast,
            "_check_invariants",
            side_effect=AssertionError("normal mutation used the full verifier"),
        ):
            fast_partial = fast._evict_suffix(
                fast._sessions["session"],
                phase="test",
                reason="parity",
                trigger_request_id="hybrid",
            )
        self.assertEqual(
            fast_partial,
            strict._evict_suffix(
                strict._sessions["session"],
                phase="test",
                reason="parity",
                trigger_request_id="hybrid",
            ),
        )
        verify()
        invoke(
            "prepare_prefill",
            session_id="session",
            target_instance_index=1,
            history_tokens=5,
            trigger_request_id="hybrid",
            action="remote-read",
        )
        invoke(
            "expand_prefill",
            session_id="session",
            instance_index=1,
            context_tokens=1,
            trigger_request_id="hybrid",
            reservation_request_id="hybrid",
        )
        invoke(
            "merge_back",
            session_id="session",
            trigger_request_id="hybrid",
            new_tokens=1,
        )
        self.assertEqual(fast.last_merge_outcome,
                         strict.last_merge_outcome)
        self.assertEqual(
            fast.last_merge_outcome["direction"], "reverse")
        self.assertFalse(fast.last_merge_outcome["zero_byte_flip"])
        self.assertEqual(fast.last_merge_outcome["winner_instance"], 1)
        self.assertTrue(fast.last_merge_outcome["home_flipped"])
        verify()
        # 翻转后会话活跃位复位（下一 fixture 逐出需要 completed+inactive）。
        invoke("mark_complete", "session", 6, "human")
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
            history_tokens=6,
            trigger_request_id="final",
            action="copy",
        )
        # 工作副本增长至终态上下文（copy 覆盖完整上下文 = 历史 6 ＋新 1）。
        invoke(
            "expand_prefill",
            session_id="session",
            instance_index=0,
            context_tokens=7,
            trigger_request_id="final",
            reservation_request_id="final",
        )
        # joint（§2.3）：跨实例工作副本必须先 merge_back 再 mark_complete
        # ——service_done 位于 merge_done 之后。REMOTE 基（无主，裁定③）
        # → in_place：零传输零池写、工作副本直接转正、home := exec。
        invoke(
            "merge_back",
            session_id="session",
            trigger_request_id="final",
            new_tokens=1,
        )
        self.assertEqual(fast.last_merge_outcome["direction"], "in_place")
        self.assertEqual(fast.last_merge_outcome["winner_instance"], 0)
        self.assertTrue(fast.last_merge_outcome["home_flipped"])
        self.assertEqual(fast.last_merge_outcome["transferred_bytes"], 0)
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

    def test_base_contribution_follows_physical_base_instance(self) -> None:
        """D-BI 回归（2026-09-26，冒烟矩阵 v20-b2 rank20 幻影贡献根因）。

        turn-0：prefill@0 → ``move_prefill_to_decode`` 迁至实例 1 →
        merge stay 提交（无工作副本，逻辑 home 粘滞于 0，KV 终驻实例 1）。
        turn-1 跨实例工作副本事务的基础物理在实例 1 而非 home 0——守恒
        期望的基础贡献必须跟随 ``base_instance_index``，不得给 home 记
        幻影字节（修复前：增量审计报 "KV accounting mismatch:
        state=<工作副本>, expected=<工作副本+基础>"，主控冒烟矩阵 12 臂
        全灭的同型失败）；merge 的基础侧释放/传输端点同口径。"""
        fast = _manager(False)
        strict = _manager(True)

        def verify() -> None:
            fast._check_invariants()
            strict._check_invariants()
            self.assertEqual(fast.hbm_snapshots(), strict.hbm_snapshots())
            self.assertEqual(fast.session_snapshots(), strict.session_snapshots())

        def invoke(name: str, *args: object, **kwargs: object) -> object:
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

        # ---- turn-0：stay@0 + P→D 迁移到 1 + merge stay 提交。 ----
        invoke(
            "prepare_prefill",
            session_id="session",
            target_instance_index=0,
            history_tokens=0,
            trigger_request_id="turn0",
        )
        invoke(
            "expand_prefill",
            session_id="session",
            instance_index=0,
            context_tokens=4,
            trigger_request_id="turn0",
        )
        invoke(
            "move_prefill_to_decode",
            session_id="session",
            target_instance_index=1,
            trigger_request_id="turn0",
        )
        invoke(
            "expand_decode",
            session_id="session",
            instance_index=1,
            final_context_tokens=5,
            trigger_request_id="turn0",
        )
        invoke("merge_back", session_id="session", trigger_request_id="turn0",
               new_tokens=1)
        session = fast._sessions["session"]
        self.assertEqual(
            (session.instance_index, session.home_instance,
             session.working_kind, session.location),
            (1, 0, None, fast.LOCAL_HBM),
        )
        self.assertIsNone(session.base_instance_index)
        self.assertEqual(
            [fast._rank_states[rank].kv_cache_bytes
             for rank in fast.topology.instance(0).ranks],
            [0, 0],
        )

        # ---- turn-1：copy 回 home（基础物理在实例 1）。 ----
        invoke(
            "prepare_prefill",
            session_id="session",
            target_instance_index=0,
            history_tokens=5,
            trigger_request_id="turn1",
            action="copy",
        )
        session = fast._sessions["session"]
        # 工作副本活跃期：基础贡献记在物理基础实例 1（D-BI），不是 home 0。
        base_contribution = fast._incremental_base_contribution(session)
        self.assertIsNotNone(base_contribution)
        self.assertEqual(base_contribution[0], 1)
        self.assertTrue(any(base_contribution[1]))
        self.assertEqual(
            fast._invariant_expected_kv_by_rank[
                fast.topology.instance(0).ranks[0]],
            sum(session.shard_bytes[:1]),
        )
        invoke(
            "expand_prefill",
            session_id="session",
            instance_index=0,
            context_tokens=6,
            trigger_request_id="turn1",
        )
        invoke("merge_back", session_id="session", trigger_request_id="turn1",
               new_tokens=1)
        # 零字节翻转：胜者 = 执行端 0，败者 = 物理基础所在 1；home 已在
        # 0，不迁移。
        self.assertEqual(fast.last_merge_outcome["direction"], "reverse")
        self.assertTrue(fast.last_merge_outcome["zero_byte_flip"])
        self.assertEqual(fast.last_merge_outcome["winner_instance"], 0)
        self.assertEqual(fast.last_merge_outcome["loser_instance"], 1)
        self.assertFalse(fast.last_merge_outcome["home_flipped"])
        self.assertEqual(
            [fast._rank_states[rank].kv_cache_bytes
             for rank in fast.topology.instance(1).ranks],
            [0, 0],
        )
        invoke("mark_complete", "session", 6, "human")
        fast.assert_final_state()
        strict.assert_final_state()

    def test_remote_read_merge_forward_targets_base_instance(self) -> None:
        """D-BI 回归（forward 腿）：remote-read 基础在他侧时，少并多
        forward 的传输目标/胜者是物理基础驻留实例，随后 ``home := 胜者``
        迁移 home（I7），终态审计全绿。"""
        fast = _manager(False)
        fast.prepare_prefill(
            session_id="session", target_instance_index=0,
            history_tokens=0, trigger_request_id="turn0")
        fast.expand_prefill(
            session_id="session", instance_index=0, context_tokens=4,
            trigger_request_id="turn0")
        fast.move_prefill_to_decode(
            session_id="session", target_instance_index=1,
            trigger_request_id="turn0")
        fast.expand_decode(
            session_id="session", instance_index=1, final_context_tokens=5,
            trigger_request_id="turn0")
        fast.merge_back(session_id="session", trigger_request_id="turn0",
                        new_tokens=1)
        fast.prepare_prefill(
            session_id="session", target_instance_index=0,
            history_tokens=5, trigger_request_id="turn1",
            action="remote-read")
        fast.expand_prefill(
            session_id="session", instance_index=0, context_tokens=1,
            trigger_request_id="turn1")
        fast.merge_back(session_id="session", trigger_request_id="turn1",
                        new_tokens=1)
        self.assertEqual(fast.last_merge_outcome["direction"], "forward")
        self.assertEqual(fast.last_merge_outcome["winner_instance"], 1)
        self.assertTrue(fast.last_merge_outcome["home_flipped"])
        self.assertEqual(fast._sessions["session"].home_instance, 1)
        fast.mark_complete("session", 6, "human")
        fast.assert_final_state()


if __name__ == "__main__":
    unittest.main()
