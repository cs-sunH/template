#!/usr/bin/env python3
"""test_distributed_settlement.py -- W9 目录提交型结算（G-DIST/V9 门）。

方案 §11.1 V9 行 + §12.1 W9 验收门的 KVCacheManager/调度器级落测
（受控多轮 fixture，合成，无需仿真）：

1. **多轮增量、超过三源**：4 实例拓扑 4 轮（stay→copy→recompute→
   recompute）——目录来源位置数 >3、区间数与位置数分别披露（原
   remote-read 轮已随 2026-10-14 远读退役换 recompute 承载）；
2. **全部有效区间可定位**：逐轮单元格级覆盖审计（跨持有者部分重叠
   区间的真并集）+ 目录守恒 + rank 台账对账（manager 咽喉点自带）；
3. **逐出→恢复**：留存段经目录 evict_to_pool（owner 记录保留、唯一
   内容在池）→ restore_from_pool → READY——恢复后全部唯一内容仍可
   定位；容量压力路径（_ensure_capacity）把留存段作为区间级 victim；
4. **最终会话清理无悬挂**：retire_terminal_session 清理与未完访问
   协调（悬挂引用 fail-closed 负例）；
5. **consolidate 零漂移**：同 fixture 缺省策略走 merge_back（无留存
   段/无 distributed_layout），distributed 只在显式配置激活；
6. **distributed vs consolidate 对照面**：同 fixture 双策略运行——
   per-session 完整会话指标分列（同轮 merge 节省单独披露，不冒称长
   期收益）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 online/test_distributed_settlement.py
"""
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WL = os.path.dirname(_HERE)
for _p in (_HERE, _WL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from face_scheduler import (  # noqa: E402
    FaceHardware,
    FaceInstanceSpec,
    KVCacheManager,
    build_instances,
    kv_cache_shard_bytes_for_tokens,
)
from joint.distributed_settlement import (  # noqa: E402
    assert_full_content_locatable,
    enumerate_source_surface,
)
from joint.kv_location_directory import (  # noqa: E402
    ENTRY_STATE_EVICTED_TO_POOL,
    POOL_HOLDER,
)
from joint.test_joint_mechanisms import _tiny_model  # noqa: E402


def _four_instance_hardware(capacity_bytes=10_000_000):
    return FaceHardware(
        mesh_rows=2, mesh_cols=2,
        local_hbm_capacity_bytes=capacity_bytes,
        local_hbm_bandwidth_gbps=100.0, d2d_bandwidth_gbps=200.0,
        peak_perf_tflops=1.0, d2d_latency_ns=0, local_hbm_latency_ns=0,
    )


def _four_instance_topology(hardware):
    return build_instances(hardware, (
        FaceInstanceSpec("ins0", "1", (0,)),
        FaceInstanceSpec("ins1", "2", (1,)),
        FaceInstanceSpec("ins2", "3", (2,)),
        FaceInstanceSpec("ins3", "4", (3,)),
    ))


def _manager(capacity=10_000_000, layers=4):
    hardware = _four_instance_hardware(capacity)
    model = _tiny_model(layers)
    topology = _four_instance_topology(hardware)
    return KVCacheManager(
        topology, model, category_mode="lru",
        layer_policy="minimal_layer_groups")


def _round(kv, session_id, request_id, target, action, history,
           final_context, *, at_ns, mark_complete=True,
           next_request_type=None):
    """一轮受控执行：prepare → expand → settle（合成多轮 fixture）。

    final_context = 本轮结束时的完整上下文 token 数；本轮新增
    new_tokens = final_context - history（无 decode 增量的合成形态——
    输入增量即全部增长）。三动作工作副本恒持完整上下文（原 remote-read
    增量口径物化已随 2026-10-14 远读退役删除）。"""
    increment = final_context - history
    kv.prepare_prefill(
        session_id=session_id, target_instance_index=target,
        history_tokens=history, trigger_request_id=request_id,
        action=action, at_ns=at_ns)
    if increment > 0:
        kv.expand_prefill(
            session_id=session_id, instance_index=target,
            context_tokens=history, trigger_request_id=request_id,
            at_ns=at_ns)
        kv.expand_decode(
            session_id=session_id, instance_index=target,
            final_context_tokens=final_context,
            trigger_request_id=request_id, at_ns=at_ns)
    outcome = kv.settle_distributed(
        session_id=session_id, trigger_request_id=request_id,
        new_tokens=increment, at_ns=at_ns)
    if mark_complete:
        kv.mark_complete(
            session_id, at_ns, next_request_type=next_request_type)
    return outcome


def _sources(kv, session_id):
    return enumerate_source_surface(kv.kv_directory, session_id)


class MultiRoundDistributedTest(unittest.TestCase):
    """V9：多轮增量、超过三源、逐轮可定位/守恒。"""

    # （原 P=copy + D=remote-read 边界来源矩形用例已随 2026-10-14
    # 远读退役删除：commit_boundary_decode_transition 的 remote 分支
    # 删除后该形态不可构造；来源面完整性由四轮用例的目录审计承接。）

    def test_four_rounds_three_plus_sources(self):
        kv = _manager()
        sid = "s_multi"
        # 轮 1：stay@0 建档（10 token）。
        out1 = _round(kv, sid, "r1", 0, "stay", 0, 10, at_ns=100)
        self.assertEqual(out1.family, "stay_local")
        self.assertEqual(out1.source_count, 1)
        surface1 = _sources(kv, sid)
        self.assertEqual(surface1.interval_count, 1)
        # 轮 2：copy 0→1（12 token；[0,10)@0 转留存 + 工作副本@1）。
        out2 = _round(kv, sid, "r2", 1, "copy", 10, 12, at_ns=200)
        self.assertEqual(out2.family, "zero_byte_flip")
        self.assertEqual(out2.source_count, 2)
        # 轮 3：recompute @2（14 token；重建并集@2 + [0,12)@1 + [0,10)@0）。
        out3 = _round(kv, sid, "r3", 2, "recompute", 12, 14, at_ns=300)
        self.assertEqual(out3.family, "zero_byte_flip")
        self.assertEqual(out3.final_tokens, 14)
        self.assertGreaterEqual(out3.source_count, 3)
        # 轮 4：recompute @3（16 token；重建并集@3——来源位置 >3）。
        out4 = _round(kv, sid, "r4", 3, "recompute", 14, 16, at_ns=400)
        self.assertEqual(out4.family, "zero_byte_flip")
        surface4 = _sources(kv, sid)
        self.assertGreater(surface4.source_count, 3)
        self.assertGreaterEqual(surface4.interval_count, surface4.source_count)
        # 逐轮：全部唯一内容可定位（单元格级真并集）+ 目录守恒。
        kv.kv_directory.assert_conservation()
        assert_full_content_locatable(
            kv.kv_directory, sid, history_tokens=16,
            model_layers=kv.model.layers)
        # 增长预算：每轮 ≤ +2（无显式拆分）。
        growths = [round_info["growth"] for round_info in
                   kv.distributed_directory_meter.session_rounds[sid]]
        self.assertTrue(all(g <= 2 for g in growths), growths)
        # 会话字段：home 粘性（不暗中重置，A13）；distributed_layout 在位。
        session = kv._sessions[sid]
        self.assertEqual(session.home_instance, 0)
        self.assertTrue(session.distributed_layout)
        self.assertIsNone(session.working_kind)  # 轮已结算
        # 版本键：重复结算同一请求 fail-closed。
        with self.assertRaisesRegex(RuntimeError, "settled twice"):
            kv.settle_distributed(
                session_id=sid, trigger_request_id="r4", new_tokens=0)

    def test_early_round_directory_tracks_layer_and_token_intervals(self):
        """V9：层/token 区间变化的多轮形态在目录逐条可查。"""
        kv = _manager()
        sid = "s_intervals"
        _round(kv, sid, "r1", 0, "stay", 0, 10, at_ns=100)
        _round(kv, sid, "r2", 1, "copy", 10, 12, at_ns=200)
        surface = _sources(kv, sid)
        token_ranges = sorted(
            item[2] for item in surface.on_chip)
        # 跨轮区间：[0,12)@1（工作副本）与 [0,10)@0（留存基础）。
        self.assertIn((0, 12), token_ranges)
        self.assertIn((0, 10), token_ranges)


class OverlapUnionDedupTest(unittest.TestCase):
    """W4/V6（§6.1/§6.3）：跨持有者重复留存段的并集去重。

    distributed 跨轮留存天然产生包含型重复区间（如 [0,10)@0 与
    [0,12)@1 并存）——缺失/来源推导按初等单元格并集去重：copy 工作字
    节 = 缺失并集（不重复搬运）、decode credit 读流基数 = 缺失并集
    token 数（不重复读）；settle 零字节翻转族守恒断言据此闭合。"""

    def _three_round_overlap(self):
        kv = _manager()
        sid = "s_overlap"
        _round(kv, sid, "r1", 0, "stay", 0, 10, at_ns=100)
        _round(kv, sid, "r2", 1, "copy", 10, 12, at_ns=200)
        _round(kv, sid, "r3", 2, "copy", 12, 14, at_ns=300)
        return kv, sid

    def test_missing_query_returns_union_not_entry_sum(self):
        """缺失查询 = 并集 14 token（含 [0,10)@0 与 [0,12)@1 重复段）。"""
        kv, sid = self._three_round_overlap()
        missing = kv.kv_directory.query_missing_intervals(sid, 3)
        union_tokens = sum(
            entry.content_key.token_end - entry.content_key.token_start
            for entry in missing)
        self.assertEqual(union_tokens, 14)
        # 缺失并集字节 = 全史字节（copy 工作字节据此计，§6.3）。
        self.assertEqual(
            sum(sum(entry.bytes_by_tp_rank) for entry in missing),
            sum(kv_cache_shard_bytes_for_tokens(
                kv.model, 14, kv.tp_degree)))
        # 读流来源 = 缺失并集的权威选读片段（并集去重——旧逐条目口径
        # 对 [0,10)/[0,12)/[0,14) 重复段会计 22+ token；此处按实例 3
        # 视角查询，权威选读恰覆盖 14 token 并集）。
        sources = kv.kv_directory.query_sources(sid, 3)
        source_union_tokens = sum(
            entry.content_key.token_end - entry.content_key.token_start
            for entry in sources)
        self.assertEqual(source_union_tokens, 14)

    def test_copy_over_retained_union_settles_conserved(self):
        """copy 过重复留存段：工作字节 = 缺失并集，settle 守恒闭合。"""
        kv, sid = self._three_round_overlap()
        out4 = _round(kv, sid, "r4", 3, "copy", 14, 16, at_ns=400)
        self.assertEqual(out4.family, "zero_byte_flip")
        kv.kv_directory.assert_conservation()
        assert_full_content_locatable(
            kv.kv_directory, sid, history_tokens=16,
            model_layers=kv.model.layers)


class EvictRestoreLocatableTest(unittest.TestCase):
    """V9：逐出→恢复后全部唯一内容仍可定位（目录审计）。"""

    def _settled_multi_round(self):
        kv = _manager()
        sid = "s_evict"
        _round(kv, sid, "r1", 0, "stay", 0, 10, at_ns=100)
        _round(kv, sid, "r2", 1, "copy", 10, 12, at_ns=200)
        _round(kv, sid, "r3", 2, "copy", 12, 14, at_ns=300)
        return kv, sid

    def test_retained_evict_to_pool_keeps_owner_and_locatable(self):
        kv, sid = self._settled_multi_round()
        session = kv._sessions[sid]
        victims = kv._completed_retained_segments(0)
        self.assertTrue(victims)
        victim_session, segment = victims[0]
        self.assertEqual(victim_session.session_id, sid)
        # 执行保护已结束（settle 释放引用）→ 合法驱逐（不永久 pin）。
        entry_before = next(
            entry for entry in kv.kv_directory.entries_for_session(sid)
            if entry.holder == segment.instance_index
            and entry.content_key.token_range == (
                segment.token_start, segment.token_end))
        transfer = kv._evict_retained_segment(
            session, segment, phase="history", reason="test_capacity",
            trigger_request_id="evict_r")
        self.assertGreater(transfer.total_bytes, 0)
        # owner 记录保留（目录池段在场 + owner 链披露）。
        pooled = kv.kv_directory.query_pool_intervals(sid)
        self.assertTrue(any(
            entry.content_key.token_range == (
                segment.token_start, segment.token_end)
            for entry in pooled))
        retained = kv._distributed_retained[sid][segment.key]
        self.assertEqual(retained.state, ENTRY_STATE_EVICTED_TO_POOL)
        self.assertIn("evicted-to-pool", retained.history[-1])
        # 全部唯一内容仍可定位（池 = 合法位置）。
        assert_full_content_locatable(
            kv.kv_directory, sid, history_tokens=14,
            model_layers=kv.model.layers)
        kv.kv_directory.assert_conservation()
        _ = entry_before

    def test_restore_returns_content_on_chip(self):
        kv, sid = self._settled_multi_round()
        session = kv._sessions[sid]
        _victim_session, segment = kv._completed_retained_segments(0)[0]
        kv._evict_retained_segment(
            session, segment, phase="history", reason="test_capacity",
            trigger_request_id="evict_r")
        restore_transfer = kv.restore_retained_segment(
            sid, segment.key, target_instance=3, trigger_request_id="res_r")
        self.assertGreater(restore_transfer.total_bytes, 0)
        kv.complete_retained_restore(sid, segment.key)
        # 2026-10-01 审查修复对齐：complete_retained_restore 换实例后按
        # 新实例键写入（值 .key 与字典键一致）；按旧键读取改为按键值
        # 反查（修前读旧键 = KeyError）。
        restored = next(
            seg for seg in kv._distributed_retained[sid].values()
            if seg.layer_start == segment.layer_start
            and seg.layer_end == segment.layer_end
            and seg.token_start == segment.token_start
            and seg.token_end == segment.token_end)
        self.assertEqual(restored.instance_index, 3)
        self.assertIn(restored.key, kv._distributed_retained[sid])
        assert_full_content_locatable(
            kv.kv_directory, sid, history_tokens=14,
            model_layers=kv.model.layers)
        kv.kv_directory.assert_conservation()

    def test_ensure_capacity_evicts_retained_segments(self):
        """容量压力：主位置 victim 之外，留存段进 E 的 victim 序。"""
        kv, sid = self._settled_multi_round()
        # 另一会话在实例 0 制造缺口（主位置 victim 不可用——active）。
        kv.prepare_prefill(
            session_id="s_pressure", target_instance_index=0,
            history_tokens=0, trigger_request_id="p1")
        kv.expand_prefill(
            session_id="s_pressure", instance_index=0,
            context_tokens=0, trigger_request_id="p1")
        # s_pressure active ⇒ 主位置 victim 保护自身；缺口由留存段满足。
        retained_on_zero = sum(
            segment.total_bytes()
            for _session, segment in kv._completed_retained_segments(0))
        self.assertGreater(retained_on_zero, 0)
        remaining = kv._effective_remaining_by_tp_rank(0)
        required = tuple(
            r + retained_on_zero // 2 + 1 for r in remaining)
        evictions = kv._ensure_capacity(
            0, required, phase="history", reason="pressure_test",
            trigger_request_id="p1", protected_session_id="s_pressure")
        self.assertTrue(any(
            "retained_segment" in (transfer.reason or "")
            for transfer in evictions))
        assert_full_content_locatable(
            kv.kv_directory, sid, history_tokens=14,
            model_layers=kv.model.layers)


class TerminalCleanupTest(unittest.TestCase):
    """V9：最终会话清理无悬挂（fail-closed 负例 + 正常清理）。"""

    def test_retire_clears_retained_without_dangling(self):
        kv = _manager()
        sid = "s_retire"
        _round(kv, sid, "r1", 0, "stay", 0, 10, at_ns=100)
        _round(kv, sid, "r2", 1, "copy", 10, 12, at_ns=200)
        self.assertTrue(kv._distributed_retained.get(sid))
        kv.retire_terminal_session(sid, 1000, request_id="r2")
        self.assertNotIn(sid, kv._distributed_retained)
        self.assertNotIn(sid, kv._sessions)
        kv.kv_directory.assert_conservation()

    def test_live_reference_blocks_retire(self):
        kv = _manager()
        sid = "s_dangling"
        _round(kv, sid, "r1", 0, "stay", 0, 10, at_ns=100)
        _round(kv, sid, "r2", 1, "copy", 10, 12, at_ns=200)
        # 悬挂引用：终局清理与未完访问协调（fail-closed）。
        entry = next(
            entry for entry in kv.kv_directory.entries_for_session(sid)
            if entry.holder != POOL_HOLDER)
        kv.kv_directory.acquire_entry_reference(entry.entry_id)
        with self.assertRaisesRegex(RuntimeError, "dangling"):
            kv.retire_terminal_session(sid, 1000, request_id="r2")
        kv.kv_directory.release_entry_reference(entry.entry_id)
        kv.retire_terminal_session(sid, 1000, request_id="r2")


class ConsolidateZeroDriftTest(unittest.TestCase):
    """G-DIST 前提：consolidate 缺省路径零漂移（无留存段/无 distributed
    分支激活）；distributed 只在显式结算调用下生效。"""

    def test_default_settle_leaves_no_retained_segments(self):
        kv = _manager()
        sid = "s_consolidate"
        kv.prepare_prefill(
            session_id=sid, target_instance_index=0,
            history_tokens=0, trigger_request_id="c1")
        kv.expand_prefill(
            session_id=sid, instance_index=0, context_tokens=0,
            trigger_request_id="c1")
        kv.expand_decode(
            session_id=sid, instance_index=0, final_context_tokens=10,
            trigger_request_id="c1")
        kv.mark_complete(sid, 100)
        # 下一轮 copy（legacy 路径）+ merge_back 收敛单位置。
        kv.prepare_prefill(
            session_id=sid, target_instance_index=1,
            history_tokens=10, trigger_request_id="c2", action="copy")
        kv.expand_decode(
            session_id=sid, instance_index=1, final_context_tokens=12,
            trigger_request_id="c2")
        kv.merge_back(
            session_id=sid, trigger_request_id="c2", new_tokens=2)
        session = kv._sessions[sid]
        self.assertFalse(session.distributed_layout)
        self.assertEqual(session.instance_index, 1)
        self.assertEqual(session.context_tokens, 12)
        self.assertFalse(kv._distributed_retained)
        surface = _sources(kv, sid)
        self.assertEqual(surface.source_count, 1)  # 归并后单位置
        kv.kv_directory.assert_conservation()


class DistributedVsConsolidateComparisonTest(unittest.TestCase):
    """G-DIST：distributed vs consolidate 同 fixture 对照的指标面。

    同目标/硬件/控制（同轮序列、同实例选择、同 token 序列），只有结算
    策略不同——完整会话指标分列：
    * distributed：零 merge 载荷、留存段非零（同轮 merge 节省单独披露，
      不冒称长期收益——`same_round_merge_saved_bytes` 显式同轮口径）；
    * consolidate：物理归并载荷非零、终态单位置；
    * 两策略终态并集内容一致（可定位面积守恒）。"""

    def _fixture(self, *, distributed: bool):
        kv = _manager()
        sid = "s_cmp"
        state = {
            "merge_payload_bytes": 0,
            "pool_store_bytes": 0,
            "settle_outcomes": [],
        }
        rounds = ((0, "stay", 0, 10), (1, "copy", 10, 12),
                  (2, "copy", 12, 14), (3, "recompute", 14, 16))
        for index, (target, action, history, final) in enumerate(rounds):
            request_id = f"cmp_r{index + 1}"
            if distributed:
                outcome = _round(
                    kv, sid, request_id, target, action, history, final,
                    at_ns=100 * (index + 1))
                state["settle_outcomes"].append(outcome)
                state["merge_payload_bytes"] += outcome.merge_payload_bytes
            else:
                kv.prepare_prefill(
                    session_id=sid, target_instance_index=target,
                    history_tokens=history, trigger_request_id=request_id,
                    action=action)
                increment = final - history
                if increment > 0:
                    kv.expand_prefill(
                        session_id=sid, instance_index=target,
                        context_tokens=history,
                        trigger_request_id=request_id)
                    kv.expand_decode(
                        session_id=sid, instance_index=target,
                        final_context_tokens=final,
                        trigger_request_id=request_id)
                transfers = kv.merge_back(
                    session_id=sid, trigger_request_id=request_id,
                    new_tokens=increment)
                state["merge_payload_bytes"] += sum(
                    int(t.total_bytes) for t in transfers)
                state["pool_store_bytes"] += sum(
                    int(t.total_bytes) for t in transfers
                    if t.kind == "remote_store")
            kv.mark_complete(sid, 100 * (index + 1))
        state["final_context"] = kv._sessions[sid].context_tokens
        state["surface"] = _sources(kv, sid)
        state["kv"] = kv
        state["session_id"] = sid
        return state

    def test_comparison_metric_surface(self):
        distributed = self._fixture(distributed=True)
        consolidate = self._fixture(distributed=False)
        # distributed：零物理 merge 载荷；位置多源留存。
        self.assertEqual(distributed["merge_payload_bytes"], 0)
        self.assertGreater(
            distributed["surface"].source_count, 1)
        # consolidate：终态单位置（逐轮 merge_back 归并）。三动作下本
        # fixture 的合并恒零字节翻转（嵌套副本释放；原真实迁移腿来自
        # remote-read 增量布局的分裂末态，已随 2026-10-14 远读退役不可
        # 构造——完整型合并语义由 test_joint_remote_settlement 的 copy
        # 三位置链与 face_scheduler merge 面钉测承接）。
        self.assertEqual(consolidate["merge_payload_bytes"], 0)
        self.assertEqual(consolidate["surface"].source_count, 1)
        # 同轮省 merge 的对照口径：distributed 结算披露显式标注同轮语义。
        for outcome in distributed["settle_outcomes"]:
            self.assertEqual(outcome.merge_payload_bytes, 0)
            self.assertIn("same_round", "same_round_merge_saved_bytes")
        # 两策略终态并集内容一致（16 token × L 全可定位）。
        for state in (distributed, consolidate):
            assert_full_content_locatable(
                state["kv"].kv_directory, state["session_id"],
                history_tokens=16,
                model_layers=state["kv"].model.layers)
            state["kv"].kv_directory.assert_conservation()
        # 分布式的同轮节省 ≠ 长期收益：留存段字节持续占用（容量摘要可
        # 见——distributed 的留存段进入可回收桶，consolidate 归并后单
        # 位置驻留）。
        retained_bytes = sum(
            segment.total_bytes()
            for segment in distributed["kv"]
            ._distributed_retained[distributed["session_id"]].values())
        self.assertGreater(retained_bytes, 0)


class SchedulerDistributedSettleTest(unittest.TestCase):
    """调度器结算分派（真 drain → _complete_requests → settle_distributed
    → settle_done）+ 早到下一轮结算闭合门 + 来源面披露。"""

    def _distributed_scheduler(self):
        from types import SimpleNamespace
        from online.test_settlement_state_machine import (
            _settle_ready_scheduler)
        from online.sh30_online_scheduler import (
            SETTLEMENT_POLICY_DISTRIBUTED)
        from joint.prediction_config import PredictionParams
        scheduler, kv, runtime = _settle_ready_scheduler(decode_length=0)
        scheduler._joint_prediction_config = SimpleNamespace(
            params=PredictionParams(),
            settlement_policy=SETTLEMENT_POLICY_DISTRIBUTED,
            manifest_dict=lambda: {},
        )
        return scheduler, kv, runtime

    def test_distributed_settle_closes_with_directory_commit(self):
        scheduler, kv, runtime = self._distributed_scheduler()
        scheduler._on_prefill_drain("r0", 5000)
        self.assertEqual(runtime.joint_stage, "SETTLE_DONE")
        settle_rows = [row for row in scheduler.online_log_rows
                       if row["kind"] == "settle_done"]
        self.assertEqual(len(settle_rows), 1)
        settle = settle_rows[0]["decision"]
        self.assertEqual(settle["settlement_policy"], "distributed")
        self.assertEqual(settle["merge_payload_bytes"], 0)
        self.assertIsNone(settle["merge_done_ns"])
        # W9：目录提交块（条目/来源/版本披露）。
        commit = settle["directory_commit"]
        self.assertEqual(commit["settlement_policy"], "distributed")
        self.assertGreaterEqual(commit["interval_count"], 1)
        self.assertIn("directory_version", commit)
        # KV 侧：零字节移动（无 merge 载荷）+ 结算账本闭合弹出；终轮
        # （无后继）会话经 retire_terminal_session 清理——清理无悬挂本身
        # 即审计证明（悬挂引用/未完 ack fail-closed 在清理路径钉测）。
        self.assertNotIn(runtime.session_id, kv._sessions)
        self.assertFalse(scheduler._joint_settlement_ledger())
        kv.kv_directory.assert_conservation()

    def test_early_next_round_blocked_while_settlement_open(self):
        """§7.4：到达早于 settle 允许排队，但消费受结算闭合门限制。"""
        scheduler, kv, runtime = self._distributed_scheduler()
        # 模拟在册未闭合结算（到达早于 settle_done 的窗口）。
        scheduler._joint_settlement_ledger()["r_prev"] = {
            "stage": "SETTLE_COMMITTED",
            "session_id": runtime.session_id,
        }
        self.assertTrue(
            scheduler._joint_session_settlement_open(runtime.session_id))
        # 闭合后放行。
        scheduler._joint_settlement_ledger().pop("r_prev")
        self.assertFalse(
            scheduler._joint_session_settlement_open(runtime.session_id))

    def test_admission_gate_returns_false_when_settlement_open(self):
        scheduler, kv, runtime = self._distributed_scheduler()
        runtime.estimated_arrival_ns = 100
        scheduler._joint_settlement_ledger()["r_prev"] = {
            "stage": "SETTLE_COMMITTED",
            "session_id": runtime.session_id,
        }
        admitted = scheduler._try_admit_request(runtime, 100)
        self.assertFalse(admitted)
        gate_rows = [row for row in scheduler.online_log_rows
                     if row["kind"] == "joint_admission"
                     and row["decision"].get("admitted") is False]
        self.assertTrue(any(
            row["decision"]["reason"] == "settlement_open_dependency_gate"
            for row in gate_rows))


if __name__ == "__main__":
    unittest.main()
