"""test_d11_capacity_summary.py -- D11：分阶段足迹 + 容量驱逐摘要。

设计依据：v2.0 §14（式 19/20、§14.2 摘要数据结构与 θ 份额消费、
§14.3 释放完成、§14.4 外部性不计价）与 joint修改文档 D11 验收：

1. rank 不均匀布局在正确 rank 产生缺口（耦合 TP 分片不拆散）；
2. 干净对象不付不存在的池写回（零写回费分支钉测）；
3. 摘要查询成本不随已完成 session 累计数增长（查询路径不扫描会话、
   桶数有界）；
4. KVCacheManager 增量维护（完成/激活/引用变化/真实释放事件）与从零
   重建逐位一致；``_ensure_capacity`` 原位不动；容量账 C=W+O+Z+F 分
   量守恒（Z→O 转账不变 C−W）；不将未来 victim 恢复费用加到当前请求。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 joint/test_d11_capacity_summary.py   （或 pytest 同路径）
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
    KVCacheManager,
    kv_cache_shard_bytes_for_layer_range,
    kv_cache_shard_bytes_for_tokens,
)
from joint.test_joint_mechanisms import (  # noqa: E402
    _seed,
    _tiny_hardware,
    _tiny_model,
    _two_instance_topology,
)
from joint.capacity_summary import (  # noqa: E402
    PHASE_DECODE,
    PHASE_MERGE,
    CapacityBucket,
    CapacityBucketKey,
    CapacitySummary,
    CapacitySummaryError,
    InstanceCapacitySummary,
    PhaseFootprint,
    SUFFIX_EASY,
    SUFFIX_REST,
    consume_gap_theta,
    shard_ratio_signature,
)
from joint.distributed_settlement import RetainedSegment  # noqa: E402
from joint.joint_cost_model import (  # noqa: E402
    InstanceLoadView,
    JointCostModel,
    JointHardwareRates,
    LinkFlowRegistry,
    ServiceFactors,
    SessionKVView,
    RequestView,
)


def _bucket(key, reclaimable, clean=None, dirty=None, atoms=None,
            max_atom=None):
    clean = tuple(0 for _ in reclaimable) if clean is None else clean
    dirty = (tuple(r - c for r, c in zip(reclaimable, clean))
             if dirty is None else dirty)
    atoms = (tuple(1 if v > 0 else 0 for v in reclaimable)
             if atoms is None else atoms)
    max_atom = tuple(reclaimable) if max_atom is None else max_atom
    return CapacityBucket(
        key=key, reclaimable_by_rank=tuple(reclaimable),
        clean_by_rank=tuple(clean), dirty_by_rank=tuple(dirty),
        atom_count_by_rank=tuple(atoms), max_atom_by_rank=tuple(max_atom))


def _key(category="human", suffix=SUFFIX_EASY,
         ratio=(1, 1), ports=(0, 1)):
    return CapacityBucketKey(
        category=category, suffix_class=suffix,
        shard_ratio=ratio, writeback_ports=ports)


# ============================================================ θ 数学 ==

class ThetaConsumptionMathTest(unittest.TestCase):
    """§14.2 θ 份额近似消费原式（纯函数钉测）。"""

    def test_theta_covers_worst_rank_with_coupled_release(self):
        # θ = min(1, max(50/100, 80/200)) = 0.5；预计释放 = θ·R capped
        # 于缺口 = (50, 80)；预计脏写回 = θ·dirty = (50, 100)（TP 耦合
        # ——非最重 rank 的释放可超其缺口）；事务数 = ceil(0.5×2) = 1。
        bucket = _bucket(_key(), (100, 200), atoms=(2, 2),
                         max_atom=(60, 120))
        consumptions = consume_gap_theta([bucket], (50, 80))
        self.assertEqual(len(consumptions), 1)
        consumption = consumptions[0]
        self.assertAlmostEqual(consumption.theta, 0.5)
        self.assertEqual(consumption.released_by_rank, (50, 80))
        self.assertEqual(consumption.dirty_released_by_rank, (50, 100))
        self.assertEqual(consumption.expected_transactions, 1)

    def test_bucket_covering_no_gapped_rank_has_theta_zero(self):
        # 桶字节全部落在无缺口 rank 上（rank2 缺口 500、桶 R[2] = 0）
        # ⇒ θ = 0、零消费——不能把耦合分片拆成各 rank 独立 victim。
        bucket = _bucket(_key(ratio=(1, 1, 0), ports=(0, 1, 2)),
                         (10, 10, 0), atoms=(1, 1, 0),
                         max_atom=(10, 10, 0))
        self.assertEqual(consume_gap_theta([bucket], (0, 0, 500)), ())

    def test_multi_bucket_fixed_order_consumes_residual(self):
        # 固定 T 序（human/easy → tool/rest）：第一桶 θ=1 释放 (60, 60)
        # 后 d = (0, 40)；第二桶只覆盖 rank1（R[0]=0 于缺口 rank0 无效
        # ——rank0 缺口已 0）→ θ = min(1, 40/40) = 1、释放 (0, 40)。
        human = _bucket(_key(category="human", suffix=SUFFIX_EASY),
                        (60, 60), atoms=(1, 1), max_atom=(60, 60))
        tool = _bucket(_key(category="tool", suffix=SUFFIX_REST,
                            ratio=(0, 1), ports=(0, 1)),
                       (0, 40), atoms=(0, 1), max_atom=(0, 40))
        consumptions = consume_gap_theta([human, tool], (60, 100))
        self.assertEqual(
            [c.key.category for c in consumptions], ["human", "tool"])
        self.assertEqual(consumptions[0].released_by_rank, (60, 60))
        self.assertAlmostEqual(consumptions[0].theta, 1.0)
        self.assertEqual(consumptions[1].released_by_rank, (0, 40))

    def test_empty_summary_consumes_nothing(self):
        empty = InstanceCapacitySummary(instance_index=0)
        self.assertEqual(consume_gap_theta(empty.buckets, (10, 10)), ())

    def test_shard_ratio_signature_normalizes_proportions(self):
        self.assertEqual(
            shard_ratio_signature((100, 150)),
            shard_ratio_signature((200, 300)))
        self.assertNotEqual(
            shard_ratio_signature((100, 150)),
            shard_ratio_signature((150, 150)))

    def test_negative_gap_rejected(self):
        with self.assertRaises(CapacitySummaryError):
            consume_gap_theta([], (10, -1))


# ==================================================== 分阶段足迹 ==

class PhaseFootprintTest(unittest.TestCase):
    """式(19)/(20)：分阶段峰值分别检查，不机械相加成全程同时驻留。"""

    def test_binding_gap_is_per_phase_max_not_sum(self):
        footprint = PhaseFootprint(
            prepare=(0, 0), prefill=(10, 20),
            decode=(30, 60), merge=(30, 60))
        # 各阶段缺口（free = (5, 15)）：prefill (5,5)、decode (25,45)、
        # merge (25,45)——绑定向量 = 逐 rank max，不是阶段求和。
        self.assertEqual(
            footprint.binding_gap_by_rank((5, 15)), (25, 45))
        self.assertEqual(footprint.binding_phase((5, 15)), PHASE_DECODE)

    def test_single_phase_factory_declares_merge_context(self):
        footprint = PhaseFootprint.single(PHASE_MERGE, (7, 9))
        self.assertEqual(footprint.merge, (7, 9))
        self.assertEqual(footprint.prepare, (0, 0))
        self.assertEqual(footprint.prefill, (0, 0))
        self.assertEqual(footprint.decode, (0, 0))
        self.assertEqual(
            footprint.binding_gap_by_rank((0, 0)), (7, 9))

    def test_mismatched_phase_lengths_rejected(self):
        with self.assertRaises(CapacitySummaryError):
            PhaseFootprint(prepare=(1,), prefill=(1, 2),
                           decode=(1, 2), merge=(1, 2))


# ======================================== 管理器增量维护与只读快照 ==

def _manager(category_mode="typed", layers=4, capacity=100_000):
    hardware = _tiny_hardware(capacity)
    model = _tiny_model(layers)
    topology = _two_instance_topology(hardware)
    return KVCacheManager(
        topology, model, category_mode=category_mode,
        layer_policy="legacy_half")


class ManagerIncrementalMaintenanceTest(unittest.TestCase):
    """完成/激活/引用变化/真实释放事件的增量维护 == 从零重建（strict
    全量对账 ``assert_final_state`` 逐步校验）。"""

    def test_summary_matches_from_scratch_scan(self):
        kv = _manager()
        _seed(kv, "a", 0, 10, 10, "human")
        _seed(kv, "b", 0, 20, 20, "tool")
        _seed(kv, "c", 1, 15, 15, "human")
        # active 会话不是合法 victim（d 不进摘要）。
        _seed(kv, "live", 0, 5, 5, "human")
        kv.prepare_prefill(session_id="live", target_instance_index=0,
                           history_tokens=5, trigger_request_id="r_live")
        summary = kv.capacity_summary_snapshot()
        # 从零逐会话重算（与 _completed_resident_candidates 同一谓词）。
        for instance_index in (0, 1):
            legacy = kv.layer_policy.legacy_resident_prefix
            totals = [0, 0]
            for session in kv._sessions.values():
                if (
                        session.instance_index != instance_index
                        or session.active
                        or session.last_completion_ns is None
                        or session.location not in
                        {kv.LOCAL_HBM, kv.PARTIAL_HBM_REMOTE}
                        or session.session_id in kv._stage_references
                ):
                    continue
                for start, end in (
                        (legacy, session.resident_prefix_layers),
                        (0, legacy)):
                    if end <= start:
                        continue
                    shards = kv_cache_shard_bytes_for_layer_range(
                        kv.model, session.context_tokens, kv.tp_degree,
                        layer_start=start, layer_end=end)
                    totals = [t + s for t, s in zip(totals, shards)]
            self.assertEqual(
                summary.instance(instance_index).total_reclaimable_by_rank,
                tuple(totals),
                f"instance {instance_index} summary != from-scratch scan")

    def test_retained_segments_in_one_bucket_accumulate(self):
        """多个跨轮留存段不能互相覆盖容量摘要。"""
        kv = _manager()
        _seed(kv, "retained", 0, 10, 10, "human")
        segment_bytes = kv_cache_shard_bytes_for_layer_range(
            kv.model, 10, kv.tp_degree, layer_start=0, layer_end=1)
        kv._distributed_retained["retained"] = {
            (0, 0, 1, 0, 10): RetainedSegment(
                "retained", 0, 0, 1, 0, 10, segment_bytes, "resident"),
            (0, 0, 1, 10, 20): RetainedSegment(
                "retained", 0, 0, 1, 10, 20, segment_bytes, "resident"),
        }
        bundle = kv._capacity_retained_contribution_bundle("retained")
        self.assertEqual(len(bundle), 1)
        vectors = tuple(bundle[0][1].values())
        self.assertEqual(len(vectors), 1)
        self.assertEqual(
            vectors[0][0], tuple(2 * value for value in segment_bytes))
        self.assertEqual(vectors[0][3], (2, 2))

    def test_lifecycle_events_maintain_summary(self):
        kv = _manager()
        _seed(kv, "a", 0, 10, 10, "human")
        snapshot = kv.capacity_summary_snapshot()
        total_complete = sum(
            sum(b.reclaimable_by_rank)
            for b in snapshot.instance(0).buckets)
        self.assertGreater(total_complete, 0)
        # 激活（prepare_prefill）→ 退出合法 victim 集合。
        kv.prepare_prefill(session_id="a", target_instance_index=0,
                           history_tokens=10, trigger_request_id="r1")
        snapshot = kv.capacity_summary_snapshot()
        self.assertEqual(sum(
            sum(b.reclaimable_by_rank)
            for b in snapshot.instance(0).buckets), 0)
        # 完成回归。
        kv.mark_complete("a", 20, "human")
        snapshot = kv.capacity_summary_snapshot()
        self.assertEqual(
            sum(sum(b.reclaimable_by_rank)
                for b in snapshot.instance(0).buckets),
            total_complete)
        # 真实释放事件：后缀逐出削减 easy 桶、会话转 PARTIAL。
        kv._evict_suffix(
            kv._sessions["a"], phase="history", reason="test",
            trigger_request_id="r2")
        snapshot = kv.capacity_summary_snapshot()
        total_after_evict = sum(
            sum(b.reclaimable_by_rank)
            for b in snapshot.instance(0).buckets)
        self.assertLess(total_after_evict, total_complete)
        # 引用变化：持阶段引用立即退出可回收视图。
        reference = kv.acquire_stage_reference(
            "a", request_id="r3", stage_epoch=1)
        self.assertEqual(
            sum(sum(b.reclaimable_by_rank)
                for b in kv.capacity_summary_snapshot()
                .instance(0).buckets), 0)
        self.assertTrue(kv.release_stage_reference("a", reference_id=reference))
        self.assertEqual(
            sum(sum(b.reclaimable_by_rank)
                for b in kv.capacity_summary_snapshot()
                .instance(0).buckets),
            total_after_evict)
        # 终态退休（remove_session 族）→ 摘要清零。
        kv.retire_terminal_session("a", 30, request_id="r4")
        self.assertEqual(
            sum(sum(b.reclaimable_by_rank)
                for b in kv.capacity_summary_snapshot()
                .instance(0).buckets), 0)
        # 全程 strict 增量对账无漂移。
        kv.assert_final_state()

    def test_reclaimable_serves_from_summary_totals(self):
        # D11：``_instance_reclaimable_capacity_by_tp_rank`` 改供桶聚合
        # ΣR（旧逐会话扫描退役）——数值与从零重算逐位一致。
        kv = _manager()
        _seed(kv, "a", 0, 10, 10, "human")
        _seed(kv, "b", 0, 20, 20, "tool")
        summary = kv.capacity_summary_snapshot().instance(0)
        totals = summary.total_reclaimable_by_rank
        served = kv._instance_reclaimable_capacity_by_tp_rank(0)
        base = kv._effective_remaining_by_tp_rank(0)
        self.assertEqual(
            served, tuple(b + t for b, t in zip(base, totals)))
        # exclude_session_id 语义不变：本会话字节不入可回收视图。
        excluded = kv._instance_reclaimable_capacity_by_tp_rank(
            0, exclude_session_id="b")
        own = kv_cache_shard_bytes_for_layer_range(
            kv.model, 20, kv.tp_degree,
            layer_start=0, layer_end=kv.model.layers)
        self.assertEqual(
            excluded, tuple(s - o for s, o in zip(served, own)))

    def test_summary_query_does_not_scan_sessions(self):
        # 验收 3：查询路径（冻结快照/聚合合计/容量账）不随已完成
        # session 数扫描——把候选扫描器换成必须不触的哨兵。
        kv = _manager(capacity=10_000_000)
        for index in range(300):
            _seed(kv, f"s{index}", 0, 5, index, "human")

        def _must_not_scan(*args, **kwargs):
            raise AssertionError(
                "capacity summary query scanned candidate sessions")
        original = KVCacheManager._completed_resident_candidates
        KVCacheManager._completed_resident_candidates = _must_not_scan
        try:
            snapshot = kv.capacity_summary_snapshot()
            kv._instance_reclaimable_capacity_by_tp_rank(0)
            kv.capacity_ledger_by_tp_rank(0)
        finally:
            KVCacheManager._completed_resident_candidates = original
        # 桶数有界（typed = 类别×层类 = 2×2），不随 300 个完成会话增长。
        self.assertLessEqual(len(snapshot.instance(0).buckets), 4)
        self.assertGreater(len(snapshot.instance(0).buckets), 0)

    def test_bucket_order_follows_t_passes(self):
        # 快照固定桶序 = T 类别遍历序 × 层类序（human/easy → human/rest
        # → tool/easy → tool/rest）。
        kv = _manager()
        _seed(kv, "h", 0, 10, 10, "human")
        _seed(kv, "t", 0, 10, 20, "tool")
        snapshot = kv.capacity_summary_snapshot().instance(0)
        observed = [
            (bucket.key.category, bucket.key.suffix_class)
            for bucket in snapshot.buckets]
        self.assertEqual(observed, [
            ("human", SUFFIX_EASY), ("human", SUFFIX_REST),
            ("tool", SUFFIX_EASY), ("tool", SUFFIX_REST)])

    def test_lru_mode_single_category(self):
        kv = _manager(category_mode="lru")
        _seed(kv, "a", 0, 10, 10, "tool")
        snapshot = kv.capacity_summary_snapshot().instance(0)
        self.assertEqual(
            {bucket.key.category for bucket in snapshot.buckets}, {"all"})

    def test_clean_dirty_split_and_atom_counts(self):
        # 本仓执行协议：容量逐出恒经 remote_store、账面忽略池侧陈旧副
        # 本（§4.3.1）⇒ 驻留字节无有效池 backing——clean 恒 0（真实推
        # 导）；原子个数 = 合法 victim 数（层类各一原子/victim）。
        kv = _manager()
        _seed(kv, "a", 0, 10, 10, "human")
        _seed(kv, "b", 0, 20, 20, "human")
        snapshot = kv.capacity_summary_snapshot().instance(0)
        for bucket in snapshot.buckets:
            self.assertEqual(
                bucket.clean_by_rank, tuple(0 for _ in bucket.clean_by_rank))
            # 两个合法 victim、每 victim 每层类一原子。
            self.assertEqual(
                bucket.atom_count_by_rank,
                tuple(2 for _ in bucket.atom_count_by_rank))
            self.assertEqual(
                tuple(c + d for c, d in zip(
                    bucket.clean_by_rank, bucket.dirty_by_rank)),
                bucket.reclaimable_by_rank)
        # easy 桶最大原子 = 大会话的后缀字节（legacy 半层边界）。
        legacy = kv.layer_policy.legacy_resident_prefix
        big_suffix = kv_cache_shard_bytes_for_layer_range(
            kv.model, 20, kv.tp_degree,
            layer_start=legacy, layer_end=kv.model.layers)
        easy = next(bucket for bucket in snapshot.buckets
                    if bucket.key.suffix_class == SUFFIX_EASY)
        self.assertEqual(easy.max_atom_by_rank, big_suffix)


class CapacityLedgerConservationTest(unittest.TestCase):
    """式(19)：C = W + O + Z + F 分量分明；Z→O 转账只转账不加总量。"""

    def test_ledger_reconciles_and_z_to_o_transfers(self):
        kv = _manager()
        _seed(kv, "s", 0, 10, 10, "human")
        kv.prepare_prefill(session_id="s", target_instance_index=0,
                           history_tokens=10, trigger_request_id="r1")
        ledger = kv.capacity_ledger_by_tp_rank(0)
        weights = kv.model_weight_bytes_by_tp_rank
        capacity = kv.topology.hardware.local_hbm_capacity_bytes

        def _check(message):
            current = kv.capacity_ledger_by_tp_rank(0)
            effective = kv._effective_remaining_by_tp_rank(0)
            for rank, (cap, weight, materialized, reserved) in enumerate(
                    current):
                self.assertEqual(cap, capacity, message)
                self.assertEqual(weight, weights[rank], message)
                self.assertEqual(
                    cap - weight - materialized - reserved,
                    effective[rank], message)

        _check("initial")
        kv.reserve_request_capacity(
            request_id="r2", session_id="s", instance_index=0,
            final_context_tokens=20)
        ledger_reserved = kv.capacity_ledger_by_tp_rank(0)
        self.assertTrue(any(reserved > 0 for _, _, _, reserved
                            in ledger_reserved))
        _check("after reserve (Z up)")
        kv.expand_prefill(session_id="s", instance_index=0,
                          context_tokens=20, trigger_request_id="r2")
        _check("after materialize")
        kv.release_request_capacity_reservation("r2")
        ledger_final = kv.capacity_ledger_by_tp_rank(0)
        _check("after Z→O transfer")
        # 转账守恒：O 增量 == Z 减量（C−W 总量不变）。
        for before, after in zip(ledger_reserved, ledger_final):
            self.assertEqual(
                (before[2] + before[3]), (after[2] + after[3]))
            self.assertGreater(after[2], before[2])
            self.assertLess(after[3], before[3])


# ==================================== 验收：rank 不均匀缺口正确落位 ==

def _four_rank_manager(layers=4):
    """4-rank 单实例 + heads=2：KV 分片 (x, x, 0, 0)（rank2/3 零 KV）。"""
    from face_scheduler import FaceModel, build_instances, FaceInstanceSpec
    hardware = _tiny_hardware()
    model = FaceModel(
        layers=layers, hidden_size=16, ffn_size=32, num_heads=2,
        vocab_size=32, bytes_per_elem=2, mlp_variant="swiglu")
    topology = build_instances(hardware, (
        FaceInstanceSpec("ins0", "1", (0, 1, 2, 3)),
    ))
    return KVCacheManager(topology, model)


class SkewedLayoutGapTest(unittest.TestCase):
    """验收 1：相同总量 KV、rank 不均匀布局——缺口产生在正确 rank；
    耦合 TP 分片不拆成各 rank 独立 victim。"""

    def test_gap_lands_on_zero_kv_ranks_and_is_marked_uncertain(self):
        kv = _four_rank_manager()
        _seed(kv, "a", 0, 10, 10, "human")
        summary = kv.capacity_summary_snapshot().instance(0)
        # 分片倾斜：(x, x, 0, 0)——可释放字节只在 rank0/1。
        per_token = kv_cache_shard_bytes_for_tokens(kv.model, 1, 4)
        self.assertEqual(per_token[2], 0)
        self.assertEqual(per_token[3], 0)
        for bucket in summary.buckets:
            self.assertEqual(
                bucket.reclaimable_by_rank[2:], (0, 0))
        remaining = kv._effective_remaining_by_tp_rank(0)
        # 足迹四个 rank 等量增长 → 缺口 (50,50,50,50)；rank0/1 由摘要
        # 覆盖（θ·R），rank2/3 R=0 → 残量留在正确 rank → 深缺口不确定。
        demand = tuple(r + 50 for r in remaining)
        load = InstanceLoadView(
            instance_index=0, queued_task_load_ns=0,
            running_task_load_ns=0, active_decode_task_load_ns=0,
            hbm_remaining_bytes_by_tp_rank=remaining,
            reclaimable_bytes_by_tp_rank=(
                summary.total_reclaimable_by_rank),
            capacity_summary=summary,
            capacity_ledger_by_tp_rank=kv.capacity_ledger_by_tp_rank(0))
        rates = JointHardwareRates.from_gbps(
            noc_link_gbps=10.0, pool_port_gbps=5.0, local_hbm_gbps=100.0,
            d2d_latency_ns=10, pool_latency_ns=100)
        model = JointCostModel(
            rates=rates, loads={0: load}, flow_registry=LinkFlowRegistry(),
            service_factors=ServiceFactors(),
            prefill_ns_per_token=1.0, decode_ns_per_token=2.0,
            model_layers=4, instance_tp_size=4,
            route_fn=lambda s, t: ((s, t), 1))
        notes = []
        space_ready = model._space_ready_ns_estimate(
            load, PhaseFootprint.single(PHASE_DECODE, demand), notes,
            instance_index=0)
        # rank0/1 缺口被桶消费（写回腿在）；rank2/3 残量 → 不确定。
        self.assertIn("execution_growth_eviction_writeback_est", notes)
        self.assertIn("execution_growth_capacity_wait_uncertain", notes)
        # 足迹无缺口面（demand ≤ remaining 全 rank）→ A = 0、无注记。
        notes_clear = []
        self.assertEqual(model._space_ready_ns_estimate(
            load, PhaseFootprint.single(PHASE_DECODE, remaining), notes_clear,
            instance_index=0), 0)
        self.assertNotIn(
            "execution_growth_capacity_wait_uncertain", notes_clear)


# ==================================== 验收：干净对象零池写回费 ==

class CleanReleaseZeroWritebackTest(unittest.TestCase):
    """验收 2：干净字节（有可直接复用的有效 backing）不付池写回——
    计价按脏向量、干净释放零池腿（结构分支钉测；本仓执行协议下生产
    摘要 clean 恒空，本测试以摘要构造 API 直接驱动该分支）。"""

    @staticmethod
    def _model_with_summary(summary, remaining=(100, 100),
                            running=0, active=0):
        load = InstanceLoadView(
            instance_index=0, queued_task_load_ns=0,
            running_task_load_ns=running,
            active_decode_task_load_ns=active,
            hbm_remaining_bytes_by_tp_rank=remaining,
            reclaimable_bytes_by_tp_rank=(
                summary.total_reclaimable_by_rank),
            capacity_summary=summary)
        rates = JointHardwareRates.from_gbps(
            noc_link_gbps=10.0, pool_port_gbps=5.0, local_hbm_gbps=100.0,
            d2d_latency_ns=10, pool_latency_ns=100)
        return JointCostModel(
            rates=rates, loads={0: load}, flow_registry=LinkFlowRegistry(),
            service_factors=ServiceFactors(),
            prefill_ns_per_token=1.0, decode_ns_per_token=2.0,
            model_layers=4, instance_tp_size=2,
            route_fn=lambda s, t: ((s, t), 1)), load

    def test_clean_only_bucket_pays_no_pool_writeback(self):
        # 干净桶（backing 在场）：缺口全部由干净释放覆盖 → 无写回腿、
        # A = max(0, 排空) = 0；只付元数据释放费用（低于 ns 粒度、不
        # 虚构费用时间）——注记披露分支在场。
        summary = InstanceCapacitySummary(instance_index=0, buckets=(
            _bucket(_key(), (100, 100),
                    clean=(100, 100), dirty=(0, 0)),))
        model, load = self._model_with_summary(summary)
        notes = []
        demand = tuple(r + 50 for r in load.hbm_remaining_bytes_by_tp_rank)
        space_ready = model._space_ready_ns_estimate(
            load, PhaseFootprint.single(PHASE_DECODE, demand), notes,
            instance_index=0)
        self.assertEqual(space_ready, 0)
        self.assertNotIn("execution_growth_eviction_writeback_est", notes)
        self.assertIn("execution_growth_clean_release_metadata_only", notes)

    def test_mixed_bucket_prices_only_dirty_share(self):
        # 同一 R、clean/dirty 拆分：只按脏份额计池写回——干净半边免费
        # ⇒ A(混合) < A(全脏)。
        mixed = InstanceCapacitySummary(instance_index=0, buckets=(
            _bucket(_key(), (100, 100),
                    clean=(50, 50), dirty=(50, 50)),))
        model_mixed, load_mixed = self._model_with_summary(mixed)
        dirty_only = InstanceCapacitySummary(instance_index=0, buckets=(
            _bucket(_key(), (100, 100),
                    clean=(0, 0), dirty=(100, 100)),))
        model_dirty, load_dirty = self._model_with_summary(dirty_only)
        notes_mixed, notes_dirty = [], []
        demand = tuple(r + 80 for r in (100, 100))
        mixed_ready = model_mixed._space_ready_ns_estimate(
            load_mixed, PhaseFootprint.single(PHASE_DECODE, demand),
            notes_mixed, instance_index=0)
        dirty_ready = model_dirty._space_ready_ns_estimate(
            load_dirty, PhaseFootprint.single(PHASE_DECODE, demand),
            notes_dirty, instance_index=0)
        self.assertGreater(dirty_ready, mixed_ready)
        # 混合桶 θ=80/100：脏释放 = θ·dirty = 40/rank → 写回 =
        # 100+40/5 = 108（干净半边 40/rank 零写回费）；全脏桶脏释放
        # = 80/rank → 100+80/5 = 116。
        self.assertEqual(mixed_ready, 100 + 40 // 5)
        self.assertEqual(dirty_ready, 100 + 80 // 5)
        self.assertIn("execution_growth_eviction_writeback_est", notes_mixed)


# ============================ 深缺口/外部性/生产接线 ==

class DeepGapAndExternalityTest(unittest.TestCase):
    """§14.3 深缺口近似与 §14.4 外部性不计价。"""

    @staticmethod
    def _model(load):
        rates = JointHardwareRates.from_gbps(
            noc_link_gbps=10.0, pool_port_gbps=5.0, local_hbm_gbps=100.0,
            d2d_latency_ns=10, pool_latency_ns=100)
        return JointCostModel(
            rates=rates, loads={0: load}, flow_registry=LinkFlowRegistry(),
            service_factors=ServiceFactors(),
            prefill_ns_per_token=1.0, decode_ns_per_token=2.0,
            model_layers=4, instance_tp_size=2,
            route_fn=lambda s, t: ((s, t), 1))

    def test_deep_gap_marks_capacity_wait_uncertain_with_drain(self):
        summary = InstanceCapacitySummary(instance_index=0, buckets=(
            _bucket(_key(), (100, 100)),))
        load = InstanceLoadView(
            instance_index=0, queued_task_load_ns=0,
            running_task_load_ns=300, active_decode_task_load_ns=400,
            hbm_remaining_bytes_by_tp_rank=(100, 100),
            reclaimable_bytes_by_tp_rank=(100, 100),
            capacity_summary=summary)
        model = self._model(load)
        notes = []
        space_ready = model._space_ready_ns_estimate(
            load, PhaseFootprint.single(PHASE_DECODE, (1000, 1000)), notes,
            instance_index=0)
        # 写回（θ=1、脏释放 100/rank → 120）< 活跃排空 700 ⇒ A = 700，
        # 并标 capacity_wait_uncertain（有限近似、非可执行保证）。
        self.assertEqual(space_ready, 700)
        self.assertIn("execution_growth_capacity_wait_uncertain", notes)

    def test_no_future_victim_recovery_cost_added(self):
        # §14.4：当前请求只承担真实前驱写回——A 恰等于写回腿，无任何
        # victim 未来恢复（池→目标恢复腿）计价项混入。
        summary = InstanceCapacitySummary(instance_index=0, buckets=(
            _bucket(_key(), (100, 100)),))
        load = InstanceLoadView(
            instance_index=0, queued_task_load_ns=0,
            running_task_load_ns=0, active_decode_task_load_ns=0,
            hbm_remaining_bytes_by_tp_rank=(100, 100),
            reclaimable_bytes_by_tp_rank=(100, 100),
            capacity_summary=summary)
        model = self._model(load)
        notes = []
        space_ready = model._space_ready_ns_estimate(
            load, PhaseFootprint.single(PHASE_DECODE, (200, 200)), notes,
            instance_index=0)
        self.assertEqual(space_ready, 100 + 100 // 5)
        for note in notes:
            self.assertNotIn("restore", note)

    def test_ledger_mismatch_fails_closed(self):
        summary = InstanceCapacitySummary(instance_index=0, buckets=(
            _bucket(_key(), (100, 100)),))
        load = InstanceLoadView(
            instance_index=0, queued_task_load_ns=0,
            running_task_load_ns=0, active_decode_task_load_ns=0,
            hbm_remaining_bytes_by_tp_rank=(100, 100),
            reclaimable_bytes_by_tp_rank=(100, 100),
            capacity_summary=summary,
            # 台账 (C, W, O, Z) 与 remaining 不守恒 → fail-closed。
            capacity_ledger_by_tp_rank=((300, 0, 0, 0), (300, 0, 0, 0)))
        model = self._model(load)
        with self.assertRaises(Exception):
            model._space_ready_ns_estimate(
                load, PhaseFootprint.single(PHASE_DECODE, (200, 200)),
                [], instance_index=0)


class ProductionWiringTest(unittest.TestCase):
    """SH `_joint_cost_model` 供真实管理器快照：候选只查询、台账守恒、
    θ 消费注记可检索（estimate 路径消费真摘要）。"""

    def test_load_view_carries_manager_summary_and_reconciles(self):
        kv = _manager()
        _seed(kv, "a", 0, 10, 10, "human")
        _seed(kv, "b", 1, 20, 20, "tool")
        snapshot = kv.capacity_summary_snapshot()
        for instance_index in (0, 1):
            summary = snapshot.instance(instance_index)
            served = kv._instance_reclaimable_capacity_by_tp_rank(
                instance_index)
            base = kv._effective_remaining_by_tp_rank(instance_index)
            self.assertEqual(
                served,
                tuple(b + t for b, t in zip(
                    base, summary.total_reclaimable_by_rank)))
            ledger = kv.capacity_ledger_by_tp_rank(instance_index)
            for rank, (cap, weight, materialized, reserved) in enumerate(
                    ledger):
                self.assertEqual(
                    cap - weight - materialized - reserved, base[rank])

    def test_estimate_consumes_real_summary_via_theta_note(self):
        # 制造真实缺口：容量 = 权重 + 大会话 + 50 B/rank 余量——候选
        # stay 足迹（input+growth）超 remaining → 摘要 θ 消费在场。
        from face_scheduler import model_weight_shard_bytes_by_tp_rank
        model = _tiny_model(4)
        weights = model_weight_shard_bytes_by_tp_rank(model, 2)
        hog_per_rank = kv_cache_shard_bytes_for_tokens(model, 100, 2)[0]
        capacity = weights[0] + hog_per_rank + 50
        kv = _manager(capacity=capacity)
        _seed(kv, "hog", 0, 100, 10, "human")
        summary = kv.capacity_summary_snapshot().instance(0)
        remaining = kv._effective_remaining_by_tp_rank(0)
        self.assertEqual(remaining, (50, 50))
        load = InstanceLoadView(
            instance_index=0, queued_task_load_ns=0,
            running_task_load_ns=0, active_decode_task_load_ns=0,
            hbm_remaining_bytes_by_tp_rank=remaining,
            reclaimable_bytes_by_tp_rank=(
                summary.total_reclaimable_by_rank),
            capacity_summary=summary,
            capacity_ledger_by_tp_rank=kv.capacity_ledger_by_tp_rank(0))
        rates = JointHardwareRates.from_gbps(
            noc_link_gbps=10.0, pool_port_gbps=5.0, local_hbm_gbps=100.0,
            d2d_latency_ns=10, pool_latency_ns=100)
        model = JointCostModel(
            rates=rates, loads={0: load}, flow_registry=LinkFlowRegistry(),
            service_factors=ServiceFactors(),
            prefill_ns_per_token=1.0, decode_ns_per_token=2.0,
            model_layers=4, instance_tp_size=2,
            route_fn=lambda s, t: ((s, t), 1))
        session = SessionKVView(
            session_id="hog", home_instance=0, resident_instance=0,
            location="local_hbm", history_tokens=100,
            resident_prefix_layers=4,
            history_bytes_by_tp_rank=kv_cache_shard_bytes_for_tokens(
                kv.model, 100, kv.tp_degree),
            missing_bytes_by_tp_rank=(0, 0))
        request = RequestView(
            request_id="r", session_id="hog", input_tokens=5,
            history_tokens_before=100, estimated_decode_tokens=4,
            horizon_source="session_online_mean",
            input_kv_bytes_by_tp_rank=kv_cache_shard_bytes_for_layer_range(
                kv.model, 5, kv.tp_degree,
                layer_start=0, layer_end=kv.model.layers))
        candidate = model.estimate_P_with_continuation(
            session=session, request=request, instance_index=0,
            action="stay")
        self.assertTrue(candidate.applicable)
        # D11 披露面：分阶段峰值 + 摘要 θ 消费注记随候选落盘。
        notes = candidate.breakdown.notes
        self.assertTrue(any(
            note.startswith("execution_growth_phase_peaks(")
            for note in notes))
        self.assertTrue(any(
            note.startswith("execution_growth_capacity_summary_theta(")
            for note in notes))


if __name__ == "__main__":
    unittest.main()
