#!/usr/bin/env python3
"""test_prefill_remote_read_manager.py -- 规格书 §6 第 1/6/7 组：prefill
remote-read 前缀读流（2026-09-25）的 KV 管理器侧回归（纯规划函数
plan_prefill_remote_read_transfers + prepare_prefill 三元返回值口径）。

目标语义（PARTIAL 基）：home HBM 保存前缀 [0,p)、remote pool 保存后缀
[p,L)；选 remote-read 且 exec != home 时 prefill 两条腿并行分叉——
home 前缀 [0,p) 经 NoC 读流（stream_only，瞬时、不入任何持久账本）、
remote pool 后缀 [p,L) 经 remote_load 池恢复（入 exec 容量账）。

覆盖：
  1. PARTIAL 基管理器口径（组 1）：
     a. 基形态：home 前缀 [0,p) / 池后缀 [p,L)；
     b. 前缀读流覆盖 [0,p)（kind/phase/reason/stream_only/source/target/
        驻留指针恒 p、非 C13/C15 腿、字节逐组派生）；
     c. 后缀恢复腿覆盖 [p,L)（prepare 三元组只含 remote_load）；
     d. 两腿层区间不相交、并集铺满 [0,L)、共享同一 trigger_request_id；
     e. exec shard_bytes 只含后缀（D1 两口径：ctx=0 增量）＋ home base
        元数据不变 ＋ 前缀瞬时 staging 不进容量账本；
     f. 规划纯度（快照/账本/账册零副作用）＋ 须以基形态会话调用。
  6. LOCAL 基回归（组 6）：p=L；读流覆盖全部层（字节 ≡ 整份历史）；
     无后缀恢复腿；decode credit 读计划仍覆盖全部层。
  7. REMOTE 基回归（组 7）：新规划器 fail-closed（REMOTE 基 / exec ==
     驻留实例 → RuntimeError；history_tokens 不符 → ValueError）；
     prepare_prefill 对 REMOTE 基零读流腿、decode credit 计划 None
     ——新增 prefill 读流规划不得绕过 remote-read 适用性约束。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 online/test_prefill_remote_read_manager.py
      （或 pytest 同路径）
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

from face_scheduler import (  # noqa: E402
    FaceHardware,
    FaceInstanceSpec,
    FaceModel,
    KVCacheManager,
    build_instances,
    kv_cache_shard_bytes_for_layer_range,
    kv_cache_shard_bytes_for_tokens,
    plan_layer_groups,
    plan_suffix_restore_groups,
    RESTORE_GROUP_LAYERS,
)
from online.sh30_online_scheduler import Sh30OnlineScheduler  # noqa: E402

HOME_INSTANCE = 0   # ranks (0, 1)
EXEC_INSTANCE = 1   # ranks (2, 3)
HISTORY_TOKENS = 100
PREFIX_LAYERS = 12  # PARTIAL 基驻留前缀 p（> RESTORE_GROUP_LAYERS ⇒ 多组）


def _manager(*, layers: int = 16, capacity_bytes: int = 200_000):
    """16 层双实例 fixture（每 token 每层每 rank 4B；与
    test_joint_layer_restore 的 _restore_manager 同款）。"""
    hardware = FaceHardware(
        mesh_rows=2,
        mesh_cols=2,
        local_hbm_capacity_bytes=capacity_bytes,
        local_hbm_bandwidth_gbps=1.0,
        d2d_bandwidth_gbps=2.0,
        peak_perf_tflops=1.0,
        d2d_latency_ns=0,
        local_hbm_latency_ns=0,
    )
    model = FaceModel(
        layers=layers,
        hidden_size=4,
        ffn_size=4,
        num_heads=2,
        vocab_size=4,
        bytes_per_elem=1,
        mlp_variant="gelu",
    )
    topology = build_instances(
        hardware,
        (
            FaceInstanceSpec("ins0", "1", (0, 1)),
            FaceInstanceSpec("ins1", "2", (2, 3)),
        ),
    )
    manager = KVCacheManager(
        topology, model, category_mode="typed",
        layer_policy="adaptive",
        pool_bandwidth_gbps=4.0,
        pool_latency_ns=10)
    return model, manager


def _seed_resident(manager, *, tokens: int = HISTORY_TOKENS) -> None:
    """在 home（instance 0）完成一轮生长到 tokens 的 LOCAL 会话。"""
    manager.prepare_prefill(
        session_id="s", target_instance_index=HOME_INSTANCE,
        history_tokens=0, trigger_request_id="s_seed")
    manager.expand_prefill(
        session_id="s", instance_index=HOME_INSTANCE,
        context_tokens=tokens, trigger_request_id="s_seed")
    manager.mark_complete("s", tokens)


def _seed_partial(manager, *, prefix_layers: int = PREFIX_LAYERS,
                  tokens: int = HISTORY_TOKENS) -> None:
    """完成一轮并逐出 [prefix_layers, L) → PARTIAL 会话（home 不变）。"""
    _seed_resident(manager, tokens=tokens)
    manager._evict_suffix(
        manager._sessions["s"], phase="completion", reason="fixture",
        trigger_request_id="s_seed", layer_start=prefix_layers)


def _seed_remote(manager, *, tokens: int = HISTORY_TOKENS) -> None:
    """完成一轮并整份逐出 → REMOTE 基（仅池 backing，无驻留实例）。"""
    _seed_resident(manager, tokens=tokens)
    manager._evict_session(
        manager._sessions["s"], phase="completion", reason="fixture",
        trigger_request_id="s_seed")


def _layer_ranges(transfers):
    """传输序列的层区间（按消费顺序去排序输入依赖前保持发射序）。"""
    return tuple((t.layer_start, t.layer_end) for t in transfers)


def _assert_contiguous_cover(test, ranges, start: int, end: int) -> None:
    """层区间序列无空隙、无重叠、恰好铺满 [start, end)。"""
    test.assertTrue(ranges, f"expected a non-empty cover of [{start}, {end})")
    cursor = start
    for layer_start, layer_end in ranges:
        test.assertEqual(layer_start, cursor,
                         "layer ranges must be contiguous in consumption order")
        test.assertGreater(layer_end, layer_start)
        cursor = layer_end
    test.assertEqual(cursor, end,
                     f"layer ranges must cover exactly [{start}, {end})")


def _instance_used_bytes(manager, instance_index: int) -> int:
    """实例级容量账本占用（rank 求和；hbm_snapshots 逐 rank 披露）。"""
    return sum(snapshot.used_bytes
               for snapshot in manager.hbm_snapshots(instance_index))


def _credit_plan_for(manager, model, session_id: str):
    """decode remote-read credit 持久读计划（_joint_remote_read_credit_plan
    的最小替身直连：该函数只读 kv_manager/model/topology/hardware 四个
    成员——替身对齐 __init__ 后按真实会话状态派生，不 mock 派生逻辑）。"""
    scheduler = Sh30OnlineScheduler.__new__(Sh30OnlineScheduler)
    scheduler.kv_manager = manager
    scheduler.model = model
    scheduler.topology = manager.topology
    scheduler.hardware = manager.topology.hardware
    runtime = SimpleNamespace(
        session_id=session_id,
        origin_home_instance=HOME_INSTANCE,
        joint_input_tokens=10,
        decode_length=8,
    )
    return scheduler._joint_remote_read_credit_plan(
        runtime, EXEC_INSTANCE)


class PartialRemoteReadManagerTests(unittest.TestCase):
    """组 1：PARTIAL 基 remote-read 管理器口径（前缀读流 × 后缀池恢复）。"""

    def setUp(self):
        self.model, self.manager = _manager()
        _seed_partial(self.manager, prefix_layers=PREFIX_LAYERS)
        # 夹具前提：p > RESTORE_GROUP_LAYERS ⇒ 前缀读流必为多组切分
        # （逐组结构断言不是单组退化形态）。
        self.assertGreater(PREFIX_LAYERS, RESTORE_GROUP_LAYERS)
        self.assertGreater(len(plan_layer_groups(0, PREFIX_LAYERS)), 1)
        self.session = self.manager._sessions["s"]
        self.base_snapshot = self.manager.session_snapshot("s")
        # 前缀读流以基形态会话规划（prepare_prefill 之前）。
        self.prefix_transfers = self.manager.plan_prefill_remote_read_transfers(
            self.session, EXEC_INSTANCE, HISTORY_TOKENS, "r1")
        self.prefix_bytes = kv_cache_shard_bytes_for_layer_range(
            self.model, HISTORY_TOKENS, 2,
            layer_start=0, layer_end=PREFIX_LAYERS)
        self.suffix_bytes = kv_cache_shard_bytes_for_layer_range(
            self.model, HISTORY_TOKENS, 2,
            layer_start=PREFIX_LAYERS, layer_end=self.model.layers)

    def test_base_form_prefix_at_home_and_suffix_in_pool(self):
        # 基形态：home 驻留前缀 [0,p)，其余在池。
        self.assertIs(self.session.location, KVCacheManager.PARTIAL_HBM_REMOTE)
        self.assertEqual(self.session.instance_index, HOME_INSTANCE)
        self.assertEqual(self.session.home_instance, HOME_INSTANCE)
        self.assertEqual(self.session.resident_prefix_layers, PREFIX_LAYERS)
        self.assertLess(self.session.resident_prefix_layers,
                        self.model.layers)  # 真 PARTIAL
        self.assertEqual(self.session.context_tokens, HISTORY_TOKENS)
        # 快照分账：local = 前缀 [0,p)、remote = 后缀 [p,L)。
        self.assertEqual(
            tuple(self.base_snapshot.local_shard_bytes), self.prefix_bytes)
        self.assertEqual(
            tuple(self.base_snapshot.remote_shard_bytes), self.suffix_bytes)
        self.assertEqual(self.base_snapshot.local_bytes, sum(self.prefix_bytes))
        self.assertEqual(self.base_snapshot.remote_bytes,
                         sum(self.suffix_bytes))

    def test_prefix_read_stream_covers_resident_prefix(self):
        transfers = self.prefix_transfers
        # 逐组切分 = plan_layer_groups(0, p)（公共规划器消费顺序）。
        self.assertEqual(
            _layer_ranges(transfers),
            plan_layer_groups(0, PREFIX_LAYERS))
        _assert_contiguous_cover(
            self, _layer_ranges(transfers), 0, PREFIX_LAYERS)
        for transfer in transfers:
            self.assertEqual(transfer.kind, "noc_migrate")
            self.assertEqual(transfer.phase, "prefill")
            self.assertEqual(transfer.reason, "remote_read_prefill_prefix")
            # 瞬时读流标记：不入任何持久账本（规格 §一.7）。
            self.assertTrue(transfer.stream_only)
            self.assertEqual(transfer.source_instance_index, HOME_INSTANCE)
            self.assertEqual(transfer.target_instance_index, EXEC_INSTANCE)
            self.assertEqual(transfer.session_id, "s")
            self.assertEqual(transfer.trigger_request_id, "r1")
            # 驻留指针恒 = p：读流无驻留推进（镜像 decode credit 口径）。
            self.assertEqual(
                (transfer.resident_prefix_layers_before,
                 transfer.resident_prefix_layers_after),
                (PREFIX_LAYERS, PREFIX_LAYERS))
            # 非 C13 交接块 / 非 C15 恢复组腿（独立前缀读流科目）。
            self.assertIsNone(transfer.handoff_chunk)
            self.assertIsNone(transfer.restore_group)
            # 逐组字节按层区间派生（与 _noc_transfer 同源）。
            expected = kv_cache_shard_bytes_for_layer_range(
                self.model, HISTORY_TOKENS, 2,
                layer_start=transfer.layer_start, layer_end=transfer.layer_end)
            self.assertEqual(transfer.total_bytes, sum(expected))
        # 总字节 = 前缀 [0, p) 整段派生（读流恰好覆盖驻留前缀一遍）。
        self.assertEqual(
            sum(t.total_bytes for t in transfers), sum(self.prefix_bytes))

    def test_planner_accepts_positional_and_keyword_args(self):
        positional = self.prefix_transfers
        keyword = self.manager.plan_prefill_remote_read_transfers(
            session=self.session,
            target_instance_index=EXEC_INSTANCE,
            history_tokens=HISTORY_TOKENS,
            trigger_request_id="r1")
        self.assertEqual(keyword, positional)

    def test_suffix_restore_leg_covers_pool_suffix(self):
        _before, transfers, evictions = self.manager.prepare_prefill(
            session_id="s", target_instance_index=EXEC_INSTANCE,
            history_tokens=HISTORY_TOKENS, trigger_request_id="r1",
            action="remote-read")
        # 夹具容量充足 ⇒ 无逐出；prepare 三元组只含后缀恢复腿。
        self.assertEqual(evictions, ())
        self.assertEqual(
            _layer_ranges(transfers),
            plan_suffix_restore_groups(PREFIX_LAYERS, self.model.layers))
        _assert_contiguous_cover(
            self, _layer_ranges(transfers), PREFIX_LAYERS, self.model.layers)
        for transfer in transfers:
            self.assertEqual(transfer.kind, "remote_load")
            self.assertEqual(transfer.phase, "history")
            self.assertEqual(
                transfer.reason, "history_suffix_pool_restore_working_copy")
            # 池恢复是持久口径（入 exec 容量账），非瞬时读流。
            self.assertFalse(transfer.stream_only)
            self.assertEqual(
                transfer.target_instance_index, EXEC_INSTANCE)
        # 前缀读流不进 prepare 三元组（由新规划器独立供给构图器）。
        self.assertNotIn(
            "noc_migrate", {t.kind for t in transfers},
            "prefix read stream must be planned by the standalone planner, "
            "not leaked into prepare_prefill transfers")
        # 三元返回值第一元 = 基形态快照（口径不变）。
        self.assertEqual(_before, self.base_snapshot)

    def test_two_legs_partition_all_layers_under_same_trigger(self):
        _before, transfers, _evictions = self.manager.prepare_prefill(
            session_id="s", target_instance_index=EXEC_INSTANCE,
            history_tokens=HISTORY_TOKENS, trigger_request_id="r1",
            action="remote-read")
        prefix_ranges = _layer_ranges(self.prefix_transfers)
        suffix_ranges = _layer_ranges(transfers)
        # 两腿共享同一准入触发标识（同一 frontier 分叉的配对凭证）。
        self.assertTrue(
            {t.trigger_request_id for t in self.prefix_transfers}
            == {t.trigger_request_id for t in transfers} == {"r1"})
        # 不相交且并集恰好铺满 [0, L)：前缀读流 + 后缀池恢复全覆盖。
        self.assertEqual(prefix_ranges + suffix_ranges,
                         plan_layer_groups(0, PREFIX_LAYERS)
                         + plan_suffix_restore_groups(
                             PREFIX_LAYERS, self.model.layers))
        _assert_contiguous_cover(
            self, prefix_ranges + suffix_ranges, 0, self.model.layers)

    def test_exec_shard_bytes_suffix_only_and_home_base_metadata_intact(self):
        exec_used_before = _instance_used_bytes(
            self.manager, EXEC_INSTANCE)
        home_used_before = _instance_used_bytes(self.manager, HOME_INSTANCE)
        _before, _transfers, _evictions = self.manager.prepare_prefill(
            session_id="s", target_instance_index=EXEC_INSTANCE,
            history_tokens=HISTORY_TOKENS, trigger_request_id="r1",
            action="remote-read")
        session = self.manager._sessions["s"]
        # exec shard_bytes 只含后缀（物理真值），不含前缀读流 staging。
        self.assertEqual(tuple(session.shard_bytes), self.suffix_bytes)
        self.assertEqual(session.total_bytes, sum(self.suffix_bytes))
        self.assertNotEqual(
            tuple(session.shard_bytes),
            kv_cache_shard_bytes_for_tokens(self.model, HISTORY_TOKENS, 2))
        # D1 两口径分离：context_tokens 为增量口径（0 起步）。
        self.assertEqual(session.context_tokens, 0)
        # home base 元数据不变：逻辑 home 不随异地执行迁移，权威基础
        # 前缀仍描述 home [0, p)。
        self.assertEqual(session.home_instance, HOME_INSTANCE)
        self.assertEqual(session.base_location,
                         KVCacheManager.PARTIAL_HBM_REMOTE)
        self.assertEqual(session.base_resident_prefix_layers, PREFIX_LAYERS)
        self.assertEqual(session.base_history_tokens, HISTORY_TOKENS)
        self.assertEqual(session.working_kind, "remote-read")
        snapshot = self.manager.session_snapshot("s")
        self.assertEqual(snapshot.home_instance, HOME_INSTANCE)
        self.assertEqual(snapshot.working_instance_index, EXEC_INSTANCE)
        self.assertEqual(tuple(snapshot.local_shard_bytes), self.suffix_bytes)
        # 容量账本：exec 增量 == 后缀字节（前缀瞬时 staging 不入账），
        # home 占用逐字节不变。
        self.assertEqual(
            _instance_used_bytes(self.manager, EXEC_INSTANCE)
            - exec_used_before,
            sum(self.suffix_bytes))
        self.assertEqual(_instance_used_bytes(self.manager, HOME_INSTANCE),
                         home_used_before)

    def test_planner_is_pure_over_state_and_ledgers(self):
        snapshot_before = self.manager.session_snapshot("s")
        used_before = (
            _instance_used_bytes(self.manager, HOME_INSTANCE),
            _instance_used_bytes(self.manager, EXEC_INSTANCE))
        self.manager.plan_prefill_remote_read_transfers(
            self.session, EXEC_INSTANCE, HISTORY_TOKENS, "r2")
        # 会话快照与两侧容量账本零变化。
        self.assertEqual(self.manager.session_snapshot("s"), snapshot_before)
        self.assertEqual(
            (_instance_used_bytes(self.manager, HOME_INSTANCE),
             _instance_used_bytes(self.manager, EXEC_INSTANCE)),
            used_before)
        # 会话字段与事务账册零变化（无 journal 登记、无 merge 版本写入）。
        self.assertIsNone(self.session.restore_journal)
        self.assertIsNone(self.session.copy_handoff)
        self.assertIsNone(self.session.last_merged_request_id)
        self.assertIs(self.session.location, KVCacheManager.PARTIAL_HBM_REMOTE)
        self.assertEqual(self.session.resident_prefix_layers, PREFIX_LAYERS)
        self.assertEqual(self.session.context_tokens, HISTORY_TOKENS)
        # 纯度不改变后续 prepare 行为：同一 planner 结果可复现。
        again = self.manager.plan_prefill_remote_read_transfers(
            self.session, EXEC_INSTANCE, HISTORY_TOKENS, "r1")
        self.assertEqual(again, self.prefix_transfers)

    def test_planner_requires_base_form_session(self):
        # 工作副本形态（prepare 之后）不得再当基规划：home==exec 的
        # 读流即自读/摧毁权威基础——fail-closed（规格：须以基形态调用）。
        _before, _transfers, _evictions = self.manager.prepare_prefill(
            session_id="s", target_instance_index=EXEC_INSTANCE,
            history_tokens=HISTORY_TOKENS, trigger_request_id="r1",
            action="remote-read")
        with self.assertRaises(RuntimeError):
            self.manager.plan_prefill_remote_read_transfers(
                self.manager._sessions["s"], EXEC_INSTANCE,
                HISTORY_TOKENS, "r1")


class LocalBaseRemoteReadRegressionTests(unittest.TestCase):
    """组 6：LOCAL 基回归锚（p == L：读流全覆盖、无后缀腿、decode credit
    仍全覆盖）。"""

    def setUp(self):
        self.model, self.manager = _manager()
        _seed_resident(self.manager, tokens=HISTORY_TOKENS)
        self.session = self.manager._sessions["s"]

    def test_local_base_has_full_resident_prefix(self):
        self.assertIs(self.session.location, KVCacheManager.LOCAL_HBM)
        # p = L：全层驻留（LOCAL 基定义）。
        self.assertEqual(self.session.resident_prefix_layers,
                         self.model.layers)
        self.assertEqual(self.session.home_instance, HOME_INSTANCE)

    def test_prefill_read_stream_covers_all_layers(self):
        transfers = self.manager.plan_prefill_remote_read_transfers(
            self.session, EXEC_INSTANCE, HISTORY_TOKENS, "r1")
        self.assertEqual(_layer_ranges(transfers),
                         plan_layer_groups(0, self.model.layers))
        _assert_contiguous_cover(self, _layer_ranges(transfers),
                                 0, self.model.layers)
        for transfer in transfers:
            self.assertEqual(transfer.kind, "noc_migrate")
            self.assertEqual(transfer.phase, "prefill")
            self.assertEqual(transfer.reason, "remote_read_prefill_prefix")
            self.assertTrue(transfer.stream_only)
            self.assertEqual(transfer.source_instance_index, HOME_INSTANCE)
            self.assertEqual(transfer.target_instance_index, EXEC_INSTANCE)
            self.assertEqual(
                (transfer.resident_prefix_layers_before,
                 transfer.resident_prefix_layers_after),
                (self.model.layers, self.model.layers))
        # 字节回归锚：p == L ⇒ 读流总量逐字节等于整份历史 KV
        # （全层派生 ≡ 旧全层口径）。
        full_bytes = kv_cache_shard_bytes_for_tokens(
            self.model, HISTORY_TOKENS, 2)
        self.assertEqual(sum(t.total_bytes for t in transfers),
                         sum(full_bytes))

    def test_no_suffix_restore_leg_for_local_base(self):
        _before, transfers, evictions = self.manager.prepare_prefill(
            session_id="s", target_instance_index=EXEC_INSTANCE,
            history_tokens=HISTORY_TOKENS, trigger_request_id="r1",
            action="remote-read")
        # LOCAL 基无缺失后缀：零池恢复腿、零逐出（夹具容量充足）。
        self.assertEqual(transfers, ())
        self.assertEqual(evictions, ())
        self.assertNotIn(
            "remote_load", {t.kind for t in transfers},
            "LOCAL base must not plan any suffix remote_load leg")
        session = self.manager._sessions["s"]
        self.assertIsNone(session.restore_journal)
        # 回归锚（LOCAL 基工作副本零化起算）：shard_bytes 恒等
        # kv(context_tokens)@全层 == 0。
        self.assertEqual(
            tuple(session.shard_bytes),
            kv_cache_shard_bytes_for_tokens(self.model, 0, 2))
        self.assertEqual(session.total_bytes, 0)
        self.assertEqual(session.context_tokens, 0)

    def test_decode_credit_still_covers_all_layers(self):
        self.manager.plan_prefill_remote_read_transfers(
            self.session, EXEC_INSTANCE, HISTORY_TOKENS, "r1")
        _before, _transfers, _evictions = self.manager.prepare_prefill(
            session_id="s", target_instance_index=EXEC_INSTANCE,
            history_tokens=HISTORY_TOKENS, trigger_request_id="r1",
            action="remote-read")
        plan = _credit_plan_for(self.manager, self.model, "s")
        self.assertIsNotNone(plan)
        # decode credit 读全部层（LOCAL 基 p==L 不变锚——新增 prefill
        # 规划不得收窄 decode 读流层区间）。
        self.assertEqual(plan["read_prefix_layers"], self.model.layers)
        self.assertEqual(plan["home_instance"], HOME_INSTANCE)
        self.assertEqual(plan["exec_instance"], EXEC_INSTANCE)
        self.assertEqual(plan["steps"], 8)
        context_per_step = HISTORY_TOKENS + 10 + 8
        expected_per_step = kv_cache_shard_bytes_for_layer_range(
            self.model, context_per_step, 2,
            layer_start=0, layer_end=self.model.layers)
        self.assertEqual(
            tuple(spec[2] for spec in plan["shard_specs"]),
            expected_per_step)
        # I1 计划基数：total = Σshard × steps。
        self.assertEqual(
            plan["total_bytes"],
            sum(spec[2] for spec in plan["shard_specs"]) * plan["steps"])


class RemoteBaseApplicabilityRegressionTests(unittest.TestCase):
    """组 7：REMOTE 基回归——remote-read 仍不适用，新增 prefill 读流
    规划器不得绕过适用性约束（全链 fail-closed）。"""

    def setUp(self):
        self.model, self.manager = _manager()
        _seed_remote(self.manager, tokens=HISTORY_TOKENS)
        self.session = self.manager._sessions["s"]

    def test_remote_base_planner_fail_closed(self):
        self.assertIs(self.session.location, KVCacheManager.REMOTE_MEMORY)
        self.assertIsNone(self.session.instance_index)
        self.assertEqual(self.session.resident_prefix_layers, 0)
        with self.assertRaises(RuntimeError) as ctx:
            self.manager.plan_prefill_remote_read_transfers(
                self.session, EXEC_INSTANCE, HISTORY_TOKENS, "r1")
        self.assertIn("LOCAL/PARTIAL", str(ctx.exception))

    def test_prepare_prefill_yields_no_remote_read_legs(self):
        # 适用性约束不因新规划器松动：REMOTE 基 remote-read 在管理器侧
        # 只允许零化工作副本，且三元组零传输腿——构图器无任何读流可消费
        # （前缀读流规划器不能被用来给 REMOTE 基补腿）。
        _before, transfers, evictions = self.manager.prepare_prefill(
            session_id="s", target_instance_index=EXEC_INSTANCE,
            history_tokens=HISTORY_TOKENS, trigger_request_id="r1",
            action="remote-read")
        self.assertEqual(transfers, ())
        self.assertEqual(evictions, ())
        session = self.manager._sessions["s"]
        self.assertEqual(tuple(session.shard_bytes),
                         (0,) * 2)
        self.assertEqual(session.context_tokens, 0)
        self.assertEqual(session.working_kind, "remote-read")
        self.assertIsNone(session.restore_journal)
        # decode 侧同样拒绝：REMOTE 基 credit 计划防御 None。
        self.assertIsNone(_credit_plan_for(self.manager, self.model, "s"))

    def test_exec_equal_resident_fail_closed(self):
        # PARTIAL 基上 exec == 驻留实例：适用性合同破损必须 raise
        # （防读流退化成自读）——新规划器不得放过该组合。
        model, manager = _manager()
        _seed_partial(manager, prefix_layers=PREFIX_LAYERS)
        session = manager._sessions["s"]
        with self.assertRaises(RuntimeError):
            manager.plan_prefill_remote_read_transfers(
                session, session.instance_index, HISTORY_TOKENS, "r1")

    def test_history_token_mismatch_fail_closed(self):
        # 历史元数据不符：ValueError（字节源派生前提 = context_tokens
        # == history_tokens，不得静默按请求侧 token 规划）。该约束在
        # LOCAL/PARTIAL 基形态上生效（REMOTE 基先撞适用性守卫，见
        # test_remote_base_planner_fail_closed 的 fail-closed 序）。
        model, manager = _manager()
        _seed_partial(manager, prefix_layers=PREFIX_LAYERS)
        session = manager._sessions["s"]
        with self.assertRaises(ValueError):
            manager.plan_prefill_remote_read_transfers(
                session, EXEC_INSTANCE, HISTORY_TOKENS + 1, "r1")
        # 会话状态在失败规划后不变（fail-closed 无副作用）。
        self.assertIs(session.location, KVCacheManager.PARTIAL_HBM_REMOTE)
        self.assertEqual(session.context_tokens, HISTORY_TOKENS)


if __name__ == "__main__":
    unittest.main()
