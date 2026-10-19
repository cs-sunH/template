#!/usr/bin/env python3
"""test_kv_location_directory.py -- W4 权威区间目录单元测试。

钉测面（《joint仓库当下+历史修改方案》§6.1/§7.4、§12.1 W4 验收）：

1. 状态机：WRITING→READY→EVICTED_TO_POOL→RESTORING→READY 合法链；
   表外转移（READY→WRITING、RELEASED→*）fail-closed；
2. ready/ack 义务：pending ack 非空不可 READY；未知/重复 ack fail-closed；
3. 同 (内容, 持有者) 唯一（重复登记 fail-closed）；不同持有者多副本合法；
4. 守恒：capacity_projection 直和；assert_conservation 抓双登记/负引用/
   RELEASED 泄漏；有效保留量 V 按区间并集只计一次（重复副本不增加 V）；
5. 驱逐不清 owner：evict_to_pool 保留 entry_id/字节/角色披露、holder
   翻 POOL；引用>0 / 执行 pin 不可驱逐；restore_from_pool 回实例；
6. 区间元数据合并：仅相邻 token 区间且 owner/物理布局/就绪保护状态兼
   容；跨层/跨实例/引用>0 拒绝；
7. 引用守恒：acquire/release 配对；悬空 release / 越权 unpin fail-closed；
   引用>0 条目 retire 拒绝；
8. query_missing_intervals：目标已持有/在途 copy **单元格并集**排除 +
   同内容多副本冻结序只选读一副本（部分重叠只补差集片段）+
   token_start 升序确定性；
9. query_sources：目标缺失并集的权威读流片段（并集去重）；>2 来源无
   上限（5 来源结构支持）；冻结确定性角色序；
   池条目/零字节条目不进片上来源集；
10. begin/end_inflight_copy：同 (来源, 目标, 区间) 去重；未知闭合
    fail-closed；
11. 事务版本单调递增；schema 版本引用 KV_DIRECTORY_SCHEMA_VERSION。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 joint/test_kv_location_directory.py
"""
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WL = os.path.dirname(_HERE)
for _p in (_HERE, _WL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from joint.kv_location_directory import (  # noqa: E402
    ENTRY_STATE_EVICTED_TO_POOL,
    ENTRY_STATE_READY,
    ENTRY_STATE_RESTORING,
    ENTRY_STATE_WRITING,
    POOL_HOLDER,
    ROLE_BASE,
    ROLE_BOUNDARY_SOURCE,
    ROLE_POOL_BACKED,
    ROLE_RESIDENT,
    ROLE_WORKING,
    KvContentKey,
    KvDirectoryEntry,
    KvDirectoryError,
    KvLocationDirectory,
)
from joint.schema_registry import (  # noqa: E402
    KV_DIRECTORY_SCHEMA_VERSION,
)

MODEL_VERSION = "unit-test-model-v1"
SHARDS = (100, 120)          # tp_degree = 2 的逐 rank 字节
SHARDS_2 = (60, 72)


def _directory():
    return KvLocationDirectory(tp_degree=2, model_version=MODEL_VERSION)


def _register(directory, *, entry="e1", session="s1", holder=0,
              role=ROLE_RESIDENT, layer=(0, 4), token=(0, 16),
              shards=SHARDS, state=ENTRY_STATE_WRITING, acks=()):
    return directory.register_entry(
        session_id=session, holder=holder, role=role,
        layer_start=layer[0], layer_end=layer[1],
        token_start=token[0], token_end=token[1],
        bytes_by_tp_rank=shards, state=state, pending_acks=acks)


class StateMachineTests(unittest.TestCase):
    def test_legal_lifecycle_chain(self):
        d = _directory()
        entry = _register(d, state=ENTRY_STATE_WRITING)
        self.assertEqual(entry.state, ENTRY_STATE_WRITING)
        ready = d.mark_ready(entry.entry_id)
        self.assertEqual(ready.state, ENTRY_STATE_READY)
        pooled = d.evict_to_pool(entry.entry_id)
        self.assertEqual(pooled.state, ENTRY_STATE_EVICTED_TO_POOL)
        self.assertEqual(pooled.holder, POOL_HOLDER)
        restored = d.restore_from_pool(entry.entry_id, target_instance=1)
        self.assertEqual(restored.state, ENTRY_STATE_RESTORING)
        self.assertEqual(restored.holder, 1)
        self.assertEqual(d.mark_ready(entry.entry_id).state, ENTRY_STATE_READY)

    def test_illegal_transitions_fail_closed(self):
        d = _directory()
        entry = _register(d, state=ENTRY_STATE_READY)
        with self.assertRaises(KvDirectoryError):
            d._transition(entry.entry_id, ENTRY_STATE_WRITING)  # 回退禁
        pooled = d.evict_to_pool(entry.entry_id)
        with self.assertRaises(KvDirectoryError):
            d._transition(pooled.entry_id, ENTRY_STATE_READY)   # 越级禁

    def test_mark_ready_requires_empty_acks(self):
        d = _directory()
        entry = _register(d, acks=("ack-1",))
        with self.assertRaises(KvDirectoryError):
            d.mark_ready(entry.entry_id)
        d.acknowledge(entry.entry_id, "ack-1")
        self.assertEqual(
            d.mark_ready(entry.entry_id).state, ENTRY_STATE_READY)

    def test_ack_protocol_fail_closed(self):
        d = _directory()
        entry = _register(d, acks=("ack-1",))
        with self.assertRaises(KvDirectoryError):
            d.acknowledge(entry.entry_id, "ack-unknown")
        d.acknowledge(entry.entry_id, "ack-1")
        with self.assertRaises(KvDirectoryError):
            d.acknowledge(entry.entry_id, "ack-1")   # 重复 ack
        self.assertEqual(entry.pending_acks, ("ack-1",))  # frozen 历史不动
        d.add_ack_obligation(entry.entry_id, "ack-2")
        self.assertEqual(
            d.entry(entry.entry_id).pending_acks, ("ack-2",))
        with self.assertRaises(KvDirectoryError):
            d.add_ack_obligation(entry.entry_id, "ack-2")  # 重复挂接

    def test_eviction_requires_ready_and_preserves_pending_obligation(self):
        d = _directory()
        entry = _register(d, acks=("copy-0",))
        with self.assertRaises(KvDirectoryError):
            d.evict_to_pool(entry.entry_id)
        self.assertEqual(d.entry(entry.entry_id).state, ENTRY_STATE_WRITING)
        self.assertEqual(d.entry(entry.entry_id).pending_acks, ("copy-0",))

        d.acknowledge(entry.entry_id, "copy-0")
        d.mark_ready(entry.entry_id)
        pooled = d.evict_to_pool(entry.entry_id)
        restored = d.restore_from_pool(pooled.entry_id, target_instance=1)
        self.assertEqual(restored.state, ENTRY_STATE_RESTORING)
        with self.assertRaises(KvDirectoryError):
            d.restore_from_pool(restored.entry_id, target_instance=1)


class UniquenessAndConservationTests(unittest.TestCase):
    def test_duplicate_content_at_holder_rejected(self):
        d = _directory()
        _register(d)
        with self.assertRaises(KvDirectoryError):
            _register(d, entry="dup", role=ROLE_WORKING)  # 同内容同持有者

    def test_replica_at_other_holder_legal(self):
        d = _directory()
        _register(d)
        replica = _register(d, entry="r", holder=1, role=ROLE_BASE)
        self.assertEqual(replica.holder, 1)
        d.assert_conservation()

    def test_capacity_projection_sums_each_replica(self):
        d = _directory()
        primary = _register(d, state=ENTRY_STATE_READY)          # holder 0
        replica = _register(d, holder=1, shards=SHARDS_2)
        self.assertEqual(d.capacity_projection(0), SHARDS)
        self.assertEqual(d.capacity_projection(1), SHARDS_2)
        self.assertEqual(d.capacity_projection(2), (0, 0))
        pooled = d.evict_to_pool(primary.entry_id)
        self.assertEqual(d.capacity_projection(0), (0, 0))   # 池不占实例
        self.assertEqual(pooled.bytes_by_tp_rank, SHARDS)

    def test_effective_resident_counts_union_once(self):
        d = _directory()
        primary = _register(d)                                   # 权威
        _register(d, holder=1, role=ROLE_BASE)                   # 重复副本
        self.assertEqual(d.effective_resident_bytes("s1"), SHARDS)
        # 不相交区间副本（分段持有）：并集 = 直和（各自唯一内容）。
        _register(d, holder=2, role=ROLE_BASE,
                  token=(16, 24), shards=SHARDS_2)
        self.assertEqual(
            d.effective_resident_bytes("s1"),
            tuple(a + b for a, b in zip(SHARDS, SHARDS_2)))
        d.retire_entry(primary.entry_id, reason="test")
        self.assertEqual(             # 权威退役后 = 重复副本 ∪ 分段副本
            d.effective_resident_bytes("s1"),
            tuple(a + b for a, b in zip(SHARDS, SHARDS_2)))

    def test_assert_conservation_catches_duplicate(self):
        d = _directory()
        _register(d)
        # 直接注入双登记破坏（绕过 register 的守卫模拟账本破损）。
        d._entries["broken"] = d._entries[
            next(iter(d._entries))]
        with self.assertRaises(KvDirectoryError):
            d.assert_conservation()

    def test_register_validates_vector_width(self):
        d = _directory()
        with self.assertRaises(KvDirectoryError):
            d.register_entry(
                session_id="s1", holder=0, role=ROLE_RESIDENT,
                layer_start=0, layer_end=4, token_start=0, token_end=8,
                bytes_by_tp_rank=(1, 2, 3))


class EvictionKeepsOwnerTests(unittest.TestCase):
    """§7.4：驱逐更新目录并保留唯一内容在池，不清 owner 记录。"""

    def test_eviction_preserves_owner_record(self):
        d = _directory()
        entry = _register(d, state=ENTRY_STATE_READY)
        entry_id = entry.entry_id
        pooled = d.evict_to_pool(entry_id)
        self.assertEqual(pooled.entry_id, entry_id)         # 身份保留
        self.assertEqual(pooled.bytes_by_tp_rank, SHARDS)   # 内容保留
        self.assertIn("role_was:resident", pooled.release_condition)
        self.assertEqual(pooled.generation, entry.generation + 1)

    def test_referenced_entry_not_evictable(self):
        d = _directory()
        entry = _register(d, state=ENTRY_STATE_READY)
        d.acquire_entry_reference(entry.entry_id)
        with self.assertRaises(KvDirectoryError):
            d.evict_to_pool(entry.entry_id)
        d.release_entry_reference(entry.entry_id)
        self.assertEqual(
            d.evict_to_pool(entry.entry_id).state,
            ENTRY_STATE_EVICTED_TO_POOL)

    def test_exec_pinned_entry_not_evictable(self):
        d = _directory()
        entry = _register(d, state=ENTRY_STATE_READY)
        d.acquire_entry_reference(entry.entry_id, exec_pin=True)
        with self.assertRaises(KvDirectoryError):
            d.evict_to_pool(entry.entry_id)
        with self.assertRaises(KvDirectoryError):
            d.release_entry_reference(entry.entry_id)  # 未释放 pin 位
        d.release_entry_reference(entry.entry_id, exec_pin=True)
        d.evict_to_pool(entry.entry_id)

    def test_unique_protected_source_enumeration(self):
        d = _directory()
        sole = _register(d, session="s1", state=ENTRY_STATE_READY,
                         role=ROLE_BASE)
        _register(d, session="s2", holder=1, role=ROLE_BASE,
                  state=ENTRY_STATE_READY)
        _register(d, session="s2", holder=2, role=ROLE_BOUNDARY_SOURCE,
                  state=ENTRY_STATE_READY)
        protected = d.query_unique_protected_sources("s2")
        self.assertEqual(protected, ())          # 双副本 → 无唯一保护来源
        self.assertEqual(
            d.query_unique_protected_sources("s1"), (sole,))


class ReferenceConservationTests(unittest.TestCase):
    def test_acquire_release_pairing(self):
        d = _directory()
        entry = _register(d, state=ENTRY_STATE_READY)
        self.assertEqual(d.read_refs(entry.entry_id), 0)
        self.assertEqual(d.acquire_entry_reference(entry.entry_id), 1)
        self.assertEqual(d.acquire_entry_reference(entry.entry_id), 2)
        self.assertEqual(d.release_entry_reference(entry.entry_id), 1)
        self.assertEqual(d.release_entry_reference(entry.entry_id), 0)
        with self.assertRaises(KvDirectoryError):
            d.release_entry_reference(entry.entry_id)   # 悬空 release
        with self.assertRaises(KvDirectoryError):
            d.release_entry_reference("no-such-entry")

    def test_retire_with_live_refs_rejected(self):
        d = _directory()
        entry = _register(d, state=ENTRY_STATE_READY)
        d.acquire_entry_reference(entry.entry_id)
        with self.assertRaises(KvDirectoryError):
            d.retire_entry(entry.entry_id, reason="test")
        d.release_entry_reference(entry.entry_id)
        d.retire_entry(entry.entry_id, reason="test")
        self.assertEqual(d.entry_count, 0)

    def test_retire_with_pending_ack_rejected(self):
        d = _directory()
        entry = _register(d, acks=("ack-0",))
        with self.assertRaises(KvDirectoryError):
            d.retire_entry(entry.entry_id, reason="test")
        self.assertEqual(d.entry(entry.entry_id).pending_acks, ("ack-0",))


class MetadataMergeTests(unittest.TestCase):
    """§7.4 区间元数据合并：仅相邻且 owner/物理布局/就绪保护状态兼容。"""

    def test_adjacent_merge_sums_bytes(self):
        d = _directory()
        a = _register(d, entry="a", token=(0, 8), shards=SHARDS_2,
                      state=ENTRY_STATE_READY)
        b = _register(d, entry="b", token=(8, 16), shards=SHARDS_2,
                      state=ENTRY_STATE_READY)
        merged = d.merge_adjacent_metadata(a.entry_id, b.entry_id)
        self.assertEqual(
            merged.content_key.token_range, (0, 16))
        self.assertEqual(
            merged.bytes_by_tp_rank,
            tuple(x + y for x, y in zip(SHARDS_2, SHARDS_2)))
        self.assertEqual(merged.state, ENTRY_STATE_READY)
        self.assertEqual(d.entry_count, 1)
        d.assert_conservation()

    def test_non_adjacent_rejected(self):
        d = _directory()
        a = _register(d, entry="a", token=(0, 8), state=ENTRY_STATE_READY)
        b = _register(d, entry="b", token=(10, 16),
                      state=ENTRY_STATE_READY)
        with self.assertRaises(KvDirectoryError):
            d.merge_adjacent_metadata(a.entry_id, b.entry_id)

    def test_incompatible_state_rejected(self):
        d = _directory()
        a = _register(d, entry="a", token=(0, 8), state=ENTRY_STATE_READY)
        b = _register(d, entry="b", token=(8, 16),
                      state=ENTRY_STATE_WRITING)
        with self.assertRaises(KvDirectoryError):
            d.merge_adjacent_metadata(a.entry_id, b.entry_id)

    def test_cross_holder_rejected(self):
        d = _directory()
        a = _register(d, entry="a", token=(0, 8), state=ENTRY_STATE_READY)
        b = _register(d, entry="b", holder=1, token=(8, 16),
                      state=ENTRY_STATE_READY)
        with self.assertRaises(KvDirectoryError):
            d.merge_adjacent_metadata(a.entry_id, b.entry_id)

    def test_referenced_rejected(self):
        d = _directory()
        a = _register(d, entry="a", token=(0, 8), state=ENTRY_STATE_READY)
        b = _register(d, entry="b", token=(8, 16),
                      state=ENTRY_STATE_READY)
        d.acquire_entry_reference(a.entry_id)
        with self.assertRaises(KvDirectoryError):
            d.merge_adjacent_metadata(a.entry_id, b.entry_id)


class QueryTests(unittest.TestCase):
    def _three_source_directory(self):
        d = _directory()
        _register(d, session="t", role=ROLE_BASE, token=(0, 8),
                  state=ENTRY_STATE_READY)                     # base@0
        _register(d, session="t", entry="b", holder=1,
                  role=ROLE_BOUNDARY_SOURCE, token=(8, 12),
                  state=ENTRY_STATE_READY)                     # 边界源@1
        _register(d, session="t", entry="c", holder=2,
                  role=ROLE_RESIDENT, token=(12, 16),
                  state=ENTRY_STATE_READY)                     # 第三源@2
        return d

    def test_missing_intervals_exclude_held_and_inflight(self):
        d = self._three_source_directory()
        # 目标 3 已持有 (12,16) 的权威副本（同内容多副本合法）。
        _register(d, session="t", entry="held", holder=3,
                  token=(12, 16), state=ENTRY_STATE_READY)
        # 缺失 = base (0,8) + 边界源 (8,12)，token_start 升序确定性。
        missing = d.query_missing_intervals("t", 3)
        self.assertEqual(
            tuple(e.content_key.token_range for e in missing),
            ((0, 8), (8, 12)))
        # 在途 copy 覆盖 base 区间 → 缺失只剩边界源（去重不重发）。
        d.begin_inflight_copy(
            session_id="t", source_holder=0, target_instance=3,
            layer_start=0, layer_end=4, token_start=0, token_end=8,
            bytes_by_tp_rank=SHARDS, ack_id="ack-b")
        missing = d.query_missing_intervals("t", 3)
        self.assertEqual(
            tuple(e.content_key.token_range for e in missing), ((8, 12),))

    def test_inflight_copy_dedup_and_close(self):
        d = self._three_source_directory()
        first = d.begin_inflight_copy(
            session_id="t", source_holder=0, target_instance=3,
            layer_start=0, layer_end=4, token_start=0, token_end=8,
            bytes_by_tp_rank=SHARDS, ack_id="ack-b")
        second = d.begin_inflight_copy(
            session_id="t", source_holder=0, target_instance=3,
            layer_start=0, layer_end=4, token_start=0, token_end=8,
            bytes_by_tp_rank=SHARDS, ack_id="ack-b")
        self.assertIs(first, second)                    # 同身份在途去重
        self.assertEqual(len(d.inflight_copies("t")), 1)
        d.end_inflight_copy(first.copy_id)
        self.assertEqual(len(d.inflight_copies("t")), 0)
        with self.assertRaises(KvDirectoryError):
            d.end_inflight_copy(first.copy_id)

    def test_sources_ordered_and_unbounded(self):
        d = self._three_source_directory()
        sources = d.query_sources("t", 3)
        self.assertEqual(
            tuple(e.holder for e in sources), (0, 1, 2))
        self.assertEqual(
            tuple(e.role for e in sources),
            (ROLE_BASE, ROLE_BOUNDARY_SOURCE, ROLE_RESIDENT))
        # >2 来源结构支持：加第四、第五来源（W9 跨轮多来源形态）。
        _register(d, session="t", entry="d", holder=3,
                  role=ROLE_BOUNDARY_SOURCE, token=(16, 20),
                  state=ENTRY_STATE_READY)
        _register(d, session="t", entry="e", holder=4,
                  role=ROLE_BOUNDARY_SOURCE, token=(20, 24),
                  state=ENTRY_STATE_READY)
        five = d.query_sources("t", 5)
        self.assertEqual(len(five), 5)
        # 冻结确定性序：角色（base→boundary→resident）→ token_start →
        # 实例——boundary 组按 token 升序 (8/16/20) 先于 resident(12)。
        self.assertEqual(
            tuple(e.holder for e in five), (0, 1, 3, 4, 2))
        # 池条目与目标本地条目不进片上来源集。
        pooled = _register(d, session="t", entry="p", holder=POOL_HOLDER,
                           role=ROLE_POOL_BACKED, token=(24, 28),
                           state=ENTRY_STATE_EVICTED_TO_POOL)
        self.assertEqual(pooled.holder, POOL_HOLDER)
        self.assertEqual(len(d.query_sources("t", 5)), 5)
        pooled_intervals = d.query_pool_intervals("t")
        self.assertEqual(
            tuple(e.content_key.token_range for e in pooled_intervals),
            ((24, 28),))

    def test_missing_union_dedup_partial_overlap(self):
        """V6：缺失推导按初等单元格并集去重（部分重叠只补差集片段）。"""
        d = _directory()
        _register(d, session="t", role=ROLE_BASE, token=(0, 10),
                  state=ENTRY_STATE_READY)                     # base@0
        _register(d, session="t", entry="w", holder=1,
                  role=ROLE_RESIDENT, token=(0, 12), shards=SHARDS_2,
                  state=ENTRY_STATE_READY)                     # [0,12)@1
        missing = d.query_missing_intervals("t", 3)
        # 冻结序 base 先取 [0,10)；resident 只补差集片段 [10,12)——
        # 同内容多副本不重复搬运（§6.1）。
        self.assertEqual(
            tuple(e.content_key.token_range for e in missing),
            ((0, 10), (10, 12)))
        self.assertEqual(missing[0].holder, 0)
        self.assertEqual(missing[1].holder, 1)
        # 片段字节按 token×layer 面积线性精确比例（2/12）。
        self.assertEqual(
            missing[1].bytes_by_tp_rank,
            tuple(v * 2 // 12 for v in SHARDS_2))

    def test_missing_excludes_target_superset_holdings(self):
        """V6：目标已持包含型超集时子集条目不列为缺失（覆盖而非恒等）。"""
        d = _directory()
        _register(d, session="t", role=ROLE_BASE, token=(0, 10),
                  state=ENTRY_STATE_READY)                     # [0,10)@0
        _register(d, session="t", entry="big", holder=3,
                  role=ROLE_RESIDENT, token=(0, 12),
                  state=ENTRY_STATE_READY)                     # [0,12)@3
        # 旧区间身份比对会把 [0,10) 误列为缺失；并集口径 = 空缺失。
        self.assertEqual(d.query_missing_intervals("t", 3), ())

    def test_sources_union_dedup_reads_one_replica(self):
        """§6.1：同内容多副本按冻结确定序只选读一副本（读流并集基数）。"""
        d = _directory()
        _register(d, session="t", role=ROLE_BASE, token=(0, 10),
                  state=ENTRY_STATE_READY)                     # [0,10)@0
        _register(d, session="t", entry="dup", holder=1,
                  role=ROLE_BASE, token=(0, 12), shards=SHARDS_2,
                  state=ENTRY_STATE_READY)                     # [0,12)@1
        sources = d.query_sources("t", 2)
        # base@0 先选读 [0,10)；@1 只补差集 [10,12)——union tokens = 12。
        self.assertEqual(
            tuple((e.holder, e.content_key.token_range) for e in sources),
            ((0, (0, 10)), (1, (10, 12))))
        self.assertEqual(
            sum(e.content_key.token_end - e.content_key.token_start
                for e in sources), 12)
        # 在途 copy 覆盖 [0,10) 后读流只剩差集片段（不重复读在途内容）。
        d.begin_inflight_copy(
            session_id="t", source_holder=0, target_instance=2,
            layer_start=0, layer_end=4, token_start=0, token_end=10,
            bytes_by_tp_rank=SHARDS, ack_id="ack-u")
        sources = d.query_sources("t", 2)
        self.assertEqual(
            tuple((e.holder, e.content_key.token_range) for e in sources),
            (((1, (10, 12)),)))

    def test_effective_resident_bytes_cell_union_partial_overlap(self):
        """V：跨持有者部分重叠区间按单元格并集只计一次（不再计两遍）。"""
        d = _directory()
        _register(d, session="t", role=ROLE_BASE, token=(0, 10),
                  state=ENTRY_STATE_READY)                     # [0,10)@0
        _register(d, session="t", entry="w", holder=1,
                  role=ROLE_RESIDENT, token=(0, 12), shards=SHARDS_2,
                  state=ENTRY_STATE_READY)                     # [0,12)@1
        # 并集 [0,12)：base 片段 SHARDS + resident 差集 (10,12) 比例字节。
        expected = tuple(
            a + b for a, b in zip(
                SHARDS, tuple(v * 2 // 12 for v in SHARDS_2)))
        self.assertEqual(
            d.effective_resident_bytes("t"), expected)

    def test_pool_intervals_enumeration(self):
        d = self._three_source_directory()
        _register(d, session="t", entry="p", holder=POOL_HOLDER,
                  role=ROLE_POOL_BACKED, token=(16, 24),
                  state=ENTRY_STATE_EVICTED_TO_POOL)
        pooled = d.query_pool_intervals("t")
        self.assertEqual(len(pooled), 1)
        self.assertEqual(pooled[0].content_key.token_range, (16, 24))


def replace_entry(entry, **overrides):
    """Return a target-shaped immutable directory entry for sync tests."""
    return KvDirectoryEntry(
        entry_id=entry.entry_id,
        content_key=entry.content_key,
        holder=entry.holder,
        role=entry.role,
        bytes_by_tp_rank=entry.bytes_by_tp_rank,
        padding_by_tp_rank=entry.padding_by_tp_rank,
        generation=entry.generation,
        state=overrides.get("state", entry.state),
        pending_acks=overrides.get("pending_acks", entry.pending_acks),
        release_condition=entry.release_condition,
    )


class DirectoryMetadataTests(unittest.TestCase):
    def test_version_monotonic(self):
        d = _directory()
        v0 = d.version
        entry = _register(d)
        self.assertGreater(d.version, v0)
        d.mark_ready(entry.entry_id)
        self.assertGreater(d.version, v0 + 1)

    def test_schema_version_from_registry(self):
        d = _directory()
        self.assertEqual(d.schema_version, KV_DIRECTORY_SCHEMA_VERSION)
        self.assertEqual(d.schema_version, "kvdir-v1-20260929")

    def test_content_key_shard_derivation(self):
        key = KvContentKey(
            session_id="s", model_version=MODEL_VERSION,
            layer_start=0, layer_end=4, token_start=0, token_end=8)
        shard = key.shard_key(1)
        self.assertEqual(shard.tp_rank, 1)
        self.assertTrue(key.covers_same_content(shard))
        self.assertEqual(shard.model_version, MODEL_VERSION)

    def test_sync_entries_reprojection(self):
        d = _directory()
        _register(d, session="sync", state=ENTRY_STATE_READY)
        # 目标形态：同一内容改字节并转 WRITING（带 ack），随后收敛回
        # READY + 新字节；多余条目被退休。
        target = KvDirectoryEntry(
            entry_id="sync#resident#L0-4#T0-16",
            content_key=KvContentKey(
                session_id="sync", model_version=MODEL_VERSION,
                layer_start=0, layer_end=4, token_start=0, token_end=16),
            holder=0, role=ROLE_RESIDENT,
            bytes_by_tp_rank=(50, 60), state=ENTRY_STATE_READY)
        d.sync_entries("sync", [target])
        entries = d.entries_for_session("sync")
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].bytes_by_tp_rank, (50, 60))
        self.assertEqual(entries[0].state, ENTRY_STATE_READY)
        d.sync_entries("sync", [])           # 收敛到空集 → 全退休
        self.assertEqual(d.entry_count, 0)
        d.assert_conservation()

    def test_sync_cannot_drop_ack_without_explicit_acknowledgement(self):
        d = _directory()
        entry = _register(d, session="ack-sync", acks=("copy-chunk-0",))
        target = replace_entry(entry, state=ENTRY_STATE_READY, pending_acks=())
        with self.assertRaises(KvDirectoryError):
            d.sync_entries("ack-sync", [target])
        self.assertEqual(
            d.entry(entry.entry_id).pending_acks, ("copy-chunk-0",))
        d.sync_entries(
            "ack-sync", [target], acknowledged_acks=("copy-chunk-0",))
        self.assertEqual(d.entry(entry.entry_id).state, ENTRY_STATE_READY)

    def test_sync_preflights_all_retirements_before_mutation(self):
        d = _directory()
        first = _register(d, session="atomic-sync", token=(0, 8))
        second = _register(
            d, session="atomic-sync", entry="second", token=(8, 16),
            acks=("copy-chunk-1",))
        with self.assertRaises(KvDirectoryError):
            d.sync_entries("atomic-sync", [])
        self.assertEqual(
            {entry.entry_id for entry in d.entries_for_session("atomic-sync")},
            {first.entry_id, second.entry_id})
    def test_forget_session_clears_entries(self):
        d = _directory()
        _register(d, session="gone", state=ENTRY_STATE_READY)
        d.forget_session("gone", reason="test")
        self.assertEqual(d.entry_count, 0)
        self.assertEqual(len(d.inflight_copies("gone")), 0)


if __name__ == "__main__":
    unittest.main()
