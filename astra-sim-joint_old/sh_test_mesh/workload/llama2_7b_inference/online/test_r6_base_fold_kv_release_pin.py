#!/usr/bin/env python3
"""test_r6_base_fold_kv_release_pin.py -- 修复轮三 R6 钉测：D 边界折叠分
支（j==base 驻留实例）的 base 物理字节零载荷释放（2026-09-27）。

背景（stageR5/repro_kv_drift_merge.py sweep 复现，修复前后逐字相同）：
D=remote-read 落到 base 驻留实例时 ``remote_read_base_fold`` 折叠分支
把会话账本清成完整上下文（full_shards）并复位 base_*，但 base 前缀的
物理字节滞留在 base 驻留实例上——P=recompute 全量重建形态下 i_P 侧
工作副本持完整 token 域（⊇ base 域），base 字节是重复副本，必须就地
零载荷释放。修复前 fail-closed：

``RuntimeError: rank 2 KV accounting mismatch: state=53248, expected=28672``

（state = base 24576 ＋ 全史 28672，expected = 全史；sweep 等比形态。）

修复面（face_scheduler.py ``commit_boundary_decode_transition``）：
  1. ``remote_read_base_fold`` 分支：无 journal 且工作副本 token 域覆
     盖基础域（[0,ctx) ⊇ [0,base_history)、驻留层 ⊇ 基础驻留层）时，
     对 ``complete_copy_base_fold`` 的 boundary_copy_base_fold_release
     科目对称补零载荷释放（cause=boundary_remote_read_base_fold_release）；
  2. 同族第三处 ``increment_copy_home_u_only`` 分支（D=copy@base、
     working_kind=remote-read）同科目缺对称释放——同式补齐
     （cause=boundary_copy_home_u_fold_release）。
  释放按覆盖门控：增量形态（P=remote-read 产线工作副本仅持 U，
  ctx < base_history）base 字节是全史组成部分，**不得释放**（base＋U
  直和恰为 full_shards）——钉测含不越权释放的回归锚。

被钉语义（修复后）：
  * 折叠后物理(base 驻留实例) ≡ 账本 full_shards（无 stranded 字节、
    无 double count）；
  * 全链（折叠 → decode 增长 → merge_back）后胜者恰持全史，其余实例
    归零；全程 _check_invariants ＋ _check_invariants_after_mutation
    双绿（账本字节 ≡ 物理字节）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 -m pytest online/test_r6_base_fold_kv_release_pin.py -q
"""
import os
import sys
import unittest

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
    kv_cache_shard_bytes_for_tokens,
)

# sweep 等比缩放（sweep: HISTORY 96 / INPUT 16 / tp3；base@2 第三方、
# i_P=0、D 目标 j=2 = base 驻留实例——折叠分支的触发形态）。
HISTORY_TOKENS = 96   # turn-0 终态上下文 = base 历史（base@2 驻留）
INPUT_TOKENS = 16     # turn-1 prefill 输入（全量重建 H_B = 112 token）
BASE_INSTANCE = 2     # base 驻留实例（D 目标 j == base → 折叠分支）
P_INSTANCE = 0        # turn-1 执行实例（工作副本所在）
GROWTH_TOKENS = 3     # 折叠后 decode 增长


def _model():
    return FaceModel(
        layers=2, hidden_size=64, ffn_size=128, num_heads=8,
        vocab_size=256, bytes_per_elem=2, mlp_variant="gelu")


def _hardware():
    return FaceHardware(
        mesh_rows=2, mesh_cols=3,
        local_hbm_capacity_bytes=10 ** 9,
        local_hbm_bandwidth_gbps=100.0, d2d_bandwidth_gbps=200.0,
        peak_perf_tflops=1.0, d2d_latency_ns=0, local_hbm_latency_ns=0)


def _topology(hardware):
    return build_instances(hardware, (
        FaceInstanceSpec("ins0", "1", (0, 3)),
        FaceInstanceSpec("ins1", "2", (1, 4)),
        FaceInstanceSpec("ins2", "3", (2, 5)),
    ))


def _instance_bytes(kv, instance_index):
    return tuple(
        kv._rank_states[rank].kv_cache_bytes
        for rank in kv.topology.instance(instance_index).ranks)


def _build_turn0():
    """turn-0：基础 96 token 建立于第三方实例 2（base 驻留位）。"""
    kv = KVCacheManager(_topology(_hardware()), _model())
    kv.prepare_prefill(
        session_id="s", target_instance_index=BASE_INSTANCE,
        history_tokens=0, trigger_request_id="seed")
    kv.expand_prefill(
        session_id="s", instance_index=BASE_INSTANCE,
        context_tokens=HISTORY_TOKENS, trigger_request_id="seed")
    kv.mark_complete("s", 100)
    return kv


def _drive_full_chain(kv, *, p_action, p_ctx, d_action):
    """turn-1 P 相 @0（指定口径展开）→ D=action@base 驻留实例（折叠）
    → decode 增长 → merge_back，返回（session, boundary_outcome）。"""
    kv.prepare_prefill(
        session_id="s", target_instance_index=P_INSTANCE,
        history_tokens=HISTORY_TOKENS, trigger_request_id="r1",
        action=p_action)
    kv.expand_prefill(
        session_id="s", instance_index=P_INSTANCE,
        context_tokens=p_ctx, trigger_request_id="r1")
    final_tokens = HISTORY_TOKENS + INPUT_TOKENS
    _transfer, _evictions, outcome = kv.commit_boundary_decode_transition(
        session_id="s", p_instance_index=P_INSTANCE,
        target_instance_index=BASE_INSTANCE, action=d_action,
        history_tokens=final_tokens, trigger_request_id="r1")
    # 修复面：折叠清账后物理 ≡ 账本（修复前 rank state=base＋full 对
    # expected=full 在此确定性 raise——红在真根因，不红在探针自身）。
    kv._check_invariants_after_mutation(session_ids=("s",))
    kv.expand_decode(
        session_id="s", instance_index=BASE_INSTANCE,
        final_context_tokens=final_tokens + GROWTH_TOKENS,
        trigger_request_id="r1")
    kv.merge_back(
        session_id="s", trigger_request_id="r1", new_tokens=GROWTH_TOKENS)
    return kv._sessions["s"], outcome


class RemoteReadBaseFoldReleasePin(unittest.TestCase):
    """R6 主缺口钉测：P=recompute 全量重建 → D=remote-read@base 驻留
    实例——base 重复副本零载荷释放，账本 ≡ 物理（修复前
    state=53248 对 expected=28672 fail-closed）。"""

    def test_base_fold_releases_duplicate_base_full_copy(self):
        kv = _build_turn0()
        session, outcome = _drive_full_chain(
            kv, p_action="recompute",
            p_ctx=HISTORY_TOKENS + INPUT_TOKENS, d_action="remote-read")
        self.assertEqual(outcome["kind"], "remote_read_base_fold")
        total_tokens = HISTORY_TOKENS + INPUT_TOKENS + GROWTH_TOKENS
        full_shards = kv_cache_shard_bytes_for_tokens(
            kv.model, total_tokens, kv.tp_degree)
        # 折叠落点 = base 驻留实例，merge 后恰持全史；其余实例零滞留
        # （修复前 base 前缀字节 stranded 在 base 驻留实例）。
        self.assertEqual(
            _instance_bytes(kv, BASE_INSTANCE),
            tuple(int(v) for v in full_shards))
        self.assertEqual(_instance_bytes(kv, P_INSTANCE), (0, 0))
        # 会话账本终态：折叠族口径（上下文 = 全史、字节 = full_shards、
        # base_*/working_kind 全清）。
        self.assertEqual(session.context_tokens, total_tokens)
        self.assertEqual(
            tuple(int(v) for v in session.shard_bytes),
            tuple(int(v) for v in full_shards))
        self.assertIsNone(session.working_kind)
        self.assertIsNone(session.base_instance_index)
        self.assertEqual(session.base_history_tokens, 0)
        # 全程"账本字节 ≡ 物理字节"：全量 strict + 增量通道双绿。
        kv._check_invariants()
        kv._check_invariants_after_mutation(session_ids=("s",))

    def test_incremental_base_fold_keeps_base_bytes(self):
        """覆盖门控回归锚：产线增量形态（P=remote-read 工作副本仅持 U，
        ctx < base_history）base 字节是全史组成部分——折叠**不释放**
        （base＋U 直和恰为 full_shards）；修复若越权释放则此处
        under-count 必红。"""
        kv = _build_turn0()
        session, outcome = _drive_full_chain(
            kv, p_action="remote-read", p_ctx=INPUT_TOKENS,
            d_action="remote-read")
        self.assertEqual(outcome["kind"], "remote_read_base_fold")
        total_tokens = HISTORY_TOKENS + INPUT_TOKENS + GROWTH_TOKENS
        full_shards = kv_cache_shard_bytes_for_tokens(
            kv.model, total_tokens, kv.tp_degree)
        self.assertEqual(
            _instance_bytes(kv, BASE_INSTANCE),
            tuple(int(v) for v in full_shards))
        self.assertEqual(_instance_bytes(kv, P_INSTANCE), (0, 0))
        self.assertEqual(session.context_tokens, total_tokens)
        kv._check_invariants()
        kv._check_invariants_after_mutation(session_ids=("s",))


class CopyHomeUFoldReleasePin(unittest.TestCase):
    """同族第三处钉测：increment_copy_home_u_only（D=copy@base、
    working_kind=remote-read、工作副本持全量 token 域）——同科目对称
    释放（修复前同 state=53248 对 expected=28672 断裂）。产线今日的
    P=remote-read 工作副本为 U 域（上一用例回归锚不越权释放）；本用
    例钉全量域契约面（sweep 判定面同形态）。"""

    def test_copy_home_u_fold_releases_duplicate_base(self):
        kv = _build_turn0()
        session, outcome = _drive_full_chain(
            kv, p_action="remote-read",
            p_ctx=HISTORY_TOKENS + INPUT_TOKENS, d_action="copy")
        self.assertEqual(outcome["kind"], "increment_copy_home_u_only")
        total_tokens = HISTORY_TOKENS + INPUT_TOKENS + GROWTH_TOKENS
        full_shards = kv_cache_shard_bytes_for_tokens(
            kv.model, total_tokens, kv.tp_degree)
        self.assertEqual(
            _instance_bytes(kv, BASE_INSTANCE),
            tuple(int(v) for v in full_shards))
        self.assertEqual(_instance_bytes(kv, P_INSTANCE), (0, 0))
        self.assertIsNone(session.working_kind)
        self.assertIsNone(session.base_instance_index)
        kv._check_invariants()
        kv._check_invariants_after_mutation(session_ids=("s",))


class FoldFamilyRegressionPin(unittest.TestCase):
    """折叠族回归锚：journal 形态（P=copy 的 C13 逐块交接）与既有
    complete_copy_base_fold（R4 修复面）在 R6 改动后行为不变。"""

    def test_copy_handoff_journal_fold_unchanged(self):
        kv = _build_turn0()
        session, outcome = _drive_full_chain(
            kv, p_action="copy",
            p_ctx=HISTORY_TOKENS + INPUT_TOKENS, d_action="remote-read")
        self.assertEqual(outcome["kind"], "remote_read_base_fold")
        total_tokens = HISTORY_TOKENS + INPUT_TOKENS + GROWTH_TOKENS
        full_shards = kv_cache_shard_bytes_for_tokens(
            kv.model, total_tokens, kv.tp_degree)
        self.assertEqual(
            _instance_bytes(kv, BASE_INSTANCE),
            tuple(int(v) for v in full_shards))
        self.assertEqual(_instance_bytes(kv, P_INSTANCE), (0, 0))
        self.assertIsNone(session.copy_handoff)
        kv._check_invariants()
        kv._check_invariants_after_mutation(session_ids=("s",))

    def test_complete_copy_base_fold_unchanged(self):
        kv = _build_turn0()
        session, outcome = _drive_full_chain(
            kv, p_action="recompute",
            p_ctx=HISTORY_TOKENS + INPUT_TOKENS, d_action="copy")
        self.assertEqual(outcome["kind"], "complete_copy_base_fold")
        total_tokens = HISTORY_TOKENS + INPUT_TOKENS + GROWTH_TOKENS
        full_shards = kv_cache_shard_bytes_for_tokens(
            kv.model, total_tokens, kv.tp_degree)
        self.assertEqual(
            _instance_bytes(kv, BASE_INSTANCE),
            tuple(int(v) for v in full_shards))
        self.assertEqual(_instance_bytes(kv, P_INSTANCE), (0, 0))
        kv._check_invariants()
        kv._check_invariants_after_mutation(session_ids=("s",))


if __name__ == "__main__":
    unittest.main()
