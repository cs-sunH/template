#!/usr/bin/env python3
"""test_r6_base_fold_kv_release_pin.py -- 修复轮三 R6 钉测：D 边界折叠分
支（j==base 驻留实例）的 base 物理字节零载荷释放（2026-09-27）。

背景（stageR5/repro_kv_drift_merge.py sweep 复现，修复前后逐字相同）：
D 落到 base 驻留实例时折叠分支把会话账本清成完整上下文（full_shards）
并复位 base_*，但 base 前缀的物理字节滞留在 base 驻留实例上——工作
副本持完整 token 域（⊇ base 域）时 base 字节是重复副本，必须就地零载
荷释放。修复前 fail-closed：

``RuntimeError: rank 2 KV accounting mismatch: state=53248, expected=28672``

（state = base 24576 ＋ 全史 28672，expected = 全史；sweep 等比形态。）

修复面（face_scheduler.py ``commit_boundary_decode_transition``）：
  ``complete_copy_base_fold`` 的 boundary_copy_base_fold_release 科目
  零载荷释放重复 base 副本。
  （2026-10-14 远读退役：原 ``remote_read_base_fold`` 折叠分支与
  ``increment_copy_home_u_only``（D=copy@base、working_kind=remote-read
  增量形态）两族用例已删除——增量布局只能由 remote-read 准入产生；
  保留 complete 型 copy 折叠族：journal 形态与 complete_copy_base_fold
  回归锚。）

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


# （原 RemoteReadBaseFoldReleasePin 两用例（D=remote-read@base 折叠的
# 重复 base 零载荷释放 + 增量形态不越权释放门控）与
# CopyHomeUFoldReleasePin（increment_copy_home_u_only 同科目）已随
# 2026-10-14 远读退役删除：增量布局（P=remote-read 工作副本仅持 U）
# 三动作下不可构造；complete 型 copy 折叠族的零载荷释放语义由下方
# FoldFamilyRegressionPin 钉住。）


class FoldFamilyRegressionPin(unittest.TestCase):
    """折叠族回归锚：journal 形态（P=copy 的 C13 逐块交接）与既有
    complete_copy_base_fold（R4 修复面）在 R6 改动后行为不变。"""

    def test_copy_handoff_journal_fold_unchanged(self):
        # （2026-10-14 远读退役：D 折叠动作由 remote-read 换 copy——
        # journal 形态的折叠闭合语义不变。）
        kv = _build_turn0()
        session, outcome = _drive_full_chain(
            kv, p_action="copy",
            p_ctx=HISTORY_TOKENS + INPUT_TOKENS, d_action="copy")
        self.assertEqual(outcome["kind"], "complete_copy_base_fold")
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
