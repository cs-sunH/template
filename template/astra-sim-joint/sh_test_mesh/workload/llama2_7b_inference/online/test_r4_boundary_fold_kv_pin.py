#!/usr/bin/env python3
"""test_r4_boundary_fold_kv_pin.py -- 修复轮三 R4 钉测：D 边界 copy@base
折叠的 base 字节释放（2026-09-28）。

背景（probe_20260927T190219Z / session_3_request_1__P__recompute 臂红，
seq 172）：P=recompute 轮的 D 边界 copy 落到 base 驻留实例时，
``commit_boundary_decode_transition`` 的 complete_copy_base_fold 把 H_B
工作副本迁入 base 实例并清 base_*/working_kind——若 base 物理字节不在
同一事务释放，rank 台账残留 base+H_B、期望表只剩 H_B，fail-closed：
``rank <first rank of base instance> KV accounting mismatch:
state=(H_B+base) , expected=H_B``。probe 实测（rank 38，instance 4，
tp6/32 头、首 rank 6 头、16384 B/token/头）：state=2959736832
（=30108 token = base 11837 + H_B 18271）、expected=1796112384
（=18271 token）——与本钉测场景逐 token 同型（小模型等比缩放）。

被钉语义（2026-10-15 recompute 退役后：P 腿承载动作改 copy，
base+工作副本几何同构；原 D=recompute@j 的 boundary_source 全量审计
用例（BoundarySourceFullAuditPin）已随 boundary_source 字段族删除）：
  1. P=copy 轮 D=copy@base：折叠后 base 实例恰持 H_B，无 stranded 字
     节；全量审计（_check_invariants）与增量通道双绿。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 -m pytest online/test_r4_boundary_fold_kv_pin.py -q
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

HISTORY_TOKENS = 96   # base（turn-0 终态上下文）
INPUT_TOKENS = 16     # turn-1 prefill 输入
HOME_INSTANCE = 2     # turn-0 执行实例 = base 驻留
P_INSTANCE = 0        # turn-1 P 执行实例


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


def _p_copy_kv():
    """turn-0 基础建立于 home → turn-1 P=copy 后的工作副本状态
    （base@home、H_B@i_P；2026-10-15 recompute 退役后 P 腿承载动作
    copy 化，几何与原 recompute 臂同构）。"""
    kv = KVCacheManager(_topology(_hardware()), _model())
    kv.prepare_prefill(
        session_id="s", target_instance_index=HOME_INSTANCE,
        history_tokens=0, trigger_request_id="seed")
    kv.expand_prefill(
        session_id="s", instance_index=HOME_INSTANCE,
        context_tokens=HISTORY_TOKENS, trigger_request_id="seed")
    kv.mark_complete("s", 100)
    kv.prepare_prefill(
        session_id="s", target_instance_index=P_INSTANCE,
        history_tokens=HISTORY_TOKENS, trigger_request_id="r1",
        action="copy")
    kv.expand_prefill(
        session_id="s", instance_index=P_INSTANCE,
        context_tokens=HISTORY_TOKENS + INPUT_TOKENS,
        trigger_request_id="r1")
    return kv


class FoldBaseReleasePin(unittest.TestCase):
    """钉 1（probe 红同型）：P=copy 轮 D=copy@base 折叠不残留
    base 字节——否则 rank 0（base 实例首 rank）state=H_B+base 对
    expected=H_B 的 fail-closed 复现。"""

    def test_copy_to_base_folds_without_stranding(self):
        kv = _p_copy_kv()
        final_tokens = HISTORY_TOKENS + INPUT_TOKENS
        full_shards = kv_cache_shard_bytes_for_tokens(
            kv.model, final_tokens, kv.tp_degree)
        transfer, evictions, outcome = kv.commit_boundary_decode_transition(
            session_id="s", p_instance_index=P_INSTANCE,
            target_instance_index=HOME_INSTANCE, action="copy",
            history_tokens=final_tokens, trigger_request_id="r1")
        self.assertEqual(outcome["kind"], "complete_copy_base_fold")
        # 折叠后 base 实例恰持 H_B（base 副本已释放、工作副本已迁入）。
        self.assertEqual(
            _instance_bytes(kv, HOME_INSTANCE),
            tuple(int(v) for v in full_shards))
        self.assertEqual(_instance_bytes(kv, P_INSTANCE), (0, 0))
        # 双通道审计：全量 strict + 增量期望表。
        kv._check_invariants()
        kv._check_invariants_after_mutation(session_ids=("s",))


# （原 BoundarySourceFullAuditPin（D=recompute@j 的 boundary_source 第三
# 位置全量审计）已随 2026-10-15 recompute 退役删除：boundary_source 字段
# 族与三位置物化合并（_merge_back_three_position）整删，第三位置不再
# 存在；两位置守恒由 FoldBaseReleasePin 与 r6 折叠族用例承接。）


if __name__ == "__main__":
    unittest.main()
