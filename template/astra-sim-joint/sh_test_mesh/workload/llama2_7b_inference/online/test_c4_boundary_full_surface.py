#!/usr/bin/env python3
"""test_c4_boundary_full_surface.py -- C4 钉测：D 边界完整动作面的执行
语义（v2.0 §19.1 P→D 专项 / D19 按缺失区间 copy）。

覆盖（执行侧 + 计价口径一致性；视图层基础钉测在 joint/test_d19；
原增量型布局族（P=remote-read@i_P 的 U/H 分置）与多源远读/credit
切片/远读折叠/远读臂用例已随 2026-10-14 远读退役删除；完整型
recompute@j 单源形态、D 静默重建首 token 消费与 {recompute} 臂全链
路三类用例已随 2026-10-15 recompute 退役删除——二动作下边界布局恒
完整型单族）：
  1. **完整型 copy 回 base 驻留实例就地折叠**（journal 闭合、轮末
     merge = stay）与完成路径 new_tokens 口径钉。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 -m pytest online/test_c4_boundary_full_surface.py -q
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
    KVCacheManager,
    build_instances,
    kv_cache_shard_bytes_for_tokens,
)
from joint.test_joint_mechanisms import (  # noqa: E402
    _tiny_model,
)
from online.test_joint_arrival_contract import (  # noqa: E402
    R0,
    SERVICE_DONE_NS,
)

HISTORY_TOKENS = 96        # 历史区间 H（token）
INPUT_TOKENS = 16          # 本轮输入 U（token）
HOME_INSTANCE = 2          # 历史权威驻留实例（三实例几何）
P_INSTANCE = 0             # i_P（P 阶段执行实例）


def _three_instance_topology(hardware):
    """三实例几何（home=2 / i_P=0 / 第三目标=1；2×3 mesh 6 rank，
    逐实例一列 × 两 rank = 轴对齐矩形）。"""
    return build_instances(hardware, (
        FaceInstanceSpec("ins0", "1", (0, 3)),
        FaceInstanceSpec("ins1", "2", (1, 4)),
        FaceInstanceSpec("ins2", "3", (2, 5)),
    ))


def _three_hardware():
    return FaceHardware(
        mesh_rows=2, mesh_cols=3,
        local_hbm_capacity_bytes=10 ** 9,
        local_hbm_bandwidth_gbps=100.0, d2d_bandwidth_gbps=200.0,
        peak_perf_tflops=1.0, d2d_latency_ns=0, local_hbm_latency_ns=0)


def _instance_bytes(kv, instance_index):
    return tuple(
        kv._rank_states[rank].kv_cache_bytes
        for rank in kv.topology.instance(instance_index).ranks)


# ------------------------------------------------------ 边界事务（KV 账本） --


# （原增量型布局用例六类——C4HomeCopyFillsOnlyU/C4ThirdTargetCopyCoversHAndU/
# C4BaseMaterializationAtExec/C4MultiSourceRemote/C5D12ThreePositionMergeSurface/
# C4RecomputeRebuildElsewhere——已随 2026-10-14 远读退役删除：其会话几何
# （P=remote-read@i_P 的 U@i_P+H@home 分置）只能由 remote-read 准入产生，
# 二动作下不可构造；完整型单族的对应语义由下方
# G3CompleteCopyOntoBaseFoldsTest 承接。）
# （完整型 recompute@j 单源形态（C4CompleteFamilyTransitionsTest）、
# recompute@j 计价与 recompute@i_P 空间复用（C4RecomputePricingTest）、
# {recompute}+load-first 臂全链路（C4RecomputeArmFullChainTest 与
# _recompute_arm_shell/_TrainGraphStub 基底）已随 2026-10-15 recompute
# 退役删除：被测动作/掩码臂/D10 静默重建面均不再存在。）


# --------------------------------------------- G3 冒烟矩阵回归钉（2026-09-27）--

DECODE_TOKENS = 8             # D 增长（token）


def _turn0_local_prefill(kv, tokens, request_id):
    """turn-0 完整型 P（本地 stay 起步，无工作副本）。"""
    kv.prepare_prefill(
        session_id="s", target_instance_index=HOME_INSTANCE,
        history_tokens=0, trigger_request_id=request_id)
    kv.expand_prefill(
        session_id="s", instance_index=HOME_INSTANCE,
        context_tokens=tokens, trigger_request_id=request_id)


class G3CompleteCopyOntoBaseFoldsTest(unittest.TestCase):
    """G3 真实形态二：完整型 copy 的 D 边界迁移落到 base 驻留实例——
    就地并基折叠（R13 消除的组合不得到达轮末 merge）。

    P 阶段 copy 建 C13 逐块交接 journal（home 侧基础轮内释放），D 边
    界再选 copy 回 base 驻留实例：折叠清 working_kind/基础记录、闭合
    journal，轮末 merge = stay 本地提交（不再撞 N9 防御）。
    """

    def _arrange(self):
        hardware = _three_hardware()
        model = _tiny_model()
        kv = KVCacheManager(_three_instance_topology(hardware), model)
        _turn0_local_prefill(kv, HISTORY_TOKENS, "seed")
        kv.mark_complete("s", HISTORY_TOKENS)
        kv.prepare_prefill(
            session_id="s", target_instance_index=P_INSTANCE,
            history_tokens=HISTORY_TOKENS, trigger_request_id="r1",
            action="copy")
        kv.expand_prefill(
            session_id="s", instance_index=P_INSTANCE,
            context_tokens=HISTORY_TOKENS + INPUT_TOKENS,
            trigger_request_id="r1")
        kv._settle_copy_handoffs(
            "s", boundary="prefill_drain", trigger_request_id="r1")
        return kv, model

    def test_copy_onto_base_folds_and_settles_stay(self):
        kv, model = self._arrange()
        session = kv._sessions["s"]
        self.assertEqual(session.working_kind, "copy")
        self.assertEqual(session.base_instance_index, HOME_INSTANCE)
        _transfer, _evictions, outcome = kv.commit_boundary_decode_transition(
            session_id="s", p_instance_index=P_INSTANCE,
            target_instance_index=HOME_INSTANCE, action="copy",
            history_tokens=HISTORY_TOKENS + INPUT_TOKENS,
            trigger_request_id="r1")
        self.assertEqual(outcome["kind"], "complete_copy_base_fold")
        # 折叠后：本地全上下文，无工作副本、无基础记录、journal 闭合。
        self.assertIsNone(session.working_kind)
        self.assertEqual(session.instance_index, HOME_INSTANCE)
        self.assertEqual(
            session.context_tokens, HISTORY_TOKENS + INPUT_TOKENS)
        self.assertIsNone(session.base_instance_index)
        self.assertEqual(session.base_history_tokens, 0)
        self.assertIsNone(session.copy_handoff)
        full = kv_cache_shard_bytes_for_tokens(
            model, HISTORY_TOKENS + INPUT_TOKENS, kv.tp_degree)
        self.assertEqual(session.shard_bytes, full)
        self.assertEqual(
            _instance_bytes(kv, HOME_INSTANCE),
            tuple(int(value) for value in full))
        self.assertEqual(_instance_bytes(kv, P_INSTANCE), (0, 0))
        # 轮末 merge = stay 本地提交（零流量），守恒不经过 N9 防御。
        kv.expand_decode(
            session_id="s", instance_index=HOME_INSTANCE,
            final_context_tokens=(HISTORY_TOKENS + INPUT_TOKENS
                                  + DECODE_TOKENS),
            trigger_request_id="r1")
        transfers = kv.merge_back(
            session_id="s", trigger_request_id="r1",
            new_tokens=INPUT_TOKENS + DECODE_TOKENS)
        self.assertEqual(transfers, ())
        self.assertEqual(kv.last_merge_outcome["direction"], "stay")
        kv.mark_complete(
            "s", HISTORY_TOKENS + INPUT_TOKENS + DECODE_TOKENS)


# （原远读落 base 折叠/预约 sizing/交接 journal 闭合与 growth-only
# 增量域四类用例已随 2026-10-14 远读退役删除：remote_read_base_fold
# 折叠支路、远读 stage 预约口径与 joint_remote_growth_only 增量域均
# 随远读机制删除；copy 折叠形态由 G3CompleteCopyOntoBaseFoldsTest
# 承接。）


class G3CompletionNewTokensDomainTest(unittest.TestCase):
    """完成路径 new_tokens 口径钉（调度器侧）：完成路径统一按
    input+decode 披露（零字节翻转族口径）。"""

    @staticmethod
    def _completed_scheduler():
        from online.test_joint_arrival_contract import (
            _bare_scheduler,
        )
        scheduler = _bare_scheduler()
        runtime = scheduler.runtime_by_request_id[R0]
        runtime.joint_input_tokens = INPUT_TOKENS
        runtime.decode_length = DECODE_TOKENS
        runtime.decode_tokens_consumed = DECODE_TOKENS
        return scheduler, runtime

    def _merge_new_tokens(self, scheduler):
        for name, kwargs in scheduler.kv_manager.calls:
            if name == "merge_back":
                return kwargs["new_tokens"]
        raise AssertionError("merge_back was not called")

    # （原 growth-only/远读完成路径 new_tokens 两用例已随 2026-10-14
    # 远读退役删除：joint_remote_growth_only 与 ACTION_REMOTE 完成分支
    # 均退役，完成路径统一按 input+decode 披露。）

    def test_flip_family_keeps_disclosure_formula(self):
        scheduler, runtime = self._completed_scheduler()
        runtime.joint_action = "copy"
        runtime.joint_decode_action = "copy"
        scheduler._complete_requests([R0], SERVICE_DONE_NS)
        self.assertEqual(
            self._merge_new_tokens(scheduler),
            INPUT_TOKENS + DECODE_TOKENS)


if __name__ == "__main__":
    unittest.main()
