#!/usr/bin/env python3
"""test_d12_merge_projection_parity.py -- D12 完整型布局下的
merge 投影 ↔ 执行平价钉测。

原远读增量型三位置平价（H@home + U@i_P + growth@j 的保留量择优/
逐源缺失腿）已随增量型布局族退役（D6：P ∈ {stay,copy,recompute}
的 P 末工作副本恒完整型、D 末态 = 执行端完整并集直接作为结果位
置）。本文件钉完整型家族的投影/执行双侧平价：

  1. **copy 末态零字节翻转**（P=copy@j、D=copy@j 两形态）：投影
     ``merge_from_layout`` 判 complete_union@j、零腿零载荷；执行
     ``merge_back`` 胜者 = j、零传输、终态守恒（home/旧源清零）；
  2. **recompute 末态同样零翻转**：来源侧旧副本随重建物化释放
     （重复副本不加 V——去重平价）；
  3. **保留量/并集披露**：投影保留量 = 真账本逐实例字节（投影输入
     与执行输入同源）。

区间代数面（保留量择优/容量回退序/逐源腿）由
test_d12_merge_three_position 以纯 ``project_end_state`` +
``merge_from_layout`` 覆盖（该路径在三动作生产面不可达、
merge 机制通用面保留，见执行方案 §1.2 保留红线与 A3 报告 D-1）。

夹具自包含（2026-10-14 远读退役改写）：原 ``_increment_kv`` 来自
online/test_c4_boundary_full_surface（随远读用例整删），本文件改
本地完整型夹具——H 基础建 home，P=copy 到 i_P 物化 H+U 工作副本。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 joint/test_d12_merge_projection_parity.py（或 pytest）
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
from joint.boundary_layout import (  # noqa: E402
    INTERVAL_ROLE_BASE,
    INTERVAL_ROLE_CONTEXT,
    BoundaryKvInterval,
)
from joint.joint_cost_model import (  # noqa: E402
    MERGE_SELECTION_COMPLETE_UNION,
    project_end_state,
)
from joint.test_d12_merge_three_position import _loads3  # noqa: E402
from joint.test_d18_boundary_decision import _model  # noqa: E402
from joint.test_joint_mechanisms import _tiny_model  # noqa: E402

HISTORY_TOKENS = 96        # 基础历史 H（token）
INPUT_TOKENS = 16          # 本轮输入 U（token）
HOME_INSTANCE = 2          # 基础驻留实例（三实例几何）
P_INSTANCE = 0             # i_P（P = copy 执行实例）
THIRD_INSTANCE = 1         # 第三目标
GROWTH_TOKENS = 6          # decode 增长（token）
MODEL_LAYERS = 4           # tiny model 层数（层字段 = 身份元数据）


def _three_hardware():
    return FaceHardware(
        mesh_rows=2, mesh_cols=3,
        local_hbm_capacity_bytes=10 ** 9,
        local_hbm_bandwidth_gbps=100.0, d2d_bandwidth_gbps=200.0,
        peak_perf_tflops=1.0, d2d_latency_ns=0, local_hbm_latency_ns=0)


def _three_topology(hardware):
    """三实例几何（home=2 / i_P=0 / 第三目标=1；2×3 mesh 6 rank）。"""
    return build_instances(hardware, (
        FaceInstanceSpec("ins0", "1", (0, 3)),
        FaceInstanceSpec("ins1", "2", (1, 4)),
        FaceInstanceSpec("ins2", "3", (2, 5)),
    ))


def _instance_bytes(kv, instance_index):
    return tuple(
        kv._rank_states[rank].kv_cache_bytes
        for rank in kv.topology.instance(instance_index).ranks)


def _u_bytes(model, kv):
    return kv_cache_shard_bytes_for_tokens(
        model, INPUT_TOKENS, kv.tp_degree)


def _h_bytes(model, kv):
    return kv_cache_shard_bytes_for_tokens(
        model, HISTORY_TOKENS, kv.tp_degree)


def _bytes_for(kv, model, tokens):
    return tuple(int(value) for value in (
        kv_cache_shard_bytes_for_tokens(model, tokens, kv.tp_degree)))


def _copy_kv():
    """三实例真 KVCacheManager + 完整型会话（P = copy@i_P：H 基础在
    home、H+U 工作副本物化在 i_P）——完整型边界事务的执行基底。"""
    hardware = _three_hardware()
    model = _tiny_model()
    topology = _three_topology(hardware)
    kv = KVCacheManager(topology, model)
    # 基础历史先建立在 home（turn-0 语义），完成后作为下一轮基础。
    kv.prepare_prefill(
        session_id="s", target_instance_index=HOME_INSTANCE,
        history_tokens=0, trigger_request_id="seed")
    kv.expand_prefill(
        session_id="s", instance_index=HOME_INSTANCE,
        context_tokens=HISTORY_TOKENS, trigger_request_id="seed")
    kv.mark_complete("s", 100)
    # P = copy@i_P：工作副本 = H 基础整份 + U 增量（完整型）。
    kv.prepare_prefill(
        session_id="s", target_instance_index=P_INSTANCE,
        history_tokens=HISTORY_TOKENS, trigger_request_id="r0",
        action="copy")
    kv.expand_prefill(
        session_id="s", instance_index=P_INSTANCE,
        context_tokens=HISTORY_TOKENS + INPUT_TOKENS,
        trigger_request_id="r0")
    return kv, model, topology


def _interval(replica_id, role, holder, start, end, bytes_by_rank,
              *, authority=True):
    return BoundaryKvInterval(
        replica_id=replica_id, role=role, holder_instance=holder,
        authority=authority, token_start=start, token_end=end,
        layer_start=0, layer_end=MODEL_LAYERS,
        bytes_by_tp_rank=tuple(int(value) for value in bytes_by_rank))


def _projection_end_state(model, kv, *, decode_instance, growth_tokens,
                          history_tokens, intervals, logic_home):
    growth_bytes = _bytes_for(kv, model, growth_tokens)
    return project_end_state(
        intervals=intervals,
        decode_instance=decode_instance,
        decode_growth_by_rank=growth_bytes,
        decode_growth_tokens=growth_tokens,
        history_tokens=history_tokens,
        logic_home=logic_home)


def _assert_retention_matches_ledger(test, end_state, kv):
    """project_end_state 的保留量 == 真账本逐实例字节（投影输入与执行
    输入同源——平价的前提）。"""
    for position in end_state.positions:
        test.assertEqual(
            tuple(position.retained_by_rank),
            _instance_bytes(kv, position.instance_index))


class CompleteUnionCopyParityTest(unittest.TestCase):
    """D=copy@第三目标：H_B 整份迁移到 j、growth 随后并入——末态 =
    执行端完整并集（零翻转），投影/执行双侧平价。"""

    def test_projection_matches_execution_zero_flip(self):
        kv, model, _topology = _copy_kv()
        kv.commit_boundary_decode_transition(
            session_id="s", p_instance_index=P_INSTANCE,
            target_instance_index=THIRD_INSTANCE, action="copy",
            history_tokens=HISTORY_TOKENS + INPUT_TOKENS,
            trigger_request_id="r0")
        kv.expand_decode(
            session_id="s", instance_index=THIRD_INSTANCE,
            final_context_tokens=(
                HISTORY_TOKENS + INPUT_TOKENS + GROWTH_TOKENS),
            trigger_request_id="r0")
        union_bytes = _bytes_for(
            kv, model, HISTORY_TOKENS + INPUT_TOKENS + GROWTH_TOKENS)
        # 投影（决策时点）：home 的旧基础副本（非权威）+ j 的完整并集
        # （权威 context）；i_P 的旧副本已随整份迁移释放。
        intervals = (
            _interval("boundary_base", INTERVAL_ROLE_BASE, HOME_INSTANCE,
                      0, HISTORY_TOKENS, _h_bytes(model, kv),
                      authority=False),
            _interval("working_union", INTERVAL_ROLE_CONTEXT,
                      THIRD_INSTANCE, 0,
                      HISTORY_TOKENS + INPUT_TOKENS + GROWTH_TOKENS,
                      union_bytes),
        )
        end_state = _projection_end_state(
            model, kv, decode_instance=THIRD_INSTANCE,
            growth_tokens=0,
            history_tokens=(
                HISTORY_TOKENS + INPUT_TOKENS + GROWTH_TOKENS),
            intervals=intervals, logic_home=HOME_INSTANCE)
        _assert_retention_matches_ledger(self, end_state, kv)
        self.assertEqual(end_state.complete_union_position, THIRD_INSTANCE)
        notes = []
        plan = _model(_loads3()).merge_from_layout(end_state, notes=notes)
        transfers = kv.merge_back(
            session_id="s", trigger_request_id="r0",
            new_tokens=INPUT_TOKENS + GROWTH_TOKENS)
        outcome = kv.last_merge_outcome
        # 双侧零载荷：完整并集直接作为结果位置（copy 零字节翻转）。
        self.assertTrue(plan.complete_union)
        self.assertEqual(plan.target_instance, THIRD_INSTANCE)
        self.assertEqual(
            plan.selection_rule, MERGE_SELECTION_COMPLETE_UNION)
        self.assertEqual(plan.legs, ())
        self.assertEqual(plan.payload_ns, 0)
        self.assertIn("merge_v2_zero_byte_flip", notes)
        self.assertEqual(transfers, ())
        self.assertTrue(outcome["zero_byte_flip"])
        self.assertEqual(outcome["winner_instance"], THIRD_INSTANCE)
        self.assertEqual(outcome["transferred_bytes"], 0)
        # 终态守恒：胜者持完整并集；home 旧副本/迁移源清零。
        self.assertEqual(_instance_bytes(kv, THIRD_INSTANCE), union_bytes)
        self.assertEqual(_instance_bytes(kv, HOME_INSTANCE), (0, 0))
        self.assertEqual(_instance_bytes(kv, P_INSTANCE), (0, 0))


class TwoPositionCopyParityTest(unittest.TestCase):
    """两位置退化（copy 原地继续：H@home + H+U+growth@i_P）——
    计价两位置 complete_union 与执行两位置零翻转一致。"""

    def test_two_position_projection_matches_merge_back(self):
        kv, model, _topology = _copy_kv()
        # j == i_P 原地继续（无边界事务）：working 增长到 U+G @i_P。
        kv.expand_decode(
            session_id="s", instance_index=P_INSTANCE,
            final_context_tokens=(
                HISTORY_TOKENS + INPUT_TOKENS + GROWTH_TOKENS),
            trigger_request_id="r0")
        union_bytes = _bytes_for(
            kv, model, HISTORY_TOKENS + INPUT_TOKENS + GROWTH_TOKENS)
        self.assertEqual(_instance_bytes(kv, P_INSTANCE), union_bytes)
        intervals = (
            _interval("boundary_base", INTERVAL_ROLE_BASE, HOME_INSTANCE,
                      0, HISTORY_TOKENS, _h_bytes(model, kv),
                      authority=False),
            _interval("working_union", INTERVAL_ROLE_CONTEXT, P_INSTANCE,
                      0,
                      HISTORY_TOKENS + INPUT_TOKENS + GROWTH_TOKENS,
                      union_bytes),
        )
        end_state = _projection_end_state(
            model, kv, decode_instance=P_INSTANCE,
            growth_tokens=0,
            history_tokens=(
                HISTORY_TOKENS + INPUT_TOKENS + GROWTH_TOKENS),
            intervals=intervals, logic_home=HOME_INSTANCE)
        _assert_retention_matches_ledger(self, end_state, kv)
        # 权威内容单位置（home 旧副本非权威、零贡献——去重口径）。
        self.assertEqual(end_state.complete_union_position, P_INSTANCE)
        notes = []
        plan = _model(_loads3()).merge_from_layout(end_state, notes=notes)
        transfers = kv.merge_back(
            session_id="s", trigger_request_id="r0",
            new_tokens=INPUT_TOKENS + GROWTH_TOKENS)
        outcome = kv.last_merge_outcome
        # 执行端完整并集 → 零翻转（计价/执行/胜者三口径一致）。
        self.assertTrue(plan.complete_union)
        self.assertEqual(plan.target_instance, P_INSTANCE)
        self.assertEqual(transfers, ())
        self.assertTrue(outcome["zero_byte_flip"])
        self.assertEqual(outcome["winner_instance"], P_INSTANCE)
        self.assertEqual(outcome["transferred_bytes"], 0)
        final = _bytes_for(
            kv, model,
            HISTORY_TOKENS + INPUT_TOKENS + GROWTH_TOKENS)
        self.assertEqual(_instance_bytes(kv, P_INSTANCE), final)
        self.assertEqual(_instance_bytes(kv, HOME_INSTANCE), (0, 0))


class DuplicateReplicaParityTest(unittest.TestCase):
    """去重（重复副本不加 V）：recompute@基础驻留端——重建副本持完整
    并集（权威），同实例 base 记录是重复副本（零贡献）。"""

    def test_duplicate_base_zero_payload_on_both_sides(self):
        kv, model, _topology = _copy_kv()
        kv.commit_boundary_decode_transition(
            session_id="s", p_instance_index=P_INSTANCE,
            target_instance_index=HOME_INSTANCE, action="recompute",
            history_tokens=HISTORY_TOKENS + INPUT_TOKENS,
            trigger_request_id="r0")
        # 重建物化：home 的重建副本长到完整并集（H+U+G）。
        kv.expand_decode(
            session_id="s", instance_index=HOME_INSTANCE,
            final_context_tokens=HISTORY_TOKENS + INPUT_TOKENS,
            trigger_request_id="r0")
        kv.expand_decode(
            session_id="s", instance_index=HOME_INSTANCE,
            final_context_tokens=(
                HISTORY_TOKENS + INPUT_TOKENS + GROWTH_TOKENS),
            trigger_request_id="r0")
        union_bytes = _bytes_for(
            kv, model, HISTORY_TOKENS + INPUT_TOKENS + GROWTH_TOKENS)
        # 投影：重复 base 副本（非权威——重建副本为权威并集）+ 并集区间
        # @home；i_P 来源侧旧副本随重建物化释放。
        intervals = (
            _interval("boundary_base", INTERVAL_ROLE_BASE, HOME_INSTANCE,
                      0, HISTORY_TOKENS, _h_bytes(model, kv),
                      authority=False),
            _interval("rebuild_union", INTERVAL_ROLE_CONTEXT,
                      HOME_INSTANCE, 0,
                      HISTORY_TOKENS + INPUT_TOKENS + GROWTH_TOKENS,
                      union_bytes),
        )
        end_state = _projection_end_state(
            model, kv, decode_instance=HOME_INSTANCE,
            growth_tokens=0,
            history_tokens=(
                HISTORY_TOKENS + INPUT_TOKENS + GROWTH_TOKENS),
            intervals=intervals, logic_home=HOME_INSTANCE)
        # 去重：唯一并集 = 重建副本；home 持完整并集。
        self.assertEqual(end_state.complete_union_position, HOME_INSTANCE)
        notes = []
        plan = _model(_loads3()).merge_from_layout(end_state, notes=notes)
        transfers = kv.merge_back(
            session_id="s", trigger_request_id="r0",
            new_tokens=INPUT_TOKENS + GROWTH_TOKENS)
        outcome = kv.last_merge_outcome
        # 双侧零载荷：重复副本不增加 V、不产生传输。
        self.assertTrue(plan.complete_union)
        self.assertEqual(plan.legs, ())
        self.assertEqual(plan.payload_ns, 0)
        self.assertIn("merge_v2_zero_byte_flip", notes)
        self.assertEqual(transfers, ())
        self.assertTrue(outcome["zero_byte_flip"])
        self.assertEqual(outcome["winner_instance"], HOME_INSTANCE)
        self.assertEqual(outcome["transferred_bytes"], 0)
        # 终态：home = 并集；i_P 的旧副本已释放。
        self.assertEqual(_instance_bytes(kv, HOME_INSTANCE), union_bytes)
        self.assertEqual(_instance_bytes(kv, P_INSTANCE), (0, 0))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
