#!/usr/bin/env python3
"""test_d12_merge_projection_parity.py -- D12 修复轮二：三位置 merge 的
预测-执行平价（式(21) 投影 ↔ ``_merge_back_three_position`` 执行）。

同一实时状态（真 KVCacheManager 夹具：增量型会话 + 边界事务 + decode
增长）下，逐字段核对：

* **目标/保留量**——``project_end_state`` 的位置保留量 == 账本实例字节；
  ``merge_from_layout`` 的式(21) 目标（``merge_eq21_target`` notes）==
  执行侧 ``last_merge_outcome`` 的 winner；
* **逐源缺失区间/传输字节**——plan.legs（source→target + 逐 rank 字节）
  == 执行侧 transfers 逐腿；payload == ``transferred_bytes``；
* **通道义务**——腿的 ``source_holds_reference_until_ack`` 披露在场；
* **去重**——重复副本（非权威）不增加 V：计价 legs 空/执行零传输，
  双侧一致（complete_union / zero_byte_flip）；
* **容量回退序列**——执行侧新增的 ``fallback_sequence`` 披露（主目标
  失败 → 回退1 → 锁定，≤ ``m_end-1`` 次替代（W5 §7.3 推广））与
  计价侧 ``capacity_fallback_order`` 同序；回退目标的审计替位计划
  （``audit_alternate_targets``）与实际执行腿逐位一致。

覆盖：两位置退化、三位置、去重（重复副本不加 V）、容量回退 ≤2 次。
纯夹具测试（不跑仿真）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 joint/test_d12_merge_projection_parity.py
"""
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WL = os.path.dirname(_HERE)
for _p in (_HERE, _WL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from face_scheduler import kv_cache_shard_bytes_for_tokens  # noqa: E402
from joint.boundary_layout import (  # noqa: E402
    INTERVAL_ROLE_BASE,
    INTERVAL_ROLE_INCREMENT,
    INTERVAL_ROLE_CONTEXT,
    BoundaryKvInterval,
)
from joint.joint_cost_model import (  # noqa: E402
    MERGE_SELECTION_RETENTION_MAX,
    project_end_state,
)
from joint.test_d18_boundary_decision import (  # noqa: E402
    _loads3,
    _model,
)
from online.test_c4_boundary_full_surface import (  # noqa: E402
    HISTORY_TOKENS,
    HOME_INSTANCE,
    INPUT_TOKENS,
    P_INSTANCE,
    THIRD_INSTANCE,
    _h_bytes,
    _increment_kv,
    _instance_bytes,
    _three_hardware,
    _u_bytes,
)

GROWTH_TOKENS = 6            # decode 增长（token）
MODEL_LAYERS = 4             # tiny model 层数（层字段 = 身份元数据）


def _bytes_for(kv, model, tokens):
    return tuple(int(value) for value in (
        kv_cache_shard_bytes_for_tokens(model, tokens, kv.tp_degree)))


def _interval(replica_id, role, holder, start, end, bytes_by_rank,
              *, authority=True):
    return BoundaryKvInterval(
        replica_id=replica_id, role=role, holder_instance=holder,
        authority=authority, token_start=start, token_end=end,
        layer_start=0, layer_end=MODEL_LAYERS,
        bytes_by_tp_rank=tuple(int(v) for v in bytes_by_rank))


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


def _assert_legs_parity(test, plan, transfers, transferred_bytes):
    """逐源缺失区间/载荷平价：plan.legs ↔ 执行 transfers（逐腿
    source→target + 字节），payload ↔ transferred_bytes；通道义务披露
    在场（来源持引用直到载荷确认）。"""
    plan_legs = sorted(
        (leg.source_instance, leg.target_instance, leg.total_bytes)
        for leg in plan.legs)
    exec_legs = sorted(
        (transfer.source_instance_index, transfer.target_instance_index,
         transfer.total_bytes)
        for transfer in transfers)
    test.assertEqual(plan_legs, exec_legs)
    for leg in plan.legs:
        test.assertTrue(leg.source_holds_reference_until_ack)
    test.assertEqual(
        sum(leg.total_bytes for leg in plan.legs), transferred_bytes)


def _assert_retention_matches_ledger(test, end_state, kv):
    """project_end_state 的保留量 == 真账本逐实例字节（投影输入与执行
    输入同源——平价的前提）。"""
    for position in end_state.positions:
        test.assertEqual(
            tuple(position.retained_by_rank),
            _instance_bytes(kv, position.instance_index))


class ThreePositionParityTest(unittest.TestCase):
    """三位置（H@home + U@i_P + growth@j）：主目标成功路径的逐字段平价。"""

    def test_projection_matches_execution_target_legs_and_notes(self):
        kv, model, topology = _increment_kv()
        kv.commit_boundary_decode_transition(
            session_id="s", p_instance_index=P_INSTANCE,
            target_instance_index=THIRD_INSTANCE, action="remote-read",
            history_tokens=HISTORY_TOKENS + INPUT_TOKENS,
            trigger_request_id="r0")
        kv.expand_decode(
            session_id="s", instance_index=THIRD_INSTANCE,
            final_context_tokens=GROWTH_TOKENS, trigger_request_id="r0")
        session = kv._sessions["s"]
        self.assertEqual(session.home_instance, HOME_INSTANCE)
        # 投影：base H@home + U@i_P（权威）+ growth@j（project_end_state
        # 生成）。
        intervals = (
            _interval("boundary_base", INTERVAL_ROLE_BASE, HOME_INSTANCE,
                      0, HISTORY_TOKENS, _h_bytes(model, kv)),
            _interval("boundary_increment", INTERVAL_ROLE_INCREMENT,
                      P_INSTANCE, HISTORY_TOKENS,
                      HISTORY_TOKENS + INPUT_TOKENS, _u_bytes(model, kv)),
        )
        end_state = _projection_end_state(
            model, kv, decode_instance=THIRD_INSTANCE,
            growth_tokens=GROWTH_TOKENS,
            history_tokens=HISTORY_TOKENS + INPUT_TOKENS,
            intervals=intervals, logic_home=HOME_INSTANCE)
        _assert_retention_matches_ledger(self, end_state, kv)
        notes = []
        plan = _model(_loads3()).merge_from_layout(end_state, notes=notes)
        # 执行：同状态 merge_back（三位置分支）。
        transfers = kv.merge_back(
            session_id="s", trigger_request_id="r0",
            new_tokens=INPUT_TOKENS + GROWTH_TOKENS)
        outcome = kv.last_merge_outcome
        # 目标平价：H(96) > U(16) > growth(6)——保留量最大者 home。
        self.assertEqual(plan.target_instance, HOME_INSTANCE)
        self.assertEqual(outcome["winner_instance"], plan.target_instance)
        self.assertEqual(
            plan.selection_rule, MERGE_SELECTION_RETENTION_MAX)
        self.assertIn(f"merge_eq21_target={HOME_INSTANCE}", notes)
        self.assertIn(
            f"merge_eq21_rule={MERGE_SELECTION_RETENTION_MAX}", notes)
        # 保留量披露（偏好序首位的 home 及其 V）。
        retention_notes = [note for note in notes
                           if note.startswith("merge_eq21_retention=")]
        self.assertEqual(len(retention_notes), 1)
        self.assertTrue(retention_notes[0].startswith(
            "merge_eq21_retention={}:{}".format(
                HOME_INSTANCE, sum(_h_bytes(model, kv))) + ","))
        # 逐源缺失区间/载荷平价 + 通道义务。
        _assert_legs_parity(
            self, plan, transfers, outcome["transferred_bytes"])
        self.assertFalse(plan.complete_union)
        self.assertFalse(outcome["zero_byte_flip"])
        # 回退序列披露：主目标（home）一次成功即锁定。
        sequence = outcome["fallback_sequence"]
        self.assertEqual(
            [(entry["step"], entry["target"], entry["outcome"])
             for entry in sequence],
            [(0, HOME_INSTANCE, "committed"),
             (1, HOME_INSTANCE, "locked")])
        # 主目标（home）的需求 = 缺失区间直和（U@i_P + growth@j，逐 rank）。
        self.assertEqual(
            sequence[0]["required"],
            _sum_vectors(_u_bytes(model, kv),
                         _bytes_for(kv, model, GROWTH_TOKENS)))
        # 终态守恒：胜者持完整并集（H+U+G）。
        final = _bytes_for(
            kv, model,
            HISTORY_TOKENS + INPUT_TOKENS + GROWTH_TOKENS)
        self.assertEqual(_instance_bytes(kv, HOME_INSTANCE), final)
        self.assertEqual(_instance_bytes(kv, P_INSTANCE), (0, 0))
        self.assertEqual(_instance_bytes(kv, THIRD_INSTANCE), (0, 0))


class TwoPositionDegenerateParityTest(unittest.TestCase):
    """两位置退化（remote 原地继续：H@home + U+growth@i_P）——计价两
    位置腿与执行两位置合并（少并多）逐位一致。"""

    def test_two_position_projection_matches_merge_back(self):
        kv, model, _topology = _increment_kv()
        # j == i_P 原地继续（无边界事务）：working 增长到 U+G @i_P。
        kv.expand_decode(
            session_id="s", instance_index=P_INSTANCE,
            final_context_tokens=INPUT_TOKENS + GROWTH_TOKENS,
            trigger_request_id="r0")
        working_bytes = _bytes_for(
            kv, model, INPUT_TOKENS + GROWTH_TOKENS)
        self.assertEqual(
            _instance_bytes(kv, P_INSTANCE), working_bytes)
        intervals = (
            _interval("boundary_base", INTERVAL_ROLE_BASE, HOME_INSTANCE,
                      0, HISTORY_TOKENS, _h_bytes(model, kv)),
            _interval("boundary_increment", INTERVAL_ROLE_INCREMENT,
                      P_INSTANCE, HISTORY_TOKENS,
                      HISTORY_TOKENS + INPUT_TOKENS, _u_bytes(model, kv)),
        )
        end_state = _projection_end_state(
            model, kv, decode_instance=P_INSTANCE,
            growth_tokens=GROWTH_TOKENS,
            history_tokens=HISTORY_TOKENS + INPUT_TOKENS,
            intervals=intervals, logic_home=HOME_INSTANCE)
        _assert_retention_matches_ledger(self, end_state, kv)
        self.assertEqual(len(end_state.positions), 2)
        notes = []
        plan = _model(_loads3()).merge_from_layout(end_state, notes=notes)
        transfers = kv.merge_back(
            session_id="s", trigger_request_id="r0",
            new_tokens=INPUT_TOKENS + GROWTH_TOKENS)
        outcome = kv.last_merge_outcome
        # 少并多：U+G(22) < H(96) → 目标 home（计价/执行/方向三口径）。
        self.assertEqual(plan.target_instance, HOME_INSTANCE)
        self.assertEqual(outcome["winner_instance"], HOME_INSTANCE)
        self.assertEqual(outcome["direction"], "forward")
        # 两位置分支的 outcome 不带 three_position 键（缺省即 False）。
        self.assertFalse(outcome.get("three_position", False))
        _assert_legs_parity(
            self, plan, transfers, outcome["transferred_bytes"])
        final = _bytes_for(
            kv, model,
            HISTORY_TOKENS + INPUT_TOKENS + GROWTH_TOKENS)
        self.assertEqual(_instance_bytes(kv, HOME_INSTANCE), final)


class DuplicateReplicaParityTest(unittest.TestCase):
    """去重（重复副本不加 V）：recompute@基础驻留端——重建副本持完整
    并集（权威），同实例 base 记录是重复副本（零贡献）。"""

    def test_duplicate_base_zero_payload_on_both_sides(self):
        kv, model, _topology = _increment_kv()
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
        # @home；U@i_P 来源侧在重建后释放（C5 语义：来源侧旧副本随重建
        # 物化释放）。project_end_state 只收权威区间。
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
        self.assertTrue(outcome["three_position"])
        self.assertTrue(outcome["zero_byte_flip"])
        self.assertEqual(outcome["winner_instance"], HOME_INSTANCE)
        self.assertEqual(outcome["transferred_bytes"], 0)
        # 回退序列披露：完整并集零载荷 + 锁定。
        self.assertEqual(
            [(entry["step"], entry["target"], entry["outcome"])
             for entry in outcome["fallback_sequence"]],
            [(0, HOME_INSTANCE, "complete_union_zero_payload"),
             (1, HOME_INSTANCE, "locked")])
        # 终态：home = 并集；i_P 的 U 已释放。
        self.assertEqual(_instance_bytes(kv, HOME_INSTANCE), union_bytes)
        self.assertEqual(_instance_bytes(kv, P_INSTANCE), (0, 0))


class CapacityFallbackSequenceParityTest(unittest.TestCase):
    """容量回退 ≤2 次的序列披露 + 与计价 capacity_fallback_order/审计
    替位计划的平价。"""

    def test_fallback_sequence_matches_projection_fallback_order(self):
        kv, model, topology = _increment_kv()
        hardware = _three_hardware()
        kv.commit_boundary_decode_transition(
            session_id="s", p_instance_index=P_INSTANCE,
            target_instance_index=THIRD_INSTANCE, action="remote-read",
            history_tokens=HISTORY_TOKENS + INPUT_TOKENS,
            trigger_request_id="r0")
        kv.expand_decode(
            session_id="s", instance_index=THIRD_INSTANCE,
            final_context_tokens=GROWTH_TOKENS, trigger_request_id="r0")
        growth_bytes = _bytes_for(kv, model, GROWTH_TOKENS)
        # 受保护 decoy 占满 home 余量：主目标（保留量最大的 home）容量
        # 失败 → 式(21) 偏好序回退到次持有位置（i_P）。
        per_rank_need = (
            sum(_u_bytes(model, kv)) + sum(growth_bytes)) // kv.tp_degree
        home_rank = topology.instance(HOME_INSTANCE).ranks[0]
        free = kv._rank_states[home_rank].remaining_bytes
        bpt = kv_cache_shard_bytes_for_tokens(model, 1, kv.tp_degree)[0]
        decoy_tokens = max(1, (free - per_rank_need // 2) // bpt)
        kv.prepare_prefill(
            session_id="decoy", target_instance_index=HOME_INSTANCE,
            history_tokens=0, trigger_request_id="d0")
        kv.expand_prefill(
            session_id="decoy", instance_index=HOME_INSTANCE,
            context_tokens=decoy_tokens, trigger_request_id="d0")
        self.assertLess(
            kv._rank_states[home_rank].remaining_bytes, per_rank_need)
        # 投影（决策时点）：主目标 home + 回退序（偏好序其余位置，
        # ≤ m_end-1（W5 §7.3 推广）），并请求 i_P 的审计替位计划。
        intervals = (
            _interval("boundary_base", INTERVAL_ROLE_BASE, HOME_INSTANCE,
                      0, HISTORY_TOKENS, _h_bytes(model, kv)),
            _interval("boundary_increment", INTERVAL_ROLE_INCREMENT,
                      P_INSTANCE, HISTORY_TOKENS,
                      HISTORY_TOKENS + INPUT_TOKENS, _u_bytes(model, kv)),
        )
        end_state = _projection_end_state(
            model, kv, decode_instance=THIRD_INSTANCE,
            growth_tokens=GROWTH_TOKENS,
            history_tokens=HISTORY_TOKENS + INPUT_TOKENS,
            intervals=intervals, logic_home=HOME_INSTANCE)
        notes = []
        plan = _model(_loads3()).merge_from_layout(
            end_state, notes=notes,
            audit_alternate_targets=(P_INSTANCE,))
        self.assertEqual(plan.target_instance, HOME_INSTANCE)
        self.assertEqual(
            plan.capacity_fallback_order,
            (P_INSTANCE, THIRD_INSTANCE))
        self.assertIn(
            "merge_capacity_fallback_order={},{}".format(
                P_INSTANCE, THIRD_INSTANCE), notes)
        alternate = next(
            item for item in plan.alternates
            if item.target_instance == P_INSTANCE)
        # 执行：home 失败 → 回退 i_P。
        transfers = kv.merge_back(
            session_id="s", trigger_request_id="r0",
            new_tokens=INPUT_TOKENS + GROWTH_TOKENS)
        outcome = kv.last_merge_outcome
        self.assertEqual(outcome["winner_instance"], P_INSTANCE)
        # 回退序列披露：主目标失败 → 回退1 成功 → 锁定；替代次数 ≤
        # m_end-1（W5 §7.3 推广）。
        sequence = outcome["fallback_sequence"]
        attempts = [(entry["step"], entry["target"], entry["outcome"])
                    for entry in sequence]
        self.assertEqual(
            attempts,
            [(0, HOME_INSTANCE, "kv_capacity_error"),
             (1, P_INSTANCE, "committed"),
             (2, P_INSTANCE, "locked")])
        failures = [entry for entry in sequence
                    if entry["outcome"] == "kv_capacity_error"]
        # W5（§7.3 重锚）：m_end-1 推广——失败替代数 ≤ 位置数-1
        # （无常数上限；本夹具 3 位置 ⇒ ≤2）。
        self.assertLessEqual(len(failures), 2)
        # 披露元素形态：{step, target, required, outcome}（容量失败元素
        # 另携 error 文本——原 KVCapacityError 文本不再只在异常里）。
        for entry in sequence:
            self.assertEqual(
                set(entry) - {"error"},
                {"step", "target", "required", "outcome"})
            self.assertIsInstance(entry["required"], list)
        for entry in failures:
            self.assertIn("error", entry)
            self.assertTrue(entry["error"])
        # 尝试序与计价回退序同源：[主目标] + capacity_fallback_order 的
        # 前缀。
        executed_order = [entry["target"] for entry in sequence
                          if entry["outcome"] != "locked"]
        self.assertEqual(
            executed_order,
            [plan.target_instance]
            + list(plan.capacity_fallback_order)[:len(executed_order) - 1])
        # 回退目标（i_P）的实际执行腿与审计替位计划逐位平价——预测器
        # 用同一目标次序即可重算执行结果。
        _assert_legs_parity(
            self, alternate, transfers, outcome["transferred_bytes"])
        final = _bytes_for(
            kv, model,
            HISTORY_TOKENS + INPUT_TOKENS + GROWTH_TOKENS)
        self.assertEqual(_instance_bytes(kv, P_INSTANCE), final)
        self.assertEqual(_instance_bytes(kv, THIRD_INSTANCE), (0, 0))


def _sum_vectors(*vectors):
    return [sum(values) for values in zip(*vectors)]


if __name__ == "__main__":
    unittest.main()
