#!/usr/bin/env python3
"""test_d12_merge_three_position.py -- D12 至多三位置确定性合并单元测试。

验收红线逐条落测（joint修改文档 D12 / v2.0 §15.1 式(21)、§15.2、
§15.3、D12 代码侧章节）：

  1. **两位置退化为既定少并多** —— V 执行侧 ≤ V home → 目标 home
     （byte_le_tie_forward 平局含等号），载荷 = 少方整份；两位置
     容量回退退化为只检查一次反方向；
  2. **三位置结果不依赖两两合并顺序** —— 目标/载荷/回退序在区间输
     入排列与位置插入次序下逐位不变；每区间恰被权威持有者搬运一次
     （载荷 = 唯一并集 − V_target）；
  3. **重复副本不增加 V** —— 同内容非权威副本零贡献（V、并集、计
     划不变）；两个权威区间内容交叠 fail-closed；
  4. **P 阶段 action 不覆盖真实 D 末态** —— copy/recompute 末态 =
     执行端完整并集 → 直接作为结果位置（零载荷），不因 home 更大
     搬回 home；
  5. **平局次序：边界逻辑 home → D 实例 → 编号**（式(21)）；
  6. **容量回退** —— 同一"保留量降序＋固定平局"次序，至多两次替
     代检查；**不搜索目标速度最优**（替代目标更廉也不改变选择，
     只作审计披露）；
  7. **merge_from_layout 返回面** —— 固定目标、各来源缺失区间、通
     道/引用义务、目标空间（service_done 投影占用的净新增）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 joint/test_d12_merge_three_position.py   （或 pytest 同路径）
"""
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WL = os.path.dirname(_HERE)
for _p in (_HERE, _WL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from joint.boundary_layout import (  # noqa: E402
    INTERVAL_ROLE_BASE,
    INTERVAL_ROLE_CONTEXT,
    INTERVAL_ROLE_GROWTH,
    BoundaryKvInterval,
)
from joint.joint_cost_model import (  # noqa: E402
    MERGE_SELECTION_COMPLETE_UNION,
    MERGE_SELECTION_RETENTION_MAX,
    JointCostError,
    project_end_state,
)
from joint.test_d18_boundary_decision import (  # noqa: E402
    KV_TOKEN_BYTES,
    _complete_boundary,
    _loads,
    _model,
)

U_BYTES = KV_TOKEN_BYTES * 40         # 新增区间（40 token）/rank
GROWTH_BYTES = KV_TOKEN_BYTES * 8     # decode 增长（8 token）/rank


def _interval(replica_id, holder, start, end, bytes_by_rank,
              *, role=INTERVAL_ROLE_BASE, authority=True):
    return BoundaryKvInterval(
        replica_id=replica_id, role=role, holder_instance=holder,
        authority=authority, token_start=start, token_end=end,
        layer_start=0, layer_end=0,
        bytes_by_tp_rank=tuple(int(value) for value in bytes_by_rank))


def _loads3():
    """三实例负载视图（区间代数构造需要三位置几何）。"""
    from joint.joint_cost_model import InstanceLoadView
    return {
        index: InstanceLoadView(
            instance_index=index, queued_task_load_ns=0,
            running_task_load_ns=0, active_decode_task_load_ns=0,
            hbm_remaining_bytes_by_tp_rank=(10 ** 9, 10 ** 9),
            reclaimable_bytes_by_tp_rank=(10 ** 9, 10 ** 9))
        for index in range(3)
    }


# ------------------------------------------------------------- 式(21) --


class Eq21TargetSelectionTest(unittest.TestCase):
    """式(21)：完整并集优先 → 保留量最大 → 逻辑 home → D 实例 → 编号。"""

    def test_two_position_few_into_many(self):
        """两位置退化：home=2000 < exec=3000 → 目标 exec（少并多），
        载荷 = home 整份；无空间紧张时零等待。"""
        end_state = project_end_state(
            intervals=(
                _interval("base", 0, 0, 100, (1000, 1000)),
                _interval("increment", 1, 100, 150, (1500, 1500),
                          role=INTERVAL_ROLE_GROWTH),
            ),
            decode_instance=1, logic_home=0)
        self.assertEqual(end_state.complete_union_position, None)
        self.assertEqual(
            [p.total_bytes for p in end_state.positions], [2000, 3000])
        plan = _model(_loads2()).merge_from_layout(end_state, notes=[])
        self.assertEqual(plan.target_instance, 1)
        self.assertEqual(plan.selection_rule, MERGE_SELECTION_RETENTION_MAX)
        self.assertEqual(
            [(leg.source_instance, leg.target_instance) for leg in plan.legs],
            [(0, 1)])
        self.assertEqual(plan.legs[0].total_bytes, 2000)
        self.assertEqual(
            plan.target_space_bytes_by_rank, (1000, 1000))

    def test_two_position_tie_keeps_logic_home(self):
        """同量留原逻辑 home（byte_le_tie_forward 含等号的式(21) 形态）。"""
        end_state = project_end_state(
            intervals=(
                _interval("base", 0, 0, 100, (1000, 1000)),
                _interval("increment", 1, 100, 200, (1000, 1000),
                          role=INTERVAL_ROLE_GROWTH),
            ),
            decode_instance=1, logic_home=0)
        plan = _model(_loads2()).merge_from_layout(end_state, notes=[])
        self.assertEqual(plan.target_instance, 0)
        self.assertEqual(
            [(leg.source_instance, leg.target_instance) for leg in plan.legs],
            [(1, 0)])

    def test_complete_union_position_wins_directly(self):
        """已有完整并集的一端直接作为结果位置：载荷为零、无回退必要。"""
        end_state = project_end_state(
            intervals=(
                _interval("context", 1, 0, 100, (5000, 5000),
                          role=INTERVAL_ROLE_CONTEXT),
            ),
            decode_instance=1,
            decode_growth_by_rank=(400, 400), decode_growth_tokens=4,
            history_tokens=100, logic_home=0)
        self.assertEqual(end_state.complete_union_position, 1)
        plan = _model(_loads2()).merge_from_layout(end_state, notes=[])
        self.assertEqual(plan.target_instance, 1)
        self.assertEqual(plan.selection_rule, MERGE_SELECTION_COMPLETE_UNION)
        self.assertTrue(plan.complete_union)
        self.assertEqual(plan.legs, ())
        self.assertEqual(plan.payload_ns, 0)
        self.assertEqual(plan.merge_ns, 0)

    def test_three_position_retention_max_with_two_sources(self):
        """三位置：home 保留量最大 → 其余两位置各把缺失区间搬给 home
        （来源升序、逐来源腿），载荷 = 并集 − V_home。"""
        end_state = _three_position_state()
        plan = _model(_loads3()).merge_from_layout(end_state, notes=[])
        self.assertEqual(plan.target_instance, 2)
        self.assertEqual(
            [(leg.source_instance, leg.target_instance) for leg in plan.legs],
            [(0, 2), (1, 2)])
        self.assertEqual(
            sum(leg.total_bytes for leg in plan.legs),
            end_state.total_bytes - 2000)
        self.assertEqual(
            plan.capacity_fallback_order, (1, 0))
        # W5（§7.3 重锚）：m_end-1 推广——无常数截断（3 位置 ⇒ 2 替代）。
        self.assertEqual(len(plan.capacity_fallback_order), 2)

    def test_tie_order_home_then_decode_then_index(self):
        """平局次序：逻辑 home → 当前 D 实例 → 实例编号（式(21)）。"""
        def _plan(logic_home, decode_instance):
            end_state = project_end_state(
                intervals=(
                    _interval("a", 0, 0, 10, (100, 100)),
                    _interval("b", 1, 10, 20, (100, 100)),
                    _interval("c", 2, 20, 30, (100, 100)),
                ),
                decode_instance=decode_instance, logic_home=logic_home)
            self.assertEqual(end_state.total_bytes, 600)
            plan = _model(_loads3()).merge_from_layout(
                end_state, notes=[])
            return plan.target_instance

        self.assertEqual(_plan(1, 2), 1)    # 平局：逻辑 home 优先
        self.assertEqual(_plan(None, 2), 2)  # 无 home：当前 D 实例
        self.assertEqual(_plan(None, 3), 0)  # 无 home 非 D：实例编号


# ----------------------------------------------------- 顺序无关与去重 --


class OrderIndependenceTest(unittest.TestCase):
    """三位置结果不依赖两两合并顺序（D12 验收第 2 条）。"""

    def test_plan_invariant_under_interval_input_order(self):
        base = _interval("base", 2, 0, 100, (1000, 1000))
        increment = _interval("increment", 0, 100, 140, (300, 300),
                              role=INTERVAL_ROLE_GROWTH)
        growth = (200, 200)
        reference = None
        for intervals in (
                (base, increment), (increment, base)):
            end_state = project_end_state(
                intervals=intervals, decode_instance=1,
                decode_growth_by_rank=growth, decode_growth_tokens=4,
                history_tokens=140, logic_home=2)
            plan = _model(_loads3()).merge_from_layout(end_state, notes=[])
            if reference is None:
                reference = plan
                continue
            self.assertEqual(plan.target_instance,
                             reference.target_instance)
            self.assertEqual(plan.payload_ns, reference.payload_ns)
            self.assertEqual(
                [(leg.source_instance, leg.total_bytes)
                 for leg in plan.legs],
                [(leg.source_instance, leg.total_bytes)
                 for leg in reference.legs])
            self.assertEqual(plan.capacity_fallback_order,
                             reference.capacity_fallback_order)

    def test_payload_transfers_each_interval_exactly_once(self):
        """每区间恰被权威持有者搬运一次：载荷 = 唯一并集 − V_target，
        与"先两两合并再选最大"的任何顺序无关（本接口没有合并顺序输
        入——结构级保证）。"""
        end_state = _three_position_state()
        plan = _model(_loads3()).merge_from_layout(end_state, notes=[])
        target = plan.target_instance
        target_position = end_state.position(target)
        self.assertIsNotNone(target_position)
        moved = set()
        for leg in plan.legs:
            self.assertEqual(leg.target_instance, target)
            for interval in leg.intervals:
                self.assertNotIn(interval.replica_id, moved)
                moved.add(interval.replica_id)
                self.assertNotEqual(
                    interval.holder_instance, target)
        self.assertEqual(
            {interval.replica_id
             for position in end_state.positions
             if position.instance_index != target
             for interval in position.intervals},
            moved)
        self.assertEqual(
            sum(leg.total_bytes for leg in plan.legs),
            end_state.total_bytes - target_position.total_bytes)


class DedupTest(unittest.TestCase):
    """唯一内容去重（D12 验收第 3 条）。"""

    def test_duplicate_replica_does_not_increase_v(self):
        base = _interval("base", 2, 0, 100, (1000, 1000))
        increment = _interval("increment", 0, 100, 140, (300, 300),
                              role=INTERVAL_ROLE_GROWTH)
        clean = project_end_state(
            intervals=(base, increment), decode_instance=1,
            logic_home=2)
        duplicated = project_end_state(
            intervals=(
                base,
                increment,
                # base 的交接重复副本落在 i_P=0：非权威 → 零贡献。
                _interval("base_replica", 0, 0, 100, (1000, 1000),
                          authority=False),
            ),
            decode_instance=1, logic_home=2)
        self.assertEqual(
            clean.union_bytes_by_rank, duplicated.union_bytes_by_rank)
        self.assertEqual(
            [p.total_bytes for p in clean.positions],
            [p.total_bytes for p in duplicated.positions])
        plan_clean = _model(_loads3()).merge_from_layout(
            clean, notes=[])
        plan_dup = _model(_loads3()).merge_from_layout(
            duplicated, notes=[])
        self.assertEqual(plan_clean.payload_ns, plan_dup.payload_ns)
        self.assertEqual(
            plan_clean.target_instance, plan_dup.target_instance)

    def test_overlapping_authority_intervals_fail_closed(self):
        with self.assertRaises(JointCostError):
            project_end_state(
                intervals=(
                    _interval("a", 0, 0, 100, (100, 100)),
                    _interval("b", 1, 50, 150, (100, 100)),
                ),
                decode_instance=1, logic_home=0)


class AuthorityMigrationTest(unittest.TestCase):
    """候选先行状态准备的权威转移（增量型切换 U i_P→j）。"""

    def test_migrated_interval_authority_moves_to_decode_instance(self):
        end_state = project_end_state(
            intervals=(
                _interval("base", 2, 0, 100, (1000, 1000)),
                _interval("increment", 0, 100, 140, (300, 300),
                          role=INTERVAL_ROLE_GROWTH),
            ),
            decode_instance=1,
            migrated_interval_ids=("increment",),
            logic_home=2)
        # i_P 不再权威持有任何内容 → 不在 O_end；U 权威在 j=1。
        self.assertEqual(
            [p.instance_index for p in end_state.positions], [1, 2])
        self.assertEqual(
            end_state.position(1).total_bytes, 600)
        self.assertEqual(
            end_state.position(2).total_bytes, 2000)


# ------------------------------------------------------- 容量回退 --


class CapacityFallbackTest(unittest.TestCase):
    """容量回退（v2.0 §15.3）：固定保留量次序、至多两次、载荷提交
    后锁定；不搜索目标速度最优。"""

    def test_two_position_fallback_is_single_reverse(self):
        """两处布局退化为只检查一次反方向。"""
        end_state = project_end_state(
            intervals=(
                _interval("base", 0, 0, 100, (1000, 1000)),
                _interval("increment", 1, 100, 150, (1500, 1500),
                          role=INTERVAL_ROLE_GROWTH),
            ),
            decode_instance=1, logic_home=0)
        plan = _model(_loads2()).merge_from_layout(end_state, notes=[])
        self.assertEqual(plan.target_instance, 1)
        self.assertEqual(plan.capacity_fallback_order, (0,))

    def test_fallback_follows_retention_descending_at_most_two(self):
        end_state = _three_position_state()
        plan = _model(_loads3()).merge_from_layout(end_state, notes=[])
        # 偏好序（保留量降序）= 2:2000, 1:1200, 0:600；主目标 2 →
        # 回退序 = (1, 0)（至多两次替代检查）。
        self.assertEqual(
            plan.retention_disclosure,
            ((2, 2000), (1, 1200), (0, 600)))
        self.assertEqual(plan.capacity_fallback_order, (1, 0))

    def test_no_target_speed_search(self):
        """保留量最大的目标即使空间等待更贵也不被更廉替代目标翻转
        （选择只看式(21)；替代计划仅审计披露）。"""
        end_state = project_end_state(
            intervals=(
                _interval("base", 0, 0, 100, (3000, 3000)),
                _interval("increment", 1, 100, 140, (500, 500),
                          role=INTERVAL_ROLE_GROWTH),
            ),
            decode_instance=1, logic_home=0)
        # home=0 空间紧张（仅 100 B/rank），exec=1 充裕：按速度搜索
        # 会选 exec，式(21) 恒选 home。
        model = _model({
            0: _load_for(0, remaining=(100, 100)),
            1: _load_for(1),
        })
        notes: list = []
        plan = model.merge_from_layout(end_state, notes=notes)
        self.assertEqual(plan.target_instance, 0)
        self.assertEqual(plan.capacity_fallback_order, (1,))
        self.assertGreater(plan.target_space_wait_ns, 0)
        self.assertEqual(
            [(leg.source_instance, leg.target_instance) for leg in plan.legs],
            [(1, 0)])
        self.assertIn("merge_capacity_fallback_order=1", notes)
        self.assertIn("merge_capacity_lock=first_payload_commit", notes)


# ------------------------------------------------------- 返回面与notes --


class MergePlanSurfaceTest(unittest.TestCase):
    """merge_from_layout 返回面：固定目标、缺失区间、通道/引用义务、
    目标空间。"""

    def test_legs_carry_intervals_paths_and_reference_obligation(self):
        end_state = _three_position_state()
        plan = _model(_loads3()).merge_from_layout(end_state, notes=[])
        for leg in plan.legs:
            self.assertTrue(leg.source_holds_reference_until_ack)
            self.assertTrue(leg.paths_by_rank)          # 通道非空
            self.assertTrue(leg.intervals)              # 缺失区间非空
            self.assertEqual(
                leg.total_bytes,
                sum(int(v) for v in leg.bytes_by_rank))
        # 目标空间 = 各来源缺失区间直和（净新增，service_done 投影）。
        self.assertEqual(
            plan.target_space_bytes_by_rank,
            tuple(sum(leg.bytes_by_rank[rank] for leg in plan.legs)
                  for rank in range(len(plan.target_space_bytes_by_rank))))

    def test_two_position_notes_keep_legacy_disclosure(self):
        """两位置退化保留既定 merge_v2 双腿审计面＋eq21 新键。"""
        end_state = project_end_state(
            intervals=(
                _interval("base", 0, 0, 100, (1000, 1000)),
                _interval("increment", 1, 100, 150, (1500, 1500),
                          role=INTERVAL_ROLE_GROWTH),
            ),
            decode_instance=1, logic_home=0)
        notes: list = []
        plan = _model(_loads2()).merge_from_layout(end_state, notes=notes)
        self.assertEqual(plan.target_instance, 1)
        self.assertIn("merge_v2_direction=reverse", notes)
        self.assertIn("merge_v2_direction_rule=byte_le_tie_forward", notes)
        self.assertIn("merge_to_home=0", notes)
        forward_note = next(
            note for note in notes
            if note.startswith("merge_v2_forward_ns="))
        reverse_note = next(
            note for note in notes
            if note.startswith("merge_v2_reverse_ns="))
        # reverse = 选中计划（目标 1），forward = 审计替代计划（目标 0）。
        self.assertEqual(int(reverse_note.split("=")[1]), plan.merge_ns)
        alternate_home = next(
            alternate for alternate in plan.alternates
            if alternate.target_instance == 0)
        self.assertEqual(
            int(forward_note.split("=")[1]), alternate_home.merge_ns)
        self.assertIn(
            "merge_eq21_rule=" + MERGE_SELECTION_RETENTION_MAX, notes)

    def test_three_position_notes_disclose_eq21_without_legacy_directions(self):
        end_state = _three_position_state()
        notes: list = []
        _model(_loads3()).merge_from_layout(end_state, notes=notes)
        self.assertIn("merge_eq21_target=2", notes)
        self.assertIn(
            "merge_eq21_retention=2:2000,1:1200,0:600", notes)
        self.assertNotIn("merge_v2_direction=forward", notes)
        self.assertNotIn("merge_v2_direction=reverse", notes)
        self.assertIn("merge_capacity_fallback_order=1,0", notes)


# --------------------------------------------------------- 入口集成 --


class EndStateOverridesActionNameTest(unittest.TestCase):
    """P 阶段 action 不覆盖真实 D 末态（§15.1）。"""

    def test_copy_end_state_union_not_merged_back_to_bigger_home(self):
        """copy 末态 = 执行端完整并集：home 更大也不搬回（action 名不
        决定合并方向）。"""
        from joint.joint_cost_model import SessionKVView, RequestView
        model = _model(_loads2())
        candidate = model.estimate_P_with_continuation(
            session=SessionKVView(
                session_id="s", home_instance=0, resident_instance=0,
                location="local_hbm", history_tokens=50,
                resident_prefix_layers=4,
                history_bytes_by_tp_rank=(6000, 6000),
                missing_bytes_by_tp_rank=(0, 0)),
            request=RequestView(
                request_id="r", session_id="s", input_tokens=5,
                history_tokens_before=50, estimated_decode_tokens=10,
                horizon_source="run_online_mean",
                input_kv_bytes_by_tp_rank=(500, 500)),
            instance_index=1, action="copy")
        self.assertTrue(candidate.applicable)
        self.assertEqual(candidate.breakdown.merge_ns, 0)
        self.assertIn("merge_v2_zero_byte_flip", candidate.breakdown.notes)
        self.assertIn(
            "merge_eq21_rule=" + MERGE_SELECTION_COMPLETE_UNION,
            candidate.breakdown.notes)


class DBoundaryEq21IntegrationTest(unittest.TestCase):
    """D 边界结算集成：三实例几何下的式(21) 目标与 ω_B 平局。"""

    def test_complete_boundary_copy_zero_payload(self):
        """完整型 copy@j：j 持并集 → 直接作为结果位置，零载荷。"""
        boundary = _complete_boundary(p_instance_index=0)
        model = _model(_loads())
        candidate = model.estimate_D_remaining(
            boundary=boundary, instance_index=1, action="copy")
        self.assertTrue(candidate.applicable)
        self.assertEqual(candidate.breakdown.merge_ns, 0)
        self.assertIn("merge_v2_zero_byte_flip", candidate.breakdown.notes)


# --------------------------------------------------------------- 装置 --


def _loads2():
    return _loads()


def _load_for(instance_index, remaining=(10 ** 9, 10 ** 9)):
    from joint.joint_cost_model import InstanceLoadView
    return InstanceLoadView(
        instance_index=instance_index, queued_task_load_ns=0,
        running_task_load_ns=0, active_decode_task_load_ns=0,
        hbm_remaining_bytes_by_tp_rank=remaining,
        reclaimable_bytes_by_tp_rank=(10 ** 9, 10 ** 9))


def _three_position_state():
    """三位置末态：home=2 持 base 2000、i_P=0 持增量 300、j=1 持
    增长 600（/token 128 的抽象字节；平局次序测试另有等量构造）。"""
    return project_end_state(
        intervals=(
            _interval("base", 2, 0, 100, (1000, 1000)),
            _interval("increment", 0, 100, 140, (300, 300),
                      role=INTERVAL_ROLE_GROWTH),
            _interval("growth", 1, 140, 220, (600, 600),
                      role=INTERVAL_ROLE_CONTEXT),
        ),
        decode_instance=1, logic_home=2)


if __name__ == "__main__":
    unittest.main()
