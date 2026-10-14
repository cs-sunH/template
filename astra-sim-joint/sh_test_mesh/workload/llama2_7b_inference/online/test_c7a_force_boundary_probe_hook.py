#!/usr/bin/env python3
"""test_c7a_force_boundary_probe_hook.py -- C7a 钉测：同状态对照 probe 的
验证专用强制钩子（run_scripts/joint_same_state_probe.py 契约）。

合同（工具侧已写死，调度器侧按同值实现）：

* env ``JOINT_FORCE_BOUNDARY_REQUEST`` + ``JOINT_FORCE_BOUNDARY_ACTION``
  成对设置才武装（缺省 off，行为零变化；半配置/动作名非法启动失败）；
* boundary 臂只强制指定请求 **D 边界**决策的**最终选择**（修复轮三
  R8：P 准入侧接线已删——不碰 P 入口，P 面强制一律走 entry 钩子
  ``JOINT_FORCE_ENTRY_ACTION``；BOUNDARY env 武装时 P 准入不受影响，
  selection = 自然 argmin）——计价照常算全部候选、决策行照常披露
  （同状态对照的计价输入不变）；
* 强制动作必须在该请求 D 边界时刻的适用面（掩码 + 物理适用性）内，
  否则 fail-closed 报错（测量仪器不得制造非法状态）；
* 决策行披露 ``forced_by_probe``（off 时键缺席 = 行与既有钉测逐位一致）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 -m pytest online/test_c7a_force_boundary_probe_hook.py -q
"""
import copy
import inspect
import os
import sys
import unittest

_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _p in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from joint.joint_config import ALL_ACTIONS  # noqa: E402
from joint.joint_cost_model import ActionCandidate  # noqa: E402
from joint.joint_scheduler import SelectionRecord  # noqa: E402
from joint.predictor_cost_accounting import (  # noqa: E402
    DECISION_STAGE_D,
    DECISION_STAGE_P,
)
from online.sh30_online_scheduler import (  # noqa: E402
    FORCE_BOUNDARY_ACTION_ENV,
    FORCE_BOUNDARY_REQUEST_ENV,
    FORCE_ENTRY_ACTION_ENV,
    Sh30OnlineScheduler,
    _parse_force_boundary_env,
)


def _price_face_rows(candidates):
    """决策行候选面的计价零漂移比较面（2026-09-30 修复轮同步
    test_joint_telemetry_divisor._price_face 口径）：剥除逐候选
    breakdown 的 ``projection_stats`` 墙时分段——§5.6 交付面的
    "场景和耗时"披露（wall_ns）与 2026-10-05 埋点新增的构建/前推
    粗相分段（build_wall_ns / closure_wall_ns /
    candidate_world_wall_ns / reference_world_wall_ns）及后续
    joint_cost_model 增补的精确重价墙时（exact_reprice_wall_ns /
    first_projection.wall_ns）均为宿主墙时非计价量、逐次运行必不同；
    其余字段（适用性/代价/理由/事件与闭包计数/build_graph_legs/
    reference_world_runs 及 first_projection.events 等确定性披露）
    保留逐位一致语义（"同状态计价输入不变"钉测不变）。"""
    rows = copy.deepcopy(candidates)
    for row in rows:
        stats = (row.get("breakdown") or {}).get("projection_stats")
        if isinstance(stats, dict):
            for volatile_key in (
                    "wall_ns", "build_wall_ns", "closure_wall_ns",
                    "candidate_world_wall_ns", "reference_world_wall_ns",
                    "exact_reprice_wall_ns"):
                stats.pop(volatile_key, None)
            first_projection = stats.get("first_projection")
            if isinstance(first_projection, dict):
                first_projection.pop("wall_ns", None)
    return rows


# --------------------------------------------------------------- env 解析 --

class ForceBoundaryEnvParseTest(unittest.TestCase):
    """env 成对合同（fail-closed：半配置比没有更糟）。"""

    def test_off_by_default(self):
        self.assertEqual(_parse_force_boundary_env({}), (None, None))

    def test_half_configuration_fails_closed(self):
        for env in (
            {FORCE_BOUNDARY_REQUEST_ENV: "r"},
            {FORCE_BOUNDARY_ACTION_ENV: "copy"},
        ):
            with self.assertRaises(RuntimeError):
                _parse_force_boundary_env(env)

    def test_empty_request_id_rejected(self):
        with self.assertRaises(RuntimeError):
            _parse_force_boundary_env(
                {FORCE_BOUNDARY_REQUEST_ENV: "",
                 FORCE_BOUNDARY_ACTION_ENV: "copy"})

    def test_unknown_action_name_rejected(self):
        with self.assertRaises(RuntimeError):
            _parse_force_boundary_env(
                {FORCE_BOUNDARY_REQUEST_ENV: "r",
                 FORCE_BOUNDARY_ACTION_ENV: "teleport"})

    def test_valid_pair_arms_the_hook(self):
        self.assertEqual(
            _parse_force_boundary_env(
                {FORCE_BOUNDARY_REQUEST_ENV: "session_9_request_9",
                 FORCE_BOUNDARY_ACTION_ENV: "copy"}),
            ("session_9_request_9", "copy"))

    def test_env_names_match_probe_contract(self):
        # 契约由 run_scripts/joint_same_state_probe.py 写死，不得偏离。
        self.assertEqual(FORCE_BOUNDARY_REQUEST_ENV,
                         "JOINT_FORCE_BOUNDARY_REQUEST")
        self.assertEqual(FORCE_BOUNDARY_ACTION_ENV,
                         "JOINT_FORCE_BOUNDARY_ACTION")


# --------------------------------------------------------- 裁决点（单元）--

def _candidate(instance_index, action, *, applicable=True, cost_ns=1000,
               reason=None, is_continuation=False):
    return ActionCandidate(
        instance_index=instance_index, action=action,
        applicable=applicable,
        inapplicable_reason=(None if applicable else (
            reason or "physically infeasible")),
        cost_ns=(cost_ns if applicable else None),
        is_continuation=is_continuation)


def _record(candidates, chosen):
    return SelectionRecord(
        mode="joint", chosen=chosen, candidates=tuple(candidates),
        instance_rule_note="test")


class ForceBoundaryChoiceUnitTest(unittest.TestCase):
    """_apply_forced_boundary_choice 单一裁决点语义（__new__ 壳）。"""

    def _scheduler(self, request_id="r0", action="copy"):
        scheduler = Sh30OnlineScheduler.__new__(Sh30OnlineScheduler)
        scheduler._force_boundary_request = request_id
        scheduler._force_boundary_action = action
        return scheduler

    def _surface(self):
        stay0 = _candidate(0, "stay", cost_ns=500, is_continuation=True)
        copy0 = _candidate(0, "copy", cost_ns=400)
        copy1 = _candidate(1, "copy", cost_ns=900)
        recompute1 = _candidate(1, "recompute", applicable=False,
                                reason="mask excluded")
        record = _record(
            [stay0, copy0, copy1, recompute1], chosen=copy0)
        return record

    def test_off_returns_record_untouched(self):
        # 未武装（类级缺省）与请求不匹配：原 record 对象原样返回、无
        # 披露——off 行为零变化（stage 无关；边界钩子只接线 D 边界，
        # 单元调用一律用接线 stage = D）。
        record = self._surface()
        unarmed = Sh30OnlineScheduler.__new__(Sh30OnlineScheduler)
        self.assertIs(
            unarmed._apply_forced_boundary_choice(
                record, "r0", DECISION_STAGE_D)[0], record)
        mismatch = self._scheduler(request_id="other")
        self.assertIs(
            mismatch._apply_forced_boundary_choice(
                record, "r0", DECISION_STAGE_D)[0], record)

    def test_forced_action_selected_pricing_untouched(self):
        # on：强制动作被选（适用面内多实例取 min cost），candidates
        # 元组同一对象（计价输入不变），披露 forced_by_probe（stage=D
        # ——boundary 臂只强制 D 边界）。
        record = self._surface()
        scheduler = self._scheduler(action="copy")
        forced, note = scheduler._apply_forced_boundary_choice(
            record, "r0", DECISION_STAGE_D)
        self.assertIsNot(forced, record)
        self.assertIs(forced.candidates, record.candidates)
        self.assertEqual(forced.chosen.action, "copy")
        self.assertEqual(forced.chosen.instance_index, 0)  # min cost
        self.assertEqual(
            note,
            {"stage": DECISION_STAGE_D, "hook": "boundary", "action": "copy",
             "instance_index": 0, "is_continuation": False})

    def test_outside_surface_fails_closed_with_reasons(self):
        record = self._surface()
        scheduler = self._scheduler(action="recompute")
        with self.assertRaises(RuntimeError) as ctx:
            scheduler._apply_forced_boundary_choice(
                record, "r0", DECISION_STAGE_D)
        message = str(ctx.exception)
        self.assertIn(FORCE_BOUNDARY_ACTION_ENV, message)
        self.assertIn("applicable surface", message)
        self.assertIn("mask excluded", message)
        # 候选面完全没有的动作（无逐候选理由）同拒。
        scheduler2 = self._scheduler(action="stay")
        record2 = _record(
            [_candidate(0, "copy", cost_ns=10)], chosen=None)
        with self.assertRaises(RuntimeError) as ctx2:
            scheduler2._apply_forced_boundary_choice(
                record2, "r0", DECISION_STAGE_D)
        self.assertIn("not in candidate surface", str(ctx2.exception))

    def test_stage_d_tie_break_prefers_continuation(self):
        # D 入口平局键（式 1c）：cost 并列时原地继续优先于迁移实例。
        stay0 = _candidate(0, "stay", cost_ns=700, is_continuation=True)
        stay1 = _candidate(1, "stay", cost_ns=700)
        record = _record([stay0, stay1], chosen=stay0)
        scheduler = self._scheduler(action="stay")
        forced, note = scheduler._apply_forced_boundary_choice(
            record, "r0", DECISION_STAGE_D)
        self.assertEqual(forced.chosen.instance_index, 0)
        self.assertEqual(note["stage"], DECISION_STAGE_D)


# ------------------------------------------------- D 入口全链（真决策行）--

def _drive_drain(force_request=None, force_action=None, force_target=None):
    """与 test_decision_route_serialization 同款场景：r0 在实例 0 准入、
    实例 0 重在服负载 → D 边界把 decode 切到轻载实例 1（真 select_
    decode_target_and_action + 真提交 + 真决策行）。返回 (scheduler,
    runtime)；强制钩子按参数武装（缺省 off）。W8：force_target =
    (request, instance, action) 武装 TARGET 精确强制钩子（与 per-action
    两参数互斥使用）。"""
    from face_scheduler import (
        FaceHardware, FaceInstanceSpec, FaceModel, KVCacheManager,
        build_instances,
    )
    from online.test_decision_route_serialization import (
        _decision_shell, _make_runtime,
    )
    hardware = FaceHardware(
        mesh_rows=2, mesh_cols=2, local_hbm_capacity_bytes=1_000_000_000,
        local_hbm_bandwidth_gbps=100.0, d2d_bandwidth_gbps=200.0,
        peak_perf_tflops=1.0, d2d_latency_ns=0, local_hbm_latency_ns=0)
    model = FaceModel(
        layers=4, hidden_size=16, ffn_size=32, num_heads=4, vocab_size=32,
        bytes_per_elem=2, mlp_variant="swiglu")
    topology = build_instances(
        hardware, (FaceInstanceSpec("ins0", "1", (0, 1)),
                   FaceInstanceSpec("ins1", "2", (2, 3))))
    kv = KVCacheManager(topology, model)
    scheduler = _decision_shell(
        kv, hardware=hardware, model=model, topology=topology,
        no_affinity=True)
    # 与 test_decision_route_serialization 同款动作面掩码（copy,stay：
    # 迁移臂 + 式 1c 回退腿保持合法）——被测机器是强制钩子，掩码收窄
    # 使基线选择与既有钉测同值（copy@1）且面外动作集确定。
    from joint.joint_config import parse_joint_config
    scheduler.joint_config = parse_joint_config(
        {"JOINT_ALLOWED_ACTIONS": "copy,stay"})
    scheduler._joint_mode = scheduler.joint_config.scheduler_mode
    scheduler._force_boundary_request = force_request
    scheduler._force_boundary_action = force_action
    scheduler._force_boundary_target = force_target
    runtime = _make_runtime("r0")
    scheduler.runtime_by_request_id["r0"] = runtime
    kv.reserve_request_capacity(
        request_id="r0", session_id=runtime.session_id,
        instance_index=0,
        final_context_tokens=runtime.prefill_context_tokens)
    kv.prepare_prefill(
        session_id=runtime.session_id, target_instance_index=0,
        history_tokens=0, trigger_request_id="r0")
    kv.expand_prefill(
        session_id=runtime.session_id, instance_index=0,
        context_tokens=runtime.prefill_context_tokens,
        trigger_request_id="r0")
    runtime.prefill_instance_index = 0
    scheduler.instances[0].qp.append(runtime)
    runtime.queued_chunk_load_ns = 0
    for seq in range(3):
        heavy = _make_runtime(f"active0_{seq}")
        scheduler.runtime_by_request_id[f"active0_{seq}"] = heavy
        scheduler.instances[0].active_decode.append(heavy)
    scheduler._on_prefill_drain("r0", 1000)
    return scheduler, runtime


def _decode_row(scheduler):
    rows = [row for row in scheduler.online_log_rows
            if row["kind"] == "decode"]
    assert len(rows) == 1, rows
    return rows[0]


class DPathForceHookIntegrationTest(unittest.TestCase):
    """D 入口全链：off 逐位一致 / on 强制生效且计价输入不变 / 面外
    fail-closed。"""

    def test_off_row_matches_baseline_schema(self):
        scheduler, runtime = _drive_drain()
        decision = _decode_row(scheduler)["decision"]
        boundary = decision["boundary_decision"]
        # off：无 forced_by_probe 键（行形态与既有钉测逐位一致——
        # test_decision_route_serialization 同场景钉 chosen=copy@1）。
        self.assertNotIn("forced_by_probe", boundary)
        self.assertNotIn("forced_by_probe", decision)
        self.assertEqual(boundary["chosen_action"], "copy")
        self.assertEqual(boundary["chosen_instance_index"], 1)

    def test_on_forces_action_with_identical_pricing_inputs(self):
        off_scheduler, _off_rt = _drive_drain()
        off_boundary = _decode_row(off_scheduler)["decision"][
            "boundary_decision"]
        applicable = sorted({
            row["action"] for row in off_boundary["candidates"]
            if row["applicable"]})
        # 钉测前提：本场景适用面 ≥2 动作（可强制到非基线动作）。
        self.assertGreaterEqual(len(applicable), 2, applicable)
        forced_action = next(
            action for action in ("stay", "recompute", "copy")
            if action in applicable and action != "copy")
        on_scheduler, on_rt = _drive_drain(
            force_request="r0", force_action=forced_action)
        decision = _decode_row(on_scheduler)["decision"]
        boundary = decision["boundary_decision"]
        # on：强制动作被选 + 披露在位。
        self.assertEqual(boundary["chosen_action"], forced_action)
        self.assertEqual(
            decision["joint_decode_action"], forced_action)
        self.assertEqual(
            boundary["forced_by_probe"]["stage"], DECISION_STAGE_D)
        self.assertEqual(
            boundary["forced_by_probe"]["hook"], "boundary")
        self.assertEqual(
            boundary["forced_by_probe"]["action"], forced_action)
        self.assertEqual(
            boundary["forced_by_probe"]["instance_index"],
            boundary["chosen_instance_index"])
        self.assertIn(
            "is_continuation", boundary["forced_by_probe"])
        # 同状态（计价输入）：候选面逐位一致（适用性/代价/理由/实例；
        # 非计价遥测 wall_ns 经 _price_face_rows 剥除）。
        self.assertEqual(
            _price_face_rows(boundary["candidates"]),
            _price_face_rows(off_boundary["candidates"]))
        # 强制 stay = 原地接续（式 1c 回退腿），decode 实例留在 0。
        self.assertEqual(forced_action, "stay")
        self.assertEqual(boundary["chosen_instance_index"], 0)

    def test_on_outside_surface_fails_closed(self):
        off_scheduler, _rt = _drive_drain()
        boundary = _decode_row(off_scheduler)["decision"][
            "boundary_decision"]
        applicable = {
            row["action"] for row in boundary["candidates"]
            if row["applicable"]}
        outside = [action for action in ALL_ACTIONS
                   if action not in applicable]
        # 掩码收窄后面外动作集确定（recompute 被掩码剔除）。
        self.assertTrue(outside, applicable)
        with self.assertRaises(RuntimeError) as ctx:
            _drive_drain(force_request="r0", force_action=outside[0])
        message = str(ctx.exception)
        self.assertIn(FORCE_BOUNDARY_ACTION_ENV, message)
        self.assertIn("applicable surface", message)


class ProductionWiringTest(unittest.TestCase):
    """两入口的强制点接线（P：只接 entry 钩子——BOUNDARY 不触 P 面，
    修复轮三 R8；D：BOUNDARY 钩子在选择器产出之后）。"""

    def test_p_entry_wires_only_entry_hook_after_quota_reselect(self):
        # 修复轮三 R8：P 准入不再接线 BOUNDARY 钩子——boundary 臂只强制
        # D 边界（P 侧接线会在到达 D 边界前先被 P 面适用性核验 fail-
        # closed——D 面可适用动作在 turn-0 P 面恒不适用——且 D__recompute
        # 会把 P 选择静默改写为 recompute ≠ entry_pin=stay）。P 面强制
        # 只走 entry 钩子，披露键照常落 forced_by_probe。
        source = inspect.getsource(Sh30OnlineScheduler._try_admit_request)
        self.assertNotIn("_apply_forced_boundary_choice", source)
        self.assertIn("_apply_forced_entry_choice", source)
        self.assertIn("forced_by_probe", source)

    def test_d_entry_applies_hook_after_selector(self):
        source = inspect.getsource(
            Sh30OnlineScheduler._on_prefill_drain)
        self.assertIn("_apply_forced_boundary_choice", source)
        self.assertIn(DECISION_STAGE_D, source)

    def test_init_parses_probe_env(self):
        source = inspect.getsource(Sh30OnlineScheduler.__init__)
        self.assertIn("_parse_force_boundary_env", source)


# ------------------------------------------------- P 入口全链（真决策行）--

class PPathForceHookIntegrationTest(unittest.TestCase):
    """P 入口全链（test_joint_decision_schema 同款真 _try_admit_request
    壳）：off 行形态不变 / BOUNDARY env 武装不触 P 面（selection=自然
    argmin——修复轮三 R8 合同，轮三前 P 侧接线会在到达 D 边界前先被 P
    面适用性核验 fail-closed）/ entry 字段强制生效且计价输入不变 /
    entry 面外 fail-closed。"""

    @staticmethod
    def _admit(force_request=None, force_action=None,
               entry_request=None, entry_action=None, allowed="all"):
        from joint.joint_config import parse_joint_config
        from online.test_joint_decision_schema import (
            _admission_rows, _make_scheduler, _runtime,
        )
        scheduler = _make_scheduler()
        scheduler.joint_config = parse_joint_config(
            {"JOINT_ALLOWED_ACTIONS": allowed})
        scheduler._joint_mode = scheduler.joint_config.scheduler_mode
        scheduler._force_boundary_request = force_request
        scheduler._force_boundary_action = force_action
        scheduler._force_entry_request = entry_request
        scheduler._force_entry_action = entry_action
        runtime = _runtime("r_p")
        admitted = scheduler._try_admit_request(runtime, 0)
        return scheduler, admitted, _admission_rows(scheduler)

    def test_off_row_has_no_probe_key(self):
        _scheduler, admitted, rows = self._admit()
        self.assertTrue(admitted)
        self.assertEqual(len(rows), 1)
        decision = rows[0]["decision"]
        self.assertNotIn("forced_by_probe", decision)

    def test_boundary_armed_leaves_p_admission_at_natural_argmin(self):
        # 修复轮三 R8 合同钉：BOUNDARY env 武装（遍历全部动作，含 P 面
        # 不适用的 recompute——轮三前即红 'no history to
        # copy' at its P-boundary）不影响 P 准入：照常 admitted、
        # selection=自然 argmin（=off 基线逐字段一致）、无披露键、候选
        # 面逐位一致（同状态对照的计价输入不变）。D 边界照常武装由
        # DPathForceHookIntegrationTest 全链钉测承载。
        _off_s, _off_ok, off_rows = self._admit()
        off_decision = off_rows[0]["decision"]
        for action in ALL_ACTIONS:
            _s, admitted, rows = self._admit(
                force_request="r_p", force_action=action)
            self.assertTrue(admitted, action)
            self.assertEqual(len(rows), 1, action)
            decision = rows[0]["decision"]
            self.assertNotIn("forced_by_probe", decision, action)
            self.assertEqual(
                decision["selected_action"],
                off_decision["selected_action"], action)
            self.assertEqual(
                decision["joint_action"],
                off_decision["joint_action"], action)
            self.assertEqual(
                decision["joint_instance_index"],
                off_decision["joint_instance_index"], action)
            # 同状态（计价输入）：候选面逐位一致（适用性/代价/理由/实例；
            # 非计价遥测 wall_ns 经 _price_face_rows 剥除）。
            self.assertEqual(
                _price_face_rows(decision["candidates"]),
                _price_face_rows(off_decision["candidates"]), action)
            self.assertEqual(
                decision["applicable_actions"],
                off_decision["applicable_actions"], action)

    def test_entry_field_forces_action_with_identical_pricing_inputs(self):
        # P 侧强制用例（修复轮三 R8 自 BOUNDARY 字段迁移到 entry 字段
        # ——boundary 臂不再触 P 面）：on——强制动作被选（selected_
        # action/joint_action 同值，实例属该动作的适用候选）+ 披露在位
        # （stage="P"、hook="entry"），候选面逐位一致（计价输入不变）。
        _off_s, _off_ok, off_rows = self._admit()
        off_decision = off_rows[0]["decision"]
        baseline = off_decision["selected_action"]
        applicable = list(off_decision["applicable_actions"])
        forced_action = next(
            action for action in ("copy", "recompute", "stay")
            if action in applicable and action != baseline)
        _on_s, admitted, on_rows = self._admit(
            entry_request="r_p", entry_action=forced_action)
        self.assertTrue(admitted)
        decision = on_rows[0]["decision"]
        self.assertEqual(decision["selected_action"], forced_action)
        self.assertEqual(decision["joint_action"], forced_action)
        self.assertEqual(
            decision["forced_by_probe"]["stage"], DECISION_STAGE_P)
        self.assertEqual(decision["forced_by_probe"]["hook"], "entry")
        self.assertEqual(
            decision["forced_by_probe"]["action"], forced_action)
        self.assertEqual(
            decision["forced_by_probe"]["instance_index"],
            decision["joint_instance_index"])
        self.assertIn("is_continuation",
                      decision["forced_by_probe"])
        forced_candidates = [
            row for row in decision["candidates"]
            if row["action"] == forced_action and row["applicable"]]
        self.assertTrue(forced_candidates)
        self.assertIn(decision["joint_instance_index"],
                      {row["instance_index"]
                       for row in forced_candidates})
        # 同状态（计价输入）：候选面逐位一致（适用性/代价/理由/实例；
        # 非计价遥测 wall_ns 经 _price_face_rows 剥除）。
        self.assertEqual(
            _price_face_rows(decision["candidates"]),
            _price_face_rows(off_decision["candidates"]))
        self.assertEqual(
            decision["applicable_actions"],
            off_decision["applicable_actions"])

    def test_entry_field_outside_surface_fails_closed(self):
        # 掩码收窄到 {copy,stay} 后经 entry 字段强制 recompute：面外
        # 动作（掩码+物理适用性∩配额可行交集外）fail-closed——测量仪器
        # 不得制造非法状态（BOUNDARY 臂的面外 fail-closed 钉在
        # DPathForceHookIntegrationTest）。
        with self.assertRaises(RuntimeError) as ctx:
            self._admit(entry_request="r_p", entry_action="recompute",
                        allowed="copy,stay")
        message = str(ctx.exception)
        self.assertIn(FORCE_ENTRY_ACTION_ENV, message)
        self.assertIn("applicable surface", message)


if __name__ == "__main__":
    unittest.main()
