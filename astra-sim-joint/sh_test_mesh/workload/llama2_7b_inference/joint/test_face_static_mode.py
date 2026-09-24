#!/usr/bin/env python3
"""test_face_static_mode.py -- C17（WP6b-静态臂）face_static 定向验收单测。

对象：joint_config 的 ``face_static`` 调度模式取值域扩展 + quota 强制
耦合（F7 唯一耦合例外），joint_scheduler 的静态距离球掩码臂
（``hop ≤ floor(rho)``，逐配置硬件派生）。全部为 python 级构造性用例
（不跑仿真，零后端）。

纪律断言（冻结项，违反即事故）：

* **joint 主臂永不掩码**——joint 臂无任何掩码语义（掩码键恒不出现，
  全局 argmin 可选中球外实例）；face_static 是唯一掩码分支；
* **身份纪律（设计文档 §5.3/§1.3）**：face_static 是本文内部的静态
  距离变体，manifest 注记 ``face_static_identity`` 必须声明"非完整
  FACE、不宣称零负载特例"；非 face_static 模式下该键为 None；
* **quota 强制 off**：face_static + 非 off quota 配置 = 显式覆盖为
  off + manifest 注记 "face_static forces quota-off"，**不 fail-closed
  拒绝启动**（对照臂语义）；quota 开关自身（C9 交付）不受影响；
* 掩码半径逐配置派生（base 配置 floor(4050/1640) = floor(2.47) = 2；
  8100/1640 → 4；ρ<1 → 0 退化仅锚实例，如实记录，无 max(1,·) 保底）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 joint/test_face_static_mode.py
"""
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for _p in (_HERE, _PARENT, os.path.join(_PARENT, "online")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from joint.joint_config import (  # noqa: E402
    COMBO_PRESETS,
    JointConfigError,
    JointMechanismConfig,
    QUOTA_OFF,
    QUOTA_STATIC,
    QUOTA_AIMD,
    SCHEDULER_MODES,
    SCHEDULER_MODE_FACE_STATIC,
    parse_joint_config,
)
from joint.joint_cost_model import (  # noqa: E402
    InstanceLoadView,
    JointCostModel,
    JointHardwareRates,
    LinkFlowRegistry,
    RequestView,
    ServiceFactors,
    SessionKVView,
)
from joint.joint_scheduler import (  # noqa: E402
    JointSchedulerError,
    _static_ball_radius,
    select_instance_and_action,
)


# ============================================================ 开关解析 ==


class FaceStaticModeParseTest(unittest.TestCase):
    """模式解析：取值域扩展 + fail-closed + 非八组合成员。"""

    def test_face_static_in_mode_domain_and_parses(self):
        self.assertIn(SCHEDULER_MODE_FACE_STATIC, SCHEDULER_MODES)
        config = parse_joint_config(
            {"JOINT_SCHEDULER_MODE": SCHEDULER_MODE_FACE_STATIC})
        self.assertEqual(config.scheduler_mode, "face_static")
        # 对照臂非 J 主臂：j_enabled 恒 False。
        self.assertFalse(config.j_enabled)
        self.assertIsNone(config.combo)

    def test_face_static_not_in_eight_combo_presets(self):
        """policy variant：八组合固定映射的 scheduler 档恒 joint/
        load-first，face_static 只经显式开关进入。"""
        for combo, (_category, scheduler, _layer) in COMBO_PRESETS.items():
            self.assertNotEqual(
                scheduler, SCHEDULER_MODE_FACE_STATIC, combo)

    def test_invalid_values_fail_closed(self):
        for env in (
            {"JOINT_SCHEDULER_MODE": "face"},
            {"JOINT_SCHEDULER_MODE": "FACE_STATIC"},
            {"JOINT_SCHEDULER_MODE": "face-static"},
            {"JOINT_SCHEDULER_MODE": " face_static"},
            {"JOINT_SCHEDULER_MODE": ""},
        ):
            with self.assertRaises(JointConfigError, msg=repr(env)):
                parse_joint_config(env)

    def test_combo_preset_conflict_fails_closed(self):
        """combo 预设与显式 scheduler 开关互斥（既有 fail-closed 覆盖
        face_static：对照臂不可经 combo 通道进入）。"""
        with self.assertRaises(JointConfigError):
            parse_joint_config({
                "JOINT_ABLATION_COMBO": "TJE",
                "JOINT_SCHEDULER_MODE": SCHEDULER_MODE_FACE_STATIC,
            })


# ====================================================== quota 强制耦合 ==


class QuotaForceOffTest(unittest.TestCase):
    """F7 唯一耦合例外：face_static ⇒ quota-off（显式覆盖 + 注记，不拒
    绝启动）；quota 开关自身（C9 交付）不受影响。"""

    def test_quota_switch_alone_unchanged(self):
        """C9 无回归：无 face_static 时 quota 解析/记录照旧。"""
        for mode in ("static", "aimd"):
            config = parse_joint_config({"JOINT_QUOTA_MODE": mode})
            self.assertEqual(config.quota_mode, mode)
            self.assertTrue(config.quota_enabled)
            self.assertFalse(config.quota_forced_off)
            manifest = config.manifest_dict()
            self.assertEqual(manifest["quota_mode"], mode)
            self.assertTrue(manifest["quota_enabled"])
            self.assertIsNone(manifest["quota_forced_off_note"])
            self.assertIsNone(manifest["face_static_identity"])

    def test_face_static_forces_non_off_quota_to_off(self):
        for requested in ("static", "aimd"):
            config = parse_joint_config({
                "JOINT_SCHEDULER_MODE": SCHEDULER_MODE_FACE_STATIC,
                "JOINT_QUOTA_MODE": requested,
            })
            # 不 fail-closed 拒绝启动：解析成功且生效值为 off。
            self.assertEqual(config.quota_mode, QUOTA_OFF)
            self.assertFalse(config.quota_enabled)
            self.assertTrue(config.quota_forced_off)
            manifest = config.manifest_dict()
            self.assertEqual(manifest["quota_mode"], QUOTA_OFF)
            self.assertFalse(manifest["quota_enabled"])
            self.assertTrue(manifest["quota_forced_off"])
            self.assertEqual(
                manifest["quota_forced_off_note"],
                "face_static forces quota-off")
            # 来源登记保留 env 原值（覆盖事实由注记键解释，不抹来源）。
            self.assertEqual(
                manifest["source_env"]["JOINT_QUOTA_MODE"], requested)

    def test_face_static_with_explicit_off_not_annotated_as_override(self):
        """quota 本就 off（显式或缺省）时不产生覆盖注记（不虚报）。"""
        for env in (
            {"JOINT_SCHEDULER_MODE": SCHEDULER_MODE_FACE_STATIC},
            {"JOINT_SCHEDULER_MODE": SCHEDULER_MODE_FACE_STATIC,
             "JOINT_QUOTA_MODE": QUOTA_OFF},
        ):
            config = parse_joint_config(env)
            self.assertEqual(config.quota_mode, QUOTA_OFF)
            self.assertFalse(config.quota_forced_off)
            manifest = config.manifest_dict()
            self.assertFalse(manifest["quota_forced_off"])
            self.assertIsNone(manifest["quota_forced_off_note"])

    def test_direct_construction_also_forces_quota_off(self):
        """单一裁决点在构造层：非 env 路径直接构造同样强制（含非法
        quota 值仍先 fail-closed 校验）。"""
        config = JointMechanismConfig(
            category_mode="typed", scheduler_mode=SCHEDULER_MODE_FACE_STATIC,
            layer_policy="adaptive", remote_actions="on",
            quota_mode=QUOTA_AIMD)
        self.assertEqual(config.quota_mode, QUOTA_OFF)
        self.assertTrue(config.quota_forced_off)
        with self.assertRaises(JointConfigError):
            JointMechanismConfig(
                category_mode="typed",
                scheduler_mode=SCHEDULER_MODE_FACE_STATIC,
                layer_policy="adaptive", remote_actions="on",
                quota_mode="bogus")

    def test_identity_note_discipline(self):
        """身份纪律（§5.3/§1.3）：manifest 必须声明静态距离变体身份、
        不冒称完整 FACE、不宣称零负载特例；他模式恒 None。"""
        manifest = parse_joint_config(
            {"JOINT_SCHEDULER_MODE": SCHEDULER_MODE_FACE_STATIC}
        ).manifest_dict()
        identity = manifest["face_static_identity"]
        self.assertIsNotNone(identity)
        self.assertIn("NOT full FACE", identity)
        self.assertIn("zero-load special case", identity)
        self.assertIn("static-distance variant", identity)
        for scheduler in ("joint", "load-first", "affinity-first"):
            other = parse_joint_config(
                {"JOINT_SCHEDULER_MODE": scheduler}).manifest_dict()
            self.assertIsNone(other["face_static_identity"], scheduler)


# ================================================== 静态距离球（选择） ==


def _line_route(source, target):
    """线形拓扑路由：hop = |source − target|（链上确定性最短路径）。"""
    low, high = min(source, target), max(source, target)
    return (tuple(range(low, high + 1)), high - low)


def _cost_model(loads, *, noc_link_gbps, local_hbm_gbps):
    rates = JointHardwareRates.from_gbps(
        noc_link_gbps=noc_link_gbps, pool_port_gbps=5.0,
        local_hbm_gbps=local_hbm_gbps, d2d_latency_ns=10,
        pool_latency_ns=100)
    return JointCostModel(
        rates=rates, loads=loads, flow_registry=LinkFlowRegistry(),
        service_factors=ServiceFactors(),
        prefill_ns_per_token=1.0, decode_ns_per_token=2.0,
        model_layers=4, instance_tp_size=2, route_fn=_line_route)


def _load(index, total, remaining=(10**9, 10**9)):
    return InstanceLoadView(
        instance_index=index, queued_task_load_ns=total,
        running_task_load_ns=0, active_decode_task_load_ns=0,
        hbm_remaining_bytes_by_tp_rank=remaining)


def _session(home=0, resident=0, location="local_hbm", tokens=100):
    return SessionKVView(
        session_id="s", home_instance=home, resident_instance=resident,
        location=location, history_tokens=tokens, resident_prefix_layers=4,
        history_bytes_by_tp_rank=(1000, 1000),
        missing_bytes_by_tp_rank=(0, 0))


def _fresh_session():
    return SessionKVView(
        session_id="s", home_instance=None, resident_instance=None,
        location="none", history_tokens=0, resident_prefix_layers=0,
        history_bytes_by_tp_rank=(0, 0), missing_bytes_by_tp_rank=(0, 0))


def _request(input_tokens=50, decode_estimate=10):
    return RequestView(
        request_id="r", session_id="s", input_tokens=input_tokens,
        history_tokens_before=100,
        estimated_decode_tokens=decode_estimate,
        horizon_source="session_online_mean",
        input_kv_bytes_by_tp_rank=(25, 25))


_BASE_LINK_GBPS = 4050.0    # base 配置 D2D 单链路带宽（face_case5_config_c）
_BASE_HBM_GBPS = 1640.0     # base 配置单 rank 本地 HBM 带宽


class StaticBallRadiusTest(unittest.TestCase):
    """掩码半径逐配置派生：floor(B_D2D/B_HBM)，无 max(1,·) 保底。"""

    def test_base_config_radius_is_two(self):
        """base 配置 floor(4050/1640) = floor(2.47) = 2（卡正文解析例）。"""
        model = _cost_model({}, noc_link_gbps=_BASE_LINK_GBPS,
                            local_hbm_gbps=_BASE_HBM_GBPS)
        rho_eff, radius = _static_ball_radius(model)
        self.assertAlmostEqual(rho_eff, 4050.0 / 1640.0, places=9)
        self.assertEqual(radius, 2)

    def test_radius_is_per_config_not_a_constant(self):
        """逐配置派生（C18 孪生参考点：x2 D2D 8100 → 半径 4；sub
        1200 < 1640 → ρ<1 半径 0）。"""
        model = _cost_model({}, noc_link_gbps=8100.0,
                            local_hbm_gbps=_BASE_HBM_GBPS)
        self.assertEqual(_static_ball_radius(model)[1], 4)
        model_sub = _cost_model({}, noc_link_gbps=1200.0,
                                local_hbm_gbps=_BASE_HBM_GBPS)
        rho_sub, radius_sub = _static_ball_radius(model_sub)
        self.assertLess(rho_sub, 1.0)
        self.assertEqual(radius_sub, 0)

    def test_selection_ball_membership_base_config(self):
        """base 半径 2 + 锚 0（驻留 0）→ 球 = {0,1,2}，掩出 {3,4,5}。"""
        loads = {index: _load(index, 0) for index in range(6)}
        record = select_instance_and_action(
            mode=SCHEDULER_MODE_FACE_STATIC,
            cost_model=_cost_model(
                loads, noc_link_gbps=_BASE_LINK_GBPS,
                local_hbm_gbps=_BASE_HBM_GBPS),
            session=_session(home=0, resident=0),
            request=_request(), remote_enabled=True)
        diagnostics = record.diagnostics
        self.assertEqual(diagnostics["radius_hops"], 2)
        self.assertEqual(diagnostics["static_ball_anchor"], 0)
        self.assertEqual(diagnostics["ball_instances"], (0, 1, 2))
        self.assertEqual(diagnostics["masked_out_instances"], (3, 4, 5))
        self.assertIn("ball", record.instance_rule_note)
        self.assertNotIn("degenerate", record.instance_rule_note)

    def test_rho_below_one_degenerates_to_anchor_only(self):
        """ρ<1：半径 0 → 球仅锚实例（无保底），note 如实披露退化。"""
        loads = {index: _load(index, 0) for index in range(6)}
        record = select_instance_and_action(
            mode=SCHEDULER_MODE_FACE_STATIC,
            cost_model=_cost_model(
                loads, noc_link_gbps=1200.0, local_hbm_gbps=_BASE_HBM_GBPS),
            session=_session(home=0, resident=0),
            request=_request(), remote_enabled=True)
        self.assertEqual(record.diagnostics["radius_hops"], 0)
        self.assertEqual(record.diagnostics["ball_instances"], (0,))
        self.assertEqual(record.diagnostics["masked_out_instances"],
                         (1, 2, 3, 4, 5))
        self.assertEqual(record.chosen.instance_index, 0)
        self.assertIn("degenerate", record.instance_rule_note)


class FaceStaticSelectionTest(unittest.TestCase):
    """选择行为：掩码臂 vs joint 主臂（永不掩码）+ 锚语义 + 边界。"""

    @staticmethod
    def _loaded_model():
        """锚 0、半径 2 配置：球外实例 3/4/5 零负载（全局最优在球外）。"""
        loads = {index: _load(index, 0 if index >= 3 else 10**9)
                 for index in range(6)}
        model = _cost_model(
            loads, noc_link_gbps=_BASE_LINK_GBPS,
            local_hbm_gbps=_BASE_HBM_GBPS)
        session = _session(home=0, resident=0)
        return model, session

    def test_joint_main_arm_never_masks(self):
        """joint 主臂掩码恒空：全局 argmin 可选中球外实例、无掩码键、
        全候选枚举（掩码概念不出现）；face_static 同输入掩出该实例。"""
        model, session = self._loaded_model()
        joint = select_instance_and_action(
            mode="joint", cost_model=model, session=session,
            request=_request(), remote_enabled=True)
        # 全局最优 = 球外零负载实例 3（joint 无掩码，照常选中）。
        self.assertEqual(joint.chosen.instance_index, 3)
        self.assertEqual(len(joint.candidates), 6 * 4)
        for key in ("ball_instances", "masked_out_instances",
                    "radius_hops", "static_ball_anchor"):
            self.assertNotIn(key, joint.diagnostics, key)
        face = select_instance_and_action(
            mode=SCHEDULER_MODE_FACE_STATIC, cost_model=model,
            session=session, request=_request(), remote_enabled=True)
        # 对照臂：实例 3 被静态球掩出，胜者落在球内。
        self.assertIn(3, face.diagnostics["masked_out_instances"])
        self.assertIn(face.chosen.instance_index, (0, 1, 2))
        self.assertEqual(len(face.candidates), 6 * 4)  # 枚举不变，只换选择

    def test_ball_winner_is_argmin_within_ball(self):
        """球内选择 = 掩码集上的 (cost_ns, order_key) argmin（与 joint
        同 tie 序）：胜者代价 = 球内全部候选最小值。"""
        model, session = self._loaded_model()
        record = select_instance_and_action(
            mode=SCHEDULER_MODE_FACE_STATIC, cost_model=model,
            session=session, request=_request(), remote_enabled=True)
        ball = set(record.diagnostics["ball_instances"])
        in_ball = [
            candidate for candidate in record.candidates
            if candidate.instance_index in ball
            and candidate.applicable and candidate.cost_ns is not None]
        best = min(in_ball, key=lambda candidate: (
            candidate.cost_ns, candidate.order_key()))
        self.assertEqual(
            (record.chosen.instance_index, record.chosen.action),
            (best.instance_index, best.action))

    def test_anchor_is_resident_else_home(self):
        """锚与 cost_model 路由同源：有驻留取驻留；REMOTE 基（无驻留）
        回退逻辑 home。"""
        loads = {index: _load(index, 0) for index in range(6)}
        model = _cost_model(
            loads, noc_link_gbps=_BASE_LINK_GBPS,
            local_hbm_gbps=_BASE_HBM_GBPS)
        # 驻留 1 优先于 home 0（物理数据位置决定距离）；线形拓扑
        # hop = |i − 1| → 球 = {0,1,2,3}。
        record = select_instance_and_action(
            mode=SCHEDULER_MODE_FACE_STATIC, cost_model=model,
            session=_session(home=0, resident=1),
            request=_request(), remote_enabled=True)
        self.assertEqual(record.diagnostics["static_ball_anchor"], 1)
        self.assertEqual(record.diagnostics["ball_instances"], (0, 1, 2, 3))
        self.assertEqual(record.diagnostics["masked_out_instances"], (4, 5))
        # REMOTE 基：resident None → 锚 = home 2，球 = {0,1,2,3,4}。
        record_remote = select_instance_and_action(
            mode=SCHEDULER_MODE_FACE_STATIC, cost_model=model,
            session=_session(home=2, resident=None,
                             location="remote_memory"),
            request=_request(), remote_enabled=True)
        self.assertEqual(
            record_remote.diagnostics["static_ball_anchor"], 2)
        self.assertEqual(
            record_remote.diagnostics["ball_instances"], (0, 1, 2, 3, 4))

    def test_fresh_session_ball_vacuous_admits_all(self):
        """turn-0 无历史无锚：球退化为空约束（全部实例参与、如实记
        录、不 fail-closed），选择与 joint 全候选 argmin 一致。"""
        loads = {index: _load(index, 0 if index >= 3 else 10**9)
                 for index in range(6)}
        model = _cost_model(
            loads, noc_link_gbps=_BASE_LINK_GBPS,
            local_hbm_gbps=_BASE_HBM_GBPS)
        request = _request()
        record = select_instance_and_action(
            mode=SCHEDULER_MODE_FACE_STATIC, cost_model=model,
            session=_fresh_session(), request=request,
            remote_enabled=True)
        self.assertTrue(record.diagnostics["ball_vacuous"])
        self.assertIsNone(record.diagnostics["static_ball_anchor"])
        self.assertEqual(record.diagnostics["ball_instances"],
                         (0, 1, 2, 3, 4, 5))
        self.assertEqual(record.diagnostics["masked_out_instances"], ())
        joint = select_instance_and_action(
            mode="joint", cost_model=model, session=_fresh_session(),
            request=request, remote_enabled=True)
        self.assertEqual(
            (record.chosen.instance_index, record.chosen.action),
            (joint.chosen.instance_index, joint.chosen.action))

    def test_unknown_scheduler_mode_still_fails_closed(self):
        model, session = self._loaded_model()
        with self.assertRaises(JointSchedulerError):
            select_instance_and_action(
                mode="face", cost_model=model, session=session,
                request=_request(), remote_enabled=True)


if __name__ == "__main__":
    unittest.main()
