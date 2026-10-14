"""test_prediction_config.py -- W0 预测配置契约的验收单测。

覆盖（方案 §9.1/§9.2/W0 退出门 G0 的配置面）：

* 缺省配置 = known_work/none/shadow/two_stage/consolidate/pure_four/
  ordered_shared_double_buffer；
* 七轴取值域 fail-closed（未知值/条件增强值 scenario、causal_lookahead、
  并行多源协议一律拒绝，不用静默别名）；
* A 对照（frozen_diagnostic）必须附带显式冻结规格，缺规格/规格字段不
  全 fail-closed；非 A 模式携带规格同样拒绝；
* §9.2 参数表校验（0<Lf<Ls、正 Delta/tau、alpha 派生不配置、直方图
  有界、预算正数、控制延迟零值披露 assumed-zero）；
* 文件加载：版本不匹配/未知键/未知参数键/坏 JSON/不可读路径 fail-closed；
* stage_policy 与 boundary_policy 同轴合一裁决（矛盾 fail-closed、一致
  通过）；
* cross_validate：J-off 时预测轴记 inactive 而不拒绝，共同底座轴仍适
  用；J-on 时全轴生效；
* 随仓 preset 文件全部可解析且臂身份正确。
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

from joint.prediction_config import (
    ACTION_SPACE_PURE_FOUR,
    ACTION_SPACE_SOURCE_SUBSET_HYBRID,
    ENV_JOINT_PREDICTION_CONFIG,
    FUTURE_DEMAND_NONE,
    FUTURE_DEMAND_RECENT,
    FUTURE_DEMAND_REVERTING,
    FrozenDiagnosticSpec,
    HISTORY_USE_SHADOW,
    HISTORY_USE_VALIDATED,
    JointPredictionConfig,
    MULTI_SOURCE_ORDERED_SHARED_DOUBLE_BUFFER,
    PREDICTION_CONFIG_SCHEMA_VERSION,
    PROJECTION_MODE_FROZEN_DIAGNOSTIC,
    PROJECTION_MODE_KNOWN_WORK,
    PredictionConfigError,
    PredictionParams,
    SETTLEMENT_CONSOLIDATE,
    SETTLEMENT_DISTRIBUTED,
    STAGE_POLICY_ONCE_MAPPING,
    STAGE_POLICY_TWO_STAGE,
    cross_validate,
    from_mapping,
    load_prediction_config,
    reconcile_stage_policy,
)

_HERE = Path(__file__).resolve().parent
_PRESETS = _HERE.parents[2] / "run_scripts" / "prediction_presets"


class DefaultConfigTest(unittest.TestCase):
    def test_incomplete_freeze_spec_uses_the_config_error_channel(self):
        with self.assertRaises(PredictionConfigError):
            from_mapping({
                "version": PREDICTION_CONFIG_SCHEMA_VERSION,
                "projection_mode": PROJECTION_MODE_FROZEN_DIAGNOSTIC,
                "frozen_diagnostic": {}})

    def test_invalid_parameter_units_and_non_finite_values_are_rejected(self):
        for params in (
                {"arrival_bucket_delta_ns": 0.5},
                {"forecast_template_budget": True},
                {"forecast_ordinal_threshold": float("inf")},
                {"host_decision_budget_seconds": float("nan")},
                {"control_latency_ns": float("nan")},
                {"control_latency_execution_deferred": True},
                {"forecast_distance_buckets_ns": [True, 100]},
                {"forecast_distance_buckets_ns": [1, 2, 3]},
                {"cold_start_tail_distribution": "unsupported"},
                {"cold_start_tail_distribution": None}):
            with self.subTest(params=params):
                with self.assertRaises(PredictionConfigError):
                    from_mapping({
                        "version": PREDICTION_CONFIG_SCHEMA_VERSION,
                        "params": params})

    def test_defaults_are_the_safe_base(self):
        cfg = JointPredictionConfig()
        self.assertEqual(cfg.projection_mode, PROJECTION_MODE_KNOWN_WORK)
        self.assertEqual(cfg.future_demand_mode, FUTURE_DEMAND_NONE)
        self.assertEqual(cfg.history_use, HISTORY_USE_SHADOW)
        self.assertEqual(cfg.stage_policy, STAGE_POLICY_TWO_STAGE)
        self.assertEqual(cfg.settlement_policy, SETTLEMENT_CONSOLIDATE)
        self.assertEqual(cfg.action_space, ACTION_SPACE_PURE_FOUR)
        self.assertEqual(
            cfg.multi_source_protocol,
            MULTI_SOURCE_ORDERED_SHARED_DOUBLE_BUFFER)
        self.assertIsNone(cfg.frozen_diagnostic)

    def test_env_unset_returns_default(self):
        cfg = load_prediction_config(env={})
        self.assertEqual(cfg.projection_mode, PROJECTION_MODE_KNOWN_WORK)
        self.assertEqual(dict(cfg.sources), {"*": "default"})

    def test_env_loads_file(self):
        payload = {
            "version": PREDICTION_CONFIG_SCHEMA_VERSION,
            "future_demand_mode": FUTURE_DEMAND_RECENT,
        }
        with tempfile.TemporaryDirectory(prefix="joint-prediction-config-") as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            cfg = load_prediction_config(
                env={ENV_JOINT_PREDICTION_CONFIG: str(path)})
            self.assertEqual(cfg.future_demand_mode, FUTURE_DEMAND_RECENT)
            self.assertEqual(cfg.sources.get("*"), str(path))

    def test_duplicate_json_keys_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="joint-prediction-config-") as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                '{"version": "' + PREDICTION_CONFIG_SCHEMA_VERSION +
                '", "future_demand_mode": "none", "future_demand_mode": "recent"}',
                encoding="utf-8")
            with self.assertRaises(PredictionConfigError):
                load_prediction_config(env={ENV_JOINT_PREDICTION_CONFIG: str(path)})

    def test_bad_env_values_fail_closed(self):
        with self.assertRaises(PredictionConfigError):
            load_prediction_config(env={ENV_JOINT_PREDICTION_CONFIG: " "})
        with self.assertRaises(PredictionConfigError):
            load_prediction_config(
                env={ENV_JOINT_PREDICTION_CONFIG: "/nonexistent/x.json"})

    def test_validation_evidence_rejects_non_json_nested_values(self):
        for value in ({1: 2}, {"eta": float("nan")}, {"eta": {1, 2}}):
            with self.subTest(value=value), self.assertRaises(PredictionConfigError):
                JointPredictionConfig(validation_registry={
                    "recent|near": {"params": value}})


class AxisDomainTest(unittest.TestCase):
    def test_unknown_projection_mode_rejected(self):
        with self.assertRaises(PredictionConfigError):
            JointPredictionConfig(projection_mode="oracle")

    def test_scenario_future_demand_not_yet_implemented(self):
        with self.assertRaises(PredictionConfigError):
            JointPredictionConfig(future_demand_mode="scenario")

    def test_causal_lookahead_not_yet_implemented(self):
        with self.assertRaises(PredictionConfigError):
            JointPredictionConfig(stage_policy="causal_lookahead")

    def test_parallel_multi_source_rejected_no_alias(self):
        with self.assertRaises(PredictionConfigError):
            JointPredictionConfig(multi_source_protocol="parallel_pairs")

    def test_history_use_domain(self):
        self.assertEqual(
            JointPredictionConfig(history_use=HISTORY_USE_VALIDATED)
            .history_use, HISTORY_USE_VALIDATED)
        with self.assertRaises(PredictionConfigError):
            JointPredictionConfig(history_use="auto")

    def test_settlement_and_action_space_domains(self):
        cfg = JointPredictionConfig(
            settlement_policy=SETTLEMENT_DISTRIBUTED,
            action_space=ACTION_SPACE_SOURCE_SUBSET_HYBRID)
        self.assertEqual(cfg.settlement_policy, SETTLEMENT_DISTRIBUTED)
        self.assertEqual(
            cfg.action_space, ACTION_SPACE_SOURCE_SUBSET_HYBRID)
        with self.assertRaises(PredictionConfigError):
            JointPredictionConfig(settlement_policy="hybrid_free")
        with self.assertRaises(PredictionConfigError):
            JointPredictionConfig(action_space="arbitrary_block_layout")


class FrozenDiagnosticSpecTest(unittest.TestCase):
    def _spec(self, **overrides):
        base = {
            "frozen_horizon_mode": "per_action",
            "resource_rule": "frozen resource rules",
            "service_estimate_rule": "frozen service estimates",
        }
        base.update(overrides)
        return FrozenDiagnosticSpec(**base)

    def test_requires_explicit_rules(self):
        with self.assertRaises(PredictionConfigError):
            JointPredictionConfig(
                projection_mode=PROJECTION_MODE_FROZEN_DIAGNOSTIC)
        with self.assertRaises(PredictionConfigError):
            JointPredictionConfig(
                projection_mode=PROJECTION_MODE_FROZEN_DIAGNOSTIC,
                frozen_diagnostic=self._spec(resource_rule=""))
        with self.assertRaises(PredictionConfigError):
            JointPredictionConfig(
                projection_mode=PROJECTION_MODE_FROZEN_DIAGNOSTIC,
                frozen_diagnostic=self._spec(service_estimate_rule=""))

    def test_horizon_ns_mode_requires_positive_ns(self):
        with self.assertRaises(PredictionConfigError):
            FrozenDiagnosticSpec(
                frozen_horizon_mode="horizon_ns",
                resource_rule="r", service_estimate_rule="s")
        with self.assertRaises(PredictionConfigError):
            FrozenDiagnosticSpec(
                frozen_horizon_mode="horizon_ns", frozen_horizon_ns=0,
                resource_rule="r", service_estimate_rule="s")

    def test_per_action_rejects_explicit_ns(self):
        with self.assertRaises(PredictionConfigError):
            FrozenDiagnosticSpec(
                frozen_horizon_mode="per_action", frozen_horizon_ns=5,
                resource_rule="r", service_estimate_rule="s")

    def test_spec_only_valid_for_frozen_mode(self):
        with self.assertRaises(PredictionConfigError):
            JointPredictionConfig(
                projection_mode=PROJECTION_MODE_KNOWN_WORK,
                frozen_diagnostic=self._spec())


class PredictionParamsTest(unittest.TestCase):
    def test_half_life_ordering_enforced(self):
        with self.assertRaises(PredictionConfigError):
            PredictionParams(
                arrival_half_life_fast_ns=1_000,
                arrival_half_life_slow_ns=999)
        with self.assertRaises(PredictionConfigError):
            PredictionParams(arrival_half_life_fast_ns=0)

    def test_alphas_derived_not_configured(self):
        params = PredictionParams()
        delta, lf, ls = (
            params.arrival_bucket_delta_ns,
            params.arrival_half_life_fast_ns,
            params.arrival_half_life_slow_ns)
        self.assertAlmostEqual(
            params.arrival_alpha_fast, 1.0 - 2.0 ** (-delta / lf))
        self.assertAlmostEqual(
            params.arrival_alpha_slow, 1.0 - 2.0 ** (-delta / ls))
        self.assertGreater(
            params.arrival_alpha_fast, params.arrival_alpha_slow)

    def test_positive_domains(self):
        with self.assertRaises(PredictionConfigError):
            PredictionParams(arrival_bucket_delta_ns=0)
        with self.assertRaises(PredictionConfigError):
            PredictionParams(revert_tau_ns=-1)
        with self.assertRaises(PredictionConfigError):
            PredictionParams(cold_start_forward_steps=0)
        with self.assertRaises(PredictionConfigError):
            PredictionParams(length_histogram_max_buckets=0)
        with self.assertRaises(PredictionConfigError):
            PredictionParams(history_min_mature_samples=0)
        with self.assertRaises(PredictionConfigError):
            PredictionParams(scenario_budget=0)
        # W11 死配置面收口：scenario 未实现，>1 配置即拒（不静默死值）。
        with self.assertRaises(PredictionConfigError):
            PredictionParams(scenario_budget=2)
        with self.assertRaises(PredictionConfigError):
            PredictionParams(projection_event_budget=0)
        with self.assertRaises(PredictionConfigError):
            PredictionParams(control_latency_ns=-1)

    def test_zero_control_latency_is_disclosed_assumption(self):
        manifest = PredictionParams().params_manifest()
        self.assertEqual(manifest["control_latency_ns"], 0)
        self.assertEqual(manifest["control_latency_source"],
                         "assumed-zero")
        self.assertNotIn("control_latency_execution_deferred", manifest)

    def test_nonzero_control_latency_requires_real_execution(self):
        """尚无延迟提交执行时，只允许明确披露的零延迟假设。"""
        with self.assertRaises(PredictionConfigError):
            PredictionParams(control_latency_ns=500)
        with self.assertRaises(PredictionConfigError):
            from_mapping({
                "version": PREDICTION_CONFIG_SCHEMA_VERSION,
                "params": {"control_latency_ns": 500,
                           "control_latency_execution_deferred": True}})

    def test_ewma_weight_domain(self):
        with self.assertRaises(PredictionConfigError):
            PredictionParams(service_efficiency_ewma_weight=1.5)
        with self.assertRaises(PredictionConfigError):
            PredictionParams(service_efficiency_init=0.0)

    def test_hybrid_budget_at_least_two(self):
        with self.assertRaises(PredictionConfigError):
            PredictionParams(hybrid_max_remote_sources=1)


class FileLoadingTest(unittest.TestCase):
    def _payload(self, **overrides):
        base = {"version": PREDICTION_CONFIG_SCHEMA_VERSION}
        base.update(overrides)
        return base

    def test_version_mismatch_fails_closed(self):
        with self.assertRaises(PredictionConfigError):
            from_mapping(
                {"version": "predcfg-v0-19990101",
                 "projection_mode": PROJECTION_MODE_KNOWN_WORK})

    def test_missing_version_fails_closed(self):
        with self.assertRaises(PredictionConfigError):
            from_mapping({"projection_mode": PROJECTION_MODE_KNOWN_WORK})

    def test_unknown_top_level_key_rejected(self):
        with self.assertRaises(PredictionConfigError):
            from_mapping(self._payload(unknown_axis="x"))

    def test_unknown_param_key_rejected(self):
        with self.assertRaises(PredictionConfigError):
            from_mapping(self._payload(params={"universal_window_s": 5}))


class StagePolicyReconcileTest(unittest.TestCase):
    def test_consistent_pairs_pass(self):
        self.assertEqual(
            reconcile_stage_policy("off", STAGE_POLICY_TWO_STAGE),
            STAGE_POLICY_TWO_STAGE)
        self.assertEqual(
            reconcile_stage_policy(
                "p-continuation", STAGE_POLICY_ONCE_MAPPING),
            STAGE_POLICY_ONCE_MAPPING)

    def test_conflicting_pairs_fail_closed(self):
        with self.assertRaises(PredictionConfigError):
            reconcile_stage_policy("off", STAGE_POLICY_ONCE_MAPPING)
        with self.assertRaises(PredictionConfigError):
            reconcile_stage_policy("p-continuation", STAGE_POLICY_TWO_STAGE)


class _MechLike:
    def __init__(self, scheduler_mode, boundary_policy="off"):
        self.scheduler_mode = scheduler_mode
        self.boundary_policy = boundary_policy


class CrossValidateTest(unittest.TestCase):
    def test_j_on_all_axes_active(self):
        disclosure = cross_validate(
            _MechLike("joint"), JointPredictionConfig())
        self.assertTrue(disclosure["j_pricing_active"])
        self.assertEqual(disclosure["inactive_axes_under_j_off"], [])

    def test_j_off_records_inactive_not_rejected(self):
        disclosure = cross_validate(
            _MechLike("load-first"), JointPredictionConfig())
        self.assertFalse(disclosure["j_pricing_active"])
        self.assertEqual(
            disclosure["inactive_axes_under_j_off"],
            ["projection_mode", "future_demand_mode", "history_use"])
        self.assertIn("inactive_note", disclosure)

    def test_boundary_conflict_fails_closed(self):
        with self.assertRaises(PredictionConfigError):
            cross_validate(
                _MechLike("joint", boundary_policy="p-continuation"),
                JointPredictionConfig())

    def test_requires_mech_like_object(self):
        with self.assertRaises(PredictionConfigError):
            cross_validate(object(), JointPredictionConfig())


class ShippedPresetsTest(unittest.TestCase):
    def test_preset_files_parse_and_identify_arms(self):
        expected = {
            "default.json": (
                PROJECTION_MODE_KNOWN_WORK, FUTURE_DEMAND_NONE,
                SETTLEMENT_CONSOLIDATE, ACTION_SPACE_PURE_FOUR),
            "arm_a_frozen.json": (
                PROJECTION_MODE_FROZEN_DIAGNOSTIC, FUTURE_DEMAND_NONE,
                SETTLEMENT_CONSOLIDATE, ACTION_SPACE_PURE_FOUR),
            "arm_c_recent.json": (
                PROJECTION_MODE_KNOWN_WORK, FUTURE_DEMAND_RECENT,
                SETTLEMENT_CONSOLIDATE, ACTION_SPACE_PURE_FOUR),
            "arm_d_reverting.json": (
                PROJECTION_MODE_KNOWN_WORK, FUTURE_DEMAND_REVERTING,
                SETTLEMENT_CONSOLIDATE, ACTION_SPACE_PURE_FOUR),
            "distributed.json": (
                PROJECTION_MODE_KNOWN_WORK, FUTURE_DEMAND_NONE,
                SETTLEMENT_DISTRIBUTED, ACTION_SPACE_PURE_FOUR),
            "hybrid.json": (
                PROJECTION_MODE_KNOWN_WORK, FUTURE_DEMAND_NONE,
                SETTLEMENT_CONSOLIDATE, ACTION_SPACE_SOURCE_SUBSET_HYBRID),
        }
        for name, wanted in expected.items():
            payload = json.loads(
                (_PRESETS / name).read_text(encoding="utf-8"))
            cfg = from_mapping(payload)
            got = (cfg.projection_mode, cfg.future_demand_mode,
                   cfg.settlement_policy, cfg.action_space)
            self.assertEqual(got, wanted, name)
            # manifest 披露完整性（回退后的实际模式 + 版本）。
            manifest = cfg.manifest_dict()
            self.assertEqual(
                manifest["prediction_config_schema_version"],
                PREDICTION_CONFIG_SCHEMA_VERSION)
            self.assertIn("params", manifest)


if __name__ == "__main__":
    unittest.main()
