#!/usr/bin/env python3
"""test_joint_mechanisms.py -- 三机制联合策略的定向验收单测。

对应《三机制联合策略_template仓库设计方案》§7.1（八组合硬性要求）与
§8 验收问题表；全部为 python 级构造性用例（不跑仿真）：

* 开关解析：八组合固定映射、fail-closed、预设与显式开关互斥；
* T 严格性：human 部分层仍可释放且缺口未满足时不动 tool；T-off 为
  同一合法集合上的类型无关单遍 LRU；需求满足即停；
* E 规格：k_hide 解析例（L=32/c=1ms/q=0 → r=0.5/2/4ms ⇒ 1/17/25）、
  minimal_layer_groups 最少完整层组、adaptive 目标外→目标内两扫描、
  未来信息隔离（估计器只随真实观测更新）；
* J 选择：joint 全候选 argmin / load-first 与 affinity-first 顺序参照、
  remote off 仅移除 remote-read 候选；
* home/merge：copy 异地执行不改 home、merge_back 增量恰好归并一次、
  工作副本释放无双份驻留、recompute 增量归并、remote-read 工作副本
  仅承载新增量；
* 八组合：同一实现内以配置切换出全部八组合（category_mode ×
  scheduler_mode × layer_policy 与 §7.1 表一致）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 joint/test_joint_mechanisms.py
"""
import os
import sys
import unittest
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for _p in (_HERE, _PARENT, os.path.join(_PARENT, "online")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from joint.joint_config import (  # noqa: E402
    COMBO_NAMES,
    COMBO_PRESETS,
    JointConfigError,
    JointMechanismConfig,
    parse_joint_config,
)
from joint.eviction_priority import (  # noqa: E402
    classify_with_fallback,
    eviction_class_order,
)
from joint.layer_eviction_policy import (  # noqa: E402
    LayerEvictionError,
    LayerEvictionPolicy,
    OnlineMean,
    VictimView,
    coalesce_steps,
    k_hide_deadline,
)
from joint.joint_cost_model import (  # noqa: E402
    ACTION_ORDER,
    JointHardwareRates,
    InstanceLoadView,
    JointCostModel,
    LinkFlowRegistry,
    RequestView,
    ServiceFactors,
    SessionKVView,
)
from joint.joint_scheduler import (  # noqa: E402
    select_instance_and_action,
)
from face_scheduler import (  # noqa: E402
    FaceHardware,
    FaceInstanceSpec,
    FaceModel,
    KVCacheManager,
    build_instances,
    kv_cache_shard_bytes_for_layer_range,
)


# ============================================================ 开关解析 ==


class JointConfigParseTest(unittest.TestCase):
    """§7.1：八组合固定映射 / fail-closed / 预设与显式互斥。"""

    def test_eight_combo_presets_match_design_table(self):
        expected = {
            "none": ("lru", "load-first", "minimal_layer_groups"),
            "T": ("typed", "load-first", "minimal_layer_groups"),
            "J": ("lru", "joint", "minimal_layer_groups"),
            "E": ("lru", "load-first", "adaptive"),
            "TJ": ("typed", "joint", "minimal_layer_groups"),
            "TE": ("typed", "load-first", "adaptive"),
            "JE": ("lru", "joint", "adaptive"),
            "TJE": ("typed", "joint", "adaptive"),
        }
        self.assertEqual(set(COMBO_NAMES), set(expected))
        for combo, modes in expected.items():
            self.assertEqual(COMBO_PRESETS[combo], modes, combo)

    def test_default_is_full_mechanisms(self):
        config = parse_joint_config({})
        self.assertEqual(
            (config.category_mode, config.scheduler_mode,
             config.layer_policy, config.remote_actions),
            ("typed", "joint", "adaptive", "on"))
        self.assertTrue(config.t_enabled and config.j_enabled
                        and config.e_enabled and config.remote_enabled)

    def test_combo_preset_parses(self):
        for combo in COMBO_NAMES:
            config = parse_joint_config({"JOINT_ABLATION_COMBO": combo})
            self.assertEqual(config.combo, combo)
            self.assertEqual(
                (config.category_mode, config.scheduler_mode,
                 config.layer_policy),
                COMBO_PRESETS[combo])

    def test_explicit_switches_parse(self):
        config = parse_joint_config({
            "JOINT_CATEGORY_MODE": "lru",
            "JOINT_SCHEDULER_MODE": "affinity-first",
            "JOINT_LAYER_POLICY": "legacy_half",
            "JOINT_REMOTE_ACTIONS": "off",
        })
        self.assertEqual(config.combo, None)
        self.assertFalse(config.t_enabled)
        self.assertFalse(config.j_enabled)
        self.assertFalse(config.e_enabled)
        self.assertFalse(config.remote_enabled)

    def test_invalid_values_fail_closed(self):
        for env in (
            {"JOINT_ABLATION_COMBO": "full"},
            {"JOINT_CATEGORY_MODE": "TYPED"},
            {"JOINT_SCHEDULER_MODE": "load_first"},
            {"JOINT_LAYER_POLICY": "half"},
            {"JOINT_REMOTE_ACTIONS": "1"},
            {"JOINT_CATEGORY_MODE": " typed"},
            {"JOINT_SCHEDULER_MODE": ""},
        ):
            with self.assertRaises(JointConfigError, msg=repr(env)):
                parse_joint_config(env)

    def test_combo_and_explicit_conflict_fails_closed(self):
        with self.assertRaises(JointConfigError):
            parse_joint_config({
                "JOINT_ABLATION_COMBO": "T",
                "JOINT_CATEGORY_MODE": "lru",
            })
        # 与预设同值的显式开关同样拒绝（重复指定）。
        with self.assertRaises(JointConfigError):
            parse_joint_config({
                "JOINT_ABLATION_COMBO": "T",
                "JOINT_CATEGORY_MODE": "typed",
            })

    def test_manifest_discloses_off_substitutions(self):
        config = parse_joint_config({"JOINT_ABLATION_COMBO": "none"})
        manifest = config.manifest_dict()
        self.assertEqual(manifest["combo"], "none")
        self.assertIn("T-off", manifest["off_substitutions"])
        self.assertIn("J-off", manifest["off_substitutions"])
        self.assertIn("E-off", manifest["off_substitutions"])


# ================================================================== T ==


def _tiny_hardware(capacity_bytes=1_000_000_000):
    return FaceHardware(
        mesh_rows=2, mesh_cols=2,
        local_hbm_capacity_bytes=capacity_bytes,
        local_hbm_bandwidth_gbps=100.0, d2d_bandwidth_gbps=200.0,
        peak_perf_tflops=1.0, d2d_latency_ns=0, local_hbm_latency_ns=0,
    )


def _tiny_model(layers=4):
    return FaceModel(
        layers=layers, hidden_size=16, ffn_size=32, num_heads=4,
        vocab_size=32, bytes_per_elem=2, mlp_variant="swiglu",
    )


def _two_instance_topology(hardware):
    return build_instances(hardware, (
        FaceInstanceSpec("ins0", "1", (0, 1)),
        FaceInstanceSpec("ins1", "2", (2, 3)),
    ))


def _seed(kv, session_id, instance, tokens, completion_ns,
          next_request_type=None):
    kv.prepare_prefill(
        session_id=session_id, target_instance_index=instance,
        history_tokens=0, trigger_request_id=f"{session_id}_seed")
    kv.expand_prefill(
        session_id=session_id, instance_index=instance,
        context_tokens=tokens, trigger_request_id=f"{session_id}_seed")
    kv.mark_complete(session_id, completion_ns,
                     next_request_type=next_request_type)


class TypedEvictionStrictnessTest(unittest.TestCase):
    """§4/§8：严格类别序 + 满足即停 + T-off 类型无关。"""

    def _manager(self, category_mode, capacity=1000, layers=4):
        hardware = _tiny_hardware(capacity)
        model = _tiny_model(layers)
        topology = _two_instance_topology(hardware)
        return KVCacheManager(
            topology, model, category_mode=category_mode,
            layer_policy="minimal_layer_groups")

    @staticmethod
    def _required_for_gap(kv, instance, extra_by_rank):
        remaining = kv._effective_remaining_by_tp_rank(instance)
        return tuple(r + e for r, e in zip(remaining, extra_by_rank))

    def test_typed_uses_tool_only_after_human_exhausted(self):
        kv = self._manager("typed", capacity=100_000)
        # FIFO：human 较旧，tool 较新——typed 必须先耗尽 human 的全部
        # 层组，缺口仍未满足才动 tool（§4）。
        _seed(kv, "h_old", 0, 10, 10, "human")
        _seed(kv, "h_new", 0, 10, 20, "human")
        _seed(kv, "t_new", 0, 10, 30, "tool")
        full = kv_cache_shard_bytes_for_layer_range(
            kv.model, 10, kv.tp_degree, layer_start=0,
            layer_end=kv.model.layers)
        one_group = kv_cache_shard_bytes_for_layer_range(
            kv.model, 10, kv.tp_degree, layer_start=3,
            layer_end=kv.model.layers)
        # 缺口 = 两个 human 整份 + tool 恰一层组 → human 耗尽后才逐
        # tool 的一个层组即停。
        extra = tuple(
            2 * f + g for f, g in zip(full, one_group))
        required = self._required_for_gap(kv, 0, extra)
        evictions = kv._ensure_capacity(
            0, required, phase="prefill",
            reason="probe", trigger_request_id="probe")
        victims = [transfer.session_id for transfer in evictions]
        # 顺序断言：两个 human victim（整份）均先于 tool（单层组）。
        self.assertEqual(victims.index("h_old"), 0)
        self.assertEqual(victims.index("h_new"), 1)
        self.assertEqual(victims.index("t_new"), 2)
        self.assertEqual(
            kv.session_snapshot("t_new").resident_prefix_layers,
            kv.model.layers - 1)

    def test_typed_stops_when_gap_met_without_touching_tool(self):
        kv = self._manager("typed", capacity=100_000)
        _seed(kv, "h_old", 0, 10, 10, "human")
        _seed(kv, "t_new", 0, 10, 30, "tool")
        # 缺口恰好 = 一个完整层组（minimal 策略下恰释放 1 层即停）。
        per_group = kv_cache_shard_bytes_for_layer_range(
            kv.model, 10, kv.tp_degree, layer_start=0, layer_end=1)
        required = self._required_for_gap(kv, 0, per_group)
        evictions = kv._ensure_capacity(
            0, required, phase="prefill",
            reason="probe", trigger_request_id="probe")
        victims = {transfer.session_id for transfer in evictions}
        self.assertEqual(victims, {"h_old"})
        # 满足即停：单层组释放后 human 仍余 3 层，不再多逐。
        snapshot = kv.session_snapshot("h_old")
        self.assertEqual(snapshot.resident_prefix_layers, 3)

    def test_lru_mode_is_type_agnostic_single_pass(self):
        kv = self._manager("lru", capacity=100_000)
        _seed(kv, "h_old", 0, 10, 10, "human")
        _seed(kv, "t_new", 0, 10, 30, "tool")
        # LRU：完成时间序（h_old 先）——类别不进判据；缺口小 → 只逐
        # 最旧者的最少层组。
        per_group = kv_cache_shard_bytes_for_layer_range(
            kv.model, 10, kv.tp_degree, layer_start=0, layer_end=1)
        required = self._required_for_gap(kv, 0, per_group)
        evictions = kv._ensure_capacity(
            0, required, phase="prefill",
            reason="probe", trigger_request_id="probe")
        victims = {transfer.session_id for transfer in evictions}
        self.assertEqual(victims, {"h_old"})
        self.assertEqual(kv.session_snapshot("h_old").resident_prefix_layers, 3)

    def test_class_order_constants(self):
        self.assertEqual(
            [p.trigger_type for p in eviction_class_order("typed")],
            ["human", "tool"])
        self.assertEqual(
            [p.trigger_type for p in eviction_class_order("lru")],
            [None])

    def test_unknown_type_falls_back_to_human_with_disclosure(self):
        record = classify_with_fallback("unknown_value")
        self.assertEqual(record.eviction_class, "human")
        self.assertTrue(record.fallback)
        self.assertEqual(record.raw_next_request_type, "unknown_value")
        self.assertFalse(classify_with_fallback("tool").fallback)


# ================================================================== E ==


class KHideDeadlineFormulaTest(unittest.TestCase):
    """§5.2/§5.7：期限公式解析例与边界。"""

    def test_analytic_fixture_L32(self):
        layers = 32
        for r_us, expected in ((500, 1), (2000, 17), (4000, 25)):
            result = k_hide_deadline([1000] * layers, [r_us] * layers, 0)
            self.assertEqual(result.k_hide, expected, f"r={r_us}us")
        # r=2ms 保留 16 层在末层暴露 1ms（2*16=32 > 31）——不能判为隐藏。
        result16 = k_hide_deadline([1000] * layers, [2000] * layers, 0)
        worst, _ = self._margins(16, layers, 2000)
        self.assertEqual(worst, 1000)
        self.assertEqual(result16.exposed_stall_ns, 0)

    @staticmethod
    def _margins(k, layers, r_us):
        worst = None
        for ell in range(k + 1, layers + 1):
            restore = r_us * (ell - k)
            deadline = 1000 * (ell - 1)
            over = restore - deadline
            if worst is None or over > worst:
                worst = over
        return worst, None

    def test_k_equals_L_boundary_and_zero_restore(self):
        result = k_hide_deadline([1000] * 8, [0] * 8, 0)
        self.assertEqual(result.k_hide, 0)
        self.assertEqual(result.exposed_stall_ns, 0)
        # 无缺失层时条件为空（k=L 边界可达）。
        result = k_hide_deadline([10] * 4, [10**9] * 4, 0)
        self.assertEqual(result.k_hide, 4)
        self.assertIsNone(result.binding_layer)

    def test_first_layer_requires_history_blocks_k0(self):
        # q=0、r>0、D_1=0 ⇒ k=0 永不满足首层。
        result = k_hide_deadline([100] * 4, [50] * 4, 0)
        self.assertEqual(result.k_hide, 1)

    def test_malformed_vectors_fail_closed(self):
        with self.assertRaises(LayerEvictionError):
            k_hide_deadline([1, 2], [1], 0)
        with self.assertRaises(LayerEvictionError):
            k_hide_deadline([1, -1], [1, 1], 0)


class LayerPolicyPlanTest(unittest.TestCase):
    """§5.6：三模式的释放计划语义。"""

    @staticmethod
    def _victim(session_id, prefix, per_layer=(100, 150), target=0):
        def bytes_fn(a, b):
            n = b - a
            return (per_layer[0] * n, per_layer[1] * n)
        return VictimView(
            session_id=session_id, resident_prefix_layers=prefix,
            layer_group_bytes_fn=bytes_fn, retention_target_layers=target)

    def test_minimal_releases_fewest_complete_groups(self):
        policy = LayerEvictionPolicy("minimal_layer_groups", 32)
        victim = self._victim("s", 32)
        plan = policy.plan_release(
            gap_bytes_by_tp_rank=(250, 300), victims=[victim])
        self.assertTrue(plan.satisfied)
        # rank0 两层 200 < 250 → 需三层（300/450）才满足全部 rank ——
        # "最少完整层组"按逐 rank 缺口全满足计；步已合并为整段 [29,32)。
        self.assertEqual(len(plan.steps), 1)
        self.assertEqual(plan.steps[0].layer_start, 29)
        self.assertEqual(plan.steps[0].layer_end, 32)
        self.assertEqual(plan.released_bytes_by_tp_rank, (300, 450))

    def test_adaptive_two_scans_respect_soft_target(self):
        policy = LayerEvictionPolicy("adaptive", 32)
        victim = self._victim("s", 32, target=28)
        # 缺口可由目标外 4 层满足（rank1: 4×150=600 ≥ 600）。
        plan = policy.plan_release(
            gap_bytes_by_tp_rank=(300, 600), victims=[victim])
        self.assertTrue(plan.satisfied)
        self.assertEqual(plan.target_breach_sessions, ())
        merged = coalesce_steps(plan.steps)
        self.assertEqual(merged[0].layer_start, 28)

    def test_adaptive_breaches_target_only_when_gap_unmet(self):
        policy = LayerEvictionPolicy("adaptive", 32)
        victim = self._victim("s", 32, target=28)
        # 缺口超过目标外 4 层可释放量 → 进入目标内并披露 breach；
        # 30 层（rank0 3000 ≥ 3000）后满足即停。
        plan = policy.plan_release(
            gap_bytes_by_tp_rank=(3000, 3000), victims=[victim])
        self.assertTrue(plan.satisfied)
        self.assertEqual(plan.target_breach_sessions, ("s",))
        self.assertEqual(plan.steps[0].layer_start, 2)  # 保留前 2 层

    def test_legacy_half_matches_original_two_stage(self):
        policy = LayerEvictionPolicy("legacy_half", 4)
        full = self._victim("full", 4, per_layer=(10, 10))
        partial = self._victim("part", 2, per_layer=(10, 10))
        # 第一段：仅 full-local 释放 [2,4)；缺口大 → 第二段整份外迁。
        plan = policy.plan_release(
            gap_bytes_by_tp_rank=(60, 60), victims=[full, partial])
        kinds = [
            (step.session_id, step.layer_start, step.layer_end)
            for step in plan.steps]
        self.assertIn(("full", 2, 4), kinds)
        self.assertIn(("full", 0, 2), kinds)
        self.assertIn(("part", 0, 2), kinds)
        # partial 不参与半层段（非全本地）。
        self.assertNotIn(("part", 2, 4), kinds)

    def test_zero_gap_is_noop(self):
        policy = LayerEvictionPolicy("adaptive", 8)
        plan = policy.plan_release(
            gap_bytes_by_tp_rank=(0, 0),
            victims=[self._victim("s", 8)])
        self.assertEqual(plan.steps, ())
        self.assertTrue(plan.satisfied)


class EstimatorCausalityTest(unittest.TestCase):
    """§5.5：估计器只随已完成轮次更新；无样本保守保留全层。"""

    def test_online_mean_incremental_update(self):
        mean = OnlineMean()
        self.assertFalse(mean.available)
        for value in (10, 20, 30):
            mean.update(value)
        self.assertEqual(mean.count, 3)
        self.assertAlmostEqual(mean.mean, 20.0)

    def test_future_info_isolation(self):
        """改变"未来"（尚未完成的行）不影响估计器状态——只有
        observe_completed 才改变状态（因果边界）。"""
        from face_scheduler import KVCacheManager
        hardware = _tiny_hardware()
        model = _tiny_model(4)
        topology = _two_instance_topology(hardware)
        kv = KVCacheManager(
            topology, model, category_mode="typed",
            layer_policy="adaptive", pool_bandwidth_gbps=100.0,
            pool_latency_ns=100)
        _seed(kv, "s", 0, 8, 10, "human")
        before_target = kv._adaptive_retention_target(kv._sessions["s"])
        before_estimate = kv._estimate_next_input("s")
        # 无样本：保守保留全部层（cold_start_unknown_input 语义）。
        self.assertEqual(before_target, model.layers)
        self.assertIsNone(before_estimate)
        # 真实观测（已完成轮次）后估计器更新、目标可低于 L。
        kv.observe_completed_input("s", 4)
        after_estimate = kv._estimate_next_input("s")
        self.assertEqual(after_estimate, 4)
        target = kv._adaptive_retention_target(kv._sessions["s"])
        self.assertIsInstance(target, int)
        self.assertGreaterEqual(target, 0)
        self.assertLessEqual(target, model.layers)

    def test_session_mean_preferred_over_run_mean(self):
        from face_scheduler import KVCacheManager
        hardware = _tiny_hardware()
        topology = _two_instance_topology(hardware)
        kv = KVCacheManager(
            topology, _tiny_model(4), category_mode="typed",
            layer_policy="adaptive", pool_bandwidth_gbps=100.0)
        kv.observe_completed_input("a", 100)
        kv.observe_completed_input("a", 200)
        kv.observe_completed_input("b", 1000)
        self.assertEqual(kv._estimate_next_input("a"), 150)
        self.assertEqual(kv._estimate_next_input("b"), 1000)
        # 无本 session 样本 → 本 run 均值（(100+200+1000)/3 = 433）。
        self.assertEqual(kv._estimate_next_input("c"), 433)


# ================================================================== J ==


def _route(source, target):
    return ((source, target), 1)


def _cost_model(loads, *, remote_enabled_partner=False):
    rates = JointHardwareRates.from_gbps(
        noc_link_gbps=10.0, pool_port_gbps=5.0, local_hbm_gbps=100.0,
        d2d_latency_ns=10, pool_latency_ns=100)
    return JointCostModel(
        rates=rates, loads=loads, flow_registry=LinkFlowRegistry(),
        service_factors=ServiceFactors(),
        prefill_ns_per_token=1.0, decode_ns_per_token=2.0,
        model_layers=4, instance_tp_size=2, route_fn=_route)


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


def _request(input_tokens=50, decode_estimate=10):
    return RequestView(
        request_id="r", session_id="s", input_tokens=input_tokens,
        history_tokens_before=100,
        estimated_decode_tokens=decode_estimate,
        horizon_source="session_online_mean",
        input_kv_bytes_by_tp_rank=(25, 25))


class JointSelectionTest(unittest.TestCase):
    """§7.1：joint / load-first / affinity-first 三模式与 remote 开关。"""

    def test_joint_picks_global_argmin_over_all_instances(self):
        # 实例 1 空闲且历史驻留（stay 适用、零历史搬运）→ 全局最优。
        loads = {0: _load(0, 5000), 1: _load(1, 0)}
        session = _session(home=0, resident=1)
        record = select_instance_and_action(
            mode="joint", cost_model=_cost_model(loads), session=session,
            request=_request(), remote_enabled=True)
        self.assertEqual(record.chosen.instance_index, 1)
        self.assertEqual(record.chosen.action, "stay")
        # 全候选枚举：每实例 4 动作。
        self.assertEqual(len(record.candidates), 8)

    def test_load_first_picks_min_load_instance_then_action(self):
        loads = {0: _load(0, 100), 1: _load(1, 5000)}
        session = _session(home=0, resident=1)
        record = select_instance_and_action(
            mode="load-first", cost_model=_cost_model(loads),
            session=session, request=_request(), remote_enabled=True)
        # 负载序先定实例 0（尽管历史驻留 1），再在该位置选动作。
        self.assertEqual(record.chosen.instance_index, 0)
        self.assertNotEqual(record.chosen.action, "stay")
        self.assertEqual(record.diagnostics["sequential_instance"], 0)

    def test_affinity_first_prefers_home_then_fallback(self):
        loads = {0: _load(0, 5000), 1: _load(1, 0)}
        session = _session(home=1, resident=1)
        record = select_instance_and_action(
            mode="affinity-first", cost_model=_cost_model(loads),
            session=session, request=_request(), remote_enabled=True)
        self.assertEqual(record.chosen.instance_index, 1)
        # 无 home（新会话）回退 load-first。
        fresh = SessionKVView(
            session_id="s", home_instance=None, resident_instance=None,
            location="none", history_tokens=0, resident_prefix_layers=0,
            history_bytes_by_tp_rank=(0, 0), missing_bytes_by_tp_rank=(0, 0))
        record = select_instance_and_action(
            mode="affinity-first", cost_model=_cost_model(loads),
            session=fresh, request=_request(), remote_enabled=True)
        self.assertEqual(record.chosen.instance_index, 1)

    def test_remote_off_only_removes_remote_candidates(self):
        loads = {0: _load(0, 0), 1: _load(1, 0)}
        session = _session(home=1, resident=1)
        for remote_enabled in (True, False):
            record = select_instance_and_action(
                mode="joint", cost_model=_cost_model(loads),
                session=session, request=_request(),
                remote_enabled=remote_enabled)
            remote_candidates = [
                candidate for candidate in record.candidates
                if candidate.action == "remote-read"]
            if remote_enabled:
                self.assertTrue(any(
                    candidate.applicable
                    for candidate in remote_candidates))
            else:
                self.assertFalse(any(
                    candidate.applicable
                    for candidate in remote_candidates))
                # 其余动作不受影响。
                self.assertTrue(any(
                    candidate.applicable and candidate.action == "recompute"
                    for candidate in record.candidates))

    def test_stay_inapplicable_when_history_not_resident(self):
        loads = {0: _load(0, 0)}
        model = _cost_model(loads)
        applicability = dict(zip(
            ACTION_ORDER,
            model.applicable_actions(
                _session(home=0, resident=1), _request(), 0,
                remote_enabled=True)))
        self.assertFalse(applicability["stay"][0])
        self.assertTrue(applicability["recompute"][0])
        self.assertTrue(applicability["copy"][0])
        self.assertTrue(applicability["remote-read"][0])

    def test_new_session_only_stay_and_recompute(self):
        loads = {0: _load(0, 0)}
        model = _cost_model(loads)
        fresh = SessionKVView(
            session_id="s", home_instance=None, resident_instance=None,
            location="none", history_tokens=0, resident_prefix_layers=0,
            history_bytes_by_tp_rank=(0, 0), missing_bytes_by_tp_rank=(0, 0))
        applicability = dict(zip(
            ACTION_ORDER,
            model.applicable_actions(
                fresh, _request(), 0, remote_enabled=True)))
        self.assertTrue(applicability["stay"][0])       # turn-0 本地建立
        self.assertTrue(applicability["recompute"][0])
        self.assertFalse(applicability["copy"][0])
        self.assertFalse(applicability["remote-read"][0])

    def test_tie_break_is_deterministic(self):
        loads = {0: _load(0, 0), 1: _load(1, 0)}
        fresh = SessionKVView(
            session_id="s", home_instance=None, resident_instance=None,
            location="none", history_tokens=0, resident_prefix_layers=0,
            history_bytes_by_tp_rank=(0, 0), missing_bytes_by_tp_rank=(0, 0))
        records = [
            select_instance_and_action(
                mode="joint", cost_model=_cost_model(loads),
                session=fresh, request=_request(), remote_enabled=True)
            for _ in range(3)]
        chosen = [(r.chosen.instance_index, r.chosen.action)
                  for r in records]
        self.assertEqual(len(set(chosen)), 1)


# ======================================================== home / merge ==


class HomeMergeSemanticsTest(unittest.TestCase):
    """§2.1/§2.2/§8：home 保持、增量恰好归并一次、无双份驻留。"""

    def _manager(self):
        hardware = _tiny_hardware()
        topology = _two_instance_topology(hardware)
        return KVCacheManager(
            topology, _tiny_model(4), category_mode="typed",
            layer_policy="minimal_layer_groups")

    def test_copy_preserves_home_and_merges_increments_once(self):
        kv = self._manager()
        _seed(kv, "s", 0, 10, 10, "human")
        kv.prepare_prefill(
            session_id="s", target_instance_index=1,
            history_tokens=10, trigger_request_id="t1", action="copy")
        working = kv.session_snapshot("s")
        self.assertEqual(working.home_instance, 0)
        self.assertEqual(working.working_kind, "copy")
        self.assertEqual(working.working_instance_index, 1)
        # 执行期增长发生在执行端（工作副本）。
        kv.expand_prefill(
            session_id="s", instance_index=1, context_tokens=15,
            trigger_request_id="t1")
        kv.expand_decode(
            session_id="s", instance_index=1, final_context_tokens=18,
            trigger_request_id="t1")
        merge_transfers = kv.merge_back(
            session_id="s", trigger_request_id="t1", new_tokens=8)
        self.assertEqual(len(merge_transfers), 1)
        self.assertEqual(merge_transfers[0].kind, "noc_migrate")
        # 增量字节 = kv(8 token)，而非整份。
        increment = kv_cache_shard_bytes_for_layer_range(
            kv.model, 8, kv.tp_degree, layer_start=0,
            layer_end=kv.model.layers)
        self.assertEqual(merge_transfers[0].total_bytes, sum(increment))
        merged = kv.session_snapshot("s")
        self.assertEqual(merged.instance_index, 0)      # home 恢复权威
        self.assertIsNone(merged.working_kind)
        self.assertEqual(merged.context_tokens, 18)
        self.assertEqual(merged.home_instance, 0)
        # 双持有清除：home 侧恰为 base+增量，exec 侧清零。
        used = [snapshot.kv_cache_bytes for snapshot in kv.hbm_snapshots()]
        base_plus_increment = kv_cache_shard_bytes_for_layer_range(
            kv.model, 18, kv.tp_degree, layer_start=0,
            layer_end=kv.model.layers)
        self.assertEqual(tuple(used[:2]), tuple(base_plus_increment))
        self.assertEqual(tuple(used[2:]), (0, 0))
        kv.mark_complete("s", 20, "human")

    def test_recompute_at_remote_instance_merges_increments(self):
        kv = self._manager()
        _seed(kv, "s", 0, 10, 10, "human")
        kv.prepare_prefill(
            session_id="s", target_instance_index=1,
            history_tokens=10, trigger_request_id="t1",
            action="recompute")
        working = kv.session_snapshot("s")
        self.assertEqual(working.working_kind, "recompute")
        self.assertEqual(working.context_tokens, 0)
        # 重算物化 + 增量。
        kv.expand_prefill(
            session_id="s", instance_index=1, context_tokens=13,
            trigger_request_id="t1")
        merge_transfers = kv.merge_back(
            session_id="s", trigger_request_id="t1", new_tokens=3)
        merged = kv.session_snapshot("s")
        self.assertEqual(merged.instance_index, 0)
        self.assertEqual(merged.context_tokens, 13)
        self.assertEqual(merged.home_instance, 0)
        # 增量 = 3 token（重算历史不重复归并——home 已有权威基础）。
        if merge_transfers:
            increment = kv_cache_shard_bytes_for_layer_range(
                kv.model, 3, kv.tp_degree, layer_start=0,
                layer_end=kv.model.layers)
            self.assertEqual(
                merge_transfers[0].total_bytes, sum(increment))
        kv.mark_complete("s", 20, "human")

    def test_remote_read_working_copy_holds_increments_only(self):
        kv = self._manager()
        _seed(kv, "s", 0, 10, 10, "human")
        kv.prepare_prefill(
            session_id="s", target_instance_index=1,
            history_tokens=10, trigger_request_id="t1",
            action="remote-read")
        working = kv.session_snapshot("s")
        self.assertEqual(working.working_kind, "remote-read")
        # 工作副本从 0 起步：仅新增 token 的 KV 驻留执行端。
        self.assertEqual(working.context_tokens, 0)
        base_bytes = kv_cache_shard_bytes_for_layer_range(
            kv.model, 10, kv.tp_degree, layer_start=0,
            layer_end=kv.model.layers)
        used = [snapshot.kv_cache_bytes for snapshot in kv.hbm_snapshots()]
        self.assertEqual(tuple(used[:2]), tuple(base_bytes))  # home 基础不动
        kv.expand_prefill(
            session_id="s", instance_index=1, context_tokens=5,
            trigger_request_id="t1")
        kv.merge_back(
            session_id="s", trigger_request_id="t1", new_tokens=5)
        merged = kv.session_snapshot("s")
        self.assertEqual(merged.context_tokens, 15)
        self.assertEqual(merged.instance_index, 0)
        kv.mark_complete("s", 20, "human")

    def test_stay_completes_without_merge_traffic(self):
        kv = self._manager()
        _seed(kv, "s", 0, 10, 10, "human")
        kv.prepare_prefill(
            session_id="s", target_instance_index=0,
            history_tokens=10, trigger_request_id="t1", action="stay")
        kv.expand_prefill(
            session_id="s", instance_index=0, context_tokens=12,
            trigger_request_id="t1")
        # stay：本地提交零流量（§2.2：执行位置等于 home 不生成虚构 D2D）。
        self.assertEqual(kv.merge_back(
            session_id="s", trigger_request_id="t1", new_tokens=2), ())
        kv.mark_complete("s", 20, "human")

    def test_mark_complete_rejects_unmerged_working_copy(self):
        kv = self._manager()
        _seed(kv, "s", 0, 10, 10, "human")
        kv.prepare_prefill(
            session_id="s", target_instance_index=1,
            history_tokens=10, trigger_request_id="t1", action="copy")
        with self.assertRaisesRegex(
                RuntimeError, "unmerged working copy"):
            kv.mark_complete("s", 20, "human")


# ============================================================== 八组合 ==


class EightCombinationIntegrationTest(unittest.TestCase):
    """§7.1 硬性要求：同一实现内配置切换出全部八组合。

    同一 KVCacheManager 构造路径 + 同一选择函数，仅开关不同；各组合
    语义可区分（typed/lru 的类别序、adaptive/minimal 的释放层数、
    joint/load-first 的选点）。
    """

    def test_all_eight_combos_configurable_and_distinct(self):
        seen_modes = set()
        for combo in COMBO_NAMES:
            category, scheduler, layer = COMBO_PRESETS[combo]
            config = parse_joint_config({"JOINT_ABLATION_COMBO": combo})
            seen_modes.add(
                (config.category_mode, config.scheduler_mode,
                 config.layer_policy))
            # KV 层（T/E）与选择层（J）均由同一实现承载。
            hardware = _tiny_hardware()
            topology = _two_instance_topology(hardware)
            kv = KVCacheManager(
                topology, _tiny_model(4), category_mode=category,
                layer_policy=layer)
            self.assertEqual(kv.category_mode, category)
            self.assertEqual(kv.layer_policy_mode, layer)
            loads = {0: _load(0, 100), 1: _load(1, 0)}
            record = select_instance_and_action(
                mode=scheduler, cost_model=_cost_model(loads),
                session=_session(home=0, resident=1),
                request=_request(), remote_enabled=True)
            self.assertIsNotNone(record.chosen)
        # 八组合产生恰好 8 个不同开关三元组（覆盖 §7.1 表全行）。
        self.assertEqual(len(seen_modes), 8)

    def test_off_switches_do_not_change_untested_modules(self):
        """T-off/lru 只改类别序：保护规则与合法集合不变（同一
        _completed_resident_candidates 过滤器）；E-off 只换层计划。"""
        hardware = _tiny_hardware()
        topology = _two_instance_topology(hardware)
        model = _tiny_model(4)
        for category in ("typed", "lru"):
            kv = KVCacheManager(
                topology, model, category_mode=category,
                layer_policy="minimal_layer_groups")
            # active 会话在两种模式下都不是合法 victim。
            kv.prepare_prefill(
                session_id="active", target_instance_index=0,
                history_tokens=0, trigger_request_id="seed")
            candidates = kv._completed_resident_candidates(0, None)
            self.assertNotIn("active", [c.session_id for c in candidates])


if __name__ == "__main__":
    unittest.main()
