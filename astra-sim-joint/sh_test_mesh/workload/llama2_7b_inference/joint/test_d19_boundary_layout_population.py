#!/usr/bin/env python3
"""test_d19_boundary_layout_population.py -- D19 轮二：BoundaryKVLayout
摆脱 schema-only（三项运行时状态 + 层覆盖的执行面填充接通钉测）。

背景（修复轮二审查结论）：``BoundaryKVLayout.from_boundary`` 此前只填
区间字节向量，``terminated``/``first_token``/``obligations`` 恒缺省、
区间 ``layer_start=layer_end=0`` 注释"全层（账本字节向量为准）"——
PARTIAL 基础区间实际只有前缀 [0,p) 层，层覆盖不真实披露。本轮把三项
状态与层覆盖经 ``BoundaryRuntimeStateEnvelope``（可选属性面，鸭子类型
透传 PDBoundaryView）带进 ``from_boundary``。

钉测面：
  1. 完整型单族布局 × 三项填充——envelope 携带的
     terminated/first_token/obligations 逐位落进 layout；
  2. 层覆盖——完整型 context = 全模型层数（增量型布局族已随远读
     退役删除，D6）；
  3. 缺省回退兼容——直接构造的 PDBoundaryView（无可选属性）回退
     terminated=False / 首 token 全缺省 / 义务表空 / 层 0/0，与既有
     夹具逐位一致（层字段是披露面：0 = 未披露，字节向量仍是账本真值，
     计价行为零变化）；
  4. sh30 执行面接线——真调度器 ``_joint_boundary_view`` 返回 envelope
     （存在性断言：五属性全在场），terminated 源 =
     ``_joint_boundary_terminal``（正常 D 路径恒 False），first_token
     源 = StageWork（P 末 chunk 已采样恰一次），obligations 源 = 继承
     义务表（含在册容量预约）；``estimate_D_remaining`` 消费同一
     envelope 时 layout 已带三项（计价入口端到端）；
  5. envelope 构造点 fail-closed——first_token/obligations 类型不符
     即 ValueError（不接收推测性登记）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 joint/test_d19_boundary_layout_population.py
"""
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WL = os.path.dirname(_HERE)
for _p in (_HERE, _WL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from types import SimpleNamespace  # noqa: E402

from joint.boundary_layout import (  # noqa: E402
    INTERVAL_ROLE_CONTEXT,
    BoundaryFirstTokenState,
    BoundaryKVLayout,
    BoundaryRuntimeStateEnvelope,
    InheritedObligation,
)
from joint.joint_cost_model import (  # noqa: E402
    ACTION_STAY,
    JointCostModel,
    JointHardwareRates,
    LinkFlowRegistry,
    ServiceFactors,
)
from joint.test_d18_boundary_decision import (  # noqa: E402
    KV_TOKEN_BYTES,
    _complete_boundary,
    _loads,
    _model,
)

MODEL_LAYERS = 4                     # tiny model 层数（test_d18 同源）
PARTIAL_BASE_LAYERS = 2              # PARTIAL 基前缀 [0,p)


def _issued_first_token():
    """P 末 chunk 已采样恰一次（issued、RNG 推进 1、待首个 D 前向消费）。
    """
    return BoundaryFirstTokenState(
        issued=True, consumed=False, rng_advances=1,
        rebuild_sampling_suppressed=False)


def _obligations(*kinds):
    return tuple(
        InheritedObligation(
            obligation_id=f"r0#{kind}", kind=kind,
            receiver_instance=None, summary="population fixture")
        for kind in kinds)


class EnvelopePopulationUnitTest(unittest.TestCase):
    """单元面：envelope → from_boundary 的三项填充与层覆盖（两族）。"""

    def test_complete_family_populates_three_items_and_full_layers(self):
        envelope = BoundaryRuntimeStateEnvelope(
            _complete_boundary(),
            terminated=False,
            first_token=_issued_first_token(),
            obligations=_obligations("committed_transfer"),
            working_layers=MODEL_LAYERS,
            base_layers=MODEL_LAYERS)
        layout = BoundaryKVLayout.from_boundary(envelope)
        self.assertFalse(layout.terminated)
        self.assertEqual(layout.first_token, _issued_first_token())
        self.assertEqual(
            tuple(item.obligation_id for item in layout.obligations),
            ("r0#committed_transfer",))
        # 完整型：单 context 区间，真实层覆盖 = 全模型层数。
        self.assertEqual(len(layout.intervals), 1)
        context = layout.intervals[0]
        self.assertEqual(context.role, INTERVAL_ROLE_CONTEXT)
        self.assertEqual(
            (context.layer_start, context.layer_end),
            (0, MODEL_LAYERS))

    def test_terminated_true_propagates(self):
        """terminated 真实接通（非 schema 占位）：真值经 envelope 落进
        layout——正常 D 路径恒 False（见 sh30 接线钉测），字段通道在。"""
        envelope = BoundaryRuntimeStateEnvelope(
            _complete_boundary(), terminated=True,
            first_token=_issued_first_token())
        layout = BoundaryKVLayout.from_boundary(envelope)
        self.assertTrue(layout.terminated)
        self.assertTrue(layout.first_token.issued)
        self.assertEqual(layout.first_token.rng_advances, 1)
        # envelope 未指定 first_token 时构造缺省态（通道在场、值缺省）。
        default_layout = BoundaryKVLayout.from_boundary(
            BoundaryRuntimeStateEnvelope(
                _complete_boundary(), terminated=True))
        self.assertTrue(default_layout.terminated)
        self.assertFalse(default_layout.first_token.issued)

    def test_default_fallback_plain_view_is_bit_compatible(self):
        """缺省回退兼容：直接构造的 PDBoundaryView（无可选属性——共享
        文件的计价视图字段面不动）回退到旧行为：三项缺省 + 层 0/0。"""
        for boundary in (_complete_boundary(),):
            layout = BoundaryKVLayout.from_boundary(boundary)
            self.assertFalse(layout.terminated)
            self.assertEqual(
                layout.first_token, BoundaryFirstTokenState())
            self.assertEqual(layout.obligations, ())
            for interval in layout.intervals:
                self.assertEqual(
                    (interval.layer_start, interval.layer_end), (0, 0))
            # 字节真值不受层披露缺省影响（计价口径 = 账本向量）。
            self.assertGreater(layout.intervals[0].total_bytes(), 0)

    def test_envelope_rejects_speculative_registration(self):
        """构造点 fail-closed：first_token/obligations 类型不符即
        ValueError（存在性/类型断言——不接收推测性登记）。"""
        with self.assertRaises(ValueError):
            BoundaryRuntimeStateEnvelope(
                _complete_boundary(), first_token=object())
        with self.assertRaises(ValueError):
            BoundaryRuntimeStateEnvelope(
                _complete_boundary(),
                obligations=(SimpleNamespace(kind="capacity_reservation"),))
        bad = SimpleNamespace(
            first_token=_issued_first_token(),
            obligations=_obligations("capacity_reservation"))
        with self.assertRaises(ValueError):
            # first_token 缺类型约束的鸭子面（非 BoundaryFirstTokenState）。
            BoundaryKVLayout.from_boundary(
                SimpleNamespace(
                    kv_token_bytes=KV_TOKEN_BYTES,
                    working_bytes_by_tp_rank=(KV_TOKEN_BYTES,) * 2,
                    p_instance_index=0, history_tokens=1,
                    decode_steps=1, session_id="s", home_instance=0,
                    first_token=SimpleNamespace(issued=True)))


class EnvelopePassthroughTest(unittest.TestCase):
    """envelope 透传纪律：PDBoundaryView 字段面/方法经 __getattr__ 全量
    可达（计价/选择器零改动的基础）。"""

    def test_view_fields_and_methods_delegate(self):
        view = _complete_boundary()
        envelope = BoundaryRuntimeStateEnvelope(
            view, terminated=False, first_token=_issued_first_token(),
            obligations=_obligations("capacity_reservation"),
            working_layers=MODEL_LAYERS, base_layers=MODEL_LAYERS)
        self.assertEqual(envelope.session_id, view.session_id)
        self.assertEqual(envelope.home_instance, view.home_instance)
        self.assertEqual(envelope.p_instance_index, view.p_instance_index)
        self.assertEqual(
            envelope.working_bytes_by_tp_rank, view.working_bytes_by_tp_rank)
        self.assertEqual(envelope.decode_steps, view.decode_steps)
        self.assertEqual(envelope.kv_token_bytes, view.kv_token_bytes)
        self.assertEqual(envelope.logic_home_instance, view.logic_home_instance)
        self.assertEqual(
            envelope.working_context_bytes(), view.working_context_bytes())


class SchedulerPopulationTest(unittest.TestCase):
    """sh30 执行面接线：真调度器 _joint_boundary_view 的三项/层覆盖填充
    （存在性断言——审查"schema-only"结论的正钉）。"""

    def _drain_ready(self):
        from face_scheduler import KVCacheManager
        from joint.test_joint_mechanisms import (
            _tiny_hardware,
            _tiny_model,
            _two_instance_topology,
        )
        from online.test_decision_route_serialization import (
            _GraphStub,
            _decision_shell,
            _make_runtime,
        )
        hardware = _tiny_hardware()
        model = _tiny_model()
        topology = _two_instance_topology(hardware)
        kv = KVCacheManager(topology, model)
        scheduler = _decision_shell(kv, hardware=hardware, model=model,
                                    topology=topology)
        scheduler.graph = _GraphStub()
        runtime = _make_runtime("r0", decode_length=8)
        scheduler.runtime_by_request_id["r0"] = runtime
        kv.reserve_request_capacity(
            request_id="r0", session_id=runtime.session_id, instance_index=0,
            final_context_tokens=runtime.prefill_context_tokens)
        kv.prepare_prefill(
            session_id=runtime.session_id, target_instance_index=0,
            history_tokens=0, trigger_request_id="r0")
        kv.expand_prefill(
            session_id=runtime.session_id, instance_index=0,
            context_tokens=runtime.prefill_context_tokens,
            trigger_request_id="r0")
        runtime.prefill_instance_index = 0
        return scheduler, kv, runtime

    def test_view_returns_populated_envelope(self):
        scheduler, kv, runtime = self._drain_ready()
        boundary = scheduler._joint_boundary_view(
            runtime, SimpleNamespace(index=0))
        # 存在性断言（审查项）：五属性全在场，非 schema 占位。
        for name in ("terminated", "first_token", "obligations",
                     "working_layers", "base_layers"):
            self.assertTrue(
                hasattr(boundary, name),
                f"boundary envelope missing {name!r}")
        # terminated 源 = _joint_boundary_terminal（decode_length=8 > 0 →
        # 正常 D 路径恒 False）。
        self.assertFalse(boundary.terminated)
        self.assertFalse(scheduler._joint_boundary_terminal(runtime))
        # first_token 源 = StageWork：P 末 chunk 已采样恰一次。
        self.assertTrue(boundary.first_token.issued)
        self.assertFalse(boundary.first_token.consumed)
        self.assertEqual(boundary.first_token.rng_advances, 1)
        # obligations 源 = 继承义务表：在册容量预约必须登记（真实账本
        # 对象，非推测）。
        kinds = {item.kind for item in boundary.obligations}
        self.assertIn("capacity_reservation", kinds)
        for item in boundary.obligations:
            self.assertIsInstance(item, InheritedObligation)
        # 层覆盖 = 真实模型层数（完整型）。
        self.assertEqual(boundary.working_layers, scheduler.model.layers)
        self.assertEqual(boundary.base_layers, scheduler.model.layers)

    def test_from_boundary_picks_up_scheduler_population(self):
        scheduler, kv, runtime = self._drain_ready()
        boundary = scheduler._joint_boundary_view(
            runtime, SimpleNamespace(index=0))
        layout = BoundaryKVLayout.from_boundary(boundary)
        self.assertFalse(layout.terminated)
        self.assertTrue(layout.first_token.issued)
        self.assertEqual(layout.first_token.rng_advances, 1)
        self.assertIn(
            "capacity_reservation",
            {item.kind for item in layout.obligations})
        context = layout.intervals[0]
        self.assertEqual(context.role, INTERVAL_ROLE_CONTEXT)
        self.assertEqual(context.layer_end, scheduler.model.layers)

    def test_estimate_d_remaining_consumes_envelope(self):
        """计价入口端到端：estimate_D_remaining 消费同一 envelope 时，
        内部派生的 BoundaryKVLayout 已带三项与层覆盖（从计价面可观测，
        不经调度器旁路）。"""
        scheduler, kv, runtime = self._drain_ready()
        boundary = scheduler._joint_boundary_view(
            runtime, SimpleNamespace(index=0))
        cost_model = JointCostModel(
            rates=JointHardwareRates.from_gbps(
                noc_link_gbps=10.0, pool_port_gbps=5.0,
                local_hbm_gbps=100.0, d2d_latency_ns=10,
                pool_latency_ns=100),
            loads=_loads(),
            flow_registry=LinkFlowRegistry(),
            service_factors=ServiceFactors(),
            prefill_ns_per_token=1.0, decode_ns_per_token=2.0,
            model_layers=MODEL_LAYERS, instance_tp_size=2,
            route_fn=lambda s, t: ((s, t), 1))
        candidate = cost_model.estimate_D_remaining(
            boundary=boundary, instance_index=0, action=ACTION_STAY)
        self.assertTrue(candidate.applicable)
        # 区间布局披露（breakdown.interval_layout）携带层覆盖与区间身份
        # ——envelope 填充从计价面可观测。
        layout = BoundaryKVLayout.from_boundary(boundary)
        self.assertEqual(
            layout.intervals[0].layer_end, MODEL_LAYERS)


if __name__ == "__main__":
    unittest.main()
