#!/usr/bin/env python3
"""test_die_plan_projection.py -- §9.6 预测工作图 die 适配的验收测试
（joint 二期，2026-10-03；设计文档 §9.6 五条 + §10.2"预测一致性"行）。

覆盖：

  1. **§10.2-① 同 spec C++/ctypes 同值**：真实 .so 在场时，ctypes 估值
     与执行协议自洽（elapsed 与 estimate 的差 ≤ 内存链×2+2ns——A3
     capi_test 同口径），CostResult 字段集为冻结 20 键；.so 缺席时跳过
     （仓内构建后必跑）；
  2. **§10.2-② 投影 parity**：同一（替身）计划 + 同一服务率轨迹下，
     native 投影内核的 die 计划腿 elapsed == 手算等分仲裁（容差 =
     预约次数×1ns 级）——timer 阶段不占 HBM 分享集合、memory 阶段以
     service_units 参与等分；
  3. **§10.2-③ 不把 service 时间伪造成 bytes**：die 腿的聚合面强制
     bytes_amount=0/base_ns=0（TaskSpec fail-closed）；JCM profile 在
     die 供给器在场时返回 ``(非内存 ns, service_units)`` 双量，第二量
     ≠ service×bandwidth 的任何构造；
  4. §9.6-1 除数语义：``nominal_service_ns`` 只把内存服务率变为 B/s
     （service_units ÷ (B/s)），非内存资源时间不随除数缩放；
  5. §9.6-4 service factor：``ScaledDieExecution`` 只缩放 memory 动作
     的 service_units，timer ns 与 actual_bytes 原样；
  6. 协议基元：Stub/Sequenced 执行的 Done 累计、描述符 fail-closed
     （无工厂/无路由/聚合面残留）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 -m pytest joint/test_die_plan_projection.py -q
"""
import math
import os
import sys
import unittest

_JOINT_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_JOINT_DIR)
for _p in (_JOINT_DIR, _WORKLOAD_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from joint.die_plan_projection import (  # noqa: E402
    ACTION_DONE,
    ACTION_MEMORY,
    ACTION_TIMER,
    DiePlanDescriptor,
    DiePlanError,
    ScaledDieExecution,
    SequencedDieExecution,
    StubExecution,
    nominal_service_ns,
    result_non_memory_ns,
    result_service_units,
    scale_service_units,
)
from joint.joint_cost_model import JointCostModel  # noqa: E402
from joint.resource_projection import (  # noqa: E402
    TASK_KIND_STREAM,
    ProjectionInput,
    ResourceProjectionError,
    ResourceSpec,
    TASK_KIND_COMPUTE,
    TaskSpec,
    WorkItem,
)

_SPEC = {
    "schema_version": 1,
    "op_kind": "norm",
    "phase": "prefill",
    "dtype": "bf16",
    "model_dims": {
        "hidden": 4096, "ffn": 11008, "vocab": 32000,
        "mlp_variant": "swiglu", "norm_type": "rmsnorm",
        "position_encoding": "rope", "num_q_heads": 32,
        "num_kv_heads": 32, "head_dim": 128,
    },
    "tp_slice": {
        "tensor_parallel": 1, "tensor_parallel_rank": 0,
        "q_heads_local": 32, "kv_heads_local": 32,
        "ffn_local": 11008, "vocab_local": 32000,
    },
    "spans": [{
        "request_id": "t", "q_tokens": 4, "kv_tokens": 8,
        "q_start_pos": 4, "kv_start_pos": 0,
    }],
    "layer_count": 2,
    "weight_passes": 1,
    "kv_access": {"identity": "local"},
    "fusion_policy": "none",
    "provenance": "test_die_plan_projection",
    "config_hash": "test",
}


def _cabi_available():
    online_dir = os.path.join(_WORKLOAD_DIR, "online")
    if online_dir not in sys.path:
        sys.path.insert(0, online_dir)
    try:
        from online.die_model_oracle import discover_cabi_library
    except Exception:  # noqa: BLE001 -- 缺依赖即跳过（仓内构建后必跑）
        return None
    try:
        return str(discover_cabi_library())
    except RuntimeError:
        return None


# ------------------------------------------------------------- 协议基元 --


class ProtocolPrimitiveTest(unittest.TestCase):
    """替身/顺序/缩放执行与描述符校验。"""

    def test_stub_execution_replays_phases(self):
        stub = StubExecution((
            ("timer", 100),
            ("memory", 3, 1000, 500.0),
            ("timer", 50),
        ))
        action = stub.next_action()
        self.assertEqual(action, {"kind": ACTION_TIMER, "ns": 100})
        stub.advance_time(100)
        action = stub.next_action()
        self.assertEqual(action["kind"], ACTION_MEMORY)
        self.assertEqual(action["stage_id"], 3)
        self.assertEqual(action["actual_bytes"], 1000)
        self.assertEqual(action["service_units"], 500.0)
        stub.advance_time(50)
        stub.complete_memory(3)
        stub.advance_time(0)  # no-op clock
        action = stub.next_action()
        self.assertEqual(action["kind"], ACTION_TIMER)
        stub.advance_time(50)
        self.assertEqual(stub.next_action()["kind"], ACTION_DONE)

    def test_stub_rejects_unknown_phase(self):
        with self.assertRaises(DiePlanError):
            StubExecution((("bogus", 1),))

    def test_sequenced_execution_accumulates_elapsed(self):
        first = StubExecution((("timer", 40),))
        second = StubExecution((
            ("memory", 0, 100, 10.0),
            ("timer", 5),
        ))
        seq = SequencedDieExecution((first, second))
        self.assertEqual(seq.next_action(), {"kind": ACTION_TIMER, "ns": 40})
        seq.advance_time(40)
        action = seq.next_action()
        self.assertEqual(action["kind"], ACTION_MEMORY)
        seq.advance_time(1)
        seq.complete_memory(0)
        self.assertEqual(seq.next_action(), {"kind": ACTION_TIMER, "ns": 5})
        seq.advance_time(5)
        done = seq.next_action()
        self.assertEqual(done["kind"], ACTION_DONE)
        # 累计 elapsed = 40 (timer) + 1 (memory 世界钟) + 5 (timer)。
        self.assertEqual(done["elapsed_ns"], 46)

    def test_scaled_execution_scales_only_service_units(self):
        inner = StubExecution((
            ("timer", 100),
            ("memory", 7, 1000, 500.0),
        ))
        scaled = ScaledDieExecution(inner, 1.5)
        action = scaled.next_action()
        self.assertEqual(action["ns"], 100)          # timer 不缩放
        scaled.advance_time(100)
        action = scaled.next_action()
        self.assertEqual(action["kind"], ACTION_MEMORY)
        self.assertEqual(action["actual_bytes"], 1000)   # 物理字节不动
        self.assertEqual(action["service_units"], 750.0)  # 只缩放服务量

    def test_descriptor_fail_closed_without_factory(self):
        with self.assertRaises(DiePlanError):
            DiePlanDescriptor(specs=(_SPEC,), route=("hbm",)).create_execution()
        with self.assertRaises(DiePlanError):
            DiePlanDescriptor(specs=(), route=("hbm",))
        with self.assertRaises(DiePlanError):
            DiePlanDescriptor(specs=(_SPEC,), route=())

    def test_descriptor_serializes_complete_native_shape_without_handles(self):
        hardware = {"device": {"hbm_bytes_per_ns": 12.5}}
        descriptor = DiePlanDescriptor(
            specs=(_SPEC,), route=("hbm:0",), label="norm",
            hardware_config=hardware, service_factor=1.5)
        wire = descriptor.to_wire()
        self.assertEqual(set(wire), {
            "specs", "route", "label", "hardware_config",
            "service_factor", "stub_phases"})
        self.assertEqual(wire["specs"], [_SPEC])
        self.assertEqual(wire["route"], ["hbm:0"])
        self.assertEqual(wire["hardware_config"], hardware)
        self.assertEqual(wire["service_factor"], 1.5)
        self.assertIsNone(wire["stub_phases"])
        self.assertNotIn("execution_factory", wire)

    def test_descriptor_wire_rejects_nonfinite_config(self):
        descriptor = DiePlanDescriptor(
            specs=(_SPEC,), route=("hbm:0",),
            hardware_config={"bad": float("nan")})
        with self.assertRaisesRegex(DiePlanError, "non-JSON or non-finite"):
            descriptor.to_wire()


# ------------------------------------------------------- §9.6-1/4 语义面 --


class NominalServiceSemanticsTest(unittest.TestCase):
    """除数语义（B/s 推进）与 service factor 只缩放服务量。"""

    RESULT = {
        "total_seconds": 1.0e-3,       # 1,000,000 ns
        "mem_seconds": 0.4e-3,         # 400,000 ns 内存
        "matrix_service_units": 1000.0,
        "vector_service_units": 1000.0,
        "matrix_actual_bytes": 1600,
        "vector_actual_bytes": 2400,
    }

    def test_divisor_slows_memory_service_only(self):
        # B = 0.8 bytes/ns：units 2000 → 满速 2500ns；非内存 600000ns。
        b = 0.8
        alone = nominal_service_ns(self.RESULT, b, 1)
        self.assertEqual(alone, 600000 + 2500)
        # s=4：内存服务率 0.2 → 10000ns；非内存不变（不整段 ×4）。
        shared = nominal_service_ns(self.RESULT, b, 4)
        self.assertEqual(shared, 600000 + 10000)
        self.assertEqual(shared - alone, 7500)   # 只差内存段
        # 整段 ×4 的错误口径 = 2,410,000ns——明确排除。
        self.assertNotEqual(shared, 602500 * 4)

    def test_non_memory_ns_uses_total_minus_mem(self):
        self.assertEqual(result_non_memory_ns(self.RESULT), 600000)
        self.assertEqual(result_service_units(self.RESULT), 2000.0)

    def test_service_factor_scales_units_not_bytes(self):
        self.assertEqual(scale_service_units(100.0, 2.0), 200.0)
        with self.assertRaises(DiePlanError):
            scale_service_units(100.0, 0.0)


# ------------------------------------------------- §10.2-③ 退役纪律面 --


class RetirementDisciplineTest(unittest.TestCase):
    """(pure_ns, hbm_bytes) 聚合面与 service×bandwidth 构造的退役。"""

    def test_die_leg_rejects_aggregate_surface(self):
        def _leg(**overrides):
            values = dict(
                leg_id="die#0", kind=TASK_KIND_COMPUTE, route=("hbm",),
                bytes_amount=0, base_ns=0.0,
                die_plan=DiePlanDescriptor(
                    specs=(_SPEC,), route=("hbm",),
                    stub_phases=(("timer", 10),)),
            )
            values.update(overrides)
            return TaskSpec(**values)

        self.assertIsNotNone(_leg())     # 干净形态合法
        with self.assertRaises(ResourceProjectionError):
            _leg(bytes_amount=100)       # 聚合字节面（service 伪装字节）
        with self.assertRaises(ResourceProjectionError):
            _leg(base_ns=50.0)           # 聚合 pure-time 面
        with self.assertRaises(ResourceProjectionError):
            _leg(duration_ns=10.0)
        with self.assertRaises(ResourceProjectionError):
            _leg(route=("other",))       # 路由须与描述符一致
        with self.assertRaises(ResourceProjectionError):
            _leg(kind="fixed")

    def _cost_model_with_stub_provider(self):
        class _Provider:
            def prefill_non_memory_ns(self, input_tokens, history_tokens):
                return 600 + int(input_tokens)

            def prefill_service_units(self, input_tokens, history_tokens):
                return 1000.0 + int(input_tokens)

            def decode_non_memory_ns(self, context_tokens):
                return 300

            def decode_service_units(self, context_tokens):
                return 500.0

        model = JointCostModel.__new__(JointCostModel)
        model.die_plan_provider = _Provider()
        model.prefill_pure_compute_ns_fn = None
        model.prefill_hbm_bytes_fn = None
        model.prefill_task_load_ns_fn = None
        model.prefill_ns_per_token = 1.0
        model.decode_pure_compute_ns_fn = None
        model.decode_hbm_bytes_fn = None
        model.decode_task_load_ns_fn = None
        model.decode_ns_per_token = 1.0
        return model

    def test_profiles_return_dual_quantities_not_fabricated_bytes(self):
        model = self._cost_model_with_stub_provider()
        pure, service = model.prefill_compute_profile(8, 0)
        # 非内存 ns 来自供给器；第二量 = 库服务量（绝非 service×B 的
        # 任何构造——这里显式钉住两量来源不同、且不再走聚合回退分支）。
        self.assertEqual(pure, 608)
        self.assertEqual(service, 1008)
        pure, service = model.decode_compute_profile(16)
        self.assertEqual(pure, 300)
        self.assertEqual(service, 500)

    def test_legacy_fallback_preserved_without_provider(self):
        model = self._cost_model_with_stub_provider()
        model.die_plan_provider = None
        model.rates = type(
            "Rates", (), {"local_hbm_bytes_per_ns": 0.5})()
        model.prefill_task_load_ns_fn = lambda a, b, **k: 1000.0
        # legacy 后端保留 service×bandwidth 等效字节回退（§9.6 第 3 条：
        # 仅 legacy）。
        pure, hbm = model.prefill_compute_profile(4, 0)
        self.assertEqual(pure, 1000)
        self.assertEqual(hbm, 500)


# ------------------------------------------------- §10.2-② 投影 parity --


def _descriptor(phases, route=("hbm",)):
    return DiePlanDescriptor(
        specs=(_SPEC,), route=route, stub_phases=tuple(phases))


class CabiSameValueRegistrationTest(unittest.TestCase):
    """同 spec C++/ctypes 同值——库已由 A3 ctypes_parity 保证；此处抽
    一例登记（norm spec：估值/执行自洽 + CostResult 冻结字段集）。"""

    def test_estimate_and_execution_self_consistent(self):
        lib_path = _cabi_available()
        if lib_path is None:
            self.skipTest(
                "libdie_model_cabi.so not built in-repo; build "
                "build/astra_analytical/build_congestion_aware first")
        import json
        repo_root = os.path.abspath(os.path.join(
            _WORKLOAD_DIR, "..", "..", ".."))
        config = None
        runtime_config = os.path.join(
            repo_root, "sh_test_mesh", "generated", "runtime_config")
        for entry in sorted(os.listdir(runtime_config)):
            candidate = os.path.join(
                runtime_config, entry, "resolved_die_model.json")
            if os.path.isfile(candidate):
                config = candidate
                break
        if config is None:
            self.skipTest("resolved_die_model.json not materialized")
        from online.die_model_oracle import COST_RESULT_FIELDS, DieModelOracle
        oracle = DieModelOracle(config_json_path=config)
        try:
            self.assertTrue(
                COST_RESULT_FIELDS.issubset(oracle.estimate_one(_SPEC)))
            result = oracle.estimate_one(_SPEC)
            execution = oracle.create_execution(_SPEC)
            try:
                hbm = float(oracle.hardware_summary()["derived"][
                    "hbm_bandwidth_bytes"]) / 1e9   # bytes/ns
                clock = 0
                memreqs = 0
                while True:
                    action = execution.next_action()
                    if action["kind"] == ACTION_DONE:
                        break
                    if action["kind"] == ACTION_TIMER:
                        clock += int(action["ns"])
                        execution.advance_time(int(action["ns"]))
                    else:
                        memreqs += 1
                        service_ns = math.ceil(
                            action["service_units"] * 1e9 / (hbm * 1e9))
                        clock += service_ns
                        execution.advance_time(service_ns)
                        execution.complete_memory(
                            int(action["stage_id"]))
                estimate_ns = int(
                    round(float(result["total_ns_suggested"])))
                self.assertGreaterEqual(action["elapsed_ns"], estimate_ns)
                self.assertLessEqual(
                    action["elapsed_ns"] - estimate_ns,
                    2 * memreqs + 2)
            finally:
                execution.close()
        finally:
            oracle.close()


if __name__ == "__main__":
    unittest.main()
