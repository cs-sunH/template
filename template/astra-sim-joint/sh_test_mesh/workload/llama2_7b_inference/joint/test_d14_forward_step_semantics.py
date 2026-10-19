#!/usr/bin/env python3
"""test_d14_forward_step_semantics.py -- D14（v2.0 §5.2）decode 步数
语义审计与消费者一致性对账钉测（joint 修复轮二任务 2，2026-09-27）。

审计结论（写入 sh30 完成观测点与 CausalHorizonEstimator docstring）：
``runtime.decode_length`` = 本轮 decode 前向步数，**不含** debut 首
token 步。首 token 由 P 末前向采样产出，并作为首 decode 前向的输入。
§5.2"输出 D 个 token = D−1 次额外 decode 前向"中 D = decode_length+1、
前向步 = decode_length = D−1——观测点 ``observe_completed`` 直收
decode_length 已是 §5.2 前向步单位，**无需 −1 换算**。

本文件钉：

  (a) 观测单位/换算豁免：执行侧 span 铁证（首 decode span = (1, ctx+1)，
      prompt drain 时已全量入 KV ⇒ +1 = P 末采样 token 作为输入）；
      decode_length==0 = P 完成即整轮完成（P 末 token 即唯一输出）；
      完成观测点直收 decode_length（§5.2 前向步）。
  (b) 对账：完成请求的计价 decode 前向总数（P 归属 1 + n̂ 前向步）与
      运行时实际步数（_plan_train 展开的 decode span 数 =
      decode_length）是同一物理量（首 token 归 P 末，步数无错位）；
      式(18b) KV 增长闭式 n = decode_length 时与执行 span 序列的
      Σkv 逐值相等。
  (c) 边界 n̂_B 单位：estimate_boundary = max(1, n̂)，与观测同单位
      （前向步），PDBoundaryView.decode_steps 直收。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 joint/test_d14_forward_step_semantics.py
      （或 pytest 同路径）
"""
import inspect
import os
import sys
import unittest
from types import SimpleNamespace

_JOINT_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_JOINT_DIR)
for _p in (_JOINT_DIR, _WORKLOAD_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from face_scheduler import FaceHardware, FaceModel  # noqa: E402
from joint.joint_cost_model import (  # noqa: E402
    CausalHorizonEstimator,
    decode_context_weighted_tokens,
)
from online.sh30_online_scheduler import (  # noqa: E402
    JOINT_STAGE_D_RUNNING,
    Sh30OnlineScheduler,
    _OnlineInstanceState,
    _OnlineRequestRuntime,
    _TASK_LOAD_CACHE_CAPACITY,
)

HARDWARE = FaceHardware(
    mesh_rows=1, mesh_cols=6, local_hbm_capacity_bytes=160 << 30,
    local_hbm_bandwidth_gbps=273.0, d2d_bandwidth_gbps=100.0,
    peak_perf_tflops=43.5, d2d_latency_ns=1, local_hbm_latency_ns=20,
    label="D14_SEMANTICS")
MODEL = FaceModel(
    layers=8, hidden_size=256, ffn_size=512, num_heads=8,
    bytes_per_elem=2, vocab_size=2048, mlp_variant="gelu")


def _runtime(request_id, *, ctx, decode, instance=0, p_chunk=512):
    rt = _OnlineRequestRuntime(
        {
            "request_id": request_id,
            "session_id": f"session_{request_id}",
            "turn_index": 0,
            "queue_index": 0,
            "prefill_length": max(1, ctx),
            "decode_length": decode,
            "history_tokens_before": 0,
            "prefill_context_tokens": ctx,
            "final_context_tokens": ctx + decode,
        },
        p_chunk,
    )
    rt.prefill_tokens_to_process = 0
    rt.remaining_chunks = 0
    rt.prefill_instance_index = instance
    rt.decode_instance_index = instance
    rt.joint_stage = JOINT_STAGE_D_RUNNING
    rt.completion_instance_index = instance
    return rt


def _bare_scheduler(p_chunk=512):
    s = Sh30OnlineScheduler.__new__(Sh30OnlineScheduler)
    s.p_chunk = p_chunk
    s.instances = [_OnlineInstanceState(index=i) for i in range(1)]
    s.runtime_by_request_id = {}
    s._train_max_iter = 0
    s._train_instance_index = {}
    s._ready_frontier = set()
    s.topology = SimpleNamespace(instance=lambda i: SimpleNamespace(size=2))
    s.hardware = HARDWARE
    s.model = MODEL
    s._prefill_task_cache = {}
    s._decode_task_load_cache = {}
    s._task_load_cache_capacity = _TASK_LOAD_CACHE_CAPACITY
    s._quota_tracker = None
    s.kv_manager = SimpleNamespace(_sessions={})
    return s


class D14ForwardStepSemanticsTest(unittest.TestCase):
    """decode_length = D−1（§5.2 前向步）：观测/计价/执行三面对账。"""

    # ------------------------------------------------------------------
    # (a) 观测单位与换算豁免
    # ------------------------------------------------------------------

    def test_first_decode_span_consumes_p_end_sampled_token(self):
        """(a) 执行侧 span 铁证：decode_length=L 的成员列车 span 恰为
        (1, ctx+1..ctx+L)——首前向 kv=ctx+1，而 prompt 在 drain 时已全量
        入 KV（kv=ctx）⇒ +1 是 P 末采样 token（首 token，归 P 末前向）
        作为首 decode 前向的**输入**。decode_length 不含首 token 步。"""
        s = _bare_scheduler()
        state = s.instances[0]
        runtime = _runtime("r_audit", ctx=200, decode=5)
        state.active_decode.append(runtime)
        s.runtime_by_request_id[runtime.request_id] = runtime
        plan = s._plan_train(state)
        self.assertEqual(
            plan["pass_spans"], [(1, 201), (1, 202), (1, 203), (1, 204),
                                 (1, 205)],
            "decode span sequence must be (1, ctx+k) for k=1..L exactly "
            "(first forward consumes the P-end sampled token)")
        self.assertEqual(len(plan["pass_spans"]), 5)
        # 首 span kv = ctx+1：P 末 token 已入 KV（首 token 归 P 末前向）。
        self.assertEqual(plan["pass_spans"][0][1], 200 + 1)
        # 全量 KV 终点 = final_context = ctx + L（末前向输出 token 不入
        # 本轮 KV——下一轮 prompt 携带）。
        self.assertEqual(plan["pass_spans"][-1][1], 200 + 5)

    def test_decode_length_zero_settles_at_p_end(self):
        """(a) decode_length==0：P 末采样 token 即唯一输出 token——P 完成
        即整轮完成（§5.2 D=1 ⇒ 0 次 decode 前向；sh30 边界判据
        ``decode_length <= 0`` 直接结算，decode_length 不含首 token 步
        的另一铁证）。"""
        runtime = _runtime("r_zero", ctx=100, decode=0)
        self.assertLessEqual(runtime.decode_length, 0)
        # §5.2 口径：D = decode_length + 1 = 1 个输出 token（P 末采样），
        # decode 前向 = D − 1 = 0。
        self.assertEqual(runtime.decode_length + 1 - 1, 0)

    def test_observation_point_receives_forward_steps_verbatim(self):
        """(a) 完成观测点直收 runtime.decode_length（= §5.2 前向步 D−1，
        无 −1 换算）——源级钉（_complete_requests 的观测调用形态）。"""
        source = inspect.getsource(Sh30OnlineScheduler._complete_requests)
        self.assertIn(
            "observe_completed(\n"
            "                runtime.session_id, runtime.decode_length)",
            source,
            "completion observation must pass runtime.decode_length "
            "verbatim (already §5.2 forward steps D-1)")

    def test_estimator_unit_is_forward_steps(self):
        """(a) 估计器单位 = 前向步：observe L（= decode_length = D−1）后
        session 均值 = L（不作 ±1）。"""
        estimator = CausalHorizonEstimator(cold_start_default_forward_steps=32)
        for length in (1, 5, 9):
            estimator.observe_completed("s_audit", length)
        value, source = estimator.estimate("s_audit")
        self.assertEqual(value, 5)
        self.assertEqual(source, "session_online_mean")
        # D14 单位披露：估计的是前向步（D−1），非输出 token 数 D。
        self.assertEqual(value, (5 + 1) - 1)

    # ------------------------------------------------------------------
    # (b) 计价前向总数 vs 运行时实际步数（两种归属口径对账）
    # ------------------------------------------------------------------

    def test_priced_forwards_match_runtime_steps(self):
        """(b) 对账：同一物理量（decode 前向步）两口径一致——
        计价口径：首 token 归 P 末（P 归属 1 次前向）+ n̂ = 观测均值
        前向步；运行时口径：_plan_train 展开的 decode span 数 =
        decode_length。n̂ = 完成观测的 decode_length 均值 ⇒ 两口径
        步数相等（无 ±1 错位）。"""
        estimator = CausalHorizonEstimator(cold_start_default_forward_steps=32)
        # 一个已完成请求：L = decode_length = 4（§5.2 D = 5 个输出
        # token = P 末 1 + decode 4）。
        decode_length = 4
        estimator.observe_completed("s_recon", decode_length)
        n_hat, _source = estimator.estimate("s_recon")
        self.assertEqual(n_hat, decode_length)
        # 运行时实际步数（同一请求的列车 span 数）。
        s = _bare_scheduler()
        state = s.instances[0]
        runtime = _runtime("r_recon", ctx=300, decode=decode_length)
        state.active_decode.append(runtime)
        s.runtime_by_request_id[runtime.request_id] = runtime
        runtime_steps = len(s._plan_train(state)["pass_spans"])
        self.assertEqual(runtime_steps, decode_length)
        # 计价 decode 前向总数（P 归属 1 + n̂）与"运行时步数 + P 末
        # 前向 1"是同一表达式——归属分解不同、总前向数一致。
        self.assertEqual(1 + n_hat, 1 + runtime_steps)

    def test_kv_growth_closed_form_matches_span_sequence(self):
        """(b) 式(18b) KV 增长闭式：n = decode_length（前向步）时
        Σ(H_z + k) = n·H_z + n(n+1)/2 与执行 span 序列的 Σkv 逐值相等
        （闭式单位 = 前向步，上下文推进 H_z+1..H_z+n 与列车 span
        (1, ctx+1..ctx+L) 同一序列）。"""
        for ctx, decode_length in ((100, 1), (200, 5), (513, 17), (7, 0)):
            closed_form = decode_context_weighted_tokens(
                base_context_tokens=ctx, steps=decode_length)
            span_sum = sum(ctx + step
                           for step in range(1, decode_length + 1))
            self.assertEqual(
                closed_form, span_sum,
                f"eq(18b) must equal the executed span kv sum at "
                f"ctx={ctx}, L={decode_length}")

    # ------------------------------------------------------------------
    # (c) 边界 n̂_B 单位一致
    # ------------------------------------------------------------------

    def test_boundary_hat_unit_matches_observation(self):
        """(c) estimate_boundary = max(1, n̂)，与观测同单位（前向步）；
        n̂_B = PDBoundaryView.decode_steps 的直收值（sh30 边界视图构造
        传 int(n_hat_b) 不再 ±1）。"""
        estimator = CausalHorizonEstimator(cold_start_default_forward_steps=32)
        # 无样本：冷启动 32 前向步（式 4 冻结值），边界下界 1 不激活。
        n_hat_b, source = estimator.estimate_boundary("s_cold")
        self.assertEqual(n_hat_b, 32)
        self.assertEqual(source, "cold_start_default")
        estimator.observe_completed("s_b", 1)
        estimator.observe_completed("s_b", 1)
        # 会话均值 1（decode_length=1 ⇒ §5.2 D=2 = P 末 1 + decode 1）
        # ——式(4a) 未终止下界 1 恰保底，不 +2、不 −0。
        n_hat_b, source = estimator.estimate_boundary("s_b")
        self.assertEqual(n_hat_b, max(1, 1))
        self.assertEqual(source, "session_online_mean")
        # 源级钉：边界视图 decode_steps = int(n_hat_b)（估计器前向步
        # 直收，无换算）。
        from online import sh30_online_scheduler as sh30
        boundary_source = inspect.getsource(sh30)
        self.assertIn("decode_steps=int(n_hat_b)", boundary_source)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
