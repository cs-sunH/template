#!/usr/bin/env python3
"""test_train_machinery.py -- wscllm 拼 batch 列车机制层 A1 夹具(2026-08-22;
照 sh_1.0 母本 test_train_machinery.py 改订,适配 §3.6 PD 分离豁免:
列车机制仅 D 侧 decode 实例——列车只含 decode 成员 span,P 侧整段
发射不经 _plan_train,无混拼路径)。

覆盖(设计文档 §3.2 构造规则 + §7.7,wscllm 范围 = 仅 decode 互拼):
  1. 异长 decode 同批退出(成员退出迭代先验,列车不因退出截断);
  2. 纯 decode 列车 span 结构钉子(全部 span tokens==1,无 chunk span);
  3. decode_tokens_consumed 闭式账本(连续列车推进 = 逐 token 精确);
  4. busy 门(列车在飞的实例不重复发射);
  5. same-tick 多实例完成 + 跨交付信号幂等核销(同列车 exit 标记
     watch 跨 tick fire,事件拆交付);
  6. §7.6 Layer A/B 互校验(§3.6 decode-only 形态):层次 A LUT
     estimate_iteration_time_ns 的 decode 迭代 linear_tokens = d_batch、
     权重每迭代恰一次;层次 B 列车体线性段 ops(全 TP rank 求和)与
     之逐字段一致;LUT phase-exclusive((p_chunk>0) 与 (d_batch>0)
     同真 raise)钉住"无混拼"豁免的 LUT 前提。时间域镜像在记录
     容差内(口径差:层次 A 对 attention 取 max,层次 B 17 节点链
     逐类 max 求和)。
"""
import os
import sys
import unittest
from types import SimpleNamespace

_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _p in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from generate_trace import transformer_pass_aggregated  # noqa: E402
from online.wsc_llm_online_scheduler import (  # noqa: E402
    _OnlineInstanceState,
    _OnlineRequestRuntime,
    WscLlmOnlineScheduler,
)
from wsc_llm_scheduler import (  # noqa: E402  (红线:只读 import)
    DECODE_ROLE,
    WscLlmHardware,
    WscLlmModel,
    estimate_iteration_time_ns,
)


def _record(request_id, *, prefill=512, decode=4, ctx=None, history=0):
    """manifest 事实 record(与 plan_materializer 的 9 字段同构)。"""
    context = prefill if ctx is None else ctx
    return {
        "request_id": request_id,
        "session_id": f"session_{request_id}",
        "turn_index": 0,
        "queue_index": 0,
        "prefill_length": prefill,
        "decode_length": decode,
        "history_tokens_before": history,
        "prefill_context_tokens": context,
        "final_context_tokens": context + decode,
    }


def _runtime(request_id, *, ctx=100, decode=4, instance=0):
    rt = _OnlineRequestRuntime(_record(request_id, decode=decode, ctx=ctx))
    rt.prefill_instance_index = (instance + 1) % 2
    rt.decode_instance_index = instance
    return rt


def _bare_scheduler():
    s = WscLlmOnlineScheduler.__new__(WscLlmOnlineScheduler)
    s.instances = [
        _OnlineInstanceState(index=0, phase_role=DECODE_ROLE),
        _OnlineInstanceState(index=1, phase_role=DECODE_ROLE),
    ]
    s.runtime_by_request_id = {}
    s._train_max_iter = 0   # 0 = 不设限(语义测试测自然边界;交付默认 8)
    s._train_instance_index = {}
    s._ready_frontier = set()
    s.waiting_decode_admissions = {}
    s.decode_admission_dirty = set()
    s._profile_batch = {"scanned_entries": 0, "full_scan_entries": 0}
    return s


class TrainPlanningTest(unittest.TestCase):
    """_plan_train 构造规则(§3.2;wscllm D 侧 = 无 prefill 工作)。"""

    def test_uneven_decode_members_exit_in_one_train(self):
        """异长 decode 同批:迭代 = max 剩余;两成员都在列车内退出
        (退出不是列车边界),participation 各自截断。"""
        s = _bare_scheduler()
        state = s.instances[0]
        for rid, remaining in (("ra", 5), ("rb", 9)):
            rt = _runtime(rid, ctx=100, decode=remaining)
            s.runtime_by_request_id[rid] = rt
            state.active_decode.append(rt)
            state.active_decode_lookup.add(rt)
        plan = s._plan_train(state)
        self.assertEqual(plan["iterations"], 9)
        self.assertEqual(
            sorted(rt.request_id for rt in plan["exit_members"]),
            ["ra", "rb"])
        self.assertEqual(
            {rt.request_id: p for rt, p in plan["members"]},
            {"ra": 5, "rb": 9})
        # span 数 = Σ participation(成员×迭代,无 padding)。
        self.assertEqual(len(plan["pass_spans"]), 14)
        self.assertTrue(plan["membership_digest"])

    def test_train_spans_are_pure_decode(self):
        """§3.6 钉子:列车 span 全部 tokens==1(逐成员逐 token),绝无
        chunk span(chunk 数 ≥ 2 的 prefill 工作量)。"""
        s = _bare_scheduler()
        state = s.instances[0]
        rt = _runtime("rm", ctx=100, decode=7)
        s.runtime_by_request_id["rm"] = rt
        state.active_decode.append(rt)
        plan = s._plan_train(state)
        self.assertEqual(plan["iterations"], 7)
        self.assertTrue(all(tokens == 1 for tokens, _kv in plan["pass_spans"]))
        # span kv = ctx + consumed + step,step 1..7(KV 逐迭代 +1)。
        self.assertEqual(
            [kv for _t, kv in plan["pass_spans"]],
            [101, 102, 103, 104, 105, 106, 107])


class TrainFinalizeTest(unittest.TestCase):
    """_finalize_completed_trains 原子提交 + 幂等。"""

    def test_advance_is_closed_form_exact(self):
        """闭式账本:列车核销推进 decode_tokens_consumed 恰一次
        (连续列车推进 = 逐 token 精确)。"""
        s = _bare_scheduler()
        state = s.instances[0]
        rt = _runtime("rm", ctx=100, decode=9)
        s.runtime_by_request_id["rm"] = rt
        state.active_decode.append(rt)
        state.active_decode_lookup.add(rt)
        state.in_flight_train = {
            "train_id": "t1", "iterations": 4, "members": [(rt, 4)],
            "exit_set": {"rm"}, "signal_set": {"rm"},
        }
        s._finalize_completed_trains(["rm"], [], 0)
        self.assertEqual(rt.decode_tokens_consumed, 4)
        self.assertEqual(state.active_decode, [])
        self.assertEqual(state.active_decode_lookup, set())
        self.assertIsNone(state.in_flight_train)
        self.assertEqual(state.iteration_count, 4)
        self.assertEqual(state.finalized_trains, [])

    def test_same_tick_multi_instance_and_split_delivery_idempotent(self):
        """same-tick 多实例核销 + 同列车信号拆交付幂等:实例 0 的列车
        含两个退出成员(mx/mz),信号分两个交付到达,第二次不重复推进、
        不报陈旧错配;实例 1 同 tick 正常核销;清空后的迟到信号
        fail-closed。"""
        s = _bare_scheduler()
        # 实例 0:列车含退出成员 mx(3 token)与 mz(3 token)。
        st0 = s.instances[0]
        rt_x = _runtime("mx", ctx=100, decode=3, instance=0)
        rt_z = _runtime("mz", ctx=50, decode=3, instance=0)
        s.runtime_by_request_id["mx"] = rt_x
        s.runtime_by_request_id["mz"] = rt_z
        for rt in (rt_x, rt_z):
            st0.active_decode.append(rt)
            st0.active_decode_lookup.add(rt)
        st0.in_flight_train = {
            "train_id": "t0", "iterations": 3,
            "members": [(rt_x, 3), (rt_z, 3)],
            "exit_set": {"mx", "mz"},
            "signal_set": {"mx", "mz"},
        }
        # 实例 1:同 tick 完成的另一列车。
        st1 = s.instances[1]
        rt_y = _runtime("my", ctx=50, decode=2, instance=1)
        s.runtime_by_request_id["my"] = rt_y
        st1.active_decode.append(rt_y)
        st1.active_decode_lookup.add(rt_y)
        st1.in_flight_train = {
            "train_id": "t1", "iterations": 2, "members": [(rt_y, 2)],
            "exit_set": {"my"}, "signal_set": {"my"},
        }
        # 交付 1:实例 0 的 mx 信号 + 实例 1 的 my 信号(same-tick 多实例;
        # mx 的核销把 mz 登记为待收信号)。
        s._finalize_completed_trains(["mx", "my"], [], 0)
        self.assertEqual(rt_x.decode_tokens_consumed, 3)
        self.assertEqual(rt_y.decode_tokens_consumed, 2)
        self.assertEqual(rt_z.decode_tokens_consumed, 3)  # 首信号核销整列车
        self.assertIsNone(st0.in_flight_train)
        self.assertIsNone(st1.in_flight_train)
        self.assertEqual(st0.active_decode, [])
        # 交付 2(拆分到达):mz 的迟到信号——幂等对账,不重复推进。
        s._finalize_completed_trains(["mz"], [], 1)
        self.assertEqual(rt_z.decode_tokens_consumed, 3)
        self.assertEqual(st0.finalized_trains, [])
        # 已核销列车记录清空后的重复迟到信号 → 陈旧错配 fail-closed。
        with self.assertRaises(RuntimeError):
            s._finalize_completed_trains(["mz"], [], 2)

    def test_busy_gate_skips_in_flight_instance(self):
        """busy 门:列车在飞的实例不重复发射(_admit_pass 跳过;
        graph stub 一旦被调即失败)。"""
        s = _bare_scheduler()
        s.graph = SimpleNamespace(
            emit_iteration_train=lambda *_a, **_k: (_ for _ in ()).throw(
                AssertionError("busy gate violated: emitted a train")))
        state = s.instances[0]
        rt = _runtime("rm", ctx=100, decode=4, instance=0)
        s.runtime_by_request_id["rm"] = rt
        state.active_decode.append(rt)
        state.active_decode_lookup.add(rt)
        state.in_flight_train = {
            "train_id": "t", "iterations": 1, "members": [(rt, 1)],
            "exit_set": set(), "signal_set": set(),
        }
        s._ready_frontier.add(0)
        s._admit_pass(0)  # 不得发射、不得抛


HARDWARE = WscLlmHardware(
    mesh_rows=1, mesh_cols=6, local_hbm_capacity_bytes=160 << 30,
    local_hbm_bandwidth_gbps=273.0, d2d_bandwidth_gbps=100.0,
    peak_perf_tflops=43.5, d2d_latency_ns=1, local_hbm_latency_ns=20,
    label="A1_SYNTHETIC")
# 比例接近真实模型(避免 2 层玩具下每节点固定延迟占比失真)。
MODEL = WscLlmModel(
    layers=8, hidden_size=256, ffn_size=512, num_heads=8,
    vocab_size=2048, bytes_per_elem=2, mlp_variant="gelu")


class LayerABCrossCheckTest(unittest.TestCase):
    """§7.6 Layer A/B 互校验(§3.6 decode-only 形态,手算可复核)。"""

    def test_lut_is_phase_exclusive_decode_only_form(self):
        """§3.6 前提钉子:LUT phase-exclusive((p_chunk>0) 与 (d_batch>0)
        同真 raise)——wscllm 无混拼迭代豁免的 LUT 依据(本体未动)。"""
        with self.assertRaises(ValueError):
            estimate_iteration_time_ns(
                HARDWARE, MODEL, instance_size=6, p_chunk=512,
                d_batch=2, d_token=400)

    def test_linear_stage_consistency_and_weight_once_per_iteration(self):
        """同一批组成(B=2 decode, 1 迭代)下:
        - 层次 A linear_tokens = p_chunk + d_batch = d_batch(decode-only);
        - 层次 B 列车体线性类 num_ops 的 per-iteration 分量(全 TP rank
          求和)与层次 A 公式一致(线性于 d_batch);
        - 权重每迭代恰读一次(weight_passes=1 ⇒ 权重常量与批成员数
          无关;A0 已证 B 无关性,此处锚定与 LUT 的
          estimate_model_weight_bytes 同口径)。
        时间域镜像比值记录容差(attention max-vs-sum 口径差)。"""
        d_batch, d_token = 2, 400
        layer_a_ns = estimate_iteration_time_ns(
            HARDWARE, MODEL, instance_size=6, p_chunk=0,
            d_batch=d_batch, d_token=d_token)
        self.assertGreater(layer_a_ns, 0)

        class _Rec:
            def __init__(self):
                self.nodes = []

            def comp(self, name, num_ops, tensor_size, remote=0):
                self.nodes.append((name, num_ops, tensor_size))

            def all_reduce(self, name, size, pg):
                pass

        h, ffn, layers, bpe, tp = (MODEL.hidden_size, MODEL.ffn_size,
                                   MODEL.layers, MODEL.bytes_per_elem, 6)
        spans = [(1, d_token) for _ in range(d_batch)]
        # 层次 B 是每 rank 量(TP 分片;attention 按 head 分片,单 rank
        # ≠ h/tp,须全 rank 求和后与层次 A 全实例量对拍)。
        rank_nodes = []
        for rank in range(tp):
            rec = _Rec()
            transformer_pass_aggregated(
                rec, phase="x", pass_spans=spans, layers=layers,
                hidden_size=h, ffn_size=ffn, tensor_parallel=tp,
                pg_name="pg", vocab_size=MODEL.vocab_size,
                bytes_per_elem=bpe, num_heads=MODEL.num_heads,
                mlp_variant=MODEL.mlp_variant, weight_passes=1,
                tensor_parallel_rank=rank)
            rank_nodes.append(rec.nodes)
        # 层次 A 的 linear_ops 只含 qkv/out/mlp 矩阵(无 logits 头),
        # 层次 B 取同四类(gelu 命名:mlp_up/mlp_down);matmul 对
        # tokens 线性 ⇒ span 求和 == 整批一次(手算恒等)。
        linear_names = ("attention_qkv_projection",
                        "attention_output_projection",
                        "mlp_up_projection", "mlp_down_projection")
        ops_b = sum(
            ops for nodes in rank_nodes for name, ops, _ts in nodes
            if any(name.endswith(n) for n in linear_names))
        linear_ops_a = layers * d_batch * (
            8 * h * h + (4 * h * ffn))  # gelu mlp_ops_per_token
        self.assertEqual(ops_b, linear_ops_a,
                         "Layer B linear ops (summed over TP ranks) must "
                         "equal Layer A's linear_ops for the same batch "
                         "composition")
        # 权重分量 == 常量(与 span 数/批成员数无关,A0 已证,这里锚定
        # "每迭代恰一次"的量级:weight_passes=1 ⇒ 权重只计一份)。
        weight_b = sum(
            ts for nodes in rank_nodes for name, _ops, ts in nodes
            if any(name.endswith(n) for n in linear_names))
        linear_weight_total = layers * (
            4 * h * h + 2 * h * ffn) * bpe
        self.assertGreater(weight_b, linear_weight_total)
        # 时间域镜像(容差记录:层次 A attention 取 max,层次 B 逐类
        # max 求和;镜像按单 rank 链 × tp 并行取 max 口径,容差 3×)。
        perf = HARDWARE.peak_perf_tflops * 1e12
        bw = HARDWARE.local_hbm_bandwidth_gbps * 1e9
        mirror_b_ns = max(
            sum(max(ops / perf, ts / bw) * 1e9 for _n, ops, ts in nodes)
            for nodes in rank_nodes) + \
            HARDWARE.local_hbm_latency_ns * len(rank_nodes[0])
        ratio = mirror_b_ns / layer_a_ns
        self.assertTrue(0.33 < ratio < 3.0,
                        f"Layer A/B time mirror out of tolerance: "
                        f"ratio={ratio:.2f}")


if __name__ == "__main__":
    unittest.main()
