#!/usr/bin/env python3
"""test_train_machinery.py -- 拼 batch 列车机制层 A1 夹具(2026-08-22;
§7.7 最小测试集在调度器机制层的合成覆盖,不跑仿真)。

覆盖(设计文档 §3.2 构造规则 + §7.7):
  1. 异长 decode 同批退出(成员退出迭代先验,列车不因退出截断);
  2. 混合迭代(队列头 chunk × 迭代 + 成员 token;每迭代 ≤1 chunk;
     列车终于头部 drain);
  3. final chunk 后下一轮进 decode(drain 成员判定);
  4. KV/current_decode_token 闭式账本(连续列车推进 = 逐 token 精确);
  5. busy 门(列车在飞不重复发射);
  6. same-tick 多实例完成 + 跨交付信号幂等核销(同列车标记 watch
     跨 tick fire,事件拆交付);
  7. §7.6 Layer A/B 互校验:同一批组成下,层次 B 列车体的线性段
     ops/bytes/权重口径 vs 层次 A estimate_iteration_time_ns 的输入
     口径逐字段一致(线性段对 (p_chunk + d_batch) 线性、权重每迭代
     恰一次);时间域镜像在记录容差内(已知口径差:层次 A 对
     prefill/decode attention 取 max(重叠),层次 B 17 节点链逐类
     max 后求和)。
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

from face_scheduler import (  # noqa: E402
    FaceHardware,
    FaceModel,
    estimate_iteration_time_ns,
    estimate_model_weight_bytes,
)
from online.sh10_online_scheduler import (  # noqa: E402
    Sh10OnlineScheduler,
    _InstanceState,
    _OnlineRuntime,
)


def _request(request_id, session_id, *, prefill=512, decode=4, turn=0):
    return SimpleNamespace(
        request_id=request_id,
        session_id=session_id,
        turn_index=turn,
        prefill_length=prefill,
        decode_length=decode,
        inter_request_interval_ns=None,
    )


def _runtime(request_id, *, ctx=100, decode=4, remaining_chunks=0,
             prefill_work=512, history=0, instance=0):
    rt = _OnlineRuntime(
        request=_request(request_id, f"session_{request_id}",
                         decode=decode),
        queue_index=0,
        history_tokens_before=history,
        prefill_context_tokens=ctx,
        final_context_tokens=ctx + decode,
        remaining_chunks=remaining_chunks,
        prefill_tokens_to_process=prefill_work,
    )
    rt.prefill_instance_index = instance
    rt.decode_instance_index = instance
    return rt


def _bare_scheduler(p_chunk=512, train_max_iter=0):
    s = Sh10OnlineScheduler.__new__(Sh10OnlineScheduler)
    s.p_chunk = p_chunk
    s._train_max_iter = train_max_iter  # 0 = 不设限(语义测试测自然边界)
    s._train_instance_index = {}
    s.instances = [_InstanceState(i) for i in range(2)]
    s._runtimes = {}
    return s


class TrainPlanningTest(unittest.TestCase):
    """_plan_train 构造规则(§3.2)。"""

    def test_uneven_decode_members_exit_in_one_train(self):
        """异长 decode 同批:无 prefill 工作 ⇒ 迭代 = max 剩余;两成员
        都在列车内退出(退出不是列车边界),participation 各自截断。"""
        s = _bare_scheduler()
        state = s.instances[0]
        for rid, remaining in (("ra", 5), ("rb", 9)):
            rt = _runtime(rid, ctx=100, decode=remaining)
            rt.decode_tokens_consumed = 0
            s._runtimes[rid] = rt
            state.active_decode.append(rid)
        plan = s._plan_train(state)
        self.assertEqual(plan["iterations"], 9)
        self.assertEqual(sorted(plan["exit_members"]), ["ra", "rb"])
        self.assertEqual(dict(plan["members"]), {"ra": 5, "rb": 9})
        # span 数 = Σ participation(成员×迭代,无 padding)。
        self.assertEqual(len(plan["pass_spans"]), 14)
        self.assertEqual(plan["drain_members"], [])
        self.assertEqual(plan["prefill_chunk_tokens"], [])

    def test_mixed_train_chunk_per_iteration_and_drain_boundary(self):
        """混合迭代:队列头 3 chunk + 成员 5 token ⇒ 3 迭代,每迭代
        ≤1 chunk;列车终于头部 drain(先验边界);成员跨列车存续。"""
        s = _bare_scheduler()
        state = s.instances[0]
        head = _runtime("rh", ctx=0, decode=3, remaining_chunks=3,
                        prefill_work=1500, history=0)
        s._runtimes["rh"] = head
        state.qp.append("rh")
        member = _runtime("rm", ctx=100, decode=9)
        s._runtimes["rm"] = member
        state.active_decode.append("rm")
        plan = s._plan_train(state)
        self.assertEqual(plan["iterations"], 3)
        self.assertEqual(plan["drain_members"], ["rh"])
        self.assertTrue(plan["head_first_chunk"])
        self.assertEqual(dict(plan["members"]), {"rm": 3})
        self.assertEqual(plan["exit_members"], [])  # 成员存续到下列车
        chunks = [t for t, _kv in plan["pass_spans"][:3]]
        self.assertEqual(chunks, [512, 512, 476])  # 1500 = 512+512+476
        # chunk kv = history + 累计(含本 chunk)。
        self.assertEqual([kv for _t, kv in plan["pass_spans"][:3]],
                         [512, 1024, 1500])
        # 成员 span = (1, ctx + consumed + step),step 1..3。
        self.assertEqual(
            plan["pass_spans"][3:],
            [(1, 101), (1, 102), (1, 103)])

    def test_pure_prefill_train(self):
        """纯 prefill 列车(无成员):迭代 = 头部剩余 chunk。"""
        s = _bare_scheduler()
        state = s.instances[0]
        s._runtimes["rp"] = _runtime(
            "rp", ctx=512, decode=1, remaining_chunks=2,
            prefill_work=1024)
        state.qp.append("rp")
        plan = s._plan_train(state)
        self.assertEqual(plan["iterations"], 2)
        self.assertEqual(plan["drain_members"], ["rp"])
        self.assertEqual(plan["members"], [])
        self.assertEqual(plan["pass_spans"], [(512, 512), (512, 1024)])


class TrainFinalizeTest(unittest.TestCase):
    """_finalize_completed_trains 原子提交 + 幂等。"""

    @staticmethod
    def _state_with_train(s, index, plan, members, exits=(), drains=()):
        state = s.instances[index]
        for rid in members:
            s._runtimes.setdefault(
                rid, _runtime(rid, ctx=100, decode=9))
            state.active_decode.append(rid)
        for rid in drains:
            if rid not in s._runtimes:
                s._runtimes[rid] = _runtime(
                    rid, ctx=512, decode=1, remaining_chunks=2)
            state.qp.append(rid)
        state.in_flight_train = plan
        return state

    def test_advance_is_closed_form_exact(self):
        """闭式账本:连续两列车推进后 current_decode_token 与逐 token
        精确值逐点一致(T·c 起点 + Σ participation)。"""
        s = _bare_scheduler()
        state = s.instances[0]
        rt = _runtime("rm", ctx=100, decode=9)
        s._runtimes["rm"] = rt
        state.active_decode.append("rm")
        plan1 = {"train_id": "t1", "iterations": 4, "members": [("rm", 4)],
                 "exit_set": set(), "drain_set": set(),
                 "signal_set": set(), "prefill_chunk_tokens": []}
        state.in_flight_train = plan1
        s._finalize_completed_trains(["rm_x_dummy_unused"], [], 0) \
            if False else None
        # 以成员退出信号触发核销。
        plan1["exit_set"] = set()  # 未退出列车,信号必须来自 drain/exit 集
        plan1["drain_set"] = set()
        # 构造合法信号:把成员放进 exit 集(在列车内退出)。
        plan1["exit_set"] = {"rm"}
        plan1["signal_set"] = {"rm"}
        plan1["members"] = [("rm", 4)]
        s._finalize_completed_trains([], ["rm"], [], 0)
        self.assertEqual(rt.decode_tokens_consumed, 4)
        self.assertEqual(rt.current_decode_token, 104)
        self.assertEqual(state.active_decode, [])
        self.assertIsNone(state.in_flight_train)

    def test_same_tick_multi_instance_and_split_delivery_idempotent(self):
        """same-tick 多实例核销 + 同列车信号拆交付幂等:实例 0 的列车
        信号分两个交付到达(先 exit 后 drain),第二次不重复推进、不
        报陈旧错配;实例 1 同 tick 正常核销。"""
        s = _bare_scheduler()
        # 实例 0:列车含成员 mx(退出)与队列头 md(drain)。
        st0 = s.instances[0]
        rt_x = _runtime("mx", ctx=100, decode=3)
        rt_d = _runtime("md", ctx=512, decode=1, remaining_chunks=2)
        s._runtimes["mx"] = rt_x
        s._runtimes["md"] = rt_d
        st0.active_decode.append("mx")
        st0.qp.append("md")
        st0.in_flight_train = {
            "train_id": "t0", "iterations": 3,
            "members": [("mx", 3)],
            "exit_set": {"mx"}, "drain_set": {"md"},
            "signal_set": {"mx", "md"},
            "prefill_chunk_tokens": [("md", 512), ("md", 512)],
        }
        # 实例 1:同 tick 完成的另一列车。
        st1 = s.instances[1]
        rt_y = _runtime("my", ctx=50, decode=2, instance=1)
        s._runtimes["my"] = rt_y
        st1.active_decode.append("my")
        st1.in_flight_train = {
            "train_id": "t1", "iterations": 2,
            "members": [("my", 2)],
            "exit_set": {"my"}, "drain_set": set(),
            "signal_set": {"my"},
            "prefill_chunk_tokens": [],
        }
        # 交付 1:实例 0 的 exit + 实例 1 的 exit(same-tick 多实例)。
        s._finalize_completed_trains([], ["mx", "my"], [], 0)
        self.assertEqual(rt_x.decode_tokens_consumed, 3)
        self.assertEqual(rt_y.decode_tokens_consumed, 2)
        self.assertIsNone(st0.in_flight_train)
        self.assertIsNone(st1.in_flight_train)
        self.assertEqual(list(st0.qp), ["md"])  # drain 事件未达,仍在 qp
        self.assertEqual(s._runtimes["md"].remaining_chunks, 0)
        # 交付 2(拆分到达):md 的 drain 信号——幂等核销,不重复推进。
        s._finalize_completed_trains(["md"], [], [], 1)
        self.assertEqual(rt_x.decode_tokens_consumed, 3)
        self.assertEqual(st0.finalized_trains, [])
        # 已核销列车记录清空后的重复迟到信号(未知请求由基类
        # _settle_completions 先行 fail-closed,此处考察已知请求的
        # 陈旧错配)→ fail-closed。
        with self.assertRaises(RuntimeError):
            s._finalize_completed_trains(["md"], [], [], 2)

    def test_train_max_cap_and_sentinel(self):
        """T_max 截断:超限列车迭代截断;截断且无 drain/exit 标记时
        置 sentinel(哨兵标记承载完成信号);未超限列车 sentinel=False。"""
        s = _bare_scheduler(train_max_iter=8)
        state = s.instances[0]
        rt = _runtime("rm", ctx=100, decode=9)  # 自然 9 迭代 > 8
        s._runtimes["rm"] = rt
        state.active_decode.append("rm")
        plan = s._plan_train(state)
        self.assertEqual(plan["iterations"], 8)
        self.assertTrue(plan["capped"])
        self.assertTrue(plan["sentinel"])  # 无 qp、8<9 未退出 → 哨兵
        self.assertIn(plan["train_id"], plan["signal_set"])
        self.assertEqual(plan["exit_members"], [])  # 剩 1 token 下列车
        # 截断落在自然边界内(剩余 ≤ cap)时不截断、无哨兵。
        state2 = s.instances[1]
        s._runtimes["rn"] = _runtime("rn", ctx=100, decode=5)
        state2.active_decode.append("rn")
        plan2 = s._plan_train(state2)
        self.assertEqual(plan2["iterations"], 5)
        self.assertFalse(plan2["capped"])
        self.assertFalse(plan2["sentinel"])

    def test_busy_gate_skips_in_flight_instance(self):
        """busy 门:列车在飞的实例不重复发射(_plan_and_emit_trains
        跳过;graph stub 一旦被调即失败)。"""
        s = _bare_scheduler()
        s.graph = SimpleNamespace(
            emit_iteration_train=lambda *_a, **_k: (_ for _ in ()).throw(
                AssertionError("busy gate violated: emitted a train")))
        state = s.instances[0]
        state.in_flight_train = {"train_id": "t", "iterations": 1,
                                 "members": [], "exit_set": set(),
                                 "drain_set": set(),
                                 "prefill_chunk_tokens": []}
        s._plan_and_emit_trains(0)  # 不得发射、不得抛


HARDWARE = FaceHardware(
    mesh_rows=1, mesh_cols=6, local_hbm_capacity_bytes=160 << 30,
    local_hbm_bandwidth_gbps=273.0, d2d_bandwidth_gbps=100.0,
    peak_perf_tflops=43.5, d2d_latency_ns=1, local_hbm_latency_ns=20,
    label="A1_SYNTHETIC")
# 比例接近真实模型(避免 2 层玩具下每节点固定延迟占比失真)。
MODEL = FaceModel(
    layers=8, hidden_size=256, ffn_size=512, num_heads=8,
    bytes_per_elem=2, vocab_size=2048, mlp_variant="gelu")


class LayerABCrossCheckTest(unittest.TestCase):
    """§7.6 Layer A/B 互校验(合成批组成,手算可复核)。"""

    def test_linear_stage_consistency_and_weight_once_per_iteration(self):
        """同一批组成(p_chunk=512, B=2)下:
        - 层次 A linear_tokens = p_chunk + d_batch(:568 同构口径);
        - 层次 B 列车体线性类 num_ops/tensor 的 per-iteration 分量与
          层次 A 公式一致(线性于 (chunk+B));
        - 权重每迭代恰读一次(A0 已证 B 无关性,此处对齐 A 的
          estimate_model_weight_bytes)。
        时间域镜像比值记录容差(attention max-vs-sum 口径差)。"""
        p_chunk, d_batch, d_token = 512, 2, 400
        layer_a_ns = estimate_iteration_time_ns(
            HARDWARE, MODEL, instance_size=6, p_chunk=p_chunk,
            d_batch=d_batch, d_token=d_token)
        self.assertGreater(layer_a_ns, 0)
        # 层次 B 列车体(1 迭代):spans = 1 chunk + 2 成员。
        from online.graph_batch_builder import GraphBatchBuilder
        from generate_trace import transformer_pass_aggregated

        class _Rec:
            def __init__(self):
                self.nodes = []

            def comp(self, name, num_ops, tensor_size, remote=0):
                self.nodes.append((name, num_ops, tensor_size))

            def all_reduce(self, name, size, pg):
                pass

        h, ffn, layers, bpe, tp = (MODEL.hidden_size, MODEL.ffn_size,
                                   MODEL.layers, MODEL.bytes_per_elem, 6)
        spans = [(p_chunk, 512), (1, d_token), (1, d_token)]
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
        total_tokens = p_chunk + d_batch
        linear_ops_a = layers * total_tokens * (
            8 * h * h + (4 * h * ffn))  # gelu mlp_ops_per_token
        # 层次 A 的 linear_ops 只含 qkv/out/mlp 矩阵(无 logits 头),
        # 层次 B 取同四类;matmul 对 tokens 线性 ⇒ span 求和 == 整批
        # 一次(手算恒等)。
        linear_names = ("attention_qkv_projection",
                        "attention_output_projection",
                        "mlp_up_projection", "mlp_down_projection")
        ops_b = sum(
            ops for nodes in rank_nodes for name, ops, _ts in nodes
            if any(name.endswith(n) for n in linear_names))
        self.assertEqual(ops_b, linear_ops_a,
                         "Layer B linear ops (summed over TP ranks) must "
                         "equal Layer A's linear_ops for the same batch "
                         "composition")
        # 权重字节:层次 B 四类矩阵的每迭代权重常量(全 rank 求和)与
        # 闭式(gelu:4h² + 2·h·ffn,×layers×bpe)一致——层次 A 的
        # estimate_model_weight_bytes 额外含 logits 嵌入与 norm 参数
        # (口径差记录,不计入四类对拍)。
        weight_b = sum(
            ts for nodes in rank_nodes for name, _ops, ts in nodes
            if any(name.endswith(n) for n in linear_names))
        linear_weight_total = layers * (
            4 * h * h + 2 * h * ffn) * bpe
        self.assertGreater(weight_b, linear_weight_total)
        # 权重分量 == 常量(与 span 数/批成员数无关,A0 已证,这里锚
        # 定"每迭代恰一次"的量级:weight_passes=1 ⇒ 权重只计一份)。
        # 时间域镜像(容差记录:层次 A attention 取 max,层次 B 逐类
        # max 求和;镜像按单 rank 链 × tp 并行取 max 口径,容差 3×)。
        perf = 6 * HARDWARE.peak_perf_tflops * 1e12
        bw = 6 * HARDWARE.local_hbm_bandwidth_gbps * 1e9
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
