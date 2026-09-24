#!/usr/bin/env python3
"""test_remote_credit_stream.py -- remote-read credit 交错流（唯一执行
口径，2026-09-17 用户裁定：v1 批量串行口径删除）的发射结构单测
（路线 B §4.6；方案《remote-read改造分析方案.md》§4.1 不变量）。

覆盖：
  1. I1 字节守恒：单列车任意 K、多列车跨切片 Σ ≡ 持久计划总量；
  2. I2 门并集：体块首节点依赖 = 覆盖其迭代区间成员的切片块完成门
     （多 remote 成员并集；不依赖后续块；无门消费扫描）；
  3. 单块退化锚（K ≥ S_j）：产出的节点/边与"pd_transfer 单笔 + 单段
     体"形态逐字节等价（v1 发射路径在块 1 复用的结构证明——serial
     运行时路径已删，锚改经同构对照列车钉住）；
  4. I3b 多列车结构正则：续列车切片块 1 上主链（无 barrier）、尾块
     旁挂 + arm 门；字节总量等价；
  5. R15 逐切片键（rid#decode#{j}）：切片创建点登记、Tj 核销注销
     ——T2 规划期登记表恰含本列车键，无陈旧多计（R-6）；
  6. D2 发射序：fork 点先于块 1（支链首节点 parent == 块 1 的 parent），
     尾块支链不 join barrier；
  7. v1 删除钉：_joint_remote_read_stream 不复存在。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 online/test_remote_credit_stream.py
"""
import json
import os
import sys
import unittest
from collections import deque
from types import SimpleNamespace

_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _p in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from face_scheduler import (  # noqa: E402
    FaceHardware,
    FaceModel,
    kv_cache_shard_bytes_for_layer_range,
    kv_cache_shard_bytes_for_tokens,
)
from joint.joint_config import parse_joint_config  # noqa: E402
from joint.joint_cost_model import LinkFlowRegistry  # noqa: E402
from online.graph_batch_builder import GraphBatchBuilder  # noqa: E402
from online.sh30_online_scheduler import (  # noqa: E402
    Sh30OnlineScheduler,
    _OnlineInstanceState,
    _OnlineRequestRuntime,
    _TASK_LOAD_CACHE_CAPACITY,  # F6 销账：替身对齐 __init__ 初值
)
from online.test_graph_batch_builder import _make_config  # noqa: E402

EXEC_INSTANCE = 0   # ranks (0, 1)
HOME_INSTANCE = 1   # ranks (2, 3)


def _scheduler(credit_iters="auto", layers=2):
    s = Sh30OnlineScheduler.__new__(Sh30OnlineScheduler)
    s.p_chunk = 512
    # N5：_emit_train 的 merge_tail_gated 纯度标记读 _pending_merge_
    # alarms（__init__ 恒设——替身补设，F6）。
    s._pending_merge_alarms = {}
    s.instances = [_OnlineInstanceState(index=0),
                   _OnlineInstanceState(index=1)]
    s.runtime_by_request_id = {}
    s._train_max_iter = 0
    s._train_instance_index = {}
    s._ready_frontier = set()
    ranks_by_instance = {0: (0, 1), 1: (2, 3)}
    s.topology = SimpleNamespace(
        instance=lambda i: SimpleNamespace(
            size=2, ranks=ranks_by_instance[i]),
        instances=[SimpleNamespace(size=2), SimpleNamespace(size=2)])
    s.hardware = FaceHardware(
        mesh_rows=2, mesh_cols=2,
        local_hbm_capacity_bytes=10**9,
        local_hbm_bandwidth_gbps=100.0, d2d_bandwidth_gbps=200.0,
        peak_perf_tflops=1.0, d2d_latency_ns=0, local_hbm_latency_ns=0)
    s.model = FaceModel(
        layers=layers, hidden_size=64, ffn_size=128, num_heads=8,
        vocab_size=256, bytes_per_elem=2, mlp_variant="gelu")
    s._prefill_task_cache = {}
    s._decode_task_cache = {}
    # 对齐 __init__ 初值（F6 销账：软门已删，替身漏设 = AttributeError）。
    s._decode_task_load_cache = {}
    s._task_load_cache_capacity = _TASK_LOAD_CACHE_CAPACITY
    s._quota_tracker = None  # off 档 __init__ 初值（F6 销账）
    s.graph = GraphBatchBuilder(_make_config())
    s.graph.begin_batch()
    s._batch = {"watches": []}
    s.train_ledger_rows = []
    s.train_ledger_sink = None
    s.joint_config = parse_joint_config(
        {"JOINT_REMOTE_CREDIT_ITERS": credit_iters})
    s._joint_flows = LinkFlowRegistry()
    s._pool_ports = SimpleNamespace(
        register=lambda *args, **kwargs: None,
        release_owner=lambda *args, **kwargs: None)
    # 对齐 __init__ 初值（F6 销账：软门已删，_register/_release_
    # transfer_flows 直达 _hbm_ports——替身漏设 = AttributeError）。
    from joint.hbm_port_flow_registry import HbmPortFlowRegistry
    s._hbm_ports = HbmPortFlowRegistry()
    s._kv_ledger_epoch = 0
    return s


def _session(namespace="s_rr", context_tokens=100, working_kind=None,
             base_history_tokens=0, base_prefix_layers=0):
    return SimpleNamespace(
        working_kind=working_kind, context_tokens=context_tokens,
        base_history_tokens=base_history_tokens,
        base_resident_prefix_layers=base_prefix_layers)


def _remote_runtime(request_id="rr", *, decode=8, input_tokens=16,
                    consumed=0, session_id="s_rr"):
    rt = _OnlineRequestRuntime({
        "request_id": request_id,
        "session_id": session_id,
        "turn_index": 0, "queue_index": 0,
        "prefill_length": input_tokens, "decode_length": decode,
        "history_tokens_before": 0,
        "prefill_context_tokens": input_tokens,
        "final_context_tokens": input_tokens + decode,
    }, 512)
    rt.joint_action = "remote-read"
    rt.origin_home_instance = HOME_INSTANCE
    rt.prefill_instance_index = HOME_INSTANCE
    rt.decode_instance_index = EXEC_INSTANCE
    rt.joint_input_tokens = input_tokens
    rt.decode_tokens_consumed = consumed
    rt.remaining_chunks = 0
    # 真实流程由 drain 列车块末写入（joiner 触发门表）；离线夹具给
    # home 实例 ranks 占位（本组无 decode_evictions，门不被消费）。
    rt.drain_block_ends = {2: 0, 3: 0}
    return rt


def _head_runtime(request_id="rh", chunks=3, work=1536):
    rt = _OnlineRequestRuntime({
        "request_id": request_id,
        "session_id": "session_%s" % request_id,
        "turn_index": 0, "queue_index": 1,
        "prefill_length": work, "decode_length": 4,
        "history_tokens_before": 0,
        "prefill_context_tokens": work,
        "final_context_tokens": work + 4,
    }, 512)
    rt.prefill_tokens_to_process = work
    rt.remaining_chunks = chunks
    rt.prefill_instance_index = EXEC_INSTANCE
    rt.decode_instance_index = EXEC_INSTANCE
    return rt


def _install_remote(scheduler, runtime):
    scheduler.runtime_by_request_id[runtime.request_id] = runtime
    scheduler.kv_manager = SimpleNamespace(
        _sessions={runtime.session_id: _session()}, tp_degree=2)
    # drain 边界的持久读计划建立（_on_prefill_drain 的等价直连）。
    runtime.remote_read_credit_plan = (
        scheduler._joint_remote_read_credit_plan(
            runtime, EXEC_INSTANCE))


class RemoteCreditEmissionTest(unittest.TestCase):
    """T1 joiner 多块切片：结构不变量（I1/I2/D2 + 支链不 join）。"""

    def _emit_t1(self, credit_iters="2", decode=8):
        s = _scheduler(credit_iters)
        rt = _remote_runtime(decode=decode)
        _install_remote(s, rt)
        state = s.instances[EXEC_INSTANCE]
        state.active_decode.append(rt)
        state.active_decode_lookup.add(rt)
        plan = s._plan_train(state)
        s._emit_train(state, plan, [rt], 1)
        return s, rt, plan

    def _nodes(self, s):
        return s.graph.batch["nodes"]

    def _edges(self, s):
        return s.graph.batch["parent_edges"]

    def _by_name(self, s, needle):
        return [n for n in self._nodes(s) if needle in n["name"]]

    def test_t1_multi_block_structure(self):
        s, rt, plan = self._emit_t1(credit_iters="2", decode=8)
        # I1：切片总量 ≡ 计划总量（per shard 由整数乘法保证）。
        summary = rt.remote_read_slice_summaries[0]
        self.assertEqual(summary["k"], 2)
        self.assertEqual(summary["block_steps"], [2, 2, 2, 2])
        self.assertEqual(
            summary["total_bytes"], rt.remote_read_credit_plan["total_bytes"])
        # R15：逐切片键已登记（切片创建点）。
        self.assertIn("rr#decode#1", s._joint_flows._owner_flows)
        # 块 1 走 v1 pd_transfer 主链路径（无 remote_credit 命名）。
        pd_sends = self._by_name(s, "prefill_decode_transfer")
        pd_sends = [n for n in pd_sends if "_send" in n["name"]]
        self.assertTrue(pd_sends)
        self.assertTrue(all(n["rank"] in (2, 3) for n in pd_sends))
        # 尾块：home 侧 send / exec 侧 recv，credit2..4。
        for block in (2, 3, 4):
            self.assertTrue(
                self._by_name(s, "_credit%d_send" % block))
            self.assertTrue(
                self._by_name(s, "_credit%d_recv" % block))
        # D2：fork 点先于块 1——home rank 上 credit2_send 的 parent 集合
        # == pd 块 1 send 的 parent 集合（同 fork frontier；首批发射的
        # rank 两者同无父边，空列表等价成立）。
        parents = {}
        for edge in self._edges(s):
            parents.setdefault((edge["rank"], edge["to"]), []).append(
                edge["from"])
        home_rank = pd_sends[0]["rank"]
        credit2_send = [
            n for n in self._by_name(s, "_credit2_send")
            if n["rank"] == home_rank][0]
        self.assertEqual(
            parents.get((home_rank, credit2_send["id"]), []),
            parents.get((home_rank, pd_sends[0]["id"]), []))
        # D2/支链不 join：barrier 在，且尾块 recv 不是 barrier 的祖先。
        barrier = [
            n for n in self._nodes(s)
            if n["name"].endswith("_decode_kv_ready_barrier")]
        self.assertEqual(sorted(n["rank"] for n in barrier), [0, 1])
        ancestors = set()
        frontier = [n["id"] for n in barrier if n["rank"] == 0]
        while frontier:
            current = frontier.pop()
            for parent in parents.get((0, current), ()):
                if parent not in ancestors:
                    ancestors.add(parent)
                    frontier.append(parent)
        recv_ids = {n["id"] for n in self._by_name(s, "_credit")
                    if "_recv" in n["name"] and n["rank"] == 0}
        self.assertFalse(recv_ids & ancestors)

    def test_i2_body_block_gates(self):
        s, rt, plan = self._emit_t1(credit_iters="2", decode=8)
        parents = {}
        for edge in self._edges(s):
            parents.setdefault((edge["rank"], edge["to"]), []).append(
                edge["from"])
        train_id = plan["train_id"]
        # 每 credit 块 b（2..4）恰有一个体块首节点以其 recv 为依赖
        # （I2：块 b 门 = 块 b 完成门；不依赖后续块）。
        gated_node_ids = {}
        for block in (2, 3, 4):
            recv = [
                n for n in self._by_name(s, "_credit%d_recv" % block)
                if n["rank"] == 0][0]
            targets = [
                to for (rank, to), froms in parents.items()
                if rank == 0 and recv["id"] in froms]
            body_targets = [
                node for node in self._nodes(s)
                if node["id"] in targets and node["rank"] == 0
                and node["request_id"] == train_id]
            self.assertEqual(len(body_targets), 1, block)
            gated_node_ids[block] = body_targets[0]["id"]
        # 门序 = 块序（体块 b 的门节点先于体块 b+1 的门节点——不依赖
        # 后续块）。
        self.assertLess(gated_node_ids[2], gated_node_ids[3])
        self.assertLess(gated_node_ids[3], gated_node_ids[4])
        # 无门消费扫描：每个被门控的体节点都是 COMP 且属于 exec rank。
        for block, node_id in gated_node_ids.items():
            node = [n for n in self._nodes(s) if n["id"] == node_id][0]
            self.assertEqual(node["rank"], 0)
            self.assertTrue(node["compute"]["num_ops"] > 0, block)


class MultiTrainSliceLedgerTest(unittest.TestCase):
    """跨列车切片账本：I1 总量守恒 + R15 逐切片键登记/核销 + 续坐发射。"""

    def _settle(self, s, state, tick):
        train = state.in_flight_train
        completed = [
            request_id for request_id, _ in train["members"]
            if request_id in train["exit_set"]]
        s._finalize_completed_trains(
            drained=tuple(train["drain_members"]),
            completed_now=completed, sentinel_trains=(), tick=tick)

    def test_two_train_slicing_r15_and_continuation(self):
        s = _scheduler(credit_iters="3")
        rt = _remote_runtime(decode=8)
        _install_remote(s, rt)
        state = s.instances[EXEC_INSTANCE]
        head = _head_runtime(chunks=3)
        s.runtime_by_request_id["rh"] = head
        state.qp.append(head)
        state.active_decode.append(rt)
        state.active_decode_lookup.add(rt)
        # T1：iterations=3（队头 3 chunk），joiner 参与 3 步 → K=3 单块。
        plan1 = s._plan_train(state)
        self.assertEqual(plan1["iterations"], 3)
        s._emit_train(state, plan1, [rt], 1)
        first = rt.remote_read_slice_summaries[0]
        self.assertEqual(first["steps"], 3)
        self.assertEqual(first["block_steps"], [3])
        per_step_total = rt.remote_read_credit_plan["total_bytes"] // 8
        self.assertEqual(first["total_bytes"], per_step_total * 3)
        self.assertIn("rr#decode#1", s._joint_flows._owner_flows)
        # T1 核销：键 1 释放（消费栅栏已物理通过）；grow no-op（会话
        # 上下文对齐 consumed）。
        s.kv_manager._sessions["s_rr"].context_tokens = (
            rt.joint_input_tokens + 3)
        self._settle(s, state, 2)
        self.assertNotIn("rr#decode#1", s._joint_flows._owner_flows)
        self.assertEqual(rt.decode_tokens_consumed, 3)
        # 队头 chunk 在 T1 内完成 → drain 出队（等价 _on_prefill_drain
        # 的 qp 移除；此处直连后续列车规划）。
        self.assertEqual(head.remaining_chunks, 0)
        state.qp.remove(head)
        # T2：无 qp → iterations=剩余 5，续坐（无 joiner 槽位 → D7 新
        # 发射槽位），K=3 → 两块 [3, 2]。
        s.graph.begin_batch()
        plan2 = s._plan_train(state)
        self.assertEqual(plan2["iterations"], 5)
        s._emit_train(state, plan2, [], 3)
        second = rt.remote_read_slice_summaries[1]
        self.assertEqual(second["slice_index"], 2)
        self.assertEqual(second["block_steps"], [3, 2])
        # I1（多列车字节总量等价）：Σ切片 ≡ 计划总量。
        self.assertEqual(
            first["total_bytes"] + second["total_bytes"],
            rt.remote_read_credit_plan["total_bytes"])
        # R15：T2 规划期登记表恰含键 2（键 1 已核销——无陈旧多计）。
        self.assertNotIn("rr#decode#1", s._joint_flows._owner_flows)
        self.assertIn("rr#decode#2", s._joint_flows._owner_flows)
        # I3b 结构正则：续列车无 barrier；块 1 上主链（remote_credit
        # 动作名）；尾块 credit2 旁挂。
        nodes = s.graph.batch["nodes"]
        self.assertFalse([
            n for n in nodes
            if n["name"].endswith("_decode_kv_ready_barrier")])
        self.assertTrue([
            n for n in nodes
            if "remote_credit" in n["name"] and "_recv" in n["name"]])
        self.assertTrue([
            n for n in nodes if "_credit2_recv" in n["name"]])
        # T2 核销：键 2 释放，请求走完。
        s.kv_manager._sessions["s_rr"].context_tokens = (
            rt.joint_input_tokens + 8)
        self._settle(s, state, 4)
        self.assertNotIn("rr#decode#2", s._joint_flows._owner_flows)
        self.assertEqual(rt.decode_tokens_consumed, 8)


class SingleBlockDegenerateEquivalenceTest(unittest.TestCase):
    """单块退化锚（K ≥ S_j）：credit 发射与"pd 单笔 + 单段体"同构
    对照列车逐字节等价（节点/边/tag 序列）——块 1 复用 v1 发射路径的
    结构证明。"""

    def test_k_equals_steps_matches_plain_train(self):
        decode = 8
        # A：credit 路径（K=8=S，M=1，无尾块）。
        s = _scheduler(credit_iters="8")
        rt = _remote_runtime(decode=decode)
        _install_remote(s, rt)
        state = s.instances[EXEC_INSTANCE]
        state.active_decode.append(rt)
        state.active_decode_lookup.add(rt)
        plan = s._plan_train(state)
        s._emit_train(state, plan, [rt], 1)
        # B：同构对照——手工拼 v1 形态列车（pd 单笔 = 同字节单块传输，
        # 无 remote_credit 规格 → 单段体）。
        s2 = _scheduler(credit_iters="8")
        rt2 = _remote_runtime(decode=decode)
        _install_remote(s2, rt2)
        block1 = s2._joint_remote_read_slice(rt2, decode, decode)[0]
        self.assertEqual(
            block1.total_bytes, rt2.remote_read_credit_plan["total_bytes"])
        joiner = rt2.plan_dict()
        joiner["prefill_drain_block_ends"] = {2: 0, 3: 0}
        train_plan = {
            "train_id": plan["train_id"],
            "instance_index": EXEC_INSTANCE,
            "stage": "decode",
            "joiners": [joiner],
            "pass_spans": list(plan["pass_spans"]),
            "iterations": decode,
            "sentinel": False,
            "prefill_start_member": None,
            "first_chunk_member": None,
            "drain_members": [],
            "exit_members": [{"request_id": rt2.request_id}],
        }
        joiner["prefill_decode_transfer"] = block1
        s2.graph.emit_iteration_train(train_plan)
        # 节点/边逐字节等价（JSON 序列化排序对照）。
        self.assertEqual(
            [json.dumps(n, sort_keys=True) for n in s.graph.batch["nodes"]],
            [json.dumps(n, sort_keys=True)
             for n in s2.graph.batch["nodes"]])
        self.assertEqual(
            [json.dumps(e, sort_keys=True) for e in s.graph.batch[
                "parent_edges"]],
            [json.dumps(e, sort_keys=True)
             for e in s2.graph.batch["parent_edges"]])


class MultiMemberGateUnionTest(unittest.TestCase):
    """I2 门并集（规格级）：多 remote 成员不同 participation 的切片块
    覆盖区间并集正确 + 各成员 I1。"""

    def test_gate_union_spec(self):
        s = _scheduler(credit_iters="2")
        ra = _remote_runtime("ra", decode=6, input_tokens=16,
                             session_id="s_a")
        rb = _remote_runtime("rb", decode=3, input_tokens=16,
                             session_id="s_b")
        s.runtime_by_request_id["ra"] = ra
        s.runtime_by_request_id["rb"] = rb
        s.kv_manager = SimpleNamespace(
            _sessions={"s_a": _session(), "s_b": _session()}, tp_degree=2)
        for rt in (ra, rb):
            rt.remote_read_credit_plan = (
                s._joint_remote_read_credit_plan(rt, EXEC_INSTANCE))
        state = s.instances[EXEC_INSTANCE]
        for rt in (ra, rb):
            state.active_decode.append(rt)
            state.active_decode_lookup.add(rt)
        plan = s._plan_train(state)
        joiner_plans = [rt.plan_dict() for rt in (ra, rb)]
        spec = s._plan_remote_credit_slices(plan, [ra, rb], joiner_plans)
        self.assertIsNotNone(spec)
        self.assertEqual(spec["k"], 2)
        # 体块 [1-2]/[3-4]/[5-6] 的门 = 覆盖成员切片块并集。
        self.assertEqual(spec["body_blocks"][0]["gates"],
                         [("ra", 1), ("rb", 1)])
        self.assertEqual(spec["body_blocks"][1]["gates"],
                         [("ra", 2), ("rb", 2)])
        self.assertEqual(spec["body_blocks"][2]["gates"], [("ra", 3)])
        # 成员切片块数：S=6 → 3 块；S=3 → 2 块 [2,1]（块内步数守恒）。
        self.assertEqual(
            ra.remote_read_slice_summaries[0]["block_steps"], [2, 2, 2])
        self.assertEqual(
            rb.remote_read_slice_summaries[0]["block_steps"], [2, 1])
        for rt in (ra, rb):
            self.assertEqual(
                rt.remote_read_slice_summaries[0]["total_bytes"],
                rt.remote_read_credit_plan["total_bytes"])


class PartialHybridReadStreamTest(unittest.TestCase):
    """§4.2.3 PARTIAL 基混合形态（2026-09-17）：读流层区间化 [0, p)——
    工作会话读前缀层（base_resident_prefix_layers），后缀 [p, L) 准入相
    池恢复物化、不经读流；I1 基数 = S × f(终态) × [0, p) 层区间。"""

    def _install(self, s, rt, session):
        s.runtime_by_request_id[rt.request_id] = rt
        s.kv_manager = SimpleNamespace(
            _sessions={rt.session_id: session}, tp_degree=2)
        rt.remote_read_credit_plan = (
            s._joint_remote_read_credit_plan(rt, EXEC_INSTANCE))

    def test_partial_plan_prefix_range_and_bytes(self):
        layers, prefix = 4, 2
        s = _scheduler(credit_iters="4", layers=layers)
        rt = _remote_runtime(decode=8)
        self._install(s, rt, _session(
            working_kind="remote-read", context_tokens=0,
            base_history_tokens=100, base_prefix_layers=prefix))
        plan = rt.remote_read_credit_plan
        self.assertIsNotNone(plan)
        # 计划披露读前缀层。
        self.assertEqual(plan["read_prefix_layers"], prefix)
        # per-step 字节 = 层区间派生（[0, p)，非全层 [0, L)）。
        context_per_step = 100 + 16 + 8
        expected_shards = kv_cache_shard_bytes_for_layer_range(
            s.model, context_per_step, 2,
            layer_start=0, layer_end=prefix)
        full_shards = kv_cache_shard_bytes_for_tokens(
            s.model, context_per_step, 2)
        self.assertNotEqual(expected_shards, full_shards)
        self.assertEqual(
            tuple(spec[2] for spec in plan["shard_specs"]), expected_shards)
        # I1：总量 = S × 每步（层区间派生的每步读量）。
        self.assertEqual(plan["steps"], 8)
        self.assertEqual(plan["total_bytes"], sum(expected_shards) * 8)
        # 切片：KVTransfer/KVTransferShard 层区间 = [0, p)。
        blocks = s._joint_remote_read_slice(rt, 8, 4)
        self.assertEqual([b.total_bytes for b in blocks],
                         [sum(expected_shards) * 4] * 2)
        self.assertEqual(
            sum(b.total_bytes for b in blocks), plan["total_bytes"])
        for block in blocks:
            self.assertEqual(block.model_layers, layers)
            self.assertEqual((block.layer_start, block.layer_end),
                             (0, prefix))
            self.assertEqual(
                (block.resident_prefix_layers_before,
                 block.resident_prefix_layers_after),
                (prefix, prefix))
            for shard in block.shards:
                self.assertEqual(
                    (shard.layer_start, shard.layer_end), (0, prefix))

    def test_local_base_working_session_byte_identical_anchor(self):
        # LOCAL 基（working 会话 p==L）：计划与非 working 会话逐字段相同
        # ——回归锚（p==L 时层区间派生 ≡ 全层旧口径）。
        s = _scheduler(credit_iters="4", layers=4)
        rt_w = _remote_runtime("rw", session_id="s_w")
        rt_p = _remote_runtime("rp", session_id="s_p")
        self._install(s, rt_w, _session(
            working_kind="remote-read", context_tokens=0,
            base_history_tokens=100, base_prefix_layers=4))
        self._install(s, rt_p, _session(context_tokens=100))
        self.assertEqual(
            rt_w.remote_read_credit_plan["read_prefix_layers"], 4)
        self.assertEqual(
            rt_w.remote_read_credit_plan, rt_p.remote_read_credit_plan)

    def test_remote_base_defensive_none(self):
        # 防御锚：无前缀层可读（REMOTE 基形态，正常不可达）→ None。
        s = _scheduler(credit_iters="4", layers=4)
        rt = _remote_runtime(decode=8)
        self._install(s, rt, _session(
            working_kind="remote-read", context_tokens=0,
            base_history_tokens=100, base_prefix_layers=0))
        self.assertIsNone(rt.remote_read_credit_plan)


class MissingArmFailClosedTest(unittest.TestCase):
    """交付后复核硬化钉：体块门引用块号 ≥2 而 arm 账本整块缺失（尾块
    未发射/未登记/块号错位）→ fail-closed raise，不静默无门消费。"""

    def test_missing_tail_arm_raises(self):
        s = _scheduler(credit_iters="2")
        rt = _remote_runtime(decode=8)
        _install_remote(s, rt)
        state = s.instances[EXEC_INSTANCE]
        state.active_decode.append(rt)
        state.active_decode_lookup.add(rt)
        plan = s._plan_train(state)
        joiner_plans = [rt.plan_dict()]
        joiner_plans[0]["prefill_drain_block_ends"] = {2: 0, 3: 0}
        spec = s._plan_remote_credit_slices(plan, [rt], joiner_plans)
        self.assertIsNotNone(spec)
        # 篡改：体块 2 的门指向从未发射的块号 5（模拟尾块发射缺失/错位）。
        spec["body_blocks"][1]["gates"] = [(rt.request_id, 5)]
        s._batch = {"watches": []}
        train_plan = {
            "train_id": plan["train_id"],
            "instance_index": EXEC_INSTANCE,
            "stage": "decode",
            "joiners": [joiner_plans[0]],
            "pass_spans": list(plan["pass_spans"]),
            "iterations": plan["iterations"],
            "sentinel": False,
            "prefill_start_member": None,
            "first_chunk_member": None,
            "drain_members": [],
            "exit_members": [{"request_id": rt.request_id}],
            "remote_credit": spec,
        }
        with self.assertRaises(RuntimeError) as ctx:
            s.graph.emit_iteration_train(train_plan)
        self.assertIn("credit arm ledger is missing block 5", str(ctx.exception))


class V1SerialPathRemovedTest(unittest.TestCase):
    """v1 批量串行口径删除钉（2026-09-17 用户裁定：不作为开关可选项
    保留）。"""

    def test_v1_stream_synthesizer_deleted(self):
        self.assertFalse(
            hasattr(Sh30OnlineScheduler, "_joint_remote_read_stream"))

    def test_no_remote_exec_switch(self):
        cfg = parse_joint_config({})
        self.assertFalse(hasattr(cfg, "remote_exec"))


if __name__ == "__main__":
    unittest.main()
