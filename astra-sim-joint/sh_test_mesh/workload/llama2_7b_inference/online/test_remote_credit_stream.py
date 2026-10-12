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
    JOINT_STAGE_D_RUNNING,
    Sh30OnlineScheduler,
    _OnlineInstanceState,
    _OnlineRequestRuntime,
    _TASK_LOAD_CACHE_CAPACITY,  # F6 销账：替身对齐 __init__ 初值
)
from online.test_graph_batch_builder import _make_config  # noqa: E402

EXEC_INSTANCE = 0   # ranks (0, 1)
HOME_INSTANCE = 1   # ranks (2, 3)


def _graph_config_for(instance_count):
    """N 实例构图器配置（3 实例多源夹具用：npus/分组随实例数扩展；
    2 实例缺省路径仍用 test_graph_batch_builder._make_config 原锚）。"""
    return SimpleNamespace(
        npus_count=2 * instance_count,
        remote_operand_loads=False,
        trace_granularity="request_aggregated",
        inference_groups=[
            SimpleNamespace(ranks=(2 * i, 2 * i + 1),
                            pg_name="tp_instance%d" % i)
            for i in range(instance_count)],
        layers=2,
        hidden_size=64,
        ffn_size=128,
        vocab_size=256,
        bytes_per_elem=2,
        num_heads=8,
        mlp_variant="gelu",
        request_queue=[
            SimpleNamespace(
                session_arrival_time_ns=0,
                inter_request_interval_ns=None,
            ),
        ],
    )


def _scheduler(tile_bytes="auto", layers=2, instance_count=2):
    s = Sh30OnlineScheduler.__new__(Sh30OnlineScheduler)
    s.p_chunk = 512
    # N5：_emit_train 的 merge_tail_gated 纯度标记读 _pending_merge_
    # alarms（__init__ 恒设——替身补设，F6）。
    s._pending_merge_alarms = {}
    s.instances = [
        _OnlineInstanceState(index=i) for i in range(instance_count)]
    s.runtime_by_request_id = {}
    s._train_max_iter = 0
    s._train_instance_index = {}
    s._ready_frontier = set()
    ranks_by_instance = {
        i: (2 * i, 2 * i + 1) for i in range(instance_count)}
    s.topology = SimpleNamespace(
        instance=lambda i: SimpleNamespace(
            size=2, ranks=ranks_by_instance[i]),
        instances=[SimpleNamespace(size=2)] * instance_count)
    s.hardware = FaceHardware(
        mesh_rows=2 * instance_count, mesh_cols=2,
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
    s.graph = GraphBatchBuilder(
        _make_config() if instance_count == 2
        else _graph_config_for(instance_count))
    s.graph.begin_batch()
    s._batch = {"watches": []}
    s.train_ledger_rows = []
    s.train_ledger_sink = None
    s.joint_config = parse_joint_config(
        {"JOINT_REMOTE_READ_TILE_BYTES": tile_bytes})
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
    # D20（v2.0 §16.4）状态机口径：本夹具成员 = 边界提交后的 decode
    # 成员——D_RUNNING + 完成归属在边界提交点锁定（完成信号按归属
    # 实例核销的 fail-closed 门）。
    rt.joint_stage = JOINT_STAGE_D_RUNNING
    rt.completion_instance_index = EXEC_INSTANCE
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


def _seed_pipeline_identity(scheduler, runtime, *, stage="decode"):
    if runtime.remote_read_credit_plan is None:
        return
    target_ranks = tuple(sorted({
        int(spec[1]) for spec in
        runtime.remote_read_credit_plan.get("shard_specs", ())}))
    runtime.pipeline_identity_inputs[stage] = {
        "stage_epoch": int(runtime.joint_boundary_epoch),
        "staging_allocation_id": 1,
        "target_ranks": target_ranks,
    }


def _install_remote(scheduler, runtime):
    scheduler.runtime_by_request_id[runtime.request_id] = runtime
    scheduler.kv_manager = SimpleNamespace(
        _sessions={runtime.session_id: _session()}, tp_degree=2,
        # D09/D20：Tj 核销边界暂存 lease 注销直达调用（本替身无在册
        # lease——幂等空放语义；F6 教义：接口直达，替身补齐）。
        release_staging_lease=lambda request_id, boundary_epoch=None: False,
        staging_lease_for_request=lambda request_id: None,
        peek_next_staging_allocation_id=lambda: 1)
    # drain 边界的持久读计划建立（_on_prefill_drain 的等价直连）。
    runtime.remote_read_credit_plan = (
        scheduler._joint_remote_read_credit_plan(
            runtime, EXEC_INSTANCE))
    # The compact production emitter now requires the same explicit identity
    # seed that a selected scheduler action carries into GraphBatchBuilder.
    # This fixture models an already-live decode staging lease; candidate
    # preview must not consume the run-scoped segment ordinal.
    _seed_pipeline_identity(scheduler, runtime)


class RemoteCreditEmissionTest(unittest.TestCase):
    """T1 joiner 字节 tile 切片（D09）：结构不变量（I1/I2/D2 + 支链不
    join + 缓冲复用释放链）。auto tile（256 token）≥ 基础区间
    （100 token）→ 每 pass 恰 1 个 tile：decode=8 → M=8、tile t 由体块
    t 消费；tile 2 头部旁挂、tile 3..8 由消费 tile t-2 的体块点释放
    （生产等消费——真实缓冲复用依赖）。

    compact 发射（descriptor + native 执行器）后，per-tile send/recv/
    释放事件在 decode_remote_credit descriptor 草稿的 endpoint_runs
    内；本组经 _credit_stream_surface 汇总逐 tile 腿/锚/消费面。"""

    def _emit_t1(self, tile_bytes="auto", decode=8):
        s = _scheduler(tile_bytes)
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

    @staticmethod
    def _credit_stream_surface(s, request_id="rr"):
        """decode_remote_credit 草稿的逐 tile 汇总面。

        返回 (legs_by_tile, recv_anchor_by_tile, consume_dep_by_tile)：
        legs = {(endpoint, run.rank)}；recv_anchor = tile 的 data_recv
        支链 fork 锚物理节点 id；consume_dep = 该 tile recv 门的消费
        体块末物理节点 id（consume_completion_runs 的依赖面）。"""
        legs_by_tile = {}
        recv_anchor_by_tile = {}
        consume_dep_by_tile = {}
        for draft in s.graph._pipeline_drafts.values():
            if draft.stream_kind != "decode_remote_credit":
                continue
            if draft.identity.request_id != request_id:
                continue
            tile_by_group = {
                group["group_ordinal"]:
                    group["tile_axis"]["global_tile_first"]
                for group in draft.groups}
            for run in draft.endpoint_runs:
                first_global = (tile_by_group[run["group_ordinal"]]
                                + int(run["first_tile"]) - 1)
                for offset in range(int(run["tile_count"])):
                    for leg in run["legs"]:
                        legs_by_tile.setdefault(first_global + offset, set())\
                            .add((leg["endpoint"], int(run["rank"])))
            for anchor in draft.anchors:
                selector = anchor.get("selector", {})
                if (selector.get("kind") != "tile_endpoint_leg"
                        or selector.get("endpoint") != "data_recv"):
                    continue
                first = tile_by_group[selector["group_ordinal"]]
                recv_anchor_by_tile[first + selector["tile_ordinal"] - 1] = (
                    anchor["node_id"])
            for run in draft.consume_completion_runs:
                first = tile_by_group[run["group_ordinal"]]
                consume_dep_by_tile[first + int(run["first_tile"]) - 1] = (
                    int(run["first_dependency_json_id"]))
        return legs_by_tile, recv_anchor_by_tile, consume_dep_by_tile

    def test_t1_multi_block_structure(self):
        s, rt, plan = self._emit_t1(tile_bytes="auto", decode=8)
        # I1：切片总量 ≡ 计划总量（per shard 由整数乘法保证）。
        summary = rt.remote_read_slice_summaries[0]
        self.assertEqual(summary["tiles_per_pass"], 1)
        self.assertEqual(summary["steps"], 8)
        self.assertEqual(summary["block_count"], 8)
        self.assertEqual(
            summary["total_bytes"], rt.remote_read_credit_plan["total_bytes"])
        # R15：逐切片键已登记（切片创建点）。
        self.assertIn("rr#decode#1", s._joint_flows._owner_flows)
        legs_by_tile, recv_anchors, consume_deps = (
            self._credit_stream_surface(s))
        # 全部 tile 1..8 的数据腿在场：home 源 rank（2/3）发 data_send、
        # exec 目标 rank（0/1）收 data_recv；fill_ack 回执反向同构
        #（compact 发射：tile 1 主链与 tile 2 头部旁挂同为即时面，
        # tile 3..8 消费锚释放——同 descriptor 逐 tile 腿表达）。
        self.assertEqual(sorted(legs_by_tile), list(range(1, 9)))
        for tile, legs in legs_by_tile.items():
            self.assertIn(("data_send", 2), legs, tile)
            self.assertIn(("data_send", 3), legs, tile)
            self.assertIn(("data_recv", 0), legs, tile)
            self.assertIn(("data_recv", 1), legs, tile)
            if tile >= 3:
                self.assertIn(("release_ack_send", 0), legs, tile)
                self.assertIn(("release_ack_recv", 2), legs, tile)
            else:
                self.assertNotIn(
                    ("release_ack_send", 0), legs, tile)
        # D2：fork 点先于块 1——tile 1 与 tile 2 的 data_send 支链锚
        # 同为头部即时面（发射序先于一切体块；首批发射无体依赖）。
        self.assertLess(
            recv_anchors[1], consume_deps[1], "tile 1 anchor precedes body")
        self.assertLess(
            recv_anchors[2], consume_deps[1], "tile 2 anchor precedes body")
        # D2/支链不 join：barrier 在，且 no consume run 锚到 barrier
        #（read 流的完成门只由体块消费，不汇入 readiness barrier）。
        barrier = [
            n for n in self._nodes(s)
            if n["name"].endswith("_decode_kv_ready_barrier")]
        self.assertEqual(sorted(n["rank"] for n in barrier), [0, 1])
        barrier_ids = {node["id"] for node in barrier}
        self.assertFalse(set(consume_deps.values()) & barrier_ids)

    def test_buffer_reuse_release_chain(self):
        # D09 缓冲复用释放链（tile ≥ 3）：tile t 的 send 在 home 侧以
        # 释放 ack（tag 配对）为前置，且 tile t 的支链发射于消费 tile
        # t-2 的体块之后（生产等消费，慢计算停泵的结构表达）。compact
        # 发射：fork 因果由 descriptor 的 initial_parent_json_id 承载；
        # 物理面上 tile t 的 data_recv 支链锚（pipeline frontier）发射
        # 序夹在消费 tile t-2 的体块末与 tile t 的消费体块末之间。
        s, rt, plan = self._emit_t1(tile_bytes="auto", decode=8)
        legs_by_tile, recv_anchors, consume_deps = (
            self._credit_stream_surface(s))
        by_id = {node["id"]: node for node in self._nodes(s)}
        # 消费体块末节点按 tile 对位（rcb{tile} 的末计算节点）。
        for tile in range(4, 9):
            self.assertIn(tile, recv_anchors)
            consumer_terminal = by_id[consume_deps[tile]]
            producer_terminal = by_id[consume_deps[tile - 2]]
            self.assertEqual(
                consumer_terminal["request_id"], plan["train_id"])
            self.assertEqual(
                producer_terminal["request_id"], plan["train_id"])
            # tile t 的支链锚定在消费 tile t-2 的体块之后（fork 发射
            # 序 > 生产者体块末）、且早于 tile t 自己的消费体块末。
            self.assertGreater(
                recv_anchors[tile], consume_deps[tile - 2],
                "tile %d production is not anchored after tile %d "
                "consumption" % (tile, tile - 2))
            self.assertLess(
                recv_anchors[tile], consume_deps[tile],
                "tile %d production must precede its own consumption" % tile)

    def test_i2_body_block_gates(self):
        s, rt, plan = self._emit_t1(tile_bytes="auto", decode=8)
        legs_by_tile, recv_anchors, consume_deps = (
            self._credit_stream_surface(s))
        by_id = {node["id"]: node for node in self._nodes(s)}
        train_id = plan["train_id"]
        # 每 credit tile b 恰一条 consume completion run（I2：tile b
        # 完成门 = 覆盖其迭代区间的体块末；不依赖后续块）。
        self.assertEqual(sorted(consume_deps), list(range(1, 9)))
        for block in sorted(consume_deps):
            terminal = by_id[consume_deps[block]]
            self.assertEqual(terminal["request_id"], train_id)
            self.assertGreater(terminal.get("compute", {}).get(
                "num_ops", 0), 0, block)
        # 门序 = 块序（体块 b 的门节点先于体块 b+1 的门节点——不依赖
        # 后续块）。
        ordered = [consume_deps[block] for block in sorted(consume_deps)]
        self.assertEqual(ordered, sorted(ordered))
        # 无门消费扫描：每个被门控的体节点都是 COMP 且属于 exec rank。
        for block, node_id in consume_deps.items():
            node = by_id[node_id]
            self.assertIn(node["rank"], (0, 1))
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
        s = _scheduler(tile_bytes="auto")
        rt = _remote_runtime(decode=8)
        _install_remote(s, rt)
        state = s.instances[EXEC_INSTANCE]
        head = _head_runtime(chunks=3)
        s.runtime_by_request_id["rh"] = head
        state.qp.append(head)
        state.active_decode.append(rt)
        state.active_decode_lookup.add(rt)
        # T1：iterations=3（队头 3 chunk），joiner 参与 3 步 → 3 个
        # pass tile（auto tile ≥ 基础区间 → 每 pass 1 tile）。
        plan1 = s._plan_train(state)
        self.assertEqual(plan1["iterations"], 3)
        s._emit_train(state, plan1, [rt], 1)
        first = rt.remote_read_slice_summaries[0]
        self.assertEqual(first["steps"], 3)
        self.assertEqual(first["tiles_per_pass"], 1)
        self.assertEqual(first["block_count"], 3)
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
        self.assertEqual(second["steps"], 5)
        self.assertEqual(second["block_count"], 5)
        # I1（多列车字节总量等价）：Σ切片 ≡ 计划总量。
        self.assertEqual(
            first["total_bytes"] + second["total_bytes"],
            rt.remote_read_credit_plan["total_bytes"])
        # R15：T2 规划期登记表恰含键 2（键 1 已核销——无陈旧多计）。
        self.assertNotIn("rr#decode#1", s._joint_flows._owner_flows)
        self.assertIn("rr#decode#2", s._joint_flows._owner_flows)
        # I3b 结构正则：续列车无 barrier；compact 发射面 = T2 批次
        # decode_remote_credit 草稿覆盖本切片全部 5 个 tile（批内
        # slice-local 序 1..5；T1 批次 3 tile 已随其批次结清）。
        nodes = s.graph.batch["nodes"]
        self.assertFalse([
            n for n in nodes
            if n["name"].endswith("_decode_kv_ready_barrier")])
        legs_by_tile, _anchors, _consume = (
            RemoteCreditEmissionTest._credit_stream_surface(s))
        self.assertEqual(sorted(legs_by_tile), list(range(1, 6)))
        # T2 核销：键 2 释放，请求走完。
        s.kv_manager._sessions["s_rr"].context_tokens = (
            rt.joint_input_tokens + 8)
        self._settle(s, state, 4)
        self.assertNotIn("rr#decode#2", s._joint_flows._owner_flows)
        self.assertEqual(rt.decode_tokens_consumed, 8)


class SingleTileDegenerateEquivalenceTest(unittest.TestCase):
    """单 tile 退化锚（decode=1 → M=1，无尾块无释放链）。compact 发射
    （descriptor + native 执行器）后 credit 流恒走 descriptor 面（M=1
    同样），与 plain pd 单笔的物理节点路径是两种发射面；本组改钉退化
    形态等价：两侧的体/主链/标记节点集逐字节相同，读流面 = 恰一组一
    tile 的 decode_remote_credit 草稿（data/fill 腿、无释放腿、账本结
    清）。"""

    def test_single_step_matches_plain_train(self):
        decode = 1
        # A：credit 路径（M=1，无尾块）。
        s = _scheduler(tile_bytes="auto")
        rt = _remote_runtime(decode=decode)
        _install_remote(s, rt)
        state = s.instances[EXEC_INSTANCE]
        state.active_decode.append(rt)
        state.active_decode_lookup.add(rt)
        plan = s._plan_train(state)
        s._emit_train(state, plan, [rt], 1)
        # B：同构对照——手工拼同形列车（pd 单笔 = 同字节单 tile 传输，
        # 无 remote_credit 规格 → 单段体）。
        s2 = _scheduler(tile_bytes="auto")
        rt2 = _remote_runtime(decode=decode)
        _install_remote(s2, rt2)
        block1 = s2._joint_remote_read_slice(rt2, decode)[0]
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
        # 体/主链/标记节点集逐字节等价（JSON 序列化排序对照；读流面
        # 两侧各自表达——A = pipeline 锚 + descriptor，B = pd 物理腿）。
        def _body_nodes(graph):
            return [
                json.dumps(node, sort_keys=True)
                for node in graph.batch["nodes"]
                if "pipeline" not in node["name"]
                and "prefill_decode_transfer" not in node["name"]]

        self.assertEqual(_body_nodes(s.graph), _body_nodes(s2.graph))
        # A 的退化读流面：每 owner rank 恰一组一 tile 的
        # decode_remote_credit 草稿；data/fill 腿在场、无释放腿（M=1
        # 无尾块无释放链）；账本结清（无 arms/pending 残留）。
        drafts = [
            draft for draft in s.graph._pipeline_drafts.values()
            if draft.stream_kind == "decode_remote_credit"]
        self.assertEqual(
            sorted(draft.identity.rank for draft in drafts), [0, 1])
        for draft in drafts:
            self.assertEqual(len(draft.groups), 1)
            self.assertEqual(
                draft.groups[0]["tile_axis"]["tile_count"], 1)
            endpoints = {
                leg["endpoint"]
                for run in draft.endpoint_runs for leg in run["legs"]}
            self.assertEqual(
                endpoints,
                {"data_send", "data_recv", "fill_ack_send", "fill_ack_recv"})
        self.assertEqual(s.graph._credit_pending_tiles, {})
        self.assertEqual(s.graph._credit_arms, {})


class MultiMemberGateUnionTest(unittest.TestCase):
    """I2 门并集（规格级）：多 remote 成员不同 participation 的切片块
    覆盖区间并集正确 + 各成员 I1。"""

    def test_gate_union_spec(self):
        s = _scheduler(tile_bytes="auto")
        ra = _remote_runtime("ra", decode=6, input_tokens=16,
                             session_id="s_a")
        rb = _remote_runtime("rb", decode=3, input_tokens=16,
                             session_id="s_b")
        s.runtime_by_request_id["ra"] = ra
        s.runtime_by_request_id["rb"] = rb
        s.kv_manager = SimpleNamespace(
            _sessions={"s_a": _session(), "s_b": _session()}, tp_degree=2,
            # D09/D20 同上：替身补齐 lease 注销接口（幂等空放）。
            release_staging_lease=lambda request_id, boundary_epoch=None: False)
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
        # 体块逐迭代切分；每体块门 = 覆盖该迭代的成员 tile 并集（I2）。
        self.assertEqual(spec["body_blocks"][0]["gates"],
                         [("ra", 1), ("rb", 1)])
        self.assertEqual(spec["body_blocks"][1]["gates"],
                         [("ra", 2), ("rb", 2)])
        self.assertEqual(spec["body_blocks"][2]["gates"],
                         [("ra", 3), ("rb", 3)])
        self.assertEqual(spec["body_blocks"][3]["gates"], [("ra", 4)])
        # 成员切片 tile 数（auto tile ≥ 基础区间 → 每 pass 1 tile）：
        # S=6 → 6 tile；S=3 → 3 tile。
        self.assertEqual(
            ra.remote_read_slice_summaries[0]["block_count"], 6)
        self.assertEqual(
            rb.remote_read_slice_summaries[0]["block_count"], 3)
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
            _sessions={rt.session_id: session}, tp_degree=2,
            # D09/D20 同上：替身补齐 lease 注销接口（幂等空放）。
            release_staging_lease=lambda request_id, boundary_epoch=None: False)
        rt.remote_read_credit_plan = (
            s._joint_remote_read_credit_plan(rt, EXEC_INSTANCE))

    def test_partial_plan_prefix_range_and_bytes(self):
        layers, prefix = 4, 2
        s = _scheduler(tile_bytes="auto", layers=layers)
        rt = _remote_runtime(decode=8)
        self._install(s, rt, _session(
            working_kind="remote-read", context_tokens=0,
            base_history_tokens=100, base_prefix_layers=prefix))
        plan = rt.remote_read_credit_plan
        self.assertIsNotNone(plan)
        # 计划披露读前缀层。
        self.assertEqual(plan["read_prefix_layers"], prefix)
        # per-step 字节 = 层区间派生（[0, p)，非全层 [0, L)）；D09 式
        # (18a)：读量基数 = 远端基础区间（base_tokens=100）——输入/
        # decode 增量本地读不过 NoC（旧 base+input+decode 终态口径废除）。
        context_per_step = 100
        expected_shards = kv_cache_shard_bytes_for_layer_range(
            s.model, context_per_step, 2,
            layer_start=0, layer_end=prefix)
        full_shards = kv_cache_shard_bytes_for_tokens(
            s.model, 100 + 16 + 8, 2)
        self.assertNotEqual(expected_shards, full_shards)
        self.assertEqual(
            tuple(spec[2] for spec in plan["shard_specs"]), expected_shards)
        # I1：总量 = S × 每步（层区间派生的每步读量）。
        self.assertEqual(plan["steps"], 8)
        self.assertEqual(plan["total_bytes"], sum(expected_shards) * 8)
        # 切片：KVTransfer/KVTransferShard 层区间 = [0, p)；auto tile
        # ≥ 基础区间 → 每 pass 1 tile、字节 = 每 pass 全量。
        blocks = s._joint_remote_read_slice(rt, 8)
        self.assertEqual([b.total_bytes for b in blocks],
                         [sum(expected_shards)] * 8)
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
        s = _scheduler(tile_bytes="auto", layers=4)
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
        s = _scheduler(tile_bytes="auto", layers=4)
        rt = _remote_runtime(decode=8)
        self._install(s, rt, _session(
            working_kind="remote-read", context_tokens=0,
            base_history_tokens=100, base_prefix_layers=0))
        self.assertIsNone(rt.remote_read_credit_plan)


class MissingArmFailClosedTest(unittest.TestCase):
    """交付后复核硬化钉：体块门引用块号 ≥2 而 arm 账本整块缺失（尾块
    未发射/未登记/块号错位）→ fail-closed raise，不静默无门消费。"""

    def test_missing_tail_arm_raises(self):
        s = _scheduler(tile_bytes="auto")
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
        # compact 发射的现行错误面：门引用块号 ≥2 而 arm 账本无该块
        # 的有界 recv 锚（尾块未发射/未登记/块号错位）→ fail-closed。
        self.assertIn(
            "credit arm ledger has no bounded receive anchor for block 5",
            str(ctx.exception))


class V1SerialPathRemovedTest(unittest.TestCase):
    """v1 批量串行口径删除钉（2026-09-17 用户裁定：不作为开关可选项
    保留）。"""

    def test_v1_stream_synthesizer_deleted(self):
        self.assertFalse(
            hasattr(Sh30OnlineScheduler, "_joint_remote_read_stream"))

    def test_no_remote_exec_switch(self):
        cfg = parse_joint_config({})
        self.assertFalse(hasattr(cfg, "remote_exec"))


def _multi_source_session(context_tokens=300, boundary_tokens=300):
    """C4 多源会话替身：权威基（实例 1，context_tokens）＋边界第二来
    源（实例 2 的 i_P 侧旧工作副本，boundary_tokens）——两源均在 exec
    （实例 0）之外。"""
    return SimpleNamespace(
        working_kind=None, context_tokens=context_tokens,
        base_history_tokens=0, base_resident_prefix_layers=0,
        boundary_source_instance_index=2,
        boundary_source_history_tokens=boundary_tokens,
        boundary_source_shard_bytes=(64, 64),
        boundary_source_resident_prefix_layers=None)


class MultiSourceGateAlignmentTest(unittest.TestCase):
    """C4 多源 decode credit × D09 固定 tile：体块门必须覆盖切片实际
    tile 数（逐源 ⌈tokens_s/tile⌉ 之和），而非联合区间 ⌈Σtokens/tile⌉。

    两源各自存在不足一 tile 的尾块余量时，逐源 tile 数严格大于联合
    tile 数——若门按联合口径生成，每 pass 的尾部 tile 无消费体块，
    列车尾待发账本残留 fail-closed（TJE_quota_aimd 冒烟红复现：
    "credit pending tiles [467..472] ... never released by a consuming
    body block"——该臂的准入时序使 session_1_request_3 以多源形态远
    读，其余臂单源不触发）。"""

    def _emit(self, decode=8, base_tokens=300, boundary_tokens=300):
        s = _scheduler(tile_bytes="auto", instance_count=3)
        rt = _remote_runtime(decode=decode, session_id="s_ms")
        s.runtime_by_request_id[rt.request_id] = rt
        s.kv_manager = SimpleNamespace(
            _sessions={"s_ms": _multi_source_session(
                context_tokens=base_tokens,
                boundary_tokens=boundary_tokens)},
            tp_degree=2,
            release_staging_lease=lambda request_id,
            boundary_epoch=None: False)
        rt.remote_read_credit_plan = s._joint_remote_read_credit_plan(
            rt, EXEC_INSTANCE)
        _seed_pipeline_identity(s, rt)
        state = s.instances[EXEC_INSTANCE]
        state.active_decode.append(rt)
        state.active_decode_lookup.add(rt)
        plan = s._plan_train(state)
        s._emit_train(state, plan, [rt], 1)
        return s, rt, plan

    def _by_name(self, s, needle):
        return [n for n in s.graph.batch["nodes"] if needle in n["name"]]

    def test_plan_tiles_per_pass_matches_slice_per_pass(self):
        # 计划/切片单一口径：每 pass tile 数 = 切片单 pass 实际块数
        # （多源 = 逐源 ⌈tokens/tile⌉ 之和；单源 = 联合 ⌈tokens/tile⌉
        # 恒等）。
        for label, instance_count, session in (
                ("multi", 3, _multi_source_session()),
                ("single", 2, _session(context_tokens=300))):
            s = _scheduler(tile_bytes="auto", instance_count=instance_count)
            request_id = "rr_%s" % label
            session_id = ("s_ms" if label == "multi" else "s_one")
            rt = _remote_runtime(request_id, decode=4,
                                 session_id=session_id)
            s.runtime_by_request_id[request_id] = rt
            s.kv_manager = SimpleNamespace(
                _sessions={session_id: session}, tp_degree=2,
                release_staging_lease=lambda request_id,
                boundary_epoch=None: False)
            rt.remote_read_credit_plan = (
                s._joint_remote_read_credit_plan(rt, EXEC_INSTANCE))
            per_pass = s._joint_remote_read_slice(rt, 1)
            self.assertEqual(
                rt.remote_read_credit_plan["tiles_per_pass"],
                len(per_pass), label)
            runs = per_pass.pipeline_tile_runs
            self.assertTrue(runs)
            self.assertEqual(sum(run["tile_count"] for run in runs),
                             len(per_pass), label)
            self.assertEqual(
                {run["axis_unit"] for run in runs}, {"source_tokens"},
                label)
            for run in runs:
                self.assertEqual(
                    run["total_units"],
                    (run["tile_count"] - 1) * run["units_per_tile"]
                    + run["tail_units"], label)
        # 多源显式锚：300+300 token、tile 256 → 逐源 2+2 = 4 块/pass，
        # 严格大于联合口径 ⌈600/256⌉ = 3（两源尾块各自存在）。
        s = _scheduler(tile_bytes="auto", instance_count=3)
        rt = _remote_runtime("rr_anchor", decode=4, session_id="s_ms")
        s.runtime_by_request_id["rr_anchor"] = rt
        s.kv_manager = SimpleNamespace(
            _sessions={"s_ms": _multi_source_session()}, tp_degree=2,
            release_staging_lease=lambda request_id,
            boundary_epoch=None: False)
        rt.remote_read_credit_plan = s._joint_remote_read_credit_plan(
            rt, EXEC_INSTANCE)
        self.assertEqual(
            rt.remote_read_credit_plan["tiles_per_pass"], 4)

    def test_multi_source_body_gates_cover_slice_tiles(self):
        # 8 pass × 每 pass 4 tile = 32 块切片，全部 tile 有消费体块
        # （门并集覆盖 + 释放链闭合）——修复前：门按联合口径 3 块/pass
        # 只覆盖 24 块，tile 27..32 无消费体块，列车尾
        # RuntimeError(credit pending tiles ... never released)。
        s, rt, plan = self._emit()
        cpp = rt.remote_read_credit_plan
        self.assertEqual(cpp["tiles_per_pass"], 4)
        summary = rt.remote_read_slice_summaries[0]
        self.assertEqual(summary["tiles_per_pass"], 4)
        self.assertEqual(summary["steps"], 8)
        self.assertEqual(summary["block_count"], 32)
        # I1：切片总量 ≡ 计划总量（多源逐源排空口径不变）。
        self.assertEqual(
            summary["total_bytes"], cpp["total_bytes"])
        # 全部 tile 2..32 的数据腿在场（发射链完整；compact 发射 =
        # decode_remote_credit 草稿逐 tile data_send/recv 腿；多源
        # 场景 data_send 在各源 rank——源 rank 集由切片 shard 面给
        # 出，此处钉 exec 侧 recv 与存在性）。
        legs_by_tile, _anchors, _consume = (
            RemoteCreditEmissionTest._credit_stream_surface(
                s, request_id=rt.request_id))
        self.assertEqual(sorted(legs_by_tile), list(range(1, 33)))
        for block in range(2, 33):
            self.assertTrue(
                any(endpoint == "data_send"
                    for endpoint, _rank in legs_by_tile[block]), block)
            self.assertIn(("data_recv", 0), legs_by_tile[block], block)
            self.assertIn(("data_recv", 1), legs_by_tile[block], block)
        # 释放链闭合：列车尾无待发 tile 残留、无残留门账本。
        self.assertEqual(s.graph._credit_pending_tiles, {})
        self.assertEqual(s.graph._credit_arms, {})

    def test_multi_source_body_block_gate_sets(self):
        # 体块逐迭代门 = 该迭代的 4 个 tile（(iter-1)*4+1 .. iter*4）
        # ——门序与切片 pass 主序对齐（逐 tile lag-2 释放链的表达前提）。
        s2 = _scheduler(tile_bytes="auto", instance_count=3)
        rt2 = _remote_runtime("rr_ms2", decode=2, session_id="s_ms")
        s2.runtime_by_request_id["rr_ms2"] = rt2
        s2.kv_manager = SimpleNamespace(
            _sessions={"s_ms": _multi_source_session()}, tp_degree=2,
            release_staging_lease=lambda request_id,
            boundary_epoch=None: False)
        rt2.remote_read_credit_plan = (
            s2._joint_remote_read_credit_plan(rt2, EXEC_INSTANCE))
        state = s2.instances[EXEC_INSTANCE]
        state.active_decode.append(rt2)
        state.active_decode_lookup.add(rt2)
        plan2 = s2._plan_train(state)
        joiner_plans = [rt2.plan_dict()]
        spec = s2._plan_remote_credit_slices(plan2, [rt2], joiner_plans)
        self.assertIsNotNone(spec)
        self.assertEqual(len(spec["body_blocks"]), 2)
        for iteration, block in enumerate(spec["body_blocks"], start=1):
            expected = [(rt2.request_id, tile)
                        for tile in range((iteration - 1) * 4 + 1,
                                          iteration * 4 + 1)]
            self.assertEqual(block["gates"], expected, iteration)


if __name__ == "__main__":
    unittest.main()
