#!/usr/bin/env python3
"""test_d16_runtime_linkage_bounded_backend.py -- D16 运行时联动与后端
有界性的验收单测（joint修改文档 第二批 D16 条目 + v2.0 §17.1/C13/
§12.2；与既有 per-stream 结构钉测互补，不重复其断言面）。

条目红线逐条落测：
  1. 后端有界活跃流/延迟生成（v2.0 §17.1）——折叠列车体的节点数与
     迭代数无关（不为整轮"步数×层数×块数"预建事件图；迭代数由
     weight_passes/聚合字节承载）；remote credit 流只预发第二缓冲块
     （块 2），块 3..M 记入待发账本、在消费锚体块点延迟发射；待发
     残留 fail-closed。
  2. 模拟 HBM 只存字节计数/区间身份/引用/依赖（§17.1）——
     SessionKVState/NodeHBMState 状态形状与 token 数无关（不申请与
     模拟 KV 字节等大的宿主数组；字节以计数器承载）。
  3. 等价批处理合并遇真实外部事件即停（§17.1"合并遇到下一真实外部
     事件必须停止"）——列车合并的上界 = 交付默认治理边界 T_max
     （SH_TRAIN_MAX_ITER 缺省 "8"，sh30_online_scheduler.__init__）；
     自然 drain 边界先于上界时合并恰止于 drain 迭代；纯 decode 合并
     恰止于最后一个工作迭代（成员退出即真实事件，不补齐到上界）。
  4. 验收"producer/write/consume/release 时间闭合"——credit tile 的
     四类事件逐块闭合：send（home 产出）/ recv（exec 暂存写入）/
     体块门消费（恰一体块）/ 释放 ack 对（tag 配对归还缓冲；块 2 双
     缓冲初始皆空无释放依赖）；图侧 recv 字节 Σ ≡ 计划总量（I1 图
     级）；账本在尾标记结清（producer 侧无未锚定生产）。
  5. D16 对齐面收口（§12.2"执行器与预测器必须采用相同规则"；D09
     交付批登记的"切片计价 tile = 配置 tile、执行切片/暂存预约 = 放
     宽执行 tile"两侧粒度差；修复轮二 2026-09-27 固定 tile 口径后放
     宽已废除）——双缓冲闭式事务切块 tile 与执行切片同一裁决点：tile
     恒 = 配置值（宽基础区间块数 = ⌈rank 流字节/(配置 tile×kv)⌉×
     shard 数，可 >2——逐 tile lag-2 双缓冲承载），窄/宽基础区间同
     源于配置 tile（D09 手算锚不漂移）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 online/test_d16_runtime_linkage_bounded_backend.py
      （或 pytest 同路径）
"""
import os
import re
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
    NodeHBMState,
    SessionKVState,
)
from joint.joint_config import parse_joint_config  # noqa: E402
from joint.boundary_layout import BoundaryRuntimeStateEnvelope  # noqa: E402
from joint.joint_cost_model import (  # noqa: E402
    ACTION_REMOTE,
    InstanceLoadView,
    JointCostModel,
    JointHardwareRates,
    LinkFlowRegistry,
    PDBoundaryView,
    RequestView,
    ServiceFactors,
    SessionKVView,
)
from online.graph_batch_builder import GraphBatchBuilder  # noqa: E402
from online.sh30_online_scheduler import (  # noqa: E402
    JOINT_STAGE_D_RUNNING,
    Sh30OnlineScheduler,
    _OnlineInstanceState,
    _OnlineRequestRuntime,
    _TASK_LOAD_CACHE_CAPACITY,
)
from online.test_graph_batch_builder import _make_config  # noqa: E402

EXEC_INSTANCE = 0   # ranks (0, 1)
HOME_INSTANCE = 1   # ranks (2, 3)


# ================================================== 计价侧夹具（对齐面）==

def _route(source, target):
    return ((source, target), 1)


def _load(index, total):
    return InstanceLoadView(
        instance_index=index, queued_task_load_ns=total,
        running_task_load_ns=0, active_decode_task_load_ns=0,
        hbm_remaining_bytes_by_tp_rank=(10**9, 10**9))


def _jcm(loads, **overrides):
    kwargs = dict(
        rates=JointHardwareRates.from_gbps(
            noc_link_gbps=10.0, pool_port_gbps=5.0, local_hbm_gbps=100.0,
            d2d_latency_ns=10, pool_latency_ns=100),
        loads=loads, flow_registry=LinkFlowRegistry(),
        service_factors=ServiceFactors(),
        prefill_ns_per_token=1.0, decode_ns_per_token=2.0,
        model_layers=4, instance_tp_size=2, route_fn=_route)
    kwargs.update(overrides)
    return JointCostModel(**kwargs)


def _anchor_loads():
    return {0: _load(0, 5000), 1: _load(1, 1200)}


def _session(home=0, resident=0, history_bytes=(1000, 1000),
             history_tokens=100):
    return SessionKVView(
        session_id="s", home_instance=home, resident_instance=resident,
        location="local_hbm", history_tokens=history_tokens,
        resident_prefix_layers=4,
        history_bytes_by_tp_rank=history_bytes,
        missing_bytes_by_tp_rank=(0, 0))


def _request(decode=10):
    # input 500 B / 50 token = 10 B/token（惰性 kv_token_bytes 基准）。
    return RequestView(
        request_id="r", session_id="s", input_tokens=50,
        history_tokens_before=100, estimated_decode_tokens=decode,
        horizon_source="session_online_mean",
        input_kv_bytes_by_tp_rank=(500, 500))


# ================================================== 执行侧夹具（有界性）==

def _scheduler(tile_bytes="auto", layers=2):
    """轻量调度器替身（test_remote_credit_stream 同族，F6 对齐初值）。"""
    s = Sh30OnlineScheduler.__new__(Sh30OnlineScheduler)
    s.p_chunk = 512
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
    s._decode_task_load_cache = {}
    s._task_load_cache_capacity = _TASK_LOAD_CACHE_CAPACITY
    s._quota_tracker = None
    s.graph = GraphBatchBuilder(_make_config())
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
    from joint.hbm_port_flow_registry import HbmPortFlowRegistry
    s._hbm_ports = HbmPortFlowRegistry()
    s._kv_ledger_epoch = 0
    return s


from online.graph_batch_builder import GraphBatchBuilder  # noqa: E402


class GraphBatchBuilderForTest(GraphBatchBuilder):
    """GraphBatchBuilder 直通别名（替身注入点；保持与生产同一实现）。"""


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
    rt.joint_stage = JOINT_STAGE_D_RUNNING
    rt.completion_instance_index = EXEC_INSTANCE
    rt.joint_input_tokens = input_tokens
    rt.decode_tokens_consumed = consumed
    rt.remaining_chunks = 0
    rt.drain_block_ends = {2: 0, 3: 0}
    # 实际发射身份种子面（生产在 staging lease 获取后写入 runtime
    # pipeline_identity_inputs["decode"]；plan_dict 据此生成显式种子，
    # 远读 credit 发射按 rank 精确匹配）。夹具固定 allocation_id=1、
    # 目标 rank = 执行实例 TP 面 (0, 1)（EXEC_INSTANCE=0），与 credit
    # plan shard_specs 的目标 rank 同源。
    rt.pipeline_identity_inputs["decode"] = {
        "stage_epoch": int(rt.joint_boundary_epoch),
        "staging_allocation_id": 1,
        "target_ranks": (0, 1),
    }
    return rt


def _install_remote(scheduler, runtime):
    scheduler.runtime_by_request_id[runtime.request_id] = runtime
    scheduler.kv_manager = SimpleNamespace(
        _sessions={runtime.session_id: SimpleNamespace(
            working_kind=None, context_tokens=100,
            base_history_tokens=0, base_resident_prefix_layers=0)},
        tp_degree=2)
    runtime.remote_read_credit_plan = (
        scheduler._joint_remote_read_credit_plan(
            runtime, EXEC_INSTANCE))


def _all_builder_nodes(graph):
    nodes = []
    for builder in graph.builders.values():
        nodes.extend(builder.nodes)
    return nodes


def _train_plan_from_spec(plan, rt, spec, joiner_plans):
    # joiner dict 必须复用 _plan_remote_credit_slices 就地补全的同一
    # dict（prefill_decode_transfer / remote_credit_tail 注入面）。
    return {
        "train_id": plan["train_id"],
        "instance_index": EXEC_INSTANCE,
        "stage": "decode",
        "joiners": joiner_plans,
        "pass_spans": list(plan["pass_spans"]),
        "iterations": plan["iterations"],
        "sentinel": False,
        "prefill_start_member": None,
        "first_chunk_member": None,
        "drain_members": [],
        "exit_members": [{"request_id": rt.request_id}],
        "remote_credit": spec,
    }


# ==================================================== 1. 有界后端发射 ==

class BoundedBackendEmissionTest(unittest.TestCase):
    """§17.1：不为整轮预建事件图；按缓冲上限延迟发出下一块。"""

    def test_folded_body_node_count_is_constant_in_iterations(self):
        # 折叠列车体：decode 8 步与 64 步的列车节点数逐位相等——迭代数
        # 由 weight_passes/end barrier 聚合字节承载，不逐迭代预建节点
        #（步数×层数×块数的整轮事件图禁令）。plain 成员（无 credit）。
        counts = {}
        barrier_bytes = {}
        for decode in (8, 64):
            s = _scheduler()
            rt = _remote_runtime(request_id="rp%d" % decode, decode=decode)
            rt.joint_action = "stay"
            rt.origin_home_instance = None
            s.runtime_by_request_id[rt.request_id] = rt
            state = s.instances[EXEC_INSTANCE]
            state.active_decode.append(rt)
            state.active_decode_lookup.add(rt)
            plan = s._plan_train(state)
            self.assertEqual(plan["iterations"], decode)
            s._emit_train(state, plan, [], 1)
            counts[decode] = len(s.graph.batch["nodes"])
            barriers = [
                n for n in s.graph.batch["nodes"]
                if n["name"].endswith("_end_barrier")]
            barrier_bytes[decode] = sum(
                n["coll"]["bytes"] for n in barriers)
        self.assertEqual(counts[8], counts[64])
        # 聚合体真实携带迭代数（折叠≠丢工作）：end barrier 聚合字节随
        # 迭代数同比增长（weight_passes 语义在节点属性面可见）。
        self.assertEqual(barrier_bytes[64], 8 * barrier_bytes[8])
        self.assertGreater(barrier_bytes[8], 0)

    def test_credit_tiles_beyond_buffer_pair_delayed_not_prebuilt(self):
        # D09 延迟生成的中间态钉（此前只有事后全量形态断言）：compact
        # 发射下头部即时面 = 双缓冲块 1..2（块 1 主链 + 块 2 旁挂并行
        # 预取，recv 门都入 arms 账本）；块 3..M 记入待发账本，头部
        # descriptor 草稿只含块 1..2 的 endpoint run；体发射在消费锚
        # 逐块释放——生产等消费，任意时刻远读暂存 ≤2 块。
        s = _scheduler()
        rt = _remote_runtime(decode=8)
        _install_remote(s, rt)
        self.assertIsNotNone(rt.remote_read_credit_plan)
        state = s.instances[EXEC_INSTANCE]
        state.active_decode.append(rt)
        state.active_decode_lookup.add(rt)
        plan = s._plan_train(state)
        joiner_plans = [rt.plan_dict()]
        joiner_plans[0]["prefill_drain_block_ends"] = {2: 0, 3: 0}
        spec = s._plan_remote_credit_slices(plan, [rt], joiner_plans)
        self.assertIsNotNone(spec)
        train_plan = _train_plan_from_spec(plan, rt, spec, joiner_plans)
        s.graph._emit_train_head(train_plan)
        # 中间态：头部账本 = 双缓冲即时门 1..2 + 待发 3..M。
        arms = s.graph._credit_arms[rt.request_id]
        pending = s.graph._credit_pending_tiles[rt.request_id]
        self.assertEqual(sorted(arms), [1, 2])
        self.assertEqual(sorted(pending), list(range(3, 9)))
        names = [n["name"] for n in _all_builder_nodes(s.graph)]
        for block in range(3, 9):
            self.assertFalse(
                ["_credit%d_" % block in name for name in names if
                 "_credit%d_" % block in name],
                "tile %d was pre-built before its buffer-reuse release "
                "anchor" % block)
        # compact 头的物理节点面 = pipeline 锚（per-tile 事件在 descriptor
        # endpoint_runs 内）；草稿组只覆盖全局 tile 1..2。
        self.assertTrue(any("remote_credit_frontier" in name
                            for name in names))
        head_tiles = sorted({
            group["tile_axis"]["global_tile_first"]
            for draft in s.graph._pipeline_drafts.values()
            for group in draft.groups})
        self.assertEqual(head_tiles, [1, 2])
        # 体发射：每个消费锚体块释放其待发 tile（生产侧账本排空）。
        s.graph._emit_train_body(
            train_plan, list(train_plan["pass_spans"]),
            train_plan["iterations"], credit_blocks=spec["body_blocks"])
        self.assertFalse(
            s.graph._credit_pending_tiles.get(rt.request_id))
        all_tiles = sorted({
            group["tile_axis"]["global_tile_first"]
            for draft in s.graph._pipeline_drafts.values()
            for group in draft.groups})
        self.assertEqual(all_tiles, list(range(1, 9)))
        # 尾标记结清（账本清空——无跨列车残留）。
        s.graph._emit_train_tail_markers(train_plan, None)
        self.assertNotIn(rt.request_id, s.graph._credit_arms)
        self.assertNotIn(rt.request_id, s.graph._credit_pending_tiles)

    def test_pending_tile_leftover_fails_closed(self):
        # 待发 range 残留 = 释放链破损 → 尾标记 fail-closed；不人工
        # 插入 per-tile 字典项，直接保留头部发射后尚未释放的真范围。
        s = _scheduler()
        rt = _remote_runtime(decode=8)
        _install_remote(s, rt)
        state = s.instances[EXEC_INSTANCE]
        state.active_decode.append(rt)
        state.active_decode_lookup.add(rt)
        plan = s._plan_train(state)
        joiner_plans = [rt.plan_dict()]
        joiner_plans[0]["prefill_drain_block_ends"] = {2: 0, 3: 0}
        spec = s._plan_remote_credit_slices(plan, [rt], joiner_plans)
        train_plan = _train_plan_from_spec(plan, rt, spec, joiner_plans)
        s.graph._emit_train_head(train_plan)
        # 以空 arm 显式隔离待发账本断言；range 仍保留真实块 3..M。
        s.graph._credit_arms[rt.request_id] = {}
        with self.assertRaises(RuntimeError) as ctx:
            s.graph._emit_train_tail_markers(train_plan, None)
        self.assertIn("were never released by a consuming body block",
                      str(ctx.exception))


# ========================================== 2. 合并止于真实外部事件 ==

class MergeStopsAtRealBoundaryTest(unittest.TestCase):
    """§17.1：等价批处理合并遇到真实外部事件必须停止。

    列车合并（折叠体）的停止面 = ①交付治理上界 T_max
    （SH_TRAIN_MAX_ITER 缺省 "8"——sh30_online_scheduler.__init__，
    §3.2.8/§7.4；每次到达即重新规划的真实调度边界）；②队列头 drain
    迭代（先验真实的完成事件）；③纯 decode 的最后一个工作迭代（成员
    退出 = 真实完成事件，不向上界补齐空转）。"""

    def test_merge_horizon_capped_at_delivered_t_max(self):
        s = _scheduler()
        s._train_max_iter = 8  # 交付默认（__init__ env 缺省值）
        rt = _remote_runtime(request_id="rlong", decode=100)
        rt.joint_action = "stay"
        rt.origin_home_instance = None
        s.runtime_by_request_id[rt.request_id] = rt
        state = s.instances[EXEC_INSTANCE]
        state.active_decode.append(rt)
        state.active_decode_lookup.add(rt)
        plan = s._plan_train(state)
        # 上界即合并停止面：迭代数恰为治理上界（哨兵承载边界重规划），
        # 成员未退出。
        self.assertEqual(plan["iterations"], 8)
        self.assertTrue(plan["sentinel"])
        self.assertEqual(plan["exit_members"], [])

    def test_merge_stops_at_natural_drain_before_cap(self):
        s = _scheduler()
        s._train_max_iter = 8
        rt = _remote_runtime(request_id="rlong", decode=100)
        rt.joint_action = "stay"
        rt.origin_home_instance = None
        s.runtime_by_request_id[rt.request_id] = rt
        head = _head_runtime(chunks=2)
        s.runtime_by_request_id[head.request_id] = head
        state = s.instances[EXEC_INSTANCE]
        state.qp.append(head)
        state.active_decode.append(rt)
        state.active_decode_lookup.add(rt)
        plan = s._plan_train(state)
        # 合并止于 drain（真实先验完成事件），不越过它补齐 decode。
        self.assertEqual(plan["iterations"], 2)
        self.assertEqual(plan["drain_members"], [head.request_id])
        self.assertFalse(plan["sentinel"])

    def test_merge_ends_at_last_work_not_padded_to_cap(self):
        s = _scheduler()
        s._train_max_iter = 8
        rt = _remote_runtime(request_id="rshort", decode=3)
        rt.joint_action = "stay"
        rt.origin_home_instance = None
        s.runtime_by_request_id[rt.request_id] = rt
        state = s.instances[EXEC_INSTANCE]
        state.active_decode.append(rt)
        state.active_decode_lookup.add(rt)
        plan = s._plan_train(state)
        # 成员迭代 3 处退出（真实事件）——合并恰止于最后一个工作迭代。
        self.assertEqual(plan["iterations"], 3)
        self.assertEqual(plan["exit_members"], [rt.request_id])
        self.assertFalse(plan["sentinel"])


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


# ====================================== 3. 模拟 HBM 存储形态（§17.1）==

class SimulatedHbmStorageShapeTest(unittest.TestCase):
    """模拟 HBM 只存字节计数/区间身份/引用/依赖——状态形状与 token 数
    无关（不申请与模拟 KV 字节等大的宿主数组）。"""

    _SEQ_TYPES = (list, tuple, dict, set)

    def _shape(self, state):
        shape = {}
        for name, value in vars(state).items():
            if isinstance(value, self._SEQ_TYPES):
                shape[name] = ("seq", len(value))
            else:
                shape[name] = ("scalar", type(value).__name__)
        return shape

    def test_session_state_shape_is_token_count_invariant(self):
        small = SessionKVState(
            session_id="s", location="local_hbm", instance_index=0,
            context_tokens=8, total_bytes=64,
            shard_bytes=(32, 32), resident_prefix_layers=4)
        big = SessionKVState(
            session_id="s", location="local_hbm", instance_index=0,
            context_tokens=10 ** 6, total_bytes=64 * 125000,
            shard_bytes=(4 * 10 ** 6, 4 * 10 ** 6),
            resident_prefix_layers=4)
        # 属性面同形：字段集合与序列长度不随 token 数增长（shard 字节
        # 向量长度 = TP 度，非 token 数）。
        self.assertEqual(
            self._shape(small), self._shape(big))
        for name, value in vars(big).items():
            if isinstance(value, self._SEQ_TYPES):
                self.assertLessEqual(
                    len(value), 8, name)
        # 字节以计数器承载（区间身份 resident_prefix_layers/字节计数
        # total_bytes），非载荷：状态的文本投影有界。
        self.assertEqual(big.total_bytes, 8 * 10 ** 6)
        self.assertLess(len(repr(big)), 4096)

    def test_node_hbm_state_is_scalar_byte_counters(self):
        state = NodeHBMState(
            rank=0, instance_index=0, capacity_bytes=10 ** 9,
            model_weight_bytes=1 << 30, kv_cache_bytes=8 * 10 ** 6)
        for name, value in vars(state).items():
            self.assertIsInstance(value, int, name)
        self.assertEqual(state.kv_cache_bytes, 8 * 10 ** 6)
        self.assertEqual(
            state.used_bytes,
            state.model_weight_bytes + state.kv_cache_bytes)
        self.assertEqual(
            state.remaining_bytes,
            state.capacity_bytes - state.used_bytes)
        snapshot = state.snapshot()
        self.assertEqual(snapshot.kv_cache_bytes, 8 * 10 ** 6)


# =============================== 4. producer/write/consume/release ==

class CreditStreamTimeClosureTest(unittest.TestCase):
    """验收"producer/write/consume/release 时间闭合"（credit tile 四类
    事件逐块闭合 + 图级字节守恒 + 账本结清）。体块门消费恰一次由
    test_remote_credit_stream.test_i2_body_block_gates 钉住，此处不重
    复；释放链 fork 锚定由 test_buffer_reuse_release_chain 钉住。"""

    def _emit(self, decode=8):
        s = _scheduler()
        rt = _remote_runtime(decode=decode)
        _install_remote(s, rt)
        state = s.instances[EXEC_INSTANCE]
        state.active_decode.append(rt)
        state.active_decode_lookup.add(rt)
        plan = s._plan_train(state)
        s._emit_train(state, plan, [rt], 1)
        return s, rt, plan

    def test_four_event_chain_closes_per_tile(self):
        s, rt, plan = self._emit(decode=8)
        # compact 发射面：per-tile 四类事件在 decode_remote_credit
        # descriptor 的 endpoint_runs/legs 内（物理节点面只有 pipeline
        # 锚）；每 draft 是一个 owner rank 视角，run.rank 跨源/目标
        # 两侧（源 rank 2/3 发 data_send、收 fill_ack_recv；目标
        # rank 0/1 收 data_recv、回 fill_ack_send）。
        drafts = sorted(
            (draft for draft in s.graph._pipeline_drafts.values()
             if draft.stream_kind == "decode_remote_credit"),
            key=lambda draft: draft.identity.rank)
        self.assertEqual(
            [draft.identity.rank for draft in drafts], [0, 1])
        legs_by_tile = {}
        for draft in drafts:
            tile_by_group = {
                group["group_ordinal"]:
                    group["tile_axis"]["global_tile_first"]
                for group in draft.groups}
            for run in draft.endpoint_runs:
                tile = tile_by_group[run["group_ordinal"]]
                for leg in run["legs"]:
                    legs_by_tile.setdefault(tile, []).append(
                        (leg["endpoint"], int(run["rank"])))
        # producer/write：每 tile 每 shard 的 data/fill_ack 四类腿
        # 成对在场；release：tile 3..M 每块一对释放 ack（exec 目标
        # rank 发 ack_send、home 源 rank 收 ack_recv——缓冲归还因果
        # 由 tag_allocation_runs 派生的 tag 配对承载）；tile 1..2 双
        # 缓冲初始皆空，无释放依赖（首块发射约束的前置面）。
        self.assertEqual(sorted(legs_by_tile), list(range(1, 9)))
        for tile in sorted(legs_by_tile):
            legs = sorted(legs_by_tile[tile])
            expected = sorted([
                ("data_recv", 0), ("data_recv", 1),
                ("data_send", 2), ("data_send", 3),
                ("fill_ack_recv", 2), ("fill_ack_recv", 3),
                ("fill_ack_send", 0), ("fill_ack_send", 1)])
            if tile >= 3:
                expected = sorted(expected + [
                    ("release_ack_recv", 2), ("release_ack_recv", 3),
                    ("release_ack_send", 0), ("release_ack_send", 1)])
            self.assertEqual(legs, expected, tile)
        # I1 图级守恒：descriptor 分组 shard 字节面 Σ（两 owner draft
        # 合计）≡ 计划总量。
        total = rt.remote_read_credit_plan["total_bytes"]
        group_bytes = sum(
            shard["total_data_bytes"]
            for draft in drafts for group in draft.groups
            for shard in group["shards"])
        self.assertEqual(group_bytes, total)
        # consume：每 tile 恰一条 consume completion run（recv 门由
        # 覆盖其迭代区间的体块消费锚结清——门/消费一一对位）。
        for draft in drafts:
            self.assertEqual(len(draft.consume_completion_runs), 8)
        # release（账本面）：列车尾标记结清后无跨列车残留——producer
        # 侧无未锚定生产、消费侧无未消费门。
        self.assertNotIn(rt.request_id, s.graph._credit_arms)
        self.assertNotIn(rt.request_id, s.graph._credit_pending_tiles)


# ==================================== 5. D16 对齐面收口（tile 同源）==

class TilePricingAlignmentTest(unittest.TestCase):
    """双缓冲闭式事务切块 tile = 执行切片 tile（§12.2 相同规则；D09
    交付批登记的两侧粒度差随 D16 收口废除；修复轮二 2026-09-27 起
    tile = 配置值固定——不随基础区间放大，宽基础区间按配置 tile 计块
    数 = ⌈rank 流字节/(配置 tile×kv)⌉×shard 数）。"""

    def test_wide_base_prices_at_configured_execution_tile(self):
        # 基础区间 1000 token（> 2×配置 tile 256）→ tile 恒 = 配置 256
        # token（固定，修复轮二废除 ⌈base/2⌉ 放宽——v2.0 §17.1 不做
        # 清单"不得为减少仿真事件把容量/缓冲暗改无限"）：kv = 10 B/
        # token（request 惰性基准）⇒ tile 流 2560 B。prefill 腿 1 遍
        # rank 流 10000 B → ⌈10000/2560⌉ = 4 块/rank × 2 shard = 8 块
        # （含尾块）；decode 腿 10 遍 100000 B → 40 块/rank × 2 = 80
        # 块（含尾块）。x = 2×(2560 B ÷ 10 GB/s) + 10 ns 时延 = 522
        # （compute 除数 2——exec_flows=2）。
        model = _jcm(_anchor_loads())
        candidate = model.estimate_P_with_continuation(
            session=_session(history_bytes=(10000, 10000),
                             history_tokens=1000),
            request=_request(), instance_index=1,
            action=ACTION_REMOTE, remote_enabled=True)
        self.assertTrue(candidate.applicable)
        notes = candidate.breakdown.notes
        self.assertIn("remote_read_tile_tokens=256", notes)
        self.assertNotIn("remote_read_tile_tokens=500", notes)
        # 块数按逐 rank 求和披露（2 shard）：prefill 腿 4 块/rank（3
        # 整 + 1 尾）× 2 = 8；decode 腿（修复轮三 H1 逐 pass 重切）=
        # passes×⌈tokens/tile⌉ = 10×4 = 40 块/rank（每 pass 3 整 + 1
        # 尾、尾块并入整块服务）× 2 = 80——旧连续链 ⌈100000/2560⌉ =
        # 40 块/rank（39 整 + 1 尾）数值相同（重切前后块数巧合相等，
        # 腿值增大：尾块按整块服务计），锚保留原值并在注释说明形态。
        self.assertIn("prefill_tile_tiles=8_x=522_c=6", notes)
        self.assertIn("decode_tile_tiles=80_x=522_c=0", notes)

    def test_narrow_base_keeps_config_tile_d09_anchor(self):
        # 窄基础区间 tile 同为配置 256 token（固定口径下窄/宽同源——
        # 与 D09 手算锚 test_joint_credit_pricing 一致，对齐不改小会
        # 话口径）。
        model = _jcm(_anchor_loads())
        candidate = model.estimate_P_with_continuation(
            session=_session(history_bytes=(1000, 1000),
                             history_tokens=100),
            request=_request(), instance_index=1,
            action=ACTION_REMOTE, remote_enabled=True)
        self.assertIn(
            "remote_read_tile_tokens=256",
            candidate.breakdown.notes)

    def test_boundary_continuation_leg_prices_at_execution_tile(self):
        # D 边界 continuation 腿与本边界暂存预约同一裁决点：基础区间
        # 1000 token → tile = 配置 256 token（固定）× 128 B = 32768 B；
        # 修复轮三 H1 逐 pass 重切：full_tiles = passes×⌈tokens/tile⌉ =
        # 8×⌈1000/256⌉ = 8×4 = 32 块/rank（每 pass 3 整 + 1 尾、尾块并
        # 入整块服务）× 2 shard = 64 块——旧连续链 ⌈1024000/32768⌉ =
        # 32 块/rank（31 整 + 1 尾）数值相同（块数巧合相等、腿值增大：
        # 尾块按整块服务计），锚保留原值并在注释说明形态。
        # x = 2×3277 + 10 = 6564（tile 流 32768 B ÷ 10 GB/s = 3276.8 ns
        # 取整 3277；除数 2 + 10 ns 时延）。
        kv = 128
        boundary = PDBoundaryView(
            session_id="s", home_instance=2, p_instance_index=0,
            p_action="remote-read", history_tokens=1000,
            working_bytes_by_tp_rank=(kv * 40,) * 2,
            base_bytes_by_tp_rank=(kv * 1000,) * 2,
            decode_steps=8, base_tokens=1000, kv_token_bytes=kv)
        model = _jcm({0: _load(0, 0), 1: _load(1, 0), 2: _load(2, 0)},
                     remote_read_kv_token_bytes=kv)
        candidate = model.estimate_D_remaining(
            boundary=boundary, instance_index=0,
            action=ACTION_REMOTE, remote_enabled=True)
        self.assertTrue(candidate.applicable)
        self.assertTrue(candidate.is_continuation)
        self.assertIn("decode_tile_tiles=64_x=6564_c=0",
                      candidate.breakdown.notes)

    def test_multi_source_leg_counts_per_source_tile_chains(self):
        # 修复轮三 H1（执行口径逐源和）：双 holder 各 128 token（增量
        # U@i_P=0 + 基础区间@home=2，目标 j=1 两者全缺）——tile = 配置
        # 256 token 下旧联合口径每 pass = ⌈256/256⌉ = 1 块，执行切片
        # 实际逐源独立 tile 链 = ⌈128/256⌉+⌈128/256⌉ = 2 块（源内尾
        # 块不可跨源合并）。decode_steps=1 → full_tiles = 2 块/rank ×
        # 2 shard = 4：断 decode_tile_tiles=4（= 2×shard 数，执行口径
        # 逐源和）；x 与单源同形（tile 流 32768 B、单跳路径同构）=
        # 6564。
        kv = 128
        boundary = PDBoundaryView(
            session_id="s", home_instance=2, p_instance_index=0,
            p_action="remote-read", history_tokens=228,
            working_bytes_by_tp_rank=(kv * 128,) * 2,
            base_bytes_by_tp_rank=(kv * 128,) * 2,
            decode_steps=1, base_tokens=128, kv_token_bytes=kv)
        model = _jcm({0: _load(0, 0), 1: _load(1, 0), 2: _load(2, 0)},
                     remote_read_kv_token_bytes=kv)
        candidate = model.estimate_D_remaining(
            boundary=boundary, instance_index=1,
            action=ACTION_REMOTE, remote_enabled=True)
        self.assertTrue(candidate.applicable)
        notes = candidate.breakdown.notes
        # 两个来源按权威 holder 披露（增量@0 + 基础@2）。
        self.assertIn("boundary_remote_sources=0,2", notes)
        # 块数 = 逐源和（执行口径）：2 块/rank/pass × 2 shard = 4——
        # 旧联合口径 ⌈256/256⌉ = 1 块/rank → 2，重切消除联合块数对逐
        # 源尾块的低估（只增不减）。
        self.assertIn("decode_tile_tiles=4_x=6564_c=0", notes)

    def test_partial_base_leg_uses_prefix_layer_rate(self):
        # 修复轮三 H2（PARTIAL 基块内率局部统一）：envelope 披露
        # base_layers=2/working_layers=4 → 腿块内每 token 率 =
        # 128×2//4 = 64 B/token（前缀层率，执行侧逐源 specs 字节//
        # tokens 同口径；boundary.kv_token_bytes 本身不动——decode 增
        # 长语义超载）。tile 配置 1000 B：全层率 128 → tile = 7 token
        # （tile 流 896 B、x = 2×⌈89.6⌉+10 = 190、每 pass ⌈1000/7⌉ =
        # 143 块 × 8 遍 × 2 shard = 2288）；前缀率 64 → tile = 15
        # token（tile 流 960 B、x = 2×96+10 = 202、67×8×2 = 1072）—
        # 率统一后 token↔字节反推自洽（tiles×tile 流 ≈ 前缀真字节，
        # 旧全层率对 PARTIAL 基 2× 高估块内字节）。envelope 缺省回退
        # 全层 = 上一用例（直接构造夹具逐位兼容）。
        kv = 128

        def _boundary():
            return PDBoundaryView(
                session_id="s", home_instance=2, p_instance_index=0,
                p_action="remote-read", history_tokens=1100,
                working_bytes_by_tp_rank=(kv * 100,) * 2,
                base_bytes_by_tp_rank=(kv * 1000,) * 2,
                decode_steps=8, base_tokens=1000, kv_token_bytes=kv)

        loads = {0: _load(0, 0), 1: _load(1, 0), 2: _load(2, 0)}
        full_rate = _jcm(loads, remote_read_kv_token_bytes=kv,
                         remote_read_tile_bytes="1000").estimate_D_remaining(
            boundary=_boundary(), instance_index=0,
            action=ACTION_REMOTE, remote_enabled=True)
        self.assertIn("decode_tile_tiles=2288_x=190_c=0",
                      full_rate.breakdown.notes)
        wrapped = BoundaryRuntimeStateEnvelope(
            _boundary(), working_layers=4, base_layers=2)
        partial = _jcm(loads, remote_read_kv_token_bytes=kv,
                       remote_read_tile_bytes="1000").estimate_D_remaining(
            boundary=wrapped, instance_index=0,
            action=ACTION_REMOTE, remote_enabled=True)
        self.assertIn("decode_tile_tiles=1072_x=202_c=0",
                      partial.breakdown.notes)


if __name__ == "__main__":
    unittest.main()
