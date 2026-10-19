#!/usr/bin/env python3
"""test_d16_runtime_linkage_bounded_backend.py -- D16 运行时联动与后端
有界性的验收单测（joint修改文档 第二批 D16 条目 + v2.0 §17.1/C13/
§12.2；与既有 per-stream 结构钉测互补，不重复其断言面）。

条目红线逐条落测：
  1. 后端有界活跃流/延迟生成（v2.0 §17.1）——折叠列车体的节点数与
     迭代数无关（不为整轮"步数×层数×块数"预建事件图；迭代数由
     weight_passes/聚合字节承载）。
  2. 模拟 HBM 只存字节计数/区间身份/引用/依赖（§17.1）——
     SessionKVState/NodeHBMState 状态形状与 token 数无关（不申请与
     模拟 KV 字节等大的宿主数组；字节以计数器承载）。
  3. 等价批处理合并遇真实外部事件即停（§17.1"合并遇到下一真实外部
     事件必须停止"）——列车合并的上界 = 交付默认治理边界 T_max
     （SH_TRAIN_MAX_ITER 缺省 "8"，sh30_online_scheduler.__init__）；
     自然 drain 边界先于上界时合并恰止于 drain 迭代；纯 decode 合并
     恰止于最后一个工作迭代（成员退出即真实事件，不补齐到上界）。

（原 remote credit 双缓冲发射/待发账本/时间闭合与 D09/D16 tile 计价
对齐面用例——credit tile 七八类——已随 2026-10-14 远读退役删除：
发射链、_credit_arms/_credit_pending_tiles 账本、pipeline descriptor
协议与 ACTION_REMOTE 计价腿一并退役。）

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
from joint.joint_cost_model import LinkFlowRegistry  # noqa: E402
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


# ================================================== 执行侧夹具（有界性）==

def _scheduler(layers=2):
    """轻量调度器替身（__new__ 范式，F6 对齐 __init__ 初值）。"""
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
    s.joint_config = parse_joint_config({})
    s._joint_flows = LinkFlowRegistry()
    s._pool_ports = SimpleNamespace(
        register=lambda *args, **kwargs: None,
        release_owner=lambda *args, **kwargs: None)
    from joint.hbm_port_flow_registry import HbmPortFlowRegistry
    s._hbm_ports = HbmPortFlowRegistry()
    s._kv_ledger_epoch = 0
    return s


def _decode_runtime(request_id="rd", *, decode=8, input_tokens=16,
                    consumed=0, session_id="s_rd"):
    """stay@本实例的纯 decode 成员（原远读 runtime 夹具的动作面收缩
    ——保留 _plan_train/_emit_train 消费的成员字段面）。"""
    rt = _OnlineRequestRuntime({
        "request_id": request_id,
        "session_id": session_id,
        "turn_index": 0, "queue_index": 0,
        "prefill_length": input_tokens, "decode_length": decode,
        "history_tokens_before": 0,
        "prefill_context_tokens": input_tokens,
        "final_context_tokens": input_tokens + decode,
    }, 512)
    rt.joint_action = "stay"
    rt.origin_home_instance = None
    rt.prefill_instance_index = EXEC_INSTANCE
    rt.decode_instance_index = EXEC_INSTANCE
    rt.joint_stage = JOINT_STAGE_D_RUNNING
    rt.completion_instance_index = EXEC_INSTANCE
    rt.joint_input_tokens = input_tokens
    rt.decode_tokens_consumed = consumed
    rt.remaining_chunks = 0
    rt.drain_block_ends = {2: 0, 3: 0}
    return rt


# ==================================================== 1. 有界后端发射 ==

class BoundedBackendEmissionTest(unittest.TestCase):
    """§17.1：不为整轮预建事件图；按缓冲上限延迟发出下一块。"""

    def test_folded_body_node_count_is_constant_in_iterations(self):
        # 折叠列车体：decode 8 步与 64 步的列车节点数逐位相等——迭代数
        # 由 weight_passes/end barrier 聚合字节承载，不逐迭代预建节点
        #（步数×层数×块数的整轮事件图禁令）。plain 成员。
        counts = {}
        barrier_bytes = {}
        for decode in (8, 64):
            s = _scheduler()
            rt = _decode_runtime(request_id="rp%d" % decode, decode=decode)
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
        rt = _decode_runtime(request_id="rlong", decode=100)
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
        rt = _decode_runtime(request_id="rlong", decode=100)
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
        rt = _decode_runtime(request_id="rshort", decode=3)
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


if __name__ == "__main__":
    unittest.main()
