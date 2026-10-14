#!/usr/bin/env python3
"""test_d03_frontier_alignment.py -- D03 修复轮二（2026-09-27）：P 准备
腿发射 frontier 对齐构图事实。

构图事实（graph_batch_builder "frontier 接续裁决"：准入动作链到该
rank 当前 frontier）：目标实例列车在飞时，P 准备腿物理排在在飞列车
之后——"决策时刻发射、与排队重叠"只在实例空闲时成立。计价入口
``estimate_P_with_continuation`` 新增逐候选目标实例
``prep_frontier_ns``（决策相对 ns；空闲 = 0 = 旧口径逐字节一致）：

* 准备腿锚点 = max(A, frontier)（stay 池恢复 / copy 前缀链+池后缀 /
  remote 前缀读流 tile 到达全部换锚）；
* notes：frontier > 0 时 ``prep_issue_frontier=P_prep_at_target_
  frontier(+Nns)`` + ``prep_frontier_ns=N``；= 0 时保持
  ``P_prep_at_decision_epoch``（空闲对照锚不漂移）；
* 调度器侧（sh30 ``_prep_frontier_by_instance``）：在飞列车冻结快照
  的 chunk 负载 + 成员 participation 步服务（真实步数，非 horizon
  估计）；无在飞列车 = 0。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 joint/test_d03_frontier_alignment.py   （或 pytest）
"""
import os
import sys
import unittest
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for _p in (_HERE, _PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from joint.joint_cost_model import (  # noqa: E402
    ACTION_COPY,
    ACTION_REMOTE,
    ACTION_STAY,
    InstanceLoadView,
    JointCostModel,
    JointHardwareRates,
    LinkFlowRegistry,
    RequestView,
    ServiceFactors,
    SessionKVView,
)


def _rates(pool_latency_ns=100, d2d_latency_ns=10):
    return JointHardwareRates.from_gbps(
        noc_link_gbps=10.0, pool_port_gbps=5.0, local_hbm_gbps=100.0,
        d2d_latency_ns=d2d_latency_ns, pool_latency_ns=pool_latency_ns)


def _load(index, total=0, running=0, active=0,
          remaining=(10 ** 9, 10 ** 9), reclaimable=None):
    return InstanceLoadView(
        instance_index=index, queued_task_load_ns=total,
        running_task_load_ns=running, active_decode_task_load_ns=active,
        hbm_remaining_bytes_by_tp_rank=remaining,
        reclaimable_bytes_by_tp_rank=(
            remaining if reclaimable is None else reclaimable))


def _session(home=1, resident=1, history_bytes=(1000, 1000),
             missing=(0, 0), location="local_hbm", prefix=4,
             history_tokens=100):
    return SessionKVView(
        session_id="s", home_instance=home, resident_instance=resident,
        location=location, history_tokens=history_tokens,
        resident_prefix_layers=prefix,
        history_bytes_by_tp_rank=history_bytes,
        missing_bytes_by_tp_rank=missing)


def _request(input_tokens=50, decode=10, input_bytes=(25, 25)):
    return RequestView(
        request_id="r", session_id="s", input_tokens=input_tokens,
        history_tokens_before=100, estimated_decode_tokens=decode,
        horizon_source="session_online_mean",
        input_kv_bytes_by_tp_rank=input_bytes)


def _route(source, target):
    return ((source, target), 1)


def _model(loads, *, layers=4):
    return JointCostModel(
        rates=_rates(), loads=loads, flow_registry=LinkFlowRegistry(),
        service_factors=ServiceFactors(),
        prefill_ns_per_token=1.0, decode_ns_per_token=2.0,
        model_layers=layers, instance_tp_size=2, route_fn=_route)


def _note_value(breakdown, key):
    prefix = key + "="
    for note in breakdown.notes:
        if note.startswith(prefix):
            return int(note[len(prefix):])
    raise AssertionError(f"note {key!r} missing in {breakdown.notes}")


FRONTIER_NS = 4000


class PPrepFrontierAlignmentTest(unittest.TestCase):
    """忙目标（frontier>0）：准备腿不再从 0 起算；空闲：与旧值一致。"""

    def _stay_partial_candidate(self, prep_frontier_ns=0):
        """stay-PARTIAL：缺失后缀 (1000,1000)/rank 池恢复（D04 逐
        shard：restore = 100 + 1000/5 = 300ns）门控后缀层——腿在关键
        路径上（Q=0、A=0 ⇒ 腿决定后缀层就绪）。"""
        model = _model({0: _load(0), 1: _load(1)})
        session = _session(
            home=1, resident=1, history_bytes=(1000, 1000),
            missing=(1000, 1000), location="partial_hbm_remote", prefix=2)
        return model.estimate_P_with_continuation(
            session=session, request=_request(), instance_index=1,
            action=ACTION_STAY, remote_enabled=True,
            prep_frontier_ns=prep_frontier_ns)

    def test_busy_target_pool_restore_starts_after_frontier(self):
        # 池恢复腿自 max(A, frontier) 起算：忙目标的就绪腿比空闲恰晚
        # frontier（Q=0/A=0、腿门控后缀层 ⇒ 全时间轴平移 frontier）。
        idle = self._stay_partial_candidate(0)
        busy = self._stay_partial_candidate(FRONTIER_NS)
        self.assertTrue(idle.applicable and busy.applicable)
        self.assertEqual(busy.cost_ns - idle.cost_ns, FRONTIER_NS,
                         "transfer-gated timeline must shift by exactly "
                         "the target frontier when the leg dominates")
        # notes 口径：忙目标披露实际发射 frontier。
        self.assertIn(
            f"prep_frontier_ns={FRONTIER_NS}", busy.breakdown.notes)
        self.assertIn(
            "prep_issue_frontier=P_prep_at_target_frontier"
            f"(+{FRONTIER_NS}ns)",
            busy.breakdown.notes)
        self.assertIn("prep_issue_frontier=P_prep_at_decision_epoch",
                      idle.breakdown.notes)

    def test_idle_target_matches_legacy_path(self):
        # 对照锚：frontier=0（缺省/显式 0）与不传参的旧调用逐值一致。
        legacy = self._stay_partial_candidate()
        explicit_zero = self._stay_partial_candidate(0)
        self.assertEqual(legacy.cost_ns, explicit_zero.cost_ns)
        self.assertEqual(
            sorted(legacy.breakdown.notes),
            sorted(explicit_zero.breakdown.notes))

    def test_busy_copy_legs_shift_and_idle_unchanged(self):
        # copy（LOCAL 别处）：NoC 前缀交接链 + 计算门——忙目标成本 ≥
        # 空闲；交接链首墙 notes 平移可核对（Q=0 时链头 = frontier）。
        def candidate(prep_frontier_ns=0):
            model = _model({0: _load(0), 1: _load(1)})
            session = _session(home=0, resident=0, location="local_hbm",
                               prefix=4)
            return model.estimate_P_with_continuation(
                session=session, request=_request(), instance_index=1,
                action=ACTION_COPY, remote_enabled=True,
                prep_frontier_ns=prep_frontier_ns)

        idle = candidate(0)
        busy = candidate(FRONTIER_NS)
        self.assertTrue(idle.applicable and busy.applicable)
        self.assertGreater(busy.cost_ns, idle.cost_ns)
        self.assertEqual(
            _note_value(busy.breakdown, "copy_handoff_first_chunk_ns")
            - _note_value(idle.breakdown, "copy_handoff_first_chunk_ns"),
            FRONTIER_NS,
            "copy handoff chain head must start at max(A, frontier)")

    def test_busy_remote_read_stream_shifts(self):
        # remote-read：前缀读流 tile 到达闭式 start_ns = max(A,
        # frontier)——忙目标首前向递推迟后、成本 ≥ 空闲（腿在关键
        # 路径上时差额 > 0）。
        def candidate(prep_frontier_ns=0):
            model = _model({0: _load(0), 1: _load(1)})
            session = _session(
                home=0, resident=0, history_bytes=(40000, 40000),
                location="local_hbm", prefix=4)
            return model.estimate_P_with_continuation(
                session=session, request=_request(), instance_index=1,
                action=ACTION_REMOTE, remote_enabled=True,
                prep_frontier_ns=prep_frontier_ns)

        idle = candidate(0)
        busy = candidate(FRONTIER_NS)
        self.assertTrue(idle.applicable and busy.applicable)
        self.assertGreater(busy.cost_ns, idle.cost_ns,
                           "remote read stream anchored at the target "
                           "frontier must delay the busy-target timeline")

    def test_queue_only_target_keeps_decision_epoch(self):
        # 排队（Q>0）不推 frontier：frontier 是发射门（在飞列车），排队
        # 工作物理在准备之后的列车里——Q=5000、frontier=0 时准备仍自决
        # 策时刻起估（旧"与排队重叠"口径保留，notes 仍 decision_epoch）。
        model = _model({0: _load(0), 1: _load(1, total=5000)})
        candidate = model.estimate_P_with_continuation(
            session=_session(home=1, resident=1),
            request=_request(), instance_index=1,
            action=ACTION_STAY, remote_enabled=True)
        self.assertIn("prep_issue_frontier=P_prep_at_decision_epoch",
                      candidate.breakdown.notes)


def _graph_config():
    """构图器最小 config（test_graph_batch_builder 同款灯具：实例 0 =
    ranks (0,1)、实例 1 = ranks (2,3)）。"""
    return SimpleNamespace(
        npus_count=4,
        remote_operand_loads=False,
        trace_granularity="request_aggregated",
        inference_groups=[
            SimpleNamespace(ranks=(0, 1), pg_name="tp_prefill"),
            SimpleNamespace(ranks=(2, 3), pg_name="tp_decode"),
        ],
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


def _edge_sources_of(builder, rank, node_id):
    return {
        edge["from"]
        for edge in builder.batch["parent_edges"]
        if edge["to"] == node_id and edge["rank"] == rank
    }


# D03 端到端对账的**同一冻结在飞列车快照**（三方分测共用唯一来源）：
# 两段 prefill chunk + 一个 decode 成员参与 2 步。
TRAIN_CHUNKS = ((128, 128), (128, 256))
TRAIN_MEMBER_ID = "r_inflight"
TRAIN_MEMBER_PARTICIPATION = 2
TRAIN_MEMBER_CTX = 256
ADMISSION_REQUEST = "session_d03_e2e_request_1"


def _local_hit_transfer():
    """joiner 的 Prefill→Decode local_hit（零迁移节点；同
    test_graph_batch_builder 灯具口径）。"""
    from face_scheduler import KVTransfer
    return KVTransfer(
        kind="local_hit",
        phase="prefill_to_decode",
        reason="d03_frontier_e2e_test",
        session_id="session_d03_e2e",
        trigger_request_id=TRAIN_MEMBER_ID,
        source_instance_index=0,
        target_instance_index=1,
        total_bytes=0,
        shards=(),
        model_layers=2,
        layer_start=0,
        layer_end=2,
        resident_prefix_layers_before=2,
        resident_prefix_layers_after=2,
    )


def _inflight_train_plan():
    """构图器在飞列车 plan：体 spans = [chunk 序列] + [成员连续 decode
    段]（sh30 _plan_train 平铺口径，与 _prep_frontier_by_instance 消费
    的 in_flight_train 冻结快照同源同形）。"""
    member_spans = [
        (1, TRAIN_MEMBER_CTX + step)
        for step in range(1, TRAIN_MEMBER_PARTICIPATION + 1)
    ]
    return {
        "train_id": "batch_train_i1_d03",
        "instance_index": 1,
        "stage": "decode",
        "joiners": [{
            "request_id": TRAIN_MEMBER_ID,
            "session_id": "session_d03_e2e",
            "turn_index": 0,
            "queue_index": 0,
            "prefill_instance_index": 0,
            "decode_instance_index": 1,
            "prefill_context_tokens": TRAIN_MEMBER_CTX,
            "decode_evictions": [],
            "prefill_decode_transfer": _local_hit_transfer(),
            "prefill_drain_block_ends": {rank: 0 for rank in (0, 1)},
        }],
        "pass_spans": list(TRAIN_CHUNKS) + member_spans,
        "iterations": TRAIN_MEMBER_PARTICIPATION,
        "prefill_start_member": None,
        "first_chunk_member": None,
        "drain_members": [],
        "exit_members": [],
    }


def _inflight_train_snapshot():
    """sh30 侧 in_flight_train 冻结快照（_prep_frontier_by_instance 的
    消费形态）——与 _inflight_train_plan 同一列车的事实快照。"""
    return {
        "prefill_chunk_spans": list(TRAIN_CHUNKS),
        "members": [(TRAIN_MEMBER_ID, TRAIN_MEMBER_PARTICIPATION)],
    }


def _admission_plan_busy_target():
    """准入请求（P 实例 = 忙目标实例 1）：turn-0 到达门 + 准入动作链
    经 emit_admission_batch 真实发射。"""
    return {
        "request_id": ADMISSION_REQUEST,
        "session_id": "session_d03_e2e",
        "turn_index": 0,
        "queue_index": 0,
        "prefill_instance_index": 1,
        "decode_instance_index": 1,
        "admission_time_ns": 1000,
        "history_location_before": None,
        "history_transfer": None,
        "history_evictions": [],
        "prefill_evictions": [],
        "history_tokens_before": 0,
        "prefill_context_tokens": 300,
    }


class FrontierEmissionOrderReconciliationTest(unittest.TestCase):
    """D03 端到端发射序对账（钉测轮 2026-09-28）：同一冻结在飞列车快
    照驱动三方分测——

      1. 调度器快照（sh30 ``_prep_frontier_by_instance``）＝真实方法；
      2. 预测器锚（JCM ``estimate_P_with_continuation``）消费**同一个
         值**并如实披露（锚点同源；对应 sh30 准入积分面
         ``prep_frontier_by_instance=self._prep_frontier_by_instance()``
         的取值合同）；
      3. 构图器父链（真实 ``emit_iteration_train`` + ``emit_admission_
         batch`` 发射）：忙目标 rank 上准入动作链首节点的父边接在在飞
         列车 frontier 节点（＝该 rank 发射序列车尾节点）之后——
         "frontier 接续裁决"的发射序事实。
    """

    def _busy_vs_idle_frontier(self):
        """腿 1：真实 sh30 快照方法——忙实例 > 0、空闲实例 = 0。"""
        from face_scheduler import FaceHardware, FaceModel
        from online.sh30_online_scheduler import (
            Sh30OnlineScheduler,
            _OnlineInstanceState,
            _OnlineRequestRuntime,
            _TASK_LOAD_CACHE_CAPACITY,
        )
        scheduler = Sh30OnlineScheduler.__new__(Sh30OnlineScheduler)
        scheduler.instances = []
        scheduler.runtime_by_request_id = {}
        scheduler.topology = SimpleNamespace(
            instance=lambda index: SimpleNamespace(size=2))
        scheduler.hardware = FaceHardware(
            mesh_rows=1, mesh_cols=6, local_hbm_capacity_bytes=160 << 30,
            local_hbm_bandwidth_gbps=273.0, d2d_bandwidth_gbps=100.0,
            peak_perf_tflops=43.5, d2d_latency_ns=1, local_hbm_latency_ns=20,
            label="D03_E2E")
        scheduler.model = FaceModel(
            layers=8, hidden_size=256, ffn_size=512, num_heads=8,
            bytes_per_elem=2, vocab_size=2048, mlp_variant="gelu")
        scheduler._prefill_task_cache = {}
        scheduler._decode_task_load_cache = {}
        scheduler._task_load_cache_capacity = _TASK_LOAD_CACHE_CAPACITY
        idle_state = _OnlineInstanceState(index=0)
        busy_state = _OnlineInstanceState(index=1)
        runtime = _OnlineRequestRuntime(
            {
                "request_id": TRAIN_MEMBER_ID,
                "session_id": "session_d03_e2e",
                "turn_index": 0, "queue_index": 0,
                "prefill_length": 256, "decode_length": 4,
                "history_tokens_before": 0,
                "prefill_context_tokens": TRAIN_MEMBER_CTX,
                "final_context_tokens": TRAIN_MEMBER_CTX + 4,
            }, 512)
        scheduler.instances = [idle_state, busy_state]
        scheduler.runtime_by_request_id[TRAIN_MEMBER_ID] = runtime
        busy_state.in_flight_train = _inflight_train_snapshot()
        frontier = scheduler._prep_frontier_by_instance()
        self.assertEqual(frontier[0], 0, "idle instance has zero frontier")
        self.assertGreater(frontier[1], 0, "busy instance frontier > 0")
        return frontier

    def test_same_snapshot_frontier_value_reaches_predictor(self):
        """腿 1+2：_prep_frontier_by_instance 对冻结快照算出的 frontier
        值，被预测器逐字节同值消费并披露（锚点同源端到端）。"""
        frontier = self._busy_vs_idle_frontier()
        busy_frontier = frontier[1]
        model = _model({0: _load(0), 1: _load(1)})
        # stay-PARTIAL 候选（缺失后缀池恢复腿在关键路径上——PPrep
        # FrontierAlignmentTest 同款闭式对照形态）。
        busy_session = _session(
            home=1, resident=1, history_bytes=(1000, 1000),
            missing=(1000, 1000), location="partial_hbm_remote", prefix=2)
        candidate = model.estimate_P_with_continuation(
            session=busy_session,
            request=_request(), instance_index=1,
            action=ACTION_STAY, remote_enabled=True,
            prep_frontier_ns=busy_frontier)
        self.assertTrue(candidate.applicable)
        # 预测器消费的锚 = 调度器快照值（逐字节同值，非近似）。
        self.assertIn(
            f"prep_frontier_ns={busy_frontier}", candidate.breakdown.notes)
        self.assertIn(
            "prep_issue_frontier=P_prep_at_target_frontier"
            f"(+{busy_frontier}ns)",
            candidate.breakdown.notes)
        # 同一快照值下忙目标平移恰 F（腿在关键路径上的闭式对照）。
        idle = model.estimate_P_with_continuation(
            session=busy_session,
            request=_request(), instance_index=1,
            action=ACTION_STAY, remote_enabled=True,
            prep_frontier_ns=0)
        self.assertEqual(
            candidate.cost_ns - idle.cost_ns, busy_frontier,
            "the scheduler-computed frontier must shift the busy-target "
            "timeline by exactly that value (anchor same-source)")

    def test_admission_chain_extends_inflight_train_frontier(self):
        """腿 3：真实构图器发射——在飞列车后准入动作链首节点（忙目标
        rank 上首个带父边的准入节点）的父链接在列车 frontier 节点后
        （previous_id 无条件续链的发射序事实）。"""
        from online.graph_batch_builder import GraphBatchBuilder
        builder = GraphBatchBuilder(_graph_config())
        builder.begin_batch()
        builder.emit_iteration_train(_inflight_train_plan())
        # 在飞列车 frontier 节点 = 列车发射后各忙 rank 的主链尾。
        frontier_ids = {
            rank: builder.builders[rank].previous_id
            for rank in (2, 3)
        }
        for rank, frontier_id in frontier_ids.items():
            self.assertIsNotNone(frontier_id)
            train_ids = {
                node["id"] for node in builder.batch["nodes"]
                if node["rank"] == rank}
            self.assertEqual(
                max(train_ids), frontier_id,
                "captured frontier must be the train's last node on the "
                "rank (emission-order fact)")
        builder.emit_admission_batch(_admission_plan_busy_target())
        for rank, frontier_id in frontier_ids.items():
            admission_nodes = [
                node for node in builder.batch["nodes"]
                if node["rank"] == rank and node["id"] > frontier_id]
            self.assertTrue(
                admission_nodes,
                f"admission must emit nodes on busy rank {rank}")
            # 首个带父边的准入节点：其父边必须含列车 frontier 节点
            # （timer gate 不链 previous_id，链式续接自首个 _new_node
            # 节点起；其 previous_id 恰 = frontier_id）。
            chained = min(
                (node["id"] for node in admission_nodes
                 if _edge_sources_of(builder, rank, node["id"])))
            sources = _edge_sources_of(builder, rank, chained)
            self.assertIn(
                frontier_id, sources,
                f"first chained admission node on rank {rank} must extend "
                f"the in-flight train frontier {frontier_id} (got {sources})")
            # 准入链逐节点续链：后续节点的父边不被截断（发射序 = 主链序）。
            self.assertEqual(
                builder.builders[rank].previous_id,
                max(node["id"] for node in admission_nodes),
                "admission chain tail must be the rank's current frontier")


class SchedulerFrontierComputationTest(unittest.TestCase):
    """sh30 _prep_frontier_by_instance：在飞列车冻结快照口径。"""

    def _scheduler(self):
        from face_scheduler import FaceHardware, FaceModel
        from online.sh30_online_scheduler import Sh30OnlineScheduler
        scheduler = Sh30OnlineScheduler.__new__(Sh30OnlineScheduler)
        scheduler.instances = []
        scheduler.runtime_by_request_id = {}
        scheduler.topology = SimpleNamespace(
            instance=lambda index: SimpleNamespace(size=2))
        scheduler.hardware = FaceHardware(
            mesh_rows=1, mesh_cols=6, local_hbm_capacity_bytes=160 << 30,
            local_hbm_bandwidth_gbps=273.0, d2d_bandwidth_gbps=100.0,
            peak_perf_tflops=43.5, d2d_latency_ns=1, local_hbm_latency_ns=20,
            label="D03_FRONTIER")
        scheduler.model = FaceModel(
            layers=8, hidden_size=256, ffn_size=512, num_heads=8,
            bytes_per_elem=2, vocab_size=2048, mlp_variant="gelu")
        scheduler._prefill_task_cache = {}
        scheduler._decode_task_load_cache = {}
        from online.sh30_online_scheduler import _TASK_LOAD_CACHE_CAPACITY
        scheduler._task_load_cache_capacity = _TASK_LOAD_CACHE_CAPACITY
        return scheduler

    def test_idle_instance_zero_and_busy_instance_positive(self):
        from online.sh30_online_scheduler import (
            _OnlineInstanceState,
            _OnlineRequestRuntime,
        )
        scheduler = self._scheduler()
        idle_state = _OnlineInstanceState(index=0)
        busy_state = _OnlineInstanceState(index=1)
        runtime = _OnlineRequestRuntime(
            {
                "request_id": "r_busy", "session_id": "s",
                "turn_index": 0, "queue_index": 0,
                "prefill_length": 512, "decode_length": 4,
                "history_tokens_before": 0,
                "prefill_context_tokens": 512,
                "final_context_tokens": 516,
            }, 512)
        scheduler.instances = [idle_state, busy_state]
        scheduler.runtime_by_request_id["r_busy"] = runtime
        busy_state.in_flight_train = {
            "prefill_chunk_spans": [(512, 512)],
            "members": [("r_busy", 2)],
        }
        # chunk 分量（roofline 1 ns/token 量级）+ participation 步服务
        # ——忙实例 > 0、空闲实例 = 0；排队（无在飞列车）不推 frontier。
        frontier = scheduler._prep_frontier_by_instance()
        self.assertEqual(frontier[0], 0)
        self.assertGreater(frontier[1], 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
